from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd
import pytest

from analysis.pipeline.compare_runs import (
    CANONICAL_COLUMNS,
    COMPARISONS_ROOT,
    ComparisonSpec,
    RUNS_ROOT,
    build_comparison_bundle,
)
from analysis.pipeline.report_table_renderer import resolve_title
from analysis.pipeline.view_table_builder import build_view_bundles, parse_view_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


@contextmanager
def _phase8_results_root():
    token = uuid4().hex
    created_paths: list[Path] = []
    try:
        yield token, created_paths
    finally:
        for path in reversed(created_paths):
            shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_bundle(
    *,
    run_id: str,
    rows: list[dict[str, object]],
    slice_manifest: dict[str, object],
    created_paths: list[Path],
) -> Path:
    bundle_dir = RUNS_ROOT / run_id
    created_paths.append(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=CANONICAL_COLUMNS).to_csv(bundle_dir / "long_table.csv", index=False)
    _write_json(
        bundle_dir / "manifest.json",
        {
            "run_id": run_id,
            "canonical_columns": CANONICAL_COLUMNS,
            "generated_files": [
                "inventory.json",
                "long_table.csv",
                "manifest.json",
                "slice_manifest.json",
            ],
            "slice": {
                "slice_policy": slice_manifest["slice_policy"],
                "requested_victims": slice_manifest["requested_victims"],
                "selected_targets": slice_manifest["selected_targets"],
                "selected_target_count": slice_manifest["selected_target_count"],
                "fairness_safe": slice_manifest["fairness_safe"],
            },
        },
    )
    _write_json(bundle_dir / "slice_manifest.json", slice_manifest)
    return bundle_dir


def _rows_for_run(run_id: str) -> list[dict[str, object]]:
    base_value = 0.1 if run_id.endswith("a") else 0.2
    return [
        {
            "run_id": run_id,
            "dataset": "diginetica",
            "attack_method": "clean",
            "victim_model": "miasrec",
            "target_item": 111,
            "target_type": "popular",
            "attack_size": 0.1,
            "poison_model": "heuristic",
            "fake_session_generation_topk": 20,
            "replacement_topk_ratio": 0.5,
            "metric": "recall",
            "k": 10,
            "value": base_value,
        },
        {
            "run_id": run_id,
            "dataset": "diginetica",
            "attack_method": "clean",
            "victim_model": "tron",
            "target_item": 111,
            "target_type": "popular",
            "attack_size": 0.1,
            "poison_model": "heuristic",
            "fake_session_generation_topk": 20,
            "replacement_topk_ratio": 0.5,
            "metric": "recall",
            "k": 10,
            "value": base_value + 0.05,
        },
    ]


def _base_slice_manifest(*, run_group_key: str = "run_group_test") -> dict[str, object]:
    return {
        "source_run_group_key": run_group_key,
        "target_cohort_key": "target_cohort_test",
        "source_summary_current_path": "outputs/runs/test/summary_current.json",
        "source_run_coverage_path": "outputs/runs/test/run_coverage.json",
        "source_target_registry_path": "outputs/shared/test/target_registry.json",
        "source_resolved_config_path": "outputs/runs/test/resolved_config.json",
        "source_artifact_manifest_path": "outputs/runs/test/artifact_manifest.json",
        "slice_policy": "largest_complete_prefix",
        "requested_victims": ["miasrec", "tron"],
        "requested_victims_source": "cli",
        "requested_target_count": 2,
        "selected_targets": [111, 222],
        "selected_target_count": 2,
        "excluded_targets": [],
        "excluded_incomplete_cells": [],
        "fairness_safe": True,
        "generation_timestamp": "2026-04-19T00:00:00+00:00",
    }


def _comparison_spec(*, token: str, run_ids: list[str], slice_compatibility: str = "strict") -> ComparisonSpec:
    return ComparisonSpec(
        comparison_id=f"phase8_{token}_comparison",
        run_ids=run_ids,
        output_dir=COMPARISONS_ROOT / f"phase8_{token}_comparison",
        slice_compatibility=slice_compatibility,
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_compare_runs_strict_accepts_compatible_slice_manifests() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_ids = [f"phase8_{token}_a", f"phase8_{token}_b"]
        for run_id in run_ids:
            _write_run_bundle(
                run_id=run_id,
                rows=_rows_for_run(run_id),
                slice_manifest=_base_slice_manifest(run_group_key=f"{run_id}_group"),
                created_paths=created_paths,
            )

        spec = _comparison_spec(token=token, run_ids=run_ids)
        created_paths.append(spec.output_dir)
        result = build_comparison_bundle(spec)
        manifest = _load_json(result["manifest_path"])

    assert result["row_count"] == 4
    assert manifest["slice_compatibility"]["mode"] == "strict"
    assert manifest["slice_compatibility"]["compatible"] is True


def test_compare_runs_strict_rejects_incompatible_slice_policy() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_a = f"phase8_{token}_policy_a"
        run_b = f"phase8_{token}_policy_b"
        _write_run_bundle(
            run_id=run_a,
            rows=_rows_for_run(run_a),
            slice_manifest=_base_slice_manifest(),
            created_paths=created_paths,
        )
        incompatible_slice = _base_slice_manifest()
        incompatible_slice["slice_policy"] = "intersection_complete"
        _write_run_bundle(
            run_id=run_b,
            rows=_rows_for_run(run_b),
            slice_manifest=incompatible_slice,
            created_paths=created_paths,
        )

        spec = _comparison_spec(token=token, run_ids=[run_a, run_b])
        with pytest.raises(ValueError) as exc_info:
            build_comparison_bundle(spec)

    assert "slice_policy" in str(exc_info.value)


def test_compare_runs_strict_rejects_incompatible_requested_victims() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_a = f"phase8_{token}_victims_a"
        run_b = f"phase8_{token}_victims_b"
        _write_run_bundle(
            run_id=run_a,
            rows=_rows_for_run(run_a),
            slice_manifest=_base_slice_manifest(),
            created_paths=created_paths,
        )
        incompatible_slice = _base_slice_manifest()
        incompatible_slice["requested_victims"] = ["miasrec"]
        incompatible_slice["selected_targets"] = [111, 222, 333]
        incompatible_slice["selected_target_count"] = 3
        _write_run_bundle(
            run_id=run_b,
            rows=_rows_for_run(run_b),
            slice_manifest=incompatible_slice,
            created_paths=created_paths,
        )

        spec = _comparison_spec(token=token, run_ids=[run_a, run_b])
        with pytest.raises(ValueError) as exc_info:
            build_comparison_bundle(spec)

    assert "requested_victims" in str(exc_info.value)


def test_compare_runs_strict_rejects_incompatible_selected_targets_and_count() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_a = f"phase8_{token}_targets_a"
        run_b = f"phase8_{token}_targets_b"
        _write_run_bundle(
            run_id=run_a,
            rows=_rows_for_run(run_a),
            slice_manifest=_base_slice_manifest(),
            created_paths=created_paths,
        )
        incompatible_slice = _base_slice_manifest()
        incompatible_slice["selected_targets"] = [111]
        incompatible_slice["selected_target_count"] = 1
        _write_run_bundle(
            run_id=run_b,
            rows=_rows_for_run(run_b),
            slice_manifest=incompatible_slice,
            created_paths=created_paths,
        )

        spec = _comparison_spec(token=token, run_ids=[run_a, run_b])
        with pytest.raises(ValueError) as exc_info:
            build_comparison_bundle(spec)

    assert "selected_targets" in str(exc_info.value) or "selected_target_count" in str(exc_info.value)


def test_compare_runs_writes_merged_slice_metadata_into_manifest() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_ids = [f"phase8_{token}_merge_a", f"phase8_{token}_merge_b"]
        for run_id in run_ids:
            _write_run_bundle(
                run_id=run_id,
                rows=_rows_for_run(run_id),
                slice_manifest=_base_slice_manifest(),
                created_paths=created_paths,
            )

        spec = _comparison_spec(token=token, run_ids=run_ids)
        created_paths.append(spec.output_dir)
        result = build_comparison_bundle(spec)
        manifest = _load_json(result["manifest_path"])
        slice_manifest = _load_json(result["slice_manifest_path"])

    assert manifest["slice"]["slice_policy"] == "largest_complete_prefix"
    assert manifest["slice"]["requested_victims"] == ["miasrec", "tron"]
    assert manifest["slice"]["selected_targets"] == [111, 222]
    assert manifest["slice"]["selected_target_count"] == 2
    assert slice_manifest["source_run_ids"] == run_ids


def test_relaxed_debug_comparison_is_clearly_labeled_non_default() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_a = f"phase8_{token}_relaxed_a"
        run_b = f"phase8_{token}_relaxed_b"
        _write_run_bundle(
            run_id=run_a,
            rows=_rows_for_run(run_a),
            slice_manifest=_base_slice_manifest(),
            created_paths=created_paths,
        )
        incompatible_slice = _base_slice_manifest()
        incompatible_slice["selected_targets"] = [111]
        incompatible_slice["selected_target_count"] = 1
        incompatible_slice["fairness_safe"] = False
        _write_run_bundle(
            run_id=run_b,
            rows=_rows_for_run(run_b),
            slice_manifest=incompatible_slice,
            created_paths=created_paths,
        )

        spec = _comparison_spec(
            token=token,
            run_ids=[run_a, run_b],
            slice_compatibility="relaxed_debug",
        )
        created_paths.append(spec.output_dir)
        result = build_comparison_bundle(spec)
        manifest = _load_json(result["manifest_path"])

    assert manifest["slice_compatibility"]["mode"] == "relaxed_debug"
    assert manifest["slice_compatibility"]["debug_only"] is True
    assert manifest["slice_compatibility"]["compatible"] is False
    assert manifest["slice"]["selected_targets"] is None


def test_view_table_builder_propagates_slice_metadata_into_meta_and_context() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_ids = [f"phase8_{token}_view_a", f"phase8_{token}_view_b"]
        for run_id in run_ids:
            _write_run_bundle(
                run_id=run_id,
                rows=_rows_for_run(run_id),
                slice_manifest=_base_slice_manifest(),
                created_paths=created_paths,
            )

        comparison_spec = _comparison_spec(token=token, run_ids=run_ids)
        created_paths.append(comparison_spec.output_dir)
        comparison_result = build_comparison_bundle(comparison_spec)

        output_bundle_dir = RESULTS_ROOT / "views" / f"phase8_{token}_view_bundle"
        created_paths.append(output_bundle_dir)
        view_spec = parse_view_spec(
            {
                "input": str(comparison_result["merged_csv_path"]),
                "output": str(output_bundle_dir),
                "name": f"phase8_{token}_view_bundle",
                "filters": {"metric": "recall", "k": 10},
                "rows": ["victim_model"],
                "cols": ["run_id"],
                "value_col": "value",
                "agg": "mean",
                "auto_context": True,
                "require_unique_cells": True,
            },
            source_spec_path=REPO_ROOT / "analysis" / "tests" / "phase8_view_spec.yaml",
        )
        bundle_dirs = build_view_bundles(view_spec)
        meta = _load_json(bundle_dirs[0] / "meta.json")

    assert meta["slice"]["slice_policy"] == "largest_complete_prefix"
    assert meta["slice_context"]["selected_target_count"] == 2
    assert meta["context"]["slice_policy"] == "largest_complete_prefix"
    assert meta["context"]["requested_victims"] == ["miasrec", "tron"]
    assert meta["source_slice_manifest_path"].endswith("slice_manifest.json")


def test_renderer_can_consume_propagated_slice_metadata() -> None:
    meta_payload = {
        "context": {
            "slice_policy": "largest_complete_prefix",
            "selected_target_count": 2,
            "requested_victims": ["miasrec", "tron"],
        },
        "slice_context": {
            "fairness_safe": True,
        },
        "effective_filters": {},
        "split_values": {},
    }

    title = resolve_title(
        template="Policy {slice_policy} N={selected_target_count} Victims {requested_victims} Safe={fairness_safe}",
        meta_payload=meta_payload,
    )

    assert title == "Policy largest_complete_prefix N=2 Victims miasrec, tron Safe=True"


def test_renderer_remains_presentation_only_without_runtime_state_inputs() -> None:
    meta_payload = {
        "context": {},
        "slice_context": {
            "slice_policy": "intersection_complete",
            "selected_target_count": 3,
            "requested_victims": ["miasrec"],
        },
        "effective_filters": {},
        "split_values": {},
    }

    title = resolve_title(
        template="Slice {slice_policy} N={selected_target_count} Victims {requested_victims}",
        meta_payload=meta_payload,
    )

    assert title == "Slice intersection_complete N=3 Victims miasrec"
