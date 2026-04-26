from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from analysis.pipeline.compare_runs import (
    CANONICAL_COLUMNS,
    COMPARISONS_ROOT,
    ComparisonSpec,
    RUNS_ROOT,
    build_comparison_bundle,
)
from analysis.pipeline.comparison_stats_builder import (
    build_comparison_stats_bundle,
    parse_comparison_stats_spec,
)
from analysis.pipeline.report_table_renderer import (
    BestValueBoldingSpec,
    TableStructure,
    apply_dimension_value_orders,
    build_data_cell_presentation,
    draw_cell_block,
    format_dataframe_for_display,
    parse_render_spec,
    resolve_data_cell_fill_color,
    resolve_ranked_value_highlights,
    resolve_title,
)
from analysis.pipeline.view_table_builder import (
    AnalysisError as ViewAnalysisError,
    build_view_bundles,
    parse_view_spec,
)


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


def _write_long_table_bundle(
    *,
    bundle_name: str,
    rows: list[dict[str, object]],
    created_paths: list[Path],
) -> Path:
    bundle_dir = RESULTS_ROOT / "tests" / bundle_name
    created_paths.append(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    csv_path = bundle_dir / "long_table.csv"
    pd.DataFrame(rows, columns=CANONICAL_COLUMNS).to_csv(csv_path, index=False)
    return csv_path


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


def _rows_for_method_stats_case(
    *,
    run_id: str,
    attack_method: str,
    replacement_topk_ratio: float,
    values: list[float],
) -> list[dict[str, object]]:
    units = [
        ("miasrec", 111, "recall", 10, values[0]),
        ("miasrec", 111, "ground_truth_recall", 10, values[1]),
        ("tron", 222, "recall", 20, values[2]),
        ("tron", 222, "ground_truth_recall", 20, values[3]),
    ]
    rows: list[dict[str, object]] = []
    for victim_model, target_item, metric, k, value in units:
        rows.append(
            {
                "run_id": run_id,
                "dataset": "diginetica",
                "attack_method": attack_method,
                "victim_model": victim_model,
                "target_item": target_item,
                "target_type": "popular",
                "attack_size": 0.1,
                "poison_model": "heuristic",
                "fake_session_generation_topk": 20,
                "replacement_topk_ratio": replacement_topk_ratio,
                "metric": metric,
                "k": k,
                "value": value,
            }
        )
    return rows


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


def test_comparison_stats_builder_reports_pairwise_win_rates_from_filtered_comparison_bundle() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_specs = [
            (
                f"phase8_{token}_stats_a",
                "method_a",
                0.2,
                [0.9, 0.8, 0.7, 0.6],
            ),
            (
                f"phase8_{token}_stats_b",
                "method_b",
                1.0,
                [0.8, 0.9, 0.7, 0.5],
            ),
            (
                f"phase8_{token}_stats_c",
                "method_c",
                0.5,
                [0.1, 0.2, 0.3, 0.4],
            ),
        ]
        run_ids: list[str] = []
        for run_id, attack_method, replacement_topk_ratio, values in run_specs:
            run_ids.append(run_id)
            _write_run_bundle(
                run_id=run_id,
                rows=_rows_for_method_stats_case(
                    run_id=run_id,
                    attack_method=attack_method,
                    replacement_topk_ratio=replacement_topk_ratio,
                    values=values,
                ),
                slice_manifest=_base_slice_manifest(run_group_key=f"{run_id}_group"),
                created_paths=created_paths,
            )

        comparison_spec = _comparison_spec(token=token, run_ids=run_ids)
        created_paths.append(comparison_spec.output_dir)
        comparison_result = build_comparison_bundle(comparison_spec)

        output_dir = RESULTS_ROOT / "tests" / f"phase8_{token}_stats_bundle"
        created_paths.append(output_dir)
        stats_spec = parse_comparison_stats_spec(
            {
                "analysis_type": "pairwise_method_win_rate",
                "input": str(comparison_result["merged_csv_path"]),
                "output": str(output_dir),
                "filters": {
                    "metric_name": ["recall"],
                    "metric_scope": ["targeted", "ground_truth"],
                },
                "pairing": {
                    "auto_ignore_method_descriptor_columns": True,
                },
            },
            source_spec_path=REPO_ROOT / "analysis" / "configs" / "stats" / f"{token}.yaml",
        )
        result = build_comparison_stats_bundle(stats_spec)
        summary = _load_json(result["summary_path"])

    assert result["method_count"] == 3
    assert result["pair_count"] == 3
    assert summary["row_count_after_filters"] == 12
    assert summary["comparison_unit_count"] == 4
    assert summary["pairing"]["explicit_ignore_columns"] == ["run_id"]
    assert summary["pairing"]["auto_ignore_method_descriptor_columns"] == ["replacement_topk_ratio"]
    assert summary["pairing"]["comparison_unit_columns"] == [
        "dataset",
        "victim_model",
        "target_item",
        "target_type",
        "attack_size",
        "poison_model",
        "fake_session_generation_topk",
        "metric",
        "k",
        "metric_name",
        "metric_scope",
    ]
    method_a_vs_b = summary["pairwise_lookup"]["method_a"]["method_b"]
    assert method_a_vs_b["comparable_result_count"] == 4
    assert method_a_vs_b["left_win_count"] == 2
    assert method_a_vs_b["right_win_count"] == 1
    assert method_a_vs_b["tie_count"] == 1
    assert method_a_vs_b["left_win_rate"] == 0.5
    assert method_a_vs_b["right_win_rate"] == 0.25
    assert method_a_vs_b["tie_rate"] == 0.25
    assert summary["pairwise_lookup"]["method_b"]["method_a"]["left_win_count"] == 1


def test_comparison_stats_builder_supports_requested_pairs_and_skips_full_matrices_by_default() -> None:
    with _phase8_results_root() as (token, created_paths):
        run_specs = [
            (
                f"phase8_{token}_selected_a",
                "method_a",
                0.2,
                [0.9, 0.8, 0.7, 0.6],
            ),
            (
                f"phase8_{token}_selected_b",
                "method_b",
                1.0,
                [0.8, 0.9, 0.7, 0.5],
            ),
            (
                f"phase8_{token}_selected_c",
                "method_c",
                0.5,
                [0.1, 0.2, 0.3, 0.4],
            ),
        ]
        run_ids: list[str] = []
        for run_id, attack_method, replacement_topk_ratio, values in run_specs:
            run_ids.append(run_id)
            _write_run_bundle(
                run_id=run_id,
                rows=_rows_for_method_stats_case(
                    run_id=run_id,
                    attack_method=attack_method,
                    replacement_topk_ratio=replacement_topk_ratio,
                    values=values,
                ),
                slice_manifest=_base_slice_manifest(run_group_key=f"{run_id}_group"),
                created_paths=created_paths,
            )

        comparison_spec = _comparison_spec(token=token, run_ids=run_ids)
        created_paths.append(comparison_spec.output_dir)
        comparison_result = build_comparison_bundle(comparison_spec)

        output_dir = RESULTS_ROOT / "tests" / f"phase8_{token}_selected_pairs_stats_bundle"
        created_paths.append(output_dir)
        stats_spec = parse_comparison_stats_spec(
            {
                "analysis_type": "pairwise_method_win_rate",
                "input": str(comparison_result["merged_csv_path"]),
                "output": str(output_dir),
                "filters": {
                    "metric_name": ["recall"],
                    "metric_scope": ["targeted", "ground_truth"],
                },
                "pairing": {
                    "auto_ignore_method_descriptor_columns": True,
                },
                "pairwise_method_win_rate": {
                    "requested_pairs": [
                        ["method_a", "method_b"],
                        ["method_c", "method_b"],
                    ]
                },
            },
            source_spec_path=REPO_ROOT / "analysis" / "configs" / "stats" / f"{token}.yaml",
        )
        result = build_comparison_stats_bundle(stats_spec)
        summary_path = result["summary_path"]
        summary = _load_json(summary_path)
        pairwise_results_csv_path = output_dir / "pairwise_results.csv"
        win_rate_matrix_path = output_dir / "pairwise_win_rate_matrix.csv"
        comparable_count_matrix_path = output_dir / "pairwise_comparable_count_matrix.csv"
        pairwise_results_csv = pd.read_csv(pairwise_results_csv_path)

    assert result["method_count"] == 3
    assert result["pair_count"] == 2
    assert summary["methods"] == ["method_a", "method_b", "method_c"]
    assert summary["available_methods"] == ["method_a", "method_b", "method_c"]
    assert summary["pair_selection"] == {
        "mode": "selected_pairs",
        "requested_pairs": [
            {"left_method": "method_a", "right_method": "method_b"},
            {"left_method": "method_c", "right_method": "method_b"},
        ],
        "method_count": 3,
        "methods": ["method_a", "method_b", "method_c"],
        "write_full_matrices": False,
    }
    assert list(pairwise_results_csv["left_method"]) == ["method_a", "method_c"]
    assert list(pairwise_results_csv["right_method"]) == ["method_b", "method_b"]
    assert not win_rate_matrix_path.exists()
    assert not comparable_count_matrix_path.exists()
    assert "method_c" not in summary["pairwise_lookup"]["method_a"]


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


def test_view_table_builder_transforms_ground_truth_relative_to_clean_before_aggregation() -> None:
    with _phase8_results_root() as (token, created_paths):
        input_csv = _write_long_table_bundle(
            bundle_name=f"phase8_{token}_gt_relative_input",
            rows=[
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.50,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.60,
                },
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 102,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.20,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 102,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.10,
                },
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.01,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.07,
                },
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 102,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.02,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 102,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.08,
                },
            ],
            created_paths=created_paths,
        )

        output_bundle_dir = RESULTS_ROOT / "views" / f"phase8_{token}_gt_relative_bundle"
        created_paths.append(output_bundle_dir)
        view_spec = parse_view_spec(
            {
                "input": str(input_csv),
                "output": str(output_bundle_dir),
                "name": f"phase8_{token}_gt_relative_bundle",
                "filters": {"metric_name": "recall", "k": 10},
                "rows": ["attack_method"],
                "cols": ["metric_scope"],
                "value_col": "value",
                "agg": "mean",
                "auto_context": True,
                "require_unique_cells": False,
                "ground_truth_relative_to_clean": {
                    "enabled": True,
                    "baseline_attack_method": "clean",
                    "ignore_pairing_columns": ["run_id"],
                },
            },
            source_spec_path=REPO_ROOT / "analysis" / "tests" / "phase8_gt_relative_view_spec.yaml",
        )

        bundle_dirs = build_view_bundles(view_spec)
        table = pd.read_csv(bundle_dirs[0] / "table.csv")
        meta = _load_json(bundle_dirs[0] / "meta.json")
        source_table = pd.read_csv(input_csv)

    assert source_table[source_table["metric"] == "ground_truth_recall"]["value"].tolist() == [
        0.50,
        0.60,
        0.20,
        0.10,
    ]
    assert table["attack_method"].tolist() == ["clean", "dpsbr_baseline"]
    assert table["ground_truth"].tolist() == pytest.approx([0.35, -15.0])
    assert table["targeted"].tolist() == pytest.approx([0.015, 0.075])
    assert meta["ground_truth_relative_to_clean"] == {
        "enabled": True,
        "baseline_attack_method": "clean",
        "ignore_pairing_columns": ["run_id"],
    }


def test_view_table_builder_rejects_gt_relative_views_that_mix_metric_scopes_per_cell() -> None:
    with _phase8_results_root() as (token, created_paths):
        input_csv = _write_long_table_bundle(
            bundle_name=f"phase8_{token}_gt_relative_conflict_input",
            rows=[
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.50,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.60,
                },
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.01,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "targeted_recall",
                    "k": 10,
                    "value": 0.07,
                },
            ],
            created_paths=created_paths,
        )

        output_bundle_dir = RESULTS_ROOT / "views" / f"phase8_{token}_gt_relative_conflict_bundle"
        created_paths.append(output_bundle_dir)
        view_spec = parse_view_spec(
            {
                "input": str(input_csv),
                "output": str(output_bundle_dir),
                "name": f"phase8_{token}_gt_relative_conflict_bundle",
                "filters": {"metric_name": "recall", "k": 10},
                "rows": ["attack_method"],
                "cols": ["metric_name"],
                "value_col": "value",
                "agg": "mean",
                "auto_context": False,
                "require_unique_cells": False,
                "ground_truth_relative_to_clean": {
                    "enabled": True,
                    "baseline_attack_method": "clean",
                    "ignore_pairing_columns": ["run_id"],
                },
            },
            source_spec_path=REPO_ROOT / "analysis" / "tests" / "phase8_gt_relative_conflict_view_spec.yaml",
        )

        with pytest.raises(ViewAnalysisError) as exc_info:
            build_view_bundles(view_spec)

    assert "exactly one metric_scope" in str(exc_info.value)


def test_view_table_builder_rejects_gt_relative_views_that_mix_clean_raw_and_attack_deltas() -> None:
    with _phase8_results_root() as (token, created_paths):
        input_csv = _write_long_table_bundle(
            bundle_name=f"phase8_{token}_gt_relative_mode_conflict_input",
            rows=[
                {
                    "run_id": "clean_run",
                    "dataset": "diginetica",
                    "attack_method": "clean",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.50,
                },
                {
                    "run_id": "attack_run",
                    "dataset": "diginetica",
                    "attack_method": "dpsbr_baseline",
                    "victim_model": "miasrec",
                    "target_item": 101,
                    "target_type": "popular",
                    "metric": "ground_truth_recall",
                    "k": 10,
                    "value": 0.60,
                },
            ],
            created_paths=created_paths,
        )

        output_bundle_dir = RESULTS_ROOT / "views" / f"phase8_{token}_gt_relative_mode_conflict_bundle"
        created_paths.append(output_bundle_dir)
        view_spec = parse_view_spec(
            {
                "input": str(input_csv),
                "output": str(output_bundle_dir),
                "name": f"phase8_{token}_gt_relative_mode_conflict_bundle",
                "filters": {"metric_name": "recall", "metric_scope": "ground_truth", "k": 10},
                "rows": ["victim_model"],
                "cols": ["metric_scope"],
                "value_col": "value",
                "agg": "mean",
                "auto_context": False,
                "require_unique_cells": False,
                "ground_truth_relative_to_clean": {
                    "enabled": True,
                    "baseline_attack_method": "clean",
                    "ignore_pairing_columns": ["run_id"],
                },
            },
            source_spec_path=REPO_ROOT / "analysis" / "tests" / "phase8_gt_relative_mode_conflict_view_spec.yaml",
        )

        with pytest.raises(ViewAnalysisError) as exc_info:
            build_view_bundles(view_spec)

    assert "exactly one value unit" in str(exc_info.value)


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


def test_renderer_formats_gt_relative_cells_as_signed_percent_and_colors_by_signed_magnitude() -> None:
    dataframe = pd.DataFrame(
        [
            {"attack_method": "clean", "ground_truth": 0.35, "targeted": 0.015},
            {"attack_method": "dpsbr_baseline", "ground_truth": -15.0, "targeted": 0.075},
            {"attack_method": "position_opt_mvp", "ground_truth": 20.0, "targeted": 0.050},
        ]
    )
    table_structure = TableStructure(
        row_levels=["attack_method"],
        col_levels=["metric_scope"],
        row_tuples=[
            ("clean",),
            ("dpsbr_baseline",),
            ("position_opt_mvp",),
        ],
        column_tuples=[("ground_truth",), ("targeted",)],
        row_column_names=["attack_method"],
        value_column_names=["ground_truth", "targeted"],
    )
    meta_payload = {
        "ground_truth_relative_to_clean": {
            "enabled": True,
            "baseline_attack_method": "clean",
            "ignore_pairing_columns": ["run_id"],
        },
        "context": {},
        "effective_filters": {"metric_name": "recall", "k": 10},
        "split_values": {},
    }

    data_cell_presentation = build_data_cell_presentation(
        dataframe=dataframe,
        table_structure=table_structure,
        meta_payload=meta_payload,
    )
    display_dataframe = format_dataframe_for_display(
        dataframe,
        identifier_columns={"attack_method"},
        round_digits=6,
        signed_percent_round_digits=4,
        value_alias={},
        table_structure=table_structure,
        data_cell_presentation=data_cell_presentation,
    )

    assert data_cell_presentation.display_modes == [
        ["absolute", "absolute"],
        ["signed_percent", "absolute"],
        ["signed_percent", "absolute"],
    ]
    assert display_dataframe["ground_truth"].tolist() == ["0.350000", "-15.0000%", "+20.0000%"]
    assert display_dataframe["targeted"].tolist() == ["0.015000", "0.075000", "0.050000"]

    scope_colors = {}
    clean_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[0],
        column_tuple=table_structure.column_tuples[0],
        raw_value=0.35,
        display_mode=data_cell_presentation.display_modes[0][0],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[0][0],
        leaf_column_fill_color=None,
        scope_colors=scope_colors,
    )
    negative_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[1],
        column_tuple=table_structure.column_tuples[0],
        raw_value=-15.0,
        display_mode=data_cell_presentation.display_modes[1][0],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[1][0],
        leaf_column_fill_color=None,
        scope_colors=scope_colors,
    )
    positive_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[2],
        column_tuple=table_structure.column_tuples[0],
        raw_value=20.0,
        display_mode=data_cell_presentation.display_modes[2][0],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[2][0],
        leaf_column_fill_color=None,
        scope_colors=scope_colors,
    )
    targeted_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[2],
        column_tuple=table_structure.column_tuples[1],
        raw_value=0.050,
        display_mode=data_cell_presentation.display_modes[2][1],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[2][1],
        leaf_column_fill_color=None,
        scope_colors=scope_colors,
    )

    assert clean_color is None
    assert positive_color is not None
    assert negative_color is not None
    assert positive_color != negative_color
    assert targeted_color is None


def test_renderer_scales_gt_heatmap_separately_for_positive_and_negative_values_within_row_group() -> None:
    dataframe = pd.DataFrame(
        [
            {"victim_model": "miasrec", "attack_method": "clean", "ground_truth": 0.35},
            {"victim_model": "miasrec", "attack_method": "dpsbr_baseline", "ground_truth": -20.0},
            {"victim_model": "miasrec", "attack_method": "prefix_nonzero_when_possible", "ground_truth": -10.0},
            {"victim_model": "miasrec", "attack_method": "random_nonzero_when_possible", "ground_truth": -5.0},
            {"victim_model": "miasrec", "attack_method": "position_opt_mvp", "ground_truth": 1.0},
            {"victim_model": "tron", "attack_method": "clean", "ground_truth": 0.42},
            {"victim_model": "tron", "attack_method": "dpsbr_baseline", "ground_truth": 8.0},
        ]
    )
    table_structure = TableStructure(
        row_levels=["victim_model", "attack_method"],
        col_levels=["metric_scope"],
        row_tuples=[
            ("miasrec", "clean"),
            ("miasrec", "dpsbr_baseline"),
            ("miasrec", "prefix_nonzero_when_possible"),
            ("miasrec", "random_nonzero_when_possible"),
            ("miasrec", "position_opt_mvp"),
            ("tron", "clean"),
            ("tron", "dpsbr_baseline"),
        ],
        column_tuples=[("ground_truth",)],
        row_column_names=["victim_model", "attack_method"],
        value_column_names=["ground_truth"],
    )
    meta_payload = {
        "ground_truth_relative_to_clean": {
            "enabled": True,
            "baseline_attack_method": "clean",
            "ignore_pairing_columns": ["run_id", "replacement_topk_ratio"],
        },
        "context": {},
        "effective_filters": {"metric_name": "recall", "k": 10},
        "split_values": {},
    }

    data_cell_presentation = build_data_cell_presentation(
        dataframe=dataframe,
        table_structure=table_structure,
        meta_payload=meta_payload,
    )

    positive_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[4],
        column_tuple=table_structure.column_tuples[0],
        raw_value=1.0,
        display_mode=data_cell_presentation.display_modes[4][0],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[4][0],
        leaf_column_fill_color=None,
        scope_colors={},
    )
    negative_color = resolve_data_cell_fill_color(
        table_structure=table_structure,
        row_tuple=table_structure.row_tuples[1],
        column_tuple=table_structure.column_tuples[0],
        raw_value=-20.0,
        display_mode=data_cell_presentation.display_modes[1][0],
        signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[1][0],
        leaf_column_fill_color=None,
        scope_colors={},
    )

    assert data_cell_presentation.signed_percent_scales[4][0] is not None
    assert data_cell_presentation.display_modes[0][0] == "absolute"
    assert data_cell_presentation.signed_percent_scales[0][0] is None
    assert data_cell_presentation.signed_percent_scales[4][0].positive_abs_max == pytest.approx(1.0)
    assert data_cell_presentation.signed_percent_scales[4][0].negative_abs_max == pytest.approx(20.0)
    assert positive_color == "#2ca25f"
    assert negative_color == "#de2d26"


def test_renderer_keeps_absolute_values_when_gt_relative_metadata_is_absent() -> None:
    dataframe = pd.DataFrame([{"attack_method": "position_opt_mvp", "ground_truth": 20.0}])
    table_structure = TableStructure(
        row_levels=["attack_method"],
        col_levels=["metric_scope"],
        row_tuples=[("position_opt_mvp",)],
        column_tuples=[("ground_truth",)],
        row_column_names=["attack_method"],
        value_column_names=["ground_truth"],
    )
    data_cell_presentation = build_data_cell_presentation(
        dataframe=dataframe,
        table_structure=table_structure,
        meta_payload={
            "context": {},
            "effective_filters": {"metric_name": "recall", "k": 10},
            "split_values": {},
        },
    )
    display_dataframe = format_dataframe_for_display(
        dataframe,
        identifier_columns={"attack_method"},
        round_digits=3,
        value_alias={},
        table_structure=table_structure,
        data_cell_presentation=data_cell_presentation,
    )

    assert data_cell_presentation.display_modes == [["absolute"]]
    assert display_dataframe["ground_truth"].tolist() == ["20.000"]


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


def test_renderer_parses_second_best_underline_flag() -> None:
    render_spec = parse_render_spec(
        {
            "style_name": "phase8_ranked_highlight",
            "output_format": "png",
            "title": {
                "template": "test",
                "align": "center",
                "font_size": 12,
                "color": "black",
            },
            "figure": {
                "width": 8,
                "height": 4,
                "dpi": 100,
                "background_color": "white",
            },
            "table": {
                "font_size": 10,
                "round_digits": 4,
                "text_color": "black",
                "show_grid": True,
                "auto_shrink": False,
                "wrap_text": False,
                "cell_align": "center",
                "display_alias": {},
                "value_alias": {},
                "dimension_value_orders": {},
                "scope_colors": {},
                "best_value_bolding": {
                    "compare_along": "rows",
                    "mode": "max",
                    "partition_by_levels": ["victim_model"],
                    "underline_second_best": True,
                },
                "top_level_group_separators": False,
            },
        }
    )

    assert render_spec.table.best_value_bolding is not None
    assert render_spec.table.best_value_bolding.underline_second_best is True


def test_renderer_parses_optional_signed_percent_round_digits() -> None:
    render_spec = parse_render_spec(
        {
            "style_name": "phase8_signed_percent_digits",
            "output_format": "png",
            "title": {
                "template": "test",
                "align": "center",
                "font_size": 12,
                "color": "black",
            },
            "figure": {
                "width": 8,
                "height": 4,
                "dpi": 100,
                "background_color": "white",
            },
            "table": {
                "font_size": 10,
                "round_digits": 6,
                "signed_percent_round_digits": 4,
                "text_color": "black",
                "show_grid": True,
                "auto_shrink": False,
                "wrap_text": False,
                "cell_align": "center",
                "display_alias": {},
                "value_alias": {},
                "dimension_value_orders": {},
                "scope_colors": {},
                "top_level_group_separators": False,
            },
        }
    )

    assert render_spec.table.round_digits == 6
    assert render_spec.table.signed_percent_round_digits == 4


def test_renderer_can_identify_best_and_second_best_cells_per_partition() -> None:
    dataframe = pd.DataFrame(
        [
            {"victim_model": "miasrec", "attack_method": "clean", "recall | 10 | ground_truth": 0.11},
            {"victim_model": "miasrec", "attack_method": "dpsbr_baseline", "recall | 10 | ground_truth": 0.14},
            {
                "victim_model": "miasrec",
                "attack_method": "position_opt_mvp",
                "recall | 10 | ground_truth": 0.13,
            },
            {"victim_model": "tron", "attack_method": "clean", "recall | 10 | ground_truth": 0.30},
            {"victim_model": "tron", "attack_method": "dpsbr_baseline", "recall | 10 | ground_truth": 0.30},
            {"victim_model": "tron", "attack_method": "position_opt_mvp", "recall | 10 | ground_truth": 0.25},
        ]
    )
    table_structure = TableStructure(
        row_levels=["victim_model", "attack_method"],
        col_levels=["metric_name", "k", "metric_scope"],
        row_tuples=[
            ("miasrec", "clean"),
            ("miasrec", "dpsbr_baseline"),
            ("miasrec", "position_opt_mvp"),
            ("tron", "clean"),
            ("tron", "dpsbr_baseline"),
            ("tron", "position_opt_mvp"),
        ],
        column_tuples=[("recall", 10, "ground_truth")],
        row_column_names=["victim_model", "attack_method"],
        value_column_names=["recall | 10 | ground_truth"],
    )

    highlights = resolve_ranked_value_highlights(
        dataframe=dataframe,
        table_structure=table_structure,
        best_value_bolding=BestValueBoldingSpec(
            compare_along="rows",
            mode="max",
            partition_by_levels=["victim_model"],
            underline_second_best=True,
        ),
    )

    assert highlights.best_value_cells == {(1, 0), (3, 0), (4, 0)}
    assert highlights.second_best_value_cells == {(2, 0), (5, 0)}


def test_draw_cell_block_supports_underlined_text() -> None:
    render_spec = parse_render_spec(
        {
            "style_name": "phase8_underline_draw",
            "output_format": "png",
            "title": {
                "template": "test",
                "align": "center",
                "font_size": 12,
                "color": "black",
            },
            "figure": {
                "width": 8,
                "height": 4,
                "dpi": 100,
                "background_color": "white",
            },
            "table": {
                "font_size": 10,
                "round_digits": 4,
                "text_color": "black",
                "show_grid": True,
                "auto_shrink": False,
                "wrap_text": False,
                "cell_align": "center",
                "display_alias": {},
                "value_alias": {},
                "dimension_value_orders": {},
                "scope_colors": {},
                "top_level_group_separators": False,
            },
        }
    )
    fig, ax = plt.subplots()
    try:
        draw_cell_block(
            ax=ax,
            x0=0.0,
            x1=1.0,
            y0=0.0,
            y1=1.0,
            text="0.13",
            font_weight="normal",
            underline_text=True,
            facecolor=None,
            render_spec=render_spec,
            total_table_width=1.0,
            total_row_count=1,
        )
        assert len(ax.texts) == 1
        assert len(ax.lines) == 1
        line = ax.lines[0]
        x_data = list(line.get_xdata())
        y_data = list(line.get_ydata())
        assert x_data[0] < x_data[1]
        assert y_data[0] == pytest.approx(y_data[1])
    finally:
        plt.close(fig)


def test_renderer_can_reorder_attack_method_rows_from_render_config() -> None:
    render_spec = parse_render_spec(
        {
            "style_name": "phase8_render_order",
            "output_format": "png",
            "title": {
                "template": "test",
                "align": "center",
                "font_size": 12,
                "color": "black",
            },
            "figure": {
                "width": 8,
                "height": 4,
                "dpi": 100,
                "background_color": "white",
            },
            "table": {
                "font_size": 10,
                "round_digits": 4,
                "text_color": "black",
                "show_grid": True,
                "auto_shrink": False,
                "wrap_text": False,
                "cell_align": "center",
                "display_alias": {},
                "value_alias": {},
                "dimension_value_orders": {
                    "attack_method": [
                        "clean",
                        "dpsbr_baseline",
                        "prefix_nonzero_when_possible",
                        "random_nonzero_when_possible",
                        "position_opt_mvp",
                    ]
                },
                "scope_colors": {},
                "top_level_group_separators": False,
            },
        }
    )
    dataframe = pd.DataFrame(
        [
            {
                "victim_model": "miasrec",
                "attack_method": "clean",
                "recall | 10 | ground_truth": 0.11,
            },
            {
                "victim_model": "miasrec",
                "attack_method": "dpsbr_baseline",
                "recall | 10 | ground_truth": 0.12,
            },
            {
                "victim_model": "miasrec",
                "attack_method": "position_opt_mvp",
                "recall | 10 | ground_truth": 0.13,
            },
            {
                "victim_model": "miasrec",
                "attack_method": "prefix_nonzero_when_possible",
                "recall | 10 | ground_truth": 0.14,
            },
            {
                "victim_model": "miasrec",
                "attack_method": "random_nonzero_when_possible",
                "recall | 10 | ground_truth": 0.15,
            },
            {
                "victim_model": "tron",
                "attack_method": "clean",
                "recall | 10 | ground_truth": 0.21,
            },
            {
                "victim_model": "tron",
                "attack_method": "dpsbr_baseline",
                "recall | 10 | ground_truth": 0.22,
            },
            {
                "victim_model": "tron",
                "attack_method": "position_opt_mvp",
                "recall | 10 | ground_truth": 0.23,
            },
            {
                "victim_model": "tron",
                "attack_method": "prefix_nonzero_when_possible",
                "recall | 10 | ground_truth": 0.24,
            },
            {
                "victim_model": "tron",
                "attack_method": "random_nonzero_when_possible",
                "recall | 10 | ground_truth": 0.25,
            },
        ]
    )
    table_structure = TableStructure(
        row_levels=["victim_model", "attack_method"],
        col_levels=["metric_name", "k", "metric_scope"],
        row_tuples=[
            ("miasrec", "clean"),
            ("miasrec", "dpsbr_baseline"),
            ("miasrec", "position_opt_mvp"),
            ("miasrec", "prefix_nonzero_when_possible"),
            ("miasrec", "random_nonzero_when_possible"),
            ("tron", "clean"),
            ("tron", "dpsbr_baseline"),
            ("tron", "position_opt_mvp"),
            ("tron", "prefix_nonzero_when_possible"),
            ("tron", "random_nonzero_when_possible"),
        ],
        column_tuples=[("recall", 10, "ground_truth")],
        row_column_names=["victim_model", "attack_method"],
        value_column_names=["recall | 10 | ground_truth"],
    )

    ordered_dataframe, ordered_table_structure = apply_dimension_value_orders(
        dataframe=dataframe,
        table_structure=table_structure,
        dimension_value_orders=render_spec.table.dimension_value_orders,
    )

    assert ordered_table_structure.row_tuples == [
        ("miasrec", "clean"),
        ("miasrec", "dpsbr_baseline"),
        ("miasrec", "prefix_nonzero_when_possible"),
        ("miasrec", "random_nonzero_when_possible"),
        ("miasrec", "position_opt_mvp"),
        ("tron", "clean"),
        ("tron", "dpsbr_baseline"),
        ("tron", "prefix_nonzero_when_possible"),
        ("tron", "random_nonzero_when_possible"),
        ("tron", "position_opt_mvp"),
    ]
    assert list(
        zip(
            ordered_dataframe["victim_model"].tolist(),
            ordered_dataframe["attack_method"].tolist(),
            ordered_dataframe["recall | 10 | ground_truth"].tolist(),
            strict=True,
        )
    ) == [
        ("miasrec", "clean", 0.11),
        ("miasrec", "dpsbr_baseline", 0.12),
        ("miasrec", "prefix_nonzero_when_possible", 0.14),
        ("miasrec", "random_nonzero_when_possible", 0.15),
        ("miasrec", "position_opt_mvp", 0.13),
        ("tron", "clean", 0.21),
        ("tron", "dpsbr_baseline", 0.22),
        ("tron", "prefix_nonzero_when_possible", 0.24),
        ("tron", "random_nonzero_when_possible", 0.25),
        ("tron", "position_opt_mvp", 0.23),
    ]
