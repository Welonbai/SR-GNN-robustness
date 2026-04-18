from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from analysis.pipeline.long_csv_generator import generate_long_table_bundle, load_json_mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RESULTS_ROOT = REPO_ROOT / "results"


@contextmanager
def _phase7_temp_roots():
    token = uuid4().hex
    outputs_root = OUTPUTS_ROOT / ".pytest_phase7" / token
    results_root = RESULTS_ROOT / "runs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    try:
        yield outputs_root, token
    finally:
        shutil.rmtree(outputs_root, ignore_errors=True)
        for bundle_dir in results_root.glob(f"phase7_{token}_*"):
            shutil.rmtree(bundle_dir, ignore_errors=True)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _resolved_config_payload() -> dict[str, object]:
    return {
        "result_config": {
            "data": {"dataset_name": "diginetica"},
            "targets": {"bucket": "popular"},
            "attack": {
                "size": 0.1,
                "poison_model": {"name": "heuristic"},
                "fake_session_generation_topk": 20,
                "replacement_topk_ratio": 0.5,
            },
        },
        "derived": {
            "run_type": "clean",
        },
    }


def _summary_target_payload(target_item: int | str, victims: dict[str, float]) -> dict[str, object]:
    return {
        "target_item": target_item,
        "victims": {
            victim_name: {
                "metrics": {"targeted_recall@10": metric_value},
                "metrics_available": True,
                "predictions_path": f"mock/{target_item}/{victim_name}/predictions.json",
            }
            for victim_name, metric_value in victims.items()
        },
    }


def _write_run_group_artifacts(
    outputs_root: Path,
    *,
    target_order: list[int],
    current_count: int,
    coverage_statuses: dict[int, dict[str, str]],
    summary_metrics: dict[int, dict[str, float]],
    victims: list[str],
) -> Path:
    run_group_key = "run_group_phase7_test"
    target_cohort_key = "target_cohort_phase7_test"
    run_root = outputs_root / "runs" / "diginetica" / "analysis_phase7" / run_group_key
    target_registry_path = (
        outputs_root
        / "shared"
        / "target_cohorts"
        / target_cohort_key
        / "target_registry.json"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    target_registry_path.parent.mkdir(parents=True, exist_ok=True)

    target_registry_path.write_text(
        __import__("json").dumps(
            {
                "target_cohort_key": target_cohort_key,
                "split_key": "split_test",
                "selection_policy_version": "appendable_target_cohort_v1",
                "mode": "sampled",
                "bucket": "popular",
                "seed": 123,
                "explicit_list": None,
                "candidate_pool_hash": "hash",
                "candidate_pool_size": len(target_order),
                "ordered_targets": target_order,
                "current_count": current_count,
                "created_at": "2026-04-19T00:00:00+00:00",
                "updated_at": "2026-04-19T00:00:00+00:00",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    run_coverage_payload = {
        "run_group_key": run_group_key,
        "target_cohort_key": target_cohort_key,
        "split_key": "split_test",
        "run_type": "clean",
        "targets_order": target_order[:current_count],
        "victims": {
            victim_name: {
                "status": "requested",
                "first_requested_at": "2026-04-19T00:00:00+00:00",
                "last_requested_at": "2026-04-19T00:00:00+00:00",
                "victim_prediction_key": f"victim_{victim_name}_test",
            }
            for victim_name in victims
        },
        "cells": {
            str(target_item): {
                victim_name: {
                    "status": coverage_statuses[target_item][victim_name],
                    "artifacts": {
                        "metrics": None,
                        "predictions": None,
                        "train_history": None,
                        "poisoned_train": None,
                    },
                    "error": None,
                    "first_requested_at": "2026-04-19T00:00:00+00:00",
                    "last_requested_at": "2026-04-19T00:00:00+00:00",
                    "last_started_at": None,
                    "last_execution_id": None,
                    "attempt_count": 0,
                    "completed_at": None,
                    "failed_at": None,
                    "last_updated_at": "2026-04-19T00:00:00+00:00",
                }
                for victim_name in victims
            }
            for target_item in target_order[:current_count]
        },
        "created_at": "2026-04-19T00:00:00+00:00",
        "updated_at": "2026-04-19T00:00:00+00:00",
    }
    (run_root / "run_coverage.json").write_text(
        __import__("json").dumps(run_coverage_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_payload = {
        "run_type": "clean",
        "run_group_key": run_group_key,
        "target_cohort_key": target_cohort_key,
        "is_snapshot": True,
        "snapshot_source": "run_coverage_and_cell_artifacts",
        "target_items": target_order[:current_count],
        "victims": victims,
        "targets": {
            str(target_item): _summary_target_payload(target_item, summary_metrics[target_item])
            for target_item in summary_metrics
        },
        "created_at": "2026-04-19T00:00:00+00:00",
        "updated_at": "2026-04-19T00:00:00+00:00",
    }
    (run_root / "summary_current.json").write_text(
        __import__("json").dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_root / "resolved_config.json").write_text(
        __import__("json").dumps(_resolved_config_payload(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_root / "artifact_manifest.json").write_text(
        __import__("json").dumps(
            {
                "shared_artifacts": {
                    "target_cohort": {
                        "target_registry": _repo_relative(target_registry_path),
                    }
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run_root / "summary_current.json"


def test_largest_complete_prefix_uses_coverage_and_not_summary_snapshot() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[11, 22, 33, 44],
            current_count=4,
            victims=["miasrec", "tron"],
            coverage_statuses={
                11: {"miasrec": "completed", "tron": "completed"},
                22: {"miasrec": "completed", "tron": "completed"},
                33: {"miasrec": "completed", "tron": "failed"},
                44: {"miasrec": "completed", "tron": "completed"},
            },
            summary_metrics={
                11: {"miasrec": 0.11, "tron": 0.12},
                22: {"miasrec": 0.21, "tron": 0.22},
                33: {"miasrec": 0.31, "tron": 0.32},
                44: {"miasrec": 0.41, "tron": 0.42},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_largest_prefix",
            slice_policy="largest_complete_prefix",
            requested_victims=None,
            requested_target_count=None,
        )

        dataframe = pd.read_csv(result["long_table_path"])
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")

    assert dataframe["target_item"].drop_duplicates().tolist() == [11, 22]
    assert sorted(dataframe["victim_model"].unique().tolist()) == ["miasrec", "tron"]
    assert slice_manifest["selected_targets"] == [11, 22]
    assert slice_manifest["excluded_targets"] == [33, 44]
    assert slice_manifest["fairness_safe"] is True
    assert {"target_item": 33, "victim_name": "tron", "status": "failed"} in slice_manifest[
        "excluded_incomplete_cells"
    ]


def test_largest_complete_prefix_preserves_target_registry_order() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[33, 11, 22],
            current_count=3,
            victims=["miasrec", "tron"],
            coverage_statuses={
                33: {"miasrec": "completed", "tron": "completed"},
                11: {"miasrec": "completed", "tron": "completed"},
                22: {"miasrec": "completed", "tron": "pending"},
            },
            summary_metrics={
                11: {"miasrec": 0.11, "tron": 0.12},
                22: {"miasrec": 0.21, "tron": 0.22},
                33: {"miasrec": 0.31, "tron": 0.32},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_registry_order",
            slice_policy="largest_complete_prefix",
            requested_victims=None,
            requested_target_count=None,
        )
        dataframe = pd.read_csv(result["long_table_path"])

    assert dataframe["target_item"].drop_duplicates().tolist() == [33, 11]


def test_intersection_complete_selects_completed_targets_in_registry_order() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[40, 10, 30],
            current_count=3,
            victims=["miasrec", "tron"],
            coverage_statuses={
                40: {"miasrec": "completed", "tron": "completed"},
                10: {"miasrec": "completed", "tron": "failed"},
                30: {"miasrec": "completed", "tron": "completed"},
            },
            summary_metrics={
                40: {"miasrec": 0.40, "tron": 0.41},
                10: {"miasrec": 0.10, "tron": 0.11},
                30: {"miasrec": 0.30, "tron": 0.31},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_intersection",
            slice_policy="intersection_complete",
            requested_victims=None,
            requested_target_count=None,
        )
        dataframe = pd.read_csv(result["long_table_path"])
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")

    assert dataframe["target_item"].drop_duplicates().tolist() == [40, 30]
    assert slice_manifest["selected_targets"] == [40, 30]
    assert slice_manifest["excluded_targets"] == [10]


def test_all_available_includes_partial_completed_rows_and_is_not_fairness_safe() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[1, 2, 3],
            current_count=3,
            victims=["miasrec", "tron"],
            coverage_statuses={
                1: {"miasrec": "completed", "tron": "completed"},
                2: {"miasrec": "completed", "tron": "pending"},
                3: {"miasrec": "failed", "tron": "completed"},
            },
            summary_metrics={
                1: {"miasrec": 0.11, "tron": 0.12},
                2: {"miasrec": 0.21, "tron": 0.22},
                3: {"miasrec": 0.31, "tron": 0.32},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_all_available",
            slice_policy="all_available",
            requested_victims=None,
            requested_target_count=None,
        )
        dataframe = pd.read_csv(result["long_table_path"])
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")

    assert dataframe["target_item"].drop_duplicates().tolist() == [1, 2, 3]
    assert list(zip(dataframe["target_item"], dataframe["victim_model"])) == [
        (1, "miasrec"),
        (1, "tron"),
        (2, "miasrec"),
        (3, "tron"),
    ]
    assert slice_manifest["fairness_safe"] is False


def test_requested_victim_subset_changes_slice_membership() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[101, 202, 303],
            current_count=3,
            victims=["miasrec", "tron"],
            coverage_statuses={
                101: {"miasrec": "completed", "tron": "completed"},
                202: {"miasrec": "completed", "tron": "failed"},
                303: {"miasrec": "completed", "tron": "pending"},
            },
            summary_metrics={
                101: {"miasrec": 0.11, "tron": 0.12},
                202: {"miasrec": 0.21, "tron": 0.22},
                303: {"miasrec": 0.31, "tron": 0.32},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_victim_subset",
            slice_policy="largest_complete_prefix",
            requested_victims=["miasrec"],
            requested_target_count=None,
        )
        dataframe = pd.read_csv(result["long_table_path"])
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")

    assert dataframe["target_item"].drop_duplicates().tolist() == [101, 202, 303]
    assert dataframe["victim_model"].unique().tolist() == ["miasrec"]
    assert slice_manifest["requested_victims"] == ["miasrec"]
    assert slice_manifest["requested_victims_source"] == "cli"


def test_omitted_slice_policy_defaults_to_largest_complete_prefix() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[5, 6, 7],
            current_count=3,
            victims=["miasrec", "tron"],
            coverage_statuses={
                5: {"miasrec": "completed", "tron": "completed"},
                6: {"miasrec": "completed", "tron": "failed"},
                7: {"miasrec": "completed", "tron": "completed"},
            },
            summary_metrics={
                5: {"miasrec": 0.11, "tron": 0.12},
                6: {"miasrec": 0.21, "tron": 0.22},
                7: {"miasrec": 0.31, "tron": 0.32},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_default_policy",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")

    assert slice_manifest["slice_policy"] == "largest_complete_prefix"
    assert slice_manifest["selected_targets"] == [5]


def test_slice_manifest_and_output_manifest_record_required_metadata() -> None:
    with _phase7_temp_roots() as (outputs_root, token):
        summary_path = _write_run_group_artifacts(
            outputs_root,
            target_order=[9, 8],
            current_count=2,
            victims=["miasrec", "tron"],
            coverage_statuses={
                9: {"miasrec": "completed", "tron": "completed"},
                8: {"miasrec": "completed", "tron": "failed"},
            },
            summary_metrics={
                9: {"miasrec": 0.11, "tron": 0.12},
                8: {"miasrec": 0.21, "tron": 0.22},
            },
        )
        result = generate_long_table_bundle(
            summary_path=summary_path,
            output_name=f"phase7_{token}_manifest",
            slice_policy="largest_complete_prefix",
            requested_victims=None,
            requested_target_count=2,
        )
        slice_manifest = load_json_mapping(result["slice_manifest_path"], label="slice_manifest JSON")
        manifest = load_json_mapping(result["manifest_path"], label="manifest JSON")

    assert slice_manifest["source_summary_current_path"].endswith("summary_current.json")
    assert slice_manifest["source_run_coverage_path"].endswith("run_coverage.json")
    assert slice_manifest["source_target_registry_path"].endswith("target_registry.json")
    assert slice_manifest["selected_target_count"] == 1
    assert manifest["generated_files"] == [
        "inventory.json",
        "long_table.csv",
        "manifest.json",
        "slice_manifest.json",
        "source_resolved_config.json",
    ]
    assert manifest["slice"]["slice_policy"] == "largest_complete_prefix"
    assert manifest["slice"]["selected_target_count"] == 1
