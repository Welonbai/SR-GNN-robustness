from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

from attack.common.artifact_io import (
    load_execution_log,
    load_run_coverage,
    load_selected_targets,
    load_summary_current,
    load_target_info,
    load_target_registry,
    load_target_selection_meta,
    save_json,
    save_selected_targets,
    save_target_info,
    save_target_selection_meta,
)
from attack.common.config import load_config
from attack.common.paths import run_metadata_paths, shared_artifact_paths, target_cohort_key
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.pipeline_utils import (
    build_ordered_target_cohort,
    ensure_target_registry_prefix,
    load_or_init_execution_log,
    load_or_init_run_coverage,
    load_or_init_target_registry,
    rebuild_summary_current,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 2 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


def _all_bucket_config(
    temp_root: Path,
    *,
    count: int,
    reuse_saved_targets: bool = True,
):
    base = _base_config()
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=replace(
            base.targets,
            mode="sampled",
            bucket="all",
            count=count,
            reuse_saved_targets=reuse_saved_targets,
        ),
    )


def _sample_stats():
    sessions = [
        [10, 20, 30],
        [20, 30, 40],
        [30, 40, 50],
        [40, 50, 60],
        [50, 60, 70],
        [60, 70, 80],
    ]
    return compute_session_stats(sessions)


@contextmanager
def _phase2_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase2" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def test_deterministic_sampled_cohort_construction() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        config = _all_bucket_config(temp_root, count=3)
        first = build_ordered_target_cohort(stats, config)
        second = build_ordered_target_cohort(stats, config)

    assert first["ordered_targets"] == second["ordered_targets"]
    assert first["candidate_pool_hash"] == second["candidate_pool_hash"]
    assert first["candidate_pool_size"] == len(first["ordered_targets"])


def test_same_sampled_cohort_identity_uses_same_ordered_target_basis_across_counts() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        small = _all_bucket_config(temp_root, count=2)
        large = _all_bucket_config(temp_root, count=5)

        small_cohort = build_ordered_target_cohort(stats, small)
        large_cohort = build_ordered_target_cohort(stats, large)

    assert target_cohort_key(small) == target_cohort_key(large)
    assert small_cohort["ordered_targets"] == large_cohort["ordered_targets"]
    assert large_cohort["ordered_targets"][: small.targets.count] == small_cohort["ordered_targets"][
        : small.targets.count
    ]


def test_increasing_requested_prefix_updates_current_count_without_reordering_targets() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        small = _all_bucket_config(temp_root, count=2)
        shared_paths = shared_artifact_paths(small, run_type="clean")
        initial_registry = load_or_init_target_registry(stats, small, shared_paths=shared_paths)
        expanded_config = _all_bucket_config(temp_root, count=5)
        expanded_registry = ensure_target_registry_prefix(
            stats,
            expanded_config,
            shared_paths=shared_paths,
            target_registry=initial_registry,
        )
        persisted_registry = load_target_registry(shared_paths["target_registry"])

    assert initial_registry["current_count"] == 2
    assert expanded_registry["current_count"] == 5
    assert expanded_registry["ordered_targets"] == initial_registry["ordered_targets"]
    assert persisted_registry is not None
    assert persisted_registry["current_count"] == 5
    assert persisted_registry["ordered_targets"] == initial_registry["ordered_targets"]


def test_reuse_saved_targets_flag_does_not_affect_new_target_registry_behavior() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        reuse_config = _all_bucket_config(temp_root, count=4, reuse_saved_targets=True)
        no_reuse_config = _all_bucket_config(temp_root, count=4, reuse_saved_targets=False)
        shared_paths = shared_artifact_paths(reuse_config, run_type="clean")

        reuse_registry = load_or_init_target_registry(stats, reuse_config, shared_paths=shared_paths)
        no_reuse_registry = load_or_init_target_registry(
            stats,
            no_reuse_config,
            shared_paths=shared_paths,
        )

    assert target_cohort_key(reuse_config) == target_cohort_key(no_reuse_config)
    assert reuse_registry["ordered_targets"] == no_reuse_registry["ordered_targets"]
    assert reuse_registry["current_count"] == no_reuse_registry["current_count"]


def test_run_coverage_initialization_shape() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        config = _all_bucket_config(temp_root, count=3)
        shared_paths = shared_artifact_paths(config, run_type="clean")
        registry = ensure_target_registry_prefix(stats, config, shared_paths=shared_paths)
        metadata_paths = run_metadata_paths(config, run_type="clean")
        coverage = load_or_init_run_coverage(
            config,
            run_type="clean",
            metadata_paths=metadata_paths,
            target_registry=registry,
        )
        persisted_coverage = load_run_coverage(metadata_paths["run_coverage"])

    assert coverage["run_group_key"] == persisted_coverage["run_group_key"]
    assert coverage["target_cohort_key"] == registry["target_cohort_key"]
    assert coverage["targets_order"] == registry["ordered_targets"][: registry["current_count"]]
    assert set(coverage["victims"]) == set(config.victims.enabled)
    for target_item in coverage["targets_order"]:
        cell_row = coverage["cells"][str(target_item)]
        for victim_name in config.victims.enabled:
            assert cell_row[victim_name]["status"] == "pending"
            assert cell_row[victim_name]["artifacts"]["metrics"] is None
            assert cell_row[victim_name]["artifacts"]["predictions"] is None


def test_execution_log_initialization_shape() -> None:
    with _phase2_temp_root() as temp_root:
        config = _all_bucket_config(temp_root, count=3)
        metadata_paths = run_metadata_paths(config, run_type="clean")
        execution_log = load_or_init_execution_log(
            config,
            run_type="clean",
            metadata_paths=metadata_paths,
        )
        persisted_execution_log = load_execution_log(metadata_paths["execution_log"])

    assert execution_log["run_group_key"] == persisted_execution_log["run_group_key"]
    assert execution_log["target_cohort_key"] == target_cohort_key(config)
    assert execution_log["executions"] == []


def test_summary_current_rebuilds_from_completed_cells() -> None:
    stats = _sample_stats()
    with _phase2_temp_root() as temp_root:
        config = _all_bucket_config(temp_root, count=2)
        shared_paths = shared_artifact_paths(config, run_type="clean")
        registry = ensure_target_registry_prefix(stats, config, shared_paths=shared_paths)
        metadata_paths = run_metadata_paths(config, run_type="clean")
        coverage = load_or_init_run_coverage(
            config,
            run_type="clean",
            metadata_paths=metadata_paths,
            target_registry=registry,
        )

        target_item = coverage["targets_order"][0]
        metrics_path = (
            metadata_paths["run_root"]
            / "targets"
            / str(target_item)
            / "victims"
            / "srgnn"
            / "metrics.json"
        )
        predictions_path = metrics_path.with_name("predictions.json")
        save_json(
            {
                "metrics": {
                    "targeted_recall@10": 0.5,
                    "targeted_mrr@10": 0.25,
                },
                "metrics_available": True,
                "predictions_path": _repo_relative(predictions_path),
            },
            metrics_path,
        )
        save_json({"available": True, "rankings": [[1, 2, 3]]}, predictions_path)

        coverage["cells"][str(target_item)]["srgnn"]["status"] = "completed"
        coverage["cells"][str(target_item)]["srgnn"]["artifacts"]["metrics"] = _repo_relative(
            metrics_path
        )
        coverage["cells"][str(target_item)]["srgnn"]["artifacts"]["predictions"] = _repo_relative(
            predictions_path
        )

        summary_current = rebuild_summary_current(
            config,
            run_type="clean",
            metadata_paths=metadata_paths,
            run_coverage=coverage,
        )
        persisted_summary = load_summary_current(metadata_paths["summary_current"])

    assert summary_current["run_group_key"] == persisted_summary["run_group_key"]
    assert summary_current["targets"][str(target_item)]["target_item"] == int(target_item)
    assert summary_current["targets"][str(target_item)]["victims"]["srgnn"]["metrics"][
        "targeted_recall@10"
    ] == 0.5
    assert "miasrec" not in summary_current["targets"][str(target_item)]["victims"]


def test_legacy_selected_target_helpers_remain_available_but_are_not_the_primary_registry_path() -> None:
    with _phase2_temp_root() as temp_root:
        config = _all_bucket_config(temp_root, count=3)
        shared_paths = shared_artifact_paths(config, run_type="clean")

        save_selected_targets(shared_paths["selected_targets"], [10, 20, 30])
        save_target_selection_meta(shared_paths["target_selection_meta"], {"legacy": True})
        save_target_info(
            shared_paths["target_info"],
            target_items=[10, 20, 30],
            target_selection_mode="sampled",
            seed=123,
            bucket="all",
            count=3,
        )

        stats = _sample_stats()
        registry = load_or_init_target_registry(stats, config, shared_paths=shared_paths)
        persisted_registry = load_target_registry(shared_paths["target_registry"])

        loaded_selected_targets = load_selected_targets(shared_paths["selected_targets"])
        loaded_target_selection_meta = load_target_selection_meta(shared_paths["target_selection_meta"])
        loaded_target_info = load_target_info(shared_paths["target_info"])

    assert loaded_selected_targets == [10, 20, 30]
    assert loaded_target_selection_meta == {"legacy": True}
    assert loaded_target_info is not None
    assert shared_paths["target_registry"] != shared_paths["selected_targets"]
    assert persisted_registry is not None
    assert registry["target_cohort_key"].startswith("target_cohort_")
