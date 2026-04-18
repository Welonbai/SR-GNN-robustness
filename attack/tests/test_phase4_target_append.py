from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from attack.common.artifact_io import (
    load_execution_log,
    load_json,
    load_run_coverage,
    load_summary_current,
    load_target_registry,
    save_json,
)
from attack.common.config import load_config
from attack.common.paths import run_metadata_paths, shared_artifact_paths
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.orchestrator import RunContext, TargetPoisonOutput, run_targets_and_victims
from attack.pipeline.core.pipeline_utils import build_ordered_target_cohort
from attack.pipeline.core.victim_execution import VictimExecutionResult


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 4 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase4_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase4" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _sample_stats():
    return compute_session_stats(
        [
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
            [40, 50, 60],
            [50, 60, 70],
        ]
    )


def _config_for_temp_root(
    temp_root: Path,
    *,
    count: int,
    victims: tuple[str, ...],
):
    base = _base_config()
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=replace(base.targets, mode="sampled", bucket="all", count=count),
        victims=replace(base.victims, enabled=victims),
    )


def _minimal_context(config, *, run_type: str) -> RunContext:
    return RunContext(
        canonical_dataset=object(),
        stats=_sample_stats(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        export_paths={},
        shared_paths=shared_artifact_paths(config, run_type=run_type),
        fake_session_count=0,
    )


def _expected_target_prefix(config) -> list[int]:
    cohort = build_ordered_target_cohort(_sample_stats(), config)
    return [int(item) for item in cohort["ordered_targets"][: int(config.targets.count)]]


def _install_fake_execution(
    monkeypatch,
    *,
    calls: list[tuple[int, str]],
    fail_cells: set[tuple[int, str]] | None = None,
    omit_prediction_cells: set[tuple[int, str]] | None = None,
) -> None:
    fail_cells = fail_cells or set()
    omit_prediction_cells = omit_prediction_cells or set()

    def fake_victim_execution(*args, **kwargs):
        target_item = int(kwargs["target_item"])
        victim_name = str(kwargs["victim_name"])
        calls.append((target_item, victim_name))
        save_json(
            {"victim_name": victim_name, "phase4_fake": True},
            kwargs["artifacts"]["resolved_config"],
        )
        cell_key = (target_item, victim_name)
        if cell_key in fail_cells:
            raise RuntimeError(f"phase4 forced failure for {target_item}:{victim_name}")
        if cell_key not in omit_prediction_cells:
            save_json(
                {"rankings": [[target_item, 1, 2, 3]], "victim": victim_name},
                kwargs["predictions_path"],
            )
        return (
            VictimExecutionResult(
                predictions=[[target_item, 1, 2, 3]],
                predictions_path=kwargs["predictions_path"],
                extra={"phase4_fake": True},
                poisoned_train_path=None,
            ),
            False,
        )

    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator._maybe_reuse_or_execute_victim",
        fake_victim_execution,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.resolve_ground_truth_labels",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.evaluate_prediction_metrics",
        lambda *args, **kwargs: ({"targeted_recall@10": 1.0}, True),
    )


def _build_poisoned(target_item: int) -> TargetPoisonOutput:
    return TargetPoisonOutput(
        poisoned=PoisonedDataset(
            sessions=[[1, 2]],
            labels=[int(target_item)],
            clean_count=1,
            fake_count=0,
        ),
        metadata={"phase4_smoke": True},
    )


def test_first_invocation_executes_full_requested_prefix_and_updates_state(monkeypatch) -> None:
    with _phase4_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=3, victims=("miasrec", "tron"))
        context = _minimal_context(config, run_type="clean")
        calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=calls)

        summary = run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        target_registry = load_target_registry(context.shared_paths["target_registry"])
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])
        summary_current = load_summary_current(metadata_paths["summary_current"])
        progress_payload = load_json(metadata_paths["progress"])

    assert target_registry is not None
    expected_targets = target_registry["ordered_targets"][:3]
    expected_calls = [
        (int(target_item), victim_name)
        for target_item in expected_targets
        for victim_name in ("miasrec", "tron")
    ]
    assert calls == expected_calls
    assert summary["target_items"] == [int(item) for item in expected_targets]

    assert run_coverage is not None
    assert run_coverage["targets_order"] == [int(item) for item in expected_targets]
    for target_item in expected_targets:
        for victim_name in ("miasrec", "tron"):
            assert run_coverage["cells"][str(target_item)][victim_name]["status"] == "completed"

    assert execution_log is not None
    assert len(execution_log["executions"]) == 1
    record = execution_log["executions"][0]
    assert record["status"] == "completed"
    assert len(record["planned_cells"]) == 6
    assert len(record["completed_cells"]) == 6
    assert record["failed_cells"] == []

    assert summary_current is not None
    assert set(summary_current["targets"]) == {str(item) for item in expected_targets}
    assert progress_payload["is_authoritative"] is False
    assert progress_payload["authoritative_state"]["run_coverage"].endswith("run_coverage.json")
    assert progress_payload["authoritative_state"]["execution_log"].endswith("execution_log.json")


def test_larger_prefix_only_executes_new_targets_and_rerun_executes_nothing_new(
    monkeypatch,
) -> None:
    with _phase4_temp_root() as temp_root:
        first_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        larger_config = _config_for_temp_root(temp_root, count=4, victims=("miasrec",))
        context = _minimal_context(first_config, run_type="clean")

        first_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=first_calls)
        run_targets_and_victims(
            first_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        expected_larger_targets = _expected_target_prefix(larger_config)
        appended_targets = expected_larger_targets[2:4]
        second_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=second_calls)
        run_targets_and_victims(
            larger_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        third_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=third_calls)
        summary = run_targets_and_victims(
            larger_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(larger_config, run_type="clean")
        execution_log = load_execution_log(metadata_paths["execution_log"])
        summary_current = load_summary_current(metadata_paths["summary_current"])

    assert len(first_calls) == 2
    assert second_calls == [(int(target_item), "miasrec") for target_item in appended_targets]
    assert third_calls == []
    assert summary["target_items"] == [int(item) for item in expected_larger_targets]

    assert execution_log is not None
    assert len(execution_log["executions"]) == 3
    second_record = execution_log["executions"][1]
    third_record = execution_log["executions"][2]
    assert [cell["target_item"] for cell in second_record["planned_cells"]] == [
        int(item) for item in appended_targets
    ]
    assert len(second_record["skipped_completed_cells"]) == 2
    assert third_record["planned_cells"] == []
    assert len(third_record["skipped_completed_cells"]) == 4
    assert third_record["completed_cells"] == []

    assert summary_current is not None
    assert set(summary_current["targets"]) == {str(item) for item in expected_larger_targets}


def test_victim_set_change_is_rejected_until_phase5(monkeypatch) -> None:
    with _phase4_temp_root() as temp_root:
        first_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        second_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec", "tron"))
        context = _minimal_context(first_config, run_type="clean")
        calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=calls)

        run_targets_and_victims(
            first_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                second_config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

    assert "Victim-append or victim-set changes are not implemented until Phase 5" in str(
        exc_info.value
    )


def test_failed_cells_remain_eligible_and_rerun_only_retries_failures(monkeypatch) -> None:
    with _phase4_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")
        target_prefix = _expected_target_prefix(config)
        failed_target = int(target_prefix[1])

        first_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=first_calls,
            fail_cells={(failed_target, "miasrec")},
        )
        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        first_coverage = load_run_coverage(metadata_paths["run_coverage"])
        first_summary_current = load_summary_current(metadata_paths["summary_current"])

        second_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=second_calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )
        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])

    assert "phase4 forced failure" in str(exc_info.value)
    assert first_calls == [(int(target_prefix[0]), "miasrec"), (failed_target, "miasrec")]
    assert first_coverage is not None
    assert first_coverage["cells"][str(target_prefix[0])]["miasrec"]["status"] == "completed"
    assert first_coverage["cells"][str(failed_target)]["miasrec"]["status"] == "failed"
    assert first_summary_current is not None
    assert set(first_summary_current["targets"]) == {str(target_prefix[0])}

    assert second_calls == [(failed_target, "miasrec")]
    assert final_coverage is not None
    assert final_coverage["cells"][str(failed_target)]["miasrec"]["status"] == "completed"
    assert execution_log is not None
    assert len(execution_log["executions"]) == 2
    assert execution_log["executions"][0]["status"] == "failed"
    assert execution_log["executions"][1]["status"] == "completed"
    assert len(execution_log["executions"][1]["planned_cells"]) == 1
    assert len(execution_log["executions"][1]["skipped_completed_cells"]) == 1


def test_missing_required_local_artifact_prevents_completed_marking(monkeypatch) -> None:
    with _phase4_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=1, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")
        target_item = int(_expected_target_prefix(config)[0])
        calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=calls,
            omit_prediction_cells={(target_item, "miasrec")},
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])

    assert "Cell completion requires persisted local artifacts" in str(exc_info.value)
    assert calls == [(target_item, "miasrec")]
    assert run_coverage is not None
    cell = run_coverage["cells"][str(target_item)]["miasrec"]
    assert cell["status"] == "failed"
    assert cell["artifacts"]["metrics"] is not None
    assert cell["artifacts"]["predictions"] is None
    assert execution_log is not None
    assert execution_log["executions"][0]["status"] == "failed"
    assert len(execution_log["executions"][0]["failed_cells"]) == 1
