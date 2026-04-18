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
    save_execution_log,
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
    assert CONFIG_PATH.is_file(), f"Missing Phase 6 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase6_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase6" / uuid4().hex
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
    exception_by_call_index: dict[int, BaseException] | None = None,
    exception_by_cell: dict[tuple[int, str], BaseException] | None = None,
    omit_prediction_cells: set[tuple[int, str]] | None = None,
) -> None:
    exception_by_call_index = exception_by_call_index or {}
    exception_by_cell = exception_by_cell or {}
    omit_prediction_cells = omit_prediction_cells or set()

    def fake_victim_execution(*args, **kwargs):
        target_item = int(kwargs["target_item"])
        victim_name = str(kwargs["victim_name"])
        calls.append((target_item, victim_name))
        save_json(
            {"victim_name": victim_name, "phase6_fake": True},
            kwargs["artifacts"]["resolved_config"],
        )

        call_index = len(calls)
        raised = exception_by_call_index.get(call_index)
        if raised is None:
            raised = exception_by_cell.get((target_item, victim_name))
        if raised is not None:
            raise raised

        if (target_item, victim_name) not in omit_prediction_cells:
            save_json(
                {"rankings": [[target_item, 1, 2, 3]], "victim": victim_name},
                kwargs["predictions_path"],
            )
        return (
            VictimExecutionResult(
                predictions=[[target_item, 1, 2, 3]],
                predictions_path=kwargs["predictions_path"],
                extra={"phase6_fake": True},
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
        metadata={"phase6_smoke": True},
    )


def test_interrupted_run_reruns_only_non_completed_cells(monkeypatch) -> None:
    with _phase6_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=3, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")
        target_prefix = _expected_target_prefix(config)

        interrupted_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=interrupted_calls,
            exception_by_call_index={2: KeyboardInterrupt("phase6 simulated interrupt")},
        )
        with pytest.raises(KeyboardInterrupt):
            run_targets_and_victims(
                config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        failed_coverage = load_run_coverage(metadata_paths["run_coverage"])
        failed_summary_current = load_summary_current(metadata_paths["summary_current"])
        failed_execution_log = load_execution_log(metadata_paths["execution_log"])

        retry_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=retry_calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        final_summary_current = load_summary_current(metadata_paths["summary_current"])
        final_execution_log = load_execution_log(metadata_paths["execution_log"])
        progress_payload = load_json(metadata_paths["progress"])

    first_target, second_target, third_target = [int(item) for item in target_prefix]
    assert interrupted_calls == [(first_target, "miasrec"), (second_target, "miasrec")]
    assert failed_coverage is not None
    assert failed_coverage["cells"][str(first_target)]["miasrec"]["status"] == "completed"
    assert failed_coverage["cells"][str(second_target)]["miasrec"]["status"] == "failed"
    assert failed_coverage["cells"][str(third_target)]["miasrec"]["status"] == "requested"
    assert failed_summary_current is not None
    assert set(failed_summary_current["targets"]) == {str(first_target)}

    assert failed_execution_log is not None
    first_record = failed_execution_log["executions"][0]
    assert first_record["status"] == "failed"
    assert len(first_record["completed_cells"]) == 1
    assert len(first_record["failed_cells"]) == 1
    assert [cell["status"] for cell in first_record["planned_cells"]] == [
        "completed",
        "failed",
        "requested",
    ]

    assert retry_calls == [(second_target, "miasrec"), (third_target, "miasrec")]
    assert final_coverage is not None
    for target_item in target_prefix:
        assert final_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
    assert final_summary_current is not None
    assert set(final_summary_current["targets"]) == {str(item) for item in target_prefix}

    assert final_execution_log is not None
    assert len(final_execution_log["executions"]) == 2
    retry_record = final_execution_log["executions"][1]
    assert retry_record["mode"] == "retry_incomplete_cells"
    assert retry_record["status"] == "completed"
    assert len(retry_record["planned_cells"]) == 2
    assert len(retry_record["skipped_completed_cells"]) == 1
    assert progress_payload["is_authoritative"] is False


def test_missing_required_artifact_never_marks_completed_and_progress_is_ignored(monkeypatch) -> None:
    with _phase6_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=1, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")
        target_item = int(_expected_target_prefix(config)[0])

        failing_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=failing_calls,
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
        failed_coverage = load_run_coverage(metadata_paths["run_coverage"])
        failed_execution_log = load_execution_log(metadata_paths["execution_log"])
        save_json(
            {
                "is_authoritative": False,
                "status": "completed",
                "runs": [{"target_item": target_item, "victim_name": "miasrec", "status": "completed"}],
            },
            metadata_paths["progress"],
        )

        retry_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=retry_calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        final_summary_current = load_summary_current(metadata_paths["summary_current"])
        final_execution_log = load_execution_log(metadata_paths["execution_log"])

    assert "Cell completion requires persisted local artifacts" in str(exc_info.value)
    assert failing_calls == [(target_item, "miasrec")]
    assert failed_coverage is not None
    failed_cell = failed_coverage["cells"][str(target_item)]["miasrec"]
    assert failed_cell["status"] == "failed"
    assert failed_cell["completed_at"] is None
    assert failed_cell["artifacts"]["metrics"] is not None
    assert failed_cell["artifacts"]["predictions"] is None

    assert failed_execution_log is not None
    assert failed_execution_log["executions"][0]["status"] == "failed"
    assert len(failed_execution_log["executions"][0]["failed_cells"]) == 1

    assert retry_calls == [(target_item, "miasrec")]
    assert final_coverage is not None
    assert final_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
    assert final_summary_current is not None
    assert set(final_summary_current["targets"]) == {str(target_item)}
    assert final_execution_log is not None
    assert len(final_execution_log["executions"]) == 2


def test_mixed_target_and_victim_append_resumes_after_partial_interrupt(monkeypatch) -> None:
    with _phase6_temp_root() as temp_root:
        initial_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        combined_config = _config_for_temp_root(temp_root, count=4, victims=("miasrec", "tron"))
        context = _minimal_context(initial_config, run_type="clean")

        initial_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=initial_calls)
        run_targets_and_victims(
            initial_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        target_prefix = _expected_target_prefix(combined_config)
        old_targets = [int(item) for item in target_prefix[:2]]
        new_targets = [int(item) for item in target_prefix[2:]]

        combined_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=combined_calls,
            exception_by_cell={(new_targets[0], "tron"): KeyboardInterrupt("phase6 mixed interrupt")},
        )
        with pytest.raises(KeyboardInterrupt):
            run_targets_and_victims(
                combined_config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(combined_config, run_type="clean")
        interrupted_coverage = load_run_coverage(metadata_paths["run_coverage"])
        interrupted_summary_current = load_summary_current(metadata_paths["summary_current"])
        interrupted_execution_log = load_execution_log(metadata_paths["execution_log"])

        retry_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=retry_calls)
        run_targets_and_victims(
            combined_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        final_summary_current = load_summary_current(metadata_paths["summary_current"])
        final_execution_log = load_execution_log(metadata_paths["execution_log"])

    assert combined_calls == [
        (old_targets[0], "tron"),
        (old_targets[1], "tron"),
        (new_targets[0], "miasrec"),
        (new_targets[0], "tron"),
    ]
    assert interrupted_coverage is not None
    assert interrupted_coverage["cells"][str(old_targets[0])]["tron"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(old_targets[1])]["tron"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(new_targets[0])]["miasrec"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(new_targets[0])]["tron"]["status"] == "failed"
    assert interrupted_coverage["cells"][str(new_targets[1])]["miasrec"]["status"] == "requested"
    assert interrupted_coverage["cells"][str(new_targets[1])]["tron"]["status"] == "requested"

    assert interrupted_summary_current is not None
    assert set(interrupted_summary_current["targets"][str(new_targets[0])]["victims"]) == {
        "miasrec"
    }
    assert interrupted_execution_log is not None
    interrupted_record = interrupted_execution_log["executions"][1]
    assert interrupted_record["mode"] == "target_and_victim_append"
    assert interrupted_record["status"] == "failed"
    assert len(interrupted_record["completed_cells"]) == 3
    assert len(interrupted_record["failed_cells"]) == 1
    assert len(interrupted_record["skipped_completed_cells"]) == 2

    assert retry_calls == [
        (new_targets[0], "tron"),
        (new_targets[1], "miasrec"),
        (new_targets[1], "tron"),
    ]
    assert final_coverage is not None
    for target_item in target_prefix:
        assert final_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
        assert final_coverage["cells"][str(target_item)]["tron"]["status"] == "completed"
    assert final_summary_current is not None
    for target_item in target_prefix:
        assert set(final_summary_current["targets"][str(target_item)]["victims"]) == {
            "miasrec",
            "tron",
        }
    assert final_execution_log is not None
    retry_record = final_execution_log["executions"][2]
    assert retry_record["mode"] == "retry_incomplete_cells"
    assert retry_record["status"] == "completed"
    assert len(retry_record["planned_cells"]) == 3
    assert len(retry_record["skipped_completed_cells"]) == 5


def test_stale_running_execution_records_are_reconciled_on_next_invocation(monkeypatch) -> None:
    with _phase6_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=1, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")

        initial_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=initial_calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        execution_log = load_execution_log(metadata_paths["execution_log"])
        assert execution_log is not None
        execution_log["executions"][0]["status"] = "running"
        execution_log["executions"][0]["completed_at"] = None
        execution_log["executions"][0]["error_type"] = None
        execution_log["executions"][0]["error"] = None
        save_execution_log(execution_log, metadata_paths["execution_log"])

        rerun_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=rerun_calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_execution_log = load_execution_log(metadata_paths["execution_log"])

    assert rerun_calls == []
    assert final_execution_log is not None
    assert final_execution_log["executions"][0]["status"] == "interrupted"
    assert final_execution_log["executions"][0]["error_type"] == "InterruptedExecution"
    assert final_execution_log["executions"][1]["mode"] == "noop"
    assert final_execution_log["executions"][1]["status"] == "completed"
