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
    assert CONFIG_PATH.is_file(), f"Missing Phase 5 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase5_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase5" / uuid4().hex
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
    victim_epoch_overrides: dict[str, int] | None = None,
):
    base = _base_config()
    victim_params = {
        name: dict(params)
        for name, params in base.victims.params.items()
    }
    if victim_epoch_overrides:
        for victim_name, epochs in victim_epoch_overrides.items():
            params = dict(victim_params[victim_name])
            train = dict(params.get("train", {}))
            train["epochs"] = int(epochs)
            params["train"] = train
            victim_params[victim_name] = params
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=replace(base.targets, mode="sampled", bucket="all", count=count),
        victims=replace(base.victims, enabled=victims, params=victim_params),
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
) -> None:
    fail_cells = fail_cells or set()

    def fake_victim_execution(*args, **kwargs):
        target_item = int(kwargs["target_item"])
        victim_name = str(kwargs["victim_name"])
        calls.append((target_item, victim_name))
        save_json(
            {"victim_name": victim_name, "phase5_fake": True},
            kwargs["artifacts"]["resolved_config"],
        )
        if (target_item, victim_name) in fail_cells:
            raise RuntimeError(f"phase5 forced failure for {target_item}:{victim_name}")
        save_json(
            {"rankings": [[target_item, 1, 2, 3]], "victim": victim_name},
            kwargs["predictions_path"],
        )
        return (
            VictimExecutionResult(
                predictions=[[target_item, 1, 2, 3]],
                predictions_path=kwargs["predictions_path"],
                extra={"phase5_fake": True},
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
        metadata={"phase5_smoke": True},
    )


def test_appending_new_victim_only_executes_missing_cells_for_that_victim(monkeypatch) -> None:
    with _phase5_temp_root() as temp_root:
        initial_config = _config_for_temp_root(temp_root, count=3, victims=("miasrec",))
        expanded_config = _config_for_temp_root(temp_root, count=3, victims=("miasrec", "tron"))
        context = _minimal_context(initial_config, run_type="clean")

        first_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=first_calls)
        run_targets_and_victims(
            initial_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        second_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=second_calls)
        summary = run_targets_and_victims(
            expanded_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(expanded_config, run_type="clean")
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])
        summary_current = load_summary_current(metadata_paths["summary_current"])
        progress_payload = load_json(metadata_paths["progress"])

    expected_targets = _expected_target_prefix(expanded_config)
    assert len(first_calls) == 3
    assert second_calls == [(int(target_item), "tron") for target_item in expected_targets]
    assert summary["target_items"] == [int(item) for item in expected_targets]

    assert run_coverage is not None
    assert list(run_coverage["victims"].keys()) == ["miasrec", "tron"]
    assert run_coverage["victims"]["tron"]["victim_prediction_key"].startswith("victim_tron_")
    for target_item in expected_targets:
        assert run_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
        assert run_coverage["cells"][str(target_item)]["tron"]["status"] == "completed"

    assert execution_log is not None
    assert len(execution_log["executions"]) == 2
    victim_append_record = execution_log["executions"][1]
    assert victim_append_record["mode"] == "victim_append"
    assert victim_append_record["added_target_items"] == []
    assert victim_append_record["added_victims"] == ["tron"]
    assert len(victim_append_record["planned_cells"]) == 3
    assert len(victim_append_record["skipped_completed_cells"]) == 3

    assert summary_current is not None
    for target_item in expected_targets:
        assert set(summary_current["targets"][str(target_item)]["victims"]) == {"miasrec", "tron"}
    assert progress_payload["is_authoritative"] is False
    assert progress_payload["requested_victims"] == ["miasrec", "tron"]


def test_rerun_with_same_prefix_and_expanded_victim_set_does_not_rerun_completed_cells(
    monkeypatch,
) -> None:
    with _phase5_temp_root() as temp_root:
        initial_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        expanded_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec", "tron"))
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

        append_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=append_calls)
        run_targets_and_victims(
            expanded_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        rerun_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=rerun_calls)
        run_targets_and_victims(
            expanded_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(expanded_config, run_type="clean")
        execution_log = load_execution_log(metadata_paths["execution_log"])
        progress_payload = load_json(metadata_paths["progress"])

    assert len(initial_calls) == 2
    assert len(append_calls) == 2
    assert rerun_calls == []
    assert execution_log is not None
    noop_record = execution_log["executions"][2]
    assert noop_record["mode"] == "noop"
    assert noop_record["planned_cells"] == []
    assert len(noop_record["skipped_completed_cells"]) == 4
    assert progress_payload["total_victims"] == 2
    assert progress_payload["requested_victims"] == ["miasrec", "tron"]


def test_combined_target_append_and_victim_append_schedules_expected_missing_matrix(
    monkeypatch,
) -> None:
    with _phase5_temp_root() as temp_root:
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

        combined_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=combined_calls)
        run_targets_and_victims(
            combined_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(combined_config, run_type="clean")
        execution_log = load_execution_log(metadata_paths["execution_log"])
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        summary_current = load_summary_current(metadata_paths["summary_current"])

    expected_targets = _expected_target_prefix(combined_config)
    old_targets = expected_targets[:2]
    new_targets = expected_targets[2:]
    assert combined_calls == [
        (int(old_targets[0]), "tron"),
        (int(old_targets[1]), "tron"),
        (int(new_targets[0]), "miasrec"),
        (int(new_targets[0]), "tron"),
        (int(new_targets[1]), "miasrec"),
        (int(new_targets[1]), "tron"),
    ]

    assert execution_log is not None
    combined_record = execution_log["executions"][1]
    assert combined_record["mode"] == "target_and_victim_append"
    assert combined_record["added_target_items"] == [int(item) for item in new_targets]
    assert combined_record["added_victims"] == ["tron"]
    assert len(combined_record["skipped_completed_cells"]) == 2

    assert run_coverage is not None
    assert run_coverage["targets_order"] == [int(item) for item in expected_targets]
    for target_item in expected_targets:
        assert run_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
        assert run_coverage["cells"][str(target_item)]["tron"]["status"] == "completed"

    assert summary_current is not None
    assert set(summary_current["targets"]) == {str(item) for item in expected_targets}
    for target_item in expected_targets:
        assert set(summary_current["targets"][str(target_item)]["victims"]) == {"miasrec", "tron"}


def test_failed_new_victim_cells_remain_eligible_for_rerun(monkeypatch) -> None:
    with _phase5_temp_root() as temp_root:
        initial_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        expanded_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec", "tron"))
        context = _minimal_context(initial_config, run_type="clean")
        target_prefix = _expected_target_prefix(expanded_config)
        failed_target = int(target_prefix[1])

        initial_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=initial_calls)
        run_targets_and_victims(
            initial_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        failing_append_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=failing_append_calls,
            fail_cells={(failed_target, "tron")},
        )
        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                expanded_config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(expanded_config, run_type="clean")
        failed_coverage = load_run_coverage(metadata_paths["run_coverage"])
        failed_summary_current = load_summary_current(metadata_paths["summary_current"])

        retry_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=retry_calls)
        run_targets_and_victims(
            expanded_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])

    assert "phase5 forced failure" in str(exc_info.value)
    assert failing_append_calls == [(int(target_prefix[0]), "tron"), (failed_target, "tron")]
    assert failed_coverage is not None
    assert failed_coverage["cells"][str(target_prefix[0])]["tron"]["status"] == "completed"
    assert failed_coverage["cells"][str(failed_target)]["tron"]["status"] == "failed"
    assert failed_summary_current is not None
    assert set(failed_summary_current["targets"][str(target_prefix[0])]["victims"]) == {"miasrec", "tron"}
    assert set(failed_summary_current["targets"][str(failed_target)]["victims"]) == {"miasrec"}

    assert retry_calls == [(failed_target, "tron")]
    assert final_coverage is not None
    assert final_coverage["cells"][str(failed_target)]["tron"]["status"] == "completed"
    assert execution_log is not None
    assert execution_log["executions"][1]["mode"] == "victim_append"
    assert execution_log["executions"][1]["status"] == "failed"
    assert execution_log["executions"][2]["mode"] == "retry_incomplete_cells"
    assert execution_log["executions"][2]["status"] == "completed"


def test_incompatible_existing_victim_configuration_is_rejected(monkeypatch) -> None:
    with _phase5_temp_root() as temp_root:
        initial_config = _config_for_temp_root(temp_root, count=2, victims=("miasrec",))
        incompatible_config = _config_for_temp_root(
            temp_root,
            count=2,
            victims=("miasrec",),
            victim_epoch_overrides={"miasrec": 999},
        )
        context = _minimal_context(initial_config, run_type="clean")
        calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=calls)
        run_targets_and_victims(
            initial_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                incompatible_config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

    assert "victim registry is incompatible with the currently requested victim configuration" in str(
        exc_info.value
    )
