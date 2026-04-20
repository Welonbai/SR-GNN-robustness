from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest

from attack.common.artifact_io import load_json, load_run_coverage, load_summary_current, save_json
from attack.common.config import load_config
from attack.common.paths import (
    run_artifact_paths,
    run_metadata_paths,
    shared_artifact_paths,
    shared_victim_dir,
)
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    _persist_shared_victim_result,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import build_ordered_target_cohort
from attack.pipeline.core.victim_execution import VictimExecutionResult, _victim_stage_seed
from attack.pipeline.runs.run_clean import run_clean


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing clean shared reuse test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_clean_shared" / uuid4().hex
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
    victims: tuple[str, ...] = ("miasrec",),
):
    base = _base_config()
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=replace(base.targets, mode="sampled", bucket="all", count=count),
        victims=replace(base.victims, enabled=victims),
    )


def _minimal_context(config) -> RunContext:
    return RunContext(
        canonical_dataset=object(),
        stats=_sample_stats(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        export_paths={},
        shared_paths=shared_artifact_paths(config, run_type="clean"),
        fake_session_count=0,
    )


def _expected_target_prefix(config) -> list[int]:
    cohort = build_ordered_target_cohort(_sample_stats(), config)
    return [int(item) for item in cohort["ordered_targets"][: int(config.targets.count)]]


def _build_clean_poisoned(target_item: int) -> TargetPoisonOutput:
    return TargetPoisonOutput(
        poisoned=PoisonedDataset(
            sessions=[[1, 2]],
            labels=[999],
            clean_count=1,
            fake_count=0,
        ),
        metadata={"clean_shared_reuse_test": True, "target_item_for_metrics": int(target_item)},
    )


def _install_fake_clean_execution(
    monkeypatch,
    *,
    execute_calls: list[dict[str, object]],
    include_local_path_extra: bool = False,
) -> None:
    def fake_execute_single_victim(config, **kwargs):
        target_item = int(kwargs["target_item"])
        run_type = str(kwargs["run_type"])
        victim_name = str(kwargs["victim_name"])
        predictions_path = kwargs["predictions_path"]
        run_dir = kwargs["run_dir"]
        eval_topk = kwargs["eval_topk"]
        execute_calls.append(
            {
                "run_type": run_type,
                "target_item": target_item,
                "victim_name": victim_name,
                "poisoned_labels": list(kwargs["poisoned_labels"]),
            }
        )
        save_json(
            {"fake_resolved": True, "run_type": run_type, "victim_name": victim_name},
            run_dir / "resolved_config.json",
        )
        save_json(
            {
                "available": True,
                "count": 1,
                "rankings": [[7, 8, 9]],
                "topk": int(max(eval_topk)),
                "victim": victim_name,
                "target_item": target_item,
            },
            predictions_path,
        )
        extra: dict[str, object] = {"clean_shared_reuse": True}
        if include_local_path_extra:
            extra.update(
                {
                    victim_name: {
                        "returncode": 0,
                        "victim_train_seed": 123,
                        "log_path": str(run_dir / f"{victim_name}_stdout.log"),
                        "export_topk_path": str(run_dir / f"{victim_name}_topk_raw.json"),
                    },
                    f"{victim_name}_export": {
                        "train": str(run_dir / "export" / victim_name / "train.txt"),
                        "valid": str(run_dir / "export" / victim_name / "valid.txt"),
                    },
                }
            )
        return VictimExecutionResult(
            predictions=[[7, 8, 9]],
            predictions_path=predictions_path,
            extra=extra,
            poisoned_train_path=None,
        )

    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.execute_single_victim",
        fake_execute_single_victim,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.resolve_ground_truth_labels",
        lambda *args, **kwargs: [1],
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.evaluate_prediction_metrics",
        lambda *args, **kwargs: (
            {
                "targeted_recall@10": float(kwargs["target_item"]),
                "ground_truth_recall@10": 0.5,
            },
            True,
        ),
    )


def test_clean_shared_victim_dir_is_target_agnostic() -> None:
    config = _config_for_temp_root(REPO_ROOT / "outputs" / ".pytest_clean_shared_identity", count=2)

    clean_first = shared_victim_dir(
        config,
        run_type="clean",
        target_id=11,
        victim_name="miasrec",
    )
    clean_second = shared_victim_dir(
        config,
        run_type="clean",
        target_id=22,
        victim_name="miasrec",
    )
    attack_first = shared_victim_dir(
        config,
        run_type="attack",
        target_id=11,
        victim_name="miasrec",
    )
    attack_second = shared_victim_dir(
        config,
        run_type="attack",
        target_id=22,
        victim_name="miasrec",
    )

    assert clean_first == clean_second
    assert clean_first.name == "shared"
    assert "targets" not in clean_first.parts
    assert attack_first != attack_second
    assert attack_first.parts[-2] == "targets"


def test_clean_victim_stage_seed_ignores_target_item() -> None:
    config = _base_config()

    clean_a = _victim_stage_seed(
        config,
        victim_name="miasrec",
        run_type="clean",
        target_item=11,
    )
    clean_b = _victim_stage_seed(
        config,
        victim_name="miasrec",
        run_type="clean",
        target_item=22,
    )
    attack_a = _victim_stage_seed(
        config,
        victim_name="miasrec",
        run_type="attack",
        target_item=11,
    )
    attack_b = _victim_stage_seed(
        config,
        victim_name="miasrec",
        run_type="attack",
        target_item=22,
    )

    assert clean_a == clean_b
    assert attack_a != attack_b


def test_clean_initial_run_executes_each_victim_once_and_reuses_predictions(monkeypatch) -> None:
    with _temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=3)
        context = _minimal_context(config)
        expected_targets = _expected_target_prefix(config)
        execute_calls: list[dict[str, object]] = []
        _install_fake_clean_execution(monkeypatch, execute_calls=execute_calls)

        summary = run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_clean_poisoned,
        )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        summary_current = load_summary_current(metadata_paths["summary_current"])
        first_artifacts = run_artifact_paths(
            config,
            run_type="clean",
            target_id=expected_targets[0],
            victim_name="miasrec",
        )
        second_artifacts = run_artifact_paths(
            config,
            run_type="clean",
            target_id=expected_targets[1],
            victim_name="miasrec",
        )
        second_predictions = load_json(second_artifacts["predictions"])

    assert len(execute_calls) == 1
    assert execute_calls[0]["run_type"] == "clean"
    assert execute_calls[0]["target_item"] == int(expected_targets[0])
    assert execute_calls[0]["poisoned_labels"] == [999]
    assert summary["target_items"] == [int(item) for item in expected_targets]

    assert first_artifacts["shared_dir"] == second_artifacts["shared_dir"]
    assert run_coverage is not None
    for target_item in expected_targets:
        assert run_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"

    assert second_predictions is not None
    assert second_predictions["target_item"] == int(expected_targets[1])

    assert summary_current is not None
    for index, target_item in enumerate(expected_targets):
        victim_summary = summary_current["targets"][str(target_item)]["victims"]["miasrec"]
        assert victim_summary["metrics"]["targeted_recall@10"] == float(target_item)
        assert victim_summary["metrics"]["ground_truth_recall@10"] == 0.5
        assert victim_summary["reused_predictions"] is (index != 0)


def test_reused_clean_cells_strip_stale_local_path_metadata(monkeypatch) -> None:
    with _temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=2)
        context = _minimal_context(config)
        execute_calls: list[dict[str, object]] = []
        _install_fake_clean_execution(
            monkeypatch,
            execute_calls=execute_calls,
            include_local_path_extra=True,
        )

        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_clean_poisoned,
        )

        target_prefix = _expected_target_prefix(config)
        first_artifacts = run_artifact_paths(
            config,
            run_type="clean",
            target_id=target_prefix[0],
            victim_name="miasrec",
        )
        second_artifacts = run_artifact_paths(
            config,
            run_type="clean",
            target_id=target_prefix[1],
            victim_name="miasrec",
        )
        shared_execution = load_json(first_artifacts["shared_execution_result"])
        reused_metrics = load_json(second_artifacts["metrics"])

    assert len(execute_calls) == 1
    assert shared_execution is not None
    assert shared_execution["extra"]["miasrec"] == {
        "returncode": 0,
        "victim_train_seed": 123,
    }
    assert "miasrec_export" not in shared_execution["extra"]

    assert reused_metrics is not None
    assert reused_metrics["reused_predictions"] is True
    assert reused_metrics["miasrec"] == {
        "returncode": 0,
        "victim_train_seed": 123,
    }
    assert "miasrec_export" not in reused_metrics


def test_clean_target_append_reuses_shared_predictions_without_new_execution(monkeypatch) -> None:
    with _temp_root() as temp_root:
        first_config = _config_for_temp_root(temp_root, count=2)
        larger_config = _config_for_temp_root(temp_root, count=4)
        context = _minimal_context(first_config)
        initial_targets = _expected_target_prefix(first_config)
        appended_targets = _expected_target_prefix(larger_config)[len(initial_targets) :]
        execute_calls: list[dict[str, object]] = []
        _install_fake_clean_execution(monkeypatch, execute_calls=execute_calls)

        run_targets_and_victims(
            first_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_clean_poisoned,
        )
        run_targets_and_victims(
            larger_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_clean_poisoned,
        )

        metadata_paths = run_metadata_paths(larger_config, run_type="clean")
        run_coverage = load_run_coverage(metadata_paths["run_coverage"])
        summary_current = load_summary_current(metadata_paths["summary_current"])
        appended_prediction_payloads = {
            int(target_item): load_json(
                run_artifact_paths(
                    larger_config,
                    run_type="clean",
                    target_id=target_item,
                    victim_name="miasrec",
                )["predictions"]
            )
            for target_item in appended_targets
        }

    assert len(execute_calls) == 1
    assert execute_calls[0]["target_item"] == int(initial_targets[0])

    assert run_coverage is not None
    for target_item in _expected_target_prefix(larger_config):
        assert run_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"

    assert summary_current is not None
    for target_item in appended_targets:
        victim_summary = summary_current["targets"][str(target_item)]["victims"]["miasrec"]
        assert victim_summary["reused_predictions"] is True
        assert victim_summary["metrics"]["targeted_recall@10"] == float(target_item)
        assert appended_prediction_payloads[int(target_item)]["target_item"] == int(target_item)


def test_run_clean_reuses_one_target_invariant_poisoned_bundle(monkeypatch) -> None:
    with _temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=3, victims=("srgnn",))
        fake_dataset = SimpleNamespace(
            train_sub=[[1, 2, 3], [2, 3, 4]],
            valid=[[1, 2]],
            test=[[2, 3]],
            metadata={"dataset_name": config.data.dataset_name},
        )
        build_dataset_calls: list[tuple[list[list[int]], list[int], list[list[int]]]] = []
        poisoned_bundle = PoisonedDataset(
            sessions=[[1, 2]],
            labels=[3],
            clean_count=1,
            fake_count=0,
        )
        poisoned_ids: list[int] = []

        monkeypatch.setattr(
            "attack.pipeline.runs.run_clean.ensure_canonical_dataset",
            lambda _config: fake_dataset,
        )

        def fake_build_poisoned_dataset(clean_sessions, clean_labels, fake_sessions):
            build_dataset_calls.append(
                (
                    [list(session) for session in clean_sessions],
                    [int(label) for label in clean_labels],
                    [list(session) for session in fake_sessions],
                )
            )
            return poisoned_bundle

        monkeypatch.setattr(
            "attack.pipeline.runs.run_clean.build_poisoned_dataset",
            fake_build_poisoned_dataset,
        )

        def fake_run_targets_and_victims(
            _config,
            *,
            config_path,
            context,
            run_type,
            build_poisoned,
        ):
            assert run_type == "clean"
            assert config_path is None
            assert context.canonical_dataset is fake_dataset
            metadata_paths = run_metadata_paths(_config, run_type="clean")
            assert metadata_paths["run_root"].exists() is False
            assert context.export_paths is not None
            shared_attack_dir = shared_artifact_paths(_config, run_type="clean")["attack_shared_dir"]
            assert Path(context.export_paths["train"]).is_relative_to(shared_attack_dir)
            for target_item in (101, 202, 303):
                payload = build_poisoned(target_item)
                poisoned_ids.append(id(payload.poisoned))
                assert payload.metadata["position_stats_path"].endswith("position_stats.json")
            return {"target_items": [101, 202, 303]}

        monkeypatch.setattr(
            "attack.pipeline.runs.run_clean.run_targets_and_victims",
            fake_run_targets_and_victims,
        )

        summary = run_clean(config, config_path=None)

    assert summary["target_items"] == [101, 202, 303]
    assert len(build_dataset_calls) == 1
    assert poisoned_ids == [id(poisoned_bundle), id(poisoned_bundle), id(poisoned_bundle)]


def test_clean_append_rejects_missing_bootstrap_shared_cache(monkeypatch) -> None:
    with _temp_root() as temp_root:
        first_config = _config_for_temp_root(temp_root, count=1)
        larger_config = _config_for_temp_root(temp_root, count=2)
        context = _minimal_context(first_config)
        execute_calls: list[dict[str, object]] = []
        _install_fake_clean_execution(monkeypatch, execute_calls=execute_calls)

        run_targets_and_victims(
            first_config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_clean_poisoned,
        )
        first_target = _expected_target_prefix(first_config)[0]
        shared_artifacts = run_artifact_paths(
            first_config,
            run_type="clean",
            target_id=first_target,
            victim_name="miasrec",
        )
        shared_artifacts["shared_predictions"].unlink()
        shared_artifacts["shared_execution_result"].unlink()

        with pytest.raises(RuntimeError) as exc_info:
            run_targets_and_victims(
                larger_config,
                config_path=None,
                context=context,
                run_type="clean",
                build_poisoned=_build_clean_poisoned,
            )

    assert len(execute_calls) == 1
    assert "target-agnostic shared clean cache" in str(exc_info.value)


def test_attack_shared_execution_result_preserves_target_scoped_path_metadata() -> None:
    with _temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, count=1)
        artifacts = run_artifact_paths(
            config,
            run_type="attack",
            target_id=11,
            victim_name="miasrec",
        )
        save_json(
            {
                "available": True,
                "count": 1,
                "rankings": [[1, 2, 3]],
                "target_item": 11,
                "victim": "miasrec",
            },
            artifacts["predictions"],
        )
        victim_result = VictimExecutionResult(
            predictions=[[1, 2, 3]],
            predictions_path=artifacts["predictions"],
            extra={
                "miasrec": {
                    "returncode": 0,
                    "log_path": "outputs/attack/log.txt",
                },
                "miasrec_export": {
                    "train": "outputs/attack/train.inter",
                },
            },
            poisoned_train_path=None,
        )

        _persist_shared_victim_result(
            run_type="attack",
            victim_result=victim_result,
            artifacts=artifacts,
        )
        shared_execution = load_json(artifacts["shared_execution_result"])

    assert shared_execution is not None
    assert shared_execution["extra"]["miasrec"]["log_path"] == "outputs/attack/log.txt"
    assert shared_execution["extra"]["miasrec_export"]["train"] == "outputs/attack/train.inter"
