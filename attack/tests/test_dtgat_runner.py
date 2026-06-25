from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from attack.common.config import load_config, normalize_config_mapping
from attack.data.canonical_dataset import CanonicalDataset
from attack.models.victim.dtgat_runner import DTGATRunner
from attack.models.victim.registry import available_victims
from attack.pipeline.core import orchestrator
from attack.pipeline.core.victim_execution import execute_single_victim
from attack.pipeline.core.victim_execution import VictimExecutionResult


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "attack"
    / "configs"
    / "diginetica_attack_dpsbr.yaml"
)


def _runner_config(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    params = dict(config.victims.params)
    params["dtgat"] = {
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "emb_size": 8,
            "time_dims": 8,
            "intent_num": 2,
            "lr": 0.001,
            "dropout": 0.1,
            "l2": 1e-4,
            "lr_dc": 0.1,
            "lr_dc_step": 10,
            "layer": 1,
            "beta": 0.005,
            "topk": 3,
            "per_epoch_diagnostics": False,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    runtime = dict(config.victims.runtime or {})
    python_path = tmp_path / "python.exe"
    python_path.write_text("", encoding="utf-8")
    runtime["dtgat"] = {
        "python_executable": str(python_path.resolve()),
        "repo_root": str((tmp_path / "dtgat").resolve()),
        "working_dir": str((tmp_path / "dtgat").resolve()),
        "device": {"use_gpu": True, "gpu_id": "3"},
    }
    return replace(config, victims=replace(config.victims, params=params, runtime=runtime))


def test_dtgat_is_registered_and_config_validates(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    payload = config.to_primitive()
    payload["victims"]["enabled"] = ["dtgat"]
    payload["victims"]["params"] = {"dtgat": config.victims.params["dtgat"]}
    payload["victims"]["runtime"] = {"dtgat": config.victims.runtime["dtgat"]}

    normalized = normalize_config_mapping(payload)

    assert "dtgat" in available_victims()
    assert normalized["victims"]["enabled"] == ["dtgat"]
    assert normalized["victims"]["params"]["dtgat"]["train"]["topk"] == 3
    assert normalized["victims"]["params"]["dtgat"]["train"]["per_epoch_diagnostics"] is False
    assert normalized["victims"]["params"]["dtgat"]["train"]["checkpoint_protocol"] == "fixed_epoch"


def test_dtgat_local_smoke_config_validates() -> None:
    smoke_path = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "configs"
        / "local_diginetica_valbest_clean_dtgat_append_one_target.yaml"
    )
    config = load_config(smoke_path)

    assert config.victims.enabled == ("dtgat",)
    assert config.targets.count == 1
    assert config.victims.params["dtgat"]["train"]["topk"] == 50
    assert "not a formal full-result" in smoke_path.read_text(encoding="utf-8")


def test_dtgat_local_epoch_diagnostic_configs_validate() -> None:
    config_root = Path(__file__).resolve().parents[2] / "attack" / "configs"
    for name in (
        "local_diginetica_dtgat_epoch3_diagnostic_one_target.yaml",
        "local_yoochoose1_64_dtgat_epoch3_diagnostic_one_target.yaml",
    ):
        config = load_config(config_root / name)
        train = config.victims.params["dtgat"]["train"]
        assert config.victims.enabled == ("dtgat",)
        assert config.targets.count == 1
        assert train["epochs"] == 3
        assert train["topk"] == 50
        assert train["per_epoch_diagnostics"] is True


def test_runner_builds_expected_dtgat_command_and_env(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    repo = Path(config.victims.runtime["dtgat"]["repo_root"])
    repo.mkdir()
    (repo / "main.py").write_text("", encoding="utf-8")
    data_dir = tmp_path / "export" / "toy"
    data_dir.mkdir(parents=True)

    cmd, env = DTGATRunner(config).build_command(
        data_dir=data_dir,
        dataset_name="toy",
        n_node=5,
        prediction_output_path=tmp_path / "run" / "dtgat_topk_raw.json",
        metrics_output_path=tmp_path / "run" / "dtgat_third_party_metrics.json",
        resolved_config_output_path=tmp_path
        / "run"
        / "dtgat_third_party_resolved_config.json",
        per_epoch_prediction_dir=None,
        requested_topk=3,
        epochs=2,
        seed=7,
    )

    expected_args = {
        "--dataset": "toy",
        "--data_dir": str(data_dir.resolve()),
        "--n_node": "5",
        "--epoch": "2",
        "--batchSize": "4",
        "--embSize": "8",
        "--time_dims": "8",
        "--intent_num": "2",
        "--lr": "0.001",
        "--dropout": "0.1",
        "--layer": "1",
        "--topk": "3",
        "--seed": "7",
        "--gpu_id": "3",
        "--cuda": "true",
    }
    for argument, expected_value in expected_args.items():
        assert argument in cmd
        assert cmd[cmd.index(argument) + 1] == expected_value
    assert "--prediction_output_path" in cmd
    assert "--metrics_output_path" in cmd
    assert "--resolved_config_output_path" in cmd
    assert "--per_epoch_prediction_dir" not in cmd
    assert env["PYTHONHASHSEED"] == "7"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"


def test_runner_builds_per_epoch_prediction_arg_only_when_enabled(tmp_path: Path) -> None:
    config = _runner_config(tmp_path)
    repo = Path(config.victims.runtime["dtgat"]["repo_root"])
    repo.mkdir()
    (repo / "main.py").write_text("", encoding="utf-8")
    data_dir = tmp_path / "export" / "toy"
    data_dir.mkdir(parents=True)
    per_epoch_dir = tmp_path / "run" / "dtgat_per_epoch_predictions"

    cmd, _ = DTGATRunner(config).build_command(
        data_dir=data_dir,
        dataset_name="toy",
        n_node=5,
        prediction_output_path=tmp_path / "run" / "dtgat_topk_raw.json",
        metrics_output_path=tmp_path / "run" / "dtgat_third_party_metrics.json",
        resolved_config_output_path=tmp_path
        / "run"
        / "dtgat_third_party_resolved_config.json",
        per_epoch_prediction_dir=per_epoch_dir,
        requested_topk=3,
        epochs=2,
        seed=7,
    )

    assert "--per_epoch_prediction_dir" in cmd
    assert cmd[cmd.index("--per_epoch_prediction_dir") + 1] == str(per_epoch_dir.resolve())


def _write_prediction(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"topk": 2, "requested_topk": 2, "n_node": 3, "num_examples": 1},
            "must contain a rankings list",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "num_examples": 1,
                "rankings": [[1, 2]],
            },
            "prediction count mismatch",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "num_examples": 1,
                "rankings": [[1]],
            },
            "shorter than JSON topk",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "num_examples": 1,
                "rankings": [[0, 1]],
            },
            "outside 1..3",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "num_examples": 1,
                "rankings": [[1, 4]],
            },
            "outside 1..3",
        ),
    ],
)
def test_runner_rejects_malformed_prediction_json(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "predictions.json"
    _write_prediction(path, payload)

    with pytest.raises(ValueError, match=message):
        DTGATRunner(_runner_config(tmp_path)).predict_topk(
            predictions_path=path,
            expected_test_count=2 if "count mismatch" in message else 1,
            n_node=3,
            requested_topk=2,
        )


def test_runner_accepts_valid_predictions_and_slices(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    _write_prediction(
        path,
        {
            "topk": 3,
            "requested_topk": 3,
            "n_node": 5,
            "num_examples": 2,
            "rankings": [[1, 2, 3], [5, 4, 3]],
        },
    )

    rankings = DTGATRunner(_runner_config(tmp_path)).predict_topk(
        predictions_path=path,
        expected_test_count=2,
        n_node=5,
        requested_topk=3,
        topk=2,
    )

    assert rankings == [[1, 2], [5, 4]]


def test_per_epoch_prediction_metrics_require_every_epoch(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "run" / "dtgat_per_epoch_predictions"
    prediction_dir.mkdir(parents=True)
    _write_prediction(
        prediction_dir / "epoch_001.json",
        {
            "topk": 2,
            "requested_topk": 2,
            "n_node": 5,
            "num_examples": 1,
            "epoch": 1,
            "rankings": [[1, 2]],
        },
    )

    with pytest.raises(RuntimeError, match="Missing"):
        DTGATRunner(_runner_config(tmp_path)).evaluate_per_epoch_predictions(
            prediction_dir=prediction_dir,
            metrics_output_path=tmp_path / "run" / "per_epoch_metrics.json",
            expected_epochs=2,
            expected_test_count=1,
            n_node=5,
            requested_topk=2,
            target_item=2,
            ground_truth_labels=[2],
            evaluation_topk=[1, 2],
            targeted_metrics=["recall"],
            ground_truth_metrics=["recall"],
            run_dir=tmp_path / "run",
        )


def test_per_epoch_prediction_metrics_are_written(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prediction_dir = run_dir / "dtgat_per_epoch_predictions"
    prediction_dir.mkdir(parents=True)
    for epoch in (1, 2):
        _write_prediction(
            prediction_dir / f"epoch_{epoch:03d}.json",
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 5,
                "num_examples": 1,
                "epoch": epoch,
                "rankings": [[2, 1]],
            },
        )

    summary = DTGATRunner(_runner_config(tmp_path)).evaluate_per_epoch_predictions(
        prediction_dir=prediction_dir,
        metrics_output_path=run_dir / "per_epoch_metrics.json",
        expected_epochs=2,
        expected_test_count=1,
        n_node=5,
        requested_topk=2,
        target_item=2,
        ground_truth_labels=[2],
        evaluation_topk=[1, 2],
        targeted_metrics=["recall"],
        ground_truth_metrics=["recall"],
        run_dir=run_dir,
    )

    payload = json.loads((run_dir / "per_epoch_metrics.json").read_text(encoding="utf-8"))
    assert payload == summary
    assert payload["enabled"] is True
    assert [row["epoch"] for row in payload["epochs"]] == [1, 2]
    assert payload["epochs"][0]["prediction_path"] == "dtgat_per_epoch_predictions/epoch_001.json"
    assert payload["epochs"][0]["metrics"]["targeted_recall@1"] == 1.0


def test_execution_branch_uses_export_result_data_dir_and_poisoned_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _runner_config(tmp_path)
    dataset = CanonicalDataset(
        train_sub=[[1, 2, 3]],
        valid=[[1, 3]],
        test=[[1, 2]],
        item_map={str(index): index for index in range(1, 6)},
        metadata={"dataset_name": "toy", "item_count": 5, "counts": {"items": 5}},
    )
    captured: dict[str, object] = {}
    data_dir = tmp_path / "exported" / "toy"
    data_dir.mkdir(parents=True)
    test_path = data_dir / "processed_data" / "test.txt"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("", encoding="utf-8")

    class FakeExporter:
        def export_with_poisoned_train(self, dataset_arg, **kwargs):
            captured["export_kwargs"] = kwargs
            return SimpleNamespace(
                data_dir=data_dir,
                n_node=5,
                test_example_count=1,
                train_example_count=2,
                raw_train_session_count=2,
                raw_fake_session_count=1,
                observed_max_item_id=5,
                max_train_prefix_length=2,
                max_test_prefix_length=1,
                max_all_train_seq_length=3,
                expected_fake_expanded_pair_count=1,
                fake_pairs_present_in_train=True,
                files={"test": test_path},
            )

    class FakeRunner:
        def __init__(self, config_arg):
            captured["runner_config"] = config_arg

        def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return {
                "log_path": str(tmp_path / "dtgat_stdout.log"),
                "prediction_output_path": str(kwargs["prediction_output_path"]),
            }

        def predict_topk(self, **kwargs):
            captured["predict_kwargs"] = kwargs
            return [[1, 2, 3]]

    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution.DTGATExporter",
        lambda: FakeExporter(),
    )
    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution.get_victim_runner",
        lambda name: FakeRunner,
    )

    result = execute_single_victim(
        config,
        run_type="poisoned",
        victim_name="dtgat",
        canonical_dataset=dataset,
        poisoned_sessions=[[1], [4, 5]],
        poisoned_labels=[2, 3],
        raw_fake_sessions=[[4, 5, 3]],
        run_dir=tmp_path / "victim",
        poisoned_train_path=tmp_path / "poisoned_train.txt",
        target_item=3,
        eval_topk=[1, 3],
        predictions_path=tmp_path / "victim" / "predictions.json",
    )

    export_kwargs = captured["export_kwargs"]
    assert export_kwargs["poisoned_prefixes"] == [[1], [4, 5]]
    assert export_kwargs["poisoned_labels"] == [2, 3]
    assert export_kwargs["raw_fake_sessions"] == [[4, 5, 3]]
    assert captured["run_kwargs"]["data_dir"] == data_dir
    assert captured["run_kwargs"]["requested_topk"] == 3
    assert result.predictions == [[1, 2, 3]]


def test_dtgat_bypasses_shared_clean_prediction_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _runner_config(tmp_path)
    shared_predictions = tmp_path / "shared" / "predictions.json"
    shared_execution = tmp_path / "shared" / "execution_result.json"
    shared_predictions.parent.mkdir(parents=True)
    shared_predictions.write_text(
        json.dumps({"rankings": [[99]], "victim": "dtgat"}),
        encoding="utf-8",
    )
    shared_execution.write_text(json.dumps({"extra": {"cached": True}}), encoding="utf-8")
    calls: list[str] = []

    def fake_execute(*args, **kwargs):
        calls.append(kwargs["victim_name"])
        return VictimExecutionResult(
            predictions=[[1, 2, 3]],
            predictions_path=kwargs["predictions_path"],
            extra={"fresh": True},
            poisoned_train_path=None,
        )

    monkeypatch.setattr(orchestrator, "execute_single_victim", fake_execute)

    result, reused = orchestrator._maybe_reuse_or_execute_victim(
        config,
        run_type="clean",
        run_coverage={"cells": {"3": {"dtgat": {"status": "completed"}}}},
        victim_name="dtgat",
        canonical_dataset=object(),
        poisoned_sessions=[[1]],
        poisoned_labels=[2],
        raw_fake_sessions=[],
        run_dir=tmp_path / "victim",
        poisoned_train_path=tmp_path / "poisoned_train.txt",
        target_item=3,
        eval_topk=[3],
        srg_nn_export_paths=None,
        predictions_path=tmp_path / "victim" / "predictions.json",
        artifacts={
            "shared_predictions": shared_predictions,
            "shared_execution_result": shared_execution,
        },
    )

    assert calls == ["dtgat"]
    assert reused is False
    assert result.extra == {"fresh": True}
