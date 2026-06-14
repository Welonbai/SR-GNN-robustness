from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from attack.common.config import load_config
from attack.models.victim.mdhg_runner import MDHGRunner
from attack.models.victim.registry import available_victims


CONFIG_PATH = Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"


def _runner_config(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    params = dict(config.victims.params)
    params["mdhg"] = {
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.001,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    runtime = dict(config.victims.runtime or {})
    runtime["mdhg"] = {
        "python_executable": "python",
        "repo_root": str(tmp_path / "mdhg"),
        "working_dir": str(tmp_path / "mdhg"),
        "device": {"use_gpu": True, "gpu_id": "3"},
    }
    return replace(config, victims=replace(config.victims, params=params, runtime=runtime))


def _with_diagnostics(config, *, epoch_metrics: bool, per_epoch_predictions: bool):
    runtime = dict(config.victims.runtime or {})
    mdhg_runtime = dict(runtime["mdhg"])
    mdhg_runtime["diagnostics"] = {
        "epoch_metrics": epoch_metrics,
        "per_epoch_predictions": per_epoch_predictions,
    }
    runtime["mdhg"] = mdhg_runtime
    return replace(config, victims=replace(config.victims, runtime=runtime))


def test_runner_command_and_json_validation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "8,9")
    config = _runner_config(tmp_path)
    repo = Path(config.victims.runtime["mdhg"]["repo_root"])
    repo.mkdir()
    (repo / "main.py").write_text("", encoding="utf-8")
    data_dir = tmp_path / "export"
    data_dir.mkdir()
    output_path = tmp_path / "run" / "mdhg_topk_raw.json"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs["env"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "topk": 3,
                    "requested_topk": 3,
                    "n_node": 5,
                    "rankings": [[1, 2, 3], [3, 2, 1]],
                }
            ),
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("attack.models.victim.mdhg_runner.run_subprocess_with_epoch_progress", fake_run)
    runner = MDHGRunner(config)
    run_info = runner.run(
        data_dir=data_dir,
        dataset_name="toy",
        n_node=5,
        expected_test_count=2,
        run_dir=tmp_path / "run",
        export_topk_path=output_path,
        topk=3,
        victim_train_seed=7,
    )

    assert "--gpu_id" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--gpu_id") + 1] == "3"
    assert "--topk" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--topk") + 1] == "3"
    expected_args = {
        "--epoch": "2",
        "--batchSize": "4",
        "--lr": "0.001",
        "--data_dir": str(data_dir.resolve()),
        "--n_node": "5",
        "--prediction_output_path": str(output_path.resolve()),
        "--seed": "7",
    }
    for argument, expected_value in expected_args.items():
        assert argument in captured["cmd"]
        assert captured["cmd"][captured["cmd"].index(argument) + 1] == expected_value
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "8,9"
    assert run_info["python_executable"] == "python"
    assert run_info["repo_root"] == str(repo)
    assert run_info["working_dir"] == str(repo)
    assert run_info["data_dir"] == str(data_dir)
    assert run_info["n_node"] == 5
    assert run_info["requested_topk"] == 3
    assert run_info["effective_topk"] == 3
    assert run_info["expected_test_count"] == 2
    assert run_info["gpu_id"] == "3"


def test_runner_rejects_invalid_prediction_rows(tmp_path) -> None:
    runner = MDHGRunner(_runner_config(tmp_path))
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"topk": 2, "requested_topk": 2, "n_node": 3, "rankings": [[1, 4]]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside 1..3"):
        runner.predict_topk(
            predictions_path=path,
            expected_test_count=1,
            n_node=3,
            requested_topk=2,
        )


def test_runner_includes_diagnostic_args_and_run_info(monkeypatch, tmp_path) -> None:
    config = _with_diagnostics(
        _runner_config(tmp_path),
        epoch_metrics=True,
        per_epoch_predictions=True,
    )
    repo = Path(config.victims.runtime["mdhg"]["repo_root"])
    repo.mkdir()
    (repo / "main.py").write_text("", encoding="utf-8")
    data_dir = tmp_path / "export"
    data_dir.mkdir()
    run_dir = tmp_path / "run"
    output_path = run_dir / "mdhg_topk_raw.json"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "topk": 2,
                    "requested_topk": 2,
                    "n_node": 3,
                    "rankings": [[1, 2]],
                }
            ),
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("attack.models.victim.mdhg_runner.run_subprocess_with_epoch_progress", fake_run)
    run_info = MDHGRunner(config).run(
        data_dir=data_dir,
        dataset_name="toy",
        n_node=3,
        expected_test_count=1,
        run_dir=run_dir,
        export_topk_path=output_path,
        topk=2,
    )

    metrics_path = run_dir / "mdhg_epoch_metrics.jsonl"
    prediction_dir = run_dir / "mdhg_per_epoch_predictions"
    assert captured["cmd"][captured["cmd"].index("--epoch_metrics_output_path") + 1] == str(
        metrics_path.resolve()
    )
    assert captured["cmd"][captured["cmd"].index("--per_epoch_prediction_dir") + 1] == str(
        prediction_dir.resolve()
    )
    assert run_info["epoch_metrics_output_path"] == str(metrics_path.resolve())
    assert run_info["per_epoch_prediction_dir"] == str(prediction_dir.resolve())


def test_runner_omits_diagnostic_args_by_default(monkeypatch, tmp_path) -> None:
    config = _runner_config(tmp_path)
    repo = Path(config.victims.runtime["mdhg"]["repo_root"])
    repo.mkdir()
    (repo / "main.py").write_text("", encoding="utf-8")
    data_dir = tmp_path / "export"
    data_dir.mkdir()
    output_path = tmp_path / "run" / "mdhg_topk_raw.json"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "topk": 1,
                    "requested_topk": 1,
                    "n_node": 2,
                    "rankings": [[1]],
                }
            ),
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr("attack.models.victim.mdhg_runner.run_subprocess_with_epoch_progress", fake_run)
    run_info = MDHGRunner(config).run(
        data_dir=data_dir,
        dataset_name="toy",
        n_node=2,
        expected_test_count=1,
        run_dir=tmp_path / "run",
        export_topk_path=output_path,
        topk=1,
    )

    assert "--epoch_metrics_output_path" not in captured["cmd"]
    assert "--per_epoch_prediction_dir" not in captured["cmd"]
    assert "epoch_metrics_output_path" not in run_info
    assert "per_epoch_prediction_dir" not in run_info


@pytest.mark.parametrize(
    ("payload", "expected_count", "n_node", "requested_topk", "message"),
    [
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "rankings": [[1, 2]],
            },
            2,
            3,
            2,
            "prediction count mismatch",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 2,
                "n_node": 3,
                "rankings": [[1]],
            },
            1,
            3,
            2,
            "length does not match JSON topk",
        ),
        (
            {
                "topk": 2,
                "requested_topk": 5,
                "n_node": 3,
                "rankings": [[1, 2]],
            },
            1,
            3,
            5,
            "does not match requested effective topk",
        ),
        (
            {"topk": 2, "requested_topk": 2, "n_node": 3},
            1,
            3,
            2,
            "must contain a rankings list",
        ),
    ],
)
def test_runner_rejects_malformed_prediction_json(
    tmp_path,
    payload,
    expected_count,
    n_node,
    requested_topk,
    message,
) -> None:
    runner = MDHGRunner(_runner_config(tmp_path))
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        runner.predict_topk(
            predictions_path=path,
            expected_test_count=expected_count,
            n_node=n_node,
            requested_topk=requested_topk,
        )


def test_mdhg_is_registered_and_smoke_config_validates() -> None:
    smoke_path = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "configs"
        / "local_diginetica_valbest_clean_mdhg_append_one_target.yaml"
    )
    config = load_config(smoke_path)
    assert "mdhg" in available_victims()
    assert config.victims.enabled == ("mdhg",)
    assert config.targets.count == 1
    assert "topk" not in config.victims.params["mdhg"]["train"]
    smoke_text = smoke_path.read_text(encoding="utf-8")
    assert "smoke template only" in smoke_text
    assert "not the formal full-result" in smoke_text


def test_mdhg_epoch_diagnostic_config_validates() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "configs"
        / "local_diginetica_valbest_clean_mdhg_epoch_diagnostic_one_target.yaml"
    )
    config = load_config(path)
    diagnostics = config.victims.runtime["mdhg"]["diagnostics"]
    assert config.experiment.name.startswith("diagnostic_")
    assert config.victims.enabled == ("mdhg",)
    assert config.victims.params["mdhg"]["train"]["epochs"] == 20
    assert diagnostics == {"epoch_metrics": True, "per_epoch_predictions": True}
    text = path.read_text(encoding="utf-8")
    assert "not validation-best" in text
    assert "does not test CEM reuse" in text
