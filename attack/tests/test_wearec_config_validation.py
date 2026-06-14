from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

import attack.common.config as config_module
from attack.common.config import (
    _normalize_wearec_train,
    _validate_wearec_runtime,
)
from attack.models.victim.wearec_runner import WEARecRunner, effective_wearec_config
from attack.tests.wearec_test_utils import wearec_config, wearec_train


def _runtime(**overrides) -> dict:
    runtime = {
        "python_executable": "/srv/conda/envs/wearec/bin/python",
        "repo_root": "/srv/benchmark/third_party/wearec",
        "working_dir": "/srv/benchmark/third_party/wearec",
        "device": {"use_gpu": True, "gpu_id": "0"},
        "dataloader": {"num_workers": 0},
        "diagnostics": {"per_epoch_predictions": False},
    }
    runtime.update(overrides)
    return runtime


def test_runtime_schema_accepts_nonexistent_server_paths():
    _validate_wearec_runtime(_runtime(), "victims.runtime.wearec")


def test_runtime_schema_validation_does_not_touch_filesystem(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "Path",
        lambda *_args, **_kwargs: pytest.fail("config validation touched Path"),
    )
    _validate_wearec_runtime(_runtime(), "victims.runtime.wearec")


@pytest.mark.parametrize(
    "field",
    ["python_executable", "repo_root", "working_dir", "device", "dataloader"],
)
def test_missing_required_runtime_fields_fail(field):
    runtime = _runtime()
    del runtime[field]
    with pytest.raises((KeyError, ValueError), match=field):
        _validate_wearec_runtime(runtime, "victims.runtime.wearec")


@pytest.mark.parametrize(
    "field",
    ["python_executable", "repo_root", "working_dir"],
)
def test_empty_runtime_strings_fail(field):
    runtime = _runtime(**{field: "  "})
    with pytest.raises(ValueError, match="non-empty"):
        _validate_wearec_runtime(runtime, "victims.runtime.wearec")


@pytest.mark.parametrize(
    "override",
    [
        {"device": {"use_gpu": "yes", "gpu_id": "0"}},
        {"device": {"use_gpu": True, "gpu_id": "gpu"}},
        {"dataloader": {"num_workers": 1}},
    ],
)
def test_invalid_device_or_worker_schema_fails(override):
    with pytest.raises((TypeError, ValueError)):
        _validate_wearec_runtime(
            _runtime(**override),
            "victims.runtime.wearec",
        )


def test_hidden_size_must_be_divisible_by_num_heads(tmp_path):
    accepted = wearec_train(hidden_size=8, num_heads=2)
    assert _normalize_wearec_train(accepted, "wearec.train")["hidden_size"] == 8

    rejected = wearec_train(hidden_size=10, num_heads=4)
    with pytest.raises(ValueError, match="divisible"):
        _normalize_wearec_train(rejected, "wearec.train")
    with pytest.raises(ValueError, match="divisible"):
        effective_wearec_config(
            wearec_config(
                tmp_path,
                train_overrides={"hidden_size": 10, "num_heads": 4},
            ),
            seed=7,
            requested_topk=5,
        )


def _runner_config(tmp_path, **runtime_overrides):
    repo = Path(__file__).resolve().parents[2] / "third_party" / "wearec"
    return wearec_config(
        tmp_path,
        python_executable=str(Path(sys.executable).resolve()),
        runtime_overrides={
            "repo_root": str(repo.resolve()),
            "working_dir": str(repo.resolve()),
            **runtime_overrides,
        },
    )


def _run_for_path_validation(runner, tmp_path):
    inputs = {}
    for name in ("train", "valid", "test", "metadata"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        inputs[name] = path
    return runner.run(
        train_path=inputs["train"],
        valid_path=inputs["valid"],
        test_path=inputs["test"],
        metadata_path=inputs["metadata"],
        item_count=5,
        expected_test_count=1,
        run_dir=tmp_path / "run",
        prediction_output_path=tmp_path / "run" / "predictions.json",
        requested_topk=5,
        epochs=1,
        victim_train_seed=7,
        target_item=None,
    )


@pytest.mark.parametrize(
    "runtime_overrides",
    [
        {"python_executable": "python"},
        {"repo_root": "third_party/wearec"},
        {"working_dir": "third_party/wearec"},
    ],
)
def test_runner_rejects_relative_runtime_paths(tmp_path, runtime_overrides):
    runner = WEARecRunner(_runner_config(tmp_path, **runtime_overrides))
    with pytest.raises(ValueError, match="absolute path"):
        _run_for_path_validation(runner, tmp_path)


@pytest.mark.parametrize(
    "field",
    ["python_executable", "repo_root", "working_dir"],
)
def test_runner_rejects_absolute_missing_runtime_paths(tmp_path, field):
    missing_values = {
        "python_executable": (
            tmp_path
            / "missing_python"
            / ("python.exe" if os.name == "nt" else "python")
        ).resolve(),
        "repo_root": (tmp_path / "missing_repo").resolve(),
        "working_dir": (tmp_path / "missing_working_dir").resolve(),
    }
    runner = WEARecRunner(
        _runner_config(tmp_path, **{field: str(missing_values[field])})
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        _run_for_path_validation(runner, tmp_path)


def test_runner_rejects_directory_as_python_executable(tmp_path):
    runner = WEARecRunner(
        _runner_config(
            tmp_path,
            python_executable=str(tmp_path.resolve()),
        )
    )
    with pytest.raises(FileNotFoundError, match="python executable"):
        _run_for_path_validation(runner, tmp_path)
