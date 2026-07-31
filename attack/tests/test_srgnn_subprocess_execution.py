from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import pytest

from attack.common.artifact_io import load_json
from attack.common.config import load_config
from attack.common.paths import PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE
from attack.models.victim import srgnn_subprocess
from attack.pipeline.core import victim_execution
from attack.pipeline.core.evaluator import save_predictions


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_generated_popular_all_victims.yaml"
)


def _write_pairs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(([[1, 2], [2, 3]], [3, 4]), handle)


def test_srgnn_subprocess_preserves_fixed_last_training_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    poisoned_train_path = tmp_path / "poisoned_train.txt"
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    for path in (poisoned_train_path, valid_path, test_path):
        _write_pairs(path)
    calls: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, config):
            calls["config"] = config
            self.train_loss_history: list[float] = []

        def build_model(self, opt) -> None:
            calls["opt"] = opt

        def load_dataset(self, *, train_path, test_path, shuffle_train=True):
            calls.setdefault("loads", []).append(
                (Path(train_path), Path(test_path), bool(shuffle_train))
            )
            return f"train:{train_path}", f"test:{test_path}"

        def train(self, train_data, valid_data, epochs, *, target_item, topk) -> None:
            calls["train"] = (train_data, valid_data, epochs, target_item, topk)
            self.train_loss_history = [2.0, 1.0, 0.5, 0.25]

        def predict_topk(self, test_data, *, topk):
            calls["predict"] = (test_data, topk)
            return [[3, 2, 1], [4, 3, 2]]

    monkeypatch.setattr(srgnn_subprocess, "SRGNNVictimRunner", FakeRunner)
    run_dir = tmp_path / "run"
    predictions_path = run_dir / "predictions.json"

    summary = srgnn_subprocess.run_srgnn_victim_subprocess(
        config_path=CONFIG_PATH,
        poisoned_train_path=poisoned_train_path,
        valid_path=valid_path,
        test_path=test_path,
        run_dir=run_dir,
        predictions_path=predictions_path,
        target_item=7759,
        topk=50,
        seed=20260405,
        clean_run=False,
    )

    assert calls["train"][2:] == (4, 7759, 50)
    assert calls["loads"] == [
        (poisoned_train_path, valid_path, True),
        (poisoned_train_path, test_path, False),
    ]
    assert calls["predict"][1] == 50
    assert load_json(predictions_path)["rankings"] == [[3, 2, 1], [4, 3, 2]]
    assert load_json(run_dir / "train_history.json")["train_loss"] == [
        2.0,
        1.0,
        0.5,
        0.25,
    ]
    assert summary["epochs_requested"] == summary["epochs_completed"] == 4
    assert summary["execution_mode"] == "isolated_subprocess"


def test_srgnn_victim_runs_in_isolated_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    run_dir = tmp_path / "victim"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_pairs(valid_path)
    _write_pairs(test_path)
    captured: dict[str, object] = {}

    def fake_run_subprocess(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured.update(kwargs)
        predictions_arg = Path(cmd[cmd.index("--predictions") + 1])
        save_predictions(
            predictions_arg,
            topk=50,
            rankings=[[3, 2, 1], [4, 3, 2]],
            victim="srgnn",
            target_item=7759,
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        victim_execution,
        "run_subprocess_with_epoch_progress",
        fake_run_subprocess,
    )
    monkeypatch.setattr(
        victim_execution,
        "_victim_stage_seed",
        lambda *args, **kwargs: 20260405,
    )

    result = victim_execution.execute_single_victim(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
        victim_name="srgnn",
        canonical_dataset=object(),
        poisoned_sessions=[[1, 2], [2, 3]],
        poisoned_labels=[3, 4],
        raw_fake_sessions=[],
        run_dir=run_dir,
        poisoned_train_path=run_dir / "poisoned_train.txt",
        target_item=7759,
        eval_topk=[20, 50],
        srg_nn_export_paths={"valid": valid_path, "test": test_path},
        predictions_path=run_dir / "predictions.json",
    )

    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[:3] == [
        victim_execution.sys.executable,
        "-m",
        "attack.models.victim.srgnn_subprocess",
    ]
    assert cmd[cmd.index("--seed") + 1] == "20260405"
    assert cmd[cmd.index("--target-item") + 1] == "7759"
    assert cmd[cmd.index("--topk") + 1] == "50"
    assert captured["log_path"] == run_dir / "srgnn_stdout.log"
    assert captured["epoch_numbers_are_one_based"] is True
    assert captured["env"]["PYTHONUNBUFFERED"] == "1"
    assert result.predictions == [[3, 2, 1], [4, 3, 2]]
    assert result.predictions_path == run_dir / "predictions.json"
    assert result.extra == {"execution_mode": "isolated_subprocess"}
    assert result.poisoned_train_path == run_dir / "poisoned_train.txt"


def test_srgnn_victim_reports_subprocess_log_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    run_dir = tmp_path / "victim"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        CONFIG_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    valid_path = tmp_path / "valid.txt"
    test_path = tmp_path / "test.txt"
    _write_pairs(valid_path)
    _write_pairs(test_path)
    monkeypatch.setattr(
        victim_execution,
        "run_subprocess_with_epoch_progress",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 7),
    )
    monkeypatch.setattr(
        victim_execution,
        "_victim_stage_seed",
        lambda *args, **kwargs: 20260405,
    )

    with pytest.raises(RuntimeError, match=r"code 7.*srgnn_stdout\.log"):
        victim_execution.execute_single_victim(
            config,
            run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
            victim_name="srgnn",
            canonical_dataset=object(),
            poisoned_sessions=[[1, 2]],
            poisoned_labels=[3],
            raw_fake_sessions=[],
            run_dir=run_dir,
            poisoned_train_path=run_dir / "poisoned_train.txt",
            target_item=7759,
            eval_topk=[50],
            srg_nn_export_paths={"valid": valid_path, "test": test_path},
            predictions_path=run_dir / "predictions.json",
        )
