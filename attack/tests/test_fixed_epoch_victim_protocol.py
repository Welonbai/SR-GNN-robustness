from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json, save_json
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    victim_prediction_key,
    victim_prediction_key_payload,
)
from attack.models.victim.miasrec_runner import MiaSRecRunner
from attack.models.victim.tron_runner import TRONRunner


BASE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_ratio1_partial4_sample.yaml"
)
FAST_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_ratio1_srgnn_miasrec_tron_fixed_epoch_fast_sample2.yaml"
)
RUN_TYPE = PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE


def test_fixed_epoch_fast_config_loads() -> None:
    config = load_config(FAST_CONFIG_PATH)

    miasrec_train = config.victims.params["miasrec"]["train"]
    tron_train = config.victims.params["tron"]["train"]

    assert miasrec_train["epochs"] == 6
    assert miasrec_train["checkpoint_protocol"] == "fixed_epoch"
    assert miasrec_train["validation_enabled"] is False
    assert miasrec_train["export_model"] == "last"

    assert tron_train["epochs"] == 4
    assert tron_train["max_epochs"] == 4
    assert tron_train["checkpoint_protocol"] == "fixed_epoch"
    assert tron_train["validation_enabled"] is False
    assert tron_train["export_model"] == "last"


def test_validation_best_defaults_preserve_old_behavior() -> None:
    config = load_config(BASE_CONFIG_PATH)

    miasrec_train = config.victims.params["miasrec"]["train"]
    tron_train = config.victims.params["tron"]["train"]

    assert miasrec_train["checkpoint_protocol"] == "validation_best"
    assert miasrec_train["validation_enabled"] is True
    assert miasrec_train["export_model"] == "best"

    assert tron_train["checkpoint_protocol"] == "validation_best"
    assert tron_train["validation_enabled"] is True
    assert tron_train["export_model"] == "last"
    assert tron_train["epochs"] == tron_train["max_epochs"] == 3


def test_invalid_validation_disabled_export_best_fails(tmp_path: Path) -> None:
    payload = yaml.safe_load(FAST_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["victims"]["params"]["miasrec"]["train"]["checkpoint_protocol"] = "validation_best"
    payload["victims"]["params"]["miasrec"]["train"]["validation_enabled"] = False
    payload["victims"]["params"]["miasrec"]["train"]["export_model"] = "best"
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="validation_enabled=false.*export_model=best"):
        load_config(config_path)


def test_invalid_fixed_epoch_export_best_fails(tmp_path: Path) -> None:
    payload = yaml.safe_load(FAST_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["victims"]["params"]["tron"]["train"]["checkpoint_protocol"] = "fixed_epoch"
    payload["victims"]["params"]["tron"]["train"]["validation_enabled"] = True
    payload["victims"]["params"]["tron"]["train"]["export_model"] = "best"
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fixed_epoch requires export_model=last"):
        load_config(config_path)


def test_victim_cache_identity_uses_protocol_but_not_validation_enabled() -> None:
    base = load_config(BASE_CONFIG_PATH)
    fast = load_config(FAST_CONFIG_PATH)
    fixed_miasrec = replace(
        base,
        victims=replace(
            base.victims,
            params={**base.victims.params, "miasrec": fast.victims.params["miasrec"]},
        ),
    )
    fixed_validation_enabled = replace(
        fixed_miasrec,
        victims=replace(
            fixed_miasrec.victims,
            params={
                **fixed_miasrec.victims.params,
                "miasrec": {
                    **fixed_miasrec.victims.params["miasrec"],
                    "train": {
                        **fixed_miasrec.victims.params["miasrec"]["train"],
                        "validation_enabled": True,
                    },
                },
            },
        ),
    )
    fixed_epochs_12 = replace(
        fixed_miasrec,
        victims=replace(
            fixed_miasrec.victims,
            params={
                **fixed_miasrec.victims.params,
                "miasrec": {
                    **fixed_miasrec.victims.params["miasrec"],
                    "train": {
                        **fixed_miasrec.victims.params["miasrec"]["train"],
                        "epochs": 12,
                    },
                },
            },
        ),
    )

    assert victim_prediction_key(base, "miasrec", run_type=RUN_TYPE) != victim_prediction_key(
        fixed_miasrec,
        "miasrec",
        run_type=RUN_TYPE,
    )
    assert victim_prediction_key(
        fixed_miasrec,
        "miasrec",
        run_type=RUN_TYPE,
    ) == victim_prediction_key(
        fixed_validation_enabled,
        "miasrec",
        run_type=RUN_TYPE,
    )
    assert victim_prediction_key(
        fixed_miasrec,
        "miasrec",
        run_type=RUN_TYPE,
    ) != victim_prediction_key(
        fixed_epochs_12,
        "miasrec",
        run_type=RUN_TYPE,
    )

    payload = victim_prediction_key_payload(fixed_miasrec, "miasrec", run_type=RUN_TYPE)
    assert payload["victim_training_protocol"] == {
        "checkpoint_protocol": "fixed_epoch",
        "export_model": "last",
        "epochs": 6,
    }
    assert "validation_enabled" not in payload["victim_params"]["train"]


def test_miasrec_fixed_mode_override_disables_validation_and_load_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(FAST_CONFIG_PATH)
    export_root = _export_root(tmp_path, "miasrec")
    run_dir = tmp_path / "miasrec_run"
    raw_predictions = run_dir / "miasrec_topk_raw.json"
    observed: dict[str, object] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        override_path = Path(cmd[cmd.index("--config2") + 1])
        override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
        observed.update(override)
        save_json({"topk": 20, "rankings": [[1, 2, 3]]}, Path(override["export_topk_path"]))
        stdout.write("epoch 0 training [time: 1.0s, train loss: 1.0]\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("attack.models.victim.miasrec_runner.subprocess.run", fake_run)

    run_info = MiaSRecRunner(config).run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=run_dir,
        export_topk_path=raw_predictions,
        topk=20,
    )

    assert observed["checkpoint_protocol"] == "fixed_epoch"
    assert observed["validation_enabled"] is False
    assert observed["export_model"] == "last"
    assert observed["load_best_model_for_export"] is False
    assert run_info["used_best_checkpoint_for_export"] is False
    assert run_info["validation_metrics_recorded"] is False


def test_miasrec_diagnostic_forces_validation_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(FAST_CONFIG_PATH)
    export_root = _export_root(tmp_path, "miasrec")
    run_dir = tmp_path / "miasrec_diagnostic"
    raw_predictions = run_dir / "miasrec_topk_raw.json"
    observed: dict[str, object] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        override_path = Path(cmd[cmd.index("--config2") + 1])
        override = yaml.safe_load(override_path.read_text(encoding="utf-8"))
        observed.update(override)
        save_json({"topk": 20, "rankings": [[1, 2, 3]]}, Path(override["export_topk_path"]))
        stdout.write("epoch 0 training [time: 1.0s, train loss: 1.0]\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("attack.models.victim.miasrec_runner.subprocess.run", fake_run)

    MiaSRecRunner(config).run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=run_dir,
        export_topk_path=raw_predictions,
        topk=20,
        diagnostic_epoch_metrics_path=run_dir / "epoch_metrics.jsonl",
        diagnostic_summary_path=run_dir / "summary.json",
    )

    assert observed["checkpoint_protocol"] == "validation_best"
    assert observed["validation_enabled"] is True
    assert observed["export_model"] == "best"
    assert observed["load_best_model_for_export"] is True


def test_tron_fixed_mode_config_disables_validation_and_checkpoint_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(FAST_CONFIG_PATH)
    export_root = _export_root(tmp_path, "tron")
    run_dir = tmp_path / "tron_run"
    raw_predictions = run_dir / "tron_topk_raw.json"
    observed: dict[str, object] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        config_dir = Path(cmd[cmd.index("--config-dir") + 1])
        config_path = config_dir / f"{cmd[cmd.index('--config-filename') + 1]}.json"
        tron_config = load_json(config_path)
        observed.update(tron_config)
        save_json({"topk": 20, "rankings": [[1, 2, 3]]}, Path(tron_config["export_topk_path"]))
        stdout.write("trainer.fit without validation\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("attack.models.victim.tron_runner.subprocess.run", fake_run)

    run_info = TRONRunner(config).run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=run_dir,
        export_topk_path=raw_predictions,
        topk=20,
    )

    assert observed["checkpoint_protocol"] == "fixed_epoch"
    assert observed["validation_enabled"] is False
    assert observed["export_model"] == "last"
    assert observed["checkpoint_monitor_enabled"] is False
    assert observed["max_epochs"] == 4
    assert run_info["selected_checkpoint_path"] is None
    assert run_info["used_best_checkpoint_for_export"] is False


def test_tron_diagnostic_forces_validation_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(FAST_CONFIG_PATH)
    export_root = _export_root(tmp_path, "tron")
    run_dir = tmp_path / "tron_diagnostic"
    raw_predictions = run_dir / "tron_topk_raw.json"
    observed: dict[str, object] = {}

    def fake_run(cmd, cwd, env, stdout, stderr, check):
        config_dir = Path(cmd[cmd.index("--config-dir") + 1])
        config_path = config_dir / f"{cmd[cmd.index('--config-filename') + 1]}.json"
        tron_config = load_json(config_path)
        observed.update(tron_config)
        save_json({"topk": 20, "rankings": [[1, 2, 3]]}, Path(tron_config["export_topk_path"]))
        stdout.write("trainer.fit with validation\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("attack.models.victim.tron_runner.subprocess.run", fake_run)

    TRONRunner(config).run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=run_dir,
        export_topk_path=raw_predictions,
        topk=20,
        diagnostic_summary_path=run_dir / "summary.json",
    )

    assert observed["checkpoint_protocol"] == "validation_best"
    assert observed["validation_enabled"] is True
    assert observed["export_model"] == "last"
    assert observed["checkpoint_monitor_enabled"] is True


def _export_root(tmp_path: Path, model_name: str) -> Path:
    export_root = tmp_path / "export" / model_name
    dataset_dir = export_root / "diginetica"
    dataset_dir.mkdir(parents=True)
    return export_root
