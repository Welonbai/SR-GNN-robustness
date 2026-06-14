from __future__ import annotations

import json

import pytest

from attack.models.victim.wearec_diagnostics import (
    atomic_write_json,
    validate_wearec_per_epoch_predictions,
)
from attack.models.victim.wearec_runner import effective_wearec_config
from attack.tests.wearec_test_utils import raw_prediction_payload, wearec_config


def _rows():
    metrics = {
        f"{metric}@{cutoff}": 1.0
        for cutoff in (1, 3, 5)
        for metric in ("hr", "mrr", "ndcg")
    }
    return [
        {"epoch": epoch, "train_loss": 1.0 / epoch, "valid": dict(metrics)}
        for epoch in (1, 2)
    ]


def _epoch_payload(epoch):
    payload = raw_prediction_payload()
    payload["split"] = "valid"
    payload["current_epoch"] = epoch
    payload["epochs_requested"] = 2
    payload["epochs_completed"] = epoch
    payload.pop("final_epoch")
    payload.pop("selected_epoch")
    payload["valid_metrics"] = payload.pop("test_metrics")
    return payload


def _write_epochs(directory):
    directory.mkdir(exist_ok=True)
    for epoch in (1, 2):
        (directory / f"epoch_{epoch:03d}_validation_topk.json").write_text(
            json.dumps(_epoch_payload(epoch)),
            encoding="utf-8",
        )


def _validate(tmp_path):
    return validate_wearec_per_epoch_predictions(
        tmp_path,
        configured_epochs=2,
        expected_labels=[1, 2, 3],
        item_count=5,
        effective_config=effective_wearec_config(
            wearec_config(tmp_path), seed=7, requested_topk=5
        ),
        dataset_name="toy",
        training_mode="clean",
        diagnostic_rows=_rows(),
    )


def test_atomic_json_replaces_complete_destination(tmp_path, monkeypatch):
    destination = tmp_path / "summary.json"
    destination.write_text('{"old":true}\n', encoding="utf-8")
    calls = []
    from attack.models.victim import wearec_diagnostics

    real_replace = wearec_diagnostics.os.replace

    def recording_replace(source, target):
        calls.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(wearec_diagnostics.os, "replace", recording_replace)
    atomic_write_json({"new": True}, destination)
    assert json.loads(destination.read_text()) == {"new": True}
    assert calls and calls[-1][1] == destination
    assert not list(tmp_path.glob("*.tmp"))


def test_exact_epoch_prediction_set_is_accepted(tmp_path):
    _write_epochs(tmp_path)
    assert [path.name for path in _validate(tmp_path)] == [
        "epoch_001_validation_topk.json",
        "epoch_002_validation_topk.json",
    ]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_epoch_prediction_is_rejected(tmp_path, mutation):
    _write_epochs(tmp_path)
    if mutation == "missing":
        (tmp_path / "epoch_002_validation_topk.json").unlink()
    else:
        (tmp_path / "epoch_003_validation_topk.json").write_text(
            json.dumps(_epoch_payload(2)), encoding="utf-8"
        )
    with pytest.raises(ValueError, match="exactly"):
        _validate(tmp_path)


def test_wrong_validation_label_is_rejected(tmp_path):
    _write_epochs(tmp_path)
    path = tmp_path / "epoch_001_validation_topk.json"
    payload = json.loads(path.read_text())
    payload["rankings"][0]["label"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="label alignment"):
        _validate(tmp_path)


def test_malformed_intermediate_lifecycle_is_rejected(tmp_path):
    _write_epochs(tmp_path)
    path = tmp_path / "epoch_001_validation_topk.json"
    payload = json.loads(path.read_text())
    payload["final_epoch"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final/best"):
        _validate(tmp_path)
