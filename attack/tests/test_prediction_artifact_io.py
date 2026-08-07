from __future__ import annotations

import errno
import json
import os
from pathlib import Path

from attack.pipeline.core.evaluator import save_predictions
from attack.pipeline.core.orchestrator import (
    _atomic_link_or_copy,
    _persist_shared_victim_result,
    _save_reused_predictions_payload,
)
from attack.pipeline.core.victim_execution import VictimExecutionResult


def test_save_predictions_uses_compact_atomic_json(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    rankings = [[1, 2, 3], [4, 5, 6]]

    save_predictions(
        path,
        topk=3,
        rankings=rankings,
        victim="srgnn",
        target_item=7,
    )

    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["rankings"] == rankings
    assert payload["target_item"] == 7
    assert "\n" not in text
    assert not list(tmp_path.glob(".predictions.json.*.tmp"))


def test_atomic_link_or_copy_uses_hard_link_on_same_filesystem(
    tmp_path: Path,
) -> None:
    source = tmp_path / "local" / "predictions.json"
    destination = tmp_path / "shared" / "predictions.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"rankings":[[1,2,3]]}', encoding="utf-8")

    _atomic_link_or_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert os.path.samefile(source, destination)


def test_atomic_link_or_copy_falls_back_when_link_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "local" / "predictions.json"
    destination = tmp_path / "shared" / "predictions.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"rankings":[[1,2,3]]}', encoding="utf-8")

    def unavailable_link(_source, _destination) -> None:
        raise OSError(errno.EXDEV, "hard links unavailable")

    monkeypatch.setattr(os, "link", unavailable_link)
    _atomic_link_or_copy(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert not os.path.samefile(source, destination)


def test_shared_prediction_persistence_uses_one_physical_file(tmp_path: Path) -> None:
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    predictions = local / "predictions.json"
    save_predictions(
        predictions,
        topk=3,
        rankings=[[1, 2, 3]],
        victim="srgnn",
        target_item=3,
    )
    artifacts = {
        "shared_dir": shared,
        "predictions": predictions,
        "shared_predictions": shared / "predictions.json",
        "train_history": local / "train_history.json",
        "shared_train_history": shared / "train_history.json",
        "shared_poisoned_train": shared / "poisoned_train.txt",
        "shared_execution_result": shared / "execution_result.json",
    }
    result = VictimExecutionResult(
        predictions=[[1, 2, 3]],
        predictions_path=predictions,
        extra={},
        poisoned_train_path=None,
    )

    _persist_shared_victim_result(
        run_type="attack",
        victim_result=result,
        artifacts=artifacts,
    )

    assert os.path.samefile(predictions, artifacts["shared_predictions"])
    assert json.loads(artifacts["shared_execution_result"].read_text())[
        "predictions_path"
    ] == str(artifacts["shared_predictions"])


def test_reused_predictions_link_when_target_metadata_matches(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared" / "predictions.json"
    local = tmp_path / "local" / "predictions.json"
    shared.parent.mkdir(parents=True)
    payload = {
        "available": True,
        "count": 1,
        "rankings": [[3, 2, 1]],
        "target_item": 3,
        "topk": 3,
        "victim": "srgnn",
    }
    shared.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _save_reused_predictions_payload(
        payload,
        predictions_path=local,
        predictions_source=shared,
        target_item=3,
    )

    assert os.path.samefile(shared, local)
    assert json.loads(local.read_text(encoding="utf-8")) == payload


def test_reused_predictions_rewrite_target_without_mutating_shared(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared" / "predictions.json"
    local = tmp_path / "local" / "predictions.json"
    shared.parent.mkdir(parents=True)
    payload = {
        "available": True,
        "count": 1,
        "rankings": [[3, 2, 1]],
        "target_item": 3,
        "topk": 3,
        "victim": "srgnn",
    }
    shared.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _save_reused_predictions_payload(
        payload,
        predictions_path=local,
        predictions_source=shared,
        target_item=9,
    )

    assert not os.path.samefile(shared, local)
    assert json.loads(shared.read_text(encoding="utf-8"))["target_item"] == 3
    assert json.loads(local.read_text(encoding="utf-8"))["target_item"] == 9
    assert "\n" not in local.read_text(encoding="utf-8")
