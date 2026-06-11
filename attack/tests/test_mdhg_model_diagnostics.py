from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_mdhg_model_module():
    model_path = Path(__file__).resolve().parents[2] / "third_party" / "mdhg" / "model.py"
    spec = importlib.util.spec_from_file_location("mdhg_phase1c_model", model_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_indexed_ranking_writer_preserves_order_and_epoch_metadata(tmp_path) -> None:
    module = _load_mdhg_model_module()
    indexed = [None, None, None]
    module._store_indexed_rankings(indexed, [2, 0], [[3, 2], [1, 2]])
    module._store_indexed_rankings(indexed, [1, 2], [[2, 1], [3, 2]])
    output_path = tmp_path / "epoch_001_topk.json"

    module._write_indexed_rankings(
        output_path,
        indexed,
        requested_topk=2,
        n_node=3,
        expected_count=3,
        epoch=1,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "epoch": 1,
        "topk": 2,
        "requested_topk": 2,
        "n_node": 3,
        "rankings": [[1, 2], [2, 1], [3, 2]],
    }


def test_indexed_ranking_writer_rejects_conflicting_duplicate(tmp_path) -> None:
    module = _load_mdhg_model_module()
    indexed = [None]
    module._store_indexed_rankings(indexed, [0], [[1, 2]])
    with pytest.raises(RuntimeError, match="inconsistent predictions"):
        module._store_indexed_rankings(indexed, [0], [[2, 1]])


def test_prepare_diagnostic_outputs_truncates_metrics_jsonl(tmp_path) -> None:
    module = _load_mdhg_model_module()
    metrics_path = tmp_path / "nested" / "mdhg_epoch_metrics.jsonl"
    metrics_path.parent.mkdir()
    metrics_path.write_text('{"epoch": 99}\n', encoding="utf-8")
    prediction_dir = tmp_path / "predictions"

    module.prepare_diagnostic_outputs(metrics_path, prediction_dir)

    assert metrics_path.read_text(encoding="utf-8") == ""
    assert prediction_dir.is_dir()
