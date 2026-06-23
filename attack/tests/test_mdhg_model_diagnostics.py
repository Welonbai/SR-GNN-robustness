from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _load_mdhg_model_module():
    model_path = Path(__file__).resolve().parents[2] / "third_party" / "mdhg" / "model.py"
    spec = importlib.util.spec_from_file_location("mdhg_phase1c_model", model_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mdhg_util_module():
    util_path = Path(__file__).resolve().parents[2] / "third_party" / "mdhg" / "util.py"
    spec = importlib.util.spec_from_file_location("mdhg_evaluation_util", util_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _generated_slices(length: int, batch_size: int):
    module = _load_mdhg_util_module()
    data = object.__new__(module.Data)
    data.length = length
    data.shuffle = False
    return data.generate_batch(batch_size)


def _owned_indices(slices, ownership_masks) -> list[int]:
    return [
        int(sample_index)
        for batch_indices, ownership_mask in zip(slices, ownership_masks)
        for sample_index, owns_row in zip(batch_indices, ownership_mask)
        if owns_row
    ]


def test_evaluation_ownership_without_final_batch_overlap() -> None:
    module = _load_mdhg_model_module()
    slices = _generated_slices(length=8, batch_size=4)

    ownership_masks = module._build_evaluation_ownership_masks(slices, expected_count=8)

    assert all(np.all(mask) for mask in ownership_masks)
    assert _owned_indices(slices, ownership_masks) == list(range(8))


def test_evaluation_ownership_prefers_last_occurrence_for_small_remainder() -> None:
    module = _load_mdhg_model_module()
    slices = _generated_slices(length=10, batch_size=4)

    ownership_masks = module._build_evaluation_ownership_masks(slices, expected_count=10)

    assert [batch.tolist() for batch in slices] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [6, 7, 8, 9],
    ]
    assert [mask.tolist() for mask in ownership_masks] == [
        [True, True, True, True],
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert sorted(_owned_indices(slices, ownership_masks)) == list(range(10))


def test_remainder_one_uses_unique_canonical_rows_with_context_dependent_rankings() -> None:
    module = _load_mdhg_model_module()
    slices = _generated_slices(length=9, batch_size=4)
    ownership_masks = module._build_evaluation_ownership_masks(slices, expected_count=9)
    indexed = [None] * 9

    for batch_number, (batch_indices, ownership_mask) in enumerate(
        zip(slices, ownership_masks)
    ):
        owned_indices = np.asarray(batch_indices)[ownership_mask]
        rankings = [
            [int(sample_index) + 1, 100 + batch_number]
            for sample_index in owned_indices
        ]
        module._store_indexed_rankings(indexed, owned_indices, rankings)

    assert [batch.tolist() for batch in slices] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ]
    assert sorted(_owned_indices(slices, ownership_masks)) == list(range(9))
    assert all(ranking is not None for ranking in indexed)
    assert indexed[5] == [6, 102]
    assert indexed[7] == [8, 102]


def test_owned_targets_scores_and_export_indices_stay_aligned() -> None:
    module = _load_mdhg_model_module()
    slices = _generated_slices(length=10, batch_size=4)
    ownership_masks = module._build_evaluation_ownership_masks(slices, expected_count=10)
    metric_sample_indices = []
    export_sample_indices = []

    for batch_indices, ownership_mask in zip(slices, ownership_masks):
        batch_indices = np.asarray(batch_indices)
        targets = torch.tensor(batch_indices + 100, dtype=torch.long)
        scores = torch.tensor(
            [[float(sample_index), -float(sample_index)] for sample_index in batch_indices]
        )
        owned_indices, owned_targets, owned_scores = (
            module._select_owned_evaluation_rows(
                batch_indices,
                ownership_mask,
                targets,
                scores,
            )
        )
        export_sample_indices.extend(owned_indices.tolist())
        metric_sample_indices.extend((owned_targets - 100).tolist())
        assert owned_scores[:, 0].tolist() == owned_indices.astype(float).tolist()

    assert metric_sample_indices == export_sample_indices
    assert sorted(export_sample_indices) == list(range(10))


def test_train_test_uses_same_unique_rows_for_metrics_and_export(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_mdhg_model_module()
    slices = _generated_slices(length=9, batch_size=4)

    class EmptyTrainData:
        def generate_batch(self, batch_size):
            assert batch_size == 4
            return []

    class OverlappingTestData:
        raw = list(range(9))

        def generate_batch(self, batch_size):
            assert batch_size == 4
            return slices

    class FakeModel:
        batch_size = 4
        n_node = 50

        def eval(self):
            return self

    def fake_forward(model, batch_indices, data, epoch, train):
        assert train is False
        batch_indices = np.asarray(batch_indices)
        targets = torch.tensor(batch_indices, dtype=torch.long)
        scores = torch.full((len(batch_indices), model.n_node), -1000.0)
        context_offset = int(batch_indices[0])
        for row_index, sample_index in enumerate(batch_indices):
            sample_index = int(sample_index)
            context_item = (sample_index + context_offset + 1) % model.n_node
            scores[row_index, sample_index] = 100.0
            scores[row_index, context_item] = 90.0
        return targets, scores, 0, 0, 0

    monkeypatch.setattr(module, "forward", fake_forward)
    output_path = tmp_path / "mdhg_topk_raw.json"

    metrics, total_loss = module.train_test(
        FakeModel(),
        EmptyTrainData(),
        OverlappingTestData(),
        epoch=0,
        prediction_output_path=output_path,
        requested_topk=2,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert total_loss == 0.0
    assert len(payload["rankings"]) == 9
    assert payload["rankings"][5] == [6, 12]
    for metric_values in metrics.values():
        assert len(metric_values) == 9
    for K in (5, 10, 20, 50):
        assert metrics[f"hit{K}"] == [True] * 9
        assert metrics[f"mrr{K}"] == [1.0] * 9
        assert all(value > 0.0 for value in metrics[f"ndcg{K}"])


def test_indexed_ranking_writer_preserves_order_and_epoch_metadata(tmp_path) -> None:
    module = _load_mdhg_model_module()
    indexed = [None, None, None]
    module._store_indexed_rankings(indexed, [2, 0], [[3, 2], [1, 2]])
    module._store_indexed_rankings(indexed, [1], [[2, 1]])
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
        "test_count": 3,
        "prediction_count": 3,
        "rankings": [[1, 2], [2, 1], [3, 2]],
    }


def test_indexed_ranking_writer_rejects_any_true_canonical_duplicate() -> None:
    module = _load_mdhg_model_module()
    indexed = [None]
    module._store_indexed_rankings(indexed, [0], [[1, 2]])
    with pytest.raises(RuntimeError, match="Canonical evaluation ownership.*duplicate"):
        module._store_indexed_rankings(indexed, [0], [[1, 2]])


def test_evaluation_ownership_and_export_reject_missing_indices(tmp_path) -> None:
    module = _load_mdhg_model_module()
    with pytest.raises(RuntimeError, match="ownership is missing.*index=2"):
        module._build_evaluation_ownership_masks(
            [np.array([0, 1]), np.array([1])],
            expected_count=3,
        )

    with pytest.raises(RuntimeError, match="export is missing.*index=1"):
        module._write_indexed_rankings(
            tmp_path / "missing.json",
            [[1], None],
            requested_topk=1,
            n_node=2,
            expected_count=2,
        )


def test_prepare_diagnostic_outputs_truncates_metrics_jsonl(tmp_path) -> None:
    module = _load_mdhg_model_module()
    metrics_path = tmp_path / "nested" / "mdhg_epoch_metrics.jsonl"
    metrics_path.parent.mkdir()
    metrics_path.write_text('{"epoch": 99}\n', encoding="utf-8")
    prediction_dir = tmp_path / "predictions"

    module.prepare_diagnostic_outputs(metrics_path, prediction_dir)

    assert metrics_path.read_text(encoding="utf-8") == ""
    assert prediction_dir.is_dir()
