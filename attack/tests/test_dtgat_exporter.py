from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.dtgat_exporter import (
    DTGATExporter,
    SYNTHETIC_TEMPORAL_POLICY,
    sequences_to_pairs,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset


def _dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train_sub=[[1, 2, 3], [2, 4]],
        valid=[[1, 4]],
        test=[[1, 3, 4], [2, 3]],
        item_map={str(index): index for index in range(1, 6)},
        metadata={"dataset_name": "toy", "item_count": 5, "counts": {"items": 5}},
    )


def _read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def test_static_clean_export_smoke_generates_dtgat_files_without_training(tmp_path: Path) -> None:
    result = DTGATExporter().export(_dataset(), tmp_path)

    assert set(result.files) == {
        "train",
        "test",
        "all_train_seq",
        "metadata",
        "eval_rows",
    }
    assert result.files["train"].as_posix().endswith("toy/processed_data/train.txt")
    assert result.files["test"].as_posix().endswith("toy/processed_data/test.txt")
    assert result.files["all_train_seq"].as_posix().endswith(
        "toy/processed_data/all_train_seq.txt"
    )

    train_prefixes, train_intervals, train_labels, train_stamps = _read_pickle(
        result.files["train"]
    )
    test_prefixes, test_intervals, test_labels, test_stamps = _read_pickle(
        result.files["test"]
    )
    assert (train_prefixes, train_labels) == sequences_to_pairs(_dataset().train_sub)
    assert (test_prefixes, test_labels) == (
        [[1, 3], [1], [2]],
        [4, 3, 3],
    )
    assert train_intervals == [[1000, 1000], [1000], [1000]]
    assert test_intervals == [[1000, 1000], [1000], [1000]]
    assert train_stamps == [0, 86400, 172800]
    assert test_stamps == [0, 86400, 172800]
    assert _read_pickle(result.files["all_train_seq"]) == _dataset().train_sub

    eval_rows = json.loads(result.files["eval_rows"].read_text(encoding="utf-8"))
    assert eval_rows == [
        {"example_id": 0, "input_prefix": [1, 3], "label": 4},
        {"example_id": 1, "input_prefix": [1], "label": 3},
        {"example_id": 2, "input_prefix": [2], "label": 3},
    ]
    metadata = json.loads(result.files["metadata"].read_text(encoding="utf-8"))
    assert metadata["temporal_policy"] == SYNTHETIC_TEMPORAL_POLICY
    assert metadata["counts"]["train_examples"] == 3
    assert metadata["counts"]["test_examples"] == 3
    assert metadata["validation"]["all_train_seq_full_sessions_not_expanded_rows"] is True
    assert result.n_node == 5


def test_poisoned_export_keeps_authoritative_pairs_and_full_fake_sessions(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    fake_sessions = [[4, 5, 3]]
    clean_prefixes, clean_labels = sequences_to_pairs(dataset.train_sub)
    poisoned = build_poisoned_dataset(clean_prefixes, clean_labels, fake_sessions)

    result = DTGATExporter().export_with_poisoned_train(
        dataset,
        poisoned_prefixes=poisoned.sessions,
        poisoned_labels=poisoned.labels,
        raw_fake_sessions=fake_sessions,
        output_dir=tmp_path,
        dataset_name="toy",
    )

    train_prefixes, _, train_labels, _ = _read_pickle(result.files["train"])
    assert (train_prefixes, train_labels) == (poisoned.sessions, poisoned.labels)
    assert train_prefixes[-2:] == [[4, 5], [4]]
    assert train_labels[-2:] == [3, 5]
    # The graph/padding context stays as full sessions; expanded fake prefixes
    # are only written to train.txt so the resolved pair budget is preserved.
    assert _read_pickle(result.files["all_train_seq"]) == [
        [1, 2, 3],
        [2, 4],
        [4, 5, 3],
    ]
    assert result.expected_fake_expanded_pair_count == 2
    assert result.fake_pairs_present_in_train is True


def test_poisoned_export_reports_missing_fake_pair_expansion_without_recomputing_budget(
    tmp_path: Path,
) -> None:
    result = DTGATExporter().export_with_poisoned_train(
        _dataset(),
        poisoned_prefixes=[[1], [2, 4]],
        poisoned_labels=[2, 3],
        raw_fake_sessions=[[4, 5, 3]],
        output_dir=tmp_path,
        dataset_name="toy",
    )

    assert result.fake_pairs_present_in_train is False
    metadata = json.loads(result.files["metadata"].read_text(encoding="utf-8"))
    assert metadata["validation"]["fake_pairs_present_in_train"] is False


def test_export_rejects_invalid_ids_and_too_short_all_train_seq(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="1..5"):
        DTGATExporter().export_with_poisoned_train(
            _dataset(),
            poisoned_prefixes=[[1]],
            poisoned_labels=[6],
            raw_fake_sessions=[],
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="must be at least every train/test prefix"):
        DTGATExporter().export_with_train_pairs(
            _dataset(),
            train_prefixes=[[1, 2, 3]],
            train_labels=[4],
            raw_train_sessions=[[1, 2]],
            output_dir=tmp_path,
            dataset_name="toy",
            mode="clean",
        )


def test_export_rejects_misaligned_manual_payload_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="equal length"):
        DTGATExporter().export_with_train_pairs(
            _dataset(),
            train_prefixes=[[1], [2]],
            train_labels=[3],
            raw_train_sessions=[[1, 2]],
            output_dir=tmp_path,
            dataset_name="toy",
            mode="clean",
        )
