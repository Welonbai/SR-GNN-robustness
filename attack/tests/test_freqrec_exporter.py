from __future__ import annotations

import json

import pytest

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.freqrec_exporter import FreqRecExporter
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.pipeline.core.pipeline_utils import build_clean_pairs
from attack.tests.freqrec_test_utils import CONFIG_PATH
from attack.common.config import load_config


def _dataset():
    return CanonicalDataset(
        train_sub=[[1, 2, 3], [2, 4]],
        valid=[[1, 2, 3], [4, 5]],
        test=[[2, 3, 5], [1, 4]],
        item_map={str(index): index for index in range(1, 6)},
        metadata={"dataset_name": "toy", "item_count": 5, "counts": {"items": 5}},
    )


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_export_uses_authoritative_train_pairs_and_exact_validation_order(tmp_path):
    dataset = _dataset()
    clean_prefixes, clean_labels = build_clean_pairs(dataset)
    result = FreqRecExporter().export_with_train_pairs(
        dataset,
        train_prefixes=clean_prefixes,
        train_labels=clean_labels,
        output_dir=tmp_path,
        dataset_name="toy",
        max_seq_length=5,
        mode="clean",
    )

    assert _jsonl(result.files["train"]) == [
        {"example_id": i, "input_prefix": prefix, "label": label}
        for i, (prefix, label) in enumerate(zip(clean_prefixes, clean_labels))
    ]
    assert _jsonl(result.files["valid"]) == [
        {"example_id": 0, "input_prefix": [1, 2], "label": 3},
        {"example_id": 1, "input_prefix": [1], "label": 2},
        {"example_id": 2, "input_prefix": [4], "label": 5},
    ]
    test_rows = _jsonl(result.files["test"])
    labels = resolve_ground_truth_labels(
        load_config(CONFIG_PATH),
        victim_name="freqrec",
        canonical_dataset=dataset,
        predictions=[[1]] * len(test_rows),
    )
    assert [row["label"] for row in test_rows] == labels
    assert result.item_count == 5
    assert result.observed_max_item_id == 5


def test_poisoned_export_keeps_pairs_without_expansion_or_deduplication(tmp_path):
    result = FreqRecExporter().export_with_train_pairs(
        _dataset(),
        train_prefixes=[[1], [1], [2, 5]],
        train_labels=[5, 5, 4],
        output_dir=tmp_path,
        dataset_name="toy",
        max_seq_length=3,
        mode="poisoned",
    )
    assert _jsonl(result.files["train"]) == [
        {"example_id": 0, "input_prefix": [1], "label": 5},
        {"example_id": 1, "input_prefix": [1], "label": 5},
        {"example_id": 2, "input_prefix": [2, 5], "label": 4},
    ]


@pytest.mark.parametrize("prefix,label", [([0], 1), ([-1], 1), ([1], 6), ([True], 1)])
def test_export_rejects_invalid_canonical_ids(tmp_path, prefix, label):
    with pytest.raises((TypeError, ValueError)):
        FreqRecExporter().export_with_train_pairs(
            _dataset(),
            train_prefixes=[prefix],
            train_labels=[label],
            output_dir=tmp_path,
            dataset_name="toy",
            max_seq_length=5,
            mode="poisoned",
        )
