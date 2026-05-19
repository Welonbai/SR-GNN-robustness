from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import pickle

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


def load_srg_nn_train(path: str | Path) -> tuple[list[list[int]], list[int]]:
    data = pickle.load(Path(path).open("rb"))
    if not (isinstance(data, (list, tuple)) and len(data) == 2):
        raise ValueError("Expected SR-GNN train format: (sessions, labels).")
    sessions, labels = data
    return [list(s) for s in sessions], [int(l) for l in labels]


def save_srg_nn_train(path: str | Path, sessions: Sequence[Sequence[int]], labels: Sequence[int]) -> None:
    if len(sessions) != len(labels):
        raise ValueError("sessions and labels must be the same length.")
    data = (list(map(list, sessions)), list(map(int, labels)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, path.open("wb"))


def _max_item_id_from_sequences(*groups: Sequence[Sequence[int]]) -> int:
    max_item = 0
    for group in groups:
        for sequence in group:
            if sequence:
                max_item = max(max_item, max(int(item) for item in sequence))
    return int(max_item)


def _write_export_metadata(dataset: CanonicalDataset, output_dir: Path) -> Path:
    observed_max_item_id = _max_item_id_from_sequences(
        dataset.train_sub,
        dataset.valid,
        dataset.test,
    )
    item_map = getattr(dataset, "item_map", None)
    if isinstance(item_map, dict):
        item_count = int(len(item_map))
        max_item_id = int(max(item_map.values(), default=0))
    else:
        item_count = int(observed_max_item_id)
        max_item_id = int(observed_max_item_id)
    metadata_path = output_dir / "export_metadata.json"
    payload = {
        "dataset_name": dataset.metadata.get("dataset_name", "dataset"),
        "item_count": item_count,
        "max_item_id": max_item_id,
        "observed_max_item_id": observed_max_item_id,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return metadata_path


def _session_sequences_to_pairs(sequences: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    sessions: list[list[int]] = []
    labels: list[int] = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            labels.append(int(seq[-i]))
            sessions.append(list(seq[:-i]))
    return sessions, labels


class SRGNNExporter(BaseExporter):
    name = "srgnn"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> ExportResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_sessions, train_labels = _session_sequences_to_pairs(dataset.train_sub)
        valid_sessions, valid_labels = _session_sequences_to_pairs(dataset.valid)
        test_sessions, test_labels = _session_sequences_to_pairs(dataset.test)

        train_path = output_dir / "train.txt"
        valid_path = output_dir / "valid.txt"
        test_path = output_dir / "test.txt"

        save_srg_nn_train(train_path, train_sessions, train_labels)
        save_srg_nn_train(valid_path, valid_sessions, valid_labels)
        save_srg_nn_train(test_path, test_sessions, test_labels)
        metadata_path = _write_export_metadata(dataset, output_dir)

        return ExportResult(
            output_dir=output_dir,
            files={
                "train": train_path,
                "valid": valid_path,
                "test": test_path,
                "metadata": metadata_path,
            },
        )

    def export_train_pairs(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        save_srg_nn_train(output_path, sessions, labels)
        return output_path


__all__ = [
    "SRGNNExporter",
    "load_srg_nn_train",
    "save_srg_nn_train",
]
