from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


class TRONExporter(BaseExporter):
    name = "tron"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> ExportResult:
        return self._export_with_train_override(
            dataset,
            output_dir=output_dir,
            train_sessions=None,
            train_labels=None,
        )

    def export_with_poisoned_train(
        self,
        dataset: CanonicalDataset,
        *,
        poisoned_sessions: Sequence[Sequence[int]],
        poisoned_labels: Sequence[int],
        output_dir: str | Path,
        dataset_name: str | None = None,
    ) -> ExportResult:
        return self._export_with_train_override(
            dataset,
            output_dir=output_dir,
            train_sessions=poisoned_sessions,
            train_labels=poisoned_labels,
            dataset_name=dataset_name,
        )

    def _export_with_train_override(
        self,
        dataset: CanonicalDataset,
        *,
        output_dir: str | Path,
        train_sessions: Sequence[Sequence[int]] | None,
        train_labels: Sequence[int] | None,
        dataset_name: str | None = None,
    ) -> ExportResult:
        dataset_name = dataset_name or dataset.metadata.get("dataset_name", "dataset")
        output_root = Path(output_dir)
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if train_sessions is None or train_labels is None:
            train_sequences = dataset.train_sub
        else:
            train_sequences = _pairs_to_sequences(train_sessions, train_labels)
        valid_sequences = dataset.valid
        test_sequences = dataset.test

        train_sessions_json = _build_tron_sessions(train_sequences)
        valid_sessions_json = _build_tron_sessions(valid_sequences)
        test_sessions_json = _build_tron_sessions(test_sequences)

        train_path = dataset_dir / f"{dataset_name}_train.jsonl"
        valid_path = dataset_dir / f"{dataset_name}_valid.jsonl"
        test_path = dataset_dir / f"{dataset_name}_test.jsonl"
        stats_path = dataset_dir / f"{dataset_name}_stats.json"

        _write_jsonl(train_path, train_sessions_json)
        _write_jsonl(valid_path, valid_sessions_json)
        _write_jsonl(test_path, test_sessions_json)

        num_items = _infer_num_items(dataset, train_sequences, valid_sequences, test_sequences)
        stats = {
            "train": {
                "num_sessions": len(train_sessions_json),
                "num_events": sum(len(session["events"]) for session in train_sessions_json),
            },
            "valid": {
                "num_sessions": len(valid_sessions_json),
                "num_events": sum(len(session["events"]) for session in valid_sessions_json),
            },
            "num_items": int(num_items),
            "test": {
                "num_sessions": len(test_sessions_json),
                "num_events": sum(len(session["events"]) for session in test_sessions_json),
            },
        }
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle)

        return ExportResult(
            output_dir=dataset_dir,
            files={
                "train": train_path,
                "valid": valid_path,
                "test": test_path,
                "stats": stats_path,
            },
        )


def _pairs_to_sequences(
    sessions: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> list[list[int]]:
    if len(sessions) != len(labels):
        raise ValueError("sessions and labels must be the same length.")
    sequences: list[list[int]] = []
    for session, label in zip(sessions, labels):
        if not session:
            continue
        combined = list(session) + [int(label)]
        if len(combined) < 2:
            continue
        sequences.append(combined)
    return sequences


def _build_tron_sessions(sequences: Sequence[Sequence[int]]) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    for idx, seq in enumerate(sequences, start=1):
        if len(seq) < 2:
            continue
        events = []
        for offset, item in enumerate(seq):
            events.append({"aid": int(item), "ts": int(offset), "type": "clicks"})
        sessions.append({"session": str(idx), "events": events})
    return sessions


def _write_jsonl(path: Path, sessions: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for session in sessions:
            handle.write(json.dumps(session, separators=(",", ":")) + "\n")


def _infer_num_items(
    dataset: CanonicalDataset,
    train_sequences: Sequence[Sequence[int]],
    valid_sequences: Sequence[Sequence[int]],
    test_sequences: Sequence[Sequence[int]],
) -> int:
    max_item = 0
    for seq in train_sequences:
        if seq:
            max_item = max(max_item, max(seq))
    for seq in valid_sequences:
        if seq:
            max_item = max(max_item, max(seq))
    for seq in test_sequences:
        if seq:
            max_item = max(max_item, max(seq))
    metadata_count = None
    counts = dataset.metadata.get("counts")
    if isinstance(counts, dict):
        metadata_count = counts.get("items")
    if metadata_count is None:
        return int(max_item)
    return int(max(max_item, int(metadata_count)))


__all__ = ["TRONExporter"]
