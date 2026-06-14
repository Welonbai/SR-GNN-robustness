from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult
from attack.data.poisoned_dataset_builder import expand_session_to_samples


FREQREC_EXPORT_SEMANTICS = "canonical_explicit_prefix_label_v1"


@dataclass(frozen=True)
class FreqRecExportResult(ExportResult):
    item_count: int
    max_seq_length: int
    train_example_count: int
    valid_example_count: int
    test_example_count: int
    observed_max_item_id: int

    @property
    def data_dir(self) -> Path:
        return self.output_dir


class FreqRecExporter(BaseExporter):
    name = "freqrec"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> FreqRecExportResult:
        raise ValueError(
            "FreqRec export requires authoritative train prefixes and labels; "
            "use export_with_train_pairs()."
        )

    def export_with_train_pairs(
        self,
        dataset: CanonicalDataset,
        *,
        train_prefixes: Sequence[Sequence[int]],
        train_labels: Sequence[int],
        output_dir: str | Path,
        dataset_name: str,
        max_seq_length: int,
        mode: str,
    ) -> FreqRecExportResult:
        if len(train_prefixes) != len(train_labels):
            raise ValueError("FreqRec train prefixes and labels must have equal lengths.")
        if not train_prefixes:
            raise ValueError("FreqRec train split must not be empty.")
        if not isinstance(max_seq_length, int) or isinstance(max_seq_length, bool):
            raise TypeError("FreqRec max_seq_length must be an integer.")
        if max_seq_length <= 0:
            raise ValueError("FreqRec max_seq_length must be positive.")
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("FreqRec dataset_name must be a non-empty string.")
        if mode not in {"clean", "poisoned"}:
            raise ValueError("FreqRec export mode must be 'clean' or 'poisoned'.")

        item_count = _resolve_item_count(dataset)
        train_rows, train_max = _validated_rows(
            train_prefixes, train_labels, split="train", item_count=item_count
        )
        valid_prefixes, valid_labels = _expand_sequences(dataset.valid)
        test_prefixes, test_labels = _expand_sequences(dataset.test)
        valid_rows, valid_max = _validated_rows(
            valid_prefixes, valid_labels, split="valid", item_count=item_count
        )
        test_rows, test_max = _validated_rows(
            test_prefixes, test_labels, split="test", item_count=item_count
        )
        if not valid_rows or not test_rows:
            raise ValueError("FreqRec validation and test splits must not be empty.")

        dataset_dir = Path(output_dir) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train": dataset_dir / "train.jsonl",
            "valid": dataset_dir / "valid.jsonl",
            "test": dataset_dir / "test.jsonl",
            "metadata": dataset_dir / "metadata.json",
        }
        _atomic_write_jsonl(paths["train"], train_rows)
        _atomic_write_jsonl(paths["valid"], valid_rows)
        _atomic_write_jsonl(paths["test"], test_rows)
        _atomic_write_json(
            paths["metadata"],
            {
                "schema_version": 1,
                "dataset_name": dataset_name,
                "item_count": item_count,
                "padding_id": 0,
                "id_min": 1,
                "id_max": item_count,
                "max_seq_length": max_seq_length,
                "train_example_count": len(train_rows),
                "valid_example_count": len(valid_rows),
                "test_example_count": len(test_rows),
                "ordering": "example_id_ascending",
                "exporter_semantics": FREQREC_EXPORT_SEMANTICS,
                "training_mode": mode,
            },
        )
        return FreqRecExportResult(
            output_dir=dataset_dir,
            files=paths,
            item_count=item_count,
            max_seq_length=max_seq_length,
            train_example_count=len(train_rows),
            valid_example_count=len(valid_rows),
            test_example_count=len(test_rows),
            observed_max_item_id=max(train_max, valid_max, test_max),
        )


def _expand_sequences(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for sequence in sequences:
        sequence_prefixes, sequence_labels = expand_session_to_samples(sequence)
        prefixes.extend(sequence_prefixes)
        labels.extend(sequence_labels)
    return prefixes, labels


def _resolve_item_count(dataset: CanonicalDataset) -> int:
    candidates: list[int] = []
    if dataset.item_map:
        mapped_ids = list(dataset.item_map.values())
        if any(type(item_id) is not int for item_id in mapped_ids):
            raise ValueError("FreqRec canonical item_map IDs must be integers.")
        if set(mapped_ids) != set(range(1, len(mapped_ids) + 1)):
            raise ValueError("FreqRec requires dense, unique, 1-based canonical item IDs.")
        candidates.append(len(mapped_ids))
    counts = dataset.metadata.get("counts")
    if isinstance(counts, dict) and "items" in counts:
        candidates.append(_exact_positive_int(counts["items"], "metadata.counts.items"))
    if "item_count" in dataset.metadata:
        candidates.append(
            _exact_positive_int(dataset.metadata["item_count"], "metadata.item_count")
        )
    if not candidates:
        raise ValueError("FreqRec item_count must come from canonical dataset metadata.")
    if len(set(candidates)) != 1:
        raise ValueError(f"FreqRec canonical item_count sources disagree: {candidates}.")
    return candidates[0]


def _validated_rows(
    prefixes: Sequence[Sequence[int]],
    labels: Sequence[int],
    *,
    split: str,
    item_count: int,
) -> tuple[list[dict[str, Any]], int]:
    if len(prefixes) != len(labels):
        raise ValueError(f"FreqRec {split} prefixes and labels must have equal lengths.")
    rows: list[dict[str, Any]] = []
    observed_max = 0
    for example_id, (prefix, label) in enumerate(zip(prefixes, labels)):
        if isinstance(prefix, (str, bytes)) or not isinstance(prefix, Sequence):
            raise TypeError(f"FreqRec {split}[{example_id}] prefix must be a sequence.")
        if not prefix:
            raise ValueError(f"FreqRec {split}[{example_id}] prefix must not be empty.")
        normalized_prefix: list[int] = []
        for item_index, item in enumerate(prefix):
            item_id = _exact_item_id(
                item,
                item_count=item_count,
                field=f"{split}[{example_id}].input_prefix[{item_index}]",
            )
            normalized_prefix.append(item_id)
            observed_max = max(observed_max, item_id)
        label_id = _exact_item_id(
            label, item_count=item_count, field=f"{split}[{example_id}].label"
        )
        observed_max = max(observed_max, label_id)
        rows.append(
            {
                "example_id": example_id,
                "input_prefix": normalized_prefix,
                "label": label_id,
            }
        )
    return rows, observed_max


def _exact_item_id(value: Any, *, item_count: int, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"FreqRec {field} must be an integer.")
    item_id = int(value)
    if item_id < 1 or item_id > item_count:
        raise ValueError(
            f"FreqRec {field}={item_id} is outside canonical range 1..{item_count}."
        )
    return item_id


def _exact_positive_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"FreqRec {field} must be an integer.")
    if value <= 0:
        raise ValueError(f"FreqRec {field} must be positive.")
    return int(value)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


__all__ = [
    "FREQREC_EXPORT_SEMANTICS",
    "FreqRecExporter",
    "FreqRecExportResult",
]
