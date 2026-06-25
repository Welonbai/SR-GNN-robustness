from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any, Sequence

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult
from attack.data.poisoned_dataset_builder import expand_session_to_samples


DTGAT_EXPORTER_SEMANTICS = "dtgat_processed_prefix_interval_stamp_v1"
SYNTHETIC_TEMPORAL_POLICY = {
    "source": "synthetic",
    "interval_units_written": "milliseconds",
    "interval_value_per_prefix_position": 1000,
    "session_stamp_units_written": "seconds",
    "session_stamp_formula": "example_id * 86400 within each split",
    "reason": "Canonical datasets currently do not carry per-event timestamps.",
}


@dataclass(frozen=True)
class DTGATExportResult(ExportResult):
    n_node: int
    train_example_count: int
    test_example_count: int
    raw_train_session_count: int
    raw_fake_session_count: int
    observed_max_item_id: int
    max_train_prefix_length: int
    max_test_prefix_length: int
    max_all_train_seq_length: int
    expected_fake_expanded_pair_count: int
    fake_pairs_present_in_train: bool
    exporter_semantics: str = DTGAT_EXPORTER_SEMANTICS

    @property
    def data_dir(self) -> Path:
        return self.output_dir


class DTGATExporter(BaseExporter):
    name = "dtgat"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> DTGATExportResult:
        train_prefixes, train_labels = sequences_to_pairs(dataset.train_sub)
        return self._export(
            dataset,
            train_prefixes=train_prefixes,
            train_labels=train_labels,
            raw_train_sessions=dataset.train_sub,
            raw_fake_sessions=[],
            output_dir=output_dir,
            mode="clean",
        )

    def export_with_poisoned_train(
        self,
        dataset: CanonicalDataset,
        *,
        poisoned_prefixes: Sequence[Sequence[int]],
        poisoned_labels: Sequence[int],
        raw_fake_sessions: Sequence[Sequence[int]],
        output_dir: str | Path,
        dataset_name: str | None = None,
    ) -> DTGATExportResult:
        raw_train_sessions = [
            *[list(session) for session in dataset.train_sub],
            *[list(session) for session in raw_fake_sessions],
        ]
        return self._export(
            dataset,
            train_prefixes=poisoned_prefixes,
            train_labels=poisoned_labels,
            raw_train_sessions=raw_train_sessions,
            raw_fake_sessions=raw_fake_sessions,
            output_dir=output_dir,
            dataset_name=dataset_name,
            mode="poisoned",
        )

    def export_with_train_pairs(
        self,
        dataset: CanonicalDataset,
        *,
        train_prefixes: Sequence[Sequence[int]],
        train_labels: Sequence[int],
        raw_train_sessions: Sequence[Sequence[int]],
        output_dir: str | Path,
        dataset_name: str | None = None,
        mode: str,
    ) -> DTGATExportResult:
        if mode not in {"clean", "poisoned"}:
            raise ValueError("DT-GAT export mode must be 'clean' or 'poisoned'.")
        return self._export(
            dataset,
            train_prefixes=train_prefixes,
            train_labels=train_labels,
            raw_train_sessions=raw_train_sessions,
            raw_fake_sessions=[],
            output_dir=output_dir,
            dataset_name=dataset_name,
            mode=mode,
        )

    def _export(
        self,
        dataset: CanonicalDataset,
        *,
        train_prefixes: Sequence[Sequence[int]],
        train_labels: Sequence[int],
        raw_train_sessions: Sequence[Sequence[int]],
        raw_fake_sessions: Sequence[Sequence[int]],
        output_dir: str | Path,
        mode: str,
        dataset_name: str | None = None,
    ) -> DTGATExportResult:
        if mode not in {"clean", "poisoned"}:
            raise ValueError("DT-GAT export mode must be 'clean' or 'poisoned'.")

        dataset_name = str(dataset_name or dataset.metadata.get("dataset_name", "dataset"))
        if not dataset_name.strip():
            raise ValueError("DT-GAT dataset_name must be non-empty.")

        normalized_train_prefixes = _normalize_prefixes(train_prefixes, "train prefixes")
        normalized_train_labels = _normalize_labels(train_labels, "train labels")
        normalized_raw_train_sessions = _normalize_sessions(
            raw_train_sessions, "all_train_seq"
        )
        normalized_raw_fake_sessions = _normalize_sessions(
            raw_fake_sessions, "raw fake sessions", allow_empty=True
        )
        test_prefixes, test_labels = sequences_to_pairs(dataset.test)

        n_node = _resolve_n_node(
            dataset,
            normalized_train_prefixes,
            normalized_train_labels,
            normalized_raw_train_sessions,
            test_prefixes,
            test_labels,
        )
        observed_max = _validate_export_inputs(
            train_prefixes=normalized_train_prefixes,
            train_labels=normalized_train_labels,
            test_prefixes=test_prefixes,
            test_labels=test_labels,
            all_train_seq=normalized_raw_train_sessions,
            n_node=n_node,
        )
        train_intervals, train_stamps = _synthetic_temporal_fields(
            normalized_train_prefixes
        )
        test_intervals, test_stamps = _synthetic_temporal_fields(test_prefixes)

        train_payload = [
            normalized_train_prefixes,
            train_intervals,
            normalized_train_labels,
            train_stamps,
        ]
        test_payload = [test_prefixes, test_intervals, test_labels, test_stamps]
        _validate_dtgat_payload(train_payload, "train")
        _validate_dtgat_payload(test_payload, "test")

        expected_fake_prefixes, expected_fake_labels = _expand_sequences(
            normalized_raw_fake_sessions
        )
        fake_pairs_present = _contains_ordered_subsequence(
            list(zip(normalized_train_prefixes, normalized_train_labels)),
            list(zip(expected_fake_prefixes, expected_fake_labels)),
        )

        dataset_dir = Path(output_dir) / dataset_name
        processed_dir = dataset_dir / "processed_data"
        paths = {
            "train": processed_dir / "train.txt",
            "test": processed_dir / "test.txt",
            "all_train_seq": processed_dir / "all_train_seq.txt",
            "metadata": dataset_dir / "metadata.json",
            "eval_rows": dataset_dir / "eval_rows.json",
        }
        _atomic_write_pickle(paths["train"], train_payload)
        _atomic_write_pickle(paths["test"], test_payload)
        # DT-GAT pads expanded prefixes from full training sessions; expanded
        # prefix rows belong only in train.txt so the resolved pair budget is kept.
        _atomic_write_pickle(paths["all_train_seq"], normalized_raw_train_sessions)
        _atomic_write_json(paths["eval_rows"], _eval_rows(test_prefixes, test_labels))

        max_train_prefix_length = _max_len(normalized_train_prefixes)
        max_test_prefix_length = _max_len(test_prefixes)
        max_all_train_seq_length = _max_len(normalized_raw_train_sessions)
        metadata = {
            "schema_version": 1,
            "dataset_name": dataset_name,
            "victim": "dtgat",
            "exporter_semantics": DTGAT_EXPORTER_SEMANTICS,
            "training_mode": mode,
            "n_node": n_node,
            "padding_id": 0,
            "id_min": 1,
            "id_max": n_node,
            "item_id_convention": "dense_one_based_with_zero_padding",
            "files": {
                key: str(path.relative_to(dataset_dir)) for key, path in paths.items()
            },
            "pickle_payloads": {
                "train": "[prefixes, intervals, labels, session_stamps]",
                "test": "[prefixes, intervals, labels, session_stamps]",
                "all_train_seq": "list[list[int]] full training sessions",
            },
            "temporal_policy": SYNTHETIC_TEMPORAL_POLICY,
            "ordering": {
                "test": "canonical_expanded_prefix_label_order",
                "eval_rows": "example_id_ascending_matches_test_payload",
            },
            "counts": {
                "train_examples": len(normalized_train_prefixes),
                "test_examples": len(test_prefixes),
                "all_train_seq_sessions": len(normalized_raw_train_sessions),
                "raw_fake_sessions": len(normalized_raw_fake_sessions),
                "expected_fake_expanded_pairs": len(expected_fake_prefixes),
            },
            "max_lengths": {
                "train_prefix": max_train_prefix_length,
                "test_prefix": max_test_prefix_length,
                "all_train_seq": max_all_train_seq_length,
            },
            "validation": {
                "fake_pairs_present_in_train": fake_pairs_present,
                "all_train_seq_full_sessions_not_expanded_rows": True,
                "observed_max_item_id": observed_max,
            },
        }
        _atomic_write_json(paths["metadata"], metadata)

        return DTGATExportResult(
            output_dir=dataset_dir,
            files=paths,
            n_node=n_node,
            train_example_count=len(normalized_train_prefixes),
            test_example_count=len(test_prefixes),
            raw_train_session_count=len(normalized_raw_train_sessions),
            raw_fake_session_count=len(normalized_raw_fake_sessions),
            observed_max_item_id=observed_max,
            max_train_prefix_length=max_train_prefix_length,
            max_test_prefix_length=max_test_prefix_length,
            max_all_train_seq_length=max_all_train_seq_length,
            expected_fake_expanded_pair_count=len(expected_fake_prefixes),
            fake_pairs_present_in_train=fake_pairs_present,
        )


def sequences_to_pairs(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for sequence in sequences:
        seq = [int(item) for item in sequence]
        expanded_prefixes, expanded_labels = expand_session_to_samples(seq)
        prefixes.extend(expanded_prefixes)
        labels.extend(expanded_labels)
    return prefixes, labels


def _expand_sequences(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for session in sequences:
        session_prefixes, session_labels = expand_session_to_samples(session)
        prefixes.extend(session_prefixes)
        labels.extend(session_labels)
    return prefixes, labels


def _normalize_prefixes(
    prefixes: Sequence[Sequence[int]], field: str
) -> list[list[int]]:
    if isinstance(prefixes, (str, bytes)):
        raise TypeError(f"DT-GAT {field} must be a sequence of prefixes.")
    normalized: list[list[int]] = []
    for row_index, prefix in enumerate(prefixes):
        if isinstance(prefix, (str, bytes)) or not isinstance(prefix, Sequence):
            raise TypeError(f"DT-GAT {field}[{row_index}] must be a sequence.")
        row = [_exact_int(item, f"{field}[{row_index}]") for item in prefix]
        if not row:
            raise ValueError(f"DT-GAT {field}[{row_index}] must not be empty.")
        normalized.append(row)
    if not normalized:
        raise ValueError(f"DT-GAT {field} must not be empty.")
    return normalized


def _normalize_sessions(
    sessions: Sequence[Sequence[int]],
    field: str,
    *,
    allow_empty: bool = False,
) -> list[list[int]]:
    if isinstance(sessions, (str, bytes)):
        raise TypeError(f"DT-GAT {field} must be a sequence of sessions.")
    normalized: list[list[int]] = []
    for row_index, session in enumerate(sessions):
        if isinstance(session, (str, bytes)) or not isinstance(session, Sequence):
            raise TypeError(f"DT-GAT {field}[{row_index}] must be a sequence.")
        row = [_exact_int(item, f"{field}[{row_index}]") for item in session]
        if len(row) < 2:
            raise ValueError(f"DT-GAT {field}[{row_index}] must contain at least 2 items.")
        normalized.append(row)
    if not normalized and not allow_empty:
        raise ValueError("DT-GAT all_train_seq.txt must be non-empty.")
    return normalized


def _normalize_labels(labels: Sequence[int], field: str) -> list[int]:
    if isinstance(labels, (str, bytes)):
        raise TypeError(f"DT-GAT {field} must be a sequence.")
    normalized = [_exact_int(label, f"{field}[{index}]") for index, label in enumerate(labels)]
    if not normalized:
        raise ValueError(f"DT-GAT {field} must not be empty.")
    return normalized


def _resolve_n_node(
    dataset: CanonicalDataset,
    train_prefixes: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    all_train_seq: Sequence[Sequence[int]],
    test_prefixes: Sequence[Sequence[int]],
    test_labels: Sequence[int],
) -> int:
    candidates: list[int] = []
    if dataset.item_map:
        mapped_ids = list(dataset.item_map.values())
        if any(type(item_id) is not int for item_id in mapped_ids):
            raise TypeError("DT-GAT canonical item_map IDs must be integers.")
        if set(mapped_ids) != set(range(1, len(mapped_ids) + 1)):
            raise ValueError("DT-GAT requires dense, unique, 1-based item_map IDs.")
        candidates.append(len(mapped_ids))
    counts = dataset.metadata.get("counts")
    if isinstance(counts, dict) and "items" in counts:
        candidates.append(_exact_positive_int(counts["items"], "metadata.counts.items"))
    if "item_count" in dataset.metadata:
        candidates.append(_exact_positive_int(dataset.metadata["item_count"], "metadata.item_count"))
    if not candidates:
        candidates.append(
            _observed_max(
                train_prefixes,
                all_train_seq,
                test_prefixes,
                train_labels=train_labels,
                test_labels=test_labels,
            )
        )
    if len(set(candidates)) != 1:
        raise ValueError(f"DT-GAT canonical item count sources disagree: {candidates}.")
    n_node = candidates[0]
    if n_node <= 0:
        raise ValueError("Unable to determine a positive DT-GAT n_node.")
    return n_node


def _validate_export_inputs(
    *,
    train_prefixes: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    test_prefixes: Sequence[Sequence[int]],
    test_labels: Sequence[int],
    all_train_seq: Sequence[Sequence[int]],
    n_node: int,
) -> int:
    if len(train_prefixes) != len(train_labels):
        raise ValueError("DT-GAT train prefixes and labels must have equal length.")
    if len(test_prefixes) != len(test_labels):
        raise ValueError("DT-GAT test prefixes and labels must have equal length.")
    if not all_train_seq:
        raise ValueError("DT-GAT all_train_seq.txt must be non-empty.")
    max_all_train = _max_len(all_train_seq)
    max_prefix = max(_max_len(train_prefixes), _max_len(test_prefixes))
    if max_all_train < max_prefix:
        raise ValueError(
            "DT-GAT all_train_seq max length must be at least every train/test prefix length."
        )
    observed_max = _observed_max(
        train_prefixes,
        all_train_seq,
        test_prefixes,
        train_labels=train_labels,
        test_labels=test_labels,
    )
    for group_name, groups in (
        ("train prefixes", train_prefixes),
        ("test prefixes", test_prefixes),
        ("all_train_seq", all_train_seq),
    ):
        for row_index, row in enumerate(groups):
            for item in row:
                _validate_item_id(item, n_node, f"{group_name}[{row_index}]")
    for label_name, labels in (("train labels", train_labels), ("test labels", test_labels)):
        for row_index, label in enumerate(labels):
            _validate_item_id(label, n_node, f"{label_name}[{row_index}]")
    return observed_max


def _validate_dtgat_payload(payload: list[object], split: str) -> None:
    if len(payload) != 4:
        raise ValueError(f"DT-GAT {split} payload must have four top-level lists.")
    prefixes, intervals, labels, stamps = payload
    if not (
        isinstance(prefixes, list)
        and isinstance(intervals, list)
        and isinstance(labels, list)
        and isinstance(stamps, list)
    ):
        raise TypeError(f"DT-GAT {split} payload entries must be lists.")
    lengths = {len(prefixes), len(intervals), len(labels), len(stamps)}
    if len(lengths) != 1:
        raise ValueError(f"DT-GAT {split} top-level lists must have equal length.")
    for row_index, (prefix, interval_row) in enumerate(zip(prefixes, intervals)):
        if len(prefix) != len(interval_row):
            raise ValueError(
                f"DT-GAT {split} interval row {row_index} is not aligned with prefix."
            )


def _synthetic_temporal_fields(
    prefixes: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    intervals = [[1000 for _ in prefix] for prefix in prefixes]
    stamps = [index * 86400 for index in range(len(prefixes))]
    return intervals, stamps


def _eval_rows(prefixes: Sequence[Sequence[int]], labels: Sequence[int]) -> list[dict[str, Any]]:
    return [
        {"example_id": index, "input_prefix": list(prefix), "label": int(label)}
        for index, (prefix, label) in enumerate(zip(prefixes, labels))
    ]


def _contains_ordered_subsequence(
    haystack: Sequence[tuple[list[int], int]],
    needle: Sequence[tuple[list[int], int]],
) -> bool:
    if not needle:
        return True
    start_max = len(haystack) - len(needle)
    for start in range(start_max + 1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return True
    return False


def _observed_max(
    *groups: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    test_labels: Sequence[int],
) -> int:
    observed = 0
    for group in groups:
        for row in group:
            if row:
                observed = max(observed, max(int(item) for item in row))
    for label in [*train_labels, *test_labels]:
        observed = max(observed, int(label))
    return observed


def _max_len(rows: Sequence[Sequence[int]]) -> int:
    return max((len(row) for row in rows), default=0)


def _validate_item_id(value: int, n_node: int, field: str) -> None:
    if value < 1 or value > n_node:
        raise ValueError(
            f"DT-GAT {field} contains item id {value}; expected dense item IDs in 1..{n_node}."
        )


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"DT-GAT {field} must be an integer.")
    return int(value)


def _exact_positive_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"DT-GAT {field} must be an integer.")
    if value <= 0:
        raise ValueError(f"DT-GAT {field} must be positive.")
    return int(value)


def _atomic_write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _atomic_write_json(path: Path, payload: object) -> None:
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
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


__all__ = [
    "DTGAT_EXPORTER_SEMANTICS",
    "DTGATExportResult",
    "DTGATExporter",
    "SYNTHETIC_TEMPORAL_POLICY",
    "sequences_to_pairs",
]
