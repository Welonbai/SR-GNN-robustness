from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


@dataclass(frozen=True)
class MDHGExportResult(ExportResult):
    n_node: int
    train_example_count: int
    test_example_count: int
    raw_train_session_count: int
    observed_max_item_id: int
    expected_raw_expanded_pair_count: int
    train_pairs_match_raw_expansion: bool

    @property
    def data_dir(self) -> Path:
        return self.output_dir


class MDHGExporter(BaseExporter):
    name = "mdhg"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> MDHGExportResult:
        train_sessions, train_labels = sequences_to_pairs(dataset.train_sub)
        return self._export(
            dataset,
            train_sessions=train_sessions,
            train_labels=train_labels,
            raw_train_sessions=dataset.train_sub,
            output_dir=output_dir,
        )

    def export_with_poisoned_train(
        self,
        dataset: CanonicalDataset,
        *,
        poisoned_sessions: Sequence[Sequence[int]],
        poisoned_labels: Sequence[int],
        raw_fake_sessions: Sequence[Sequence[int]],
        output_dir: str | Path,
        dataset_name: str | None = None,
    ) -> MDHGExportResult:
        raw_train_sessions = [
            *[list(session) for session in dataset.train_sub],
            *[list(session) for session in raw_fake_sessions],
        ]
        return self._export(
            dataset,
            train_sessions=poisoned_sessions,
            train_labels=poisoned_labels,
            raw_train_sessions=raw_train_sessions,
            output_dir=output_dir,
            dataset_name=dataset_name,
        )

    def _export(
        self,
        dataset: CanonicalDataset,
        *,
        train_sessions: Sequence[Sequence[int]],
        train_labels: Sequence[int],
        raw_train_sessions: Sequence[Sequence[int]],
        output_dir: str | Path,
        dataset_name: str | None = None,
    ) -> MDHGExportResult:
        if len(train_sessions) != len(train_labels):
            raise ValueError("MDHG train sessions and labels must have equal lengths.")

        dataset_name = dataset_name or str(dataset.metadata.get("dataset_name", "dataset"))
        dataset_dir = Path(output_dir) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        normalized_train_sessions = [list(map(int, session)) for session in train_sessions]
        normalized_train_labels = [int(label) for label in train_labels]
        normalized_raw_sessions = [list(map(int, session)) for session in raw_train_sessions]
        test_sessions, test_labels = sequences_to_pairs(dataset.test)

        observed_max = _validate_positive_ids(
            normalized_train_sessions,
            normalized_train_labels,
            normalized_raw_sessions,
            test_sessions,
            test_labels,
        )
        n_node = _resolve_n_node(dataset, observed_max)
        if observed_max > n_node:
            raise ValueError(
                f"MDHG observed item id {observed_max} exceeds canonical n_node {n_node}."
            )

        expected_sessions, expected_labels = sequences_to_pairs(normalized_raw_sessions)
        # Poisoned expanded pairs remain authoritative because some attacks may
        # deliberately inject examples that are not a full raw-session expansion.
        train_matches_raw = (
            normalized_train_sessions == expected_sessions
            and normalized_train_labels == expected_labels
        )

        train_path = dataset_dir / "train.txt"
        test_path = dataset_dir / "test.txt"
        all_train_seq_path = dataset_dir / "all_train_seq.txt"
        _write_pickle(train_path, (normalized_train_sessions, normalized_train_labels))
        _write_pickle(test_path, (test_sessions, test_labels))
        _write_pickle(all_train_seq_path, normalized_raw_sessions)

        return MDHGExportResult(
            output_dir=dataset_dir,
            files={
                "train": train_path,
                "test": test_path,
                "all_train_seq": all_train_seq_path,
            },
            n_node=n_node,
            train_example_count=len(normalized_train_sessions),
            test_example_count=len(test_sessions),
            raw_train_session_count=len(normalized_raw_sessions),
            observed_max_item_id=observed_max,
            expected_raw_expanded_pair_count=len(expected_sessions),
            train_pairs_match_raw_expansion=train_matches_raw,
        )


def sequences_to_pairs(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    sessions: list[list[int]] = []
    labels: list[int] = []
    for sequence in sequences:
        seq = [int(item) for item in sequence]
        for suffix_length in range(1, len(seq)):
            sessions.append(seq[:-suffix_length])
            labels.append(seq[-suffix_length])
    return sessions, labels


def _resolve_n_node(dataset: CanonicalDataset, observed_max: int) -> int:
    candidates: list[int] = []
    if dataset.item_map:
        mapped_ids = {int(item_id) for item_id in dataset.item_map.values()}
        if len(mapped_ids) != len(dataset.item_map):
            raise ValueError("MDHG requires canonical item_map values to be unique.")
        expected_ids = set(range(1, len(mapped_ids) + 1))
        if mapped_ids != expected_ids:
            raise ValueError("MDHG requires canonical item_map values to be dense 1-based IDs.")
        candidates.append(len(mapped_ids))

    counts = dataset.metadata.get("counts")
    if isinstance(counts, dict) and counts.get("items") is not None:
        candidates.append(int(counts["items"]))
    if dataset.metadata.get("item_count") is not None:
        candidates.append(int(dataset.metadata["item_count"]))

    n_node = max(candidates) if candidates else int(observed_max)
    if n_node <= 0:
        raise ValueError("Unable to determine a positive MDHG n_node.")
    return n_node


def _validate_positive_ids(
    train_sessions: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    raw_train_sessions: Sequence[Sequence[int]],
    test_sessions: Sequence[Sequence[int]],
    test_labels: Sequence[int],
) -> int:
    observed_max = 0
    for name, sessions in (
        ("train prefixes", train_sessions),
        ("raw train sessions", raw_train_sessions),
        ("test prefixes", test_sessions),
    ):
        for row_index, session in enumerate(sessions):
            if not session:
                raise ValueError(f"MDHG {name}[{row_index}] must not be empty.")
            for item in session:
                item_id = int(item)
                if item_id <= 0:
                    raise ValueError(
                        f"MDHG {name}[{row_index}] contains invalid item id {item_id}; "
                        "external IDs must be positive and 0 is reserved for padding."
                    )
                observed_max = max(observed_max, item_id)
    for name, labels in (("train labels", train_labels), ("test labels", test_labels)):
        for row_index, label in enumerate(labels):
            item_id = int(label)
            if item_id <= 0:
                raise ValueError(
                    f"MDHG {name}[{row_index}] is {item_id}; labels must be positive."
                )
            observed_max = max(observed_max, item_id)
    return observed_max


def _write_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


__all__ = ["MDHGExporter", "MDHGExportResult", "sequences_to_pairs"]
