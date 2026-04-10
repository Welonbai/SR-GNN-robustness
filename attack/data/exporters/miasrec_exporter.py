from __future__ import annotations

from pathlib import Path
from typing import Sequence

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


class MiaSRecExporter(BaseExporter):
    name = "miasrec"

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
            train_sessions, train_labels = _sequences_to_pairs(dataset.train_sub)
        valid_sessions, valid_labels = _sequences_to_pairs(dataset.valid)
        test_sessions, test_labels = _sequences_to_pairs(dataset.test)

        train_path = dataset_dir / f"{dataset_name}.train.inter"
        valid_path = dataset_dir / f"{dataset_name}.valid.inter"
        test_path = dataset_dir / f"{dataset_name}.test.inter"

        _write_inter_file(train_path, train_sessions, train_labels)
        _write_inter_file(valid_path, valid_sessions, valid_labels)
        _write_inter_file(test_path, test_sessions, test_labels)

        return ExportResult(
            output_dir=dataset_dir,
            files={
                "train": train_path,
                "valid": valid_path,
                "test": test_path,
            },
        )


def _sequences_to_pairs(
    sequences: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int]]:
    sessions: list[list[int]] = []
    labels: list[int] = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            labels.append(int(seq[-i]))
            sessions.append(list(seq[:-i]))
    return sessions, labels


def _write_inter_file(
    path: Path,
    sessions: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> None:
    if len(sessions) != len(labels):
        raise ValueError("sessions and labels must be the same length.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
        for idx, (session, label) in enumerate(zip(sessions, labels), start=1):
            if not session:
                continue
            item_list = " ".join(str(item) for item in session)
            handle.write(f"{idx}\t{item_list}\t{int(label)}\n")


__all__ = ["MiaSRecExporter"]
