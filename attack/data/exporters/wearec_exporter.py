from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.freqrec_exporter import (
    FREQREC_EXPORT_SEMANTICS,
    FreqRecExporter,
)


@dataclass(frozen=True)
class WEARecExportResult:
    output_dir: Path
    files: dict[str, Path]
    item_count: int
    max_seq_length: int
    train_example_count: int
    valid_example_count: int
    test_example_count: int
    observed_max_item_id: int
    exporter_semantics: str = FREQREC_EXPORT_SEMANTICS

    @property
    def data_dir(self) -> Path:
        return self.output_dir


class WEARecExporter:
    name = "wearec"

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
    ) -> WEARecExportResult:
        if type(max_seq_length) is not int or max_seq_length <= 0 or max_seq_length % 2:
            raise ValueError("WEARec max_seq_length must be a positive even integer.")
        delegated = FreqRecExporter().export_with_train_pairs(
            dataset,
            train_prefixes=train_prefixes,
            train_labels=train_labels,
            output_dir=output_dir,
            dataset_name=dataset_name,
            max_seq_length=max_seq_length,
            mode=mode,
        )
        return WEARecExportResult(
            output_dir=delegated.output_dir,
            files=dict(delegated.files),
            item_count=delegated.item_count,
            max_seq_length=delegated.max_seq_length,
            train_example_count=delegated.train_example_count,
            valid_example_count=delegated.valid_example_count,
            test_example_count=delegated.test_example_count,
            observed_max_item_id=delegated.observed_max_item_id,
        )


__all__ = ["WEARecExporter", "WEARecExportResult"]
