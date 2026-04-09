from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from attack.data.canonical_dataset import CanonicalDataset


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    files: dict[str, Path]


class BaseExporter(ABC):
    name: str

    @abstractmethod
    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> ExportResult:
        raise NotImplementedError


__all__ = ["BaseExporter", "ExportResult"]
