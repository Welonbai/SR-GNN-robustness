from __future__ import annotations

from pathlib import Path

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


class TRONExporter(BaseExporter):
    name = "tron"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> ExportResult:
        """
        Export canonical split into TRON-specific input format.

        TODO: inspect TRON input contract and implement export logic.
        """
        raise NotImplementedError("TRON exporter is not implemented yet.")


__all__ = ["TRONExporter"]
