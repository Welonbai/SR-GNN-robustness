from __future__ import annotations

from pathlib import Path

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import BaseExporter, ExportResult


class MiaSRecExporter(BaseExporter):
    name = "miasrec"

    def export(self, dataset: CanonicalDataset, output_dir: str | Path) -> ExportResult:
        """
        Export canonical split into RecBole .inter files.

        Expected outputs (benchmark mode):
          - <dataset>.train.inter
          - <dataset>.valid.inter
          - <dataset>.test.inter

        Columns:
          - session_id
          - item_id_list
          - item_id

        TODO: implement export logic once MiaSRec integration starts.
        """
        raise NotImplementedError("MiaSRec exporter is not implemented yet.")


__all__ = ["MiaSRecExporter"]
