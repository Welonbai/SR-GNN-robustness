from __future__ import annotations

import json

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.freqrec_exporter import FreqRecExporter
from attack.data.exporters.wearec_exporter import WEARecExporter
from attack.pipeline.core.pipeline_utils import build_clean_pairs


def _dataset():
    return CanonicalDataset(
        train_sub=[[1, 2, 3]],
        valid=[[1, 2, 3]],
        test=[[2, 3, 4]],
        item_map={str(value): value for value in range(1, 6)},
        metadata={"item_count": 5, "counts": {"items": 5}, "variant": "full"},
    )


def test_wearec_directly_reuses_contract_in_isolated_destination(tmp_path):
    prefixes, labels = build_clean_pairs(_dataset())
    freqrec = FreqRecExporter().export_with_train_pairs(
        _dataset(), train_prefixes=prefixes, train_labels=labels,
        output_dir=tmp_path / "export" / "freqrec", dataset_name="toy",
        max_seq_length=6, mode="clean",
    )
    wearec = WEARecExporter().export_with_train_pairs(
        _dataset(), train_prefixes=prefixes, train_labels=labels,
        output_dir=tmp_path / "export" / "wearec", dataset_name="toy",
        max_seq_length=6, mode="clean",
    )
    assert freqrec.output_dir != wearec.output_dir
    for name in ("train", "valid", "test", "metadata"):
        assert freqrec.files[name].read_bytes() == wearec.files[name].read_bytes()
    metadata = json.loads(wearec.files["metadata"].read_text(encoding="utf-8"))
    assert metadata["exporter_semantics"] == "canonical_explicit_prefix_label_v1"
    assert "model" not in metadata
