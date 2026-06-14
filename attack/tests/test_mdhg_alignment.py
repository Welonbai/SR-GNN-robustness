from __future__ import annotations

from pathlib import Path

from attack.common.config import load_config
from attack.data.canonical_dataset import CanonicalDataset
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels


CONFIG_PATH = Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"


def test_mdhg_uses_canonical_expanded_test_order() -> None:
    dataset = CanonicalDataset(
        train_sub=[],
        valid=[],
        test=[[1, 2, 3], [4, 5]],
        item_map={},
        metadata={},
    )
    labels = resolve_ground_truth_labels(
        load_config(CONFIG_PATH),
        victim_name="mdhg",
        canonical_dataset=dataset,
        predictions=[[1], [2], [3]],
    )
    assert labels == [3, 2, 5]
