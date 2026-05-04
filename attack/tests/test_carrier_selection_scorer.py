from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.carrier_selection.scorer import HybridTargetSessionCompatibilityScorer
from attack.common.config import CarrierSelectionConfig


def _carrier_config(**overrides) -> CarrierSelectionConfig:
    payload = {
        "enabled": True,
        "candidate_pool_size": 0.03,
        "final_attack_size": 0.01,
        "embedding_weight": 0.4,
        "cooccurrence_weight": 0.3,
        "transition_weight": 0.3,
    }
    payload.update(overrides)
    return CarrierSelectionConfig(**payload)


def test_hybrid_scorer_ranks_target_compatible_session_higher() -> None:
    target = 9
    train_sub = [
        [9, 2, 3],
        [2, 9, 4],
        [7, 8, 9],
        [1, 5, 6],
    ]
    candidates = [
        [2, 3, 4],
        [5, 6, 7],
        [9, 5, 6],
    ]
    embeddings = np.zeros((10, 2), dtype=np.float32)
    embeddings[9] = [1.0, 0.0]
    embeddings[2] = [1.0, 0.0]
    embeddings[3] = [0.9, 0.1]
    embeddings[4] = [0.8, 0.2]
    embeddings[5] = [0.0, 1.0]
    embeddings[6] = [0.0, -1.0]
    embeddings[7] = [0.2, 0.8]

    scorer = HybridTargetSessionCompatibilityScorer(
        train_sub_sessions=train_sub,
        config=_carrier_config(),
        embedding_table=embeddings,
    )
    records, metadata = scorer.score(candidate_sessions=candidates, target_item=target)

    assert records[0].carrier_score > records[1].carrier_score
    assert metadata["target_train_sub_count"] == 3
    assert metadata["embedding"]["embedding_shape"] == [10, 2]
    assert metadata["embedding"]["target_item_embedding_row"] == 9
    assert metadata["embedding"]["item_id_row_mapping"] == "padding_row_0_item_id_to_row"


def test_scorer_skips_target_items_during_component_aggregation() -> None:
    target = 9
    embeddings = np.zeros((10, 2), dtype=np.float32)
    embeddings[9] = [1.0, 0.0]
    embeddings[5] = [0.0, 1.0]
    scorer = HybridTargetSessionCompatibilityScorer(
        train_sub_sessions=[[9, 5]],
        config=_carrier_config(embedding_weight=1.0, cooccurrence_weight=0.0, transition_weight=0.0),
        embedding_table=embeddings,
    )

    records, metadata = scorer.score(candidate_sessions=[[9], [9, 5]], target_item=target)

    assert records[0].raw_embedding_score == pytest.approx(0.0)
    assert records[1].raw_embedding_score == pytest.approx(0.5)
    assert metadata["embedding"]["skipped_target_item_count"] == 2


def test_minmax_normalization_marks_constant_component_columns() -> None:
    scorer = HybridTargetSessionCompatibilityScorer(
        train_sub_sessions=[[1, 2, 3]],
        config=_carrier_config(embedding_weight=0.0, cooccurrence_weight=1.0, transition_weight=0.0),
        embedding_table=np.ones((5, 2), dtype=np.float32),
    )

    records, metadata = scorer.score(candidate_sessions=[[4], [4], [4]], target_item=3)

    assert all(record.normalized_cooccurrence_score == pytest.approx(0.0) for record in records)
    assert metadata["constant_normalized_columns"]["cooccurrence"] is True
    assert metadata["constant_normalized_columns"]["transition"] is True
