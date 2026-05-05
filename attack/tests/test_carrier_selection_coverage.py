from __future__ import annotations

from math import log2
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.carrier_selection.coverage_local_position_scorer import (
    CoverageAwareLocalPositionScorer,
)
from attack.carrier_selection.coverage_prefix_bank import (
    ValidationPrefixRankRecord,
    build_prefix_bank_from_ranked_cases,
    target_rank_from_scores,
)
from attack.carrier_selection.selector import (
    build_targetized_selected_sessions_with_fixed_positions,
    select_local_position_carriers,
)
from attack.common.config import CarrierSelectionConfig


def _coverage_config(**overrides) -> CarrierSelectionConfig:
    payload = {
        "enabled": True,
        "candidate_pool_size": 0.03,
        "final_attack_size": 0.01,
        "scorer": "coverage_aware_local_position",
        "placement_mode": "best_local_position",
        "operation": "replacement",
        "candidate_positions": "nonzero",
        "coverage_prefix_source": "validation",
        "vulnerable_rank_min": 20,
        "vulnerable_rank_max": 200,
        "max_vulnerable_prefixes": 5000,
        "prefix_representation": "mean_item_embedding",
        "candidate_representation": "targetized_prefix_mean_embedding",
        "top_m_coverage": 1,
        "rank_weighting": "inverse_log_rank",
        "coverage_similarity": "cosine",
        "use_length_control": True,
        "length_buckets": "exact_until_4_plus",
        "normalize": "minmax",
    }
    payload.update(overrides)
    return CarrierSelectionConfig(**payload)


def _embeddings() -> np.ndarray:
    table = np.zeros((12, 2), dtype=np.float32)
    table[1] = [-1.0, 0.0]
    table[2] = [1.0, 0.0]
    table[3] = [1.0, 0.0]
    table[4] = [0.0, 1.0]
    table[5] = [0.0, -1.0]
    table[9] = [1.0, 0.0]
    return table


def test_prefix_bank_filters_truncates_weights_and_uses_prefix_only_representation() -> None:
    config = _coverage_config(max_vulnerable_prefixes=2)
    ranked_cases = [
        ValidationPrefixRankRecord(0, [2], 4, 10, 0.9),
        ValidationPrefixRankRecord(1, [2], 4, 21, 0.8),
        ValidationPrefixRankRecord(2, [4], 2, 30, 0.7),
        ValidationPrefixRankRecord(3, [5], 2, 22, 0.6),
        ValidationPrefixRankRecord(4, [2], 4, 201, 0.5),
    ]

    bank = build_prefix_bank_from_ranked_cases(
        ranked_cases=ranked_cases,
        target_item=9,
        config=config,
        embedding_table=_embeddings(),
        score_metadata={
            "score_dim": 12,
            "target_item": 9,
            "target_score_column": 8,
        },
    )

    assert [record.case_index for record in bank.records] == [1, 3]
    assert [record.target_rank for record in bank.records] == [21, 22]
    assert bank.weights[0] == pytest.approx(1.0 / log2(22))
    assert bank.metadata["vulnerable_prefix_count"] == 2
    assert bank.metadata["low_vulnerable_prefix_count_warning"] is True
    assert bank.metadata["score_dim"] == 12
    assert bank.metadata["target_score_column"] == 8
    np.testing.assert_allclose(bank.representations[0], np.array([1.0, 0.0]))


def test_prefix_bank_empty_vulnerable_set_raises() -> None:
    with pytest.raises(ValueError, match="Coverage prefix bank is empty"):
        build_prefix_bank_from_ranked_cases(
            ranked_cases=[
                ValidationPrefixRankRecord(0, [2], 4, 10, 0.9),
                ValidationPrefixRankRecord(1, [2], 4, 250, 0.1),
            ],
            target_item=9,
            config=_coverage_config(),
            embedding_table=_embeddings(),
        )


def test_target_rank_from_scores_uses_strictly_greater_convention_and_checks_bounds() -> None:
    rank, score = target_rank_from_scores([0.3, 0.7, 0.7, 0.1], target_score_column=1)

    assert rank == 1
    assert score == pytest.approx(0.7)
    with pytest.raises(ValueError, match="target_score_column"):
        target_rank_from_scores([0.3, 0.7], target_score_column=2)


def test_coverage_scorer_selects_best_nonzero_position_and_allows_last_position() -> None:
    bank = build_prefix_bank_from_ranked_cases(
        ranked_cases=[ValidationPrefixRankRecord(0, [2], 4, 21, 0.8)],
        target_item=9,
        config=_coverage_config(),
        embedding_table=_embeddings(),
    )
    scorer = CoverageAwareLocalPositionScorer(
        config=_coverage_config(),
        prefix_bank=bank,
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=[[4, 2, 3]], target_item=9)

    assert records[0].best_position == 2
    assert records[0].best_position_label == "pos2"
    assert records[0].best_position_record is not None
    assert records[0].best_position_record.right_item is None
    assert records[0].valid_position_count == 2
    assert metadata["coverage_score_summary"]["raw"]["max"] > metadata["coverage_score_summary"]["raw"]["min"]


def test_coverage_scorer_excludes_pos0_and_skips_noop_target_replacements() -> None:
    bank = build_prefix_bank_from_ranked_cases(
        ranked_cases=[ValidationPrefixRankRecord(0, [2], 4, 21, 0.8)],
        target_item=9,
        config=_coverage_config(),
        embedding_table=_embeddings(),
    )
    scorer = CoverageAwareLocalPositionScorer(
        config=_coverage_config(),
        prefix_bank=bank,
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=[[2, 9, 3]], target_item=9)

    assert records[0].best_position == 2
    assert records[0].valid_position_count == 1
    assert metadata["skipped_noop_target_position_count"] == 1
    assert metadata["pre_existing_target_in_pool_count"] == 1


def test_coverage_scorer_constant_normalization_ties_to_lower_position() -> None:
    bank = build_prefix_bank_from_ranked_cases(
        ranked_cases=[ValidationPrefixRankRecord(0, [2], 4, 21, 0.8)],
        target_item=9,
        config=_coverage_config(),
        embedding_table=_embeddings(),
    )
    scorer = CoverageAwareLocalPositionScorer(
        config=_coverage_config(),
        prefix_bank=bank,
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=[[2, 3, 3]], target_item=9)

    assert records[0].best_position == 1
    assert records[0].best_position_score == pytest.approx(0.0)
    assert metadata["coverage_score_summary"]["constant"] is True


def test_coverage_records_reuse_local_selector_and_fixed_replacement() -> None:
    bank = build_prefix_bank_from_ranked_cases(
        ranked_cases=[ValidationPrefixRankRecord(0, [2], 4, 21, 0.8)],
        target_item=9,
        config=_coverage_config(),
        embedding_table=_embeddings(),
    )
    scorer = CoverageAwareLocalPositionScorer(
        config=_coverage_config(),
        prefix_bank=bank,
        embedding_table=_embeddings(),
    )
    candidates = [[4, 2, 3], [2, 9], [4, 2], [5, 2, 3, 4]]
    records, _ = scorer.score(candidate_sessions=candidates, target_item=9)

    selection = select_local_position_carriers(
        candidate_sessions=candidates,
        session_records=records,
        final_count=2,
        target_item=9,
        use_length_control=True,
        length_buckets="exact_until_4_plus",
    )
    selected_records = {record.index: record for record in records}
    targetized = build_targetized_selected_sessions_with_fixed_positions(
        candidate_sessions=candidates,
        selected_records=[selected_records[index] for index in selection.selected_indices],
        target_item=9,
    )

    assert len(selection.selected_indices) == 2
    assert len(targetized.fake_sessions) == 2
    assert all(9 in session for session in targetized.fake_sessions)
    assert targetized.selected_candidate_indices == sorted(targetized.selected_candidate_indices)
