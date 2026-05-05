from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.carrier_selection.local_position_scorer import (
    HybridLocalPositionCompatibilityScorer,
    LocalPositionRecord,
    LocalPositionSessionRecord,
)
from attack.carrier_selection.selector import (
    build_targetized_selected_sessions_with_fixed_positions,
    select_local_position_carriers,
)
from attack.common.config import CarrierSelectionConfig


def _local_config(**overrides) -> CarrierSelectionConfig:
    payload = {
        "enabled": True,
        "candidate_pool_size": 0.03,
        "final_attack_size": 0.01,
        "scorer": "hybrid_local_position_compatibility",
        "placement_mode": "best_local_position",
        "operation": "replacement",
        "candidate_positions": "nonzero",
        "embedding_weight": 0.4,
        "cooccurrence_weight": 0.3,
        "transition_weight": 0.3,
        "local_embedding_weight": 0.0,
        "local_transition_weight": 1.0,
        "session_compatibility_weight": 0.0,
        "left_to_target_weight": 0.5,
        "target_to_right_weight": 0.5,
    }
    payload.update(overrides)
    return CarrierSelectionConfig(**payload)


def _embeddings(row_count: int = 10) -> np.ndarray:
    table = np.zeros((row_count, 2), dtype=np.float32)
    for row in range(row_count):
        table[row] = [1.0, 0.0]
    return table


def test_local_scorer_selects_best_replacement_position_from_transitions() -> None:
    target = 9
    train_sub = [[2, 9, 4] for _ in range(5)] + [[1, 9, 3]]
    candidates = [[1, 2, 3, 4]]
    scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=train_sub,
        config=_local_config(),
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=candidates, target_item=target)

    assert records[0].best_position == 2
    assert records[0].best_position_label == "pos2"
    assert records[0].best_position_record is not None
    assert records[0].best_position_record.left_item == 2
    assert records[0].best_position_record.right_item == 4
    assert metadata["index_base"] == "zero_based"
    assert metadata["position_level_score_summaries"]["left_to_target"]["raw"]["max"] > 0.0
    assert metadata["position_level_score_summaries"]["target_to_right"]["raw"]["max"] > 0.0


def test_local_scorer_excludes_pos0_even_when_pos0_context_is_strong() -> None:
    target = 9
    train_sub = [[5, 9] for _ in range(10)]
    candidates = [[5, 1, 2]]
    scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=train_sub,
        config=_local_config(),
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=candidates, target_item=target)
    session_record = records[0]

    assert session_record.best_position != 0
    assert session_record.valid_position_count == 2
    assert "session_records" not in metadata


def test_local_scorer_allows_last_nonzero_position_with_missing_right_neighbor() -> None:
    target = 9
    train_sub = [[2, 9] for _ in range(10)] + [[1, 9, 3]]
    candidates = [[1, 2, 3]]
    scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=train_sub,
        config=_local_config(),
        embedding_table=_embeddings(),
    )

    records, _ = scorer.score(candidate_sessions=candidates, target_item=target)

    assert records[0].best_position == 2
    assert records[0].best_position_record is not None
    assert records[0].best_position_record.right_item is None


def test_local_scorer_skips_noop_target_replacement_positions() -> None:
    target = 9
    train_sub = [[1, 9, 2] for _ in range(5)]
    candidates = [[1, 9, 2]]
    scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=train_sub,
        config=_local_config(),
        embedding_table=_embeddings(),
    )

    records, metadata = scorer.score(candidate_sessions=candidates, target_item=target)

    assert records[0].best_position != 1
    assert records[0].valid_position_count == 1
    assert metadata["pre_existing_target_in_pool_count"] == 1
    assert metadata["skipped_noop_target_position_count"] == 1


def test_local_scorer_only_saves_all_session_records_when_debug_enabled() -> None:
    target = 9
    candidates = [[5, 1, 2]]
    default_scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=[[5, 9]],
        config=_local_config(),
        embedding_table=_embeddings(),
    )
    debug_scorer = HybridLocalPositionCompatibilityScorer(
        train_sub_sessions=[[5, 9]],
        config=_local_config(debug_save_all_session_records=True),
        embedding_table=_embeddings(),
    )

    _, default_metadata = default_scorer.score(
        candidate_sessions=candidates,
        target_item=target,
    )
    _, debug_metadata = debug_scorer.score(
        candidate_sessions=candidates,
        target_item=target,
    )

    assert "session_records" not in default_metadata
    assert "session_records" in debug_metadata
    assert debug_metadata["session_records"][0]["candidate_index"] == 0


def test_fixed_position_replacement_only_injects_selected_sessions() -> None:
    candidates = [[1, 2, 3], [4, 5, 6]]
    selected_records = [
        _session_record(index=0, session=candidates[0], score=1.0, best_position=1),
    ]

    result = build_targetized_selected_sessions_with_fixed_positions(
        candidate_sessions=candidates,
        selected_records=selected_records,
        target_item=99,
    )

    assert result.selected_candidate_indices == [0]
    assert result.selected_positions == [1]
    assert result.fake_sessions == [[1, 99, 3]]


def test_local_selector_excludes_invalid_records_and_is_deterministic() -> None:
    candidates = [
        [10, 11],
        [20, 21, 22],
        [30, 31, 32, 33],
        [40, 41, 42, 43, 44],
        [50, 51, 52, 53, 54],
    ]
    records = [
        _session_record(index=0, session=candidates[0], score=0.8, best_position=1),
        _session_record(index=1, session=candidates[1], score=0.7, best_position=1),
        _session_record(index=2, session=candidates[2], score=0.6, best_position=1),
        _session_record(index=3, session=candidates[3], score=0.9, best_position=1),
        _session_record(index=4, session=candidates[4], score=1.0, best_position=None),
    ]

    first = select_local_position_carriers(
        candidate_sessions=candidates,
        session_records=records,
        final_count=3,
        target_item=99,
        use_length_control=True,
        length_buckets="exact_until_4_plus",
    )
    second = select_local_position_carriers(
        candidate_sessions=candidates,
        session_records=records,
        final_count=3,
        target_item=99,
        use_length_control=True,
        length_buckets="exact_until_4_plus",
    )

    assert first.selected_indices == second.selected_indices
    assert 4 not in first.selected_indices
    assert len(first.selected_indices) == 3
    assert first.metadata["valid_position_count_summary"]["selected"]["min"] == 1.0


def test_local_selector_raises_when_valid_sessions_are_insufficient() -> None:
    candidates = [[1, 2], [3, 4]]
    records = [
        _session_record(index=0, session=candidates[0], score=1.0, best_position=1),
        _session_record(index=1, session=candidates[1], score=0.0, best_position=None),
    ]

    with pytest.raises(ValueError, match="fewer valid candidate sessions"):
        select_local_position_carriers(
            candidate_sessions=candidates,
            session_records=records,
            final_count=2,
            target_item=99,
            use_length_control=False,
            length_buckets="exact_until_4_plus",
        )


def _session_record(
    *,
    index: int,
    session: list[int],
    score: float,
    best_position: int | None,
) -> LocalPositionSessionRecord:
    best_record = None
    if best_position is not None:
        best_record = LocalPositionRecord(
            candidate_index=index,
            session=session,
            position=best_position,
            position_label=f"pos{best_position}",
            left_item=session[best_position - 1] if best_position > 0 else None,
            right_item=session[best_position + 1] if best_position + 1 < len(session) else None,
            replaced_item=session[best_position],
            raw_local_embedding_score=score,
            raw_left_to_target_score=score,
            raw_target_to_right_score=score,
            raw_local_transition_score=score,
            raw_session_compatibility_score=0.0,
            normalized_local_embedding_score=score,
            normalized_left_to_target_score=score,
            normalized_target_to_right_score=score,
            normalized_local_transition_score=score,
            normalized_session_compatibility_score=0.0,
            position_score=score,
        )
    return LocalPositionSessionRecord(
        index=index,
        session=session,
        valid_position_count=0 if best_position is None else 1,
        best_position=best_position,
        best_position_label=None if best_position is None else f"pos{best_position}",
        best_position_score=float("-inf") if best_position is None else score,
        invalid=best_position is None,
        best_position_record=best_record,
    )
