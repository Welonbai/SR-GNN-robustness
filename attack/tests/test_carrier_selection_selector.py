from __future__ import annotations

from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.carrier_selection.scorer import CarrierScoreRecord
from attack.carrier_selection.selector import (
    build_targetized_selected_sessions,
    select_carriers,
)
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy


def _record(index: int, session: list[int], score: float) -> CarrierScoreRecord:
    return CarrierScoreRecord(
        index=index,
        session=session,
        raw_embedding_score=score,
        raw_cooccurrence_score=score,
        raw_transition_score=score,
        normalized_embedding_score=score,
        normalized_cooccurrence_score=score,
        normalized_transition_score=score,
        carrier_score=score,
    )


def test_length_control_selects_high_scoring_sessions_per_bucket_deterministically() -> None:
    candidates = [
        [10, 11],
        [12, 13],
        [20, 21, 22],
        [23, 24, 25],
        [30, 31, 32, 33],
        [40, 41, 42, 43, 44],
        [45, 46, 47, 48, 49],
        [50, 51, 52, 53, 54],
    ]
    scores = [0.1, 0.9, 0.8, 0.2, 0.1, 0.3, 0.7, 0.6]
    records = [_record(index, session, scores[index]) for index, session in enumerate(candidates)]

    first = select_carriers(
        candidate_sessions=candidates,
        score_records=records,
        final_count=4,
        target_item=99,
        use_length_control=True,
        length_buckets="exact_until_4_plus",
    )
    second = select_carriers(
        candidate_sessions=candidates,
        score_records=records,
        final_count=4,
        target_item=99,
        use_length_control=True,
        length_buckets="exact_until_4_plus",
    )

    assert first.selected_indices == [1, 2, 4, 6]
    assert second.selected_indices == first.selected_indices
    assert first.metadata["selected_length_distribution"] == {
        "len2": 1,
        "len3": 1,
        "len4": 1,
        "len5_plus": 1,
    }


def test_global_selection_uses_highest_scores() -> None:
    candidates = [[1, 2], [3, 4], [5, 6], [7, 8]]
    records = [
        _record(0, candidates[0], 0.2),
        _record(1, candidates[1], 0.8),
        _record(2, candidates[2], 0.7),
        _record(3, candidates[3], 0.1),
    ]

    result = select_carriers(
        candidate_sessions=candidates,
        score_records=records,
        final_count=2,
        target_item=99,
        use_length_control=False,
        length_buckets="exact_until_4_plus",
    )

    assert result.selected_indices == [1, 2]


def test_targetized_selected_sessions_only_inject_selected_candidates() -> None:
    candidates = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    policy = RandomNonzeroWhenPossiblePolicy(1.0, rng=random.Random(123))

    result = build_targetized_selected_sessions(
        candidate_sessions=candidates,
        selected_indices=[2, 0],
        target_item=99,
        policy=policy,
    )

    assert result.selected_candidate_indices == [0, 2]
    assert len(result.fake_sessions) == 2
    assert all(99 in session for session in result.fake_sessions)
    assert candidates[1] not in result.fake_sessions
