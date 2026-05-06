from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.tools.target_anchor_survey import (
    anchor_score,
    compute_neighbor_stats,
    fake_session_anchor_availability,
    filter_vulnerable_cases,
)


def test_synthetic_session_neighbor_stats_counts_predecessor_successor_cooccurrence() -> None:
    sessions = [
        [1, 2, 9, 3],
        [2, 9, 4],
        [2, 5, 9],
        [7, 8],
    ]

    stats = compute_neighbor_stats(sessions, 9, top_k=10)

    predecessors = {row["predecessor_item"]: row for row in stats["predecessors"]}
    successors = {row["successor_item"]: row for row in stats["successors"]}
    cooccur = {row["cooccur_item"]: row for row in stats["cooccurrence"]}
    assert predecessors[2]["count_i_to_target"] == 2
    assert predecessors[5]["count_i_to_target"] == 1
    assert successors[3]["count_target_to_j"] == 1
    assert successors[4]["count_target_to_j"] == 1
    assert cooccur[2]["cooccur_session_count"] == 3
    assert cooccur[1]["cooccur_session_count"] == 1


def test_vulnerable_prefix_filtering_uses_open_min_closed_max() -> None:
    ranks = [1, 20, 21, 200, 201]

    assert filter_vulnerable_cases(ranks, rank_min=20, rank_max=200) == [2, 3]


def test_anchor_score_increases_with_coverage_and_fake_availability() -> None:
    base = anchor_score(
        vulnerable_coverage=0.05,
        fake_session_count_with_anchor=2,
        avg_vulnerable_target_rank=50,
    )
    more_coverage = anchor_score(
        vulnerable_coverage=0.10,
        fake_session_count_with_anchor=2,
        avg_vulnerable_target_rank=50,
    )
    more_fake = anchor_score(
        vulnerable_coverage=0.05,
        fake_session_count_with_anchor=20,
        avg_vulnerable_target_rank=50,
    )

    assert more_coverage > base
    assert more_fake > base


def test_fake_session_availability_feasible_count_excludes_anchor_at_tail() -> None:
    fake_sessions = [
        [1, 2, 3],
        [4, 1],
        [1, 5, 1],
    ]

    availability = fake_session_anchor_availability(fake_sessions, [1])[1]

    assert availability["fake_session_count_with_anchor"] == 3
    assert availability["total_anchor_occurrences_in_fake_sessions"] == 4
    assert availability["anchor_after_insertion_feasible_count"] == 2
    assert availability["internal_insertion_feasible_ratio"] == pytest.approx(0.5)

