from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from attack.position_opt.candidate_builder import build_candidate_position_result


@dataclass(frozen=True)
class RankCandidateSessionState:
    fake_session_index: int
    original_session: tuple[int, ...]
    session_length: int
    base_candidate_positions: tuple[int, ...]
    candidate_positions: tuple[int, ...]
    candidate_count: int
    availability_group: str
    rank1_position: int
    rank2_position: int | None
    tail_positions: tuple[int, ...]
    pos0_removed: bool
    fallback_to_pos0_only: bool


def build_rank_candidate_states(
    fake_sessions: Sequence[Sequence[int]],
    replacement_topk_ratio: float,
    nonzero_action_when_possible: bool,
) -> list[RankCandidateSessionState]:
    states: list[RankCandidateSessionState] = []
    for fake_session_index, session in enumerate(fake_sessions):
        session_items = tuple(int(item) for item in session)
        candidate_result = build_candidate_position_result(
            session_items,
            replacement_topk_ratio,
            nonzero_action_when_possible=nonzero_action_when_possible,
        )
        base_candidate_positions = tuple(
            sorted(int(position) for position in candidate_result.positions_before_mask)
        )
        candidate_positions = tuple(
            sorted(int(position) for position in candidate_result.positions)
        )
        candidate_count = len(candidate_positions)
        if candidate_count <= 0:
            raise ValueError(
                "RankBucket-CEM encountered a fake session with zero available "
                "replacement candidates after candidate masking."
            )
        availability_group = _availability_group(candidate_count)
        states.append(
            RankCandidateSessionState(
                fake_session_index=int(fake_session_index),
                original_session=session_items,
                session_length=int(len(session_items)),
                base_candidate_positions=base_candidate_positions,
                candidate_positions=candidate_positions,
                candidate_count=int(candidate_count),
                availability_group=availability_group,
                rank1_position=int(candidate_positions[0]),
                rank2_position=(
                    None if candidate_count < 2 else int(candidate_positions[1])
                ),
                tail_positions=tuple(int(position) for position in candidate_positions[2:]),
                pos0_removed=bool(candidate_result.pos0_removed),
                fallback_to_pos0_only=bool(candidate_result.fallback_to_pos0_only),
            )
        )
    if not states:
        raise ValueError("fake_sessions must contain at least one session.")
    return states


def build_availability_summary(
    states: Sequence[RankCandidateSessionState],
) -> dict[str, Any]:
    if not states:
        raise ValueError("states must not be empty.")

    candidate_counts = [int(state.candidate_count) for state in states]
    total = len(states)
    g1_count = sum(1 for state in states if state.availability_group == "G1")
    g2_count = sum(1 for state in states if state.availability_group == "G2")
    g3_count = sum(1 for state in states if state.availability_group == "G3")
    pos0_removed_count = sum(1 for state in states if state.pos0_removed)
    fallback_to_pos0_only_count = sum(1 for state in states if state.fallback_to_pos0_only)
    tail_available_count = sum(1 for state in states if state.tail_positions)

    return {
        "total_fake_sessions": int(total),
        "G1_count": int(g1_count),
        "G1_pct": _ratio(g1_count, total),
        "G2_count": int(g2_count),
        "G2_pct": _ratio(g2_count, total),
        "G3_count": int(g3_count),
        "G3_pct": _ratio(g3_count, total),
        "candidate_count_min": int(min(candidate_counts)),
        "candidate_count_max": int(max(candidate_counts)),
        "candidate_count_mean": _mean(candidate_counts),
        "candidate_count_median": float(median(candidate_counts)),
        "candidate_count_p25": _percentile(candidate_counts, 25.0),
        "candidate_count_p75": _percentile(candidate_counts, 75.0),
        "candidate_count_p90": _percentile(candidate_counts, 90.0),
        "candidate_count_p95": _percentile(candidate_counts, 95.0),
        "pos0_removed_count": int(pos0_removed_count),
        "pos0_removed_pct": _ratio(pos0_removed_count, total),
        "fallback_to_pos0_only_count": int(fallback_to_pos0_only_count),
        "fallback_to_pos0_only_pct": _ratio(fallback_to_pos0_only_count, total),
        "tail_available_count": int(tail_available_count),
        "tail_available_pct": _ratio(tail_available_count, total),
    }


def _availability_group(candidate_count: int) -> str:
    if candidate_count == 1:
        return "G1"
    if candidate_count == 2:
        return "G2"
    return "G3"


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(float(value) for value in values) / len(values))


def _percentile(values: Sequence[int | float], percentile: float) -> float:
    normalized = sorted(float(value) for value in values)
    if len(normalized) == 1:
        return float(normalized[0])
    rank = (float(percentile) / 100.0) * float(len(normalized) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(normalized[lower_index])
    weight = rank - float(lower_index)
    lower_value = normalized[lower_index]
    upper_value = normalized[upper_index]
    return float((1.0 - weight) * lower_value + weight * upper_value)


def _ratio(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total) * 100.0


__all__ = [
    "RankCandidateSessionState",
    "build_availability_summary",
    "build_rank_candidate_states",
]
