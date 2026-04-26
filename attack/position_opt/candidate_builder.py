from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class CandidatePositionBuildResult:
    positions_before_mask: tuple[int, ...]
    positions: tuple[int, ...]
    nonzero_action_when_possible: bool
    pos0_removed: bool
    forced_single_candidate: bool
    fallback_to_pos0_only: bool


def build_candidate_positions(
    session: Sequence[int],
    replacement_topk_ratio: float,
    *,
    nonzero_action_when_possible: bool = False,
) -> list[int]:
    return list(
        build_candidate_position_result(
            session,
            replacement_topk_ratio,
            nonzero_action_when_possible=nonzero_action_when_possible,
        ).positions
    )


def build_candidate_position_result(
    session: Sequence[int],
    replacement_topk_ratio: float,
    *,
    nonzero_action_when_possible: bool = False,
) -> CandidatePositionBuildResult:
    if not session:
        raise ValueError("session must contain at least one item.")
    if not (0.0 < replacement_topk_ratio <= 1.0):
        raise ValueError("replacement_topk_ratio must be within (0, 1].")

    session_length = len(session)
    topk_count = max(1, int(math.ceil(session_length * float(replacement_topk_ratio))))
    max_index = min(topk_count, session_length) - 1
    positions_before_mask = tuple(range(0, max_index + 1))
    positions = filter_candidate_positions_nonzero_when_possible(
        positions_before_mask,
        enabled=nonzero_action_when_possible,
    )
    pos0_removed = positions != positions_before_mask
    return CandidatePositionBuildResult(
        positions_before_mask=positions_before_mask,
        positions=positions,
        nonzero_action_when_possible=bool(nonzero_action_when_possible),
        pos0_removed=bool(pos0_removed),
        forced_single_candidate=(len(positions) == 1),
        fallback_to_pos0_only=bool(
            nonzero_action_when_possible
            and positions_before_mask == (0,)
            and positions == (0,)
        ),
    )


def filter_candidate_positions_nonzero_when_possible(
    candidate_positions: Sequence[int],
    *,
    enabled: bool,
) -> tuple[int, ...]:
    normalized_positions = tuple(int(position) for position in candidate_positions)
    if not enabled:
        return normalized_positions
    nonzero_positions = tuple(position for position in normalized_positions if position != 0)
    if nonzero_positions:
        return nonzero_positions
    return normalized_positions


__all__ = [
    "CandidatePositionBuildResult",
    "build_candidate_position_result",
    "build_candidate_positions",
    "filter_candidate_positions_nonzero_when_possible",
]
