from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .base_policy import InsertionPolicy


@dataclass(frozen=True)
class InternalRandomReplacementResult:
    session: list[int]
    replacement_position: int
    original_length: int
    replaced_length: int
    original_item: int
    left_item: int | None
    right_item: int | None
    pre_existing_target_count: int
    target_occurrence_count_after_replacement: int
    was_noop: bool
    used_internal_position: bool
    used_tail_fallback: bool
    internal_candidate_count: int
    candidate_positions: list[int]
    restricted_candidate_positions: list[int]


class InternalRandomReplacementNonzeroWhenPossiblePolicy(InsertionPolicy):
    def __init__(self, topk_ratio: float, rng: random.Random | None = None) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be within (0, 1].")
        self.topk_ratio = float(topk_ratio)
        self.rng = rng or random.Random()

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> InternalRandomReplacementResult:
        original = [int(item) for item in session]
        if len(original) < 2:
            raise ValueError(
                "Internal-Random-Replacement-NZ requires session length >= 2; "
                "there is no valid nonzero replacement position."
            )

        target = int(target_item)
        length = len(original)
        if length >= 3:
            candidate_positions = list(range(1, length - 1))
            used_internal_position = True
            used_tail_fallback = False
            internal_candidate_count = len(candidate_positions)
        else:
            candidate_positions = [1]
            used_internal_position = False
            used_tail_fallback = True
            internal_candidate_count = 0

        topk_count = max(
            1,
            int(math.ceil(len(candidate_positions) * self.topk_ratio)),
        )
        restricted_candidate_positions = candidate_positions[:topk_count]
        replacement_position = int(self.rng.choice(restricted_candidate_positions))

        original_item = int(original[replacement_position])
        updated = list(original)
        updated[replacement_position] = target
        return InternalRandomReplacementResult(
            session=updated,
            replacement_position=replacement_position,
            original_length=int(length),
            replaced_length=int(len(updated)),
            original_item=original_item,
            left_item=int(original[replacement_position - 1]),
            right_item=(
                None
                if used_tail_fallback
                else int(original[replacement_position + 1])
            ),
            pre_existing_target_count=int(
                sum(1 for item in original if item == target)
            ),
            target_occurrence_count_after_replacement=int(
                sum(1 for item in updated if item == target)
            ),
            was_noop=bool(original_item == target),
            used_internal_position=bool(used_internal_position),
            used_tail_fallback=bool(used_tail_fallback),
            internal_candidate_count=int(internal_candidate_count),
            candidate_positions=[int(position) for position in candidate_positions],
            restricted_candidate_positions=[
                int(position) for position in restricted_candidate_positions
            ],
        )


__all__ = [
    "InternalRandomReplacementNonzeroWhenPossiblePolicy",
    "InternalRandomReplacementResult",
]
