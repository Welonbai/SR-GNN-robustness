from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .base_policy import InsertionPolicy


@dataclass(frozen=True)
class InternalRandomInsertionResult:
    session: list[int]
    insertion_slot: int
    original_length: int
    inserted_length: int
    left_item: int
    right_item: int
    pre_existing_target_count: int
    target_occurrence_count_after_insertion: int


class InternalRandomInsertionNonzeroWhenPossiblePolicy(InsertionPolicy):
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
    ) -> InternalRandomInsertionResult:
        original = [int(item) for item in session]
        if len(original) < 2:
            raise ValueError(
                "Internal-Random-Insertion-NZ requires session length >= 2; "
                "there is no valid internal nonzero insertion slot."
            )

        target = int(target_item)
        length = len(original)
        valid_internal_slot_count = length - 1
        topk_count = max(
            1,
            int(math.ceil(valid_internal_slot_count * self.topk_ratio)),
        )
        max_slot = min(topk_count, valid_internal_slot_count)
        insertion_slot = self.rng.randint(1, max_slot)
        if insertion_slot < 1 or insertion_slot > length - 1:
            raise RuntimeError("Sampled insertion slot is not internal.")

        updated = list(original)
        updated.insert(insertion_slot, target)
        return InternalRandomInsertionResult(
            session=updated,
            insertion_slot=int(insertion_slot),
            original_length=int(length),
            inserted_length=int(len(updated)),
            left_item=int(original[insertion_slot - 1]),
            right_item=int(original[insertion_slot]),
            pre_existing_target_count=int(
                sum(1 for item in original if item == target)
            ),
            target_occurrence_count_after_insertion=int(
                sum(1 for item in updated if item == target)
            ),
        )


__all__ = [
    "InternalRandomInsertionNonzeroWhenPossiblePolicy",
    "InternalRandomInsertionResult",
]
