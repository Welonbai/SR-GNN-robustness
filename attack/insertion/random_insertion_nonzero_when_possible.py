from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .base_policy import InsertionPolicy


@dataclass(frozen=True)
class RandomInsertionNonzeroWhenPossibleResult:
    session: list[int]
    insertion_slot: int


class RandomInsertionNonzeroWhenPossiblePolicy(InsertionPolicy):
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
    ) -> RandomInsertionNonzeroWhenPossibleResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        length = len(session)
        topk_count = max(1, int(math.ceil(length * self.topk_ratio)))
        max_slot = min(topk_count, length)
        insertion_slot = self.rng.randint(1, max_slot)
        updated = list(session)
        updated.insert(insertion_slot, int(target_item))
        return RandomInsertionNonzeroWhenPossibleResult(
            session=updated,
            insertion_slot=int(insertion_slot),
        )


__all__ = [
    "RandomInsertionNonzeroWhenPossiblePolicy",
    "RandomInsertionNonzeroWhenPossibleResult",
]
