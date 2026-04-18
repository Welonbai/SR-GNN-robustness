from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from .base_policy import InsertionPolicy


@dataclass(frozen=True)
class RandomNonzeroWhenPossibleResult:
    session: list[int]
    position: int


class RandomNonzeroWhenPossiblePolicy(InsertionPolicy):
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
    ) -> RandomNonzeroWhenPossibleResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        length = len(session)
        topk_count = max(1, int(math.ceil(length * self.topk_ratio)))
        max_index = min(topk_count, length) - 1
        if topk_count == 1:
            replace_index = 0
        else:
            replace_index = self.rng.randint(1, max_index)
        updated = list(session)
        updated[replace_index] = int(target_item)
        return RandomNonzeroWhenPossibleResult(session=updated, position=int(replace_index))


__all__ = ["RandomNonzeroWhenPossiblePolicy", "RandomNonzeroWhenPossibleResult"]
