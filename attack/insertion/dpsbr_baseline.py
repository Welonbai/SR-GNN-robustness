from __future__ import annotations

import math
import random
from typing import Sequence

from .base_policy import InsertionPolicy


class DPSBRBaselinePolicy(InsertionPolicy):
    def __init__(self, topk_ratio: float, rng: random.Random | None = None) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be within (0, 1].")
        self.topk_ratio = float(topk_ratio)
        self.rng = rng or random.Random()

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        if not session:
            raise ValueError("Session must contain at least one item.")
        length = len(session)
        topk_count = max(1, int(math.ceil(length * self.topk_ratio)))
        max_index = min(topk_count, length) - 1
        replace_index = self.rng.randint(0, max_index)
        updated = list(session)
        updated[replace_index] = int(target_item)
        return updated


__all__ = ["DPSBRBaselinePolicy"]
