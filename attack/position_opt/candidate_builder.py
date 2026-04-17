from __future__ import annotations

import math
from typing import Sequence


def build_candidate_positions(
    session: Sequence[int],
    replacement_topk_ratio: float,
) -> list[int]:
    if not session:
        raise ValueError("session must contain at least one item.")
    if not (0.0 < replacement_topk_ratio <= 1.0):
        raise ValueError("replacement_topk_ratio must be within (0, 1].")

    session_length = len(session)
    topk_count = max(1, int(math.ceil(session_length * float(replacement_topk_ratio))))
    max_index = min(topk_count, session_length) - 1
    return list(range(0, max_index + 1))


__all__ = ["build_candidate_positions"]
