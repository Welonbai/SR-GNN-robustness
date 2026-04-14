from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrefixNonzeroWhenPossibleResult:
    session: list[int]
    position: int
    target_score: float


class PrefixNonzeroWhenPossiblePolicy:
    def __init__(self, runner, topk_ratio: float) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be within (0, 1].")
        self.runner = runner
        self.topk_ratio = float(topk_ratio)

    def _candidate_positions(self, length: int) -> range:
        topk_count = max(1, int(math.ceil(length * self.topk_ratio)))
        max_index = min(topk_count, length) - 1
        return range(0, max_index + 1)

    def _score_target_prefix(self, prefix: Sequence[int], target_item: int) -> float:
        scores = self.runner.score_session(prefix)
        target_index = int(target_item) - 1
        if target_index < 0 or target_index >= scores.shape[0]:
            raise ValueError("target_item is outside the score vector range.")
        return float(scores[target_index].item())

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> PrefixNonzeroWhenPossibleResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        if target_item <= 0:
            raise ValueError("target_item must be a positive item id.")

        candidate_positions = list(self._candidate_positions(len(session)))
        if len(candidate_positions) == 1:
            updated = list(session)
            updated[0] = int(target_item)
            return PrefixNonzeroWhenPossibleResult(
                session=updated,
                position=0,
                target_score=float("nan"),
            )

        best_pos = None
        best_score = None
        best_session: list[int] | None = None

        for pos in candidate_positions[1:]:
            candidate = list(session)
            candidate[pos] = int(target_item)
            prefix = candidate[:pos]
            score = self._score_target_prefix(prefix, target_item)
            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos
                best_session = candidate

        if best_session is None or best_pos is None or best_score is None:
            raise RuntimeError("Failed to select a best position.")

        return PrefixNonzeroWhenPossibleResult(
            session=best_session,
            position=int(best_pos),
            target_score=float(best_score),
        )


__all__ = [
    "PrefixNonzeroWhenPossiblePolicy",
    "PrefixNonzeroWhenPossibleResult",
]
