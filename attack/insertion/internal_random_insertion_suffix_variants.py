from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
    InternalRandomInsertionResult,
)


@dataclass(frozen=True)
class InternalRandomInsertionTruncateSuffixResult:
    session: list[int]
    inserted_session_before_truncation: list[int]
    insertion_result: InternalRandomInsertionResult
    insertion_slot: int
    original_length: int
    inserted_length: int
    final_length: int
    left_item: int
    original_right_item_before_truncation: int
    truncated_suffix: list[int]
    pre_existing_target_count: int
    target_occurrence_count_after_insertion: int
    target_occurrence_count_final: int


class InternalRandomInsertionTruncateSuffixPolicy:
    def __init__(
        self,
        topk_ratio: float,
        rng: random.Random | None = None,
    ) -> None:
        self.insertion_policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
            topk_ratio=topk_ratio,
            rng=rng,
        )

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> InternalRandomInsertionTruncateSuffixResult:
        insertion_result = self.insertion_policy.apply_with_metadata(
            session,
            target_item,
        )
        target = int(target_item)
        insertion_slot = int(insertion_result.insertion_slot)
        inserted = [int(item) for item in insertion_result.session]
        final = inserted[: insertion_slot + 1]
        if len(final) < 2:
            raise RuntimeError("Truncated internal insertion session is too short.")
        if final[-1] != target:
            raise RuntimeError("Truncated internal insertion target is not final.")
        truncated_suffix = inserted[insertion_slot + 1 :]

        return InternalRandomInsertionTruncateSuffixResult(
            session=final,
            inserted_session_before_truncation=inserted,
            insertion_result=insertion_result,
            insertion_slot=insertion_slot,
            original_length=int(insertion_result.original_length),
            inserted_length=int(insertion_result.inserted_length),
            final_length=int(len(final)),
            left_item=int(insertion_result.left_item),
            original_right_item_before_truncation=int(insertion_result.right_item),
            truncated_suffix=truncated_suffix,
            pre_existing_target_count=int(insertion_result.pre_existing_target_count),
            target_occurrence_count_after_insertion=int(
                insertion_result.target_occurrence_count_after_insertion
            ),
            target_occurrence_count_final=int(
                sum(1 for item in final if item == target)
            ),
        )


__all__ = [
    "InternalRandomInsertionTruncateSuffixPolicy",
    "InternalRandomInsertionTruncateSuffixResult",
]
