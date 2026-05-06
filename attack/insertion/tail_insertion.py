from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TailInsertionResult:
    session: list[int]
    insertion_slot: int
    original_length: int
    inserted_length: int
    pre_existing_target_count: int
    target_occurrence_count_after_insertion: int


class TailInsertionPolicy:
    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> TailInsertionResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        original = [int(item) for item in session]
        target = int(target_item)
        pre_existing_target_count = sum(1 for item in original if item == target)
        updated = list(original)
        insertion_slot = len(updated)
        updated.append(target)
        return TailInsertionResult(
            session=updated,
            insertion_slot=int(insertion_slot),
            original_length=int(len(original)),
            inserted_length=int(len(updated)),
            pre_existing_target_count=int(pre_existing_target_count),
            target_occurrence_count_after_insertion=int(
                pre_existing_target_count + 1
            ),
        )


__all__ = ["TailInsertionPolicy", "TailInsertionResult"]
