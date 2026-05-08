from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class InternalRandomInsertionSuccessorRepairResult:
    session: list[int]
    inserted_session_before_repair: list[int]
    insertion_result: InternalRandomInsertionResult
    insertion_slot: int
    original_length: int
    inserted_length: int
    final_length: int
    left_item: int
    original_right_item: int
    repaired_right_item: int
    repair_applied: bool
    repair_changed_item: bool
    successor_pool_empty: bool
    sampled_successor: int | None
    successor_pool: list[int]
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


class InternalRandomInsertionSuccessorRepairPolicy:
    def __init__(
        self,
        topk_ratio: float,
        successor_counts: Counter[int] | Mapping[int, int],
        successor_top_k: int = 5,
        insertion_rng: random.Random | None = None,
        successor_rng: random.Random | None = None,
        exclude_target_from_successor_pool: bool = False,
    ) -> None:
        if successor_top_k < 1:
            raise ValueError("successor_top_k must be >= 1.")
        self.insertion_policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
            topk_ratio=topk_ratio,
            rng=insertion_rng,
        )
        self.successor_counts: Counter[int] = Counter(
            {int(item): int(count) for item, count in successor_counts.items()}
        )
        self.successor_top_k = int(successor_top_k)
        self.successor_rng = successor_rng or random.Random()
        self.exclude_target_from_successor_pool = bool(
            exclude_target_from_successor_pool
        )

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> InternalRandomInsertionSuccessorRepairResult:
        insertion_result = self.insertion_policy.apply_with_metadata(
            session,
            target_item,
        )
        target = int(target_item)
        insertion_slot = int(insertion_result.insertion_slot)
        inserted = [int(item) for item in insertion_result.session]
        final = list(inserted)
        successor_pool = self._successor_pool(target)
        successor_pool_empty = not successor_pool
        sampled_successor: int | None = None
        repair_applied = False
        repair_changed_item = False
        original_right_item = int(insertion_result.right_item)
        repaired_right_item = original_right_item

        if successor_pool:
            sampled_successor = self._sample_successor(successor_pool)
            repaired_right_item = int(sampled_successor)
            final[insertion_slot + 1] = repaired_right_item
            repair_applied = True
            repair_changed_item = repaired_right_item != original_right_item

        return InternalRandomInsertionSuccessorRepairResult(
            session=final,
            inserted_session_before_repair=inserted,
            insertion_result=insertion_result,
            insertion_slot=insertion_slot,
            original_length=int(insertion_result.original_length),
            inserted_length=int(insertion_result.inserted_length),
            final_length=int(len(final)),
            left_item=int(insertion_result.left_item),
            original_right_item=original_right_item,
            repaired_right_item=repaired_right_item,
            repair_applied=repair_applied,
            repair_changed_item=repair_changed_item,
            successor_pool_empty=successor_pool_empty,
            sampled_successor=sampled_successor,
            successor_pool=successor_pool,
            pre_existing_target_count=int(insertion_result.pre_existing_target_count),
            target_occurrence_count_after_insertion=int(
                insertion_result.target_occurrence_count_after_insertion
            ),
            target_occurrence_count_final=int(
                sum(1 for item in final if item == target)
            ),
        )

    def _successor_pool(self, target_item: int) -> list[int]:
        items = [
            (int(item), int(count))
            for item, count in self.successor_counts.items()
            if int(count) > 0
            and (
                not self.exclude_target_from_successor_pool
                or int(item) != int(target_item)
            )
        ]
        items.sort(key=lambda pair: (-pair[1], pair[0]))
        return [item for item, _ in items[: self.successor_top_k]]

    def _sample_successor(self, successor_pool: Sequence[int]) -> int:
        weighted_pool = [
            (int(item), int(self.successor_counts[int(item)]))
            for item in successor_pool
            if int(self.successor_counts[int(item)]) > 0
        ]
        total = sum(count for _, count in weighted_pool)
        if total <= 0:
            raise RuntimeError("Successor pool has no positive empirical counts.")
        draw = self.successor_rng.randrange(total)
        running = 0
        for item, count in weighted_pool:
            running += count
            if draw < running:
                return int(item)
        return int(weighted_pool[-1][0])


def build_target_successor_counts(
    train_sessions: Sequence[Sequence[int]],
    target_item: int,
    exclude_target: bool = False,
) -> Counter[int]:
    target = int(target_item)
    counts: Counter[int] = Counter()
    for session in train_sessions:
        normalized = [int(item) for item in session]
        for index, item in enumerate(normalized[:-1]):
            if item != target:
                continue
            successor = int(normalized[index + 1])
            if exclude_target and successor == target:
                continue
            counts[successor] += 1
    return counts


__all__ = [
    "InternalRandomInsertionSuccessorRepairPolicy",
    "InternalRandomInsertionSuccessorRepairResult",
    "InternalRandomInsertionTruncateSuffixPolicy",
    "InternalRandomInsertionTruncateSuffixResult",
    "build_target_successor_counts",
]
