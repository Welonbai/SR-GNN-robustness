from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from attack.insertion.generated_continuation_suffix import (
    PURE_GENERATED_MODE_RNG_TAG,
    GeneratedContinuationAppliedResult,
    TargetExposureForSuffix,
    apply_generated_continuation_to_exposure,
    deterministic_session_rng,
    generate_poison_model_suffix,
)
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
    InternalRandomInsertionResult,
)


@dataclass(frozen=True)
class InternalRandomInsertionGeneratedContinuationResult:
    session: list[int]
    inserted_session_before_generation: list[int]
    insertion_result: InternalRandomInsertionResult
    insertion_slot: int
    original_length: int
    inserted_length: int
    final_length: int
    left_item: int
    original_right_item: int
    original_suffix: list[int]
    generated_suffix: list[int]
    target_position: int
    suffix_length: int
    pre_existing_target_count: int
    target_occurrence_count_after_insertion: int
    target_occurrence_count_final: int
    generated_suffix_contains_target_count: int
    generated_suffix_unique_item_count: int


class InternalRandomInsertionGeneratedContinuationPolicy:
    def __init__(
        self,
        topk_ratio: float,
        poison_runner,
        generation_topk: int,
        insertion_rng: random.Random | None = None,
        generation_rng: random.Random | None = None,
        generation_rng_base_seed: int = 0,
        pure_generated_mode_rng_tag: str = PURE_GENERATED_MODE_RNG_TAG,
    ) -> None:
        self.insertion_policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
            topk_ratio=topk_ratio,
            rng=insertion_rng,
        )
        self.poison_runner = poison_runner
        self.generation_topk = int(generation_topk)
        self.generation_rng = generation_rng
        self.generation_rng_base_seed = int(generation_rng_base_seed)
        self.pure_generated_mode_rng_tag = str(pure_generated_mode_rng_tag)

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
        fake_session_index: int = 0,
    ) -> InternalRandomInsertionGeneratedContinuationResult:
        insertion_result = self.insertion_policy.apply_with_metadata(
            session,
            target_item,
        )
        if self.generation_rng is not None:
            return _build_generated_result_with_rng(
                insertion_result=insertion_result,
                target_item=int(target_item),
                poison_runner=self.poison_runner,
                generation_topk=self.generation_topk,
                generation_rng=self.generation_rng,
            )

        exposure = _insertion_result_to_exposure(
            insertion_result=insertion_result,
            original_session=session,
            target_item=int(target_item),
        )
        generated_result = apply_generated_continuation_to_exposure(
            exposure,
            poison_runner=self.poison_runner,
            generation_topk=self.generation_topk,
            generation_rng_base_seed=self.generation_rng_base_seed,
            target_item=int(target_item),
            fake_session_index=int(fake_session_index),
            rng_tag=self.pure_generated_mode_rng_tag,
        )
        return _result_from_generated_result(
            insertion_result=insertion_result,
            target_item=int(target_item),
            generated_result=generated_result,
        )


def _insertion_result_to_exposure(
    *,
    insertion_result: InternalRandomInsertionResult,
    original_session: Sequence[int],
    target_item: int,
) -> TargetExposureForSuffix:
    slot = int(insertion_result.insertion_slot)
    inserted = [int(item) for item in insertion_result.session]
    return TargetExposureForSuffix(
        original_session=[int(item) for item in original_session],
        session_before_suffix=inserted,
        target_item=int(target_item),
        target_position=slot,
        operation="internal_random_insertion_nonzero_when_possible",
        original_suffix=inserted[slot + 1 :],
        left_item=int(insertion_result.left_item),
        right_item=int(insertion_result.right_item),
        action_position=slot,
        operation_metadata={
            "insertion_slot": slot,
            "original_length": int(insertion_result.original_length),
            "inserted_length": int(insertion_result.inserted_length),
        },
    )


def _result_from_generated_result(
    *,
    insertion_result: InternalRandomInsertionResult,
    target_item: int,
    generated_result: GeneratedContinuationAppliedResult,
) -> InternalRandomInsertionGeneratedContinuationResult:
    return InternalRandomInsertionGeneratedContinuationResult(
        session=generated_result.session,
        inserted_session_before_generation=[
            int(item) for item in insertion_result.session
        ],
        insertion_result=insertion_result,
        insertion_slot=int(insertion_result.insertion_slot),
        original_length=int(insertion_result.original_length),
        inserted_length=int(insertion_result.inserted_length),
        final_length=int(generated_result.final_length),
        left_item=int(insertion_result.left_item),
        original_right_item=int(insertion_result.right_item),
        original_suffix=generated_result.original_suffix,
        generated_suffix=generated_result.generated_suffix,
        target_position=int(generated_result.final_target_position),
        suffix_length=int(generated_result.suffix_length),
        pre_existing_target_count=int(insertion_result.pre_existing_target_count),
        target_occurrence_count_after_insertion=int(
            insertion_result.target_occurrence_count_after_insertion
        ),
        target_occurrence_count_final=int(
            generated_result.target_occurrence_count_final
        ),
        generated_suffix_contains_target_count=int(
            generated_result.generated_suffix_contains_target_count
        ),
        generated_suffix_unique_item_count=int(
            generated_result.generated_suffix_unique_item_count
        ),
    )


def _build_generated_result_with_rng(
    *,
    insertion_result: InternalRandomInsertionResult,
    target_item: int,
    poison_runner,
    generation_topk: int,
    generation_rng: random.Random,
) -> InternalRandomInsertionGeneratedContinuationResult:
    slot = int(insertion_result.insertion_slot)
    inserted = [int(item) for item in insertion_result.session]
    prefix_through_target = inserted[: slot + 1]
    original_suffix = inserted[slot + 1 :]
    generated_suffix = generate_poison_model_suffix(
        runner=poison_runner,
        prefix=prefix_through_target,
        suffix_length=len(original_suffix),
        topk=int(generation_topk),
        rng=generation_rng,
    )
    final = prefix_through_target + generated_suffix
    return InternalRandomInsertionGeneratedContinuationResult(
        session=final,
        inserted_session_before_generation=inserted,
        insertion_result=insertion_result,
        insertion_slot=slot,
        original_length=int(insertion_result.original_length),
        inserted_length=int(insertion_result.inserted_length),
        final_length=int(len(final)),
        left_item=int(insertion_result.left_item),
        original_right_item=int(insertion_result.right_item),
        original_suffix=original_suffix,
        generated_suffix=generated_suffix,
        target_position=slot,
        suffix_length=int(len(original_suffix)),
        pre_existing_target_count=int(insertion_result.pre_existing_target_count),
        target_occurrence_count_after_insertion=int(
            insertion_result.target_occurrence_count_after_insertion
        ),
        target_occurrence_count_final=int(
            sum(1 for item in final if int(item) == int(target_item))
        ),
        generated_suffix_contains_target_count=int(
            sum(1 for item in generated_suffix if int(item) == int(target_item))
        ),
        generated_suffix_unique_item_count=int(len(set(generated_suffix))),
    )


__all__ = [
    "InternalRandomInsertionGeneratedContinuationPolicy",
    "InternalRandomInsertionGeneratedContinuationResult",
    "PURE_GENERATED_MODE_RNG_TAG",
    "deterministic_session_rng",
    "generate_poison_model_suffix",
]
