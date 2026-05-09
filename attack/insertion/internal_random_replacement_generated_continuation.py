from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from attack.insertion.generated_continuation_suffix import (
    PURE_GENERATED_MODE_RNG_TAG,
    GeneratedContinuationAppliedResult,
    TargetExposureForSuffix,
    apply_generated_continuation_to_exposure,
)
from attack.insertion.internal_random_replacement_nonzero_when_possible import (
    InternalRandomReplacementNonzeroWhenPossiblePolicy,
    InternalRandomReplacementResult,
)


@dataclass(frozen=True)
class InternalRandomReplacementGeneratedContinuationResult:
    session: list[int]
    replaced_session_before_generation: list[int]
    replacement_result: InternalRandomReplacementResult
    exposure: TargetExposureForSuffix
    generated_result: GeneratedContinuationAppliedResult
    replacement_position: int
    original_length: int
    replaced_length: int
    final_length: int
    left_item: int | None
    original_right_item: int | None
    original_suffix: list[int]
    generated_suffix: list[int]
    suffix_length: int
    pre_existing_target_count: int
    target_occurrence_count_after_replacement: int
    target_occurrence_count_final: int


class InternalRandomReplacementGeneratedContinuationPolicy:
    def __init__(
        self,
        topk_ratio: float,
        poison_runner,
        generation_topk: int,
        replacement_rng: random.Random | None = None,
        generation_rng_base_seed: int = 0,
        pure_generated_mode_rng_tag: str = PURE_GENERATED_MODE_RNG_TAG,
    ) -> None:
        self.replacement_policy = InternalRandomReplacementNonzeroWhenPossiblePolicy(
            topk_ratio=topk_ratio,
            rng=replacement_rng,
        )
        self.poison_runner = poison_runner
        self.generation_topk = int(generation_topk)
        self.generation_rng_base_seed = int(generation_rng_base_seed)
        self.pure_generated_mode_rng_tag = str(pure_generated_mode_rng_tag)

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
        fake_session_index: int = 0,
    ) -> InternalRandomReplacementGeneratedContinuationResult:
        replacement_result = self.replacement_policy.apply_with_metadata(
            session,
            target_item,
        )
        exposure = _replacement_result_to_exposure(
            replacement_result=replacement_result,
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
        return InternalRandomReplacementGeneratedContinuationResult(
            session=generated_result.session,
            replaced_session_before_generation=[
                int(item) for item in replacement_result.session
            ],
            replacement_result=replacement_result,
            exposure=exposure,
            generated_result=generated_result,
            replacement_position=int(replacement_result.replacement_position),
            original_length=int(replacement_result.original_length),
            replaced_length=int(replacement_result.replaced_length),
            final_length=int(generated_result.final_length),
            left_item=(
                None
                if replacement_result.left_item is None
                else int(replacement_result.left_item)
            ),
            original_right_item=(
                None
                if replacement_result.right_item is None
                else int(replacement_result.right_item)
            ),
            original_suffix=generated_result.original_suffix,
            generated_suffix=generated_result.generated_suffix,
            suffix_length=int(generated_result.suffix_length),
            pre_existing_target_count=int(
                replacement_result.pre_existing_target_count
            ),
            target_occurrence_count_after_replacement=int(
                replacement_result.target_occurrence_count_after_replacement
            ),
            target_occurrence_count_final=int(
                generated_result.target_occurrence_count_final
            ),
        )


def _replacement_result_to_exposure(
    *,
    replacement_result: InternalRandomReplacementResult,
    original_session: Sequence[int],
    target_item: int,
) -> TargetExposureForSuffix:
    position = int(replacement_result.replacement_position)
    replaced = [int(item) for item in replacement_result.session]
    return TargetExposureForSuffix(
        original_session=[int(item) for item in original_session],
        session_before_suffix=replaced,
        target_item=int(target_item),
        target_position=position,
        operation="internal_random_replacement_nonzero_when_possible",
        original_suffix=replaced[position + 1 :],
        left_item=(
            None
            if replacement_result.left_item is None
            else int(replacement_result.left_item)
        ),
        right_item=(
            None
            if replacement_result.right_item is None
            else int(replacement_result.right_item)
        ),
        action_position=position,
        operation_metadata={
            "replacement_position": position,
            "original_length": int(replacement_result.original_length),
            "replaced_length": int(replacement_result.replaced_length),
            "original_item": int(replacement_result.original_item),
            "was_noop": bool(replacement_result.was_noop),
            "used_internal_position": bool(replacement_result.used_internal_position),
            "used_tail_fallback": bool(replacement_result.used_tail_fallback),
            "candidate_positions": [
                int(value) for value in replacement_result.candidate_positions
            ],
            "restricted_candidate_positions": [
                int(value)
                for value in replacement_result.restricted_candidate_positions
            ],
        },
    )


__all__ = [
    "InternalRandomReplacementGeneratedContinuationPolicy",
    "InternalRandomReplacementGeneratedContinuationResult",
]
