from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from attack.generation.fake_session_generator import _to_numpy
from attack.generation.score_smoothing import min_max_smooth
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
    InternalRandomInsertionResult,
)


PURE_GENERATED_MODE_RNG_TAG = "generated_continuation_base"


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


def deterministic_session_rng(
    *,
    base_seed: int,
    target_item: int,
    fake_session_index: int,
    tag: str,
) -> random.Random:
    payload = (
        f"{int(base_seed)}|{int(target_item)}|"
        f"{int(fake_session_index)}|{str(tag)}"
    )
    seed = int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def generate_poison_model_suffix(
    *,
    runner,
    prefix: Sequence[int],
    suffix_length: int,
    topk: int,
    rng: random.Random,
    smoothing_fn: Callable[[np.ndarray], np.ndarray] = min_max_smooth,
) -> list[int]:
    if suffix_length <= 0:
        return []
    if topk <= 0:
        raise ValueError("topk must be positive.")

    session_prefix = [int(item) for item in prefix]
    generated: list[int] = []
    for _ in range(int(suffix_length)):
        scores = runner.score_session(session_prefix)
        scores_np = _to_numpy(scores)
        if scores_np.size == 0:
            raise ValueError("Score vector is empty; cannot generate suffix item.")
        smoothed = _to_numpy(smoothing_fn(scores_np))
        k = min(int(topk), int(smoothed.size))
        topk_indices = np.argsort(smoothed)[-k:]
        topk_weights = smoothed[topk_indices].astype(np.float64, copy=False)
        candidates = [int(index) for index in topk_indices.tolist()]
        if np.all(topk_weights == 0):
            next_index = candidates[int(rng.randrange(len(candidates)))]
        else:
            next_index = _weighted_choice(
                candidates,
                [float(weight) for weight in topk_weights.tolist()],
                rng,
            )
        next_item = int(next_index + 1)
        if next_item < 1 or next_item > int(scores_np.size):
            raise ValueError(
                "Generated item id is outside canonical score-vector bounds."
            )
        generated.append(next_item)
        session_prefix.append(next_item)
    return generated


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
        rng = self.generation_rng or deterministic_session_rng(
            base_seed=self.generation_rng_base_seed,
            target_item=int(target_item),
            fake_session_index=int(fake_session_index),
            tag=self.pure_generated_mode_rng_tag,
        )
        return _build_generated_result(
            insertion_result=insertion_result,
            target_item=int(target_item),
            poison_runner=self.poison_runner,
            generation_topk=self.generation_topk,
            generation_rng=rng,
        )


def _build_generated_result(
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
        **_result_payload(
            insertion_result=insertion_result,
            target_item=int(target_item),
            generated_suffix=generated_suffix,
            final=final,
        )
    )


def _result_payload(
    *,
    insertion_result: InternalRandomInsertionResult,
    target_item: int,
    generated_suffix: Sequence[int],
    final: Sequence[int],
) -> dict[str, object]:
    target = int(target_item)
    slot = int(insertion_result.insertion_slot)
    inserted = [int(item) for item in insertion_result.session]
    original_suffix = inserted[slot + 1 :]
    generated = [int(item) for item in generated_suffix]
    final_list = [int(item) for item in final]
    return {
        "session": final_list,
        "inserted_session_before_generation": inserted,
        "insertion_result": insertion_result,
        "insertion_slot": slot,
        "original_length": int(insertion_result.original_length),
        "inserted_length": int(insertion_result.inserted_length),
        "final_length": int(len(final_list)),
        "left_item": int(insertion_result.left_item),
        "original_right_item": int(insertion_result.right_item),
        "original_suffix": original_suffix,
        "generated_suffix": generated,
        "target_position": slot,
        "suffix_length": int(len(original_suffix)),
        "pre_existing_target_count": int(insertion_result.pre_existing_target_count),
        "target_occurrence_count_after_insertion": int(
            insertion_result.target_occurrence_count_after_insertion
        ),
        "target_occurrence_count_final": int(
            sum(1 for item in final_list if item == target)
        ),
        "generated_suffix_contains_target_count": int(
            sum(1 for item in generated if item == target)
        ),
        "generated_suffix_unique_item_count": int(len(set(generated))),
    }


def _weighted_choice(
    candidates: Sequence[int],
    weights: Sequence[float],
    rng: random.Random,
) -> int:
    if len(candidates) != len(weights):
        raise ValueError("candidates and weights must have the same length.")
    if not candidates:
        raise ValueError("candidates must not be empty.")
    total = float(sum(float(weight) for weight in weights))
    if total <= 0.0:
        return int(candidates[int(rng.randrange(len(candidates)))])
    draw = float(rng.random()) * total
    running = 0.0
    for candidate, weight in zip(candidates, weights):
        running += float(weight)
        if draw < running:
            return int(candidate)
    return int(candidates[-1])


__all__ = [
    "InternalRandomInsertionGeneratedContinuationPolicy",
    "InternalRandomInsertionGeneratedContinuationResult",
    "PURE_GENERATED_MODE_RNG_TAG",
    "deterministic_session_rng",
    "generate_poison_model_suffix",
]
