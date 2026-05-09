from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from attack.generation.fake_session_generator import _to_numpy
from attack.generation.score_smoothing import min_max_smooth


PURE_GENERATED_MODE_RNG_TAG = "generated_continuation_base"


@dataclass(frozen=True)
class TargetExposureForSuffix:
    original_session: list[int]
    session_before_suffix: list[int]
    target_item: int
    target_position: int
    operation: str
    original_suffix: list[int]
    left_item: int | None
    right_item: int | None
    action_position: int | None
    operation_metadata: dict[str, object]


@dataclass(frozen=True)
class GeneratedContinuationAppliedResult:
    session: list[int]
    exposure: TargetExposureForSuffix
    prefix_through_target: list[int]
    original_suffix: list[int]
    generated_suffix: list[int]
    final_target_position: int
    original_length: int
    before_suffix_length: int
    final_length: int
    suffix_length: int
    generated_suffix_contains_target_count: int
    generated_suffix_unique_item_count: int
    target_occurrence_count_before_suffix: int
    target_occurrence_count_final: int


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


def apply_generated_continuation_to_exposure(
    exposure: TargetExposureForSuffix,
    *,
    poison_runner,
    generation_topk: int,
    generation_rng_base_seed: int,
    target_item: int,
    fake_session_index: int,
    rng_tag: str = PURE_GENERATED_MODE_RNG_TAG,
) -> GeneratedContinuationAppliedResult:
    target = int(target_item)
    before_suffix = [int(item) for item in exposure.session_before_suffix]
    target_position = int(exposure.target_position)
    if target_position < 0 or target_position >= len(before_suffix):
        raise ValueError("target_position is outside session_before_suffix.")
    if before_suffix[target_position] != target:
        raise ValueError("target_position does not point to target_item.")

    prefix_through_target = before_suffix[: target_position + 1]
    original_suffix = [int(item) for item in exposure.original_suffix]
    expected_suffix = before_suffix[target_position + 1 :]
    if original_suffix != expected_suffix:
        raise ValueError("exposure.original_suffix does not match session suffix.")

    generated_suffix = generate_poison_model_suffix(
        runner=poison_runner,
        prefix=prefix_through_target,
        suffix_length=len(original_suffix),
        topk=int(generation_topk),
        rng=deterministic_session_rng(
            base_seed=int(generation_rng_base_seed),
            target_item=target,
            fake_session_index=int(fake_session_index),
            tag=str(rng_tag),
        ),
    )
    final_session = prefix_through_target + generated_suffix
    if len(final_session) != len(before_suffix):
        raise RuntimeError("Generated continuation changed exposure length.")
    if final_session[target_position] != target:
        raise RuntimeError("Generated continuation moved the target item.")

    return GeneratedContinuationAppliedResult(
        session=final_session,
        exposure=exposure,
        prefix_through_target=prefix_through_target,
        original_suffix=original_suffix,
        generated_suffix=generated_suffix,
        final_target_position=target_position,
        original_length=int(len(exposure.original_session)),
        before_suffix_length=int(len(before_suffix)),
        final_length=int(len(final_session)),
        suffix_length=int(len(original_suffix)),
        generated_suffix_contains_target_count=int(
            sum(1 for item in generated_suffix if int(item) == target)
        ),
        generated_suffix_unique_item_count=int(len(set(generated_suffix))),
        target_occurrence_count_before_suffix=int(
            sum(1 for item in before_suffix if int(item) == target)
        ),
        target_occurrence_count_final=int(
            sum(1 for item in final_session if int(item) == target)
        ),
    )


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
    "GeneratedContinuationAppliedResult",
    "PURE_GENERATED_MODE_RNG_TAG",
    "TargetExposureForSuffix",
    "apply_generated_continuation_to_exposure",
    "deterministic_session_rng",
    "generate_poison_model_suffix",
]
