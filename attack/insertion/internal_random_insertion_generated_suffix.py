from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from attack.generation.fake_session_generator import _to_numpy
from attack.generation.score_smoothing import min_max_smooth
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
    InternalRandomInsertionResult,
)


PURE_GENERATED_MODE_RNG_TAG = "generated_continuation_base"
SEEDED_REMAINDER_RNG_TAG = "successor_seeded_generated_continuation_remainder"


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


@dataclass(frozen=True)
class InternalRandomInsertionSuccessorSeededGeneratedContinuationResult(
    InternalRandomInsertionGeneratedContinuationResult
):
    successor_seed_attempted: bool
    successor_seed_applied: bool
    successor_seed_item: int | None
    successor_pool_empty: bool
    successor_pool: list[int]
    generated_suffix_after_seed: list[int]
    self_successor_seed: bool
    repair_generation_mode: Literal["successor_seeded", "pure_generated"]


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


class InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy:
    def __init__(
        self,
        topk_ratio: float,
        poison_runner,
        generation_topk: int,
        successor_counts: Counter[int] | Mapping[int, int],
        successor_pool_top_k: int = 10,
        successor_seed_ratio: float = 0.25,
        successor_smoothing_alpha: float = 0.5,
        insertion_rng: random.Random | None = None,
        generation_rng: random.Random | None = None,
        successor_seed_rng: random.Random | None = None,
        successor_item_rng: random.Random | None = None,
        generation_rng_base_seed: int = 0,
        pure_generated_mode_rng_tag: str = PURE_GENERATED_MODE_RNG_TAG,
        seeded_remainder_rng_tag: str = SEEDED_REMAINDER_RNG_TAG,
    ) -> None:
        if successor_pool_top_k < 1:
            raise ValueError("successor_pool_top_k must be >= 1.")
        if not 0.0 <= successor_seed_ratio <= 1.0:
            raise ValueError("successor_seed_ratio must be within [0, 1].")
        if successor_smoothing_alpha <= 0.0:
            raise ValueError("successor_smoothing_alpha must be positive.")
        self.insertion_policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
            topk_ratio=topk_ratio,
            rng=insertion_rng,
        )
        self.poison_runner = poison_runner
        self.generation_topk = int(generation_topk)
        self.successor_counts: Counter[int] = Counter(
            {int(item): int(count) for item, count in successor_counts.items()}
        )
        self.successor_pool_top_k = int(successor_pool_top_k)
        self.successor_seed_ratio = float(successor_seed_ratio)
        self.successor_smoothing_alpha = float(successor_smoothing_alpha)
        self.generation_rng = generation_rng
        self.successor_seed_rng = successor_seed_rng or random.Random()
        self.successor_item_rng = successor_item_rng or random.Random()
        self.generation_rng_base_seed = int(generation_rng_base_seed)
        self.pure_generated_mode_rng_tag = str(pure_generated_mode_rng_tag)
        self.seeded_remainder_rng_tag = str(seeded_remainder_rng_tag)

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
        fake_session_index: int = 0,
    ) -> InternalRandomInsertionSuccessorSeededGeneratedContinuationResult:
        insertion_result = self.insertion_policy.apply_with_metadata(
            session,
            target_item,
        )
        target = int(target_item)
        slot = int(insertion_result.insertion_slot)
        inserted = [int(item) for item in insertion_result.session]
        prefix_through_target = inserted[: slot + 1]
        original_suffix = inserted[slot + 1 :]
        suffix_length = int(len(original_suffix))
        successor_pool = successor_topk_items(
            self.successor_counts,
            self.successor_pool_top_k,
        )
        successor_pool_empty = not successor_pool
        eligible = suffix_length > 0 and not successor_pool_empty
        attempted = bool(
            eligible and self.successor_seed_rng.random() < self.successor_seed_ratio
        )

        if attempted:
            seed_item = sample_successor_from_pool(
                successor_counts=self.successor_counts,
                successor_pool=successor_pool,
                alpha=self.successor_smoothing_alpha,
                rng=self.successor_item_rng,
            )
            remainder_rng = self.generation_rng or deterministic_session_rng(
                base_seed=self.generation_rng_base_seed,
                target_item=target,
                fake_session_index=int(fake_session_index),
                tag=self.seeded_remainder_rng_tag,
            )
            generated_after_seed = generate_poison_model_suffix(
                runner=self.poison_runner,
                prefix=prefix_through_target + [int(seed_item)],
                suffix_length=suffix_length - 1,
                topk=self.generation_topk,
                rng=remainder_rng,
            )
            generated_suffix = [int(seed_item)] + generated_after_seed
            mode: Literal["successor_seeded", "pure_generated"] = "successor_seeded"
            applied = True
        else:
            generation_rng = self.generation_rng or deterministic_session_rng(
                base_seed=self.generation_rng_base_seed,
                target_item=target,
                fake_session_index=int(fake_session_index),
                tag=self.pure_generated_mode_rng_tag,
            )
            generated_suffix = generate_poison_model_suffix(
                runner=self.poison_runner,
                prefix=prefix_through_target,
                suffix_length=suffix_length,
                topk=self.generation_topk,
                rng=generation_rng,
            )
            seed_item = None
            generated_after_seed = []
            mode = "pure_generated"
            applied = False

        final = prefix_through_target + generated_suffix
        base = _result_payload(
            insertion_result=insertion_result,
            target_item=target,
            generated_suffix=generated_suffix,
            final=final,
        )
        return InternalRandomInsertionSuccessorSeededGeneratedContinuationResult(
            **base,
            successor_seed_attempted=attempted,
            successor_seed_applied=applied,
            successor_seed_item=seed_item,
            successor_pool_empty=successor_pool_empty,
            successor_pool=successor_pool,
            generated_suffix_after_seed=generated_after_seed,
            self_successor_seed=bool(seed_item == target) if seed_item is not None else False,
            repair_generation_mode=mode,
        )


def successor_topk_items(
    successor_counts: Counter[int] | Mapping[int, int],
    top_k: int,
) -> list[int]:
    ordered = sorted(
        (
            (int(item), int(count))
            for item, count in successor_counts.items()
            if int(count) > 0
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return [item for item, _ in ordered[: int(top_k)]]


def successor_smoothed_payload(
    successor_counts: Counter[int] | Mapping[int, int],
    top_k: int,
    alpha: float,
) -> dict[str, object]:
    ordered = sorted(
        (
            (int(item), int(count))
            for item, count in successor_counts.items()
            if int(count) > 0
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    top = ordered[: int(top_k)]
    weights = [float(count) ** float(alpha) for _, count in top]
    weight_total = float(sum(weights))
    probabilities = [
        float(weight) / weight_total if weight_total else 0.0 for weight in weights
    ]
    total_count = int(sum(count for _, count in ordered))
    pool_total = int(sum(count for _, count in top))
    top1_count = int(top[0][1]) if top else 0
    return {
        "successor_total_count": total_count,
        "successor_pool_total_count": pool_total,
        "successor_pool_size": int(len(top)),
        "top_successor_items": [int(item) for item, _ in top],
        "top_successor_counts": [int(count) for _, count in top],
        "top_successor_smoothed_weights": weights,
        "top_successor_smoothed_probabilities": probabilities,
        "top1_successor_share": (
            float(top1_count) / float(total_count) if total_count else 0.0
        ),
        "top10_successor_share": (
            float(pool_total) / float(total_count) if total_count else 0.0
        ),
    }


def successor_rank(
    successor_counts: Counter[int] | Mapping[int, int],
    target_item: int,
) -> int | None:
    ordered = sorted(
        (
            (int(item), int(count))
            for item, count in successor_counts.items()
            if int(count) > 0
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    target = int(target_item)
    for rank, (item, _) in enumerate(ordered, start=1):
        if item == target:
            return int(rank)
    return None


def sample_successor_from_pool(
    *,
    successor_counts: Counter[int] | Mapping[int, int],
    successor_pool: Sequence[int],
    alpha: float,
    rng: random.Random,
) -> int:
    if not successor_pool:
        raise ValueError("successor_pool must not be empty.")
    weights = [
        float(successor_counts[int(item)]) ** float(alpha)
        for item in successor_pool
    ]
    return _weighted_choice([int(item) for item in successor_pool], weights, rng)


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
    "InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy",
    "InternalRandomInsertionSuccessorSeededGeneratedContinuationResult",
    "PURE_GENERATED_MODE_RNG_TAG",
    "SEEDED_REMAINDER_RNG_TAG",
    "deterministic_session_rng",
    "generate_poison_model_suffix",
    "sample_successor_from_pool",
    "successor_rank",
    "successor_smoothed_payload",
    "successor_topk_items",
]
