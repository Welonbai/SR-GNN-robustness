from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from attack.insertion.generated_continuation_suffix import (
    deterministic_session_rng,
    generate_poison_model_suffix,
)
from attack.pts.specs import (
    GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE,
    GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX,
    PTSConstructionSpec,
)


@dataclass(frozen=True)
class PTSConstructionResult:
    final_session: list[int]
    prefix_through_target: list[int]
    constructed_suffix: list[int]
    generated_suffix: list[int]


def apply_suffix_construction(
    *,
    prefix: Sequence[int],
    target_item: int,
    residual_suffix: Sequence[int],
    spec: PTSConstructionSpec,
    fake_session_index: int = 0,
    poison_runner=None,
    generation_topk: int = 100,
    generation_rng_base_seed: int = 0,
    generation_rng_tag: str = "pts_generated_suffix",
) -> PTSConstructionResult:
    prefix_items = [int(item) for item in prefix]
    target = int(target_item)
    residual = [int(item) for item in residual_suffix]
    suffix_spec = spec.suffix_constructor
    _validate_phase1_suffix_spec(spec)

    prefix_through_target = prefix_items + [target]
    if suffix_spec.continuation_source == "keep":
        constructed_suffix = _kept_suffix_after_consumption(
            residual,
            consume_policy=suffix_spec.consume_policy,
        )
        return PTSConstructionResult(
            final_session=prefix_through_target + constructed_suffix,
            prefix_through_target=prefix_through_target,
            constructed_suffix=constructed_suffix,
            generated_suffix=[],
        )

    if poison_runner is None:
        raise ValueError(
            "poison_runner is required for generated PTS suffix construction."
        )
    if int(generation_topk) <= 0:
        raise ValueError("generation_topk must be positive.")
    suffix_length = _generated_suffix_length(
        residual,
        consume_policy=suffix_spec.consume_policy,
        generation_length_policy=suffix_spec.generation_length_policy,
    )
    generated_suffix = generate_poison_model_suffix(
        runner=poison_runner,
        prefix=prefix_through_target,
        suffix_length=suffix_length,
        topk=int(generation_topk),
        rng=deterministic_session_rng(
            base_seed=int(generation_rng_base_seed),
            target_item=target,
            fake_session_index=int(fake_session_index),
            tag=str(generation_rng_tag),
        ),
    )
    generated = [int(item) for item in generated_suffix]
    if len(generated) != suffix_length:
        raise RuntimeError(
            "Generated PTS suffix length does not match residual suffix length: "
            f"expected {suffix_length}, received {len(generated)}."
        )
    return PTSConstructionResult(
        final_session=prefix_through_target + generated,
        prefix_through_target=prefix_through_target,
        constructed_suffix=generated,
        generated_suffix=generated,
    )


def _kept_suffix_after_consumption(
    residual_suffix: Sequence[int],
    *,
    consume_policy: str,
) -> list[int]:
    residual = [int(item) for item in residual_suffix]
    if consume_policy == "zero":
        return residual
    if consume_policy == "one":
        if not residual:
            raise ValueError("consume_policy='one' requires a residual suffix item.")
        return residual[1:]
    if consume_policy == "all":
        return []
    raise ValueError(f"Unsupported Phase 1 PTS consume_policy {consume_policy!r}.")


def _generated_suffix_length(
    residual_suffix: Sequence[int],
    *,
    consume_policy: str,
    generation_length_policy: str | None,
) -> int:
    residual_length = int(len(residual_suffix))
    if (
        consume_policy == "zero"
        and generation_length_policy
        == GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX
    ):
        return residual_length
    if (
        consume_policy == "one"
        and generation_length_policy
        == GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE
    ):
        if residual_length < 1:
            raise ValueError(
                "consume_policy='one' generated PTS suffix construction requires "
                "a residual suffix item."
            )
        return residual_length - 1
    raise ValueError(
        "Unsupported generated PTS suffix length policy for consume_policy="
        f"{consume_policy!r}: {generation_length_policy!r}."
    )


def _validate_phase1_suffix_spec(spec: PTSConstructionSpec) -> None:
    suffix_spec = spec.suffix_constructor
    consume_policy = suffix_spec.consume_policy
    continuation_source = suffix_spec.continuation_source
    generation_length_policy = suffix_spec.generation_length_policy

    if consume_policy not in {"zero", "one", "all"}:
        raise ValueError(
            f"Unsupported Phase 1 PTS consume_policy {consume_policy!r}."
        )
    if continuation_source not in {"keep", "generate"}:
        raise ValueError(
            "Unsupported Phase 1 PTS continuation_source "
            f"{continuation_source!r}."
        )
    if continuation_source == "keep" and generation_length_policy is not None:
        raise ValueError(
            "Phase 1 PTS keep actions require generation_length_policy=None."
        )
    if continuation_source != "generate":
        return
    valid_generate_specs = {
        ("zero", GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX),
        ("one", GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE),
    }
    if (consume_policy, generation_length_policy) not in valid_generate_specs:
        raise ValueError(
            "Phase 1 PTS generated suffix actions support only "
            "consume_policy='zero' with "
            "generation_length_policy='same_as_residual_suffix' or "
            "consume_policy='one' with "
            "generation_length_policy='residual_suffix_minus_one'."
        )


__all__ = [
    "PTSConstructionResult",
    "apply_suffix_construction",
]
