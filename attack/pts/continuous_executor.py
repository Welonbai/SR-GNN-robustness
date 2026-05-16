from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from attack.insertion.generated_continuation_suffix import (
    deterministic_session_rng,
    generate_poison_model_suffix,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_RHO_TAG,
    CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    CONTINUOUS_BETA_SOURCE_TAG,
    ContinuousBetaPolicy,
    build_suffix_length_percentile_lookup,
    deterministic_policy_seed,
    deterministic_unit_interval,
    sample_beta,
)
from attack.pts.executor import PTSConstructionBatchResult
from attack.pts.metadata import build_pts_batch_summary
from attack.pts.prefix_selector import select_internal_uniform_anchor


CONTINUOUS_ACTION_KEEP_FULL_SUFFIX = "continuous_keep_full_suffix"
CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX = "continuous_generate_full_suffix"
CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX = "continuous_partial_keep_suffix"
CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX = "continuous_partial_generate_suffix"
CONTINUOUS_ACTION_STOP = "continuous_stop"


def compute_half_up_consume_count(rho: float, residual_suffix_len: int) -> int:
    suffix_len = int(residual_suffix_len)
    if suffix_len < 0:
        raise ValueError("residual_suffix_len must be non-negative.")
    raw_count = int(math.floor(float(rho) * float(suffix_len) + 0.5))
    return min(max(raw_count, 0), suffix_len)


@dataclass(frozen=True)
class PTSContinuousSessionContext:
    fake_session_index: int
    template_session: list[int]
    anchor_position: int
    prefix: list[int]
    residual_suffix: list[int]
    suffix_length_percentile: float

    @property
    def residual_suffix_length(self) -> int:
        return int(len(self.residual_suffix))


def build_continuous_shared_session_contexts(
    *,
    template_sessions: Sequence[Sequence[int]],
    target_item: int,
    base_seed: int,
    prefix_rng_tag: str = CONTINUOUS_BETA_SHARED_PREFIX_TAG,
) -> tuple[PTSContinuousSessionContext, ...]:
    partial: list[tuple[int, list[int], int, list[int], list[int]]] = []
    residual_lengths: list[int] = []
    for index, session in enumerate(template_sessions):
        template = [int(item) for item in session]
        seed = deterministic_policy_seed(
            base_seed=int(base_seed),
            target_item=int(target_item),
            candidate_key="shared_prefix_assignment",
            fake_session_index=int(index),
            tag=str(prefix_rng_tag),
        )
        anchor_position = select_internal_uniform_anchor(
            len(template),
            rng=random.Random(seed),
        )
        prefix = template[:anchor_position]
        residual_suffix = template[anchor_position:]
        residual_lengths.append(int(len(residual_suffix)))
        partial.append((int(index), template, int(anchor_position), prefix, residual_suffix))

    percentile_lookup = build_suffix_length_percentile_lookup(residual_lengths)
    return tuple(
        PTSContinuousSessionContext(
            fake_session_index=index,
            template_session=template,
            anchor_position=anchor_position,
            prefix=prefix,
            residual_suffix=residual_suffix,
            suffix_length_percentile=float(percentile_lookup[int(len(residual_suffix))]),
        )
        for index, template, anchor_position, prefix, residual_suffix in partial
    )


def apply_pts_continuous_beta_construction_batch(
    *,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
    policy: ContinuousBetaPolicy,
    base_seed: int,
    candidate_key: str,
    poison_runner=None,
    generation_topk: int = 100,
    generation_rng_base_seed: int = 0,
    generation_rng_tag: str = "pts_generated_suffix",
    rho_sampling_tag: str = CONTINUOUS_BETA_RHO_TAG,
    source_sampling_tag: str = CONTINUOUS_BETA_SOURCE_TAG,
    materialize_generated_suffix: bool = True,
) -> PTSConstructionBatchResult:
    if not session_contexts:
        raise ValueError("session_contexts must not be empty.")
    if int(generation_topk) <= 0:
        raise ValueError("generation_topk must be positive.")

    target = int(target_item)
    final_sessions: list[list[int]] = []
    records: list[dict[str, object]] = []
    for context in session_contexts:
        prefix = [int(item) for item in context.prefix]
        residual_suffix = [int(item) for item in context.residual_suffix]
        residual_suffix_len = int(len(residual_suffix))
        q_suffix = float(context.suffix_length_percentile)
        alpha, beta = policy.beta_params(q_suffix)
        rho_seed = deterministic_policy_seed(
            base_seed=int(base_seed),
            target_item=target,
            candidate_key=str(candidate_key),
            fake_session_index=int(context.fake_session_index),
            tag=str(rho_sampling_tag),
        )
        rho = sample_beta(alpha, beta, seed=rho_seed)
        consume_count = compute_half_up_consume_count(rho, residual_suffix_len)
        prefix_through_target = prefix + [target]
        generated_suffix: list[int] = []
        p_generate: float | None = None
        if consume_count == residual_suffix_len:
            continuation_source = "stop"
            action_name = CONTINUOUS_ACTION_STOP
            constructed_suffix: list[int] = []
        else:
            p_generate = policy.p_generate(q_suffix, rho)
            source_sample = deterministic_unit_interval(
                base_seed=int(base_seed),
                target_item=target,
                candidate_key=str(candidate_key),
                fake_session_index=int(context.fake_session_index),
                tag=str(source_sampling_tag),
            )
            use_generate = bool(source_sample < p_generate)
            remaining_length = int(residual_suffix_len - consume_count)
            if use_generate:
                if bool(materialize_generated_suffix) and poison_runner is None:
                    raise ValueError(
                        "poison_runner is required for continuous generated suffix construction."
                    )
                if bool(materialize_generated_suffix):
                    generated_suffix = [
                        int(item)
                        for item in generate_poison_model_suffix(
                            runner=poison_runner,
                            prefix=prefix_through_target,
                            suffix_length=remaining_length,
                            topk=int(generation_topk),
                            rng=deterministic_session_rng(
                                base_seed=int(generation_rng_base_seed),
                                target_item=target,
                                fake_session_index=int(context.fake_session_index),
                                tag=str(generation_rng_tag),
                            ),
                        )
                    ]
                else:
                    generated_suffix = [-1 for _ in range(remaining_length)]
                if len(generated_suffix) != remaining_length:
                    raise RuntimeError(
                        "Generated continuous PTS suffix length mismatch: "
                        f"expected {remaining_length}, received {len(generated_suffix)}."
                    )
                constructed_suffix = list(generated_suffix)
                continuation_source = "generate"
                action_name = (
                    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX
                    if consume_count == 0
                    else CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX
                )
            else:
                constructed_suffix = residual_suffix[consume_count:]
                continuation_source = "keep"
                action_name = (
                    CONTINUOUS_ACTION_KEEP_FULL_SUFFIX
                    if consume_count == 0
                    else CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX
                )

        final_session = prefix_through_target + constructed_suffix
        final_sessions.append([int(item) for item in final_session])
        target_occurrence_count_final = int(
            sum(1 for item in final_session if int(item) == target)
        )
        record = {
            "fake_session_index": int(context.fake_session_index),
            "target_item": target,
            "template_session": [int(item) for item in context.template_session],
            "template_length": int(len(context.template_session)),
            "anchor_range": "internal",
            "anchor_sampler": "uniform",
            "anchor_position": int(context.anchor_position),
            "prefix": prefix,
            "prefix_length": int(len(prefix)),
            "residual_suffix": residual_suffix,
            "residual_suffix_length": residual_suffix_len,
            "suffix_len_group": "continuous",
            "suffix_length_percentile": q_suffix,
            "continuous_policy": policy.to_dict(),
            "beta_alpha": float(alpha),
            "beta_beta": float(beta),
            "consume_ratio": float(rho),
            "consume_count": int(consume_count),
            "source_generate_probability": (
                None if p_generate is None else float(p_generate)
            ),
            "action": action_name,
            "consume_policy": "continuous_beta",
            "continuation_source": continuation_source,
            "generation_length_policy": "remaining_suffix_after_consume",
            "generated_suffix": [int(item) for item in generated_suffix],
            "generated_suffix_length": int(len(generated_suffix)),
            "generated_suffix_materialized": bool(
                not generated_suffix or bool(materialize_generated_suffix)
            ),
            "final_session": [int(item) for item in final_session],
            "final_length": int(len(final_session)),
            "length_shift_from_template": int(
                len(final_session) - len(context.template_session)
            ),
            "target_position_final": int(context.anchor_position),
            "target_tail": bool(final_session and int(final_session[-1]) == target),
            "target_occurrence_count_final": target_occurrence_count_final,
            "dynamic_mask_disable_consume_one": False,
            "dynamic_mask_applied": False,
            "dynamic_mask_masked_actions": [],
            "policy_fallback_to_uniform_after_mask": False,
            "policy_original_probabilities": {},
            "policy_effective_probabilities": {},
            "policy_probability": 0.0,
        }
        records.append(record)

    return PTSConstructionBatchResult(
        final_sessions=final_sessions,
        per_session_records=records,
        summary=build_pts_batch_summary(records),
    )


__all__ = [
    "CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX",
    "CONTINUOUS_ACTION_KEEP_FULL_SUFFIX",
    "CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX",
    "CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX",
    "CONTINUOUS_ACTION_STOP",
    "PTSContinuousSessionContext",
    "apply_pts_continuous_beta_construction_batch",
    "build_continuous_shared_session_contexts",
    "compute_half_up_consume_count",
]
