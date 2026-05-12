from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from attack.pts.grouping import (
    SuffixLengthBucket,
    assign_suffix_length_group,
    default_suffix_length_buckets,
)
from attack.pts.metadata import build_pts_batch_summary
from attack.pts.policy import CONSUME_ONE_ACTION_NAMES, GroupActionPolicy
from attack.pts.prefix_selector import select_anchor_position
from attack.pts.specs import PTSConstructionSpec, lookup_spec_by_name
from attack.pts.suffix_constructor import apply_suffix_construction


@dataclass(frozen=True)
class PTSConstructionBatchResult:
    final_sessions: list[list[int]]
    per_session_records: list[dict[str, object]]
    summary: dict[str, object]


def apply_pts_construction_batch(
    *,
    template_sessions: Sequence[Sequence[int]],
    target_item: int,
    specs: Sequence[PTSConstructionSpec],
    group_policy: GroupActionPolicy,
    rng: random.Random,
    poison_runner=None,
    generation_topk: int = 100,
    generation_rng_base_seed: int = 0,
    generation_rng_tag: str = "pts_generated_suffix",
    suffix_length_buckets: Sequence[SuffixLengthBucket] | None = None,
    disable_consume_one_when_suffix_len_leq_1: bool = True,
) -> PTSConstructionBatchResult:
    spec_list = list(specs)
    if not spec_list:
        raise ValueError("specs must not be empty.")
    _validate_phase1_executor_specs(spec_list)
    spec_names = {str(spec.name) for spec in spec_list}
    unknown_policy_actions = [
        action for action in group_policy.action_names() if action not in spec_names
    ]
    if unknown_policy_actions:
        raise ValueError(
            "GroupActionPolicy contains actions without matching PTS specs: "
            f"{unknown_policy_actions}."
        )

    buckets = (
        default_suffix_length_buckets()
        if suffix_length_buckets is None
        else tuple(suffix_length_buckets)
    )
    target = int(target_item)
    prefix_selector_spec = spec_list[0].prefix_selector

    final_sessions: list[list[int]] = []
    records: list[dict[str, object]] = []
    for index, session in enumerate(template_sessions):
        template = [int(item) for item in session]
        anchor_position = select_anchor_position(
            template,
            spec=prefix_selector_spec,
            rng=rng,
        )
        prefix = template[:anchor_position]
        residual_suffix = template[anchor_position:]
        residual_suffix_len = int(len(residual_suffix))
        suffix_len_group = assign_suffix_length_group(
            residual_suffix_len,
            buckets,
        )
        sample_result = group_policy.sample_action_with_metadata(
            suffix_len_group,
            residual_suffix_len,
            rng,
            disable_consume_one_when_suffix_len_leq_1=(
                disable_consume_one_when_suffix_len_leq_1
            ),
        )
        action_name = str(sample_result.action_name)
        spec = lookup_spec_by_name(spec_list, action_name)
        construction = apply_suffix_construction(
            prefix=prefix,
            target_item=target,
            residual_suffix=residual_suffix,
            spec=spec,
            fake_session_index=index,
            poison_runner=poison_runner,
            generation_topk=int(generation_topk),
            generation_rng_base_seed=int(generation_rng_base_seed),
            generation_rng_tag=str(generation_rng_tag),
        )
        final_session = [int(item) for item in construction.final_session]
        final_sessions.append(final_session)
        target_occurrence_count_final = int(
            sum(1 for item in final_session if int(item) == target)
        )
        generated_suffix = [int(item) for item in construction.generated_suffix]
        record = {
            "fake_session_index": int(index),
            "target_item": target,
            "template_session": template,
            "template_length": int(len(template)),
            "anchor_range": str(prefix_selector_spec.range_name),
            "anchor_sampler": str(prefix_selector_spec.sampler_name),
            "anchor_position": int(anchor_position),
            "prefix": [int(item) for item in prefix],
            "prefix_length": int(len(prefix)),
            "residual_suffix": [int(item) for item in residual_suffix],
            "residual_suffix_length": residual_suffix_len,
            "suffix_len_group": str(suffix_len_group),
            "action": action_name,
            "consume_policy": str(spec.suffix_constructor.consume_policy),
            "continuation_source": str(spec.suffix_constructor.continuation_source),
            "generation_length_policy": (
                None
                if spec.suffix_constructor.generation_length_policy is None
                else str(spec.suffix_constructor.generation_length_policy)
            ),
            "generated_suffix": generated_suffix,
            "generated_suffix_length": int(len(generated_suffix)),
            "final_session": final_session,
            "final_length": int(len(final_session)),
            "length_shift_from_template": int(len(final_session) - len(template)),
            "target_position_final": int(anchor_position),
            "target_tail": bool(final_session and int(final_session[-1]) == target),
            "target_occurrence_count_final": target_occurrence_count_final,
            "dynamic_mask_disable_consume_one": bool(
                sample_result.dynamic_mask_applied
                and any(
                    action in set(sample_result.masked_actions)
                    for action in CONSUME_ONE_ACTION_NAMES
                )
            ),
            "dynamic_mask_applied": bool(sample_result.dynamic_mask_applied),
            "dynamic_mask_masked_actions": list(sample_result.masked_actions),
            "policy_fallback_to_uniform_after_mask": bool(
                sample_result.fallback_to_uniform_after_mask
            ),
            "policy_original_probabilities": dict(
                sample_result.original_probabilities
            ),
            "policy_effective_probabilities": dict(
                sample_result.effective_probabilities
            ),
            "policy_probability": float(sample_result.policy_probability),
        }
        records.append(record)

    return PTSConstructionBatchResult(
        final_sessions=final_sessions,
        per_session_records=records,
        summary=build_pts_batch_summary(records),
    )


def _validate_phase1_executor_specs(specs: Sequence[PTSConstructionSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if spec.name in seen:
            raise ValueError(f"Duplicate PTS construction spec {spec.name!r}.")
        seen.add(spec.name)
        selector = spec.prefix_selector
        if selector.range_name != "internal" or selector.sampler_name != "uniform":
            raise ValueError(
                "Phase 1 PTS executor supports only internal/uniform prefix "
                f"selection; spec {spec.name!r} uses "
                f"{selector.range_name!r}/{selector.sampler_name!r}."
            )


__all__ = [
    "PTSConstructionBatchResult",
    "apply_pts_construction_batch",
]
