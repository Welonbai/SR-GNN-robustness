from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence

from attack.insertion.generated_continuation_suffix import generate_poison_model_suffix
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_FAMILIES,
    DIRECT_ACTION_GENERATE,
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_POLICY_MLP_H2,
    DIRECT_ACTION_STOP,
    DirectAction,
    DirectActionMLPPolicy,
    direct_action_consume_ratio,
    direct_action_family_probabilities,
    direct_action_generated_length,
    map_direct_action_to_family,
    enumerate_valid_direct_actions,
    sample_direct_action_categorical,
    score_direct_action,
    stable_softmax,
)
from attack.pts.executor import PTSConstructionBatchResult
from attack.pts.metadata import build_pts_batch_summary
from attack.pts.prefix_selector import select_internal_uniform_anchor


DIRECT_ACTION_FORMAL_PREFIX_TAG = "direct_action_formal_shared_prefix"
DIRECT_ACTION_FORMAL_SAMPLE_TAG = "direct_action_formal_sample"
DIRECT_ACTION_FORMAL_GENERATION_TAG = "direct_action_formal_generated_suffix"
DIRECT_ACTION_LENGTH_STD_EPSILON = 1e-12


@dataclass(frozen=True)
class DirectActionFormalSessionContext:
    fake_session_index: int
    template_session: list[int]
    anchor_position: int
    prefix: list[int]
    residual_suffix: list[int]

    @property
    def residual_suffix_len(self) -> int:
        return int(len(self.residual_suffix))


@dataclass(frozen=True)
class DirectActionContextStats:
    mean_m: float
    std_m: float
    raw_std_m: float
    max_m: int
    context_seed: int
    prefix_rng_tag: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mean_m": float(self.mean_m),
            "std_m": float(self.std_m),
            "raw_std_m": float(self.raw_std_m),
            "max_m": int(self.max_m),
            "context_seed": int(self.context_seed),
            "prefix_rng_tag": str(self.prefix_rng_tag),
            "target_independent": True,
        }


def build_direct_action_formal_session_contexts(
    *,
    template_sessions: Sequence[Sequence[int]],
    base_seed: int,
    prefix_rng_tag: str = DIRECT_ACTION_FORMAL_PREFIX_TAG,
) -> tuple[tuple[DirectActionFormalSessionContext, ...], DirectActionContextStats]:
    contexts: list[DirectActionFormalSessionContext] = []
    context_seed = _stable_seed(int(base_seed), str(prefix_rng_tag), "target_independent")
    for index, session in enumerate(template_sessions):
        template = [int(item) for item in session]
        if len(template) < 2:
            raise ValueError("Direct-action formal contexts require session length >= 2.")
        seed = _stable_seed(context_seed, int(index), "prefix_assignment")
        anchor_position = select_internal_uniform_anchor(
            len(template),
            rng=random.Random(seed),
        )
        prefix = template[:anchor_position]
        residual_suffix = template[anchor_position:]
        if not prefix or not residual_suffix:
            raise ValueError("Direct-action prefix assignment produced an empty side.")
        contexts.append(
            DirectActionFormalSessionContext(
                fake_session_index=int(index),
                template_session=template,
                anchor_position=int(anchor_position),
                prefix=prefix,
                residual_suffix=residual_suffix,
            )
        )
    if not contexts:
        raise ValueError("Direct-action formal construction requires contexts.")
    lengths = [context.residual_suffix_len for context in contexts]
    mean_m = _mean(lengths)
    raw_std_m = _std(lengths)
    std_m = 1.0 if raw_std_m <= DIRECT_ACTION_LENGTH_STD_EPSILON else raw_std_m
    return tuple(contexts), DirectActionContextStats(
        mean_m=float(mean_m),
        std_m=float(std_m),
        raw_std_m=float(raw_std_m),
        max_m=int(max(lengths)),
        context_seed=int(context_seed),
        prefix_rng_tag=str(prefix_rng_tag),
    )


def apply_pts_direct_action_construction_batch(
    *,
    session_contexts: Sequence[DirectActionFormalSessionContext],
    context_stats: DirectActionContextStats,
    target_item: int,
    policy: DirectActionMLPPolicy,
    base_seed: int,
    iteration: int,
    candidate_key: str,
    poison_runner=None,
    generation_topk: int = 100,
    sample_rng_tag: str = DIRECT_ACTION_FORMAL_SAMPLE_TAG,
    generation_rng_tag: str = DIRECT_ACTION_FORMAL_GENERATION_TAG,
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
        m = int(len(residual_suffix))
        actions = enumerate_valid_direct_actions(m)
        scores = [
            score_direct_action(
                policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
                theta=policy.to_vector(),
                action=action,
                residual_suffix_len=m,
                length_feature_mode=policy.length_feature_mode,
                max_residual_suffix_len=int(context_stats.max_m),
                mean_residual_suffix_len=float(context_stats.mean_m),
                std_residual_suffix_len=float(context_stats.std_m),
            )
            for action in actions
        ]
        probabilities = stable_softmax(scores)
        sample_seed_payload = _action_sample_seed_payload(
            base_seed=base_seed,
            target_item=target,
            iteration=iteration,
            candidate_key=candidate_key,
            fake_session_index=context.fake_session_index,
            tag=sample_rng_tag,
        )
        sampled_action = sample_direct_action_categorical(
            actions=actions,
            probabilities=probabilities,
            seed=int(sample_seed_payload["seed"]),
        )
        sampled_index = list(actions).index(sampled_action)
        family = map_direct_action_to_family(sampled_action, m)
        consume_ratio = direct_action_consume_ratio(sampled_action, m)
        consume_count = int(m if sampled_action.action_type == DIRECT_ACTION_STOP else sampled_action.consume_count)
        generated_length = direct_action_generated_length(sampled_action, m)
        prefix_through_target = prefix + [target]
        generated_suffix: list[int] = []
        generation_seed_payload: dict[str, object] | None = None
        if sampled_action.action_type == DIRECT_ACTION_GENERATE:
            if generated_length <= 0:
                raise AssertionError("generate(k) must request generated_length > 0.")
            if poison_runner is None:
                raise ValueError("poison_runner is required for generated direct actions.")
            generation_seed_payload = _generation_seed_payload(
                base_seed=base_seed,
                target_item=target,
                iteration=iteration,
                candidate_key=candidate_key,
                fake_session_index=context.fake_session_index,
                consume_count=int(sampled_action.consume_count),
                generated_length=generated_length,
                tag=generation_rng_tag,
            )
            generated_suffix = generate_poison_model_suffix(
                runner=poison_runner,
                prefix=prefix_through_target,
                suffix_length=generated_length,
                topk=int(generation_topk),
                rng=random.Random(int(generation_seed_payload["seed"])),
            )
            if len(generated_suffix) != generated_length:
                # Keep the actual length in records below; this branch exists for helpers
                # that may intentionally stop early in future implementations.
                generated_suffix = [int(item) for item in generated_suffix]
            transformed_suffix = [int(item) for item in generated_suffix]
            continuation_source = "generate"
        elif sampled_action.action_type == DIRECT_ACTION_STOP:
            transformed_suffix = []
            continuation_source = "stop"
        else:
            transformed_suffix = residual_suffix[int(sampled_action.consume_count) :]
            continuation_source = "keep"

        final_session = prefix_through_target + transformed_suffix
        final_sessions.append([int(item) for item in final_session])
        expected_family = direct_action_family_probabilities(
            actions=actions,
            probabilities=probabilities,
            residual_suffix_len=m,
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
            "residual_suffix_length": m,
            "suffix_len_group": _suffix_group(m),
            "action": sampled_action.name,
            "selected_action_type": sampled_action.action_type,
            "consume_policy": "direct_action_categorical",
            "consume_count": int(consume_count),
            "consume_ratio": float(consume_ratio),
            "derived_action_family": family,
            "continuation_source": continuation_source,
            "generation_length_policy": "remaining_suffix_after_consume",
            "generated_length": int(generated_length),
            "expected_generated_length": int(generated_length),
            "generated_suffix": [int(item) for item in generated_suffix],
            "generated_suffix_length": int(len(generated_suffix)),
            "actual_generated_length": int(len(generated_suffix)),
            "generated_suffix_length_matches_expected": bool(
                int(len(generated_suffix)) == int(generated_length)
            ),
            "generated_suffix_materialized": bool(sampled_action.action_type == DIRECT_ACTION_GENERATE),
            "final_session": [int(item) for item in final_session],
            "final_length": int(len(final_session)),
            "final_session_length": int(len(final_session)),
            "length_shift_from_template": int(len(final_session) - len(context.template_session)),
            "target_position_final": int(context.anchor_position),
            "target_tail": bool(final_session and int(final_session[-1]) == target),
            "target_occurrence_count_final": int(sum(1 for item in final_session if int(item) == target)),
            "dynamic_mask_disable_consume_one": False,
            "dynamic_mask_applied": False,
            "dynamic_mask_masked_actions": [],
            "policy_fallback_to_uniform_after_mask": False,
            "policy_original_probabilities": {
                action.name: float(prob)
                for action, prob in zip(actions, probabilities)
            },
            "policy_effective_probabilities": {
                action.name: float(prob)
                for action, prob in zip(actions, probabilities)
            },
            "policy_probability": float(probabilities[sampled_index]),
            "action_score": float(scores[sampled_index]),
            "expected_family_probabilities": expected_family,
            "sampled_family_probabilities": {
                family_name: (1.0 if family_name == family else 0.0)
                for family_name in DIRECT_ACTION_FAMILIES
            },
            "direct_action_context_stats": context_stats.to_dict(),
            "action_sample_seed": dict(sample_seed_payload),
            "generation_seed": generation_seed_payload,
        }
        records.append(record)
    summary = build_pts_batch_summary(records)
    summary["direct_action"] = build_direct_action_action_summary(records)
    return PTSConstructionBatchResult(
        final_sessions=final_sessions,
        per_session_records=records,
        summary=summary,
    )


def build_direct_action_action_summary(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    total = int(len(records))
    expected = {family: 0.0 for family in DIRECT_ACTION_FAMILIES}
    sampled = {family: 0 for family in DIRECT_ACTION_FAMILIES}
    by_group: dict[str, dict[str, object]] = {}
    for record in records:
        expected_payload = dict(record.get("expected_family_probabilities", {}))
        for family in DIRECT_ACTION_FAMILIES:
            expected[family] += float(expected_payload.get(family, 0.0))
        sampled[str(record["derived_action_family"])] += 1
        group = str(record["suffix_len_group"])
        group_payload = by_group.setdefault(
            group,
            {
                "session_count": 0,
                "expected": {family: 0.0 for family in DIRECT_ACTION_FAMILIES},
                "sampled": {family: 0 for family in DIRECT_ACTION_FAMILIES},
            },
        )
        group_payload["session_count"] = int(group_payload["session_count"]) + 1
        for family in DIRECT_ACTION_FAMILIES:
            group_payload["expected"][family] += float(expected_payload.get(family, 0.0))
        group_payload["sampled"][str(record["derived_action_family"])] += 1

    def ratios_from_counts(counts: dict[str, int], denominator: int) -> dict[str, float]:
        return {
            family: (0.0 if denominator <= 0 else float(counts[family]) / float(denominator))
            for family in DIRECT_ACTION_FAMILIES
        }

    summary = {
        "expected_family_ratios": {
            family: (0.0 if total <= 0 else float(expected[family]) / float(total))
            for family in DIRECT_ACTION_FAMILIES
        },
        "sampled_family_ratios": ratios_from_counts(sampled, total),
        "consume_ratio_mean": _mean([float(record["consume_ratio"]) for record in records]),
        "generated_action_count": int(
            sum(1 for record in records if str(record["selected_action_type"]) == DIRECT_ACTION_GENERATE)
        ),
    }
    summary["by_suffix_group"] = {
        group: {
            "session_count": int(payload["session_count"]),
            "expected_family_ratios": {
                family: (
                    0.0
                    if int(payload["session_count"]) <= 0
                    else float(payload["expected"][family]) / float(payload["session_count"])
                )
                for family in DIRECT_ACTION_FAMILIES
            },
            "sampled_family_ratios": ratios_from_counts(
                payload["sampled"],
                int(payload["session_count"]),
            ),
        }
        for group, payload in sorted(by_group.items())
    }
    return summary


def _action_sample_seed_payload(
    *,
    base_seed: int,
    target_item: int,
    iteration: int,
    candidate_key: str,
    fake_session_index: int,
    tag: str,
) -> dict[str, object]:
    return _seed_payload(
        base_seed,
        target_item,
        iteration,
        candidate_key,
        fake_session_index,
        tag,
    )


def _generation_seed_payload(
    *,
    base_seed: int,
    target_item: int,
    iteration: int,
    candidate_key: str,
    fake_session_index: int,
    consume_count: int,
    generated_length: int,
    tag: str,
) -> dict[str, object]:
    return _seed_payload(
        base_seed,
        target_item,
        iteration,
        candidate_key,
        fake_session_index,
        consume_count,
        generated_length,
        tag,
    )


def _seed_payload(*parts: object) -> dict[str, object]:
    text = "|".join(str(part) for part in parts)
    return {
        "fields": [str(part) for part in parts],
        "seed": int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16),
    }


def _stable_seed(*parts: object) -> int:
    return int(hashlib.sha1("|".join(str(part) for part in parts).encode()).hexdigest()[:16], 16)


def _suffix_group(residual_suffix_len: int) -> str:
    m = int(residual_suffix_len)
    if m == 1:
        return "suffix_1"
    if m == 2:
        return "suffix_2"
    return "suffix_3plus"


def _mean(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(sum(float(value) for value in values)) / float(len(values))


def _std(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return float((sum((float(value) - center) ** 2.0 for value in values) / float(len(values))) ** 0.5)


__all__ = [
    "DIRECT_ACTION_FORMAL_GENERATION_TAG",
    "DIRECT_ACTION_FORMAL_PREFIX_TAG",
    "DIRECT_ACTION_FORMAL_SAMPLE_TAG",
    "DirectActionContextStats",
    "DirectActionFormalSessionContext",
    "apply_pts_direct_action_construction_batch",
    "build_direct_action_action_summary",
    "build_direct_action_formal_session_contexts",
]
