from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.executor import apply_pts_construction_batch
from attack.pts.policy import GroupActionPolicy
from attack.pts.specs import get_default_pts_v1_specs


class FixedBatchRng:
    def __init__(self, anchors: list[int], random_values: list[float] | None = None) -> None:
        self.anchors = list(anchors)
        self.random_values = list(random_values or [])

    def randint(self, lower: int, upper: int) -> int:
        if not self.anchors:
            raise AssertionError("No fixed anchors remain.")
        value = int(self.anchors.pop(0))
        if value < lower or value > upper:
            raise AssertionError(f"Anchor {value} is outside [{lower}, {upper}].")
        return value

    def random(self) -> float:
        if not self.random_values:
            return 0.0
        return float(self.random_values.pop(0))


def test_pts_executor_builds_sessions_records_and_summary(monkeypatch) -> None:
    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        return [70 + index for index in range(int(suffix_length))]

    monkeypatch.setattr(
        "attack.pts.suffix_constructor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )
    policy = GroupActionPolicy(
        {
            "suffix_1": {
                "consume_one_keep_rest": 1.0,
                "consume_all_stop": 0.0,
            },
            "suffix_2": {
                "regenerate_residual_suffix": 1.0,
            },
            "suffix_3plus": {
                "keep_residual_suffix": 1.0,
            },
        }
    )
    result = apply_pts_construction_batch(
        template_sessions=[
            [1, 2, 3, 4],
            [5, 6, 7],
            [8, 9],
        ],
        target_item=99,
        specs=get_default_pts_v1_specs(),
        group_policy=policy,
        rng=FixedBatchRng(anchors=[2, 1, 1]),
        poison_runner=object(),
        generation_topk=10,
    )

    assert len(result.final_sessions) == 3
    assert all(99 in session for session in result.final_sessions)
    assert len(result.per_session_records) == 3

    required_fields = {
        "fake_session_index",
        "target_item",
        "template_session",
        "template_length",
        "anchor_range",
        "anchor_sampler",
        "anchor_position",
        "prefix",
        "prefix_length",
        "residual_suffix",
        "residual_suffix_length",
        "suffix_len_group",
        "action",
        "consume_policy",
        "continuation_source",
        "generation_length_policy",
        "generated_suffix",
        "generated_suffix_length",
        "final_session",
        "final_length",
        "length_shift_from_template",
        "target_position_final",
        "target_tail",
        "target_occurrence_count_final",
        "dynamic_mask_disable_consume_one",
        "dynamic_mask_applied",
        "dynamic_mask_masked_actions",
        "policy_fallback_to_uniform_after_mask",
        "policy_original_probabilities",
        "policy_effective_probabilities",
        "policy_probability",
    }
    for record in result.per_session_records:
        assert required_fields <= set(record)

    generated_records = [
        record
        for record in result.per_session_records
        if record["action"] == "regenerate_residual_suffix"
    ]
    assert generated_records
    assert all(record["generated_suffix"] for record in generated_records)

    summary = result.summary
    for key in [
        "action_counts",
        "action_ratios",
        "action_counts_by_group",
        "action_ratios_by_group",
        "group_counts",
        "template_length_distribution",
        "residual_suffix_length_distribution",
        "final_length_distribution",
        "length_shift_distribution",
        "target_tail_ratio",
        "generated_suffix_count",
        "generated_suffix_length_distribution",
        "generated_suffix_contains_target_ratio_among_generated",
        "generated_suffix_contains_target_ratio_overall",
        "final_sessions_with_multiple_target_count",
        "final_sessions_with_multiple_target_ratio",
        "dynamic_mask_counts",
    ]:
        assert key in summary
    assert summary["generated_suffix_count"] == 2
    assert summary["dynamic_mask_counts"]["fallback_to_uniform_after_mask"] == 1
