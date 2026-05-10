from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.grouping import (
    assign_suffix_length_group,
    default_suffix_length_buckets,
)
from attack.pts.policy import (
    CONSUME_ONE_ACTION_NAME,
    GroupActionPolicy,
    build_valid_actions_by_group,
)


def test_default_suffix_length_grouping() -> None:
    buckets = default_suffix_length_buckets()

    assert assign_suffix_length_group(1, buckets) == "suffix_1"
    assert assign_suffix_length_group(2, buckets) == "suffix_2"
    assert assign_suffix_length_group(3, buckets) == "suffix_3plus"
    assert assign_suffix_length_group(10, buckets) == "suffix_3plus"


def test_grouping_rejects_nonpositive_lengths() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        assign_suffix_length_group(0, default_suffix_length_buckets())


def test_uniform_policy_normalizes_per_group() -> None:
    policy = GroupActionPolicy.uniform(
        group_names=["suffix_1", "suffix_2"],
        action_names=["keep_residual_suffix", "consume_all_stop"],
    )

    payload = policy.to_dict()
    group_probabilities = payload["group_probabilities"]

    assert isinstance(group_probabilities, dict)
    for probabilities in group_probabilities.values():
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert probabilities["keep_residual_suffix"] == pytest.approx(0.5)
        assert probabilities["consume_all_stop"] == pytest.approx(0.5)


def test_valid_actions_by_group_removes_duplicate_suffix_1_action() -> None:
    valid_actions = build_valid_actions_by_group(
        group_buckets=default_suffix_length_buckets(),
        enabled_actions=[
            "keep_residual_suffix",
            "regenerate_residual_suffix",
            "consume_one_keep_rest",
            "consume_all_stop",
        ],
    )

    assert valid_actions["suffix_1"] == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_all_stop",
    ]
    assert CONSUME_ONE_ACTION_NAME not in valid_actions["suffix_1"]
    assert CONSUME_ONE_ACTION_NAME in valid_actions["suffix_2"]
    assert CONSUME_ONE_ACTION_NAME in valid_actions["suffix_3plus"]


def test_uniform_policy_supports_group_specific_valid_actions() -> None:
    actions = [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_all_stop",
    ]
    valid_actions = build_valid_actions_by_group(
        group_buckets=default_suffix_length_buckets(),
        enabled_actions=actions,
    )
    policy = GroupActionPolicy.uniform(
        group_names=["suffix_1", "suffix_2", "suffix_3plus"],
        action_names=actions,
        valid_actions_by_group=valid_actions,
    )

    suffix_1 = policy.group_probabilities["suffix_1"]
    assert len(suffix_1) == 3
    assert CONSUME_ONE_ACTION_NAME not in suffix_1
    assert sum(suffix_1.values()) == pytest.approx(1.0)
    assert len(policy.group_probabilities["suffix_2"]) == 4
    assert len(policy.group_probabilities["suffix_3plus"]) == 4
    assert policy.disabled_actions_by_group["suffix_1"] == [
        CONSUME_ONE_ACTION_NAME
    ]


def test_group_specific_policy_never_samples_removed_suffix_1_action() -> None:
    actions = [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_all_stop",
    ]
    policy = GroupActionPolicy.uniform(
        group_names=["suffix_1", "suffix_2", "suffix_3plus"],
        action_names=actions,
        valid_actions_by_group=build_valid_actions_by_group(
            group_buckets=default_suffix_length_buckets(),
            enabled_actions=actions,
        ),
    )

    rng = random.Random(123)
    sampled = [
        policy.sample_action("suffix_1", 1, rng)
        for _ in range(100)
    ]

    assert CONSUME_ONE_ACTION_NAME not in sampled


def test_policy_from_old_payload_does_not_prune_suffix_1() -> None:
    policy = GroupActionPolicy.from_dict(
        {
            "group_probabilities": {
                "suffix_1": {
                    "consume_one_keep_rest": 0.6,
                    "consume_all_stop": 0.4,
                }
            },
            "disable_consume_one_when_suffix_len_leq_1": True,
        }
    )

    assert policy.valid_actions("suffix_1") == [
        "consume_one_keep_rest",
        "consume_all_stop",
    ]
    assert CONSUME_ONE_ACTION_NAME in policy.group_probabilities["suffix_1"]


def test_dynamic_mask_disables_consume_one_and_keeps_consume_all_enabled() -> None:
    policy = GroupActionPolicy(
        {
            "suffix_1": {
                "consume_one_keep_rest": 0.9,
                "consume_all_stop": 0.1,
            }
        }
    )

    result = policy.sample_action_with_metadata(
        "suffix_1",
        1,
        random.Random(1),
    )

    assert result.dynamic_mask_applied is True
    assert result.masked_actions == ["consume_one_keep_rest"]
    assert "consume_one_keep_rest" not in result.effective_probabilities
    assert "consume_all_stop" in result.effective_probabilities
    assert sum(result.effective_probabilities.values()) == pytest.approx(1.0)


def test_dynamic_mask_falls_back_to_uniform_when_all_positive_mass_removed() -> None:
    policy = GroupActionPolicy(
        {
            "suffix_1": {
                "consume_one_keep_rest": 1.0,
                "keep_residual_suffix": 0.0,
                "consume_all_stop": 0.0,
            }
        }
    )

    result = policy.sample_action_with_metadata(
        "suffix_1",
        1,
        random.Random(0),
    )

    assert result.fallback_to_uniform_after_mask is True
    assert result.effective_probabilities == {
        "keep_residual_suffix": pytest.approx(0.5),
        "consume_all_stop": pytest.approx(0.5),
    }
    assert sum(result.effective_probabilities.values()) == pytest.approx(1.0)
