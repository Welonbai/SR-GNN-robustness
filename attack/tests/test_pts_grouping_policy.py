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
from attack.pts.policy import GroupActionPolicy


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
