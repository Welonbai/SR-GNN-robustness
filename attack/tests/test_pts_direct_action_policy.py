from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.direct_action_policy import (
    DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX,
    DIRECT_ACTION_FAMILY_STOP,
    DIRECT_ACTION_GENERATE,
    DIRECT_ACTION_KEEP,
    DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M,
    DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_POLICY_LINEAR_LENGTH,
    DIRECT_ACTION_POLICY_MLP_H2,
    DIRECT_ACTION_STOP,
    DirectAction,
    deterministic_direct_action_seed,
    direct_action_length_feature,
    direct_action_family_probabilities,
    enumerate_valid_direct_actions,
    map_direct_action_to_family,
    parameter_count_for_policy,
    sample_direct_action_categorical,
    score_direct_action,
    stable_softmax,
    uniform_family_baseline,
)


def test_direct_action_enumeration() -> None:
    actions_m1 = enumerate_valid_direct_actions(1)
    assert [action.name for action in actions_m1] == ["keep(0)", "generate(0)", "stop"]

    actions_m3 = enumerate_valid_direct_actions(3)
    assert [action.name for action in actions_m3] == [
        "keep(0)",
        "keep(1)",
        "keep(2)",
        "generate(0)",
        "generate(1)",
        "generate(2)",
        "stop",
    ]

    with pytest.raises(ValueError, match=">= 1"):
        enumerate_valid_direct_actions(0)


def test_direct_action_family_mapping() -> None:
    assert (
        map_direct_action_to_family(DirectAction(DIRECT_ACTION_KEEP, 0), 3)
        == DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX
    )
    assert (
        map_direct_action_to_family(DirectAction(DIRECT_ACTION_GENERATE, 0), 3)
        == DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX
    )
    assert (
        map_direct_action_to_family(DirectAction(DIRECT_ACTION_KEEP, 1), 3)
        == DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX
    )
    assert (
        map_direct_action_to_family(DirectAction(DIRECT_ACTION_GENERATE, 2), 3)
        == DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX
    )
    assert (
        map_direct_action_to_family(DirectAction(DIRECT_ACTION_STOP), 3)
        == DIRECT_ACTION_FAMILY_STOP
    )


def test_direct_action_parameter_counts() -> None:
    assert parameter_count_for_policy(DIRECT_ACTION_POLICY_LINEAR_LENGTH) == 8
    assert parameter_count_for_policy(DIRECT_ACTION_POLICY_MLP_H2) == 15


def test_direct_action_m_over_max_m_length_feature() -> None:
    assert direct_action_length_feature(
        2,
        mode=DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M,
        max_residual_suffix_len=5,
    ) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="max_residual_suffix_len"):
        direct_action_length_feature(2, mode=DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M)


def test_direct_action_z_score_length_feature() -> None:
    assert direct_action_length_feature(
        5,
        mode=DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
        mean_residual_suffix_len=3.0,
        std_residual_suffix_len=2.0,
    ) == pytest.approx(1.0)
    assert direct_action_length_feature(
        5,
        mode=DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
        mean_residual_suffix_len=5.0,
        std_residual_suffix_len=0.0,
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="mean_residual_suffix_len"):
        direct_action_length_feature(2, mode=DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M)


def test_direct_action_raw_m_length_feature() -> None:
    assert direct_action_length_feature(
        7,
        mode=DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
    ) == pytest.approx(7.0)


@pytest.mark.parametrize(
    "policy_variant,theta",
    [
        (DIRECT_ACTION_POLICY_LINEAR_LENGTH, [0.0] * 8),
        (DIRECT_ACTION_POLICY_MLP_H2, [0.0] * 15),
    ],
)
@pytest.mark.parametrize("residual_suffix_len", [1, 3])
def test_zero_theta_matches_uniform_action_baseline(
    policy_variant: str,
    theta: list[float],
    residual_suffix_len: int,
) -> None:
    actions = enumerate_valid_direct_actions(residual_suffix_len)
    scores = [
        score_direct_action(
            policy_variant=policy_variant,
            theta=theta,
            action=action,
            residual_suffix_len=residual_suffix_len,
        )
        for action in actions
    ]
    assert len(set(scores)) == 1

    probabilities = stable_softmax(scores)
    assert all(
        probability == pytest.approx(1.0 / len(actions))
        for probability in probabilities
    )
    family_probabilities = direct_action_family_probabilities(
        actions=actions,
        probabilities=probabilities,
        residual_suffix_len=residual_suffix_len,
    )
    assert family_probabilities == pytest.approx(
        uniform_family_baseline(residual_suffix_len)
    )


def test_direct_action_deterministic_sampling() -> None:
    actions = enumerate_valid_direct_actions(3)
    probabilities = [1.0 / len(actions)] * len(actions)
    seed = deterministic_direct_action_seed(
        base_seed=123,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_std=1.0,
        candidate_key="std1_iter0_cand0",
        session_index=4,
        tag="direct_action_sample",
    )

    assert sample_direct_action_categorical(
        actions=actions,
        probabilities=probabilities,
        seed=seed,
    ) == sample_direct_action_categorical(
        actions=actions,
        probabilities=probabilities,
        seed=seed,
    )

    other_seed = deterministic_direct_action_seed(
        base_seed=124,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_std=1.0,
        candidate_key="std1_iter0_cand0",
        session_index=4,
        tag="direct_action_sample",
    )
    assert isinstance(
        sample_direct_action_categorical(
            actions=actions,
            probabilities=probabilities,
            seed=other_seed,
        ),
        DirectAction,
    )
