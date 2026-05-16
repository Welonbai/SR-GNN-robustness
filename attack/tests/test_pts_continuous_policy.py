from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
    CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES,
    ContinuousBetaPolicy,
    build_suffix_length_percentile_lookup,
    deterministic_policy_seed,
    sample_beta,
)


def test_continuous_policy_vector_roundtrip_and_math() -> None:
    vector = [-1.0, 0.5, 1.0, -0.25, 0.2, 0.3, -0.4]
    policy = ContinuousBetaPolicy.from_vector(vector, parameter_bounds=(-5.0, 5.0))

    assert policy.to_vector() == vector
    payload = policy.to_dict()
    assert payload["type"] == "continuous_beta_policy"
    assert ContinuousBetaPolicy.from_dict(payload).to_vector() == vector

    alpha, beta = policy.beta_params(0.75)
    assert alpha > 0.0
    assert beta > 0.0
    assert 0.0 <= policy.p_generate(0.25, 0.8) <= 1.0


def test_continuous_policy_tiny_mlp_vector_roundtrip_and_math() -> None:
    vector = [
        5.0,
        -1.25,
        -5.0,
        3.75,
        0.25,
        -1.5,
        1.25,
        -0.5,
        1.75,
        -1.0,
        0.2,
        0.3,
        -0.4,
    ]
    policy = ContinuousBetaPolicy.from_vector(
        vector,
        parameter_bounds=(-5.0, 5.0),
        parameterization=CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
    )

    assert policy.to_vector() == vector
    payload = policy.to_dict()
    assert payload["parameterization"] == CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2
    assert payload["parameter_names"] == list(CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES)
    assert ContinuousBetaPolicy.from_dict(payload).to_dict() == payload

    low_alpha, low_beta = policy.beta_params(0.1)
    mid_alpha, mid_beta = policy.beta_params(0.5)
    high_alpha, high_beta = policy.beta_params(0.9)
    assert min(low_alpha, low_beta, mid_alpha, mid_beta, high_alpha, high_beta) > 0.0
    assert (low_alpha, low_beta) != (mid_alpha, mid_beta)
    assert (mid_alpha, mid_beta) != (high_alpha, high_beta)
    assert 0.0 <= policy.p_generate(0.25, 0.8) <= 1.0


def test_continuous_policy_sampling_seed_is_deterministic() -> None:
    seed = deterministic_policy_seed(
        base_seed=123,
        target_item=99,
        candidate_key="iter0_cand1",
        fake_session_index=4,
        tag="rho",
    )

    assert seed == deterministic_policy_seed(
        base_seed=123,
        target_item=99,
        candidate_key="iter0_cand1",
        fake_session_index=4,
        tag="rho",
    )
    assert sample_beta(2.0, 3.0, seed=seed) == sample_beta(2.0, 3.0, seed=seed)


def test_suffix_length_percentile_lookup_is_midpoint_empirical_cdf() -> None:
    lookup = build_suffix_length_percentile_lookup([1, 1, 3, 5])

    assert lookup[1] == 0.25
    assert lookup[3] == 0.625
    assert lookup[5] == 0.875
    assert build_suffix_length_percentile_lookup([2, 2, 2]) == {2: 0.5}
