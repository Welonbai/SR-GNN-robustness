from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the main project env
    np = None


CONTINUOUS_BETA_POLICY_TYPE = "continuous_beta_policy"
CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA = "linear_log_beta"
CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2 = "tiny_mlp_log_beta_h2"
CONTINUOUS_BETA_PARAMETERIZATION = CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA
CONTINUOUS_BETA_SOURCE_POLICY = "q_and_rho_logistic"
CONTINUOUS_BETA_INPUT = "suffix_length_percentile"
CONTINUOUS_BETA_PARAMETER_NAMES = (
    "a0",
    "a1",
    "b0",
    "b1",
    "c0",
    "c1",
    "c2",
)
CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES = (
    "h0_w",
    "h0_b",
    "h1_w",
    "h1_b",
    "a0",
    "a_h0",
    "a_h1",
    "b0",
    "b_h0",
    "b_h1",
    "c0",
    "c1",
    "c2",
)
CONTINUOUS_BETA_ALL_PARAMETER_NAMES = tuple(
    dict.fromkeys(
        list(CONTINUOUS_BETA_PARAMETER_NAMES)
        + list(CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES)
    )
)
CONTINUOUS_BETA_DEFAULT_BOUNDS = (-5.0, 5.0)
CONTINUOUS_BETA_MIN_SHAPE = 1e-3
CONTINUOUS_BETA_RHO_TAG = "pts_continuous_beta_rho"
CONTINUOUS_BETA_SOURCE_TAG = "pts_continuous_beta_source"
CONTINUOUS_BETA_SHARED_PREFIX_TAG = "pts_continuous_shared_prefix"
CONTINUOUS_BETA_NORMALIZED_SAMPLER = "gaussian_parameter_space_v1"
CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1 = "behavior_covering_v1"
CONTINUOUS_BETA_SMOOTHING_DECISION_TAG = "pts_continuous_beta_smoothing_decision"
CONTINUOUS_BETA_SMOOTHING_UNIFORM_TAG = "pts_continuous_beta_smoothing_uniform"


@dataclass(frozen=True)
class ContinuousBetaPolicy:
    parameter_values: tuple[float, ...]
    parameterization: str = CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA
    parameter_bounds: tuple[float, float] = CONTINUOUS_BETA_DEFAULT_BOUNDS
    smoothing_epsilon: float = 0.0

    def __post_init__(self) -> None:
        bounds = _coerce_bounds(self.parameter_bounds)
        parameterization = normalize_parameterization(self.parameterization)
        smoothing_epsilon = validate_smoothing_epsilon(self.smoothing_epsilon)
        names = parameter_names_for_parameterization(parameterization)
        values = tuple(float(value) for value in self.parameter_values)
        if len(values) != len(names):
            raise ValueError(
                f"{parameterization} requires exactly {len(names)} parameters."
            )
        clipped = tuple(_clip(value, bounds) for value in values)
        object.__setattr__(self, "parameter_values", clipped)
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "parameter_bounds", bounds)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)
        for name, value in zip(names, clipped):
            object.__setattr__(self, name, float(value))

    def to_vector(self) -> list[float]:
        return [float(value) for value in self.parameter_values]

    @classmethod
    def from_vector(
        cls,
        values: Sequence[float],
        *,
        parameter_bounds: tuple[float, float] = CONTINUOUS_BETA_DEFAULT_BOUNDS,
        parameterization: str = CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA,
        smoothing_epsilon: float = 0.0,
    ) -> "ContinuousBetaPolicy":
        parameterization = normalize_parameterization(parameterization)
        names = parameter_names_for_parameterization(parameterization)
        vector = [float(value) for value in values]
        if len(vector) != len(names):
            raise ValueError(
                f"ContinuousBetaPolicy parameterization={parameterization!r} "
                f"requires exactly {len(names)} parameters."
            )
        bounds = _coerce_bounds(parameter_bounds)
        clipped = [_clip(value, bounds) for value in vector]
        return cls(
            tuple(clipped),
            parameterization=parameterization,
            parameter_bounds=bounds,
            smoothing_epsilon=float(smoothing_epsilon),
        )

    def to_dict(self) -> dict[str, object]:
        names = parameter_names_for_parameterization(self.parameterization)
        return {
            "type": CONTINUOUS_BETA_POLICY_TYPE,
            "input": CONTINUOUS_BETA_INPUT,
            "parameterization": self.parameterization,
            "source_policy": CONTINUOUS_BETA_SOURCE_POLICY,
            "parameter_names": list(names),
            "parameter_vector": self.to_vector(),
            "parameter_bounds": {
                "min": float(self.parameter_bounds[0]),
                "max": float(self.parameter_bounds[1]),
            },
            "smoothing_epsilon": float(self.smoothing_epsilon),
            "parameters": {
                name: float(getattr(self, name))
                for name in names
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContinuousBetaPolicy":
        parameterization = normalize_parameterization(
            str(payload.get("parameterization", CONTINUOUS_BETA_PARAMETERIZATION))
        )
        names = parameter_names_for_parameterization(parameterization)
        raw_bounds = payload.get("parameter_bounds", {})
        if isinstance(raw_bounds, Mapping):
            bounds = (
                float(raw_bounds.get("min", CONTINUOUS_BETA_DEFAULT_BOUNDS[0])),
                float(raw_bounds.get("max", CONTINUOUS_BETA_DEFAULT_BOUNDS[1])),
            )
        else:
            bounds = CONTINUOUS_BETA_DEFAULT_BOUNDS
        raw_vector = payload.get("parameter_vector")
        if raw_vector is not None:
            if not isinstance(raw_vector, Sequence) or isinstance(raw_vector, (str, bytes)):
                raise ValueError("ContinuousBetaPolicy parameter_vector must be a sequence.")
            return cls.from_vector(
                raw_vector,
                parameter_bounds=bounds,
                parameterization=parameterization,
                smoothing_epsilon=float(payload.get("smoothing_epsilon", 0.0)),
            )
        raw_parameters = payload.get("parameters")
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("ContinuousBetaPolicy payload is missing parameters.")
        return cls.from_vector(
            [float(raw_parameters[name]) for name in names],
            parameter_bounds=bounds,
            parameterization=parameterization,
            smoothing_epsilon=float(payload.get("smoothing_epsilon", 0.0)),
        )

    def beta_params(self, q: float) -> tuple[float, float]:
        q_value = _clamp_unit(float(q))
        if self.parameterization == CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2:
            hidden0 = math.tanh(float(self.h0_w) * q_value + float(self.h0_b))
            hidden1 = math.tanh(float(self.h1_w) * q_value + float(self.h1_b))
            log_alpha = (
                float(self.a0)
                + float(self.a_h0) * hidden0
                + float(self.a_h1) * hidden1
            )
            log_beta = (
                float(self.b0)
                + float(self.b_h0) * hidden0
                + float(self.b_h1) * hidden1
            )
        else:
            log_alpha = float(self.a0) + float(self.a1) * q_value
            log_beta = float(self.b0) + float(self.b1) * q_value
        log_alpha = _clip(log_alpha, self.parameter_bounds)
        log_beta = _clip(log_beta, self.parameter_bounds)
        alpha = max(CONTINUOUS_BETA_MIN_SHAPE, math.exp(log_alpha))
        beta = max(CONTINUOUS_BETA_MIN_SHAPE, math.exp(log_beta))
        return float(alpha), float(beta)

    def p_generate(self, q: float, rho: float) -> float:
        q_value = _clamp_unit(float(q))
        rho_value = _clamp_unit(float(rho))
        raw_p = stable_sigmoid(
            float(self.c0) + float(self.c1) * q_value + float(self.c2) * rho_value
        )
        epsilon = float(self.smoothing_epsilon)
        return float(epsilon + (1.0 - 2.0 * epsilon) * float(raw_p))


def normalize_parameterization(parameterization: str) -> str:
    value = str(parameterization).strip().lower()
    if value not in {
        CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA,
        CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
    }:
        raise ValueError(
            "continuous_beta parameterization must be 'linear_log_beta' or "
            "'tiny_mlp_log_beta_h2'."
        )
    return value


def parameter_names_for_parameterization(parameterization: str) -> tuple[str, ...]:
    value = normalize_parameterization(parameterization)
    if value == CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2:
        return CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES
    return CONTINUOUS_BETA_PARAMETER_NAMES


def stable_sigmoid(value: float) -> float:
    x = float(value)
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sample_beta(alpha: float, beta: float, *, seed: int) -> float:
    if float(alpha) <= 0.0 or float(beta) <= 0.0:
        raise ValueError("Beta parameters must be positive.")
    if np is not None:
        rng = np.random.default_rng(int(seed))
        return float(rng.beta(float(alpha), float(beta)))
    return float(random.Random(int(seed)).betavariate(float(alpha), float(beta)))


def sample_smoothed_beta_ratio(
    *,
    alpha: float,
    beta: float,
    epsilon: float,
    seed: int,
) -> float:
    epsilon = validate_smoothing_epsilon(epsilon)
    if epsilon == 0.0:
        return sample_beta(alpha, beta, seed=int(seed))
    # smoothing_epsilon is an action-agnostic exploration floor.  It mixes the
    # learned Beta consume distribution with a small Uniform component, matching
    # the spirit of bounded probabilities in grouped PTS-CEM.
    decision_seed = _salted_seed(int(seed), CONTINUOUS_BETA_SMOOTHING_DECISION_TAG)
    if random.Random(decision_seed).random() < float(epsilon):
        uniform_seed = _salted_seed(int(seed), CONTINUOUS_BETA_SMOOTHING_UNIFORM_TAG)
        return float(random.Random(uniform_seed).random())
    return sample_beta(alpha, beta, seed=int(seed))


def validate_smoothing_epsilon(epsilon: float) -> float:
    value = float(epsilon)
    if not 0.0 <= value < 0.5:
        raise ValueError("smoothing_epsilon must satisfy 0.0 <= epsilon < 0.5.")
    return value


def deterministic_unit_interval(
    *,
    base_seed: int,
    target_item: int,
    candidate_key: str,
    fake_session_index: int,
    tag: str,
) -> float:
    seed = deterministic_policy_seed(
        base_seed=base_seed,
        target_item=target_item,
        candidate_key=candidate_key,
        fake_session_index=fake_session_index,
        tag=tag,
    )
    return float(random.Random(seed).random())


def deterministic_policy_seed(
    *,
    base_seed: int,
    target_item: int,
    candidate_key: str,
    fake_session_index: int,
    tag: str,
) -> int:
    payload = (
        f"{int(base_seed)}|{int(target_item)}|{str(candidate_key)}|"
        f"{int(fake_session_index)}|{str(tag)}"
    )
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)


def _salted_seed(seed: int, tag: str) -> int:
    payload = f"{int(seed)}|{str(tag)}"
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)


def build_suffix_length_percentile_lookup(lengths: Sequence[int]) -> dict[int, float]:
    values = sorted(int(length) for length in lengths)
    if not values:
        raise ValueError("Cannot build suffix length percentiles from an empty sequence.")
    if values[0] == values[-1]:
        return {int(values[0]): 0.5}

    total = float(len(values))
    lookup: dict[int, float] = {}
    index = 0
    while index < len(values):
        length = int(values[index])
        end = index + 1
        while end < len(values) and int(values[end]) == length:
            end += 1
        # Midpoint empirical CDF for ties: less-than mass plus half the tie mass.
        lookup[length] = float((index + (end - index) / 2.0) / total)
        index = end
    return lookup


def _coerce_bounds(bounds: tuple[float, float] | Sequence[float]) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError("parameter_bounds must contain exactly two values.")
    lower = float(bounds[0])
    upper = float(bounds[1])
    if not lower < upper:
        raise ValueError("parameter_bounds must satisfy min < max.")
    return lower, upper


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(min(max(float(value), float(bounds[0])), float(bounds[1])))


def _clamp_unit(value: float) -> float:
    return float(min(max(float(value), 0.0), 1.0))


__all__ = [
    "CONTINUOUS_BETA_DEFAULT_BOUNDS",
    "CONTINUOUS_BETA_ALL_PARAMETER_NAMES",
    "CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1",
    "CONTINUOUS_BETA_INPUT",
    "CONTINUOUS_BETA_NORMALIZED_SAMPLER",
    "CONTINUOUS_BETA_PARAMETERIZATION",
    "CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA",
    "CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2",
    "CONTINUOUS_BETA_PARAMETER_NAMES",
    "CONTINUOUS_BETA_POLICY_TYPE",
    "CONTINUOUS_BETA_RHO_TAG",
    "CONTINUOUS_BETA_SHARED_PREFIX_TAG",
    "CONTINUOUS_BETA_SMOOTHING_DECISION_TAG",
    "CONTINUOUS_BETA_SMOOTHING_UNIFORM_TAG",
    "CONTINUOUS_BETA_SOURCE_POLICY",
    "CONTINUOUS_BETA_SOURCE_TAG",
    "CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES",
    "ContinuousBetaPolicy",
    "build_suffix_length_percentile_lookup",
    "deterministic_policy_seed",
    "deterministic_unit_interval",
    "normalize_parameterization",
    "parameter_names_for_parameterization",
    "sample_beta",
    "sample_smoothed_beta_ratio",
    "stable_sigmoid",
    "validate_smoothing_epsilon",
]
