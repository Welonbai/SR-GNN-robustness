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
CONTINUOUS_BETA_PARAMETERIZATION = "linear_log_beta"
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
CONTINUOUS_BETA_DEFAULT_BOUNDS = (-5.0, 5.0)
CONTINUOUS_BETA_MIN_SHAPE = 1e-3
CONTINUOUS_BETA_RHO_TAG = "pts_continuous_beta_rho"
CONTINUOUS_BETA_SOURCE_TAG = "pts_continuous_beta_source"
CONTINUOUS_BETA_SHARED_PREFIX_TAG = "pts_continuous_shared_prefix"
CONTINUOUS_BETA_NORMALIZED_SAMPLER = "gaussian_parameter_space_v1"
CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1 = "behavior_covering_v1"


@dataclass(frozen=True)
class ContinuousBetaPolicy:
    a0: float
    a1: float
    b0: float
    b1: float
    c0: float
    c1: float
    c2: float
    parameter_bounds: tuple[float, float] = CONTINUOUS_BETA_DEFAULT_BOUNDS

    def __post_init__(self) -> None:
        bounds = _coerce_bounds(self.parameter_bounds)
        object.__setattr__(self, "parameter_bounds", bounds)
        for name in CONTINUOUS_BETA_PARAMETER_NAMES:
            object.__setattr__(self, name, _clip(float(getattr(self, name)), bounds))

    def to_vector(self) -> list[float]:
        return [float(getattr(self, name)) for name in CONTINUOUS_BETA_PARAMETER_NAMES]

    @classmethod
    def from_vector(
        cls,
        values: Sequence[float],
        *,
        parameter_bounds: tuple[float, float] = CONTINUOUS_BETA_DEFAULT_BOUNDS,
    ) -> "ContinuousBetaPolicy":
        vector = [float(value) for value in values]
        if len(vector) != len(CONTINUOUS_BETA_PARAMETER_NAMES):
            raise ValueError(
                "ContinuousBetaPolicy requires exactly "
                f"{len(CONTINUOUS_BETA_PARAMETER_NAMES)} parameters."
            )
        bounds = _coerce_bounds(parameter_bounds)
        clipped = [_clip(value, bounds) for value in vector]
        return cls(*clipped, parameter_bounds=bounds)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": CONTINUOUS_BETA_POLICY_TYPE,
            "input": CONTINUOUS_BETA_INPUT,
            "parameterization": CONTINUOUS_BETA_PARAMETERIZATION,
            "source_policy": CONTINUOUS_BETA_SOURCE_POLICY,
            "parameter_names": list(CONTINUOUS_BETA_PARAMETER_NAMES),
            "parameter_vector": self.to_vector(),
            "parameter_bounds": {
                "min": float(self.parameter_bounds[0]),
                "max": float(self.parameter_bounds[1]),
            },
            "parameters": {
                name: float(getattr(self, name))
                for name in CONTINUOUS_BETA_PARAMETER_NAMES
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ContinuousBetaPolicy":
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
            return cls.from_vector(raw_vector, parameter_bounds=bounds)
        raw_parameters = payload.get("parameters")
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("ContinuousBetaPolicy payload is missing parameters.")
        return cls.from_vector(
            [float(raw_parameters[name]) for name in CONTINUOUS_BETA_PARAMETER_NAMES],
            parameter_bounds=bounds,
        )

    def beta_params(self, q: float) -> tuple[float, float]:
        q_value = _clamp_unit(float(q))
        log_alpha = _clip(float(self.a0) + float(self.a1) * q_value, self.parameter_bounds)
        log_beta = _clip(float(self.b0) + float(self.b1) * q_value, self.parameter_bounds)
        alpha = max(CONTINUOUS_BETA_MIN_SHAPE, math.exp(log_alpha))
        beta = max(CONTINUOUS_BETA_MIN_SHAPE, math.exp(log_beta))
        return float(alpha), float(beta)

    def p_generate(self, q: float, rho: float) -> float:
        q_value = _clamp_unit(float(q))
        rho_value = _clamp_unit(float(rho))
        return float(
            stable_sigmoid(
                float(self.c0) + float(self.c1) * q_value + float(self.c2) * rho_value
            )
        )


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
    "CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1",
    "CONTINUOUS_BETA_INPUT",
    "CONTINUOUS_BETA_NORMALIZED_SAMPLER",
    "CONTINUOUS_BETA_PARAMETERIZATION",
    "CONTINUOUS_BETA_PARAMETER_NAMES",
    "CONTINUOUS_BETA_POLICY_TYPE",
    "CONTINUOUS_BETA_RHO_TAG",
    "CONTINUOUS_BETA_SHARED_PREFIX_TAG",
    "CONTINUOUS_BETA_SOURCE_POLICY",
    "CONTINUOUS_BETA_SOURCE_TAG",
    "ContinuousBetaPolicy",
    "build_suffix_length_percentile_lookup",
    "deterministic_policy_seed",
    "deterministic_unit_interval",
    "sample_beta",
    "stable_sigmoid",
]
