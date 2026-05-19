from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping, Sequence


DIRECT_ACTION_POLICY_LINEAR_LENGTH = "direct_action_linear_length"
DIRECT_ACTION_POLICY_MLP_H2 = "direct_action_mlp_h2"
DIRECT_ACTION_LENGTH_FEATURE_LOG1P = "log1p"
DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M = "m_over_max_m"
DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M = "z_score_m"
DIRECT_ACTION_LENGTH_FEATURE_RAW_M = "raw_m"

DIRECT_ACTION_KEEP = "keep"
DIRECT_ACTION_GENERATE = "generate"
DIRECT_ACTION_STOP = "stop"

DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX = "keep_full_suffix"
DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX = "generate_full_suffix"
DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX = "partial_keep_suffix"
DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX = "partial_generate_suffix"
DIRECT_ACTION_FAMILY_STOP = "stop"
DIRECT_ACTION_FAMILIES = (
    DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX,
    DIRECT_ACTION_FAMILY_STOP,
)

DIRECT_ACTION_LINEAR_LENGTH_PARAMETER_NAMES = (
    "a_keep",
    "b_keep",
    "c_keep",
    "a_gen",
    "b_gen",
    "c_gen",
    "a_stop",
    "c_stop",
)

DIRECT_ACTION_MLP_H2_PARAMETER_NAMES = (
    "w_is_keep_h0",
    "w_is_generate_h0",
    "w_is_stop_h0",
    "w_r_h0",
    "w_l_h0",
    "w_is_keep_h1",
    "w_is_generate_h1",
    "w_is_stop_h1",
    "w_r_h1",
    "w_l_h1",
    "b_h0",
    "b_h1",
    "v_h0",
    "v_h1",
    "d",
)


@dataclass(frozen=True)
class DirectAction:
    action_type: str
    consume_count: int | None = None

    def __post_init__(self) -> None:
        action_type = str(self.action_type).strip().lower()
        if action_type not in {
            DIRECT_ACTION_KEEP,
            DIRECT_ACTION_GENERATE,
            DIRECT_ACTION_STOP,
        }:
            raise ValueError(f"Unsupported direct action type: {self.action_type}")
        if action_type == DIRECT_ACTION_STOP:
            if self.consume_count is not None:
                raise ValueError("stop action must not carry consume_count.")
        else:
            if self.consume_count is None:
                raise ValueError(f"{action_type} action requires consume_count.")
            if int(self.consume_count) < 0:
                raise ValueError("consume_count must be non-negative.")
        object.__setattr__(self, "action_type", action_type)
        if self.consume_count is not None:
            object.__setattr__(self, "consume_count", int(self.consume_count))

    @property
    def name(self) -> str:
        if self.action_type == DIRECT_ACTION_STOP:
            return DIRECT_ACTION_STOP
        return f"{self.action_type}({int(self.consume_count)})"

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "consume_count": self.consume_count,
            "name": self.name,
        }


@dataclass(frozen=True)
class DirectActionMLPPolicy:
    parameter_values: tuple[float, ...]
    length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
    context_stats: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.parameter_values)
        if len(values) != len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES):
            raise ValueError(
                "direct_action_mlp_h2 requires exactly "
                f"{len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)} parameters."
            )
        object.__setattr__(self, "parameter_values", values)
        object.__setattr__(
            self,
            "length_feature_mode",
            normalize_direct_action_length_feature_mode(self.length_feature_mode),
        )
        object.__setattr__(
            self,
            "context_stats",
            {} if self.context_stats is None else dict(self.context_stats),
        )

    @classmethod
    def from_vector(
        cls,
        values: Sequence[float],
        *,
        length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
        context_stats: Mapping[str, float] | None = None,
    ) -> "DirectActionMLPPolicy":
        return cls(
            tuple(float(value) for value in values),
            length_feature_mode=length_feature_mode,
            context_stats=context_stats,
        )

    def to_vector(self) -> list[float]:
        return [float(value) for value in self.parameter_values]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "direct_action_policy",
            "parameterization": DIRECT_ACTION_POLICY_MLP_H2,
            "length_feature": self.length_feature_mode,
            "input_features": ["is_keep", "is_generate", "is_stop", "r", "l"],
            "hidden_size": 2,
            "parameter_names": list(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES),
            "parameter_count": len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES),
            "parameter_vector": self.to_vector(),
            "parameters": {
                name: float(self.parameter_values[index])
                for index, name in enumerate(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
            },
            "context_stats": dict(self.context_stats or {}),
        }


def normalize_direct_action_policy_variant(policy_variant: str) -> str:
    value = str(policy_variant).strip().lower()
    if value not in {DIRECT_ACTION_POLICY_LINEAR_LENGTH, DIRECT_ACTION_POLICY_MLP_H2}:
        raise ValueError(
            "direct action policy must be 'direct_action_linear_length' or "
            "'direct_action_mlp_h2'."
        )
    return value


def parameter_names_for_policy(policy_variant: str) -> tuple[str, ...]:
    value = normalize_direct_action_policy_variant(policy_variant)
    if value == DIRECT_ACTION_POLICY_MLP_H2:
        return DIRECT_ACTION_MLP_H2_PARAMETER_NAMES
    return DIRECT_ACTION_LINEAR_LENGTH_PARAMETER_NAMES


def parameter_count_for_policy(policy_variant: str) -> int:
    return int(len(parameter_names_for_policy(policy_variant)))


def enumerate_valid_direct_actions(residual_suffix_len: int) -> tuple[DirectAction, ...]:
    m = _validate_residual_suffix_len(residual_suffix_len)
    actions: list[DirectAction] = []
    actions.extend(DirectAction(DIRECT_ACTION_KEEP, k) for k in range(m))
    actions.extend(DirectAction(DIRECT_ACTION_GENERATE, k) for k in range(m))
    actions.append(DirectAction(DIRECT_ACTION_STOP))
    return tuple(actions)


def map_direct_action_to_family(
    action: DirectAction,
    residual_suffix_len: int,
) -> str:
    _validate_action_for_suffix_len(action, residual_suffix_len)
    if action.action_type == DIRECT_ACTION_STOP:
        return DIRECT_ACTION_FAMILY_STOP
    if action.action_type == DIRECT_ACTION_KEEP:
        if int(action.consume_count) == 0:
            return DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX
        return DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX
    if int(action.consume_count) == 0:
        return DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX
    return DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX


def direct_action_consume_ratio(
    action: DirectAction,
    residual_suffix_len: int,
) -> float:
    m = _validate_action_for_suffix_len(action, residual_suffix_len)
    if action.action_type == DIRECT_ACTION_STOP:
        return 1.0
    return float(int(action.consume_count)) / float(m)


def direct_action_generated_length(
    action: DirectAction,
    residual_suffix_len: int,
) -> int:
    m = _validate_action_for_suffix_len(action, residual_suffix_len)
    if action.action_type != DIRECT_ACTION_GENERATE:
        return 0
    return int(m - int(action.consume_count))


def build_direct_action_features(
    action: DirectAction,
    residual_suffix_len: int,
    *,
    length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    max_residual_suffix_len: int | None = None,
    mean_residual_suffix_len: float | None = None,
    std_residual_suffix_len: float | None = None,
) -> tuple[float, float, float, float, float]:
    m = _validate_action_for_suffix_len(action, residual_suffix_len)
    is_keep = 1.0 if action.action_type == DIRECT_ACTION_KEEP else 0.0
    is_generate = 1.0 if action.action_type == DIRECT_ACTION_GENERATE else 0.0
    is_stop = 1.0 if action.action_type == DIRECT_ACTION_STOP else 0.0
    r = direct_action_consume_ratio(action, m)
    length_feature = direct_action_length_feature(
        m,
        mode=length_feature_mode,
        max_residual_suffix_len=max_residual_suffix_len,
        mean_residual_suffix_len=mean_residual_suffix_len,
        std_residual_suffix_len=std_residual_suffix_len,
    )
    return is_keep, is_generate, is_stop, float(r), float(length_feature)


def direct_action_length_feature(
    residual_suffix_len: int,
    *,
    mode: str = DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    max_residual_suffix_len: int | None = None,
    mean_residual_suffix_len: float | None = None,
    std_residual_suffix_len: float | None = None,
) -> float:
    m = _validate_residual_suffix_len(residual_suffix_len)
    value = normalize_direct_action_length_feature_mode(mode)
    if value == DIRECT_ACTION_LENGTH_FEATURE_LOG1P:
        return float(math.log1p(float(m)))
    if value == DIRECT_ACTION_LENGTH_FEATURE_RAW_M:
        return float(m)
    if max_residual_suffix_len is None:
        if value == DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M:
            raise ValueError("max_residual_suffix_len is required for m_over_max_m.")
    if value == DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M:
        max_m = _validate_residual_suffix_len(max_residual_suffix_len)
        if m > max_m:
            raise ValueError("residual_suffix_len must be <= max_residual_suffix_len.")
        return float(m) / float(max_m)
    if mean_residual_suffix_len is None or std_residual_suffix_len is None:
        raise ValueError(
            "mean_residual_suffix_len and std_residual_suffix_len are required "
            "for z_score_m."
        )
    std_m = float(std_residual_suffix_len)
    if std_m <= 0.0:
        return 0.0
    return (float(m) - float(mean_residual_suffix_len)) / std_m


def normalize_direct_action_length_feature_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value == "z_score":
        value = DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
    if value not in {
        DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
        DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M,
        DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
        DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
    }:
        raise ValueError(
            "direct action length feature must be 'log1p', 'm_over_max_m', "
            "'z_score_m'/'z_score', or 'raw_m'."
        )
    return value


def score_direct_action(
    *,
    policy_variant: str,
    theta: Sequence[float],
    action: DirectAction,
    residual_suffix_len: int,
    length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    max_residual_suffix_len: int | None = None,
    mean_residual_suffix_len: float | None = None,
    std_residual_suffix_len: float | None = None,
) -> float:
    value = normalize_direct_action_policy_variant(policy_variant)
    vector = _validate_theta(value, theta)
    m = _validate_action_for_suffix_len(action, residual_suffix_len)
    r = direct_action_consume_ratio(action, m)
    length_feature = direct_action_length_feature(
        m,
        mode=length_feature_mode,
        max_residual_suffix_len=max_residual_suffix_len,
        mean_residual_suffix_len=mean_residual_suffix_len,
        std_residual_suffix_len=std_residual_suffix_len,
    )
    if value == DIRECT_ACTION_POLICY_LINEAR_LENGTH:
        if action.action_type == DIRECT_ACTION_KEEP:
            return float(vector[0] + vector[1] * r + vector[2] * length_feature)
        if action.action_type == DIRECT_ACTION_GENERATE:
            return float(vector[3] + vector[4] * r + vector[5] * length_feature)
        return float(vector[6] + vector[7] * length_feature)

    features = build_direct_action_features(
        action,
        m,
        length_feature_mode=length_feature_mode,
        max_residual_suffix_len=max_residual_suffix_len,
        mean_residual_suffix_len=mean_residual_suffix_len,
        std_residual_suffix_len=std_residual_suffix_len,
    )
    hidden0 = math.tanh(
        sum(float(features[index]) * float(vector[index]) for index in range(5))
        + float(vector[10])
    )
    hidden1 = math.tanh(
        sum(float(features[index]) * float(vector[index + 5]) for index in range(5))
        + float(vector[11])
    )
    return float(float(vector[12]) * hidden0 + float(vector[13]) * hidden1 + vector[14])


def stable_softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("scores must not be empty.")
    values = [float(score) for score in scores]
    max_score = max(values)
    exp_values = [math.exp(score - max_score) for score in values]
    total = float(sum(exp_values))
    if total <= 0.0:
        raise ValueError("softmax denominator must be positive.")
    return [float(value / total) for value in exp_values]


def direct_action_family_probabilities(
    *,
    actions: Sequence[DirectAction],
    probabilities: Sequence[float],
    residual_suffix_len: int,
) -> dict[str, float]:
    if len(actions) != len(probabilities):
        raise ValueError("actions and probabilities must have the same length.")
    values = {family: 0.0 for family in DIRECT_ACTION_FAMILIES}
    for action, probability in zip(actions, probabilities):
        family = map_direct_action_to_family(action, residual_suffix_len)
        values[family] += float(probability)
    return values


def direct_action_entropy(probabilities: Sequence[float]) -> float:
    positives = [float(value) for value in probabilities if float(value) > 0.0]
    if not positives:
        return 0.0
    return float(-sum(value * math.log(value) for value in positives))


def sample_theta(
    *,
    policy_variant: str,
    initial_std: float,
    seed: int,
) -> list[float]:
    std = float(initial_std)
    if std < 0.0:
        raise ValueError("initial_std must be non-negative.")
    rng = random.Random(int(seed))
    return [
        float(rng.gauss(0.0, std))
        for _ in range(parameter_count_for_policy(policy_variant))
    ]


def deterministic_direct_action_seed(
    *,
    base_seed: int,
    policy_variant: str,
    initial_std: float,
    candidate_key: str,
    session_index: int,
    tag: str,
) -> int:
    payload = (
        f"{int(base_seed)}|{normalize_direct_action_policy_variant(policy_variant)}|"
        f"{float(initial_std):.17g}|{str(candidate_key)}|{int(session_index)}|"
        f"{str(tag)}"
    )
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)


def sample_direct_action_categorical(
    *,
    actions: Sequence[DirectAction],
    probabilities: Sequence[float],
    seed: int,
) -> DirectAction:
    if len(actions) != len(probabilities):
        raise ValueError("actions and probabilities must have the same length.")
    if not actions:
        raise ValueError("actions must not be empty.")
    threshold = random.Random(int(seed)).random()
    cumulative = 0.0
    for action, probability in zip(actions, probabilities):
        cumulative += float(probability)
        if threshold <= cumulative:
            return action
    return actions[-1]


def uniform_family_baseline(residual_suffix_len: int) -> dict[str, float]:
    m = _validate_residual_suffix_len(residual_suffix_len)
    denominator = float(2 * m + 1)
    return {
        DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX: 1.0 / denominator,
        DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX: 1.0 / denominator,
        DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX: float(m - 1) / denominator,
        DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX: float(m - 1) / denominator,
        DIRECT_ACTION_FAMILY_STOP: 1.0 / denominator,
    }


def direct_action_policy_payload(
    *,
    policy_variant: str,
    theta: Sequence[float],
) -> dict[str, object]:
    value = normalize_direct_action_policy_variant(policy_variant)
    vector = _validate_theta(value, theta)
    names = parameter_names_for_policy(value)
    return {
        "policy_variant": value,
        "parameter_names": list(names),
        "parameter_count": int(len(names)),
        "parameter_vector": [float(item) for item in vector],
        "parameters": {
            name: float(vector[index])
            for index, name in enumerate(names)
        },
    }


def action_probability_payload(
    *,
    actions: Sequence[DirectAction],
    scores: Sequence[float],
    probabilities: Sequence[float],
    residual_suffix_len: int,
) -> list[dict[str, object]]:
    if len(actions) != len(scores) or len(actions) != len(probabilities):
        raise ValueError("actions, scores, and probabilities must have matching lengths.")
    return [
        {
            **action.to_dict(),
            "family": map_direct_action_to_family(action, residual_suffix_len),
            "score": float(score),
            "probability": float(probability),
            "consume_ratio": direct_action_consume_ratio(action, residual_suffix_len),
            "generated_length": direct_action_generated_length(
                action,
                residual_suffix_len,
            ),
        }
        for action, score, probability in zip(actions, scores, probabilities)
    ]


def _validate_theta(policy_variant: str, theta: Sequence[float]) -> tuple[float, ...]:
    names = parameter_names_for_policy(policy_variant)
    values = tuple(float(value) for value in theta)
    if len(values) != len(names):
        raise ValueError(
            f"{policy_variant} requires exactly {len(names)} parameters."
        )
    return values


def _validate_residual_suffix_len(residual_suffix_len: int) -> int:
    m = int(residual_suffix_len)
    if m < 1:
        raise ValueError("residual_suffix_len must be >= 1.")
    return m


def _validate_action_for_suffix_len(
    action: DirectAction,
    residual_suffix_len: int,
) -> int:
    m = _validate_residual_suffix_len(residual_suffix_len)
    if action.action_type != DIRECT_ACTION_STOP and int(action.consume_count) >= m:
        raise ValueError("consume_count must be in [0, residual_suffix_len - 1].")
    return m


__all__ = [
    "DIRECT_ACTION_FAMILIES",
    "DIRECT_ACTION_LENGTH_FEATURE_LOG1P",
    "DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M",
    "DIRECT_ACTION_LENGTH_FEATURE_RAW_M",
    "DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M",
    "DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX",
    "DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX",
    "DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX",
    "DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX",
    "DIRECT_ACTION_FAMILY_STOP",
    "DIRECT_ACTION_GENERATE",
    "DIRECT_ACTION_KEEP",
    "DIRECT_ACTION_LINEAR_LENGTH_PARAMETER_NAMES",
    "DIRECT_ACTION_MLP_H2_PARAMETER_NAMES",
    "DIRECT_ACTION_POLICY_LINEAR_LENGTH",
    "DIRECT_ACTION_POLICY_MLP_H2",
    "DIRECT_ACTION_STOP",
    "DirectAction",
    "DirectActionMLPPolicy",
    "action_probability_payload",
    "build_direct_action_features",
    "deterministic_direct_action_seed",
    "direct_action_consume_ratio",
    "direct_action_entropy",
    "direct_action_family_probabilities",
    "direct_action_generated_length",
    "direct_action_length_feature",
    "direct_action_policy_payload",
    "enumerate_valid_direct_actions",
    "map_direct_action_to_family",
    "normalize_direct_action_policy_variant",
    "normalize_direct_action_length_feature_mode",
    "parameter_count_for_policy",
    "parameter_names_for_policy",
    "sample_direct_action_categorical",
    "sample_theta",
    "score_direct_action",
    "stable_softmax",
    "uniform_family_baseline",
]
