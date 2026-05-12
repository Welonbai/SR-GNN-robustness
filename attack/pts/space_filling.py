from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the main env
    np = None

from attack.pts.policy import GroupActionPolicy
from attack.pts.specs import (
    CONSUME_ALL_STOP_ACTION_NAME,
    CONSUME_ONE_GENERATE_ACTION_NAME,
    CONSUME_ONE_KEEP_REST_ACTION_NAME,
    KEEP_RESIDUAL_SUFFIX_ACTION_NAME,
    REGENERATE_RESIDUAL_SUFFIX_ACTION_NAME,
)


FLOAT_TOLERANCE = 1.0e-8
MANDATORY_VERTEX_NAMES = (
    "c0_preserve",
    "c0_generate",
    "c1_preserve_where_valid",
    "c1_generate_where_valid",
    "stop",
)


@dataclass(frozen=True)
class PTSSpaceFillingConfig:
    seed: int
    mandatory_enabled: bool = True
    extreme_count: int = 7
    moderate_count: int = 3
    balanced_count: int = 1
    extreme_pool_size: int = 1024
    moderate_pool_size: int = 512
    extreme_alpha: float = 0.3
    moderate_alpha: float = 2.0
    min_probability: float = 0.03
    max_probability: float = 0.90
    distance: str = "l1"

    @property
    def mandatory_count(self) -> int:
        return len(MANDATORY_VERTEX_NAMES) if bool(self.mandatory_enabled) else 0

    @property
    def initial_population_size(self) -> int:
        return int(
            self.mandatory_count
            + int(self.extreme_count)
            + int(self.moderate_count)
            + int(self.balanced_count)
        )


@dataclass(frozen=True)
class PTSSpaceFillingSample:
    policy: GroupActionPolicy
    sample_origin: str
    vertex_name: str | None = None
    pool_index: int | None = None
    distance_to_uniform: float = 0.0
    min_distance_to_previous_selected: float | None = None
    diagnostic_metadata: dict[str, object] = field(default_factory=dict)

    def sample_metadata(self, *, init_mode: str) -> dict[str, object]:
        return {
            "init_mode": str(init_mode),
            "sample_origin": str(self.sample_origin),
            "vertex_name": self.vertex_name,
            "pool_index": None if self.pool_index is None else int(self.pool_index),
            "distance_to_uniform": float(self.distance_to_uniform),
            "min_distance_to_previous_selected": (
                None
                if self.min_distance_to_previous_selected is None
                else float(self.min_distance_to_previous_selected)
            ),
            **dict(self.diagnostic_metadata),
        }


@dataclass(frozen=True)
class _PoolCandidate:
    policy: dict[str, dict[str, float]]
    sample_origin: str
    pool_index: int | None = None
    vertex_name: str | None = None


def initial_population_size_for_space_filling(
    *,
    mandatory_enabled: bool,
    extreme_count: int,
    moderate_count: int,
    balanced_count: int,
) -> int:
    mandatory_count = len(MANDATORY_VERTEX_NAMES) if bool(mandatory_enabled) else 0
    return int(
        mandatory_count
        + int(extreme_count)
        + int(moderate_count)
        + int(balanced_count)
    )


def build_vertex_stratified_initial_population(
    *,
    config: PTSSpaceFillingConfig,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    enabled_actions: Sequence[str],
    disable_consume_one_when_suffix_len_leq_1: bool = True,
) -> list[PTSSpaceFillingSample]:
    group_order = [str(group) for group in valid_actions_by_group]
    valid_actions = {
        str(group): [str(action) for action in actions]
        for group, actions in valid_actions_by_group.items()
    }
    enabled = [str(action) for action in enabled_actions]
    validate_space_filling_config(config, valid_actions)

    uniform_policy = build_uniform_policy(valid_actions)
    selected_pool_candidates: list[_PoolCandidate] = []
    if bool(config.mandatory_enabled):
        selected_pool_candidates.extend(
            build_mandatory_vertices(
                valid_actions_by_group=valid_actions,
                min_probability=float(config.min_probability),
                max_probability=float(config.max_probability),
            )
        )

    extreme_pool = generate_policy_pool(
        pool_size=int(config.extreme_pool_size),
        sample_origin="extreme_maximin",
        alpha=float(config.extreme_alpha),
        seed=int(config.seed),
        valid_actions_by_group=valid_actions,
        min_probability=float(config.min_probability),
        max_probability=float(config.max_probability),
    )
    selected_extreme = select_pool_candidates_greedy_maximin(
        pool=extreme_pool,
        count=int(config.extreme_count),
        reference_policies=[candidate.policy for candidate in selected_pool_candidates],
        uniform_policy=uniform_policy,
        group_order=group_order,
    )
    selected_pool_candidates.extend(selected_extreme)

    moderate_pool = generate_policy_pool(
        pool_size=int(config.moderate_pool_size),
        sample_origin="moderate_maximin",
        alpha=float(config.moderate_alpha),
        seed=int(config.seed) + 1_000_003,
        valid_actions_by_group=valid_actions,
        min_probability=float(config.min_probability),
        max_probability=float(config.max_probability),
    )
    selected_moderate = select_pool_candidates_greedy_maximin(
        pool=moderate_pool,
        count=int(config.moderate_count),
        reference_policies=[candidate.policy for candidate in selected_pool_candidates],
        uniform_policy=uniform_policy,
        group_order=group_order,
    )
    selected_pool_candidates.extend(selected_moderate)

    for _ in range(int(config.balanced_count)):
        selected_pool_candidates.append(
            _PoolCandidate(
                policy=uniform_policy,
                sample_origin="balanced",
            )
        )

    if len(selected_pool_candidates) != config.initial_population_size:
        raise RuntimeError("Space-filling initial population size mismatch.")
    return build_space_filling_samples(
        selected_pool_candidates=selected_pool_candidates,
        uniform_policy=uniform_policy,
        group_order=group_order,
        valid_actions_by_group=valid_actions,
        enabled_actions=enabled,
        disable_consume_one_when_suffix_len_leq_1=(
            disable_consume_one_when_suffix_len_leq_1
        ),
    )


def validate_space_filling_config(
    config: PTSSpaceFillingConfig,
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> None:
    if int(config.extreme_count) < 0:
        raise ValueError("extreme_count must be >= 0.")
    if int(config.moderate_count) < 0:
        raise ValueError("moderate_count must be >= 0.")
    if int(config.balanced_count) not in {0, 1}:
        raise ValueError("balanced_count must be 0 or 1.")
    if int(config.extreme_pool_size) < int(config.extreme_count):
        raise ValueError("extreme_pool_size must be >= extreme_count.")
    if int(config.moderate_pool_size) < int(config.moderate_count):
        raise ValueError("moderate_pool_size must be >= moderate_count.")
    if float(config.extreme_alpha) <= 0.0:
        raise ValueError("extreme_alpha must be positive.")
    if float(config.moderate_alpha) <= 0.0:
        raise ValueError("moderate_alpha must be positive.")
    if str(config.distance) != "l1":
        raise ValueError("Only distance='l1' is supported.")
    if not 0.0 <= float(config.min_probability) < float(config.max_probability) <= 1.0:
        raise ValueError("Require 0 <= min_probability < max_probability <= 1.")
    if int(config.initial_population_size) <= 0:
        raise ValueError("Initial space-filling population must be positive.")

    for group, actions in valid_actions_by_group.items():
        action_count = len(list(actions))
        if action_count <= 0:
            raise ValueError(f"Valid action set for group {group!r} must not be empty.")
        if action_count * float(config.min_probability) > 1.0 + FLOAT_TOLERANCE:
            raise ValueError(
                f"min_probability is infeasible for group {group!r} with "
                f"{action_count} valid actions."
            )
        if action_count * float(config.max_probability) < 1.0 - FLOAT_TOLERANCE:
            raise ValueError(
                f"max_probability is infeasible for group {group!r} with "
                f"{action_count} valid actions."
            )


def build_mandatory_vertices(
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    min_probability: float,
    max_probability: float,
) -> list[_PoolCandidate]:
    definitions: list[tuple[str, dict[str, str]]] = [
        (
            "c0_preserve",
            {
                group: KEEP_RESIDUAL_SUFFIX_ACTION_NAME
                for group in valid_actions_by_group
            },
        ),
        (
            "c0_generate",
            {
                group: REGENERATE_RESIDUAL_SUFFIX_ACTION_NAME
                for group in valid_actions_by_group
            },
        ),
        (
            "c1_preserve_where_valid",
            {
                group: (
                    CONSUME_ALL_STOP_ACTION_NAME
                    if str(group) == "suffix_1"
                    else CONSUME_ONE_KEEP_REST_ACTION_NAME
                )
                for group in valid_actions_by_group
            },
        ),
        (
            "c1_generate_where_valid",
            {
                group: (
                    CONSUME_ALL_STOP_ACTION_NAME
                    if str(group) == "suffix_1"
                    else CONSUME_ONE_GENERATE_ACTION_NAME
                )
                for group in valid_actions_by_group
            },
        ),
        (
            "stop",
            {
                group: CONSUME_ALL_STOP_ACTION_NAME
                for group in valid_actions_by_group
            },
        ),
    ]

    return [
        _PoolCandidate(
            policy=build_near_vertex_policy(
                valid_actions_by_group=valid_actions_by_group,
                dominant_action_by_group=dominant_action_by_group,
                min_probability=float(min_probability),
                max_probability=float(max_probability),
            ),
            sample_origin="mandatory_vertex",
            vertex_name=vertex_name,
        )
        for vertex_name, dominant_action_by_group in definitions
    ]


def build_near_vertex_policy(
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    dominant_action_by_group: Mapping[str, str],
    min_probability: float,
    max_probability: float,
) -> dict[str, dict[str, float]]:
    policy: dict[str, dict[str, float]] = {}
    for group, actions in valid_actions_by_group.items():
        dominant_action = str(dominant_action_by_group[str(group)])
        if dominant_action not in set(actions):
            raise ValueError(
                f"Cannot build mandatory PTS vertex {dominant_action!r} for "
                f"group {group!r}; valid actions are {list(actions)}."
            )
        policy[str(group)] = build_near_vertex_distribution(
            actions=actions,
            dominant_action=dominant_action,
            min_probability=float(min_probability),
            max_probability=float(max_probability),
        )
    return policy


def build_near_vertex_distribution(
    *,
    actions: Sequence[str],
    dominant_action: str,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    ordered_actions = [str(action) for action in actions]
    if not ordered_actions:
        raise ValueError("actions must not be empty.")
    if str(dominant_action) not in set(ordered_actions):
        raise ValueError(f"Dominant action {dominant_action!r} is not valid.")
    if len(ordered_actions) == 1:
        return {ordered_actions[0]: 1.0}
    high = min(
        float(max_probability),
        1.0 - float(len(ordered_actions) - 1) * float(min_probability),
    )
    low = (1.0 - high) / float(len(ordered_actions) - 1)
    return {
        action: float(high if action == str(dominant_action) else low)
        for action in ordered_actions
    }


def generate_policy_pool(
    *,
    pool_size: int,
    sample_origin: str,
    alpha: float,
    seed: int,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    min_probability: float,
    max_probability: float,
) -> list[_PoolCandidate]:
    if np is not None:
        rng = np.random.default_rng(int(seed))
    else:
        rng = random.Random(int(seed))
    pool: list[_PoolCandidate] = []
    for pool_index in range(int(pool_size)):
        policy = {
            str(group): _sample_group_distribution(
                rng=rng,
                actions=actions,
                alpha=float(alpha),
                min_probability=float(min_probability),
                max_probability=float(max_probability),
            )
            for group, actions in valid_actions_by_group.items()
        }
        pool.append(
            _PoolCandidate(
                policy=policy,
                sample_origin=str(sample_origin),
                pool_index=int(pool_index),
            )
        )
    return pool


def _sample_group_distribution(
    *,
    rng,
    actions: Sequence[str],
    alpha: float,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    action_names = [str(action) for action in actions]
    if np is not None and hasattr(rng, "dirichlet"):
        raw_values = [float(value) for value in rng.dirichlet([float(alpha)] * len(action_names))]
    else:
        values = [float(rng.gammavariate(float(alpha), 1.0)) for _ in action_names]
        total = float(sum(values))
        raw_values = (
            [1.0 / float(len(action_names)) for _ in action_names]
            if total <= 0.0
            else [float(value) / total for value in values]
        )
    return project_probability_mapping_to_bounds(
        {
            action: float(value)
            for action, value in zip(action_names, raw_values)
        },
        min_probability=float(min_probability),
        max_probability=float(max_probability),
    )


def select_pool_candidates_greedy_maximin(
    *,
    pool: Sequence[_PoolCandidate],
    count: int,
    reference_policies: Sequence[Mapping[str, Mapping[str, float]]],
    uniform_policy: Mapping[str, Mapping[str, float]],
    group_order: Sequence[str],
) -> list[_PoolCandidate]:
    if int(count) < 0:
        raise ValueError("count must be >= 0.")
    if int(count) > len(pool):
        raise ValueError("count must be <= pool length.")
    selected_indices: list[int] = []
    references = [
        _copy_policy(policy)
        for policy in reference_policies
    ]
    distance_to_uniform = [
        policy_l1_distance(candidate.policy, uniform_policy, group_order=group_order)
        for candidate in pool
    ]

    while len(selected_indices) < int(count):
        selected_set = set(selected_indices)
        best_index: int | None = None
        best_distance = -1.0
        for index, candidate in enumerate(pool):
            if index in selected_set:
                continue
            if references:
                candidate_distance = min(
                    policy_l1_distance(
                        candidate.policy,
                        reference,
                        group_order=group_order,
                    )
                    for reference in references
                )
            else:
                candidate_distance = distance_to_uniform[index]
            if (
                candidate_distance > best_distance + FLOAT_TOLERANCE
                or (
                    abs(candidate_distance - best_distance) <= FLOAT_TOLERANCE
                    and (
                        best_index is None
                        or int(candidate.pool_index or 0)
                        < int(pool[best_index].pool_index or 0)
                    )
                )
            ):
                best_distance = candidate_distance
                best_index = index
        if best_index is None:
            raise ValueError("Could not select a next maximin candidate.")
        selected_indices.append(best_index)
        references.append(pool[best_index].policy)

    return [pool[index] for index in selected_indices]


def build_space_filling_samples(
    *,
    selected_pool_candidates: Sequence[_PoolCandidate],
    uniform_policy: Mapping[str, Mapping[str, float]],
    group_order: Sequence[str],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    enabled_actions: Sequence[str],
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> list[PTSSpaceFillingSample]:
    selected: list[PTSSpaceFillingSample] = []
    previous_policies: list[Mapping[str, Mapping[str, float]]] = []
    for pool_candidate in selected_pool_candidates:
        if previous_policies:
            min_distance_to_previous = min(
                policy_l1_distance(
                    pool_candidate.policy,
                    previous_policy,
                    group_order=group_order,
                )
                for previous_policy in previous_policies
            )
        else:
            min_distance_to_previous = None
        policy = GroupActionPolicy(
            pool_candidate.policy,
            valid_actions_by_group=valid_actions_by_group,
            enabled_actions=enabled_actions,
            disable_consume_one_when_suffix_len_leq_1=(
                disable_consume_one_when_suffix_len_leq_1
            ),
        )
        selected.append(
            PTSSpaceFillingSample(
                policy=policy,
                sample_origin=str(pool_candidate.sample_origin),
                vertex_name=pool_candidate.vertex_name,
                pool_index=pool_candidate.pool_index,
                distance_to_uniform=policy_l1_distance(
                    pool_candidate.policy,
                    uniform_policy,
                    group_order=group_order,
                ),
                min_distance_to_previous_selected=min_distance_to_previous,
                diagnostic_metadata={
                    "entropy_by_group": {
                        group: entropy(pool_candidate.policy[group])
                        for group in group_order
                    },
                    "dominant_action_by_group": {
                        group: max(
                            pool_candidate.policy[group].items(),
                            key=lambda item: (item[1], item[0]),
                        )[0]
                        for group in group_order
                    },
                },
            )
        )
        previous_policies.append(pool_candidate.policy)
    return selected


def build_uniform_policy(
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    return {
        str(group): {
            str(action): 1.0 / float(len(actions))
            for action in actions
        }
        for group, actions in valid_actions_by_group.items()
    }


def project_probability_mapping_to_bounds(
    probabilities: Mapping[str, float],
    *,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    actions = [str(action) for action in probabilities]
    if not actions:
        raise ValueError("Cannot project an empty probability mapping.")
    values = [max(0.0, float(probabilities[action])) for action in actions]
    projected = project_to_bounded_simplex(
        values,
        min_probability=float(min_probability),
        max_probability=float(max_probability),
    )
    return {
        action: float(value)
        for action, value in zip(actions, projected)
    }


def project_to_bounded_simplex(
    values: Sequence[float],
    *,
    min_probability: float,
    max_probability: float,
) -> list[float]:
    raw = [max(0.0, float(value)) for value in values]
    count = len(raw)
    if count <= 0:
        raise ValueError("Cannot project an empty probability vector.")
    if count * float(min_probability) > 1.0 + FLOAT_TOLERANCE:
        raise ValueError("min_probability is infeasible for this vector.")
    if count * float(max_probability) < 1.0 - FLOAT_TOLERANCE:
        raise ValueError("max_probability is infeasible for this vector.")
    total = float(sum(raw))
    if total <= 0.0:
        raw = [1.0 / float(count) for _ in raw]
    else:
        raw = [value / total for value in raw]

    lower_tau = min(value - float(max_probability) for value in raw)
    upper_tau = max(value - float(min_probability) for value in raw)
    result = [
        min(max(value, float(min_probability)), float(max_probability))
        for value in raw
    ]
    for _ in range(100):
        tau = (lower_tau + upper_tau) / 2.0
        candidate = [
            min(max(value - tau, float(min_probability)), float(max_probability))
            for value in raw
        ]
        candidate_total = float(sum(candidate))
        result = candidate
        if abs(candidate_total - 1.0) <= FLOAT_TOLERANCE:
            break
        if candidate_total > 1.0:
            lower_tau = tau
        else:
            upper_tau = tau

    total = float(sum(result))
    if abs(total - 1.0) > FLOAT_TOLERANCE:
        result = [value / total for value in result]
    _validate_probability_values(
        result,
        min_probability=float(min_probability),
        max_probability=float(max_probability),
    )
    return [float(value) for value in result]


def _validate_probability_values(
    probabilities: Sequence[float],
    *,
    min_probability: float,
    max_probability: float,
) -> None:
    total = float(sum(probabilities))
    if abs(total - 1.0) > FLOAT_TOLERANCE:
        raise ValueError(f"Probability vector sums to {total}, not 1.")
    if any(float(value) < float(min_probability) - FLOAT_TOLERANCE for value in probabilities):
        raise ValueError("Probability vector violates the lower bound.")
    if any(float(value) > float(max_probability) + FLOAT_TOLERANCE for value in probabilities):
        raise ValueError("Probability vector violates the upper bound.")


def policy_l1_distance(
    policy_a: Mapping[str, Mapping[str, float]],
    policy_b: Mapping[str, Mapping[str, float]],
    *,
    group_order: Sequence[str],
) -> float:
    distance = 0.0
    for group in group_order:
        actions = sorted(set(policy_a[str(group)]) | set(policy_b[str(group)]))
        distance += sum(
            abs(
                float(policy_a[str(group)].get(action, 0.0))
                - float(policy_b[str(group)].get(action, 0.0))
            )
            for action in actions
        )
    return float(distance)


def entropy(probabilities: Mapping[str, float]) -> float:
    total = 0.0
    for probability in probabilities.values():
        value = float(probability)
        if value > 0.0:
            total -= value * math.log(value)
    return float(total)


def _copy_policy(
    policy: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        str(group): {
            str(action): float(probability)
            for action, probability in probabilities.items()
        }
        for group, probabilities in policy.items()
    }


__all__ = [
    "FLOAT_TOLERANCE",
    "MANDATORY_VERTEX_NAMES",
    "PTSSpaceFillingConfig",
    "PTSSpaceFillingSample",
    "build_uniform_policy",
    "build_vertex_stratified_initial_population",
    "initial_population_size_for_space_filling",
    "policy_l1_distance",
    "project_probability_mapping_to_bounds",
    "project_to_bounded_simplex",
    "validate_space_filling_config",
]
