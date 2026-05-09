from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence


CONSUME_ONE_ACTION_NAME = "consume_one_keep_rest"


@dataclass(frozen=True)
class PolicySampleResult:
    action_name: str
    group_name: str
    residual_suffix_len: int
    original_probabilities: dict[str, float]
    effective_probabilities: dict[str, float]
    policy_probability: float
    dynamic_mask_applied: bool
    masked_actions: list[str]
    fallback_to_uniform_after_mask: bool
    disable_consume_one_when_suffix_len_leq_1: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "action_name": str(self.action_name),
            "group_name": str(self.group_name),
            "residual_suffix_len": int(self.residual_suffix_len),
            "original_probabilities": dict(self.original_probabilities),
            "effective_probabilities": dict(self.effective_probabilities),
            "policy_probability": float(self.policy_probability),
            "dynamic_mask_applied": bool(self.dynamic_mask_applied),
            "masked_actions": list(self.masked_actions),
            "fallback_to_uniform_after_mask": bool(
                self.fallback_to_uniform_after_mask
            ),
            "disable_consume_one_when_suffix_len_leq_1": bool(
                self.disable_consume_one_when_suffix_len_leq_1
            ),
        }


class GroupActionPolicy:
    def __init__(
        self,
        group_probabilities: Mapping[str, Mapping[str, float]],
        *,
        disable_consume_one_when_suffix_len_leq_1: bool = True,
    ) -> None:
        if not group_probabilities:
            raise ValueError("group_probabilities must not be empty.")
        self.group_probabilities = {
            str(group): _normalize_probability_mapping(actions, label=str(group))
            for group, actions in group_probabilities.items()
        }
        self.disable_consume_one_when_suffix_len_leq_1 = bool(
            disable_consume_one_when_suffix_len_leq_1
        )

    @classmethod
    def uniform(
        cls,
        *,
        group_names: Sequence[str],
        action_names: Sequence[str],
        disable_consume_one_when_suffix_len_leq_1: bool = True,
    ) -> "GroupActionPolicy":
        groups = [str(group) for group in group_names]
        actions = [str(action) for action in action_names]
        if not groups:
            raise ValueError("group_names must not be empty.")
        if not actions:
            raise ValueError("action_names must not be empty.")
        probability = 1.0 / float(len(actions))
        return cls(
            {
                group: {action: probability for action in actions}
                for group in groups
            },
            disable_consume_one_when_suffix_len_leq_1=(
                disable_consume_one_when_suffix_len_leq_1
            ),
        )

    def action_names(self) -> tuple[str, ...]:
        names: list[str] = []
        seen: set[str] = set()
        for probabilities in self.group_probabilities.values():
            for action in probabilities:
                if action not in seen:
                    seen.add(action)
                    names.append(action)
        return tuple(names)

    def sample_action(
        self,
        group_name: str,
        residual_suffix_len: int,
        rng: random.Random,
        *,
        disable_consume_one_when_suffix_len_leq_1: bool | None = None,
    ) -> str:
        return self.sample_action_with_metadata(
            group_name,
            residual_suffix_len,
            rng,
            disable_consume_one_when_suffix_len_leq_1=(
                disable_consume_one_when_suffix_len_leq_1
            ),
        ).action_name

    def sample_action_with_metadata(
        self,
        group_name: str,
        residual_suffix_len: int,
        rng: random.Random,
        *,
        disable_consume_one_when_suffix_len_leq_1: bool | None = None,
    ) -> PolicySampleResult:
        group = str(group_name)
        if group not in self.group_probabilities:
            raise ValueError(f"Unknown PTS suffix length group {group!r}.")
        original = dict(self.group_probabilities[group])
        disable_mask = (
            self.disable_consume_one_when_suffix_len_leq_1
            if disable_consume_one_when_suffix_len_leq_1 is None
            else bool(disable_consume_one_when_suffix_len_leq_1)
        )
        effective, metadata = _apply_dynamic_mask(
            original,
            residual_suffix_len=int(residual_suffix_len),
            disable_consume_one_when_suffix_len_leq_1=disable_mask,
        )
        action = _sample_categorical(effective, rng=rng)
        return PolicySampleResult(
            action_name=action,
            group_name=group,
            residual_suffix_len=int(residual_suffix_len),
            original_probabilities=original,
            effective_probabilities=effective,
            policy_probability=float(effective[action]),
            dynamic_mask_applied=bool(metadata["dynamic_mask_applied"]),
            masked_actions=list(metadata["masked_actions"]),
            fallback_to_uniform_after_mask=bool(
                metadata["fallback_to_uniform_after_mask"]
            ),
            disable_consume_one_when_suffix_len_leq_1=disable_mask,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_probabilities": {
                group: dict(probabilities)
                for group, probabilities in self.group_probabilities.items()
            },
            "disable_consume_one_when_suffix_len_leq_1": bool(
                self.disable_consume_one_when_suffix_len_leq_1
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GroupActionPolicy":
        raw_probabilities = payload.get("group_probabilities")
        if not isinstance(raw_probabilities, Mapping):
            raise ValueError("GroupActionPolicy payload is missing group_probabilities.")
        probabilities: dict[str, dict[str, float]] = {}
        for group, action_payload in raw_probabilities.items():
            if not isinstance(action_payload, Mapping):
                raise ValueError(
                    f"GroupActionPolicy probabilities for group {group!r} "
                    "must be a mapping."
                )
            probabilities[str(group)] = {
                str(action): float(probability)
                for action, probability in action_payload.items()
            }
        return cls(
            probabilities,
            disable_consume_one_when_suffix_len_leq_1=bool(
                payload.get("disable_consume_one_when_suffix_len_leq_1", True)
            ),
        )


def _normalize_probability_mapping(
    probabilities: Mapping[str, float],
    *,
    label: str,
) -> dict[str, float]:
    normalized = {str(action): float(value) for action, value in probabilities.items()}
    if not normalized:
        raise ValueError(f"Probability mapping for {label} must not be empty.")
    if any(value < 0.0 for value in normalized.values()):
        raise ValueError(f"Probability mapping for {label} contains a negative value.")
    total = float(sum(normalized.values()))
    if total <= 0.0:
        probability = 1.0 / float(len(normalized))
        return {action: probability for action in normalized}
    return {action: value / total for action, value in normalized.items()}


def _apply_dynamic_mask(
    probabilities: Mapping[str, float],
    *,
    residual_suffix_len: int,
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> tuple[dict[str, float], dict[str, object]]:
    masked_actions: list[str] = []
    if (
        disable_consume_one_when_suffix_len_leq_1
        and int(residual_suffix_len) <= 1
        and CONSUME_ONE_ACTION_NAME in probabilities
    ):
        masked_actions.append(CONSUME_ONE_ACTION_NAME)

    valid_actions = {
        action: float(probability)
        for action, probability in probabilities.items()
        if action not in set(masked_actions)
    }
    if not valid_actions:
        raise ValueError(
            "Dynamic PTS policy mask removed all actions; at least one action "
            "must remain enabled."
        )

    total = float(sum(valid_actions.values()))
    fallback = bool(masked_actions and total <= 0.0)
    if fallback:
        probability = 1.0 / float(len(valid_actions))
        effective = {action: probability for action in valid_actions}
    else:
        effective = _normalize_probability_mapping(
            valid_actions,
            label="dynamic_mask_effective_probabilities",
        )
    return effective, {
        "dynamic_mask_applied": bool(masked_actions),
        "masked_actions": masked_actions,
        "fallback_to_uniform_after_mask": fallback,
    }


def _sample_categorical(
    probabilities: Mapping[str, float],
    *,
    rng: random.Random,
) -> str:
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    draw = float(rng.random())
    cumulative = 0.0
    last_action = None
    for action, probability in probabilities.items():
        last_action = action
        cumulative += float(probability)
        if draw < cumulative:
            return str(action)
    if last_action is None:
        raise ValueError("probabilities must not be empty.")
    return str(last_action)


__all__ = [
    "CONSUME_ONE_ACTION_NAME",
    "GroupActionPolicy",
    "PolicySampleResult",
]
