from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


CONSUME_ONE_ACTION_NAME = "consume_one_keep_rest"
CONSUME_ONE_GENERATE_ACTION_NAME = "consume_one_generate_continuation"
CONSUME_ONE_ACTION_NAMES = (
    CONSUME_ONE_ACTION_NAME,
    CONSUME_ONE_GENERATE_ACTION_NAME,
)


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
        valid_actions_by_group: Mapping[str, Sequence[str]] | None = None,
        enabled_actions: Sequence[str] | None = None,
        disable_consume_one_when_suffix_len_leq_1: bool = True,
    ) -> None:
        if not group_probabilities:
            raise ValueError("group_probabilities must not be empty.")
        normalized_probabilities = {
            str(group): _normalize_probability_mapping(actions, label=str(group))
            for group, actions in group_probabilities.items()
        }
        self.valid_actions_by_group = _resolve_valid_actions_by_group(
            normalized_probabilities,
            valid_actions_by_group=valid_actions_by_group,
        )
        self.enabled_actions = _resolve_enabled_actions(
            self.valid_actions_by_group,
            enabled_actions=enabled_actions,
        )
        self.disabled_actions_by_group = _derive_disabled_actions_by_group(
            self.enabled_actions,
            self.valid_actions_by_group,
        )
        self.group_probabilities = _validate_probability_actions(
            normalized_probabilities,
            self.valid_actions_by_group,
        )
        self.disable_consume_one_when_suffix_len_leq_1 = bool(
            disable_consume_one_when_suffix_len_leq_1
        )

    @classmethod
    def uniform(
        cls,
        *,
        group_names: Sequence[str],
        action_names: Sequence[str],
        valid_actions_by_group: Mapping[str, Sequence[str]] | None = None,
        disable_consume_one_when_suffix_len_leq_1: bool = True,
    ) -> "GroupActionPolicy":
        groups = [str(group) for group in group_names]
        actions = _normalize_ordered_names(action_names, label="action_names")
        if not groups:
            raise ValueError("group_names must not be empty.")
        if not actions:
            raise ValueError("action_names must not be empty.")
        if valid_actions_by_group is None:
            valid_actions = {group: list(actions) for group in groups}
        else:
            valid_actions = _normalize_valid_actions_by_group(
                valid_actions_by_group,
                expected_groups=groups,
            )
            unknown_actions = {
                action
                for group_actions in valid_actions.values()
                for action in group_actions
                if action not in set(actions)
            }
            if unknown_actions:
                raise ValueError(
                    "valid_actions_by_group contains actions not present in "
                    f"action_names: {sorted(unknown_actions)}."
                )
        return cls(
            {
                group: {
                    action: 1.0 / float(len(valid_actions[group]))
                    for action in valid_actions[group]
                }
                for group in groups
            },
            valid_actions_by_group=valid_actions,
            enabled_actions=actions,
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

    def valid_actions(self, group_name: str) -> list[str]:
        group = str(group_name)
        if group not in self.valid_actions_by_group:
            raise ValueError(f"Unknown PTS suffix length group {group!r}.")
        return list(self.valid_actions_by_group[group])

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
            "valid_actions_by_group": {
                group: list(actions)
                for group, actions in self.valid_actions_by_group.items()
            },
            "disabled_actions_by_group": {
                group: list(actions)
                for group, actions in self.disabled_actions_by_group.items()
            },
            "enabled_actions": list(self.enabled_actions),
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
        raw_valid_actions = payload.get("valid_actions_by_group")
        valid_actions_by_group = (
            None
            if raw_valid_actions is None
            else _coerce_valid_actions_payload(raw_valid_actions)
        )
        raw_enabled_actions = payload.get("enabled_actions")
        enabled_actions = (
            None
            if raw_enabled_actions is None
            else _normalize_ordered_names(
                _coerce_sequence_payload(raw_enabled_actions, "enabled_actions"),
                label="enabled_actions",
            )
        )
        return cls(
            probabilities,
            valid_actions_by_group=valid_actions_by_group,
            enabled_actions=enabled_actions,
            disable_consume_one_when_suffix_len_leq_1=bool(
                payload.get("disable_consume_one_when_suffix_len_leq_1", True)
            ),
        )


def build_valid_actions_by_group(
    *,
    group_buckets: Sequence[Any],
    enabled_actions: Sequence[str],
    disable_consume_one_when_suffix_len_leq_1: bool = True,
) -> dict[str, list[str]]:
    actions = _normalize_ordered_names(enabled_actions, label="enabled_actions")
    if not actions:
        raise ValueError("enabled_actions must not be empty.")
    result: dict[str, list[str]] = {}
    for bucket in group_buckets:
        group_name = str(getattr(bucket, "name"))
        max_len = _bucket_max_len(bucket)
        group_actions = list(actions)
        if (
            bool(disable_consume_one_when_suffix_len_leq_1)
            and max_len is not None
            and int(max_len) <= 1
        ):
            group_actions = [
                action
                for action in group_actions
                if action not in set(CONSUME_ONE_ACTION_NAMES)
            ]
        if not group_actions:
            raise ValueError(
                f"PTS valid action set for group {group_name!r} is empty."
            )
        result[group_name] = group_actions
    if not result:
        raise ValueError("group_buckets must not be empty.")
    return result


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


def _resolve_valid_actions_by_group(
    group_probabilities: Mapping[str, Mapping[str, float]],
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    if valid_actions_by_group is None:
        return {
            str(group): list(probabilities.keys())
            for group, probabilities in group_probabilities.items()
        }
    return _normalize_valid_actions_by_group(
        valid_actions_by_group,
        expected_groups=[str(group) for group in group_probabilities],
    )


def _normalize_valid_actions_by_group(
    valid_actions_by_group: Mapping[str, Sequence[str]],
    *,
    expected_groups: Sequence[str],
) -> dict[str, list[str]]:
    if not isinstance(valid_actions_by_group, Mapping):
        raise TypeError("valid_actions_by_group must be a mapping.")
    expected = [str(group) for group in expected_groups]
    expected_set = set(expected)
    provided_set = {str(group) for group in valid_actions_by_group}
    missing = sorted(expected_set - provided_set)
    extra = sorted(provided_set - expected_set)
    if missing or extra:
        raise ValueError(
            "valid_actions_by_group must match policy groups; "
            f"missing={missing}, extra={extra}."
        )
    result: dict[str, list[str]] = {}
    for group in expected:
        actions = _normalize_ordered_names(
            valid_actions_by_group[group],
            label=f"valid_actions_by_group[{group!r}]",
        )
        if not actions:
            raise ValueError(
                f"valid_actions_by_group[{group!r}] must not be empty."
            )
        result[group] = actions
    return result


def _resolve_enabled_actions(
    valid_actions_by_group: Mapping[str, Sequence[str]],
    *,
    enabled_actions: Sequence[str] | None,
) -> list[str]:
    if enabled_actions is None:
        return _ordered_union(valid_actions_by_group.values())
    actions = _normalize_ordered_names(enabled_actions, label="enabled_actions")
    action_set = set(actions)
    unknown_valid_actions = sorted(
        {
            action
            for group_actions in valid_actions_by_group.values()
            for action in group_actions
            if action not in action_set
        }
    )
    if unknown_valid_actions:
        raise ValueError(
            "valid_actions_by_group contains actions absent from enabled_actions: "
            f"{unknown_valid_actions}."
        )
    return actions


def _derive_disabled_actions_by_group(
    enabled_actions: Sequence[str],
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    disabled: dict[str, list[str]] = {}
    for group, valid_actions in valid_actions_by_group.items():
        valid_set = set(valid_actions)
        disabled[str(group)] = [
            action for action in enabled_actions if action not in valid_set
        ]
    return disabled


def _validate_probability_actions(
    group_probabilities: Mapping[str, Mapping[str, float]],
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for group, probabilities in group_probabilities.items():
        valid_actions = list(valid_actions_by_group[str(group)])
        valid_set = set(valid_actions)
        probability_set = set(probabilities)
        missing = [action for action in valid_actions if action not in probability_set]
        extra = sorted(probability_set - valid_set)
        if missing or extra:
            raise ValueError(
                f"Policy probabilities for group {group!r} must match valid actions; "
                f"missing={missing}, extra={extra}."
            )
        result[str(group)] = {action: float(probabilities[action]) for action in valid_actions}
    return result


def _normalize_ordered_names(value: Sequence[str], *, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of strings.")
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{label} must contain only strings.")
        name = str(item)
        if name in seen:
            raise ValueError(f"{label} must not contain duplicate entries.")
        seen.add(name)
        result.append(name)
    return result


def _ordered_union(groups: Sequence[Sequence[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for actions in groups:
        for action in actions:
            if action in seen:
                continue
            seen.add(action)
            result.append(action)
    return result


def _coerce_valid_actions_payload(payload: object) -> dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        raise ValueError("valid_actions_by_group must be a mapping.")
    result: dict[str, list[str]] = {}
    for group, actions in payload.items():
        result[str(group)] = _normalize_ordered_names(
            _coerce_sequence_payload(actions, f"valid_actions_by_group[{group!r}]"),
            label=f"valid_actions_by_group[{group!r}]",
        )
    return result


def _coerce_sequence_payload(payload: object, label: str) -> Sequence[str]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{label} must be a sequence.")
    return payload


def _bucket_max_len(bucket: object) -> int | None:
    if hasattr(bucket, "max_len"):
        value = getattr(bucket, "max_len")
    else:
        value = getattr(bucket, "max", None)
    return None if value is None else int(value)


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
    ):
        for action in CONSUME_ONE_ACTION_NAMES:
            if action in probabilities:
                masked_actions.append(action)

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
    "CONSUME_ONE_ACTION_NAMES",
    "CONSUME_ONE_GENERATE_ACTION_NAME",
    "GroupActionPolicy",
    "PolicySampleResult",
    "build_valid_actions_by_group",
]
