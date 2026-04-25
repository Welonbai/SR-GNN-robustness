from __future__ import annotations

from dataclasses import dataclass


ALLOWED_POSITION_OPT_ITEM_FEATURES = (
    "target_item",
    "original_item",
    "left_item",
    "right_item",
)
ALLOWED_POSITION_OPT_SCALAR_FEATURES = (
    "position_index",
    "normalized_position",
    "session_length",
    "prefix_score",
    "has_prefix",
)
_ALLOWED_ITEM_FEATURE_SET = frozenset(ALLOWED_POSITION_OPT_ITEM_FEATURES)
_ALLOWED_SCALAR_FEATURE_SET = frozenset(ALLOWED_POSITION_OPT_SCALAR_FEATURES)


@dataclass(frozen=True)
class PositionOptPolicyFeatureSetSpec:
    name: str
    item_features: tuple[str, ...]
    scalar_features: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown_item_features = sorted(
            set(self.item_features) - _ALLOWED_ITEM_FEATURE_SET
        )
        if unknown_item_features:
            raise ValueError(
                f"Unknown item features for {self.name!r}: "
                + ", ".join(unknown_item_features)
            )
        unknown_scalar_features = sorted(
            set(self.scalar_features) - _ALLOWED_SCALAR_FEATURE_SET
        )
        if unknown_scalar_features:
            raise ValueError(
                f"Unknown scalar features for {self.name!r}: "
                + ", ".join(unknown_scalar_features)
            )
        if not self.item_features and not self.scalar_features:
            raise ValueError(
                f"Policy feature set {self.name!r} must enable at least one input feature."
            )

    @property
    def requires_prefix_scores(self) -> bool:
        return "prefix_score" in self.scalar_features

    @property
    def requires_has_prefix_flags(self) -> bool:
        return "has_prefix" in self.scalar_features

    @property
    def requires_prefix_features(self) -> bool:
        return self.requires_prefix_scores or self.requires_has_prefix_flags

    def input_dim(self, *, embedding_dim: int) -> int:
        return (len(self.item_features) * int(embedding_dim)) + len(self.scalar_features)


POSITION_OPT_POLICY_FEATURE_SET_SPECS = {
    "local_context": PositionOptPolicyFeatureSetSpec(
        name="local_context",
        item_features=(
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ),
        scalar_features=(
            "position_index",
            "normalized_position",
            "session_length",
        ),
    ),
    "local_context_prefix_score_prob": PositionOptPolicyFeatureSetSpec(
        name="local_context_prefix_score_prob",
        item_features=(
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ),
        scalar_features=(
            "position_index",
            "normalized_position",
            "session_length",
            "prefix_score",
            "has_prefix",
        ),
    ),
    "normalized_position_only": PositionOptPolicyFeatureSetSpec(
        name="normalized_position_only",
        item_features=(),
        scalar_features=("normalized_position",),
    ),
    "target_normalized_position": PositionOptPolicyFeatureSetSpec(
        name="target_normalized_position",
        item_features=("target_item",),
        scalar_features=("normalized_position",),
    ),
    "target_original_normalized_position": PositionOptPolicyFeatureSetSpec(
        name="target_original_normalized_position",
        item_features=(
            "target_item",
            "original_item",
        ),
        scalar_features=("normalized_position",),
    ),
    "target_original_position_scalar": PositionOptPolicyFeatureSetSpec(
        name="target_original_position_scalar",
        item_features=(
            "target_item",
            "original_item",
        ),
        scalar_features=(
            "position_index",
            "normalized_position",
            "session_length",
        ),
    ),
    "full_context_normalized_position": PositionOptPolicyFeatureSetSpec(
        name="full_context_normalized_position",
        item_features=(
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ),
        scalar_features=("normalized_position",),
    ),
}
ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS = tuple(POSITION_OPT_POLICY_FEATURE_SET_SPECS)


def resolve_position_opt_policy_feature_set(
    policy_feature_set: str,
) -> PositionOptPolicyFeatureSetSpec:
    normalized_policy_feature_set = str(policy_feature_set).strip().lower()
    try:
        return POSITION_OPT_POLICY_FEATURE_SET_SPECS[normalized_policy_feature_set]
    except KeyError as exc:
        allowed_feature_sets = ", ".join(ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS)
        raise ValueError(
            "policy_feature_set must be one of: " f"{allowed_feature_sets}."
        ) from exc


__all__ = [
    "ALLOWED_POSITION_OPT_ITEM_FEATURES",
    "ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS",
    "ALLOWED_POSITION_OPT_SCALAR_FEATURES",
    "POSITION_OPT_POLICY_FEATURE_SET_SPECS",
    "PositionOptPolicyFeatureSetSpec",
    "resolve_position_opt_policy_feature_set",
]
