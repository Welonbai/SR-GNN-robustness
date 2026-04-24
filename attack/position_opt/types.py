from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

from attack.common.config import PositionOptConfig


@dataclass(frozen=True)
class CandidateMetadata:
    session_length: int
    replacement_topk_ratio: float
    positions: tuple[int, ...]


@dataclass(frozen=True)
class SelectedPositionResult:
    position: int
    candidate_index: int | None = None
    score: float | None = None


@dataclass(frozen=True)
class SurrogateScoreResult:
    values: tuple[float, ...]
    mean: float
    metrics: dict[str, float] | None = None

    @classmethod
    def from_values(
        cls,
        values: Sequence[float],
        *,
        metrics: Mapping[str, float] | None = None,
    ) -> "SurrogateScoreResult":
        normalized = tuple(float(value) for value in values)
        if normalized:
            mean = float(sum(normalized) / len(normalized))
        else:
            mean = float("nan")
        normalized_metrics = None
        if metrics is not None:
            normalized_metrics = {
                str(key): float(value) for key, value in metrics.items()
            }
        return cls(values=normalized, mean=mean, metrics=normalized_metrics)


@dataclass(frozen=True)
class TruncatedFineTuneConfig:
    steps: int = 20
    epochs: int = 1

    def __post_init__(self) -> None:
        if int(self.steps) < 0:
            raise ValueError("steps must be non-negative.")
        if int(self.epochs) < 0:
            raise ValueError("epochs must be non-negative.")


@dataclass(frozen=True)
class PositionOptArtifactPaths:
    base_dir: Path
    clean_surrogate_checkpoint: Path
    optimized_poisoned_sessions: Path
    selected_positions: Path | None = None
    training_history: Path | None = None
    learned_logits: Path | None = None
    run_metadata: Path | None = None


@dataclass(frozen=True)
class InnerTrainResult:
    model: object
    history: Mapping[str, Any] | None = None


POSITION_OPT_DEFAULTS = PositionOptConfig()


def resolve_position_opt_config(
    config: PositionOptConfig | Mapping[str, Any] | None = None,
    overrides: PositionOptConfig | Mapping[str, Any] | None = None,
) -> PositionOptConfig:
    merged = asdict(POSITION_OPT_DEFAULTS)
    merged.update(_coerce_position_opt_layer(config, context="attack.position_opt"))
    merged.update(
        _coerce_position_opt_layer(overrides, context="position_opt_config override")
    )
    return PositionOptConfig(**merged)


def position_opt_identity_payload(config: PositionOptConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("clean_surrogate_checkpoint", None)
    if payload.get("policy_feature_set") == "local_context":
        payload.pop("policy_feature_set", None)
    return payload


def _coerce_position_opt_layer(
    value: PositionOptConfig | Mapping[str, Any] | None,
    *,
    context: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, PositionOptConfig):
        return asdict(value)
    if isinstance(value, Mapping):
        payload = dict(value)
        _consume_deprecated_position_opt_keys(payload, context=context)
        allowed_fields = {field.name for field in fields(PositionOptConfig)}
        unknown = set(payload) - allowed_fields
        if unknown:
            raise ValueError(
                "Unknown position-opt config keys: " + ", ".join(sorted(map(str, unknown)))
            )
        return payload
    raise TypeError(
        "position_opt_config must be a PositionOptConfig instance, a mapping, or None."
    )


def _consume_deprecated_position_opt_keys(
    payload: dict[str, Any],
    *,
    context: str,
) -> None:
    training_selector = payload.pop("training_selector", None)
    if training_selector is not None:
        if training_selector != "categorical_reinforce":
            raise ValueError(
                f"{context}.training_selector is deprecated. The active position-opt "
                "trainer is fixed to joint REINFORCE and no longer supports "
                "selector-family switching."
            )
        warnings.warn(
            f"{context}.training_selector is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=3,
        )

    eval_selector = payload.pop("eval_selector", None)
    if eval_selector is not None:
        if eval_selector != "argmax":
            raise ValueError(
                f"{context}.eval_selector is deprecated. The current MVP only supports "
                "final_selection='argmax'."
            )
        warnings.warn(
            f"{context}.eval_selector is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=3,
        )

    if "gumbel_temperature" in payload:
        payload.pop("gumbel_temperature")
        warnings.warn(
            f"{context}.gumbel_temperature is deprecated and ignored. Gumbel is no "
            "longer part of the active joint-REINFORCE config path.",
            DeprecationWarning,
            stacklevel=3,
        )


__all__ = [
    "POSITION_OPT_DEFAULTS",
    "CandidateMetadata",
    "InnerTrainResult",
    "PositionOptConfig",
    "PositionOptArtifactPaths",
    "SelectedPositionResult",
    "SurrogateScoreResult",
    "TruncatedFineTuneConfig",
    "position_opt_identity_payload",
    "resolve_position_opt_config",
]
