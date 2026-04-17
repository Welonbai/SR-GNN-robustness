from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


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

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "SurrogateScoreResult":
        normalized = tuple(float(value) for value in values)
        if normalized:
            mean = float(sum(normalized) / len(normalized))
        else:
            mean = float("nan")
        return cls(values=normalized, mean=mean)


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
class PositionOptDefaults:
    enabled: bool = True
    training_selector: str = "categorical_reinforce"
    eval_selector: str = "argmax"
    outer_steps: int = 30
    policy_lr: float = 0.05
    gumbel_temperature: float = 1.0
    fine_tune_steps: int = 20
    enable_gt_penalty: bool = False
    gt_penalty_weight: float = 0.0
    gt_tolerance: float = 0.0
    reward_baseline_momentum: float = 0.9
    validation_subset_size: int | None = None


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


POSITION_OPT_DEFAULTS = PositionOptDefaults()


def resolve_position_opt_config(
    overrides: PositionOptDefaults | Mapping[str, Any] | None,
) -> PositionOptDefaults:
    # Position-opt config still uses Python defaults plus explicit runtime
    # overrides. YAML/config parser wiring is intentionally deferred.
    if overrides is None:
        resolved = replace(POSITION_OPT_DEFAULTS)
    elif isinstance(overrides, PositionOptDefaults):
        resolved = overrides
    elif isinstance(overrides, Mapping):
        allowed_fields = {field.name for field in fields(PositionOptDefaults)}
        unknown = set(overrides) - allowed_fields
        if unknown:
            raise ValueError(
                "Unknown position-opt config keys: " + ", ".join(sorted(map(str, unknown)))
            )
        resolved = replace(POSITION_OPT_DEFAULTS, **dict(overrides))
    else:
        raise TypeError(
            "position_opt_config must be a PositionOptDefaults instance, a mapping, or None."
        )

    if resolved.training_selector not in {"categorical_reinforce", "st_gumbel"}:
        raise ValueError(
            "Phase 2.5 only supports training_selector='categorical_reinforce' "
            "or the legacy 'st_gumbel' compatibility value."
        )
    if resolved.eval_selector != "argmax":
        raise ValueError("Phase 2.5 only supports eval_selector='argmax'.")
    if int(resolved.outer_steps) < 0:
        raise ValueError("outer_steps must be non-negative.")
    if float(resolved.policy_lr) <= 0.0:
        raise ValueError("policy_lr must be positive.")
    if float(resolved.gumbel_temperature) <= 0.0:
        raise ValueError("gumbel_temperature must be positive.")
    if int(resolved.fine_tune_steps) < 0:
        raise ValueError("fine_tune_steps must be non-negative.")
    if float(resolved.gt_penalty_weight) < 0.0:
        raise ValueError("gt_penalty_weight must be non-negative.")
    if float(resolved.gt_tolerance) < 0.0:
        raise ValueError("gt_tolerance must be non-negative.")
    if not 0.0 <= float(resolved.reward_baseline_momentum) <= 1.0:
        raise ValueError("reward_baseline_momentum must be in [0, 1].")
    if resolved.validation_subset_size is not None and int(resolved.validation_subset_size) <= 0:
        raise ValueError("validation_subset_size must be positive when provided.")
    return resolved


__all__ = [
    "POSITION_OPT_DEFAULTS",
    "CandidateMetadata",
    "InnerTrainResult",
    "PositionOptArtifactPaths",
    "PositionOptDefaults",
    "SelectedPositionResult",
    "SurrogateScoreResult",
    "TruncatedFineTuneConfig",
    "resolve_position_opt_config",
]
