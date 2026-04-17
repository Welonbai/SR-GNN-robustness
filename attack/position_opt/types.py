from __future__ import annotations

from dataclasses import dataclass
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
    training_selector: str = "st_gumbel"
    eval_selector: str = "argmax"
    outer_steps: int = 30
    policy_lr: float = 0.05
    gumbel_temperature: float = 1.0
    fine_tune_steps: int = 20
    enable_gt_penalty: bool = False
    gt_penalty_weight: float = 0.0
    gt_tolerance: float = 0.0


@dataclass(frozen=True)
class PositionOptArtifactPaths:
    base_dir: Path
    clean_surrogate_checkpoint: Path
    optimized_poisoned_sessions: Path
    training_history: Path | None = None
    learned_logits: Path | None = None


@dataclass(frozen=True)
class InnerTrainResult:
    model: object
    history: Mapping[str, Any] | None = None


POSITION_OPT_DEFAULTS = PositionOptDefaults()


__all__ = [
    "POSITION_OPT_DEFAULTS",
    "CandidateMetadata",
    "InnerTrainResult",
    "PositionOptArtifactPaths",
    "PositionOptDefaults",
    "SelectedPositionResult",
    "SurrogateScoreResult",
    "TruncatedFineTuneConfig",
]
