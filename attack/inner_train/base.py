from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from attack.position_opt.types import InnerTrainResult, TruncatedFineTuneConfig
from attack.surrogate.base import PoisonedTrainInput, SurrogateBackend


@runtime_checkable
class InnerTrainer(Protocol):
    def run(
        self,
        surrogate_backend: SurrogateBackend,
        clean_checkpoint_path: str | Path,
        poisoned_train_data: PoisonedTrainInput,
        *,
        config: TruncatedFineTuneConfig | None = None,
        eval_data: Any | None = None,
    ) -> InnerTrainResult:
        ...


__all__ = ["InnerTrainer"]
