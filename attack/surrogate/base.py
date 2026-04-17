from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.position_opt.types import SurrogateScoreResult, TruncatedFineTuneConfig


SessionBatch = Sequence[Sequence[int]]
PoisonedTrainInput = PoisonedDataset | tuple[Sequence[Sequence[int]], Sequence[int]]


@runtime_checkable
class SurrogateBackend(Protocol):
    def load_clean_checkpoint(self, path: str | Path) -> None:
        ...

    def clone_clean_model(self) -> object:
        ...

    def fine_tune(
        self,
        model: object,
        poisoned_train_data: PoisonedTrainInput,
        *,
        fine_tune_config: TruncatedFineTuneConfig | None = None,
        eval_data: Any | None = None,
    ) -> dict[str, Any] | None:
        ...

    def score_target(
        self,
        model: object,
        eval_sessions: SessionBatch,
        target_item: int,
    ) -> SurrogateScoreResult:
        ...

    def score_gt(
        self,
        model: object,
        eval_sessions: SessionBatch,
        ground_truth_items: Sequence[int],
    ) -> SurrogateScoreResult:
        ...


__all__ = ["PoisonedTrainInput", "SessionBatch", "SurrogateBackend"]
