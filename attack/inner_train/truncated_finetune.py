from __future__ import annotations

from pathlib import Path
from typing import Any

from attack.position_opt.types import InnerTrainResult, TruncatedFineTuneConfig
from attack.surrogate.base import PoisonedTrainInput, SurrogateBackend


class TruncatedFineTuneInnerTrainer:
    """MVP inner trainer for the new surrogate-model role.

    This helper always starts from an explicit clean surrogate checkpoint, clones
    a clean surrogate model through the backend, and applies only a short
    fine-tuning budget. It does not reuse the DPSBR poison-model checkpoint.
    """

    def __init__(self, default_config: TruncatedFineTuneConfig | None = None) -> None:
        self.default_config = default_config or TruncatedFineTuneConfig()

    def run(
        self,
        surrogate_backend: SurrogateBackend,
        clean_checkpoint_path: str | Path,
        poisoned_train_data: PoisonedTrainInput,
        *,
        config: TruncatedFineTuneConfig | None = None,
        eval_data: Any | None = None,
    ) -> InnerTrainResult:
        effective_config = config or self.default_config
        # Phase 1 keeps checkpoint resolution outside the trainer. The caller must
        # provide the clean surrogate checkpoint explicitly until later artifact or
        # config wiring is added.
        surrogate_backend.load_clean_checkpoint(clean_checkpoint_path)
        model = surrogate_backend.clone_clean_model()
        history = surrogate_backend.fine_tune(
            model,
            poisoned_train_data,
            fine_tune_config=effective_config,
            eval_data=eval_data,
        )
        return InnerTrainResult(model=model, history=history)


__all__ = ["TruncatedFineTuneInnerTrainer"]
