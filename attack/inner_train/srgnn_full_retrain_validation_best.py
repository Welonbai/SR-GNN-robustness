from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.models.srgnn_validation_training import (
    srgnn_validation_train_history_extra,
    train_srgnn_validation_best,
)
from attack.position_opt.types import InnerTrainResult
from attack.surrogate.base import PoisonedTrainInput, SurrogateBackend
from pytorch_code.utils import Data


class SRGNNFullRetrainValidationBestInnerTrainer:
    """Fresh SR-GNN surrogate evaluator with validation-best checkpointing."""

    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        max_epochs: int | None = None,
        patience: int | None = None,
        log_prefix: str = "[surrogate:srgnn-full-retrain]",
    ) -> None:
        self.train_config = dict(train_config)
        self.max_epochs = int(max_epochs if max_epochs is not None else self.train_config["epochs"])
        self.patience = int(patience if patience is not None else self.train_config["patience"])
        self.log_prefix = str(log_prefix)

    def run(
        self,
        surrogate_backend: SurrogateBackend,
        clean_checkpoint_path: str | Path | None,
        poisoned_train_data: PoisonedTrainInput,
        *,
        config: Any | None = None,
        eval_data: Any | None = None,
        seed: int | None = None,
    ) -> InnerTrainResult:
        del clean_checkpoint_path, config
        if seed is not None:
            set_seed(int(seed))
        if eval_data is None:
            raise ValueError(
                "SR-GNN full-retrain validation-best surrogate evaluation requires "
                "validation eval_data."
            )
        build_fresh_model = getattr(surrogate_backend, "build_fresh_model", None)
        if build_fresh_model is None:
            raise TypeError(
                "SR-GNN full-retrain validation-best surrogate evaluation requires "
                "a backend with build_fresh_model()."
            )

        sessions, labels = _coerce_poisoned_train_data(poisoned_train_data)
        train_data = Data((sessions, labels), shuffle=True)
        model = build_fresh_model()
        result = train_srgnn_validation_best(
            model.runner,
            train_data,
            eval_data,
            train_config=self.train_config,
            max_epochs=self.max_epochs,
            patience=self.patience,
            best_checkpoint_path=None,
            log_prefix=self.log_prefix,
        )
        history = {
            "surrogate_evaluator_mode": "full_retrain_validation_best",
            "steps": None,
            "epochs": int(result.summary["epochs_completed"]),
            "avg_loss": None,
            **srgnn_validation_train_history_extra(result),
        }
        return InnerTrainResult(model=model, history=history)


def _coerce_poisoned_train_data(
    poisoned_train_data: PoisonedTrainInput,
) -> tuple[list[list[int]], list[int]]:
    if isinstance(poisoned_train_data, PoisonedDataset):
        sessions = poisoned_train_data.sessions
        labels = poisoned_train_data.labels
    else:
        sessions, labels = poisoned_train_data

    normalized_sessions = [list(session) for session in sessions]
    normalized_labels = [int(label) for label in labels]
    if len(normalized_sessions) != len(normalized_labels):
        raise ValueError("poisoned_train_data sessions and labels must have the same length.")
    if not normalized_sessions:
        raise ValueError("poisoned_train_data must contain at least one training sample.")
    if any(len(session) == 0 for session in normalized_sessions):
        raise ValueError("poisoned_train_data sessions must be non-empty.")
    return normalized_sessions, normalized_labels


__all__ = ["SRGNNFullRetrainValidationBestInnerTrainer"]
