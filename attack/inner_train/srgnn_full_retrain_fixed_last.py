from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.models.srgnn_validation_training import train_srgnn_one_epoch
from attack.position_opt.types import InnerTrainResult
from attack.surrogate.base import PoisonedTrainInput, SurrogateBackend
from pytorch_code.utils import Data


class SRGNNFullRetrainFixedLastInnerTrainer:
    """Fresh SR-GNN surrogate evaluator with fixed-last checkpoint semantics."""

    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        max_epochs: int | None = None,
        log_prefix: str = "[surrogate:srgnn-full-retrain-fixed-last]",
        log_epochs: bool = True,
    ) -> None:
        self.train_config = dict(train_config)
        self.max_epochs = int(max_epochs if max_epochs is not None else self.train_config["epochs"])
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        self.log_prefix = str(log_prefix)
        self.log_epochs = bool(log_epochs)

    def run(
        self,
        surrogate_backend: SurrogateBackend,
        clean_checkpoint_path: str | Path | None,
        poisoned_train_data: PoisonedTrainInput,
        *,
        config: Any | None = None,
        eval_data: Any | None = None,
        seed: int | None = None,
        epoch_callback: Callable[[object, Mapping[str, Any]], None] | None = None,
    ) -> InnerTrainResult:
        del clean_checkpoint_path, config, eval_data
        if seed is not None:
            set_seed(int(seed))
        build_fresh_model = getattr(surrogate_backend, "build_fresh_model", None)
        if build_fresh_model is None:
            raise TypeError(
                "SR-GNN full-retrain fixed-last surrogate evaluation requires "
                "a backend with build_fresh_model()."
            )

        sessions, labels = _coerce_poisoned_train_data(poisoned_train_data)
        train_data = Data((sessions, labels), shuffle=True)
        model = build_fresh_model()
        rows: list[dict[str, Any]] = []
        start = time.perf_counter()

        for epoch in range(1, int(self.max_epochs) + 1):
            if bool(self.log_epochs):
                print(
                    f"{self.log_prefix} epoch={epoch}/{int(self.max_epochs)} training...",
                    flush=True,
                )
            train_loss = train_srgnn_one_epoch(model.runner, train_data)
            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "lr": _current_lr(model.runner),
                "elapsed_seconds": float(time.perf_counter() - start),
            }
            rows.append(row)
            if bool(self.log_epochs):
                print(
                    f"{self.log_prefix} epoch={epoch} train_loss={train_loss:.6g}",
                    flush=True,
                )
            if epoch_callback is not None:
                epoch_callback(model, row)

        history = {
            "surrogate_evaluator_mode": "full_retrain_fixed_last",
            "checkpoint_protocol": "fixed_last",
            "steps": None,
            "epochs": int(self.max_epochs),
            "epochs_completed": int(self.max_epochs),
            "avg_loss": (
                None
                if not rows
                else float(sum(float(row["train_loss"]) for row in rows) / len(rows))
            ),
            "selected_checkpoint_epoch": int(self.max_epochs),
            "selected_checkpoint_protocol": "fixed_last",
            "selected_checkpoint_source": "last_epoch",
            "selected_checkpoint_metric": None,
            "validation_best_metrics_recorded": False,
            "official_reward_checkpoint_epoch": int(self.max_epochs),
            "max_epochs": int(self.max_epochs),
            "stopped_epoch": int(self.max_epochs),
            "stop_reason": "max_epochs_completed",
            "history_rows": [dict(row) for row in rows],
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


def _current_lr(runner: Any) -> float:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    return float(runner.model.optimizer.param_groups[0]["lr"])


__all__ = ["SRGNNFullRetrainFixedLastInnerTrainer"]
