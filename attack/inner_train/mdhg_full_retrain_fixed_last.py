from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Mapping

from attack.common.seed import set_seed
from attack.models.mdhg_core import build_mdhg_train_rows
from attack.position_opt.types import InnerTrainResult
from attack.surrogate.base import PoisonedTrainInput
from attack.surrogate.mdhg_backend import (
    MDHGBackend,
    MDHGModelHandle,
    coerce_mdhg_poisoned_train_data,
)


class MDHGFullRetrainFixedLastInnerTrainer:
    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        max_epochs: int | None = None,
        log_prefix: str = "[surrogate:mdhg-full-retrain-fixed-last]",
        log_epochs: bool = True,
    ) -> None:
        self.train_config = dict(train_config)
        self.max_epochs = int(max_epochs if max_epochs is not None else self.train_config["epochs"])
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        self.log_prefix = str(log_prefix)
        self.log_epochs = bool(log_epochs)
        self.last_train_rows_fingerprint = None
        self.last_raw_session_fingerprint = None

    def run(
        self,
        surrogate_backend,
        clean_checkpoint_path: str | Path | None,
        poisoned_train_data: PoisonedTrainInput,
        *,
        config: Any | None = None,
        eval_data: Any | None = None,
        seed: int | None = None,
        epoch_callback: Callable[[object, Mapping[str, Any]], None] | None = None,
    ) -> InnerTrainResult:
        del clean_checkpoint_path, config, eval_data
        if not isinstance(surrogate_backend, MDHGBackend):
            raise TypeError("MDHG fixed-last trainer requires MDHGBackend.")
        if seed is not None:
            set_seed(int(seed))
        sessions, labels, raw_train_sessions, fake_sessions = coerce_mdhg_poisoned_train_data(
            poisoned_train_data,
            clean_train_raw_sessions=surrogate_backend.clean_train_raw_sessions,
        )
        train_rows = build_mdhg_train_rows(
            sessions,
            labels,
            raw_train_sessions=raw_train_sessions,
            item_count=int(surrogate_backend.item_count),
        )
        self.last_train_rows_fingerprint = train_rows.row_fingerprint
        self.last_raw_session_fingerprint = train_rows.raw_session_fingerprint
        model = surrogate_backend.build_fresh_model()
        if not isinstance(model, MDHGModelHandle):
            raise TypeError("MDHG backend returned an invalid model handle.")

        start = time.perf_counter()
        rows: list[dict[str, Any]] = []
        try:
            losses = model.model.train_pairs(
                train_rows.sessions,
                train_rows.labels,
                raw_train_sessions=train_rows.raw_train_sessions,
                epochs=int(self.max_epochs),
            )
            for epoch, train_loss in enumerate(losses, start=1):
                row = {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "elapsed_seconds": float(time.perf_counter() - start),
                }
                rows.append(row)
                if bool(self.log_epochs):
                    print(
                        f"{self.log_prefix} epoch={epoch}/{int(self.max_epochs)} "
                        f"train_loss={float(train_loss):.6g}",
                        flush=True,
                    )
                if epoch_callback is not None:
                    epoch_callback(model, row)
            history = {
                "surrogate_evaluator_mode": "mdhg_full_retrain_fixed_last",
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
                "mdhg_train_row_count": int(len(train_rows.sessions)),
                "mdhg_raw_train_session_count": int(len(train_rows.raw_train_sessions)),
                "mdhg_candidate_fake_session_count": int(len(fake_sessions)),
                "mdhg_duplicate_rows_preserved": True,
                "mdhg_duplicate_raw_sessions_preserved": True,
            }
            return InnerTrainResult(model=model, history=history)
        except Exception:
            model.cleanup()
            raise


__all__ = ["MDHGFullRetrainFixedLastInnerTrainer"]
