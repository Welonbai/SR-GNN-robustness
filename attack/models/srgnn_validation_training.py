from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from attack.common.srgnn_training_protocol import (
    SRGNN_VALIDATION_BEST_METRIC,
    SRGNN_VALIDATION_PATIENCE_METRIC,
    SRGNN_VALIDATION_PROTOCOL_NAME,
    srgnn_best_metric,
    srgnn_patience_metric,
)
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics
from pytorch_code.model import forward as srg_forward, trans_to_cuda


_GT_METRICS = ("recall", "mrr")
_GT_TOPK = (20,)
VALID_RECALL20_KEY = "valid_ground_truth_recall@20"
VALID_MRR20_KEY = "valid_ground_truth_mrr@20"


@dataclass(frozen=True)
class SRGNNValidationTrainingResult:
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def improved_over_best(current: float, best: float) -> bool:
    return float(current) > float(best)


def official_patience_improved(
    *,
    current_valid_recall20: float,
    best_valid_recall20: float,
    current_valid_mrr20: float,
    best_valid_mrr20: float,
) -> bool:
    return bool(
        improved_over_best(current_valid_recall20, best_valid_recall20)
        or improved_over_best(current_valid_mrr20, best_valid_mrr20)
    )


def train_srgnn_validation_best(
    runner: Any,
    train_data: Any,
    valid_data: Any,
    *,
    train_config: Mapping[str, Any],
    max_epochs: int,
    patience: int,
    best_checkpoint_path: str | Path | None = None,
    log_prefix: str = "[srgnn-validation-best]",
) -> SRGNNValidationTrainingResult:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    if int(max_epochs) <= 0:
        raise ValueError("max_epochs must be positive.")
    if int(patience) <= 0:
        raise ValueError("patience must be positive.")
    _validate_supported_training_selection(train_config)

    rows: list[dict[str, Any]] = []
    best_valid_mrr20 = float("-inf")
    best_valid_recall20 = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_counter = 0
    stopped_epoch: int | None = None
    stop_reason = "max_epochs_completed"
    start = time.perf_counter()

    for epoch in range(1, int(max_epochs) + 1):
        print(f"{log_prefix} epoch={epoch}/{int(max_epochs)} training...", flush=True)
        train_loss = _train_one_epoch(runner, train_data)
        valid_metrics = _evaluate_ground_truth_for_data(runner, valid_data)
        current_valid_mrr20 = float(valid_metrics["ground_truth_mrr@20"])
        current_valid_recall20 = float(valid_metrics["ground_truth_recall@20"])

        improved_mrr20 = improved_over_best(current_valid_mrr20, best_valid_mrr20)
        improved_recall20 = improved_over_best(
            current_valid_recall20,
            best_valid_recall20,
        )
        improved_for_patience = official_patience_improved(
            current_valid_recall20=current_valid_recall20,
            best_valid_recall20=best_valid_recall20,
            current_valid_mrr20=current_valid_mrr20,
            best_valid_mrr20=best_valid_mrr20,
        )

        if improved_mrr20:
            best_valid_mrr20 = current_valid_mrr20
            best_state = _copy_model_state(runner)
            if best_checkpoint_path is not None:
                runner.save_model(best_checkpoint_path)
        if improved_recall20:
            best_valid_recall20 = current_valid_recall20

        if improved_for_patience:
            bad_counter = 0
        else:
            bad_counter += 1

        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "valid_ground_truth_recall@20": current_valid_recall20,
            "valid_ground_truth_mrr@20": current_valid_mrr20,
            "improved_valid_mrr20_this_epoch": bool(improved_mrr20),
            "improved_valid_recall20_this_epoch": bool(improved_recall20),
            "improved_valid_recall_or_mrr20_this_epoch": bool(improved_for_patience),
            "is_best_mrr20": bool(improved_mrr20),
            "bad_counter": int(bad_counter),
            "lr": _current_lr(runner),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        rows.append(row)
        print(
            f"{log_prefix} epoch={epoch} train_loss={train_loss:.6g} "
            f"valid_mrr20={current_valid_mrr20:.9f} "
            f"valid_recall20={current_valid_recall20:.9f} "
            f"improved_mrr20={improved_mrr20} "
            f"improved_recall20={improved_recall20} "
            f"bad_counter={bad_counter}",
            flush=True,
        )

        if bad_counter >= int(patience):
            stopped_epoch = int(epoch)
            stop_reason = "patience_exhausted"
            print(f"{log_prefix} early stop at epoch={epoch} patience={int(patience)}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("SR-GNN validation-best training did not produce a checkpoint.")
    runner.model.load_state_dict(best_state)
    if best_checkpoint_path is not None:
        runner.save_model(best_checkpoint_path)

    summary = build_srgnn_validation_summary(
        rows,
        max_epochs=int(max_epochs),
        patience=int(patience),
        stopped_epoch=stopped_epoch if stopped_epoch is not None else int(rows[-1]["epoch"]),
        stop_reason=stop_reason,
    )
    return SRGNNValidationTrainingResult(rows=rows, summary=summary)


def build_srgnn_validation_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_epochs: int,
    patience: int,
    stopped_epoch: int,
    stop_reason: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot build SR-GNN validation summary from empty rows.")
    best_mrr20_row = _best_row(rows, VALID_MRR20_KEY)
    best_recall20_row = _best_row(rows, VALID_RECALL20_KEY)
    return {
        "validation_protocol": SRGNN_VALIDATION_PROTOCOL_NAME,
        "checkpoint_protocol": "validation_best",
        "selected_checkpoint_metric": SRGNN_VALIDATION_BEST_METRIC,
        "selected_checkpoint_epoch": int(best_mrr20_row["epoch"]),
        "patience_metric": SRGNN_VALIDATION_PATIENCE_METRIC,
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "epochs_completed": int(len(rows)),
        "stopped_epoch": int(stopped_epoch),
        "stop_reason": str(stop_reason),
        "best_valid_mrr20_epoch": int(best_mrr20_row["epoch"]),
        "best_valid_mrr20": float(best_mrr20_row[VALID_MRR20_KEY]),
        "best_valid_recall20_at_best_mrr20_epoch": float(best_mrr20_row[VALID_RECALL20_KEY]),
        "best_valid_recall20_epoch": int(best_recall20_row["epoch"]),
        "best_valid_recall20": float(best_recall20_row[VALID_RECALL20_KEY]),
        "is_best_mrr20_note": (
            "is_best_mrr20 means this epoch improved over the previous best validation "
            "MRR@20, not necessarily that it is the final global best epoch."
        ),
    }


def srgnn_validation_train_history_extra(
    result: SRGNNValidationTrainingResult,
) -> dict[str, Any]:
    return {
        **result.summary,
        "valid_ground_truth_recall@20": [
            float(row[VALID_RECALL20_KEY])
            for row in result.rows
        ],
        "valid_ground_truth_mrr@20": [
            float(row[VALID_MRR20_KEY])
            for row in result.rows
        ],
        "history_rows": [dict(row) for row in result.rows],
    }


def _validate_supported_training_selection(train_config: Mapping[str, Any]) -> None:
    best_metric = srgnn_best_metric(train_config)
    if best_metric != SRGNN_VALIDATION_BEST_METRIC:
        raise ValueError(
            "SR-GNN validation-best training currently supports only "
            f"{SRGNN_VALIDATION_BEST_METRIC!r} as best_metric."
        )
    patience_metric = srgnn_patience_metric(train_config)
    if patience_metric != SRGNN_VALIDATION_PATIENCE_METRIC:
        raise ValueError(
            "SR-GNN validation-best training currently supports only "
            f"{SRGNN_VALIDATION_PATIENCE_METRIC!r} as patience_metric."
        )


def _train_one_epoch(runner: Any, train_data: Any) -> float:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    model = runner.model
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch_indices in train_data.generate_batch(model.batch_size):
        model.optimizer.zero_grad()
        targets, scores = srg_forward(model, batch_indices, train_data)
        targets_tensor = trans_to_cuda(torch.as_tensor(targets, dtype=torch.long))
        loss = model.loss_function(scores, targets_tensor - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
    model.scheduler.step()
    avg_loss = total_loss / float(max(1, batch_count))
    runner.train_loss_history.append(float(avg_loss))
    return float(avg_loss)


def _evaluate_ground_truth_for_data(runner: Any, data: Any) -> dict[str, float]:
    labels = [int(label) for label in data.targets.tolist()]
    rankings = runner.predict_topk(data, topk=max(_GT_TOPK))
    metrics, available = evaluate_ground_truth_metrics(
        rankings,
        labels=labels,
        metrics=_GT_METRICS,
        topk=_GT_TOPK,
    )
    if not available:
        raise RuntimeError("Ground-truth metrics are unavailable.")
    return {str(key): float(value) for key, value in metrics.items()}


def _copy_model_state(runner: Any) -> dict[str, torch.Tensor]:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    return {
        key: value.detach().cpu().clone()
        for key, value in runner.model.state_dict().items()
    }


def _current_lr(runner: Any) -> float:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    return float(runner.model.optimizer.param_groups[0]["lr"])


def _best_row(rows: Sequence[Mapping[str, Any]], metric_key: str) -> Mapping[str, Any]:
    return max(rows, key=lambda row: (float(row[metric_key]), -int(row["epoch"])))


__all__ = [
    "SRGNNValidationTrainingResult",
    "VALID_MRR20_KEY",
    "VALID_RECALL20_KEY",
    "build_srgnn_validation_summary",
    "improved_over_best",
    "official_patience_improved",
    "srgnn_validation_train_history_extra",
    "train_srgnn_validation_best",
]
