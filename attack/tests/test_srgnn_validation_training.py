from __future__ import annotations

import pytest

from attack.common.srgnn_training_protocol import (
    SRGNN_VALIDATION_BEST_METRIC,
    SRGNN_VALIDATION_PATIENCE_METRIC,
    SRGNN_VALIDATION_PROTOCOL_NAME,
)
from attack.models.srgnn_validation_training import (
    build_srgnn_validation_summary,
    official_patience_improved,
    srgnn_validation_train_history_extra,
    SRGNNValidationTrainingResult,
)


def _row(
    epoch: int,
    *,
    recall20: float,
    mrr20: float,
    bad_counter: int = 0,
) -> dict[str, object]:
    return {
        "epoch": int(epoch),
        "train_loss": 1.0 / float(epoch),
        "valid_ground_truth_recall@20": float(recall20),
        "valid_ground_truth_mrr@20": float(mrr20),
        "improved_valid_mrr20_this_epoch": False,
        "improved_valid_recall20_this_epoch": False,
        "improved_valid_recall_or_mrr20_this_epoch": False,
        "is_best_mrr20": False,
        "bad_counter": int(bad_counter),
        "lr": 0.001,
    }


def test_patience_rule_resets_on_recall_or_mrr20_strict_improvement() -> None:
    assert official_patience_improved(
        current_valid_recall20=0.31,
        best_valid_recall20=0.30,
        current_valid_mrr20=0.10,
        best_valid_mrr20=0.11,
    )
    assert official_patience_improved(
        current_valid_recall20=0.30,
        best_valid_recall20=0.31,
        current_valid_mrr20=0.12,
        best_valid_mrr20=0.11,
    )
    assert not official_patience_improved(
        current_valid_recall20=0.30,
        best_valid_recall20=0.30,
        current_valid_mrr20=0.11,
        best_valid_mrr20=0.11,
    )


def test_summary_selects_mrr20_checkpoint_and_reports_recall_best_separately() -> None:
    rows = [
        _row(1, recall20=0.40, mrr20=0.10),
        _row(2, recall20=0.45, mrr20=0.12),
        _row(3, recall20=0.50, mrr20=0.11),
    ]

    summary = build_srgnn_validation_summary(
        rows,
        max_epochs=30,
        patience=10,
        stopped_epoch=3,
        stop_reason="max_epochs_completed",
    )

    assert summary["validation_protocol"] == SRGNN_VALIDATION_PROTOCOL_NAME
    assert summary["selected_checkpoint_metric"] == SRGNN_VALIDATION_BEST_METRIC
    assert summary["patience_metric"] == SRGNN_VALIDATION_PATIENCE_METRIC
    assert summary["selected_checkpoint_epoch"] == 2
    assert summary["best_valid_mrr20_epoch"] == 2
    assert summary["best_valid_mrr20"] == pytest.approx(0.12)
    assert summary["best_valid_recall20_at_best_mrr20_epoch"] == pytest.approx(0.45)
    assert summary["best_valid_recall20_epoch"] == 3
    assert summary["best_valid_recall20"] == pytest.approx(0.50)
    assert summary["stopped_epoch"] == 3
    assert summary["stop_reason"] == "max_epochs_completed"


def test_train_history_extra_contains_checkpoint_metadata_and_rows() -> None:
    rows = [
        _row(1, recall20=0.40, mrr20=0.10),
        _row(2, recall20=0.45, mrr20=0.12),
    ]
    summary = build_srgnn_validation_summary(
        rows,
        max_epochs=30,
        patience=10,
        stopped_epoch=2,
        stop_reason="patience_exhausted",
    )
    result = SRGNNValidationTrainingResult(rows=rows, summary=summary)

    extra = srgnn_validation_train_history_extra(result)

    assert extra["selected_checkpoint_epoch"] == 2
    assert extra["selected_checkpoint_metric"] == "valid_ground_truth_mrr@20"
    assert extra["best_valid_mrr20"] == pytest.approx(0.12)
    assert extra["best_valid_recall20_at_best_mrr20_epoch"] == pytest.approx(0.45)
    assert extra["stopped_epoch"] == 2
    assert extra["stop_reason"] == "patience_exhausted"
    assert extra["valid_ground_truth_mrr@20"] == [0.10, 0.12]
    assert len(extra["history_rows"]) == 2
