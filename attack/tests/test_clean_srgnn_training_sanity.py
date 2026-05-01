from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.tools.clean_srgnn_training_sanity import (
    CHECKPOINT_IMPROVEMENT_RULE,
    IS_BEST_MRR20_NOTE,
    OFFICIAL_EARLY_STOPPING_NOTE,
    PATIENCE_IMPROVEMENT_RULE,
    TEST_METRICS_NOTE,
    VALIDATION_PROTOCOL,
    build_summary,
    epoch8_status,
    improved_over_best,
    official_patience_improved,
    render_markdown_report,
)


def _row(epoch: int, *, mrr20: float, recall20: float) -> dict[str, object]:
    return {
        "epoch": int(epoch),
        "train_loss": 1.0 / float(epoch),
        "valid_ground_truth_recall@10": recall20 / 2.0,
        "valid_ground_truth_recall@20": recall20,
        "valid_ground_truth_mrr@10": mrr20 / 2.0,
        "valid_ground_truth_mrr@20": mrr20,
        "test_ground_truth_recall@10": recall20 / 3.0,
        "test_ground_truth_recall@20": recall20 / 1.5,
        "test_ground_truth_mrr@10": mrr20 / 3.0,
        "test_ground_truth_mrr@20": mrr20 / 1.5,
        "improved_valid_mrr20_this_epoch": False,
        "improved_valid_recall20_this_epoch": False,
        "improved_valid_recall_or_mrr20_this_epoch": False,
        "is_best_mrr20": False,
        "bad_counter": 0,
        "lr": 0.001,
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return build_summary(
        rows,
        config_path="attack/configs/example.yaml",
        output_dir="outputs/diagnostics/example",
        max_epochs=30,
        patience=10,
        train_config_source="poison_model",
        seed=123,
        data_sources={
            "train_source": "canonical_dataset.train_sub",
            "srgnn_train_pair_source": "in_memory_pairs_from_train_sub_not_train_plus_valid",
        },
        checkpoint_paths={
            "best_validation": "best_validation.pt",
            "epoch_008": "epoch_008.pt",
            "final": "final.pt",
        },
        stopped_early=False,
        stopped_epoch=None,
    )


def test_improvement_rule_is_strict_greater_than() -> None:
    assert improved_over_best(0.101, 0.1) is True
    assert improved_over_best(0.1, 0.1) is False
    assert improved_over_best(0.099, 0.1) is False


def test_official_patience_resets_on_recall_or_mrr20_improvement() -> None:
    assert official_patience_improved(
        current_valid_recall20=0.30,
        best_valid_recall20=0.29,
        current_valid_mrr20=0.10,
        best_valid_mrr20=0.11,
    ) is True
    assert official_patience_improved(
        current_valid_recall20=0.29,
        best_valid_recall20=0.30,
        current_valid_mrr20=0.11,
        best_valid_mrr20=0.10,
    ) is True
    assert official_patience_improved(
        current_valid_recall20=0.30,
        best_valid_recall20=0.30,
        current_valid_mrr20=0.10,
        best_valid_mrr20=0.10,
    ) is False


@pytest.mark.parametrize(
    ("best_epoch", "expected"),
    [
        (9, "before_best"),
        (8, "at_best"),
        (7, "after_best"),
    ],
)
def test_epoch8_status_compares_to_best_mrr20_epoch(best_epoch: int, expected: str) -> None:
    assert epoch8_status(8, best_epoch) == expected


def test_build_summary_reports_epoch8_before_best_and_metric_deltas() -> None:
    rows = [
        _row(epoch, mrr20=0.10, recall20=0.20)
        for epoch in range(1, 8)
    ]
    rows.append(_row(8, mrr20=0.100, recall20=0.200))
    rows.append(_row(9, mrr20=0.110, recall20=0.205))
    rows.append(_row(10, mrr20=0.105, recall20=0.230))

    summary = _summary(rows)

    assert summary["best_valid_mrr20_epoch"] == 9
    assert summary["best_valid_recall20_epoch"] == 10
    assert summary["epoch8_status"] == "before_best"
    assert summary["epoch8_mrr20"] == pytest.approx(0.100)
    assert summary["epoch8_recall20"] == pytest.approx(0.200)
    assert summary["epoch8_to_best_mrr20_abs_delta"] == pytest.approx(0.010)
    assert summary["epoch8_to_best_mrr20_rel_delta"] == pytest.approx(0.010 / 0.110)
    assert summary["epoch8_to_best_recall20_abs_delta"] == pytest.approx(0.030)
    assert summary["epoch8_to_best_recall20_rel_delta"] == pytest.approx(0.030 / 0.230)
    assert summary["epoch8_close_to_best_mrr20"] is False
    assert summary["epoch8_recall20_close_to_best"] is False
    assert summary["validation_protocol"] == VALIDATION_PROTOCOL
    assert summary["patience_improvement_rule"] == PATIENCE_IMPROVEMENT_RULE
    assert summary["checkpoint_improvement_rule"] == CHECKPOINT_IMPROVEMENT_RULE
    assert summary["official_early_stopping_note"] == OFFICIAL_EARLY_STOPPING_NOTE
    assert summary["is_best_mrr20_note"] == IS_BEST_MRR20_NOTE
    assert summary["test_metrics_note"] == TEST_METRICS_NOTE


def test_build_summary_marks_epoch8_close_when_within_one_percent() -> None:
    rows = [
        _row(epoch, mrr20=0.09, recall20=0.19)
        for epoch in range(1, 8)
    ]
    rows.append(_row(8, mrr20=0.0995, recall20=0.199))
    rows.append(_row(9, mrr20=0.1000, recall20=0.200))

    summary = _summary(rows)

    assert summary["epoch8_status"] == "before_best"
    assert summary["epoch8_close_to_best_mrr20"] is True
    assert summary["epoch8_recall20_close_to_best"] is True


def test_report_mentions_epoch8_status_and_test_diagnostic_note() -> None:
    rows = [
        _row(epoch, mrr20=0.10, recall20=0.20)
        for epoch in range(1, 9)
    ]
    rows[7]["is_best_mrr20"] = True
    rows[7]["improved_valid_mrr20_this_epoch"] = True
    rows[7]["improved_valid_recall_or_mrr20_this_epoch"] = True
    summary = _summary(rows)

    report = render_markdown_report(summary, rows)

    assert "epoch8_status" in report
    assert VALIDATION_PROTOCOL in report
    assert OFFICIAL_EARLY_STOPPING_NOTE in report
    assert IS_BEST_MRR20_NOTE in report
    assert "improved MRR@20" in report
    assert "patience reset" in report
    assert TEST_METRICS_NOTE in report
    assert "canonical_dataset.train_sub" in report
