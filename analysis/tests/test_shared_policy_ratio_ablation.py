from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.diagnosis.shared_policy_ratio_ablation.run import (
    _assess_exclusion_usefulness,
    _collect_metric_deltas,
    _metric_distance,
    _summarize_metric_delta,
)


def test_collect_metric_deltas_and_metric_distance_ignore_missing_values() -> None:
    left = {
        "target_recall@10": 0.20,
        "target_recall@20": 0.30,
        "target_recall@30": None,
        "target_mrr@10": 0.10,
    }
    right = {
        "target_recall@10": 0.10,
        "target_recall@20": 0.40,
        "target_recall@30": 0.50,
        "target_mrr@10": 0.10,
    }

    deltas = _collect_metric_deltas(
        left,
        right,
        ("target_recall@10", "target_recall@20", "target_recall@30", "target_mrr@10"),
    )

    assert deltas == pytest.approx([0.10, -0.10, 0.0])
    assert _metric_distance(left, right) == pytest.approx(0.20)


def test_summarize_metric_delta_and_exclusion_usefulness_follow_expected_heuristics() -> None:
    ratio05 = {
        "target_recall@10": 0.20,
        "target_recall@20": 0.30,
        "target_recall@30": 0.40,
        "target_mrr@10": 0.10,
        "target_mrr@20": 0.12,
        "target_mrr@30": 0.14,
    }
    ratio1 = {
        "target_recall@10": 0.10,
        "target_recall@20": 0.20,
        "target_recall@30": 0.30,
        "target_mrr@10": 0.05,
        "target_mrr@20": 0.06,
        "target_mrr@30": 0.07,
    }
    worse_ratio05 = {
        "target_recall@10": 0.05,
        "target_recall@20": 0.10,
        "target_recall@30": 0.15,
        "target_mrr@10": 0.02,
        "target_mrr@20": 0.03,
        "target_mrr@30": 0.04,
    }

    assert _summarize_metric_delta(
        ratio05,
        ratio1,
        (
            "target_recall@10",
            "target_recall@20",
            "target_recall@30",
            "target_mrr@10",
            "target_mrr@20",
            "target_mrr@30",
        ),
    ).startswith("Improved")
    assert _assess_exclusion_usefulness(
        excluded_rate=0.0,
        ratio05_metrics=ratio05,
        ratio1_metrics=ratio1,
    ) == "No."
    assert _assess_exclusion_usefulness(
        excluded_rate=0.25,
        ratio05_metrics=worse_ratio05,
        ratio1_metrics=ratio1,
    ) == "Likely yes."
