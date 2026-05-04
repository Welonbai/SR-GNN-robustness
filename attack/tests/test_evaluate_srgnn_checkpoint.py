from __future__ import annotations

import pytest

from attack.tools.evaluate_srgnn_checkpoint import _raw_lowk


def test_raw_lowk_averages_lowk_target_metrics() -> None:
    assert _raw_lowk(
        {
            "targeted_mrr@10": 0.1,
            "targeted_mrr@20": 0.2,
            "targeted_recall@10": 0.3,
            "targeted_recall@20": 0.4,
        }
    ) == pytest.approx(0.25)


def test_raw_lowk_returns_none_for_missing_or_null_metrics() -> None:
    assert _raw_lowk({"targeted_mrr@10": 0.1}) is None
    assert _raw_lowk(
        {
            "targeted_mrr@10": 0.1,
            "targeted_mrr@20": 0.2,
            "targeted_recall@10": 0.3,
            "targeted_recall@20": None,
        }
    ) is None
