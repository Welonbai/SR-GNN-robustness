from __future__ import annotations

import pytest

from attack.tools.assemble_tailboosted_cem_report import (
    _absolute_position_band_summary,
    _candidate_trace_summary,
)


def _trace_row(
    *,
    iteration: int,
    candidate_id: int,
    global_id: int,
    raw_lowk: float,
    normalized_reward: float,
    elite: bool = False,
    global_best: bool = False,
    pos1_pct: float = 0.0,
    tail_pct: float = 0.0,
) -> dict[str, object]:
    return {
        "target_item": 11103,
        "iteration": iteration,
        "candidate_id_in_iteration": candidate_id,
        "global_candidate_id": global_id,
        "raw_lowk": raw_lowk,
        "normalized_reward": normalized_reward,
        "iteration_normalized_lowk_reward": normalized_reward,
        "global_normalized_lowk_reward": normalized_reward,
        "selected_as_iteration_elite": elite,
        "selected_as_global_best": global_best,
        "selected_checkpoint_epoch": 4,
        "valid_ground_truth_mrr@20": 0.16,
        "valid_ground_truth_recall@20": 0.48,
        "targeted_mrr@10": 0.05,
        "targeted_mrr@20": 0.06,
        "targeted_recall@10": 0.15,
        "targeted_recall@20": 0.25,
        "position_summary": {
            "pos1_pct": pos1_pct,
            "tail_pct": tail_pct,
            "unique_selected_position_count": 8,
        },
    }


def test_absolute_position_band_summary_groups_positions() -> None:
    summary = {
        "total": 5,
        "counts": {
            "1": 2,
            "2": 1,
            "4": 1,
            "6": 1,
        },
    }

    bands = _absolute_position_band_summary(summary)

    assert bands["total"] == 5
    assert bands["pos1_pct"] == pytest.approx(40.0)
    assert bands["pos2_pct"] == pytest.approx(20.0)
    assert bands["pos3_pct"] == pytest.approx(0.0)
    assert bands["pos4_pos5_pct"] == pytest.approx(20.0)
    assert bands["pos6plus_pct"] == pytest.approx(20.0)


def test_candidate_trace_summary_keeps_best_and_position_summaries() -> None:
    rows = [
        _trace_row(
            iteration=0,
            candidate_id=0,
            global_id=0,
            raw_lowk=0.10,
            normalized_reward=0.40,
            pos1_pct=20.0,
            tail_pct=70.0,
        ),
        _trace_row(
            iteration=0,
            candidate_id=1,
            global_id=1,
            raw_lowk=0.12,
            normalized_reward=0.90,
            elite=True,
            pos1_pct=30.0,
            tail_pct=60.0,
        ),
        _trace_row(
            iteration=1,
            candidate_id=0,
            global_id=2,
            raw_lowk=0.11,
            normalized_reward=1.00,
            elite=True,
            global_best=True,
            pos1_pct=80.0,
            tail_pct=5.0,
        ),
    ]

    summary = _candidate_trace_summary(rows)

    assert summary["candidate_count"] == 3
    assert summary["global_best"]["global_candidate_id"] == 2
    assert summary["global_best"]["position_summary"]["pos1_pct"] == pytest.approx(80.0)
    assert [row["iteration"] for row in summary["per_iteration"]] == [0, 1]
    assert summary["per_iteration"][0]["elite_global_candidate_ids"] == [1]
    assert summary["per_iteration"][0]["best_by_normalized_reward"]["global_candidate_id"] == 1
    assert summary["top_by_raw_lowk"][0]["global_candidate_id"] == 1
    assert summary["top_by_global_normalized_lowk"][0]["global_candidate_id"] == 2
