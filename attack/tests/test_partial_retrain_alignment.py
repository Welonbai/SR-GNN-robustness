from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.tools.partial_retrain_alignment import (
    CSV_FIELDS,
    assert_random_nz_histogram_matches,
    build_comparison_table,
    first_aligned_epoch,
    first_stably_aligned_epoch,
    raw_lowk,
    write_outputs,
)


def _temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / "test_partial_retrain_alignment" / uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path


def _row(candidate_type: str, epoch: int, values: tuple[float, float, float, float]):
    mrr10, mrr20, recall10, recall20 = values
    metrics = {
        "targeted_mrr@10": mrr10,
        "targeted_mrr@20": mrr20,
        "targeted_recall@10": recall10,
        "targeted_recall@20": recall20,
    }
    return {
        "target_item": 14514,
        "candidate_type": candidate_type,
        "epoch": epoch,
        **metrics,
        "raw_lowk": raw_lowk(metrics),
        "train_time_seconds": float(epoch),
        "actual_train_epochs": epoch,
        "seed": 123,
        "notes": "test",
    }


def test_raw_lowk_averages_four_lowk_metrics() -> None:
    metrics = {
        "targeted_mrr@10": 0.1,
        "targeted_mrr@20": 0.2,
        "targeted_recall@10": 0.3,
        "targeted_recall@20": 0.4,
    }

    assert raw_lowk(metrics) == pytest.approx(0.25)


def test_comparison_table_records_raw_and_metric_winners() -> None:
    rows = [
        _row("cem_best", 1, (0.3, 0.3, 0.3, 0.3)),
        _row("random_nz", 1, (0.2, 0.2, 0.4, 0.4)),
    ]

    comparison = build_comparison_table(rows)

    assert comparison == [
        {
            "target_item": 14514,
            "epoch": 1,
            "cem_raw_lowk": pytest.approx(0.3),
            "random_raw_lowk": pytest.approx(0.3),
            "delta_raw_lowk": pytest.approx(0.0),
            "cem_wins_count_4": 2,
            "random_wins_count_4": 2,
            "winner_by_raw_lowk": "tie",
            "winner_by_metric_count": "tie",
        }
    ]


def test_alignment_epochs_handle_transient_flip() -> None:
    comparison = [
        {"epoch": 1, "winner_by_raw_lowk": "cem_best"},
        {"epoch": 2, "winner_by_raw_lowk": "random_nz"},
        {"epoch": 3, "winner_by_raw_lowk": "cem_best"},
        {"epoch": 4, "winner_by_raw_lowk": "random_nz"},
        {"epoch": 5, "winner_by_raw_lowk": "random_nz"},
    ]

    assert first_aligned_epoch(comparison, final_winner="random_nz") == 2
    assert first_stably_aligned_epoch(comparison, final_winner="random_nz") == 4


def test_random_nz_histogram_mismatch_fails_fast() -> None:
    temp_dir = _temp_dir()
    try:
        metadata_path = temp_dir / "random_nonzero_position_metadata.json"
        metadata_path.write_text(
            json.dumps({"counts": {"1": 2, "2": 1}, "ratios": {}, "total": 3}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="does not match existing metadata"):
            assert_random_nz_histogram_matches({"1": 3}, metadata_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_write_outputs_emits_expected_schema() -> None:
    temp_dir = _temp_dir()
    rows = [
        _row("cem_best", 1, (0.1, 0.2, 0.3, 0.4)),
        _row("random_nz", 1, (0.2, 0.3, 0.4, 0.5)),
    ]
    comparison = build_comparison_table(rows)
    final_context = {
        "cem_best": {
            "targeted_mrr@10": 0.1,
            "targeted_mrr@20": 0.2,
            "targeted_recall@10": 0.3,
            "targeted_recall@20": 0.4,
            "raw_lowk": 0.25,
        },
        "random_nz": {
            "targeted_mrr@10": 0.2,
            "targeted_mrr@20": 0.3,
            "targeted_recall@10": 0.4,
            "targeted_recall@20": 0.5,
            "raw_lowk": 0.35,
        },
        "winner_by_raw_lowk": "random_nz",
    }
    interpretation = {
        "final_winner_by_raw_lowk": "random_nz",
        "first_aligned_epoch": 1,
        "first_stably_aligned_epoch": 1,
        "diagnostic_result": "test interpretation",
    }
    metadata = {
        "seed": 123,
        "random_nz_histogram_match": True,
        "candidate_metadata": {
            "cem_best": {"clean_count": 10, "fake_session_count": 1},
            "random_nz": {"clean_count": 10, "fake_session_count": 1},
        },
    }

    try:
        paths = write_outputs(
            temp_dir,
            target_item=14514,
            rows=rows,
            comparison_rows=comparison,
            final_context=final_context,
            interpretation=interpretation,
            metadata=metadata,
        )

        assert set(paths) == {"csv", "json", "markdown"}
        with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(CSV_FIELDS)
            assert len(list(reader)) == 2
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert payload["comparison_by_epoch"][0]["winner_by_raw_lowk"] == "random_nz"
        assert "Final SR-GNN Victim" in paths["markdown"].read_text(encoding="utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
