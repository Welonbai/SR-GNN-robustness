from __future__ import annotations

import csv
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from analysis.utils.position_collapse_summary import (
    PositionCollapseSummaryError,
    build_position_collapse_summary,
    main,
    resolve_training_history_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_position_collapse_summary" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _training_history_payload() -> dict[str, object]:
    return {
        "target_item": 11103,
        "policy_representation": "shared_contextual_mlp",
        "training_history": [
            {
                "outer_step": 0,
                "mean_entropy": 1.2,
                "reward": 0.01,
                "baseline": None,
                "advantage": 0.01,
                "policy_loss": 0.5,
                "target_utility": 0.001,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 1, 1, 2],
            },
            {
                "outer_step": 1,
                "mean_entropy": 1.0,
                "reward": 0.02,
                "baseline": 0.01,
                "advantage": 0.01,
                "policy_loss": 0.4,
                "target_utility": 0.002,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 0, 1, 2],
            },
            {
                "outer_step": 2,
                "mean_entropy": 0.8,
                "reward": 0.03,
                "baseline": 0.02,
                "advantage": 0.01,
                "policy_loss": 0.3,
                "target_utility": 0.003,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 0, 0, 2],
            },
            {
                "outer_step": 3,
                "mean_entropy": 0.4,
                "reward": 0.04,
                "baseline": 0.03,
                "advantage": 0.01,
                "policy_loss": 0.2,
                "target_utility": 0.004,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 0, 0, 0],
            },
        ],
        "final_selected_positions": [
            {"candidate_index": 0, "position": 0, "score": 1.0},
            {"candidate_index": 0, "position": 0, "score": 2.0},
            {"candidate_index": 0, "position": 0, "score": 3.0},
            {"candidate_index": 1, "position": 2, "score": 4.0},
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_build_position_collapse_summary_computes_entropy_and_thresholds() -> None:
    summary = build_position_collapse_summary(_training_history_payload())

    assert summary["target_item"] == 11103
    assert summary["policy_representation"] == "shared_contextual_mlp"
    assert summary["steps"] == 4
    assert summary["sample_count"] == 4
    assert summary["initial_entropy"] == 1.2
    assert summary["final_entropy"] == 0.4
    assert summary["entropy_drop"] == pytest.approx(0.8)
    assert summary["entropy_drop_pct"] == pytest.approx(66.6666666667)

    assert summary["initial_sampled_dominant"]["dominant_position"] == 1
    assert summary["initial_sampled_dominant"]["dominant_share_pct"] == 50.0
    assert summary["final_sampled_dominant"]["dominant_position"] == 0
    assert summary["final_sampled_dominant"]["dominant_share_pct"] == 100.0
    assert summary["share_threshold_crossings"][">=50%"]["outer_step"] == 0
    assert summary["share_threshold_crossings"][">=70%"]["outer_step"] == 2
    assert summary["share_threshold_crossings"][">=90%"]["outer_step"] == 3

    rows = summary["step_summaries"]
    assert rows[2]["dominant_position"] == 0
    assert rows[2]["dominant_count"] == 3
    assert rows[2]["dominant_share_pct"] == 75.0
    assert rows[2]["pos0_share_pct"] == 75.0
    assert rows[2]["average_selected_position"] == pytest.approx(0.5)
    assert rows[2]["median_selected_position"] == 0.0
    assert rows[2]["fraction_pos0"] == 0.75
    assert rows[2]["fraction_pos_le_1"] == 0.75
    assert rows[2]["fraction_pos_le_2"] == 1.0
    assert rows[2]["baseline"] == 0.02
    assert rows[2]["advantage"] == 0.01
    assert rows[2]["target_utility"] == 0.003
    assert rows[2]["gt_drop"] == 0.0
    assert rows[2]["gt_penalty"] == 0.0
    assert rows[2]["unique_positions"] == 2
    assert rows[2]["entropy_drop_from_initial"] == pytest.approx(0.4)


def test_build_position_collapse_summary_includes_final_argmax_distribution() -> None:
    summary = build_position_collapse_summary(_training_history_payload())

    final_distribution = summary["final_argmax_distribution"]
    assert final_distribution["total"] == 4
    assert final_distribution["unique_positions"] == 2
    assert final_distribution["dominant"]["position"] == 0
    assert final_distribution["dominant"]["share_pct"] == 75.0
    assert final_distribution["positions"][0] == {"position": 0, "count": 3, "share_pct": 75.0}


def test_resolve_training_history_path_supports_run_root_and_target_item() -> None:
    with _temp_test_dir() as temp_dir:
        history_path = (
            temp_dir
            / "run_group_test"
            / "targets"
            / "11103"
            / "position_opt"
            / "training_history.json"
        )
        _write_json(history_path, _training_history_payload())

        resolved = resolve_training_history_path(run_root=temp_dir / "run_group_test", target_item=11103)

        assert resolved == history_path


def test_missing_training_history_raises_clear_error() -> None:
    payload = _training_history_payload()
    payload.pop("training_history")

    with pytest.raises(PositionCollapseSummaryError, match="training_history"):
        build_position_collapse_summary(payload)


def test_empty_selected_positions_raises_clear_error() -> None:
    payload = _training_history_payload()
    payload["training_history"][0]["selected_positions"] = []

    with pytest.raises(PositionCollapseSummaryError, match="selected_positions"):
        build_position_collapse_summary(payload)


def test_invalid_threshold_raises_clear_error() -> None:
    with pytest.raises(PositionCollapseSummaryError, match="threshold_pct"):
        build_position_collapse_summary(_training_history_payload(), share_thresholds_pct=(0.0,))


def test_cli_smoke_writes_console_json_and_csv(capsys: pytest.CaptureFixture[str]) -> None:
    with _temp_test_dir() as temp_dir:
        history_path = temp_dir / "training_history.json"
        output_json = temp_dir / "summary.json"
        output_csv = temp_dir / "steps.csv"
        _write_json(history_path, _training_history_payload())

        exit_code = main(
            [
                "--training-history",
                str(history_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Position Collapse Summary" in captured.out
        assert "target_item: 11103" in captured.out

        summary = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["steps"] == 4
        assert summary["share_threshold_crossings"][">=70%"]["outer_step"] == 2

        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["outer_step"] == "0"
        assert rows[0]["baseline"] == ""
        assert float(rows[0]["dominant_share_pct"]) == 50.0
        assert float(rows[0]["average_selected_position"]) == 1.0
        assert float(rows[0]["median_selected_position"]) == 1.0
        assert float(rows[0]["fraction_pos0"]) == 0.25
        assert float(rows[0]["fraction_pos_le_1"]) == 0.75
        assert float(rows[0]["fraction_pos_le_2"]) == 1.0
        assert float(rows[3]["entropy_drop_from_initial"]) == pytest.approx(0.8)
