from __future__ import annotations

import csv
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from analysis.utils.position_stats_summary import (
    PositionStatsSummaryError,
    build_position_stats_summary,
    main,
    resolve_position_stats_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_position_stats_summary" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _position_stats_payload() -> dict[str, object]:
    return {
        "run_type": "position_opt_shared_policy",
        "target_item": 11103,
        "total_sessions": 10,
        "overall": {
            "counts": {"0": 7, "1": 2, "3": 1},
            "ratios": {"0": 0.7, "1": 0.2, "3": 0.1},
        },
        "by_session_length": {
            "5": {
                "session_count": 6,
                "position_counts": {"0": 5, "1": 1},
                "position_ratios": {"0": 5 / 6, "1": 1 / 6},
            },
            "7": {
                "session_count": 4,
                "position_counts": {"0": 2, "1": 1, "3": 1},
                "position_ratios": {"0": 0.5, "1": 0.25, "3": 0.25},
            },
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_build_position_stats_summary_computes_distribution_and_cumulative_ratios() -> None:
    summary = build_position_stats_summary(_position_stats_payload())

    assert summary["run_type"] == "position_opt_shared_policy"
    assert summary["target_item"] == 11103
    assert summary["total_sessions"] == 10
    assert summary["unique_selected_position_count"] == 3
    assert summary["positions"] == [
        {"position": 0, "count": 7, "ratio": 0.7, "ratio_pct": 70.0},
        {"position": 1, "count": 2, "ratio": 0.2, "ratio_pct": 20.0},
        {"position": 3, "count": 1, "ratio": 0.1, "ratio_pct": 10.0},
    ]
    assert summary["cumulative_ratios"]["<=0"]["ratio_pct"] == 70.0
    assert summary["cumulative_ratios"]["<=1"]["ratio_pct"] == 90.0
    assert summary["cumulative_ratios"]["<=2"]["ratio_pct"] == 90.0
    assert summary["cumulative_ratios"]["<=5"]["ratio_pct"] == 100.0
    assert summary["max_position_share"]["position"] == 0
    assert summary["top_positions"][0]["count"] == 7


def test_build_position_stats_summary_includes_session_length_summary() -> None:
    summary = build_position_stats_summary(
        _position_stats_payload(),
        include_session_lengths=True,
    )

    assert summary["by_session_length"][0]["session_length"] == 5
    assert summary["by_session_length"][0]["dominant_position"] == 0
    assert summary["by_session_length"][0]["dominant_ratio_pct"] == pytest.approx(83.3333333333)
    assert summary["by_session_length"][1]["session_length"] == 7
    assert summary["by_session_length"][1]["unique_selected_position_count"] == 3


def test_resolve_position_stats_path_supports_run_root_and_target_item() -> None:
    with _temp_test_dir() as temp_dir:
        stats_path = temp_dir / "run_group_test" / "targets" / "11103" / "position_stats.json"
        _write_json(stats_path, _position_stats_payload())

        resolved = resolve_position_stats_path(run_root=temp_dir / "run_group_test", target_item=11103)

        assert resolved == stats_path


def test_missing_required_schema_raises_clear_error() -> None:
    payload = _position_stats_payload()
    payload.pop("overall")

    with pytest.raises(PositionStatsSummaryError, match="overall"):
        build_position_stats_summary(payload)


def test_zero_total_sessions_raises_clear_error() -> None:
    payload = _position_stats_payload()
    payload["total_sessions"] = 0

    with pytest.raises(PositionStatsSummaryError, match="total_sessions must be positive"):
        build_position_stats_summary(payload)


def test_cli_smoke_writes_console_json_and_csv(capsys: pytest.CaptureFixture[str]) -> None:
    with _temp_test_dir() as temp_dir:
        stats_path = temp_dir / "position_stats.json"
        output_json = temp_dir / "summary.json"
        output_csv = temp_dir / "positions.csv"
        _write_json(stats_path, _position_stats_payload())

        exit_code = main(
            [
                "--position-stats",
                str(stats_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--include-session-lengths",
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Position Stats Summary" in captured.out
        assert "target_item: 11103" in captured.out

        summary = json.loads(output_json.read_text(encoding="utf-8"))
        assert summary["total_sessions"] == 10
        assert summary["by_session_length"][0]["session_length"] == 5

        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["position"] == "0"
        assert rows[0]["count"] == "7"
        assert float(rows[0]["cumulative_ratio_pct"]) == 70.0
