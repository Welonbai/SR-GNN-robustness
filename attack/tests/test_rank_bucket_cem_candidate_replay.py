from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from attack.pipeline.runs.run_rank_bucket_cem_candidate_replay import (
    _candidate_selected_positions,
    _load_cem_candidate_row,
    _sha1_json,
    _validate_selected_positions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / ".pytest_rank_bucket_cem_candidate_replay" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def test_load_cem_candidate_row_finds_unique_target_candidate() -> None:
    temp_dir = _repo_temp_dir()
    try:
        trace_path = temp_dir / "cem_trace.jsonl"
        _write_jsonl(
            trace_path,
            [
                {
                    "target_item": 11103,
                    "global_candidate_id": 7,
                    "selected_positions": [1, 2],
                },
                {
                    "target_item": 11103,
                    "global_candidate_id": 8,
                    "selected_positions": [2, 1],
                },
            ],
        )

        row = _load_cem_candidate_row(
            trace_path,
            target_item=11103,
            global_candidate_id=8,
        )

        assert row["selected_positions"] == [2, 1]
        assert row["_source_trace_line_number"] == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_candidate_selected_positions_requires_saved_positions() -> None:
    assert _candidate_selected_positions({"selected_positions": ["1", 2]}) == [1, 2]
    with pytest.raises(ValueError, match="selected_positions"):
        _candidate_selected_positions({"global_candidate_id": 8})


def test_validate_selected_positions_rejects_count_and_bounds_mismatch() -> None:
    _validate_selected_positions(
        [1, 0],
        template_sessions=[[10, 11], [12, 13, 14]],
        target_item=11103,
        source_trace_path=Path("cem_trace.jsonl"),
        global_candidate_id=8,
    )

    with pytest.raises(ValueError, match="length"):
        _validate_selected_positions(
            [1],
            template_sessions=[[10, 11], [12, 13, 14]],
            target_item=11103,
            source_trace_path=Path("cem_trace.jsonl"),
            global_candidate_id=8,
        )

    with pytest.raises(ValueError, match="invalid positions"):
        _validate_selected_positions(
            [2, 0],
            template_sessions=[[10, 11], [12, 13, 14]],
            target_item=11103,
            source_trace_path=Path("cem_trace.jsonl"),
            global_candidate_id=8,
        )


def test_sha1_json_is_stable_for_selected_positions() -> None:
    assert _sha1_json([1, 2, 3]) == _sha1_json([1, 2, 3])
    assert _sha1_json([1, 2, 3]) != _sha1_json([1, 3, 2])
