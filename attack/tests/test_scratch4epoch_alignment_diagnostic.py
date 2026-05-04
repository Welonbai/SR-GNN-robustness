from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from attack.tools.run_scratch4epoch_surrogate_alignment_diagnostic import (
    _assert_random_position_replay_matches,
    _resolve_cem_trace_path,
    _validate_cached_random_surrogate_replay,
    _victim_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / ".pytest_scratch4epoch_alignment_diagnostic" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_random_position_replay_check_matches_counts_and_fails_on_mismatch() -> None:
    temp_dir = _repo_temp_dir()
    try:
        metadata_path = temp_dir / "random_nonzero_position_metadata.json"
        _write_json(
            metadata_path,
            {
                "total": 4,
                "counts": {"1": 2, "2": 1, "3": 1},
            },
        )

        check = _assert_random_position_replay_matches(
            [1, 2, 1, 3],
            random_position_metadata_path=metadata_path,
        )

        assert check["checks_passed"] is True
        assert check["expected_counts"] == {"1": 2, "2": 1, "3": 1}
        assert check["replay_counts"] == {"1": 2, "2": 1, "3": 1}

        with pytest.raises(RuntimeError, match="does not match"):
            _assert_random_position_replay_matches(
                [1, 1, 1, 3],
                random_position_metadata_path=metadata_path,
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_victim_payload_accepts_top_level_metric_payload() -> None:
    temp_dir = _repo_temp_dir()
    try:
        metrics_path = temp_dir / "targets" / "11103" / "victims" / "srgnn" / "metrics.json"
        _write_json(
            metrics_path,
            {
                "run_type": "random_nonzero_when_possible",
                "target_item": 11103,
                "targeted_mrr@10": 0.1,
                "targeted_mrr@20": 0.2,
                "targeted_recall@10": 0.3,
                "targeted_recall@20": 0.4,
                "ground_truth_mrr@20": 0.5,
                "ground_truth_recall@20": 0.6,
            },
        )

        payload = _victim_payload(metrics_path)

        assert payload["raw_lowk"] == pytest.approx(0.25)
        assert payload["ground_truth_mrr@20"] == pytest.approx(0.5)
        assert payload["ground_truth_recall@20"] == pytest.approx(0.6)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cem_trace_path_falls_back_from_metrics_path() -> None:
    temp_dir = _repo_temp_dir()
    try:
        metrics_path = temp_dir / "targets" / "11103" / "victims" / "srgnn" / "metrics.json"
        trace_path = temp_dir / "targets" / "11103" / "position_opt" / "cem" / "cem_trace.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")

        resolved = _resolve_cem_trace_path(
            None,
            {},
            cem_metrics_path=metrics_path,
        )

        assert resolved == trace_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_cached_random_surrogate_requires_successful_replay_check_and_matching_seed(
) -> None:
    temp_dir = _repo_temp_dir()
    try:
        metadata_path = temp_dir / "random_nonzero_position_metadata.json"
        metadata_path.write_text("{}", encoding="utf-8")
        payload = {
            "surrogate_train_seed": 123,
            "random_position_replay_check": {
                "checks_passed": True,
                "metadata_path": str(metadata_path),
            },
        }

        _validate_cached_random_surrogate_replay(
            payload,
            expected_surrogate_train_seed=123,
            expected_random_position_metadata_path=metadata_path,
        )

        with pytest.raises(RuntimeError, match="different surrogate_train_seed"):
            _validate_cached_random_surrogate_replay(
                payload,
                expected_surrogate_train_seed=456,
                expected_random_position_metadata_path=metadata_path,
            )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
