from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack.tools.invalidate_victim_cells import (
    InvalidationError,
    format_invalidation_result,
    invalidate_victim_cells,
)


def _cell(status: str) -> dict[str, object]:
    return {
        "status": status,
        "artifacts": {
            "metrics": "metrics.json",
            "predictions": "predictions.json",
            "train_history": "train_history.json",
            "poisoned_train": "poisoned_train.pkl",
        },
        "error": "old error",
        "first_requested_at": "old",
        "last_requested_at": "old",
        "last_started_at": "old",
        "last_execution_id": "execution-old",
        "attempt_count": 3,
        "completed_at": "old" if status == "completed" else None,
        "failed_at": "old" if status == "failed" else None,
        "last_updated_at": "old",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_run(run_dir: Path, *, tron_status: str = "completed") -> dict[str, object]:
    coverage = {
        "run_group_key": "run_group_test",
        "target_cohort_key": "target_cohort_test",
        "run_type": "attack",
        "targets_order": [101],
        "victims": {
            "srgnn": {"status": "completed", "victim_prediction_key": "srgnn-key"},
            "miasrec": {"status": "completed", "victim_prediction_key": "miasrec-key"},
            "tron": {"status": "completed", "victim_prediction_key": "old-tron-key"},
        },
        "cells": {
            "101": {
                "srgnn": _cell("completed"),
                "miasrec": _cell("completed"),
                "tron": _cell(tron_status),
            }
        },
        "materialized_target_prefix_count": 1,
        "created_at": "old",
        "updated_at": "old",
    }
    _write_json(run_dir / "run_coverage.json", coverage)
    for relative in (
        "targets/101/victims/srgnn/marker.txt",
        "targets/101/victims/miasrec/marker.txt",
        "targets/101/victims/tron/marker.txt",
        "targets/101/pts_construction_cem/marker.txt",
        "shared/victim_predictions/tron/marker.txt",
    ):
        marker = run_dir / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("keep-or-delete", encoding="utf-8")
    for name in (
        "summary_current.json",
        "artifact_manifest.json",
        "execution_log.json",
        "progress.json",
    ):
        _write_json(run_dir / name, {"marker": name, "tron": "stale"})
    return coverage


def _load_coverage(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run_coverage.json").read_text(encoding="utf-8"))


def test_dry_run_reports_changes_without_modifying_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_group_a"
    original_coverage = _create_run(run_dir)
    original_derived = {
        name: (run_dir / name).read_bytes()
        for name in (
            "summary_current.json",
            "artifact_manifest.json",
            "execution_log.json",
            "progress.json",
        )
    }

    [result] = invalidate_victim_cells(
        [run_dir],
        victim="tron",
        dry_run=True,
        allowed_roots=[tmp_path],
    )
    report = format_invalidation_result(result)

    assert _load_coverage(run_dir) == original_coverage
    assert (run_dir / "targets/101/victims/tron/marker.txt").exists()
    assert (run_dir / "targets/101/victims/srgnn/marker.txt").exists()
    assert (run_dir / "targets/101/victims/miasrec/marker.txt").exists()
    assert (run_dir / "targets/101/pts_construction_cem/marker.txt").exists()
    assert "completed_to_reset=1" in report
    assert "CEM caches, fake-session caches, SR-GNN/MiaSRec outputs" in report
    assert "may temporarily contain stale TRON entries" in report
    for name, content in original_derived.items():
        assert (run_dir / name).read_bytes() == content


def test_actual_invalidation_resets_only_tron_and_preserves_derived_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_group_a"
    _create_run(run_dir)
    original_derived = {
        name: (run_dir / name).read_bytes()
        for name in (
            "summary_current.json",
            "artifact_manifest.json",
            "execution_log.json",
            "progress.json",
        )
    }

    invalidate_victim_cells([run_dir], victim="tron", allowed_roots=[tmp_path])
    coverage = _load_coverage(run_dir)
    tron_cell = coverage["cells"]["101"]["tron"]

    assert tron_cell["status"] == "requested"
    assert tron_cell["attempt_count"] == 0
    assert tron_cell["artifacts"] == {
        "metrics": None,
        "poisoned_train": None,
        "predictions": None,
        "train_history": None,
    }
    assert tron_cell["error"] is None
    assert tron_cell["last_started_at"] is None
    assert tron_cell["last_execution_id"] is None
    assert tron_cell["completed_at"] is None
    assert tron_cell["failed_at"] is None
    assert coverage["cells"]["101"]["srgnn"]["status"] == "completed"
    assert coverage["cells"]["101"]["miasrec"]["status"] == "completed"
    assert coverage["victims"]["tron"]["status"] == "requested"
    assert "victim_prediction_key" not in coverage["victims"]["tron"]
    assert coverage["victims"]["tron"]["previous_victim_prediction_key"] == "old-tron-key"
    assert coverage["victims"]["srgnn"]["victim_prediction_key"] == "srgnn-key"
    assert coverage["victims"]["miasrec"]["victim_prediction_key"] == "miasrec-key"
    assert coverage["materialized_target_prefix_count"] == 0

    assert not (run_dir / "targets/101/victims/tron").exists()
    assert (run_dir / "targets/101/victims/srgnn/marker.txt").exists()
    assert (run_dir / "targets/101/victims/miasrec/marker.txt").exists()
    assert (run_dir / "targets/101/pts_construction_cem/marker.txt").exists()
    assert (run_dir / "shared/victim_predictions/tron/marker.txt").exists()
    for name, content in original_derived.items():
        assert (run_dir / name).read_bytes() == content


def test_actual_invalidation_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_group_a"
    _create_run(run_dir)

    invalidate_victim_cells([run_dir], victim="tron", allowed_roots=[tmp_path])
    invalidate_victim_cells([run_dir], victim="tron", allowed_roots=[tmp_path])
    coverage = _load_coverage(run_dir)

    assert coverage["cells"]["101"]["tron"]["status"] == "requested"
    assert coverage["victims"]["tron"]["previous_victim_prediction_key"] == "old-tron-key"
    assert "victim_prediction_key" not in coverage["victims"]["tron"]
    assert (run_dir / "targets/101/victims/srgnn/marker.txt").exists()
    assert (run_dir / "targets/101/pts_construction_cem/marker.txt").exists()


def test_multiple_run_dirs_are_processed_independently(tmp_path: Path) -> None:
    run_a = tmp_path / "run_group_a"
    run_b = tmp_path / "run_group_b"
    _create_run(run_a)
    _create_run(run_b, tron_status="failed")

    results = invalidate_victim_cells(
        [run_a, run_b],
        victim="tron",
        allowed_roots=[tmp_path],
    )

    assert len(results) == 2
    assert _load_coverage(run_a)["cells"]["101"]["tron"]["status"] == "requested"
    assert _load_coverage(run_b)["cells"]["101"]["tron"]["status"] == "requested"
    assert results[0]["completed_cell_count"] == 1
    assert results[1]["non_completed_cell_count"] == 1


def test_invalid_victim_and_unsafe_or_ambiguous_paths_fail_safely(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_group_a"
    _create_run(run_dir)

    with pytest.raises(InvalidationError, match="supports only 'tron'"):
        invalidate_victim_cells([run_dir], victim="srgnn", allowed_roots=[tmp_path])
    with pytest.raises(InvalidationError, match="outside the allowed"):
        invalidate_victim_cells(
            [run_dir],
            victim="tron",
            allowed_roots=[tmp_path / "different-root"],
        )

    ambiguous = tmp_path / "ambiguous"
    _write_json(
        ambiguous / "run_coverage.json",
        {
            "run_group_key": "run_group_test",
            "target_cohort_key": "target_cohort_test",
            "targets_order": [101],
            "victims": {"tron": {}},
            "cells": {},
            "created_at": "old",
            "updated_at": "old",
        },
    )
    with pytest.raises(InvalidationError, match="contains no target cells"):
        invalidate_victim_cells([ambiguous], victim="tron", allowed_roots=[tmp_path])

    unsafe_target = tmp_path / "unsafe-target"
    unsafe_coverage = _create_run(unsafe_target)
    unsafe_coverage["cells"]["../outside"] = unsafe_coverage["cells"].pop("101")
    _write_json(unsafe_target / "run_coverage.json", unsafe_coverage)
    with pytest.raises(InvalidationError, match="Unsafe target id"):
        invalidate_victim_cells([unsafe_target], victim="tron", allowed_roots=[tmp_path])
