from __future__ import annotations

import json
from pathlib import Path

import pytest

import attack.tools.invalidate_victim_cells as invalidation_module
from attack.tools.invalidate_victim_cells import (
    InvalidationError,
    inspect_invalidation_plan,
    invalidate_victim_cells,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _create_run(
    tmp_path: Path,
    *,
    victim: str = "wearec",
    clean: bool = True,
    targets: tuple[str, ...] = ("101", "202"),
) -> tuple[Path, list[Path]]:
    run_dir = tmp_path / "runs" / "run_group"
    cells = {}
    manifest_victims = {}
    shared_paths: list[Path] = []
    clean_shared = (
        tmp_path
        / "shared"
        / "victim_predictions"
        / victim
        / "scientific-key"
        / "shared"
    )
    for target in targets:
        cells[target] = {
            victim: {"status": "completed"},
            "mdhg": {"status": "completed"},
        }
        local = run_dir / "targets" / target / "victims" / victim
        local.mkdir(parents=True, exist_ok=True)
        (local / "marker.txt").write_text("local", encoding="utf-8")
        shared = (
            clean_shared
            if clean
            else (
                tmp_path
                / "shared"
                / "victim_predictions"
                / victim
                / "scientific-key"
                / "targets"
                / target
            )
        )
        shared.mkdir(parents=True, exist_ok=True)
        (shared / "marker.txt").write_text("shared", encoding="utf-8")
        shared_paths.append(shared)
        manifest_victims[target] = {
            victim: {"shared": {"shared_dir": str(shared)}},
            "mdhg": {
                "shared": {
                    "shared_dir": str(
                        tmp_path / "shared" / "victim_predictions" / "mdhg" / target
                    )
                }
            },
        }
    coverage = {
        "run_group_key": "run-group",
        "target_cohort_key": "cohort",
        "targets_order": [int(target) for target in targets],
        "victims": {
            victim: {
                "status": "completed",
                "victim_prediction_key": "old-key",
            },
            "mdhg": {"status": "completed"},
        },
        "cells": cells,
        "materialized_target_prefix_count": len(targets),
        "created_at": "old",
        "updated_at": "old",
    }
    _write_json(run_dir / "run_coverage.json", coverage)
    _write_json(run_dir / "artifact_manifest.json", {"victims": manifest_victims})
    other_victim = tmp_path / "shared" / "victim_predictions" / "mdhg" / "keep"
    attack = run_dir / "targets" / targets[0] / "pts_construction_cem"
    for path in (other_victim, attack):
        path.mkdir(parents=True, exist_ok=True)
        (path / "marker.txt").write_text("keep", encoding="utf-8")
    return run_dir, shared_paths


def test_exact_clean_wearec_layout_is_accepted_and_deduplicated(tmp_path):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    plan = inspect_invalidation_plan(
        run_dir, victim="wearec", allowed_roots=[tmp_path]
    )
    assert len(plan.artifact_dirs) == 2
    assert plan.shared_artifact_dirs == (shared_paths[0],)


def test_clean_wearec_shared_directory_is_deleted_exactly_once(
    tmp_path, monkeypatch
):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    calls = []
    real_rmtree = invalidation_module.shutil.rmtree

    def recording_rmtree(path):
        calls.append(Path(path))
        real_rmtree(path)

    monkeypatch.setattr(invalidation_module.shutil, "rmtree", recording_rmtree)
    invalidate_victim_cells(
        [run_dir], victim="wearec", allowed_roots=[tmp_path]
    )
    assert calls.count(shared_paths[0]) == 1


def test_exact_poisoned_wearec_layout_is_accepted_and_deleted(tmp_path):
    run_dir, shared_paths = _create_run(tmp_path, clean=False)
    [result] = invalidate_victim_cells(
        [run_dir], victim="wearec", allowed_roots=[tmp_path]
    )
    assert set(result["shared_artifact_dirs_to_delete"]) == {
        str(path) for path in shared_paths
    }
    assert all(not path.exists() for path in shared_paths)


@pytest.mark.parametrize("mutation", ["missing_manifest", "missing_shared_dir"])
def test_manifest_failures_leave_everything_untouched(tmp_path, mutation):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    local = run_dir / "targets" / "101" / "victims" / "wearec"
    if mutation == "missing_manifest":
        (run_dir / "artifact_manifest.json").unlink()
    else:
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
        del manifest["victims"]["101"]["wearec"]["shared"]["shared_dir"]
        _write_json(run_dir / "artifact_manifest.json", manifest)
    with pytest.raises(InvalidationError):
        invalidate_victim_cells(
            [run_dir], victim="wearec", allowed_roots=[tmp_path]
        )
    assert local.is_dir()
    assert shared_paths[0].is_dir()


def test_wearec_rejects_freqrec_shared_path(tmp_path):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    wrong = (
        tmp_path / "shared" / "victim_predictions" / "freqrec" / "key" / "shared"
    )
    wrong.mkdir(parents=True)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    manifest["victims"]["101"]["wearec"]["shared"]["shared_dir"] = str(wrong)
    _write_json(run_dir / "artifact_manifest.json", manifest)
    with pytest.raises(InvalidationError, match="unsafe wearec"):
        inspect_invalidation_plan(
            run_dir, victim="wearec", allowed_roots=[tmp_path]
        )
    assert shared_paths[0].is_dir()
    assert wrong.is_dir()


@pytest.mark.parametrize(
    "relative",
    [
        ("victim_predictions", "freqrec", "key", "wearec", "shared"),
        ("victim_predictions", "wearec", "key", "not-targets", "101"),
        ("victim_predictions", "wearec", "key", "targets", "202"),
        ("victim_predictions", "wearec", "shared"),
    ],
    ids=[
        "victim-component-not-immediate",
        "not-targets",
        "wrong-target",
        "missing-identity",
    ],
)
def test_malformed_wearec_layout_fails_before_any_deletion(tmp_path, relative):
    run_dir, shared_paths = _create_run(
        tmp_path, clean=False, targets=("101",)
    )
    malformed = tmp_path / "malformed" / Path(*relative)
    malformed.mkdir(parents=True)
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    manifest["victims"]["101"]["wearec"]["shared"]["shared_dir"] = str(malformed)
    _write_json(run_dir / "artifact_manifest.json", manifest)
    local = run_dir / "targets" / "101" / "victims" / "wearec"

    with pytest.raises(InvalidationError, match="unsafe wearec"):
        invalidate_victim_cells(
            [run_dir], victim="wearec", allowed_roots=[tmp_path]
        )

    assert local.is_dir()
    assert shared_paths[0].is_dir()
    assert malformed.is_dir()


def test_wearec_rejects_shared_path_outside_allowed_roots(tmp_path):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    outside = (
        tmp_path.parent
        / "outside"
        / "victim_predictions"
        / "wearec"
        / "key"
        / "shared"
    )
    manifest = json.loads((run_dir / "artifact_manifest.json").read_text())
    manifest["victims"]["101"]["wearec"]["shared"]["shared_dir"] = str(outside)
    _write_json(run_dir / "artifact_manifest.json", manifest)
    with pytest.raises(InvalidationError, match="outside the allowed"):
        inspect_invalidation_plan(
            run_dir, victim="wearec", allowed_roots=[tmp_path]
        )
    assert shared_paths[0].is_dir()


def test_freqrec_manifest_invalidation_behavior_is_unchanged(tmp_path):
    run_dir, shared_paths = _create_run(
        tmp_path, victim="freqrec", clean=False, targets=("101",)
    )
    invalidate_victim_cells(
        [run_dir], victim="freqrec", allowed_roots=[tmp_path]
    )
    assert not shared_paths[0].exists()


@pytest.mark.parametrize("clean", [True, False], ids=["clean", "poisoned"])
def test_freqrec_exact_shared_layouts_remain_accepted(tmp_path, clean):
    run_dir, shared_paths = _create_run(
        tmp_path,
        victim="freqrec",
        clean=clean,
        targets=("101",),
    )
    plan = inspect_invalidation_plan(
        run_dir, victim="freqrec", allowed_roots=[tmp_path]
    )
    assert plan.shared_artifact_dirs == (shared_paths[0],)


def test_wearec_invalidation_preserves_other_victim_and_attack_artifacts(tmp_path):
    run_dir, _ = _create_run(tmp_path, clean=True)
    other = tmp_path / "shared" / "victim_predictions" / "mdhg" / "keep"
    attack = run_dir / "targets" / "101" / "pts_construction_cem"
    invalidate_victim_cells(
        [run_dir], victim="wearec", allowed_roots=[tmp_path]
    )
    assert other.is_dir()
    assert attack.is_dir()


def test_wearec_invalidation_removes_entry_required_for_cache_reuse(tmp_path):
    run_dir, shared_paths = _create_run(tmp_path, clean=True)
    invalidate_victim_cells(
        [run_dir], victim="wearec", allowed_roots=[tmp_path]
    )
    assert not shared_paths[0].exists()
