from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSCEMRuntimeConfig,
    PTSArtifactsConfig,
    PTSRewardConfig,
    load_config,
)
from attack.common.artifact_io import save_json
from attack.pipeline.runs import run_pts_construction_cem
from attack.pipeline.runs.run_pts_construction_cem import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    _load_json_sessions,
    _resolve_pts_cem_base_seed,
    _try_load_cached_pts_best_candidate,
    _validate_pts_construction_run_config,
    _write_pts_construction_complete_marker,
)


CONFIG_PATH = Path(
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_ratio1_srgnn_partial4.yaml"
)


def _with_pts(config, pts_config):
    return replace(
        config,
        attack=replace(config.attack, pts_construction=pts_config),
    )


def _make_test_output_dir() -> Path:
    path = REPO_ROOT / "outputs" / f"tmp_pts_pipeline_runner_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def test_runner_imports_without_training() -> None:
    assert run_pts_construction_cem.DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH
    assert callable(run_pts_construction_cem.run_pts_construction_grouped_cem)


def test_runner_help_exits_successfully() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "attack.pipeline.runs.run_pts_construction_cem",
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Grouped PTS-CEM" in completed.stdout
    assert "--force-recompute-pts-cem" in completed.stdout


def test_runner_validation_rejects_save_candidate_sessions() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    artifacts = object.__new__(PTSArtifactsConfig)
    object.__setattr__(artifacts, "save_cem_trace", True)
    object.__setattr__(artifacts, "save_best_policy", True)
    object.__setattr__(artifacts, "save_final_policy", True)
    object.__setattr__(artifacts, "save_per_session_records", True)
    object.__setattr__(artifacts, "save_candidate_sessions", True)
    object.__setattr__(artifacts, "save_best_sessions", True)
    object.__setattr__(artifacts, "save_top_candidate_sessions", True)
    bad_config = _with_pts(
        config,
        replace(
            pts,
            artifacts=artifacts,
        ),
    )

    with pytest.raises(ValueError, match="save_candidate_sessions"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_disabled_top_candidate_sessions() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    artifacts = object.__new__(PTSArtifactsConfig)
    object.__setattr__(artifacts, "save_cem_trace", True)
    object.__setattr__(artifacts, "save_best_policy", True)
    object.__setattr__(artifacts, "save_final_policy", True)
    object.__setattr__(artifacts, "save_per_session_records", True)
    object.__setattr__(artifacts, "save_candidate_sessions", False)
    object.__setattr__(artifacts, "save_best_sessions", True)
    object.__setattr__(artifacts, "save_top_candidate_sessions", False)
    bad_config = _with_pts(config, replace(pts, artifacts=artifacts))

    with pytest.raises(ValueError, match="save_top_candidate_sessions"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_unsupported_penalties() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    reward = object.__new__(PTSRewardConfig)
    object.__setattr__(reward, "target_summary", "raw_lowk_mrr_recall_10_20")
    object.__setattr__(reward, "enable_gt_penalty", True)
    object.__setattr__(reward, "gt_penalty_weight", 1.0)
    object.__setattr__(reward, "enable_length_penalty", False)
    object.__setattr__(reward, "length_penalty_weight", 0.0)
    bad_config = _with_pts(config, replace(pts, reward=reward))

    with pytest.raises(NotImplementedError, match="GT penalty"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_unsupported_seed_source() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    cem = object.__new__(PTSCEMRuntimeConfig)
    for field in fields(PTSCEMRuntimeConfig):
        object.__setattr__(cem, field.name, getattr(pts.cem, field.name))
    object.__setattr__(cem, "seed_source", "fake_session_seed")
    bad_config = _with_pts(config, replace(pts, cem=cem))

    with pytest.raises(ValueError, match="position_opt_seed"):
        _resolve_pts_cem_base_seed(bad_config)


def test_pts_cache_load_json_sessions_validates_shape() -> None:
    work_dir = _make_test_output_dir()
    try:
        sessions_path = work_dir / "sessions.json"
        save_json([[1, "2", 3.0], [4, 5]], sessions_path)
        assert _load_json_sessions(sessions_path) == [[1, 2, 3], [4, 5]]

        save_json({"not": "sessions"}, sessions_path)
        with pytest.raises(ValueError, match="JSON list"):
            _load_json_sessions(sessions_path)

        save_json([[1, 2], "bad-row"], sessions_path)
        with pytest.raises(ValueError, match="session lists"):
            _load_json_sessions(sessions_path)

        save_json([[1, "not-an-int"]], sessions_path)
        with pytest.raises(ValueError, match="int-like"):
            _load_json_sessions(sessions_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_legacy_fallback_loads_rank1_sessions() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99], [3, 4, 99]], rank1 / "sessions.json")
        save_json(
            {
                "rank": 1,
                "iteration": 2,
                "candidate_id": 3,
                "candidate_seed": 202,
                "reward": 0.25,
                "target_item": 99,
            },
            rank1 / "metadata.json",
        )
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")

        cached = _try_load_cached_pts_best_candidate(
            artifact_dir=root,
            target_item=99,
        )

        assert cached is not None
        assert cached.sessions == [[1, 2, 99], [3, 4, 99]]
        assert cached.cache_mode == "legacy_top_candidate_files"
        assert cached.cache_marker_missing is True
        assert cached.complete_marker_path is None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_complete_marker_resolves_relative_paths() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 99}, rank1 / "metadata.json")
        save_json(
            {
                "schema_version": "pts_construction_cache_v1",
                "status": "completed",
                "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                "target_item": 99,
                "cache_mode": "fresh_cem",
                "best_candidate": {
                    "rank": 1,
                    "iteration": 1,
                    "candidate_id": 2,
                    "candidate_seed": 3,
                    "reward": 0.5,
                    "reward_metrics": {"reward": 0.5},
                    "sessions_path": "top_candidates/rank_1/sessions.json",
                    "metadata_path": "top_candidates/rank_1/metadata.json",
                    "policy_path": "top_candidates/rank_1/policy.json",
                },
            },
            root / "pts_construction_complete.json",
        )

        cached = _try_load_cached_pts_best_candidate(
            artifact_dir=root,
            target_item=99,
        )

        assert cached is not None
        assert cached.cache_mode == "complete_marker"
        assert cached.cache_marker_missing is False
        assert cached.sessions_path == rank1 / "sessions.json"
        assert cached.metadata["reward"] == 0.5
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_rejects_target_mismatch() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 100}, rank1 / "metadata.json")
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")

        with pytest.raises(ValueError, match="target_item mismatch"):
            _try_load_cached_pts_best_candidate(
                artifact_dir=root,
                target_item=99,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_complete_marker_rejects_target_mismatch() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        save_json(
            {
                "status": "completed",
                "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                "target_item": 100,
                "best_candidate": {
                    "sessions_path": "top_candidates/rank_1/sessions.json",
                    "metadata_path": "top_candidates/rank_1/metadata.json",
                },
            },
            root / "pts_construction_complete.json",
        )

        with pytest.raises(ValueError, match="target_item mismatch"):
            _try_load_cached_pts_best_candidate(
                artifact_dir=root,
                target_item=99,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_complete_marker_writer_creates_candidate_fields() -> None:
    work_dir = _make_test_output_dir()
    try:
        config = load_config(CONFIG_PATH)
        artifact_dir = work_dir / "pts_construction_cem"
        rank1 = artifact_dir / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        artifact_paths = {
            "top_candidate_rank_1_sessions": str(rank1 / "sessions.json"),
            "top_candidate_rank_1_metadata": str(rank1 / "metadata.json"),
            "top_candidate_rank_1_policy": str(rank1 / "policy.json"),
        }

        class FakeBestCandidate:
            iteration = 2
            candidate_id = 5
            candidate_seed = 20260405
            reward = 0.75
            reward_metrics = {"reward": 0.75}

        marker_path = _write_pts_construction_complete_marker(
            config=config,
            target_item=5334,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
            best_candidate=FakeBestCandidate(),
            attack_identity_context=(
                run_pts_construction_cem.build_pts_construction_attack_identity_context(
                    config
                )
            ),
        )

        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        assert marker["status"] == "completed"
        assert marker["run_type"] == PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
        assert marker["target_item"] == 5334
        assert (
            marker["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            marker["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert marker["configured_surrogate_train_seed"] == 20260405
        assert marker["configured_victim_train_seed"] == 20260405
        assert marker["resolved_surrogate_effective_seed"] == 1386226870
        assert marker["resolved_victim_effective_seed"] == 1386226870
        assert marker["surrogate_victim_seed_aligned"] is True
        assert marker["identity"]["target_item"] == 5334
        assert (
            marker["identity"]["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            marker["identity"]["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert marker["identity"]["resolved_surrogate_effective_seed"] == 1386226870
        assert marker["identity"]["resolved_victim_effective_seed"] == 1386226870
        assert marker["best_candidate"]["rank"] == 1
        assert marker["best_candidate"]["candidate_id"] == 5
        assert marker["best_candidate"]["sessions_path"] == (
            "top_candidates/rank_1/sessions.json"
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_target_metadata_uses_aligned_surrogate_retrain_seed() -> None:
    config = load_config(CONFIG_PATH)
    pts_config = config.attack.pts_construction
    assert pts_config is not None
    cem_config = pts_config.cem
    artifact_dir = Path("outputs/test_pts_metadata/pts_construction_cem")
    artifact_paths = {
        "pts_cem_trace": str(artifact_dir / "pts_cem_trace.jsonl"),
        "pts_top_candidates": str(artifact_dir / "pts_top_candidates.json"),
        "top_candidate_rank_1_sessions": str(
            artifact_dir / "top_candidates" / "rank_1" / "sessions.json"
        ),
        "top_candidate_rank_1_metadata": str(
            artifact_dir / "top_candidates" / "rank_1" / "metadata.json"
        ),
    }

    class FakeBestCandidate:
        iteration = 2
        candidate_id = 5
        candidate_seed = 20260405
        reward = 0.75
        reward_metrics = {"reward": 0.75}

    metadata = run_pts_construction_cem._target_metadata(
        config=config,
        pts_config=pts_config,
        cem_config=cem_config,
        artifact_dir=artifact_dir,
        artifact_paths=artifact_paths,
        best_candidate=FakeBestCandidate(),
        target_item=5334,
        complete_marker_path=artifact_dir / "pts_construction_complete.json",
    )
    cached = run_pts_construction_cem.CachedPTSBestCandidate(
        sessions=[[1, 2, 5334]],
        metadata={
            "iteration": 2,
            "candidate_id": 5,
            "candidate_seed": 20260405,
            "reward": 0.75,
            "reward_metrics": {"reward": 0.75},
        },
        sessions_path=Path(artifact_paths["top_candidate_rank_1_sessions"]),
        metadata_path=Path(artifact_paths["top_candidate_rank_1_metadata"]),
        top_candidates_path=Path(artifact_paths["pts_top_candidates"]),
        complete_marker_path=None,
        cache_mode="complete_marker",
        cache_marker_missing=False,
    )
    cached_metadata = run_pts_construction_cem._target_metadata_from_cache(
        config=config,
        pts_config=pts_config,
        cem_config=cem_config,
        artifact_dir=artifact_dir,
        target_item=5334,
        cached=cached,
    )

    for payload in (metadata, cached_metadata):
        assert payload["target_item"] == 5334
        assert payload["pts_candidate_retrain_seed"] == 1386226870
        assert (
            payload["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            payload["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert payload["configured_surrogate_train_seed"] == 20260405
        assert payload["configured_victim_train_seed"] == 20260405
        assert payload["resolved_surrogate_effective_seed"] == 1386226870
        assert payload["resolved_victim_effective_seed"] == 1386226870
        assert payload["surrogate_victim_seed_aligned"] is True
