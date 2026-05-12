from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import save_json
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
    attack_key,
)
from attack.pipeline.runs.run_pts_construction_candidate_replay import (
    build_pts_candidate_replay_run_identity_context,
    load_pts_cem_top_candidate_source,
    resolve_pts_cem_top_candidate_paths,
    _source_pts_cem_artifact_dir,
    _target_replay_metadata,
)


CONFIG_PATH = Path(
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)


def _repo_temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / ".pytest_pts_candidate_replay" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_config(target_item: int = 5334):
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        experiment=replace(
            config.experiment,
            name=f"{config.experiment.name}_pytest_replay_{uuid4().hex}",
        ),
        targets=replace(
            config.targets,
            mode="explicit_list",
            explicit_list=(int(target_item),),
            count=1,
        ),
    )


def _write_rank_candidate(root: Path, *, rank: int, target_item: int) -> None:
    rank_dir = root / "top_candidates" / f"rank_{int(rank)}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    save_json([[1, 2, int(target_item)], [3, 4, int(target_item)]], rank_dir / "sessions.json")
    save_json(
        {
            "rank": int(rank),
            "target_item": int(target_item),
            "iteration": 1,
            "candidate_id": 7,
            "candidate_seed": 1234,
            "reward": 0.42,
            "reward_metrics": {"reward": 0.42},
            "evaluator_metadata": {"candidate_seed": 1234},
        },
        rank_dir / "metadata.json",
    )
    save_json({"group_probabilities": {}}, rank_dir / "policy.json")
    save_json({"candidates": [{"rank": int(rank)}]}, root / "pts_top_candidates.json")
    save_json({"top_k": int(rank)}, root / "pts_top_candidate_policies.json")


def test_pts_candidate_replay_help_exits_successfully() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "attack.pipeline.runs.run_pts_construction_candidate_replay",
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "--candidate-rank" in completed.stdout
    assert "--target-item" in completed.stdout


def test_resolve_pts_cem_top_candidate_paths() -> None:
    root = Path("targets/5334/pts_construction_cem")
    paths = resolve_pts_cem_top_candidate_paths(root, candidate_rank=2)

    assert paths["rank_dir"] == root / "top_candidates" / "rank_2"
    assert paths["sessions"] == root / "top_candidates" / "rank_2" / "sessions.json"
    assert paths["metadata"] == root / "top_candidates" / "rank_2" / "metadata.json"
    assert paths["policy"] == root / "top_candidates" / "rank_2" / "policy.json"


def test_load_pts_candidate_source_reads_saved_rank() -> None:
    config = _unique_config(target_item=5334)
    source_dir = _source_pts_cem_artifact_dir(config, target_item=5334)
    try:
        _write_rank_candidate(source_dir, rank=2, target_item=5334)

        source = load_pts_cem_top_candidate_source(
            config,
            target_item=5334,
            candidate_rank=2,
        )

        assert source.sessions == [[1, 2, 5334], [3, 4, 5334]]
        assert source.metadata["candidate_id"] == 7
        assert source.candidate_rank == 2
        assert source.sessions_path == source_dir / "top_candidates" / "rank_2" / "sessions.json"
    finally:
        shutil.rmtree(source_dir.parents[2], ignore_errors=True)


def test_load_pts_candidate_source_missing_sessions_raises_clear_error() -> None:
    config = _unique_config(target_item=5334)
    source_dir = _source_pts_cem_artifact_dir(config, target_item=5334)
    try:
        rank_dir = source_dir / "top_candidates" / "rank_2"
        rank_dir.mkdir(parents=True, exist_ok=True)
        save_json({"rank": 2, "target_item": 5334}, rank_dir / "metadata.json")
        save_json({"group_probabilities": {}}, rank_dir / "policy.json")

        with pytest.raises(FileNotFoundError, match="save_top_k_candidates >= 2"):
            load_pts_cem_top_candidate_source(
                config,
                target_item=5334,
                candidate_rank=2,
            )
    finally:
        shutil.rmtree(source_dir.parents[2], ignore_errors=True)


def test_load_pts_candidate_source_rejects_malformed_sessions() -> None:
    config = _unique_config(target_item=5334)
    source_dir = _source_pts_cem_artifact_dir(config, target_item=5334)
    try:
        rank_dir = source_dir / "top_candidates" / "rank_2"
        rank_dir.mkdir(parents=True, exist_ok=True)
        save_json([[1, "bad"]], rank_dir / "sessions.json")
        save_json({"rank": 2, "target_item": 5334}, rank_dir / "metadata.json")
        save_json({"group_probabilities": {}}, rank_dir / "policy.json")

        with pytest.raises(ValueError, match="int-like"):
            load_pts_cem_top_candidate_source(
                config,
                target_item=5334,
                candidate_rank=2,
            )
    finally:
        shutil.rmtree(source_dir.parents[2], ignore_errors=True)


def test_load_pts_candidate_source_rejects_target_mismatch() -> None:
    config = _unique_config(target_item=5334)
    source_dir = _source_pts_cem_artifact_dir(config, target_item=5334)
    try:
        _write_rank_candidate(source_dir, rank=2, target_item=9999)

        with pytest.raises(ValueError, match="target_item mismatch"):
            load_pts_cem_top_candidate_source(
                config,
                target_item=5334,
                candidate_rank=2,
            )
    finally:
        shutil.rmtree(source_dir.parents[2], ignore_errors=True)


def test_target_replay_metadata_contains_candidate_fields() -> None:
    config = _unique_config(target_item=5334)
    source_dir = _source_pts_cem_artifact_dir(config, target_item=5334)
    try:
        _write_rank_candidate(source_dir, rank=2, target_item=5334)
        source = load_pts_cem_top_candidate_source(
            config,
            target_item=5334,
            candidate_rank=2,
        )
        replay_metadata = {
            "candidate_iteration": 1,
            "candidate_id": 7,
            "candidate_seed": 1234,
            "candidate_validation_reward": 0.42,
            "candidate_reward_metrics": {"reward": 0.42},
            "replay_paths": {
                "replay_candidate_metadata": "replay_candidate_metadata.json",
                "comparison_summary": "comparison_summary.json",
            },
        }

        payload = _target_replay_metadata(
            source=source,
            replay_metadata=replay_metadata,
            target_item=5334,
            candidate_rank=2,
        )

        assert payload["pts_candidate_replay"] is True
        assert payload["pts_replay_candidate_rank"] == 2
        assert payload["pts_candidate_id"] == 7
        assert payload["pts_candidate_validation_reward"] == 0.42
        assert payload["pts_final_selection_mode"] == "candidate_replay"
    finally:
        shutil.rmtree(source_dir.parents[2], ignore_errors=True)


def test_pts_candidate_replay_attack_identity_changes_by_rank() -> None:
    config = _unique_config(target_item=5334)
    key_rank2 = attack_key(
        config,
        run_type=PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
        attack_identity_context=build_pts_candidate_replay_run_identity_context(
            config,
            candidate_rank=2,
        ),
    )
    key_rank3 = attack_key(
        config,
        run_type=PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
        attack_identity_context=build_pts_candidate_replay_run_identity_context(
            config,
            candidate_rank=3,
        ),
    )

    assert key_rank2 != key_rank3
