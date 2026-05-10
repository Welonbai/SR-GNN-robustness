from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

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
from attack.pipeline.runs import run_pts_construction_cem
from attack.pipeline.runs.run_pts_construction_cem import (
    _resolve_pts_cem_base_seed,
    _validate_pts_construction_run_config,
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
    object.__setattr__(cem, "seed_source", "fake_session_seed")
    bad_config = _with_pts(config, replace(pts, cem=cem))

    with pytest.raises(ValueError, match="position_opt_seed"):
        _resolve_pts_cem_base_seed(bad_config)
