from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSFinalSelectionConfig,
    PTSRewardConfig,
    PTSCEMRuntimeConfig,
    PTSSuffixLengthBucketConfig,
    load_config,
)
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _build_pts_cem_config_from_config,
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


def test_pts_pipeline_yaml_loads() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    assert pts.enabled is True
    assert pts.method == "grouped_cem_v1"
    assert pts.cem.population_schedule == (16, 8, 8)
    assert list(pts.actions.enabled) == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_all_stop",
    ]
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4


def test_pts_cem_runtime_config_mapping() -> None:
    config = load_config(CONFIG_PATH)
    cem_config = _build_pts_cem_config_from_config(config)

    assert cem_config.iterations == 3
    assert cem_config.population_schedule == [16, 8, 8]
    assert cem_config.base_seed == config.seeds.position_opt_seed
    assert cem_config.candidate_seed_stride == 1000
    assert cem_config.resampling.mode == "standard"


def test_pts_cem_resampling_runtime_config_mapping() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    resampling_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                resampling={
                    "mode": "elite_centered",
                    "local_concentration_scale": 30.0,
                },
            ),
        ),
    )

    cem_config = _build_pts_cem_config_from_config(resampling_config)

    assert cem_config.resampling.mode == "elite_centered"
    assert cem_config.resampling.local_concentration_scale == 30.0


def test_pts_cem_resampling_rejects_global_exploration_fields() -> None:
    with pytest.raises(TypeError, match="global_exploration_fraction"):
        PTSCEMRuntimeConfig(
            resampling={
                "mode": "elite_centered",
                "local_concentration_scale": 30.0,
                "global_exploration_fraction": 0.25,
            }
        )


def test_invalid_final_selection_mode_raises() -> None:
    with pytest.raises(ValueError, match="global_best_candidate"):
        PTSFinalSelectionConfig(mode="last")


def test_invalid_reward_target_summary_raises() -> None:
    with pytest.raises(ValueError, match="raw_lowk_mrr_recall_10_20"):
        PTSRewardConfig(target_summary="targeted_mrr@20")


def test_pts_attack_identity_changes_but_shared_key_does_not_for_pts_fields() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    base_attack_key = attack_key(config, run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE)
    base_shared_key = shared_attack_artifact_key(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )

    schedule_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(pts.cem, population_schedule=(8, 8, 8)),
        ),
    )
    actions_config = _with_pts(
        config,
        replace(
            pts,
            actions=replace(
                pts.actions,
                enabled=("keep_residual_suffix", "consume_all_stop"),
            ),
        ),
    )
    bucket_config = _with_pts(
        config,
        replace(
            pts,
            grouping=replace(
                pts.grouping,
                buckets=(
                    PTSSuffixLengthBucketConfig(name="suffix_1", min=1, max=1),
                    PTSSuffixLengthBucketConfig(name="suffix_2", min=2, max=2),
                    PTSSuffixLengthBucketConfig(name="suffix_3plus", min=4, max=None),
                ),
            ),
        ),
    )

    for modified in (schedule_config, actions_config, bucket_config):
        assert (
            attack_key(modified, run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE)
            != base_attack_key
        )
        assert (
            shared_attack_artifact_key(
                modified,
                run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            )
            == base_shared_key
        )
