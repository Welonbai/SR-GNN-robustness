from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSActionsConfig,
    PTSCEMEpochRewardDiagnosticsRuntimeConfig,
    PTSFinalSelectionConfig,
    PTSRewardConfig,
    PTSCEMRuntimeConfig,
    PTSSuffixLengthBucketConfig,
    load_config,
)
from attack.common.seed import derive_seed
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE,
    PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME,
    _build_pts_cem_config_from_config,
    _build_pts_specs_from_config,
    _build_suffix_length_buckets_from_config,
    _validate_pts_construction_run_config,
    build_pts_construction_attack_identity_context,
    pts_cem_surrogate_seed_alignment_metadata,
    resolve_pts_cem_surrogate_effective_seed,
)
from attack.pts.policy import build_valid_actions_by_group


CONFIG_PATH = Path(
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)
SPACE_FILLING_CONFIG_PATH = CONFIG_PATH
EPOCH_DIAG_CONFIG_PATH = Path(
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_epochdiag_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)
NEW_DIAGNOSTIC_CONFIG_PATH = Path(
    "attack/configs/"
    "diginetica_valbest_attack_ptscem_diagnostic_ratio1_srgnn_partial4_target39588.yaml"
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
        "consume_one_generate_continuation",
        "consume_all_stop",
    ]
    assert pts.cem.init.mode == "vertex_stratified_space_filling"
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4
    assert pts.cem.epoch_reward_diagnostics.enabled is False
    assert pts.cem.epoch_reward_diagnostics.epochs == ()


def test_pts_epoch_reward_diagnostics_yaml_loads() -> None:
    config = load_config(EPOCH_DIAG_CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    diagnostics = pts.cem.epoch_reward_diagnostics
    assert diagnostics.enabled is True
    assert diagnostics.epochs == (2, 3)
    assert diagnostics.include_final_epoch is True
    assert diagnostics.write_candidate_epoch_metrics is True
    assert diagnostics.write_ranking_summary is True
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4


def test_pts_epoch_reward_diagnostics_config_validation() -> None:
    assert PTSCEMEpochRewardDiagnosticsRuntimeConfig().enabled is False

    with pytest.raises(ValueError, match="positive integers"):
        PTSCEMEpochRewardDiagnosticsRuntimeConfig(enabled=True, epochs=(0,))

    with pytest.raises(ValueError, match="duplicates"):
        PTSCEMEpochRewardDiagnosticsRuntimeConfig(enabled=True, epochs=(2, 2))

    with pytest.raises(ValueError, match="must not be empty"):
        PTSCEMEpochRewardDiagnosticsRuntimeConfig(enabled=True, epochs=())


def test_pts_epoch_reward_diagnostics_rejects_epoch_above_retrain_budget() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    bad_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                    enabled=True,
                    epochs=(5,),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="retrain epochs"):
        _validate_pts_construction_run_config(bad_config)


def test_pts_epoch_reward_diagnostics_rejects_final_epoch_when_included() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    bad_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                    enabled=True,
                    epochs=(2, 3, 4),
                    include_final_epoch=True,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="final_partial_retrain_protocol"):
        _validate_pts_construction_run_config(bad_config)


def test_pts_defaults_use_five_action_space_filling_runtime() -> None:
    assert list(PTSActionsConfig().enabled) == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_one_generate_continuation",
        "consume_all_stop",
    ]
    assert PTSCEMRuntimeConfig().init.mode == "vertex_stratified_space_filling"
    assert PTSCEMRuntimeConfig().resampling.mode == "elite_centered"


def test_new_pts_diagnostic_yaml_loads_with_default_runtime_settings() -> None:
    config = load_config(NEW_DIAGNOSTIC_CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    assert pts.enabled is True
    assert config.experiment.name == (
        "valbest_attack_ptscem_diagnostic_ratio1_srgnn_partial4_target39588"
    )
    assert config.targets.mode == "explicit_list"
    assert list(config.targets.explicit_list) == [39588]
    assert config.targets.count == 1
    assert list(config.victims.enabled) == ["srgnn"]
    assert list(pts.actions.enabled) == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_one_generate_continuation",
        "consume_all_stop",
    ]
    assert pts.cem.init.mode == "vertex_stratified_space_filling"
    assert pts.cem.resampling.mode == "elite_centered"
    assert pts.cem.epoch_reward_diagnostics.enabled is True
    assert pts.cem.epoch_reward_diagnostics.epochs == (2, 3)


def test_pts_space_filling_yaml_loads_with_five_actions() -> None:
    config = load_config(SPACE_FILLING_CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    assert pts.enabled is True
    assert config.experiment.name == (
        "valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334"
    )
    assert list(pts.actions.enabled) == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_one_generate_continuation",
        "consume_all_stop",
    ]
    assert pts.cem.init.mode == "vertex_stratified_space_filling"
    assert pts.cem.init.mandatory_enabled is True
    assert pts.cem.init.extreme_count == 7
    assert pts.cem.init.moderate_count == 3
    assert pts.cem.init.balanced_count == 1
    assert pts.cem.population_schedule == (16, 8, 8)
    assert pts.cem.resampling.mode == "elite_centered"
    assert pts.cem.resampling.local_concentration_scale == 30.0

    specs = _build_pts_specs_from_config(pts)
    valid_actions = build_valid_actions_by_group(
        group_buckets=_build_suffix_length_buckets_from_config(pts),
        enabled_actions=[spec.name for spec in specs],
        disable_consume_one_when_suffix_len_leq_1=(
            pts.actions.dynamic_masks.disable_consume_one_when_suffix_len_leq_1
        ),
    )
    assert valid_actions["suffix_1"] == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_all_stop",
    ]


def test_vertex_space_filling_mandatory_requires_c1_generate_action() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None

    with pytest.raises(
        ValueError,
        match=(
            "vertex_stratified_space_filling with mandatory_enabled=true requires "
            "consume_one_generate_continuation"
        ),
    ):
        replace(
            pts,
            actions=replace(
                pts.actions,
                enabled=(
                    "keep_residual_suffix",
                    "regenerate_residual_suffix",
                    "consume_one_keep_rest",
                    "consume_all_stop",
                ),
            ),
            cem=replace(
                pts.cem,
                init={
                    "mode": "vertex_stratified_space_filling",
                    "mandatory_enabled": True,
                },
            ),
        )


def test_pts_cem_runtime_config_mapping() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    uniform_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                init={"mode": "uniform"},
            ),
        ),
    )
    cem_config = _build_pts_cem_config_from_config(uniform_config)

    assert cem_config.iterations == 3
    assert cem_config.population_schedule == [16, 8, 8]
    assert cem_config.base_seed == config.seeds.position_opt_seed
    assert cem_config.candidate_seed_stride == 1000
    assert cem_config.resampling.mode == "elite_centered"
    assert cem_config.init.mode == "uniform"


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


def test_pts_cem_vertex_init_runtime_config_mapping() -> None:
    config = load_config(SPACE_FILLING_CONFIG_PATH)
    cem_config = _build_pts_cem_config_from_config(config)

    assert cem_config.init.mode == "vertex_stratified_space_filling"
    assert cem_config.init.mandatory_enabled is True
    assert cem_config.init.extreme_count == 7
    assert cem_config.init.moderate_count == 3
    assert cem_config.init.balanced_count == 1
    assert cem_config.init.extreme_pool_size == 1024
    assert cem_config.init.moderate_pool_size == 512
    assert cem_config.init.extreme_alpha == 0.3
    assert cem_config.init.moderate_alpha == 2.0
    assert cem_config.init.distance == "l1"


def test_pts_cem_surrogate_seed_aligns_to_srgnn_victim_effective_seed() -> None:
    config = load_config(SPACE_FILLING_CONFIG_PATH)
    expected = derive_seed(20260405, "victim_train", "srgnn", 5334)

    assert config.experiment.name == (
        "valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334"
    )
    resolved = resolve_pts_cem_surrogate_effective_seed(config, target_item=5334)
    metadata = pts_cem_surrogate_seed_alignment_metadata(config, target_item=5334)

    assert expected == 1386226870
    assert resolved == expected
    assert metadata["target_item"] == 5334
    assert (
        metadata["pts_cem_surrogate_seed_alignment_mode"]
        == PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE
        == "victim_effective_seed"
    )
    assert (
        metadata["pts_cem_surrogate_seed_alignment_target_victim_name"]
        == PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME
        == "srgnn"
    )
    assert metadata["configured_surrogate_train_seed"] == 20260405
    assert metadata["configured_victim_train_seed"] == 20260405
    assert metadata["resolved_surrogate_effective_seed"] == expected
    assert metadata["resolved_victim_effective_seed"] == expected
    assert metadata["surrogate_victim_seed_aligned"] is True
    assert metadata["resolved_surrogate_effective_seed"] != (
        metadata["configured_surrogate_train_seed"]
    )


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
                enabled=(
                    "keep_residual_suffix",
                    "consume_one_generate_continuation",
                    "consume_all_stop",
                ),
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


def test_pts_epoch_reward_diagnostics_do_not_enter_attack_identity() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    base_attack_key = attack_key(config, run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE)
    diagnostic_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                    enabled=True,
                    epochs=(2, 3),
                ),
            ),
        ),
    )

    assert (
        attack_key(
            diagnostic_config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        )
        == base_attack_key
    )


def test_pts_attack_identity_context_includes_a0a4_vertex_init_fields() -> None:
    config = load_config(SPACE_FILLING_CONFIG_PATH)
    context = build_pts_construction_attack_identity_context(config)
    pts_identity = context["pts_construction"]
    cem_identity = pts_identity["cem"]

    assert pts_identity["actions"]["enabled"] == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_one_generate_continuation",
        "consume_all_stop",
    ]
    assert pts_identity["runtime_seeds"]["position_opt_seed"] == 20260405
    assert pts_identity["runtime_seeds"]["resolved_cem_base_seed"] == 20260405
    seed_alignment = pts_identity["runtime_seeds"]["surrogate_seed_alignment"]
    assert seed_alignment["target_item"] == 5334
    assert (
        seed_alignment["pts_cem_surrogate_seed_alignment_mode"]
        == "victim_effective_seed"
    )
    assert (
        seed_alignment["pts_cem_surrogate_seed_alignment_target_victim_name"]
        == "srgnn"
    )
    assert seed_alignment["configured_surrogate_train_seed"] == 20260405
    assert seed_alignment["configured_victim_train_seed"] == 20260405
    assert seed_alignment["resolved_surrogate_effective_seed"] == 1386226870
    assert seed_alignment["resolved_victim_effective_seed"] == 1386226870
    assert seed_alignment["surrogate_victim_seed_aligned"] is True
    assert cem_identity["cem_base_seed"] == 20260405
    assert cem_identity["resolved_cem_base_seed"] == 20260405
    assert cem_identity["init"] == {
        "mode": "vertex_stratified_space_filling",
        "mandatory_enabled": True,
        "extreme_count": 7,
        "moderate_count": 3,
        "balanced_count": 1,
        "extreme_pool_size": 1024,
        "moderate_pool_size": 512,
        "extreme_alpha": 0.3,
        "moderate_alpha": 2.0,
        "distance": "l1",
    }
