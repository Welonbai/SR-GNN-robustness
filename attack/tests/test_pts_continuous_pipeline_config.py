from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSConstructionConfig,
    PTSContinuousPolicyConfig,
    PTSCEMInitRuntimeConfig,
    PTSCEMRuntimeConfig,
    PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
    PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
    PTS_CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
    PTS_CONTINUOUS_POLICY_PARAMETERIZATION_SUFFIX_LENGTH_MLP,
    PTS_CEM_SAMPLER_GAUSSIAN,
    load_config,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _validate_pts_construction_run_config,
    build_pts_cem_shared_cache_identity,
    pts_cem_shared_cache_key,
)


CONFIG_PATH = Path("attack/configs/diginetica_valbest_attack_ptscem_internal_sample.yaml")
TEST_CONFIG_DIR = REPO_ROOT / "attack" / "tests" / "fixtures" / "configs"
GROUPED_CONFIG_PATH = (
    TEST_CONFIG_DIR
    / "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)
CONTINUOUS_CONFIG_PATH = (
    TEST_CONFIG_DIR
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_srgnn_partial4_target5334.yaml"
)
CONTINUOUS_MLP_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_srgnn_partial4_target5334.yaml"
)


def _with_pts(config, pts_config):
    return replace(config, attack=replace(config.attack, pts_construction=pts_config))


def _continuous_pts(
    *,
    sampler_type: str = PTS_CEM_SAMPLER_GAUSSIAN,
    initial_std: float = 2.0,
    smoothing_epsilon: float = 0.0,
):
    return PTSConstructionConfig(
        enabled=True,
        method=PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
        continuous_policy=PTSContinuousPolicyConfig(
            smoothing_epsilon=smoothing_epsilon,
        ),
        cem=PTSCEMRuntimeConfig(
            iterations=2,
            population_schedule=(4, 2),
            sampler={"type": sampler_type, "concentration_scale": 20.0},
            init=PTSCEMInitRuntimeConfig(
                mode=PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
                soft_extreme_initial_std=initial_std,
                soft_extreme_pool_size=8,
                moderate_pool_size=8,
                soft_extreme_select_size=2,
                moderate_select_size=2,
                q_grid_size=7,
            ),
        ),
    )


def test_continuous_pts_config_validates_without_grouping_or_actions() -> None:
    config = load_config(CONFIG_PATH)
    continuous_config = _with_pts(config, _continuous_pts())

    _validate_pts_construction_run_config(continuous_config)


def test_continuous_smoothing_epsilon_validation() -> None:
    assert PTSContinuousPolicyConfig(smoothing_epsilon=0.1).smoothing_epsilon == 0.1
    for value in (-0.1, 0.5, 1.0):
        with pytest.raises(ValueError, match="smoothing_epsilon"):
            PTSContinuousPolicyConfig(smoothing_epsilon=value)


def test_continuous_sample_yaml_loads_without_grouped_fields() -> None:
    raw = yaml.safe_load(CONTINUOUS_CONFIG_PATH.read_text(encoding="utf-8"))
    pts_raw = raw["attack"]["pts_construction"]
    assert "grouping" not in pts_raw
    assert "actions" not in pts_raw

    config = load_config(CONTINUOUS_CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    assert pts.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM
    assert pts.cem.sampler.type == PTS_CEM_SAMPLER_GAUSSIAN
    assert pts.continuous_policy.parameterization == "suffix_length_mlp"
    assert pts.continuous_policy.hidden_size == 2
    assert pts.cem.init.mode == "two_pool_behavior_curve_space_filling"
    assert pts.cem.init.init_materialize_generated_suffix is False
    _validate_pts_construction_run_config(config)


def test_continuous_mlp_sample_yaml_loads() -> None:
    config = load_config(CONTINUOUS_MLP_CONFIG_PATH)
    pts = config.attack.pts_construction

    assert pts is not None
    assert config.experiment.name == (
        "valbest_attack_pts_construction_continuous_mlp_cem_ratio1_srgnn_partial4_target5334"
    )
    assert pts.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM
    assert pts.cem.sampler.type == PTS_CEM_SAMPLER_GAUSSIAN
    assert (
        pts.continuous_policy.parameterization
        == PTS_CONTINUOUS_POLICY_PARAMETERIZATION_SUFFIX_LENGTH_MLP
    )
    assert pts.continuous_policy.internal_parameterization == (
        PTS_CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2
    )
    assert pts.continuous_policy.source_policy == "q_and_rho_logistic"
    assert pts.continuous_policy.smoothing_epsilon == 0.1
    assert config.targets.mode == "explicit_list"
    assert config.targets.explicit_list == (5334,)
    _validate_pts_construction_run_config(config)


def test_legacy_continuous_beta_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="continuous_mlp_cem"):
        PTSConstructionConfig(
            enabled=True,
            method="continuous_beta_cem_v1",
        )


def test_grouped_pts_config_still_rejects_non_dirichlet_sampler() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None

    with pytest.raises(ValueError, match="dirichlet"):
        _with_pts(
            config,
            replace(
                pts,
                cem=replace(
                    pts.cem,
                    sampler={
                        "type": PTS_CEM_SAMPLER_GAUSSIAN,
                        "concentration_scale": 20.0,
                    },
                ),
            ),
        )


def test_grouped_validation_does_not_call_continuous_init_selector(monkeypatch) -> None:
    def fail_selector(*args, **kwargs):
        raise AssertionError("grouped CEM must not call continuous init selector")

    monkeypatch.setattr(
        "attack.pipeline.runs.run_pts_construction_cem."
        "build_continuous_mlp_initial_sample_plan",
        fail_selector,
    )
    config = load_config(GROUPED_CONFIG_PATH)

    _validate_pts_construction_run_config(config)


def test_continuous_cache_identity_normalizes_ignored_sampler_type() -> None:
    config = load_config(CONFIG_PATH)
    fake_sessions_path = Path(__file__)
    gaussian = _with_pts(config, _continuous_pts(sampler_type="gaussian"))
    dirichlet = _with_pts(config, _continuous_pts(sampler_type="dirichlet"))

    gaussian_identity = build_pts_cem_shared_cache_identity(
        gaussian,
        target_item=99,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )
    dirichlet_identity = build_pts_cem_shared_cache_identity(
        dirichlet,
        target_item=99,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )

    assert gaussian_identity == dirichlet_identity
    assert pts_cem_shared_cache_key(gaussian_identity) == pts_cem_shared_cache_key(
        dirichlet_identity
    )


def test_continuous_cache_identity_changes_for_output_affecting_settings() -> None:
    config = load_config(CONFIG_PATH)
    fake_sessions_path = Path(__file__)
    base = _with_pts(config, _continuous_pts(initial_std=2.0))
    changed = _with_pts(config, _continuous_pts(initial_std=1.5))

    base_identity = build_pts_cem_shared_cache_identity(
        base,
        target_item=99,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )
    changed_identity = build_pts_cem_shared_cache_identity(
        changed,
        target_item=99,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )

    assert base_identity != changed_identity
    assert pts_cem_shared_cache_key(base_identity) != pts_cem_shared_cache_key(
        changed_identity
    )
