from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSConstructionConfig,
    PTSContinuousBetaConfig,
    PTSCEMRuntimeConfig,
    PTS_CONSTRUCTION_METHOD_CONTINUOUS_BETA_CEM_V1,
    PTS_CEM_SAMPLER_GAUSSIAN,
    load_config,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _validate_pts_construction_run_config,
    build_pts_cem_shared_cache_identity,
    pts_cem_shared_cache_key,
)


CONFIG_PATH = Path("attack/configs/diginetica_valbest_attack_ptscem_internal_sample.yaml")


def _with_pts(config, pts_config):
    return replace(config, attack=replace(config.attack, pts_construction=pts_config))


def _continuous_pts(*, sampler_type: str = PTS_CEM_SAMPLER_GAUSSIAN, initial_std: float = 2.0):
    return PTSConstructionConfig(
        enabled=True,
        method=PTS_CONSTRUCTION_METHOD_CONTINUOUS_BETA_CEM_V1,
        continuous_beta=PTSContinuousBetaConfig(initial_std=initial_std),
        cem=PTSCEMRuntimeConfig(
            iterations=2,
            population_schedule=(4, 2),
            sampler={"type": sampler_type, "concentration_scale": 20.0},
        ),
    )


def test_continuous_pts_config_validates_without_grouping_or_actions() -> None:
    config = load_config(CONFIG_PATH)
    continuous_config = _with_pts(config, _continuous_pts())

    _validate_pts_construction_run_config(continuous_config)


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
