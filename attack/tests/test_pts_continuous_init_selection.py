from __future__ import annotations

import json
import importlib
from dataclasses import replace
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json, save_fake_sessions
from attack.common.config import (
    ArtifactsConfig,
    PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
    TargetsConfig,
    load_config,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _build_continuous_beta_cem_config,
    _build_pts_cem_config_from_config,
    build_pts_cem_shared_cache_identity,
    pts_cem_shared_cache_key,
)
from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSCEMInitConfig,
)
from attack.pts.continuous_cem import (
    ContinuousCandidateSampleSpec,
    PTSContinuousBetaCEMConfig,
    PTSContinuousBetaCEMTrainer,
)
from attack.pts.continuous_init_selection import (
    CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE,
    build_continuous_mlp_initial_sample_plan,
    continuous_mlp_init_cache_key,
    continuous_mlp_init_cache_path,
    continuous_mlp_init_identity_payload,
    resolve_continuous_mlp_init_seed,
)
from attack.pts.continuous_executor import build_continuous_shared_session_contexts
from attack.pts.executor import PTSConstructionBatchResult


CONTINUOUS_FIXTURE = (
    REPO_ROOT
    / "attack"
    / "tests"
    / "fixtures"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_target5334.yaml"
)


def _small_config(tmp_path: Path):
    config = load_config(CONTINUOUS_FIXTURE)
    pts = config.attack.pts_construction
    assert pts is not None
    init = replace(
        pts.cem.init,
        mode=PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
        soft_extreme_pool_size=8,
        moderate_pool_size=8,
        soft_extreme_select_size=2,
        moderate_select_size=2,
        q_grid_size=5,
    )
    cem = replace(pts.cem, init=init)
    return replace(
        config,
        attack=replace(config.attack, pts_construction=replace(pts, cem=cem)),
        artifacts=ArtifactsConfig(
            root=str(tmp_path),
            shared_dir=config.artifacts.shared_dir,
            runs_dir=config.artifacts.runs_dir,
        ),
    )


def test_continuous_init_selection_import_does_not_import_diagnostic_runner() -> None:
    sys.modules.pop("attack.pts.continuous_init_selection", None)
    sys.modules.pop("attack.pipeline.runs.run_pts_continuous_init_diagnostic", None)

    importlib.import_module("attack.pts.continuous_init_selection")

    assert "attack.pipeline.runs.run_pts_continuous_init_diagnostic" not in sys.modules


def test_continuous_mlp_init_cache_records_target_independent_metadata(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    templates = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    cem_config = _build_pts_cem_config_from_config(config)
    continuous_config = _build_continuous_beta_cem_config(config.attack.pts_construction)

    result = build_continuous_mlp_initial_sample_plan(
        config=config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        template_sessions=templates,
        generation_topk=10,
        force_rebuild=True,
    )

    payload = load_json(result.cache_path)
    assert payload["init_materialize_generated_suffix"] is False
    assert payload["identity"]["method"] == "continuous_mlp_cem"
    assert payload["identity"]["init"]["init_materialize_generated_suffix"] is False
    assert "target_item" not in json.dumps(payload["identity"])
    assert "per_session_records" not in json.dumps(payload)
    assert CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE in result.cache_path.parts
    assert "pts_construction_grouped_cem" not in result.cache_path.parts
    assert [item["candidate_key"] for item in payload["selected_candidates"]] == [
        f"iter0_cand{index}" for index in range(4)
    ]
    assert result.selected_sample_plan[0].sample_metadata["candidate_key"] == "iter0_cand0"
    assert result.selected_sample_plan[0].sample_metadata["pool_candidate_key"]


def test_continuous_context_seed_scope_controls_target_dependence() -> None:
    templates = [[index, index + 1, index + 2, index + 3, index + 4] for index in range(1, 30, 5)]
    independent_a = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=1,
        base_seed=123,
        seed_scope="target_independent",
    )
    independent_b = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=4,
        base_seed=123,
        seed_scope="target_independent",
    )
    dependent_a = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=1,
        base_seed=123,
        seed_scope="target_dependent",
    )
    dependent_b = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=4,
        base_seed=123,
        seed_scope="target_dependent",
    )

    assert independent_a == independent_b
    assert dependent_a != dependent_b


def test_continuous_mlp_init_key_and_vectors_ignore_target_item_only(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    changed_target = replace(
        config,
        targets=TargetsConfig(
            mode=config.targets.mode,
            explicit_list=(999999,),
            bucket=config.targets.bucket,
            count=config.targets.count,
            reuse_saved_targets=config.targets.reuse_saved_targets,
        ),
    )
    templates = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    cem_config = _build_pts_cem_config_from_config(config)
    continuous_config = _build_continuous_beta_cem_config(config.attack.pts_construction)

    base_identity = continuous_mlp_init_identity_payload(
        config=config,
        template_sessions=templates,
    )
    changed_identity = continuous_mlp_init_identity_payload(
        config=changed_target,
        template_sessions=templates,
    )
    base = build_continuous_mlp_initial_sample_plan(
        config=config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        template_sessions=templates,
        generation_topk=10,
        force_rebuild=True,
    )
    changed = build_continuous_mlp_initial_sample_plan(
        config=changed_target,
        cem_config=cem_config,
        continuous_config=continuous_config,
        template_sessions=templates,
        generation_topk=10,
    )

    assert base_identity == changed_identity
    assert base.cache_key == changed.cache_key
    assert [
        sample.vector for sample in base.selected_sample_plan
    ] == [sample.vector for sample in changed.selected_sample_plan]


def test_diagnostic_and_formal_initializer_share_selected_vectors(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    templates = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
    cem_config = _build_pts_cem_config_from_config(config)
    continuous_config = _build_continuous_beta_cem_config(config.attack.pts_construction)

    formal = build_continuous_mlp_initial_sample_plan(
        config=config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        template_sessions=templates,
        generation_topk=10,
        force_rebuild=True,
    )
    from attack.pipeline.runs.run_pts_continuous_init_diagnostic import (
        run_continuous_init_diagnostic,
    )

    diagnostic = run_continuous_init_diagnostic(
        config=config,
        config_path=CONTINUOUS_FIXTURE,
        output_dir=tmp_path / "diagnostic",
        max_candidates=4,
        sample_sessions=2,
        template_sessions=templates,
        target_item=5334,
    )

    candidates = load_json(diagnostic.paths["initial_candidates"])
    assert [item["parameter_vector"] for item in candidates] == [
        sample.vector for sample in formal.selected_sample_plan
    ]
    assert [item["candidate_key"] for item in candidates] == [
        f"iter0_cand{index}" for index in range(4)
    ]


def test_formal_continuous_cem_applies_iter0_candidate_keys(monkeypatch) -> None:
    applied_keys: list[str] = []

    def fake_apply(**kwargs):
        applied_keys.append(str(kwargs["candidate_key"]))
        return PTSConstructionBatchResult(
            final_sessions=[[1, 2, 3]],
            per_session_records=[],
            summary={"num_sessions": 1},
        )

    monkeypatch.setattr(
        "attack.pts.continuous_cem.apply_pts_continuous_beta_construction_batch",
        fake_apply,
    )
    cem_config = PTSCEMConfig(
        iterations=1,
        population_schedule=[2],
        elite_ratio=0.5,
        init=PTSCEMInitConfig(mode=PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING),
    )
    continuous_config = PTSContinuousBetaCEMConfig(
        parameterization="tiny_mlp_log_beta_h2",
        initialization_mode=PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
    )
    trainer = PTSContinuousBetaCEMTrainer(
        cem_config=cem_config,
        continuous_config=continuous_config,
        initial_sample_plan=[
            ContinuousCandidateSampleSpec(
                vector=[0.0] * 13,
                sample_origin="continuous_mlp_two_pool_behavior_curve",
                sample_metadata={"candidate_key": "iter0_cand0"},
            ),
            ContinuousCandidateSampleSpec(
                vector=[0.1] * 13,
                sample_origin="continuous_mlp_two_pool_behavior_curve",
                sample_metadata={"candidate_key": "iter0_cand1"},
            ),
        ],
    )

    trainer.train(
        template_sessions=[[1, 2, 3], [4, 5, 6]],
        target_item=5334,
        poison_runner=None,
        evaluator_fn=lambda **kwargs: PTSCEMEvaluationResult(
            reward=1.0 - float(kwargs["candidate_id"]),
            reward_metrics={},
        ),
    )

    assert applied_keys == ["iter0_cand0", "iter0_cand1"]


def test_continuous_mlp_init_identity_changes_for_seed_and_init_settings(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    templates = [[1, 2, 3], [4, 5, 6, 7]]
    base_identity = continuous_mlp_init_identity_payload(
        config=config,
        template_sessions=templates,
    )
    changed_seed = replace(
        config,
        seeds=replace(config.seeds, position_opt_seed=config.seeds.position_opt_seed + 1),
    )
    changed_smoothing = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                config.attack.pts_construction,
                continuous_policy=replace(
                    config.attack.pts_construction.continuous_policy,
                    smoothing_epsilon=0.2,
                ),
            ),
        ),
    )
    changed_init = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                config.attack.pts_construction,
                cem=replace(
                    config.attack.pts_construction.cem,
                    init=replace(
                        config.attack.pts_construction.cem.init,
                        q_grid_size=7,
                    ),
                ),
            ),
        ),
    )

    assert base_identity["prefix_assignment"]["seed_source"] == "position_opt_seed"
    assert "resolved_init_seed" in base_identity["prefix_assignment"]
    assert continuous_mlp_init_cache_key(base_identity) != continuous_mlp_init_cache_key(
        continuous_mlp_init_identity_payload(config=changed_seed, template_sessions=templates)
    )
    assert continuous_mlp_init_cache_key(base_identity) != continuous_mlp_init_cache_key(
        continuous_mlp_init_identity_payload(config=changed_smoothing, template_sessions=templates)
    )
    assert continuous_mlp_init_cache_key(base_identity) != continuous_mlp_init_cache_key(
        continuous_mlp_init_identity_payload(config=changed_init, template_sessions=templates)
    )


def test_continuous_mlp_init_seed_resolver_supports_only_position_opt_seed(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    assert resolve_continuous_mlp_init_seed(config) == int(config.seeds.position_opt_seed)

    with pytest.raises(ValueError, match="position_opt_seed"):
        replace(
            config.attack.pts_construction.cem,
            seed_source="unsupported_seed",
        )


def test_continuous_construction_identity_includes_target_specific_init_key(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)
    fake_sessions_path = tmp_path / "fake_sessions.pkl"
    templates = [[1, 2, 3], [4, 5, 6, 7]]
    save_fake_sessions(templates, fake_sessions_path)
    init_identity = continuous_mlp_init_identity_payload(
        config=config,
        template_sessions=templates,
    )
    init_key = continuous_mlp_init_cache_key(init_identity)

    target_1 = build_pts_cem_shared_cache_identity(
        config,
        target_item=5334,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )
    target_2 = build_pts_cem_shared_cache_identity(
        config,
        target_item=999999,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=None,
    )

    assert target_1["pts_construction"]["initialization"]["cache_key"] == init_key
    assert target_1["pts_construction"]["initialization"]["target_independent"] is True
    assert continuous_mlp_init_cache_path(config, cache_key=init_key).parts[-3] == (
        CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE
    )
    assert pts_cem_shared_cache_key(target_1) != pts_cem_shared_cache_key(target_2)


def test_continuous_construction_identity_requires_readable_fake_sessions(
    tmp_path: Path,
) -> None:
    config = _small_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="fake sessions"):
        build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=tmp_path / "missing.pkl",
            poison_model_path=None,
        )
