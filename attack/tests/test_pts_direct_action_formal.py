from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import save_fake_sessions
from attack.common.config import (
    PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN,
    PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN,
    PTSCEMUpdateRuntimeConfig,
    load_config,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    build_pts_cem_shared_cache_identity,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _build_direct_action_mlp_cem_config,
    _build_pts_cem_config_from_config,
    _validate_pts_construction_run_config,
)
from attack.pts.cem import PTSCEMCandidateResult, PTSCEMEvaluationResult
from attack.pts.artifacts import write_pts_cem_artifacts
from attack.pts.direct_action_cem import (
    PTSDirectActionMLPCEMTrainer,
    direct_action_elite_count,
)
from attack.pts.direct_action_executor import (
    DirectActionContextStats,
    DirectActionFormalSessionContext,
    _action_sample_seed_payload,
    _generation_seed_payload,
    apply_pts_direct_action_construction_batch,
    build_direct_action_formal_session_contexts,
)
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_MLP_H2_PARAMETER_NAMES,
    DIRECT_ACTION_POLICY_MLP_H2,
    DirectAction,
    DirectActionMLPPolicy,
)


SMOKE_CONFIG = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample.yaml"
)
CONTINUOUS_CONFIG = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_target5334.yaml"
)
GROUPED_CONFIG = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_internal_sample.yaml"
)


def test_common_pts_update_config_accepts_old_and_new_modes() -> None:
    assert PTSCEMUpdateRuntimeConfig(mode="standard").mode == "standard"
    assert (
        PTSCEMUpdateRuntimeConfig(
            mode=PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN,
        ).mode
        == PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN
    )
    assert (
        PTSCEMUpdateRuntimeConfig(
            mode=PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN,
        ).mode
        == PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN
    )


def test_direct_action_smoke_config_loads_and_records_cem_init() -> None:
    config = load_config(SMOKE_CONFIG)
    _validate_pts_construction_run_config(config)
    pts_config = config.attack.pts_construction
    assert pts_config.method == "direct_action_mlp_cem"
    assert pts_config.direct_action_policy.length_feature == "z_score"

    direct_config = _build_direct_action_mlp_cem_config(pts_config)
    assert direct_config.length_feature_mode == DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
    assert not hasattr(pts_config.direct_action_policy, "initial_std")
    assert not hasattr(direct_config, "initial_std")
    assert direct_config.elite_min_std == pytest.approx(0.25)
    assert not hasattr(direct_config, "elite_std_scale")
    assert hasattr(pts_config.cem.update, "elite_std_scale")


def test_direct_action_old_exposed_std_fields_are_rejected(tmp_path: Path) -> None:
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    old_policy_config = text.replace(
        "      length_feature: z_score\n",
        "      length_feature: z_score\n      initial_std: 1.0\n",
    )
    old_policy_path = tmp_path / "old_policy.yaml"
    old_policy_path.write_text(old_policy_config, encoding="utf-8")
    with pytest.raises(ValueError, match="initial_std"):
        load_config(old_policy_path)

    old_update_config = text.replace(
        "        elite_min_std: 0.25\n",
        "        elite_min_std: 0.25\n        elite_std_scale: 1.0\n",
    )
    old_update_path = tmp_path / "old_update.yaml"
    old_update_path.write_text(old_update_config, encoding="utf-8")
    with pytest.raises(ValueError, match="elite_std_scale"):
        load_config(old_update_path)


def test_direct_action_rejects_old_elite_centered_gaussian_mode(
    tmp_path: Path,
) -> None:
    old_mode_path = tmp_path / "old_mode.yaml"
    old_mode_path.write_text(
        SMOKE_CONFIG.read_text(encoding="utf-8").replace(
            "        mode: elite_centered_empirical_gaussian",
            "        mode: elite_centered_gaussian",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="elite_centered_empirical_gaussian"):
        load_config(old_mode_path)


def test_grouped_and_continuous_configs_still_accept_old_elite_centered_mode(
    tmp_path: Path,
) -> None:
    continuous_path = tmp_path / "continuous_old_mode.yaml"
    continuous_path.write_text(
        CONTINUOUS_CONFIG.read_text(encoding="utf-8").replace(
            "      update:\n",
            "      update:\n        mode: elite_centered_gaussian\n",
        ),
        encoding="utf-8",
    )
    continuous = load_config(continuous_path)
    assert (
        continuous.attack.pts_construction.cem.update.mode
        == PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN
    )

    grouped_path = tmp_path / "grouped_old_mode.yaml"
    grouped_path.write_text(
        GROUPED_CONFIG.read_text(encoding="utf-8").replace(
            "      update:\n",
            "      update:\n        mode: elite_centered_gaussian\n",
        ),
        encoding="utf-8",
    )
    grouped = load_config(grouped_path)
    assert (
        grouped.attack.pts_construction.cem.update.mode
        == PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN
    )


def test_direct_action_contexts_are_target_independent_and_z_stats() -> None:
    sessions = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    first, stats = build_direct_action_formal_session_contexts(
        template_sessions=sessions,
        base_seed=123,
    )
    second, second_stats = build_direct_action_formal_session_contexts(
        template_sessions=sessions,
        base_seed=123,
    )

    assert first == second
    assert stats == second_stats
    lengths = [context.residual_suffix_len for context in first]
    mean = sum(lengths) / len(lengths)
    expected_std = (sum((value - mean) ** 2 for value in lengths) / len(lengths)) ** 0.5
    assert stats.mean_m == pytest.approx(mean)
    assert stats.raw_std_m == pytest.approx(expected_std)
    assert stats.std_m == pytest.approx(expected_std if expected_std > 0 else 1.0)
    assert all(context.prefix and context.residual_suffix for context in first)


def test_direct_action_seed_payloads_include_required_target_specific_fields() -> None:
    sample_a = _action_sample_seed_payload(
        base_seed=1,
        target_item=99,
        iteration=2,
        candidate_key="iter2_cand3",
        fake_session_index=4,
        tag="formal_action_sampling",
    )
    sample_b = _action_sample_seed_payload(
        base_seed=1,
        target_item=100,
        iteration=2,
        candidate_key="iter2_cand3",
        fake_session_index=4,
        tag="formal_action_sampling",
    )
    assert sample_a["seed"] != sample_b["seed"]
    assert sample_a["fields"] == [
        "1",
        "99",
        "2",
        "iter2_cand3",
        "4",
        "formal_action_sampling",
    ]

    generation = _generation_seed_payload(
        base_seed=1,
        target_item=99,
        iteration=2,
        candidate_key="iter2_cand3",
        fake_session_index=4,
        consume_count=1,
        generated_length=3,
        tag="formal_generation",
    )
    assert generation["fields"] == [
        "1",
        "99",
        "2",
        "iter2_cand3",
        "4",
        "1",
        "3",
        "formal_generation",
    ]


def test_direct_action_executor_materializes_generate_with_length_verification(
    monkeypatch,
) -> None:
    context = DirectActionFormalSessionContext(
        fake_session_index=0,
        template_session=[1, 2, 3, 4],
        anchor_position=2,
        prefix=[1, 2],
        residual_suffix=[3, 4],
    )
    stats = DirectActionContextStats(
        mean_m=2.0,
        std_m=1.0,
        raw_std_m=0.0,
        max_m=2,
        context_seed=123,
        prefix_rng_tag="formal_prefix",
    )
    monkeypatch.setattr(
        "attack.pts.direct_action_executor.sample_direct_action_categorical",
        lambda **kwargs: DirectAction("generate", 1),
    )

    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        assert prefix == [1, 2, 99]
        assert suffix_length == 1
        return [700]

    monkeypatch.setattr(
        "attack.pts.direct_action_executor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )
    result = apply_pts_direct_action_construction_batch(
        session_contexts=[context],
        context_stats=stats,
        target_item=99,
        policy=DirectActionMLPPolicy.from_vector([0.0] * 15),
        base_seed=7,
        iteration=0,
        candidate_key="iter0_cand0",
        poison_runner=object(),
        generation_topk=10,
    )

    assert result.final_sessions == [[1, 2, 99, 700]]
    record = result.per_session_records[0]
    assert record["selected_action_type"] == "generate"
    assert record["consume_count"] == 1
    assert record["generated_length"] == 1
    assert record["actual_generated_length"] == 1
    assert record["generated_suffix_length"] == 1


def test_direct_action_trainer_uses_reward_elites_and_records_update_metadata(
    monkeypatch,
) -> None:
    config = load_config(SMOKE_CONFIG)
    cem_config = _build_pts_cem_config_from_config(config)
    cem_config = type(cem_config)(
        iterations=2,
        population_schedule=[4, 2],
        elite_ratio=0.25,
        sampler=cem_config.sampler,
        update=cem_config.update,
        init=cem_config.init,
        resampling=cem_config.resampling,
        base_seed=cem_config.base_seed,
        candidate_seed_stride=cem_config.candidate_seed_stride,
        save_top_k_candidates=2,
    )
    trainer = PTSDirectActionMLPCEMTrainer(
        cem_config=cem_config,
        direct_action_config=_build_direct_action_mlp_cem_config(
            config.attack.pts_construction
        ),
        generation_topk=10,
    )
    monkeypatch.setattr(
        "attack.pts.direct_action_executor.generate_poison_model_suffix",
        lambda *, runner, prefix, suffix_length, topk, rng: [700] * int(suffix_length),
    )

    def evaluator_fn(
        *,
        candidate_sessions,
        candidate_session_records,
        candidate_summary,
        iteration,
        candidate_id,
        candidate_seed,
        policy,
    ):
        del candidate_sessions, candidate_session_records, candidate_summary, candidate_seed, policy
        reward = 10.0 - float(candidate_id) + float(iteration)
        return PTSCEMEvaluationResult(
            reward=reward,
            reward_metrics={"reward": reward},
            metadata={},
        )

    result = trainer.train(
        template_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=evaluator_fn,
    )

    assert len(result.iteration_results) == 2
    assert result.iteration_results[0].elite_count == 2
    assert result.iteration_results[0].elite_candidate_keys == ["iter0_cand0", "iter0_cand1"]
    policy_after = result.iteration_results[0].policy_after
    assert policy_after["elite_rewards"] == [10.0, 9.0]
    assert policy_after["cem_init"] == {
        "mode": "standard_normal",
        "parameter_space": "standardized_policy_parameter_space",
    }
    assert policy_after["cem_update"]["mode"] == "elite_centered_empirical_gaussian"
    assert policy_after["cem_update"]["anti_collapse_min_std"] == pytest.approx(0.25)
    assert not _contains_key(policy_after, "initial_std")
    assert not _contains_key(policy_after, "elite_std_scale")
    assert len(policy_after["elite_mean"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert len(policy_after["elite_std"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert len(policy_after["resample_std"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert result.best_candidate.candidate_key == "iter1_cand0"
    assert result.best_candidate.sample_metadata["direct_action_action_summary"]
    assert result.best_candidate.policy.to_dict()["parameterization"] == DIRECT_ACTION_POLICY_MLP_H2
    assert not _contains_key(result.best_candidate.sample_metadata, "initial_std")
    assert not _contains_key(result.best_candidate.sample_metadata, "elite_std_scale")


def test_direct_action_iter0_standard_normal_sampling_is_deterministic() -> None:
    config = load_config(SMOKE_CONFIG)
    cem_config = _build_pts_cem_config_from_config(config)
    cem_config = type(cem_config)(
        iterations=1,
        population_schedule=[3],
        elite_ratio=0.25,
        sampler=cem_config.sampler,
        update=cem_config.update,
        init=cem_config.init,
        resampling=cem_config.resampling,
        base_seed=777,
        candidate_seed_stride=cem_config.candidate_seed_stride,
        save_top_k_candidates=2,
    )
    trainer_a = PTSDirectActionMLPCEMTrainer(
        cem_config=cem_config,
        direct_action_config=_build_direct_action_mlp_cem_config(
            config.attack.pts_construction
        ),
    )
    trainer_b = PTSDirectActionMLPCEMTrainer(
        cem_config=cem_config,
        direct_action_config=_build_direct_action_mlp_cem_config(
            config.attack.pts_construction
        ),
    )

    plan_a = trainer_a._candidate_sample_plan(
        iteration=0,
        population_size=3,
        mean=[0.0] * 15,
        std=[1.0] * 15,
    )
    plan_b = trainer_b._candidate_sample_plan(
        iteration=0,
        population_size=3,
        mean=[0.0] * 15,
        std=[1.0] * 15,
    )

    assert len(plan_a) == 3
    assert [item.vector for item in plan_a] == [item.vector for item in plan_b]
    assert all(len(item.vector) == 15 for item in plan_a)
    assert all(
        item.sample_metadata["cem_init"]["mode"] == "standard_normal"
        for item in plan_a
    )
    assert not any(_contains_key(item.sample_metadata, "initial_std") for item in plan_a)


def test_direct_action_elite_update_uses_empirical_std_without_scale() -> None:
    config = load_config(SMOKE_CONFIG)
    cem_config = _build_pts_cem_config_from_config(config)
    trainer = PTSDirectActionMLPCEMTrainer(
        cem_config=cem_config,
        direct_action_config=_build_direct_action_mlp_cem_config(
            config.attack.pts_construction
        ),
    )
    left = [0.0] * 15
    right = [0.0] * 15
    right[0] = 0.2
    right[1] = 1.0
    update = trainer._updated_distribution_from_elites(
        elites=[
            _candidate_with_policy(0, left),
            _candidate_with_policy(1, right),
        ]
    )

    assert update.elite_mean[0] == pytest.approx(0.1)
    assert update.elite_mean[1] == pytest.approx(0.5)
    assert update.elite_std[0] == pytest.approx(0.1)
    assert update.elite_std[1] == pytest.approx(0.5)
    assert update.resample_std[0] == pytest.approx(0.25)
    assert update.resample_std[1] == pytest.approx(0.5)


def test_direct_action_elite_count_floor_and_population_one() -> None:
    assert direct_action_elite_count(16, 0.25) == 4
    assert direct_action_elite_count(8, 0.25) == 2
    assert direct_action_elite_count(1, 0.25) == 1
    with pytest.raises(ValueError):
        direct_action_elite_count(0, 0.25)


def test_direct_action_identity_uses_elite_min_std_not_removed_std_fields(
    tmp_path: Path,
) -> None:
    fake_sessions_path = tmp_path / "fake_sessions.pkl"
    save_fake_sessions([[1, 2, 3], [4, 5, 6]], fake_sessions_path)
    base = load_config(SMOKE_CONFIG)
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(
        SMOKE_CONFIG.read_text(encoding="utf-8").replace(
            "        elite_min_std: 0.25",
            "        elite_min_std: 0.5",
        ),
        encoding="utf-8",
    )
    changed = load_config(changed_path)

    base_identity = build_pts_cem_shared_cache_identity(
        base,
        target_item=5334,
        fake_sessions_path=fake_sessions_path,
    )
    changed_identity = build_pts_cem_shared_cache_identity(
        changed,
        target_item=5334,
        fake_sessions_path=fake_sessions_path,
    )

    assert base_identity != changed_identity
    identity_text = repr(base_identity)
    assert "elite_min_std" in identity_text
    assert "standard_normal" in identity_text
    assert "elite_centered_empirical_gaussian" in identity_text
    assert "initial_std" not in identity_text
    assert "elite_std_scale" not in identity_text


def test_direct_action_artifacts_do_not_serialize_common_update_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(SMOKE_CONFIG)
    cem_config = _build_pts_cem_config_from_config(config)
    cem_config = type(cem_config)(
        iterations=1,
        population_schedule=[2],
        elite_ratio=0.5,
        sampler=cem_config.sampler,
        update=cem_config.update,
        init=cem_config.init,
        resampling=cem_config.resampling,
        base_seed=cem_config.base_seed,
        candidate_seed_stride=cem_config.candidate_seed_stride,
        save_top_k_candidates=1,
    )
    trainer = PTSDirectActionMLPCEMTrainer(
        cem_config=cem_config,
        direct_action_config=_build_direct_action_mlp_cem_config(
            config.attack.pts_construction
        ),
        generation_topk=10,
    )
    monkeypatch.setattr(
        "attack.pts.direct_action_executor.generate_poison_model_suffix",
        lambda *, runner, prefix, suffix_length, topk, rng: [700] * int(suffix_length),
    )

    def evaluator_fn(
        *,
        candidate_sessions,
        candidate_session_records,
        candidate_summary,
        iteration,
        candidate_id,
        candidate_seed,
        policy,
    ):
        del candidate_sessions, candidate_session_records, candidate_summary, iteration, candidate_seed, policy
        reward = float(10 - int(candidate_id))
        return PTSCEMEvaluationResult(
            reward=reward,
            reward_metrics={"reward": reward},
            metadata={},
        )

    result = trainer.train(
        template_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=evaluator_fn,
    )
    paths = write_pts_cem_artifacts(
        result=result,
        output_dir=tmp_path,
        save_top_candidate_sessions=True,
        save_per_session_records=True,
    )
    artifact_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for key, path in paths.items()
        if key
        in {
            "pts_cem_trace",
            "pts_policy_history",
            "pts_best_policy",
            "pts_final_policy",
            "pts_top_candidates",
            "pts_top_candidate_policies",
        }
    )

    assert "elite_centered_empirical_gaussian" in artifact_text
    assert "anti_collapse_min_std" in artifact_text
    assert "elite_std_scale" not in artifact_text
    assert "initial_std" not in artifact_text


def _candidate_with_policy(candidate_id: int, vector: list[float]) -> PTSCEMCandidateResult:
    return PTSCEMCandidateResult(
        iteration=0,
        candidate_id=int(candidate_id),
        candidate_seed=100 + int(candidate_id),
        policy=DirectActionMLPPolicy.from_vector(vector),
        reward=float(candidate_id),
        reward_metrics={},
        evaluator_metadata={},
        construction_summary={},
        per_session_records=[],
        final_sessions=[],
    )


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, key) for item in value)
    return False
