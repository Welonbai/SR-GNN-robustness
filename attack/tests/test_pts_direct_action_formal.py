from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.pipeline.runs.run_pts_construction_cem import (
    _build_direct_action_mlp_cem_config,
    _build_pts_cem_config_from_config,
    _validate_pts_construction_run_config,
)
from attack.pts.cem import PTSCEMEvaluationResult
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
    / "diginetica_valbest_attack_pts_construction_direct_action_mlp_cem_ratio1_target5334.yaml"
)


def test_direct_action_smoke_config_loads_and_records_cem_init() -> None:
    config = load_config(SMOKE_CONFIG)
    _validate_pts_construction_run_config(config)
    pts_config = config.attack.pts_construction
    assert pts_config.method == "direct_action_mlp_cem"
    assert pts_config.direct_action_policy.length_feature == "z_score"

    direct_config = _build_direct_action_mlp_cem_config(pts_config)
    assert direct_config.length_feature_mode == DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
    assert direct_config.initial_std == pytest.approx(1.0)
    assert direct_config.elite_min_std == pytest.approx(0.25)
    assert direct_config.elite_std_scale == pytest.approx(1.0)


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
    assert len(policy_after["elite_mean"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert len(policy_after["elite_std"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert len(policy_after["resample_std"]) == len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    assert result.best_candidate.candidate_key == "iter1_cand0"
    assert result.best_candidate.sample_metadata["direct_action_action_summary"]
    assert result.best_candidate.policy.to_dict()["parameterization"] == DIRECT_ACTION_POLICY_MLP_H2


def test_direct_action_elite_count_floor_and_population_one() -> None:
    assert direct_action_elite_count(16, 0.25) == 4
    assert direct_action_elite_count(8, 0.25) == 2
    assert direct_action_elite_count(1, 0.25) == 1
    with pytest.raises(ValueError):
        direct_action_elite_count(0, 0.25)
