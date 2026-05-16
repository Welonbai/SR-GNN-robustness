from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.cem import PTSCEMConfig, PTSCEMEvaluationResult
from attack.pts.continuous_cem import (
    PTSContinuousBetaCEMConfig,
    PTSContinuousBetaCEMTrainer,
    build_continuous_beta_initial_sample_plan,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
    CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES,
)


def test_continuous_cem_runs_behavior_covering_init_and_uses_global_best(monkeypatch) -> None:
    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        return [800 + index for index in range(int(suffix_length))]

    monkeypatch.setattr(
        "attack.pts.continuous_executor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )

    def evaluator_fn(**kwargs) -> PTSCEMEvaluationResult:
        policy = kwargs["policy"]
        reward = float(policy.a0) - abs(float(policy.c0)) * 0.01
        return PTSCEMEvaluationResult(
            reward=reward,
            reward_metrics={"score": reward},
            metadata={"candidate_id": kwargs["candidate_id"]},
        )

    trainer = PTSContinuousBetaCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=2,
            population_schedule=[4, 2],
            elite_ratio=0.5,
            base_seed=123,
            candidate_seed_stride=100,
            save_top_k_candidates=2,
        ),
        continuous_config=PTSContinuousBetaCEMConfig(
            parameter_bounds=(-5.0, 5.0),
            initial_std=2.0,
            min_std=0.25,
        ),
        generation_topk=10,
    )
    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=evaluator_fn,
    )

    assert len(result.iteration_results) == 2
    assert [candidate.sample_origin for candidate in result.iteration_results[0].candidates] == [
        "continuous_beta_behavior_covering",
        "continuous_beta_behavior_covering",
        "continuous_beta_behavior_covering",
        "continuous_beta_behavior_covering",
    ]
    assert result.final_policy.to_dict() == result.best_candidate.policy.to_dict()
    assert result.best_candidate.selected_as_global_best is True
    assert result.top_candidates
    assert "parameter_vector" in result.iteration_results[0].candidates[0].sample_metadata
    json.dumps(result.policy_history)


def test_continuous_cem_tiny_mlp_behavior_covering_init_plan() -> None:
    plan = build_continuous_beta_initial_sample_plan(
        cem_config=PTSCEMConfig(
            iterations=1,
            population_schedule=[4],
            base_seed=123,
        ),
        continuous_config=PTSContinuousBetaCEMConfig(
            parameterization=CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2,
            parameter_bounds=(-5.0, 5.0),
            initial_std=2.0,
            min_std=0.25,
        ),
        population_size=4,
    )

    assert len(plan) == 4
    assert all(sample.sample_origin == "continuous_beta_behavior_covering" for sample in plan)
    assert all(len(sample.vector) == len(CONTINUOUS_BETA_TINY_MLP_H2_PARAMETER_NAMES) for sample in plan)
    assert [sample.sample_metadata["prototype_name"] for sample in plan] == [
        "tiny_near_zero_consume_preserve",
        "tiny_near_zero_consume_generate",
        "tiny_near_one_consume_stop",
        "tiny_middle_consume_preserve",
    ]
