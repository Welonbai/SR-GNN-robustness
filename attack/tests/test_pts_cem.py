from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSCEMSamplerConfig,
    PTSCEMUpdateConfig,
    PTSGroupedCEMTrainer,
)
from attack.pts.specs import get_default_pts_v1_specs


def _patch_generated_suffix(monkeypatch) -> None:
    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        return [100 + index for index in range(int(suffix_length))]

    monkeypatch.setattr(
        "attack.pts.suffix_constructor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )


def _regen_suffix3_evaluator(
    *,
    candidate_sessions,
    candidate_session_records,
    candidate_summary,
    iteration,
    candidate_id,
    candidate_seed,
    policy,
) -> PTSCEMEvaluationResult:
    reward = float(
        policy.group_probabilities["suffix_3plus"]["regenerate_residual_suffix"]
    )
    return PTSCEMEvaluationResult(
        reward=reward,
        reward_metrics={"suffix_3plus_regenerate": reward},
        metadata={
            "candidate_seed": int(candidate_seed),
            "session_count": int(len(candidate_sessions)),
            "record_count": int(len(candidate_session_records)),
            "fake_session_count": int(candidate_summary["fake_session_count"]),
        },
    )


def _train_small_result(monkeypatch, *, iterations: int = 3, population_size: int = 12):
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=iterations,
            population_size=population_size,
            elite_ratio=0.25,
            sampler=PTSCEMSamplerConfig(concentration_scale=5.0),
            update=PTSCEMUpdateConfig(
                smoothing=0.2,
                min_probability=0.03,
                max_probability=0.90,
            ),
            base_seed=17,
            save_top_k_candidates=4,
        ),
        specs=get_default_pts_v1_specs(),
    )
    return trainer.train(
        template_sessions=[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12],
            [13, 14],
        ],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_regen_suffix3_evaluator,
    )


def test_pts_cem_config_validation() -> None:
    with pytest.raises(ValueError, match="population_schedule length"):
        PTSCEMConfig(iterations=2, population_schedule=[8], base_seed=1)

    with pytest.raises(ValueError, match="Either population_schedule or population_size"):
        PTSCEMConfig(iterations=2, base_seed=1)

    with pytest.raises(ValueError, match="elite_ratio"):
        PTSCEMConfig(iterations=1, population_size=4, elite_ratio=0.0, base_seed=1)

    with pytest.raises(ValueError, match="smoothing"):
        PTSCEMUpdateConfig(smoothing=-0.1)

    with pytest.raises(ValueError, match="min_probability"):
        PTSCEMUpdateConfig(min_probability=0.4, max_probability=0.3)

    with pytest.raises(ValueError, match="candidate_seed_stride"):
        PTSCEMConfig(
            iterations=1,
            population_size=4,
            base_seed=1,
            candidate_seed_stride=0,
        )

    with pytest.raises(ValueError, match="save_top_k_candidates"):
        PTSCEMConfig(
            iterations=1,
            population_size=4,
            base_seed=1,
            save_top_k_candidates=-1,
        )


def test_pts_cem_population_schedule_evaluates_expected_candidate_count(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    calls: list[tuple[int, int]] = []

    def evaluator_fn(**kwargs) -> PTSCEMEvaluationResult:
        calls.append((int(kwargs["iteration"]), int(kwargs["candidate_id"])))
        return _regen_suffix3_evaluator(**kwargs)

    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=3,
            population_schedule=[16, 8, 8],
            elite_ratio=0.25,
            base_seed=3,
        ),
        specs=get_default_pts_v1_specs(),
    )

    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7], [8, 9]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=evaluator_fn,
    )

    assert len(calls) == 32
    assert [iteration.population_size for iteration in result.iteration_results] == [
        16,
        8,
        8,
    ]


def test_pts_cem_floor_style_elite_counts(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=2,
            population_schedule=[16, 8],
            elite_ratio=0.25,
            base_seed=5,
        ),
        specs=get_default_pts_v1_specs(),
    )

    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7], [8, 9]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_regen_suffix3_evaluator,
    )

    assert [iteration.elite_count for iteration in result.iteration_results] == [4, 2]
    for iteration in result.iteration_results:
        assert len(iteration.elite_candidate_keys) == iteration.elite_count


def test_pts_cem_updates_policy_toward_fake_evaluator_reward(monkeypatch) -> None:
    result = _train_small_result(monkeypatch)

    initial_probability = result.policy_history[0]["group_probabilities"]["suffix_3plus"][
        "regenerate_residual_suffix"
    ]
    final_probability = result.final_policy.group_probabilities["suffix_3plus"][
        "regenerate_residual_suffix"
    ]
    best_probability = result.best_candidate.policy.group_probabilities["suffix_3plus"][
        "regenerate_residual_suffix"
    ]

    assert initial_probability == pytest.approx(0.25)
    assert best_probability > initial_probability
    assert final_probability > initial_probability


def test_pts_cem_updated_policies_are_normalized_and_bounded(monkeypatch) -> None:
    result = _train_small_result(monkeypatch)

    for policy_payload in result.policy_history:
        for probabilities in policy_payload["group_probabilities"].values():
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(0.03 - 1e-8 <= value <= 0.90 + 1e-8 for value in probabilities.values())


def test_pts_cem_global_best_and_top_k_are_consistent(monkeypatch) -> None:
    result = _train_small_result(monkeypatch)
    all_candidates = [
        candidate
        for iteration in result.iteration_results
        for candidate in iteration.candidates
    ]

    assert result.top_candidates
    assert result.best_candidate is result.top_candidates[0]
    assert [candidate.reward for candidate in result.top_candidates] == sorted(
        [candidate.reward for candidate in result.top_candidates],
        reverse=True,
    )
    assert sum(1 for candidate in all_candidates if candidate.selected_as_global_best) == 1
    assert result.best_candidate.selected_as_global_best is True


def test_pts_cem_dynamic_masks_remain_executor_time_only(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=1,
            population_size=6,
            elite_ratio=0.5,
            base_seed=11,
        ),
        specs=get_default_pts_v1_specs(),
    )

    result = trainer.train(
        template_sessions=[[1, 2], [3, 4], [5, 6]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_regen_suffix3_evaluator,
    )

    for iteration in result.iteration_results:
        for candidate in iteration.candidates:
            assert "consume_one_keep_rest" in candidate.policy.group_probabilities["suffix_1"]
            for record in candidate.per_session_records:
                assert record["suffix_len_group"] == "suffix_1"
                assert record["dynamic_mask_disable_consume_one"] is True
                assert record["action"] != "consume_one_keep_rest"
