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
    PTSCEMInitConfig,
    PTSCEMResamplingConfig,
    PTSCEMSamplerConfig,
    PTSCEMUpdateConfig,
    PTSGroupedCEMTrainer,
    _allocate_children_to_elites,
)
from attack.pts.policy import CONSUME_ONE_ACTION_NAMES
from attack.pts.space_filling import (
    MANDATORY_VERTEX_NAMES,
    PTSSpaceFillingConfig,
    build_vertex_stratified_initial_population,
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


def _assert_consume_one_actions_are_ragged(policy) -> None:
    for action in CONSUME_ONE_ACTION_NAMES:
        assert action not in policy.group_probabilities["suffix_1"]
        assert action in policy.group_probabilities["suffix_2"]
        assert action in policy.group_probabilities["suffix_3plus"]


def test_pts_cem_config_validation() -> None:
    assert (
        PTSCEMConfig(iterations=1, population_size=2, base_seed=1)
        .resampling
        .mode
        == "standard"
    )

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

    with pytest.raises(ValueError, match="resampling.mode"):
        PTSCEMResamplingConfig(mode="mixed")

    with pytest.raises(ValueError, match="local_concentration_scale"):
        PTSCEMResamplingConfig(local_concentration_scale=0.0)


def test_pts_cem_vertex_stratified_mandatory_requires_c1_generate_spec() -> None:
    old_four_specs = tuple(
        spec
        for spec in get_default_pts_v1_specs()
        if spec.name != "consume_one_generate_continuation"
    )

    with pytest.raises(
        ValueError,
        match=(
            "vertex_stratified_space_filling with mandatory_enabled=true requires "
            "consume_one_generate_continuation"
        ),
    ):
        PTSGroupedCEMTrainer(
            cem_config=PTSCEMConfig(
                iterations=3,
                population_schedule=[16, 8, 8],
                init=PTSCEMInitConfig(mode="vertex_stratified_space_filling"),
                base_seed=1,
            ),
            specs=old_four_specs,
        )


def test_pts_cem_elite_child_allocation() -> None:
    assert _allocate_children_to_elites(population_size=8, elite_count=4) == [
        2,
        2,
        2,
        2,
    ]
    assert _allocate_children_to_elites(population_size=8, elite_count=2) == [4, 4]
    assert _allocate_children_to_elites(population_size=10, elite_count=4) == [
        3,
        3,
        2,
        2,
    ]
    assert _allocate_children_to_elites(population_size=3, elite_count=4) == [
        1,
        1,
        1,
        0,
    ]

    with pytest.raises(ValueError, match="elite_count"):
        _allocate_children_to_elites(population_size=3, elite_count=0)


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
    assert all(
        candidate.sample_origin == "global_policy"
        for iteration in result.iteration_results
        for candidate in iteration.candidates
    )


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

    assert initial_probability == pytest.approx(0.2)
    assert best_probability > initial_probability
    assert final_probability > initial_probability


def test_pts_cem_updated_policies_are_normalized_and_bounded(monkeypatch) -> None:
    result = _train_small_result(monkeypatch)

    for policy_payload in result.policy_history:
        for probabilities in policy_payload["group_probabilities"].values():
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(0.03 - 1e-8 <= value <= 0.90 + 1e-8 for value in probabilities.values())
        for action in CONSUME_ONE_ACTION_NAMES:
            assert action not in policy_payload["group_probabilities"]["suffix_1"]
            assert action in policy_payload["group_probabilities"]["suffix_2"]
            assert action in policy_payload["group_probabilities"]["suffix_3plus"]


def test_pts_cem_candidate_policies_are_ragged(monkeypatch) -> None:
    result = _train_small_result(monkeypatch, iterations=1, population_size=6)

    for candidate in result.iteration_results[0].candidates:
        _assert_consume_one_actions_are_ragged(candidate.policy)
        for probabilities in candidate.policy.group_probabilities.values():
            assert sum(probabilities.values()) == pytest.approx(1.0)


def test_pts_cem_vertex_stratified_iteration0_uses_fixed_helper_policies(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    specs = get_default_pts_v1_specs()
    base_seed = 101
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=3,
            population_schedule=[16, 8, 8],
            elite_ratio=0.25,
            init=PTSCEMInitConfig(mode="vertex_stratified_space_filling"),
            resampling=PTSCEMResamplingConfig(mode="elite_centered"),
            base_seed=base_seed,
            save_top_k_candidates=4,
        ),
        specs=specs,
    )
    expected = build_vertex_stratified_initial_population(
        config=PTSSpaceFillingConfig(
            seed=base_seed,
            min_probability=0.03,
            max_probability=0.90,
        ),
        valid_actions_by_group=trainer.valid_actions_by_group,
        enabled_actions=[spec.name for spec in specs],
    )
    initial_policy = trainer._initial_policy()
    sample_plan = trainer._candidate_sample_plan(
        iteration=0,
        population_size=16,
        current_policy=initial_policy,
        previous_elites=[],
    )

    assert len(sample_plan) == 16
    assert len(expected) == 16
    assert [sample.sample_origin for sample in expected[:5]] == [
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
    ]
    assert [sample.vertex_name for sample in expected[:5]] == list(MANDATORY_VERTEX_NAMES)
    for sample_spec, expected_sample in zip(sample_plan, expected):
        assert sample_spec.fixed_policy is not None
        assert sample_spec.fixed_policy.to_dict() == expected_sample.policy.to_dict()
        assert sample_spec.sample_origin == expected_sample.sample_origin
        assert sample_spec.sample_metadata["fixed_policy"] is True
        assert sample_spec.sample_metadata["sampled_policy_projection_enabled"] is True
        assert sample_spec.sample_metadata["sampled_policy_min_probability"] == pytest.approx(0.03)
        assert sample_spec.sample_metadata["sampled_policy_max_probability"] == pytest.approx(0.90)

    result = trainer.train(
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
    iter0 = result.iteration_results[0]

    assert [candidate.sample_origin for candidate in iter0.candidates[:5]] == [
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
        "mandatory_vertex",
    ]
    assert [
        candidate.sample_metadata["vertex_name"]
        for candidate in iter0.candidates[:5]
    ] == list(MANDATORY_VERTEX_NAMES)
    assert [candidate.sample_origin for candidate in iter0.candidates[5:12]] == [
        "extreme_maximin"
    ] * 7
    assert [candidate.sample_origin for candidate in iter0.candidates[12:15]] == [
        "moderate_maximin"
    ] * 3
    assert iter0.candidates[15].sample_origin == "balanced"
    for candidate, expected_sample in zip(iter0.candidates, expected):
        assert candidate.policy.to_dict() == expected_sample.policy.to_dict()
        assert candidate.sample_metadata["sampled_policy_projection_enabled"] is True
        assert candidate.sample_metadata["sampled_policy_min_probability"] == pytest.approx(0.03)
        assert candidate.sample_metadata["sampled_policy_max_probability"] == pytest.approx(0.90)
        _assert_consume_one_actions_are_ragged(candidate.policy)
        assert list(candidate.policy.group_probabilities["suffix_1"]) == [
            "keep_residual_suffix",
            "regenerate_residual_suffix",
            "consume_all_stop",
        ]
        for probabilities in candidate.policy.group_probabilities.values():
            assert sum(probabilities.values()) == pytest.approx(1.0)
            assert all(0.03 - 1e-8 <= value <= 0.90 + 1e-8 for value in probabilities.values())


def test_pts_cem_vertex_stratified_elite_centered_children_have_metadata(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=3,
            population_schedule=[16, 8, 8],
            elite_ratio=0.25,
            init=PTSCEMInitConfig(mode="vertex_stratified_space_filling"),
            resampling=PTSCEMResamplingConfig(
                mode="elite_centered",
                local_concentration_scale=30.0,
            ),
            base_seed=103,
            save_top_k_candidates=4,
        ),
        specs=get_default_pts_v1_specs(),
    )

    result = trainer.train(
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

    iter0, iter1, iter2 = result.iteration_results
    assert len(iter0.candidates) == 16
    assert all(candidate.sample_origin == "elite_centered" for candidate in iter1.candidates)
    assert all(candidate.sample_origin == "elite_centered" for candidate in iter2.candidates)
    assert all(candidate.parent_candidate_key for candidate in iter1.candidates)
    assert all(
        candidate.sample_metadata["local_concentration_scale"] == pytest.approx(30.0)
        for candidate in iter1.candidates
    )
    assert set(_parent_key_counts(iter1.candidates)) == set(iter0.elite_candidate_keys)


def test_pts_cem_elite_centered_resampling_origins_and_parents(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=3,
            population_schedule=[16, 8, 8],
            elite_ratio=0.25,
            sampler=PTSCEMSamplerConfig(concentration_scale=5.0),
            resampling=PTSCEMResamplingConfig(
                mode="elite_centered",
                local_concentration_scale=30.0,
            ),
            base_seed=19,
            save_top_k_candidates=4,
        ),
        specs=get_default_pts_v1_specs(),
    )

    result = trainer.train(
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

    iter0, iter1, iter2 = result.iteration_results
    assert all(
        candidate.sample_origin == "initial_global_policy"
        for candidate in iter0.candidates
    )
    assert all(candidate.sample_origin == "elite_centered" for candidate in iter1.candidates)
    assert all(candidate.sample_origin == "elite_centered" for candidate in iter2.candidates)
    assert not any(
        candidate.sample_origin == "global_policy"
        for iteration in (iter1, iter2)
        for candidate in iteration.candidates
    )

    iter1_parent_counts = _parent_key_counts(iter1.candidates)
    iter2_parent_counts = _parent_key_counts(iter2.candidates)
    assert list(iter1_parent_counts.values()) == [2, 2, 2, 2]
    assert list(iter2_parent_counts.values()) == [4, 4]

    iter0_elite_keys = set(iter0.elite_candidate_keys)
    iter1_elite_keys = set(iter1.elite_candidate_keys)
    assert set(iter1_parent_counts) == iter0_elite_keys
    assert set(iter2_parent_counts) == iter1_elite_keys

    for iteration_index, iteration in enumerate((iter1, iter2), start=1):
        for candidate in iteration.candidates:
            assert candidate.parent_candidate_key
            assert candidate.parent_iteration == iteration_index - 1
            assert candidate.parent_candidate_id is not None
            assert candidate.parent_reward is not None
            assert candidate.parent_rank_among_elites is not None
            assert candidate.parent_rank_among_elites >= 1
            _assert_consume_one_actions_are_ragged(candidate.policy)
            assert candidate.sample_metadata["sample_origin"] == "elite_centered"
            assert candidate.sample_metadata["parent_candidate_key"]
            assert (
                candidate.sample_metadata["local_concentration_scale"]
                == pytest.approx(30.0)
            )
            for probabilities in candidate.policy.group_probabilities.values():
                assert sum(probabilities.values()) == pytest.approx(1.0)

    assert result.best_candidate is result.top_candidates[0]
    all_candidates = [
        candidate
        for iteration in result.iteration_results
        for candidate in iteration.candidates
    ]
    assert sum(1 for candidate in all_candidates if candidate.selected_as_global_best) == 1


def _parent_key_counts(candidates):
    counts: dict[str, int] = {}
    for candidate in candidates:
        assert candidate.parent_candidate_key is not None
        counts[candidate.parent_candidate_key] = (
            counts.get(candidate.parent_candidate_key, 0) + 1
        )
    return counts


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


def test_pts_cem_suffix_1_policies_do_not_sample_duplicate_action(monkeypatch) -> None:
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
            for action in CONSUME_ONE_ACTION_NAMES:
                assert action not in candidate.policy.group_probabilities["suffix_1"]
            for record in candidate.per_session_records:
                assert record["suffix_len_group"] == "suffix_1"
                assert record["action"] not in set(CONSUME_ONE_ACTION_NAMES)
