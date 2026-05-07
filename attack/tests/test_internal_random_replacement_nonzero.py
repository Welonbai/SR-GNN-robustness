from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.internal_random_replacement_nonzero_when_possible import (
    InternalRandomReplacementNonzeroWhenPossiblePolicy,
)
from attack.pipeline.runs.run_internal_random_replacement_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_internal_replaced_sessions,
    build_internal_random_replacement_metadata,
)


INTERNAL_REPLACEMENT_PARTIAL4_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_replacement_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)


class FixedChoiceRng:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def choice(self, values: list[int]) -> int:
        if self.value not in values:
            raise AssertionError(f"Fixed choice {self.value} not in {values}.")
        return self.value


def test_policy_replaces_only_internal_nonzero_positions_for_len_ge_3() -> None:
    policy = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(2),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.session == [1, 2, 99, 4]
    assert result.replacement_position == 2
    assert result.original_length == 4
    assert result.replaced_length == 4
    assert result.original_item == 3
    assert result.left_item == 2
    assert result.right_item == 4
    assert result.used_internal_position is True
    assert result.used_tail_fallback is False
    assert result.candidate_positions == [1, 2]
    assert result.restricted_candidate_positions == [1, 2]
    assert result.replacement_position in result.restricted_candidate_positions


def test_len3_has_only_position_one() -> None:
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(1),
    ).apply_with_metadata([1, 2, 3], 99)

    assert result.session == [1, 99, 3]
    assert result.candidate_positions == [1]
    assert result.restricted_candidate_positions == [1]
    assert result.replacement_position == 1


def test_len4_with_ratio1_can_sample_positions_one_or_two() -> None:
    positions = {
        InternalRandomReplacementNonzeroWhenPossiblePolicy(
            1.0,
            rng=FixedChoiceRng(position),
        ).apply_with_metadata([1, 2, 3, 4], 99).replacement_position
        for position in (1, 2)
    }

    assert positions == {1, 2}


def test_position_zero_and_tail_are_never_selected_when_len_ge_3() -> None:
    for selected_position in (1, 2, 3):
        result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
            1.0,
            rng=FixedChoiceRng(selected_position),
        ).apply_with_metadata([1, 2, 3, 4, 5], 99)

        assert result.replacement_position in {1, 2, 3}
        assert result.replacement_position != 0
        assert result.replacement_position != 4
        assert result.used_internal_position is True
        assert result.used_tail_fallback is False


def test_len2_falls_back_to_tail_position_one() -> None:
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(1),
    ).apply_with_metadata([1, 2], 99)

    assert result.session == [1, 99]
    assert result.replacement_position == 1
    assert result.left_item == 1
    assert result.right_item is None
    assert result.used_internal_position is False
    assert result.used_tail_fallback is True
    assert result.internal_candidate_count == 0
    assert result.candidate_positions == [1]
    assert result.restricted_candidate_positions == [1]


def test_policy_preserves_length_and_order_except_replaced_position() -> None:
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(3),
    ).apply_with_metadata([10, 20, 30, 40, 50], 99)

    assert len(result.session) == 5
    assert result.session == [10, 20, 30, 99, 50]


def test_topk_ratio_restricts_internal_positions() -> None:
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        0.5,
        rng=FixedChoiceRng(2),
    ).apply_with_metadata([1, 2, 3, 4, 5, 6], 99)

    assert result.candidate_positions == [1, 2, 3, 4]
    assert result.restricted_candidate_positions == [1, 2]
    assert result.replacement_position == 2

    with pytest.raises(AssertionError, match="not in"):
        InternalRandomReplacementNonzeroWhenPossiblePolicy(
            0.5,
            rng=FixedChoiceRng(3),
        ).apply([1, 2, 3, 4, 5, 6], 99)


def test_policy_is_deterministic_given_same_random_seed() -> None:
    sessions = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]

    def apply(seed: int) -> list[tuple[list[int], int, list[int]]]:
        policy = InternalRandomReplacementNonzeroWhenPossiblePolicy(
            1.0,
            rng=random.Random(seed),
        )
        results = [policy.apply_with_metadata(session, 99) for session in sessions]
        return [
            (
                result.session,
                result.replacement_position,
                result.restricted_candidate_positions,
            )
            for result in results
        ]

    assert apply(20260405) == apply(20260405)


def test_pre_existing_target_and_noop_replacement_are_recorded() -> None:
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(1),
    ).apply_with_metadata([1, 99, 2], 99)

    assert result.session == [1, 99, 2]
    assert result.original_item == 99
    assert result.pre_existing_target_count == 1
    assert result.target_occurrence_count_after_replacement == 1
    assert result.was_noop is True


def test_policy_rejects_empty_and_length_one_sessions() -> None:
    policy = InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0)

    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([], 99)
    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([1], 99)


def test_validate_internal_replaced_sessions_checks_invariants() -> None:
    templates = [[1, 2, 3, 4], [5, 6]]
    policies = [
        InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0, rng=FixedChoiceRng(2)),
        InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0, rng=FixedChoiceRng(1)),
    ]
    results = [
        policy.apply_with_metadata(session, 99)
        for policy, session in zip(policies, templates)
    ]
    replaced = [result.session for result in results]

    _validate_internal_replaced_sessions(
        template_sessions=templates,
        replaced_sessions=replaced,
        results=results,
        target_item=99,
    )

    bad_position = replace(
        results[0],
        replacement_position=2,
        restricted_candidate_positions=[1],
        session=[1, 2, 99, 4],
    )
    with pytest.raises(RuntimeError, match="restricted candidate"):
        _validate_internal_replaced_sessions(
            template_sessions=[templates[0]],
            replaced_sessions=[bad_position.session],
            results=[bad_position],
            target_item=99,
        )

    bad_non_prefix_restricted = replace(
        results[0],
        replacement_position=2,
        restricted_candidate_positions=[2],
        session=[1, 2, 99, 4],
    )
    with pytest.raises(RuntimeError, match="prefix of candidate positions"):
        _validate_internal_replaced_sessions(
            template_sessions=[templates[0]],
            replaced_sessions=[bad_non_prefix_restricted.session],
            results=[bad_non_prefix_restricted],
            target_item=99,
        )

    with pytest.raises(RuntimeError, match="length changed"):
        _validate_internal_replaced_sessions(
            template_sessions=[templates[0]],
            replaced_sessions=[[1, 2, 99, 4, 5]],
            results=[results[0]],
            target_item=99,
        )

    with pytest.raises(RuntimeError, match="missing target"):
        _validate_internal_replaced_sessions(
            template_sessions=[templates[0]],
            replaced_sessions=[[1, 2, 3, 4]],
            results=[results[0]],
            target_item=99,
        )

    with pytest.raises(RuntimeError, match="more than one position"):
        _validate_internal_replaced_sessions(
            template_sessions=[templates[0]],
            replaced_sessions=[[1, 2, 99, 88]],
            results=[results[0]],
            target_item=99,
        )


def test_metadata_records_replacement_semantics_and_shared_identities() -> None:
    config = load_config(INTERNAL_REPLACEMENT_PARTIAL4_CONFIG_PATH)
    templates = [[1, 2, 3, 4], [5, 6], [7, 99, 8]]
    policies = [
        InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0, rng=FixedChoiceRng(2)),
        InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0, rng=FixedChoiceRng(1)),
        InternalRandomReplacementNonzeroWhenPossiblePolicy(1.0, rng=FixedChoiceRng(1)),
    ]
    results = [
        policy.apply_with_metadata(session, 99)
        for policy, session in zip(policies, templates)
    ]
    replaced = [result.session for result in results]

    metadata = build_internal_random_replacement_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        replaced_sessions=replaced,
        replacement_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["operation"] == "replacement"
    assert metadata["replacement_policy"] == "internal_random_nonzero_when_possible"
    assert metadata["length_shift_summary"] == {"min": 0.0, "max": 0.0, "mean": 0.0}
    assert metadata["pos0_excluded"] is True
    assert metadata["tail_position_excluded_when_internal_available"] is True
    assert metadata["tail_fallback_allowed_for_len2"] is True
    assert metadata["tail_fallback_count"] == 1
    assert metadata["internal_replacement_count"] == 2
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["same_shared_fake_sessions_as_random_nz_expected"] is True
    assert metadata["same_shared_fake_sessions_as_internal_random_insertion_expected"] is True
    assert metadata["same_shared_fake_sessions_as_tail_replacement_expected"] is True
    assert metadata["shared_fake_sessions_key"] == metadata["random_nz_shared_fake_sessions_key"]
    assert (
        metadata["shared_fake_sessions_key"]
        == metadata["internal_random_insertion_shared_fake_sessions_key"]
    )
    assert metadata["shared_fake_sessions_key"] == metadata["tail_replacement_shared_fake_sessions_key"]
    assert metadata["attack_key"] != metadata["random_nz_attack_key"]
    assert metadata["attack_key"] != metadata["internal_random_insertion_attack_key"]
    assert metadata["tail_position_count"] == 1
    assert metadata["replacement_position_group_counts"]["tail_position"] == 1
    assert metadata["replacement_position_group_counts"]["pos1"] == 1
    assert metadata["every_internal_replaced_target_has_right_neighbor"] is True
    assert metadata["every_replaced_target_has_right_neighbor"] is False
    assert metadata["previews"][0]["restricted_candidate_positions"] == [1, 2]
    assert metadata["previews"][1]["right_item"] is None


def test_len2_fallback_is_grouped_only_as_tail_position() -> None:
    config = load_config(INTERNAL_REPLACEMENT_PARTIAL4_CONFIG_PATH)
    result = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedChoiceRng(1),
    ).apply_with_metadata([1, 2], 99)

    metadata = build_internal_random_replacement_metadata(
        config=config,
        target_item=99,
        template_sessions=[[1, 2]],
        replaced_sessions=[result.session],
        replacement_results=[result],
        clean_train_sessions=[[1, 2]],
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["replacement_position_group_counts"]["tail_position"] == 1
    assert metadata["replacement_position_group_counts"]["pos1"] == 0


def test_new_config_parses_successfully() -> None:
    config = load_config(INTERNAL_REPLACEMENT_PARTIAL4_CONFIG_PATH)

    assert (
        config.experiment.name
        == "valbest_attack_internal_random_replacement_nonzero_when_possible_ratio1_srgnn_partial4"
    )
    assert config.targets.mode == "sampled"
    assert config.targets.explicit_list == ()
    assert config.targets.count == 6
    assert config.attack.replacement_topk_ratio == pytest.approx(1.0)
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4


def test_internal_replacement_shares_generation_key_but_has_distinct_attack_key() -> None:
    config = load_config(INTERNAL_REPLACEMENT_PARTIAL4_CONFIG_PATH)

    shared_keys = {
        shared_attack_artifact_key(config, run_type=RANDOM_NZ_RUN_TYPE),
        shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
    }
    attack_keys = {
        attack_key(config, run_type=RANDOM_NZ_RUN_TYPE),
        attack_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
    }

    assert len(shared_keys) == 1
    assert len(attack_keys) == 4
