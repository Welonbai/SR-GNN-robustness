from __future__ import annotations

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
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_internal_random_insertion_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_internal_inserted_sessions,
    build_internal_random_insertion_metadata,
)


RANDOM_NZ_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_nonzero_when_possible_ratio1_srgnn_sample3.yaml"
)
RANDOM_INSERTION_PARTIAL4_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_insertion_nonzero_when_possible_ratio1_srgnn_target11103_partial4.yaml"
)
TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tail_replacement_nonzero_when_possible_ratio1_srgnn_target11103_partial5.yaml"
)
TAIL_INSERTION_PARTIAL5_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tail_insertion_nonzero_when_possible_ratio1_srgnn_target11103_partial5.yaml"
)
INSERTION_THEN_CROP_PARTIAL5_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_insertion_then_crop_nonzero_when_possible_ratio1_srgnn_target11103_partial5.yaml"
)
INTERNAL_INSERTION_PARTIAL5_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_nonzero_when_possible_ratio1_srgnn_target11103_partial5.yaml"
)
INTERNAL_INSERTION_FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_nonzero_when_possible_ratio1_srgnn_target11103.yaml"
)


class FixedSlotRng:
    def __init__(self, slot: int) -> None:
        self.slot = int(slot)

    def randint(self, lower: int, upper: int) -> int:
        if self.slot < lower or self.slot > upper:
            raise AssertionError(f"Fixed slot {self.slot} not within [{lower}, {upper}].")
        return self.slot


def test_policy_uses_internal_nonzero_slots_and_preserves_order() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(2),
    )

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.session == [1, 2, 99, 3]
    assert result.insertion_slot == 2
    assert result.original_length == 3
    assert result.inserted_length == 4
    assert result.left_item == 2
    assert result.right_item == 3
    assert result.target_occurrence_count_after_insertion == 1


def test_policy_length_two_has_only_slot_one() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 2], 99)

    assert result.session == [1, 99, 2]
    assert result.insertion_slot == 1
    assert result.left_item == 1
    assert result.right_item == 2


def test_policy_rejects_empty_and_length_one_sessions() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(1.0)

    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([], 99)
    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([1], 99)


def test_policy_never_selects_tail_slot() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=random.Random(20260405),
    )
    slots = [
        policy.apply_with_metadata([1, 2, 3], 99).insertion_slot
        for _ in range(200)
    ]

    assert set(slots) == {1, 2}
    assert 0 not in slots
    assert 3 not in slots


def test_policy_topk_ratio_limits_internal_slots() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        0.5,
        rng=FixedSlotRng(2),
    )
    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.insertion_slot == 2
    assert result.session == [1, 2, 99, 3, 4]

    with pytest.raises(AssertionError, match="not within"):
        InternalRandomInsertionNonzeroWhenPossiblePolicy(
            0.5,
            rng=FixedSlotRng(3),
        ).apply([1, 2, 3, 4], 99)


def test_policy_is_deterministic_given_fixed_seed() -> None:
    sessions = [[1, 2, 3], [4, 5, 6, 7], [8, 99, 10]]

    def apply(seed: int) -> list[tuple[list[int], int, int, int]]:
        policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
            1.0,
            rng=random.Random(seed),
        )
        results = [policy.apply_with_metadata(session, 99) for session in sessions]
        return [
            (result.session, result.insertion_slot, result.left_item, result.right_item)
            for result in results
        ]

    assert apply(20260405) == apply(20260405)


def test_policy_records_pre_existing_target_occurrences() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 99, 2], 99)

    assert result.session == [1, 99, 99, 2]
    assert result.pre_existing_target_count == 1
    assert result.target_occurrence_count_after_insertion == 2


def test_validate_internal_inserted_sessions_checks_invariants() -> None:
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )
    templates = [[1, 2, 3], [4, 5]]
    results = [policy.apply_with_metadata(session, 99) for session in templates]
    inserted = [result.session for result in results]

    _validate_internal_inserted_sessions(
        template_sessions=templates,
        inserted_sessions=inserted,
        results=results,
        target_item=99,
    )

    bad = results[0]
    bad_tail = type(bad)(
        session=[1, 2, 3, 99],
        insertion_slot=3,
        original_length=bad.original_length,
        inserted_length=bad.inserted_length,
        left_item=bad.left_item,
        right_item=bad.right_item,
        pre_existing_target_count=bad.pre_existing_target_count,
        target_occurrence_count_after_insertion=bad.target_occurrence_count_after_insertion,
    )
    with pytest.raises(RuntimeError, match="slot0 or tail"):
        _validate_internal_inserted_sessions(
            template_sessions=[templates[0]],
            inserted_sessions=[bad_tail.session],
            results=[bad_tail],
            target_item=99,
        )


def test_metadata_records_internal_slots_neighbors_and_length_stats() -> None:
    config = load_config(INTERNAL_INSERTION_PARTIAL5_CONFIG_PATH)
    templates = [[1, 2, 3], [4, 5], [1, 99, 2]]
    policies = [
        InternalRandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(1)),
        InternalRandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(1)),
        InternalRandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(2)),
    ]
    results = [
        policy.apply_with_metadata(session, 99)
        for policy, session in zip(policies, templates)
    ]
    inserted = [result.session for result in results]
    slot_stats = build_slot_stats_payload(
        sessions=templates,
        insertion_slots=[result.insertion_slot for result in results],
        run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
    )

    metadata = build_internal_random_insertion_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        inserted_sessions=inserted,
        insertion_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=slot_stats,
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["operation"] == "insertion"
    assert metadata["insertion_policy"] == "internal_random_nonzero_when_possible"
    assert metadata["tail_slot_count"] == 0
    assert metadata["tail_slot_ratio"] == pytest.approx(0.0)
    assert metadata["tail_slot_excluded"] is True
    assert metadata["slot0_excluded"] is True
    assert metadata["length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["every_inserted_target_has_left_neighbor"] is True
    assert metadata["every_inserted_target_has_right_neighbor"] is True
    assert metadata["unique_left_item_count"] >= 1
    assert metadata["unique_right_item_count"] >= 1
    assert metadata["unique_left_right_pair_count"] >= 1
    assert metadata["sessions_with_multiple_target_occurrences_count"] == 1
    assert metadata["replacement_topk_ratio"] == pytest.approx(1.0)
    assert metadata["internal_insertion_slot_topk_ratio"] == pytest.approx(1.0)
    assert metadata["replacement_topk_ratio_used_for_replacement"] is False
    assert "tail slot is excluded" in metadata["topk_sampling_note"]
    assert metadata["poison_model_checkpoint_path"] is None
    assert metadata["previews"][0]["index_base"] == "zero_based"
    assert metadata["previews"][0]["left_item"] == 1
    assert metadata["previews"][0]["right_item"] == 2


def test_new_configs_and_existing_basis_configs_parse() -> None:
    partial = load_config(INTERNAL_INSERTION_PARTIAL5_CONFIG_PATH)
    full = load_config(INTERNAL_INSERTION_FULL_CONFIG_PATH)
    random_nz = load_config(RANDOM_NZ_CONFIG_PATH)
    random_insertion = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)
    tail_replacement = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)
    tail_insertion = load_config(TAIL_INSERTION_PARTIAL5_CONFIG_PATH)
    insertion_then_crop = load_config(INSERTION_THEN_CROP_PARTIAL5_CONFIG_PATH)

    assert partial.experiment.name.endswith("partial5")
    assert partial.targets.explicit_list == (11103,)
    assert partial.victims.params["srgnn"]["train"]["epochs"] == 5
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert full.victims.params["srgnn"]["train"]["patience"] == 10
    assert (
        full.victims.params["srgnn"]["train"]["checkpoint_protocol"]
        == "validation_best"
    )
    assert random_nz.attack.replacement_topk_ratio == pytest.approx(1.0)
    assert random_insertion.attack.replacement_topk_ratio == pytest.approx(1.0)
    assert tail_replacement.attack.replacement_topk_ratio == pytest.approx(1.0)
    assert tail_insertion.attack.replacement_topk_ratio == pytest.approx(1.0)
    assert insertion_then_crop.attack.replacement_topk_ratio == pytest.approx(1.0)


def test_internal_insertion_shares_generation_key_but_has_distinct_attack_key() -> None:
    config = load_config(INTERNAL_INSERTION_PARTIAL5_CONFIG_PATH)

    shared_keys = {
        shared_attack_artifact_key(config, run_type=RANDOM_NZ_RUN_TYPE),
        shared_attack_artifact_key(
            config,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
    }
    attack_keys = {
        attack_key(config, run_type=RANDOM_NZ_RUN_TYPE),
        attack_key(
            config,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        attack_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
    }

    assert len(shared_keys) == 1
    assert len(attack_keys) == 6
