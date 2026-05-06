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
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.random_insertion_then_crop_nonzero_when_possible import (
    RandomInsertionThenCropNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_random_insertion_then_crop_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_insertion_then_crop_sessions,
    build_insertion_then_crop_stats_payload,
    build_random_insertion_then_crop_metadata,
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
INSERTION_THEN_CROP_FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_insertion_then_crop_nonzero_when_possible_ratio1_srgnn_target11103.yaml"
)


class FixedSlotRng:
    def __init__(self, slot: int) -> None:
        self.slot = int(slot)

    def randint(self, lower: int, upper: int) -> int:
        if self.slot < lower or self.slot > upper:
            raise AssertionError(f"Fixed slot {self.slot} not within [{lower}, {upper}].")
        return self.slot


def test_policy_insert_slot_one_then_crop_tail_non_target() -> None:
    policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.inserted_session == [1, 99, 2, 3]
    assert result.session == [1, 99, 2]
    assert result.insertion_slot == 1
    assert result.crop_position == 3
    assert result.cropped_item == 3
    assert result.target_position_after_crop == 1
    assert result.original_length == 3
    assert result.inserted_length == 4
    assert result.final_length == 3


def test_policy_insert_tail_then_crop_nearest_non_target_left() -> None:
    policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(3),
    )

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.inserted_session == [1, 2, 3, 99]
    assert result.session == [1, 2, 99]
    assert result.insertion_slot == 3
    assert result.crop_position == 2
    assert result.cropped_item == 3
    assert result.target_position_after_crop == 2


def test_policy_never_crops_target_when_original_contains_target() -> None:
    policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 99, 2], 99)

    assert result.inserted_session == [1, 99, 99, 2]
    assert result.session == [1, 99, 99]
    assert result.cropped_item == 2
    assert result.cropped_item != 99
    assert result.target_occurrence_count_after_crop == 2
    assert result.final_target_positions == [1, 2]


def test_policy_rejects_empty_and_all_target_sessions() -> None:
    policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(1))

    with pytest.raises(ValueError, match="at least one item"):
        policy.apply([], 99)
    with pytest.raises(ValueError, match="no non-target item"):
        policy.apply([99, 99], 99)


def test_policy_is_deterministic_given_fixed_seed() -> None:
    sessions = [[1, 2, 3], [4, 5, 6, 7], [8, 99, 10]]

    def apply(seed: int) -> list[tuple[list[int], int, int]]:
        policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
            1.0,
            rng=random.Random(seed),
        )
        results = [policy.apply_with_metadata(session, 99) for session in sessions]
        return [
            (result.session, result.insertion_slot, result.crop_position)
            for result in results
        ]

    assert apply(20260405) == apply(20260405)


def test_validate_insertion_then_crop_sessions_checks_lengths_and_crop_target() -> None:
    policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
        1.0,
        rng=FixedSlotRng(1),
    )
    templates = [[1, 2, 3], [4, 5, 6]]
    results = [policy.apply_with_metadata(session, 99) for session in templates]
    finals = [result.session for result in results]

    _validate_insertion_then_crop_sessions(
        template_sessions=templates,
        final_sessions=finals,
        results=results,
        target_item=99,
    )

    bad = results[0]
    bad_target_crop = type(bad)(
        session=bad.session,
        insertion_slot=bad.insertion_slot,
        inserted_session=bad.inserted_session,
        crop_position=bad.insertion_slot,
        cropped_item=99,
        final_target_positions=bad.final_target_positions,
        target_position_after_crop=bad.target_position_after_crop,
        original_length=bad.original_length,
        inserted_length=bad.inserted_length,
        final_length=bad.final_length,
        pre_existing_target_count=bad.pre_existing_target_count,
        target_occurrence_count_after_crop=bad.target_occurrence_count_after_crop,
    )
    with pytest.raises(RuntimeError, match="cropped target item"):
        _validate_insertion_then_crop_sessions(
            template_sessions=[templates[0]],
            final_sessions=[bad.session],
            results=[bad_target_crop],
            target_item=99,
        )


def test_metadata_records_insertion_crop_target_and_length_stats() -> None:
    config = load_config(INSERTION_THEN_CROP_PARTIAL5_CONFIG_PATH)
    templates = [[1, 2, 3], [1, 99, 2], [4, 5, 6]]
    policies = [
        RandomInsertionThenCropNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(1)),
        RandomInsertionThenCropNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(1)),
        RandomInsertionThenCropNonzeroWhenPossiblePolicy(1.0, rng=FixedSlotRng(3)),
    ]
    results = [
        policy.apply_with_metadata(session, 99)
        for policy, session in zip(policies, templates)
    ]
    finals = [result.session for result in results]
    insertion_slots = [result.insertion_slot for result in results]
    slot_stats = build_slot_stats_payload(
        sessions=templates,
        insertion_slots=insertion_slots,
        run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
    )
    stats = build_insertion_then_crop_stats_payload(
        run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
        template_sessions=templates,
        results=results,
        slot_stats_payload=slot_stats,
    )

    metadata = build_random_insertion_then_crop_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        final_sessions=finals,
        results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        stats_payload=stats,
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["operation"] == "insertion_then_crop"
    assert metadata["insertion_policy"] == "random_nonzero_when_possible"
    assert metadata["crop_policy"] == "crop_tail_non_target"
    assert metadata["insertion_length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["final_length_shift_summary"] == {"min": 0.0, "max": 0.0, "mean": 0.0}
    assert metadata["cropped_item_target_count"] == 0
    assert metadata["crop_at_inserted_target_count"] == 0
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["sessions_with_multiple_target_occurrences_count"] == 1
    assert metadata["replacement_topk_ratio"] == pytest.approx(1.0)
    assert metadata["insertion_slot_topk_ratio"] == pytest.approx(1.0)
    assert metadata["replacement_topk_ratio_used_for_replacement"] is False
    assert "insertion slot top-k ratio" in metadata["topk_sampling_note"]
    assert metadata["target_occurrence_count_after_crop"]["min"] == pytest.approx(1.0)
    assert metadata["target_occurrence_count_after_crop"]["max"] == pytest.approx(2.0)
    assert metadata["target_position_after_crop_counts"]
    assert metadata["previews"][0]["index_base"] == "zero_based"
    assert metadata["previews"][0]["inserted_session"] == [1, 99, 2, 3]
    assert metadata["previews"][0]["final_cropped_session"] == [1, 99, 2]


def test_new_configs_and_existing_basis_configs_parse() -> None:
    partial = load_config(INSERTION_THEN_CROP_PARTIAL5_CONFIG_PATH)
    full = load_config(INSERTION_THEN_CROP_FULL_CONFIG_PATH)
    random_nz = load_config(RANDOM_NZ_CONFIG_PATH)
    random_insertion = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)
    tail_replacement = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)
    tail_insertion = load_config(TAIL_INSERTION_PARTIAL5_CONFIG_PATH)

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


def test_insertion_then_crop_shares_generation_key_but_has_distinct_attack_key() -> None:
    config = load_config(INSERTION_THEN_CROP_PARTIAL5_CONFIG_PATH)

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
    }

    assert len(shared_keys) == 1
    assert len(attack_keys) == 5
