from __future__ import annotations

from collections import Counter
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
    INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
)
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.insertion.internal_random_insertion_suffix_variants import (
    InternalRandomInsertionSuccessorRepairPolicy,
    InternalRandomInsertionTruncateSuffixPolicy,
    build_target_successor_counts,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_internal_random_insertion_successor_repair import (
    _validate_internal_insertion_successor_repair_sessions,
    build_internal_random_insertion_successor_repair_metadata,
)
from attack.pipeline.runs.run_internal_random_insertion_truncate_suffix import (
    _validate_internal_insertion_truncate_suffix_sessions,
    build_internal_random_insertion_truncate_suffix_metadata,
)


TRUNCATE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_truncate_suffix_nonzero_when_possible_ratio1_srgnn_targets5418_4092_9496_partial4.yaml"
)
SUCCESSOR_REPAIR_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_successor_repair_nonzero_when_possible_ratio1_srgnn_targets5418_4092_9496_partial4.yaml"
)


class FixedSlotRng:
    def __init__(self, slot: int) -> None:
        self.slot = int(slot)

    def randint(self, lower: int, upper: int) -> int:
        if self.slot < lower or self.slot > upper:
            raise AssertionError(f"Fixed slot {self.slot} not within [{lower}, {upper}].")
        return self.slot


def test_truncate_policy_preserves_insertion_semantics_and_truncates_suffix() -> None:
    policy = InternalRandomInsertionTruncateSuffixPolicy(
        1.0,
        rng=FixedSlotRng(2),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_truncation == [1, 2, 99, 3, 4]
    assert result.session == [1, 2, 99]
    assert result.truncated_suffix == [3, 4]
    assert result.session[-1] == 99
    assert result.left_item == 2
    assert result.original_right_item_before_truncation == 3


def test_truncate_policy_length_two_case() -> None:
    policy = InternalRandomInsertionTruncateSuffixPolicy(
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 2], 99)

    assert result.inserted_session_before_truncation == [1, 99, 2]
    assert result.session == [1, 99]
    assert result.truncated_suffix == [2]
    assert result.final_length == 2


def test_successor_count_builder_supports_allow_self_and_exclude_self() -> None:
    train_sessions = [
        [1, 99, 5],
        [2, 99, 99],
        [3, 99, 6],
        [4, 99],
    ]

    allow_self_counts = build_target_successor_counts(
        train_sessions,
        99,
        exclude_target=False,
    )
    exclude_self_counts = build_target_successor_counts(
        train_sessions,
        99,
        exclude_target=True,
    )

    assert allow_self_counts == Counter({5: 1, 99: 1, 6: 1})
    assert exclude_self_counts == Counter({5: 1, 6: 1})


def test_successor_repair_policy_replaces_first_right_item() -> None:
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({5: 3}),
        insertion_rng=FixedSlotRng(2),
        successor_rng=random.Random(20260405),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_repair == [1, 2, 99, 3, 4]
    assert result.session == [1, 2, 99, 5, 4]
    assert result.original_right_item == 3
    assert result.repaired_right_item == 5
    assert result.repair_applied is True
    assert result.repair_changed_item is True


def test_successor_repair_can_sample_target_itself_and_validation_allows_it() -> None:
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({99: 10}),
        insertion_rng=FixedSlotRng(2),
        successor_rng=random.Random(20260405),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_repair == [1, 2, 99, 3, 4]
    assert result.session == [1, 2, 99, 99, 4]
    assert result.repaired_right_item == 99
    assert result.sampled_successor == 99
    assert result.repair_applied is True
    assert result.repair_changed_item is True
    assert any(
        left == 99 and right == 99
        for left, right in zip(result.session, result.session[1:])
    )

    _validate_internal_insertion_successor_repair_sessions(
        template_sessions=[[1, 2, 3, 4]],
        final_sessions=[result.session],
        results=[result],
        target_item=99,
    )


def test_successor_repair_empty_pool_keeps_inserted_session() -> None:
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter(),
        insertion_rng=FixedSlotRng(1),
        successor_rng=random.Random(20260405),
    )

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.inserted_session_before_repair == [1, 99, 2, 3]
    assert result.session == [1, 99, 2, 3]
    assert result.repair_applied is False
    assert result.successor_pool_empty is True


def test_successor_repair_successor_equals_original_right_is_not_changed() -> None:
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({3: 10}),
        insertion_rng=FixedSlotRng(2),
        successor_rng=random.Random(20260405),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_repair == [1, 2, 99, 3, 4]
    assert result.session == [1, 2, 99, 3, 4]
    assert result.repair_applied is True
    assert result.repair_changed_item is False


def test_successor_repair_slot_sequence_matches_original_internal_insertion() -> None:
    sessions = [[1, 2, 3, 4], [5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]
    target = 99
    seed = 20260405
    topk_ratio = 1.0
    original_policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        topk_ratio,
        rng=random.Random(seed),
    )
    repair_policy = InternalRandomInsertionSuccessorRepairPolicy(
        topk_ratio,
        successor_counts=Counter({5: 2, 6: 1}),
        insertion_rng=random.Random(seed),
        successor_rng=random.Random(12345),
    )

    original_slots = [
        original_policy.apply_with_metadata(session, target).insertion_slot
        for session in sessions
    ]
    repair_slots = [
        repair_policy.apply_with_metadata(session, target).insertion_slot
        for session in sessions
    ]

    assert repair_slots == original_slots


def test_validate_truncate_rejects_invalid_final_session() -> None:
    policy = InternalRandomInsertionTruncateSuffixPolicy(
        1.0,
        rng=FixedSlotRng(2),
    )
    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    with pytest.raises(RuntimeError, match="does not end in target"):
        _validate_internal_insertion_truncate_suffix_sessions(
            template_sessions=[[1, 2, 3, 4]],
            final_sessions=[[1, 2, 99, 3]],
            results=[result],
            target_item=99,
        )

    with pytest.raises(RuntimeError, match="prefix through target"):
        _validate_internal_insertion_truncate_suffix_sessions(
            template_sessions=[[1, 2, 3, 4]],
            final_sessions=[[1, 99]],
            results=[result],
            target_item=99,
        )

    bad_slot0 = replace(result, insertion_slot=0)
    with pytest.raises(RuntimeError, match="slot0 or tail"):
        _validate_internal_insertion_truncate_suffix_sessions(
            template_sessions=[[1, 2, 3, 4]],
            final_sessions=[result.session],
            results=[bad_slot0],
            target_item=99,
        )

    bad_tail = replace(result, insertion_slot=4)
    with pytest.raises(RuntimeError, match="slot0 or tail"):
        _validate_internal_insertion_truncate_suffix_sessions(
            template_sessions=[[1, 2, 3, 4]],
            final_sessions=[result.session],
            results=[bad_tail],
            target_item=99,
        )


def test_validate_successor_repair_rejects_invalid_repaired_item() -> None:
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({5: 1}),
        insertion_rng=FixedSlotRng(1),
        successor_rng=random.Random(20260405),
    )
    result = policy.apply_with_metadata([1, 2, 3], 99)
    bad = replace(
        result,
        session=[1, 99, 7, 3],
        repaired_right_item=7,
        successor_pool=[5],
        repair_applied=True,
        repair_changed_item=True,
    )

    with pytest.raises(RuntimeError, match="not in successor pool"):
        _validate_internal_insertion_successor_repair_sessions(
            template_sessions=[[1, 2, 3]],
            final_sessions=[bad.session],
            results=[bad],
            target_item=99,
        )


def test_metadata_builders_include_required_keys() -> None:
    truncate_config = load_config(TRUNCATE_CONFIG_PATH)
    successor_config = load_config(SUCCESSOR_REPAIR_CONFIG_PATH)
    truncate_policy = InternalRandomInsertionTruncateSuffixPolicy(
        1.0,
        rng=FixedSlotRng(1),
    )
    successor_policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({5: 2, 6: 1}),
        insertion_rng=FixedSlotRng(1),
        successor_rng=random.Random(20260405),
    )
    templates = [[1, 2, 3], [4, 5]]
    truncate_results = [
        truncate_policy.apply_with_metadata(session, 99) for session in templates
    ]
    successor_results = [
        successor_policy.apply_with_metadata(session, 99) for session in templates
    ]
    truncate_slots = [result.insertion_slot for result in truncate_results]
    successor_slots = [result.insertion_slot for result in successor_results]

    truncate_metadata = build_internal_random_insertion_truncate_suffix_metadata(
        config=truncate_config,
        target_item=99,
        template_sessions=templates,
        insertion_results=truncate_results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=truncate_slots,
            run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )
    successor_metadata = build_internal_random_insertion_successor_repair_metadata(
        config=successor_config,
        target_item=99,
        template_sessions=templates,
        insertion_results=successor_results,
        successor_counts=Counter({5: 2, 6: 1}),
        clean_train_sessions=[[1, 99, 5], [2, 99, 6]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=successor_slots,
            run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        successor_rng_seed=20260504,
    )

    assert (
        truncate_metadata["run_type"]
        == INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE
    )
    assert truncate_metadata["suffix_strategy"] == "truncate_after_target"
    assert truncate_metadata["shared_fake_sessions_key"]
    assert truncate_metadata["all_injected_sessions_contain_target"] is True
    assert truncate_metadata["target_tail_ratio"] == pytest.approx(1.0)
    assert truncate_metadata["method_is_diagnostic"] is True
    assert truncate_metadata["not_length_matched_to_random_nz"] is True
    assert "insertion_slot_in_template" in truncate_metadata
    assert "target_position_in_inserted_session" in truncate_metadata
    assert "target_position_in_final_session" in truncate_metadata

    assert (
        successor_metadata["run_type"]
        == INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE
    )
    assert successor_metadata["suffix_strategy"] == "one_step_target_successor_repair"
    assert successor_metadata["shared_fake_sessions_key"]
    assert successor_metadata["all_injected_sessions_contain_target"] is True
    assert successor_metadata["exclude_target_from_successor_pool"] is False
    assert (
        successor_metadata["successor_repair_definition"]
        == "empirical_immediate_successor_allow_self"
    )
    assert successor_metadata["successor_pool_size"] == 2
    assert successor_metadata["successor_total_count"] == 3
    assert successor_metadata["top_successor_items"] == [5, 6]
    assert successor_metadata["repair_applied_count"] == 2
    assert successor_metadata["successor_repair_available"] is True
    assert successor_metadata["successor_top_k_configurable"] is False
    assert successor_metadata["successor_top_k_source"] == "runner_default"
    assert "self_successor_count" in successor_metadata
    assert "self_successor_share" in successor_metadata
    assert "self_successor_in_topk" in successor_metadata
    assert "self_successor_rank_in_successor_counts" in successor_metadata
    assert "sampled_self_successor_count" in successor_metadata
    assert "sampled_self_successor_ratio" in successor_metadata
    assert "final_sessions_with_adjacent_target_target_count" in successor_metadata
    assert "repair_created_adjacent_target_target_count" in successor_metadata
    assert "insertion_slot_in_template" in successor_metadata
    assert "target_position_in_inserted_session" in successor_metadata
    assert "target_position_in_final_session" in successor_metadata


def test_successor_repair_metadata_records_self_successor_values() -> None:
    config = load_config(SUCCESSOR_REPAIR_CONFIG_PATH)
    policy = InternalRandomInsertionSuccessorRepairPolicy(
        1.0,
        successor_counts=Counter({99: 10, 5: 1}),
        insertion_rng=FixedSlotRng(2),
        successor_rng=random.Random(20260405),
    )
    templates = [[1, 2, 3, 4]]
    results = [policy.apply_with_metadata(templates[0], 99)]

    metadata = build_internal_random_insertion_successor_repair_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        insertion_results=results,
        successor_counts=Counter({99: 10, 5: 1}),
        clean_train_sessions=[[1, 99, 99], [2, 99, 5]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=[results[0].insertion_slot],
            run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        successor_rng_seed=20260504,
    )

    assert metadata["top_successor_items"][0] == 99
    assert metadata["top_successor_counts"][0] == 10
    assert metadata["self_successor_count"] == 10
    assert metadata["self_successor_share"] == pytest.approx(10.0 / 11.0)
    assert metadata["self_successor_in_topk"] is True
    assert metadata["self_successor_rank_in_successor_counts"] == 1
    assert metadata["sampled_self_successor_count"] == 1
    assert metadata["sampled_self_successor_ratio"] == pytest.approx(1.0)
    assert metadata["final_sessions_with_adjacent_target_target_count"] == 1
    assert metadata["final_sessions_with_adjacent_target_target_ratio"] == pytest.approx(
        1.0
    )
    assert metadata["repair_created_adjacent_target_target_count"] == 1
    assert metadata["repair_created_adjacent_target_target_ratio"] == pytest.approx(1.0)


def test_new_configs_parse() -> None:
    truncate_config = load_config(TRUNCATE_CONFIG_PATH)
    successor_config = load_config(SUCCESSOR_REPAIR_CONFIG_PATH)

    assert truncate_config.targets.mode == "explicit_list"
    assert truncate_config.targets.explicit_list == (5418, 4092, 9496)
    assert truncate_config.victims.params["srgnn"]["train"]["epochs"] == 4
    assert successor_config.targets.mode == "explicit_list"
    assert successor_config.targets.explicit_list == (5418, 4092, 9496)
    assert successor_config.victims.params["srgnn"]["train"]["epochs"] == 4
