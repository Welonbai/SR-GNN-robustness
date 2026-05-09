from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
)
from attack.insertion.internal_random_insertion_suffix_variants import (
    InternalRandomInsertionTruncateSuffixPolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_internal_random_insertion_truncate_suffix import (
    _validate_internal_insertion_truncate_suffix_sessions,
    build_internal_random_insertion_truncate_suffix_metadata,
)


TRUNCATE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_truncate_suffix_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
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


def test_truncate_metadata_builder_includes_required_keys() -> None:
    config = load_config(TRUNCATE_CONFIG_PATH)
    policy = InternalRandomInsertionTruncateSuffixPolicy(
        1.0,
        rng=FixedSlotRng(1),
    )
    templates = [[1, 2, 3], [4, 5]]
    results = [policy.apply_with_metadata(session, 99) for session in templates]
    slots = [result.insertion_slot for result in results]

    metadata = build_internal_random_insertion_truncate_suffix_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        insertion_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=slots,
            run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert (
        metadata["run_type"]
        == INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE
    )
    assert metadata["suffix_strategy"] == "truncate_after_target"
    assert metadata["shared_fake_sessions_key"]
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["target_tail_ratio"] == pytest.approx(1.0)
    assert metadata["method_is_diagnostic"] is True
    assert metadata["not_length_matched_to_random_nz"] is True
    assert "insertion_slot_in_template" in metadata
    assert "target_position_in_inserted_session" in metadata
    assert "target_position_in_final_session" in metadata


def test_truncate_appendable_config_parses() -> None:
    config = load_config(TRUNCATE_CONFIG_PATH)

    assert config.targets.mode == "sampled"
    assert config.targets.explicit_list == ()
    assert config.targets.count == 12
    assert config.targets.reuse_saved_targets is True
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4
    assert "targets" not in config.experiment.name
    assert "5418" not in config.experiment.name
    assert "sample12" not in config.experiment.name
