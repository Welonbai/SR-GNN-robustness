from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import (
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.tail_insertion import TailInsertionPolicy
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_tail_insertion_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_tail_inserted_sessions,
    build_tail_insertion_metadata,
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
TAIL_INSERTION_FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tail_insertion_nonzero_when_possible_ratio1_srgnn_target11103.yaml"
)


def test_policy_appends_target_and_preserves_original_items() -> None:
    result = TailInsertionPolicy().apply_with_metadata([1, 2, 3], 99)

    assert result.session == [1, 2, 3, 99]
    assert result.insertion_slot == 3
    assert result.original_length == 3
    assert result.inserted_length == 4
    assert result.pre_existing_target_count == 0
    assert result.target_occurrence_count_after_insertion == 1


def test_policy_records_multiple_target_occurrences() -> None:
    result = TailInsertionPolicy().apply_with_metadata([1, 99], 99)

    assert result.session == [1, 99, 99]
    assert result.insertion_slot == 2
    assert result.pre_existing_target_count == 1
    assert result.target_occurrence_count_after_insertion == 2


def test_policy_rejects_empty_session_and_is_deterministic() -> None:
    policy = TailInsertionPolicy()

    with pytest.raises(ValueError, match="at least one item"):
        policy.apply([], 99)
    assert policy.apply([1, 2, 3], 99) == [1, 2, 3, 99]
    assert policy.apply([1, 2, 3], 99) == [1, 2, 3, 99]


def test_validate_tail_inserted_sessions_checks_count_target_length_and_order() -> None:
    template_sessions = [[1, 2, 3], [4, 5]]
    inserted_sessions = [[1, 2, 3, 99], [4, 5, 99]]
    slots = [3, 2]

    _validate_tail_inserted_sessions(
        template_sessions=template_sessions,
        inserted_sessions=inserted_sessions,
        insertion_slots=slots,
        target_item=99,
    )

    with pytest.raises(RuntimeError, match="tail insertion slot"):
        _validate_tail_inserted_sessions(
            template_sessions=template_sessions,
            inserted_sessions=inserted_sessions,
            insertion_slots=[2, 2],
            target_item=99,
        )
    with pytest.raises(RuntimeError, match="preserve original item order"):
        _validate_tail_inserted_sessions(
            template_sessions=template_sessions,
            inserted_sessions=[[1, 2, 99, 3], [4, 5, 99]],
            insertion_slots=slots,
            target_item=99,
        )


def test_metadata_records_tail_slot_topk_and_target_occurrence_summary() -> None:
    config = load_config(TAIL_INSERTION_PARTIAL5_CONFIG_PATH)
    template_sessions = [[99, 1, 2], [3, 4, 5, 6, 7, 8], [9, 10]]
    policy = TailInsertionPolicy()
    results = [
        policy.apply_with_metadata(session, 99)
        for session in template_sessions
    ]
    inserted_sessions = [result.session for result in results]
    slots = [result.insertion_slot for result in results]
    slot_stats = build_slot_stats_payload(
        sessions=template_sessions,
        insertion_slots=slots,
        run_type=TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
    )

    metadata = build_tail_insertion_metadata(
        config=config,
        target_item=99,
        template_sessions=template_sessions,
        inserted_sessions=inserted_sessions,
        insertion_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=slot_stats,
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["operation"] == "insertion"
    assert metadata["insertion_policy"] == "tail"
    assert metadata["replacement_topk_ratio"] == pytest.approx(1.0)
    assert metadata["replacement_topk_ratio_used"] is False
    assert metadata["topk_ratio_field_name"] == "attack.replacement_topk_ratio"
    assert "topk ratio is ignored" in metadata["topk_sampling_note"]
    assert metadata["poison_model_checkpoint_path"] is None
    assert metadata["same_shared_fake_sessions_as_random_nz_expected"] is True
    assert metadata["same_shared_fake_sessions_as_random_insertion_expected"] is True
    assert metadata["shared_fake_sessions_key"] == metadata["random_nz_shared_fake_sessions_key"]
    assert metadata["shared_fake_sessions_key"] == metadata["random_insertion_shared_fake_sessions_key"]
    assert metadata["shared_fake_sessions_key"] == metadata["tail_replacement_shared_fake_sessions_key"]
    assert metadata["original_length_distribution"] == {"len2": 1, "len3": 1, "len6": 1}
    assert metadata["inserted_length_distribution"] == {"len3": 1, "len4": 1, "len7": 1}
    assert metadata["clean_train_length_distribution"] == {"len2": 1, "len3": 1}
    assert metadata["length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["insertion_slot_counts"] == {"2": 1, "3": 1, "6": 1}
    assert metadata["insertion_slot_group_counts"]["slot2"] == 1
    assert metadata["insertion_slot_group_counts"]["slot3"] == 1
    assert metadata["insertion_slot_group_counts"]["slot6_plus"] == 1
    assert metadata["tail_slot_count"] == 3
    assert metadata["tail_slot_ratio"] == pytest.approx(1.0)
    assert metadata["tail_slot_is_overlapping_group"] is True
    assert metadata["pre_existing_target_in_template_sessions_count"] == 1
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["target_occurrence_count_after_insertion"]["min"] == pytest.approx(1.0)
    assert metadata["target_occurrence_count_after_insertion"]["max"] == pytest.approx(2.0)
    assert metadata["target_occurrence_count_after_insertion"]["mean"] == pytest.approx(4.0 / 3.0)
    assert metadata["sessions_with_multiple_target_occurrences_count"] == 1
    assert metadata["previews"][0]["index_base"] == "zero_based"
    assert metadata["previews"][0]["tail_inserted_session"] == [99, 1, 2, 99]


def test_new_configs_and_existing_comparison_configs_parse() -> None:
    partial = load_config(TAIL_INSERTION_PARTIAL5_CONFIG_PATH)
    full = load_config(TAIL_INSERTION_FULL_CONFIG_PATH)
    random_nz = load_config(RANDOM_NZ_CONFIG_PATH)
    random_insertion = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)
    tail_replacement = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)

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


def test_tail_insertion_shares_generation_key_but_has_distinct_attack_key() -> None:
    config = load_config(TAIL_INSERTION_PARTIAL5_CONFIG_PATH)

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
    }

    assert len(shared_keys) == 1
    assert len(attack_keys) == 4
