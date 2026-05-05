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
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.tail_replacement import TailReplacementPolicy
from attack.pipeline.runs.run_tail_replacement_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_tail_replaced_sessions,
    build_tail_replacement_metadata,
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
TAIL_REPLACEMENT_FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tail_replacement_nonzero_when_possible_ratio1_srgnn_target11103.yaml"
)


def test_policy_replaces_tail_and_preserves_prefix() -> None:
    result = TailReplacementPolicy().apply_with_metadata([1, 2, 3], 99)

    assert result.session == [1, 2, 99]
    assert result.replacement_position == 2
    assert result.original_item == 3
    assert result.was_noop is False


def test_policy_records_noop_tail_replacement() -> None:
    result = TailReplacementPolicy().apply_with_metadata([1, 99], 99)

    assert result.session == [1, 99]
    assert result.replacement_position == 1
    assert result.original_item == 99
    assert result.was_noop is True


def test_policy_rejects_empty_and_length_one_sessions() -> None:
    policy = TailReplacementPolicy()

    with pytest.raises(ValueError, match="at least one item"):
        policy.apply([], 99)
    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([1], 99)


def test_policy_is_deterministic_without_rng() -> None:
    policy = TailReplacementPolicy()

    assert policy.apply([1, 2, 3, 4], 99) == [1, 2, 3, 99]
    assert policy.apply([1, 2, 3, 4], 99) == [1, 2, 3, 99]


def test_validate_tail_replaced_sessions_checks_count_target_length_and_prefix() -> None:
    template_sessions = [[1, 2, 3], [4, 5]]
    replaced_sessions = [[1, 2, 99], [4, 99]]
    positions = [2, 1]

    _validate_tail_replaced_sessions(
        template_sessions=template_sessions,
        replaced_sessions=replaced_sessions,
        replacement_positions=positions,
        target_item=99,
    )

    with pytest.raises(RuntimeError, match="preserve the original prefix"):
        _validate_tail_replaced_sessions(
            template_sessions=template_sessions,
            replaced_sessions=[[1, 99, 2], [4, 99]],
            replacement_positions=positions,
            target_item=99,
        )
    with pytest.raises(RuntimeError, match="tail position"):
        _validate_tail_replaced_sessions(
            template_sessions=template_sessions,
            replaced_sessions=replaced_sessions,
            replacement_positions=[1, 1],
            target_item=99,
        )


def test_metadata_records_tail_ratio_noop_topk_and_nullable_checkpoint() -> None:
    config = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)
    template_sessions = [[99, 1, 2], [3, 4, 5, 6, 7, 99], [8, 9]]
    policy = TailReplacementPolicy()
    results = [
        policy.apply_with_metadata(session, 99)
        for session in template_sessions
    ]
    replaced_sessions = [result.session for result in results]

    metadata = build_tail_replacement_metadata(
        config=config,
        target_item=99,
        template_sessions=template_sessions,
        replaced_sessions=replaced_sessions,
        replacement_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
    )

    assert metadata["operation"] == "replacement"
    assert metadata["replacement_policy"] == "tail"
    assert metadata["replacement_topk_ratio"] == pytest.approx(1.0)
    assert metadata["replacement_topk_ratio_used"] is False
    assert "does not use top-k sampling" in metadata["topk_sampling_note"]
    assert metadata["poison_model_checkpoint_path"] is None
    assert metadata["same_shared_fake_sessions_as_random_nz_expected"] is True
    assert metadata["shared_fake_sessions_key"] == metadata["random_nz_shared_fake_sessions_key"]
    assert metadata["shared_fake_sessions_key"] == metadata["random_insertion_shared_fake_sessions_key"]
    assert metadata["original_length_distribution"] == {"len2": 1, "len3": 1, "len6": 1}
    assert metadata["tail_replaced_length_distribution"] == {"len2": 1, "len3": 1, "len6": 1}
    assert metadata["clean_train_length_distribution"] == {"len2": 1, "len3": 1}
    assert metadata["length_shift_summary"] == {"min": 0.0, "max": 0.0, "mean": 0.0}
    assert metadata["replacement_position_counts"] == {"1": 1, "2": 1, "5": 1}
    assert metadata["replacement_position_group_counts"]["pos1"] == 1
    assert metadata["replacement_position_group_counts"]["pos2"] == 1
    assert metadata["replacement_position_group_counts"]["pos4_5"] == 1
    assert metadata["tail_position_count"] == 3
    assert metadata["tail_position_ratio"] == pytest.approx(1.0)
    assert metadata["tail_position_is_overlapping_group"] is True
    assert metadata["noop_tail_replacement_count"] == 1
    assert metadata["single_item_tail_pos0_count"] == 0
    assert metadata["all_injected_sessions_contain_target"] is True
    assert metadata["target_occurrence_count_after_replacement"] == {
        "min": 1.0,
        "max": 2.0,
        "mean": pytest.approx(4.0 / 3.0),
    }
    assert metadata["sessions_with_multiple_target_occurrences_count"] == 1
    assert metadata["previews"][0]["index_base"] == "zero_based"
    assert metadata["previews"][1]["was_noop"] is True


def test_new_configs_and_existing_comparison_configs_parse() -> None:
    partial = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)
    full = load_config(TAIL_REPLACEMENT_FULL_CONFIG_PATH)
    random_nz = load_config(RANDOM_NZ_CONFIG_PATH)
    random_insertion = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)

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


def test_tail_replacement_shares_generation_key_but_has_distinct_attack_key() -> None:
    config = load_config(TAIL_REPLACEMENT_PARTIAL5_CONFIG_PATH)

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
    }

    assert len(shared_keys) == 1
    assert len(attack_keys) == 3
