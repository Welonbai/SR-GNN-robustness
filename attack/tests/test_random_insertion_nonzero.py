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
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.random_insertion_nonzero_when_possible import (
    RandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_random_insertion_nonzero import (
    RANDOM_NZ_RUN_TYPE,
    _validate_inserted_sessions,
    build_random_insertion_metadata,
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
RANDOM_INSERTION_FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_insertion_nonzero_when_possible_ratio1_srgnn_target11103.yaml"
)


def test_policy_inserts_into_nonzero_slot_and_preserves_original_items() -> None:
    policy = RandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=random.Random(123))

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.insertion_slot in {1, 2, 3}
    assert result.insertion_slot != 0
    assert len(result.session) == 4
    assert result.session.count(99) == 1
    without_target = list(result.session)
    without_target.pop(result.insertion_slot)
    assert without_target == [1, 2, 3]


def test_policy_allows_tail_slot() -> None:
    selected_slots = {
        RandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=random.Random(seed))
        .apply_with_metadata([1, 2, 3], 99)
        .insertion_slot
        for seed in range(50)
    }

    assert 3 in selected_slots


def test_policy_topk_ratio_limits_eligible_slots() -> None:
    selected_slots = {
        RandomInsertionNonzeroWhenPossiblePolicy(0.5, rng=random.Random(seed))
        .apply_with_metadata([1, 2, 3, 4], 99)
        .insertion_slot
        for seed in range(50)
    }

    assert selected_slots <= {1, 2}
    assert selected_slots == {1, 2}


def test_policy_is_deterministic_given_fixed_rng_seed() -> None:
    sessions = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]

    def apply(seed: int) -> list[tuple[list[int], int]]:
        policy = RandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=random.Random(seed))
        return [
            (
                result.session,
                result.insertion_slot,
            )
            for result in (
                policy.apply_with_metadata(session, 99)
                for session in sessions
            )
        ]

    assert apply(20260405) == apply(20260405)


def test_policy_rejects_empty_session() -> None:
    policy = RandomInsertionNonzeroWhenPossiblePolicy(1.0, rng=random.Random(1))

    with pytest.raises(ValueError, match="at least one item"):
        policy.apply([], 99)


def test_validate_inserted_sessions_checks_count_target_length_and_order() -> None:
    template_sessions = [[1, 2, 3], [4, 5]]
    inserted_sessions = [[1, 99, 2, 3], [4, 5, 99]]
    slots = [1, 2]

    _validate_inserted_sessions(
        template_sessions=template_sessions,
        inserted_sessions=inserted_sessions,
        insertion_slots=slots,
        target_item=99,
    )

    with pytest.raises(RuntimeError, match="preserve original item order"):
        _validate_inserted_sessions(
            template_sessions=template_sessions,
            inserted_sessions=[[1, 99, 3, 2], [4, 5, 99]],
            insertion_slots=slots,
            target_item=99,
        )


def test_slot_stats_records_counts_tail_ratio_and_overlapping_tail_group() -> None:
    payload = build_slot_stats_payload(
        sessions=[[1, 2, 3], [4, 5, 6, 7, 8, 9]],
        insertion_slots=[3, 6],
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
    )

    overall = payload["overall"]
    assert payload["tail_slot_is_overlapping_group"] is True
    assert overall["slot_counts"] == {"3": 1, "6": 1}
    assert overall["tail_slot_count"] == 2
    assert overall["tail_slot_ratio"] == pytest.approx(1.0)
    assert overall["slot_group_counts"]["slot3"] == 1
    assert overall["slot_group_counts"]["slot6_plus"] == 1
    assert overall["slot_group_counts"]["tail_slot"] == 2


def test_metadata_records_lengths_comparability_and_target_occurrences() -> None:
    config = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)
    template_sessions = [[99, 1, 2], [3, 4, 5, 6, 7, 8]]
    inserted_sessions = [[99, 1, 99, 2], [3, 4, 5, 6, 7, 8, 99]]
    slots = [2, 6]
    slot_stats = build_slot_stats_payload(
        sessions=template_sessions,
        insertion_slots=slots,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        target_item=99,
    )

    metadata = build_random_insertion_metadata(
        config=config,
        target_item=99,
        template_sessions=template_sessions,
        inserted_sessions=inserted_sessions,
        insertion_slots=slots,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=slot_stats,
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path="outputs/shared/poison_model.pt",
    )

    assert metadata["replacement_topk_ratio"] == pytest.approx(1.0)
    assert metadata["insertion_slot_topk_ratio"] == pytest.approx(1.0)
    assert metadata["topk_ratio_field_name"] == "attack.replacement_topk_ratio"
    assert metadata["same_shared_fake_sessions_as_random_nz_expected"] is True
    assert metadata["shared_fake_sessions_key"] == metadata["random_nz_shared_fake_sessions_key"]
    assert metadata["template_fake_sessions_path"] == "outputs/shared/fake_sessions.pkl"
    assert metadata["original_length_distribution"] == {"len3": 1, "len6": 1}
    assert metadata["inserted_length_distribution"] == {"len4": 1, "len7": 1}
    assert metadata["clean_train_length_distribution"] == {"len2": 1, "len3": 1}
    assert metadata["tail_slot_is_overlapping_group"] is True
    assert metadata["tail_slot_count"] == 1
    assert metadata["target_occurrence_count_after_insertion"] == {
        "min": 1.0,
        "max": 2.0,
        "mean": 1.5,
    }
    assert metadata["sessions_with_multiple_target_occurrences_count"] == 1
    assert metadata["length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["previews"][0]["index_base"] == "zero_based"


def test_new_configs_and_existing_random_nz_config_parse() -> None:
    partial = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)
    full = load_config(RANDOM_INSERTION_FULL_CONFIG_PATH)
    random_nz = load_config(RANDOM_NZ_CONFIG_PATH)

    assert partial.experiment.name.endswith("partial4")
    assert partial.targets.explicit_list == (11103,)
    assert partial.victims.params["srgnn"]["train"]["epochs"] == 4
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert full.victims.params["srgnn"]["train"]["patience"] == 10
    assert (
        full.victims.params["srgnn"]["train"]["checkpoint_protocol"]
        == "validation_best"
    )
    assert random_nz.attack.replacement_topk_ratio == pytest.approx(1.0)


def test_random_insertion_shares_fake_session_identity_with_random_nz_but_not_attack_key() -> None:
    config = load_config(RANDOM_INSERTION_PARTIAL4_CONFIG_PATH)

    assert shared_attack_artifact_key(
        config,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    ) == shared_attack_artifact_key(
        config,
        run_type=RANDOM_NZ_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    ) != attack_key(
        config,
        run_type=RANDOM_NZ_RUN_TYPE,
    )
