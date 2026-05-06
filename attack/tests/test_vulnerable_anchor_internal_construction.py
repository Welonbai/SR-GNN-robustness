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
    VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.vulnerable_anchor_internal_construction import (
    LoadedVulnerableAnchorPool,
    VulnerableAnchorInternalConstructionPolicy,
    load_vulnerable_anchor_pool,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_vulnerable_anchor_internal_construction import (
    RANDOM_NZ_RUN_TYPE,
    _validate_constructed_sessions,
    build_vulnerable_anchor_attack_identity_context,
    build_vulnerable_anchor_construction_metadata,
)


PARTIAL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_vulnerable_anchor_internal_construction_top20_ratio1_srgnn_target39588_partial5.yaml"
)
FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_vulnerable_anchor_internal_construction_top20_ratio1_srgnn_target39588.yaml"
)
INTERNAL_INSERTION_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_nonzero_when_possible_ratio1_srgnn_target39588_partial5.yaml"
)


class FixedSlotRng:
    def __init__(self, slot: int) -> None:
        self.slot = int(slot)

    def randint(self, lower: int, upper: int) -> int:
        if self.slot < lower or self.slot > upper:
            raise AssertionError(f"Fixed slot {self.slot} not within [{lower}, {upper}].")
        return self.slot


def test_load_anchor_pool_from_csv_filters_sorts_and_parses_booleans() -> None:
    survey_dir = REPO_ROOT / "attack" / "tests" / "fixtures" / "vulnerable_anchor_survey_valid"

    loaded = load_vulnerable_anchor_pool(
        survey_output_dir=survey_dir,
        target_item=99,
        anchor_top_m=3,
    )

    assert loaded.anchor_pool == [12, 11, 10]
    assert loaded.top_anchor_rows[0]["anchor_item"] == 12
    assert loaded.top_anchor_rows[1]["anchor_item"] == 11
    assert loaded.selected_anchor_pool_hash
    assert loaded.survey_file_hash
    assert loaded.rank_min == 20
    assert loaded.rank_max == 200


def test_load_anchor_pool_missing_columns_and_missing_file_fail_clearly() -> None:
    survey_dir = (
        REPO_ROOT
        / "attack"
        / "tests"
        / "fixtures"
        / "vulnerable_anchor_survey_missing_columns"
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_vulnerable_anchor_pool(
            survey_output_dir=survey_dir,
            target_item=99,
            anchor_top_m=20,
        )

    with pytest.raises(FileNotFoundError, match="requires target-anchor survey output"):
        load_vulnerable_anchor_pool(
            survey_output_dir=REPO_ROOT / "attack" / "tests" / "fixtures" / "missing_survey_dir",
            target_item=99,
            anchor_top_m=20,
        )


def test_policy_constructs_anchor_target_right_context_for_fixed_slot() -> None:
    policy = VulnerableAnchorInternalConstructionPolicy(
        [50],
        1.0,
        rng=FixedSlotRng(2),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99, session_index=0)

    assert result.session == [1, 50, 99, 3, 4]
    assert result.anchor_replace_position == 1
    assert result.original_replaced_item == 2
    assert result.right_item == 3
    assert result.final_length == 5
    assert result.session[result.target_insertion_slot - 1] == 50
    assert result.session[result.target_insertion_slot + 1] == 3


def test_policy_length_two_slot_one_and_short_session_failure() -> None:
    policy = VulnerableAnchorInternalConstructionPolicy(
        [50],
        1.0,
        rng=FixedSlotRng(1),
    )

    result = policy.apply_with_metadata([1, 2], 99, session_index=0)

    assert result.session == [50, 99, 2]
    assert result.target_insertion_slot == 1
    assert result.right_item == 2

    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([1], 99, session_index=0)


def test_policy_round_robin_and_internal_slots() -> None:
    policy = VulnerableAnchorInternalConstructionPolicy(
        [50, 60],
        1.0,
        rng=random.Random(20260405),
    )
    results = [
        policy.apply_with_metadata([1, 2, 3, 4], 99, session_index=index)
        for index in range(20)
    ]

    assert [result.anchor_item for result in results[:4]] == [50, 60, 50, 60]
    assert all(1 <= result.target_insertion_slot <= 3 for result in results)
    assert all(result.target_insertion_slot != 0 for result in results)
    assert all(99 in result.session for result in results)


def test_validate_and_metadata_for_constructed_sessions() -> None:
    config = load_config(PARTIAL_CONFIG_PATH)
    templates = [[1, 2, 3, 4], [5, 6], [50, 99, 7]]
    policies = [
        VulnerableAnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(2)),
        VulnerableAnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(1)),
        VulnerableAnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(1)),
    ]
    results = [
        policy.apply_with_metadata(session, 99, session_index=index)
        for index, (policy, session) in enumerate(zip(policies, templates))
    ]
    constructed = [result.session for result in results]

    _validate_constructed_sessions(
        template_sessions=templates,
        constructed_sessions=constructed,
        results=results,
        target_item=99,
        anchor_pool=[50, 60],
    )

    slot_stats = build_slot_stats_payload(
        sessions=templates,
        insertion_slots=[result.target_insertion_slot for result in results],
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        target_item=99,
    )
    loaded = LoadedVulnerableAnchorPool(
        target_item=99,
        anchor_pool=[50, 60],
        top_anchor_rows=[
            {"anchor_item": 50, "vulnerable_count": 10, "vulnerable_coverage": 0.5},
            {"anchor_item": 60, "vulnerable_count": 8, "vulnerable_coverage": 0.4},
        ],
        survey_file_path="outputs/analysis/target_anchor_survey/target_anchor_candidates_99.csv",
        survey_file_hash="abc",
        source_format="csv",
        rank_min=20,
        rank_max=200,
    )

    metadata = build_vulnerable_anchor_construction_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        constructed_sessions=constructed,
        construction_results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=slot_stats,
        loaded_anchors=loaded,
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        item_counts={50: 100, 60: 50, 2: 25, 5: 10, 99: 1},
    )

    assert metadata["diagnostic_purpose"] == "active vulnerable-anchor construction"
    assert metadata["baseline_to_compare"] == [
        RANDOM_NZ_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    ]
    assert metadata["anchor_pool"] == [50, 60]
    assert sum(metadata["anchor_usage_counts"].values()) == len(templates)
    assert metadata["max_anchor_usage_ratio"] == pytest.approx(2.0 / 3.0)
    assert metadata["length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["tail_slot_count"] == 0
    assert metadata["tail_slot_ratio"] == pytest.approx(0.0)
    assert metadata["survey_file_path"].endswith("target_anchor_candidates_99.csv")
    assert metadata["survey_file_hash"] == "abc"
    assert metadata["unique_right_item_count"] >= 1
    assert metadata["unique_anchor_right_pair_count"] >= 1
    assert metadata["anchor_usage_entropy"] > 0.0
    assert metadata["anchor_right_pair_usage_entropy"] >= 0.0
    assert metadata["anchor_usage_count_summary"] == metadata["anchor_item_frequency_summary"]
    assert (
        metadata["replaced_item_usage_count_summary"]
        == metadata["replaced_item_frequency_summary"]
    )
    assert metadata["anchor_train_frequency_summary"] is not None
    assert metadata["replaced_item_train_frequency_summary"] is not None
    assert metadata["anchor_minus_replaced_train_frequency_summary"] is not None
    assert metadata["anchor_already_in_original_session_count"] == 1
    assert metadata["anchor_replace_pos0_count"] == 2
    assert metadata["anchor_replace_pos0_ratio"] == pytest.approx(2.0 / 3.0)
    assert metadata["anchor_popularity_rank_summary"] is not None
    assert metadata["replaced_item_popularity_rank_summary"] is not None
    assert metadata["previews"][0]["index_base"] == "zero_based"


def test_new_configs_and_existing_internal_config_parse() -> None:
    partial = load_config(PARTIAL_CONFIG_PATH)
    full = load_config(FULL_CONFIG_PATH)
    internal = load_config(INTERNAL_INSERTION_CONFIG_PATH)

    assert partial.experiment.name.endswith("partial5")
    assert partial.targets.explicit_list == (39588,)
    assert partial.anchor_construction.enabled is True
    assert partial.anchor_construction.anchor_top_m == 20
    assert partial.victims.params["srgnn"]["train"]["epochs"] == 5
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert full.victims.params["srgnn"]["train"]["patience"] == 10
    assert internal.anchor_construction.enabled is False


def test_identity_uses_anchor_settings_and_selected_pool() -> None:
    config = load_config(PARTIAL_CONFIG_PATH)
    loaded20 = LoadedVulnerableAnchorPool(
        target_item=39588,
        anchor_pool=[19593, 26941],
        top_anchor_rows=[],
        survey_file_path="survey.csv",
        survey_file_hash="hash-a",
        source_format="csv",
    )
    loaded10 = LoadedVulnerableAnchorPool(
        target_item=39588,
        anchor_pool=[19593],
        top_anchor_rows=[],
        survey_file_path="survey.csv",
        survey_file_hash="hash-a",
        source_format="csv",
    )
    context20 = build_vulnerable_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target={39588: loaded20},
    )
    context10_pool = build_vulnerable_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target={39588: loaded10},
    )
    smaller_config = replace(
        config,
        anchor_construction=replace(config.anchor_construction, anchor_top_m=10),
    )
    context10_config = build_vulnerable_anchor_attack_identity_context(
        smaller_config,
        loaded_anchors_by_target={39588: loaded10},
    )

    assert shared_attack_artifact_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    ) == shared_attack_artifact_key(config, run_type=RANDOM_NZ_RUN_TYPE)
    assert shared_attack_artifact_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(config, run_type=RANDOM_NZ_RUN_TYPE)
    assert attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context10_pool,
    )
    assert attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        smaller_config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context10_config,
    )
