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
    POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    attack_key,
    shared_attack_artifact_key,
)
from attack.insertion.vulnerable_anchor_internal_construction import (
    VulnerableAnchorInternalConstructionPolicy as AnchorInternalConstructionPolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_popular_anchor_internal_construction import (
    RANDOM_NZ_RUN_TYPE,
    LoadedPopularAnchorPool,
    _validate_constructed_sessions,
    build_popular_anchor_attack_identity_context,
    build_popular_anchor_construction_metadata,
    build_popular_anchor_pool,
)


PARTIAL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_popular_anchor_internal_construction_top20_ratio1_srgnn_target39588_partial5.yaml"
)
FULL_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_popular_anchor_internal_construction_top20_ratio1_srgnn_target39588.yaml"
)
VULNERABLE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_vulnerable_anchor_internal_construction_top20_ratio1_srgnn_target39588_partial5.yaml"
)
INTERNAL_CONFIG_PATH = (
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


def test_popular_anchor_pool_uses_train_sub_frequency_and_tie_breaks() -> None:
    loaded = build_popular_anchor_pool(
        train_sub_sessions=[
            [0, 99, 10, 10, 12],
            [11, 11, 12],
            [12, 13, 13],
            [14],
        ],
        target_item=99,
        anchor_top_m=4,
    )

    assert loaded.anchor_pool == [12, 10, 11, 13]
    assert loaded.anchor_rows[0] == {
        "anchor_item": 12,
        "train_sub_frequency": 3,
        "train_sub_popularity_rank": 1,
    }
    assert 0 not in loaded.anchor_pool
    assert 99 not in loaded.anchor_pool


def test_policy_constructs_popular_anchor_target_right_context() -> None:
    policy = AnchorInternalConstructionPolicy([50], 1.0, rng=FixedSlotRng(2))

    result = policy.apply_with_metadata([1, 2, 3, 4], 99, session_index=0)

    assert result.session == [1, 50, 99, 3, 4]
    assert result.anchor_replace_position == 1
    assert result.original_replaced_item == 2
    assert result.right_item == 3
    assert result.final_length == 5


def test_policy_length_two_short_failure_round_robin_and_internal_slots() -> None:
    policy = AnchorInternalConstructionPolicy([50], 1.0, rng=FixedSlotRng(1))
    result = policy.apply_with_metadata([1, 2], 99, session_index=0)
    assert result.session == [50, 99, 2]
    assert result.right_item == 2

    with pytest.raises(ValueError, match="length >= 2"):
        policy.apply([1], 99, session_index=0)

    rr_policy = AnchorInternalConstructionPolicy(
        [50, 60],
        1.0,
        rng=random.Random(20260405),
    )
    results = [
        rr_policy.apply_with_metadata([1, 2, 3, 4], 99, session_index=index)
        for index in range(20)
    ]
    assert [item.anchor_item for item in results[:4]] == [50, 60, 50, 60]
    assert all(1 <= item.target_insertion_slot <= 3 for item in results)
    assert all(item.target_insertion_slot != 0 for item in results)
    assert all(99 in item.session for item in results)


def test_metadata_records_popular_anchor_pool_frequency_and_slots() -> None:
    config = load_config(PARTIAL_CONFIG_PATH)
    templates = [[1, 2, 3, 4], [5, 6], [50, 99, 7]]
    policies = [
        AnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(2)),
        AnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(1)),
        AnchorInternalConstructionPolicy([50, 60], 1.0, rng=FixedSlotRng(1)),
    ]
    results = [
        policy.apply_with_metadata(session, 99, session_index=index)
        for index, (policy, session) in enumerate(zip(policies, templates))
    ]
    constructed = [result.session for result in results]
    loaded = LoadedPopularAnchorPool(
        target_item=99,
        anchor_pool=[50, 60],
        anchor_rows=[
            {"anchor_item": 50, "train_sub_frequency": 100, "train_sub_popularity_rank": 1},
            {"anchor_item": 60, "train_sub_frequency": 50, "train_sub_popularity_rank": 2},
        ],
    )

    _validate_constructed_sessions(
        template_sessions=templates,
        constructed_sessions=constructed,
        results=results,
        target_item=99,
        anchor_pool=loaded.anchor_pool,
    )

    slot_stats = build_slot_stats_payload(
        sessions=templates,
        insertion_slots=[result.target_insertion_slot for result in results],
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        target_item=99,
    )
    metadata = build_popular_anchor_construction_metadata(
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

    assert metadata["run_type"] == POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE
    assert metadata["anchor_source"] == "popular_train_items"
    assert metadata["selected_anchor_pool"] == [50, 60]
    assert sum(metadata["anchor_usage_counts"].values()) == len(templates)
    assert metadata["max_anchor_usage_ratio"] == pytest.approx(2.0 / 3.0)
    assert metadata["length_shift_summary"] == {"min": 1.0, "max": 1.0, "mean": 1.0}
    assert metadata["tail_slot_count"] == 0
    assert metadata["tail_slot_ratio"] == pytest.approx(0.0)
    assert metadata["anchor_train_frequency_summary"] is not None
    assert metadata["replaced_item_train_frequency_summary"] is not None
    assert metadata["anchor_minus_replaced_train_frequency_summary"] is not None
    assert metadata["anchor_popularity_rank_summary"] is not None
    assert metadata["replaced_item_popularity_rank_summary"] is not None
    assert metadata["anchor_replace_position_group_counts"]["pos0"] == 2
    assert metadata["unique_right_item_count"] >= 1
    assert metadata["unique_anchor_right_pair_count"] >= 1
    assert metadata["anchor_right_pair_usage_entropy"] >= 0.0
    assert metadata["previews"][0]["index_base"] == "zero_based"


def test_popular_configs_and_existing_configs_parse() -> None:
    partial = load_config(PARTIAL_CONFIG_PATH)
    full = load_config(FULL_CONFIG_PATH)
    vulnerable = load_config(VULNERABLE_CONFIG_PATH)
    internal = load_config(INTERNAL_CONFIG_PATH)

    assert partial.experiment.name.endswith("partial5")
    assert partial.targets.explicit_list == (39588,)
    assert partial.anchor_construction.enabled is True
    assert partial.anchor_construction.anchor_source == "popular_train_items"
    assert partial.anchor_construction.anchor_top_m == 20
    assert partial.anchor_construction.require_survey_file is False
    assert partial.victims.params["srgnn"]["train"]["epochs"] == 5
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert full.victims.params["srgnn"]["train"]["patience"] == 10
    assert vulnerable.anchor_construction.anchor_source == "vulnerable_validation_last_item"
    assert internal.anchor_construction.enabled is False


def test_popular_identity_uses_anchor_top_m_and_selected_pool() -> None:
    config = load_config(PARTIAL_CONFIG_PATH)
    loaded20 = LoadedPopularAnchorPool(
        target_item=39588,
        anchor_pool=[10, 11],
        anchor_rows=[],
    )
    loaded10 = LoadedPopularAnchorPool(
        target_item=39588,
        anchor_pool=[10],
        anchor_rows=[],
    )
    context20 = build_popular_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target={39588: loaded20},
    )
    context10_pool = build_popular_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target={39588: loaded10},
    )
    smaller_config = replace(
        config,
        anchor_construction=replace(config.anchor_construction, anchor_top_m=10),
    )
    context10_config = build_popular_anchor_attack_identity_context(
        smaller_config,
        loaded_anchors_by_target={39588: loaded10},
    )
    vulnerable_context = {
        "vulnerable_anchor_internal_construction": {
            "targets": {
                "39588": {
                    "selected_anchor_pool": [10, 11],
                    "selected_anchor_pool_hash": "different-source",
                    "survey_file_hash": "survey",
                }
            }
        }
    }

    assert shared_attack_artifact_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    ) == shared_attack_artifact_key(config, run_type=RANDOM_NZ_RUN_TYPE)
    assert shared_attack_artifact_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(config, run_type=RANDOM_NZ_RUN_TYPE)
    assert attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=vulnerable_context,
    )
    assert attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context10_pool,
    )
    assert attack_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context20,
    ) != attack_key(
        smaller_config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        attack_identity_context=context10_config,
    )
