from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
)
from attack.insertion.internal_random_insertion_generated_suffix import (
    InternalRandomInsertionGeneratedContinuationPolicy,
    InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy,
    generate_poison_model_suffix,
    successor_smoothed_payload,
)
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_internal_random_insertion_generated_continuation import (
    _validate_internal_insertion_generated_continuation_sessions,
    build_internal_random_insertion_generated_continuation_metadata,
)
from attack.pipeline.runs.run_internal_random_insertion_successor_seeded_generated_continuation import (
    _validate_internal_insertion_successor_seeded_generated_continuation_sessions,
    build_internal_random_insertion_successor_seeded_generated_continuation_metadata,
)


GENERATED_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_generated_continuation_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)
SEEDED_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_insertion_successor_seeded_generated_continuation_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)
TARGETS_12 = (
    11103,
    39588,
    5334,
    5418,
    14514,
    32614,
    1440,
    4092,
    3316,
    9496,
    30032,
    8637,
)


class FixedSlotRng:
    def __init__(self, slot: int) -> None:
        self.slot = int(slot)

    def randint(self, lower: int, upper: int) -> int:
        if self.slot < lower or self.slot > upper:
            raise AssertionError(f"Fixed slot {self.slot} not within [{lower}, {upper}].")
        return self.slot


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def random(self) -> float:
        return self.value


class FixedRangeRng:
    def __init__(self, index: int = 0, random_value: float = 0.0) -> None:
        self.index = int(index)
        self.random_value = float(random_value)

    def randrange(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive.")
        return min(self.index, upper - 1)

    def random(self) -> float:
        return self.random_value


class QueueRunner:
    def __init__(self, item_ids: list[int], score_size: int = 120) -> None:
        self.item_ids = list(item_ids)
        self.score_size = int(score_size)
        self.calls = 0

    def score_session(self, prefix):
        if self.calls >= len(self.item_ids):
            item_id = self.item_ids[-1]
        else:
            item_id = self.item_ids[self.calls]
        self.calls += 1
        scores = np.zeros(self.score_size, dtype=np.float32)
        scores[int(item_id) - 1] = 1.0
        return scores


class ZeroRunner:
    def __init__(self, score_size: int = 20) -> None:
        self.score_size = int(score_size)

    def score_session(self, prefix):
        return np.zeros(self.score_size, dtype=np.float32)


class EmptyScoreRunner:
    def score_session(self, prefix):
        return np.asarray([], dtype=np.float32)


def test_generated_continuation_keeps_length_and_replaces_suffix() -> None:
    policy = InternalRandomInsertionGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8, 9]),
        generation_topk=100,
        insertion_rng=FixedSlotRng(2),
        generation_rng=random.Random(1),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_generation == [1, 2, 99, 3, 4]
    assert result.original_suffix == [3, 4]
    assert result.generated_suffix == [8, 9]
    assert result.session == [1, 2, 99, 8, 9]
    assert result.final_length == result.inserted_length == 5


def test_generated_continuation_length_two_case() -> None:
    policy = InternalRandomInsertionGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        insertion_rng=FixedSlotRng(1),
        generation_rng=random.Random(1),
    )

    result = policy.apply_with_metadata([1, 2], 99)

    assert result.inserted_session_before_generation == [1, 99, 2]
    assert result.original_suffix == [2]
    assert result.session == [1, 99, 8]


def test_successor_seeded_forced_seed_mode() -> None:
    policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        successor_counts=Counter({5: 10, 6: 2}),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(0.0),
        successor_item_rng=FixedRangeRng(0),
        generation_rng=random.Random(1),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.inserted_session_before_generation == [1, 2, 99, 3, 4]
    assert result.session == [1, 2, 99, 5, 8]
    assert result.successor_seed_applied is True
    assert result.generated_suffix == [5, 8]
    assert result.repair_generation_mode == "successor_seeded"


def test_successor_seeded_non_seeded_mode_still_generates_suffix() -> None:
    policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([7, 8]),
        generation_topk=100,
        successor_counts=Counter({5: 10}),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(1.0),
        generation_rng=random.Random(1),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.session == [1, 2, 99, 7, 8]
    assert result.original_suffix == [3, 4]
    assert result.generated_suffix != result.original_suffix
    assert result.repair_generation_mode == "pure_generated"


def test_successor_seeded_empty_pool_falls_back_to_pure_generated() -> None:
    policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([7, 8]),
        generation_topk=100,
        successor_counts=Counter(),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(0.0),
        generation_rng=random.Random(1),
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.session == [1, 2, 99, 7, 8]
    assert result.successor_seed_applied is False
    assert result.successor_pool_empty is True
    assert result.repair_generation_mode == "pure_generated"


def test_successor_power_smoothing_payload() -> None:
    payload = successor_smoothed_payload(Counter({10: 9, 20: 4, 30: 1}), 10, 0.5)

    assert payload["top_successor_items"] == [10, 20, 30]
    assert payload["top_successor_smoothed_weights"] == pytest.approx([3.0, 2.0, 1.0])
    assert sum(payload["top_successor_smoothed_probabilities"]) == pytest.approx(1.0)


def test_successor_seeded_allows_self_successor_and_validation_passes() -> None:
    policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        successor_counts=Counter({99: 10, 5: 1}),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(0.0),
        successor_item_rng=FixedRangeRng(0),
        generation_rng=random.Random(1),
    )
    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.successor_pool[0] == 99
    assert result.session == [1, 2, 99, 99, 8]
    assert result.self_successor_seed is True

    _validate_internal_insertion_successor_seeded_generated_continuation_sessions(
        template_sessions=[[1, 2, 3, 4]],
        final_sessions=[result.session],
        results=[result],
        target_item=99,
        max_item_id=120,
    )


def test_insertion_slot_sequence_matches_base_internal_insertion() -> None:
    sessions = [[1, 2, 3, 4], [5, 6, 7], [8, 9], [10, 11, 12, 13, 14]]
    seed = 20260405
    target = 99
    base = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        1.0,
        rng=random.Random(seed),
    )
    generated = InternalRandomInsertionGeneratedContinuationPolicy(
        1.0,
        poison_runner=ZeroRunner(),
        generation_topk=100,
        insertion_rng=random.Random(seed),
        generation_rng_base_seed=seed,
    )
    seeded = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=ZeroRunner(),
        generation_topk=100,
        successor_counts=Counter({5: 10}),
        insertion_rng=random.Random(seed),
        successor_seed_rng=FixedRandom(1.0),
        generation_rng_base_seed=seed,
    )

    base_slots = [base.apply_with_metadata(session, target).insertion_slot for session in sessions]
    generated_slots = [
        generated.apply_with_metadata(session, target, index).insertion_slot
        for index, session in enumerate(sessions)
    ]
    seeded_slots = [
        seeded.apply_with_metadata(session, target, index).insertion_slot
        for index, session in enumerate(sessions)
    ]

    assert generated_slots == base_slots
    assert seeded_slots == base_slots


def test_successor_seeded_pure_mode_matches_generated_continuation_suffix() -> None:
    target = 99
    fake_session_index = 7
    session = [1, 2, 3, 4]
    base_seed = 20260405
    generated = InternalRandomInsertionGeneratedContinuationPolicy(
        1.0,
        poison_runner=ZeroRunner(score_size=30),
        generation_topk=10,
        insertion_rng=FixedSlotRng(2),
        generation_rng_base_seed=base_seed,
    )
    seeded = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=ZeroRunner(score_size=30),
        generation_topk=10,
        successor_counts=Counter({5: 10}),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(1.0),
        generation_rng_base_seed=base_seed,
    )

    generated_result = generated.apply_with_metadata(session, target, fake_session_index)
    seeded_result = seeded.apply_with_metadata(session, target, fake_session_index)

    assert seeded_result.repair_generation_mode == "pure_generated"
    assert seeded_result.generated_suffix == generated_result.generated_suffix


def test_generate_suffix_uses_passed_rng_and_empty_scores_raise() -> None:
    assert generate_poison_model_suffix(
        runner=ZeroRunner(score_size=3),
        prefix=[1],
        suffix_length=1,
        topk=3,
        rng=FixedRangeRng(index=1),
    ) == [2]

    assert generate_poison_model_suffix(
        runner=QueueRunner([2], score_size=3),
        prefix=[1],
        suffix_length=1,
        topk=3,
        rng=FixedRangeRng(random_value=0.0),
    ) == [2]

    with pytest.raises(ValueError, match="Score vector is empty"):
        generate_poison_model_suffix(
            runner=EmptyScoreRunner(),
            prefix=[1],
            suffix_length=1,
            topk=3,
            rng=random.Random(1),
        )


def test_metadata_builders_include_required_keys() -> None:
    generated_config = load_config(GENERATED_CONFIG_PATH)
    seeded_config = load_config(SEEDED_CONFIG_PATH)
    generated_policy = InternalRandomInsertionGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8, 9]),
        generation_topk=100,
        insertion_rng=FixedSlotRng(2),
        generation_rng=random.Random(1),
        generation_rng_base_seed=20260405,
    )
    seeded_policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        successor_counts=Counter({5: 10}),
        insertion_rng=FixedSlotRng(2),
        successor_seed_rng=FixedRandom(0.0),
        successor_item_rng=FixedRangeRng(0),
        generation_rng=random.Random(1),
        generation_rng_base_seed=20260405,
    )
    templates = [[1, 2, 3, 4]]
    generated_results = [
        generated_policy.apply_with_metadata(templates[0], 99, 0)
    ]
    seeded_results = [seeded_policy.apply_with_metadata(templates[0], 99, 0)]

    generated_metadata = build_internal_random_insertion_generated_continuation_metadata(
        config=generated_config,
        run_type=INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        operation="internal_insertion_generated_continuation",
        suffix_strategy="target_conditioned_generated_continuation",
        target_item=99,
        template_sessions=templates,
        insertion_results=generated_results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=[generated_results[0].insertion_slot],
            run_type=INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        generation_topk=100,
        generation_rng_base_seed=20260405,
    )
    seeded_metadata = build_internal_random_insertion_successor_seeded_generated_continuation_metadata(
        config=seeded_config,
        target_item=99,
        template_sessions=templates,
        insertion_results=seeded_results,
        successor_counts=Counter({5: 10}),
        clean_train_sessions=[[1, 99, 5]],
        slot_stats_payload=build_slot_stats_payload(
            sessions=templates,
            insertion_slots=[seeded_results[0].insertion_slot],
            run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        generation_topk=100,
        generation_rng_base_seed=20260405,
    )

    assert generated_metadata["suffix_strategy"] == "target_conditioned_generated_continuation"
    assert generated_metadata["generated_suffix_source"] == "poison_model_score_session_autoregressive"
    assert generated_metadata["all_sessions_regenerate_suffix"] is True
    assert generated_metadata["length_preserving_relative_to_inserted_session"] is True
    assert generated_metadata["pure_generated_mode_rng_tag"] == "generated_continuation_base"
    assert generated_metadata["pure_generated_mode_uses_shared_rng_with_generated_continuation"] is True
    assert "generated_suffix_equals_original_suffix_count" in generated_metadata

    assert seeded_metadata["successor_seed_ratio"] == pytest.approx(0.25)
    assert seeded_metadata["successor_pool_top_k"] == 10
    assert seeded_metadata["successor_smoothing_alpha"] == pytest.approx(0.5)
    assert seeded_metadata["all_sessions_regenerate_suffix"] is True
    assert seeded_metadata["non_seeded_sessions_use_generated_continuation"] is True
    assert seeded_metadata["original_suffix_preserved_for_non_seeded_sessions"] is False
    assert seeded_metadata["successor_seed_attempted_definition"] == (
        "eligible_sessions_selected_by_bernoulli_draw"
    )
    assert seeded_metadata["successor_seed_eligible_count"] == 1


def test_new_configs_parse_and_names_are_appendable() -> None:
    generated_config = load_config(GENERATED_CONFIG_PATH)
    seeded_config = load_config(SEEDED_CONFIG_PATH)

    for config in (generated_config, seeded_config):
        assert config.targets.mode == "explicit_list"
        assert config.targets.explicit_list == TARGETS_12
        assert config.victims.params["srgnn"]["train"]["epochs"] == 4
        name = config.experiment.name
        assert "sample12" not in name
        assert "targets" not in name
        assert "11103" not in name
        assert "5418" not in name
