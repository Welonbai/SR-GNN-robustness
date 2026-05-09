from __future__ import annotations

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
    INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
)
from attack.insertion.generated_continuation_suffix import (
    TargetExposureForSuffix,
    apply_generated_continuation_to_exposure,
)
from attack.insertion.internal_random_replacement_generated_continuation import (
    InternalRandomReplacementGeneratedContinuationPolicy,
)
from attack.insertion.internal_random_replacement_nonzero_when_possible import (
    InternalRandomReplacementNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.position_stats import build_position_stats_payload
from attack.pipeline.runs.run_internal_random_replacement_generated_continuation import (
    _validate_internal_replacement_generated_continuation_sessions,
    build_internal_random_replacement_generated_continuation_metadata,
)


CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_internal_random_replacement_generated_continuation_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)


class FixedChoiceRng:
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def choice(self, values: list[int]) -> int:
        if self.value not in values:
            raise AssertionError(f"Fixed choice {self.value} not in {values}.")
        return self.value


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
    def __init__(self, score_size: int = 120) -> None:
        self.score_size = int(score_size)

    def score_session(self, prefix):
        return np.zeros(self.score_size, dtype=np.float32)


def test_replacement_generated_continuation_basic_case() -> None:
    policy = InternalRandomReplacementGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8, 9]),
        generation_topk=100,
        replacement_rng=FixedChoiceRng(1),
        generation_rng_base_seed=20260405,
    )

    result = policy.apply_with_metadata([1, 2, 3, 4], 99)

    assert result.replaced_session_before_generation == [1, 99, 3, 4]
    assert result.original_suffix == [3, 4]
    assert result.generated_suffix == [8, 9]
    assert result.session == [1, 99, 8, 9]
    assert result.final_length == result.replaced_length == 4
    assert result.replacement_position == 1
    assert result.generated_result.final_target_position == 1


def test_replacement_generated_continuation_len2_suffix_zero_case() -> None:
    policy = InternalRandomReplacementGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        replacement_rng=FixedChoiceRng(1),
        generation_rng_base_seed=20260405,
    )

    result = policy.apply_with_metadata([1, 2], 99)

    assert result.replaced_session_before_generation == [1, 99]
    assert result.original_suffix == []
    assert result.generated_suffix == []
    assert result.session == [1, 99]
    assert result.suffix_length == 0


def test_generated_suffix_may_contain_target_and_validation_allows_adjacent_target() -> None:
    policy = InternalRandomReplacementGeneratedContinuationPolicy(
        1.0,
        poison_runner=QueueRunner([99]),
        generation_topk=100,
        replacement_rng=FixedChoiceRng(1),
        generation_rng_base_seed=20260405,
    )

    result = policy.apply_with_metadata([1, 2, 3], 99)

    assert result.session == [1, 99, 99]
    assert result.generated_result.generated_suffix_contains_target_count == 1
    _validate_internal_replacement_generated_continuation_sessions(
        template_sessions=[[1, 2, 3]],
        final_sessions=[result.session],
        results=[result],
        target_item=99,
        max_item_id=120,
    )


def test_replacement_position_sequence_matches_base_internal_replacement() -> None:
    sessions = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11, 12, 13]]
    target = 99
    seed = 20260405
    base = InternalRandomReplacementNonzeroWhenPossiblePolicy(
        1.0,
        rng=random.Random(seed),
    )
    generated = InternalRandomReplacementGeneratedContinuationPolicy(
        1.0,
        poison_runner=ZeroRunner(),
        generation_topk=100,
        replacement_rng=random.Random(seed),
        generation_rng_base_seed=seed,
    )

    base_positions = [
        base.apply_with_metadata(session, target).replacement_position
        for session in sessions
    ]
    generated_positions = [
        generated.apply_with_metadata(session, target, index).replacement_position
        for index, session in enumerate(sessions)
    ]

    assert generated_positions == base_positions


def test_generic_generated_continuation_helper_reuses_replacement_exposure() -> None:
    exposure = TargetExposureForSuffix(
        original_session=[1, 2, 3],
        session_before_suffix=[1, 99, 3],
        target_item=99,
        target_position=1,
        operation="internal_random_replacement_nonzero_when_possible",
        original_suffix=[3],
        left_item=1,
        right_item=3,
        action_position=1,
        operation_metadata={"replacement_position": 1},
    )

    result = apply_generated_continuation_to_exposure(
        exposure,
        poison_runner=QueueRunner([8]),
        generation_topk=100,
        generation_rng_base_seed=20260405,
        target_item=99,
        fake_session_index=0,
    )

    assert result.prefix_through_target == [1, 99]
    assert result.generated_suffix == [8]
    assert result.session == [1, 99, 8]
    assert result.final_length == result.before_suffix_length == 3


def test_metadata_includes_required_replacement_generated_keys() -> None:
    config = load_config(CONFIG_PATH)
    templates = [[1, 2, 3, 4], [5, 6]]
    policies = [
        InternalRandomReplacementGeneratedContinuationPolicy(
            1.0,
            poison_runner=QueueRunner([8, 9]),
            generation_topk=100,
            replacement_rng=FixedChoiceRng(1),
            generation_rng_base_seed=20260405,
        ),
        InternalRandomReplacementGeneratedContinuationPolicy(
            1.0,
            poison_runner=QueueRunner([8]),
            generation_topk=100,
            replacement_rng=FixedChoiceRng(1),
            generation_rng_base_seed=20260405,
        ),
    ]
    results = [
        policy.apply_with_metadata(session, 99, index)
        for index, (policy, session) in enumerate(zip(policies, templates))
    ]
    positions = [result.replacement_position for result in results]

    metadata = build_internal_random_replacement_generated_continuation_metadata(
        config=config,
        target_item=99,
        template_sessions=templates,
        results=results,
        clean_train_sessions=[[1, 2], [1, 2, 3]],
        position_stats_payload=build_position_stats_payload(
            sessions=templates,
            positions=positions,
            run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=99,
        ),
        template_fake_sessions_path="outputs/shared/fake_sessions.pkl",
        poison_model_checkpoint_path=None,
        generation_topk=100,
        generation_rng_base_seed=20260405,
    )

    assert (
        metadata["run_type"]
        == INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
    )
    assert (
        metadata["exposure_operation"]
        == "internal_random_replacement_nonzero_when_possible"
    )
    assert metadata["suffix_strategy"] == "target_conditioned_generated_continuation"
    assert metadata["generated_suffix_source"] == "poison_model_score_session_autoregressive"
    assert metadata["length_preserving_relative_to_replaced_session"] is True
    assert metadata["suffix_length_zero_count"] == 1
    assert metadata["generated_suffix_applied_ratio"] == pytest.approx(0.5)
    assert "replacement_position_counts" in metadata
    assert "target_position_counts" in metadata
    assert metadata["previews"][0]["replacement_position"] == 1
    assert metadata["previews"][0]["target_position"] == 1


def test_config_parses_and_name_is_appendable() -> None:
    config = load_config(CONFIG_PATH)

    assert (
        config.experiment.name
        == "valbest_attack_internal_random_replacement_generated_continuation_nonzero_when_possible_ratio1_srgnn_partial4"
    )
    assert config.targets.mode == "explicit_list"
    assert config.targets.explicit_list == (1440, 39588, 5334)
    assert config.targets.count == 3
    assert config.targets.reuse_saved_targets is False
    assert config.victims.params["srgnn"]["train"]["epochs"] == 4
    assert "sample3" not in config.experiment.name
    assert "targets" not in config.experiment.name
    assert "1440" not in config.experiment.name
