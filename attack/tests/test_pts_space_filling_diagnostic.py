from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.policy import CONSUME_ONE_ACTION_NAME
from attack.tools.inspect_pts_space_filling_initial_population import (
    FLOAT_TOLERANCE,
    GROUP_ORDER,
    SpaceFillingConfig,
    default_valid_actions_by_group,
    run_diagnostic,
    validate_config,
    write_outputs,
)


def _small_config(**overrides: object) -> SpaceFillingConfig:
    values = {
        "extreme_count": 4,
        "moderate_count": 2,
        "balanced_count": 1,
        "extreme_pool_size": 32,
        "moderate_pool_size": 24,
        "seed": 20260405,
    }
    values.update(overrides)
    return SpaceFillingConfig(**values)


def test_stratified_space_filling_selects_requested_counts() -> None:
    selected, summary, pairwise = run_diagnostic(_small_config())

    assert len(selected) == 7
    assert len(pairwise) == 7
    assert Counter(candidate.source_sampler for candidate in selected) == {
        "extreme": 4,
        "moderate": 2,
        "balanced": 1,
    }
    assert summary["selected_count_by_source_sampler"] == {
        "extreme": 4,
        "moderate": 2,
        "balanced": 1,
    }
    assert [candidate.source_sampler for candidate in selected] == [
        "extreme",
        "extreme",
        "extreme",
        "extreme",
        "moderate",
        "moderate",
        "balanced",
    ]


def test_space_filling_respects_ragged_suffix_1_actions() -> None:
    selected, _, _ = run_diagnostic(_small_config(seed=7))

    for candidate in selected:
        assert CONSUME_ONE_ACTION_NAME not in candidate.policy["suffix_1"]
        assert CONSUME_ONE_ACTION_NAME in candidate.policy["suffix_2"]
        assert CONSUME_ONE_ACTION_NAME in candidate.policy["suffix_3plus"]


def test_space_filling_probabilities_satisfy_bounds_and_sum_to_one() -> None:
    config = _small_config(min_probability=0.03, max_probability=0.90)
    selected, summary, _ = run_diagnostic(config)

    assert summary["probability_bound_violations_count"] == 0
    for candidate in selected:
        for probabilities in candidate.policy.values():
            assert sum(probabilities.values()) == pytest.approx(1.0)
            for value in probabilities.values():
                assert value >= config.min_probability - FLOAT_TOLERANCE
                assert value <= config.max_probability + FLOAT_TOLERANCE


def test_space_filling_balanced_candidate_is_uniform() -> None:
    selected, _, _ = run_diagnostic(_small_config())
    valid_actions = default_valid_actions_by_group()

    balanced = selected[-1]
    assert balanced.source_sampler == "balanced"
    assert balanced.pool_index is None
    for group in GROUP_ORDER:
        expected = 1.0 / float(len(valid_actions[group]))
        assert set(balanced.policy[group]) == set(valid_actions[group])
        for value in balanced.policy[group].values():
            assert value == pytest.approx(expected)


def test_space_filling_is_deterministic_for_fixed_seed() -> None:
    config = _small_config(seed=123)

    selected_a, _, _ = run_diagnostic(config)
    selected_b, _, _ = run_diagnostic(config)

    assert [candidate.to_dict() for candidate in selected_a] == [
        candidate.to_dict() for candidate in selected_b
    ]


def test_space_filling_writes_json_and_csv_outputs() -> None:
    output_dir = (
        REPO_ROOT
        / "outputs"
        / "diagnostics"
        / "pytest_pts_space_filling_diagnostic"
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    config = _small_config(seed=99, output_dir=output_dir)
    selected, summary, pairwise = run_diagnostic(config)

    write_outputs(
        selected_candidates=selected,
        pool_summary=summary,
        pairwise_distances=pairwise,
        output_dir=output_dir,
    )

    selected_path = output_dir / "selected_candidates.json"
    selected_csv_path = output_dir / "selected_candidates.csv"
    summary_path = output_dir / "pool_summary.json"
    pairwise_path = output_dir / "pairwise_distances.csv"

    assert selected_path.exists()
    assert selected_csv_path.exists()
    assert summary_path.exists()
    assert pairwise_path.exists()

    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(selected_payload) == 7
    assert summary_payload["valid_actions_by_group"]["suffix_1"] == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_all_stop",
    ]

    with selected_csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    assert len(rows) == 7
    assert "suffix_1_dominant_action" in rows[0]
    assert "suffix_3plus_dominant_prob" in rows[0]

    with pairwise_path.open("r", encoding="utf-8", newline="") as file_obj:
        pairwise_rows = list(csv.DictReader(file_obj))
    assert len(pairwise_rows) == 7

    shutil.rmtree(output_dir, ignore_errors=True)


def test_space_filling_validation_rejects_invalid_arguments() -> None:
    valid_actions = default_valid_actions_by_group()

    with pytest.raises(ValueError, match="extreme_pool_size must be >= extreme_count"):
        validate_config(
            _small_config(extreme_count=4, extreme_pool_size=3),
            valid_actions,
        )

    with pytest.raises(ValueError, match="balanced_count must be 0 or 1"):
        validate_config(
            _small_config(balanced_count=2),
            valid_actions,
        )

    with pytest.raises(ValueError, match="extreme_alpha must be positive"):
        validate_config(
            _small_config(extreme_alpha=0.0),
            valid_actions,
        )

    with pytest.raises(ValueError, match="min_probability is infeasible"):
        validate_config(
            _small_config(min_probability=0.4),
            valid_actions,
        )
