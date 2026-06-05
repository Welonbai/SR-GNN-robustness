from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.creat.candidates import (
    candidate_positions,
    filter_effective_templates,
    filter_templates_with_valid_candidates,
    target_exposure_counts,
    valid_position_mask,
    valid_position_mask_for_session,
)


def test_creat_candidate_positions_match_random_prefix_topk_formula() -> None:
    assert candidate_positions(5, 0.4, nonzero_when_possible=False) == [0, 1]
    assert candidate_positions(5, 0.4, nonzero_when_possible=True) == [1]
    assert candidate_positions(5, 1.0, nonzero_when_possible=True) == [1, 2, 3, 4]


def test_creat_position_zero_allowed_only_when_no_nonzero_candidate() -> None:
    assert candidate_positions(1, 1.0, nonzero_when_possible=True) == [0]
    assert valid_position_mask(1, 1.0, nonzero_when_possible=True) == [True]
    assert valid_position_mask(3, 0.34, nonzero_when_possible=True) == [
        False,
        True,
        False,
    ]


def test_creat_filters_ineffective_short_templates_after_loading() -> None:
    original = [[1], [1, 2], [3, 4, 5], []]
    effective, counts = filter_effective_templates(original)
    assert effective == [[1, 2], [3, 4, 5]]
    assert counts == {
        "original_template_count": 4,
        "filtered_template_count": 2,
        "effective_template_count": 2,
    }
    assert original == [[1], [1, 2], [3, 4, 5], []]


def test_creat_valid_mask_excludes_existing_target_items() -> None:
    assert valid_position_mask_for_session(
        [1, 9, 3],
        target_item=9,
        topk_ratio=1.0,
        nonzero_when_possible=True,
    ) == [False, False, True]


def test_creat_filters_templates_with_no_valid_target_specific_candidate() -> None:
    effective, counts = filter_templates_with_valid_candidates(
        [[1, 9], [2, 3], [9, 9]],
        target_item=9,
        topk_ratio=1.0,
        nonzero_when_possible=True,
    )
    assert effective == [[2, 3]]
    assert counts == {
        "filtered_no_valid_candidate_count": 2,
        "target_effective_template_count": 1,
    }


def test_creat_target_exposure_counts_use_expanded_labels() -> None:
    counts = target_exposure_counts(
        [[1, 9, 2], [3, 4, 9], [9, 9]],
        target_item=9,
    )
    assert counts == {
        "target_session_count": 3,
        "target_item_count": 4,
        "target_label_pair_count": 3,
    }


def test_creat_candidate_positions_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        candidate_positions(0, 1.0)
    with pytest.raises(ValueError, match="topk_ratio"):
        candidate_positions(3, 0.0)
