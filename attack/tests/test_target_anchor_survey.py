from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.tools.target_anchor_survey import (
    SUMMARY_COLUMNS,
    _extract_position_ratio,
    _flatten_length_shift_summary,
    _percentile,
    _shannon_entropy,
    _coverage_for_top_k,
    build_summary_row,
    compute_natural_transition_profile,
    parse_insertion_exposure_metadata,
    resolve_exposure_metadata_paths,
    simulate_insertion_exposure,
    simulate_replacement_exposure,
)


def test_synthetic_session_neighbor_stats_counts_predecessor_successor_cooccurrence() -> None:
    sessions = [
        [1, 2, 9, 3],
        [2, 9, 4],
        [2, 5, 9],
        [7, 8],
    ]

    grouped, flat = compute_natural_transition_profile(sessions, 9, top_k=10)

    predecessors = {
        row["item"]: row["count"] for row in grouped["predecessors"]["top_items"]
    }
    successors = {
        row["item"]: row["count"] for row in grouped["successors"]["top_items"]
    }
    cooccur = {
        row["item"]: row["count"] for row in grouped["cooccurrence"]["top_items"]
    }
    assert predecessors[2] == 2
    assert predecessors[5] == 1
    assert successors[3] == 1
    assert successors[4] == 1
    assert cooccur[2] == 3
    assert cooccur[1] == 1
    assert flat["num_unique_predecessors"] == 2


def test_entropy_effective_count_and_topk_coverage() -> None:
    counter = {1: 2, 2: 2}

    assert _shannon_entropy(counter) == pytest.approx(0.69314718056)
    assert _coverage_for_top_k(counter, 1) == pytest.approx(0.5)
    assert _coverage_for_top_k(counter, 5) == pytest.approx(1.0)


def test_quantile_helper_interpolates() -> None:
    assert _percentile([1, 3, 5, 7], 0.25) == pytest.approx(2.5)
    assert _percentile([], 0.5) is None


def test_length_shift_flattening_handles_missing_and_present() -> None:
    metadata = {"length_shift_summary": {"min": 0.0, "max": 1.0, "mean": 0.25}}

    assert _flatten_length_shift_summary("insertion", metadata) == {
        "insertion_length_shift_min": 0.0,
        "insertion_length_shift_max": 1.0,
        "insertion_length_shift_mean": 0.25,
    }
    assert _flatten_length_shift_summary("replacement", {}) == {
        "replacement_length_shift_min": None,
        "replacement_length_shift_max": None,
        "replacement_length_shift_mean": None,
    }


def test_position_ratio_extraction_prefers_group_ratios_and_falls_back_to_raw() -> None:
    grouped = {"replacement_position_group_ratios": {"pos4_5": 0.3}}
    raw = {"replacement_position_ratios": {"1": 0.5, "4": 0.1, "5": 0.2, "6": 0.05}}

    assert _extract_position_ratio(grouped, "pos4_5") == pytest.approx(0.3)
    assert _extract_position_ratio(raw, "pos1") == pytest.approx(0.5)
    assert _extract_position_ratio(raw, "pos4_5") == pytest.approx(0.3)
    assert _extract_position_ratio(raw, "pos6_plus") == pytest.approx(0.05)


def test_internal_insertion_simulation_excludes_tail_append_slot() -> None:
    fake_sessions = [[10, 11], [20, 21, 22], [30, 31, 32, 33]]

    insertion_group, insertion = simulate_insertion_exposure(
        fake_sessions,
        top20_near_top_anchors=[10, 20],
    )
    replacement_group, replacement = simulate_replacement_exposure(
        fake_sessions,
        top20_near_top_anchors=[10, 20],
    )

    assert insertion_group["source"] == "simulated_from_fake_sessions"
    assert insertion["insertion_exposure_source"] == "simulated_from_fake_sessions"
    # Valid Internal Random Insertion-NZ slots are [1], [1, 2], [1, 2, 3].
    # Appending after the last item is not part of this action space.
    assert insertion["insertion_unique_left_right_pair_count"] == 6
    assert insertion["insertion_candidate_unique_left_item_count"] == 6
    assert insertion["insertion_tail_slot_ratio"] == pytest.approx(0.0)
    assert insertion["insertion_every_target_has_left_neighbor"] is True
    assert insertion["insertion_every_target_has_right_neighbor"] is True
    assert replacement_group["source"] == "simulated_from_fake_sessions"
    assert replacement["replacement_exposure_source"] == "simulated_from_fake_sessions"
    assert replacement["replacement_candidate_unique_left_item_count"] is not None
    assert replacement["replacement_tail_fallback_count"] == 1
    assert replacement["replacement_internal_replacement_count"] == 2
    assert replacement["replacement_tail_fallback_ratio"] == pytest.approx(1 / 3)


def test_missing_fake_sessions_reports_missing_sources() -> None:
    _, insertion = simulate_insertion_exposure(None, top20_near_top_anchors=[])
    _, replacement = simulate_replacement_exposure(None, top20_near_top_anchors=[])

    assert insertion["insertion_exposure_source"] == "missing"
    assert replacement["replacement_exposure_source"] == "missing"
    assert insertion["insertion_candidate_unique_left_item_count"] is None
    assert replacement["replacement_candidate_unique_left_item_count"] is None


def test_partial_previews_do_not_drive_entropy_or_overlap() -> None:
    metadata = {
        "target_item": 9,
        "fake_session_count": 100,
        "unique_left_item_count": 2,
        "unique_right_item_count": 2,
        "unique_left_right_pair_count": 2,
        "previews": [
            {"left_item": 1, "right_item": 2},
            {"left_item": 3, "right_item": 4},
        ],
    }

    _, flat = parse_insertion_exposure_metadata(
        metadata,
        top20_near_top_anchors=[1, 3],
    )

    assert flat["insertion_left_entropy"] is None
    assert flat["insertion_candidate_left_entropy"] is None
    assert flat["insertion_left_overlap_count_with_top20_near_top_anchors"] is None
    assert flat["insertion_unique_left_item_count"] == 2
    assert flat["insertion_sampled_unique_left_item_count"] == 2
    assert flat["insertion_candidate_unique_left_item_count"] is None


def test_complete_records_allow_entropy_and_overlap() -> None:
    metadata = {
        "target_item": 9,
        "fake_session_count": 2,
        "previews": [
            {"left_item": 1, "right_item": 2},
            {"left_item": 3, "right_item": 4},
        ],
    }

    _, flat = parse_insertion_exposure_metadata(
        metadata,
        top20_near_top_anchors=[1, 5],
    )

    assert flat["insertion_left_entropy"] == pytest.approx(0.69314718056)
    assert flat["insertion_left_overlap_count_with_top20_near_top_anchors"] == 1
    assert flat["insertion_sampled_unique_left_item_count"] == 2
    assert flat["insertion_candidate_unique_left_item_count"] is None


def test_duplicate_metadata_detection_from_directory_scan() -> None:
    tmp_path = REPO_ROOT / "outputs" / "analysis" / "tmp_target_anchor_survey_metadata_test"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    first = tmp_path / "a" / "internal_random_insertion_metadata.json"
    second = tmp_path / "b" / "internal_random_insertion_metadata.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(json.dumps({"target_item": 99}), encoding="utf-8")
    second.write_text(json.dumps({"target_item": 99}), encoding="utf-8")

    try:
        with pytest.raises(ValueError, match="Multiple internal insertion metadata files"):
            resolve_exposure_metadata_paths(
                explicit_paths=[],
                metadata_dir=tmp_path,
                filename="internal_random_insertion_metadata.json",
                method_name="internal insertion",
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_summary_row_contains_exact_columns() -> None:
    row = build_summary_row({"target_item": 9}, {"insertion_exposure_source": "missing"})

    assert list(row.keys()) == SUMMARY_COLUMNS
    assert set(row) == set(SUMMARY_COLUMNS)
    assert "insertion_candidate_unique_left_item_count" in row
    assert "replacement_candidate_unique_left_item_count" in row


def test_default_summary_surface_is_outcome_free() -> None:
    forbidden = {
        "anchor_score",
        "raw_lowk",
        "win",
        "loss",
        "oracle",
        "best_action",
        "delta",
        "targeted_recall",
        "targeted_mrr",
        "ground_truth_recall",
        "ground_truth_mrr",
        "recommendation",
        "candidate_rows",
        "recommend_next_experiment",
    }

    joined_columns = "\n".join(SUMMARY_COLUMNS).lower()
    for token in forbidden:
        assert token not in joined_columns
