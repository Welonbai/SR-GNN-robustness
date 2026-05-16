from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import save_fake_sessions
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.pipeline.runs.run_pts_continuous_init_diagnostic import (
    BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1,
    BehaviorAwareSelectionConfig,
    _behavior_selection_pool_label,
    behavior_statistics,
    main as diagnostic_main,
    select_behavior_stratified_space_filling_candidates,
)
from attack.pts.continuous_executor import (
    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
    CONTINUOUS_ACTION_KEEP_FULL_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
    CONTINUOUS_ACTION_STOP,
)


CONTINUOUS_FIXTURE = (
    REPO_ROOT
    / "attack"
    / "tests"
    / "fixtures"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_srgnn_partial4_target5334.yaml"
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_stratification_classification_and_entropy_stats() -> None:
    config = BehaviorAwareSelectionConfig(
        enabled=True,
        mode=BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1,
    )
    rejected = _fake_behavior_candidate("pool_cand0", [0.96, 0.01, 0.01, 0.01, 0.01])
    extreme = _fake_behavior_candidate("pool_cand1", [0.80, 0.05, 0.05, 0.05, 0.05])
    moderate = _fake_behavior_candidate("pool_cand2", [0.50, 0.20, 0.10, 0.10, 0.10])
    balanced = _fake_behavior_candidate("pool_cand3", [0.20, 0.20, 0.20, 0.20, 0.20])

    assert behavior_statistics(balanced["summary"], balanced["behavior_vector"])[
        "entropy_overall"
    ] == pytest.approx(1.0)
    assert (
        _behavior_selection_pool_label(
            rejected,
            behavior_config=config,
            balanced_pool_keys=set(),
        )
        == "rejected_overcollapsed"
    )
    assert (
        _behavior_selection_pool_label(
            extreme,
            behavior_config=config,
            balanced_pool_keys=set(),
        )
        == "extreme"
    )
    assert (
        _behavior_selection_pool_label(
            moderate,
            behavior_config=config,
            balanced_pool_keys=set(),
        )
        == "moderate"
    )
    assert (
        _behavior_selection_pool_label(
            balanced,
            behavior_config=config,
            balanced_pool_keys={"pool_cand3"},
        )
        == "balanced_candidate"
    )


def test_stratified_selection_composition_when_strata_are_available() -> None:
    pool = [
        *[
            _fake_behavior_candidate(
                f"pool_cand{index}",
                _rotated([0.80, 0.05, 0.05, 0.05, 0.05], index),
            )
            for index in range(5)
        ],
        *[
            _fake_behavior_candidate(
                f"pool_cand{index + 5}",
                _rotated([0.50, 0.20, 0.10, 0.10, 0.10], index),
            )
            for index in range(5)
        ],
        _fake_behavior_candidate("pool_cand10", [0.20, 0.20, 0.20, 0.20, 0.20]),
    ]

    selected = select_behavior_stratified_space_filling_candidates(
        pool,
        select_size=6,
        extreme_count=2,
        moderate_count=3,
        balanced_count=1,
    )

    strata = [str(candidate["selection_stratum"]) for candidate in selected]
    assert strata.count("extreme") == 2
    assert strata.count("moderate") == 3
    assert strata.count("balanced") == 1


def test_stratified_selection_records_fallback_when_a_stratum_is_small() -> None:
    pool = [
        _fake_behavior_candidate("pool_cand0", [0.80, 0.05, 0.05, 0.05, 0.05]),
        _fake_behavior_candidate("pool_cand1", [0.05, 0.80, 0.05, 0.05, 0.05]),
        _fake_behavior_candidate("pool_cand2", [0.20, 0.20, 0.20, 0.20, 0.20]),
        _fake_behavior_candidate("pool_cand3", [0.30, 0.25, 0.15, 0.15, 0.15]),
    ]
    fallbacks: list[dict[str, object]] = []

    selected = select_behavior_stratified_space_filling_candidates(
        pool,
        select_size=4,
        extreme_count=2,
        moderate_count=1,
        balanced_count=1,
        fallback_records=fallbacks,
    )

    assert len(selected) == 4
    assert any(item["reason"] == "insufficient_moderate_pool" for item in fallbacks)
    assert any(str(candidate["selection_reason"]).startswith("fallback") for candidate in selected)


def test_stratified_selection_avoids_overcollapsed_candidates_when_possible() -> None:
    pool = [
        _fake_behavior_candidate("pool_cand0", [1.0, 0.0, 0.0, 0.0, 0.0]),
        _fake_behavior_candidate("pool_cand1", [0.80, 0.05, 0.05, 0.05, 0.05]),
        _fake_behavior_candidate("pool_cand2", [0.05, 0.80, 0.05, 0.05, 0.05]),
        _fake_behavior_candidate("pool_cand3", [0.50, 0.20, 0.10, 0.10, 0.10]),
        _fake_behavior_candidate("pool_cand4", [0.20, 0.20, 0.20, 0.20, 0.20]),
    ]

    selected = select_behavior_stratified_space_filling_candidates(
        pool,
        select_size=4,
        extreme_count=2,
        moderate_count=1,
        balanced_count=1,
    )

    assert "pool_cand0" not in {str(candidate["pool_candidate_key"]) for candidate in selected}


def test_stratified_diagnostic_cli_smoke_and_candidate_key_consistency(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 0.25)
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 0.0,
    )
    monkeypatch.setattr(
        "attack.pts.continuous_executor.generate_poison_model_suffix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic should not materialize generated suffixes")
        ),
    )

    config_path = tmp_path / "continuous.yaml"
    config_text = CONTINUOUS_FIXTURE.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "  root: outputs",
        f"  root: {tmp_path.as_posix()}",
    )
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    save_fake_sessions(
        [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]],
        shared_paths["fake_sessions"],
    )
    output_dir = tmp_path / "diagnostic"

    exit_code = diagnostic_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--behavior-aware-select",
            "--behavior-selection-mode",
            BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1,
            "--behavior-pool-size",
            "32",
            "--behavior-select-size",
            "8",
            "--behavior-extreme-count",
            "3",
            "--behavior-moderate-count",
            "4",
            "--behavior-balanced-count",
            "1",
            "--sample-sessions",
            "5",
        ]
    )

    assert exit_code == 0
    for filename in (
        "behavior_pool_summary.csv",
        "behavior_selected_candidates.json",
        "behavior_selected_distribution_summary.csv",
        "behavior_selected_by_suffix_len_summary.csv",
        "behavior_selection_config.json",
    ):
        assert (output_dir / filename).exists()

    pool_rows = _read_csv(output_dir / "behavior_pool_summary.csv")
    assert {"max_action_ratio_overall", "entropy_overall", "behavior_selection_pool"}.issubset(
        pool_rows[0]
    )

    selection_config = json.loads(
        (output_dir / "behavior_selection_config.json").read_text(encoding="utf-8")
    )
    assert selection_config["mode"] == BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1
    assert selection_config["extreme_count"] == 3
    assert selection_config["moderate_count"] == 4
    assert selection_config["balanced_count"] == 1

    selected = json.loads(
        (output_dir / "behavior_selected_candidates.json").read_text(encoding="utf-8")
    )
    assert len(selected) == 8
    assert {
        "source_pool_candidate_key",
        "selection_stratum",
        "selection_reason",
        "max_action_ratio_overall",
        "entropy_overall",
    }.issubset(selected[0])

    selected_summary_rows = _read_csv(
        output_dir / "behavior_selected_distribution_summary.csv"
    )
    assert {row["candidate_key"] for row in selected_summary_rows} == {
        item["candidate_key"] for item in selected
    }
    assert {
        "selected_rank",
        "source_pool_candidate_key",
        "selection_stratum",
        "selection_reason",
        "max_action_ratio_overall",
        "entropy_overall",
    }.issubset(selected_summary_rows[0])

    selected_by_suffix_rows = _read_csv(
        output_dir / "behavior_selected_by_suffix_len_summary.csv"
    )
    assert {
        "selected_rank",
        "source_pool_candidate_key",
        "selection_stratum",
        "selection_reason",
    }.issubset(selected_by_suffix_rows[0])


def _fake_behavior_candidate(
    key: str,
    first_five: list[float],
) -> dict[str, object]:
    behavior_vector = [float(value) for value in first_five] + [0.0] * 15
    summary = {
        f"{CONTINUOUS_ACTION_KEEP_FULL_SUFFIX}_ratio": float(first_five[0]),
        f"{CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX}_ratio": float(first_five[1]),
        f"{CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX}_ratio": float(first_five[2]),
        f"{CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX}_ratio": float(first_five[3]),
        f"{CONTINUOUS_ACTION_STOP}_ratio": float(first_five[4]),
        "continuous_keep_full_suffix_ratio": float(first_five[0]),
        "continuous_generate_full_suffix_ratio": float(first_five[1]),
        "continuous_partial_keep_suffix_ratio": float(first_five[2]),
        "continuous_partial_generate_suffix_ratio": float(first_five[3]),
        "continuous_stop_ratio": float(first_five[4]),
        "generate_ratio_non_stop": float(first_five[1] + first_five[3]),
    }
    index = int(key.replace("pool_cand", ""))
    stats = behavior_statistics(summary, behavior_vector)
    return {
        "pool_candidate_key": key,
        "pool_candidate_id": index,
        "candidate_info": {
            "sample_origin": "continuous_beta_behavior_covering",
            "prototype_name": key,
        },
        "summary": summary,
        "behavior_vector": behavior_vector,
        "behavior_stats": stats,
        "dominant_action_family": _dominant_action(first_five),
        "dominant_action_ratio": max(first_five),
    }


def _dominant_action(first_five: list[float]) -> str:
    actions = [
        CONTINUOUS_ACTION_KEEP_FULL_SUFFIX,
        CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
        CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
        CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
        CONTINUOUS_ACTION_STOP,
    ]
    return actions[max(range(len(first_five)), key=lambda index: first_five[index])]


def _rotated(values: list[float], offset: int) -> list[float]:
    offset = int(offset) % len(values)
    return values[offset:] + values[:offset]
