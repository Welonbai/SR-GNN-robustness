from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


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
    build_behavior_vector,
    main as diagnostic_main,
    select_behavior_aware_candidates,
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


def test_behavior_vector_groups_action_ratios_by_suffix_bucket() -> None:
    records = [
        {"residual_suffix_length": 1, "action": CONTINUOUS_ACTION_KEEP_FULL_SUFFIX},
        {"residual_suffix_length": 1, "action": CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX},
        {"residual_suffix_length": 2, "action": CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX},
        {"residual_suffix_length": 3, "action": CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX},
        {"residual_suffix_length": 4, "action": CONTINUOUS_ACTION_STOP},
    ]

    vector = build_behavior_vector(records)

    assert len(vector) == 20
    assert vector[:5] == [0.2, 0.2, 0.2, 0.2, 0.2]
    assert vector[5:10] == [0.5, 0.5, 0.0, 0.0, 0.0]
    assert vector[10:15] == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert vector[15:20] == [0.0, 0.0, 0.0, 0.5, 0.5]
    assert build_behavior_vector(records[:2])[10:20] == [0.0] * 10


def test_behavior_aware_selection_prefers_diverse_candidates_over_first_n() -> None:
    pool = [
        _fake_behavior_candidate("pool_cand0", [0, 0, 0, 0, 1], CONTINUOUS_ACTION_STOP),
        _fake_behavior_candidate("pool_cand1", [0, 0, 0, 0, 1], CONTINUOUS_ACTION_STOP),
        _fake_behavior_candidate("pool_cand2", [0, 0, 0, 0, 1], CONTINUOUS_ACTION_STOP),
        _fake_behavior_candidate("pool_cand3", [0, 0, 0, 0, 1], CONTINUOUS_ACTION_STOP),
        _fake_behavior_candidate("pool_cand4", [1, 0, 0, 0, 0], CONTINUOUS_ACTION_KEEP_FULL_SUFFIX),
        _fake_behavior_candidate("pool_cand5", [0, 1, 0, 0, 0], CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX),
        _fake_behavior_candidate("pool_cand6", [0, 0, 1, 0, 0], CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX),
        _fake_behavior_candidate("pool_cand7", [0, 0, 0, 1, 0], CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX),
    ]

    selected = select_behavior_aware_candidates(
        pool,
        select_size=4,
    )

    selected_families = {str(candidate["dominant_action_family"]) for candidate in selected}
    assert len(selected) == 4
    assert selected_families != {CONTINUOUS_ACTION_STOP}
    assert CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX in selected_families
    assert (
        CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX in selected_families
        or CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX in selected_families
    )


def test_behavior_aware_selection_ignores_deprecated_action_quota_arguments() -> None:
    pool = [
        _fake_behavior_candidate("pool_cand0", [0.2, 0.2, 0.2, 0.2, 0.2], "mixed"),
        _fake_behavior_candidate("pool_cand1", [1, 0, 0, 0, 0], CONTINUOUS_ACTION_KEEP_FULL_SUFFIX),
        _fake_behavior_candidate("pool_cand2", [0, 1, 0, 0, 0], CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX),
        _fake_behavior_candidate("pool_cand3", [0, 0, 1, 0, 0], CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX),
        _fake_behavior_candidate("pool_cand4", [0, 0, 0, 1, 0], CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX),
    ]

    baseline = select_behavior_aware_candidates(pool, select_size=3)
    with_deprecated_kwargs = select_behavior_aware_candidates(
        pool,
        select_size=3,
        max_stop_ratio=0.0,
        max_per_dominant_family=1,
        min_partial_candidates=99,
        min_generate_candidates=99,
    )

    assert [candidate["pool_candidate_key"] for candidate in baseline] == [
        candidate["pool_candidate_key"] for candidate in with_deprecated_kwargs
    ]


def test_behavior_aware_diagnostic_cli_writes_selection_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 0.25)
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 0.0,
    )

    def fail_if_materialized(*args, **kwargs):
        raise AssertionError("diagnostic should not materialize generated suffixes")

    monkeypatch.setattr(
        "attack.pts.continuous_executor.generate_poison_model_suffix",
        fail_if_materialized,
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
            "--behavior-pool-size",
            "8",
            "--behavior-select-size",
            "4",
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
    assert len(pool_rows) == 8
    assert sum(1 for row in pool_rows if row["selected"] == "True") == 4

    selection_config = json.loads(
        (output_dir / "behavior_selection_config.json").read_text(encoding="utf-8")
    )
    assert selection_config["selection_method"] == "behavior_space_greedy_maximin"
    assert selection_config["uses_action_specific_quotas"] is False
    assert selection_config["uses_action_specific_caps"] is False
    assert selection_config["candidate_key_policy"] == (
        "preserve_pool_candidate_key_with_selected_rank"
    )

    selected = json.loads(
        (output_dir / "behavior_selected_candidates.json").read_text(encoding="utf-8")
    )
    assert len(selected) == 4
    assert selected[0]["selected_rank"] == 0
    assert selected[0]["candidate_key"].startswith("pool_cand")
    assert selected[0]["pool_candidate_key"] == selected[0]["candidate_key"]
    assert "selected_candidate_key" not in selected[0]
    assert "min_distance_to_previous_selected" in selected[0]

    selected_summary_rows = _read_csv(
        output_dir / "behavior_selected_distribution_summary.csv"
    )
    assert {row["candidate_key"] for row in selected_summary_rows} == {
        item["candidate_key"] for item in selected
    }


def _fake_behavior_candidate(
    key: str,
    first_five: list[float],
    dominant_family: str,
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
    return {
        "pool_candidate_key": key,
        "pool_candidate_id": index,
        "candidate_info": {
            "sample_origin": "continuous_beta_behavior_covering",
            "prototype_name": key,
        },
        "summary": summary,
        "behavior_vector": behavior_vector,
        "dominant_action_family": dominant_family,
        "dominant_action_ratio": max(first_five),
    }
