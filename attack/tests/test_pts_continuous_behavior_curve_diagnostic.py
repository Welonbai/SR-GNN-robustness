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
    BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
    build_behavior_curve_profile,
    main as diagnostic_main,
    select_behavior_curve_two_pool_candidates,
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
    / "diginetica_valbest_attack_pts_construction_continuous_beta_cem_ratio1_srgnn_partial4_target5334.yaml"
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_behavior_curve_profile_uses_q_grid_distributions() -> None:
    records = [
        _record(0.05, CONTINUOUS_ACTION_KEEP_FULL_SUFFIX),
        _record(0.25, CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX),
        _record(0.50, CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX),
        _record(0.75, CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX),
        _record(0.95, CONTINUOUS_ACTION_STOP),
    ]

    profile = build_behavior_curve_profile(
        records,
        q_grid_size=5,
        q_grid_min=0.05,
        q_grid_max=0.95,
        q_kernel_bandwidth=0.08,
    )

    vector = profile["behavior_curve_vector"]
    assert len(vector) == 25
    for offset in range(0, len(vector), 5):
        assert abs(sum(vector[offset : offset + 5]) - 1.0) < 1e-9
    assert 0.0 <= profile["mean_entropy_over_q"] <= 1.0
    assert 0.0 <= profile["max_action_prob_over_q"] <= 1.0
    assert profile["q_variation"] > 0.0


def test_two_pool_selection_counts_and_moderate_uses_soft_extreme_context() -> None:
    soft_pool = [
        _curve_candidate("soft_extreme_pool_cand0", "soft_extreme", [1.0, 0.0, 0.0, 0.0]),
        _curve_candidate("soft_extreme_pool_cand1", "soft_extreme", [0.0, 1.0, 0.0, 0.0]),
    ]
    moderate_pool = [
        _curve_candidate("moderate_pool_cand0", "moderate", [1.0, 0.0, 0.0, 0.0]),
        _curve_candidate("moderate_pool_cand1", "moderate", [0.0, 0.0, 1.0, 0.0]),
        _curve_candidate("moderate_pool_cand2", "moderate", [0.0, 0.0, 0.0, 1.0]),
    ]

    selected, fallbacks = select_behavior_curve_two_pool_candidates(
        soft_extreme_pool=soft_pool,
        moderate_pool=moderate_pool,
        soft_extreme_select_size=1,
        moderate_select_size=1,
    )

    assert not fallbacks
    assert [candidate["selection_stage"] for candidate in selected] == [
        "soft_extreme",
        "moderate",
    ]
    assert selected[1]["pool_candidate_key"] != "moderate_pool_cand0"


def test_two_pool_mode_ignores_action_specific_deprecated_flags(
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
            BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
            "--soft-extreme-pool-size",
            "8",
            "--moderate-pool-size",
            "8",
            "--soft-extreme-select-size",
            "2",
            "--moderate-select-size",
            "3",
            "--q-grid-size",
            "5",
            "--behavior-max-stop-ratio",
            "0",
            "--behavior-min-generate-candidates",
            "99",
            "--behavior-min-partial-candidates",
            "99",
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
        "behavior_curve_metrics.csv",
    ):
        assert (output_dir / filename).exists()

    selected = json.loads(
        (output_dir / "behavior_selected_candidates.json").read_text(encoding="utf-8")
    )
    assert len(selected) == 5
    assert [item["candidate_key"] for item in selected] == [
        f"selected_cand{index}" for index in range(5)
    ]
    assert all(item["source_pool_candidate_key"] != item["candidate_key"] for item in selected)

    selected_rows = _read_csv(output_dir / "behavior_selected_distribution_summary.csv")
    assert {row["candidate_key"] for row in selected_rows} == {
        item["candidate_key"] for item in selected
    }

    metrics_rows = _read_csv(output_dir / "behavior_curve_metrics.csv")
    assert {"soft_extreme", "moderate", "selected"}.issubset(
        {row["source_pool"] for row in metrics_rows}
    )
    assert "behavior_curve_vector_json" in metrics_rows[0]

    selection_config = json.loads(
        (output_dir / "behavior_selection_config.json").read_text(encoding="utf-8")
    )
    assert (
        selection_config["selection_mode"]
        == BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1
    )
    assert selection_config["soft_extreme_select_size"] == 2
    assert selection_config["moderate_select_size"] == 3


def _record(q: float, action: str) -> dict[str, object]:
    return {
        "suffix_length_percentile": float(q),
        "residual_suffix_length": 2,
        "consume_ratio": 0.5,
        "consume_count": 1,
        "continuation_source": "keep",
        "action": action,
    }


def _curve_candidate(
    key: str,
    source_pool: str,
    vector: list[float],
) -> dict[str, object]:
    return {
        "pool_candidate_key": key,
        "pool_candidate_id": int(key.rsplit("cand", 1)[1]),
        "source_pool": source_pool,
        "behavior_curve_profile": {
            "q_grid": [0.25, 0.75],
            "q_kernel_bandwidth": 0.10,
            "behavior_curve_vector": [float(value) for value in vector],
            "mean_entropy_over_q": 0.5,
            "min_entropy_over_q": 0.4,
            "mean_max_action_prob_over_q": 0.6,
            "max_action_prob_over_q": 0.7,
            "q_variation": 0.2,
            "collapse_score": 0.1,
        },
    }
