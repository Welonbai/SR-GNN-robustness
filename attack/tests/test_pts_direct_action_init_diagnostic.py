from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json, save_fake_sessions
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.pipeline.runs.run_pts_direct_action_init_diagnostic import (
    main as diagnostic_main,
    run_from_config_path,
)
from attack.pts.direct_action_diagnostic import run_direct_action_init_diagnostic
from attack.pts.direct_action_diagnostic import (
    compute_elite_gaussian,
    sample_elite_centered_candidates,
    select_behavior_elites,
)
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_POLICY_LINEAR_LENGTH,
    DIRECT_ACTION_POLICY_MLP_H2,
)


CONTINUOUS_FIXTURE = (
    REPO_ROOT
    / "attack"
    / "tests"
    / "fixtures"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_continuous_mlp_cem_ratio1_target5334.yaml"
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_tmp_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "direct_action.yaml"
    config_text = CONTINUOUS_FIXTURE.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "  root: outputs",
        f"  root: {tmp_path.as_posix()}",
    )
    config_path.write_text(config_text, encoding="utf-8")
    return config_path


def test_direct_action_init_diagnostic_writes_required_artifacts(tmp_path: Path) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        config_path=CONTINUOUS_FIXTURE,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        length_feature_mode="m_over_max_m",
        initial_stds=[0.5, 1.0],
        num_candidates=2,
        sample_sessions=1,
        seed=123,
    )

    for artifact_name in (
        "diagnostic_config",
        "initial_candidates",
        "candidate_overall_summary",
        "candidate_by_suffix_len_summary",
        "candidate_by_suffix_group_summary",
        "uniform_baseline_by_suffix_len",
        "bias_vs_uniform_summary",
        "pairwise_behavior_distance",
        "session_samples",
    ):
        assert Path(result.paths[artifact_name]).exists()
    assert not (tmp_path / "elite_centered_config.json").exists()

    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["policy_variant"] == DIRECT_ACTION_POLICY_MLP_H2
    assert diagnostic_config["target_independent"] is True
    assert diagnostic_config["materialize_generated_suffix"] is False
    assert diagnostic_config["formal_cem_connected"] is False
    assert diagnostic_config["parameter_count"] == 15
    assert diagnostic_config["sample_sessions_per_candidate"] == 1
    assert diagnostic_config["length_feature"] == "m_over_max_m"
    assert diagnostic_config["length_feature_definition"] == "l = m / max_m"
    assert diagnostic_config["max_residual_suffix_len"] >= 1

    candidates = load_json(result.paths["initial_candidates"])
    assert len(candidates) == 4
    assert candidates[0]["parameter_count"] == 15

    overall_rows = _read_csv(result.paths["candidate_overall_summary"])
    assert len(overall_rows) == 4
    assert {row["num_sessions"] for row in overall_rows} == {"3"}

    sample_lines = Path(result.paths["session_samples"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(sample_lines) == 4
    sample = json.loads(sample_lines[0])
    assert sample["length_feature_mode"] == "m_over_max_m"
    assert 0.0 < sample["length_feature"] <= 1.0
    assert {
        "valid_actions",
        "action_scores",
        "action_probabilities",
        "sampled_action",
        "sampled_action_family",
        "sampled_consume_ratio",
        "expected_consume_ratio",
        "expected_family_probabilities",
    }.issubset(sample)
    generate_details = [
        item
        for item in sample["action_probability_details"]
        if item["action_type"] == "generate"
    ]
    assert generate_details
    assert all(item["generated_length"] >= 1 for item in generate_details)


def test_direct_action_pairwise_distances_are_separated_by_initial_std(
    tmp_path: Path,
) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_stds=[0.5, 1.0],
        num_candidates=2,
        sample_sessions=1,
        seed=123,
    )

    pairwise_rows = _read_csv(result.paths["pairwise_behavior_distance"])
    assert len(pairwise_rows) == 2
    assert {row["initial_std"] for row in pairwise_rows} == {"0.5", "1.0"}
    assert all(
        row["candidate_key_a"].split("_")[0] == row["candidate_key_b"].split("_")[0]
        for row in pairwise_rows
    )

    sample_lines = Path(result.paths["session_samples"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(sample_lines) == 4


def test_direct_action_session_samples_are_limited_per_candidate(
    tmp_path: Path,
) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_stds=[0.5],
        num_candidates=2,
        sample_sessions=1,
        seed=123,
    )

    sample_lines = Path(result.paths["session_samples"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(sample_lines) == 2
    sample_keys = [json.loads(line)["candidate_key"] for line in sample_lines]
    assert len(set(sample_keys)) == 2


def test_direct_action_z_score_length_feature_metadata(tmp_path: Path) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        length_feature_mode="z_score_m",
        initial_stds=[0.5],
        num_candidates=1,
        sample_sessions=1,
        seed=123,
    )

    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["length_feature"] == "z_score_m"
    assert diagnostic_config["length_feature_definition"] == "l = (m - mean_m) / std_m"
    assert "mean_residual_suffix_len" in diagnostic_config
    assert "std_residual_suffix_len" in diagnostic_config

    sample = json.loads(
        Path(result.paths["session_samples"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert sample["length_feature_mode"] == "z_score_m"
    assert "mean_residual_suffix_len" in sample
    assert "std_residual_suffix_len" in sample


def test_direct_action_raw_m_length_feature_metadata(tmp_path: Path) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        length_feature_mode="raw_m",
        initial_stds=[0.5],
        num_candidates=1,
        sample_sessions=1,
        seed=123,
    )

    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["length_feature"] == "raw_m"
    assert diagnostic_config["length_feature_definition"] == "l = m"

    sample = json.loads(
        Path(result.paths["session_samples"]).read_text(encoding="utf-8").splitlines()[0]
    )
    assert sample["length_feature_mode"] == "raw_m"
    assert sample["length_feature"] == pytest.approx(float(sample["residual_suffix_len"]))


def test_direct_action_runner_uses_shared_fake_sessions(tmp_path: Path) -> None:
    config_path = _write_tmp_config(tmp_path)
    config = load_config(config_path)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    save_fake_sessions(
        [[1, 2, 3], [4, 5, 6, 7]],
        shared_paths["fake_sessions"],
    )
    output_dir = tmp_path / "diagnostic"

    result = run_from_config_path(
        config_path=config_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        length_feature_mode="log1p",
        initial_stds=[0.5],
        num_candidates=1,
        sample_sessions=5,
        output_dir=output_dir,
        seed=123,
        prefix_seed_scope="target_independent",
    )

    assert result.output_dir == output_dir
    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["fake_sessions_path"] == str(shared_paths["fake_sessions"])
    assert diagnostic_config["fake_sessions_identity"]["type"] == "file_sha1"


def test_direct_action_runner_missing_fake_sessions_fails(tmp_path: Path) -> None:
    config_path = _write_tmp_config(tmp_path)

    with pytest.raises(FileNotFoundError, match="requires existing shared fake sessions"):
        run_from_config_path(
            config_path=config_path,
            policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
            length_feature_mode="log1p",
            initial_stds=[0.5],
            num_candidates=1,
            sample_sessions=5,
            output_dir=tmp_path / "diagnostic",
            seed=123,
            prefix_seed_scope="target_independent",
        )


def test_direct_action_init_diagnostic_cli_smoke(tmp_path: Path) -> None:
    config_path = _write_tmp_config(tmp_path)
    config = load_config(config_path)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    save_fake_sessions(
        [[1, 2, 3], [4, 5, 6, 7]],
        shared_paths["fake_sessions"],
    )
    output_dir = tmp_path / "diagnostic"

    exit_code = diagnostic_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--policy",
            DIRECT_ACTION_POLICY_MLP_H2,
            "--initial-stds",
            "0.5",
            "--num-candidates",
            "1",
            "--sample-sessions",
            "2",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "diagnostic_config.json").exists()
    assert (output_dir / "candidate_overall_summary.csv").exists()


def test_elite_selection_stop_heavy() -> None:
    rows = [
        {"candidate_key": "a", "expected_stop_ratio": 0.2},
        {"candidate_key": "b", "expected_stop_ratio": 0.8},
        {"candidate_key": "c", "expected_stop_ratio": 0.5},
    ]

    selected = select_behavior_elites(
        candidate_keys=["a", "b", "c"],
        elite_select_mode="stop_heavy",
        elite_count=2,
        overall_rows=rows,
    )

    assert [item["candidate_key"] for item in selected] == ["b", "c"]


def test_elite_selection_generate_oriented() -> None:
    rows = [
        {
            "candidate_key": "a",
            "expected_generate_full_ratio": 0.1,
            "expected_partial_generate_ratio": 0.1,
        },
        {
            "candidate_key": "b",
            "expected_generate_full_ratio": 0.2,
            "expected_partial_generate_ratio": 0.6,
        },
        {
            "candidate_key": "c",
            "expected_generate_full_ratio": 0.5,
            "expected_partial_generate_ratio": 0.0,
        },
    ]

    selected = select_behavior_elites(
        candidate_keys=["a", "b", "c"],
        elite_select_mode="generate_oriented",
        elite_count=2,
        overall_rows=rows,
    )

    assert [item["candidate_key"] for item in selected] == ["b", "c"]


def test_elite_gaussian_computation_applies_std_floor() -> None:
    gaussian = compute_elite_gaussian(
        [[0.9, 0.5], [1.1, 1.5]],
        elite_min_std=0.25,
        elite_std_scale=1.0,
    )

    assert gaussian["elite_mean"] == pytest.approx([1.0, 1.0])
    assert gaussian["elite_std"] == pytest.approx([0.1, 0.5])
    assert gaussian["resample_std"] == pytest.approx([0.25, 0.5])


def test_elite_resampled_theta_shape() -> None:
    mlp_candidates = sample_elite_centered_candidates(
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_std=1.0,
        elite_select_mode="stop_heavy",
        elite_resample_count=2,
        elite_mean=[0.0] * 15,
        resample_std=[0.25] * 15,
        seed=123,
    )
    linear_candidates = sample_elite_centered_candidates(
        policy_variant=DIRECT_ACTION_POLICY_LINEAR_LENGTH,
        initial_std=1.0,
        elite_select_mode="stop_heavy",
        elite_resample_count=2,
        elite_mean=[0.0] * 8,
        resample_std=[0.25] * 8,
        seed=123,
    )

    assert len(mlp_candidates[0]["parameter_vector"]) == 15
    assert len(linear_candidates[0]["parameter_vector"]) == 8


def test_direct_action_elite_centered_output_smoke(tmp_path: Path) -> None:
    config = load_config(CONTINUOUS_FIXTURE)
    result = run_direct_action_init_diagnostic(
        config=config,
        fake_sessions=[[1, 2, 3], [4, 5, 6, 7]],
        fake_sessions_path=None,
        output_dir=tmp_path,
        policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
        initial_stds=[0.5],
        num_candidates=4,
        sample_sessions=1,
        seed=123,
        include_elite_centered_diagnostic=True,
        elite_select_mode="stop_heavy",
        elite_count=2,
        elite_resample_count=3,
    )

    for artifact_name in (
        "elite_centered_config",
        "elite_selection_summary",
        "elite_distribution_parameters",
        "elite_resampled_candidates",
        "elite_resampled_overall_summary",
        "elite_resampled_by_suffix_len_summary",
        "elite_resampled_by_suffix_group_summary",
        "elite_resampling_pairwise_distance",
        "elite_resampled_session_samples",
    ):
        assert Path(result.paths[artifact_name]).exists()

    config_payload = load_json(result.paths["elite_centered_config"])
    assert config_payload["note"].startswith(
        "This elite-centered diagnostic uses behavior-selected pseudo-elites"
    )
    candidates = load_json(result.paths["elite_resampled_candidates"])
    assert len(candidates) == 3
    overall_rows = _read_csv(result.paths["elite_resampled_overall_summary"])
    assert len(overall_rows) == 3
    assert {
        "param_l2_to_elite_mean",
        "behavior_l1_to_elite_mean_overall",
        "behavior_l1_to_nearest_elite_by_len",
    }.issubset(overall_rows[0])
    sample_lines = Path(result.paths["elite_resampled_session_samples"]).read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(sample_lines) == 3
