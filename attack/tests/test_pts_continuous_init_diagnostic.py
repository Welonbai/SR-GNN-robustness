from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json, save_fake_sessions
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.pipeline.runs.run_pts_continuous_init_diagnostic import (
    CANDIDATE_DISTRIBUTION_COLUMNS,
    _candidate_summary_row,
    main as diagnostic_main,
    run_continuous_init_diagnostic,
)
from attack.pts.continuous_executor import (
    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
    CONTINUOUS_ACTION_STOP,
)
from attack.pts.continuous_policy import ContinuousBetaPolicy


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


def test_continuous_init_diagnostic_writes_summary_files(
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
    config = load_config(CONTINUOUS_FIXTURE)

    result = run_continuous_init_diagnostic(
        config=config,
        config_path=CONTINUOUS_FIXTURE,
        output_dir=tmp_path,
        max_candidates=4,
        sample_sessions=5,
        include_rounding_variants=True,
        template_sessions=[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]],
        target_item=5334,
    )

    for artifact_name in (
        "diagnostic_config",
        "initial_candidates",
        "candidate_distribution_summary",
        "candidate_by_suffix_len_summary",
        "overall_suffix_context_summary",
        "rounding_variant_summary",
        "session_samples",
    ):
        assert Path(result.paths[artifact_name]).exists()

    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["rounding_mode"] == "half_up"
    assert diagnostic_config["materialize_generated_suffix"] is False
    assert diagnostic_config["init_materialize_generated_suffix"] is False
    assert diagnostic_config["smoothing_epsilon"] == 0.0
    assert diagnostic_config["consume_smoothing"] == "beta_uniform_mixture"

    candidates = load_json(result.paths["initial_candidates"])
    assert len(candidates) == 4
    assert candidates[0]["candidate_key"] == "iter0_cand0"
    assert candidates[0]["sample_origin"] == "continuous_mlp_two_pool_behavior_curve"
    assert candidates[0]["sample_metadata"]["pool_candidate_key"]

    candidate_rows = _read_csv(result.paths["candidate_distribution_summary"])
    assert len(candidate_rows) == 4
    assert set(CANDIDATE_DISTRIBUTION_COLUMNS).issubset(candidate_rows[0])
    assert "continuous_stop_ratio" in candidate_rows[0]
    assert candidate_rows[0]["smoothing_epsilon"] == "0.0"

    by_suffix_rows = _read_csv(result.paths["candidate_by_suffix_len_summary"])
    assert by_suffix_rows
    assert "residual_suffix_len" in by_suffix_rows[0]

    overall_rows = _read_csv(result.paths["overall_suffix_context_summary"])
    assert overall_rows
    assert {"residual_suffix_len", "num_sessions", "ratio", "q_suffix"}.issubset(
        overall_rows[0]
    )

    rounding_rows = _read_csv(result.paths["rounding_variant_summary"])
    assert {row["rounding_mode"] for row in rounding_rows} == {
        "floor",
        "half_up",
        "ceil",
    }

    sample_lines = Path(result.paths["session_samples"]).read_text(encoding="utf-8").splitlines()
    assert 0 < len(sample_lines) <= 5
    sample = json.loads(sample_lines[0])
    assert {
        "candidate_key",
        "fake_session_index",
        "rho",
        "consume_count",
        "remaining_length",
        "action",
    }.issubset(sample)


def test_continuous_init_diagnostic_supports_tiny_mlp_parameterization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 0.25)
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 0.0,
    )

    config = load_config(CONTINUOUS_FIXTURE)
    pts_config = config.attack.pts_construction
    assert pts_config is not None
    assert pts_config.continuous_policy.parameterization == "suffix_length_mlp"
    assert pts_config.continuous_policy.hidden_size == 2

    result = run_continuous_init_diagnostic(
        config=config,
        config_path=CONTINUOUS_FIXTURE,
        output_dir=tmp_path,
        max_candidates=3,
        sample_sessions=3,
        template_sessions=[[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]],
        target_item=5334,
    )

    diagnostic_config = load_json(result.paths["diagnostic_config"])
    assert diagnostic_config["continuous_policy"]["parameterization"] == "suffix_length_mlp"
    assert diagnostic_config["continuous_policy"]["hidden_size"] == 2

    candidates = load_json(result.paths["initial_candidates"])
    assert len(candidates) == 3
    assert candidates[0]["policy"]["parameterization"] == "tiny_mlp_log_beta_h2"
    assert len(candidates[0]["parameter_vector"]) == 13
    assert candidates[0]["sample_metadata"]["pool_candidate_key"]

    candidate_rows = _read_csv(result.paths["candidate_distribution_summary"])
    assert candidate_rows
    assert candidate_rows[0]["h0_w"] != ""
    assert candidate_rows[0]["a_h0"] != ""
    assert candidate_rows[0]["a1"] == ""


def test_diagnostic_summary_generate_ratio_non_stop_excludes_stop() -> None:
    policy = ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0])
    candidate_info = {
        "candidate_key": "iter0_cand0",
        "candidate_id": 0,
        "sample_origin": "continuous_beta_behavior_covering",
        "prototype_name": "test",
        "sample_metadata": {},
        "policy": policy,
        "parameter_vector": policy.to_vector(),
    }
    records = [
        {
            "residual_suffix_length": 2,
            "suffix_length_percentile": 0.5,
            "consume_ratio": 1.0,
            "consume_count": 2,
            "continuation_source": "stop",
            "action": CONTINUOUS_ACTION_STOP,
        },
        {
            "residual_suffix_length": 2,
            "suffix_length_percentile": 0.5,
            "consume_ratio": 0.0,
            "consume_count": 0,
            "continuation_source": "generate",
            "action": CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
        },
    ]

    row = _candidate_summary_row(candidate_info, records)

    assert row["stop_ratio"] == 0.5
    assert row["non_stop_ratio"] == 0.5
    assert row["generate_ratio_non_stop"] == 1.0
    assert row["continuous_stop_ratio"] == 0.5


def test_continuous_init_diagnostic_cli_smoke(tmp_path: Path) -> None:
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
            "--max-candidates",
            "2",
            "--sample-sessions",
            "5",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "diagnostic_config.json").exists()
    assert (output_dir / "candidate_distribution_summary.csv").exists()
    assert (output_dir / "candidate_by_suffix_len_summary.csv").exists()
