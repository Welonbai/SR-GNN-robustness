from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import json
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import save_json
from attack.common.config import load_config
from attack.pipeline.runs.run_victim_valbest_epoch_diagnostic import (
    EXPECTED_ACTIONS,
    SourcePTSArtifact,
    load_miasrec_epoch_metrics,
    load_tron_epoch_metrics,
    resolve_source_pts_artifact,
    source_identity_payload,
    summarize_epoch_metrics,
    validate_source_metadata,
    _write_summary_csv,
)


CONFIG_PATH = Path(
    "attack/configs/diginetica_valbest_victim_valbest_epoch_diagnostic_target39588.yaml"
)


def _config_for_tmp(tmp_path: Path):
    config = load_config(CONFIG_PATH)
    return replace(
        config,
        artifacts=replace(config.artifacts, root=str(tmp_path / "outputs")),
        experiment=replace(config.experiment, name="pytest_victim_valbest_diag"),
    )


def _write_source_artifact(
    root: Path,
    *,
    target_item: int = 39588,
    rank: int = 1,
    candidate_key: str = "iter1_cand5",
    include_metadata: bool = True,
) -> Path:
    artifact_dir = root / "targets" / str(target_item) / "pts_construction_cem"
    rank_dir = artifact_dir / "top_candidates" / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    save_json([[1, 2, target_item], [3, 4, target_item]], rank_dir / "sessions.json")
    if include_metadata:
        save_json(
            {
                "target_item": target_item,
                "rank": rank,
                "candidate_key": candidate_key,
                "sample_origin": "elite_centered",
                "init_mode": "vertex_stratified_space_filling",
                "surrogate_victim_seed_aligned": True,
                "reward_metrics": {"raw_lowk_mrr_recall_10_20": 0.14601918333981478},
                "policy": {"enabled_actions": list(EXPECTED_ACTIONS)},
            },
            rank_dir / "metadata.json",
        )
        save_json({"enabled_actions": list(EXPECTED_ACTIONS)}, rank_dir / "policy.json")
    save_json(
        {
            "status": "completed",
            "run_type": "pts_construction_grouped_cem",
            "target_item": target_item,
            "shared_pts_cem_cache_key": "pts_cem_shared_pytest",
            "surrogate_victim_seed_aligned": True,
            "best_candidate": {
                "rank": rank,
                "sessions_path": f"top_candidates/rank_{rank}/sessions.json",
                "metadata_path": f"top_candidates/rank_{rank}/metadata.json",
                "policy_path": f"top_candidates/rank_{rank}/policy.json",
                "reward_metrics": {"raw_lowk_mrr_recall_10_20": 0.14601918333981478},
            },
        },
        artifact_dir / "pts_construction_complete.json",
    )
    save_json(
        {
            "candidates": [
                {
                    "rank": rank,
                    "sample_origin": "elite_centered",
                    "init_mode": "vertex_stratified_space_filling",
                    "surrogate_victim_seed_aligned": True,
                    **({"candidate_key": candidate_key} if include_metadata else {}),
                }
            ]
        },
        artifact_dir / "pts_top_candidates.json",
    )
    return artifact_dir


def test_diagnostic_yaml_loads() -> None:
    config = load_config(CONFIG_PATH)

    assert config.targets.explicit_list == (39588,)
    assert set(config.victims.enabled) == {"miasrec", "tron"}
    assert config.victims.params["miasrec"]["train"]["epochs"] == 30
    assert config.victims.params["tron"]["train"]["max_epochs"] == 30


def test_explicit_source_resolution_records_identity(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    run_root = tmp_path / "completed_run" / "run_group_abc"
    artifact_dir = _write_source_artifact(run_root)

    source = resolve_source_pts_artifact(
        config,
        target_item=39588,
        candidate_rank=1,
        source_run=run_root,
        expected_candidate_key="iter1_cand5",
        manual_raw_lowk=0.144961,
    )

    assert source.artifact_dir == artifact_dir.resolve()
    assert source.source_candidate_key == "iter1_cand5"
    assert source.source_pts_cem_cache_key == "pts_cem_shared_pytest"
    assert source.sessions == [[1, 2, 39588], [3, 4, 39588]]
    identity = source_identity_payload(source)
    assert identity["source_candidate_rank"] == 1
    assert identity["source_sessions_sha1"] == source.sessions_sha1
    assert "manual source_final_target_raw_lowk differs" in " ".join(
        identity["source_validation_warnings"]
    )


def test_omitted_source_with_multiple_artifacts_raises(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    experiment_root = (
        Path(config.artifacts.root)
        / config.artifacts.runs_dir
        / config.data.dataset_name
        / config.experiment.name
    )
    _write_source_artifact(experiment_root / "run_group_a")
    _write_source_artifact(experiment_root / "run_group_b")

    with pytest.raises(ValueError, match="Multiple compatible PTS-CEM artifacts"):
        resolve_source_pts_artifact(
            config,
            target_item=39588,
            candidate_rank=1,
            source_run=None,
            expected_candidate_key="iter1_cand5",
            manual_raw_lowk=None,
        )


def test_metadata_contradiction_fails() -> None:
    with pytest.raises(ValueError, match="candidate_key mismatch"):
        validate_source_metadata(
            {
                "target_item": 39588,
                "rank": 1,
                "candidate_key": "wrong",
                "policy": {"enabled_actions": list(EXPECTED_ACTIONS)},
                "init_mode": "vertex_stratified_space_filling",
                "sample_origin": "elite_centered",
                "surrogate_victim_seed_aligned": True,
            },
            target_item=39588,
            candidate_rank=1,
            expected_candidate_key="iter1_cand5",
            manual_raw_lowk=None,
            artifact_raw_lowk=None,
        )


def test_missing_metadata_becomes_warnings(tmp_path: Path) -> None:
    config = _config_for_tmp(tmp_path)
    run_root = tmp_path / "completed_run" / "run_group_abc"
    _write_source_artifact(run_root, include_metadata=False)

    source = resolve_source_pts_artifact(
        config,
        target_item=39588,
        candidate_rank=1,
        source_run=run_root,
        expected_candidate_key="iter1_cand5",
        manual_raw_lowk=None,
    )

    warning_text = " ".join(source.validation_warnings)
    assert "candidate_key" in warning_text
    assert "enabled PTS-CEM actions" in warning_text


def test_miasrec_epoch_metric_parsing_selects_best_mrr(tmp_path: Path) -> None:
    path = tmp_path / "miasrec_epoch_metrics.jsonl"
    rows = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "valid_score": 0.2,
            "valid_result": {"mrr@20": 0.2, "recall@20": 0.4},
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "valid_score": 0.3,
            "valid_result": {"mrr@20": 0.3, "recall@20": 0.35},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    parsed = load_miasrec_epoch_metrics(path)
    summary = summarize_epoch_metrics(
        parsed,
        victim_name="miasrec",
        primary_metric="mrr@20",
        checkpoint_path="best.pth",
        checkpoint_selection_mode="recbole_validation_best",
        max_epochs=2,
    )

    assert summary["best_epoch_by_mrr20"] == 2
    assert summary["best_valid_mrr20"] == pytest.approx(0.3)
    assert summary["last_epoch"] == 2


def test_tron_summary_records_best_checkpoint_flags() -> None:
    rows = [
        {"epoch": 1, "recall@20": 0.2, "mrr@20": 0.1, "valid_loss": 1.0},
        {"epoch": 2, "recall@20": 0.4, "mrr@20": 0.12, "valid_loss": 0.9},
    ]
    summary = summarize_epoch_metrics(
        rows,
        victim_name="tron",
        primary_metric="recall@20",
        checkpoint_path="best.ckpt",
        checkpoint_selection_mode="lightning_model_checkpoint_recall_cutoff_20",
        max_epochs=2,
        extra={
            "formal_export_behavior": "last_model",
            "diagnostic_compared_best_checkpoint": True,
            "used_best_checkpoint_for_formal_export": False,
        },
    )

    assert summary["best_epoch_by_recall20"] == 2
    assert summary["selected_checkpoint_path"] == "best.ckpt"
    assert summary["formal_export_behavior"] == "last_model"
    assert summary["diagnostic_compared_best_checkpoint"] is True
    assert summary["used_best_checkpoint_for_formal_export"] is False


def test_tron_epoch_metric_parsing_excludes_posthoc_validation_rows(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs" / "tron" / "version_0"
    log_dir.mkdir(parents=True)
    metrics_path = log_dir / "metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "epoch,step,train_loss,recall_cutoff_20,mrr_cutoff_20,test_loss",
                "0,1,4.8,,,",
                "0,2,,0.4,0.1,4.2",
                "1,3,4.4,,,",
                "1,4,,0.5,0.2,4.0",
                "2,5,,0.3,0.08,5.0",
                "2,5,,0.45,0.18,4.1",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_tron_epoch_metrics(log_dir, max_epochs=2)

    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[-1]["recall@20"] == pytest.approx(0.5)
    assert rows[-1]["mrr@20"] == pytest.approx(0.2)


def test_summary_csv_contains_source_identity_and_delta(tmp_path: Path) -> None:
    summary = {
        "dataset": "diginetica",
        "target_item": 39588,
        "victim_name": "miasrec",
        "source_pts_cem_run": "run_group_abc",
        "source_candidate_rank": 1,
        "source_candidate_key": "iter1_cand5",
        "source_sessions_sha1": "abc123",
        "max_epochs": 2,
        "primary_metric": "mrr@20",
        "best_epoch": 1,
        "best_metric_value": 0.5,
        "last_epoch": 2,
        "last_epoch_metric_value": 0.4,
        "best_vs_last_delta": 0.1,
        "selected_checkpoint_path": "best.pth",
        "checkpoint_selection_mode": "recbole_validation_best",
    }
    path = tmp_path / "summary.csv"

    _write_summary_csv([summary], path)

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["best_epoch"] == "1"
    assert rows[0]["last_epoch"] == "2"
    assert rows[0]["best_vs_last_delta"] == "0.1"
    assert rows[0]["selected_checkpoint_path"] == "best.pth"
    assert rows[0]["source_candidate_key"] == "iter1_cand5"
