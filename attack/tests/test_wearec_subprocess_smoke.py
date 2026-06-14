from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import pytest

import attack.pipeline.runs.run_victim_valbest_epoch_diagnostic as diagnostic_module
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.wearec_exporter import WEARecExporter
from attack.models.victim.wearec_diagnostics import load_wearec_epoch_metrics
from attack.models.victim.wearec_runner import WEARecRunner
from attack.pipeline.core.pipeline_utils import build_clean_pairs
from attack.pipeline.core.orchestrator import (
    _cleanup_victim_intermediates_if_enabled,
    _maybe_reuse_or_execute_victim,
)
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.pipeline.runs.run_victim_valbest_epoch_diagnostic import (
    _run_wearec_diagnostic,
)
from attack.tests.wearec_test_utils import wearec_config


pytestmark = pytest.mark.wearec_subprocess


def _python() -> str:
    value = os.environ.get("WEAREC_TEST_PYTHON")
    if not value or not Path(value).is_file():
        pytest.skip("WEAREC_TEST_PYTHON is not configured.")
    return value


def _dataset():
    return CanonicalDataset(
        train_sub=[[1, 2, 3, 4], [2, 3, 5]],
        valid=[[1, 3, 5], [2, 4]],
        test=[[1, 2, 5], [3, 4]],
        item_map={str(value): value for value in range(1, 6)},
        metadata={"item_count": 5, "counts": {"items": 5}, "variant": "full"},
    )


def _run(tmp_path, *, epochs, diagnostic):
    repo = Path(__file__).resolve().parents[2]
    config = wearec_config(
        tmp_path,
        python_executable=_python(),
        train_overrides={"epochs": epochs},
        runtime_overrides={
            "repo_root": str(repo / "third_party" / "wearec"),
            "working_dir": str(repo / "third_party" / "wearec"),
            "diagnostics": {"per_epoch_predictions": diagnostic},
        },
    )
    prefixes, labels = build_clean_pairs(_dataset())
    export = WEARecExporter().export_with_train_pairs(
        _dataset(),
        train_prefixes=prefixes,
        train_labels=labels,
        output_dir=tmp_path / "export" / "wearec",
        dataset_name="toy",
        max_seq_length=6,
        mode="clean",
    )
    runner = WEARecRunner(config)
    raw = tmp_path / "run" / "wearec_topk_raw.json"
    info = runner.run(
        train_path=export.files["train"],
        valid_path=export.files["valid"],
        test_path=export.files["test"],
        metadata_path=export.files["metadata"],
        item_count=5,
        expected_test_count=export.test_example_count,
        run_dir=tmp_path / "run",
        prediction_output_path=raw,
        requested_topk=5,
        epochs=epochs,
        victim_train_seed=7,
        target_item=None,
        per_epoch_diagnostics=diagnostic,
        per_epoch_predictions=diagnostic,
    )
    return info, json.loads(raw.read_text(encoding="utf-8"))


def test_formal_one_epoch_smoke(tmp_path):
    info, payload = _run(tmp_path, epochs=1, diagnostic=False)
    assert payload["epochs_completed"] == payload["final_epoch"] == 1
    assert info["epoch_metrics_output_path"] is None
    assert Path(info["checkpoint_output_path"]).is_file()


def test_diagnostic_two_epoch_smoke(tmp_path):
    info, payload = _run(tmp_path, epochs=2, diagnostic=True)
    rows = load_wearec_epoch_metrics(
        info["epoch_metrics_output_path"],
        configured_epochs=2,
        metric_cutoffs=[1, 3, 5],
    )
    assert [row["epoch"] for row in rows] == [1, 2]
    assert payload["epochs_completed"] == payload["final_epoch"] == 2
    assert "best_epoch" not in payload
    assert "best_metric" not in payload
    assert not (Path(info["checkpoint_output_path"]).parent / "best_validation.pt").exists()


def _parent_artifacts(run_dir):
    return {
        "run_dir": run_dir,
        "config_snapshot": run_dir / "config.yaml",
        "resolved_config": run_dir / "resolved_config.json",
        "metrics": run_dir / "metrics.json",
        "predictions": run_dir / "predictions.json",
        "train_history": run_dir / "train_history.json",
        "poisoned_train": run_dir / "poisoned_train.txt",
        "shared_dir": run_dir / "placeholder_shared",
        "shared_predictions": run_dir / "placeholder_shared" / "predictions.json",
        "shared_train_history": run_dir / "placeholder_shared" / "train_history.json",
        "shared_execution_result": run_dir / "placeholder_shared" / "execution_result.json",
        "shared_poisoned_train": run_dir / "placeholder_shared" / "poisoned_train.txt",
        "wearec_raw_predictions": run_dir / "wearec_topk_raw.json",
        "wearec_checkpoint": run_dir / "wearec_checkpoint.pt",
        "wearec_log": run_dir / "wearec_stdout.log",
    }


def test_parent_formal_cache_miss_then_hit_smoke(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    config = wearec_config(
        tmp_path,
        python_executable=_python(),
        train_overrides={"epochs": 1},
        runtime_overrides={
            "repo_root": str((repo / "third_party" / "wearec").resolve()),
            "working_dir": str((repo / "third_party" / "wearec").resolve()),
        },
    )
    config = replace(
        config,
        artifacts=replace(
            config.artifacts,
            root=str((tmp_path / "artifacts").resolve()),
            cleanup_victim_intermediates=True,
        ),
    )
    prefixes, labels = build_clean_pairs(_dataset())
    provenance = {
        "parent_repository_commit": "test-parent",
        "parent_tracked_worktree_clean": True,
        "wearec_gitlink_commit": "test-wearec",
        "wearec_submodule_commit": "test-wearec",
        "wearec_tracked_worktree_clean": True,
    }
    calls = 0
    real_run = WEARecRunner.run

    def counted_run(self, **kwargs):
        nonlocal calls
        calls += 1
        return real_run(self, **kwargs)

    monkeypatch.setattr(WEARecRunner, "run", counted_run)

    def execute(run_name, target_item):
        run_dir = tmp_path / run_name
        artifacts = _parent_artifacts(run_dir)
        result, reused = _maybe_reuse_or_execute_victim(
            config,
            run_type="clean",
            run_coverage={"cells": {}},
            victim_name="wearec",
            canonical_dataset=_dataset(),
            poisoned_sessions=prefixes,
            poisoned_labels=labels,
            raw_fake_sessions=[],
            run_dir=run_dir,
            poisoned_train_path=run_dir / "unused_poisoned_train.txt",
            target_item=target_item,
            eval_topk=(1, 3, 5),
            srg_nn_export_paths=None,
            predictions_path=artifacts["predictions"],
            artifacts=artifacts,
            wearec_provenance_resolver=lambda *_: dict(provenance),
        )
        assert (run_dir / "export" / "wearec").is_dir()
        _cleanup_victim_intermediates_if_enabled(
            config, victim_name="wearec", artifacts=artifacts
        )
        assert not (run_dir / "export" / "wearec").exists()
        assert artifacts["wearec_raw_predictions"].is_file()
        assert artifacts["wearec_checkpoint"].is_file()
        assert artifacts["wearec_log"].is_file()
        return result, reused, artifacts

    first, first_reused, first_artifacts = execute("first", 1)
    second, second_reused, second_artifacts = execute("second", 4)

    assert first_reused is False
    assert second_reused is True
    assert calls == 1
    assert first.predictions == second.predictions
    first_resolved = json.loads(first_artifacts["resolved_config"].read_text())
    assert "reused_from_shared_dir" not in first_resolved["pipeline_injected"]
    raw = json.loads(second_artifacts["wearec_raw_predictions"].read_text())
    unified = json.loads(second_artifacts["predictions"].read_text())
    expected_labels = resolve_ground_truth_labels(
        config,
        victim_name="wearec",
        canonical_dataset=_dataset(),
        predictions=second.predictions,
    )
    assert [row["label"] for row in raw["rankings"]] == expected_labels
    assert unified["rankings"] == [row["items"] for row in raw["rankings"]]
    assert unified["target_item"] == 4


def test_parent_diagnostic_two_epoch_retention_smoke(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    config = wearec_config(
        tmp_path,
        python_executable=_python(),
        train_overrides={"epochs": 2},
        runtime_overrides={
            "repo_root": str((repo / "third_party" / "wearec").resolve()),
            "working_dir": str((repo / "third_party" / "wearec").resolve()),
            "diagnostics": {"per_epoch_predictions": True},
        },
    )
    config = replace(
        config,
        evaluation=replace(config.evaluation, topk=(1, 3, 5)),
    )
    monkeypatch.setattr(
        "attack.pipeline.runs.run_victim_valbest_epoch_diagnostic.ensure_canonical_dataset",
        lambda _config: _dataset(),
    )
    atomic_writes = []
    real_atomic_write = diagnostic_module.atomic_write_json

    def recording_atomic_write(payload, destination):
        atomic_writes.append(Path(destination).name)
        return real_atomic_write(payload, destination)

    monkeypatch.setattr(diagnostic_module, "atomic_write_json", recording_atomic_write)
    provenance = {
        "parent_repository_commit": "test-parent",
        "parent_tracked_worktree_clean": True,
        "wearec_gitlink_commit": "test-wearec",
        "wearec_submodule_commit": "test-wearec",
        "wearec_tracked_worktree_clean": True,
    }
    out_dir = tmp_path / "diagnostic"
    summary = _run_wearec_diagnostic(
        config,
        out_dir=out_dir,
        effective_epochs=2,
        provenance_resolver=lambda *_: dict(provenance),
    )
    result_dir = Path(summary["final_raw_prediction_path"]).parent
    rows = [
        json.loads(line)
        for line in (result_dir / "wearec_epoch_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert (result_dir / "diagnostic_summary.json").is_file()
    assert (result_dir / "artifact_manifest.json").is_file()
    assert (result_dir / "wearec_checkpoint.pt").is_file()
    assert len(list((result_dir / "wearec_per_epoch_predictions").glob("*.json"))) == 2
    assert not (out_dir / "export" / "wearec").exists()
    assert not (result_dir / "wearec_internal_output").exists()
    assert "best_epoch" not in summary
    assert "best_metric" not in summary
    assert atomic_writes[-2:] == ["diagnostic_summary.json", "artifact_manifest.json"]
