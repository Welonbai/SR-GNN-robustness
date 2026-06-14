from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack.data.canonical_fingerprints import file_provenance
from attack.data.canonical_dataset import CanonicalDataset
from attack.models.victim.wearec_runner import effective_wearec_config
from attack.pipeline.core.evaluator import save_predictions
from attack.pipeline.core.orchestrator import (
    _load_shared_victim_result,
    _persist_shared_victim_result,
    _validate_wearec_shared_entry,
)
from attack.pipeline.core.victim_execution import VictimExecutionResult
from attack.tests.wearec_test_utils import raw_prediction_payload, wearec_config


def _artifacts(tmp_path):
    local = tmp_path / "local"
    shared = tmp_path / "shared"
    return {
        "run_dir": local,
        "config_snapshot": local / "config.yaml",
        "resolved_config": local / "resolved_config.json",
        "metrics": local / "metrics.json",
        "predictions": local / "predictions.json",
        "train_history": local / "train_history.json",
        "poisoned_train": local / "poisoned_train.txt",
        "shared_dir": shared,
        "shared_predictions": shared / "predictions.json",
        "shared_train_history": shared / "train_history.json",
        "shared_execution_result": shared / "execution_result.json",
        "shared_poisoned_train": shared / "poisoned_train.txt",
        "wearec_raw_predictions": local / "wearec_topk_raw.json",
        "wearec_checkpoint": local / "wearec_checkpoint.pt",
        "wearec_log": local / "wearec_stdout.log",
        "shared_wearec_raw_predictions": shared / "wearec_topk_raw.json",
        "shared_wearec_checkpoint": shared / "wearec_checkpoint.pt",
        "shared_wearec_log": shared / "wearec_stdout.log",
        "shared_artifact_manifest": shared / "artifact_manifest.json",
    }


def _identity(config):
    return {
        "dataset_name": "toy",
        "dataset_variant": "full",
        "ordered_exported_train_jsonl_sha256": "a",
        "ordered_exported_valid_jsonl_sha256": "b",
        "ordered_exported_test_jsonl_sha256": "c",
        "item_vocabulary_fingerprint": "d",
        "fingerprint_semantics": "canonical_exported_rows_sha256_v1",
        "item_vocabulary_fingerprint_semantics": "canonical_dense_item_map_sha256_v1",
        "item_count": 5,
        "victim": "wearec",
        "training_mode": "clean",
        "checkpoint_protocol": "fixed_epoch",
        "effective_config": effective_wearec_config(config, seed=7, requested_topk=5),
        "canonical_exporter_semantics": "canonical_explicit_prefix_label_v1",
        "parent_repository_commit": "parent",
        "parent_tracked_worktree_clean": True,
        "wearec_gitlink_commit": "wearec",
        "wearec_submodule_commit": "wearec",
        "wearec_tracked_worktree_clean": True,
        "wearec_runner_semantics_version": 1,
        "wearec_artifact_contract_version": 1,
    }


def _dataset():
    return CanonicalDataset(
        train_sub=[[1, 2]],
        valid=[[1, 2]],
        test=[[1, 2, 3], [4, 5]],
        item_map={str(value): value for value in range(1, 6)},
        metadata={"item_count": 5, "counts": {"items": 5}},
    )


def _persist(tmp_path):
    config = wearec_config(tmp_path)
    artifacts = _artifacts(tmp_path)
    artifacts["run_dir"].mkdir(parents=True)
    save_predictions(
        artifacts["predictions"],
        topk=5,
        rankings=[row["items"] for row in raw_prediction_payload()["rankings"]],
        victim="wearec",
        target_item=1,
    )
    artifacts["wearec_raw_predictions"].write_text(
        json.dumps(raw_prediction_payload()), encoding="utf-8"
    )
    artifacts["wearec_checkpoint"].write_bytes(b"checkpoint")
    artifacts["wearec_log"].write_text("log", encoding="utf-8")
    identity = _identity(config)
    result = VictimExecutionResult(
        predictions=[row["items"] for row in raw_prediction_payload()["rankings"]],
        predictions_path=artifacts["predictions"],
        extra={
            "wearec": {
                "scientific_identity": identity,
                "prediction_count": 3,
            }
        },
        poisoned_train_path=None,
    )
    _persist_shared_victim_result(
        run_type="clean", victim_result=result, artifacts=artifacts
    )
    return config, artifacts, identity


def _rewrite_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _refresh_manifest_provenance(artifacts, name, path_key):
    manifest = json.loads(artifacts["shared_artifact_manifest"].read_text())
    manifest["retained_artifacts"][name] = file_provenance(artifacts[path_key])
    _rewrite_json(artifacts["shared_artifact_manifest"], manifest)


def test_complete_entry_is_reused_and_partial_entries_are_stale(tmp_path):
    config, artifacts, identity = _persist(tmp_path)
    manifest = json.loads(artifacts["shared_artifact_manifest"].read_text())
    assert manifest["retained_artifacts"]["execution_result"]["size"] > 0
    assert len(manifest["retained_artifacts"]["checkpoint"]["sha256"]) == 64
    reused = _load_shared_victim_result(
        config,
        run_type="clean",
        victim_name="wearec",
        target_item=1,
        run_dir=artifacts["run_dir"],
        artifacts=artifacts,
        predictions_path=artifacts["predictions"],
        canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5),
        wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    )
    assert reused is not None

    artifacts["shared_wearec_raw_predictions"].unlink()
    assert _load_shared_victim_result(
        config, run_type="clean", victim_name="wearec", target_item=1,
        run_dir=artifacts["run_dir"], artifacts=artifacts,
        predictions_path=artifacts["predictions"], canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5), wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    ) is None


def test_missing_checkpoint_or_manifest_reference_is_stale(tmp_path):
    config, artifacts, identity = _persist(tmp_path)
    artifacts["shared_wearec_checkpoint"].unlink()
    assert _load_shared_victim_result(
        config, run_type="clean", victim_name="wearec", target_item=1,
        run_dir=artifacts["run_dir"], artifacts=artifacts,
        predictions_path=artifacts["predictions"], canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5), wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    ) is None

    config, artifacts, identity = _persist(tmp_path / "second")
    manifest = json.loads(artifacts["shared_artifact_manifest"].read_text())
    del manifest["retained_artifacts"]["raw_predictions"]
    artifacts["shared_artifact_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert _load_shared_victim_result(
        config, run_type="clean", victim_name="wearec", target_item=1,
        run_dir=artifacts["run_dir"], artifacts=artifacts,
        predictions_path=artifacts["predictions"], canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5), wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    ) is None


def test_pure_shared_validator_has_no_filesystem_side_effects(tmp_path):
    _, artifacts, identity = _persist(tmp_path)
    before = {
        path.relative_to(tmp_path): (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is not None
    after = {
        path.relative_to(tmp_path): (path.stat().st_mtime_ns, path.read_bytes())
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not artifacts["resolved_config"].exists()


def test_failed_validation_does_not_cleanup_staging_or_materialize_hit(
    tmp_path, monkeypatch
):
    config, artifacts, identity = _persist(tmp_path)
    staging = artifacts["run_dir"] / "export" / "wearec"
    staging.mkdir(parents=True)
    marker = staging / "train.jsonl"
    marker.write_text("staging", encoding="utf-8")
    artifacts["shared_wearec_checkpoint"].unlink()
    writes = []
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator._save_reused_predictions_payload",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    assert _load_shared_victim_result(
        config,
        run_type="clean",
        victim_name="wearec",
        target_item=1,
        run_dir=artifacts["run_dir"],
        artifacts=artifacts,
        predictions_path=artifacts["predictions"],
        canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5),
        wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    ) is None
    assert marker.is_file()
    assert writes == []


def test_actual_cache_hit_writes_reused_metadata_only_after_validation(tmp_path):
    config, artifacts, identity = _persist(tmp_path)
    assert not artifacts["resolved_config"].exists()
    result = _load_shared_victim_result(
        config,
        run_type="clean",
        victim_name="wearec",
        target_item=4,
        run_dir=artifacts["run_dir"],
        artifacts=artifacts,
        predictions_path=artifacts["predictions"],
        canonical_dataset=_dataset(),
        eval_topk=(1, 3, 5),
        wearec_identity=identity,
        wearec_expected_labels=[1, 2, 3],
    )
    assert result is not None
    resolved = json.loads(artifacts["resolved_config"].read_text())
    assert resolved["pipeline_injected"]["reused_from_shared_dir"] == str(
        artifacts["shared_dir"]
    )
    local_predictions = json.loads(artifacts["predictions"].read_text())
    assert local_predictions["target_item"] == 4


@pytest.mark.parametrize(
    "mutation",
    ["ranking", "count", "topk"],
)
def test_unified_prediction_mismatch_is_stale(tmp_path, mutation):
    _, artifacts, identity = _persist(tmp_path)
    payload = json.loads(artifacts["shared_predictions"].read_text())
    if mutation == "ranking":
        payload["rankings"][0][0], payload["rankings"][0][1] = (
            payload["rankings"][0][1],
            payload["rankings"][0][0],
        )
    elif mutation == "count":
        payload["count"] = 2
    else:
        payload["topk"] = 4
    _rewrite_json(artifacts["shared_predictions"], payload)
    _refresh_manifest_provenance(artifacts, "predictions", "shared_predictions")
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is None


@pytest.mark.parametrize("field", ["victim", "effective_config"])
def test_manifest_contract_mismatch_is_stale(tmp_path, field):
    _, artifacts, identity = _persist(tmp_path)
    manifest = json.loads(artifacts["shared_artifact_manifest"].read_text())
    manifest[field] = "freqrec" if field == "victim" else {"epochs": 999}
    _rewrite_json(artifacts["shared_artifact_manifest"], manifest)
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [("dataset_name", "other"), ("training_mode", "poisoned")],
)
def test_raw_dataset_or_training_mode_mismatch_is_stale(tmp_path, field, value):
    _, artifacts, identity = _persist(tmp_path)
    raw = json.loads(artifacts["shared_wearec_raw_predictions"].read_text())
    raw[field] = value
    _rewrite_json(artifacts["shared_wearec_raw_predictions"], raw)
    _refresh_manifest_provenance(
        artifacts, "raw_predictions", "shared_wearec_raw_predictions"
    )
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is None


def test_cached_prediction_count_cannot_override_authoritative_test_count(tmp_path):
    _, artifacts, identity = _persist(tmp_path)
    execution = json.loads(artifacts["shared_execution_result"].read_text())
    execution["extra"]["wearec"]["prediction_count"] = 999
    _rewrite_json(artifacts["shared_execution_result"], execution)
    _refresh_manifest_provenance(
        artifacts, "execution_result", "shared_execution_result"
    )
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is not None


def test_self_consistent_wrong_raw_labels_are_stale(tmp_path):
    _, artifacts, identity = _persist(tmp_path)
    raw = json.loads(artifacts["shared_wearec_raw_predictions"].read_text())
    raw["rankings"][0]["label"] = 4
    raw["rankings"][0]["items"] = [4, 1, 2, 3, 5]
    _rewrite_json(artifacts["shared_wearec_raw_predictions"], raw)
    unified = json.loads(artifacts["shared_predictions"].read_text())
    unified["rankings"][0] = [4, 1, 2, 3, 5]
    _rewrite_json(artifacts["shared_predictions"], unified)
    _refresh_manifest_provenance(
        artifacts, "raw_predictions", "shared_wearec_raw_predictions"
    )
    _refresh_manifest_provenance(artifacts, "predictions", "shared_predictions")
    assert _validate_wearec_shared_entry(
        artifacts=artifacts,
        identity=identity,
        expected_labels=[1, 2, 3],
    ) is None
