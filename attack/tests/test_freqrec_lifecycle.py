from __future__ import annotations

import json
from pathlib import Path

from attack.pipeline.core.orchestrator import (
    _load_shared_victim_result,
    _valid_freqrec_shared_result,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.tests.freqrec_test_utils import freqrec_config
from attack.tools.invalidate_victim_cells import invalidate_victim_cells


def _dataset():
    return CanonicalDataset(
        train_sub=[[1, 2]],
        valid=[[1, 2]],
        test=[[1, 2, 3]],
        item_map={str(i): i for i in range(1, 6)},
        metadata={"item_count": 5},
    )


def test_shared_cache_validation_allows_runtime_worker_difference(tmp_path):
    config = freqrec_config(tmp_path)
    predictions = {
        "victim": "freqrec",
        "target_item": 9,
        "topk": 5,
        "count": 2,
        "rankings": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    }
    from attack.pipeline.core.victim_execution import victim_effective_train_seed

    seed = victim_effective_train_seed(
        config, victim_name="freqrec", run_type="clean", target_item=9
    )
    execution = {
        "extra": {
            "freqrec": {
                "checkpoint_protocol": "fixed_epoch",
                "current_epoch": 2,
                "selected_epoch": 2,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "best_epoch": None,
                "best_metric": None,
                "validation_metric": "ndcg@20",
                "requested_topk": 8,
                "topk": 5,
                "evaluation_topk": 5,
                "batch_size": 4,
                "batch_count": 1,
                "final_batch_size": 2,
                "drop_last": False,
                "train_sampler": "seeded_random",
                "evaluation_sampler": "sequential",
                "seed": seed,
                "num_workers": 99,
                "prediction_count": 2,
            },
            "freqrec_export": {"item_count": 5, "test_example_count": 2},
        }
    }
    assert _valid_freqrec_shared_result(
        config,
        run_type="clean",
        predictions_payload=predictions,
        execution_payload=execution,
        canonical_dataset=_dataset(),
        eval_topk=(8,),
    )
    execution["extra"]["freqrec"]["batch_size"] = 8
    assert not _valid_freqrec_shared_result(
        config,
        run_type="clean",
        predictions_payload=predictions,
        execution_payload=execution,
        canonical_dataset=_dataset(),
        eval_topk=(8,),
    )


def test_fixed_epoch_shared_cache_allows_validation_metric_change(tmp_path):
    base = freqrec_config(tmp_path)
    changed = freqrec_config(
        tmp_path, train_overrides={"validation_metric": "mrr@20"}
    )
    from attack.common.paths import victim_prediction_key
    from attack.pipeline.core.victim_execution import victim_effective_train_seed

    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        changed, "freqrec", run_type="clean"
    )
    seed = victim_effective_train_seed(
        changed, victim_name="freqrec", run_type="clean", target_item=9
    )
    predictions = {
        "victim": "freqrec",
        "target_item": 9,
        "topk": 5,
        "count": 2,
        "rankings": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    }
    execution = {
        "extra": {
            "freqrec": {
                "checkpoint_protocol": "fixed_epoch",
                "current_epoch": 2,
                "selected_epoch": 2,
                "epochs_requested": 2,
                "epochs_completed": 2,
                "best_epoch": None,
                "best_metric": None,
                "validation_metric": "ndcg@20",
                "requested_topk": 8,
                "topk": 5,
                "evaluation_topk": 5,
                "batch_size": 4,
                "batch_count": 1,
                "final_batch_size": 2,
                "drop_last": False,
                "train_sampler": "seeded_random",
                "evaluation_sampler": "sequential",
                "seed": seed,
                "num_workers": 0,
                "prediction_count": 2,
            },
            "freqrec_export": {"item_count": 5, "test_example_count": 2},
        }
    }
    assert _valid_freqrec_shared_result(
        changed,
        run_type="clean",
        predictions_payload=predictions,
        execution_payload=execution,
        canonical_dataset=_dataset(),
        eval_topk=(8,),
    )


def test_freqrec_malformed_shared_json_is_cache_miss(tmp_path):
    config = freqrec_config(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    predictions = shared / "predictions.json"
    execution = shared / "execution_result.json"
    predictions.write_text("{bad", encoding="utf-8")
    execution.write_text("{}", encoding="utf-8")
    artifacts = {
        "shared_predictions": predictions,
        "shared_execution_result": execution,
        "shared_train_history": shared / "train_history.json",
        "shared_poisoned_train": shared / "poisoned_train.txt",
        "train_history": tmp_path / "local" / "train_history.json",
        "poisoned_train": tmp_path / "local" / "poisoned_train.txt",
        "resolved_config": tmp_path / "local" / "resolved_config.json",
        "shared_dir": shared,
    }
    assert (
        _load_shared_victim_result(
            config,
            run_type="clean",
            victim_name="freqrec",
            target_item=9,
            run_dir=tmp_path / "local",
            artifacts=artifacts,
            predictions_path=tmp_path / "local" / "predictions.json",
            canonical_dataset=_dataset(),
            eval_topk=(8,),
        )
        is None
    )
    predictions.write_text("[]", encoding="utf-8")
    assert (
        _load_shared_victim_result(
            config,
            run_type="clean",
            victim_name="freqrec",
            target_item=9,
            run_dir=tmp_path / "local",
            artifacts=artifacts,
            predictions_path=tmp_path / "local" / "predictions.json",
            canonical_dataset=_dataset(),
            eval_topk=(8,),
        )
        is None
    )


def test_freqrec_incomplete_execution_metadata_is_cache_miss(tmp_path):
    config = freqrec_config(tmp_path)
    predictions = {
        "victim": "freqrec",
        "target_item": 9,
        "topk": 5,
        "count": 2,
        "rankings": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    }
    assert not _valid_freqrec_shared_result(
        config,
        run_type="clean",
        predictions_payload=predictions,
        execution_payload={"extra": {"freqrec": {}, "freqrec_export": {}}},
        canonical_dataset=_dataset(),
        eval_topk=(8,),
    )


def test_valid_freqrec_shared_cache_is_reused(tmp_path):
    config = freqrec_config(tmp_path)
    from attack.pipeline.core.victim_execution import victim_effective_train_seed

    seed = victim_effective_train_seed(
        config, victim_name="freqrec", run_type="clean", target_item=9
    )
    predictions_payload = {
        "victim": "freqrec",
        "target_item": 9,
        "topk": 5,
        "count": 2,
        "rankings": [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]],
    }
    freqrec_metadata = {
        "checkpoint_protocol": "fixed_epoch",
        "current_epoch": 2,
        "selected_epoch": 2,
        "epochs_requested": 2,
        "epochs_completed": 2,
        "best_epoch": None,
        "best_metric": None,
        "validation_metric": "ndcg@20",
        "requested_topk": 8,
        "topk": 5,
        "evaluation_topk": 5,
        "batch_size": 4,
        "batch_count": 1,
        "final_batch_size": 2,
        "num_workers": 7,
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "seed": seed,
        "prediction_count": 2,
    }
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "predictions.json").write_text(
        json.dumps(predictions_payload), encoding="utf-8"
    )
    (shared / "execution_result.json").write_text(
        json.dumps(
            {
                "extra": {
                    "freqrec": freqrec_metadata,
                    "freqrec_export": {"item_count": 5, "test_example_count": 2},
                }
            }
        ),
        encoding="utf-8",
    )
    local = tmp_path / "local"
    artifacts = {
        "shared_predictions": shared / "predictions.json",
        "shared_execution_result": shared / "execution_result.json",
        "shared_train_history": shared / "train_history.json",
        "shared_poisoned_train": shared / "poisoned_train.txt",
        "train_history": local / "train_history.json",
        "poisoned_train": local / "poisoned_train.txt",
        "resolved_config": local / "resolved_config.json",
        "predictions": local / "predictions.json",
        "shared_dir": shared,
    }
    result = _load_shared_victim_result(
        config,
        run_type="clean",
        victim_name="freqrec",
        target_item=9,
        run_dir=local,
        artifacts=artifacts,
        predictions_path=local / "predictions.json",
        canonical_dataset=_dataset(),
        eval_topk=(8,),
    )
    assert result is not None
    assert result.predictions == predictions_payload["rankings"]


def test_freqrec_invalidation_deletes_only_exact_local_and_shared_cell(tmp_path):
    run_dir = tmp_path / "run"
    local = run_dir / "targets" / "101" / "victims" / "freqrec"
    shared = (
        tmp_path
        / "shared"
        / "victim_predictions"
        / "freqrec"
        / "victim_freqrec_key"
        / "targets"
        / "101"
    )
    other = tmp_path / "shared" / "victim_predictions" / "mdhg" / "keep"
    for path in (local, shared, other):
        path.mkdir(parents=True)
        (path / "marker.txt").write_text("x", encoding="utf-8")
    coverage = {
        "run_group_key": "run_group",
        "target_cohort_key": "cohort",
        "targets_order": [101],
        "created_at": "old",
        "updated_at": "old",
        "victims": {
            "freqrec": {
                "status": "completed",
                "victim_prediction_key": "victim_freqrec_key",
            },
            "mdhg": {"status": "completed"},
        },
        "cells": {
            "101": {
                "freqrec": {"status": "completed"},
                "mdhg": {"status": "completed"},
            }
        },
    }
    (run_dir / "run_coverage.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    manifest = {
        "victims": {
            "101": {
                "freqrec": {
                    "shared": {"shared_dir": str(shared)},
                }
            }
        }
    }
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    invalidate_victim_cells(
        [run_dir],
        victim="freqrec",
        allowed_roots=[tmp_path],
    )
    assert not local.exists()
    assert not shared.exists()
    assert other.exists()
