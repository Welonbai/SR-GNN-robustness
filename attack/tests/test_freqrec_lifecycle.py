from __future__ import annotations

import json
from pathlib import Path

from attack.pipeline.core.orchestrator import _valid_freqrec_shared_result
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
                "epochs_requested": 2,
                "epochs_completed": 2,
                "requested_topk": 8,
                "topk": 5,
                "batch_size": 4,
                "seed": seed,
                "num_workers": 99,
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
