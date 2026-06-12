from __future__ import annotations

from dataclasses import replace
import json

import pytest

from attack.common.paths import run_group_key_payload, victim_prediction_key
from attack.common.config import _normalize_freqrec_train
from attack.data.canonical_dataset import CanonicalDataset
from attack.models.victim.freqrec_diagnostics import (
    load_freqrec_epoch_metrics,
    summarize_freqrec_epoch_diagnostics,
)
from attack.models.victim.freqrec_runner import FreqRecRunner
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.tests.freqrec_test_utils import freqrec_config, prediction_payload


def _epoch_row(epoch, *, train_loss, validation):
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation": validation,
        "train_runtime_seconds": 1.0,
        "validation_runtime_seconds": 0.5,
        "epoch_runtime_seconds": 1.5,
        "train_example_count": 5,
        "train_batch_count": 2,
        "train_final_batch_size": 1,
        "validation_example_count": 2,
        "validation_batch_count": 1,
        "validation_final_batch_size": 2,
        "num_workers": 0,
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "checkpoint_protocol": "fixed_epoch",
        "improved": None,
        "best_epoch": None,
        "best_metric": None,
    }


def test_alignment_has_explicit_hand_written_prefix_label_order(tmp_path):
    dataset = CanonicalDataset(
        train_sub=[],
        valid=[[1, 2, 3]],
        test=[[1, 2, 3], [4, 5]],
        item_map={str(i): i for i in range(1, 6)},
        metadata={"item_count": 5},
    )
    labels = resolve_ground_truth_labels(
        freqrec_config(tmp_path),
        victim_name="freqrec",
        canonical_dataset=dataset,
        predictions=[[1], [2], [3]],
    )
    assert labels == [3, 2, 5]


def test_freqrec_identity_includes_batch_seed_epochs_and_model_but_not_runtime(tmp_path):
    base = freqrec_config(tmp_path)
    base_key = victim_prediction_key(base, "freqrec", run_type="clean")
    for overrides in (
        {"batch_size": 8},
        {"epochs": 3},
        {"hidden_size": 16},
    ):
        changed = freqrec_config(tmp_path, train_overrides=overrides)
        assert victim_prediction_key(changed, "freqrec", run_type="clean") != base_key
    changed_seed = replace(
        base, seeds=replace(base.seeds, victim_train_seed=base.seeds.victim_train_seed + 1)
    )
    assert victim_prediction_key(changed_seed, "freqrec", run_type="clean") != base_key
    runtime_changed = freqrec_config(
        tmp_path,
        runtime_overrides={
            "python_executable": "different-python",
            "device": {"gpu_id": "7"},
            "dataloader": {"num_workers": 9},
        },
    )
    assert victim_prediction_key(runtime_changed, "freqrec", run_type="clean") == base_key


def test_fixed_epoch_excludes_validation_only_identity_and_topk_stays_run_group_level(tmp_path):
    base = freqrec_config(tmp_path)
    changed = freqrec_config(
        tmp_path,
        train_overrides={
            "validation_metric": "mrr@20",
            "metric_cutoffs": [5, 20],
            "patience": 99,
        },
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        changed, "freqrec", run_type="clean"
    )
    changed_eval = replace(
        base, evaluation=replace(base.evaluation, topk=(5, 10))
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        changed_eval, "freqrec", run_type="clean"
    )
    assert run_group_key_payload(base, run_type="clean") != run_group_key_payload(
        changed_eval, run_type="clean"
    )


def test_validation_best_monitor_changes_identity(tmp_path):
    base = freqrec_config(
        tmp_path,
        train_overrides={"checkpoint_protocol": "validation_best"},
    )
    changed = freqrec_config(
        tmp_path,
        train_overrides={
            "checkpoint_protocol": "validation_best",
            "validation_metric": "mrr@20",
        },
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") != victim_prediction_key(
        changed, "freqrec", run_type="clean"
    )


def test_freqrec_config_validation_rejects_ablation_and_invalid_fft_loss():
    from attack.tests.freqrec_test_utils import freqrec_train

    assert _normalize_freqrec_train(
        freqrec_train(), "victims.params.freqrec.train"
    )["fourier_loss"] is True
    with pytest.raises(ValueError, match="fre must be exactly 1.0"):
        _normalize_freqrec_train(
            freqrec_train(fre=0.5), "victims.params.freqrec.train"
        )
    with pytest.raises(ValueError, match="fourier_loss must be true"):
        _normalize_freqrec_train(
            freqrec_train(fourier_loss=False), "victims.params.freqrec.train"
        )
    with pytest.raises(ValueError, match="fft_loss_type"):
        _normalize_freqrec_train(
            freqrec_train(fft_loss_type="cos"), "victims.params.freqrec.train"
        )


def test_epoch_metrics_parser_requires_contiguous_finite_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                _epoch_row(
                    epoch,
                    train_loss=1.0 / epoch,
                    validation={"ndcg@20": 0.1 * epoch},
                )
            )
            for epoch in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    assert [row["epoch"] for row in load_freqrec_epoch_metrics(path)] == [1, 2]
    path.write_text(
        json.dumps(_epoch_row(2, train_loss=float("nan"), validation={})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_freqrec_epoch_metrics(path)


def test_diagnostic_recomputes_parent_validation_metrics_from_per_epoch_artifact(
    tmp_path,
):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            _epoch_row(
                1,
                train_loss=1.0,
                validation={
                    "hr@20": 1.0,
                    "mrr@20": 0.75,
                    "ndcg@20": 0.8154648767857288,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    payload = prediction_payload(
        item_count=20,
        example_count=2,
        requested_topk=20,
        evaluation_topk=20,
        epochs=2,
        split="validation",
        current_epoch=1,
        selected_epoch=1,
        epochs_completed=1,
    )
    payload["rankings"][0]["items"] = list(range(1, 21))
    payload["rankings"][1]["items"] = [2, 1, *range(3, 21)]
    (prediction_dir / "epoch_001_validation_topk.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = summarize_freqrec_epoch_diagnostics(
        runner=FreqRecRunner(freqrec_config(tmp_path)),
        epoch_metrics_path=metrics_path,
        per_epoch_prediction_dir=prediction_dir,
        validation_labels=[1, 1],
        item_count=20,
        requested_topk=20,
        configured_epochs=2,
        seed=7,
        metric_cutoffs=[20],
    )
    assert rows[0]["parent_validation"]["ground_truth_recall@20"] == 1.0
    assert rows[0]["parent_validation"]["ground_truth_mrr@20"] == 0.75
    assert abs(rows[0]["consistency_delta"]["mrr@20"]) < 1e-12


def test_diagnostic_rejects_persisted_depth_below_parent_cutoff(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        summarize_freqrec_epoch_diagnostics(
            runner=FreqRecRunner(freqrec_config(tmp_path)),
            epoch_metrics_path=tmp_path / "missing.jsonl",
            per_epoch_prediction_dir=tmp_path,
            validation_labels=[1],
            item_count=10,
            requested_topk=10,
            configured_epochs=2,
            seed=7,
            metric_cutoffs=[20],
        )
