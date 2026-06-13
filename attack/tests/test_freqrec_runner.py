from __future__ import annotations

import json
from pathlib import Path

import pytest

from attack.models.victim.freqrec_runner import FreqRecRunner, load_freqrec_prediction_payload
from attack.tests.freqrec_test_utils import freqrec_config, prediction_payload


def test_gpu_command_uses_same_physical_id_in_arg_and_environment(tmp_path):
    config = freqrec_config(
        tmp_path,
        runtime_overrides={"device": {"use_gpu": True, "gpu_id": "3"}},
    )
    runner = FreqRecRunner(config)
    cmd, env = runner.build_command(
        train_path=tmp_path / "train.jsonl",
        valid_path=tmp_path / "valid.jsonl",
        test_path=tmp_path / "test.jsonl",
        metadata_path=tmp_path / "metadata.json",
        prediction_output_path=tmp_path / "predictions.json",
        checkpoint_output_path=None,
        epoch_metrics_output_path=None,
        per_epoch_prediction_dir=None,
        internal_output_dir=tmp_path / "run" / "freqrec_internal_output",
        train_name="freqrec_canonical",
        requested_topk=20,
        epochs=2,
        seed=7,
    )
    assert cmd[cmd.index("--gpu_id") + 1] == "3"
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert "--no_cuda" not in cmd
    assert "--fourier_loss" not in cmd
    assert "--do_eval" not in cmd
    assert cmd[cmd.index("--output_dir") + 1] == str(
        (tmp_path / "run" / "freqrec_internal_output").resolve()
    )
    assert cmd[cmd.index("--train_name") + 1] == "freqrec_canonical"


def test_cpu_command_uses_no_cuda_and_does_not_override_visibility(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    runner = FreqRecRunner(freqrec_config(tmp_path))
    cmd, env = runner.build_command(
        train_path=tmp_path / "train.jsonl",
        valid_path=tmp_path / "valid.jsonl",
        test_path=tmp_path / "test.jsonl",
        metadata_path=tmp_path / "metadata.json",
        prediction_output_path=tmp_path / "predictions.json",
        checkpoint_output_path=None,
        epoch_metrics_output_path=None,
        per_epoch_prediction_dir=None,
        internal_output_dir=tmp_path / "run" / "freqrec_internal_output",
        train_name="freqrec_canonical",
        requested_topk=20,
        epochs=2,
        seed=7,
    )
    assert "--no_cuda" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize("gpu_id", ["", "0,1", "3,4", "gpu"])
def test_runner_rejects_non_single_gpu_ids(tmp_path, gpu_id):
    config = freqrec_config(
        tmp_path,
        runtime_overrides={"device": {"use_gpu": True, "gpu_id": gpu_id}},
    )
    with pytest.raises(ValueError, match="one non-negative physical GPU"):
        FreqRecRunner(config).build_command(
            train_path=tmp_path / "a",
            valid_path=tmp_path / "b",
            test_path=tmp_path / "c",
            metadata_path=tmp_path / "d",
            prediction_output_path=tmp_path / "e",
            checkpoint_output_path=None,
            epoch_metrics_output_path=None,
            per_epoch_prediction_dir=None,
            internal_output_dir=tmp_path / "run" / "freqrec_internal_output",
            train_name="freqrec_canonical",
            requested_topk=20,
            epochs=1,
            seed=1,
        )


def test_parser_reconstructs_evaluation_depth_and_ignores_worker_mismatch(tmp_path):
    path = tmp_path / "predictions.json"
    payload = prediction_payload(num_workers=9)
    path.write_text(json.dumps(payload), encoding="utf-8")
    parsed = load_freqrec_prediction_payload(
        path,
        split="test",
        item_count=5,
        expected_example_count=3,
        requested_topk=8,
        configured_epochs=2,
        checkpoint_protocol="fixed_epoch",
        validation_metric="ndcg@20",
        metric_cutoffs=[20],
        batch_size=4,
        seed=7,
    )
    assert parsed["num_workers"] == 9
    assert "metric_cutoffs" not in parsed


def test_parser_requires_exact_validation_best_epoch_semantics(tmp_path):
    path = tmp_path / "predictions.json"
    payload = prediction_payload(
        protocol="validation_best",
        selected_epoch=1,
        best_metric=0.25,
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    load_freqrec_prediction_payload(
        path,
        split="test",
        item_count=5,
        expected_example_count=3,
        requested_topk=8,
        configured_epochs=2,
        checkpoint_protocol="validation_best",
        validation_metric="ndcg@20",
        metric_cutoffs=[20],
        batch_size=4,
        seed=7,
    )
    payload["epochs_completed"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected value"):
        load_freqrec_prediction_payload(
            path,
            split="test",
            item_count=5,
            expected_example_count=3,
            requested_topk=8,
            configured_epochs=2,
            checkpoint_protocol="validation_best",
            validation_metric="ndcg@20",
            metric_cutoffs=[20],
            batch_size=4,
            seed=7,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(topk=4),
        lambda p: p.update(evaluation_topk=4),
        lambda p: p.update(batch_count=2),
        lambda p: p.update(final_batch_size=2),
        lambda p: p.update(current_epoch=True),
        lambda p: p["rankings"][0]["items"].__setitem__(0, 0),
        lambda p: p["rankings"][0].update(items=[1, 1, 2, 3, 4]),
    ],
)
def test_parser_rejects_result_affecting_or_ranking_mismatches(tmp_path, mutation):
    payload = prediction_payload()
    mutation(payload)
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_freqrec_prediction_payload(
            path,
            split="test",
            item_count=5,
            expected_example_count=3,
            requested_topk=8,
            configured_epochs=2,
            checkpoint_protocol="fixed_epoch",
            validation_metric="ndcg@20",
            metric_cutoffs=[20],
            batch_size=4,
            seed=7,
        )
