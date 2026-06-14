from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path

import pytest

from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.freqrec_exporter import FreqRecExporter
from attack.models.victim.freqrec_runner import FreqRecRunner
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.tests.freqrec_test_utils import freqrec_config


@pytest.mark.freqrec_subprocess
def test_tiny_fixed_epoch_parent_to_freqrec_subprocess_smoke(tmp_path):
    configured_python = os.environ.get("FREQREC_TEST_PYTHON")
    if not configured_python:
        pytest.skip("FREQREC_TEST_PYTHON is not set")
    python_executable = Path(configured_python)
    if not python_executable.exists():
        pytest.skip(f"FREQREC_TEST_PYTHON does not exist: {python_executable}")
    repo = Path(__file__).resolve().parents[2] / "third_party" / "freqrec"
    submodule_output = repo / "output"
    before_output_files = (
        {path.relative_to(submodule_output) for path in submodule_output.rglob("*") if path.is_file()}
        if submodule_output.exists()
        else set()
    )
    config = freqrec_config(
        tmp_path,
        train_overrides={
            "epochs": 1,
            "batch_size": 4,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "max_seq_length": 4,
        },
        runtime_overrides={
            "python_executable": str(python_executable),
            "repo_root": str(repo),
            "working_dir": str(repo),
            "device": {"use_gpu": False, "gpu_id": "0"},
            "dataloader": {"num_workers": 0},
        },
    )
    dataset = CanonicalDataset(
        train_sub=[[1, 2, 3]],
        valid=[[1, 2, 4]],
        test=[[2, 3, 5]],
        item_map={str(i): i for i in range(1, 7)},
        metadata={"dataset_name": "toy", "item_count": 6, "counts": {"items": 6}},
    )
    train_prefixes = [[1], [1, 2], [2], [2, 3], [3]]
    train_labels = [2, 3, 3, 4, 5]
    export = FreqRecExporter().export_with_train_pairs(
        dataset,
        train_prefixes=train_prefixes,
        train_labels=train_labels,
        output_dir=tmp_path / "export",
        dataset_name="toy",
        max_seq_length=4,
        mode="poisoned",
    )
    runner = FreqRecRunner(config)
    output = tmp_path / "run" / "freqrec_topk_raw.json"
    run_info = runner.run(
        train_path=export.files["train"],
        valid_path=export.files["valid"],
        test_path=export.files["test"],
        metadata_path=export.files["metadata"],
        item_count=export.item_count,
        expected_test_count=export.test_example_count,
        run_dir=tmp_path / "run",
        prediction_output_path=output,
        requested_topk=10,
        epochs=1,
        victim_train_seed=7,
        target_item=None,
    )
    rankings = runner.predict_topk(
        predictions_path=output,
        item_count=6,
        expected_example_count=2,
        requested_topk=10,
        configured_epochs=1,
        seed=7,
    )
    labels = resolve_ground_truth_labels(
        config,
        victim_name="freqrec",
        canonical_dataset=dataset,
        predictions=rankings,
    )
    metrics, available = evaluate_ground_truth_metrics(
        rankings,
        labels=labels,
        metrics=("recall", "mrr", "ndcg"),
        topk=(5,),
    )
    assert available is True
    assert all(
        value is not None and math.isfinite(float(value))
        for value in metrics.values()
    )
    assert output.is_file()
    assert len(rankings) == 2
    assert all(len(row) == 6 and 0 not in row and len(set(row)) == 6 for row in rankings)
    assert run_info["epochs_completed"] == 1
    assert run_info["batch_size"] == 4
    assert run_info["batch_count"] == 1
    assert run_info["final_batch_size"] == 2
    assert Path(run_info["internal_log_path"]).is_file()
    assert Path(run_info["internal_log_path"]).parent == tmp_path / "run" / "freqrec_internal_output"
    after_output_files = (
        {path.relative_to(submodule_output) for path in submodule_output.rglob("*") if path.is_file()}
        if submodule_output.exists()
        else set()
    )
    assert after_output_files == before_output_files
