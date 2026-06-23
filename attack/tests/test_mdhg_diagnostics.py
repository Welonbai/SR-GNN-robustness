from __future__ import annotations

import json
import pickle
from pathlib import Path

from attack.models.victim.mdhg_diagnostics import (
    summarize_mdhg_epoch_diagnostics,
    summarize_mdhg_epoch_diagnostics_from_run_dir,
)
from attack.pipeline.core.evaluator import evaluate_prediction_metrics


def _write_epoch(path: Path, *, epoch: int, rankings: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "topk": 3,
                "requested_topk": 3,
                "n_node": 5,
                "rankings": rankings,
            }
        ),
        encoding="utf-8",
    )


def test_summarizer_uses_official_pipeline_metrics(tmp_path) -> None:
    run_dir = tmp_path / "victim"
    prediction_dir = run_dir / "diagnostics" / "per_epoch_predictions"
    rankings = [[4, 2, 1], [3, 5, 1]]
    _write_epoch(
        prediction_dir / "epoch_001_predictions.json",
        epoch=1,
        rankings=rankings,
    )
    epoch_metrics_path = run_dir / "diagnostics" / "mdhg_epoch_metrics.jsonl"
    epoch_metrics_path.write_text(
        json.dumps({"epoch": 1, "train_loss": 1.25}) + "\n",
        encoding="utf-8",
    )
    test_path = run_dir / "export" / "mdhg" / "toy" / "test.txt"
    test_path.parent.mkdir(parents=True)
    with test_path.open("wb") as handle:
        pickle.dump(([[1], [2]], [2, 5]), handle)

    rows = summarize_mdhg_epoch_diagnostics(
        run_dir,
        dataset_name="toy",
        run_type="clean",
        target_item=3,
        evaluation_topk=[1, 2, 3],
        targeted_metrics=["recall", "mrr", "ndcg"],
        ground_truth_metrics=["recall", "mrr", "ndcg"],
        test_data_path=test_path,
        expected_test_count=2,
        n_node=5,
        requested_topk=3,
        epoch_metrics_path=epoch_metrics_path,
        seed=7,
        gpu_id="0",
    )
    expected, available = evaluate_prediction_metrics(
        rankings,
        target_item=3,
        ground_truth_labels=[2, 5],
        targeted_metrics=["recall", "mrr", "ndcg"],
        ground_truth_metrics=["recall", "mrr", "ndcg"],
        topk=[1, 2, 3],
    )

    assert rows[0]["metrics"] == expected
    assert rows[0]["metrics_available"] is available
    assert rows[0]["dataset_name"] == "toy"
    assert rows[0]["run_type"] == "clean"
    assert rows[0]["target_item"] is None
    assert rows[0]["diagnostic_target_item"] == 3
    assert rows[0]["train_loss"] == 1.25
    assert rows[0]["gt_recall@2"] == expected["ground_truth_recall@2"]
    assert len(rows[0]["ranking_hash"]) == 64
    output_rows = json.loads(
        (run_dir / "diagnostics" / "mdhg_epoch_diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert output_rows == rows
    csv_text = (run_dir / "diagnostics" / "mdhg_epoch_diagnostic.csv").read_text(
        encoding="utf-8"
    )
    assert "gt_recall@2" in csv_text
    assert "ranking_hash" in csv_text


def test_summarizer_loads_context_from_resolved_config(tmp_path) -> None:
    run_dir = tmp_path / "victim"
    prediction_dir = run_dir / "diagnostics" / "per_epoch_predictions"
    _write_epoch(
        prediction_dir / "epoch_002_predictions.json",
        epoch=2,
        rankings=[[1, 2, 3]],
    )
    test_path = run_dir / "export" / "mdhg" / "toy" / "test.txt"
    test_path.parent.mkdir(parents=True)
    with test_path.open("wb") as handle:
        pickle.dump(([[1]], [2]), handle)
    output_path = run_dir / "diagnostics" / "mdhg_epoch_diagnostic.json"
    csv_path = run_dir / "diagnostics" / "mdhg_epoch_diagnostic.csv"
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "pipeline_injected": {
                    "dataset_name": "toy",
                    "run_type": "clean",
                    "target_item": 1,
                    "evaluation_topk": [1, 3],
                    "targeted_metrics": ["recall", "mrr", "ndcg"],
                    "ground_truth_metrics": ["recall", "mrr", "ndcg"],
                    "mdhg_test_data_path": str(test_path),
                    "expected_test_count": 1,
                    "n_node": 5,
                    "export_topk_k": 3,
                    "per_epoch_prediction_dir": str(prediction_dir),
                    "epoch_diagnostic_json_path": str(output_path),
                    "epoch_diagnostic_csv_path": str(csv_path),
                    "victim_train_seed": 9,
                    "gpu_id": "2",
                }
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_mdhg_epoch_diagnostics_from_run_dir(run_dir)

    assert [row["epoch"] for row in rows] == [2]
    assert rows[0]["target_item"] is None
    assert rows[0]["seed"] == 9
    assert rows[0]["gpu_id"] == "2"
    assert rows[0]["metrics"]["targeted_recall@1"] == 1.0
    assert rows[0]["metrics"]["ground_truth_mrr@3"] == 0.5
    assert json.loads(output_path.read_text(encoding="utf-8")) == rows
    assert csv_path.is_file()
