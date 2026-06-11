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
    prediction_dir = run_dir / "mdhg_per_epoch_predictions"
    rankings = [[4, 2, 1], [3, 5, 1]]
    _write_epoch(prediction_dir / "epoch_001_topk.json", epoch=1, rankings=rankings)
    test_path = run_dir / "export" / "mdhg" / "toy" / "test.txt"
    test_path.parent.mkdir(parents=True)
    with test_path.open("wb") as handle:
        pickle.dump(([[1], [2]], [2, 5]), handle)

    rows = summarize_mdhg_epoch_diagnostics(
        run_dir,
        target_item=3,
        evaluation_topk=[1, 2, 3],
        targeted_metrics=["recall", "mrr", "ndcg"],
        ground_truth_metrics=["recall", "mrr", "ndcg"],
        test_data_path=test_path,
        expected_test_count=2,
        n_node=5,
        requested_topk=3,
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
    output_rows = [
        json.loads(line)
        for line in (run_dir / "mdhg_epoch_pipeline_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert output_rows == rows


def test_summarizer_loads_context_from_resolved_config(tmp_path) -> None:
    run_dir = tmp_path / "victim"
    prediction_dir = run_dir / "mdhg_per_epoch_predictions"
    _write_epoch(
        prediction_dir / "epoch_002_topk.json",
        epoch=2,
        rankings=[[1, 2, 3]],
    )
    test_path = run_dir / "export" / "mdhg" / "toy" / "test.txt"
    test_path.parent.mkdir(parents=True)
    with test_path.open("wb") as handle:
        pickle.dump(([[1]], [2]), handle)
    output_path = run_dir / "mdhg_epoch_pipeline_metrics.jsonl"
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "pipeline_injected": {
                    "target_item": 1,
                    "evaluation_topk": [1, 3],
                    "targeted_metrics": ["recall", "mrr", "ndcg"],
                    "ground_truth_metrics": ["recall", "mrr", "ndcg"],
                    "mdhg_test_data_path": str(test_path),
                    "expected_test_count": 1,
                    "n_node": 5,
                    "export_topk_k": 3,
                    "per_epoch_prediction_dir": str(prediction_dir),
                    "epoch_pipeline_metrics_output_path": str(output_path),
                }
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_mdhg_epoch_diagnostics_from_run_dir(run_dir)

    assert [row["epoch"] for row in rows] == [2]
    assert rows[0]["metrics"]["targeted_recall@1"] == 1.0
    assert rows[0]["metrics"]["ground_truth_mrr@3"] == 0.5
