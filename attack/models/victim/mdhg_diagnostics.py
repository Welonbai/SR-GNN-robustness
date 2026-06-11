from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

from attack.common.artifact_io import load_json
from attack.models.victim.mdhg_runner import _load_prediction_payload
from attack.pipeline.core.evaluator import evaluate_prediction_metrics


_EPOCH_PREDICTION_PATTERN = re.compile(r"^epoch_(\d+)_topk\.json$")


def summarize_mdhg_epoch_diagnostics(
    victim_run_dir: str | Path,
    *,
    target_item: int,
    evaluation_topk: Sequence[int],
    targeted_metrics: Sequence[str],
    ground_truth_metrics: Sequence[str],
    test_data_path: str | Path,
    expected_test_count: int,
    n_node: int,
    requested_topk: int,
    per_epoch_prediction_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(victim_run_dir)
    prediction_dir = (
        Path(per_epoch_prediction_dir)
        if per_epoch_prediction_dir is not None
        else run_dir / "mdhg_per_epoch_predictions"
    )
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"MDHG per-epoch prediction directory not found: {prediction_dir}")

    labels = _load_mdhg_test_labels(test_data_path)
    if len(labels) != int(expected_test_count):
        raise ValueError(
            "MDHG diagnostic ground-truth label count mismatch: "
            f"{len(labels)} != {expected_test_count}."
        )

    epoch_files = _ordered_epoch_prediction_files(prediction_dir)
    if not epoch_files:
        raise FileNotFoundError(f"No MDHG per-epoch prediction files found in: {prediction_dir}")

    rows: list[dict[str, Any]] = []
    for epoch, prediction_path in epoch_files:
        payload = _load_prediction_payload(
            prediction_path,
            expected_test_count=expected_test_count,
            n_node=n_node,
            requested_topk=requested_topk,
        )
        payload_epoch = payload.get("epoch")
        if payload_epoch != epoch:
            raise ValueError(
                f"MDHG diagnostic epoch mismatch for {prediction_path}: "
                f"filename epoch={epoch}, JSON epoch={payload_epoch!r}."
            )
        rankings = payload["rankings"]
        metrics, available = evaluate_prediction_metrics(
            rankings,
            target_item=int(target_item),
            ground_truth_labels=labels,
            targeted_metrics=targeted_metrics,
            ground_truth_metrics=ground_truth_metrics,
            topk=evaluation_topk,
        )
        rows.append(
            {
                "epoch": int(epoch),
                "target_item": int(target_item),
                "prediction_count": len(rankings),
                "topk": int(payload["topk"]),
                "requested_topk": int(payload["requested_topk"]),
                "n_node": int(payload["n_node"]),
                "metrics_available": bool(available),
                "metrics": metrics,
            }
        )

    destination = (
        Path(output_path)
        if output_path is not None
        else run_dir / "mdhg_epoch_pipeline_metrics.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def summarize_mdhg_epoch_diagnostics_from_run_dir(
    victim_run_dir: str | Path,
) -> list[dict[str, Any]]:
    run_dir = Path(victim_run_dir)
    resolved_path = run_dir / "resolved_config.json"
    resolved = load_json(resolved_path)
    if not isinstance(resolved, dict):
        raise ValueError(f"Missing or invalid MDHG resolved config: {resolved_path}")
    injected = resolved.get("pipeline_injected")
    if not isinstance(injected, dict):
        raise ValueError(f"MDHG resolved config is missing pipeline_injected: {resolved_path}")

    required = (
        "target_item",
        "evaluation_topk",
        "targeted_metrics",
        "ground_truth_metrics",
        "mdhg_test_data_path",
        "expected_test_count",
        "n_node",
        "export_topk_k",
    )
    missing = [key for key in required if key not in injected]
    if missing:
        raise ValueError(
            "MDHG resolved config is missing diagnostic context: " + ", ".join(missing)
        )
    return summarize_mdhg_epoch_diagnostics(
        run_dir,
        target_item=int(injected["target_item"]),
        evaluation_topk=[int(k) for k in injected["evaluation_topk"]],
        targeted_metrics=[str(metric) for metric in injected["targeted_metrics"]],
        ground_truth_metrics=[str(metric) for metric in injected["ground_truth_metrics"]],
        test_data_path=Path(injected["mdhg_test_data_path"]),
        expected_test_count=int(injected["expected_test_count"]),
        n_node=int(injected["n_node"]),
        requested_topk=int(injected["export_topk_k"]),
        per_epoch_prediction_dir=injected.get("per_epoch_prediction_dir"),
        output_path=injected.get("epoch_pipeline_metrics_output_path"),
    )


def _load_mdhg_test_labels(test_data_path: str | Path) -> list[int]:
    path = Path(test_data_path)
    if not path.is_file():
        raise FileNotFoundError(f"MDHG exported test data not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError(f"MDHG exported test data must be a (sessions, labels) tuple: {path}")
    labels = payload[1]
    if not isinstance(labels, list):
        raise ValueError(f"MDHG exported test labels must be a list: {path}")
    return [int(label) for label in labels]


def _ordered_epoch_prediction_files(prediction_dir: Path) -> list[tuple[int, Path]]:
    by_epoch: dict[int, Path] = {}
    for path in prediction_dir.iterdir():
        if not path.is_file():
            continue
        match = _EPOCH_PREDICTION_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch <= 0:
            raise ValueError(f"MDHG diagnostic epoch must be positive: {path}")
        if epoch in by_epoch:
            raise ValueError(f"Duplicate MDHG diagnostic prediction epoch {epoch}.")
        by_epoch[epoch] = path
    return sorted(by_epoch.items())


__all__ = [
    "summarize_mdhg_epoch_diagnostics",
    "summarize_mdhg_epoch_diagnostics_from_run_dir",
]
