from __future__ import annotations

import json
import csv
import hashlib
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

from attack.common.artifact_io import load_json
from attack.models.victim.mdhg_runner import _load_prediction_payload
from attack.pipeline.core.evaluator import evaluate_prediction_metrics


_EPOCH_PREDICTION_PATTERN = re.compile(r"^epoch_(\d+)_(?:predictions|topk)\.json$")


def summarize_mdhg_epoch_diagnostics(
    victim_run_dir: str | Path,
    *,
    dataset_name: str | None = None,
    run_type: str | None = None,
    target_item: int | None,
    evaluation_topk: Sequence[int],
    targeted_metrics: Sequence[str],
    ground_truth_metrics: Sequence[str],
    test_data_path: str | Path,
    expected_test_count: int,
    n_node: int,
    requested_topk: int,
    per_epoch_prediction_dir: str | Path | None = None,
    epoch_metrics_path: str | Path | None = None,
    json_output_path: str | Path | None = None,
    csv_output_path: str | Path | None = None,
    output_path: str | Path | None = None,
    seed: int | None = None,
    gpu_id: str | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(victim_run_dir)
    prediction_dir = (
        Path(per_epoch_prediction_dir)
        if per_epoch_prediction_dir is not None
        else run_dir / "diagnostics" / "per_epoch_predictions"
    )
    if not prediction_dir.is_dir():
        legacy_prediction_dir = run_dir / "mdhg_per_epoch_predictions"
        if legacy_prediction_dir.is_dir():
            prediction_dir = legacy_prediction_dir
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"MDHG per-epoch prediction directory not found: {prediction_dir}")
    if target_item is None and targeted_metrics:
        raise ValueError("MDHG targeted diagnostic metrics require target_item.")

    labels = _load_mdhg_test_labels(test_data_path)
    if len(labels) != int(expected_test_count):
        raise ValueError(
            "MDHG diagnostic ground-truth label count mismatch: "
            f"{len(labels)} != {expected_test_count}."
        )

    epoch_files = _ordered_epoch_prediction_files(prediction_dir)
    if not epoch_files:
        raise FileNotFoundError(f"No MDHG per-epoch prediction files found in: {prediction_dir}")

    epoch_metrics = _load_epoch_metrics(epoch_metrics_path)
    output_target_item = None if str(run_type or "").lower() == "clean" else int(target_item)
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
        metric_aliases = _metric_aliases(metrics)
        epoch_metric_row = epoch_metrics.get(int(epoch), {})
        rows.append(
            {
                "dataset_name": dataset_name,
                "run_type": run_type,
                "epoch": int(epoch),
                "target_item": output_target_item,
                "diagnostic_target_item": int(target_item),
                "prediction_path": str(prediction_path),
                "predictions_hash": _file_sha256(prediction_path),
                "ranking_hash": _ranking_hash(rankings),
                "prediction_count": len(rankings),
                "topk": int(payload["topk"]),
                "requested_topk": int(payload["requested_topk"]),
                "n_node": int(payload["n_node"]),
                "seed": None if seed is None else int(seed),
                "gpu_id": gpu_id,
                "train_loss": epoch_metric_row.get("train_loss"),
                "train_loss_total": epoch_metric_row.get("train_loss_total"),
                "train_loss_mean_per_batch": epoch_metric_row.get(
                    "train_loss_mean_per_batch"
                ),
                "metrics_available": bool(available),
                "metrics": metrics,
                **metrics,
                **metric_aliases,
            }
        )

    json_destination = (
        Path(json_output_path)
        if json_output_path is not None
        else run_dir / "diagnostics" / "mdhg_epoch_diagnostic.json"
    )
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    with json_destination.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    csv_destination = (
        Path(csv_output_path)
        if csv_output_path is not None
        else run_dir / "diagnostics" / "mdhg_epoch_diagnostic.csv"
    )
    _write_csv(rows, csv_destination)

    if output_path is not None:
        legacy_destination = Path(output_path)
        legacy_destination.parent.mkdir(parents=True, exist_ok=True)
        with legacy_destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def _write_csv(rows: Sequence[dict[str, Any]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixed_fields = [
        "dataset_name",
        "run_type",
        "target_item",
        "diagnostic_target_item",
        "epoch",
        "prediction_path",
        "predictions_hash",
        "ranking_hash",
        "prediction_count",
        "topk",
        "requested_topk",
        "n_node",
        "seed",
        "gpu_id",
        "train_loss",
        "train_loss_total",
        "train_loss_mean_per_batch",
        "metrics_available",
    ]
    metric_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if "@" in key and isinstance(row.get(key), (float, int, type(None)))
        }
    )
    fieldnames = fixed_fields + [key for key in metric_fields if key not in fixed_fields]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _load_epoch_metrics(path: str | Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    metrics_path = Path(path)
    if not metrics_path.is_file():
        return {}
    by_epoch: dict[int, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict) or "epoch" not in payload:
                continue
            by_epoch[int(payload["epoch"])] = payload
    return by_epoch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking_hash(rankings: Sequence[Sequence[int]]) -> str:
    normalized = [[int(item) for item in row] for row in rankings]
    payload = json.dumps(normalized, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for key, value in metrics.items():
        if key.startswith("ground_truth_"):
            aliases["gt_" + key[len("ground_truth_") :]] = value
    return aliases


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
        dataset_name=injected.get("dataset_name"),
        run_type=injected.get("run_type"),
        target_item=int(injected["target_item"]),
        evaluation_topk=[int(k) for k in injected["evaluation_topk"]],
        targeted_metrics=[str(metric) for metric in injected["targeted_metrics"]],
        ground_truth_metrics=[str(metric) for metric in injected["ground_truth_metrics"]],
        test_data_path=Path(injected["mdhg_test_data_path"]),
        expected_test_count=int(injected["expected_test_count"]),
        n_node=int(injected["n_node"]),
        requested_topk=int(injected["export_topk_k"]),
        per_epoch_prediction_dir=injected.get("per_epoch_prediction_dir"),
        epoch_metrics_path=injected.get("epoch_metrics_output_path"),
        json_output_path=injected.get("epoch_diagnostic_json_path"),
        csv_output_path=injected.get("epoch_diagnostic_csv_path"),
        output_path=injected.get("epoch_pipeline_metrics_output_path"),
        seed=injected.get("victim_train_seed"),
        gpu_id=injected.get("gpu_id"),
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
