from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from attack.common.artifact_io import load_json
from attack.data.canonical_fingerprints import file_provenance
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics


def load_wearec_epoch_metrics(
    path: str | Path,
    *,
    configured_epochs: int,
    metric_cutoffs: Sequence[int],
) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"WEARec epoch metrics not found: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"epoch", "train_loss", "valid"}:
                raise ValueError("WEARec diagnostic rows must contain epoch, train_loss, valid.")
            if _exact_int(row["epoch"], "epoch") != line_number:
                raise ValueError("WEARec diagnostic epochs must be contiguous from 1.")
            _finite(row["train_loss"], "train_loss")
            valid = row["valid"]
            if not isinstance(valid, Mapping):
                raise ValueError("WEARec diagnostic valid metrics must be an object.")
            expected = {
                f"{metric}@{cutoff}"
                for cutoff in metric_cutoffs
                for metric in ("hr", "mrr", "ndcg")
            }
            if set(valid) != expected:
                raise ValueError("WEARec diagnostic validation metric keys are inconsistent.")
            for key, value in valid.items():
                number = _finite(value, key)
                if not 0.0 <= number <= 1.0:
                    raise ValueError(f"WEARec metric {key} must be in [0, 1].")
            rows.append(dict(row))
    if len(rows) != configured_epochs:
        raise ValueError("WEARec diagnostic row count must equal configured epochs.")
    return rows


def validate_wearec_metrics(
    rankings: Sequence[Sequence[int]],
    labels: Sequence[int],
    metrics: Mapping[str, Any],
    metric_cutoffs: Sequence[int],
) -> None:
    parent, available = evaluate_ground_truth_metrics(
        rankings,
        labels=labels,
        metrics=("recall", "mrr", "ndcg"),
        topk=metric_cutoffs,
    )
    if not available:
        raise ValueError("WEARec parent metric recomputation was unavailable.")
    for cutoff in metric_cutoffs:
        for native, parent_name in (
            ("hr", "ground_truth_recall"),
            ("mrr", "ground_truth_mrr"),
            ("ndcg", "ground_truth_ndcg"),
        ):
            key = f"{native}@{cutoff}"
            if key not in metrics:
                raise ValueError(f"WEARec raw artifact is missing metric {key}.")
            actual = _finite(metrics[key], key)
            expected = float(parent[f"{parent_name}@{cutoff}"])
            if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"WEARec metric {key} does not match parent recomputation.")


def validate_wearec_per_epoch_predictions(
    prediction_dir: str | Path,
    *,
    configured_epochs: int,
    expected_labels: Sequence[int],
    item_count: int,
    effective_config: Mapping[str, Any],
    dataset_name: str,
    training_mode: str,
    diagnostic_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    directory = Path(prediction_dir)
    expected_paths = [
        directory / f"epoch_{epoch:03d}_validation_topk.json"
        for epoch in range(1, configured_epochs + 1)
    ]
    actual_paths = sorted(directory.glob("epoch_*_validation_topk.json"))
    if actual_paths != expected_paths:
        raise ValueError(
            "WEARec per-epoch validation predictions must contain exactly the "
            "configured one-based epoch files."
        )
    if len(diagnostic_rows) != configured_epochs:
        raise ValueError("WEARec diagnostic rows do not match configured epochs.")
    for epoch, (path, row) in enumerate(
        zip(expected_paths, diagnostic_rows), start=1
    ):
        payload = load_json(path)
        _validate_validation_prediction_payload(
            payload,
            epoch=epoch,
            configured_epochs=configured_epochs,
            expected_labels=expected_labels,
            item_count=item_count,
            effective_config=effective_config,
            dataset_name=dataset_name,
            training_mode=training_mode,
            expected_metrics=row["valid"],
        )
    return expected_paths


def atomic_write_json(payload: Mapping[str, Any], destination: str | Path) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def validate_wearec_diagnostic_bundle(
    result_dir: str | Path,
    *,
    identity: Mapping[str, Any],
    expected_labels: Sequence[int],
    expected_validation_labels: Sequence[int],
    per_epoch_predictions: bool,
) -> dict[str, Any]:
    directory = Path(result_dir)
    summary = load_json(directory / "diagnostic_summary.json")
    manifest = load_json(directory / "artifact_manifest.json")
    effective = identity["effective_config"]
    if (
        not isinstance(summary, Mapping)
        or not isinstance(manifest, Mapping)
        or summary.get("scientific_identity") != identity
        or summary.get("effective_config") != effective
        or manifest.get("victim") != "wearec"
        or manifest.get("scientific_identity") != identity
        or manifest.get("effective_config") != effective
    ):
        raise ValueError("WEARec diagnostic summary or manifest identity is invalid.")
    retained = manifest.get("retained_artifacts")
    if not isinstance(retained, Mapping):
        raise ValueError("WEARec diagnostic manifest retained_artifacts is invalid.")
    required = {
        "predictions": directory / "predictions.json",
        "raw_predictions": directory / "wearec_topk_raw.json",
        "checkpoint": directory / "wearec_checkpoint.pt",
        "log": directory / "wearec_stdout.log",
        "epoch_metrics": directory / "wearec_epoch_metrics.jsonl",
        "diagnostic_summary": directory / "diagnostic_summary.json",
    }
    for name, path in required.items():
        expected = retained.get(name)
        actual = file_provenance(path)
        if (
            not isinstance(expected, Mapping)
            or expected.get("size") != actual["size"]
            or expected.get("sha256") != actual["sha256"]
        ):
            raise ValueError(f"WEARec retained diagnostic artifact {name} is invalid.")
    rows = load_wearec_epoch_metrics(
        required["epoch_metrics"],
        configured_epochs=int(effective["epochs"]),
        metric_cutoffs=effective["metric_cutoffs"],
    )
    if per_epoch_predictions:
        paths = validate_wearec_per_epoch_predictions(
            directory / "wearec_per_epoch_predictions",
            configured_epochs=int(effective["epochs"]),
            expected_labels=expected_validation_labels,
            item_count=int(identity["item_count"]),
            effective_config=effective,
            dataset_name=str(identity["dataset_name"]),
            training_mode=str(identity["training_mode"]),
            diagnostic_rows=rows,
        )
        expected_directory = retained.get("per_epoch_predictions")
        if (
            not isinstance(expected_directory, Mapping)
            or [file_provenance(path) for path in paths]
            != expected_directory.get("files")
        ):
            raise ValueError("WEARec retained per-epoch predictions are invalid.")
    elif "per_epoch_predictions" in retained:
        raise ValueError("WEARec manifest unexpectedly retains epoch predictions.")
    raw = load_json(required["raw_predictions"])
    if len(raw.get("rankings", [])) != len(expected_labels):
        raise ValueError("WEARec diagnostic final prediction count is invalid.")
    unified = load_json(required["predictions"])
    if unified.get("rankings") != [row["items"] for row in raw["rankings"]]:
        raise ValueError("WEARec diagnostic raw and unified rankings differ.")
    return {"summary": summary, "manifest": manifest, "rows": rows}


def _validate_validation_prediction_payload(
    payload: Any,
    *,
    epoch: int,
    configured_epochs: int,
    expected_labels: Sequence[int],
    item_count: int,
    effective_config: Mapping[str, Any],
    dataset_name: str,
    training_mode: str,
    expected_metrics: Mapping[str, Any],
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("WEARec validation prediction artifact must be an object.")
    if any(
        field in payload
        for field in (
            "final_epoch",
            "selected_epoch",
            "best_epoch",
            "best_metric",
            "best_checkpoint",
            "early_stopping",
        )
    ):
        raise ValueError("WEARec intermediate validation artifact claims final/best state.")
    exact = {
        "schema_version": 1,
        "current_epoch": epoch,
        "epochs_requested": configured_epochs,
        "epochs_completed": epoch,
        "item_count": item_count,
        "max_seq_length": effective_config["max_seq_length"],
        "requested_topk": effective_config["requested_topk"],
        "topk": effective_config["requested_topk"],
        "evaluation_topk": max(
            effective_config["requested_topk"],
            max(effective_config["metric_cutoffs"]),
        ),
        "example_count": len(expected_labels),
        "batch_size": effective_config["batch_size"],
        "num_workers": 0,
        "seed": effective_config["seed"],
    }
    for field, value in exact.items():
        if type(payload.get(field)) is not int or payload[field] != value:
            raise ValueError(f"WEARec validation artifact {field} is invalid.")
    for field, value in (
        ("model", "wearec"),
        ("mode", "canonical_sbr"),
        ("split", "valid"),
        ("checkpoint_protocol", "fixed_epoch"),
        ("dataset_name", dataset_name),
        ("training_mode", training_mode),
    ):
        if payload.get(field) != value:
            raise ValueError(f"WEARec validation artifact {field} is invalid.")
    if payload.get("metric_cutoffs") != effective_config["metric_cutoffs"]:
        raise ValueError("WEARec validation artifact metric cutoffs are invalid.")
    rankings = payload.get("rankings")
    if not isinstance(rankings, list) or len(rankings) != len(expected_labels):
        raise ValueError("WEARec validation artifact ranking count is invalid.")
    normalized_rankings: list[list[int]] = []
    for example_id, (row, expected_label) in enumerate(
        zip(rankings, expected_labels)
    ):
        if (
            not isinstance(row, Mapping)
            or type(row.get("example_id")) is not int
            or row["example_id"] != example_id
            or type(row.get("label")) is not int
            or row["label"] != expected_label
        ):
            raise ValueError("WEARec validation artifact label alignment is invalid.")
        items = row.get("items")
        if (
            not isinstance(items, list)
            or len(items) != effective_config["requested_topk"]
            or any(type(item) is not int or not 1 <= item <= item_count for item in items)
            or len(set(items)) != len(items)
        ):
            raise ValueError("WEARec validation artifact ranking is invalid.")
        normalized_rankings.append(list(items))
    metrics = payload.get("valid_metrics")
    if not isinstance(metrics, Mapping) or dict(metrics) != dict(expected_metrics):
        raise ValueError("WEARec validation artifact metrics differ from JSONL.")
    validate_wearec_metrics(
        normalized_rankings,
        expected_labels,
        metrics,
        effective_config["metric_cutoffs"],
    )


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"WEARec {field} must be an integer.")
    return int(value)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"WEARec {field} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"WEARec {field} must be finite.")
    return number


__all__ = [
    "atomic_write_json",
    "load_wearec_epoch_metrics",
    "validate_wearec_diagnostic_bundle",
    "validate_wearec_metrics",
    "validate_wearec_per_epoch_predictions",
]
