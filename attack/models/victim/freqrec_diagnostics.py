from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.models.victim.freqrec_runner import FreqRecRunner
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics


def summarize_freqrec_epoch_diagnostics(
    *,
    runner: FreqRecRunner,
    epoch_metrics_path: Path,
    per_epoch_prediction_dir: Path,
    validation_labels: Sequence[int],
    item_count: int,
    requested_topk: int,
    configured_epochs: int,
    seed: int,
    metric_cutoffs: Sequence[int],
) -> list[dict[str, Any]]:
    cutoffs = sorted(set(int(value) for value in metric_cutoffs))
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError("FreqRec diagnostic metric cutoffs must be positive.")
    required_depth = max(20, max(cutoffs))
    if requested_topk < required_depth:
        raise ValueError(
            "FreqRec diagnostic requested_topk must cover all parent metric cutoffs "
            f"and be at least 20; got {requested_topk}, need {required_depth}."
        )
    rows = load_freqrec_epoch_metrics(epoch_metrics_path)
    summaries: list[dict[str, Any]] = []
    for expected_epoch, row in enumerate(rows, start=1):
        prediction_path = (
            Path(per_epoch_prediction_dir)
            / f"epoch_{expected_epoch:03d}_validation_topk.json"
        )
        payload = runner.load_prediction_payload(
            prediction_path,
            split="validation",
            item_count=item_count,
            expected_example_count=len(validation_labels),
            requested_topk=requested_topk,
            configured_epochs=configured_epochs,
            seed=seed,
            expected_current_epoch=expected_epoch,
        )
        rankings = [ranking["items"] for ranking in payload["rankings"]]
        parent_metrics, available = evaluate_ground_truth_metrics(
            rankings,
            labels=validation_labels,
            metrics=("recall", "mrr", "ndcg"),
            topk=cutoffs,
        )
        if not available:
            raise ValueError("FreqRec parent validation metrics were unavailable.")
        internal = row["validation"]
        deltas: dict[str, float] = {}
        for cutoff in cutoffs:
            mappings = (
                (f"hr@{cutoff}", f"ground_truth_recall@{cutoff}"),
                (f"mrr@{cutoff}", f"ground_truth_mrr@{cutoff}"),
                (f"ndcg@{cutoff}", f"ground_truth_ndcg@{cutoff}"),
            )
            for internal_key, parent_key in mappings:
                if internal_key in internal and parent_key in parent_metrics:
                    deltas[internal_key] = float(internal[internal_key]) - float(
                        parent_metrics[parent_key]
                    )
        summaries.append(
            {
                "epoch": expected_epoch,
                "internal_validation": dict(internal),
                "parent_validation": parent_metrics,
                "consistency_delta": deltas,
                "prediction_path": str(prediction_path),
            }
        )
    return summaries


def load_freqrec_epoch_metrics(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FreqRec epoch metrics not found: {path}")
    rows: list[dict[str, Any]] = []
    previous_best_epoch: int | None = None
    previous_best_metric: float | None = None
    previous_protocol: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid FreqRec epoch metrics JSONL line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"FreqRec epoch metrics line {line_number} must be an object."
                )
            epoch = _exact_int(row.get("epoch"), f"line {line_number}.epoch")
            if epoch != line_number:
                raise ValueError("FreqRec epoch metrics must be contiguous from epoch 1.")
            _finite(row.get("train_loss"), f"epoch {epoch}.train_loss")
            required = {
                "train_runtime_seconds",
                "validation_runtime_seconds",
                "epoch_runtime_seconds",
                "batch_size",
                "train_example_count",
                "train_batch_count",
                "train_final_batch_size",
                "validation_example_count",
                "validation_batch_count",
                "validation_final_batch_size",
                "num_workers",
                "drop_last",
                "train_sampler",
                "evaluation_sampler",
                "checkpoint_protocol",
                "improved",
                "best_epoch",
                "best_metric",
            }
            missing = sorted(required - row.keys())
            if missing:
                raise ValueError(
                    f"FreqRec epoch {epoch} metrics missing fields: "
                    + ", ".join(missing)
                )
            for key in (
                "train_runtime_seconds",
                "validation_runtime_seconds",
                "epoch_runtime_seconds",
            ):
                if _finite(row[key], f"epoch {epoch}.{key}") < 0:
                    raise ValueError(f"FreqRec epoch {epoch}.{key} must be non-negative.")
            for key in (
                "train_example_count",
                "train_batch_count",
                "train_final_batch_size",
                "validation_example_count",
                "validation_batch_count",
                "num_workers",
            ):
                if _exact_int(row[key], f"epoch {epoch}.{key}") < 0:
                    raise ValueError(f"FreqRec epoch {epoch}.{key} must be non-negative.")
            batch_size = _exact_int(row["batch_size"], f"epoch {epoch}.batch_size")
            if batch_size <= 0:
                raise ValueError(f"FreqRec epoch {epoch}.batch_size must be positive.")
            train_count = int(row["train_example_count"])
            train_batches = int(row["train_batch_count"])
            train_final = int(row["train_final_batch_size"])
            expected_train_batches = math.ceil(train_count / batch_size)
            expected_train_final = train_count - batch_size * (
                expected_train_batches - 1
            )
            if (
                train_count <= 0
                or train_batches != expected_train_batches
                or train_final != expected_train_final
            ):
                raise ValueError(
                    f"FreqRec epoch {epoch} train batch metadata is inconsistent."
                )
            validation_final = row["validation_final_batch_size"]
            validation_count = int(row["validation_example_count"])
            validation_batches = int(row["validation_batch_count"])
            if validation_count == 0:
                if validation_batches != 0 or validation_final is not None:
                    raise ValueError(
                        f"FreqRec epoch {epoch} absent validation metadata is inconsistent."
                    )
            else:
                validation_final_int = _exact_int(
                    validation_final,
                    f"epoch {epoch}.validation_final_batch_size",
                )
                expected_validation_batches = math.ceil(validation_count / batch_size)
                expected_validation_final = validation_count - batch_size * (
                    expected_validation_batches - 1
                )
                if (
                    validation_batches != expected_validation_batches
                    or validation_final_int != expected_validation_final
                ):
                    raise ValueError(
                        f"FreqRec epoch {epoch} validation batch metadata is inconsistent."
                    )
            if row["drop_last"] is not False:
                raise ValueError(f"FreqRec epoch {epoch}.drop_last must be false.")
            if row["train_sampler"] != "seeded_random":
                raise ValueError(
                    f"FreqRec epoch {epoch}.train_sampler must be seeded_random."
                )
            if row["evaluation_sampler"] != "sequential":
                raise ValueError(
                    f"FreqRec epoch {epoch}.evaluation_sampler must be sequential."
                )
            protocol = row["checkpoint_protocol"]
            if previous_protocol is not None and protocol != previous_protocol:
                raise ValueError(
                    "FreqRec epoch metrics checkpoint_protocol must remain constant."
                )
            previous_protocol = protocol
            if protocol == "fixed_epoch":
                if (
                    row["improved"] is not None
                    or row["best_epoch"] is not None
                    or row["best_metric"] is not None
                ):
                    raise ValueError(
                        f"FreqRec fixed-epoch diagnostic row {epoch} must not "
                        "contain checkpoint-selection state."
                    )
            elif protocol == "validation_best":
                if type(row["improved"]) is not bool:
                    raise ValueError(
                        f"FreqRec validation-best epoch {epoch}.improved must be bool."
                    )
                best_epoch = _exact_int(
                    row["best_epoch"], f"epoch {epoch}.best_epoch"
                )
                if not 1 <= best_epoch <= epoch:
                    raise ValueError(
                        f"FreqRec validation-best epoch {epoch}.best_epoch is invalid."
                    )
                best_metric = _finite(
                    row["best_metric"], f"epoch {epoch}.best_metric"
                )
                if epoch == 1 and (
                    row["improved"] is not True or best_epoch != 1
                ):
                    raise ValueError(
                        "FreqRec validation-best epoch 1 must establish the initial best state."
                    )
                if row["improved"] is True and best_epoch != epoch:
                    raise ValueError(
                        f"FreqRec validation-best epoch {epoch} improved but "
                        "best_epoch is not the current epoch."
                    )
                if previous_best_epoch is not None:
                    if best_epoch < previous_best_epoch:
                        raise ValueError(
                            f"FreqRec validation-best epoch {epoch}.best_epoch decreased."
                        )
                    if row["improved"] is True:
                        if previous_best_metric is None or best_metric <= previous_best_metric:
                            raise ValueError(
                                f"FreqRec validation-best epoch {epoch} improved but "
                                "best_metric did not strictly increase."
                            )
                    else:
                        if best_epoch != previous_best_epoch:
                            raise ValueError(
                                f"FreqRec validation-best epoch {epoch} did not preserve "
                                "the previous best_epoch."
                            )
                        if previous_best_metric is None or not math.isclose(
                            best_metric,
                            previous_best_metric,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        ):
                            raise ValueError(
                                f"FreqRec validation-best epoch {epoch} did not preserve "
                                "the previous best_metric."
                            )
                previous_best_epoch = best_epoch
                previous_best_metric = best_metric
            else:
                raise ValueError(
                    f"FreqRec epoch {epoch}.checkpoint_protocol is unsupported."
                )
            validation = row.get("validation")
            if not isinstance(validation, Mapping):
                raise ValueError(f"FreqRec epoch {epoch} validation must be an object.")
            for key, value in validation.items():
                _finite(value, f"epoch {epoch}.validation.{key}")
            rows.append(dict(row))
    if not rows:
        raise ValueError("FreqRec epoch metrics JSONL must not be empty.")
    return rows


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"FreqRec {field} must be an integer.")
    return int(value)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FreqRec {field} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"FreqRec {field} must be finite.")
    return number


__all__ = [
    "load_freqrec_epoch_metrics",
    "summarize_freqrec_epoch_diagnostics",
]
