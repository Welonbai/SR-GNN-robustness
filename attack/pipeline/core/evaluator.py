from __future__ import annotations

import json
from math import log2
from pathlib import Path
from typing import Any, Mapping, Sequence


_SUPPORTED_METRIC_BASES = ("precision", "recall", "mrr", "ndcg")


def evaluate_targeted_metrics(
    rankings: Sequence[Sequence[int]] | None,
    *,
    target_item: int,
    metrics: Sequence[str],
    topk: Sequence[int],
) -> tuple[dict[str, float | None], bool]:
    """Compute requested targeted metrics from standardized top-k rankings."""
    scoped_metrics = _normalize_metric_names(metrics, scope="targeted")
    if rankings is None:
        return _empty_metric_result(scoped_metrics, topk), False

    ranks = _compute_ranks(rankings, target_item)
    return _evaluate_rank_metrics(ranks, metrics=scoped_metrics, topk=topk), True


def evaluate_ground_truth_metrics(
    rankings: Sequence[Sequence[int]] | None,
    *,
    labels: Sequence[int] | None,
    metrics: Sequence[str],
    topk: Sequence[int],
) -> tuple[dict[str, float | None], bool]:
    """Compute requested ground-truth metrics from standardized top-k rankings."""
    scoped_metrics = _normalize_metric_names(metrics, scope="ground_truth")
    if rankings is None or labels is None:
        return _empty_metric_result(scoped_metrics, topk), False

    ranks = _compute_label_ranks(rankings, labels)
    return _evaluate_rank_metrics(ranks, metrics=scoped_metrics, topk=topk), True


def evaluate_prediction_metrics(
    rankings: Sequence[Sequence[int]] | None,
    *,
    target_item: int,
    ground_truth_labels: Sequence[int] | None,
    targeted_metrics: Sequence[str],
    ground_truth_metrics: Sequence[str],
    topk: Sequence[int],
) -> tuple[dict[str, float | None], bool]:
    """Compute both targeted and ground-truth metrics from one prediction list."""
    targeted_metric_list = list(targeted_metrics)
    ground_truth_metric_list = list(ground_truth_metrics)
    targeted_results, targeted_available = evaluate_targeted_metrics(
        rankings,
        target_item=target_item,
        metrics=targeted_metric_list,
        topk=topk,
    )
    ground_truth_results, ground_truth_available = evaluate_ground_truth_metrics(
        rankings,
        labels=ground_truth_labels,
        metrics=ground_truth_metric_list,
        topk=topk,
    )
    combined = {**targeted_results, **ground_truth_results}
    targeted_ok = targeted_available or not targeted_metric_list
    ground_truth_ok = ground_truth_available or not ground_truth_metric_list
    return combined, bool(targeted_ok and ground_truth_ok)


def evaluate_runner(runner, test_data, topk: int) -> dict[str, float]:
    hit, mrr = runner.evaluate(test_data, topk=topk)
    return {"hit": float(hit), "mrr": float(mrr), "topk": int(topk)}


def save_metrics(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_predictions(
    path: str | Path,
    *,
    topk: int,
    rankings: Sequence[Sequence[int]] | None,
    victim: str,
    target_item: int,
    reason: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "victim": victim,
        "target_item": int(target_item),
        "topk": int(topk),
        "available": rankings is not None,
        "count": int(len(rankings)) if rankings is not None else 0,
    }
    if rankings is not None:
        payload["rankings"] = [list(map(int, ranking)) for ranking in rankings]
    if reason:
        payload["reason"] = reason
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _compute_ranks(rankings: Sequence[Sequence[int]], target_item: int) -> list[int]:
    target_id = int(target_item)
    ranks: list[int] = []
    for ranking in rankings:
        try:
            idx = list(ranking).index(target_id)
        except ValueError:
            ranks.append(0)
        else:
            ranks.append(int(idx + 1))
    return ranks


def _compute_label_ranks(rankings: Sequence[Sequence[int]], labels: Sequence[int]) -> list[int]:
    if len(rankings) != len(labels):
        raise ValueError(
            "rankings and labels must have the same length for ground-truth evaluation: "
            f"{len(rankings)} != {len(labels)}"
        )
    ranks: list[int] = []
    for ranking, label in zip(rankings, labels):
        target_id = int(label)
        try:
            idx = list(ranking).index(target_id)
        except ValueError:
            ranks.append(0)
        else:
            ranks.append(int(idx + 1))
    return ranks


def _normalize_metric_names(metrics: Sequence[str], *, scope: str) -> list[str]:
    if scope not in {"targeted", "ground_truth"}:
        raise ValueError(f"Unsupported metric scope: {scope}")
    normalized: list[str] = []
    for metric in metrics:
        base = _metric_base_name(metric)
        normalized.append(f"{scope}_{base}")
    return normalized


def _metric_base_name(metric: str) -> str:
    metric_name = str(metric).strip().lower()
    if metric_name.startswith("targeted_"):
        base = metric_name[len("targeted_") :]
    elif metric_name.startswith("ground_truth_"):
        base = metric_name[len("ground_truth_") :]
    else:
        base = metric_name
    if base not in _SUPPORTED_METRIC_BASES:
        raise ValueError(f"Unsupported metric: {metric}")
    return base


def _empty_metric_result(
    metrics: Sequence[str],
    topk: Sequence[int],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in metrics:
        for k in topk:
            result[f"{metric}@{int(k)}"] = None
    return result


def _evaluate_rank_metrics(
    ranks: Sequence[int],
    *,
    metrics: Sequence[str],
    topk: Sequence[int],
) -> dict[str, float | None]:
    computed: dict[str, float | None] = {}
    sample_count = len(ranks)
    for metric in metrics:
        base = _metric_base_name(metric)
        for k_value in topk:
            k = int(k_value)
            hits = sum(1 for rank in ranks if 0 < rank <= k)
            if base == "precision":
                value = (hits / (sample_count * k)) if sample_count else 0.0
            elif base == "recall":
                value = (hits / sample_count) if sample_count else 0.0
            elif base == "mrr":
                value = (
                    sum((1.0 / rank) for rank in ranks if 0 < rank <= k) / sample_count
                    if sample_count
                    else 0.0
                )
            elif base == "ndcg":
                value = (
                    sum((1.0 / log2(rank + 1)) for rank in ranks if 0 < rank <= k)
                    / sample_count
                    if sample_count
                    else 0.0
                )
            else:  # pragma: no cover - guarded by _metric_base_name
                raise ValueError(f"Unsupported metric: {metric}")
            computed[f"{metric}@{k}"] = float(value)
    return computed


__all__ = [
    "evaluate_ground_truth_metrics",
    "evaluate_prediction_metrics",
    "evaluate_targeted_metrics",
    "evaluate_runner",
    "save_metrics",
    "save_predictions",
]
