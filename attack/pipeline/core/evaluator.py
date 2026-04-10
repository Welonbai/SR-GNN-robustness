from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def compute_targeted_precision(rankings: Iterable[Iterable[int]], target_item: int) -> float:
    """Compute mean hit rate for the target item over top-k rankings.

    Rankings must use 1-based item ids to match target_item.
    """
    target_id = int(target_item)
    hits = [1.0 if target_id in ranking else 0.0 for ranking in rankings]
    return float(np.mean(hits)) if hits else 0.0


def compute_targeted_mrr(rankings: Iterable[Iterable[int]], target_item: int) -> float:
    """Compute mean reciprocal rank for the target item over top-k rankings.

    Rankings must use 1-based item ids to match target_item.
    """
    target_id = int(target_item)
    scores: list[float] = []
    for ranking in rankings:
        try:
            idx = list(ranking).index(target_id)
        except ValueError:
            scores.append(0.0)
        else:
            scores.append(1.0 / float(idx + 1))
    return float(np.mean(scores)) if scores else 0.0


def evaluate_targeted_metrics(
    rankings: Sequence[Sequence[int]] | None,
    *,
    target_item: int,
    metrics: Sequence[str],
    topk: int,
) -> tuple[dict[str, float | None], bool]:
    """Compute requested targeted metrics from standardized top-k rankings."""
    if rankings is None:
        result: dict[str, float | None] = {}
        for metric in metrics:
            key = f"{metric}_at_k"
            result[key] = None
        result["topk"] = int(topk)
        return result, False

    computed: dict[str, float | None] = {"topk": int(topk)}
    for metric in metrics:
        if metric == "targeted_precision":
            computed["targeted_precision_at_k"] = compute_targeted_precision(rankings, target_item)
        elif metric == "targeted_mrr":
            computed["targeted_mrr_at_k"] = compute_targeted_mrr(rankings, target_item)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return computed, True


def evaluate_runner(runner, test_data, topk: int) -> dict[str, float]:
    hit, mrr = runner.evaluate(test_data, topk=topk)
    return {"hit": float(hit), "mrr": float(mrr), "topk": int(topk)}


def save_metrics(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


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


__all__ = [
    "compute_targeted_precision",
    "compute_targeted_mrr",
    "evaluate_targeted_metrics",
    "evaluate_runner",
    "save_metrics",
    "save_predictions",
]
