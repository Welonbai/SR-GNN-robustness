from __future__ import annotations

import json
from math import log2
from pathlib import Path
from typing import Any, Mapping, Sequence


def evaluate_targeted_metrics(
    rankings: Sequence[Sequence[int]] | None,
    *,
    target_item: int,
    metrics: Sequence[str],
    topk: Sequence[int],
) -> tuple[dict[str, float | None], bool]:
    """Compute requested targeted metrics from standardized top-k rankings."""
    topk_values = list(topk)
    if rankings is None:
        result: dict[str, float | None] = {}
        for metric in metrics:
            for k in topk_values:
                result[f"{metric}@{k}"] = None
        return result, False

    ranks = _compute_ranks(rankings, target_item)
    computed: dict[str, float | None] = {}
    sample_count = len(ranks)
    for metric in metrics:
        for k in topk_values:
            if metric == "targeted_precision":
                hits = sum(1 for r in ranks if 0 < r <= k)
                value = (hits / (sample_count * k)) if sample_count else 0.0
            elif metric == "targeted_recall":
                hits = sum(1 for r in ranks if 0 < r <= k)
                value = (hits / sample_count) if sample_count else 0.0
            elif metric == "targeted_mrr":
                value = (
                    sum((1.0 / r) for r in ranks if 0 < r <= k) / sample_count
                    if sample_count
                    else 0.0
                )
            elif metric == "targeted_ndcg":
                value = (
                    sum((1.0 / log2(r + 1)) for r in ranks if 0 < r <= k) / sample_count
                    if sample_count
                    else 0.0
                )
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            computed[f"{metric}@{k}"] = float(value)
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


__all__ = [
    "evaluate_targeted_metrics",
    "evaluate_runner",
    "save_metrics",
    "save_predictions",
]
