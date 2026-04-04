from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from pytorch_code.model import forward as srg_forward, trans_to_cpu


def _targeted_precision_from_rankings(rankings: Iterable[Iterable[int]], target_item: int) -> float:
    target_index = int(target_item) - 1
    hits = [1.0 if target_index in ranking else 0.0 for ranking in rankings]
    return float(np.mean(hits)) if hits else 0.0


def evaluate_targeted_precision_at_k(runner, test_data, target_item: int, topk: int) -> float:
    if runner.model is None:
        raise RuntimeError("Model is not initialized. Call build_model() first.")
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if target_item <= 0:
        raise ValueError("target_item must be a positive item id.")

    runner.model.eval()
    rankings = []
    slices = test_data.generate_batch(runner.model.batch_size)
    with torch.no_grad():
        for i in slices:
            _, scores = srg_forward(runner.model, i, test_data)
            k = min(topk, scores.shape[1])
            topk_indices = scores.topk(k)[1]
            topk_indices = trans_to_cpu(topk_indices).detach().numpy()
            rankings.extend(topk_indices)
    return _targeted_precision_from_rankings(rankings, target_item)


def evaluate_runner(runner, test_data, topk: int) -> dict[str, float]:
    hit, mrr = runner.evaluate(test_data, topk=topk)
    return {"hit": float(hit), "mrr": float(mrr), "topk": int(topk)}


def save_metrics(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


__all__ = ["evaluate_runner", "evaluate_targeted_precision_at_k", "save_metrics"]
