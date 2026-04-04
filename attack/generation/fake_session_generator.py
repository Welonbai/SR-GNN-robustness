from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence
import random

import numpy as np

from attack.generation.score_smoothing import min_max_smooth


@dataclass(frozen=True)
class FakeSession:
    items: list[int]


def _to_numpy(scores) -> np.ndarray:
    if isinstance(scores, np.ndarray):
        return scores.astype(np.float32, copy=False)
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(scores, torch.Tensor):
        return scores.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(scores, dtype=np.float32)


class FakeSessionGenerator:
    def __init__(
        self,
        runner,
        sampler,
        topk: int,
        smoothing_fn: Callable[[np.ndarray], np.ndarray] = min_max_smooth,
    ) -> None:
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.runner = runner
        self.sampler = sampler
        self.topk = topk
        self.smoothing_fn = smoothing_fn

    def generate_one(self, initial_item: int | None = None, length: int | None = None) -> FakeSession:
        initial_item = initial_item if initial_item is not None else self.sampler.sample_initial_item()
        length = length if length is not None else self.sampler.sample_length()
        if length < 1:
            raise ValueError("length must be at least 1.")

        session = [int(initial_item)]
        while len(session) < length:
            scores = self.runner.score_session(session)
            scores_np = _to_numpy(scores)
            smoothed = _to_numpy(self.smoothing_fn(scores_np))
            if smoothed.size == 0:
                raise ValueError("Score vector is empty.")
            k = min(self.topk, smoothed.size)
            topk_indices = np.argsort(smoothed)[-k:]
            topk_weights = smoothed[topk_indices]
            if np.all(topk_weights == 0):
                next_index = random.choice(topk_indices.tolist())
            else:
                next_index = random.choices(topk_indices.tolist(), weights=topk_weights.tolist(), k=1)[0]
            next_item = int(next_index + 1)
            session.append(next_item)
        return FakeSession(items=session)

    def generate_many(self, count: int) -> list[FakeSession]:
        if count < 1:
            return []
        return [self.generate_one() for _ in range(count)]


__all__ = ["FakeSession", "FakeSessionGenerator"]
