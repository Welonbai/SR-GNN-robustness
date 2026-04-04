from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import random

from attack.data.session_stats import SessionStats


def _weighted_choice(items: Sequence[int], weights: Sequence[float]) -> int:
    if not items:
        raise ValueError("Cannot sample from an empty distribution.")
    return random.choices(items, weights=weights, k=1)[0]


@dataclass(frozen=True)
class FakeSessionParameters:
    initial_item: int
    length: int


class FakeSessionParameterSampler:
    def __init__(self, stats: SessionStats) -> None:
        self._first_items = list(stats.first_item_probs.keys())
        self._first_item_weights = list(stats.first_item_probs.values())
        self._lengths = list(stats.session_length_probs.keys())
        self._length_weights = list(stats.session_length_probs.values())

    def sample_initial_item(self) -> int:
        return int(_weighted_choice(self._first_items, self._first_item_weights))

    def sample_length(self) -> int:
        return int(_weighted_choice(self._lengths, self._length_weights))

    def sample(self) -> FakeSessionParameters:
        return FakeSessionParameters(
            initial_item=self.sample_initial_item(),
            length=self.sample_length(),
        )


__all__ = ["FakeSessionParameters", "FakeSessionParameterSampler"]
