from __future__ import annotations

import random

from .session_stats import SessionStats


def _popular_pool(stats: SessionStats) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select a target item.")
    avg_count = stats.total_items / float(len(stats.item_counts))
    pool = [item for item, count in stats.item_counts.items() if count > avg_count]
    if not pool:
        raise ValueError("Popular pool is empty under item_count > average_count.")
    return pool


def sample_one_from_popular(stats: SessionStats, seed: int) -> int:
    rng = random.Random(seed)
    pool = _popular_pool(stats)
    return int(rng.choice(pool))


def _unpopular_pool(stats: SessionStats, threshold: int = 10) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select a target item.")
    pool = [item for item, count in stats.item_counts.items() if count < threshold]
    if not pool:
        raise ValueError("Unpopular pool is empty under item_count < 10.")
    return pool


def sample_one_from_unpopular(stats: SessionStats, seed: int, threshold: int = 10) -> int:
    rng = random.Random(seed)
    pool = _unpopular_pool(stats, threshold=threshold)
    return int(rng.choice(pool))


__all__ = ["sample_one_from_popular", "sample_one_from_unpopular"]
