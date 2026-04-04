from __future__ import annotations

from typing import Iterable
import random

from .session_stats import SessionStats


def select_target_item(
    mode: str,
    stats: SessionStats,
    explicit_item: int | None = None,
    rng: random.Random | None = None,
    unpopular_threshold: int = 10,
) -> int:
    rng = rng or random.Random()
    mode = mode.lower().strip()

    if mode == "explicit":
        if explicit_item is None:
            raise ValueError("explicit_item must be provided when mode='explicit'.")
        return int(explicit_item)

    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select a target item.")

    if mode == "popular":
        avg_count = stats.total_items / float(len(stats.item_counts))
        pool = [item for item, count in stats.item_counts.items() if count > avg_count]
        if not pool:
            pool = [item for item, _ in sorted(stats.item_counts.items(), key=lambda x: -x[1])[:50]]
        return int(rng.choice(pool))

    if mode == "unpopular":
        pool = [item for item, count in stats.item_counts.items() if count < unpopular_threshold]
        if not pool:
            pool = [item for item, _ in sorted(stats.item_counts.items(), key=lambda x: x[1])[:50]]
        return int(rng.choice(pool))

    raise ValueError(f"Unknown target selection mode: {mode}")


def resolve_target_item(
    stats: SessionStats,
    mode: str,
    explicit_item: int | None = None,
    seed: int | None = None,
    unpopular_threshold: int = 10,
) -> int:
    if explicit_item is not None:
        return int(explicit_item)
    rng = random.Random(seed) if seed is not None else None
    return select_target_item(
        mode=mode,
        stats=stats,
        explicit_item=explicit_item,
        rng=rng,
        unpopular_threshold=unpopular_threshold,
    )


__all__ = ["select_target_item", "resolve_target_item"]
