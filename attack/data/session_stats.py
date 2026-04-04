from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import pickle


@dataclass(frozen=True)
class SessionStats:
    first_item_counts: dict[int, int]
    first_item_probs: dict[int, float]
    session_length_counts: dict[int, int]
    session_length_probs: dict[int, float]
    item_counts: dict[int, int]
    total_sessions: int
    total_items: int


def _normalize_counts(counts: Counter[int]) -> dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / float(total) for key, value in counts.items()}


def load_train_sessions(train_path: str | Path) -> list[list[int]]:
    path = Path(train_path)
    data = pickle.load(path.open("rb"))
    if isinstance(data, (list, tuple)) and len(data) == 2:
        sessions = data[0]
    else:
        sessions = data
    return [list(session) for session in sessions]


def compute_session_stats(sessions: Iterable[Sequence[int]]) -> SessionStats:
    first_item_counts: Counter[int] = Counter()
    session_length_counts: Counter[int] = Counter()
    item_counts: Counter[int] = Counter()
    total_sessions = 0
    total_items = 0

    for session in sessions:
        if not session:
            continue
        total_sessions += 1
        total_items += len(session)
        first_item_counts[session[0]] += 1
        session_length_counts[len(session)] += 1
        item_counts.update(session)

    return SessionStats(
        first_item_counts=dict(first_item_counts),
        first_item_probs=_normalize_counts(first_item_counts),
        session_length_counts=dict(session_length_counts),
        session_length_probs=_normalize_counts(session_length_counts),
        item_counts=dict(item_counts),
        total_sessions=total_sessions,
        total_items=total_items,
    )


__all__ = ["SessionStats", "load_train_sessions", "compute_session_stats"]
