from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Sequence


def candidate_positions(
    length: int,
    topk_ratio: float,
    *,
    nonzero_when_possible: bool = True,
) -> list[int]:
    if int(length) <= 0:
        raise ValueError("Session length must be positive.")
    if not 0.0 < float(topk_ratio) <= 1.0:
        raise ValueError("topk_ratio must be within (0, 1].")
    topk_count = max(1, int(math.ceil(int(length) * float(topk_ratio))))
    max_index = min(topk_count, int(length)) - 1
    candidates = list(range(0, max_index + 1))
    if nonzero_when_possible and any(position > 0 for position in candidates):
        candidates = [position for position in candidates if position > 0]
    if not candidates:
        raise ValueError("No valid candidate positions.")
    return candidates


def valid_position_mask(
    length: int,
    topk_ratio: float,
    *,
    nonzero_when_possible: bool = True,
) -> list[bool]:
    candidates = set(
        candidate_positions(
            length,
            topk_ratio,
            nonzero_when_possible=nonzero_when_possible,
        )
    )
    return [index in candidates for index in range(int(length))]


def valid_position_mask_for_session(
    session: Sequence[int],
    target_item: int,
    topk_ratio: float,
    *,
    nonzero_when_possible: bool = True,
) -> list[bool]:
    if int(target_item) <= 0:
        raise ValueError("target_item must be positive.")
    base_mask = valid_position_mask(
        len(session),
        topk_ratio,
        nonzero_when_possible=nonzero_when_possible,
    )
    return [
        bool(is_valid) and int(item) != int(target_item)
        for is_valid, item in zip(base_mask, session)
    ]


def filter_effective_templates(
    sessions: Sequence[Sequence[int]],
) -> tuple[list[list[int]], dict[str, int]]:
    effective = [list(session) for session in sessions if len(session) >= 2]
    return effective, {
        "original_template_count": int(len(sessions)),
        "filtered_template_count": int(len(sessions) - len(effective)),
        "effective_template_count": int(len(effective)),
    }


def filter_templates_with_valid_candidates(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    topk_ratio: float,
    nonzero_when_possible: bool = True,
) -> tuple[list[list[int]], dict[str, int]]:
    effective = []
    filtered = 0
    for session in sessions:
        mask = valid_position_mask_for_session(
            session,
            int(target_item),
            topk_ratio,
            nonzero_when_possible=nonzero_when_possible,
        )
        if any(mask):
            effective.append(list(session))
        else:
            filtered += 1
    return effective, {
        "filtered_no_valid_candidate_count": int(filtered),
        "target_effective_template_count": int(len(effective)),
    }


def sessions_sha1(sessions: Sequence[Sequence[int]]) -> str:
    payload = [[int(item) for item in session] for session in sessions]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def position_distribution(positions: Sequence[int]) -> dict[str, object]:
    counts = Counter(int(position) for position in positions)
    total = int(sum(counts.values()))
    return {
        "total": total,
        "counts": {str(position): int(count) for position, count in sorted(counts.items())},
        "ratios": {
            str(position): (float(count) / float(total) if total else 0.0)
            for position, count in sorted(counts.items())
        },
    }


def target_label_pair_count(positions: Sequence[int]) -> int:
    return int(sum(1 for position in positions if int(position) > 0))


__all__ = [
    "candidate_positions",
    "filter_effective_templates",
    "filter_templates_with_valid_candidates",
    "position_distribution",
    "sessions_sha1",
    "target_label_pair_count",
    "valid_position_mask",
    "valid_position_mask_for_session",
]
