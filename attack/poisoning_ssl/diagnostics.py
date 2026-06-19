from __future__ import annotations

from collections import Counter
from math import ceil
from statistics import mean
from typing import Mapping, Sequence


def nearest_rank_percentile(values: Sequence[int], percentile: int | float) -> int:
    if not values:
        return 0
    if not 0 <= float(percentile) <= 100:
        raise ValueError("percentile must be in [0, 100].")
    ordered = sorted(int(value) for value in values)
    if float(percentile) == 0:
        return int(ordered[0])
    rank = int(ceil(float(percentile) / 100.0 * len(ordered)))
    rank = max(1, min(rank, len(ordered)))
    return int(ordered[rank - 1])


def length_stats(sessions: Sequence[Sequence[int]]) -> dict[str, object]:
    lengths = [int(len(session)) for session in sessions]
    return length_stats_from_lengths(lengths)


def length_stats_from_lengths(lengths: Sequence[int]) -> dict[str, object]:
    lengths = [int(length) for length in lengths]
    if not lengths:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "p50": 0,
            "p75": 0,
            "p80": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "ratio_len_le_3": 0.0,
            "ratio_len_le_5": 0.0,
            "ratio_len_le_10": 0.0,
            "ratio_len_gt_10": 0.0,
            "ratio_len_gt_20": 0.0,
            "length_count_by_length": {},
        }
    count = len(lengths)
    return {
        "count": int(count),
        "mean": float(mean(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
        "p50": nearest_rank_percentile(lengths, 50),
        "p75": nearest_rank_percentile(lengths, 75),
        "p80": nearest_rank_percentile(lengths, 80),
        "p90": nearest_rank_percentile(lengths, 90),
        "p95": nearest_rank_percentile(lengths, 95),
        "p99": nearest_rank_percentile(lengths, 99),
        "ratio_len_le_3": float(sum(length <= 3 for length in lengths) / count),
        "ratio_len_le_5": float(sum(length <= 5 for length in lengths) / count),
        "ratio_len_le_10": float(sum(length <= 10 for length in lengths) / count),
        "ratio_len_gt_10": float(sum(length > 10 for length in lengths) / count),
        "ratio_len_gt_20": float(sum(length > 20 for length in lengths) / count),
        "length_count_by_length": {
            int(length): int(value) for length, value in sorted(Counter(lengths).items())
        },
    }


def target_diagnostics(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
) -> dict[str, object]:
    target = int(target_item)
    occurrence_counts = [sum(1 for item in session if int(item) == target) for session in sessions]
    position_counts: Counter[int] = Counter()
    for session in sessions:
        for position, item in enumerate(session):
            if int(item) == target:
                position_counts[int(position)] += 1
    total = len(sessions)
    pos0 = int(position_counts.get(0, 0))
    nonzero = int(sum(count for position, count in position_counts.items() if int(position) > 0))
    return {
        "target_occurrence_stats": {
            "count_by_occurrence": {
                int(count): int(value)
                for count, value in sorted(Counter(occurrence_counts).items())
            },
            "total_occurrences": int(sum(occurrence_counts)),
        },
        "target_position_distribution": {
            int(position): int(count) for position, count in sorted(position_counts.items())
        },
        "target_pos0_count": pos0,
        "target_pos0_ratio": 0.0 if total <= 0 else float(pos0 / total),
        "target_nonzero_count": nonzero,
        "target_nonzero_ratio": 0.0 if total <= 0 else float(nonzero / total),
        "single_target_count": int(sum(count == 1 for count in occurrence_counts)),
        "multi_target_count": int(sum(count > 1 for count in occurrence_counts)),
        "no_target_count": int(sum(count == 0 for count in occurrence_counts)),
    }


def duplicate_diagnostics(sessions: Sequence[Sequence[int]]) -> dict[str, object]:
    counts = Counter(tuple(int(item) for item in session) for session in sessions)
    duplicate_count = int(sum(value - 1 for value in counts.values() if value > 1))
    total = int(len(sessions))
    unique_count = int(len(counts))
    return {
        "unique_fake_session_count": unique_count,
        "unique_fake_session_ratio": 0.0 if total <= 0 else float(unique_count / total),
        "duplicate_session_count": duplicate_count,
        "duplicate_session_ratio": 0.0 if total <= 0 else float(duplicate_count / total),
    }


def target_label_pair_count(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
) -> int:
    target = int(target_item)
    return int(
        sum(
            1
            for session in sessions
            for position, item in enumerate(session)
            if position >= 1 and int(item) == target
        )
    )


def budget_diagnostics(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    clean_label_count: int,
) -> dict[str, object]:
    expanded_pair_count_added = int(sum(max(0, len(session) - 1) for session in sessions))
    target_label_added = target_label_pair_count(sessions, target_item=int(target_item))
    denominator = int(clean_label_count)
    return {
        "expanded_pair_count_added": expanded_pair_count_added,
        "effective_expanded_budget_ratio": (
            0.0 if denominator <= 0 else float(expanded_pair_count_added / denominator)
        ),
        "target_label_pair_count_added": int(target_label_added),
        "target_label_pair_ratio_added": (
            0.0 if denominator <= 0 else float(target_label_added / denominator)
        ),
    }


def merged_diagnostics(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    clean_label_count: int,
) -> dict[str, object]:
    return {
        **target_diagnostics(sessions, target_item=int(target_item)),
        **duplicate_diagnostics(sessions),
        **budget_diagnostics(
            sessions,
            target_item=int(target_item),
            clean_label_count=int(clean_label_count),
        ),
    }


def stringify_mapping_keys(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): stringify_mapping_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stringify_mapping_keys(item) for item in value]
    return value


__all__ = [
    "budget_diagnostics",
    "duplicate_diagnostics",
    "length_stats",
    "length_stats_from_lengths",
    "merged_diagnostics",
    "nearest_rank_percentile",
    "stringify_mapping_keys",
    "target_diagnostics",
    "target_label_pair_count",
]
