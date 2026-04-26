from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Mapping, Sequence

from .bucket_selector import (
    BUCKET_ABS_POS2,
    BUCKET_ABS_POS3PLUS,
    BUCKET_NONFIRST_NONZERO,
    BucketSessionSelectionRecord,
    selection_record_to_jsonable,
)


def write_selected_positions_jsonl(
    path: str | Path,
    records: Sequence[BucketSessionSelectionRecord],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    selection_record_to_jsonable(record),
                    sort_keys=True,
                )
            )
            handle.write("\n")
    return output_path


def build_bucket_position_summary(
    records: Sequence[BucketSessionSelectionRecord],
    *,
    method_name: str,
    target_item: int,
    seed: int,
    seed_source: str = "position_opt_seed",
    replacement_topk_ratio: float,
    nonzero_action_when_possible: bool,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty.")

    selected_positions = [int(record.selected_position) for record in records]
    normalized_positions = [
        float(record.selected_position) / float(record.session_length - 1)
        for record in records
        if record.session_length > 1
    ]
    nonzero_candidate_counts = [int(record.candidate_count) for record in records]
    mode_candidate_counts = [int(record.mode_candidate_count) for record in records]
    fallback_count = sum(1 for record in records if record.fallback_used)
    fallback_to_pos0_only_count = sum(
        1 for record in records if record.fallback_to_pos0_only
    )
    pos0_removed_session_count = sum(1 for record in records if record.pos0_removed)
    total = len(records)

    summary = {
        "total_fake_sessions": int(total),
        "target_item": int(target_item),
        "method_name": str(method_name),
        "seed": int(seed),
        "seed_source": str(seed_source),
        "replacement_topk_ratio": float(replacement_topk_ratio),
        "nonzero_action_when_possible": bool(nonzero_action_when_possible),
        "fallback_count": int(fallback_count),
        "fallback_ratio": _ratio(fallback_count, total),
        "fallback_to_pos0_only_count": int(fallback_to_pos0_only_count),
        "pos0_removed_session_count": int(pos0_removed_session_count),
        "pos0_pct": _ratio(sum(1 for position in selected_positions if position == 0), total),
        "pos1_pct": _ratio(sum(1 for position in selected_positions if position == 1), total),
        "pos2_pct": _ratio(sum(1 for position in selected_positions if position == 2), total),
        "pos3_pct": _ratio(sum(1 for position in selected_positions if position == 3), total),
        "pos4_pos5_pct": _ratio(
            sum(1 for position in selected_positions if 4 <= position <= 5),
            total,
        ),
        "pos6plus_pct": _ratio(sum(1 for position in selected_positions if position >= 6), total),
        "mean_absolute_position": _mean(selected_positions),
        "median_absolute_position": float(median(selected_positions)),
        "min_absolute_position": int(min(selected_positions)),
        "max_absolute_position": int(max(selected_positions)),
        "unique_selected_positions": sorted(set(int(position) for position in selected_positions)),
        "unique_selected_position_count": int(len(set(selected_positions))),
        "mean_normalized_position": _mean(normalized_positions),
        "median_normalized_position": float(median(normalized_positions)),
        "p25_normalized_position": _percentile(normalized_positions, 25.0),
        "p75_normalized_position": _percentile(normalized_positions, 75.0),
        "p90_normalized_position": _percentile(normalized_positions, 90.0),
    }
    summary.update(
        _distribution_fields(
            values=nonzero_candidate_counts,
            prefix="nonzero_candidate_count",
        )
    )
    summary.update(
        _distribution_fields(
            values=mode_candidate_counts,
            prefix="mode_candidate_count",
        )
    )
    candidate_count_eq_1_pct = _ratio(
        sum(1 for count in nonzero_candidate_counts if count == 1),
        total,
    )
    candidate_count_le_2_pct = _ratio(
        sum(1 for count in nonzero_candidate_counts if count <= 2),
        total,
    )
    candidate_count_le_3_pct = _ratio(
        sum(1 for count in nonzero_candidate_counts if count <= 3),
        total,
    )
    candidate_count_ge_5_pct = _ratio(
        sum(1 for count in nonzero_candidate_counts if count >= 5),
        total,
    )
    summary.update(
        {
            "candidate_count_eq_1_pct": candidate_count_eq_1_pct,
            "candidate_count_le_2_pct": candidate_count_le_2_pct,
            "candidate_count_le_3_pct": candidate_count_le_3_pct,
            "candidate_count_ge_5_pct": candidate_count_ge_5_pct,
            "nonzero_candidate_count_eq_1_pct": candidate_count_eq_1_pct,
            "nonzero_candidate_count_le_2_pct": candidate_count_le_2_pct,
            "nonzero_candidate_count_le_3_pct": candidate_count_le_3_pct,
            "nonzero_candidate_count_ge_5_pct": candidate_count_ge_5_pct,
        }
    )
    return summary


def build_bucket_diagnostics(
    records: Sequence[BucketSessionSelectionRecord],
    *,
    method_name: str,
    target_item: int,
    seed: int,
    seed_source: str = "position_opt_seed",
    replacement_topk_ratio: float,
    nonzero_action_when_possible: bool,
    shared_fake_sessions_path: str,
    target_cohort_key: str,
    resolved_target_prefix: Sequence[int],
    cohort_validation: Mapping[str, Any],
) -> dict[str, Any]:
    summary = build_bucket_position_summary(
        records,
            method_name=method_name,
            target_item=target_item,
            seed=seed,
            seed_source=seed_source,
            replacement_topk_ratio=replacement_topk_ratio,
            nonzero_action_when_possible=nonzero_action_when_possible,
        )
    fallback_records = [record for record in records if record.fallback_used]
    diagnostics = {
        "method_name": str(method_name),
        "target_item": int(target_item),
        "position_selection_seed": int(seed),
        "seed_source": str(seed_source),
        "replacement_topk_ratio": float(replacement_topk_ratio),
        "nonzero_action_when_possible": bool(nonzero_action_when_possible),
        "shared_fake_sessions_path": str(shared_fake_sessions_path),
        "target_cohort_key": str(target_cohort_key),
        "resolved_target_prefix": [int(item) for item in resolved_target_prefix],
        "cohort_validation": dict(cohort_validation),
        "summary": summary,
        "fallback_sessions_by_session_length": _histogram(
            [int(record.session_length) for record in fallback_records]
        ),
        "fallback_sessions_by_candidate_count": _histogram(
            [int(record.candidate_count) for record in fallback_records]
        ),
        "fallback_reasons": _histogram(
            [
                str(record.fallback_reason)
                for record in fallback_records
                if record.fallback_reason is not None
            ]
        ),
        "selection_source_counts": _histogram(
            [str(record.selection_source) for record in records]
        ),
    }
    if method_name == BUCKET_ABS_POS2:
        diagnostics["selected_pos2_count"] = int(
            sum(1 for record in records if record.selected_position == 2)
        )
    if method_name == BUCKET_NONFIRST_NONZERO:
        diagnostics["non_first_candidate_count_distribution"] = _distribution_payload(
            [int(record.mode_candidate_count) for record in records]
        )
    if method_name == BUCKET_ABS_POS3PLUS:
        diagnostics["pos3plus_candidate_count_distribution"] = _distribution_payload(
            [int(record.mode_candidate_count) for record in records]
        )
    return diagnostics


def _distribution_fields(
    *,
    values: Sequence[int],
    prefix: str,
) -> dict[str, Any]:
    payload = _distribution_payload(values)
    return {
        f"{prefix}_{key}": value
        for key, value in payload.items()
    }


def _distribution_payload(values: Sequence[int]) -> dict[str, Any]:
    normalized = [float(value) for value in values]
    if not normalized:
        raise ValueError("distribution values must not be empty.")
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": _mean(normalized),
        "std": float(pstdev(normalized)),
        "p25": _percentile(normalized, 25.0),
        "p50": _percentile(normalized, 50.0),
        "p75": _percentile(normalized, 75.0),
        "p90": _percentile(normalized, 90.0),
        "p95": _percentile(normalized, 95.0),
    }


def _histogram(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return {
        key: int(counts[key])
        for key in sorted(counts, key=_histogram_sort_key)
    }


def _histogram_sort_key(value: str) -> tuple[int, Any]:
    if value.lstrip("-").isdigit():
        return (0, int(value))
    return (1, value)


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        raise ValueError("values must not be empty.")
    return float(sum(float(value) for value in values) / len(values))


def _percentile(values: Sequence[float | int], percentile: float) -> float:
    normalized = sorted(float(value) for value in values)
    if not normalized:
        raise ValueError("values must not be empty.")
    if len(normalized) == 1:
        return float(normalized[0])
    rank = (float(percentile) / 100.0) * float(len(normalized) - 1)
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(normalized[lower_index])
    lower_value = normalized[lower_index]
    upper_value = normalized[upper_index]
    weight = rank - float(lower_index)
    return float((1.0 - weight) * lower_value + weight * upper_value)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total) * 100.0


__all__ = [
    "build_bucket_diagnostics",
    "build_bucket_position_summary",
    "write_selected_positions_jsonl",
]
