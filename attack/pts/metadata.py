from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def length_distribution_from_lengths(lengths: Sequence[int]) -> dict[str, int]:
    counts: Counter[int] = Counter(int(length) for length in lengths)
    return {f"len{length}": int(count) for length, count in sorted(counts.items())}


def integer_count_distribution(values: Sequence[int]) -> dict[str, int]:
    counts: Counter[int] = Counter(int(value) for value in values)
    return {str(value): int(count) for value, count in sorted(counts.items())}


def ratio_dict(counts: Mapping[str, int], *, total: int | None = None) -> dict[str, float]:
    denominator = (
        int(sum(int(count) for count in counts.values()))
        if total is None
        else int(total)
    )
    if denominator <= 0:
        return {str(key): 0.0 for key in sorted(counts)}
    return {
        str(key): float(counts[key]) / float(denominator)
        for key in sorted(counts)
    }


def build_pts_batch_summary(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    total = int(len(records))
    action_counts_counter: Counter[str] = Counter(
        str(record["action"]) for record in records
    )
    group_counts_counter: Counter[str] = Counter(
        str(record["suffix_len_group"]) for record in records
    )
    action_counts = _string_counter(action_counts_counter)
    group_counts = _string_counter(group_counts_counter)
    action_counts_by_group = _action_counts_by_group(records)
    generated_records = [
        record for record in records if str(record["continuation_source"]) == "generate"
    ]
    generated_contains_target_count = int(
        sum(
            1
            for record in generated_records
            if int(record["target_item"]) in set(_int_list(record["generated_suffix"]))
        )
    )
    generated_count = int(len(generated_records))
    target_tail_count = int(sum(1 for record in records if bool(record["target_tail"])))
    multiple_target_count = int(
        sum(
            1
            for record in records
            if int(record["target_occurrence_count_final"]) > 1
        )
    )

    return {
        "fake_session_count": total,
        "action_counts": action_counts,
        "action_ratios": ratio_dict(action_counts, total=total),
        "action_counts_by_group": action_counts_by_group,
        "action_ratios_by_group": {
            group: ratio_dict(actions)
            for group, actions in action_counts_by_group.items()
        },
        "group_counts": group_counts,
        "template_length_distribution": length_distribution_from_lengths(
            [int(record["template_length"]) for record in records]
        ),
        "residual_suffix_length_distribution": length_distribution_from_lengths(
            [int(record["residual_suffix_length"]) for record in records]
        ),
        "final_length_distribution": length_distribution_from_lengths(
            [int(record["final_length"]) for record in records]
        ),
        "length_shift_distribution": integer_count_distribution(
            [int(record["length_shift_from_template"]) for record in records]
        ),
        "target_tail_count": target_tail_count,
        "target_tail_ratio": _ratio(target_tail_count, total),
        "generated_suffix_count": generated_count,
        "generated_suffix_length_distribution": length_distribution_from_lengths(
            [int(record["generated_suffix_length"]) for record in generated_records]
        ),
        "generated_suffix_contains_target_count": generated_contains_target_count,
        "generated_suffix_contains_target_ratio": _ratio(
            generated_contains_target_count,
            total,
        ),
        "generated_suffix_contains_target_ratio_overall": _ratio(
            generated_contains_target_count,
            total,
        ),
        "generated_suffix_contains_target_ratio_among_generated": _ratio(
            generated_contains_target_count,
            generated_count,
        ),
        "generated_suffix_unique_item_mean": _mean(
            [
                len(set(_int_list(record["generated_suffix"])))
                for record in generated_records
            ]
        ),
        "final_sessions_with_multiple_target_count": multiple_target_count,
        "final_sessions_with_multiple_target_ratio": _ratio(
            multiple_target_count,
            total,
        ),
        "dynamic_mask_counts": {
            "dynamic_mask_applied": int(
                sum(1 for record in records if bool(record["dynamic_mask_applied"]))
            ),
            "disabled_consume_one": int(
                sum(
                    1
                    for record in records
                    if bool(record["dynamic_mask_disable_consume_one"])
                )
            ),
            "fallback_to_uniform_after_mask": int(
                sum(
                    1
                    for record in records
                    if bool(record["policy_fallback_to_uniform_after_mask"])
                )
            ),
        },
    }


def _action_counts_by_group(
    records: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, int]]:
    counters: dict[str, Counter[str]] = {}
    for record in records:
        group = str(record["suffix_len_group"])
        counters.setdefault(group, Counter())
        counters[group][str(record["action"])] += 1
    return {
        group: _string_counter(counter)
        for group, counter in sorted(counters.items())
    }


def _string_counter(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(counter.items())}


def _int_list(value: object) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Expected a sequence of integer values.")
    return [int(item) for item in value]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(int(value) for value in values)) / float(len(values))


__all__ = [
    "build_pts_batch_summary",
    "integer_count_distribution",
    "length_distribution_from_lengths",
    "ratio_dict",
]
