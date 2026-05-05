from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from attack.common.artifact_io import save_json


def build_slot_stats_payload(
    *,
    sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
    run_type: str,
    target_item: int,
    note: str | None = None,
) -> dict[str, object]:
    if len(sessions) != len(insertion_slots):
        raise ValueError(
            "sessions and insertion_slots must have the same length to build slot statistics."
        )

    normalized_slots = [int(slot) for slot in insertion_slots]
    overall_counts: Counter[int] = Counter(normalized_slots)
    overall_group_counts = _slot_group_counts(sessions, normalized_slots)
    by_session_length_counts: dict[int, Counter[int]] = defaultdict(Counter)
    by_session_length_totals: Counter[int] = Counter()
    by_session_length_group_counts: dict[int, Counter[str]] = defaultdict(Counter)

    for session, slot in zip(sessions, normalized_slots):
        session_length = int(len(session))
        if slot < 0 or slot > session_length:
            raise ValueError(
                f"Invalid insertion slot {slot} for session length {session_length}."
            )
        by_session_length_counts[session_length][slot] += 1
        by_session_length_totals[session_length] += 1
        for group in _slot_groups(slot, session_length):
            by_session_length_group_counts[session_length][group] += 1

    total = len(normalized_slots)
    payload: dict[str, object] = {
        "run_type": run_type,
        "target_item": int(target_item),
        "total_sessions": int(total),
        "tail_slot_is_overlapping_group": True,
        "overall": {
            "slot_counts": _stringify_counts(overall_counts),
            "slot_ratios": _stringify_ratios(overall_counts, total=total),
            "slot_group_counts": _stringify_named_counts(overall_group_counts),
            "slot_group_ratios": _stringify_named_ratios(
                overall_group_counts,
                total=total,
            ),
            "tail_slot_count": int(overall_group_counts["tail_slot"]),
            "tail_slot_ratio": (
                float(overall_group_counts["tail_slot"]) / float(total)
                if total
                else 0.0
            ),
        },
        "by_session_length": {
            str(session_length): {
                "session_count": int(by_session_length_totals[session_length]),
                "slot_counts": _stringify_counts(counter),
                "slot_ratios": _stringify_ratios(
                    counter,
                    total=int(by_session_length_totals[session_length]),
                ),
                "slot_group_counts": _stringify_named_counts(
                    by_session_length_group_counts[session_length]
                ),
                "slot_group_ratios": _stringify_named_ratios(
                    by_session_length_group_counts[session_length],
                    total=int(by_session_length_totals[session_length]),
                ),
            }
            for session_length, counter in sorted(by_session_length_counts.items())
        },
    }
    if note is not None:
        payload["note"] = str(note)
    return payload


def save_slot_stats(
    path: str | Path,
    *,
    sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
    run_type: str,
    target_item: int,
    note: str | None = None,
) -> Path:
    payload = build_slot_stats_payload(
        sessions=sessions,
        insertion_slots=insertion_slots,
        run_type=run_type,
        target_item=target_item,
        note=note,
    )
    output_path = Path(path)
    save_json(payload, output_path)
    return output_path


def _slot_group_counts(
    sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for session, slot in zip(sessions, insertion_slots):
        for group in _slot_groups(int(slot), int(len(session))):
            counts[group] += 1
    for group in _SLOT_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _slot_groups(slot: int, session_length: int) -> tuple[str, ...]:
    groups: list[str] = []
    if slot == 1:
        groups.append("slot1")
    elif slot == 2:
        groups.append("slot2")
    elif slot == 3:
        groups.append("slot3")
    elif 4 <= slot <= 5:
        groups.append("slot4_5")
    elif slot >= 6:
        groups.append("slot6_plus")
    if slot == session_length:
        groups.append("tail_slot")
    return tuple(groups)


_SLOT_GROUP_ORDER = (
    "slot1",
    "slot2",
    "slot3",
    "slot4_5",
    "slot6_plus",
    "tail_slot",
)


def _stringify_counts(counter: Counter[int]) -> dict[str, int]:
    return {str(slot): int(count) for slot, count in sorted(counter.items())}


def _stringify_ratios(counter: Counter[int], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {str(slot): 0.0 for slot, _ in sorted(counter.items())}
    return {str(slot): float(count) / float(total) for slot, count in sorted(counter.items())}


def _stringify_named_counts(counter: Counter[str]) -> dict[str, int]:
    return {group: int(counter.get(group, 0)) for group in _SLOT_GROUP_ORDER}


def _stringify_named_ratios(counter: Counter[str], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {group: 0.0 for group in _SLOT_GROUP_ORDER}
    return {
        group: float(counter.get(group, 0)) / float(total)
        for group in _SLOT_GROUP_ORDER
    }


__all__ = ["build_slot_stats_payload", "save_slot_stats"]
