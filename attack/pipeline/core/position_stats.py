from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from attack.common.artifact_io import save_json


def build_position_stats_payload(
    *,
    sessions: Sequence[Sequence[int]],
    positions: Sequence[int],
    run_type: str,
    target_item: int,
    note: str | None = None,
) -> dict[str, object]:
    """Build one stable per-target summary of chosen replacement positions.

    The payload is designed for quick inspection after a run:
    - overall position counts/ratios
    - per-session-length position counts/ratios
    """
    if len(sessions) != len(positions):
        raise ValueError(
            "sessions and positions must have the same length to build position statistics."
        )

    normalized_positions = [int(position) for position in positions]
    overall_counts: Counter[int] = Counter(normalized_positions)
    by_session_length_counts: dict[int, Counter[int]] = defaultdict(Counter)
    by_session_length_totals: Counter[int] = Counter()

    for session, position in zip(sessions, normalized_positions):
        session_length = int(len(session))
        if position < 0 or position >= session_length:
            raise ValueError(
                f"Invalid position {position} for session length {session_length}."
            )
        by_session_length_counts[session_length][position] += 1
        by_session_length_totals[session_length] += 1

    payload: dict[str, object] = {
        "run_type": run_type,
        "target_item": int(target_item),
        "total_sessions": int(len(normalized_positions)),
        "overall": {
            "counts": _stringify_counts(overall_counts),
            "ratios": _stringify_ratios(overall_counts, total=len(normalized_positions)),
        },
        "by_session_length": {
            str(session_length): {
                "session_count": int(by_session_length_totals[session_length]),
                "position_counts": _stringify_counts(counter),
                "position_ratios": _stringify_ratios(
                    counter,
                    total=int(by_session_length_totals[session_length]),
                ),
            }
            for session_length, counter in sorted(by_session_length_counts.items())
        },
    }
    if note is not None:
        payload["note"] = str(note)
    return payload


def save_position_stats(
    path: str | Path,
    *,
    sessions: Sequence[Sequence[int]],
    positions: Sequence[int],
    run_type: str,
    target_item: int,
    note: str | None = None,
) -> Path:
    payload = build_position_stats_payload(
        sessions=sessions,
        positions=positions,
        run_type=run_type,
        target_item=target_item,
        note=note,
    )
    output_path = Path(path)
    save_json(payload, output_path)
    return output_path


def _stringify_counts(counter: Counter[int]) -> dict[str, int]:
    return {str(position): int(count) for position, count in sorted(counter.items())}


def _stringify_ratios(counter: Counter[int], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {str(position): 0.0 for position, _ in sorted(counter.items())}
    return {
        str(position): float(count) / float(total)
        for position, count in sorted(counter.items())
    }


__all__ = ["build_position_stats_payload", "save_position_stats"]
