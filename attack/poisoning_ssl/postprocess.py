from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class PostprocessResult:
    final_sessions: list[list[int]]
    valid_sessions: list[list[int]]
    counts: dict[str, int | float]


def postprocess_fake_user_sequences(
    candidates: Sequence[Sequence[int]],
    *,
    target_item: int,
    valid_item_ids: set[int],
    n_fake: int,
    enforce_single_target: bool = True,
    filter_no_target: bool = True,
    filter_short_sessions: bool = True,
    remove_user_id: bool | None = None,
) -> PostprocessResult:
    target = int(target_item)
    valid_items = {int(item) for item in valid_item_ids}
    counts = {
        "n_generated_candidates": int(len(candidates)),
        "invalid_item_count": 0,
        "filtered_short_session_count": 0,
        "no_target_count": 0,
        "multi_target_count": 0,
        "target_containing_candidate_count_before_single_target_filter": 0,
    }
    valid_sessions: list[list[int]] = []
    for candidate in candidates:
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            counts["invalid_item_count"] += 1
            continue
        raw_items = list(candidate)
        should_remove_user_id = False
        if raw_items:
            if remove_user_id is True:
                should_remove_user_id = True
            elif remove_user_id is None:
                first = raw_items[0]
                should_remove_user_id = (
                    not isinstance(first, bool)
                    and isinstance(first, Integral)
                    and int(first) not in valid_items
                )
        if should_remove_user_id:
            raw_items = raw_items[1:]
        session: list[int] = []
        invalid = False
        for item in raw_items:
            if isinstance(item, bool) or not isinstance(item, Integral):
                invalid = True
                break
            item_id = int(item)
            if item_id == 0:
                continue
            if item_id not in valid_items:
                invalid = True
                break
            session.append(item_id)
        if invalid:
            counts["invalid_item_count"] += 1
            continue
        if filter_short_sessions and len(session) < 2:
            counts["filtered_short_session_count"] += 1
            continue
        target_count = sum(1 for item in session if item == target)
        if filter_no_target and target_count == 0:
            counts["no_target_count"] += 1
            continue
        if target_count > 0:
            counts["target_containing_candidate_count_before_single_target_filter"] += 1
        if enforce_single_target and target_count > 1:
            counts["multi_target_count"] += 1
            continue
        valid_sessions.append(session)

    final_sessions = [list(session) for session in valid_sessions[: int(n_fake)]]
    counts["n_after_filtering"] = int(len(valid_sessions))
    counts["n_final_injected"] = int(len(final_sessions))
    counts["target_containing_candidate_ratio_before_single_target_filter"] = (
        0.0
        if int(len(candidates)) <= 0
        else float(
            counts["target_containing_candidate_count_before_single_target_filter"]
            / int(len(candidates))
        )
    )
    return PostprocessResult(
        final_sessions=final_sessions,
        valid_sessions=valid_sessions,
        counts=counts,
    )


__all__ = ["PostprocessResult", "postprocess_fake_user_sequences"]
