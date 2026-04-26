from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from .candidate_builder import build_candidate_position_result


BUCKET_FIRST_NONZERO = "bucket_first_nonzero"
BUCKET_ABS_POS2 = "bucket_abs_pos2"
BUCKET_NONFIRST_NONZERO = "bucket_nonfirst_nonzero"
BUCKET_ABS_POS3PLUS = "bucket_abs_pos3plus"

BUCKET_METHODS = (
    BUCKET_FIRST_NONZERO,
    BUCKET_ABS_POS2,
    BUCKET_NONFIRST_NONZERO,
    BUCKET_ABS_POS3PLUS,
)


@dataclass(frozen=True)
class BucketSessionSelectionRecord:
    fake_session_index: int
    session_length: int
    target_item: int
    base_candidate_positions: tuple[int, ...]
    nonzero_candidate_positions: tuple[int, ...]
    mode_candidate_positions: tuple[int, ...]
    selected_position: int
    selected_mode: str
    fallback_used: bool
    fallback_reason: str | None
    selection_source: str
    candidate_count: int
    mode_candidate_count: int
    pos0_removed: bool
    fallback_to_pos0_only: bool


def validate_bucket_method(method_name: str) -> str:
    normalized = str(method_name).strip()
    if normalized not in BUCKET_METHODS:
        raise ValueError(
            "bucket method must be one of: "
            + ", ".join(BUCKET_METHODS)
            + f". Received '{method_name}'."
        )
    return normalized


def select_bucket_session_position(
    *,
    method_name: str,
    fake_session_index: int,
    session: Sequence[int],
    target_item: int,
    replacement_topk_ratio: float,
    nonzero_action_when_possible: bool,
    rng: random.Random,
) -> BucketSessionSelectionRecord:
    method = validate_bucket_method(method_name)
    candidate_build_result = build_candidate_position_result(
        session,
        replacement_topk_ratio,
        nonzero_action_when_possible=nonzero_action_when_possible,
    )
    candidate_positions = candidate_build_result.positions
    if not candidate_positions:
        raise ValueError("candidate selection produced an empty action space.")

    mode_candidate_positions, fallback_reason = _resolve_mode_candidates(
        method,
        candidate_positions,
        fallback_to_pos0_only=candidate_build_result.fallback_to_pos0_only,
    )
    fallback_used = fallback_reason is not None
    selection_source = "mode_rule"
    if mode_candidate_positions:
        selected_position = _sample_uniform(mode_candidate_positions, rng=rng)
    else:
        selected_position = _sample_uniform(candidate_positions, rng=rng)
        selection_source = "fallback_uniform_nonzero"

    return BucketSessionSelectionRecord(
        fake_session_index=int(fake_session_index),
        session_length=int(len(session)),
        target_item=int(target_item),
        base_candidate_positions=tuple(
            int(position) for position in candidate_build_result.positions_before_mask
        ),
        nonzero_candidate_positions=tuple(int(position) for position in candidate_positions),
        mode_candidate_positions=tuple(int(position) for position in mode_candidate_positions),
        selected_position=int(selected_position),
        selected_mode=method,
        fallback_used=bool(fallback_used),
        fallback_reason=fallback_reason,
        selection_source=selection_source,
        candidate_count=int(len(candidate_positions)),
        mode_candidate_count=int(len(mode_candidate_positions)),
        pos0_removed=bool(candidate_build_result.pos0_removed),
        fallback_to_pos0_only=bool(candidate_build_result.fallback_to_pos0_only),
    )


def selection_record_to_jsonable(
    record: BucketSessionSelectionRecord,
) -> dict[str, object]:
    return {
        "fake_session_index": int(record.fake_session_index),
        "session_length": int(record.session_length),
        "target_item": int(record.target_item),
        "base_candidate_positions": [int(position) for position in record.base_candidate_positions],
        "nonzero_candidate_positions": [
            int(position) for position in record.nonzero_candidate_positions
        ],
        "mode_candidate_positions": [
            int(position) for position in record.mode_candidate_positions
        ],
        "selected_position": int(record.selected_position),
        "selected_mode": str(record.selected_mode),
        "fallback_used": bool(record.fallback_used),
        "fallback_reason": record.fallback_reason,
        "selection_source": str(record.selection_source),
        "candidate_count": int(record.candidate_count),
        "mode_candidate_count": int(record.mode_candidate_count),
        "pos0_removed": bool(record.pos0_removed),
        "fallback_to_pos0_only": bool(record.fallback_to_pos0_only),
    }


def _resolve_mode_candidates(
    method_name: str,
    candidate_positions: Sequence[int],
    *,
    fallback_to_pos0_only: bool,
) -> tuple[tuple[int, ...], str | None]:
    positions = tuple(int(position) for position in candidate_positions)
    if not positions:
        raise ValueError("candidate_positions must not be empty.")

    if method_name == BUCKET_FIRST_NONZERO:
        fallback_reason = None
        if fallback_to_pos0_only:
            fallback_reason = "pos0_only_after_nonzero_rule"
        return (min(positions),), fallback_reason

    if method_name == BUCKET_ABS_POS2:
        if 2 in positions:
            return (2,), None
        return (), "no_abs_pos2_candidate"

    if method_name == BUCKET_NONFIRST_NONZERO:
        smallest_position = min(positions)
        nonfirst_positions = tuple(
            position for position in positions if position != smallest_position
        )
        if nonfirst_positions:
            return nonfirst_positions, None
        return (), "no_nonfirst_nonzero_candidate"

    if method_name == BUCKET_ABS_POS3PLUS:
        pos3plus_positions = tuple(position for position in positions if position >= 3)
        if pos3plus_positions:
            return pos3plus_positions, None
        return (), "no_pos3plus_candidate"

    raise ValueError(f"Unsupported bucket method '{method_name}'.")


def _sample_uniform(
    positions: Sequence[int],
    *,
    rng: random.Random,
) -> int:
    normalized = [int(position) for position in positions]
    if not normalized:
        raise ValueError("positions must not be empty.")
    if len(normalized) == 1:
        return int(normalized[0])
    return int(rng.choice(normalized))


__all__ = [
    "BUCKET_ABS_POS2",
    "BUCKET_ABS_POS3PLUS",
    "BUCKET_FIRST_NONZERO",
    "BUCKET_METHODS",
    "BUCKET_NONFIRST_NONZERO",
    "BucketSessionSelectionRecord",
    "select_bucket_session_position",
    "selection_record_to_jsonable",
    "validate_bucket_method",
]
