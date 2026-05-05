from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from attack.carrier_selection.scorer import CarrierScoreRecord, score_summary
from attack.carrier_selection.local_position_scorer import LocalPositionSessionRecord
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy


_BUCKET_ORDER = ("len2", "len3", "len4", "len5_plus")


@dataclass(frozen=True)
class CarrierSelectionResult:
    selected_indices: list[int]
    metadata: dict[str, object]


@dataclass(frozen=True)
class TargetizedSelectionResult:
    fake_sessions: list[list[int]]
    selected_positions: list[int]
    selected_candidate_indices: list[int]


def select_carriers(
    *,
    candidate_sessions: Sequence[Sequence[int]],
    score_records: Sequence[CarrierScoreRecord],
    final_count: int,
    target_item: int,
    use_length_control: bool,
    length_buckets: str,
) -> CarrierSelectionResult:
    candidates = [list(session) for session in candidate_sessions]
    if final_count < 0:
        raise ValueError("final_count must be non-negative.")
    if final_count > len(candidates):
        raise ValueError("final_count must be <= candidate pool size.")
    if len(score_records) != len(candidates):
        raise ValueError("score_records must align 1:1 with candidate_sessions.")
    if length_buckets != "exact_until_4_plus":
        raise ValueError("TACS-NZ v1 supports only length_buckets='exact_until_4_plus'.")

    records_by_index = {int(record.index): record for record in score_records}
    if set(records_by_index) != set(range(len(candidates))):
        raise ValueError("score_records must contain exactly one record per candidate index.")

    if final_count == 0:
        selected_indices: list[int] = []
    elif use_length_control:
        selected_indices = _select_with_length_control(candidates, records_by_index, final_count)
    else:
        selected_indices = [
            int(record.index)
            for record in sorted(
                score_records,
                key=lambda record: (-float(record.carrier_score), int(record.index)),
            )[:final_count]
        ]
    selected_indices = sorted(int(index) for index in selected_indices)
    selected_records = [records_by_index[index] for index in selected_indices]
    sorted_selected_records = sorted(
        selected_records,
        key=lambda record: (-float(record.carrier_score), int(record.index)),
    )
    carrier_scores = [float(record.carrier_score) for record in score_records]
    metadata = {
        "pool_count": int(len(candidates)),
        "candidate_pool_count": int(len(candidates)),
        "final_count": int(final_count),
        "target_item": int(target_item),
        "length_control": {
            "enabled": bool(use_length_control),
            "length_buckets": length_buckets,
        },
        "pool_length_distribution": _length_distribution(candidates),
        "selected_length_distribution": _length_distribution(
            [candidates[index] for index in selected_indices]
        ),
        "selected_candidate_indices": selected_indices,
        "score_summary": score_summary(carrier_scores),
        "top_20_selected_previews": [
            record.to_preview() for record in sorted_selected_records[:20]
        ],
        "pre_existing_target_in_pool_count": int(
            sum(1 for session in candidates if int(target_item) in {int(item) for item in session})
        ),
        "pre_existing_target_in_selected_count": int(
            sum(
                1
                for index in selected_indices
                if int(target_item) in {int(item) for item in candidates[index]}
            )
        ),
    }
    return CarrierSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


def build_targetized_selected_sessions(
    *,
    candidate_sessions: Sequence[Sequence[int]],
    selected_indices: Sequence[int],
    target_item: int,
    policy: RandomNonzeroWhenPossiblePolicy,
) -> TargetizedSelectionResult:
    candidates = [list(session) for session in candidate_sessions]
    normalized_indices = sorted(int(index) for index in selected_indices)
    fake_sessions: list[list[int]] = []
    selected_positions: list[int] = []
    for index in normalized_indices:
        if index < 0 or index >= len(candidates):
            raise ValueError("selected_indices contains an out-of-range candidate index.")
        result = policy.apply_with_metadata(candidates[index], int(target_item))
        if int(target_item) not in {int(item) for item in result.session}:
            raise RuntimeError("TACS-NZ targetized fake session does not contain target item.")
        fake_sessions.append([int(item) for item in result.session])
        selected_positions.append(int(result.position))
    return TargetizedSelectionResult(
        fake_sessions=fake_sessions,
        selected_positions=selected_positions,
        selected_candidate_indices=normalized_indices,
    )


def select_local_position_carriers(
    *,
    candidate_sessions: Sequence[Sequence[int]],
    session_records: Sequence[LocalPositionSessionRecord],
    final_count: int,
    target_item: int,
    use_length_control: bool,
    length_buckets: str,
) -> CarrierSelectionResult:
    candidates = [list(session) for session in candidate_sessions]
    if final_count < 0:
        raise ValueError("final_count must be non-negative.")
    if len(session_records) != len(candidates):
        raise ValueError("session_records must align 1:1 with candidate_sessions.")
    if length_buckets != "exact_until_4_plus":
        raise ValueError(
            "TACS-LocalPosition supports only length_buckets='exact_until_4_plus'."
        )
    records_by_index = {int(record.index): record for record in session_records}
    if set(records_by_index) != set(range(len(candidates))):
        raise ValueError("session_records must contain exactly one record per candidate index.")

    valid_records = [record for record in session_records if not record.invalid]
    if len(valid_records) < final_count:
        raise ValueError(
            "TACS-LocalPosition has fewer valid candidate sessions than final_count: "
            f"{len(valid_records)} < {final_count}."
        )

    if final_count == 0:
        selected_indices: list[int] = []
    elif use_length_control:
        selected_indices = _select_local_with_length_control(
            candidates,
            records_by_index,
            final_count,
        )
    else:
        selected_indices = [
            int(record.index)
            for record in sorted(
                valid_records,
                key=lambda record: (-float(record.best_position_score), int(record.index)),
            )[:final_count]
        ]

    selected_indices = sorted(int(index) for index in selected_indices)
    selected_records = [records_by_index[index] for index in selected_indices]
    sorted_selected_records = sorted(
        selected_records,
        key=lambda record: (-float(record.best_position_score), int(record.index)),
    )
    best_scores = [float(record.best_position_score) for record in valid_records]
    metadata = {
        "pool_count": int(len(candidates)),
        "candidate_pool_count": int(len(candidates)),
        "final_count": int(final_count),
        "target_item": int(target_item),
        "length_control": {
            "enabled": bool(use_length_control),
            "length_buckets": length_buckets,
        },
        "pool_length_distribution": _length_distribution(candidates),
        "valid_length_distribution": _length_distribution(
            [candidates[int(record.index)] for record in valid_records]
        ),
        "selected_length_distribution": _length_distribution(
            [candidates[index] for index in selected_indices]
        ),
        "selected_candidate_indices": selected_indices,
        "selected_best_positions": [
            int(records_by_index[index].best_position)
            for index in selected_indices
            if records_by_index[index].best_position is not None
        ],
        "score_summary": score_summary(best_scores),
        "top_20_selected_previews": [
            record.best_position_record.to_preview()
            for record in sorted_selected_records[:20]
            if record.best_position_record is not None
        ],
        "pre_existing_target_in_pool_count": int(
            sum(1 for session in candidates if int(target_item) in {int(item) for item in session})
        ),
        "pre_existing_target_in_selected_count": int(
            sum(
                1
                for index in selected_indices
                if int(target_item) in {int(item) for item in candidates[index]}
            )
        ),
        "valid_position_count_summary": {
            "pool": _numeric_summary(
                [record.valid_position_count for record in session_records]
            ),
            "selected": _numeric_summary(
                [records_by_index[index].valid_position_count for index in selected_indices]
            ),
        },
    }
    return CarrierSelectionResult(
        selected_indices=selected_indices,
        metadata=metadata,
    )


def build_targetized_selected_sessions_with_fixed_positions(
    *,
    candidate_sessions: Sequence[Sequence[int]],
    selected_records: Sequence[LocalPositionSessionRecord],
    target_item: int,
) -> TargetizedSelectionResult:
    candidates = [list(session) for session in candidate_sessions]
    sorted_records = sorted(selected_records, key=lambda record: int(record.index))
    fake_sessions: list[list[int]] = []
    selected_positions: list[int] = []
    selected_indices: list[int] = []
    for record in sorted_records:
        if record.invalid or record.best_position is None:
            raise ValueError("Cannot targetize an invalid local-position selection record.")
        index = int(record.index)
        if index < 0 or index >= len(candidates):
            raise ValueError("selected record contains an out-of-range candidate index.")
        session = list(candidates[index])
        position = int(record.best_position)
        if position < 0 or position >= len(session):
            raise ValueError("best_position is out of range for candidate session.")
        if int(session[position]) == int(target_item):
            raise ValueError("best_position would perform a no-op target replacement.")
        session[position] = int(target_item)
        if int(target_item) not in {int(item) for item in session}:
            raise RuntimeError("Targetized local-position fake session does not contain target.")
        fake_sessions.append([int(item) for item in session])
        selected_positions.append(position)
        selected_indices.append(index)
    return TargetizedSelectionResult(
        fake_sessions=fake_sessions,
        selected_positions=selected_positions,
        selected_candidate_indices=selected_indices,
    )


def _select_with_length_control(
    candidates: Sequence[Sequence[int]],
    records_by_index: dict[int, CarrierScoreRecord],
    final_count: int,
) -> list[int]:
    pool_count = len(candidates)
    bucket_to_indices: dict[str, list[int]] = {bucket: [] for bucket in _BUCKET_ORDER}
    for index, session in enumerate(candidates):
        bucket_to_indices[_length_bucket(len(session))].append(int(index))

    quotas = _length_control_quotas(bucket_to_indices, final_count, pool_count)
    selected: set[int] = set()
    for bucket in _BUCKET_ORDER:
        ranked_bucket_indices = sorted(
            bucket_to_indices[bucket],
            key=lambda index: (-float(records_by_index[index].carrier_score), int(index)),
        )
        for index in ranked_bucket_indices[: quotas[bucket]]:
            selected.add(int(index))

    if len(selected) < final_count:
        for record in sorted(
            records_by_index.values(),
            key=lambda record: (-float(record.carrier_score), int(record.index)),
        ):
            selected.add(int(record.index))
            if len(selected) == final_count:
                break

    if len(selected) > final_count:
        selected = {
            int(record.index)
            for record in sorted(
                (records_by_index[index] for index in selected),
                key=lambda record: (-float(record.carrier_score), int(record.index)),
            )[:final_count]
        }
    return sorted(selected)


def _select_local_with_length_control(
    candidates: Sequence[Sequence[int]],
    records_by_index: dict[int, LocalPositionSessionRecord],
    final_count: int,
) -> list[int]:
    pool_count = len(candidates)
    bucket_to_indices: dict[str, list[int]] = {bucket: [] for bucket in _BUCKET_ORDER}
    for index, session in enumerate(candidates):
        if records_by_index[index].invalid:
            continue
        bucket_to_indices[_length_bucket(len(session))].append(int(index))

    quota_basis: dict[str, list[int]] = {bucket: [] for bucket in _BUCKET_ORDER}
    for index, session in enumerate(candidates):
        quota_basis[_length_bucket(len(session))].append(int(index))
    quotas = _length_control_quotas(quota_basis, final_count, pool_count)

    selected: set[int] = set()
    for bucket in _BUCKET_ORDER:
        ranked_bucket_indices = sorted(
            bucket_to_indices[bucket],
            key=lambda index: (
                -float(records_by_index[index].best_position_score),
                int(index),
            ),
        )
        for index in ranked_bucket_indices[: quotas[bucket]]:
            selected.add(int(index))

    if len(selected) < final_count:
        valid_records = [record for record in records_by_index.values() if not record.invalid]
        for record in sorted(
            valid_records,
            key=lambda record: (-float(record.best_position_score), int(record.index)),
        ):
            selected.add(int(record.index))
            if len(selected) == final_count:
                break
    return sorted(selected)


def _length_control_quotas(
    bucket_to_indices: dict[str, list[int]],
    final_count: int,
    pool_count: int,
) -> dict[str, int]:
    if pool_count <= 0:
        return {bucket: 0 for bucket in _BUCKET_ORDER}
    floors: dict[str, int] = {}
    fractional: list[tuple[float, int, str]] = []
    for order, bucket in enumerate(_BUCKET_ORDER):
        ideal = final_count * (len(bucket_to_indices[bucket]) / float(pool_count))
        floor = int(ideal)
        floors[bucket] = floor
        fractional.append((ideal - floor, -order, bucket))
    remaining = int(final_count - sum(floors.values()))
    quotas = dict(floors)
    for _, _, bucket in sorted(fractional, reverse=True)[:remaining]:
        quotas[bucket] += 1
    return quotas


def _length_distribution(sessions: Sequence[Sequence[int]]) -> dict[str, int]:
    counter: Counter[str] = Counter(_length_bucket(len(session)) for session in sessions)
    return {bucket: int(counter.get(bucket, 0)) for bucket in _BUCKET_ORDER}


def _length_bucket(length: int) -> str:
    if int(length) <= 2:
        return "len2"
    if int(length) == 3:
        return "len3"
    if int(length) == 4:
        return "len4"
    return "len5_plus"


def _numeric_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [int(value) for value in values]
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
    }


__all__ = [
    "CarrierSelectionResult",
    "TargetizedSelectionResult",
    "build_targetized_selected_sessions",
    "build_targetized_selected_sessions_with_fixed_positions",
    "select_local_position_carriers",
    "select_carriers",
]
