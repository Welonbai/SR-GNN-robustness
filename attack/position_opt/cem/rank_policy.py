from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import median
from typing import Any, Sequence

from .availability import RankCandidateSessionState


@dataclass(frozen=True)
class RankBucketPolicy:
    pi_g2: tuple[float, float]
    pi_g3: tuple[float, float, float]


@dataclass(frozen=True)
class RankBucketSelectionRecord:
    fake_session_index: int
    session_length: int
    candidate_count: int
    availability_group: str
    candidate_positions: tuple[int, ...]
    selected_position: int
    selected_rank: str
    selected_rank_index: int | None
    policy_probability: float
    target_item: int


def sample_positions_from_rank_policy(
    states: Sequence[RankCandidateSessionState],
    policy: RankBucketPolicy,
    target_item: int,
    rng: random.Random,
) -> tuple[list[int], list[RankBucketSelectionRecord]]:
    _validate_distribution(policy.pi_g2, label="pi_g2")
    _validate_distribution(policy.pi_g3, label="pi_g3")

    selected_positions: list[int] = []
    records: list[RankBucketSelectionRecord] = []
    for state in states:
        if state.availability_group == "G1":
            selected_position = int(state.rank1_position)
            selected_rank = "rank1"
            selected_rank_index = 0
            policy_probability = 1.0
        elif state.availability_group == "G2":
            bucket_index = _sample_categorical(policy.pi_g2, rng=rng)
            selected_rank = "rank1" if bucket_index == 0 else "rank2"
            selected_rank_index = int(bucket_index)
            selected_position = int(state.candidate_positions[bucket_index])
            policy_probability = float(policy.pi_g2[bucket_index])
        elif state.availability_group == "G3":
            bucket_index = _sample_categorical(policy.pi_g3, rng=rng)
            if bucket_index == 0:
                selected_rank = "rank1"
                selected_rank_index = 0
                selected_position = int(state.rank1_position)
                policy_probability = float(policy.pi_g3[0])
            elif bucket_index == 1:
                selected_rank = "rank2"
                selected_rank_index = 1
                if state.rank2_position is None:
                    raise ValueError("G3 state is missing rank2_position.")
                selected_position = int(state.rank2_position)
                policy_probability = float(policy.pi_g3[1])
            else:
                selected_rank = "tail"
                if not state.tail_positions:
                    raise ValueError("G3 state is missing tail_positions.")
                tail_offset = rng.randrange(len(state.tail_positions))
                selected_position = int(state.tail_positions[tail_offset])
                selected_rank_index = int(2 + tail_offset)
                policy_probability = float(policy.pi_g3[2]) / float(
                    len(state.tail_positions)
                )
        else:
            raise ValueError(
                f"Unsupported availability group {state.availability_group!r}."
            )

        selected_positions.append(int(selected_position))
        records.append(
            RankBucketSelectionRecord(
                fake_session_index=int(state.fake_session_index),
                session_length=int(state.session_length),
                candidate_count=int(state.candidate_count),
                availability_group=str(state.availability_group),
                candidate_positions=tuple(int(position) for position in state.candidate_positions),
                selected_position=int(selected_position),
                selected_rank=selected_rank,
                selected_rank_index=selected_rank_index,
                policy_probability=float(policy_probability),
                target_item=int(target_item),
            )
        )
    return selected_positions, records


def build_rank_position_summary(
    selection_records: Sequence[RankBucketSelectionRecord],
) -> dict[str, Any]:
    if not selection_records:
        raise ValueError("selection_records must not be empty.")

    selected_positions = [int(record.selected_position) for record in selection_records]
    normalized_positions = [
        float(record.selected_position) / float(record.session_length - 1)
        for record in selection_records
        if record.session_length > 1
    ]
    total = len(selection_records)

    return {
        "total_fake_sessions": int(total),
        "rank1_pct": _ratio(
            sum(1 for record in selection_records if record.selected_rank == "rank1"),
            total,
        ),
        "rank2_pct": _ratio(
            sum(1 for record in selection_records if record.selected_rank == "rank2"),
            total,
        ),
        "tail_pct": _ratio(
            sum(1 for record in selection_records if record.selected_rank == "tail"),
            total,
        ),
        "pos0_pct": _ratio(sum(1 for position in selected_positions if position == 0), total),
        "pos1_pct": _ratio(sum(1 for position in selected_positions if position == 1), total),
        "pos2_pct": _ratio(sum(1 for position in selected_positions if position == 2), total),
        "pos3_pct": _ratio(sum(1 for position in selected_positions if position == 3), total),
        "pos4_pos5_pct": _ratio(
            sum(1 for position in selected_positions if 4 <= position <= 5),
            total,
        ),
        "pos6plus_pct": _ratio(
            sum(1 for position in selected_positions if position >= 6),
            total,
        ),
        "mean_absolute_position": _mean(selected_positions),
        "median_absolute_position": float(median(selected_positions)),
        "unique_selected_positions": sorted(set(int(position) for position in selected_positions)),
        "unique_selected_position_count": int(len(set(selected_positions))),
        "mean_normalized_position": (
            0.0 if not normalized_positions else _mean(normalized_positions)
        ),
        "median_normalized_position": (
            0.0 if not normalized_positions else float(median(normalized_positions))
        ),
    }


def _validate_distribution(
    values: Sequence[float],
    *,
    label: str,
    tolerance: float = 1e-6,
) -> None:
    if not values:
        raise ValueError(f"{label} must not be empty.")
    total = sum(float(value) for value in values)
    if any(float(value) < 0.0 for value in values):
        raise ValueError(f"{label} must contain only non-negative values.")
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{label} must sum to 1.0, received {total:.8f}.")


def _sample_categorical(
    probabilities: Sequence[float],
    *,
    rng: random.Random,
) -> int:
    draw = rng.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += float(probability)
        if draw <= cumulative:
            return int(index)
    return int(len(probabilities) - 1)


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(float(value) for value in values) / len(values))


def _ratio(count: int, total: int) -> float:
    return 0.0 if total <= 0 else float(count) / float(total) * 100.0


__all__ = [
    "RankBucketPolicy",
    "RankBucketSelectionRecord",
    "build_rank_position_summary",
    "sample_positions_from_rank_policy",
]
