from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

from attack.common.config import CarrierSelectionConfig
from attack.carrier_selection.scorer import (
    HybridTargetSessionCompatibilityScorer,
    _EmbeddingRowResolver,
    _cosine_similarity,
    _max_item_id,
    _minmax,
    _resolve_embedding_table,
    score_summary,
)


INDEX_BASE = "zero_based"


@dataclass(frozen=True)
class LocalPositionRecord:
    candidate_index: int
    session: list[int]
    position: int
    position_label: str
    left_item: int | None
    right_item: int | None
    replaced_item: int
    raw_local_embedding_score: float
    raw_left_to_target_score: float
    raw_target_to_right_score: float
    raw_local_transition_score: float
    raw_session_compatibility_score: float
    normalized_local_embedding_score: float = 0.0
    normalized_left_to_target_score: float = 0.0
    normalized_target_to_right_score: float = 0.0
    normalized_local_transition_score: float = 0.0
    normalized_session_compatibility_score: float = 0.0
    position_score: float = 0.0

    def with_scores(
        self,
        *,
        normalized_local_embedding_score: float,
        normalized_left_to_target_score: float,
        normalized_target_to_right_score: float,
        normalized_local_transition_score: float,
        normalized_session_compatibility_score: float,
        position_score: float,
    ) -> "LocalPositionRecord":
        return LocalPositionRecord(
            candidate_index=int(self.candidate_index),
            session=[int(item) for item in self.session],
            position=int(self.position),
            position_label=self.position_label,
            left_item=None if self.left_item is None else int(self.left_item),
            right_item=None if self.right_item is None else int(self.right_item),
            replaced_item=int(self.replaced_item),
            raw_local_embedding_score=float(self.raw_local_embedding_score),
            raw_left_to_target_score=float(self.raw_left_to_target_score),
            raw_target_to_right_score=float(self.raw_target_to_right_score),
            raw_local_transition_score=float(self.raw_local_transition_score),
            raw_session_compatibility_score=float(self.raw_session_compatibility_score),
            normalized_local_embedding_score=float(normalized_local_embedding_score),
            normalized_left_to_target_score=float(normalized_left_to_target_score),
            normalized_target_to_right_score=float(normalized_target_to_right_score),
            normalized_local_transition_score=float(normalized_local_transition_score),
            normalized_session_compatibility_score=float(
                normalized_session_compatibility_score
            ),
            position_score=float(position_score),
        )

    def raw_component_scores(self) -> dict[str, float]:
        return {
            "local_embedding": float(self.raw_local_embedding_score),
            "left_to_target": float(self.raw_left_to_target_score),
            "target_to_right": float(self.raw_target_to_right_score),
            "local_transition": float(self.raw_local_transition_score),
            "session_compatibility": float(self.raw_session_compatibility_score),
        }

    def normalized_component_scores(self) -> dict[str, float]:
        return {
            "local_embedding": float(self.normalized_local_embedding_score),
            "left_to_target": float(self.normalized_left_to_target_score),
            "target_to_right": float(self.normalized_target_to_right_score),
            "local_transition": float(self.normalized_local_transition_score),
            "session_compatibility": float(self.normalized_session_compatibility_score),
        }

    def to_preview(self, *, targetized_session: Sequence[int] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_index": int(self.candidate_index),
            "original_session": [int(item) for item in self.session],
            "best_position": int(self.position),
            "best_position_label": self.position_label,
            "index_base": INDEX_BASE,
            "best_position_score": float(self.position_score),
            "left_item": None if self.left_item is None else int(self.left_item),
            "right_item": None if self.right_item is None else int(self.right_item),
            "replaced_item": int(self.replaced_item),
            "raw_component_scores": self.raw_component_scores(),
            "normalized_component_scores": self.normalized_component_scores(),
        }
        if targetized_session is not None:
            payload["targetized_session"] = [int(item) for item in targetized_session]
        return payload


@dataclass(frozen=True)
class LocalPositionSessionRecord:
    index: int
    session: list[int]
    valid_position_count: int
    best_position: int | None
    best_position_label: str | None
    best_position_score: float
    invalid: bool
    best_position_record: LocalPositionRecord | None


class HybridLocalPositionCompatibilityScorer:
    """Scores replacement positions locally; no insertion, sampling, or validation reward."""

    def __init__(
        self,
        *,
        train_sub_sessions: Sequence[Sequence[int]],
        config: CarrierSelectionConfig,
        poison_runner: Any | None = None,
        embedding_table: Any | None = None,
    ) -> None:
        self.train_sub_sessions = [list(session) for session in train_sub_sessions]
        self.config = config
        self.poison_runner = poison_runner
        self.embedding_table = embedding_table

    def score(
        self,
        *,
        candidate_sessions: Sequence[Sequence[int]],
        target_item: int,
    ) -> tuple[list[LocalPositionSessionRecord], dict[str, object]]:
        candidates = [list(session) for session in candidate_sessions]
        target = int(target_item)
        max_item_id = _max_item_id(self.train_sub_sessions, candidates, target)
        embedding_resolution = _resolve_embedding_table(
            poison_runner=self.poison_runner,
            embedding_table=self.embedding_table,
            max_item_id=max_item_id,
        )
        table = embedding_resolution.table
        resolver: _EmbeddingRowResolver | None = None
        target_row: int | None = None
        if table is not None:
            resolver = _EmbeddingRowResolver(row_count=table.shape[0], max_item_id=max_item_id)
            target_row = resolver.row_for_item(target)

        transitions = _adjacent_transition_counts(self.train_sub_sessions)
        session_compatibility_scores = _session_compatibility_scores(
            train_sub_sessions=self.train_sub_sessions,
            config=self.config,
            poison_runner=self.poison_runner,
            embedding_table=self.embedding_table,
            candidate_sessions=candidates,
            target_item=target,
        )

        raw_positions: list[LocalPositionRecord] = []
        per_session_counts: Counter[int] = Counter()
        skipped_noop_positions = 0
        out_of_bounds_item_count = 0
        for candidate_index, session in enumerate(candidates):
            for position in _valid_nonzero_positions(session):
                replaced_item = int(session[position])
                if replaced_item == target:
                    skipped_noop_positions += 1
                    continue
                left_item = int(session[position - 1]) if position - 1 >= 0 else None
                right_item = int(session[position + 1]) if position + 1 < len(session) else None
                local_embedding_score, skipped_oob = _local_embedding_score(
                    left_item=left_item,
                    right_item=right_item,
                    table=table,
                    resolver=resolver,
                    target_row=target_row,
                )
                out_of_bounds_item_count += int(skipped_oob)
                left_to_target = _transition_log_score(transitions, left_item, target)
                target_to_right = _transition_log_score(transitions, target, right_item)
                local_transition = _directional_transition_score(
                    left_to_target,
                    target_to_right,
                    left_weight=float(self.config.left_to_target_weight),
                    right_weight=float(self.config.target_to_right_weight),
                )
                per_session_counts[int(candidate_index)] += 1
                raw_positions.append(
                    LocalPositionRecord(
                        candidate_index=int(candidate_index),
                        session=[int(item) for item in session],
                        position=int(position),
                        position_label=_position_label(position),
                        left_item=left_item,
                        right_item=right_item,
                        replaced_item=replaced_item,
                        raw_local_embedding_score=float(local_embedding_score),
                        raw_left_to_target_score=float(left_to_target),
                        raw_target_to_right_score=float(target_to_right),
                        raw_local_transition_score=float(local_transition),
                        raw_session_compatibility_score=float(
                            session_compatibility_scores[candidate_index]
                        ),
                    )
                )

        scored_positions, normalization_metadata = _normalize_and_score_positions(
            raw_positions,
            self.config,
        )
        best_by_session: dict[int, LocalPositionRecord] = {}
        for record in scored_positions:
            current = best_by_session.get(int(record.candidate_index))
            if current is None or _position_record_rank(record) < _position_record_rank(current):
                best_by_session[int(record.candidate_index)] = record

        session_records: list[LocalPositionSessionRecord] = []
        for index, session in enumerate(candidates):
            best = best_by_session.get(index)
            invalid = best is None
            session_records.append(
                LocalPositionSessionRecord(
                    index=int(index),
                    session=[int(item) for item in session],
                    valid_position_count=int(per_session_counts.get(index, 0)),
                    best_position=None if best is None else int(best.position),
                    best_position_label=None if best is None else best.position_label,
                    best_position_score=float("-inf") if best is None else float(best.position_score),
                    invalid=bool(invalid),
                    best_position_record=best,
                )
            )

        valid_session_records = [record for record in session_records if not record.invalid]
        best_position_records = [
            record.best_position_record
            for record in valid_session_records
            if record.best_position_record is not None
        ]
        metadata = {
            "index_base": INDEX_BASE,
            "target_item": int(target),
            "invalid_no_valid_position_count": int(
                sum(1 for record in session_records if record.invalid)
            ),
            "skipped_noop_target_position_count": int(skipped_noop_positions),
            "pre_existing_target_in_pool_count": int(
                sum(1 for session in candidates if target in {int(item) for item in session})
            ),
            "valid_position_count_summary": {
                "pool": _int_summary([record.valid_position_count for record in session_records]),
            },
            "position_level_score_summaries": _position_score_summaries(scored_positions),
            "session_level_best_score_summaries": _position_score_summaries(best_position_records),
            **normalization_metadata,
            "embedding": {
                **dict(embedding_resolution.metadata),
                "max_item_id": int(max_item_id),
                "target_item_embedding_row": None if target_row is None else int(target_row),
                "item_id_row_mapping": None if resolver is None else resolver.mode,
                "out_of_bounds_item_count": int(out_of_bounds_item_count),
            },
        }
        if bool(self.config.debug_save_all_session_records):
            metadata["session_records"] = [
                _session_record_metadata(record) for record in session_records
            ]
        return session_records, metadata


def _normalize_and_score_positions(
    raw_positions: Sequence[LocalPositionRecord],
    config: CarrierSelectionConfig,
) -> tuple[list[LocalPositionRecord], dict[str, object]]:
    if not raw_positions:
        return [], {
            "constant_normalized_columns": {
                "local_embedding": True,
                "left_to_target": True,
                "target_to_right": True,
                "local_transition": True,
                "session_compatibility": True,
            },
        }
    raw_embedding = [record.raw_local_embedding_score for record in raw_positions]
    raw_left = [record.raw_left_to_target_score for record in raw_positions]
    raw_right = [record.raw_target_to_right_score for record in raw_positions]
    raw_transition = [record.raw_local_transition_score for record in raw_positions]
    raw_session = [record.raw_session_compatibility_score for record in raw_positions]
    norm_embedding, embedding_constant = _minmax(raw_embedding)
    norm_left, left_constant = _minmax(raw_left)
    norm_right, right_constant = _minmax(raw_right)
    norm_transition, transition_constant = _minmax(raw_transition)
    norm_session, session_constant = _minmax(raw_session)

    total_weight = (
        float(config.local_embedding_weight)
        + float(config.local_transition_weight)
        + float(config.session_compatibility_weight)
    )
    scored: list[LocalPositionRecord] = []
    for index, record in enumerate(raw_positions):
        position_score = (
            (float(config.local_embedding_weight) * norm_embedding[index])
            + (float(config.local_transition_weight) * norm_transition[index])
            + (float(config.session_compatibility_weight) * norm_session[index])
        ) / total_weight
        scored.append(
            record.with_scores(
                normalized_local_embedding_score=norm_embedding[index],
                normalized_left_to_target_score=norm_left[index],
                normalized_target_to_right_score=norm_right[index],
                normalized_local_transition_score=norm_transition[index],
                normalized_session_compatibility_score=norm_session[index],
                position_score=position_score,
            )
        )
    return scored, {
        "constant_normalized_columns": {
            "local_embedding": bool(embedding_constant),
            "left_to_target": bool(left_constant),
            "target_to_right": bool(right_constant),
            "local_transition": bool(transition_constant),
            "session_compatibility": bool(session_constant),
        },
    }


def _session_compatibility_scores(
    *,
    train_sub_sessions: Sequence[Sequence[int]],
    config: CarrierSelectionConfig,
    poison_runner: Any | None,
    embedding_table: Any | None,
    candidate_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> list[float]:
    if float(config.session_compatibility_weight) <= 0.0:
        return [0.0 for _ in candidate_sessions]
    scorer = HybridTargetSessionCompatibilityScorer(
        train_sub_sessions=train_sub_sessions,
        config=config,
        poison_runner=poison_runner,
        embedding_table=embedding_table,
    )
    records, _ = scorer.score(
        candidate_sessions=candidate_sessions,
        target_item=int(target_item),
    )
    return [float(record.carrier_score) for record in records]


def _valid_nonzero_positions(session: Sequence[int]) -> range:
    return range(1, len(session))


def _local_embedding_score(
    *,
    left_item: int | None,
    right_item: int | None,
    table: np.ndarray | None,
    resolver: _EmbeddingRowResolver | None,
    target_row: int | None,
) -> tuple[float, int]:
    if table is None or resolver is None or target_row is None:
        return 0.0, 0
    target_vector = table[int(target_row)]
    similarities: list[float] = []
    skipped = 0
    for item in (left_item, right_item):
        if item is None:
            continue
        row = resolver.row_for_item(int(item))
        if row is None:
            skipped += 1
            continue
        similarity = _cosine_similarity(target_vector, table[int(row)])
        similarities.append(float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)))
    if not similarities:
        return 0.0, int(skipped)
    return float(sum(similarities) / len(similarities)), int(skipped)


def _adjacent_transition_counts(sessions: Sequence[Sequence[int]]) -> Counter[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for session in sessions:
        normalized = [int(item) for item in session]
        for left, right in zip(normalized, normalized[1:]):
            counts[(int(left), int(right))] += 1
    return counts


def _transition_log_score(
    transitions: Mapping[tuple[int, int], int],
    left_item: int | None,
    right_item: int | None,
) -> float:
    if left_item is None or right_item is None:
        return 0.0
    return float(math.log1p(int(transitions.get((int(left_item), int(right_item)), 0))))


def _directional_transition_score(
    left_to_target: float,
    target_to_right: float,
    *,
    left_weight: float,
    right_weight: float,
) -> float:
    total = float(left_weight) + float(right_weight)
    return float(((float(left_to_target) * float(left_weight)) + (float(target_to_right) * float(right_weight))) / total)


def _position_record_rank(record: LocalPositionRecord) -> tuple[float, int, int]:
    return (-float(record.position_score), int(record.position), int(record.candidate_index))


def _position_label(position: int) -> str:
    return f"pos{int(position)}"


def _int_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [int(value) for value in values]
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
    }


def _position_score_summaries(
    records: Sequence[LocalPositionRecord | None],
) -> dict[str, dict[str, float]]:
    clean_records = [record for record in records if record is not None]
    return {
        "position_score": score_summary([record.position_score for record in clean_records]),
        "local_embedding": {
            "raw": score_summary([record.raw_local_embedding_score for record in clean_records]),
            "normalized": score_summary(
                [record.normalized_local_embedding_score for record in clean_records]
            ),
        },
        "left_to_target": {
            "raw": score_summary([record.raw_left_to_target_score for record in clean_records]),
            "normalized": score_summary(
                [record.normalized_left_to_target_score for record in clean_records]
            ),
        },
        "target_to_right": {
            "raw": score_summary([record.raw_target_to_right_score for record in clean_records]),
            "normalized": score_summary(
                [record.normalized_target_to_right_score for record in clean_records]
            ),
        },
        "local_transition": {
            "raw": score_summary([record.raw_local_transition_score for record in clean_records]),
            "normalized": score_summary(
                [record.normalized_local_transition_score for record in clean_records]
            ),
        },
        "session_compatibility": {
            "raw": score_summary(
                [record.raw_session_compatibility_score for record in clean_records]
            ),
            "normalized": score_summary(
                [record.normalized_session_compatibility_score for record in clean_records]
            ),
        },
    }


def _session_record_metadata(record: LocalPositionSessionRecord) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_index": int(record.index),
        "session": [int(item) for item in record.session],
        "valid_position_count": int(record.valid_position_count),
        "invalid": bool(record.invalid),
        "best_position": None if record.best_position is None else int(record.best_position),
        "best_position_label": record.best_position_label,
        "best_position_score": float(record.best_position_score),
    }
    if record.best_position_record is not None:
        payload["best_position_record"] = record.best_position_record.to_preview()
    return payload


__all__ = [
    "HybridLocalPositionCompatibilityScorer",
    "INDEX_BASE",
    "LocalPositionRecord",
    "LocalPositionSessionRecord",
]
