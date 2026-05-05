from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from attack.carrier_selection.coverage_prefix_bank import CoveragePrefixBank
from attack.carrier_selection.local_position_scorer import INDEX_BASE
from attack.carrier_selection.scorer import (
    _EmbeddingRowResolver,
    _max_item_id,
    _minmax,
    _resolve_embedding_table,
    score_summary,
)
from attack.common.config import CarrierSelectionConfig


@dataclass(frozen=True)
class CoverageLocalPositionRecord:
    candidate_index: int
    session: list[int]
    position: int
    position_label: str
    left_item: int | None
    right_item: int | None
    replaced_item: int
    raw_coverage_score: float
    top_m_similarity_mean: float
    top_m_rank_weighted_score: float
    top_m_similarity_min: float
    top_m_similarity_max: float
    normalized_coverage_score: float = 0.0
    position_score: float = 0.0

    def with_scores(
        self,
        *,
        normalized_coverage_score: float,
        position_score: float,
    ) -> "CoverageLocalPositionRecord":
        return CoverageLocalPositionRecord(
            candidate_index=int(self.candidate_index),
            session=[int(item) for item in self.session],
            position=int(self.position),
            position_label=self.position_label,
            left_item=None if self.left_item is None else int(self.left_item),
            right_item=None if self.right_item is None else int(self.right_item),
            replaced_item=int(self.replaced_item),
            raw_coverage_score=float(self.raw_coverage_score),
            top_m_similarity_mean=float(self.top_m_similarity_mean),
            top_m_rank_weighted_score=float(self.top_m_rank_weighted_score),
            top_m_similarity_min=float(self.top_m_similarity_min),
            top_m_similarity_max=float(self.top_m_similarity_max),
            normalized_coverage_score=float(normalized_coverage_score),
            position_score=float(position_score),
        )

    def to_preview(self, *, targetized_session: Sequence[int] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_index": int(self.candidate_index),
            "original_session": [int(item) for item in self.session],
            "best_position": int(self.position),
            "best_position_label": self.position_label,
            "index_base": INDEX_BASE,
            "best_position_score": float(self.position_score),
            "raw_coverage_score": float(self.raw_coverage_score),
            "normalized_coverage_score": float(self.normalized_coverage_score),
            "top_m_similarity_summary": {
                "mean": float(self.top_m_similarity_mean),
                "min": float(self.top_m_similarity_min),
                "max": float(self.top_m_similarity_max),
                "rank_weighted_score": float(self.top_m_rank_weighted_score),
            },
            "left_item": None if self.left_item is None else int(self.left_item),
            "right_item": None if self.right_item is None else int(self.right_item),
            "replaced_item": int(self.replaced_item),
        }
        if targetized_session is not None:
            payload["targetized_session"] = [int(item) for item in targetized_session]
        return payload


@dataclass(frozen=True)
class CoverageLocalPositionSessionRecord:
    index: int
    session: list[int]
    valid_position_count: int
    best_position: int | None
    best_position_label: str | None
    best_position_score: float
    raw_coverage_score: float
    normalized_coverage_score: float
    invalid: bool
    best_position_record: CoverageLocalPositionRecord | None


class CoverageAwareLocalPositionScorer:
    """Scores local replacement positions by coverage of vulnerable validation prefixes."""

    def __init__(
        self,
        *,
        config: CarrierSelectionConfig,
        prefix_bank: CoveragePrefixBank,
        poison_runner: Any | None = None,
        embedding_table: Any | None = None,
    ) -> None:
        self.config = config
        self.prefix_bank = prefix_bank
        self.poison_runner = poison_runner
        self.embedding_table = embedding_table

    def score(
        self,
        *,
        candidate_sessions: Sequence[Sequence[int]],
        target_item: int,
    ) -> tuple[list[CoverageLocalPositionSessionRecord], dict[str, object]]:
        candidates = [[int(item) for item in session] for session in candidate_sessions]
        target = int(target_item)
        max_item_id = _max_item_id([], candidates, target)
        bank_max_item = self.prefix_bank.metadata.get("embedding", {})
        if isinstance(bank_max_item, dict) and bank_max_item.get("max_item_id") is not None:
            max_item_id = max(max_item_id, int(bank_max_item["max_item_id"]))
        embedding_resolution = _resolve_embedding_table(
            poison_runner=self.poison_runner,
            embedding_table=self.embedding_table,
            max_item_id=max_item_id,
        )
        table = embedding_resolution.table
        if table is None:
            raise ValueError("Coverage local-position scoring requires item embeddings.")
        resolver = _EmbeddingRowResolver(row_count=table.shape[0], max_item_id=max_item_id)

        raw_positions: list[CoverageLocalPositionRecord] = []
        per_session_counts: Counter[int] = Counter()
        skipped_noop_positions = 0
        out_of_bounds_item_count = 0
        for candidate_index, session in enumerate(candidates):
            for position in range(1, len(session)):
                replaced_item = int(session[position])
                if replaced_item == target:
                    skipped_noop_positions += 1
                    continue
                targetized_prefix = list(session[: position + 1])
                targetized_prefix[position] = target
                candidate_vector, skipped_items = _targetized_prefix_representation(
                    targetized_prefix,
                    table=table,
                    resolver=resolver,
                )
                out_of_bounds_item_count += int(skipped_items)
                coverage = _coverage_score(
                    candidate_vector,
                    prefix_bank=self.prefix_bank,
                    top_m=int(self.config.top_m_coverage),
                )
                left_item = int(session[position - 1]) if position - 1 >= 0 else None
                right_item = int(session[position + 1]) if position + 1 < len(session) else None
                per_session_counts[int(candidate_index)] += 1
                raw_positions.append(
                    CoverageLocalPositionRecord(
                        candidate_index=int(candidate_index),
                        session=[int(item) for item in session],
                        position=int(position),
                        position_label=_position_label(position),
                        left_item=left_item,
                        right_item=right_item,
                        replaced_item=replaced_item,
                        raw_coverage_score=float(coverage["coverage_score"]),
                        top_m_similarity_mean=float(coverage["similarity_mean"]),
                        top_m_rank_weighted_score=float(coverage["rank_weighted_score"]),
                        top_m_similarity_min=float(coverage["similarity_min"]),
                        top_m_similarity_max=float(coverage["similarity_max"]),
                    )
                )

        scored_positions, normalization_metadata = _normalize_positions(raw_positions)
        best_by_session: dict[int, CoverageLocalPositionRecord] = {}
        for record in scored_positions:
            current = best_by_session.get(int(record.candidate_index))
            if current is None or _position_record_rank(record) < _position_record_rank(current):
                best_by_session[int(record.candidate_index)] = record

        session_records: list[CoverageLocalPositionSessionRecord] = []
        for index, session in enumerate(candidates):
            best = best_by_session.get(index)
            invalid = best is None
            session_records.append(
                CoverageLocalPositionSessionRecord(
                    index=int(index),
                    session=[int(item) for item in session],
                    valid_position_count=int(per_session_counts.get(index, 0)),
                    best_position=None if best is None else int(best.position),
                    best_position_label=None if best is None else best.position_label,
                    best_position_score=(
                        float("-inf") if best is None else float(best.position_score)
                    ),
                    raw_coverage_score=(
                        float("-inf") if best is None else float(best.raw_coverage_score)
                    ),
                    normalized_coverage_score=(
                        float("-inf") if best is None else float(best.normalized_coverage_score)
                    ),
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
        metadata: dict[str, object] = {
            "index_base": INDEX_BASE,
            "target_item": int(target),
            "scorer": "coverage_aware_local_position",
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
            "coverage_score_summary": {
                "raw": score_summary([record.raw_coverage_score for record in raw_positions]),
                "normalized": score_summary(
                    [record.normalized_coverage_score for record in scored_positions]
                ),
                "constant": bool(normalization_metadata["constant_normalized_columns"]["coverage"]),
            },
            "position_level_score_summaries": _position_score_summaries(scored_positions),
            "session_level_best_score_summaries": _position_score_summaries(best_position_records),
            **normalization_metadata,
            "embedding": {
                **dict(embedding_resolution.metadata),
                "item_id_row_mapping": resolver.mode,
                "max_item_id": int(max_item_id),
                "out_of_bounds_item_count": int(out_of_bounds_item_count),
            },
        }
        if bool(self.config.debug_save_all_position_records):
            metadata["_debug_all_position_records"] = [
                record.to_preview() for record in scored_positions
            ]
        return session_records, metadata


def _coverage_score(
    candidate_vector: np.ndarray | None,
    *,
    prefix_bank: CoveragePrefixBank,
    top_m: int,
) -> dict[str, float]:
    if candidate_vector is None or prefix_bank.representations.size == 0:
        return {
            "coverage_score": 0.0,
            "rank_weighted_score": 0.0,
            "similarity_mean": 0.0,
            "similarity_min": 0.0,
            "similarity_max": 0.0,
        }
    similarities = np.matmul(prefix_bank.representations, candidate_vector.astype(np.float32))
    m = min(int(top_m), int(similarities.shape[0]))
    if m <= 0:
        return {
            "coverage_score": 0.0,
            "rank_weighted_score": 0.0,
            "similarity_mean": 0.0,
            "similarity_min": 0.0,
            "similarity_max": 0.0,
        }
    top_indices = np.argsort(-similarities, kind="mergesort")[:m]
    top_similarities = similarities[top_indices].astype(np.float32, copy=False)
    top_weights = prefix_bank.weights[top_indices].astype(np.float32, copy=False)
    weight_sum = float(np.sum(top_weights))
    if weight_sum <= 0.0:
        weighted = 0.0
    else:
        weighted = float(np.sum(top_similarities * top_weights) / weight_sum)
    return {
        "coverage_score": float(weighted),
        "rank_weighted_score": float(weighted),
        "similarity_mean": float(np.mean(top_similarities)),
        "similarity_min": float(np.min(top_similarities)),
        "similarity_max": float(np.max(top_similarities)),
    }


def _targetized_prefix_representation(
    items: Sequence[int],
    *,
    table: np.ndarray,
    resolver: _EmbeddingRowResolver,
) -> tuple[np.ndarray | None, int]:
    rows: list[np.ndarray] = []
    skipped = 0
    for item in items:
        if int(item) <= 0:
            skipped += 1
            continue
        row = resolver.row_for_item(int(item))
        if row is None:
            skipped += 1
            continue
        rows.append(table[int(row)])
    if not rows:
        return None, int(skipped)
    vector = np.mean(np.asarray(rows, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None, int(skipped)
    return (vector / norm).astype(np.float32, copy=False), int(skipped)


def _normalize_positions(
    raw_positions: Sequence[CoverageLocalPositionRecord],
) -> tuple[list[CoverageLocalPositionRecord], dict[str, object]]:
    raw_scores = [record.raw_coverage_score for record in raw_positions]
    normalized, constant = _minmax(raw_scores)
    scored = [
        record.with_scores(
            normalized_coverage_score=float(normalized[index]),
            position_score=float(normalized[index]),
        )
        for index, record in enumerate(raw_positions)
    ]
    return scored, {
        "constant_normalized_columns": {
            "coverage": bool(constant),
        },
    }


def _position_record_rank(record: CoverageLocalPositionRecord) -> tuple[float, int, int]:
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
    records: Sequence[CoverageLocalPositionRecord | None],
) -> dict[str, dict[str, float]]:
    clean_records = [record for record in records if record is not None]
    return {
        "position_score": score_summary([record.position_score for record in clean_records]),
        "coverage": {
            "raw": score_summary([record.raw_coverage_score for record in clean_records]),
            "normalized": score_summary(
                [record.normalized_coverage_score for record in clean_records]
            ),
        },
        "top_m_similarity": {
            "mean": score_summary(
                [record.top_m_similarity_mean for record in clean_records]
            ),
            "rank_weighted": score_summary(
                [record.top_m_rank_weighted_score for record in clean_records]
            ),
        },
    }


__all__ = [
    "CoverageAwareLocalPositionScorer",
    "CoverageLocalPositionRecord",
    "CoverageLocalPositionSessionRecord",
]
