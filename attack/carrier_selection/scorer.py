from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

from attack.common.config import CarrierSelectionConfig


@dataclass(frozen=True)
class CarrierScoreRecord:
    index: int
    session: list[int]
    raw_embedding_score: float
    raw_cooccurrence_score: float
    raw_transition_score: float
    normalized_embedding_score: float
    normalized_cooccurrence_score: float
    normalized_transition_score: float
    carrier_score: float

    def to_preview(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "session": [int(item) for item in self.session],
            "carrier_score": float(self.carrier_score),
            "normalized_component_scores": {
                "embedding": float(self.normalized_embedding_score),
                "cooccurrence": float(self.normalized_cooccurrence_score),
                "transition": float(self.normalized_transition_score),
            },
            "raw_component_scores": {
                "embedding": float(self.raw_embedding_score),
                "cooccurrence": float(self.raw_cooccurrence_score),
                "transition": float(self.raw_transition_score),
            },
        }


@dataclass(frozen=True)
class _EmbeddingResolution:
    table: np.ndarray | None
    metadata: dict[str, object]


class _EmbeddingRowResolver:
    def __init__(self, *, row_count: int, max_item_id: int) -> None:
        self.row_count = int(row_count)
        self.max_item_id = int(max_item_id)
        if self.row_count > self.max_item_id:
            self.mode = "padding_row_0_item_id_to_row"
        elif self.row_count == self.max_item_id:
            self.mode = "one_based_item_id_to_zero_based_row"
        else:
            self.mode = "fallback_bounds_checked"

    def row_for_item(self, item_id: int) -> int | None:
        item = int(item_id)
        if self.mode == "padding_row_0_item_id_to_row":
            row = item
        elif self.mode == "one_based_item_id_to_zero_based_row":
            row = item - 1
        else:
            if 0 <= item < self.row_count:
                row = item
            else:
                row = item - 1
        if 0 <= row < self.row_count:
            return int(row)
        return None


class HybridTargetSessionCompatibilityScorer:
    """Diagnostic TACS scorer; position placement stays Random-NZ outside this class."""

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
    ) -> tuple[list[CarrierScoreRecord], dict[str, object]]:
        candidates = [list(session) for session in candidate_sessions]
        target = int(target_item)
        max_item_id = _max_item_id(self.train_sub_sessions, candidates, target)
        embedding_resolution = _resolve_embedding_table(
            poison_runner=self.poison_runner,
            embedding_table=self.embedding_table,
            max_item_id=max_item_id,
        )

        cooccurrence_counts, target_train_sub_count = _target_cooccurrence_counts(
            self.train_sub_sessions,
            target,
        )
        transition_counts = _target_transition_counts(self.train_sub_sessions, target)

        raw_embedding_scores: list[float] = []
        raw_cooccurrence_scores: list[float] = []
        raw_transition_scores: list[float] = []
        skipped_target_item_count = 0
        out_of_bounds_item_count = 0

        target_row = None
        resolver: _EmbeddingRowResolver | None = None
        table = embedding_resolution.table
        if table is not None:
            resolver = _EmbeddingRowResolver(row_count=table.shape[0], max_item_id=max_item_id)
            target_row = resolver.row_for_item(target)

        for session in candidates:
            filtered_items = [int(item) for item in session if int(item) != target]
            skipped_target_item_count += int(len(session) - len(filtered_items))
            embedding_score, skipped_oob = _embedding_score(
                filtered_items,
                table=table,
                resolver=resolver,
                target_row=target_row,
            )
            raw_embedding_scores.append(embedding_score)
            out_of_bounds_item_count += int(skipped_oob)
            raw_cooccurrence_scores.append(
                _count_compatibility_score(filtered_items, cooccurrence_counts)
            )
            raw_transition_scores.append(
                _count_compatibility_score(filtered_items, transition_counts)
            )

        normalized_embedding, embedding_constant = _minmax(raw_embedding_scores)
        normalized_cooccurrence, cooccurrence_constant = _minmax(raw_cooccurrence_scores)
        normalized_transition, transition_constant = _minmax(raw_transition_scores)
        total_weight = (
            float(self.config.embedding_weight)
            + float(self.config.cooccurrence_weight)
            + float(self.config.transition_weight)
        )

        records: list[CarrierScoreRecord] = []
        for index, session in enumerate(candidates):
            carrier_score = (
                (float(self.config.embedding_weight) * normalized_embedding[index])
                + (float(self.config.cooccurrence_weight) * normalized_cooccurrence[index])
                + (float(self.config.transition_weight) * normalized_transition[index])
            ) / total_weight
            records.append(
                CarrierScoreRecord(
                    index=int(index),
                    session=[int(item) for item in session],
                    raw_embedding_score=float(raw_embedding_scores[index]),
                    raw_cooccurrence_score=float(raw_cooccurrence_scores[index]),
                    raw_transition_score=float(raw_transition_scores[index]),
                    normalized_embedding_score=float(normalized_embedding[index]),
                    normalized_cooccurrence_score=float(normalized_cooccurrence[index]),
                    normalized_transition_score=float(normalized_transition[index]),
                    carrier_score=float(carrier_score),
                )
            )

        embedding_metadata = dict(embedding_resolution.metadata)
        embedding_metadata.update(
            {
                "max_item_id": int(max_item_id),
                "target_item_embedding_row": (
                    None if target_row is None else int(target_row)
                ),
                "item_id_row_mapping": None if resolver is None else resolver.mode,
                "skipped_target_item_count": int(skipped_target_item_count),
                "out_of_bounds_item_count": int(out_of_bounds_item_count),
            }
        )
        metadata = {
            "target_train_sub_count": int(target_train_sub_count),
            "embedding": embedding_metadata,
            "component_score_summaries": {
                "embedding": {
                    "raw": score_summary(raw_embedding_scores),
                    "normalized": score_summary(normalized_embedding),
                },
                "cooccurrence": {
                    "raw": score_summary(raw_cooccurrence_scores),
                    "normalized": score_summary(normalized_cooccurrence),
                },
                "transition": {
                    "raw": score_summary(raw_transition_scores),
                    "normalized": score_summary(normalized_transition),
                },
            },
            "constant_normalized_columns": {
                "embedding": bool(embedding_constant),
                "cooccurrence": bool(cooccurrence_constant),
                "transition": bool(transition_constant),
            },
        }
        return records, metadata


def _resolve_embedding_table(
    *,
    poison_runner: Any | None,
    embedding_table: Any | None,
    max_item_id: int,
) -> _EmbeddingResolution:
    source = "none"
    table_obj = embedding_table
    if table_obj is None and poison_runner is not None:
        model = getattr(poison_runner, "model", None)
        if model is not None:
            table_obj = _find_model_embedding_weight(model)
            source = "poison_runner.model"
    elif table_obj is not None:
        source = "provided_embedding_table"

    if table_obj is None:
        return _EmbeddingResolution(
            table=None,
            metadata={
                "available": False,
                "source": source,
                "embedding_shape": None,
                "max_item_id": int(max_item_id),
            },
        )

    table = _to_numpy(table_obj)
    if table.ndim != 2:
        raise ValueError("Item embedding table must be a 2D matrix.")
    return _EmbeddingResolution(
        table=table.astype(np.float32, copy=False),
        metadata={
            "available": True,
            "source": source,
            "embedding_shape": [int(table.shape[0]), int(table.shape[1])],
            "max_item_id": int(max_item_id),
        },
    )


def _find_model_embedding_weight(model: Any) -> Any | None:
    embedding = getattr(model, "embedding", None)
    if embedding is not None and getattr(embedding, "weight", None) is not None:
        return embedding.weight
    try:
        import torch.nn as nn
    except ImportError:  # pragma: no cover
        return None
    if isinstance(model, nn.Module):
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                return module.weight
    return None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _max_item_id(
    train_sub_sessions: Sequence[Sequence[int]],
    candidate_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> int:
    max_item = int(target_item)
    for collection in (train_sub_sessions, candidate_sessions):
        for session in collection:
            if session:
                max_item = max(max_item, max(int(item) for item in session))
    return int(max_item)


def _target_cooccurrence_counts(
    sessions: Sequence[Sequence[int]],
    target_item: int,
) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    target_sessions = 0
    target = int(target_item)
    for session in sessions:
        unique_items = {int(item) for item in session}
        if target not in unique_items:
            continue
        target_sessions += 1
        for item in unique_items:
            if item != target:
                counts[item] += 1
    return counts, int(target_sessions)


def _target_transition_counts(
    sessions: Sequence[Sequence[int]],
    target_item: int,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    target = int(target_item)
    for session in sessions:
        normalized = [int(item) for item in session]
        for left, right in zip(normalized, normalized[1:]):
            if right == target and left != target:
                counts[left] += 1
            if left == target and right != target:
                counts[right] += 1
    return counts


def _embedding_score(
    items: Sequence[int],
    *,
    table: np.ndarray | None,
    resolver: _EmbeddingRowResolver | None,
    target_row: int | None,
) -> tuple[float, int]:
    if table is None or resolver is None or target_row is None:
        return 0.0, 0
    target_vector = table[int(target_row)]
    similarities: list[float] = []
    skipped_oob = 0
    for item in items:
        row = resolver.row_for_item(int(item))
        if row is None:
            skipped_oob += 1
            continue
        similarity = _cosine_similarity(target_vector, table[int(row)])
        similarities.append(float(np.clip((similarity + 1.0) / 2.0, 0.0, 1.0)))
    return _aggregate(similarities), int(skipped_oob)


def _count_compatibility_score(items: Sequence[int], counts: Mapping[int, int]) -> float:
    values = [math.log1p(int(counts.get(int(item), 0))) for item in items]
    return _aggregate(values)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _aggregate(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float((0.5 * max(values)) + (0.5 * (sum(values) / len(values))))


def _minmax(values: Sequence[float]) -> tuple[list[float], bool]:
    if not values:
        return [], True
    min_value = float(min(values))
    max_value = float(max(values))
    if max_value == min_value:
        return [0.0 for _ in values], True
    return [
        float((float(value) - min_value) / (max_value - min_value))
        for value in values
    ], False


def score_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [float(value) for value in values]
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
    }


__all__ = [
    "CarrierScoreRecord",
    "HybridTargetSessionCompatibilityScorer",
    "score_summary",
]
