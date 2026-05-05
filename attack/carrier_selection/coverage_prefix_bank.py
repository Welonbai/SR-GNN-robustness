from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Any, Sequence

import numpy as np

from attack.carrier_selection.scorer import (
    _EmbeddingRowResolver,
    _max_item_id,
    _resolve_embedding_table,
    _to_numpy,
    score_summary,
)
from attack.common.config import (
    COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK,
    COVERAGE_RANK_WEIGHTING_NONE,
    CarrierSelectionConfig,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import expand_session_to_samples


TARGET_RANK_CONVENTION = "rank = 1 + count(scores > target_score)"
LOW_VULNERABLE_PREFIX_WARNING_THRESHOLD = 100


@dataclass(frozen=True)
class ValidationPrefixRankRecord:
    case_index: int
    prefix: list[int]
    label: int
    target_rank: int
    target_score: float | None = None


@dataclass(frozen=True)
class CoveragePrefixRecord:
    case_index: int
    prefix: list[int]
    label: int
    target_rank: int
    target_score: float | None
    weight: float
    representation: list[float]

    def to_metadata(self) -> dict[str, object]:
        return {
            "case_index": int(self.case_index),
            "prefix": [int(item) for item in self.prefix],
            "label": int(self.label),
            "target_rank": int(self.target_rank),
            "target_score": (
                None if self.target_score is None else float(self.target_score)
            ),
            "weight": float(self.weight),
        }


@dataclass(frozen=True)
class CoveragePrefixBank:
    target_item: int
    records: list[CoveragePrefixRecord]
    representations: np.ndarray
    weights: np.ndarray
    ranks: np.ndarray
    metadata: dict[str, object]


def build_vulnerable_validation_prefix_bank(
    *,
    canonical_dataset: CanonicalDataset,
    target_item: int,
    config: CarrierSelectionConfig,
    poison_runner: Any,
    embedding_table: Any | None = None,
) -> CoveragePrefixBank:
    cases = _validation_prefix_cases(canonical_dataset.valid)
    if not cases:
        raise ValueError("Coverage prefix bank cannot be built: validation has no prefixes.")
    max_item_id = _max_item_id(canonical_dataset.train_sub, [case.prefix for case in cases], target_item)
    embedding_resolution = _resolve_embedding_table(
        poison_runner=poison_runner,
        embedding_table=embedding_table,
        max_item_id=max_item_id,
    )
    if embedding_resolution.table is None:
        raise ValueError("Coverage prefix bank requires an item embedding table.")

    ranked_cases, score_metadata = _rank_validation_prefix_cases(
        cases=cases,
        target_item=int(target_item),
        poison_runner=poison_runner,
    )
    return build_prefix_bank_from_ranked_cases(
        ranked_cases=ranked_cases,
        target_item=int(target_item),
        config=config,
        embedding_table=embedding_resolution.table,
        max_item_id=max_item_id,
        embedding_metadata=embedding_resolution.metadata,
        score_metadata=score_metadata,
    )


def build_prefix_bank_from_ranked_cases(
    *,
    ranked_cases: Sequence[ValidationPrefixRankRecord],
    target_item: int,
    config: CarrierSelectionConfig,
    embedding_table: Any,
    max_item_id: int | None = None,
    embedding_metadata: dict[str, object] | None = None,
    score_metadata: dict[str, object] | None = None,
) -> CoveragePrefixBank:
    cases = [
        ValidationPrefixRankRecord(
            case_index=int(case.case_index),
            prefix=[int(item) for item in case.prefix],
            label=int(case.label),
            target_rank=int(case.target_rank),
            target_score=(
                None if case.target_score is None else float(case.target_score)
            ),
        )
        for case in ranked_cases
    ]
    if max_item_id is None:
        max_item_id = _max_item_id([], [case.prefix for case in cases], int(target_item))
    table = _to_numpy(embedding_table).astype(np.float32, copy=False)
    if table.ndim != 2:
        raise ValueError("Coverage prefix bank item embedding table must be 2D.")
    resolver = _EmbeddingRowResolver(row_count=table.shape[0], max_item_id=int(max_item_id))

    vulnerable = [
        case
        for case in cases
        if int(config.vulnerable_rank_min) < int(case.target_rank) <= int(config.vulnerable_rank_max)
    ]
    vulnerable = sorted(
        vulnerable,
        key=lambda case: (int(case.target_rank), int(case.case_index)),
    )[: int(config.max_vulnerable_prefixes)]

    skipped_prefix_count = 0
    records: list[CoveragePrefixRecord] = []
    out_of_bounds_item_count = 0
    for case in vulnerable:
        representation, skipped_items = _mean_item_embedding_representation(
            case.prefix,
            table=table,
            resolver=resolver,
        )
        out_of_bounds_item_count += int(skipped_items)
        if representation is None:
            skipped_prefix_count += 1
            continue
        records.append(
            CoveragePrefixRecord(
                case_index=int(case.case_index),
                prefix=[int(item) for item in case.prefix],
                label=int(case.label),
                target_rank=int(case.target_rank),
                target_score=(
                    None if case.target_score is None else float(case.target_score)
                ),
                weight=_rank_weight(int(case.target_rank), config.rank_weighting),
                representation=representation.astype(np.float32).tolist(),
            )
        )

    if not records:
        raise ValueError(
            "Coverage prefix bank is empty after vulnerable-rank filtering and "
            "embedding validation."
        )

    representations = np.asarray(
        [record.representation for record in records],
        dtype=np.float32,
    )
    weights = np.asarray([record.weight for record in records], dtype=np.float32)
    ranks = np.asarray([record.target_rank for record in records], dtype=np.int64)

    score_payload = dict(score_metadata or {})
    metadata: dict[str, object] = {
        "target_item": int(target_item),
        "validation_prefix_count": int(len(cases)),
        "vulnerable_prefix_count": int(len(records)),
        "vulnerable_prefix_count_before_embedding_filter": int(len(vulnerable)),
        "low_vulnerable_prefix_count_warning": bool(
            len(records) < LOW_VULNERABLE_PREFIX_WARNING_THRESHOLD
        ),
        "vulnerable_rank_min": int(config.vulnerable_rank_min),
        "vulnerable_rank_max": int(config.vulnerable_rank_max),
        "max_vulnerable_prefixes": int(config.max_vulnerable_prefixes),
        "selected_prefix_rank_summary": score_summary([int(rank) for rank in ranks.tolist()]),
        "representation_method": config.prefix_representation,
        "rank_weighting": config.rank_weighting,
        "rank_weight_summary": score_summary([float(weight) for weight in weights.tolist()]),
        "rank_convention": TARGET_RANK_CONVENTION,
        "skipped_prefix_count": int(skipped_prefix_count),
        "out_of_bounds_item_count": int(out_of_bounds_item_count),
        "target_missing_or_oob_count": 0,
        "embedding": {
            **dict(embedding_metadata or {}),
            "item_id_row_mapping": resolver.mode,
            "embedding_shape": [int(table.shape[0]), int(table.shape[1])],
            "max_item_id": int(max_item_id),
        },
        "score_dim": score_payload.get("score_dim"),
        "target_score_column": score_payload.get("target_score_column"),
        "score_metadata": score_payload,
    }
    return CoveragePrefixBank(
        target_item=int(target_item),
        records=records,
        representations=representations,
        weights=weights,
        ranks=ranks,
        metadata=metadata,
    )


def target_rank_from_scores(scores: Sequence[float], *, target_score_column: int) -> tuple[int, float]:
    row = np.asarray(scores, dtype=np.float32)
    if row.ndim != 1:
        raise ValueError("scores must be a 1D score row.")
    column = int(target_score_column)
    if column < 0 or column >= int(row.shape[0]):
        raise ValueError(
            "Invalid target score column for coverage prefix bank: "
            f"target_score_column={column}, score_dim={int(row.shape[0])}."
        )
    target_score = float(row[column])
    rank = int(1 + np.count_nonzero(row > target_score))
    return rank, target_score


def _rank_validation_prefix_cases(
    *,
    cases: Sequence[ValidationPrefixRankRecord],
    target_item: int,
    poison_runner: Any,
) -> tuple[list[ValidationPrefixRankRecord], dict[str, object]]:
    if getattr(poison_runner, "model", None) is None:
        raise RuntimeError("Coverage prefix bank requires an initialized SR-GNN model.")
    from pytorch_code.model import forward as srg_forward, trans_to_cpu
    from pytorch_code.utils import Data
    import torch

    target_score_column = int(target_item) - 1
    if target_score_column < 0:
        raise ValueError(
            "Invalid target score column for coverage prefix bank: "
            f"target_item={int(target_item)}, target_score_column={target_score_column}."
        )
    data = Data(
        (
            [[int(item) for item in case.prefix] for case in cases],
            [int(case.label) for case in cases],
        ),
        shuffle=False,
    )
    model = poison_runner.model
    model.eval()
    ranked: list[ValidationPrefixRankRecord] = []
    score_dim: int | None = None
    with torch.no_grad():
        for batch_indices in data.generate_batch(model.batch_size):
            _, scores = srg_forward(model, batch_indices, data)
            scores_np = trans_to_cpu(scores).detach().numpy()
            if score_dim is None:
                score_dim = int(scores_np.shape[1])
                if target_score_column >= score_dim:
                    raise ValueError(
                        "Invalid target score column for coverage prefix bank: "
                        f"target_item={int(target_item)}, "
                        f"target_score_column={target_score_column}, "
                        f"score_dim={score_dim}."
                    )
            for row_offset, score_row in enumerate(scores_np):
                case = cases[int(batch_indices[row_offset])]
                rank, target_score = target_rank_from_scores(
                    score_row,
                    target_score_column=target_score_column,
                )
                ranked.append(
                    ValidationPrefixRankRecord(
                        case_index=int(case.case_index),
                        prefix=[int(item) for item in case.prefix],
                        label=int(case.label),
                        target_rank=int(rank),
                        target_score=float(target_score),
                    )
                )

    return ranked, {
        "score_dim": score_dim,
        "target_item": int(target_item),
        "target_score_column": int(target_score_column),
        "rank_convention": TARGET_RANK_CONVENTION,
    }


def _validation_prefix_cases(
    validation_sessions: Sequence[Sequence[int]],
) -> list[ValidationPrefixRankRecord]:
    cases: list[ValidationPrefixRankRecord] = []
    for session in validation_sessions:
        prefixes, labels = expand_session_to_samples(session)
        for prefix, label in zip(prefixes, labels):
            cases.append(
                ValidationPrefixRankRecord(
                    case_index=len(cases),
                    prefix=[int(item) for item in prefix],
                    label=int(label),
                    target_rank=0,
                    target_score=None,
                )
            )
    return cases


def _mean_item_embedding_representation(
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


def _rank_weight(rank: int, mode: str) -> float:
    if mode == COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK:
        return float(1.0 / log2(int(rank) + 1))
    if mode == COVERAGE_RANK_WEIGHTING_NONE:
        return 1.0
    raise ValueError(f"Unsupported coverage rank weighting: {mode}")


__all__ = [
    "CoveragePrefixBank",
    "CoveragePrefixRecord",
    "LOW_VULNERABLE_PREFIX_WARNING_THRESHOLD",
    "TARGET_RANK_CONVENTION",
    "ValidationPrefixRankRecord",
    "build_prefix_bank_from_ranked_cases",
    "build_vulnerable_validation_prefix_bank",
    "target_rank_from_scores",
]
