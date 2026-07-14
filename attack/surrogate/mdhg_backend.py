from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from attack.common.config import Config
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.models.mdhg_constants import (
    MDHG_ADAPTER_VERSION,
    MDHG_TRAIN_DATA_CONSTRUCTION_MODE,
)
from attack.models.mdhg_core import MDHGInProcessModel
from attack.pipeline.core.evaluator import evaluate_targeted_metrics
from attack.position_opt.types import SurrogateScoreResult
from attack.surrogate.base import PoisonedTrainInput, SessionBatch


_TARGET_SCORE_METRIC_TOPK = 30
_TARGET_SCORE_TARGETED_METRICS = ("mrr", "recall", "ndcg")
_TARGET_SCORE_REQUIRED_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
    "targeted_mrr@30",
    "targeted_recall@30",
)


@dataclass
class MDHGModelHandle:
    model: MDHGInProcessModel

    def cleanup(self) -> None:
        self.model.cleanup()


class MDHGBackend:
    name = "mdhg"

    def __init__(
        self,
        config: Config,
        *,
        train_config: Mapping[str, Any],
        item_count: int,
        seed: int,
        clean_train_raw_sessions: Sequence[Sequence[int]],
    ) -> None:
        self.config = config
        self.train_config = dict(train_config)
        self.item_count = int(item_count)
        self.seed = int(seed)
        self.clean_train_raw_sessions = [
            [int(item) for item in session]
            for session in clean_train_raw_sessions
        ]
        runtime = (config.victims.runtime or {}).get("mdhg", {})
        device = runtime.get("device", {}) if isinstance(runtime, Mapping) else {}
        self.use_gpu = bool(device.get("use_gpu", False))
        self.gpu_id = device.get("gpu_id", "0")

    def build_fresh_model(self) -> MDHGModelHandle:
        return MDHGModelHandle(
            model=MDHGInProcessModel(
                train_config=self.train_config,
                item_count=self.item_count,
                seed=self.seed,
                dataset_name=self.config.data.dataset_name,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
            )
        )

    def score_target(
        self,
        model: object,
        eval_sessions: SessionBatch,
        target_item: int,
    ) -> SurrogateScoreResult:
        handle = self._as_model_handle(model)
        rankings = handle.model.score_sessions_topk(
            eval_sessions,
            topk=_TARGET_SCORE_METRIC_TOPK,
        )
        metrics, _ = evaluate_targeted_metrics(
            rankings,
            target_item=int(target_item),
            metrics=_TARGET_SCORE_TARGETED_METRICS,
            topk=[10, 20, 30],
        )
        coerced = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("targeted_")
        }
        for key in _TARGET_SCORE_REQUIRED_KEYS:
            coerced.setdefault(key, 0.0)
        target_scores = [
            1.0 if int(target_item) in ranking else 0.0
            for ranking in rankings
        ]
        return SurrogateScoreResult.from_values(target_scores, metrics=coerced)

    def score_gt(
        self,
        model: object,
        eval_sessions: SessionBatch,
        ground_truth_items: Sequence[int],
    ) -> SurrogateScoreResult:
        if len(eval_sessions) != len(ground_truth_items):
            raise ValueError(
                "MDHG ground-truth scoring requires one label per validation session: "
                f"{len(eval_sessions)} sessions vs {len(ground_truth_items)} labels."
            )
        for index, label in enumerate(ground_truth_items):
            item = int(label)
            if item < 1 or item > int(self.item_count):
                raise ValueError(
                    f"MDHG ground_truth_items[{index}]={item} is outside "
                    f"canonical item range 1..{int(self.item_count)}."
                )
        handle = self._as_model_handle(model)
        rankings = handle.model.score_sessions_topk(
            eval_sessions,
            topk=max(self.config.evaluation.topk),
        )
        if len(rankings) != len(eval_sessions):
            raise ValueError(
                "MDHG ranking count does not match validation session count: "
                f"{len(rankings)} rankings vs {len(eval_sessions)} sessions."
            )
        values = [
            1.0 if int(label) in ranking else 0.0
            for label, ranking in zip(ground_truth_items, rankings)
        ]
        return SurrogateScoreResult.from_values(values)

    @staticmethod
    def _as_model_handle(model: object) -> MDHGModelHandle:
        if not isinstance(model, MDHGModelHandle):
            raise TypeError("MDHGBackend expects an MDHGModelHandle.")
        return model


def mdhg_surrogate_identity_extra() -> dict[str, object]:
    return {
        "mdhg_adapter_version": int(MDHG_ADAPTER_VERSION),
        "mdhg_train_data_construction_mode": MDHG_TRAIN_DATA_CONSTRUCTION_MODE,
    }


def coerce_mdhg_poisoned_train_data(
    poisoned_train_data: PoisonedTrainInput,
    *,
    clean_train_raw_sessions: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int], list[list[int]], list[list[int]]]:
    if isinstance(poisoned_train_data, PoisonedDataset):
        sessions = poisoned_train_data.sessions
        labels = poisoned_train_data.labels
        fake_sessions = [list(session) for session in poisoned_train_data.fake_sessions]
        raw_train_sessions = [
            *[list(session) for session in clean_train_raw_sessions],
            *fake_sessions,
        ]
    else:
        sessions, labels = poisoned_train_data
        fake_sessions = []
        raw_train_sessions = _raw_sessions_from_pairs(sessions, labels)
    normalized_sessions = [list(session) for session in sessions]
    normalized_labels = [int(label) for label in labels]
    if len(normalized_sessions) != len(normalized_labels):
        raise ValueError("poisoned_train_data sessions and labels must align.")
    if not normalized_sessions:
        raise ValueError("poisoned_train_data must contain at least one row.")
    if not raw_train_sessions:
        raise ValueError("MDHG raw training sessions must not be empty.")
    return normalized_sessions, normalized_labels, raw_train_sessions, fake_sessions


def _raw_sessions_from_pairs(
    sessions: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> list[list[int]]:
    rows: list[list[int]] = []
    for session, label in zip(sessions, labels):
        rows.append([*[int(item) for item in session], int(label)])
    return rows


__all__ = [
    "MDHGBackend",
    "MDHGModelHandle",
    "coerce_mdhg_poisoned_train_data",
    "mdhg_surrogate_identity_extra",
]
