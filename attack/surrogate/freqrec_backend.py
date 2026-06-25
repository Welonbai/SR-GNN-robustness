from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from attack.common.config import Config
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.models.freqrec_core import (
    FREQREC_ADAPTER_VERSION,
    FREQREC_TRAIN_DATA_CONSTRUCTION_MODE,
    FreqRecInProcessModel,
)
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
class FreqRecModelHandle:
    model: FreqRecInProcessModel

    def cleanup(self) -> None:
        self.model.cleanup()


class FreqRecBackend:
    name = "freqrec"

    def __init__(
        self,
        config: Config,
        *,
        train_config: Mapping[str, Any],
        item_count: int,
        seed: int,
    ) -> None:
        self.config = config
        self.train_config = dict(train_config)
        self.item_count = int(item_count)
        self.seed = int(seed)
        runtime = (config.victims.runtime or {}).get("freqrec", {})
        device = runtime.get("device", {}) if isinstance(runtime, Mapping) else {}
        dataloader = runtime.get("dataloader", {}) if isinstance(runtime, Mapping) else {}
        self.use_gpu = bool(device.get("use_gpu", False))
        self.gpu_id = device.get("gpu_id", "0")
        self.num_workers = int(dataloader.get("num_workers", 0))

    def build_fresh_model(self) -> FreqRecModelHandle:
        return FreqRecModelHandle(
            model=FreqRecInProcessModel(
                train_config=self.train_config,
                item_count=self.item_count,
                seed=self.seed,
                use_gpu=self.use_gpu,
                gpu_id=self.gpu_id,
                num_workers=self.num_workers,
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
            key: float(metrics[key])
            for key in metrics
            if key.startswith("targeted_")
        }
        for key in _TARGET_SCORE_REQUIRED_KEYS:
            coerced.setdefault(key, 0.0)
        target_scores = []
        for ranking in rankings:
            target_scores.append(1.0 if int(target_item) in ranking else 0.0)
        return SurrogateScoreResult.from_values(target_scores, metrics=coerced)

    def score_gt(
        self,
        model: object,
        eval_sessions: SessionBatch,
        ground_truth_items: Sequence[int],
    ) -> SurrogateScoreResult:
        if len(eval_sessions) != len(ground_truth_items):
            raise ValueError(
                "FreqRec ground-truth scoring requires one label per validation session: "
                f"{len(eval_sessions)} sessions vs {len(ground_truth_items)} labels."
            )
        for index, label in enumerate(ground_truth_items):
            item = int(label)
            if item < 1 or item > int(self.item_count):
                raise ValueError(
                    f"FreqRec ground_truth_items[{index}]={item} is outside "
                    f"canonical item range 1..{int(self.item_count)}."
                )
        handle = self._as_model_handle(model)
        rankings = handle.model.score_sessions_topk(
            eval_sessions,
            topk=max(self.config.evaluation.topk),
        )
        if len(rankings) != len(eval_sessions):
            raise ValueError(
                "FreqRec ranking count does not match validation session count: "
                f"{len(rankings)} rankings vs {len(eval_sessions)} sessions."
            )
        values = [
            1.0 if int(label) in ranking else 0.0
            for label, ranking in zip(ground_truth_items, rankings)
        ]
        return SurrogateScoreResult.from_values(values)

    @staticmethod
    def _as_model_handle(model: object) -> FreqRecModelHandle:
        if not isinstance(model, FreqRecModelHandle):
            raise TypeError("FreqRecBackend expects a FreqRecModelHandle.")
        return model


def freqrec_surrogate_identity_extra() -> dict[str, object]:
    return {
        "freqrec_adapter_version": int(FREQREC_ADAPTER_VERSION),
        "freqrec_train_data_construction_mode": FREQREC_TRAIN_DATA_CONSTRUCTION_MODE,
    }


def coerce_poisoned_train_data(
    poisoned_train_data: PoisonedTrainInput,
) -> tuple[list[list[int]], list[int]]:
    if isinstance(poisoned_train_data, PoisonedDataset):
        sessions = poisoned_train_data.sessions
        labels = poisoned_train_data.labels
    else:
        sessions, labels = poisoned_train_data
    normalized_sessions = [list(session) for session in sessions]
    normalized_labels = [int(label) for label in labels]
    if len(normalized_sessions) != len(normalized_labels):
        raise ValueError("poisoned_train_data sessions and labels must align.")
    if not normalized_sessions:
        raise ValueError("poisoned_train_data must contain at least one row.")
    return normalized_sessions, normalized_labels


__all__ = [
    "FreqRecBackend",
    "FreqRecModelHandle",
    "coerce_poisoned_train_data",
    "freqrec_surrogate_identity_extra",
]
