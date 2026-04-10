from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

from attack.common.config import Config
from attack.common.paths import run_artifact_paths, run_config_dir
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import SessionStats
from attack.pipeline.core.evaluator import evaluate_targeted_metrics, save_metrics
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts, resolve_target_items
from attack.pipeline.core.victim_execution import execute_single_victim


@dataclass(frozen=True)
class RunContext:
    canonical_dataset: CanonicalDataset
    stats: SessionStats
    clean_sessions: list[list[int]]
    clean_labels: list[int]
    export_paths: dict[str, Path] | None
    shared_paths: dict[str, Path]
    fake_session_count: int

    @classmethod
    def from_shared(cls, shared: SharedAttackArtifacts) -> "RunContext":
        return cls(
            canonical_dataset=shared.canonical_dataset,
            stats=shared.stats,
            clean_sessions=shared.clean_sessions,
            clean_labels=shared.clean_labels,
            export_paths=shared.export_paths,
            shared_paths=shared.shared_paths,
            fake_session_count=shared.fake_session_count,
        )


@dataclass(frozen=True)
class TargetPoisonOutput:
    poisoned: PoisonedDataset
    metadata: dict[str, object]


def run_targets_and_victims(
    config: Config,
    *,
    config_path: str | Path | None,
    context: RunContext,
    run_type: str,
    poison_epochs: int,
    attack_epochs: int,
    build_poisoned: Callable[[int], TargetPoisonOutput],
) -> dict[str, object]:
    target_items = resolve_target_items(
        context.stats,
        config,
        shared_paths=context.shared_paths,
    )
    summary: dict[str, object] = {
        "run_type": run_type,
        "target_items": [int(item) for item in target_items],
        "victims": list(config.victims.enabled),
        "poison_epochs": int(poison_epochs),
        "attack_epochs": int(attack_epochs),
        "fake_session_count": int(context.fake_session_count),
        "clean_session_count": int(len(context.clean_sessions)),
        "targets": {},
    }
    if "srgnn" in config.victims.enabled and context.export_paths is None:
        raise ValueError("SRGNN victim execution requires export paths for valid/test.")

    for target_item in target_items:
        target_payload = build_poisoned(int(target_item))
        target_summary = {
            "target_item": int(target_item),
            "victims": {},
        }
        if target_payload.metadata:
            target_summary["metadata"] = dict(target_payload.metadata)
        for victim_name in config.victims.enabled:
            artifacts = run_artifact_paths(
                config,
                target_id=target_item,
                victim_name=victim_name,
            )
            run_dir = artifacts["run_dir"]
            run_dir.mkdir(parents=True, exist_ok=True)
            if config_path:
                shutil.copyfile(config_path, artifacts["config_snapshot"])

            victim_result = execute_single_victim(
                config,
                victim_name=victim_name,
                canonical_dataset=context.canonical_dataset,
                poisoned_sessions=target_payload.poisoned.sessions,
                poisoned_labels=target_payload.poisoned.labels,
                run_dir=run_dir,
                poisoned_train_path=artifacts["poisoned_train"],
                target_item=int(target_item),
                attack_epochs=attack_epochs,
                eval_topk=config.evaluation.topk,
                srg_nn_export_paths=context.export_paths,
                predictions_path=artifacts["predictions"],
            )

            metrics, available = evaluate_targeted_metrics(
                victim_result.predictions,
                target_item=int(target_item),
                metrics=config.evaluation.metrics,
                topk=config.evaluation.topk,
            )

            payload: dict[str, object] = {
                "run_type": run_type,
                "victim": victim_name,
                "target_item": int(target_item),
                "poison_epochs": int(poison_epochs),
                "attack_epochs": int(attack_epochs),
                "fake_session_count": int(context.fake_session_count),
                "clean_session_count": int(len(context.clean_sessions)),
                "metrics_available": bool(available),
                "metrics": metrics,
                "predictions_path": str(artifacts["predictions"]),
            }
            if victim_result.poisoned_train_path is not None:
                payload["poisoned_train_path"] = str(victim_result.poisoned_train_path)
            if target_payload.metadata:
                payload.update(target_payload.metadata)
            if victim_result.extra:
                payload.update(victim_result.extra)
            save_metrics(payload, artifacts["metrics"])

            target_summary["victims"][victim_name] = {
                "metrics_path": str(artifacts["metrics"]),
                "predictions_path": str(artifacts["predictions"]),
                "metrics": metrics,
                "metrics_available": bool(available),
            }

        summary["targets"][str(target_item)] = target_summary

    summary_path = run_config_dir(config) / f"summary_{run_type}.json"
    save_metrics(summary, summary_path)
    return summary


__all__ = ["RunContext", "TargetPoisonOutput", "run_targets_and_victims"]
