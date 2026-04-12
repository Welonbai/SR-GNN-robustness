from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable

from attack.common.artifact_io import load_json, save_json
from attack.common.config import Config
from attack.common.paths import (
    attack_key,
    attack_key_payload,
    canonical_split_paths,
    evaluation_key,
    evaluation_key_payload,
    run_artifact_paths,
    run_metadata_paths,
    split_key,
    split_key_payload,
    target_selection_key,
    target_selection_key_payload,
    victim_prediction_key,
    victim_prediction_key_payload,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import SessionStats
from attack.pipeline.core.evaluator import evaluate_targeted_metrics, save_metrics
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts, resolve_target_items
from attack.pipeline.core.victim_execution import VictimExecutionResult, execute_single_victim


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
    build_poisoned: Callable[[int], TargetPoisonOutput],
) -> dict[str, object]:
    metadata_paths = run_metadata_paths(config, run_type=run_type)
    run_root = metadata_paths["run_root"]
    summary_path = metadata_paths["summary"]
    if summary_path.exists():
        print(f"[run] Existing summary found at {summary_path}. Aborting to avoid overwrite.")
        raise RuntimeError("Run output already exists for this config/run_type.")
    if run_root.exists():
        existing_metrics = list(run_root.rglob("metrics.json"))
        if existing_metrics:
            print(f"[run] Existing outputs found under {run_root}. Aborting to avoid overwrite.")
            raise RuntimeError("Run output already exists for this config/run_type.")

    metadata_paths["run_root"].mkdir(parents=True, exist_ok=True)
    context.shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        target_snapshot = context.shared_paths["target_config_snapshot"]
        if not target_snapshot.exists():
            shutil.copyfile(config_path, target_snapshot)
    resolved_payload = _resolved_config_payload(config, run_type=run_type)
    key_payloads = _key_payloads(config, run_type=run_type)
    artifact_manifest = _initial_artifact_manifest(
        config,
        context=context,
        run_type=run_type,
        metadata_paths=metadata_paths,
    )
    save_json(resolved_payload, metadata_paths["resolved_config"])
    save_json(key_payloads, metadata_paths["key_payloads"])
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])

    target_items = resolve_target_items(
        context.stats,
        config,
        shared_paths=context.shared_paths,
    )
    summary: dict[str, object] = {
        "run_type": run_type,
        "target_items": [int(item) for item in target_items],
        "victims": list(config.victims.enabled),
        "fake_session_count": int(context.fake_session_count),
        "clean_session_count": int(len(context.clean_sessions)),
        "training": _training_summary(config, run_type=run_type),
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
                run_type=run_type,
                target_id=target_item,
                victim_name=victim_name,
            )
            run_dir = artifacts["run_dir"]
            run_dir.mkdir(parents=True, exist_ok=True)
            if config_path:
                shutil.copyfile(config_path, artifacts["config_snapshot"])

            victim_result, reused = _maybe_reuse_or_execute_victim(
                config,
                victim_name=victim_name,
                canonical_dataset=context.canonical_dataset,
                poisoned_sessions=target_payload.poisoned.sessions,
                poisoned_labels=target_payload.poisoned.labels,
                run_dir=run_dir,
                poisoned_train_path=artifacts["poisoned_train"],
                target_item=int(target_item),
                eval_topk=config.evaluation.topk,
                srg_nn_export_paths=context.export_paths,
                predictions_path=artifacts["predictions"],
                artifacts=artifacts,
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
                "fake_session_count": int(context.fake_session_count),
                "clean_session_count": int(len(context.clean_sessions)),
                "training": _victim_training_summary(config, victim_name),
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
                "reused_predictions": bool(reused),
            }
            _update_artifact_manifest(
                artifact_manifest,
                target_item=int(target_item),
                victim_name=victim_name,
                artifacts=artifacts,
                victim_result=victim_result,
                reused=reused,
            )
            save_json(artifact_manifest, metadata_paths["artifact_manifest"])

        summary["targets"][str(target_item)] = target_summary

    save_metrics(summary, summary_path)
    artifact_manifest["output_files"]["summary"] = str(summary_path)
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])
    return summary


def _resolved_config_payload(config: Config, *, run_type: str) -> dict[str, object]:
    victim_prediction_keys = {
        victim_name: victim_prediction_key(config, victim_name, run_type=run_type)
        for victim_name in config.victims.enabled
    }
    return {
        "result_config": config.result_config_dict(),
        "runtime_config": config.runtime_config_dict(),
        "derived": {
            "run_type": run_type,
            "split_key": split_key(config),
            "target_selection_key": target_selection_key(config),
            "attack_key": attack_key(config, run_type=run_type),
            "victim_prediction_keys": victim_prediction_keys,
            "evaluation_key": evaluation_key(config, run_type=run_type),
        },
    }


def _key_payloads(config: Config, *, run_type: str) -> dict[str, object]:
    return {
        "split_key_payload": split_key_payload(config),
        "target_selection_key_payload": target_selection_key_payload(config),
        "attack_key_payload": attack_key_payload(config, run_type=run_type),
        "victim_prediction_key_payloads": {
            victim_name: victim_prediction_key_payload(config, victim_name, run_type=run_type)
            for victim_name in config.victims.enabled
        },
        "evaluation_key_payload": evaluation_key_payload(config, run_type=run_type),
    }


def _initial_artifact_manifest(
    config: Config,
    *,
    context: RunContext,
    run_type: str,
    metadata_paths: dict[str, Path],
) -> dict[str, object]:
    canonical_paths = canonical_split_paths(config)
    target_selection_artifact = {
        "shared_dir": str(context.shared_paths["target_shared_dir"]),
        "config_snapshot": str(context.shared_paths["target_config_snapshot"]),
        "selected_targets": str(context.shared_paths["selected_targets"]),
        "target_selection_meta": str(context.shared_paths["target_selection_meta"]),
        "legacy_target_info": str(context.shared_paths["target_info"]),
    }
    poison_artifact: dict[str, object] | None = None
    if run_type != "clean":
        poison_artifact = {
            "shared_dir": str(context.shared_paths["attack_shared_dir"]),
            "poison_model": str(context.shared_paths["poison_model"]),
            "fake_sessions": str(context.shared_paths["fake_sessions"]),
            "poison_train_history": str(context.shared_paths["poison_train_history"]),
            "config_snapshot": str(context.shared_paths["attack_config_snapshot"]),
        }
    return {
        "run_type": run_type,
        "canonical_split_artifact": {key: str(path) for key, path in canonical_paths.items()},
        "target_selection_artifact": target_selection_artifact,
        "poison_artifact": poison_artifact,
        "victims": {},
        "generated_configs": {},
        "output_files": {
            "resolved_config": str(metadata_paths["resolved_config"]),
            "key_payloads": str(metadata_paths["key_payloads"]),
            "artifact_manifest": str(metadata_paths["artifact_manifest"]),
            "summary": str(metadata_paths["summary"]),
        },
    }


def _maybe_reuse_or_execute_victim(
    config: Config,
    *,
    victim_name: str,
    canonical_dataset: CanonicalDataset,
    poisoned_sessions: list[list[int]],
    poisoned_labels: list[int],
    run_dir: Path,
    poisoned_train_path: Path,
    target_item: int,
    eval_topk: tuple[int, ...] | list[int],
    srg_nn_export_paths: dict[str, Path] | None,
    predictions_path: Path,
    artifacts: dict[str, Path],
) -> tuple[VictimExecutionResult, bool]:
    reused = _load_shared_victim_result(
        config,
        victim_name=victim_name,
        run_dir=run_dir,
        artifacts=artifacts,
        predictions_path=predictions_path,
    )
    if reused is not None:
        print(
            f"[victim:{victim_name}] Reusing shared predictions from "
            f"{artifacts['shared_predictions']}"
        )
        return reused, True

    victim_result = execute_single_victim(
        config,
        victim_name=victim_name,
        canonical_dataset=canonical_dataset,
        poisoned_sessions=poisoned_sessions,
        poisoned_labels=poisoned_labels,
        run_dir=run_dir,
        poisoned_train_path=poisoned_train_path,
        target_item=target_item,
        eval_topk=eval_topk,
        srg_nn_export_paths=srg_nn_export_paths,
        predictions_path=predictions_path,
    )
    _persist_shared_victim_result(victim_result, artifacts=artifacts)
    return victim_result, False


def _load_shared_victim_result(
    config: Config,
    *,
    victim_name: str,
    run_dir: Path,
    artifacts: dict[str, Path],
    predictions_path: Path,
) -> VictimExecutionResult | None:
    shared_predictions = artifacts["shared_predictions"]
    shared_execution_result = artifacts["shared_execution_result"]
    if not shared_predictions.exists() or not shared_execution_result.exists():
        return None

    _copy_if_exists(shared_predictions, predictions_path)
    _copy_if_exists(artifacts["shared_train_history"], artifacts["train_history"])
    poisoned_train_local: Path | None = None
    if artifacts["shared_poisoned_train"].exists():
        _copy_if_exists(artifacts["shared_poisoned_train"], artifacts["poisoned_train"])
        poisoned_train_local = artifacts["poisoned_train"]

    execution_payload = load_json(shared_execution_result)
    predictions_payload = load_json(shared_predictions)
    if not isinstance(execution_payload, dict) or not isinstance(predictions_payload, dict):
        raise ValueError("Shared victim artifacts are malformed.")

    _write_reused_victim_resolved_config(
        config,
        victim_name=victim_name,
        run_dir=run_dir,
        artifacts=artifacts,
    )

    rankings_raw = predictions_payload.get("rankings")
    rankings = None
    if rankings_raw is not None:
        rankings = [list(map(int, row)) for row in rankings_raw]

    extra = execution_payload.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}
    return VictimExecutionResult(
        predictions=rankings,
        predictions_path=predictions_path,
        extra=extra,
        poisoned_train_path=poisoned_train_local,
    )


def _persist_shared_victim_result(
    victim_result: VictimExecutionResult,
    *,
    artifacts: dict[str, Path],
) -> None:
    artifacts["shared_dir"].mkdir(parents=True, exist_ok=True)
    _copy_if_exists(artifacts["predictions"], artifacts["shared_predictions"])
    _copy_if_exists(artifacts["train_history"], artifacts["shared_train_history"])
    shared_poisoned_train = None
    if victim_result.poisoned_train_path is not None and Path(victim_result.poisoned_train_path).exists():
        _copy_if_exists(Path(victim_result.poisoned_train_path), artifacts["shared_poisoned_train"])
        shared_poisoned_train = str(artifacts["shared_poisoned_train"])
    save_json(
        {
            "extra": victim_result.extra,
            "predictions_path": str(artifacts["shared_predictions"]),
            "train_history_path": (
                str(artifacts["shared_train_history"])
                if artifacts["shared_train_history"].exists()
                else None
            ),
            "poisoned_train_path": shared_poisoned_train,
        },
        artifacts["shared_execution_result"],
    )


def _update_artifact_manifest(
    artifact_manifest: dict[str, object],
    *,
    target_item: int,
    victim_name: str,
    artifacts: dict[str, Path],
    victim_result: VictimExecutionResult,
    reused: bool,
) -> None:
    victims_payload = artifact_manifest.setdefault("victims", {})
    if not isinstance(victims_payload, dict):
        return
    target_key = str(target_item)
    target_payload = victims_payload.setdefault(target_key, {})
    if not isinstance(target_payload, dict):
        return
    target_payload[victim_name] = {
        "reused_predictions": bool(reused),
        "local": {
            "run_dir": str(artifacts["run_dir"]),
            "resolved_config": str(artifacts["resolved_config"]),
            "config_snapshot": str(artifacts["config_snapshot"]),
            "predictions": str(artifacts["predictions"]),
            "metrics": str(artifacts["metrics"]),
            "train_history": str(artifacts["train_history"]),
            "poisoned_train": str(artifacts["poisoned_train"]),
        },
        "shared": {
            "shared_dir": str(artifacts["shared_dir"]),
            "predictions": str(artifacts["shared_predictions"]),
            "train_history": str(artifacts["shared_train_history"]),
            "execution_result": str(artifacts["shared_execution_result"]),
            "poisoned_train": str(artifacts["shared_poisoned_train"]),
        },
    }
    generated_configs = artifact_manifest.setdefault("generated_configs", {})
    if isinstance(generated_configs, dict) and victim_result.extra:
        generated_configs[f"{target_key}:{victim_name}"] = victim_result.extra


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_reused_victim_resolved_config(
    config: Config,
    *,
    victim_name: str,
    run_dir: Path,
    artifacts: dict[str, Path],
) -> None:
    runtime = (config.victims.runtime or {}).get(victim_name, {})
    payload = {
        "victim_name": victim_name,
        "params": config.victims.params.get(victim_name, {}),
        "runtime": runtime,
        "pipeline_injected": {
            "export_topk_k": int(max(config.evaluation.topk)),
            "export_topk_path": str(artifacts["predictions"]),
            "run_dir": str(run_dir),
            "reused_from_shared_dir": str(artifacts["shared_dir"]),
        },
    }
    save_json(payload, run_dir / "resolved_config.json")


def _training_summary(config: Config, *, run_type: str) -> dict[str, object]:
    poison_model = None
    if run_type != "clean":
        poison_model = {
            "name": config.attack.poison_model.name,
            "train": config.attack.poison_model.params.get("train", {}),
        }
    return {
        "poison_model": poison_model,
        "victims": {
            victim_name: _victim_training_summary(config, victim_name)
            for victim_name in config.victims.enabled
        },
    }


def _victim_training_summary(config: Config, victim_name: str) -> dict[str, object]:
    params = config.victims.params.get(victim_name, {})
    train = params.get("train", {})
    if isinstance(train, dict):
        return dict(train)
    return {}


__all__ = ["RunContext", "TargetPoisonOutput", "run_targets_and_victims"]
