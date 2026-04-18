from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from attack.common.artifact_io import load_json, save_json
from attack.common.config import Config
from attack.common.paths import (
    attack_key,
    attack_key_payload,
    canonical_split_paths,
    evaluation_key,
    evaluation_key_payload,
    run_group_key,
    run_group_key_payload,
    run_artifact_paths,
    run_metadata_paths,
    shared_attack_artifact_key,
    shared_attack_artifact_key_payload,
    split_key,
    split_key_payload,
    target_cohort_key,
    target_cohort_key_payload,
    target_selection_key,
    target_selection_key_payload,
    victim_prediction_key,
    victim_prediction_key_payload,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import SessionStats
from attack.pipeline.core.evaluator import evaluate_prediction_metrics, save_metrics
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts, resolve_target_items
from attack.pipeline.core.victim_execution import VictimExecutionResult, execute_single_victim


PROGRESS_TIMEZONE = "Asia/Taipei"
_PROGRESS_ZONE = ZoneInfo(PROGRESS_TIMEZONE)


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
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata_paths = run_metadata_paths(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    _guard_phase1_run_group_reuse(
        config,
        run_type=run_type,
        metadata_paths=metadata_paths,
        attack_identity_context=attack_identity_context,
    )
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
    resolved_payload = _resolved_config_payload(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    key_payloads = _key_payloads(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    artifact_manifest = _initial_artifact_manifest(
        config,
        context=context,
        run_type=run_type,
        metadata_paths=metadata_paths,
        attack_identity_context=attack_identity_context,
    )
    save_json(resolved_payload, metadata_paths["resolved_config"])
    save_json(key_payloads, metadata_paths["key_payloads"])
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])
    run_started_monotonic = time.monotonic()
    progress_payload = _initial_progress_payload(config, run_type=run_type)
    save_json(progress_payload, metadata_paths["progress"])

    target_items = resolve_target_items(
        context.stats,
        config,
        shared_paths=context.shared_paths,
    )
    _populate_progress_plan(
        progress_payload,
        target_items=[int(item) for item in target_items],
        victim_names=list(config.victims.enabled),
        elapsed_seconds=time.monotonic() - run_started_monotonic,
    )
    save_json(progress_payload, metadata_paths["progress"])
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

    current_run: dict[str, object] | None = None
    victim_names = list(config.victims.enabled)
    try:
        for target_index, target_item in enumerate(target_items, start=1):
            current_run = {
                "target_index": int(target_index),
                "target_item": int(target_item),
                "victim_index": None,
                "victim_name": None,
                "overall_index": None,
            }
            target_payload = build_poisoned(int(target_item))
            target_summary = {
                "target_item": int(target_item),
                "victims": {},
            }
            if target_payload.metadata:
                target_summary["metadata"] = dict(target_payload.metadata)
            for victim_index, victim_name in enumerate(victim_names, start=1):
                overall_index = ((target_index - 1) * len(victim_names)) + victim_index
                current_run = {
                    "target_index": int(target_index),
                    "target_item": int(target_item),
                    "victim_index": int(victim_index),
                    "victim_name": victim_name,
                    "overall_index": int(overall_index),
                }
                _mark_progress_run_started(
                    progress_payload,
                    current_run=current_run,
                    elapsed_seconds=time.monotonic() - run_started_monotonic,
                )
                save_json(progress_payload, metadata_paths["progress"])

                artifacts = run_artifact_paths(
                    config,
                    run_type=run_type,
                    target_id=target_item,
                    victim_name=victim_name,
                    attack_identity_context=attack_identity_context,
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

                ground_truth_labels = resolve_ground_truth_labels(
                    config,
                    victim_name=victim_name,
                    canonical_dataset=context.canonical_dataset,
                    predictions=victim_result.predictions,
                )

                metrics, available = evaluate_prediction_metrics(
                    victim_result.predictions,
                    target_item=int(target_item),
                    ground_truth_labels=ground_truth_labels,
                    targeted_metrics=config.evaluation.targeted_metrics,
                    ground_truth_metrics=config.evaluation.ground_truth_metrics,
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
                _mark_progress_run_completed(
                    progress_payload,
                    current_run=current_run,
                    reused=reused,
                    elapsed_seconds=time.monotonic() - run_started_monotonic,
                )
                save_json(progress_payload, metadata_paths["progress"])

            summary["targets"][str(target_item)] = target_summary
            current_run = None

        save_metrics(summary, summary_path)
        artifact_manifest["output_files"]["summary"] = str(summary_path)
        save_json(artifact_manifest, metadata_paths["artifact_manifest"])
        _mark_progress_finished(
            progress_payload,
            status="completed",
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        save_json(progress_payload, metadata_paths["progress"])
        return summary
    except Exception as exc:
        _mark_progress_finished(
            progress_payload,
            status="failed",
            current_run=current_run,
            error=str(exc),
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        save_json(progress_payload, metadata_paths["progress"])
        raise


def _identity_object(*, key: str, payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        "key": key,
        "payload": dict(payload),
    }


def _canonical_identity_sections(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    attack_key_value = attack_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return {
        "split_identity": _identity_object(
            key=split_key(config),
            payload=split_key_payload(config),
        ),
        "target_cohort_identity": _identity_object(
            key=target_cohort_key(config),
            payload=target_cohort_key_payload(config),
        ),
        "run_group_identity": _identity_object(
            key=run_group_key(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            payload=run_group_key_payload(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        ),
        "attack_identity": {
            **_identity_object(
                key=attack_key_value,
                payload=attack_key_payload(
                    config,
                    run_type=run_type,
                    attack_identity_context=attack_identity_context,
                ),
            ),
            "shared_attack_artifact_identity": _identity_object(
                key=shared_attack_artifact_key(config, run_type=run_type),
                payload=shared_attack_artifact_key_payload(
                    config,
                    run_type=run_type,
                ),
            ),
        },
        "victim_prediction_identities": {
            victim_name: _identity_object(
                key=victim_prediction_key(
                    config,
                    victim_name,
                    run_type=run_type,
                    attack_identity_context=attack_identity_context,
                ),
                payload=victim_prediction_key_payload(
                    config,
                    victim_name,
                    run_type=run_type,
                    attack_identity_context=attack_identity_context,
                ),
            )
            for victim_name in config.victims.enabled
        },
    }


def _legacy_identity_sections(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "target_selection_identity": _identity_object(
            key=target_selection_key(config),
            payload=target_selection_key_payload(config),
        ),
        "evaluation_identity": _identity_object(
            key=evaluation_key(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            payload=evaluation_key_payload(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        ),
    }


def _current_phase1_legacy_batch_state(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, str]:
    legacy_identities = _legacy_identity_sections(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    target_selection_identity = legacy_identities["target_selection_identity"]
    evaluation_identity = legacy_identities["evaluation_identity"]
    if not isinstance(target_selection_identity, dict) or not isinstance(evaluation_identity, dict):
        raise TypeError("Legacy identity payloads must be mappings.")
    return {
        "run_type": run_type,
        "target_selection_key": str(target_selection_identity["key"]),
        "evaluation_key": str(evaluation_identity["key"]),
    }


def _extract_identity_key(identity_payload: object) -> str | None:
    if isinstance(identity_payload, dict):
        value = identity_payload.get("key")
        if isinstance(value, str):
            return value
    if isinstance(identity_payload, str):
        return identity_payload
    return None


def _extract_existing_phase1_legacy_batch_state(
    resolved_payload: Mapping[str, Any],
) -> dict[str, str] | None:
    derived = resolved_payload.get("derived")
    if not isinstance(derived, Mapping):
        return None
    run_type = derived.get("run_type")
    if not isinstance(run_type, str):
        return None

    legacy_identities = derived.get("legacy_identities")
    if isinstance(legacy_identities, Mapping):
        target_selection_key_value = _extract_identity_key(
            legacy_identities.get("target_selection_identity", legacy_identities.get("target_selection_key"))
        )
        evaluation_key_value = _extract_identity_key(
            legacy_identities.get("evaluation_identity", legacy_identities.get("evaluation_key"))
        )
    else:
        target_selection_key_value = _extract_identity_key(derived.get("target_selection_key"))
        evaluation_key_value = _extract_identity_key(derived.get("evaluation_key"))

    if target_selection_key_value is None or evaluation_key_value is None:
        return None
    return {
        "run_type": run_type,
        "target_selection_key": target_selection_key_value,
        "evaluation_key": evaluation_key_value,
    }


def _guard_phase1_run_group_reuse(
    config: Config,
    *,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    attack_identity_context: Mapping[str, object] | None = None,
) -> None:
    run_root = metadata_paths["run_root"]
    if not run_root.exists():
        return

    existing_entries = list(run_root.iterdir())
    if not existing_entries:
        return

    current_state = _current_phase1_legacy_batch_state(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    resolved_config_path = metadata_paths["resolved_config"]
    if not resolved_config_path.exists():
        raise RuntimeError(
            "Existing run-group root detected at "
            f"{run_root}, but no resolved_config.json was found to confirm the prior "
            "batch-era request. Phase 1 does not implement append/resume semantics yet, "
            "so ambiguous run-group reuse is forbidden."
        )

    resolved_payload = load_json(resolved_config_path)
    if not isinstance(resolved_payload, dict):
        raise RuntimeError(
            "Existing run-group root detected at "
            f"{run_root}, but resolved_config.json is malformed. Phase 1 does not "
            "implement append/resume semantics yet, so ambiguous run-group reuse is forbidden."
        )

    existing_state = _extract_existing_phase1_legacy_batch_state(resolved_payload)
    if existing_state is None:
        raise RuntimeError(
            "Existing run-group root detected at "
            f"{run_root}, but the stored metadata does not expose the legacy batch-era "
            "identities needed to verify compatibility. Phase 1 does not implement "
            "append/resume semantics yet, so ambiguous run-group reuse is forbidden."
        )

    if existing_state != current_state:
        raise RuntimeError(
            "Run-group root collision detected at "
            f"{run_root}. The existing root belongs to a different batch-era request "
            f"(existing run_type={existing_state['run_type']}, "
            f"target_selection_key={existing_state['target_selection_key']}, "
            f"evaluation_key={existing_state['evaluation_key']}; current "
            f"run_type={current_state['run_type']}, "
            f"target_selection_key={current_state['target_selection_key']}, "
            f"evaluation_key={current_state['evaluation_key']}). Phase 1 has moved "
            "run roots to run_group_key, but append semantics are not implemented until later phases."
        )

    raise RuntimeError(
        "Existing run-group root detected at "
        f"{run_root} for the same batch-era request "
        f"(run_type={current_state['run_type']}, "
        f"target_selection_key={current_state['target_selection_key']}, "
        f"evaluation_key={current_state['evaluation_key']}). Phase 1 does not yet "
        "implement resume, overwrite, or append semantics within a run-group root."
    )


def _resolved_config_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "result_config": config.result_config_dict(),
        "runtime_config": config.runtime_config_dict(),
        "derived": {
            "run_type": run_type,
            **_canonical_identity_sections(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "legacy_identities": _legacy_identity_sections(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        },
    }


def _key_payloads(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_type": run_type,
        **_canonical_identity_sections(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
        "legacy_identities": _legacy_identity_sections(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
    }


def _initial_artifact_manifest(
    config: Config,
    *,
    context: RunContext,
    run_type: str,
    metadata_paths: dict[str, Path],
    attack_identity_context: Mapping[str, object] | None = None,
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
    derived_identity: dict[str, object] | None = None
    if attack_identity_context is not None:
        derived_identity = {
            "final_attack_identity_context": dict(attack_identity_context),
        }
    return {
        "run_type": run_type,
        "identities": {
            **_canonical_identity_sections(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "legacy_identities": _legacy_identity_sections(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        },
        "shared_artifacts": {
            "canonical_split": {key: str(path) for key, path in canonical_paths.items()},
            "target_cohort": {
                "shared_dir": str(context.shared_paths["target_cohort_dir"]),
                "target_registry": str(context.shared_paths["target_registry"]),
            },
            "legacy_target_selection": target_selection_artifact,
            "poison_artifact": poison_artifact,
        },
        "run_group_artifacts": {
            "run_root": str(metadata_paths["run_root"]),
            "resolved_config": str(metadata_paths["resolved_config"]),
            "key_payloads": str(metadata_paths["key_payloads"]),
            "artifact_manifest": str(metadata_paths["artifact_manifest"]),
            "run_coverage": str(metadata_paths["run_coverage"]),
            "execution_log": str(metadata_paths["execution_log"]),
            "summary_current": str(metadata_paths["summary_current"]),
            "progress": str(metadata_paths["progress"]),
            "legacy_summary": str(metadata_paths["summary"]),
        },
        "derived_identity": derived_identity,
        "victims": {},
        "generated_configs": {},
        "output_files": {
            "run_root": str(metadata_paths["run_root"]),
            "resolved_config": str(metadata_paths["resolved_config"]),
            "key_payloads": str(metadata_paths["key_payloads"]),
            "artifact_manifest": str(metadata_paths["artifact_manifest"]),
            "run_coverage": str(metadata_paths["run_coverage"]),
            "execution_log": str(metadata_paths["execution_log"]),
            "summary_current": str(metadata_paths["summary_current"]),
            "progress": str(metadata_paths["progress"]),
            "summary": str(metadata_paths["summary"]),
        },
    }


def _initial_progress_payload(config: Config, *, run_type: str) -> dict[str, object]:
    timestamp = _progress_timestamp()
    return {
        "run_type": run_type,
        "timezone": PROGRESS_TIMEZONE,
        "status": "initializing",
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "elapsed_seconds": 0.0,
        "total_targets": 0,
        "total_victims": int(len(config.victims.enabled)),
        "total_runs": 0,
        "completed_runs": 0,
        "target_items": [],
        "current": None,
        "runs": [],
    }


def _populate_progress_plan(
    progress_payload: dict[str, object],
    *,
    target_items: list[int],
    victim_names: list[str],
    elapsed_seconds: float,
) -> None:
    timestamp = _progress_timestamp()
    runs: list[dict[str, object]] = []
    overall_index = 1
    for target_index, target_item in enumerate(target_items, start=1):
        for victim_index, victim_name in enumerate(victim_names, start=1):
            runs.append(
                {
                    "overall_index": int(overall_index),
                    "target_index": int(target_index),
                    "target_item": int(target_item),
                    "victim_index": int(victim_index),
                    "victim_name": victim_name,
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "reused_predictions": None,
                }
            )
            overall_index += 1
    progress_payload["status"] = "running"
    progress_payload["updated_at"] = timestamp
    progress_payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    progress_payload["total_targets"] = int(len(target_items))
    progress_payload["total_victims"] = int(len(victim_names))
    progress_payload["total_runs"] = int(len(runs))
    progress_payload["completed_runs"] = 0
    progress_payload["target_items"] = [int(item) for item in target_items]
    progress_payload["current"] = None
    progress_payload["runs"] = runs


def _mark_progress_run_started(
    progress_payload: dict[str, object],
    *,
    current_run: dict[str, object],
    elapsed_seconds: float,
) -> None:
    timestamp = _progress_timestamp()
    progress_payload["status"] = "running"
    progress_payload["updated_at"] = timestamp
    progress_payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    progress_payload["current"] = dict(current_run)
    entry = _progress_run_entry(progress_payload, current_run)
    if entry is None:
        return
    entry["status"] = "running"
    if entry.get("started_at") is None:
        entry["started_at"] = timestamp
    entry["completed_at"] = None
    entry["reused_predictions"] = None


def _mark_progress_run_completed(
    progress_payload: dict[str, object],
    *,
    current_run: dict[str, object],
    reused: bool,
    elapsed_seconds: float,
) -> None:
    timestamp = _progress_timestamp()
    progress_payload["status"] = "running"
    progress_payload["updated_at"] = timestamp
    progress_payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    progress_payload["current"] = None
    entry = _progress_run_entry(progress_payload, current_run)
    if entry is not None:
        if entry.get("started_at") is None:
            entry["started_at"] = timestamp
        entry["status"] = "completed"
        entry["completed_at"] = timestamp
        entry["reused_predictions"] = bool(reused)
    progress_payload["completed_runs"] = _completed_run_count(progress_payload)


def _mark_progress_finished(
    progress_payload: dict[str, object],
    *,
    status: str,
    elapsed_seconds: float,
    current_run: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    timestamp = _progress_timestamp()
    progress_payload["status"] = status
    progress_payload["updated_at"] = timestamp
    progress_payload["completed_at"] = timestamp
    progress_payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    progress_payload["current"] = dict(current_run) if current_run is not None else None
    if error:
        progress_payload["error"] = error
    entry = _progress_run_entry(progress_payload, current_run)
    if status == "failed" and entry is not None and entry.get("status") != "completed":
        if entry.get("started_at") is None:
            entry["started_at"] = timestamp
        entry["status"] = "failed"
        entry["completed_at"] = timestamp
    if status == "completed":
        progress_payload["completed_runs"] = int(progress_payload.get("total_runs", 0))
        progress_payload["current"] = None
    else:
        progress_payload["completed_runs"] = _completed_run_count(progress_payload)


def _progress_run_entry(
    progress_payload: dict[str, object],
    current_run: dict[str, object] | None,
) -> dict[str, object] | None:
    if current_run is None:
        return None
    overall_index = current_run.get("overall_index")
    if overall_index is None:
        return None
    runs = progress_payload.get("runs")
    if not isinstance(runs, list):
        return None
    index = int(overall_index) - 1
    if 0 <= index < len(runs):
        entry = runs[index]
        if isinstance(entry, dict) and int(entry.get("overall_index", -1)) == int(overall_index):
            return entry
    for entry in runs:
        if isinstance(entry, dict) and int(entry.get("overall_index", -1)) == int(overall_index):
            return entry
    return None


def _completed_run_count(progress_payload: dict[str, object]) -> int:
    runs = progress_payload.get("runs")
    if not isinstance(runs, list):
        return 0
    return sum(
        1
        for entry in runs
        if isinstance(entry, dict) and entry.get("status") == "completed"
    )


def _progress_timestamp() -> str:
    return datetime.now(_PROGRESS_ZONE).isoformat()


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
