from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4
from zoneinfo import ZoneInfo

from attack.common.artifact_io import (
    load_json,
    load_run_coverage,
    save_execution_log,
    save_json,
    save_run_coverage,
)
from attack.common.config import Config
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
    shared_attack_identity_requires_poison_runner,
    shared_victim_dir,
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
from attack.data.canonical_fingerprints import file_provenance
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import SessionStats
from attack.pipeline.core.evaluator import evaluate_prediction_metrics, save_metrics
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.pipeline.core.pipeline_utils import (
    SharedAttackArtifacts,
    ensure_target_registry_prefix,
    load_or_init_execution_log,
    load_or_init_run_coverage,
    plan_target_append_cells,
    rebuild_summary_current,
    requested_target_prefix,
    sync_run_coverage_materialized_prefix,
)
from attack.pipeline.core.victim_execution import (
    VictimExecutionResult,
    execute_single_victim,
    prepare_wearec_execution,
    victim_effective_train_seed,
)
from attack.models.victim.wearec_runner import load_wearec_prediction_payload


PROGRESS_TIMEZONE = "Asia/Taipei"
_PROGRESS_ZONE = ZoneInfo(PROGRESS_TIMEZONE)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DROP_SHARED_METADATA = object()
_PTS_CEM_TARGET_LEVEL_VICTIM_RUN_TYPES = frozenset(
    {
        PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    }
)


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
    raw_fake_sessions: list[list[int]]
    metadata: dict[str, object]


def run_targets_and_victims(
    config: Config,
    *,
    config_path: str | Path | None,
    context: RunContext,
    run_type: str,
    build_poisoned: Callable[[int], TargetPoisonOutput],
    attack_identity_context: Mapping[str, object] | None = None,
    max_targets_per_execution: int | None = None,
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
    artifact_manifest = _load_or_init_artifact_manifest(
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
    existing_run_coverage_payload = load_run_coverage(metadata_paths["run_coverage"])
    target_registry = ensure_target_registry_prefix(
        context.stats,
        config,
        shared_paths=context.shared_paths,
    )
    requested_target_items = requested_target_prefix(
        config,
        target_registry=target_registry,
    )
    run_coverage = load_or_init_run_coverage(
        config,
        run_type=run_type,
        metadata_paths=metadata_paths,
        target_registry=target_registry,
        attack_identity_context=attack_identity_context,
        allow_new_victims=True,
    )
    execution_log = load_or_init_execution_log(
        config,
        run_type=run_type,
        metadata_paths=metadata_paths,
        attack_identity_context=attack_identity_context,
    )
    _reconcile_interrupted_execution_records(
        execution_log,
        metadata_paths=metadata_paths,
    )

    requested_victims = list(config.victims.enabled)
    _validate_phase5_victim_request(
        run_coverage,
        requested_victims=requested_victims,
    )
    plan = plan_target_append_cells(
        run_coverage,
        requested_target_items=requested_target_items,
        requested_victims=requested_victims,
    )
    planned_cells = list(plan["planned_cells"])
    planned_cells = _limit_planned_cells_by_target(
        planned_cells,
        max_targets=max_targets_per_execution,
    )
    skipped_completed_cells = list(plan["skipped_completed_cells"])
    progress_payload = _initial_progress_payload(
        config,
        run_type=run_type,
        metadata_paths=metadata_paths,
    )
    execution_record = _append_execution_record(
        execution_log,
        config,
        run_type=run_type,
        requested_target_items=requested_target_items,
        requested_victims=requested_victims,
        planned_cells=planned_cells,
        skipped_completed_cells=skipped_completed_cells,
        existing_run_coverage_payload=existing_run_coverage_payload,
        metadata_paths=metadata_paths,
    )
    _populate_progress_plan_from_cells(
        progress_payload,
        requested_target_items=requested_target_items,
        requested_victims=requested_victims,
        planned_cells=planned_cells,
        skipped_completed_cells=skipped_completed_cells,
        elapsed_seconds=time.monotonic() - run_started_monotonic,
    )
    save_json(progress_payload, metadata_paths["progress"])
    if "srgnn" in requested_victims and planned_cells and context.export_paths is None:
        raise ValueError("SRGNN victim execution requires export paths for valid/test.")

    current_run: dict[str, object] | None = None
    current_artifacts: dict[str, Path] | None = None
    current_cell_committed = False
    plan_by_target = _group_planned_cells_by_target(planned_cells)
    try:
        overall_index = 0
        for target_index, (target_item, victim_cells) in enumerate(
            plan_by_target.items(), start=1
        ):
            current_run = {
                "target_index": int(target_index),
                "target_item": int(target_item),
                "victim_index": None,
                "victim_name": None,
                "overall_index": None,
            }
            target_payload = build_poisoned(int(target_item))
            for victim_index, cell in enumerate(victim_cells, start=1):
                overall_index += 1
                victim_name = str(cell["victim_name"])
                current_run = {
                    "target_index": int(target_index),
                    "target_item": int(target_item),
                    "victim_index": int(victim_index),
                    "victim_name": victim_name,
                    "overall_index": int(overall_index),
                }
                current_cell_committed = False
                _mark_progress_run_started(
                    progress_payload,
                    current_run=current_run,
                    elapsed_seconds=time.monotonic() - run_started_monotonic,
                )
                save_json(progress_payload, metadata_paths["progress"])

                victim_effective_seed = victim_effective_train_seed(
                    config,
                    victim_name=victim_name,
                    run_type=run_type,
                    target_item=int(target_item),
                )
                victim_attack_identity_context = (
                    _pts_cem_victim_attack_identity_context(
                        config,
                        run_type=run_type,
                        target_item=int(target_item),
                        target_metadata=target_payload.metadata,
                    )
                )
                victim_cache_effective_seed = (
                    victim_effective_seed
                    if victim_attack_identity_context is not None
                    else None
                )

                artifacts = run_artifact_paths(
                    config,
                    run_type=run_type,
                    target_id=target_item,
                    victim_name=victim_name,
                    attack_identity_context=attack_identity_context,
                    victim_attack_identity_context=victim_attack_identity_context,
                    victim_effective_train_seed=victim_cache_effective_seed,
                )
                current_artifacts = artifacts
                run_dir = artifacts["run_dir"]
                run_dir.mkdir(parents=True, exist_ok=True)
                if config_path and not artifacts["config_snapshot"].exists():
                    shutil.copyfile(config_path, artifacts["config_snapshot"])
                _mark_coverage_cell_pending(
                    run_coverage,
                    target_item=int(target_item),
                    victim_name=victim_name,
                    execution_id=str(execution_record["execution_id"]),
                )
                _save_run_coverage_snapshot(run_coverage, metadata_paths=metadata_paths)
                _record_execution_cell_pending(
                    execution_log,
                    execution_record=execution_record,
                    target_item=int(target_item),
                    victim_name=victim_name,
                    metadata_paths=metadata_paths,
                )

                victim_result, reused = _maybe_reuse_or_execute_victim(
                    config,
                    run_type=run_type,
                    run_coverage=run_coverage,
                    victim_name=victim_name,
                    canonical_dataset=context.canonical_dataset,
                    poisoned_sessions=target_payload.poisoned.sessions,
                    poisoned_labels=target_payload.poisoned.labels,
                    raw_fake_sessions=target_payload.raw_fake_sessions,
                    run_dir=run_dir,
                    poisoned_train_path=artifacts["poisoned_train"],
                    target_item=int(target_item),
                    eval_topk=config.evaluation.topk,
                    srg_nn_export_paths=context.export_paths,
                    predictions_path=artifacts["predictions"],
                    artifacts=artifacts,
                    victim_attack_identity_context=victim_attack_identity_context,
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
                    "predictions_path": _repo_relative_path(artifacts["predictions"]),
                    "reused_predictions": bool(reused),
                    "victim_prediction_key": victim_prediction_key(
                        config,
                        victim_name,
                        run_type=run_type,
                        attack_identity_context=attack_identity_context,
                        victim_attack_identity_context=(
                            victim_result.extra["wearec"]["scientific_identity"]
                            if victim_name == "wearec"
                            else victim_attack_identity_context
                        ),
                        victim_effective_train_seed=victim_cache_effective_seed,
                    ),
                }
                if victim_attack_identity_context is not None:
                    payload.update(
                        _pts_cem_victim_identity_metrics_payload(
                            config,
                            run_type=run_type,
                            attack_identity_context=attack_identity_context,
                            victim_attack_identity_context=victim_attack_identity_context,
                        )
                    )
                if victim_result.poisoned_train_path is not None:
                    payload["poisoned_train_path"] = _repo_relative_path(
                        victim_result.poisoned_train_path
                    )
                if target_payload.metadata:
                    payload.update(target_payload.metadata)
                if victim_result.extra:
                    payload.update(victim_result.extra)
                save_metrics(payload, artifacts["metrics"])
                wearec_manifest_persisted = victim_name == "wearec"
                if wearec_manifest_persisted:
                    _update_artifact_manifest(
                        artifact_manifest,
                        target_item=int(target_item),
                        victim_name=victim_name,
                        artifacts=artifacts,
                        victim_result=victim_result,
                        reused=reused,
                    )
                    save_json(artifact_manifest, metadata_paths["artifact_manifest"])
                _cleanup_victim_intermediates_if_enabled(
                    config,
                    victim_name=victim_name,
                    artifacts=artifacts,
                )
                _validate_completed_local_artifacts(artifacts)
                _mark_coverage_cell_completed(
                    run_coverage,
                    target_item=int(target_item),
                    victim_name=victim_name,
                    artifacts=artifacts,
                    execution_id=str(execution_record["execution_id"]),
                )
                _save_run_coverage_snapshot(run_coverage, metadata_paths=metadata_paths)
                _record_execution_cell_completion(
                    execution_log,
                    execution_record=execution_record,
                    target_item=int(target_item),
                    victim_name=victim_name,
                    metadata_paths=metadata_paths,
                )
                current_cell_committed = True

                if not wearec_manifest_persisted:
                    _update_artifact_manifest(
                        artifact_manifest,
                        target_item=int(target_item),
                        victim_name=victim_name,
                        artifacts=artifacts,
                        victim_result=victim_result,
                        reused=reused,
                    )
                    save_json(artifact_manifest, metadata_paths["artifact_manifest"])
                summary = _refresh_summary_snapshots(
                    config,
                    context=context,
                    run_type=run_type,
                    metadata_paths=metadata_paths,
                    run_coverage=run_coverage,
                    artifact_manifest=artifact_manifest,
                    attack_identity_context=attack_identity_context,
                )
                _mark_progress_run_completed(
                    progress_payload,
                    current_run=current_run,
                    reused=reused,
                    elapsed_seconds=time.monotonic() - run_started_monotonic,
                )
                save_json(progress_payload, metadata_paths["progress"])
                current_artifacts = None
                current_cell_committed = False
            current_run = None

        summary = _refresh_summary_snapshots(
            config,
            context=context,
            run_type=run_type,
            metadata_paths=metadata_paths,
            run_coverage=run_coverage,
            artifact_manifest=artifact_manifest,
            attack_identity_context=attack_identity_context,
        )
        _mark_execution_completed(
            execution_log,
            execution_record=execution_record,
            metadata_paths=metadata_paths,
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        _mark_progress_finished(
            progress_payload,
            status="completed",
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        save_json(progress_payload, metadata_paths["progress"])
        return summary
    except BaseException as exc:
        if (
            current_run is not None
            and current_run.get("victim_name")
            and not current_cell_committed
        ):
            _mark_coverage_cell_failed(
                run_coverage,
                target_item=int(current_run["target_item"]),
                victim_name=str(current_run["victim_name"]),
                artifacts=current_artifacts,
                error=exc,
                execution_id=str(execution_record["execution_id"]),
            )
            _save_run_coverage_snapshot(run_coverage, metadata_paths=metadata_paths)
            _record_execution_cell_failure(
                execution_log,
                execution_record=execution_record,
                target_item=int(current_run["target_item"]),
                victim_name=str(current_run["victim_name"]),
                error=exc,
                metadata_paths=metadata_paths,
            )
        _mark_progress_finished(
            progress_payload,
            status="failed",
            current_run=current_run,
            error=str(exc),
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        save_json(progress_payload, metadata_paths["progress"])
        try:
            _refresh_summary_snapshots(
                config,
                context=context,
                run_type=run_type,
                metadata_paths=metadata_paths,
                run_coverage=run_coverage,
                artifact_manifest=artifact_manifest,
                attack_identity_context=attack_identity_context,
            )
        except Exception:
            # summary_current/progress remain non-authoritative snapshots; do not mask the main failure
            pass
        _mark_execution_failed(
            execution_log,
            execution_record=execution_record,
            error=exc,
            metadata_paths=metadata_paths,
            elapsed_seconds=time.monotonic() - run_started_monotonic,
        )
        raise


def _identity_object(*, key: str, payload: Mapping[str, Any]) -> dict[str, object]:
    return {
        "key": key,
        "payload": dict(payload),
    }


def _repo_relative_path(path: str | Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (_REPO_ROOT / path_obj).resolve()


def _file_sha1(path: str | Path) -> str:
    digest = hashlib.sha1()
    with _repo_path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _pts_cem_victim_attack_identity_context(
    config: Config,
    *,
    run_type: str,
    target_item: int,
    target_metadata: Mapping[str, object],
) -> dict[str, object] | None:
    if run_type not in _PTS_CEM_TARGET_LEVEL_VICTIM_RUN_TYPES:
        return None
    construction_method = str(
        target_metadata.get("pts_construction_method", "")
    ).strip()
    if not construction_method:
        return None

    shared_cache_key = target_metadata.get("shared_pts_cem_cache_key")
    if shared_cache_key is None:
        return None
    sessions_sha1 = target_metadata.get("selected_pts_cem_sessions_sha1")
    if sessions_sha1 is None:
        sessions_path = target_metadata.get("pts_best_candidate_sessions_path")
        if sessions_path is None:
            raise ValueError(
                "PTS-CEM victim cache identity requires selected sessions hash or path."
            )
        sessions_sha1 = _file_sha1(str(sessions_path))

    seed_alignment_keys = (
        "pts_cem_surrogate_seed_alignment_mode",
        "pts_cem_surrogate_seed_alignment_target_victim_name",
        "configured_surrogate_train_seed",
        "configured_victim_train_seed",
        "resolved_surrogate_effective_seed",
        "resolved_victim_effective_seed",
        "surrogate_victim_seed_aligned",
    )
    seed_alignment = {
        key: target_metadata[key]
        for key in seed_alignment_keys
        if key in target_metadata and target_metadata[key] is not None
    }
    return {
        "schema_version": "pts_cem_target_level_victim_attack_identity_v1",
        "attack_type": "pts_cem",
        "run_type": run_type,
        "construction_method": construction_method,
        "dataset_split_identity": split_key(config),
        "poison_train_only": bool(config.data.poison_train_only),
        "target_item": int(target_item),
        "shared_pts_cem_cache_key": str(shared_cache_key),
        "selected_pts_cem_sessions_sha1": str(sessions_sha1),
        "source_candidate_rank": (
            None
            if target_metadata.get("source_candidate_rank") is None
            else int(target_metadata["source_candidate_rank"])
        ),
        "source_candidate_key": (
            None
            if target_metadata.get("source_candidate_key") is None
            else str(target_metadata["source_candidate_key"])
        ),
        "seed_alignment": seed_alignment,
    }


def _evaluation_metrics_config_payload(config: Config) -> dict[str, object]:
    return {
        "topk": [int(k) for k in config.evaluation.topk],
        "targeted_metrics": list(config.evaluation.targeted_metrics),
        "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
    }


def _pts_cem_victim_identity_metrics_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None,
    victim_attack_identity_context: Mapping[str, object],
) -> dict[str, object]:
    return {
        "victim_prediction_attack_identity_mode": "pts_cem_target_level_construction",
        "shared_pts_cem_cache_key": victim_attack_identity_context[
            "shared_pts_cem_cache_key"
        ],
        "selected_pts_cem_sessions_sha1": victim_attack_identity_context[
            "selected_pts_cem_sessions_sha1"
        ],
        "target_item": int(victim_attack_identity_context["target_item"]),
        "source_candidate_rank": victim_attack_identity_context.get(
            "source_candidate_rank"
        ),
        "source_candidate_key": victim_attack_identity_context.get(
            "source_candidate_key"
        ),
        "evaluation": _evaluation_metrics_config_payload(config),
        "old_run_level_attack_key": attack_key(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
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
                key=shared_attack_artifact_key(
                    config,
                    run_type=run_type,
                    require_poison_runner=shared_attack_identity_requires_poison_runner(run_type),
                ),
                payload=shared_attack_artifact_key_payload(
                    config,
                    run_type=run_type,
                    require_poison_runner=shared_attack_identity_requires_poison_runner(run_type),
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


def _stable_run_group_metadata(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "container_model": "appendable_experiment_container",
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


def _execution_request_metadata(config: Config, *, run_type: str) -> dict[str, object]:
    sampled_mode = config.targets.mode == "sampled"
    explicit_targets = [int(item) for item in config.targets.explicit_list]
    requested_target_count = (
        int(config.targets.count) if sampled_mode else int(len(explicit_targets))
    )
    return {
        "request_model": "append_invocation",
        "run_type": run_type,
        "execution_semantics": "append_against_stable_run_group",
        "requested_victims": list(config.victims.enabled),
        "target_request": {
            "mode": config.targets.mode,
            "bucket": config.targets.bucket if sampled_mode else None,
            "explicit_list": explicit_targets,
            "requested_target_count": requested_target_count,
            "count_interpretation": (
                "prefix_length" if sampled_mode else "explicit_cohort_size"
            ),
        },
        "legacy_runtime_flags": {
            "reuse_saved_targets": bool(config.targets.reuse_saved_targets),
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
    if not isinstance(target_selection_identity, dict) or not isinstance(
        evaluation_identity, dict
    ):
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

    stable_run_group = derived.get("stable_run_group")
    if isinstance(stable_run_group, Mapping):
        legacy_identities = stable_run_group.get("legacy_identities")
    else:
        legacy_identities = derived.get("legacy_identities")
    if isinstance(legacy_identities, Mapping):
        target_selection_key_value = _extract_identity_key(
            legacy_identities.get(
                "target_selection_identity",
                legacy_identities.get("target_selection_key"),
            )
        )
        evaluation_key_value = _extract_identity_key(
            legacy_identities.get(
                "evaluation_identity", legacy_identities.get("evaluation_key")
            )
        )
    else:
        target_selection_key_value = _extract_identity_key(
            derived.get("target_selection_key")
        )
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

    if (
        metadata_paths["run_coverage"].exists()
        and metadata_paths["execution_log"].exists()
    ):
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


def _load_or_init_artifact_manifest(
    config: Config,
    *,
    context: RunContext,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    manifest = _initial_artifact_manifest(
        config,
        context=context,
        run_type=run_type,
        metadata_paths=dict(metadata_paths),
        attack_identity_context=attack_identity_context,
    )
    existing = load_json(metadata_paths["artifact_manifest"])
    if not isinstance(existing, dict):
        return manifest

    for key in ("victims", "generated_configs"):
        existing_value = existing.get(key)
        if isinstance(existing_value, dict):
            manifest[key] = existing_value
    existing_output_files = existing.get("output_files")
    if isinstance(existing_output_files, dict):
        summary_output = existing_output_files.get("summary")
        if isinstance(summary_output, str) and summary_output.strip():
            manifest["output_files"]["summary"] = summary_output
    return manifest


def _validate_phase5_victim_request(
    run_coverage: Mapping[str, Any],
    *,
    requested_victims: list[str],
) -> None:
    existing_victims_payload = run_coverage.get("victims", {})
    if not isinstance(existing_victims_payload, Mapping):
        raise ValueError("run_coverage.json must contain a victims object.")
    if len(set(requested_victims)) != len(requested_victims):
        raise RuntimeError(
            f"Requested victims must be unique per invocation. Received {requested_victims}."
        )
    missing_victims = [
        victim_name
        for victim_name in requested_victims
        if victim_name not in existing_victims_payload
    ]
    if missing_victims:
        raise RuntimeError(
            "run_coverage.json does not include registry entries for all requested victims "
            f"after Phase 5 victim-append initialization. Missing victims={missing_victims}."
        )


def _group_planned_cells_by_target(
    planned_cells: list[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for cell in planned_cells:
        target_item = int(cell["target_item"])
        grouped.setdefault(target_item, []).append(dict(cell))
    return grouped


def _limit_planned_cells_by_target(
    planned_cells: list[dict[str, Any]],
    *,
    max_targets: int | None,
) -> list[dict[str, Any]]:
    if max_targets is None:
        return list(planned_cells)
    limit = int(max_targets)
    if limit <= 0:
        raise ValueError("max_targets_per_execution must be positive when provided.")
    selected_targets = set(_unique_targets(planned_cells)[:limit])
    return [
        dict(cell)
        for cell in planned_cells
        if int(cell["target_item"]) in selected_targets
    ]


def _execution_timestamp() -> str:
    return datetime.now(_PROGRESS_ZONE).isoformat()


def _append_execution_record(
    execution_log: dict[str, object],
    config: Config,
    *,
    run_type: str,
    requested_target_items: list[int],
    requested_victims: list[str],
    planned_cells: list[dict[str, Any]],
    skipped_completed_cells: list[dict[str, Any]],
    existing_run_coverage_payload: Mapping[str, Any] | None,
    metadata_paths: Mapping[str, Path],
) -> dict[str, object]:
    executions = execution_log.get("executions")
    if not isinstance(executions, list):
        raise ValueError("execution_log.json must contain an executions list.")
    timestamp = _execution_timestamp()
    requested_target_count = (
        int(config.targets.count)
        if config.targets.mode == "sampled"
        else int(len(requested_target_items))
    )
    plan_summary = _execution_plan_summary(
        existing_run_coverage_payload,
        requested_target_items=requested_target_items,
        requested_victims=requested_victims,
        planned_cells=planned_cells,
    )
    execution_record = {
        "execution_id": uuid4().hex,
        "mode": plan_summary["mode"],
        "run_type": run_type,
        "requested_target_count": requested_target_count,
        "requested_target_items": [int(item) for item in requested_target_items],
        "requested_victims": list(requested_victims),
        "added_target_items": [
            int(item) for item in plan_summary["added_target_items"]
        ],
        "added_victims": list(plan_summary["added_victims"]),
        "planned_cells": [
            {
                "target_item": int(cell["target_item"]),
                "victim_name": str(cell["victim_name"]),
                "status": "requested",
                "requested_at": timestamp,
                "started_at": None,
                "completed_at": None,
                "failed_at": None,
                "error_type": None,
                "error": None,
            }
            for cell in planned_cells
        ],
        "planned_target_items": [int(item) for item in _unique_targets(planned_cells)],
        "planned_cell_count": int(len(planned_cells)),
        "skipped_completed_cells": [
            {
                "target_item": int(cell["target_item"]),
                "victim_name": str(cell["victim_name"]),
                "skipped_at": timestamp,
            }
            for cell in skipped_completed_cells
        ],
        "skipped_completed_cell_count": int(len(skipped_completed_cells)),
        "completed_cells": [],
        "failed_cells": [],
        "completed_cell_count": 0,
        "failed_cell_count": 0,
        "status": "running",
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": None,
        "elapsed_seconds": 0.0,
    }
    executions.append(execution_record)
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])
    return execution_record


def _planned_cell_entry(
    execution_record: Mapping[str, Any],
    *,
    target_item: int,
    victim_name: str,
) -> dict[str, Any] | None:
    planned_cells = execution_record.get("planned_cells")
    if not isinstance(planned_cells, list):
        raise ValueError("execution_log execution records must contain planned_cells.")
    for entry in planned_cells:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("target_item", -1)) != int(target_item):
            continue
        if str(entry.get("victim_name")) != victim_name:
            continue
        return entry
    return None


def _record_execution_cell_pending(
    execution_log: dict[str, object],
    *,
    execution_record: dict[str, object],
    target_item: int,
    victim_name: str,
    metadata_paths: Mapping[str, Path],
) -> None:
    timestamp = _execution_timestamp()
    planned_entry = _planned_cell_entry(
        execution_record,
        target_item=target_item,
        victim_name=victim_name,
    )
    if planned_entry is None:
        raise ValueError(
            "execution_log planned_cells must include every scheduled cell before execution starts."
        )
    planned_entry["status"] = "pending"
    if planned_entry.get("started_at") is None:
        planned_entry["started_at"] = timestamp
    planned_entry["error_type"] = None
    planned_entry["error"] = None
    execution_record["updated_at"] = timestamp
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _unique_targets(cells: list[dict[str, Any]]) -> list[int]:
    ordered_targets: list[int] = []
    seen: set[int] = set()
    for cell in cells:
        target_item = int(cell["target_item"])
        if target_item in seen:
            continue
        seen.add(target_item)
        ordered_targets.append(target_item)
    return ordered_targets


def _execution_plan_summary(
    existing_run_coverage_payload: Mapping[str, Any] | None,
    *,
    requested_target_items: list[int],
    requested_victims: list[str],
    planned_cells: list[dict[str, Any]],
) -> dict[str, object]:
    existing_targets_order: list[int] = []
    existing_victims: list[str] = []
    if isinstance(existing_run_coverage_payload, Mapping):
        raw_targets_order = existing_run_coverage_payload.get("targets_order", [])
        if isinstance(raw_targets_order, list):
            existing_targets_order = [int(item) for item in raw_targets_order]
        raw_victims = existing_run_coverage_payload.get("victims", {})
        if isinstance(raw_victims, Mapping):
            existing_victims = [str(victim_name) for victim_name in raw_victims.keys()]

    existing_target_set = set(existing_targets_order)
    existing_victim_set = set(existing_victims)
    added_target_items = [
        int(target_item)
        for target_item in requested_target_items
        if int(target_item) not in existing_target_set
    ]
    added_victims = [
        victim_name
        for victim_name in requested_victims
        if victim_name not in existing_victim_set
    ]

    if not existing_targets_order and not existing_victims:
        mode = "initial_population"
    elif added_target_items and added_victims:
        mode = "target_and_victim_append"
    elif added_victims:
        mode = "victim_append"
    elif added_target_items:
        mode = "target_append"
    elif planned_cells:
        mode = "retry_incomplete_cells"
    else:
        mode = "noop"

    return {
        "mode": mode,
        "added_target_items": added_target_items,
        "added_victims": added_victims,
    }


def _record_execution_cell_completion(
    execution_log: dict[str, object],
    *,
    execution_record: dict[str, object],
    target_item: int,
    victim_name: str,
    metadata_paths: Mapping[str, Path],
) -> None:
    timestamp = _execution_timestamp()
    planned_entry = _planned_cell_entry(
        execution_record,
        target_item=target_item,
        victim_name=victim_name,
    )
    if planned_entry is not None:
        planned_entry["status"] = "completed"
        if planned_entry.get("started_at") is None:
            planned_entry["started_at"] = timestamp
        planned_entry["completed_at"] = timestamp
        planned_entry["error_type"] = None
        planned_entry["error"] = None
    completed_cells = execution_record.get("completed_cells")
    if not isinstance(completed_cells, list):
        raise ValueError(
            "execution_log execution records must contain completed_cells."
        )
    completed_cells.append(
        {
            "target_item": int(target_item),
            "victim_name": victim_name,
            "completed_at": timestamp,
        }
    )
    execution_record["completed_cell_count"] = int(len(completed_cells))
    execution_record["updated_at"] = timestamp
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _record_execution_cell_failure(
    execution_log: dict[str, object],
    *,
    execution_record: dict[str, object],
    target_item: int,
    victim_name: str,
    error: BaseException,
    metadata_paths: Mapping[str, Path],
) -> None:
    timestamp = _execution_timestamp()
    planned_entry = _planned_cell_entry(
        execution_record,
        target_item=target_item,
        victim_name=victim_name,
    )
    if planned_entry is not None:
        planned_entry["status"] = "failed"
        if planned_entry.get("started_at") is None:
            planned_entry["started_at"] = timestamp
        planned_entry["failed_at"] = timestamp
        planned_entry["error_type"] = type(error).__name__
        planned_entry["error"] = str(error)
    failed_cells = execution_record.get("failed_cells")
    if not isinstance(failed_cells, list):
        raise ValueError("execution_log execution records must contain failed_cells.")
    failed_cells.append(
        {
            "target_item": int(target_item),
            "victim_name": victim_name,
            "failed_at": timestamp,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    )
    execution_record["failed_cell_count"] = int(len(failed_cells))
    execution_record["updated_at"] = timestamp
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _mark_execution_completed(
    execution_log: dict[str, object],
    *,
    execution_record: dict[str, object],
    metadata_paths: Mapping[str, Path],
    elapsed_seconds: float,
) -> None:
    timestamp = _execution_timestamp()
    execution_record["status"] = "completed"
    execution_record["updated_at"] = timestamp
    execution_record["completed_at"] = timestamp
    execution_record["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _mark_execution_failed(
    execution_log: dict[str, object],
    *,
    execution_record: dict[str, object],
    error: BaseException,
    metadata_paths: Mapping[str, Path],
    elapsed_seconds: float,
) -> None:
    timestamp = _execution_timestamp()
    execution_record["status"] = "failed"
    execution_record["updated_at"] = timestamp
    execution_record["completed_at"] = timestamp
    execution_record["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    execution_record["error_type"] = type(error).__name__
    execution_record["error"] = str(error)
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _reconcile_interrupted_execution_records(
    execution_log: dict[str, object],
    *,
    metadata_paths: Mapping[str, Path],
) -> None:
    executions = execution_log.get("executions")
    if not isinstance(executions, list):
        raise ValueError("execution_log.json must contain an executions list.")
    timestamp = _execution_timestamp()
    changed = False
    for execution_record in executions:
        if not isinstance(execution_record, dict):
            continue
        if execution_record.get("status") != "running":
            continue
        execution_record["status"] = "interrupted"
        execution_record["updated_at"] = timestamp
        execution_record["completed_at"] = timestamp
        if not execution_record.get("error_type"):
            execution_record["error_type"] = "InterruptedExecution"
        if not execution_record.get("error"):
            execution_record["error"] = (
                "Detected a previously unfinished execution record during a later invocation."
            )
        changed = True
    if not changed:
        return
    execution_log["updated_at"] = timestamp
    save_execution_log(execution_log, metadata_paths["execution_log"])


def _artifact_path_if_exists(path: Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return _repo_relative_path(path_obj)


def _coverage_artifact_payload(
    artifacts: Mapping[str, Path] | None
) -> dict[str, str | None]:
    if artifacts is None:
        return {
            "metrics": None,
            "predictions": None,
            "train_history": None,
            "poisoned_train": None,
        }
    return {
        "metrics": _artifact_path_if_exists(artifacts.get("metrics")),
        "predictions": _artifact_path_if_exists(artifacts.get("predictions")),
        "train_history": _artifact_path_if_exists(artifacts.get("train_history")),
        "poisoned_train": _artifact_path_if_exists(artifacts.get("poisoned_train")),
    }


def _coverage_cell(
    run_coverage: dict[str, object],
    *,
    target_item: int,
    victim_name: str,
) -> dict[str, object]:
    cells_payload = run_coverage.get("cells")
    if not isinstance(cells_payload, dict):
        raise ValueError("run_coverage.json must contain a cells object.")
    target_cells = cells_payload.get(str(target_item))
    if not isinstance(target_cells, dict):
        raise ValueError(f"run_coverage.json cells[{target_item}] must be an object.")
    cell = target_cells.get(victim_name)
    if not isinstance(cell, dict):
        raise ValueError(
            f"run_coverage.json cells[{target_item}][{victim_name}] must be an object."
        )
    return cell


def _save_run_coverage_snapshot(
    run_coverage: dict[str, object],
    *,
    metadata_paths: Mapping[str, Path],
) -> None:
    sync_run_coverage_materialized_prefix(run_coverage)
    save_run_coverage(run_coverage, metadata_paths["run_coverage"])


def _mark_coverage_cell_completed(
    run_coverage: dict[str, object],
    *,
    target_item: int,
    victim_name: str,
    artifacts: Mapping[str, Path],
    execution_id: str,
) -> None:
    timestamp = _execution_timestamp()
    cell = _coverage_cell(
        run_coverage,
        target_item=target_item,
        victim_name=victim_name,
    )
    cell["status"] = "completed"
    cell["artifacts"] = _coverage_artifact_payload(artifacts)
    cell["error"] = None
    cell["last_execution_id"] = execution_id
    cell["completed_at"] = timestamp
    cell["failed_at"] = None
    cell["last_updated_at"] = timestamp
    run_coverage["updated_at"] = timestamp


def _mark_coverage_cell_pending(
    run_coverage: dict[str, object],
    *,
    target_item: int,
    victim_name: str,
    execution_id: str,
) -> None:
    timestamp = _execution_timestamp()
    cell = _coverage_cell(
        run_coverage,
        target_item=target_item,
        victim_name=victim_name,
    )
    cell["status"] = "pending"
    cell["error"] = None
    cell["last_execution_id"] = execution_id
    cell["last_started_at"] = timestamp
    cell["last_requested_at"] = timestamp
    cell["attempt_count"] = int(cell.get("attempt_count", 0)) + 1
    cell["last_updated_at"] = timestamp
    run_coverage["updated_at"] = timestamp


def _mark_coverage_cell_failed(
    run_coverage: dict[str, object],
    *,
    target_item: int,
    victim_name: str,
    artifacts: Mapping[str, Path] | None,
    error: BaseException,
    execution_id: str,
) -> None:
    timestamp = _execution_timestamp()
    cell = _coverage_cell(
        run_coverage,
        target_item=target_item,
        victim_name=victim_name,
    )
    cell["status"] = "failed"
    cell["artifacts"] = _coverage_artifact_payload(artifacts)
    cell["error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    cell["last_execution_id"] = execution_id
    cell["failed_at"] = timestamp
    cell["last_updated_at"] = timestamp
    run_coverage["updated_at"] = timestamp


def _validate_completed_local_artifacts(artifacts: Mapping[str, Path]) -> None:
    missing = [
        artifact_name
        for artifact_name in ("metrics", "predictions")
        if not Path(artifacts[artifact_name]).exists()
    ]
    if missing:
        raise RuntimeError(
            "Cell completion requires persisted local artifacts before coverage can be "
            f"marked completed. Missing artifacts: {', '.join(missing)}."
        )


def _write_legacy_summary_snapshot(
    config: Config,
    *,
    context: RunContext,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    summary_current: Mapping[str, Any] | None,
) -> dict[str, object]:
    summary_current_payload = (
        summary_current if isinstance(summary_current, Mapping) else {}
    )
    summary_payload = {
        "run_type": run_type,
        "run_group_key": summary_current_payload.get("run_group_key"),
        "target_cohort_key": summary_current_payload.get("target_cohort_key"),
        "is_snapshot": True,
        "snapshot_source": "summary_current",
        "target_items": list(summary_current_payload.get("target_items", [])),
        "victims": list(summary_current_payload.get("victims", [])),
        "fake_session_count": int(context.fake_session_count),
        "clean_session_count": int(len(context.clean_sessions)),
        "training": _training_summary(config, run_type=run_type),
        "targets": dict(summary_current_payload.get("targets", {})),
    }
    save_metrics(summary_payload, metadata_paths["summary"])
    return summary_payload


def _refresh_summary_snapshots(
    config: Config,
    *,
    context: RunContext,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    run_coverage: Mapping[str, Any],
    artifact_manifest: dict[str, object],
    attack_identity_context: Mapping[str, object] | None = None,
) -> dict[str, object]:
    summary_current = rebuild_summary_current(
        config,
        run_type=run_type,
        metadata_paths=metadata_paths,
        run_coverage=run_coverage,
        attack_identity_context=attack_identity_context,
    )
    summary = _write_legacy_summary_snapshot(
        config,
        context=context,
        run_type=run_type,
        metadata_paths=metadata_paths,
        summary_current=summary_current,
    )
    artifact_manifest["output_files"]["summary"] = _repo_relative_path(
        metadata_paths["summary"]
    )
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])
    return summary


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
            "stable_run_group": _stable_run_group_metadata(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "execution_request": _execution_request_metadata(
                config,
                run_type=run_type,
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
        "stable_run_group": _stable_run_group_metadata(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
        "execution_request": _execution_request_metadata(
            config,
            run_type=run_type,
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
        "shared_dir": _repo_relative_path(context.shared_paths["target_shared_dir"]),
        "config_snapshot": _repo_relative_path(
            context.shared_paths["target_config_snapshot"]
        ),
        "selected_targets": _repo_relative_path(
            context.shared_paths["selected_targets"]
        ),
        "target_selection_meta": _repo_relative_path(
            context.shared_paths["target_selection_meta"]
        ),
        "legacy_target_info": _repo_relative_path(context.shared_paths["target_info"]),
    }
    poison_artifact: dict[str, object] | None = None
    if run_type != "clean":
        poison_artifact = {
            "shared_dir": _repo_relative_path(
                context.shared_paths["attack_shared_dir"]
            ),
            "poison_model": _repo_relative_path(context.shared_paths["poison_model"]),
            "poison_model_identity": _repo_relative_path(
                context.shared_paths["poison_model_identity"]
            ),
            "fake_sessions": _repo_relative_path(context.shared_paths["fake_sessions"]),
            "poison_train_history": _repo_relative_path(
                context.shared_paths["poison_train_history"]
            ),
            "config_snapshot": _repo_relative_path(
                context.shared_paths["attack_config_snapshot"]
            ),
        }
    derived_identity: dict[str, object] | None = None
    if attack_identity_context is not None:
        derived_identity = {
            "final_attack_identity_context": dict(attack_identity_context),
        }
    return {
        "run_type": run_type,
        "identities": _stable_run_group_metadata(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
        "execution_request": _execution_request_metadata(
            config,
            run_type=run_type,
        ),
        "shared_artifacts": {
            "canonical_split": {
                key: _repo_relative_path(path) for key, path in canonical_paths.items()
            },
            "target_cohort": {
                "shared_dir": _repo_relative_path(
                    context.shared_paths["target_cohort_dir"]
                ),
                "target_registry": _repo_relative_path(
                    context.shared_paths["target_registry"]
                ),
            },
            "legacy_target_selection": target_selection_artifact,
            "poison_artifact": poison_artifact,
        },
        "run_group_artifacts": {
            "run_root": _repo_relative_path(metadata_paths["run_root"]),
            "resolved_config": _repo_relative_path(metadata_paths["resolved_config"]),
            "key_payloads": _repo_relative_path(metadata_paths["key_payloads"]),
            "artifact_manifest": _repo_relative_path(
                metadata_paths["artifact_manifest"]
            ),
            "run_coverage": _repo_relative_path(metadata_paths["run_coverage"]),
            "execution_log": _repo_relative_path(metadata_paths["execution_log"]),
            "summary_current": _repo_relative_path(metadata_paths["summary_current"]),
            "progress": _repo_relative_path(metadata_paths["progress"]),
            "legacy_summary": _repo_relative_path(metadata_paths["summary"]),
        },
        "derived_identity": derived_identity,
        "victims": {},
        "generated_configs": {},
        "output_files": {
            "run_root": _repo_relative_path(metadata_paths["run_root"]),
            "resolved_config": _repo_relative_path(metadata_paths["resolved_config"]),
            "key_payloads": _repo_relative_path(metadata_paths["key_payloads"]),
            "artifact_manifest": _repo_relative_path(
                metadata_paths["artifact_manifest"]
            ),
            "run_coverage": _repo_relative_path(metadata_paths["run_coverage"]),
            "execution_log": _repo_relative_path(metadata_paths["execution_log"]),
            "summary_current": _repo_relative_path(metadata_paths["summary_current"]),
            "progress": _repo_relative_path(metadata_paths["progress"]),
            "summary": _repo_relative_path(metadata_paths["summary"]),
        },
    }


def _initial_progress_payload(
    config: Config,
    *,
    run_type: str,
    metadata_paths: Mapping[str, Path],
) -> dict[str, object]:
    timestamp = _progress_timestamp()
    return {
        "run_type": run_type,
        "timezone": PROGRESS_TIMEZONE,
        "is_authoritative": False,
        "purpose": "debug_progress_only",
        "authoritative_state": {
            "run_coverage": _repo_relative_path(metadata_paths["run_coverage"]),
            "execution_log": _repo_relative_path(metadata_paths["execution_log"]),
        },
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


def _populate_progress_plan_from_cells(
    progress_payload: dict[str, object],
    *,
    requested_target_items: list[int],
    requested_victims: list[str],
    planned_cells: list[dict[str, Any]],
    skipped_completed_cells: list[dict[str, Any]],
    elapsed_seconds: float,
) -> None:
    timestamp = _progress_timestamp()
    runs: list[dict[str, object]] = []
    target_index_lookup = {
        int(target_item): int(index)
        for index, target_item in enumerate(requested_target_items, start=1)
    }
    victim_index_by_target: dict[int, int] = {}
    for overall_index, cell in enumerate(planned_cells, start=1):
        target_item = int(cell["target_item"])
        victim_name = str(cell["victim_name"])
        victim_index = victim_index_by_target.get(target_item, 0) + 1
        victim_index_by_target[target_item] = victim_index
        runs.append(
            {
                "overall_index": int(overall_index),
                "target_index": target_index_lookup.get(target_item, 0),
                "target_item": target_item,
                "victim_index": int(victim_index),
                "victim_name": victim_name,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "reused_predictions": None,
            }
        )
    progress_payload["status"] = "running"
    progress_payload["updated_at"] = timestamp
    progress_payload["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    progress_payload["total_targets"] = int(len(requested_target_items))
    progress_payload["total_victims"] = int(len(requested_victims))
    progress_payload["total_runs"] = int(len(runs))
    progress_payload["completed_runs"] = 0
    progress_payload["target_items"] = [int(item) for item in requested_target_items]
    progress_payload["requested_victims"] = list(requested_victims)
    progress_payload["planned_cells"] = [dict(cell) for cell in planned_cells]
    progress_payload["skipped_completed_cells"] = [
        dict(cell) for cell in skipped_completed_cells
    ]
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
        if isinstance(entry, dict) and int(entry.get("overall_index", -1)) == int(
            overall_index
        ):
            return entry
    for entry in runs:
        if isinstance(entry, dict) and int(entry.get("overall_index", -1)) == int(
            overall_index
        ):
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
    run_type: str,
    run_coverage: Mapping[str, Any],
    victim_name: str,
    canonical_dataset: CanonicalDataset,
    poisoned_sessions: list[list[int]],
    poisoned_labels: list[int],
    raw_fake_sessions: list[list[int]],
    run_dir: Path,
    poisoned_train_path: Path,
    target_item: int,
    eval_topk: tuple[int, ...] | list[int],
    srg_nn_export_paths: dict[str, Path] | None,
    predictions_path: Path,
    artifacts: dict[str, Path],
    victim_attack_identity_context: Mapping[str, object] | None = None,
    wearec_provenance_resolver: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[VictimExecutionResult, bool]:
    prepared_wearec = None
    if victim_name == "wearec":
        prepare_kwargs: dict[str, Any] = {}
        if wearec_provenance_resolver is not None:
            prepare_kwargs["provenance_resolver"] = wearec_provenance_resolver
        prepared_wearec = prepare_wearec_execution(
            config,
            run_type=run_type,
            canonical_dataset=canonical_dataset,
            train_prefixes=poisoned_sessions,
            train_labels=poisoned_labels,
            run_dir=run_dir,
            requested_topk=max(int(value) for value in eval_topk),
            target_item=target_item,
            attack_identity_context=victim_attack_identity_context,
            **prepare_kwargs,
        )
        identity = prepared_wearec["identity"]
        shared_base = shared_victim_dir(
            config,
            run_type=run_type,
            target_id=target_item,
            victim_name=victim_name,
            victim_attack_identity_context=identity,
            victim_effective_train_seed=int(identity["effective_config"]["seed"]),
        )
        artifacts.update(
            {
                "shared_dir": shared_base,
                "shared_predictions": shared_base / "predictions.json",
                "shared_train_history": shared_base / "train_history.json",
                "shared_execution_result": shared_base / "execution_result.json",
                "shared_poisoned_train": shared_base / "poisoned_train.txt",
                "shared_wearec_raw_predictions": shared_base / "wearec_topk_raw.json",
                "shared_wearec_checkpoint": shared_base / "wearec_checkpoint.pt",
                "shared_wearec_log": shared_base / "wearec_stdout.log",
                "shared_artifact_manifest": shared_base / "artifact_manifest.json",
            }
        )
    _guard_clean_shared_cache_bootstrap(
        run_type=run_type,
        run_coverage=run_coverage,
        victim_name=victim_name,
        artifacts=artifacts,
    )
    reused = _load_shared_victim_result(
        config,
        run_type=run_type,
        victim_name=victim_name,
        target_item=target_item,
        run_dir=run_dir,
        artifacts=artifacts,
        predictions_path=predictions_path,
        canonical_dataset=canonical_dataset,
        eval_topk=eval_topk,
        wearec_identity=(
            prepared_wearec["identity"] if prepared_wearec is not None else None
        ),
        wearec_expected_labels=(
            prepared_wearec["test_labels"] if prepared_wearec is not None else None
        ),
    )
    if reused is None and victim_attack_identity_context is not None:
        reused = _load_legacy_pts_cem_victim_result(
            config,
            run_type=run_type,
            victim_name=victim_name,
            target_item=target_item,
            run_dir=run_dir,
            artifacts=artifacts,
            predictions_path=predictions_path,
            victim_attack_identity_context=victim_attack_identity_context,
        )
    if reused is not None:
        print(f"[victim] target={int(target_item)} model={victim_name} reuse")
        return reused, True

    victim_result = execute_single_victim(
        config,
        run_type=run_type,
        victim_name=victim_name,
        canonical_dataset=canonical_dataset,
        poisoned_sessions=poisoned_sessions,
        poisoned_labels=poisoned_labels,
        raw_fake_sessions=raw_fake_sessions,
        run_dir=run_dir,
        poisoned_train_path=poisoned_train_path,
        target_item=target_item,
        eval_topk=eval_topk,
        srg_nn_export_paths=srg_nn_export_paths,
        predictions_path=predictions_path,
        prepared_wearec=prepared_wearec,
    )
    _persist_shared_victim_result(
        run_type=run_type,
        victim_result=victim_result,
        artifacts=artifacts,
    )
    if victim_name == "wearec":
        persisted = _validate_wearec_shared_entry(
            artifacts=artifacts,
            identity=prepared_wearec["identity"],
            expected_labels=prepared_wearec["test_labels"],
        )
        if persisted is None:
            raise RuntimeError(
                "Persisted WEARec shared result failed complete-entry validation."
            )
    return victim_result, False


def _load_shared_victim_result(
    config: Config,
    *,
    run_type: str,
    victim_name: str,
    target_item: int,
    run_dir: Path,
    artifacts: dict[str, Path],
    predictions_path: Path,
    canonical_dataset: CanonicalDataset,
    eval_topk: Sequence[int],
    wearec_identity: Mapping[str, Any] | None = None,
    wearec_expected_labels: Sequence[int] | None = None,
) -> VictimExecutionResult | None:
    shared_predictions = artifacts["shared_predictions"]
    shared_execution_result = artifacts["shared_execution_result"]
    if not shared_predictions.exists() or not shared_execution_result.exists():
        return None
    validated_wearec: Mapping[str, Any] | None = None
    if victim_name == "wearec":
        if wearec_identity is None or wearec_expected_labels is None:
            return None
        validated_wearec = _validate_wearec_shared_entry(
            artifacts=artifacts,
            identity=wearec_identity,
            expected_labels=wearec_expected_labels,
        )
        if validated_wearec is None:
            return None
        execution_payload = validated_wearec["execution_payload"]
        predictions_payload = validated_wearec["predictions_payload"]
    elif victim_name == "freqrec":
        try:
            execution_payload = load_json(shared_execution_result)
            predictions_payload = load_json(shared_predictions)
        except (OSError, ValueError):
            return None
    else:
        execution_payload = load_json(shared_execution_result)
        predictions_payload = load_json(shared_predictions)
    if not isinstance(execution_payload, dict) or not isinstance(
        predictions_payload, dict
    ):
        if victim_name in {"freqrec", "wearec"}:
            return None
        raise ValueError("Shared victim artifacts are malformed.")
    if victim_name == "freqrec" and not _valid_freqrec_shared_result(
        config,
        run_type=run_type,
        predictions_payload=predictions_payload,
        execution_payload=execution_payload,
        canonical_dataset=canonical_dataset,
        eval_topk=eval_topk,
    ):
        return None
    _save_reused_predictions_payload(
        predictions_payload,
        predictions_path=predictions_path,
        target_item=target_item,
    )
    _copy_if_exists(artifacts["shared_train_history"], artifacts["train_history"])
    if victim_name == "wearec":
        _copy_if_exists(
            artifacts["shared_wearec_raw_predictions"],
            artifacts["wearec_raw_predictions"],
        )
        _copy_if_exists(
            artifacts["shared_wearec_checkpoint"], artifacts["wearec_checkpoint"]
        )
        _copy_if_exists(artifacts["shared_wearec_log"], artifacts["wearec_log"])
    poisoned_train_local: Path | None = None
    if artifacts["shared_poisoned_train"].exists():
        _copy_if_exists(artifacts["shared_poisoned_train"], artifacts["poisoned_train"])
        poisoned_train_local = artifacts["poisoned_train"]

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

    extra = _shared_execution_extra(
        run_type=run_type,
        extra=execution_payload.get("extra", {}),
    )
    return VictimExecutionResult(
        predictions=rankings,
        predictions_path=predictions_path,
        extra=extra,
        poisoned_train_path=poisoned_train_local,
    )


def _validate_wearec_shared_entry(
    *,
    artifacts: Mapping[str, Path],
    identity: Mapping[str, Any],
    expected_labels: Sequence[int],
) -> dict[str, Any] | None:
    """Validate a complete WEARec shared entry without mutating local state."""
    required_paths = {
        "predictions": artifacts["shared_predictions"],
        "raw_predictions": artifacts["shared_wearec_raw_predictions"],
        "checkpoint": artifacts["shared_wearec_checkpoint"],
        "log": artifacts["shared_wearec_log"],
        "execution_result": artifacts["shared_execution_result"],
    }
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required_paths.values()):
        return None
    manifest_path = artifacts["shared_artifact_manifest"]
    if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
        return None
    try:
        manifest = load_json(manifest_path)
        execution_payload = load_json(required_paths["execution_result"])
        predictions_payload = load_json(required_paths["predictions"])
    except (OSError, ValueError):
        return None
    effective = identity.get("effective_config")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(execution_payload, Mapping)
        or not isinstance(predictions_payload, Mapping)
        or manifest.get("victim") != "wearec"
        or manifest.get("scientific_identity") != identity
        or manifest.get("effective_config") != effective
        or execution_payload.get("scientific_identity") != identity
        or not isinstance(effective, Mapping)
    ):
        return None
    retained = manifest.get("retained_artifacts")
    if not isinstance(retained, Mapping):
        return None
    for name, path in required_paths.items():
        expected = retained.get(name)
        if not isinstance(expected, Mapping):
            return None
        try:
            actual = file_provenance(path)
        except (OSError, ValueError):
            return None
        if (
            type(expected.get("size")) is not int
            or expected["size"] != actual["size"]
            or expected.get("sha256") != actual["sha256"]
        ):
            return None
    try:
        raw_payload = load_wearec_prediction_payload(
            required_paths["raw_predictions"],
            item_count=int(identity["item_count"]),
            expected_labels=expected_labels,
            effective_config=effective,
            expected_training_mode=str(identity["training_mode"]),
            expected_dataset_name=str(identity["dataset_name"]),
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError):
        return None
    raw_rankings = [row["items"] for row in raw_payload["rankings"]]
    if (
        predictions_payload.get("victim") != "wearec"
        or type(predictions_payload.get("topk")) is not int
        or predictions_payload["topk"] != effective["requested_topk"]
        or predictions_payload.get("available") is not True
        or type(predictions_payload.get("count")) is not int
        or predictions_payload["count"] != len(expected_labels)
        or predictions_payload.get("rankings") != raw_rankings
    ):
        return None
    return {
        "manifest": manifest,
        "execution_payload": execution_payload,
        "predictions_payload": predictions_payload,
        "raw_payload": raw_payload,
    }


def _valid_freqrec_shared_result(
    config: Config,
    *,
    run_type: str,
    predictions_payload: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    canonical_dataset: CanonicalDataset,
    eval_topk: Sequence[int],
) -> bool:
    try:
        from attack.data.poisoned_dataset_builder import expand_session_to_samples

        item_count = len(canonical_dataset.item_map)
        if item_count <= 0 or set(canonical_dataset.item_map.values()) != set(
            range(1, item_count + 1)
        ):
            return False
        test_count = sum(
            len(expand_session_to_samples(session)[1])
            for session in canonical_dataset.test
        )
        requested_topk = max(int(value) for value in eval_topk)
        effective_topk = min(requested_topk, item_count)
        train = config.victims.params["freqrec"]["train"]
        monitor_cutoff = int(str(train["validation_metric"]).rsplit("@", 1)[1])
        expected_evaluation_topk = min(
            max(
                requested_topk,
                max(int(value) for value in train["metric_cutoffs"]),
                monitor_cutoff,
            ),
            item_count,
        )
        if predictions_payload.get("victim") != "freqrec":
            return False
        if type(predictions_payload.get("topk")) is not int:
            return False
        if int(predictions_payload["topk"]) != effective_topk:
            return False
        if type(predictions_payload.get("count")) is not int:
            return False
        if int(predictions_payload["count"]) != test_count:
            return False
        rankings = predictions_payload.get("rankings")
        if not isinstance(rankings, list) or len(rankings) != test_count:
            return False
        for row in rankings:
            if not isinstance(row, list) or len(row) != effective_topk:
                return False
            if any(type(item) is not int for item in row):
                return False
            if any(item < 1 or item > item_count for item in row):
                return False
            if len(set(row)) != len(row):
                return False

        extra = execution_payload.get("extra")
        if not isinstance(extra, Mapping):
            return False
        freqrec = extra.get("freqrec")
        export = extra.get("freqrec_export")
        if not isinstance(freqrec, Mapping) or not isinstance(export, Mapping):
            return False
        expected_seed = victim_effective_train_seed(
            config,
            victim_name="freqrec",
            run_type=run_type,
            target_item=int(predictions_payload.get("target_item", 0)),
        )
        exact = {
            "checkpoint_protocol": str(train["checkpoint_protocol"]),
            "epochs_requested": int(train["epochs"]),
            "epochs_completed": int(train["epochs"]),
            "requested_topk": requested_topk,
            "topk": effective_topk,
            "evaluation_topk": expected_evaluation_topk,
            "batch_size": int(train["batch_size"]),
            "seed": expected_seed,
            "prediction_count": test_count,
        }
        for key, expected in exact.items():
            value = freqrec.get(key)
            if isinstance(expected, int) and type(value) is not int:
                return False
            if value != expected:
                return False
        if type(freqrec.get("num_workers")) is not int or int(freqrec["num_workers"]) < 0:
            return False
        cached_validation_metric = freqrec.get("validation_metric")
        if (
            not isinstance(cached_validation_metric, str)
            or not cached_validation_metric.strip()
        ):
            return False
        expected_batch_count = math.ceil(test_count / int(train["batch_size"]))
        expected_final_batch_size = test_count - int(train["batch_size"]) * (
            expected_batch_count - 1
        )
        for key, expected in (
            ("batch_count", expected_batch_count),
            ("final_batch_size", expected_final_batch_size),
        ):
            if type(freqrec.get(key)) is not int or int(freqrec[key]) != expected:
                return False
        if freqrec.get("drop_last") is not False:
            return False
        if freqrec.get("train_sampler") != "seeded_random":
            return False
        if freqrec.get("evaluation_sampler") != "sequential":
            return False
        current_epoch = freqrec.get("current_epoch")
        selected_epoch = freqrec.get("selected_epoch")
        if type(current_epoch) is not int or type(selected_epoch) is not int:
            return False
        protocol = str(train["checkpoint_protocol"])
        if protocol == "fixed_epoch":
            if current_epoch != int(train["epochs"]) or selected_epoch != int(
                train["epochs"]
            ):
                return False
            if freqrec.get("best_epoch") is not None or freqrec.get("best_metric") is not None:
                return False
        elif protocol == "validation_best":
            if cached_validation_metric != str(train["validation_metric"]):
                return False
            best_epoch = freqrec.get("best_epoch")
            best_metric = freqrec.get("best_metric")
            if type(best_epoch) is not int:
                return False
            if isinstance(best_metric, bool) or not isinstance(best_metric, (int, float)):
                return False
            if not math.isfinite(float(best_metric)):
                return False
            if not (
                1
                <= current_epoch
                == selected_epoch
                == best_epoch
                <= int(train["epochs"])
            ):
                return False
        else:
            return False
        if type(export.get("item_count")) is not int or int(export["item_count"]) != item_count:
            return False
        if (
            type(export.get("test_example_count")) is not int
            or int(export["test_example_count"]) != test_count
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _load_legacy_pts_cem_victim_result(
    config: Config,
    *,
    run_type: str,
    victim_name: str,
    target_item: int,
    run_dir: Path,
    artifacts: dict[str, Path],
    predictions_path: Path,
    victim_attack_identity_context: Mapping[str, object],
) -> VictimExecutionResult | None:
    """Backfill the target-level PTS-CEM victim cache from older run artifacts.

    Older direct-action PTS-CEM runs keyed victim predictions by the batch-level
    attack_key, so changing target count or artifact persistence knobs could hide
    otherwise identical target/victim results. The selected-session hash is the
    stable target-level identity for those results.
    """
    expected_shared_key = str(
        victim_attack_identity_context.get("shared_pts_cem_cache_key", "")
    )
    expected_sessions_sha1 = str(
        victim_attack_identity_context.get("selected_pts_cem_sessions_sha1", "")
    )
    expected_victim_prediction_key = artifacts["shared_dir"].parent.parent.name
    if not expected_shared_key or not expected_sessions_sha1:
        return None

    for metrics_path in _iter_legacy_pts_cem_metrics_paths(
        config,
        target_item=target_item,
        victim_name=victim_name,
    ):
        if metrics_path.resolve() == artifacts["metrics"].resolve():
            continue
        metrics_payload = load_json(metrics_path)
        if not isinstance(metrics_payload, Mapping):
            continue
        if not _legacy_pts_cem_metrics_match(
            config,
            metrics_payload,
            victim_name=victim_name,
            target_item=target_item,
            expected_shared_key=expected_shared_key,
            expected_sessions_sha1=expected_sessions_sha1,
            expected_victim_prediction_key=expected_victim_prediction_key,
        ):
            continue
        predictions_source = _legacy_metrics_artifact_path(
            metrics_payload,
            "predictions_path",
            default=metrics_path.parent / "predictions.json",
        )
        if predictions_source is None or not predictions_source.exists():
            continue
        predictions_payload = load_json(predictions_source)
        if not isinstance(predictions_payload, Mapping):
            continue

        _save_reused_predictions_payload(
            predictions_payload,
            predictions_path=predictions_path,
            target_item=target_item,
        )
        train_history_source = metrics_path.parent / "train_history.json"
        _copy_if_exists(train_history_source, artifacts["train_history"])
        poisoned_train_source = _legacy_metrics_artifact_path(
            metrics_payload,
            "poisoned_train_path",
            default=metrics_path.parent / "poisoned_train.txt",
        )
        poisoned_train_local: Path | None = None
        if poisoned_train_source is not None and poisoned_train_source.exists():
            _copy_if_exists(poisoned_train_source, artifacts["poisoned_train"])
            poisoned_train_local = artifacts["poisoned_train"]

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
        result = VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={
                "reused_from_legacy_pts_cem_victim_cache": True,
                "legacy_pts_cem_metrics_path": _repo_relative_path(metrics_path),
                "legacy_pts_cem_predictions_path": _repo_relative_path(
                    predictions_source
                ),
            },
            poisoned_train_path=poisoned_train_local,
        )
        _persist_shared_victim_result(
            run_type=run_type,
            victim_result=result,
            artifacts=artifacts,
        )
        return result
    return None


def _iter_legacy_pts_cem_metrics_paths(
    config: Config,
    *,
    target_item: int,
    victim_name: str,
) -> list[Path]:
    runs_dataset_root = _repo_path(
        Path(config.artifacts.root)
        / config.artifacts.runs_dir
        / config.data.dataset_name
    )
    if not runs_dataset_root.exists():
        return []
    pattern = f"*/*/targets/{int(target_item)}/victims/{victim_name}/metrics.json"
    paths = [path for path in runs_dataset_root.glob(pattern) if path.is_file()]
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def _legacy_pts_cem_metrics_match(
    config: Config,
    metrics_payload: Mapping[str, object],
    *,
    victim_name: str,
    target_item: int,
    expected_shared_key: str,
    expected_sessions_sha1: str,
    expected_victim_prediction_key: str,
) -> bool:
    if str(metrics_payload.get("victim", "")) != victim_name:
        return False
    try:
        if int(metrics_payload.get("target_item")) != int(target_item):
            return False
    except (TypeError, ValueError):
        return False
    if str(metrics_payload.get("shared_pts_cem_cache_key", "")) != expected_shared_key:
        return False
    if (
        str(metrics_payload.get("selected_pts_cem_sessions_sha1", ""))
        != expected_sessions_sha1
    ):
        return False
    if (
        victim_name in {"tron", "mdhg"}
        and str(metrics_payload.get("victim_prediction_key", ""))
        != expected_victim_prediction_key
    ):
        return False
    metrics_training = metrics_payload.get("training")
    return metrics_training == _victim_training_summary(config, victim_name)


def _legacy_metrics_artifact_path(
    metrics_payload: Mapping[str, object],
    key: str,
    *,
    default: Path,
) -> Path | None:
    value = metrics_payload.get(key)
    if isinstance(value, str) and value.strip():
        return _repo_path(value)
    return default


def _persist_shared_victim_result(
    *,
    run_type: str,
    victim_result: VictimExecutionResult,
    artifacts: dict[str, Path],
) -> None:
    artifacts["shared_dir"].mkdir(parents=True, exist_ok=True)
    if "wearec" in victim_result.extra:
        _atomic_copy(artifacts["predictions"], artifacts["shared_predictions"])
    else:
        _copy_if_exists(artifacts["predictions"], artifacts["shared_predictions"])
    _copy_if_exists(artifacts["train_history"], artifacts["shared_train_history"])
    if "wearec" in victim_result.extra:
        _atomic_copy(
            artifacts["wearec_raw_predictions"],
            artifacts["shared_wearec_raw_predictions"],
        )
        _atomic_copy(
            artifacts["wearec_checkpoint"], artifacts["shared_wearec_checkpoint"]
        )
        _atomic_copy(artifacts["wearec_log"], artifacts["shared_wearec_log"])
    shared_poisoned_train = None
    if (
        victim_result.poisoned_train_path is not None
        and Path(victim_result.poisoned_train_path).exists()
    ):
        _copy_if_exists(
            Path(victim_result.poisoned_train_path), artifacts["shared_poisoned_train"]
        )
        shared_poisoned_train = str(artifacts["shared_poisoned_train"])
    execution_payload = {
            "extra": _shared_execution_extra(
                run_type=run_type,
                extra=victim_result.extra,
            ),
            "predictions_path": str(artifacts["shared_predictions"]),
            "train_history_path": (
                str(artifacts["shared_train_history"])
                if artifacts["shared_train_history"].exists()
                else None
            ),
            "poisoned_train_path": shared_poisoned_train,
        }
    if "wearec" in victim_result.extra:
        identity = victim_result.extra["wearec"]["scientific_identity"]
        execution_payload["scientific_identity"] = identity
    if "wearec" in victim_result.extra:
        _save_json_atomic(execution_payload, artifacts["shared_execution_result"])
    else:
        save_json(execution_payload, artifacts["shared_execution_result"])
    if "wearec" in victim_result.extra:
        retained = {
            "predictions": file_provenance(artifacts["shared_predictions"]),
            "raw_predictions": file_provenance(
                artifacts["shared_wearec_raw_predictions"]
            ),
            "checkpoint": file_provenance(artifacts["shared_wearec_checkpoint"]),
            "log": file_provenance(artifacts["shared_wearec_log"]),
            "execution_result": file_provenance(
                artifacts["shared_execution_result"]
            ),
        }
        _save_json_atomic(
            {
                "schema_version": 1,
                "victim": "wearec",
                "scientific_identity": identity,
                "effective_config": identity["effective_config"],
                "retained_artifacts": retained,
            },
            artifacts["shared_artifact_manifest"],
        )


def _cleanup_victim_intermediates_if_enabled(
    config: Config,
    *,
    victim_name: str,
    artifacts: Mapping[str, Path],
) -> None:
    if not bool(config.artifacts.cleanup_victim_intermediates):
        return
    run_dir = Path(artifacts["run_dir"])
    cleanup_paths = _victim_intermediate_cleanup_paths(run_dir, victim_name=victim_name)
    removed: list[str] = []
    for path in cleanup_paths:
        if _remove_optional_path(path):
            removed.append(_repo_relative_path(path))
    if removed:
        print(
            f"[victim] model={victim_name} cleanup_intermediates "
            f"removed={len(removed)}"
        )


def _victim_intermediate_cleanup_paths(
    run_dir: Path,
    *,
    victim_name: str,
) -> list[Path]:
    if victim_name == "miasrec":
        return [
            run_dir / "miasrec_topk_raw.json",
            run_dir / "export" / "miasrec",
            run_dir / "miasrec_checkpoints",
            run_dir / "miasrec_saved",
            run_dir / "miasrec_logs",
            run_dir / "miasrec_tensorboard",
        ]
    if victim_name == "tron":
        return [
            run_dir / "tron_topk_raw.json",
            run_dir / "export" / "tron",
            run_dir / "tron_logs",
            run_dir / "mlruns",
            run_dir / ".trash",
        ]
    if victim_name == "mdhg":
        return [
            run_dir / "mdhg_topk_raw.json",
            run_dir / "export" / "mdhg",
        ]
    if victim_name == "freqrec":
        return [
            run_dir / "freqrec_topk_raw.json",
            run_dir / "export" / "freqrec",
            run_dir / "freqrec_checkpoint.pt",
            run_dir / "freqrec_epoch_metrics.jsonl",
            run_dir / "freqrec_per_epoch_predictions",
        ]
    if victim_name == "wearec":
        return [
            run_dir / "export" / "wearec",
            run_dir / "wearec_internal_output",
        ]
    if victim_name == "srgnn":
        return [
            run_dir / "best_validation.pt",
        ]
    return []


def _remove_optional_path(path: Path) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    if target.is_dir():
        shutil.rmtree(target)
        return True
    if target.is_file():
        target.unlink()
        return True
    return False


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
            "run_dir": _repo_relative_path(artifacts["run_dir"]),
            "resolved_config": _repo_relative_path(artifacts["resolved_config"]),
            "config_snapshot": _repo_relative_path(artifacts["config_snapshot"]),
            "predictions": _repo_relative_path(artifacts["predictions"]),
            "metrics": _repo_relative_path(artifacts["metrics"]),
            "train_history": _repo_relative_path(artifacts["train_history"]),
            "poisoned_train": _repo_relative_path(artifacts["poisoned_train"]),
        },
        "shared": {
            "shared_dir": _repo_relative_path(artifacts["shared_dir"]),
            "predictions": _repo_relative_path(artifacts["shared_predictions"]),
            "train_history": _repo_relative_path(artifacts["shared_train_history"]),
            "execution_result": _repo_relative_path(
                artifacts["shared_execution_result"]
            ),
            "poisoned_train": _repo_relative_path(artifacts["shared_poisoned_train"]),
        },
    }
    if victim_name == "wearec":
        target_payload[victim_name]["local"].update(
            {
                "raw_predictions": _repo_relative_path(
                    artifacts["wearec_raw_predictions"]
                ),
                "checkpoint": _repo_relative_path(artifacts["wearec_checkpoint"]),
                "log": _repo_relative_path(artifacts["wearec_log"]),
            }
        )
        target_payload[victim_name]["shared"].update(
            {
                "raw_predictions": _repo_relative_path(
                    artifacts["shared_wearec_raw_predictions"]
                ),
                "checkpoint": _repo_relative_path(
                    artifacts["shared_wearec_checkpoint"]
                ),
                "log": _repo_relative_path(artifacts["shared_wearec_log"]),
                "artifact_manifest": _repo_relative_path(
                    artifacts["shared_artifact_manifest"]
                ),
            }
        )
    generated_configs = artifact_manifest.setdefault("generated_configs", {})
    if isinstance(generated_configs, dict) and victim_result.extra:
        generated_configs[f"{target_key}:{victim_name}"] = victim_result.extra


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _atomic_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Required retained artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        shutil.copyfile(source, temp)
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _save_json_atomic(payload: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, destination)
    finally:
        if temp.exists():
            temp.unlink()


def _shared_execution_extra(
    *,
    run_type: str,
    extra: Any,
) -> dict[str, object]:
    if not isinstance(extra, Mapping):
        return {}
    if run_type != "clean":
        return dict(extra)
    sanitized = _sanitize_clean_shared_extra(extra)
    if sanitized is _DROP_SHARED_METADATA or not isinstance(sanitized, dict):
        return {}
    return sanitized


def _sanitize_clean_shared_extra(
    value: Any,
    *,
    parent_key: str | None = None,
) -> Any:
    if isinstance(value, Path):
        return _DROP_SHARED_METADATA
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            cleaned = _sanitize_clean_shared_extra(item, parent_key=key)
            if cleaned is _DROP_SHARED_METADATA:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            sanitized[key] = cleaned
        return sanitized if sanitized else _DROP_SHARED_METADATA
    if isinstance(value, (list, tuple)):
        sanitized_items = []
        for item in value:
            cleaned = _sanitize_clean_shared_extra(item, parent_key=parent_key)
            if cleaned is _DROP_SHARED_METADATA:
                continue
            if isinstance(cleaned, dict) and not cleaned:
                continue
            if isinstance(cleaned, list) and not cleaned:
                continue
            sanitized_items.append(cleaned)
        return sanitized_items if sanitized_items else _DROP_SHARED_METADATA
    if isinstance(value, str) and _looks_like_local_path_metadata(
        value=value, key=parent_key
    ):
        return _DROP_SHARED_METADATA
    return value


def _looks_like_local_path_metadata(
    *,
    value: str,
    key: str | None,
) -> bool:
    key_lower = (key or "").strip().lower()
    if any(
        token in key_lower for token in ("path", "dir", "root", "checkpoint", "log")
    ):
        return True
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return False
    if normalized.startswith(("outputs/", "./", "../", "/")):
        return True
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        return True
    if "/" not in normalized:
        return False
    if normalized.count("/") >= 2:
        return True
    return bool(Path(normalized).suffix)


def _guard_clean_shared_cache_bootstrap(
    *,
    run_type: str,
    run_coverage: Mapping[str, Any],
    victim_name: str,
    artifacts: Mapping[str, Path],
) -> None:
    if run_type != "clean":
        return
    if (
        artifacts["shared_predictions"].exists()
        and artifacts["shared_execution_result"].exists()
    ):
        return
    if not _coverage_has_completed_victim_cells(run_coverage, victim_name=victim_name):
        return
    raise RuntimeError(
        "Clean append would mix existing target-scoped victim results with the new "
        "target-agnostic shared clean cache because completed cells already exist for "
        f"victim '{victim_name}' but {artifacts['shared_predictions']} is missing. "
        "Regenerate or migrate this clean run group before appending new targets or victims."
    )


def _coverage_has_completed_victim_cells(
    run_coverage: Mapping[str, Any],
    *,
    victim_name: str,
) -> bool:
    cells_payload = run_coverage.get("cells", {})
    if not isinstance(cells_payload, Mapping):
        return False
    for target_cells in cells_payload.values():
        if not isinstance(target_cells, Mapping):
            continue
        cell = target_cells.get(victim_name)
        if isinstance(cell, Mapping) and cell.get("status") == "completed":
            return True
    return False


def _save_reused_predictions_payload(
    predictions_payload: Mapping[str, Any],
    *,
    predictions_path: Path,
    target_item: int,
) -> None:
    local_payload = dict(predictions_payload)
    local_payload["target_item"] = int(target_item)
    save_json(local_payload, predictions_path)


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
