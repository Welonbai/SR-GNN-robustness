from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping

from attack.common.artifact_io import (
    load_execution_log,
    load_json,
    load_fake_sessions,
    load_poison_model,
    load_run_coverage,
    load_selected_targets,
    load_target_info,
    load_target_registry,
    save_execution_log,
    save_fake_sessions,
    save_json,
    save_poison_model,
    save_run_coverage,
    save_selected_targets,
    save_summary_current,
    save_target_selection_meta,
    save_target_info,
    save_target_registry,
)
from attack.common.paths import (
    TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    TARGET_COHORT_SELECTION_POLICY_VERSION,
    poison_model_key,
    poison_model_key_payload,
    run_group_key,
    shared_attack_dir,
    shared_artifact_paths,
    split_key,
    target_cohort_key,
    target_selection_key,
    victim_prediction_key,
)
from attack.common.seed import derive_seed, set_seed
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.data.session_stats import SessionStats, compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.data.target_selector import (
    sample_many_from_all,
    sample_many_from_popular,
    sample_many_from_unpopular,
)
from attack.generation.fake_session_generator import FakeSessionGenerator
from attack.generation.fake_session_parameter_sampler import FakeSessionParameterSampler
from attack.models.poison.srgnn_poison_runner import SRGNNPoisonRunner
from attack.models.srgnn_validation_training import (
    srgnn_validation_train_history_extra,
    train_srgnn_validation_best,
)
from attack.pipeline.core.train_history import save_train_history
from attack.common.srgnn_training_protocol import srgnn_validation_best_enabled

from attack.common.config import Config


_REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_COVERAGE_MATERIALIZED_PREFIX_FIELD = "materialized_target_prefix_count"


def build_srgnn_opt_from_train_config(train_config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=int(train_config["batch_size"]),
        hiddenSize=int(train_config["hidden_size"]),
        epoch=int(train_config["epochs"]),
        lr=float(train_config["lr"]),
        lr_dc=float(train_config["lr_dc"]),
        lr_dc_step=int(train_config["lr_dc_step"]),
        l2=float(train_config["l2"]),
        step=int(train_config["step"]),
        patience=int(train_config["patience"]),
        nonhybrid=bool(train_config["nonhybrid"]),
    )


def _fake_session_count(ratio: float, clean_count: int) -> int:
    return fake_session_count_from_ratio(ratio, clean_count)


def fake_session_count_from_ratio(ratio: float, clean_count: int) -> int:
    if ratio <= 0:
        return 0
    count = int(round(clean_count * ratio))
    return max(1, count)


def _fake_session_generation_ratio(config: Config, *, run_type: str) -> float:
    if run_type in {
        TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
        TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    }:
        carrier_selection = config.attack.carrier_selection
        if carrier_selection is None:
            raise ValueError("TACS-NZ requires attack.carrier_selection to be configured.")
        if not carrier_selection.enabled:
            raise ValueError("TACS-NZ requires attack.carrier_selection.enabled == true.")
        return float(carrier_selection.candidate_pool_size)
    return float(config.attack.size)


def build_clean_pairs(canonical_dataset: CanonicalDataset) -> tuple[list[list[int]], list[int]]:
    clean_sessions: list[list[int]] = []
    clean_labels: list[int] = []
    for session in canonical_dataset.train_sub:
        prefixes, labels = expand_session_to_samples(session)
        clean_sessions.extend(prefixes)
        clean_labels.extend(labels)
    return clean_sessions, clean_labels


def _resolve_target_items(stats: SessionStats, config: Config) -> list[int]:
    mode = config.targets.mode
    if mode == "explicit_list":
        return [int(item) for item in config.targets.explicit_list]
    if mode == "sampled":
        seed = config.seeds.target_selection_seed
        count = config.targets.count
        if config.targets.bucket == "popular":
            return sample_many_from_popular(stats, seed=seed, count=count)
        if config.targets.bucket == "unpopular":
            return sample_many_from_unpopular(stats, seed=seed, count=count)
        if config.targets.bucket == "all":
            return sample_many_from_all(stats, seed=seed, count=count)
        raise ValueError("Unsupported targets.bucket.")
    raise ValueError("Unsupported targets.mode.")


def _popular_pool(stats: SessionStats) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    avg_count = stats.total_items / float(len(stats.item_counts))
    pool = [int(item) for item, count in stats.item_counts.items() if count > avg_count]
    if not pool:
        raise ValueError("Popular pool is empty under item_count > average_count.")
    return pool


def _unpopular_pool(stats: SessionStats, *, threshold: int = 10) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    pool = [int(item) for item, count in stats.item_counts.items() if count < threshold]
    if not pool:
        raise ValueError("Unpopular pool is empty under item_count < 10.")
    return pool


def _all_pool(stats: SessionStats) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    return [int(item) for item in stats.item_counts]


def _target_candidate_pool(stats: SessionStats, config: Config) -> list[int]:
    if config.targets.mode == "explicit_list":
        return [int(item) for item in config.targets.explicit_list]
    if config.targets.bucket == "popular":
        return _popular_pool(stats)
    if config.targets.bucket == "unpopular":
        return _unpopular_pool(stats)
    if config.targets.bucket == "all":
        return _all_pool(stats)
    raise ValueError("Unsupported targets.bucket.")


def _candidate_pool_summary(stats: SessionStats, pool: list[int]) -> dict[str, Any]:
    counts = [int(stats.item_counts[item]) for item in pool if item in stats.item_counts]
    summary: dict[str, Any] = {
        "preview": [int(item) for item in sorted(pool)[:10]],
    }
    if counts:
        summary.update(
            {
                "min_item_count": int(min(counts)),
                "max_item_count": int(max(counts)),
                "avg_item_count": float(sum(counts) / len(counts)),
            }
        )
    return summary


def _target_selection_meta_payload(
    stats: SessionStats,
    config: Config,
    *,
    candidate_pool: list[int],
) -> dict[str, Any]:
    explicit_list = [int(item) for item in config.targets.explicit_list]
    bucket = config.targets.bucket if config.targets.mode == "sampled" else None
    count = int(config.targets.count) if config.targets.mode == "sampled" else None
    return {
        "target_selection_seed": int(config.seeds.target_selection_seed),
        "targets": {
            "mode": config.targets.mode,
            "bucket": bucket,
            "count": count,
            "explicit_list": explicit_list,
        },
        "bucket": bucket,
        "count": count,
        "explicit_list": explicit_list,
        "candidate_pool_size": int(len(candidate_pool)),
        "candidate_pool_summary": _candidate_pool_summary(stats, candidate_pool),
        "target_selection_key": target_selection_key(config),
        "split_key": split_key(config),
    }


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha1_token(payload: Any) -> str:
    return hashlib.sha1(_stable_json(payload).encode("utf-8")).hexdigest()


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


def _coerce_target_item(value: Any) -> int | str:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Target item identifiers must not be empty strings.")
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    raise TypeError(f"Unsupported target item identifier type: {type(value).__name__}")


def _requested_target_prefix_length(
    config: Config,
    *,
    ordered_targets: list[int],
) -> int:
    if config.targets.mode == "explicit_list":
        return int(len(ordered_targets))
    requested = max(0, int(config.targets.count))
    if requested > len(ordered_targets):
        raise ValueError(
            "Requested target prefix length exceeds the stable sampled cohort size: "
            f"{requested} > {len(ordered_targets)}"
        )
    return requested


def _deterministic_seeded_order(candidate_pool: list[int], *, seed: int) -> list[int]:
    sorted_pool = sorted(int(item) for item in candidate_pool)
    if len(set(sorted_pool)) != len(sorted_pool):
        raise ValueError("Sampled target candidate pool must not contain duplicate items.")
    return sorted(
        sorted_pool,
        key=lambda item: (
            hashlib.sha1(f"{int(seed)}:{int(item)}".encode("utf-8")).hexdigest(),
            int(item),
        ),
    )


def build_ordered_target_cohort(stats: SessionStats, config: Config) -> dict[str, Any]:
    candidate_pool = _target_candidate_pool(stats, config)
    mode = config.targets.mode
    if mode == "explicit_list":
        ordered_targets = [int(item) for item in config.targets.explicit_list]
        if len(set(ordered_targets)) != len(ordered_targets):
            raise ValueError("Explicit target cohorts must not contain duplicate target items.")
        candidate_basis = [int(item) for item in ordered_targets]
        bucket: str | None = None
        seed: int | None = None
        explicit_list: list[int] | None = [int(item) for item in ordered_targets]
    elif mode == "sampled":
        seed = int(config.seeds.target_selection_seed)
        candidate_basis = sorted(int(item) for item in candidate_pool)
        ordered_targets = _deterministic_seeded_order(candidate_basis, seed=seed)
        bucket = config.targets.bucket
        explicit_list = None
    else:
        raise ValueError(f"Unsupported targets.mode for cohort construction: {mode}")

    return {
        "mode": mode,
        "bucket": bucket,
        "seed": seed,
        "explicit_list": explicit_list,
        "candidate_pool_hash": _sha1_token(candidate_basis),
        "candidate_pool_size": int(len(candidate_basis)),
        "ordered_targets": [int(item) for item in ordered_targets],
    }


def _expected_target_registry_payload(
    stats: SessionStats,
    config: Config,
) -> dict[str, Any]:
    cohort = build_ordered_target_cohort(stats, config)
    ordered_targets = [int(item) for item in cohort["ordered_targets"]]
    return {
        "target_cohort_key": target_cohort_key(config),
        "split_key": split_key(config),
        "selection_policy_version": TARGET_COHORT_SELECTION_POLICY_VERSION,
        **cohort,
        "current_count": _requested_target_prefix_length(
            config,
            ordered_targets=ordered_targets,
        ),
    }


def _target_registry_identity_payload(
    payload: Mapping[str, Any],
    *,
    imported_from_legacy: bool,
) -> dict[str, Any]:
    base_fields = (
        "target_cohort_key",
        "split_key",
        "mode",
        "bucket",
        "seed",
        "explicit_list",
        "ordered_targets",
    )
    strict_only_fields = (
        "selection_policy_version",
        "candidate_pool_hash",
        "candidate_pool_size",
    )
    selected_fields = base_fields if imported_from_legacy else base_fields + strict_only_fields
    return {
        field_name: payload.get(field_name)
        for field_name in selected_fields
    }


def load_or_init_target_registry(
    stats: SessionStats,
    config: Config,
    *,
    shared_paths: Mapping[str, Path],
) -> dict[str, Any]:
    registry_path = shared_paths["target_registry"]
    expected_payload = _expected_target_registry_payload(stats, config)
    existing = load_target_registry(registry_path)
    if existing is None:
        now = _timestamp_utc()
        payload = {
            **expected_payload,
            "created_at": now,
            "updated_at": now,
        }
        Path(shared_paths["target_cohort_dir"]).mkdir(parents=True, exist_ok=True)
        save_target_registry(payload, registry_path)
        return payload

    imported_from_legacy = bool(existing.get("imported_from_legacy"))
    expected_identity = _target_registry_identity_payload(
        expected_payload,
        imported_from_legacy=imported_from_legacy,
    )
    existing_identity = _target_registry_identity_payload(
        existing,
        imported_from_legacy=imported_from_legacy,
    )
    if existing_identity != expected_identity:
        raise ValueError(
            "Existing target_registry.json does not match the expected stable cohort identity."
        )

    current_count = int(existing["current_count"])
    ordered_targets = existing.get("ordered_targets")
    if not isinstance(ordered_targets, list):
        raise ValueError("target_registry.json is missing ordered_targets.")
    if current_count < 0 or current_count > len(ordered_targets):
        raise ValueError("target_registry.json has an invalid current_count.")
    return existing


def ensure_target_registry_prefix(
    stats: SessionStats,
    config: Config,
    *,
    shared_paths: Mapping[str, Path],
    target_registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = (
        load_or_init_target_registry(stats, config, shared_paths=shared_paths)
        if target_registry is None
        else dict(target_registry)
    )
    ordered_targets = registry.get("ordered_targets")
    if not isinstance(ordered_targets, list):
        raise ValueError("target_registry.json is missing ordered_targets.")
    requested_prefix = _requested_target_prefix_length(
        config,
        ordered_targets=[int(item) for item in ordered_targets],
    )
    if requested_prefix > int(registry["current_count"]):
        registry["current_count"] = int(requested_prefix)
        registry["updated_at"] = _timestamp_utc()
        save_target_registry(registry, shared_paths["target_registry"])
    return registry


def _registry_ordered_targets(target_registry: Mapping[str, Any]) -> list[int]:
    ordered_targets = target_registry.get("ordered_targets")
    if not isinstance(ordered_targets, list):
        raise ValueError("target_registry.json is missing ordered_targets.")
    return [int(item) for item in ordered_targets]


def requested_target_prefix(
    config: Config,
    *,
    target_registry: Mapping[str, Any],
) -> list[int]:
    ordered_targets = _registry_ordered_targets(target_registry)
    requested_prefix = _requested_target_prefix_length(
        config,
        ordered_targets=ordered_targets,
    )
    return [int(item) for item in ordered_targets[:requested_prefix]]


def _default_victim_coverage_entry(
    timestamp: str,
    *,
    victim_prediction_key_value: str,
) -> dict[str, Any]:
    return {
        "status": "requested",
        "first_requested_at": timestamp,
        "last_requested_at": timestamp,
        "victim_prediction_key": victim_prediction_key_value,
    }


def _default_cell_artifacts() -> dict[str, Any]:
    return {
        "metrics": None,
        "predictions": None,
        "train_history": None,
        "poisoned_train": None,
    }


def _default_cell_coverage_entry(timestamp: str) -> dict[str, Any]:
    return {
        "status": "requested",
        "artifacts": _default_cell_artifacts(),
        "error": None,
        "first_requested_at": timestamp,
        "last_requested_at": timestamp,
        "last_started_at": None,
        "last_execution_id": None,
        "attempt_count": 0,
        "completed_at": None,
        "failed_at": None,
        "last_updated_at": timestamp,
    }


def _normalize_cell_coverage_entry(
    cell: Mapping[str, Any],
    *,
    timestamp: str,
    default_requested_at: str,
) -> tuple[dict[str, Any], bool]:
    normalized = dict(cell)
    changed = False

    status = normalized.get("status")
    if not isinstance(status, str):
        normalized["status"] = "requested"
        changed = True

    artifacts = normalized.get("artifacts")
    if not isinstance(artifacts, Mapping):
        normalized["artifacts"] = _default_cell_artifacts()
        changed = True
    else:
        artifact_payload = dict(artifacts)
        for artifact_name in ("metrics", "predictions", "train_history", "poisoned_train"):
            if artifact_name not in artifact_payload:
                artifact_payload[artifact_name] = None
                changed = True
        normalized["artifacts"] = artifact_payload

    if "error" not in normalized:
        normalized["error"] = None
        changed = True
    if "first_requested_at" not in normalized:
        normalized["first_requested_at"] = default_requested_at
        changed = True
    if "last_requested_at" not in normalized:
        normalized["last_requested_at"] = normalized.get("first_requested_at", default_requested_at)
        changed = True
    if "last_started_at" not in normalized:
        normalized["last_started_at"] = None
        changed = True
    if "last_execution_id" not in normalized:
        normalized["last_execution_id"] = None
        changed = True
    if "attempt_count" not in normalized:
        normalized["attempt_count"] = 0
        changed = True
    if "completed_at" not in normalized:
        normalized["completed_at"] = None
        changed = True
    if "failed_at" not in normalized:
        normalized["failed_at"] = None
        changed = True
    if "last_updated_at" not in normalized:
        normalized["last_updated_at"] = timestamp
        changed = True
    return normalized, changed


def _largest_completed_target_prefix_count(
    target_order: list[int],
    *,
    cells_payload: Mapping[str, Any],
    victim_names: list[str],
) -> int:
    if not victim_names:
        return 0

    prefix_count = 0
    for target_item in target_order:
        target_cells = cells_payload.get(str(target_item))
        if not isinstance(target_cells, Mapping):
            break
        if all(
            isinstance(target_cells.get(victim_name), Mapping)
            and target_cells[victim_name].get("status") == "completed"
            for victim_name in victim_names
        ):
            prefix_count += 1
            continue
        break
    return prefix_count


def sync_run_coverage_materialized_prefix(run_coverage: dict[str, Any]) -> bool:
    targets_order_raw = run_coverage.get("targets_order", [])
    if not isinstance(targets_order_raw, list):
        raise ValueError("run_coverage.json must contain a targets_order list.")
    victims_payload = run_coverage.get("victims", {})
    if not isinstance(victims_payload, Mapping):
        raise ValueError("run_coverage.json must contain a victims object.")
    cells_payload = run_coverage.get("cells", {})
    if not isinstance(cells_payload, Mapping):
        raise ValueError("run_coverage.json must contain a cells object.")

    targets_order = [int(item) for item in targets_order_raw]
    victim_names = [str(victim_name) for victim_name in victims_payload.keys()]
    materialized_prefix_count = _largest_completed_target_prefix_count(
        targets_order,
        cells_payload=cells_payload,
        victim_names=victim_names,
    )
    stored_value = run_coverage.get(RUN_COVERAGE_MATERIALIZED_PREFIX_FIELD)
    if stored_value == materialized_prefix_count:
        return False
    run_coverage[RUN_COVERAGE_MATERIALIZED_PREFIX_FIELD] = int(materialized_prefix_count)
    return True


def load_or_init_run_coverage(
    config: Config,
    *,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    target_registry: Mapping[str, Any],
    attack_identity_context: Mapping[str, Any] | None = None,
    allow_new_victims: bool = True,
) -> dict[str, Any]:
    coverage_path = metadata_paths["run_coverage"]
    requested_targets_order = requested_target_prefix(
        config,
        target_registry=target_registry,
    )
    expected_run_group_key = run_group_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    expected_target_cohort_key = target_cohort_key(config)
    expected_victim_prediction_keys = {
        victim_name: victim_prediction_key(
            config,
            victim_name,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        )
        for victim_name in config.victims.enabled
    }
    now = _timestamp_utc()

    coverage = load_run_coverage(coverage_path)
    if coverage is None:
        coverage = {
            "run_group_key": expected_run_group_key,
            "target_cohort_key": expected_target_cohort_key,
            "split_key": split_key(config),
            "run_type": run_type,
            "targets_order": requested_targets_order,
            "victims": {
                victim_name: _default_victim_coverage_entry(
                    now,
                    victim_prediction_key_value=expected_victim_prediction_keys[victim_name],
                )
                for victim_name in config.victims.enabled
            },
            "cells": {
                str(target_item): {
                    victim_name: _default_cell_coverage_entry(now)
                    for victim_name in config.victims.enabled
                }
                for target_item in requested_targets_order
            },
            RUN_COVERAGE_MATERIALIZED_PREFIX_FIELD: 0,
            "created_at": now,
            "updated_at": now,
        }
        save_run_coverage(coverage, coverage_path)
        return coverage

    if coverage.get("run_group_key") != expected_run_group_key:
        raise ValueError("run_coverage.json does not match the current run_group_key.")
    if coverage.get("target_cohort_key") != expected_target_cohort_key:
        raise ValueError("run_coverage.json does not match the current target_cohort_key.")

    victims_payload = coverage.get("victims")
    if not isinstance(victims_payload, dict):
        raise ValueError("run_coverage.json must contain a victims object.")
    cells_payload = coverage.get("cells")
    if not isinstance(cells_payload, dict):
        raise ValueError("run_coverage.json must contain a cells object.")
    default_requested_at = str(coverage.get("created_at", now))

    changed = False
    for victim_name in config.victims.enabled:
        victim_entry = victims_payload.get(victim_name)
        if victim_entry is None:
            if not allow_new_victims:
                raise RuntimeError(
                    "Existing run_coverage.json does not include the currently requested "
                    f"victim '{victim_name}'. Phase 4 supports target append only for the "
                    "same victim set. Victim-append or victim-set changes are not "
                    "implemented until Phase 5."
                )
            victims_payload[victim_name] = _default_victim_coverage_entry(
                now,
                victim_prediction_key_value=expected_victim_prediction_keys[victim_name],
            )
            changed = True
            continue
        if not isinstance(victim_entry, dict):
            raise ValueError(f"run_coverage.json victims[{victim_name}] must be an object.")
        expected_key = expected_victim_prediction_keys[victim_name]
        stored_key = victim_entry.get("victim_prediction_key")
        if stored_key is None:
            victim_entry["victim_prediction_key"] = expected_key
            changed = True
        elif stored_key != expected_key:
            raise RuntimeError(
                "Existing run_coverage.json victim registry is incompatible with the "
                f"currently requested victim configuration for '{victim_name}'. "
                f"Existing victim_prediction_key={stored_key}, "
                f"requested victim_prediction_key={expected_key}."
            )
        victim_entry["last_requested_at"] = now

    registry_ordered_targets = _registry_ordered_targets(target_registry)
    existing_targets_order = [int(item) for item in coverage.get("targets_order", [])]
    if existing_targets_order:
        if registry_ordered_targets[: len(existing_targets_order)] != existing_targets_order:
            raise ValueError("run_coverage.json targets_order is incompatible with target_registry.")
    target_prefix_order = list(existing_targets_order)
    if len(requested_targets_order) > len(target_prefix_order):
        target_prefix_order = list(requested_targets_order)
    if existing_targets_order != target_prefix_order:
        coverage["targets_order"] = target_prefix_order
        changed = True

    victim_names = list(victims_payload.keys())
    for target_item in target_prefix_order:
        target_key = str(target_item)
        target_cells = cells_payload.get(target_key)
        if not isinstance(target_cells, dict):
            target_cells = {}
            cells_payload[target_key] = target_cells
            changed = True
        for victim_name in victim_names:
            if victim_name not in target_cells:
                target_cells[victim_name] = _default_cell_coverage_entry(now)
                changed = True
                continue
            cell_payload = target_cells[victim_name]
            if not isinstance(cell_payload, Mapping):
                raise ValueError(
                    f"run_coverage.json cells[{target_key}][{victim_name}] must be an object."
                )
            normalized_cell, normalized_changed = _normalize_cell_coverage_entry(
                cell_payload,
                timestamp=now,
                default_requested_at=default_requested_at,
            )
            if normalized_changed:
                target_cells[victim_name] = normalized_cell
                changed = True

    if sync_run_coverage_materialized_prefix(coverage):
        changed = True

    if changed:
        coverage["updated_at"] = now
        save_run_coverage(coverage, coverage_path)
    return coverage


def plan_target_append_cells(
    run_coverage: Mapping[str, Any],
    *,
    requested_target_items: list[int],
    requested_victims: list[str],
) -> dict[str, list[dict[str, Any]]]:
    cells_payload = run_coverage.get("cells", {})
    if not isinstance(cells_payload, Mapping):
        raise ValueError("run_coverage.json must contain a cells object.")

    planned_cells: list[dict[str, Any]] = []
    skipped_completed_cells: list[dict[str, Any]] = []
    for target_item in requested_target_items:
        target_key = str(target_item)
        target_cells = cells_payload.get(target_key)
        if not isinstance(target_cells, Mapping):
            raise ValueError(f"run_coverage.json cells[{target_key}] must be an object.")
        for victim_name in requested_victims:
            cell = target_cells.get(victim_name)
            if not isinstance(cell, Mapping):
                raise ValueError(
                    f"run_coverage.json cells[{target_key}][{victim_name}] must be an object."
                )
            cell_ref = {
                "target_item": int(target_item),
                "victim_name": victim_name,
            }
            if cell.get("status") == "completed":
                skipped_completed_cells.append(cell_ref)
            else:
                planned_cells.append(cell_ref)
    return {
        "planned_cells": planned_cells,
        "skipped_completed_cells": skipped_completed_cells,
    }


def load_or_init_execution_log(
    config: Config,
    *,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    execution_log_path = metadata_paths["execution_log"]
    expected_run_group_key = run_group_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    expected_target_cohort_key = target_cohort_key(config)
    execution_log = load_execution_log(execution_log_path)
    if execution_log is None:
        now = _timestamp_utc()
        execution_log = {
            "run_group_key": expected_run_group_key,
            "target_cohort_key": expected_target_cohort_key,
            "split_key": split_key(config),
            "run_type": run_type,
            "executions": [],
            "created_at": now,
            "updated_at": now,
        }
        save_execution_log(execution_log, execution_log_path)
        return execution_log

    if execution_log.get("run_group_key") != expected_run_group_key:
        raise ValueError("execution_log.json does not match the current run_group_key.")
    if execution_log.get("target_cohort_key") != expected_target_cohort_key:
        raise ValueError("execution_log.json does not match the current target_cohort_key.")
    if not isinstance(execution_log.get("executions"), list):
        raise ValueError("execution_log.json must contain an executions list.")
    return execution_log


def rebuild_summary_current(
    config: Config,
    *,
    run_type: str,
    metadata_paths: Mapping[str, Path],
    run_coverage: Mapping[str, Any],
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_order = list(run_coverage.get("targets_order", []))
    victims_payload = run_coverage.get("victims", {})
    if not isinstance(victims_payload, Mapping):
        raise ValueError("run_coverage.json must contain a victims object.")
    cells_payload = run_coverage.get("cells", {})
    if not isinstance(cells_payload, Mapping):
        raise ValueError("run_coverage.json must contain a cells object.")

    summary_targets: dict[str, Any] = {}
    for raw_target_item in target_order:
        target_key = str(raw_target_item)
        target_cells = cells_payload.get(target_key, {})
        if not isinstance(target_cells, Mapping):
            raise ValueError(f"run_coverage.json cells[{target_key}] must be an object.")

        victim_summaries: dict[str, Any] = {}
        for victim_name in victims_payload:
            cell = target_cells.get(victim_name)
            if not isinstance(cell, Mapping):
                continue
            if cell.get("status") != "completed":
                continue
            artifacts = cell.get("artifacts", {})
            if not isinstance(artifacts, Mapping):
                raise ValueError(
                    f"run_coverage.json cells[{target_key}][{victim_name}].artifacts must be an object."
                )
            metrics_path = artifacts.get("metrics")
            if not isinstance(metrics_path, str) or not metrics_path.strip():
                raise ValueError(
                    "Completed coverage cells must record a repo-relative metrics artifact path."
                )
            metrics_payload = load_json(_repo_path(metrics_path))
            if not isinstance(metrics_payload, dict):
                raise ValueError(f"Missing or invalid metrics artifact for completed cell: {metrics_path}")

            metric_values = metrics_payload.get("metrics", {})
            if not isinstance(metric_values, Mapping):
                metric_values = {}
            predictions_path = artifacts.get("predictions")
            victim_summary = {
                "metrics_path": metrics_path,
                "predictions_path": (
                    predictions_path
                    if isinstance(predictions_path, str) and predictions_path.strip()
                    else metrics_payload.get("predictions_path")
                ),
                "metrics": dict(metric_values),
                "metrics_available": bool(metrics_payload.get("metrics_available", True)),
            }
            if isinstance(metrics_payload.get("reused_predictions"), bool):
                victim_summary["reused_predictions"] = bool(metrics_payload["reused_predictions"])
            victim_summaries[victim_name] = victim_summary

        if victim_summaries:
            summary_targets[target_key] = {
                "target_item": _coerce_target_item(raw_target_item),
                "victims": victim_summaries,
            }

    now = _timestamp_utc()
    summary_current = {
        "run_type": run_type,
        "run_group_key": run_group_key(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
        "target_cohort_key": target_cohort_key(config),
        "is_snapshot": True,
        "snapshot_source": "run_coverage_and_cell_artifacts",
        "target_items": [_coerce_target_item(item) for item in target_order],
        "victims": list(victims_payload.keys()),
        "targets": summary_targets,
        "created_at": now,
        "updated_at": now,
    }
    save_summary_current(summary_current, metadata_paths["summary_current"])
    return summary_current


def resolve_target_items(
    stats: SessionStats,
    config: Config,
    *,
    shared_paths: dict[str, Path] | None = None,
) -> list[int]:
    if shared_paths is None:
        return _resolve_target_items(stats, config)

    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    selected_targets = None
    if config.targets.reuse_saved_targets:
        selected_targets = load_selected_targets(shared_paths["selected_targets"])
        if selected_targets is None:
            legacy_target_info = load_target_info(shared_paths["target_info"])
            if legacy_target_info is not None:
                raw_target_items = legacy_target_info.get("target_items")
                if isinstance(raw_target_items, list):
                    selected_targets = [int(item) for item in raw_target_items]
                    save_selected_targets(shared_paths["selected_targets"], selected_targets)

    candidate_pool = _target_candidate_pool(stats, config)
    meta_payload = _target_selection_meta_payload(
        stats,
        config,
        candidate_pool=candidate_pool,
    )

    if selected_targets is None:
        target_items = _resolve_target_items(stats, config)
        save_selected_targets(shared_paths["selected_targets"], target_items)
        save_target_selection_meta(shared_paths["target_selection_meta"], meta_payload)
        save_target_info(
            shared_paths["target_info"],
            target_items=target_items,
            target_selection_mode=config.targets.mode,
            seed=config.seeds.target_selection_seed,
            bucket=config.targets.bucket if config.targets.mode == "sampled" else None,
            count=config.targets.count if config.targets.mode == "sampled" else None,
            explicit_list=list(config.targets.explicit_list),
        )
    else:
        target_items = [int(item) for item in selected_targets]
        if load_json(shared_paths["target_selection_meta"]) is None:
            save_target_selection_meta(shared_paths["target_selection_meta"], meta_payload)
    return [int(item) for item in target_items]


def _export_srg_nn_dataset(
    *,
    dataset: CanonicalDataset,
    export_dir: Path,
) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    train_path = export_dir / "train.txt"
    valid_path = export_dir / "valid.txt"
    test_path = export_dir / "test.txt"
    if train_path.exists() and valid_path.exists() and test_path.exists():
        return {"train": train_path, "valid": valid_path, "test": test_path}
    exporter = SRGNNExporter()
    result = exporter.export(dataset, export_dir)
    return result.files


def _write_poison_model_identity(
    config: Config,
    *,
    shared_paths: Mapping[str, Path],
) -> None:
    save_json(
        {
            "poison_model_key": poison_model_key(config),
            "payload": poison_model_key_payload(config),
            "checkpoint_path": _repo_relative_path(shared_paths["poison_model"]),
            "train_history_path": _repo_relative_path(shared_paths["poison_train_history"]),
        },
        shared_paths["poison_model_identity"],
    )


def _legacy_poison_model_candidates(
    config: Config,
    *,
    shared_paths: Mapping[str, Path],
) -> list[tuple[Path, Path]]:
    candidates: list[tuple[Path, Path]] = []
    legacy_current = shared_paths.get("legacy_attack_poison_model")
    legacy_current_history = shared_paths.get("legacy_attack_poison_train_history")
    if legacy_current is not None and legacy_current_history is not None:
        candidates.append((Path(legacy_current), Path(legacy_current_history)))

    # Compatibility bridge for pre-canonical artifacts. Random-NZ ratio1 and
    # TACS-NZ should share the same clean poison checkpoint; only fake_sessions
    # are separated by generation size.
    random_nz_shared_dir = shared_attack_dir(
        config,
        run_type="random_nonzero_when_possible",
    )
    candidates.append(
        (
            random_nz_shared_dir / "poison_model.pt",
            random_nz_shared_dir / "poison_train_history.json",
        )
    )

    unique: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    canonical = Path(shared_paths["poison_model"]).resolve()
    for checkpoint_path, history_path in candidates:
        resolved = checkpoint_path.resolve()
        if resolved == canonical or resolved in seen:
            continue
        seen.add(resolved)
        unique.append((checkpoint_path, history_path))
    return unique


def _copy_legacy_poison_history_if_available(
    *,
    legacy_history_path: Path,
    shared_paths: Mapping[str, Path],
) -> None:
    destination = Path(shared_paths["poison_train_history"])
    if destination.exists() or not legacy_history_path.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(legacy_history_path, destination)


def _load_or_train_poison_runner(
    config: Config,
    *,
    shared_paths: dict[str, Path],
    export_paths: dict[str, Path],
) -> SRGNNPoisonRunner:
    poison_train_config = _poison_train_config(config)
    configured_poison_epochs = int(poison_train_config["epochs"])
    runner = SRGNNPoisonRunner(config)
    runner.build_model(build_srgnn_opt_from_train_config(poison_train_config))
    if load_poison_model(runner, shared_paths["poison_model"]):
        print(f"Loaded poison model checkpoint from {shared_paths['poison_model']}")
        _write_poison_model_identity(config, shared_paths=shared_paths)
        return runner

    for legacy_checkpoint, legacy_history in _legacy_poison_model_candidates(
        config,
        shared_paths=shared_paths,
    ):
        if not legacy_checkpoint.exists():
            continue
        if load_poison_model(runner, legacy_checkpoint):
            print(
                "Loaded legacy compatible poison model checkpoint from "
                f"{legacy_checkpoint}; saving canonical checkpoint to "
                f"{shared_paths['poison_model']}"
            )
            save_poison_model(runner, shared_paths["poison_model"])
            _copy_legacy_poison_history_if_available(
                legacy_history_path=legacy_history,
                shared_paths=shared_paths,
            )
            _write_poison_model_identity(config, shared_paths=shared_paths)
            return runner

    print(
        "No poison model checkpoint found. "
        f"Training new poison model for {configured_poison_epochs} epochs."
    )
    train_data, test_data = runner.load_dataset(
        train_path=export_paths["train"],
        test_path=export_paths["valid"],
    )
    if srgnn_validation_best_enabled(poison_train_config):
        result = train_srgnn_validation_best(
            runner,
            train_data,
            test_data,
            train_config=poison_train_config,
            max_epochs=configured_poison_epochs,
            patience=int(poison_train_config["patience"]),
            best_checkpoint_path=shared_paths["poison_model"],
            log_prefix="[poison:srgnn-validation-best]",
        )
        print(f"Saved poison model checkpoint to {shared_paths['poison_model']}")
        save_train_history(
            shared_paths["poison_train_history"],
            role="poison",
            model="srgnn",
            epochs=len(result.rows),
            train_loss=[float(row["train_loss"]) for row in result.rows],
            valid_loss=[None] * len(result.rows),
            notes=(
                "SRGNN poison training selected the checkpoint with highest "
                "validation ground-truth MRR@20. Test metrics were not used."
            ),
            extra=srgnn_validation_train_history_extra(result),
        )
        _write_poison_model_identity(config, shared_paths=shared_paths)
    elif configured_poison_epochs > 0:
        runner.train(
            train_data,
            test_data,
            configured_poison_epochs,
            topk=max(config.evaluation.topk),
        )
        save_poison_model(runner, shared_paths["poison_model"])
        print(f"Saved poison model checkpoint to {shared_paths['poison_model']}")
        _write_poison_model_identity(config, shared_paths=shared_paths)
        if runner.train_loss_history:
            save_train_history(
                shared_paths["poison_train_history"],
                role="poison",
                model="srgnn",
                epochs=len(runner.train_loss_history),
                train_loss=runner.train_loss_history,
                valid_loss=[None] * len(runner.train_loss_history),
                notes="valid_loss not available for SRGNN poison training.",
            )
    return runner


@dataclass(frozen=True)
class SharedAttackArtifacts:
    stats: SessionStats
    clean_sessions: list[list[int]]
    clean_labels: list[int]
    canonical_dataset: CanonicalDataset
    export_paths: dict[str, Path]
    template_sessions: list[list[int]]
    poison_runner: SRGNNPoisonRunner | None
    fake_session_count: int
    shared_paths: dict[str, Path]


def prepare_shared_attack_artifacts(
    config: Config,
    *,
    run_type: str,
    require_poison_runner: bool,
    config_path: str | Path | None = None,
) -> SharedAttackArtifacts:
    generation_seed = int(config.seeds.fake_session_seed)
    canonical_dataset = ensure_canonical_dataset(config)
    shared_paths = shared_artifact_paths(config, run_type=run_type)
    shared_paths["attack_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        snapshot_path = shared_paths["attack_config_snapshot"]
        if not snapshot_path.exists():
            shutil.copyfile(config_path, snapshot_path)
    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        target_snapshot_path = shared_paths["target_config_snapshot"]
        if not target_snapshot_path.exists():
            shutil.copyfile(config_path, target_snapshot_path)

    stats = compute_session_stats(canonical_dataset.train_sub)
    clean_sessions, clean_labels = build_clean_pairs(canonical_dataset)

    export_dir = shared_paths["attack_shared_dir"] / "export"
    export_paths = _export_srg_nn_dataset(
        dataset=canonical_dataset,
        export_dir=export_dir,
    )

    template_sessions = load_fake_sessions(shared_paths["fake_sessions"])
    poison_runner = None
    if template_sessions is None:
        print("No fake sessions found. Generating new fake sessions.")
        set_seed(derive_seed(generation_seed, "poison_model_generation"))
        poison_runner = _load_or_train_poison_runner(
            config,
            shared_paths=shared_paths,
            export_paths=export_paths,
        )
        sampler = FakeSessionParameterSampler(stats)
        generator = FakeSessionGenerator(
            poison_runner,
            sampler,
            topk=config.attack.fake_session_generation_topk,
        )
        fake_count = fake_session_count_from_ratio(
            _fake_session_generation_ratio(config, run_type=run_type),
            len(clean_sessions),
        )
        set_seed(derive_seed(generation_seed, "fake_session_generation"))
        template_sessions = [s.items for s in generator.generate_many(fake_count)]
        save_fake_sessions(template_sessions, shared_paths["fake_sessions"])
        print(f"Saved fake sessions to {shared_paths['fake_sessions']}")
    else:
        print(f"Loaded fake sessions from {shared_paths['fake_sessions']}")
        fake_count = len(template_sessions)

    if require_poison_runner and poison_runner is None:
        set_seed(derive_seed(generation_seed, "poison_model_generation"))
        poison_runner = _load_or_train_poison_runner(
            config,
            shared_paths=shared_paths,
            export_paths=export_paths,
        )

    return SharedAttackArtifacts(
        stats=stats,
        clean_sessions=clean_sessions,
        clean_labels=clean_labels,
        canonical_dataset=canonical_dataset,
        export_paths=export_paths,
        template_sessions=template_sessions,
        poison_runner=poison_runner,
        fake_session_count=fake_count,
        shared_paths=shared_paths,
    )


def _poison_train_config(config: Config) -> dict[str, Any]:
    return dict(config.attack.poison_model.params["train"])


__all__ = [
    "RUN_COVERAGE_MATERIALIZED_PREFIX_FIELD",
    "SharedAttackArtifacts",
    "build_ordered_target_cohort",
    "build_srgnn_opt_from_train_config",
    "build_clean_pairs",
    "ensure_target_registry_prefix",
    "fake_session_count_from_ratio",
    "load_or_init_execution_log",
    "load_or_init_run_coverage",
    "load_or_init_target_registry",
    "plan_target_append_cells",
    "prepare_shared_attack_artifacts",
    "requested_target_prefix",
    "rebuild_summary_current",
    "resolve_target_items",
    "sync_run_coverage_materialized_prefix",
]
