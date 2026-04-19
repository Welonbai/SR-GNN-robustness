from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import (
    load_execution_log,
    load_json,
    load_run_coverage,
    load_summary_current,
    load_target_registry,
    load_target_info,
    load_target_selection_meta,
    save_execution_log,
    save_json,
    save_run_coverage,
    save_summary_current,
    save_target_info,
    save_target_registry,
    save_target_selection_meta,
)
from attack.common.config import Config, _build_config, load_config, normalize_config_mapping
from attack.common.paths import (
    POSITION_OPT_RUN_TYPE,
    build_position_opt_attack_identity_context,
    run_artifact_paths,
    run_group_key,
    run_metadata_paths,
    shared_artifact_paths,
    split_key,
    target_cohort_key,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    _initial_artifact_manifest,
    _key_payloads,
    _resolved_config_payload,
)
from attack.pipeline.core.pipeline_utils import (
    load_or_init_execution_log,
    load_or_init_run_coverage,
    rebuild_summary_current,
)
from attack.position_opt import position_opt_identity_payload, resolve_clean_surrogate_checkpoint_path


REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_VERSION = "legacy_import_v1"
LEGACY_CELL_CORE_FILES = (
    "config_snapshot",
    "resolved_config",
    "metrics",
    "predictions",
    "train_history",
    "poisoned_train",
)
METRIC_KEY_PATTERN = re.compile(
    r"^(?P<prefix>targeted|ground_truth)_(?P<metric>[A-Za-z0-9_]+)@(?P<k>\d+)$"
)


class MigrationError(ValueError):
    """Raised when a legacy run cannot be migrated safely."""


@dataclass(frozen=True)
class LegacyRunDiscovery:
    legacy_run_root: Path
    summary_path: Path
    resolved_config_path: Path | None
    key_payloads_path: Path | None
    artifact_manifest_path: Path | None
    config_snapshot_path: Path | None
    selected_targets_path: Path | None
    target_selection_meta_path: Path | None
    target_info_path: Path | None


@dataclass(frozen=True)
class LegacyRunSource:
    discovery: LegacyRunDiscovery
    config: Config
    run_type: str
    attack_identity_context: Mapping[str, Any] | None
    resolved_config_payload: dict[str, Any] | None
    key_payloads_payload: dict[str, Any] | None
    artifact_manifest_payload: dict[str, Any] | None
    summary_payload: dict[str, Any]
    selected_targets: list[int | str]
    target_info_payload: dict[str, Any] | None
    target_selection_meta_payload: dict[str, Any] | None
    target_registry_payload: dict[str, Any]
    destination_run_root: Path
    migration_metadata: dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Import one or more legacy batch-era run roots into the new appendable "
            "run-group container model."
        )
    )
    parser.add_argument(
        "legacy_run",
        nargs="+",
        help="One or more legacy run roots, or one summary_*.json path per run.",
    )
    parser.add_argument(
        "--artifacts-root",
        help=(
            "Optional artifacts.root override for the migrated destination. "
            "Defaults to the reconstructed config artifacts.root."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and validate inferred migration state without writing files.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        results = migrate_legacy_runs(
            args.legacy_run,
            artifacts_root_override=args.artifacts_root,
            dry_run=args.dry_run,
        )
    except MigrationError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.dry_run:
        for result in results:
            print(
                "[dry-run] "
                f"legacy={result['source_legacy_run_root']} "
                f"run_group_key={result['run_group_key']} "
                f"target_count={result['imported_target_count']} "
                f"completed_cells={result['completed_cell_count']}"
            )
        return

    for result in results:
        print(
            "Migrated "
            f"{result['source_legacy_run_root']} -> {result['run_root']} "
            f"({result['completed_cell_count']} completed cells imported)"
        )


def migrate_legacy_runs(
    legacy_runs: Sequence[str | Path],
    *,
    artifacts_root_override: str | Path | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    if not legacy_runs:
        raise MigrationError("At least one legacy run root must be provided.")

    override_root = None if artifacts_root_override is None else _repo_path(artifacts_root_override)
    sources = [
        inspect_legacy_run(run_path, artifacts_root_override=override_root)
        for run_path in legacy_runs
    ]
    _validate_unique_destination_mapping(sources)

    if dry_run:
        return [
            {
                "source_legacy_run_root": _display_path(source.discovery.legacy_run_root),
                "run_group_key": run_group_key(
                    source.config,
                    run_type=source.run_type,
                    attack_identity_context=source.attack_identity_context,
                ),
                "run_root": _display_path(source.destination_run_root),
                "target_cohort_key": target_cohort_key(source.config),
                "imported_target_count": int(source.target_registry_payload["current_count"]),
                "completed_cell_count": int(source.migration_metadata["completed_cell_count"]),
                "requested_victims": list(source.config.victims.enabled),
            }
            for source in sources
        ]

    return [_migrate_legacy_source(source) for source in sources]


def inspect_legacy_run(
    legacy_run: str | Path,
    *,
    artifacts_root_override: Path | None = None,
) -> LegacyRunSource:
    discovery = discover_legacy_run(legacy_run)
    summary_payload = _load_json_object(discovery.summary_path, label="legacy summary")
    resolved_config_payload = (
        _load_json_object(discovery.resolved_config_path, label="legacy resolved_config")
        if discovery.resolved_config_path is not None
        else None
    )
    key_payloads_payload = (
        _load_json_object(discovery.key_payloads_path, label="legacy key_payloads")
        if discovery.key_payloads_path is not None
        else None
    )
    artifact_manifest_payload = (
        _load_json_object(discovery.artifact_manifest_path, label="legacy artifact_manifest")
        if discovery.artifact_manifest_path is not None
        else None
    )
    target_info_payload = (
        load_target_info(discovery.target_info_path)
        if discovery.target_info_path is not None and discovery.target_info_path.exists()
        else None
    )
    target_selection_meta_payload = (
        load_target_selection_meta(discovery.target_selection_meta_path)
        if discovery.target_selection_meta_path is not None and discovery.target_selection_meta_path.exists()
        else None
    )

    config, config_reconstruction = reconstruct_config(
        discovery=discovery,
        resolved_config_payload=resolved_config_payload,
        summary_payload=summary_payload,
        artifacts_root_override=artifacts_root_override,
    )
    run_type = infer_legacy_run_type(discovery=discovery, resolved_config_payload=resolved_config_payload)
    attack_identity_context = resolve_attack_identity_context_for_migration(config, run_type=run_type)

    selected_targets = resolve_legacy_selected_targets(
        discovery=discovery,
        summary_payload=summary_payload,
        target_info_payload=target_info_payload,
        artifact_manifest_payload=artifact_manifest_payload,
    )
    legacy_sampled_selected_targets_only = config.targets.mode == "sampled"
    if legacy_sampled_selected_targets_only:
        config = _normalize_sampled_legacy_targets_as_explicit_migration_cohort(
            config,
            selected_targets=selected_targets,
        )
    target_registry_payload, registry_notes = reconstruct_target_registry(
        config=config,
        selected_targets=selected_targets,
        target_info_payload=target_info_payload,
        target_selection_meta_payload=target_selection_meta_payload,
        legacy_sampled_selected_targets_only=legacy_sampled_selected_targets_only,
    )
    destination_run_root = run_metadata_paths(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )["run_root"]
    preview_counts = preview_import_counts(
        config=config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
        target_registry_payload=target_registry_payload,
        source_target_cells=_extract_legacy_target_cells_payload(
            discovery=discovery,
            summary_payload=summary_payload,
            artifact_manifest_payload=artifact_manifest_payload,
        ),
    )

    migration_metadata = {
        "imported_from_legacy": True,
        "migration_version": MIGRATION_VERSION,
        "imported_at": _timestamp_utc(),
        "source_legacy_run_root": _display_path(discovery.legacy_run_root),
        "source_summary_path": _display_path(discovery.summary_path),
        "source_resolved_config_path": _display_optional_path(discovery.resolved_config_path),
        "source_key_payloads_path": _display_optional_path(discovery.key_payloads_path),
        "source_artifact_manifest_path": _display_optional_path(discovery.artifact_manifest_path),
        "source_target_info_path": _display_optional_path(discovery.target_info_path),
        "source_target_selection_meta_path": _display_optional_path(discovery.target_selection_meta_path),
        "source_selected_targets_path": _display_optional_path(discovery.selected_targets_path),
        "config_reconstruction": config_reconstruction,
        "target_registry_reconstruction": registry_notes,
        "target_cohort_normalization": (
            {
                "mode": "legacy_sampled_selected_targets_promoted_to_explicit_cohort",
                "note": (
                    "Legacy sampled migration only proved the materialized selected target set. "
                    "The migrated target cohort is represented as an explicit deterministic "
                    "cohort to avoid claiming compatibility with a native sampled cohort order."
                ),
            }
            if legacy_sampled_selected_targets_only
            else None
        ),
        "completed_cell_count": int(preview_counts["completed_cell_count"]),
        "failed_cell_count": int(preview_counts["failed_cell_count"]),
        "requested_cell_count": int(preview_counts["requested_cell_count"]),
    }

    return LegacyRunSource(
        discovery=discovery,
        config=config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
        resolved_config_payload=resolved_config_payload,
        key_payloads_payload=key_payloads_payload,
        artifact_manifest_payload=artifact_manifest_payload,
        summary_payload=summary_payload,
        selected_targets=selected_targets,
        target_info_payload=target_info_payload,
        target_selection_meta_payload=target_selection_meta_payload,
        target_registry_payload=target_registry_payload,
        destination_run_root=destination_run_root,
        migration_metadata=migration_metadata,
    )


def discover_legacy_run(legacy_run: str | Path) -> LegacyRunDiscovery:
    input_path = Path(legacy_run)
    if input_path.is_file():
        if not input_path.name.startswith("summary_") or input_path.suffix.lower() != ".json":
            raise MigrationError(
                "Legacy run inputs must be run roots or summary_*.json paths. "
                f"Got file '{input_path}'."
            )
        legacy_run_root = input_path.parent.resolve()
        summary_path = input_path.resolve()
    else:
        legacy_run_root = input_path.resolve()
        if not legacy_run_root.exists():
            raise MigrationError(f"Legacy run root does not exist: '{legacy_run_root}'.")
        summary_candidates = sorted(legacy_run_root.glob("summary_*.json"))
        if not summary_candidates:
            raise MigrationError(
                f"Legacy run root '{legacy_run_root}' does not contain any summary_*.json file."
            )
        if len(summary_candidates) > 1:
            raise MigrationError(
                f"Legacy run root '{legacy_run_root}' contains multiple summary_*.json files. "
                "Pass the intended summary file explicitly."
            )
        summary_path = summary_candidates[0].resolve()

    artifact_manifest_path = _existing_path(legacy_run_root / "artifact_manifest.json")
    artifact_manifest_payload = (
        _load_json_object(artifact_manifest_path, label="legacy artifact_manifest")
        if artifact_manifest_path is not None
        else None
    )

    resolved_config_path = _existing_path(legacy_run_root / "resolved_config.json")
    key_payloads_path = _existing_path(legacy_run_root / "key_payloads.json")

    config_snapshot_path = None
    selected_targets_path = None
    target_selection_meta_path = None
    target_info_path = None
    if artifact_manifest_payload is not None:
        target_selection_payload = artifact_manifest_payload.get("target_selection_artifact")
        if isinstance(target_selection_payload, Mapping):
            config_snapshot_path = _manifest_path(
                target_selection_payload.get("config_snapshot"),
                legacy_run_root=legacy_run_root,
            )
            selected_targets_path = _manifest_path(
                target_selection_payload.get("selected_targets"),
                legacy_run_root=legacy_run_root,
            )
            target_selection_meta_path = _manifest_path(
                target_selection_payload.get("target_selection_meta"),
                legacy_run_root=legacy_run_root,
            )
            target_info_path = _manifest_path(
                target_selection_payload.get("legacy_target_info"),
                legacy_run_root=legacy_run_root,
            )

    return LegacyRunDiscovery(
        legacy_run_root=legacy_run_root,
        summary_path=summary_path,
        resolved_config_path=resolved_config_path,
        key_payloads_path=key_payloads_path,
        artifact_manifest_path=artifact_manifest_path,
        config_snapshot_path=config_snapshot_path,
        selected_targets_path=selected_targets_path,
        target_selection_meta_path=target_selection_meta_path,
        target_info_path=target_info_path,
    )


def reconstruct_config(
    *,
    discovery: LegacyRunDiscovery,
    resolved_config_payload: Mapping[str, Any] | None,
    summary_payload: Mapping[str, Any],
    artifacts_root_override: Path | None,
) -> tuple[Config, dict[str, Any]]:
    if resolved_config_payload is None:
        if discovery.config_snapshot_path is None or not discovery.config_snapshot_path.exists():
            raise MigrationError(
                "Legacy migration requires either resolved_config.json or a config snapshot. "
                f"Run root: '{discovery.legacy_run_root}'."
            )
        config = load_config(discovery.config_snapshot_path)
        if artifacts_root_override is not None:
            config = replace(
                config,
                artifacts=replace(config.artifacts, root=str(artifacts_root_override)),
            )
        return config, {
            "config_source": "config_snapshot",
            "config_snapshot_path": _display_path(discovery.config_snapshot_path),
            "defaulted_seed_keys": [],
        }

    result_config = _require_mapping(
        resolved_config_payload.get("result_config"),
        label="resolved_config.result_config",
    )
    runtime_config = _require_mapping(
        resolved_config_payload.get("runtime_config", {}),
        label="resolved_config.runtime_config",
    )
    victims_runtime = _extract_nested_mapping(runtime_config, ("victims", "runtime"))
    raw_seed_payload = _require_mapping(
        result_config.get("seeds"),
        label="resolved_config.result_config.seeds",
    )
    defaulted_seed_keys = [
        key
        for key in ("position_opt_seed", "surrogate_train_seed", "victim_train_seed")
        if key not in raw_seed_payload
    ]
    config_mapping = {
        "experiment": {
            "name": discovery.legacy_run_root.parent.name,
        },
        "data": _require_mapping(
            result_config.get("data"),
            label="resolved_config.result_config.data",
        ),
        "seeds": dict(raw_seed_payload),
        "attack": _require_mapping(
            result_config.get("attack"),
            label="resolved_config.result_config.attack",
        ),
        "targets": _require_mapping(
            result_config.get("targets"),
            label="resolved_config.result_config.targets",
        ),
        "victims": {
            "enabled": _require_sequence(
                _extract_nested_value(result_config, ("victims", "enabled")),
                label="resolved_config.result_config.victims.enabled",
            ),
            "params": _require_mapping(
                _extract_nested_value(result_config, ("victims", "params")),
                label="resolved_config.result_config.victims.params",
            ),
            "runtime": victims_runtime,
        },
        "evaluation": infer_legacy_evaluation_config(
            result_config.get("evaluation"),
            summary_payload=summary_payload,
        ),
        "artifacts": {
            "root": str(artifacts_root_override or REPO_ROOT / "outputs"),
            "shared_dir": "shared",
            "runs_dir": "runs",
        },
    }
    normalized = normalize_config_mapping(config_mapping)
    return _build_config(normalized), {
        "config_source": "resolved_config",
        "resolved_config_path": _display_path(discovery.resolved_config_path),
        "defaulted_seed_keys": defaulted_seed_keys,
    }


def infer_legacy_run_type(
    *,
    discovery: LegacyRunDiscovery,
    resolved_config_payload: Mapping[str, Any] | None,
) -> str:
    if resolved_config_payload is not None:
        derived_payload = _extract_nested_mapping(resolved_config_payload, ("derived",))
        run_type = derived_payload.get("run_type")
        if isinstance(run_type, str) and run_type.strip():
            return run_type
    summary_name = discovery.summary_path.name
    if summary_name.startswith("summary_") and summary_name.endswith(".json"):
        return summary_name[len("summary_") : -len(".json")]
    raise MigrationError(
        f"Unable to infer legacy run_type from '{discovery.summary_path}'."
    )


def infer_legacy_evaluation_config(
    raw_evaluation: Any,
    *,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_payload = _require_mapping(
        raw_evaluation,
        label="resolved_config.result_config.evaluation",
    )
    topk = [
        int(item)
        for item in _require_sequence(
            evaluation_payload.get("topk"),
            label="resolved_config.result_config.evaluation.topk",
        )
    ]
    if not topk:
        raise MigrationError("Legacy evaluation config must include a non-empty topk list.")

    targeted_metrics = list(_normalize_metric_name_list(evaluation_payload.get("targeted_metrics")))
    ground_truth_metrics = list(_normalize_metric_name_list(evaluation_payload.get("ground_truth_metrics")))
    legacy_metric_entries = [
        str(item)
        for item in _normalize_string_list(evaluation_payload.get("metrics"))
    ]
    for metric_entry in legacy_metric_entries:
        if metric_entry.startswith("targeted_"):
            targeted_metrics.append(metric_entry[len("targeted_") :])
        elif metric_entry.startswith("ground_truth_"):
            ground_truth_metrics.append(metric_entry[len("ground_truth_") :])
        else:
            targeted_metrics.append(metric_entry)

    discovered_targeted, discovered_ground_truth = discover_metric_names_from_summary(summary_payload)
    targeted_metrics = _unique_preserve_order(targeted_metrics + discovered_targeted)
    ground_truth_metrics = _unique_preserve_order(ground_truth_metrics + discovered_ground_truth)

    if not targeted_metrics and not ground_truth_metrics:
        raise MigrationError(
            "Unable to infer evaluation.targeted_metrics or evaluation.ground_truth_metrics "
            "from the legacy resolved_config and summary."
        )
    return {
        "topk": topk,
        "targeted_metrics": targeted_metrics,
        "ground_truth_metrics": ground_truth_metrics,
    }


def discover_metric_names_from_summary(
    summary_payload: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    targets_payload = _extract_nested_mapping(summary_payload, ("targets",))
    targeted: list[str] = []
    ground_truth: list[str] = []
    for target_payload in targets_payload.values():
        if not isinstance(target_payload, Mapping):
            continue
        victims_payload = _extract_nested_mapping(target_payload, ("victims",))
        for victim_payload in victims_payload.values():
            if not isinstance(victim_payload, Mapping):
                continue
            metrics_payload = _extract_nested_mapping(victim_payload, ("metrics",))
            for metric_key in metrics_payload:
                if not isinstance(metric_key, str):
                    continue
                match = METRIC_KEY_PATTERN.match(metric_key)
                if match is None:
                    continue
                metric_name = match.group("metric")
                if match.group("prefix") == "targeted":
                    targeted.append(metric_name)
                else:
                    ground_truth.append(metric_name)
    return _unique_preserve_order(targeted), _unique_preserve_order(ground_truth)


def resolve_attack_identity_context_for_migration(
    config: Config,
    *,
    run_type: str,
) -> Mapping[str, Any] | None:
    if run_type != POSITION_OPT_RUN_TYPE:
        return None

    position_opt_config = config.attack.position_opt
    if position_opt_config is None:
        raise MigrationError(
            "Position-opt legacy migration requires attack.position_opt to be present "
            "in the reconstructed config."
        )
    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=run_type,
        override=position_opt_config.clean_surrogate_checkpoint,
    ).resolve()
    if not clean_checkpoint.exists():
        raise MigrationError(
            "Position-opt legacy migration requires the clean surrogate checkpoint to exist: "
            f"'{clean_checkpoint}'."
        )
    return build_position_opt_attack_identity_context(
        position_opt_config=position_opt_identity_payload(position_opt_config),
        clean_surrogate_checkpoint=clean_checkpoint,
        runtime_seeds={
            "position_opt_seed": int(config.seeds.position_opt_seed),
            "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        },
    )


def resolve_legacy_selected_targets(
    *,
    discovery: LegacyRunDiscovery,
    summary_payload: Mapping[str, Any],
    target_info_payload: Mapping[str, Any] | None,
    artifact_manifest_payload: Mapping[str, Any] | None,
) -> list[int | str]:
    ordered_targets = _normalize_target_item_list(
        _extract_target_items_from_summary(summary_payload)
    )
    if discovery.selected_targets_path is not None and discovery.selected_targets_path.exists():
        selected_payload = _load_json_object(discovery.selected_targets_path, label="legacy selected_targets")
        loaded = _normalize_target_item_list(selected_payload.get("target_items"))
        if loaded:
            ordered_targets = _merge_target_orders(loaded, ordered_targets)
    if target_info_payload is not None:
        info_targets = _normalize_target_item_list(target_info_payload.get("target_items"))
        if info_targets:
            ordered_targets = _merge_target_orders(info_targets, ordered_targets)
    if artifact_manifest_payload is not None:
        victims_payload = artifact_manifest_payload.get("victims")
        if isinstance(victims_payload, Mapping):
            manifest_targets = _normalize_target_item_list(list(victims_payload.keys()))
            ordered_targets = _merge_target_orders(ordered_targets, manifest_targets)
    if not ordered_targets:
        raise MigrationError(
            f"Unable to determine any legacy target items for '{discovery.legacy_run_root}'."
        )
    return ordered_targets


def reconstruct_target_registry(
    *,
    config: Config,
    selected_targets: list[int | str],
    target_info_payload: Mapping[str, Any] | None,
    target_selection_meta_payload: Mapping[str, Any] | None,
    legacy_sampled_selected_targets_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    materialized_targets = _normalize_target_item_list(selected_targets)
    if not materialized_targets:
        raise MigrationError("Legacy migration requires at least one materialized target item.")

    if legacy_sampled_selected_targets_only:
        ordered_targets = sorted(materialized_targets, key=_target_sort_key)
        reconstruction_mode = "legacy_selected_targets_only"
        order_reconstructed = True
        explicit_list = list(ordered_targets)
    elif config.targets.mode == "explicit_list":
        ordered_targets = _normalize_target_item_list(config.targets.explicit_list)
        if not ordered_targets:
            ordered_targets = list(materialized_targets)
        if not _is_prefix(materialized_targets, ordered_targets):
            if not set(materialized_targets).issubset(set(ordered_targets)):
                raise MigrationError(
                    "Legacy explicit target migration found materialized targets outside the "
                    "configured explicit_list."
                )
            ordered_targets = [
                target_item
                for target_item in ordered_targets
                if target_item in set(materialized_targets)
            ]
            reconstruction_mode = "explicit_materialized_subset"
            order_reconstructed = False
        else:
            reconstruction_mode = "explicit_preserved"
            order_reconstructed = False
        explicit_list = list(ordered_targets)
    else:
        ordered_targets = sorted(materialized_targets, key=_target_sort_key)
        reconstruction_mode = "legacy_selected_targets_only"
        order_reconstructed = True
        explicit_list = []

    seed = (
        int(config.seeds.target_selection_seed)
        if config.targets.mode == "sampled"
        else None
    )
    bucket = config.targets.bucket if config.targets.mode == "sampled" else None
    candidate_basis = [str(item) for item in ordered_targets]
    now = _timestamp_utc()
    registry_payload: dict[str, Any] = {
        "target_cohort_key": target_cohort_key(config),
        "split_key": split_key(config),
        "selection_policy_version": "legacy_migration_import_v1",
        "mode": config.targets.mode,
        "bucket": bucket,
        "seed": seed,
        "explicit_list": explicit_list,
        "candidate_pool_hash": _sha1_payload(candidate_basis),
        "candidate_pool_size": int(len(candidate_basis)),
        "ordered_targets": list(ordered_targets),
        "current_count": int(len(materialized_targets)),
        "created_at": now,
        "updated_at": now,
        "imported_from_legacy": True,
        "migration": {
            "migration_version": MIGRATION_VERSION,
            "reconstruction_mode": reconstruction_mode,
            "order_reconstructed": bool(order_reconstructed),
            "materialized_targets": list(materialized_targets),
            "legacy_target_info_available": bool(target_info_payload),
            "legacy_target_selection_meta_available": bool(target_selection_meta_payload),
            "note": (
                "Legacy sampled cohorts only expose the selected/materialized target set. "
                "The original full sampled cohort order is not claimed unless legacy artifacts "
                "prove it."
                if order_reconstructed
                else "Legacy explicit target ordering was preserved."
            ),
        },
    }
    return registry_payload, dict(registry_payload["migration"])


def _normalize_sampled_legacy_targets_as_explicit_migration_cohort(
    config: Config,
    *,
    selected_targets: Sequence[int | str],
) -> Config:
    ordered_targets = tuple(
        int(target_item)
        for target_item in sorted(
            _normalize_target_item_list(list(selected_targets)),
            key=_target_sort_key,
        )
        if isinstance(target_item, int)
    )
    if not ordered_targets:
        raise MigrationError(
            "Legacy sampled migration requires at least one integer target item to "
            "construct an explicit migrated cohort."
        )
    return replace(
        config,
        targets=replace(
            config.targets,
            mode="explicit_list",
            explicit_list=ordered_targets,
            count=len(ordered_targets),
        ),
    )


def _migrate_legacy_source(source: LegacyRunSource) -> dict[str, Any]:
    metadata_paths = run_metadata_paths(
        source.config,
        run_type=source.run_type,
        attack_identity_context=source.attack_identity_context,
    )
    shared_paths = shared_artifact_paths(source.config, run_type=source.run_type)
    run_root = metadata_paths["run_root"]
    if run_root.exists() and any(run_root.iterdir()):
        raise MigrationError(
            "Migration destination already exists and is non-empty. Safe merging of legacy "
            f"inputs is not implemented for Phase 9: '{run_root}'."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    shared_paths["target_cohort_dir"].mkdir(parents=True, exist_ok=True)
    _copy_legacy_target_selection_artifacts(source, shared_paths=shared_paths)
    _copy_legacy_metadata_snapshots(source, run_root=run_root)

    existing_registry = load_target_registry(shared_paths["target_registry"])
    if existing_registry is None:
        save_target_registry(source.target_registry_payload, shared_paths["target_registry"])
    elif _normalize_registry_for_compatibility(existing_registry) != _normalize_registry_for_compatibility(
        source.target_registry_payload
    ):
        raise MigrationError(
            "Migration destination already contains a different target_registry.json for the "
            "same target cohort. Safe registry merging is not implemented for Phase 9: "
            f"'{shared_paths['target_registry']}'."
        )

    context = RunContext(
        canonical_dataset=object(),
        stats=object(),
        clean_sessions=[],
        clean_labels=[],
        export_paths=None,
        shared_paths=shared_paths,
        fake_session_count=int(source.summary_payload.get("fake_session_count", 0) or 0),
    )
    resolved_payload = _resolved_config_payload(
        source.config,
        run_type=source.run_type,
        attack_identity_context=source.attack_identity_context,
    )
    resolved_payload["migration"] = dict(source.migration_metadata)
    key_payloads = _key_payloads(
        source.config,
        run_type=source.run_type,
        attack_identity_context=source.attack_identity_context,
    )
    key_payloads["migration"] = dict(source.migration_metadata)
    artifact_manifest = _initial_artifact_manifest(
        source.config,
        context=context,
        run_type=source.run_type,
        metadata_paths=metadata_paths,
        attack_identity_context=source.attack_identity_context,
    )
    _annotate_manifest_with_legacy_shared_sources(
        artifact_manifest,
        legacy_artifact_manifest=source.artifact_manifest_payload,
    )
    artifact_manifest["migration"] = dict(source.migration_metadata)

    save_json(resolved_payload, metadata_paths["resolved_config"])
    save_json(key_payloads, metadata_paths["key_payloads"])
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])

    coverage = load_or_init_run_coverage(
        source.config,
        run_type=source.run_type,
        metadata_paths=metadata_paths,
        target_registry=source.target_registry_payload,
        attack_identity_context=source.attack_identity_context,
        allow_new_victims=True,
    )
    execution_log = load_or_init_execution_log(
        source.config,
        run_type=source.run_type,
        metadata_paths=metadata_paths,
        attack_identity_context=source.attack_identity_context,
    )
    execution_id = (
        "legacy-import-"
        f"{_sha1_payload([str(source.discovery.legacy_run_root), source.destination_run_root.as_posix()])[:12]}"
    )
    import_outcome = _import_legacy_cells(
        source=source,
        coverage=coverage,
        execution_id=execution_id,
        artifact_manifest=artifact_manifest,
    )
    source.migration_metadata["completed_cell_count"] = int(import_outcome["completed_cell_count"])
    coverage["imported_from_legacy"] = True
    coverage["migration"] = {
        **dict(source.migration_metadata),
        "completed_cell_count": int(import_outcome["completed_cell_count"]),
        "failed_cell_count": int(import_outcome["failed_cell_count"]),
        "requested_cell_count": int(import_outcome["requested_cell_count"]),
    }
    save_run_coverage(coverage, metadata_paths["run_coverage"])

    execution_record = build_import_execution_record(
        source=source,
        execution_id=execution_id,
        import_outcome=import_outcome,
    )
    executions = execution_log.setdefault("executions", [])
    if not isinstance(executions, list):
        raise MigrationError("execution_log.json executions must be a list during migration.")
    executions.append(execution_record)
    execution_log["updated_at"] = _timestamp_utc()
    execution_log["imported_from_legacy"] = True
    execution_log["migration"] = dict(source.migration_metadata)
    save_execution_log(execution_log, metadata_paths["execution_log"])

    save_json(
        build_migration_progress_payload(source, import_outcome, metadata_paths),
        metadata_paths["progress"],
    )

    summary_current = rebuild_summary_current(
        source.config,
        run_type=source.run_type,
        metadata_paths=metadata_paths,
        run_coverage=coverage,
        attack_identity_context=source.attack_identity_context,
    )
    summary_current["imported_from_legacy"] = True
    summary_current["migration"] = dict(source.migration_metadata)
    save_summary_current(summary_current, metadata_paths["summary_current"])
    save_json(
        build_legacy_summary_snapshot(
            source=source,
            summary_current=summary_current,
        ),
        metadata_paths["summary"],
    )

    artifact_manifest["output_files"]["summary"] = _to_repo_relative(metadata_paths["summary"])
    artifact_manifest["migration"] = {
        **dict(source.migration_metadata),
        "completed_cell_count": int(import_outcome["completed_cell_count"]),
        "failed_cell_count": int(import_outcome["failed_cell_count"]),
        "requested_cell_count": int(import_outcome["requested_cell_count"]),
    }
    save_json(artifact_manifest, metadata_paths["artifact_manifest"])

    persisted_coverage = load_run_coverage(metadata_paths["run_coverage"])
    persisted_execution_log = load_execution_log(metadata_paths["execution_log"])
    persisted_summary_current = load_summary_current(metadata_paths["summary_current"])
    if persisted_coverage is None or persisted_execution_log is None or persisted_summary_current is None:
        raise MigrationError("Migration did not persist the required new-architecture runtime artifacts.")

    return {
        "source_legacy_run_root": _display_path(source.discovery.legacy_run_root),
        "run_root": _display_path(run_root),
        "run_group_key": persisted_coverage["run_group_key"],
        "target_cohort_key": persisted_coverage["target_cohort_key"],
        "completed_cell_count": int(import_outcome["completed_cell_count"]),
        "failed_cell_count": int(import_outcome["failed_cell_count"]),
        "requested_cell_count": int(import_outcome["requested_cell_count"]),
    }


def _import_legacy_cells(
    *,
    source: LegacyRunSource,
    coverage: dict[str, Any],
    execution_id: str,
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    imported_at = _timestamp_utc()
    source_target_cells = _extract_legacy_target_cells(source)
    completed_cells: list[dict[str, Any]] = []
    failed_cells: list[dict[str, Any]] = []
    requested_cells: list[dict[str, Any]] = []

    for target_item in source.target_registry_payload["ordered_targets"][
        : source.target_registry_payload["current_count"]
    ]:
        target_key = str(target_item)
        target_cells = coverage["cells"][target_key]
        for victim_name in source.config.victims.enabled:
            artifacts = run_artifact_paths(
                source.config,
                run_type=source.run_type,
                target_id=target_item,
                victim_name=victim_name,
                attack_identity_context=source.attack_identity_context,
            )
            source_cell = source_target_cells.get((target_key, victim_name))
            source_local = source_cell.get("local", {}) if source_cell is not None else {}
            copied_artifacts = _copy_legacy_cell_artifacts(
                source_local=source_local,
                destination_artifacts=artifacts,
            )
            required_missing = [
                name
                for name in ("metrics", "predictions")
                if copied_artifacts.get(name) is None
            ]

            artifact_manifest.setdefault("victims", {}).setdefault(target_key, {})[victim_name] = {
                "imported_from_legacy": True,
                "reused_predictions": False,
                "source_legacy_run_root": _display_path(source.discovery.legacy_run_root),
                "source_legacy_cell_dir": (
                    _display_optional_path(_existing_path(source_local.get("run_dir")))
                    if source_local
                    else None
                ),
                "local": {
                    "run_dir": _to_repo_relative(artifacts["run_dir"]),
                    "resolved_config": _to_repo_relative(artifacts["resolved_config"]),
                    "config_snapshot": _to_repo_relative(artifacts["config_snapshot"]),
                    "predictions": _to_repo_relative(artifacts["predictions"]),
                    "metrics": _to_repo_relative(artifacts["metrics"]),
                    "train_history": _to_repo_relative(artifacts["train_history"]),
                    "poisoned_train": _to_repo_relative(artifacts["poisoned_train"]),
                },
                "shared": dict(source_cell.get("shared", {})) if source_cell is not None else {},
            }
            generated_configs = artifact_manifest.setdefault("generated_configs", {})
            if isinstance(generated_configs, dict) and source_cell is not None:
                extra = source_cell.get("generated_config")
                if isinstance(extra, Mapping):
                    generated_configs[f"{target_key}:{victim_name}"] = dict(extra)

            cell_payload = target_cells[victim_name]
            cell_payload["last_execution_id"] = execution_id
            cell_payload["last_requested_at"] = imported_at
            cell_payload["last_updated_at"] = imported_at
            cell_payload["attempt_count"] = max(1, int(cell_payload.get("attempt_count", 0)))
            cell_payload["migration"] = {
                "imported_from_legacy": True,
                "source_legacy_run_root": _display_path(source.discovery.legacy_run_root),
                "source_legacy_cell_dir": (
                    _display_optional_path(_existing_path(source_local.get("run_dir")))
                    if source_local
                    else None
                ),
            }
            for artifact_name, artifact_path in copied_artifacts.items():
                cell_payload["artifacts"][artifact_name] = (
                    _to_repo_relative(artifact_path) if artifact_path is not None else None
                )

            cell_ref = {
                "target_item": target_item,
                "victim_name": victim_name,
            }
            if required_missing:
                if source_cell is not None:
                    cell_payload["status"] = "failed"
                    cell_payload["completed_at"] = None
                    cell_payload["failed_at"] = imported_at
                    cell_payload["error"] = {
                        "type": "LegacyMigrationIncompleteCell",
                        "message": (
                            "Legacy cell artifacts were present but completion could not be trusted. "
                            f"Missing required artifacts: {', '.join(required_missing)}."
                        ),
                    }
                    failed_cells.append(
                        {
                            **cell_ref,
                            "status": "failed",
                            "error_type": "LegacyMigrationIncompleteCell",
                            "error": cell_payload["error"]["message"],
                        }
                    )
                else:
                    requested_cells.append({**cell_ref, "status": "requested"})
                continue

            if source_cell is None:
                requested_cells.append({**cell_ref, "status": "requested"})
                continue

            cell_payload["status"] = "completed"
            cell_payload["completed_at"] = imported_at
            cell_payload["failed_at"] = None
            cell_payload["error"] = None
            completed_cells.append({**cell_ref, "status": "completed"})

    return {
        "planned_cells": completed_cells + failed_cells + requested_cells,
        "completed_cells": completed_cells,
        "failed_cells": failed_cells,
        "requested_cells": requested_cells,
        "completed_cell_count": len(completed_cells),
        "failed_cell_count": len(failed_cells),
        "requested_cell_count": len(requested_cells),
    }


def build_import_execution_record(
    *,
    source: LegacyRunSource,
    execution_id: str,
    import_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    timestamp = _timestamp_utc()
    return {
        "execution_id": execution_id,
        "mode": "legacy_import",
        "requested_target_count": int(source.target_registry_payload["current_count"]),
        "requested_target_items": list(
            source.target_registry_payload["ordered_targets"][
                : source.target_registry_payload["current_count"]
            ]
        ),
        "requested_victims": list(source.config.victims.enabled),
        "planned_cells": [dict(cell) for cell in import_outcome["planned_cells"]],
        "completed_cells": [dict(cell) for cell in import_outcome["completed_cells"]],
        "failed_cells": [dict(cell) for cell in import_outcome["failed_cells"]],
        "skipped_completed_cells": [],
        "status": "completed",
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": timestamp,
        "elapsed_seconds": 0.0,
        "imported_from_legacy": True,
        "migration_version": MIGRATION_VERSION,
        "source_legacy_run_root": _display_path(source.discovery.legacy_run_root),
        "source_summary_path": _display_path(source.discovery.summary_path),
        "source_legacy_identifiers": {
            "legacy_target_selection_key": _extract_optional_string(
                _extract_nested_value(source.resolved_config_payload or {}, ("derived", "target_selection_key"))
            ),
            "legacy_evaluation_key": _extract_optional_string(
                _extract_nested_value(source.resolved_config_payload or {}, ("derived", "evaluation_key"))
            ),
        },
    }


def build_legacy_summary_snapshot(
    *,
    source: LegacyRunSource,
    summary_current: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_summary = source.summary_payload
    return {
        "run_type": source.run_type,
        "run_group_key": summary_current.get("run_group_key"),
        "target_cohort_key": summary_current.get("target_cohort_key"),
        "is_snapshot": True,
        "snapshot_source": "summary_current",
        "imported_from_legacy": True,
        "migration": dict(source.migration_metadata),
        "target_items": list(summary_current.get("target_items", [])),
        "victims": list(summary_current.get("victims", [])),
        "fake_session_count": int(legacy_summary.get("fake_session_count", 0) or 0),
        "clean_session_count": int(legacy_summary.get("clean_session_count", 0) or 0),
        "training": dict(_extract_nested_mapping(legacy_summary, ("training",))),
        "targets": dict(_extract_nested_mapping(summary_current, ("targets",))),
    }


def build_migration_progress_payload(
    source: LegacyRunSource,
    import_outcome: Mapping[str, Any],
    metadata_paths: Mapping[str, Path],
) -> dict[str, Any]:
    timestamp = _timestamp_utc()
    return {
        "run_type": source.run_type,
        "is_authoritative": False,
        "purpose": "legacy_migration_debug_snapshot",
        "authoritative_state": {
            "run_coverage": _to_repo_relative(metadata_paths["run_coverage"]),
            "execution_log": _to_repo_relative(metadata_paths["execution_log"]),
        },
        "status": "completed",
        "started_at": timestamp,
        "updated_at": timestamp,
        "completed_at": timestamp,
        "elapsed_seconds": 0.0,
        "total_targets": int(source.target_registry_payload["current_count"]),
        "total_victims": int(len(source.config.victims.enabled)),
        "total_runs": int(len(import_outcome["planned_cells"])),
        "completed_runs": int(len(import_outcome["completed_cells"])),
        "target_items": list(
            source.target_registry_payload["ordered_targets"][
                : source.target_registry_payload["current_count"]
            ]
        ),
        "requested_victims": list(source.config.victims.enabled),
        "planned_cells": [dict(cell) for cell in import_outcome["planned_cells"]],
        "skipped_completed_cells": [],
        "current": None,
        "runs": [
            {
                "overall_index": int(index),
                "target_item": cell["target_item"],
                "victim_name": cell["victim_name"],
                "status": cell["status"],
                "started_at": timestamp,
                "completed_at": timestamp,
                "reused_predictions": False,
            }
            for index, cell in enumerate(import_outcome["planned_cells"], start=1)
        ],
        "migration": dict(source.migration_metadata),
    }


def _validate_unique_destination_mapping(sources: Sequence[LegacyRunSource]) -> None:
    by_destination: dict[tuple[str, str], list[LegacyRunSource]] = {}
    for source in sources:
        destination_key = (
            source.destination_run_root.as_posix(),
            run_group_key(
                source.config,
                run_type=source.run_type,
                attack_identity_context=source.attack_identity_context,
            ),
        )
        by_destination.setdefault(destination_key, []).append(source)

    ambiguous = [grouped for grouped in by_destination.values() if len(grouped) > 1]
    if ambiguous:
        details = []
        for grouped in ambiguous:
            destination = grouped[0].destination_run_root
            sources_list = ", ".join(
                _display_path(item.discovery.legacy_run_root)
                for item in grouped
            )
            details.append(f"{destination} <= [{sources_list}]")
        raise MigrationError(
            "Multiple legacy inputs map to the same new run-group destination. Safe merging "
            "is not implemented for Phase 9. Conflicts: " + "; ".join(details)
        )


def _copy_legacy_target_selection_artifacts(
    source: LegacyRunSource,
    *,
    shared_paths: Mapping[str, Path],
) -> None:
    legacy_dir = Path(shared_paths["target_shared_dir"])
    legacy_dir.mkdir(parents=True, exist_ok=True)
    if source.discovery.config_snapshot_path is not None and source.discovery.config_snapshot_path.exists():
        _copy_file(source.discovery.config_snapshot_path, shared_paths["target_config_snapshot"])
    if source.discovery.selected_targets_path is not None and source.discovery.selected_targets_path.exists():
        _copy_file(source.discovery.selected_targets_path, shared_paths["selected_targets"])
    else:
        save_json({"target_items": list(source.selected_targets)}, shared_paths["selected_targets"])
    if source.discovery.target_selection_meta_path is not None and source.discovery.target_selection_meta_path.exists():
        _copy_file(source.discovery.target_selection_meta_path, shared_paths["target_selection_meta"])
    else:
        save_target_selection_meta(
            shared_paths["target_selection_meta"],
            {
                "imported_from_legacy": True,
                "migration_version": MIGRATION_VERSION,
                "selected_targets": list(source.selected_targets),
            },
        )
    if source.discovery.target_info_path is not None and source.discovery.target_info_path.exists():
        _copy_file(source.discovery.target_info_path, shared_paths["target_info"])
    else:
        save_target_info(
            shared_paths["target_info"],
            target_items=[int(item) for item in source.selected_targets if isinstance(item, int)],
            target_selection_mode=source.config.targets.mode,
            seed=int(source.config.seeds.target_selection_seed),
            bucket=source.config.targets.bucket if source.config.targets.mode == "sampled" else None,
            count=int(len(source.selected_targets)),
            explicit_list=list(source.config.targets.explicit_list),
        )


def _copy_legacy_metadata_snapshots(source: LegacyRunSource, *, run_root: Path) -> None:
    migration_dir = run_root / "migration"
    migration_dir.mkdir(parents=True, exist_ok=True)
    _copy_file(source.discovery.summary_path, migration_dir / "legacy_summary.json")
    if source.discovery.resolved_config_path is not None and source.discovery.resolved_config_path.exists():
        _copy_file(source.discovery.resolved_config_path, migration_dir / "legacy_resolved_config.json")
    if source.discovery.key_payloads_path is not None and source.discovery.key_payloads_path.exists():
        _copy_file(source.discovery.key_payloads_path, migration_dir / "legacy_key_payloads.json")
    if source.discovery.artifact_manifest_path is not None and source.discovery.artifact_manifest_path.exists():
        _copy_file(source.discovery.artifact_manifest_path, migration_dir / "legacy_artifact_manifest.json")


def _annotate_manifest_with_legacy_shared_sources(
    artifact_manifest: dict[str, Any],
    *,
    legacy_artifact_manifest: Mapping[str, Any] | None,
) -> None:
    if legacy_artifact_manifest is None:
        return
    canonical_split = legacy_artifact_manifest.get("canonical_split_artifact")
    if isinstance(canonical_split, Mapping):
        artifact_manifest.setdefault("shared_artifacts", {})["canonical_split"] = {
            str(key): _normalize_legacy_path_value(value)
            for key, value in canonical_split.items()
        }
    poison_artifact = legacy_artifact_manifest.get("poison_artifact")
    if poison_artifact is None or isinstance(poison_artifact, Mapping):
        artifact_manifest.setdefault("shared_artifacts", {})["poison_artifact"] = (
            {
                str(key): _normalize_legacy_path_value(value)
                for key, value in poison_artifact.items()
            }
            if isinstance(poison_artifact, Mapping)
            else None
        )


def preview_import_counts(
    *,
    config: Config,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None,
    target_registry_payload: Mapping[str, Any],
    source_target_cells: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, int]:
    del run_type, attack_identity_context
    completed = 0
    failed = 0
    requested = 0
    for target_item in target_registry_payload["ordered_targets"][
        : target_registry_payload["current_count"]
    ]:
        target_key = str(target_item)
        for victim_name in config.victims.enabled:
            source_cell = source_target_cells.get((target_key, victim_name))
            if source_cell is None:
                requested += 1
                continue
            source_local = source_cell.get("local", {})
            metrics_path = source_local.get("metrics")
            predictions_path = source_local.get("predictions")
            if isinstance(metrics_path, Path) and metrics_path.exists() and isinstance(predictions_path, Path) and predictions_path.exists():
                completed += 1
            else:
                failed += 1
    return {
        "completed_cell_count": completed,
        "failed_cell_count": failed,
        "requested_cell_count": requested,
    }


def _extract_legacy_target_cells(source: LegacyRunSource) -> dict[tuple[str, str], dict[str, Any]]:
    return _extract_legacy_target_cells_payload(
        discovery=source.discovery,
        summary_payload=source.summary_payload,
        artifact_manifest_payload=source.artifact_manifest_payload,
    )


def _extract_legacy_target_cells_payload(
    *,
    discovery: LegacyRunDiscovery,
    summary_payload: Mapping[str, Any],
    artifact_manifest_payload: Mapping[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    legacy_manifest = artifact_manifest_payload or {}
    legacy_victims_payload = legacy_manifest.get("victims")
    legacy_generated_configs = legacy_manifest.get("generated_configs")
    if isinstance(legacy_victims_payload, Mapping):
        for raw_target, target_payload in legacy_victims_payload.items():
            if not isinstance(target_payload, Mapping):
                continue
            target_key = str(_normalize_target_item(raw_target))
            for victim_name, victim_payload in target_payload.items():
                if not isinstance(victim_name, str) or not isinstance(victim_payload, Mapping):
                    continue
                local_payload = _extract_nested_mapping(victim_payload, ("local",))
                generated_config = None
                if isinstance(legacy_generated_configs, Mapping):
                    extra = legacy_generated_configs.get(f"{target_key}:{victim_name}")
                    if isinstance(extra, Mapping):
                        generated_config = dict(extra)
                cells[(target_key, victim_name)] = {
                    "local": {
                        name: _existing_path(
                            _manifest_path(path_value, legacy_run_root=discovery.legacy_run_root)
                        )
                        for name, path_value in local_payload.items()
                        if isinstance(name, str)
                    },
                    "shared": {
                        str(key): _normalize_legacy_path_value(value)
                        for key, value in _extract_nested_mapping(victim_payload, ("shared",)).items()
                    },
                    "generated_config": generated_config,
                }

    summary_targets = _extract_nested_mapping(summary_payload, ("targets",))
    for raw_target, target_payload in summary_targets.items():
        if not isinstance(target_payload, Mapping):
            continue
        target_key = str(_normalize_target_item(raw_target))
        victims_payload = _extract_nested_mapping(target_payload, ("victims",))
        for victim_name, victim_payload in victims_payload.items():
            if not isinstance(victim_name, str) or not isinstance(victim_payload, Mapping):
                continue
            cell_entry = cells.setdefault(
                (target_key, victim_name),
                {"local": {}, "shared": {}, "generated_config": None},
            )
            source_local = cell_entry.setdefault("local", {})
            metrics_path = _manifest_path(
                victim_payload.get("metrics_path"),
                legacy_run_root=discovery.legacy_run_root,
            )
            predictions_path = _manifest_path(
                victim_payload.get("predictions_path"),
                legacy_run_root=discovery.legacy_run_root,
            )
            if metrics_path is not None:
                source_local.setdefault("metrics", _existing_path(metrics_path))
                source_local.setdefault("run_dir", _existing_path(metrics_path.parent))
                source_local.setdefault("resolved_config", _existing_path(metrics_path.parent / "resolved_config.json"))
                source_local.setdefault("config_snapshot", _existing_path(metrics_path.parent / "config.yaml"))
                source_local.setdefault("train_history", _existing_path(metrics_path.parent / "train_history.json"))
                source_local.setdefault("poisoned_train", _existing_path(metrics_path.parent / "poisoned_train.txt"))
            if predictions_path is not None:
                source_local.setdefault("predictions", _existing_path(predictions_path))
                if source_local.get("run_dir") is None:
                    source_local["run_dir"] = _existing_path(predictions_path.parent)

    return cells


def _copy_legacy_cell_artifacts(
    *,
    source_local: Mapping[str, Any],
    destination_artifacts: Mapping[str, Path],
) -> dict[str, Path | None]:
    copied: dict[str, Path | None] = {}
    destination_artifacts["run_dir"].mkdir(parents=True, exist_ok=True)
    for artifact_name in LEGACY_CELL_CORE_FILES:
        source_path = source_local.get(artifact_name)
        if not isinstance(source_path, Path):
            copied[artifact_name] = None
            continue
        destination_path = destination_artifacts[artifact_name]
        _copy_file(source_path, destination_path)
        copied[artifact_name] = destination_path
    return copied


def _copy_file(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _normalize_registry_for_compatibility(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized.pop("created_at", None)
    normalized.pop("updated_at", None)
    return normalized


def _normalize_target_item_list(values: Any) -> list[int | str]:
    if values is None:
        return []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise MigrationError("Target item lists must be sequences.")
    return _unique_preserve_order(_normalize_target_item(value) for value in values)


def _normalize_target_item(value: Any) -> int | str:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise MigrationError("Target item identifiers must not be empty strings.")
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        raise MigrationError(
            "This repository expects numeric target item identifiers. "
            f"Unsupported legacy target item value: {value!r}."
        )
    raise MigrationError(f"Unsupported target item type: {type(value).__name__}")


def _extract_target_items_from_summary(summary_payload: Mapping[str, Any]) -> list[Any]:
    raw_target_items = summary_payload.get("target_items")
    if isinstance(raw_target_items, Sequence) and not isinstance(raw_target_items, (str, bytes)):
        return list(raw_target_items)
    targets_payload = _extract_nested_mapping(summary_payload, ("targets",))
    return list(targets_payload.keys())


def _merge_target_orders(primary: list[int | str], secondary: list[int | str]) -> list[int | str]:
    if not primary:
        return list(secondary)
    merged = list(primary)
    seen = set(primary)
    for item in secondary:
        if item in seen:
            continue
        merged.append(item)
        seen.add(item)
    return merged


def _target_sort_key(value: int | str) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def _is_prefix(prefix: Sequence[int | str], values: Sequence[int | str]) -> bool:
    return list(values[: len(prefix)]) == list(prefix)


def _unique_preserve_order(items: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_metric_name_list(value: Any) -> list[str]:
    return _unique_preserve_order(_normalize_string_list(value))


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MigrationError("Expected a sequence of strings.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise MigrationError("Expected a sequence of strings.")
        stripped = item.strip()
        if stripped:
            result.append(stripped)
    return result


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MigrationError(f"{label} must be a JSON object.")
    return dict(value)


def _require_sequence(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MigrationError(f"{label} must be a sequence.")
    return list(value)


def _extract_nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _extract_nested_mapping(payload: Mapping[str, Any], path: Sequence[str]) -> dict[str, Any]:
    current = _extract_nested_value(payload, path)
    if current is None:
        return {}
    if not isinstance(current, Mapping):
        raise MigrationError("Expected nested mapping at " + ".".join(path))
    return dict(current)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha1_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (REPO_ROOT / path_obj).resolve()


def _to_repo_relative(path: str | Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


def _manifest_path(raw_path: Any, *, legacy_run_root: Path) -> Path | None:
    if raw_path is None:
        return None
    if not isinstance(raw_path, (str, Path)):
        return None
    path_obj = Path(raw_path)
    if path_obj.is_absolute():
        return path_obj.resolve()
    repo_candidate = (REPO_ROOT / path_obj).resolve()
    if repo_candidate.exists():
        return repo_candidate
    run_candidate = (legacy_run_root / path_obj).resolve()
    if run_candidate.exists():
        return run_candidate
    return repo_candidate


def _normalize_legacy_path_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_legacy_path_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_legacy_path_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_legacy_path_value(item) for item in value]
    if isinstance(value, (str, Path)):
        resolved = _manifest_path(value, legacy_run_root=REPO_ROOT)
        return _display_optional_path(resolved)
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise MigrationError(f"{label} must contain a JSON object: '{path}'.")
    return dict(payload)


def _existing_path(path: Any) -> Path | None:
    if not isinstance(path, (str, Path)):
        return None
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj.resolve()
    return None


def _display_path(path: Path) -> str:
    return str(path.resolve())


def _display_optional_path(path: Path | None) -> str | None:
    if path is None:
        return None
    return _display_path(path)


def _extract_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


if __name__ == "__main__":
    main()
