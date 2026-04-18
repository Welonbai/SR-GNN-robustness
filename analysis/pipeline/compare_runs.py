#!/usr/bin/env python3
"""Merge multiple per-run long tables into one comparison table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.utils.inventory_utils import build_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
COMPARISONS_ROOT = RESULTS_ROOT / "comparisons"
SLICE_COMPATIBILITY_MODES = {"strict", "relaxed_debug"}
CANONICAL_COLUMNS = [
    "run_id",
    "dataset",
    "attack_method",
    "victim_model",
    "target_item",
    "target_type",
    "attack_size",
    "poison_model",
    "fake_session_generation_topk",
    "replacement_topk_ratio",
    "metric",
    "k",
    "value",
]


class AnalysisError(ValueError):
    """Raised when a comparison spec or source bundle is malformed."""


@dataclass(frozen=True)
class ComparisonSpec:
    """Validated comparison spec content."""

    comparison_id: str
    run_ids: list[str]
    output_dir: Path
    slice_compatibility: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 2 comparison CLI parser."""
    parser = argparse.ArgumentParser(
        description="Merge multiple per-run long_table.csv files into one comparison bundle.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a comparison YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the comparison CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config, label="comparison config")
        spec = parse_comparison_spec(load_yaml_mapping(config_path, label="comparison config"))

        result = build_comparison_bundle(spec)

        print(
            f"Wrote {result['row_count']} merged rows to '{result['merged_csv_path']}' "
            f"for comparison_id '{spec.comparison_id}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_comparison_spec(payload: Mapping[str, Any]) -> ComparisonSpec:
    """Validate and normalize a comparison YAML spec."""
    comparison_id = require_nonempty_string(payload.get("comparison_id"), label="comparison_id")
    run_ids = require_string_list(payload.get("runs"), label="runs")
    slice_compatibility = require_nonempty_string(
        payload.get("slice_compatibility", "strict"),
        label="slice_compatibility",
    ).lower()
    if slice_compatibility not in SLICE_COMPATIBILITY_MODES:
        raise AnalysisError(
            "Unsupported slice_compatibility "
            f"'{slice_compatibility}'. Allowed values: {sorted(SLICE_COMPATIBILITY_MODES)}."
        )
    output_dir = COMPARISONS_ROOT / comparison_id

    legacy_output_dir = payload.get("output_dir")
    if legacy_output_dir is not None:
        legacy_output_dir_path = resolve_repo_path(
            require_nonempty_string(legacy_output_dir, label="output_dir")
        )
        ensure_path_within(legacy_output_dir_path, COMPARISONS_ROOT, label="comparison output_dir")
        if legacy_output_dir_path != output_dir.resolve():
            raise AnalysisError(
                "The comparison config no longer needs 'output_dir'. "
                f"It is derived as 'results/comparisons/{comparison_id}'."
            )

    return ComparisonSpec(
        comparison_id=comparison_id,
        run_ids=run_ids,
        output_dir=output_dir,
        slice_compatibility=slice_compatibility,
    )


def build_comparison_bundle(spec: ComparisonSpec) -> dict[str, Any]:
    """Build one comparison bundle from validated source run bundles."""
    frames: list[pd.DataFrame] = []
    source_csvs: list[str] = []
    source_row_counts: dict[str, int] = {}
    source_manifests: list[str] = []
    source_slice_manifests: list[str] = []
    normalized_source_slices: list[dict[str, Any]] = []

    for run_id in spec.run_ids:
        long_table_path = RUNS_ROOT / run_id / "long_table.csv"
        if not long_table_path.is_file():
            raise AnalysisError(
                f"Missing per-run long table for run_id '{run_id}': '{long_table_path}'. "
                "Run Phase 7 first for every listed run."
            )

        manifest_path = RUNS_ROOT / run_id / "manifest.json"
        if not manifest_path.is_file():
            raise AnalysisError(
                f"Missing manifest.json for run_id '{run_id}': '{manifest_path}'."
            )
        slice_manifest_path = RUNS_ROOT / run_id / "slice_manifest.json"
        if not slice_manifest_path.is_file():
            raise AnalysisError(
                f"Missing slice_manifest.json for run_id '{run_id}': '{slice_manifest_path}'. "
                "Comparison now requires slice-aware run bundles."
            )

        dataframe = pd.read_csv(long_table_path)
        validate_long_table_columns(dataframe, path=long_table_path)
        frames.append(dataframe)
        source_csvs.append(to_repo_relative(long_table_path))
        source_manifests.append(to_repo_relative(manifest_path))
        source_slice_manifests.append(to_repo_relative(slice_manifest_path))
        source_row_counts[run_id] = int(len(dataframe))
        normalized_source_slices.append(
            normalize_slice_manifest(
                load_json_mapping(slice_manifest_path, label=f"slice manifest for {run_id}"),
                run_id=run_id,
                slice_manifest_path=slice_manifest_path,
            )
        )

    incompatibilities = collect_slice_incompatibilities(normalized_source_slices)
    if spec.slice_compatibility == "strict":
        if incompatibilities:
            joined = "; ".join(incompatibilities)
            raise AnalysisError(
                "Strict slice compatibility failed for the requested comparison. "
                f"Incompatibilities: {joined}."
            )
        for normalized_slice in normalized_source_slices:
            if normalized_slice["fairness_safe"] is not True:
                raise AnalysisError(
                    "Strict slice compatibility requires fairness_safe=true for every source run. "
                    "Debug-only/non-fairness-safe slices such as all_available must use "
                    "slice_compatibility: relaxed_debug."
                )

    merged_dataframe = sort_merged_long_table(pd.concat(frames, ignore_index=True))
    if merged_dataframe.empty:
        raise AnalysisError("The merged comparison table is empty after loading all listed runs.")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = spec.output_dir / "merged_long_table.csv"
    inventory_path = spec.output_dir / "inventory.json"
    manifest_path = spec.output_dir / "manifest.json"
    slice_manifest_path = spec.output_dir / "slice_manifest.json"

    merged_dataframe.to_csv(merged_csv_path, index=False)
    write_json(inventory_path, build_inventory(merged_dataframe))

    merged_slice_metadata = build_merged_slice_metadata(
        spec,
        source_slices=normalized_source_slices,
        source_slice_manifests=source_slice_manifests,
        incompatibilities=incompatibilities,
    )
    write_json(slice_manifest_path, merged_slice_metadata)

    manifest = {
        "comparison_id": spec.comparison_id,
        "source_run_ids": spec.run_ids,
        "source_csvs": source_csvs,
        "source_manifests": source_manifests,
        "source_slice_manifests": source_slice_manifests,
        "source_row_counts": source_row_counts,
        "output_dir": to_repo_relative(spec.output_dir),
        "generated_files": [
            "inventory.json",
            "manifest.json",
            "merged_long_table.csv",
            "slice_manifest.json",
        ],
        "canonical_columns": CANONICAL_COLUMNS,
        "row_count": int(len(merged_dataframe)),
        "slice_compatibility": {
            "mode": spec.slice_compatibility,
            "debug_only": spec.slice_compatibility != "strict",
            "compatible": not incompatibilities,
            "incompatibilities": incompatibilities,
        },
        "slice": merged_slice_metadata,
        "generation_timestamp": utc_now_iso(),
    }
    write_json(manifest_path, manifest)

    return {
        "merged_csv_path": merged_csv_path,
        "inventory_path": inventory_path,
        "manifest_path": manifest_path,
        "slice_manifest_path": slice_manifest_path,
        "row_count": int(len(merged_dataframe)),
    }


def normalize_slice_manifest(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    slice_manifest_path: Path,
) -> dict[str, Any]:
    """Normalize one slice manifest for strict compatibility comparison."""
    slice_policy = require_nonempty_string(payload.get("slice_policy"), label="slice_policy")
    requested_victims = require_string_list(payload.get("requested_victims"), label="requested_victims")
    selected_targets_raw = payload.get("selected_targets")
    if not isinstance(selected_targets_raw, list):
        raise AnalysisError(
            f"The slice manifest for run_id '{run_id}' at '{slice_manifest_path}' must contain "
            "a selected_targets list."
        )
    selected_targets = [
        normalize_target_item(item, label=f"selected_targets[{index}]")
        for index, item in enumerate(selected_targets_raw)
    ]
    selected_target_count = require_int(
        payload.get("selected_target_count"),
        label="selected_target_count",
    )
    if selected_target_count != len(selected_targets):
        raise AnalysisError(
            f"The slice manifest for run_id '{run_id}' at '{slice_manifest_path}' is inconsistent: "
            f"selected_target_count={selected_target_count}, len(selected_targets)={len(selected_targets)}."
        )
    fairness_safe = payload.get("fairness_safe")
    if not isinstance(fairness_safe, bool):
        raise AnalysisError(
            f"The slice manifest for run_id '{run_id}' at '{slice_manifest_path}' must contain "
            "a boolean fairness_safe field."
        )
    return {
        "run_id": run_id,
        "slice_manifest_path": to_repo_relative(slice_manifest_path),
        "source_run_group_key": optional_nonempty_string(payload.get("source_run_group_key")),
        "slice_policy": slice_policy,
        "fairness_safe": fairness_safe,
        "requested_victims": requested_victims,
        "requested_victims_source": optional_nonempty_string(payload.get("requested_victims_source")),
        "selected_targets": selected_targets,
        "selected_target_count": selected_target_count,
        "target_cohort_key": optional_nonempty_string(payload.get("target_cohort_key")),
    }


def collect_slice_incompatibilities(source_slices: list[dict[str, Any]]) -> list[str]:
    """Collect strict-compatibility mismatches across source slice manifests."""
    if not source_slices:
        return []
    baseline = source_slices[0]
    incompatibilities: list[str] = []
    for current in source_slices[1:]:
        for field_name in (
            "slice_policy",
            "fairness_safe",
            "requested_victims",
            "selected_targets",
            "selected_target_count",
        ):
            if current[field_name] == baseline[field_name]:
                continue
            incompatibilities.append(
                f"{current['run_id']} differs from {baseline['run_id']} on {field_name}: "
                f"{current[field_name]!r} != {baseline[field_name]!r}"
            )
    return incompatibilities


def build_merged_slice_metadata(
    spec: ComparisonSpec,
    *,
    source_slices: list[dict[str, Any]],
    source_slice_manifests: list[str],
    incompatibilities: list[str],
) -> dict[str, Any]:
    """Build merged slice metadata for one comparison bundle."""
    return {
        "comparison_id": spec.comparison_id,
        "source_run_ids": list(spec.run_ids),
        "source_slice_manifests": list(source_slice_manifests),
        "source_run_group_keys": unique_ordered_values(
            [
                source_slice["source_run_group_key"]
                for source_slice in source_slices
                if source_slice.get("source_run_group_key")
            ]
        ),
        "target_cohort_keys": unique_ordered_values(
            [
                source_slice["target_cohort_key"]
                for source_slice in source_slices
                if source_slice.get("target_cohort_key")
            ]
        ),
        "slice_policy": common_value(source_slices, "slice_policy"),
        "requested_victims": common_value(source_slices, "requested_victims"),
        "requested_victims_source": common_value(source_slices, "requested_victims_source"),
        "selected_targets": common_value(source_slices, "selected_targets"),
        "selected_target_count": common_value(source_slices, "selected_target_count"),
        "fairness_safe": common_value(source_slices, "fairness_safe"),
        "compatibility_mode": spec.slice_compatibility,
        "debug_only": spec.slice_compatibility != "strict",
        "compatible": not incompatibilities,
        "incompatibilities": list(incompatibilities),
        "generation_timestamp": utc_now_iso(),
    }


def common_value(source_slices: list[dict[str, Any]], field_name: str) -> Any | None:
    """Return a field value when every source slice agrees, otherwise None."""
    if not source_slices:
        return None
    value = source_slices[0].get(field_name)
    if all(source_slice.get(field_name) == value for source_slice in source_slices[1:]):
        return value
    return None


def unique_ordered_values(values: list[str]) -> list[str]:
    """Return ordered unique strings."""
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def resolve_existing_path(raw_path: str, *, label: str) -> Path:
    """Resolve and validate one existing file path."""
    path = resolve_repo_path(raw_path)
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def ensure_path_within(path: Path, root: Path, *, label: str) -> None:
    """Require a path to stay inside one repository subtree."""
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(
            f"The {label} must stay under '{root.resolve()}', got '{path}'."
        ) from exc


def load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a YAML file and require a top-level mapping."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid YAML: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level mapping.")
    return payload


def load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON file and require a top-level mapping."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid JSON: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level mapping.")
    return payload


def validate_long_table_columns(dataframe: pd.DataFrame, *, path: Path) -> None:
    """Require one long table CSV to match the canonical schema exactly."""
    columns = list(dataframe.columns)
    if columns != CANONICAL_COLUMNS:
        raise AnalysisError(
            f"The long table at '{path}' does not match the canonical columns. "
            f"Expected {CANONICAL_COLUMNS}, got {columns}."
        )


def sort_merged_long_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministically ordered merged long table."""
    target_item_order = dataframe["target_item"].map(build_target_item_sort_key)
    sorted_dataframe = (
        dataframe.assign(
            _target_item_type_order=target_item_order.map(lambda item: item[0]),
            _target_item_int_order=target_item_order.map(lambda item: item[1]),
            _target_item_str_order=target_item_order.map(lambda item: item[2]),
        )
        .sort_values(
            by=[
                "run_id",
                "_target_item_type_order",
                "_target_item_int_order",
                "_target_item_str_order",
                "victim_model",
                "metric",
                "k",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_target_item_type_order",
                "_target_item_int_order",
                "_target_item_str_order",
            ]
        )
        .reset_index(drop=True)
    )
    return sorted_dataframe


def build_target_item_sort_key(value: Any) -> tuple[int, int, str]:
    """Build a deterministic sort key for int-or-string target identifiers."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, int):
        return (0, normalized_value, "")
    if isinstance(normalized_value, str):
        return (1, 0, normalized_value.casefold())
    raise AnalysisError(
        f"Expected 'target_item' to be int or str while sorting, got {type(normalized_value).__name__}."
    )


def normalize_scalar(value: Any) -> Any:
    """Convert pandas scalar types into plain Python primitives."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def optional_nonempty_string(value: Any) -> str | None:
    """Return a stripped non-empty string when present, otherwise None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnalysisError(f"Expected a string value, got {type(value).__name__}.")
    stripped = value.strip()
    return stripped or None


def normalize_target_item(value: Any, *, label: str) -> int | str:
    """Normalize one target identifier from slice metadata."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, bool) or normalized_value is None:
        raise AnalysisError(f"Expected '{label}' to be an int or non-empty string.")
    if isinstance(normalized_value, int):
        return int(normalized_value)
    if isinstance(normalized_value, str):
        stripped = normalized_value.strip()
        if not stripped:
            raise AnalysisError(f"Expected '{label}' to be a non-empty string.")
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    raise AnalysisError(
        f"Expected '{label}' to be an int or non-empty string, got {type(normalized_value).__name__}."
    )


def require_nonempty_string(value: Any, *, label: str) -> str:
    """Require a non-empty string value."""
    if not isinstance(value, str):
        raise AnalysisError(f"Expected '{label}' to be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise AnalysisError(f"Expected '{label}' to be a non-empty string.")
    return stripped


def require_string_list(value: Any, *, label: str) -> list[str]:
    """Require a non-empty list of non-empty strings."""
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"Expected '{label}' to be a non-empty list of strings.")

    normalized: list[str] = []
    for index, item in enumerate(value):
        normalized.append(require_nonempty_string(item, label=f"{label}[{index}]"))
    return normalized


def require_int(value: Any, *, label: str) -> int:
    """Require an integer value."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be an integer, got bool.")
    if isinstance(value, int):
        return value
    raise AnalysisError(f"Expected '{label}' to be an integer, got {type(value).__name__}.")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write formatted JSON with a stable layout."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_repo_relative(path: Path) -> str:
    """Convert an in-repo path into a portable manifest string."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def utc_now_iso() -> str:
    """Return a UTC timestamp for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
