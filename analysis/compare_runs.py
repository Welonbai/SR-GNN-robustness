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


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
COMPARISONS_ROOT = RESULTS_ROOT / "comparisons"
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

        frames: list[pd.DataFrame] = []
        source_csvs: list[str] = []
        source_row_counts: dict[str, int] = {}

        for run_id in spec.run_ids:
            long_table_path = RUNS_ROOT / run_id / "long_table.csv"
            if not long_table_path.is_file():
                raise AnalysisError(
                    f"Missing per-run long table for run_id '{run_id}': '{long_table_path}'. "
                    "Run Phase 1 first for every listed run."
                )

            dataframe = pd.read_csv(long_table_path)
            validate_long_table_columns(dataframe, path=long_table_path)
            frames.append(dataframe)
            source_csvs.append(to_repo_relative(long_table_path))
            source_row_counts[run_id] = int(len(dataframe))

        merged_dataframe = sort_merged_long_table(pd.concat(frames, ignore_index=True))
        if merged_dataframe.empty:
            raise AnalysisError("The merged comparison table is empty after loading all listed runs.")

        spec.output_dir.mkdir(parents=True, exist_ok=True)
        merged_csv_path = spec.output_dir / "merged_long_table.csv"
        manifest_path = spec.output_dir / "manifest.json"

        merged_dataframe.to_csv(merged_csv_path, index=False)
        manifest = {
            "comparison_id": spec.comparison_id,
            "source_run_ids": spec.run_ids,
            "source_csvs": source_csvs,
            "source_row_counts": source_row_counts,
            "output_dir": to_repo_relative(spec.output_dir),
            "generated_files": [
                "merged_long_table.csv",
                "manifest.json",
            ],
            "canonical_columns": CANONICAL_COLUMNS,
            "row_count": int(len(merged_dataframe)),
            "generation_timestamp": utc_now_iso(),
        }
        write_json(manifest_path, manifest)

        print(
            f"Wrote {len(merged_dataframe)} merged rows to '{merged_csv_path}' "
            f"for comparison_id '{spec.comparison_id}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_comparison_spec(payload: Mapping[str, Any]) -> ComparisonSpec:
    """Validate and normalize a comparison YAML spec."""
    comparison_id = require_nonempty_string(payload.get("comparison_id"), label="comparison_id")
    run_ids = require_string_list(payload.get("runs"), label="runs")
    output_dir = resolve_repo_path(require_nonempty_string(payload.get("output_dir"), label="output_dir"))
    ensure_path_within(output_dir, COMPARISONS_ROOT, label="comparison output_dir")

    return ComparisonSpec(
        comparison_id=comparison_id,
        run_ids=run_ids,
        output_dir=output_dir,
    )


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
