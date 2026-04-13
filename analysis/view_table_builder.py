#!/usr/bin/env python3
"""Build one pivoted report-table bundle from a long-table CSV and YAML spec."""

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
ALLOWED_AGGREGATIONS = {
    "mean",
    "sum",
    "min",
    "max",
    "median",
    "count",
    "first",
    "last",
}
COLUMN_LABEL_SEPARATOR = " | "


class AnalysisError(ValueError):
    """Raised when a view spec or input table is malformed."""


@dataclass(frozen=True)
class ViewSpec:
    """Validated view spec content."""

    name: str
    input_csv: Path
    output_dir: Path
    filters: dict[str, Any]
    rows: list[str]
    cols: list[str]
    value_col: str
    agg: str


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 2 view-builder CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build one pivoted report-table bundle from a long-table CSV.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a view YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the view builder CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config, label="view config")
        spec = parse_view_spec(load_yaml_mapping(config_path, label="view config"))

        dataframe = pd.read_csv(spec.input_csv)
        validate_required_columns(
            dataframe,
            required_columns=spec.rows + spec.cols + [spec.value_col] + list(spec.filters.keys()),
            label="view input CSV",
        )

        filtered_dataframe = apply_filters(dataframe, spec.filters)
        if filtered_dataframe.empty:
            raise AnalysisError(
                f"The filters in '{config_path}' produced an empty table from '{spec.input_csv}'."
            )

        report_dataframe = build_report_table(filtered_dataframe, spec)
        bundle_dir = spec.output_dir / spec.name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        table_path = bundle_dir / "table.csv"
        meta_path = bundle_dir / "meta.json"
        report_dataframe.to_csv(table_path, index=False)

        meta = {
            "input_csv": to_repo_relative(spec.input_csv),
            "output_bundle_dir": to_repo_relative(bundle_dir),
            "filters": normalize_for_json(spec.filters),
            "rows": spec.rows,
            "cols": spec.cols,
            "value_col": spec.value_col,
            "agg": spec.agg,
            "filtered_row_count": int(len(filtered_dataframe)),
            "output_row_count": int(len(report_dataframe)),
            "output_column_count": int(len(report_dataframe.columns)),
            "generation_timestamp": utc_now_iso(),
            "context": build_context(filtered_dataframe),
        }
        write_json(meta_path, meta)

        print(
            f"Wrote report bundle '{bundle_dir}' from '{spec.input_csv}' "
            f"using view '{spec.name}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_view_spec(payload: Mapping[str, Any]) -> ViewSpec:
    """Validate and normalize a view YAML spec."""
    name = require_nonempty_string(payload.get("name"), label="name")
    input_csv = resolve_existing_path(
        require_nonempty_string(payload.get("input"), label="input"),
        label="view input CSV",
    )
    ensure_path_within(input_csv, RESULTS_ROOT, label="view input CSV")

    output_dir = resolve_repo_path(require_nonempty_string(payload.get("output_dir"), label="output_dir"))
    ensure_path_within(output_dir, RESULTS_ROOT, label="view output_dir")

    filters = normalize_filters(payload.get("filters", {}))
    rows = require_string_list(payload.get("rows"), label="rows")
    cols = require_string_list(payload.get("cols"), label="cols")
    value_col = require_nonempty_string(payload.get("value_col"), label="value_col")
    agg = require_nonempty_string(payload.get("agg"), label="agg").lower()
    if agg not in ALLOWED_AGGREGATIONS:
        raise AnalysisError(
            f"Unsupported agg '{agg}'. Allowed values: {sorted(ALLOWED_AGGREGATIONS)}."
        )

    return ViewSpec(
        name=name,
        input_csv=input_csv,
        output_dir=output_dir,
        filters=filters,
        rows=rows,
        cols=cols,
        value_col=value_col,
        agg=agg,
    )


def apply_filters(dataframe: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    """Apply equality and inclusion filters to a long-table dataframe."""
    filtered = dataframe.copy()
    for column, raw_value in filters.items():
        if column not in filtered.columns:
            raise AnalysisError(f"Filter column '{column}' does not exist in the input CSV.")

        if isinstance(raw_value, list):
            if not raw_value:
                raise AnalysisError(f"Filter '{column}' must not be an empty list.")
            filtered = filtered[filtered[column].isin(raw_value)]
        else:
            filtered = filtered[filtered[column] == raw_value]
    return filtered


def build_report_table(dataframe: pd.DataFrame, spec: ViewSpec) -> pd.DataFrame:
    """Build a pivoted report table and flatten it into a CSV-friendly dataframe."""
    try:
        pivoted = pd.pivot_table(
            dataframe,
            index=spec.rows,
            columns=spec.cols,
            values=spec.value_col,
            aggfunc=spec.agg,
            sort=True,
        )
    except Exception as exc:  # pragma: no cover - defensive pandas error wrapping
        raise AnalysisError(
            f"Could not build the report table with agg '{spec.agg}': {exc}."
        ) from exc

    if pivoted.empty:
        raise AnalysisError("The pivoted report table is empty after aggregation.")

    flattened = pivoted.reset_index()
    flattened.columns = [flatten_column_label(column) for column in flattened.columns]
    return flattened


def flatten_column_label(label: Any) -> str:
    """Flatten one pandas column label into a readable single string."""
    if isinstance(label, tuple):
        parts = [str(item) for item in label if str(item) != ""]
        if not parts:
            return "column"
        return COLUMN_LABEL_SEPARATOR.join(parts)
    return str(label)


def build_context(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Build render-friendly context values from the filtered long table."""
    context: dict[str, Any] = {}

    unique_datasets = collect_unique_values(dataframe, "dataset")
    if len(unique_datasets) == 1:
        context["dataset"] = unique_datasets[0]

    unique_target_types = collect_unique_values(dataframe, "target_type")
    if len(unique_target_types) == 1:
        context["target_type"] = unique_target_types[0]

    unique_attack_methods = collect_unique_values(dataframe, "attack_method")
    if unique_attack_methods:
        context["attack_methods"] = unique_attack_methods
        if len(unique_attack_methods) == 1:
            context["attack_method"] = unique_attack_methods[0]

    unique_victim_models = collect_unique_values(dataframe, "victim_model")
    if unique_victim_models:
        context["victim_models"] = unique_victim_models
        if len(unique_victim_models) == 1:
            context["victim_model"] = unique_victim_models[0]

    unique_metrics = collect_unique_values(dataframe, "metric")
    if unique_metrics:
        context["metrics"] = unique_metrics
        if len(unique_metrics) == 1:
            context["metric"] = unique_metrics[0]

    unique_ks = collect_unique_values(dataframe, "k")
    if unique_ks:
        context["ks"] = unique_ks
        if len(unique_ks) == 1:
            context["k"] = unique_ks[0]

    return normalize_for_json(context)


def collect_unique_values(dataframe: pd.DataFrame, column: str) -> list[Any]:
    """Collect sorted unique non-null values for one dataframe column."""
    if column not in dataframe.columns:
        return []

    values = [normalize_scalar(value) for value in dataframe[column].dropna().unique().tolist()]
    if not values:
        return []
    return sorted(values)


def validate_required_columns(
    dataframe: pd.DataFrame,
    *,
    required_columns: list[str],
    label: str,
) -> None:
    """Require a dataframe to contain every requested column."""
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise AnalysisError(
            f"The {label} is missing required columns: {missing_columns}. "
            f"Available columns: {list(dataframe.columns)}."
        )


def normalize_filters(value: Any) -> dict[str, Any]:
    """Normalize the optional filters mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AnalysisError("Expected 'filters' to be a mapping from column names to values.")

    normalized: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = require_nonempty_string(raw_key, label="filters key")
        if isinstance(raw_value, list):
            normalized[key] = [normalize_scalar(item) for item in raw_value]
        else:
            normalized[key] = normalize_scalar(raw_value)
    return normalized


def normalize_scalar(value: Any) -> Any:
    """Convert scalar values into JSON-safe Python primitives."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def normalize_for_json(value: Any) -> Any:
    """Recursively convert data into JSON-safe Python primitives."""
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]
    return normalize_scalar(value)


def load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a YAML file and require a top-level mapping."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid YAML: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level mapping.")
    return payload


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
    """Convert an in-repo path into a portable metadata string."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def utc_now_iso() -> str:
    """Return a UTC timestamp for metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
