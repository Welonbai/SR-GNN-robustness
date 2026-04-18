#!/usr/bin/env python3
"""Build one or more pivoted report-table bundles from a long-table CSV and YAML spec."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
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
METRIC_COLUMN = "metric"
METRIC_NAME_COLUMN = "metric_name"
METRIC_SCOPE_COLUMN = "metric_scope"
GROUND_TRUTH_SCOPE = "ground_truth"
TARGETED_SCOPE = "targeted"
METRIC_SCOPE_PREFIXES = (
    ("ground_truth_", GROUND_TRUTH_SCOPE),
    ("targeted_", TARGETED_SCOPE),
)
DERIVED_VIEW_COLUMNS = {METRIC_NAME_COLUMN, METRIC_SCOPE_COLUMN}
UNSAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
REPEATED_UNDERSCORE_PATTERN = re.compile(r"_+")


class AnalysisError(ValueError):
    """Raised when a view spec or input table is malformed."""


@dataclass(frozen=True)
class ViewSpec:
    """Validated view spec content."""

    input_csv: Path
    output_bundle_dir: Path
    source_spec_path: Path
    parent_spec_name: str
    filters: dict[str, Any]
    split_by: list[str]
    rows: list[str]
    cols: list[str]
    value_col: str
    agg: str
    auto_context: bool
    require_unique_cells: bool


@dataclass(frozen=True)
class HiddenColumnSummary:
    """Describe omitted columns after filtering."""

    singleton_values: dict[str, Any]
    varying_columns: list[str]


@dataclass(frozen=True)
class PivotStructure:
    """Describe the logical row/column hierarchy of one pivoted table."""

    row_levels: list[str]
    col_levels: list[str]
    row_tuples: list[list[Any]]
    column_tuples: list[list[Any]]


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
        spec = parse_view_spec(
            load_yaml_mapping(config_path, label="view config"),
            source_spec_path=config_path,
        )

        output_bundle_dirs = build_view_bundles(spec)
        bundle_count = len(output_bundle_dirs)

        print(
            f"Wrote {bundle_count} report bundle(s) under '{spec.output_bundle_dir.parent}' "
            f"from '{spec.input_csv}' using parent spec '{spec.parent_spec_name}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def build_view_bundles(spec: ViewSpec) -> list[Path]:
    """Build one or more view bundles from a validated view spec."""
    dataframe = prepare_view_dataframe(pd.read_csv(spec.input_csv), spec)
    validate_required_columns(
        dataframe,
        required_columns=(
            spec.rows + spec.cols + spec.split_by + [spec.value_col] + list(spec.filters.keys())
        ),
        label="view input CSV",
    )

    filtered_dataframe = apply_filters(dataframe, spec.filters)
    if filtered_dataframe.empty:
        raise AnalysisError(
            f"The filters produced an empty table from '{spec.input_csv}'."
        )

    input_analysis_metadata = load_input_analysis_metadata(spec.input_csv)
    output_bundle_dirs: list[Path] = []
    for split_values, bundle_dataframe in iter_view_bundle_inputs(filtered_dataframe, spec):
        bundle_dir = resolve_bundle_dir(spec=spec, split_values=split_values)
        write_view_bundle(
            dataframe=bundle_dataframe,
            spec=spec,
            bundle_dir=bundle_dir,
            split_values=split_values,
            input_analysis_metadata=input_analysis_metadata,
        )
        output_bundle_dirs.append(bundle_dir)
    return output_bundle_dirs


def parse_view_spec(payload: Mapping[str, Any], *, source_spec_path: Path) -> ViewSpec:
    """Validate and normalize a view YAML spec."""
    input_csv = resolve_existing_path(
        require_nonempty_string(payload.get("input"), label="input"),
        label="view input CSV",
    )
    ensure_path_within(input_csv, RESULTS_ROOT, label="view input CSV")

    output_bundle_dir = resolve_output_bundle_dir(payload)
    ensure_path_within(output_bundle_dir, RESULTS_ROOT, label="view output")

    parent_spec_name = resolve_parent_spec_name(payload, output_bundle_dir=output_bundle_dir)
    filters = normalize_filters(payload.get("filters", {}))
    split_by = normalize_optional_string_list(payload.get("split_by"), label="split_by")
    rows = require_string_list(payload.get("rows"), label="rows")
    cols = require_string_list(payload.get("cols"), label="cols")
    value_col = require_nonempty_string(payload.get("value_col"), label="value_col")
    agg = require_nonempty_string(payload.get("agg"), label="agg").lower()
    auto_context = require_bool(payload.get("auto_context", False), label="auto_context")
    require_unique_cells = require_bool(
        payload.get("require_unique_cells", False),
        label="require_unique_cells",
    )
    if agg not in ALLOWED_AGGREGATIONS:
        raise AnalysisError(
            f"Unsupported agg '{agg}'. Allowed values: {sorted(ALLOWED_AGGREGATIONS)}."
        )

    return ViewSpec(
        input_csv=input_csv,
        output_bundle_dir=output_bundle_dir,
        source_spec_path=source_spec_path,
        parent_spec_name=parent_spec_name,
        filters=filters,
        split_by=split_by,
        rows=rows,
        cols=cols,
        value_col=value_col,
        agg=agg,
        auto_context=auto_context,
        require_unique_cells=require_unique_cells,
    )


def resolve_output_bundle_dir(payload: Mapping[str, Any]) -> Path:
    """Resolve the final view bundle directory from one config mapping."""
    output_value = payload.get("output")
    if output_value is not None:
        output_bundle_dir = resolve_repo_path(require_nonempty_string(output_value, label="output"))
        return output_bundle_dir

    legacy_name = payload.get("name")
    legacy_output_dir = payload.get("output_dir")
    if legacy_name is None or legacy_output_dir is None:
        raise AnalysisError("The view config must contain 'output'.")

    output_dir = resolve_repo_path(require_nonempty_string(legacy_output_dir, label="output_dir"))
    name = require_nonempty_string(legacy_name, label="name")
    return output_dir / name


def resolve_parent_spec_name(payload: Mapping[str, Any], *, output_bundle_dir: Path) -> str:
    """Resolve the stable parent bundle name used for split outputs."""
    raw_name = payload.get("name")
    if raw_name is None:
        return output_bundle_dir.name
    return require_nonempty_string(raw_name, label="name")


def prepare_view_dataframe(dataframe: pd.DataFrame, spec: ViewSpec) -> pd.DataFrame:
    """Attach derived view-time columns that are requested by the spec."""
    requested_columns = set(spec.rows)
    requested_columns.update(spec.cols)
    requested_columns.update(spec.split_by)
    requested_columns.update(spec.filters.keys())

    missing_derived_columns = sorted(
        column_name
        for column_name in requested_columns
        if column_name in DERIVED_VIEW_COLUMNS and column_name not in dataframe.columns
    )
    if not missing_derived_columns:
        return dataframe.copy()

    if METRIC_COLUMN not in dataframe.columns:
        raise AnalysisError(
            "The view spec references derived metric columns "
            f"{missing_derived_columns}, but the input CSV does not contain '{METRIC_COLUMN}'."
        )

    prepared = dataframe.copy()
    derived_metric_columns = derive_metric_identity_columns(prepared[METRIC_COLUMN])
    for column_name in missing_derived_columns:
        prepared[column_name] = derived_metric_columns[column_name]
    return prepared


def derive_metric_identity_columns(metric_series: pd.Series) -> dict[str, pd.Series]:
    """Split raw metric labels into view-friendly metric_name and metric_scope columns."""
    derived_pairs = metric_series.map(derive_metric_identity)
    return {
        METRIC_NAME_COLUMN: derived_pairs.map(lambda item: item[0]),
        METRIC_SCOPE_COLUMN: derived_pairs.map(lambda item: item[1]),
    }


def derive_metric_identity(metric_value: Any) -> tuple[str, str]:
    """Derive one metric name and semantic scope from a raw metric label."""
    normalized_metric_value = normalize_scalar(metric_value)
    if not isinstance(normalized_metric_value, str):
        raise AnalysisError(
            "Could not derive metric_name/metric_scope because one metric value is not a string: "
            f"{normalized_metric_value!r}."
        )

    metric_token = normalized_metric_value.strip().lower()
    if not metric_token:
        raise AnalysisError("Could not derive metric_name/metric_scope from an empty metric value.")

    for prefix, scope in METRIC_SCOPE_PREFIXES:
        if metric_token.startswith(prefix):
            metric_name = metric_token[len(prefix) :]
            if not metric_name:
                raise AnalysisError(
                    f"Could not derive metric_name from raw metric '{normalized_metric_value}'."
                )
            return metric_name, scope
    return metric_token, TARGETED_SCOPE


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


def iter_view_bundle_inputs(
    filtered_dataframe: pd.DataFrame,
    spec: ViewSpec,
) -> list[tuple[dict[str, Any], pd.DataFrame]]:
    """Return one or more per-bundle filtered dataframes after optional splitting."""
    if not spec.split_by:
        return [({}, filtered_dataframe.copy())]

    bundle_inputs: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for split_values in extract_split_assignments(filtered_dataframe, split_by=spec.split_by):
        bundle_inputs.append(
            (
                split_values,
                apply_split_values(filtered_dataframe, split_values),
            )
        )
    return bundle_inputs


def extract_split_assignments(
    dataframe: pd.DataFrame,
    *,
    split_by: list[str],
) -> list[dict[str, Any]]:
    """Collect deterministic split assignments from one filtered dataframe."""
    unique_rows = dataframe[split_by].drop_duplicates().reset_index(drop=True)
    assignments: list[dict[str, Any]] = []
    for _, row in unique_rows.iterrows():
        assignments.append(
            {column_name: normalize_scalar(row[column_name]) for column_name in split_by}
        )
    return sorted(assignments, key=build_split_assignment_sort_key)


def apply_split_values(dataframe: pd.DataFrame, split_values: Mapping[str, Any]) -> pd.DataFrame:
    """Filter one dataframe down to a single split assignment."""
    filtered = dataframe.copy()
    for column_name, raw_value in split_values.items():
        if column_name not in filtered.columns:
            raise AnalysisError(f"Split column '{column_name}' does not exist in the input CSV.")

        if pd.isna(raw_value):
            filtered = filtered[filtered[column_name].isna()]
        else:
            filtered = filtered[filtered[column_name] == raw_value]

    if filtered.empty:
        raise AnalysisError(
            f"The split values {dict(split_values)} produced an empty bundle after filtering."
        )
    return filtered


def resolve_bundle_dir(*, spec: ViewSpec, split_values: Mapping[str, Any]) -> Path:
    """Resolve the final output directory for one bundle."""
    if not split_values:
        return spec.output_bundle_dir
    return spec.output_bundle_dir.parent / build_bundle_name(
        parent_spec_name=spec.parent_spec_name,
        split_by=spec.split_by,
        split_values=split_values,
    )


def build_bundle_name(
    *,
    parent_spec_name: str,
    split_by: list[str],
    split_values: Mapping[str, Any],
) -> str:
    """Build one deterministic bundle directory name from split values."""
    ordered_split_values = normalize_split_value_mapping(split_by, split_values)
    parts = [sanitize_component(parent_spec_name)]
    for column_name, raw_value in ordered_split_values.items():
        parts.append(f"{sanitize_component(column_name)}_{sanitize_component(stringify_split_value(raw_value))}")
    return "__".join(parts)


def write_view_bundle(
    *,
    dataframe: pd.DataFrame,
    spec: ViewSpec,
    bundle_dir: Path,
    split_values: Mapping[str, Any],
    input_analysis_metadata: Mapping[str, Any],
) -> None:
    """Write one fully-renderable view bundle."""
    hidden_column_summary = summarize_hidden_columns(dataframe, spec)
    if spec.require_unique_cells:
        validate_unique_cells(dataframe, spec)

    report_dataframe, pivot_structure = build_report_table(dataframe, spec)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    table_path = bundle_dir / "table.csv"
    meta_path = bundle_dir / "meta.json"
    report_dataframe.to_csv(table_path, index=False)

    normalized_split_values = normalize_split_value_mapping(spec.split_by, split_values)
    json_split_values = normalize_for_json(normalized_split_values)
    slice_metadata = input_analysis_metadata.get("slice")
    if not isinstance(slice_metadata, Mapping):
        slice_metadata = None
    slice_context = input_analysis_metadata.get("slice_context")
    if not isinstance(slice_context, Mapping):
        slice_context = {}
    forced_context = dict(json_split_values)
    forced_context.update(normalize_for_json(slice_context))
    meta = {
        "mode": infer_optional_mode(dataframe, spec),
        "input_csv": to_repo_relative(spec.input_csv),
        "source_view_spec_path": to_repo_relative(spec.source_spec_path),
        "output_bundle_dir": to_repo_relative(bundle_dir),
        "bundle_output_dir": to_repo_relative(bundle_dir),
        "source_manifest_path": input_analysis_metadata.get("source_manifest_path"),
        "source_slice_manifest_path": input_analysis_metadata.get("source_slice_manifest_path"),
        "parent_spec_name": spec.parent_spec_name,
        "bundle_name": bundle_dir.name,
        "filters": normalize_for_json(spec.filters),
        "effective_filters": build_effective_filters(spec.filters, normalized_split_values),
        "split_by": spec.split_by,
        "split_values": json_split_values,
        "rows": spec.rows,
        "cols": spec.cols,
        "row_levels": pivot_structure.row_levels,
        "col_levels": pivot_structure.col_levels,
        "row_tuples": pivot_structure.row_tuples,
        "column_tuples": pivot_structure.column_tuples,
        "value_col": spec.value_col,
        "agg": spec.agg,
        "unused_singleton_columns": list(hidden_column_summary.singleton_values.keys()),
        "unused_varying_columns": hidden_column_summary.varying_columns,
        "aggregated_over": (
            hidden_column_summary.varying_columns if not spec.require_unique_cells else []
        ),
        "filtered_row_count": int(len(dataframe)),
        "output_row_count": int(len(report_dataframe)),
        "output_column_count": int(len(report_dataframe.columns)),
        "generation_timestamp": utc_now_iso(),
        "slice": normalize_for_json(slice_metadata) if slice_metadata is not None else None,
        "slice_context": normalize_for_json(slice_context),
        "context": build_context(
            dataframe,
            hidden_column_summary=hidden_column_summary,
            auto_context=spec.auto_context,
            forced_context=forced_context,
        ),
    }
    write_json(meta_path, meta)


def build_report_table(dataframe: pd.DataFrame, spec: ViewSpec) -> tuple[pd.DataFrame, PivotStructure]:
    """Build one pivoted report table plus structural metadata for the view bundle."""
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

    pivot_structure = PivotStructure(
        row_levels=extract_axis_levels(pivoted.index, fallback_levels=spec.rows),
        col_levels=extract_axis_levels(pivoted.columns, fallback_levels=spec.cols),
        row_tuples=extract_axis_tuples(pivoted.index, level_count=len(spec.rows), axis_label="rows"),
        column_tuples=extract_axis_tuples(
            pivoted.columns,
            level_count=len(spec.cols),
            axis_label="columns",
        ),
    )
    flattened = pivoted.reset_index()
    flattened.columns = [flatten_column_label(column) for column in flattened.columns]
    return flattened, pivot_structure


def summarize_hidden_columns(dataframe: pd.DataFrame, spec: ViewSpec) -> HiddenColumnSummary:
    """Classify columns omitted from rows, cols, filters, and value_col."""
    singleton_values: dict[str, Any] = {}
    varying_columns: list[str] = []

    for column in collect_hidden_columns(dataframe, spec):
        values = collect_unique_values(dataframe, column)
        if len(values) == 1:
            singleton_values[column] = values[0]
        elif len(values) > 1:
            varying_columns.append(column)

    return HiddenColumnSummary(
        singleton_values=singleton_values,
        varying_columns=varying_columns,
    )


def collect_hidden_columns(dataframe: pd.DataFrame, spec: ViewSpec) -> list[str]:
    """Return columns omitted from the rendered table."""
    excluded = set(spec.rows)
    excluded.update(spec.cols)
    excluded.update(spec.split_by)
    excluded.update(spec.filters.keys())
    excluded.add(spec.value_col)
    if is_metric_semantically_represented(spec):
        excluded.add(METRIC_COLUMN)
    return [column for column in dataframe.columns if column not in excluded]


def is_metric_semantically_represented(spec: ViewSpec) -> bool:
    """Return whether the spec already exposes both derived metric dimensions."""
    represented_columns = set(spec.rows)
    represented_columns.update(spec.cols)
    represented_columns.update(spec.split_by)
    represented_columns.update(spec.filters.keys())
    return METRIC_NAME_COLUMN in represented_columns and METRIC_SCOPE_COLUMN in represented_columns


def validate_unique_cells(dataframe: pd.DataFrame, spec: ViewSpec) -> None:
    """Require each output pivot cell to map to at most one source row."""
    cell_dimensions = spec.rows + spec.cols
    counts = (
        dataframe.groupby(cell_dimensions, dropna=False, sort=True)
        .size()
        .reset_index(name="_row_count")
    )
    conflicts = counts[counts["_row_count"] > 1].reset_index(drop=True)
    if conflicts.empty:
        return

    formatted_conflicts: list[str] = []
    for _, row in conflicts.head(5).iterrows():
        row_key = {column: normalize_scalar(row[column]) for column in spec.rows}
        col_key = {column: normalize_scalar(row[column]) for column in spec.cols}
        formatted_conflicts.append(
            f"rows={int(row['_row_count'])}, row_key={row_key}, col_key={col_key}"
        )

    examples = " ; ".join(formatted_conflicts)
    raise AnalysisError(
        "Found multiple source rows for at least one pivot cell while "
        "'require_unique_cells' is true. "
        f"Conflicting cells: {examples}. "
        "Add filters, add more row/col dimensions, or intentionally aggregate with "
        "'require_unique_cells: false'."
    )


def flatten_column_label(label: Any) -> str:
    """Flatten one pandas column label into a readable single string."""
    if isinstance(label, tuple):
        parts = [str(item) for item in label if str(item) != ""]
        if not parts:
            return "column"
        return COLUMN_LABEL_SEPARATOR.join(parts)
    return str(label)


def infer_optional_mode(dataframe: pd.DataFrame, spec: ViewSpec) -> Any | None:
    """Infer one stable mode value when it exists in filters or the input table."""
    if "mode" in spec.filters:
        mode_value = spec.filters["mode"]
        if isinstance(mode_value, list):
            return mode_value[0] if len(mode_value) == 1 else None
        return normalize_for_json(mode_value)

    unique_modes = collect_unique_values(dataframe, "mode")
    if len(unique_modes) == 1:
        return normalize_for_json(unique_modes[0])
    return None


def extract_axis_levels(axis: pd.Index, *, fallback_levels: list[str]) -> list[str]:
    """Return the logical level names for one pivot axis."""
    raw_names = list(axis.names) if isinstance(axis, pd.MultiIndex) else [axis.name]
    if not raw_names:
        return list(fallback_levels)

    normalized_levels: list[str] = []
    for index, raw_name in enumerate(raw_names):
        if raw_name is None:
            if index >= len(fallback_levels):
                raise AnalysisError(
                    f"Could not infer a fallback level name for pivot axis position {index}."
                )
            normalized_levels.append(fallback_levels[index])
        else:
            normalized_levels.append(str(raw_name))
    return normalized_levels


def extract_axis_tuples(
    axis: pd.Index,
    *,
    level_count: int,
    axis_label: str,
) -> list[list[Any]]:
    """Convert one pivot axis into ordered JSON-safe tuple payloads."""
    tuples: list[list[Any]] = []
    for raw_key in axis.tolist():
        tuple_key = coerce_axis_key(raw_key, level_count=level_count, axis_label=axis_label)
        tuples.append([normalize_for_json(value) for value in tuple_key])
    return tuples


def coerce_axis_key(raw_key: Any, *, level_count: int, axis_label: str) -> tuple[Any, ...]:
    """Normalize one pandas axis key into a tuple with the expected arity."""
    if isinstance(raw_key, tuple):
        tuple_key = raw_key
    else:
        tuple_key = (raw_key,)

    if len(tuple_key) != level_count:
        raise AnalysisError(
            f"Unexpected pivot {axis_label} key {tuple_key!r}; expected {level_count} levels."
        )
    return tuple_key


def build_context(
    dataframe: pd.DataFrame,
    *,
    hidden_column_summary: HiddenColumnSummary,
    auto_context: bool,
    forced_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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

    unique_metric_names = collect_unique_values(dataframe, METRIC_NAME_COLUMN)
    if unique_metric_names:
        context["metric_names"] = unique_metric_names
        if len(unique_metric_names) == 1:
            context[METRIC_NAME_COLUMN] = unique_metric_names[0]

    unique_metric_scopes = collect_unique_values(dataframe, METRIC_SCOPE_COLUMN)
    if unique_metric_scopes:
        context["metric_scopes"] = unique_metric_scopes
        if len(unique_metric_scopes) == 1:
            context[METRIC_SCOPE_COLUMN] = unique_metric_scopes[0]

    unique_ks = collect_unique_values(dataframe, "k")
    if unique_ks:
        context["ks"] = unique_ks
        if len(unique_ks) == 1:
            context["k"] = unique_ks[0]

    if auto_context:
        for column, value in hidden_column_summary.singleton_values.items():
            context[column] = value

    if forced_context is not None:
        for column, value in forced_context.items():
            context[str(column)] = normalize_for_json(value)

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
        if raw_value is None:
            continue
        if isinstance(raw_value, list):
            normalized[key] = [normalize_scalar(item) for item in raw_value]
        else:
            normalized[key] = normalize_scalar(raw_value)
    return normalized


def build_effective_filters(
    filters: Mapping[str, Any],
    split_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine global spec filters with one bundle's split values."""
    effective_filters = dict(filters)
    for column_name, raw_value in split_values.items():
        effective_filters[column_name] = normalize_scalar(raw_value)
    return normalize_for_json(effective_filters)


def normalize_split_value_mapping(
    split_by: list[str],
    split_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize split assignments while preserving the declared split order."""
    normalized: dict[str, Any] = {}
    for column_name in split_by:
        if column_name in split_values:
            normalized[column_name] = normalize_scalar(split_values[column_name])
    for column_name, raw_value in split_values.items():
        if column_name not in normalized:
            normalized[str(column_name)] = normalize_scalar(raw_value)
    return normalized


def load_input_analysis_metadata(input_csv: Path) -> dict[str, Any]:
    """Load optional sibling manifest and slice metadata for one long-table input."""
    bundle_dir = input_csv.parent
    manifest_path = bundle_dir / "manifest.json"
    slice_manifest_path = bundle_dir / "slice_manifest.json"

    manifest_payload = None
    if manifest_path.is_file():
        manifest_payload = load_json_mapping(manifest_path, label="input manifest")
    slice_payload = None
    if slice_manifest_path.is_file():
        slice_payload = load_json_mapping(slice_manifest_path, label="input slice manifest")
    elif isinstance(manifest_payload, Mapping):
        manifest_slice = manifest_payload.get("slice")
        if isinstance(manifest_slice, Mapping):
            slice_payload = manifest_slice

    normalized_slice = normalize_input_slice_metadata(slice_payload)
    return {
        "source_manifest_path": (
            to_repo_relative(manifest_path) if manifest_path.is_file() else None
        ),
        "source_slice_manifest_path": (
            to_repo_relative(slice_manifest_path) if slice_manifest_path.is_file() else None
        ),
        "slice": normalized_slice,
        "slice_context": build_slice_context(normalized_slice),
    }


def normalize_input_slice_metadata(slice_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize slice metadata from either a run bundle or a comparison bundle."""
    if slice_payload is None:
        return None

    normalized: dict[str, Any] = {}
    for field_name in (
        "slice_policy",
        "fairness_safe",
        "selected_target_count",
        "selected_targets",
        "requested_victims",
        "requested_victims_source",
        "source_run_group_key",
        "source_run_group_keys",
        "source_run_ids",
        "compatibility_mode",
        "debug_only",
        "compatible",
    ):
        if field_name in slice_payload:
            normalized[field_name] = normalize_for_json(slice_payload[field_name])
    return normalized or None


def build_slice_context(slice_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten key slice metadata fields for render-friendly context."""
    if slice_metadata is None:
        return {}

    context: dict[str, Any] = {}
    for field_name in (
        "slice_policy",
        "fairness_safe",
        "selected_target_count",
        "requested_victims",
        "requested_victims_source",
        "source_run_group_key",
        "source_run_group_keys",
        "source_run_ids",
        "compatibility_mode",
        "debug_only",
        "compatible",
    ):
        if field_name in slice_metadata:
            context[field_name] = normalize_for_json(slice_metadata[field_name])
    return context


def normalize_optional_string_list(value: Any, *, label: str) -> list[str]:
    """Normalize an optional list of strings."""
    if value is None:
        return []
    return require_string_list(value, label=label)


def build_split_assignment_sort_key(split_values: Mapping[str, Any]) -> tuple[tuple[int, Any], ...]:
    """Build a deterministic sort key for one split assignment."""
    return tuple(build_scalar_sort_key(split_values[column_name]) for column_name in split_values)


def build_scalar_sort_key(value: Any) -> tuple[int, Any]:
    """Build a deterministic cross-type sort key for scalar split values."""
    normalized_value = normalize_scalar(value)
    if normalized_value is None:
        return (0, "")
    if isinstance(normalized_value, bool):
        return (1, int(normalized_value))
    if isinstance(normalized_value, int):
        return (2, normalized_value)
    if isinstance(normalized_value, float):
        return (3, normalized_value)
    if isinstance(normalized_value, str):
        return (4, normalized_value.casefold())
    return (5, str(normalized_value))


def stringify_split_value(value: Any) -> str:
    """Convert one split value into a readable bundle-name token."""
    normalized_value = normalize_scalar(value)
    if normalized_value is None:
        return "null"
    if isinstance(normalized_value, bool):
        return "true" if normalized_value else "false"
    if isinstance(normalized_value, float):
        return format(normalized_value, "g")
    return str(normalized_value)


def sanitize_component(raw_value: str) -> str:
    """Make one bundle-name component filesystem-safe without losing readability."""
    sanitized = UNSAFE_COMPONENT_PATTERN.sub("_", raw_value.strip())
    sanitized = REPEATED_UNDERSCORE_PATTERN.sub("_", sanitized)
    sanitized = sanitized.strip("._-")
    return sanitized or "item"


def normalize_scalar(value: Any) -> Any:
    """Convert scalar values into JSON-safe Python primitives."""
    if pd.isna(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            normalized = value.item()
            if pd.isna(normalized):
                return None
            return normalized
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


def load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON file and require a top-level mapping."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid JSON: {exc}.") from exc

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


def require_bool(value: Any, *, label: str) -> bool:
    """Require a boolean value."""
    if not isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be a boolean, got {type(value).__name__}.")
    return value


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
