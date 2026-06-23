#!/usr/bin/env python3
"""Build renderable method-rank summary bundles from a comparison long table."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.pipeline.view_table_builder import (
    AnalysisError,
    DERIVED_VIEW_COLUMNS,
    HiddenColumnSummary,
    METRIC_COLUMN,
    apply_filters,
    build_context,
    derive_metric_identity_columns,
    ensure_path_within,
    load_input_analysis_metadata,
    load_yaml_mapping,
    normalize_filters,
    normalize_for_json,
    normalize_scalar,
    require_bool,
    require_mapping,
    require_nonempty_string,
    require_string_list,
    resolve_existing_path,
    resolve_repo_path,
    to_repo_relative,
    utc_now_iso,
    validate_required_columns,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SUPPORTED_ANALYSIS_TYPES = {"method_rank_counts"}
DEFAULT_METHOD_COLUMN = "attack_method"
DEFAULT_VALUE_COLUMN = "value"


@dataclass(frozen=True)
class RankThresholdSpec:
    """One rank threshold to count."""

    rank: int
    label: str


@dataclass(frozen=True)
class MethodFamilySpec:
    """A named group whose rank hit is counted when any member hits."""

    name: str
    methods: list[str]


@dataclass(frozen=True)
class MethodRankCountsSpec:
    """Configuration for a method-rank count table."""

    higher_is_better: bool
    tie_tolerance: float
    rank_unit_columns: list[str]
    row_columns: list[str]
    counted_methods: list[str]
    method_families: list[MethodFamilySpec]
    thresholds: list[RankThresholdSpec]
    require_complete_methods: bool
    column_group_label: str | None
    layout: str


@dataclass(frozen=True)
class RankSummarySpec:
    """Validated rank-summary config content."""

    analysis_type: str
    input_csv: Path
    output_dir: Path
    source_spec_path: Path
    filters: dict[str, Any]
    method_column: str
    value_column: str
    auto_context: bool
    method_rank_counts: MethodRankCountsSpec


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the rank-summary CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build one renderable method-rank summary bundle.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a rank-summary YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the rank-summary CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config, label="rank-summary config")
        spec = parse_rank_summary_spec(
            load_yaml_mapping(config_path, label="rank-summary config"),
            source_spec_path=config_path,
        )
        result = build_rank_summary_bundle(spec)
        print(
            f"Wrote rank summary table with {result['row_count']} rows to "
            f"'{result['table_path']}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_rank_summary_spec(
    payload: Mapping[str, Any],
    *,
    source_spec_path: Path,
) -> RankSummarySpec:
    """Validate and normalize one rank-summary YAML spec."""
    analysis_type = require_nonempty_string(
        payload.get("analysis_type"),
        label="analysis_type",
    ).lower()
    if analysis_type not in SUPPORTED_ANALYSIS_TYPES:
        raise AnalysisError(
            f"Unsupported analysis_type '{analysis_type}'. "
            f"Allowed values: {sorted(SUPPORTED_ANALYSIS_TYPES)}."
        )

    input_csv = resolve_existing_path(
        require_nonempty_string(payload.get("input"), label="input"),
        label="rank-summary input CSV",
    )
    ensure_path_within(input_csv, RESULTS_ROOT, label="rank-summary input CSV")

    output_dir = resolve_repo_path(require_nonempty_string(payload.get("output"), label="output"))
    ensure_path_within(output_dir, RESULTS_ROOT, label="rank-summary output")

    method_rank_counts = parse_method_rank_counts_spec(payload.get("method_rank_counts"))
    return RankSummarySpec(
        analysis_type=analysis_type,
        input_csv=input_csv,
        output_dir=output_dir,
        source_spec_path=source_spec_path,
        filters=normalize_filters(payload.get("filters", {})),
        method_column=require_nonempty_string(
            payload.get("method_column", DEFAULT_METHOD_COLUMN),
            label="method_column",
        ),
        value_column=require_nonempty_string(
            payload.get("value_column", DEFAULT_VALUE_COLUMN),
            label="value_column",
        ),
        auto_context=require_bool(payload.get("auto_context", True), label="auto_context"),
        method_rank_counts=method_rank_counts,
    )


def parse_method_rank_counts_spec(value: Any) -> MethodRankCountsSpec:
    """Normalize the method_rank_counts block."""
    payload = require_mapping(value, label="method_rank_counts")
    thresholds = parse_rank_thresholds(payload.get("thresholds"))
    counted_methods = require_string_list(
        payload.get("counted_methods"),
        label="method_rank_counts.counted_methods",
    )
    method_families = parse_method_families(payload.get("method_families"))
    layout = require_nonempty_string(
        payload.get("layout", "methods_as_columns"),
        label="method_rank_counts.layout",
    ).lower()
    if layout not in {"methods_as_columns", "methods_as_rows", "thresholds_as_rows"}:
        raise AnalysisError(
            "method_rank_counts.layout must be 'methods_as_columns', 'methods_as_rows', "
            "or 'thresholds_as_rows'."
        )
    return MethodRankCountsSpec(
        higher_is_better=require_bool(
            payload.get("higher_is_better", True),
            label="method_rank_counts.higher_is_better",
        ),
        tie_tolerance=require_nonnegative_float(
            payload.get("tie_tolerance", 0.0),
            label="method_rank_counts.tie_tolerance",
        ),
        rank_unit_columns=require_string_list(
            payload.get("rank_unit_columns"),
            label="method_rank_counts.rank_unit_columns",
        ),
        row_columns=require_string_list(
            payload.get("row_columns"),
            label="method_rank_counts.row_columns",
        ),
        counted_methods=counted_methods,
        method_families=method_families,
        thresholds=thresholds,
        require_complete_methods=require_bool(
            payload.get("require_complete_methods", True),
            label="method_rank_counts.require_complete_methods",
        ),
        column_group_label=(
            None
            if payload.get("column_group_label") is None
            else require_nonempty_string(
                payload.get("column_group_label"),
                label="method_rank_counts.column_group_label",
            )
        ),
        layout=layout,
    )


def parse_rank_thresholds(value: Any) -> list[RankThresholdSpec]:
    """Normalize rank threshold entries."""
    if not isinstance(value, list) or not value:
        raise AnalysisError("Expected 'method_rank_counts.thresholds' to be a non-empty list.")

    thresholds: list[RankThresholdSpec] = []
    seen_ranks: set[int] = set()
    for index, item in enumerate(value):
        payload = require_mapping(item, label=f"method_rank_counts.thresholds[{index}]")
        rank = require_positive_int(
            payload.get("rank"),
            label=f"method_rank_counts.thresholds[{index}].rank",
        )
        if rank in seen_ranks:
            raise AnalysisError(f"Duplicate rank threshold {rank} is not allowed.")
        seen_ranks.add(rank)
        thresholds.append(
            RankThresholdSpec(
                rank=rank,
                label=require_nonempty_string(
                    payload.get("label"),
                    label=f"method_rank_counts.thresholds[{index}].label",
                ),
            )
        )
    return thresholds


def parse_method_families(value: Any) -> list[MethodFamilySpec]:
    """Normalize optional method-family entries."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisError("Expected 'method_rank_counts.method_families' to be a list.")

    families: list[MethodFamilySpec] = []
    seen_names: set[str] = set()
    for index, item in enumerate(value):
        payload = require_mapping(item, label=f"method_rank_counts.method_families[{index}]")
        name = require_nonempty_string(
            payload.get("name"),
            label=f"method_rank_counts.method_families[{index}].name",
        )
        if name in seen_names:
            raise AnalysisError(f"Duplicate method family '{name}' is not allowed.")
        seen_names.add(name)
        families.append(
            MethodFamilySpec(
                name=name,
                methods=require_string_list(
                    payload.get("methods"),
                    label=f"method_rank_counts.method_families[{index}].methods",
                ),
            )
        )
    return families


def build_rank_summary_bundle(spec: RankSummarySpec) -> dict[str, Any]:
    """Build one rank-summary render bundle from a validated spec."""
    dataframe = prepare_rank_dataframe(pd.read_csv(spec.input_csv), spec)
    required_columns = sorted(
        set(spec.filters.keys())
        | set(spec.method_rank_counts.rank_unit_columns)
        | set(spec.method_rank_counts.row_columns)
        | {spec.method_column, spec.value_column}
    )
    validate_required_columns(dataframe, required_columns=required_columns, label="rank-summary input CSV")

    filtered_dataframe = apply_filters(dataframe, spec.filters)
    if filtered_dataframe.empty:
        raise AnalysisError(f"The filters produced an empty table from '{spec.input_csv}'.")
    validate_numeric_value_column(filtered_dataframe, value_column=spec.value_column)
    validate_row_columns_within_rank_units(spec)

    aggregated = aggregate_method_values(filtered_dataframe, spec)
    validate_complete_rank_units(aggregated, spec)
    ranked = build_ranked_method_hits(aggregated, spec)
    report_dataframe, column_tuples = build_report_table(ranked, spec)
    if report_dataframe.empty:
        raise AnalysisError("The rank-summary table is empty after aggregation.")

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = spec.output_dir / "table.csv"
    meta_path = spec.output_dir / "meta.json"
    report_dataframe.to_csv(table_path, index=False)
    write_json(
        meta_path,
        build_bundle_metadata(
            spec=spec,
            dataframe=filtered_dataframe,
            report_dataframe=report_dataframe,
            column_tuples=column_tuples,
        ),
    )
    return {
        "table_path": table_path,
        "meta_path": meta_path,
        "row_count": int(len(report_dataframe)),
    }


def prepare_rank_dataframe(dataframe: pd.DataFrame, spec: RankSummarySpec) -> pd.DataFrame:
    """Attach derived metric columns needed by the rank-summary config."""
    requested_columns = set(spec.filters.keys())
    requested_columns.update(spec.method_rank_counts.rank_unit_columns)
    requested_columns.update(spec.method_rank_counts.row_columns)
    missing_derived_columns = sorted(
        column_name
        for column_name in requested_columns
        if column_name in DERIVED_VIEW_COLUMNS and column_name not in dataframe.columns
    )
    if not missing_derived_columns:
        return dataframe.copy()

    if METRIC_COLUMN not in dataframe.columns:
        raise AnalysisError(
            "The rank-summary spec references derived metric columns "
            f"{missing_derived_columns}, but the input CSV does not contain '{METRIC_COLUMN}'."
        )

    prepared = dataframe.copy()
    derived_metric_columns = derive_metric_identity_columns(prepared[METRIC_COLUMN])
    for column_name in missing_derived_columns:
        prepared[column_name] = derived_metric_columns[column_name]
    return prepared


def aggregate_method_values(dataframe: pd.DataFrame, spec: RankSummarySpec) -> pd.DataFrame:
    """Aggregate repeated method rows before ranking."""
    group_columns = spec.method_rank_counts.rank_unit_columns + [spec.method_column]
    aggregated = (
        dataframe.assign(**{spec.value_column: pd.to_numeric(dataframe[spec.value_column])})
        .groupby(group_columns, dropna=False, sort=True)[spec.value_column]
        .mean()
        .reset_index()
    )
    return aggregated


def build_ranked_method_hits(dataframe: pd.DataFrame, spec: RankSummarySpec) -> pd.DataFrame:
    """Rank all methods inside each rank unit and emit one hit row per unit."""
    thresholds = spec.method_rank_counts.thresholds
    subjects = build_count_subjects(spec)
    output_rows: list[dict[str, Any]] = []

    for rank_values, rank_unit_frame in dataframe.groupby(
        spec.method_rank_counts.rank_unit_columns,
        dropna=False,
        sort=True,
    ):
        rank_key = normalize_rank_key(rank_values, spec.method_rank_counts.rank_unit_columns)
        method_values = {
            str(row[spec.method_column]): float(row[spec.value_column])
            for _, row in rank_unit_frame.iterrows()
        }
        method_ranks = rank_methods(
            method_values,
            higher_is_better=spec.method_rank_counts.higher_is_better,
            tie_tolerance=spec.method_rank_counts.tie_tolerance,
        )
        output_row = dict(rank_key)
        for subject_name, subject_methods in subjects.items():
            for threshold in thresholds:
                output_row[
                    build_value_column_name(
                        subject_name,
                        threshold.label,
                        column_group_label=spec.method_rank_counts.column_group_label,
                    )
                ] = int(
                    any(method_ranks.get(method_name, float("inf")) <= threshold.rank for method_name in subject_methods)
                )
        output_rows.append(output_row)

    return pd.DataFrame(output_rows)


def build_report_table(
    ranked_dataframe: pd.DataFrame,
    spec: RankSummarySpec,
) -> tuple[pd.DataFrame, list[list[str]]]:
    """Summarize rank-hit rows into a renderable table."""
    if spec.method_rank_counts.layout == "methods_as_rows":
        return build_methods_as_rows_report_table(ranked_dataframe, spec)
    if spec.method_rank_counts.layout == "thresholds_as_rows":
        return build_thresholds_as_rows_report_table(ranked_dataframe, spec)

    subjects = build_count_subjects(spec)
    value_columns: list[str] = []
    column_tuples: list[list[str]] = []
    for subject_name in subjects:
        for threshold in spec.method_rank_counts.thresholds:
            value_columns.append(
                build_value_column_name(
                    subject_name,
                    threshold.label,
                    column_group_label=spec.method_rank_counts.column_group_label,
                )
            )
            column_tuple = [subject_name, threshold.label]
            if spec.method_rank_counts.column_group_label is not None:
                column_tuple.insert(0, spec.method_rank_counts.column_group_label)
            column_tuples.append(column_tuple)

    grouped = (
        ranked_dataframe.groupby(spec.method_rank_counts.row_columns, dropna=False, sort=True)[
            value_columns
        ]
        .sum()
        .reset_index()
    )
    return grouped.loc[:, spec.method_rank_counts.row_columns + value_columns], column_tuples


def build_methods_as_rows_report_table(
    ranked_dataframe: pd.DataFrame,
    spec: RankSummarySpec,
) -> tuple[pd.DataFrame, list[list[Any]]]:
    """Transpose rank counts so methods are rows and summary dimensions are columns."""
    subjects = build_count_subjects(spec)
    source_value_columns = [
        build_value_column_name(subject_name, threshold.label)
        for subject_name in subjects
        for threshold in spec.method_rank_counts.thresholds
    ]
    grouped = (
        ranked_dataframe.groupby(spec.method_rank_counts.row_columns, dropna=False, sort=True)[
            source_value_columns
        ]
        .sum()
        .reset_index()
    )

    column_tuples: list[list[Any]] = []
    for _, grouped_row in grouped.iterrows():
        row_prefix = [normalize_scalar(grouped_row[column]) for column in spec.method_rank_counts.row_columns]
        for threshold in spec.method_rank_counts.thresholds:
            column_tuples.append(row_prefix + [threshold.label])

    output_rows: list[dict[str, Any]] = []
    for subject_name in subjects:
        output_row: dict[str, Any] = {"rank_subject": subject_name}
        for _, grouped_row in grouped.iterrows():
            row_prefix = [normalize_scalar(grouped_row[column]) for column in spec.method_rank_counts.row_columns]
            for threshold in spec.method_rank_counts.thresholds:
                output_column = " | ".join(str(value) for value in row_prefix + [threshold.label])
                source_column = build_value_column_name(subject_name, threshold.label)
                output_row[output_column] = int(grouped_row[source_column])
        output_rows.append(output_row)
    return pd.DataFrame(output_rows), column_tuples


def build_thresholds_as_rows_report_table(
    ranked_dataframe: pd.DataFrame,
    spec: RankSummarySpec,
) -> tuple[pd.DataFrame, list[list[Any]]]:
    """Summarize rank counts with thresholds on rows and methods on columns."""
    subjects = build_count_subjects(spec)
    source_value_columns = [
        build_value_column_name(
            subject_name,
            threshold.label,
            column_group_label=spec.method_rank_counts.column_group_label,
        )
        for subject_name in subjects
        for threshold in spec.method_rank_counts.thresholds
    ]
    grouped = (
        ranked_dataframe.groupby(spec.method_rank_counts.row_columns, dropna=False, sort=True)[
            source_value_columns
        ]
        .sum()
        .reset_index()
    )

    column_tuples: list[list[Any]] = []
    for subject_name in subjects:
        column_tuple: list[Any] = [subject_name]
        if spec.method_rank_counts.column_group_label is not None:
            column_tuple.insert(0, spec.method_rank_counts.column_group_label)
        column_tuples.append(column_tuple)

    output_rows: list[dict[str, Any]] = []
    for _, grouped_row in grouped.iterrows():
        row_prefix = {
            column: normalize_scalar(grouped_row[column])
            for column in spec.method_rank_counts.row_columns
        }
        for threshold in spec.method_rank_counts.thresholds:
            output_row: dict[str, Any] = dict(row_prefix)
            output_row["rank_threshold"] = threshold.label
            for subject_name in subjects:
                source_column = build_value_column_name(
                    subject_name,
                    threshold.label,
                    column_group_label=spec.method_rank_counts.column_group_label,
                )
                output_row[subject_name] = int(grouped_row[source_column])
            output_rows.append(output_row)
    return pd.DataFrame(output_rows), column_tuples


def build_bundle_metadata(
    *,
    spec: RankSummarySpec,
    dataframe: pd.DataFrame,
    report_dataframe: pd.DataFrame,
    column_tuples: list[list[str]],
) -> dict[str, Any]:
    """Build metadata compatible with report_table_renderer."""
    input_analysis_metadata = load_input_analysis_metadata(spec.input_csv)
    slice_metadata = input_analysis_metadata.get("slice")
    if not isinstance(slice_metadata, Mapping):
        slice_metadata = None
    slice_context = input_analysis_metadata.get("slice_context")
    if not isinstance(slice_context, Mapping):
        slice_context = {}

    layout = spec.method_rank_counts.layout
    methods_as_rows = layout == "methods_as_rows"
    thresholds_as_rows = layout == "thresholds_as_rows"
    if methods_as_rows:
        row_levels = ["rank_subject"]
        col_levels = spec.method_rank_counts.row_columns + ["rank_threshold"]
    elif thresholds_as_rows:
        row_levels = spec.method_rank_counts.row_columns + ["rank_threshold"]
        col_levels = (
            ["rank_group", "rank_subject"]
            if spec.method_rank_counts.column_group_label is not None
            else ["rank_subject"]
        )
    else:
        row_levels = spec.method_rank_counts.row_columns
        col_levels = (
            ["rank_group", "rank_subject", "rank_threshold"]
            if spec.method_rank_counts.column_group_label is not None
            else ["rank_subject", "rank_threshold"]
        )
    row_tuples = [
        [normalize_scalar(row[column_name]) for column_name in row_levels]
        for _, row in report_dataframe.loc[:, row_levels].iterrows()
    ]
    return {
        "mode": "method_rank_counts",
        "input_csv": to_repo_relative(spec.input_csv),
        "source_rank_summary_spec_path": to_repo_relative(spec.source_spec_path),
        "output_bundle_dir": to_repo_relative(spec.output_dir),
        "bundle_output_dir": to_repo_relative(spec.output_dir),
        "source_manifest_path": input_analysis_metadata.get("source_manifest_path"),
        "source_slice_manifest_path": input_analysis_metadata.get("source_slice_manifest_path"),
        "bundle_name": spec.output_dir.name,
        "analysis_type": spec.analysis_type,
        "filters": normalize_for_json(spec.filters),
        "rows": row_levels,
        "cols": col_levels,
        "row_levels": row_levels,
        "col_levels": col_levels,
        "row_tuples": normalize_for_json(row_tuples),
        "column_tuples": column_tuples,
        "value_col": "count",
        "agg": "sum",
        "method_column": spec.method_column,
        "rank_unit_columns": spec.method_rank_counts.rank_unit_columns,
        "counted_methods": spec.method_rank_counts.counted_methods,
        "method_families": [
            {"name": family.name, "methods": family.methods}
            for family in spec.method_rank_counts.method_families
        ],
        "thresholds": [
            {"rank": threshold.rank, "label": threshold.label}
            for threshold in spec.method_rank_counts.thresholds
        ],
        "require_complete_methods": spec.method_rank_counts.require_complete_methods,
        "column_group_label": spec.method_rank_counts.column_group_label,
        "layout": spec.method_rank_counts.layout,
        "filtered_row_count": int(len(dataframe)),
        "output_row_count": int(len(report_dataframe)),
        "output_column_count": int(len(report_dataframe.columns)),
        "generation_timestamp": utc_now_iso(),
        "slice": normalize_for_json(slice_metadata) if slice_metadata is not None else None,
        "slice_context": normalize_for_json(slice_context),
        "context": build_context(
            dataframe,
            hidden_column_summary=HiddenColumnSummary(singleton_values={}, varying_columns=[]),
            auto_context=spec.auto_context,
            forced_context=normalize_for_json(slice_context),
        ),
    }


def build_count_subjects(spec: RankSummarySpec) -> dict[str, list[str]]:
    """Return output count subjects in display order."""
    subjects = {method_name: [method_name] for method_name in spec.method_rank_counts.counted_methods}
    for family in spec.method_rank_counts.method_families:
        subjects[family.name] = family.methods
    return subjects


def rank_methods(
    method_values: Mapping[str, float],
    *,
    higher_is_better: bool,
    tie_tolerance: float,
) -> dict[str, int]:
    """Rank methods with competition ranks; ties share the best applicable rank."""
    ranks: dict[str, int] = {}
    for method_name, method_value in method_values.items():
        better_count = 0
        for other_value in method_values.values():
            delta = other_value - method_value if higher_is_better else method_value - other_value
            if delta > tie_tolerance:
                better_count += 1
        ranks[method_name] = better_count + 1
    return ranks


def normalize_rank_key(raw_values: Any, columns: list[str]) -> dict[str, Any]:
    """Convert pandas groupby keys into a column mapping."""
    if len(columns) == 1:
        values = (raw_values,)
    else:
        values = tuple(raw_values)
    return {
        column_name: normalize_scalar(value)
        for column_name, value in zip(columns, values, strict=True)
    }


def build_value_column_name(
    subject_name: str,
    threshold_label: str,
    *,
    column_group_label: str | None = None,
) -> str:
    """Match report_table_renderer's flattened multi-level column format."""
    parts = [subject_name, threshold_label]
    if column_group_label is not None:
        parts.insert(0, column_group_label)
    return " | ".join(parts)


def validate_row_columns_within_rank_units(spec: RankSummarySpec) -> None:
    """Require row summary dimensions to be preserved through rank-unit grouping."""
    missing = [
        column_name
        for column_name in spec.method_rank_counts.row_columns
        if column_name not in spec.method_rank_counts.rank_unit_columns
    ]
    if missing:
        raise AnalysisError(
            "method_rank_counts.row_columns must be included in "
            f"method_rank_counts.rank_unit_columns. Missing: {missing}."
        )


def validate_numeric_value_column(dataframe: pd.DataFrame, *, value_column: str) -> None:
    """Require a numeric value column."""
    try:
        pd.to_numeric(dataframe[value_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas raises multiple exception types
        raise AnalysisError(f"The value column '{value_column}' must be numeric.") from exc


def validate_complete_rank_units(dataframe: pd.DataFrame, spec: RankSummarySpec) -> None:
    """Require every rank unit to contain the expected method set."""
    if not spec.method_rank_counts.require_complete_methods:
        return

    expected_methods = resolve_expected_methods(dataframe, spec)
    if not expected_methods:
        raise AnalysisError("Could not resolve any expected methods for rank-unit completeness.")

    missing_examples: list[str] = []
    for rank_values, rank_unit_frame in dataframe.groupby(
        spec.method_rank_counts.rank_unit_columns,
        dropna=False,
        sort=True,
    ):
        observed_methods = {
            str(normalize_scalar(value)) for value in rank_unit_frame[spec.method_column].tolist()
        }
        missing_methods = [method_name for method_name in expected_methods if method_name not in observed_methods]
        if missing_methods:
            rank_key = normalize_rank_key(rank_values, spec.method_rank_counts.rank_unit_columns)
            missing_examples.append(f"rank_unit={rank_key}, missing_methods={missing_methods}")
        if len(missing_examples) >= 5:
            break

    if missing_examples:
        raise AnalysisError(
            "method_rank_counts.require_complete_methods is enabled, but at least one rank unit "
            "is missing expected methods. "
            f"Examples: {' ; '.join(missing_examples)}."
        )


def resolve_expected_methods(dataframe: pd.DataFrame, spec: RankSummarySpec) -> list[str]:
    """Resolve the method set each rank unit must contain."""
    configured_methods = spec.filters.get(spec.method_column)
    if isinstance(configured_methods, list):
        return [str(normalize_scalar(value)) for value in configured_methods]
    if configured_methods is not None:
        return [str(normalize_scalar(configured_methods))]
    return sorted(
        {str(normalize_scalar(value)) for value in dataframe[spec.method_column].dropna().tolist()}
    )


def require_positive_int(value: Any, *, label: str) -> int:
    """Require a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"Expected '{label}' to be a positive integer.")
    if value <= 0:
        raise AnalysisError(f"Expected '{label}' to be greater than zero, got {value}.")
    return int(value)


def require_nonnegative_float(value: Any, *, label: str) -> float:
    """Require a non-negative numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"Expected '{label}' to be a non-negative number.")
    numeric_value = float(value)
    if numeric_value < 0.0:
        raise AnalysisError(f"Expected '{label}' to be non-negative, got {numeric_value}.")
    return numeric_value


if __name__ == "__main__":
    main()
