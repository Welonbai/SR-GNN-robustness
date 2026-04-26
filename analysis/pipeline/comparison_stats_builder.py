#!/usr/bin/env python3
"""Build comparison-level stats bundles from a merged long-table CSV."""

from __future__ import annotations

import argparse
import itertools
import math
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
    METRIC_COLUMN,
    apply_filters,
    derive_metric_identity_columns,
    ensure_path_within,
    load_input_analysis_metadata,
    load_yaml_mapping,
    normalize_filters,
    normalize_for_json,
    normalize_scalar,
    require_bool,
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
SUPPORTED_ANALYSIS_TYPES = {"pairwise_method_win_rate"}
DEFAULT_METHOD_COLUMN = "attack_method"
DEFAULT_VALUE_COLUMN = "value"
DEFAULT_ALWAYS_IGNORED_COLUMNS = {"run_id"}


@dataclass(frozen=True)
class RequestedMethodPair:
    """One explicitly requested method-vs-method comparison."""

    left_method: str
    right_method: str


@dataclass(frozen=True)
class PairingSpec:
    """Control how rows are paired across methods before comparison."""

    ignore_columns: list[str]
    auto_ignore_method_descriptor_columns: bool


@dataclass(frozen=True)
class PairwiseMethodWinRateSpec:
    """Control how wins and ties are determined."""

    higher_is_better: bool
    tie_tolerance: float
    requested_pairs: list[RequestedMethodPair]
    write_full_matrices: bool


@dataclass(frozen=True)
class ComparisonStatsSpec:
    """Validated stats-builder config content."""

    analysis_type: str
    input_csv: Path
    output_dir: Path
    source_spec_path: Path
    filters: dict[str, Any]
    method_column: str
    value_column: str
    pairing: PairingSpec
    pairwise_method_win_rate: PairwiseMethodWinRateSpec


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the comparison-stats CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build one comparison-stats bundle from a merged comparison long-table CSV."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a comparison-stats YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the comparison-stats CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config, label="comparison-stats config")
        spec = parse_comparison_stats_spec(
            load_yaml_mapping(config_path, label="comparison-stats config"),
            source_spec_path=config_path,
        )
        result = build_comparison_stats_bundle(spec)
        print(
            f"Wrote pairwise stats for {result['method_count']} methods and "
            f"{result['pair_count']} method pairs to '{result['summary_path']}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_comparison_stats_spec(
    payload: Mapping[str, Any],
    *,
    source_spec_path: Path,
) -> ComparisonStatsSpec:
    """Validate and normalize one comparison-stats YAML spec."""
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
        label="comparison-stats input CSV",
    )
    ensure_path_within(input_csv, RESULTS_ROOT, label="comparison-stats input CSV")

    output_dir = resolve_repo_path(
        require_nonempty_string(payload.get("output"), label="output")
    )
    ensure_path_within(output_dir, RESULTS_ROOT, label="comparison-stats output")

    filters = normalize_filters(payload.get("filters", {}))
    method_column = require_nonempty_string(
        payload.get("method_column", DEFAULT_METHOD_COLUMN),
        label="method_column",
    )
    value_column = require_nonempty_string(
        payload.get("value_column", DEFAULT_VALUE_COLUMN),
        label="value_column",
    )
    pairing = parse_pairing_spec(payload.get("pairing"))
    pairwise_method_win_rate = parse_pairwise_method_win_rate_spec(
        payload.get("pairwise_method_win_rate")
    )

    return ComparisonStatsSpec(
        analysis_type=analysis_type,
        input_csv=input_csv,
        output_dir=output_dir,
        source_spec_path=source_spec_path,
        filters=filters,
        method_column=method_column,
        value_column=value_column,
        pairing=pairing,
        pairwise_method_win_rate=pairwise_method_win_rate,
    )


def parse_pairing_spec(value: Any) -> PairingSpec:
    """Normalize one optional pairing block."""
    if value is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise AnalysisError(
            f"Expected 'pairing' to be a mapping, got {type(value).__name__}."
        )

    ignore_columns = normalize_optional_string_list(
        payload.get("ignore_columns"),
        label="pairing.ignore_columns",
    )
    auto_ignore_method_descriptor_columns = require_bool(
        payload.get("auto_ignore_method_descriptor_columns", True),
        label="pairing.auto_ignore_method_descriptor_columns",
    )
    return PairingSpec(
        ignore_columns=ignore_columns,
        auto_ignore_method_descriptor_columns=auto_ignore_method_descriptor_columns,
    )


def parse_pairwise_method_win_rate_spec(value: Any) -> PairwiseMethodWinRateSpec:
    """Normalize one optional pairwise-win-rate block."""
    if value is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(value, Mapping):
        payload = value
    else:
        raise AnalysisError(
            "Expected 'pairwise_method_win_rate' to be a mapping, got "
            f"{type(value).__name__}."
        )

    higher_is_better = require_bool(
        payload.get("higher_is_better", True),
        label="pairwise_method_win_rate.higher_is_better",
    )
    tie_tolerance = require_nonnegative_float(
        payload.get("tie_tolerance", 0.0),
        label="pairwise_method_win_rate.tie_tolerance",
    )
    requested_pairs = parse_requested_method_pairs(
        payload.get("requested_pairs"),
        label="pairwise_method_win_rate.requested_pairs",
    )
    write_full_matrices = require_bool(
        payload.get("write_full_matrices", not requested_pairs),
        label="pairwise_method_win_rate.write_full_matrices",
    )
    return PairwiseMethodWinRateSpec(
        higher_is_better=higher_is_better,
        tie_tolerance=tie_tolerance,
        requested_pairs=requested_pairs,
        write_full_matrices=write_full_matrices,
    )


def build_comparison_stats_bundle(spec: ComparisonStatsSpec) -> dict[str, Any]:
    """Build one comparison-stats bundle from a validated spec."""
    dataframe = prepare_stats_dataframe(pd.read_csv(spec.input_csv), spec)
    validate_required_columns(
        dataframe,
        required_columns=sorted(
            set(spec.filters.keys()) | {spec.method_column, spec.value_column}
        ),
        label="comparison-stats input CSV",
    )

    filtered_dataframe = apply_filters(dataframe, spec.filters)
    if filtered_dataframe.empty:
        raise AnalysisError(
            f"The filters produced an empty table from '{spec.input_csv}'."
        )

    input_analysis_metadata = load_input_analysis_metadata(spec.input_csv)
    summary = build_pairwise_method_win_rate_summary(
        filtered_dataframe,
        spec=spec,
        input_analysis_metadata=input_analysis_metadata,
    )

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = spec.output_dir / "summary.json"
    manifest_path = spec.output_dir / "manifest.json"
    win_rate_matrix_path = spec.output_dir / "pairwise_win_rate_matrix.csv"
    comparable_count_matrix_path = spec.output_dir / "pairwise_comparable_count_matrix.csv"
    pairwise_results_csv_path = spec.output_dir / "pairwise_results.csv"

    write_json(summary_path, summary)
    pd.DataFrame(summary["pairwise_results"]).to_csv(pairwise_results_csv_path, index=False)
    generated_files = [
        "manifest.json",
        "pairwise_results.csv",
        "summary.json",
    ]
    if spec.pairwise_method_win_rate.write_full_matrices:
        build_pairwise_metric_matrix(
            methods=summary["methods"],
            pairwise_results=summary["pairwise_results"],
            left_value_key="left_win_rate",
            right_value_key="right_win_rate",
        ).to_csv(win_rate_matrix_path, index=True)
        build_pairwise_metric_matrix(
            methods=summary["methods"],
            pairwise_results=summary["pairwise_results"],
            left_value_key="comparable_result_count",
            right_value_key="comparable_result_count",
        ).to_csv(comparable_count_matrix_path, index=True)
        generated_files.insert(1, "pairwise_comparable_count_matrix.csv")
        generated_files.insert(2, "pairwise_win_rate_matrix.csv")

    manifest = {
        "analysis_type": spec.analysis_type,
        "input_csv": to_repo_relative(spec.input_csv),
        "output_dir": to_repo_relative(spec.output_dir),
        "source_spec_path": to_repo_relative(spec.source_spec_path),
        "generated_files": generated_files,
        "filters": normalize_for_json(spec.filters),
        "method_column": spec.method_column,
        "value_column": spec.value_column,
        "pairwise_method_win_rate": {
            "requested_pairs": [
                {
                    "left_method": pair.left_method,
                    "right_method": pair.right_method,
                }
                for pair in spec.pairwise_method_win_rate.requested_pairs
            ],
            "write_full_matrices": spec.pairwise_method_win_rate.write_full_matrices,
        },
        "source_manifest_path": input_analysis_metadata.get("source_manifest_path"),
        "source_slice_manifest_path": input_analysis_metadata.get(
            "source_slice_manifest_path"
        ),
        "slice": input_analysis_metadata.get("slice"),
        "generation_timestamp": utc_now_iso(),
    }
    write_json(manifest_path, manifest)

    return {
        "summary_path": summary_path,
        "manifest_path": manifest_path,
        "method_count": len(summary["methods"]),
        "pair_count": len(summary["pairwise_results"]),
    }


def prepare_stats_dataframe(
    dataframe: pd.DataFrame,
    spec: ComparisonStatsSpec,
) -> pd.DataFrame:
    """Attach derived metric columns needed by the stats config."""
    requested_columns = set(spec.filters.keys())
    requested_columns.add(spec.method_column)
    requested_columns.add(spec.value_column)
    missing_derived_columns = sorted(
        column_name
        for column_name in requested_columns
        if column_name in DERIVED_VIEW_COLUMNS and column_name not in dataframe.columns
    )
    if not missing_derived_columns:
        return dataframe.copy()

    if METRIC_COLUMN not in dataframe.columns:
        raise AnalysisError(
            "The stats spec references derived metric columns "
            f"{missing_derived_columns}, but the input CSV does not contain '{METRIC_COLUMN}'."
        )

    prepared = dataframe.copy()
    derived_metric_columns = derive_metric_identity_columns(prepared[METRIC_COLUMN])
    for column_name in missing_derived_columns:
        prepared[column_name] = derived_metric_columns[column_name]
    return prepared


def build_pairwise_method_win_rate_summary(
    dataframe: pd.DataFrame,
    *,
    spec: ComparisonStatsSpec,
    input_analysis_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute pairwise method win rates across all comparable result cells."""
    validate_numeric_value_column(
        dataframe,
        value_column=spec.value_column,
    )

    available_methods = collect_sorted_unique_values(dataframe[spec.method_column])
    if len(available_methods) < 2:
        raise AnalysisError(
            f"Expected at least two methods in column '{spec.method_column}', got {available_methods}."
        )

    explicit_ignore_columns = unique_preserve_order(
        spec.pairing.ignore_columns + sorted(DEFAULT_ALWAYS_IGNORED_COLUMNS)
    )
    auto_ignore_columns = (
        resolve_auto_ignore_method_descriptor_columns(
            dataframe,
            method_column=spec.method_column,
            value_column=spec.value_column,
            explicit_ignore_columns=explicit_ignore_columns,
        )
        if spec.pairing.auto_ignore_method_descriptor_columns
        else []
    )
    ignored_columns = unique_preserve_order(explicit_ignore_columns + auto_ignore_columns)
    comparison_unit_columns = [
        column_name
        for column_name in dataframe.columns
        if column_name not in ignored_columns
        and column_name not in {spec.method_column, spec.value_column}
    ]
    validate_unique_method_rows(
        dataframe,
        method_column=spec.method_column,
        comparison_unit_columns=comparison_unit_columns,
    )

    wide_frame = build_pairwise_wide_frame(
        dataframe,
        method_column=spec.method_column,
        value_column=spec.value_column,
        comparison_unit_columns=comparison_unit_columns,
    )
    requested_pairs = resolve_requested_pairs(
        spec.pairwise_method_win_rate.requested_pairs,
        available_methods=available_methods,
    )
    pairwise_results = build_pairwise_results(
        wide_frame,
        methods=available_methods,
        win_rate_spec=spec.pairwise_method_win_rate,
        requested_pairs=requested_pairs,
    )
    pairwise_lookup = build_pairwise_lookup(pairwise_results)
    output_methods = (
        collect_requested_pair_methods(requested_pairs)
        if requested_pairs
        else list(available_methods)
    )

    return {
        "analysis_type": spec.analysis_type,
        "input_csv": to_repo_relative(spec.input_csv),
        "output_dir": to_repo_relative(spec.output_dir),
        "source_spec_path": to_repo_relative(spec.source_spec_path),
        "row_count_after_filters": int(len(dataframe)),
        "comparison_unit_count": int(len(wide_frame)),
        "filters": normalize_for_json(spec.filters),
        "method_column": spec.method_column,
        "value_column": spec.value_column,
        "methods": output_methods,
        "available_methods": available_methods,
        "pairing": {
            "explicit_ignore_columns": explicit_ignore_columns,
            "auto_ignore_method_descriptor_columns": auto_ignore_columns,
            "ignored_columns": ignored_columns,
            "comparison_unit_columns": comparison_unit_columns,
        },
        "pairwise_method_win_rate": {
            "higher_is_better": spec.pairwise_method_win_rate.higher_is_better,
            "tie_tolerance": spec.pairwise_method_win_rate.tie_tolerance,
            "requested_pairs": [
                {
                    "left_method": pair.left_method,
                    "right_method": pair.right_method,
                }
                for pair in requested_pairs
            ],
            "write_full_matrices": spec.pairwise_method_win_rate.write_full_matrices,
        },
        "pair_selection": {
            "mode": "selected_pairs" if requested_pairs else "all_pairs",
            "requested_pairs": [
                {
                    "left_method": pair.left_method,
                    "right_method": pair.right_method,
                }
                for pair in requested_pairs
            ],
            "method_count": len(output_methods),
            "methods": output_methods,
            "write_full_matrices": spec.pairwise_method_win_rate.write_full_matrices,
        },
        "pairwise_results": pairwise_results,
        "pairwise_lookup": pairwise_lookup,
        "source_manifest_path": input_analysis_metadata.get("source_manifest_path"),
        "source_slice_manifest_path": input_analysis_metadata.get(
            "source_slice_manifest_path"
        ),
        "slice": input_analysis_metadata.get("slice"),
        "slice_context": input_analysis_metadata.get("slice_context"),
        "generation_timestamp": utc_now_iso(),
    }


def validate_numeric_value_column(dataframe: pd.DataFrame, *, value_column: str) -> None:
    """Require the value column to be numeric for pairwise comparisons."""
    try:
        pd.to_numeric(dataframe[value_column], errors="raise")
    except Exception as exc:  # pragma: no cover - pandas raises multiple exception types
        raise AnalysisError(
            f"The value column '{value_column}' must be numeric for pairwise win rates."
        ) from exc


def collect_sorted_unique_values(series: pd.Series) -> list[Any]:
    """Collect deterministic unique scalar values from one series."""
    values = [normalize_scalar(value) for value in series.dropna().unique().tolist()]
    if not values:
        return []
    return sorted(values, key=build_scalar_sort_key)


def build_scalar_sort_key(value: Any) -> tuple[int, Any]:
    """Build a deterministic cross-type sort key."""
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


def resolve_auto_ignore_method_descriptor_columns(
    dataframe: pd.DataFrame,
    *,
    method_column: str,
    value_column: str,
    explicit_ignore_columns: list[str],
) -> list[str]:
    """Auto-ignore columns whose value is fully determined by the method identity."""
    explicit_ignore_set = set(explicit_ignore_columns)
    auto_ignored: list[str] = []
    for column_name in dataframe.columns:
        if column_name in explicit_ignore_set or column_name in {method_column, value_column}:
            continue
        overall_unique_count = dataframe[column_name].nunique(dropna=False)
        if overall_unique_count <= 1:
            continue
        per_method_unique_counts = dataframe.groupby(
            method_column,
            dropna=False,
            sort=True,
        )[column_name].nunique(dropna=False)
        if bool((per_method_unique_counts <= 1).all()):
            auto_ignored.append(column_name)
    return auto_ignored


def validate_unique_method_rows(
    dataframe: pd.DataFrame,
    *,
    method_column: str,
    comparison_unit_columns: list[str],
) -> None:
    """Require one row per method per comparison unit."""
    grouping_columns = [method_column] + comparison_unit_columns
    duplicate_rows = (
        dataframe.groupby(grouping_columns, dropna=False, sort=True)
        .size()
        .reset_index(name="_row_count")
    )
    duplicate_rows = duplicate_rows[duplicate_rows["_row_count"] > 1].reset_index(drop=True)
    if duplicate_rows.empty:
        return

    formatted_duplicates: list[str] = []
    for _, row in duplicate_rows.head(5).iterrows():
        key_payload = {
            column_name: normalize_scalar(row[column_name])
            for column_name in grouping_columns
        }
        formatted_duplicates.append(
            f"rows={int(row['_row_count'])}, comparison_key={key_payload}"
        )
    examples = " ; ".join(formatted_duplicates)
    raise AnalysisError(
        "Pairwise method win rates require exactly one row per method and comparison unit. "
        f"Conflicts: {examples}. "
        "Tighten the filters or adjust the pairing.ignore_columns settings."
    )


def build_pairwise_wide_frame(
    dataframe: pd.DataFrame,
    *,
    method_column: str,
    value_column: str,
    comparison_unit_columns: list[str],
) -> pd.DataFrame:
    """Pivot one filtered dataframe into method columns over comparable units."""
    normalized = dataframe.copy()
    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="raise")

    if comparison_unit_columns:
        wide_frame = normalized.pivot(
            index=comparison_unit_columns,
            columns=method_column,
            values=value_column,
        )
    else:
        # Degenerate case: compare methods on a single aggregate-less unit.
        normalized = normalized.assign(_comparison_unit="__all_rows__")
        wide_frame = normalized.pivot(
            index="_comparison_unit",
            columns=method_column,
            values=value_column,
        )
    return wide_frame.sort_index()


def build_pairwise_results(
    wide_frame: pd.DataFrame,
    *,
    methods: list[Any],
    win_rate_spec: PairwiseMethodWinRateSpec,
    requested_pairs: list[RequestedMethodPair],
) -> list[dict[str, Any]]:
    """Build pairwise win-rate summaries for every method pair."""
    results: list[dict[str, Any]] = []
    if requested_pairs:
        method_pairs = [
            (pair.left_method, pair.right_method) for pair in requested_pairs
        ]
    else:
        method_pairs = list(itertools.combinations(methods, 2))
    for left_method, right_method in method_pairs:
        if left_method not in wide_frame.columns or right_method not in wide_frame.columns:
            continue

        left_values = wide_frame[left_method]
        right_values = wide_frame[right_method]
        comparable_mask = left_values.notna() & right_values.notna()
        comparable_count = int(comparable_mask.sum())

        left_win_count = 0
        right_win_count = 0
        tie_count = 0
        mean_delta_left_minus_right = None
        median_delta_left_minus_right = None

        if comparable_count > 0:
            deltas = (left_values[comparable_mask] - right_values[comparable_mask]).astype(float)
            if win_rate_spec.higher_is_better:
                left_wins = deltas > win_rate_spec.tie_tolerance
                right_wins = deltas < -win_rate_spec.tie_tolerance
            else:
                left_wins = deltas < -win_rate_spec.tie_tolerance
                right_wins = deltas > win_rate_spec.tie_tolerance

            left_win_count = int(left_wins.sum())
            right_win_count = int(right_wins.sum())
            tie_count = comparable_count - left_win_count - right_win_count
            mean_delta_left_minus_right = float(deltas.mean())
            median_delta_left_minus_right = float(deltas.median())

        non_tie_count = left_win_count + right_win_count
        results.append(
            {
                "left_method": normalize_scalar(left_method),
                "right_method": normalize_scalar(right_method),
                "comparable_result_count": comparable_count,
                "incomparable_result_count": int(len(wide_frame) - comparable_count),
                "left_win_count": left_win_count,
                "right_win_count": right_win_count,
                "tie_count": tie_count,
                "non_tie_count": non_tie_count,
                "left_win_rate": safe_ratio(left_win_count, comparable_count),
                "right_win_rate": safe_ratio(right_win_count, comparable_count),
                "tie_rate": safe_ratio(tie_count, comparable_count),
                "left_non_tie_win_rate": safe_ratio(left_win_count, non_tie_count),
                "right_non_tie_win_rate": safe_ratio(right_win_count, non_tie_count),
                "mean_delta_left_minus_right": mean_delta_left_minus_right,
                "median_delta_left_minus_right": median_delta_left_minus_right,
            }
        )
    return results


def build_pairwise_lookup(
    pairwise_results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build a direct left->right lookup for pairwise results."""
    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for result in pairwise_results:
        left_method = str(result["left_method"])
        right_method = str(result["right_method"])
        lookup.setdefault(left_method, {})[right_method] = dict(result)
        lookup.setdefault(right_method, {})[left_method] = reverse_pairwise_result(result)
    return normalize_for_json(lookup)


def reverse_pairwise_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the same pairwise result from the opposite method perspective."""
    mean_delta = result.get("mean_delta_left_minus_right")
    median_delta = result.get("median_delta_left_minus_right")
    reversed_mean_delta = None if mean_delta is None else -float(mean_delta)
    reversed_median_delta = None if median_delta is None else -float(median_delta)
    return {
        "left_method": result["right_method"],
        "right_method": result["left_method"],
        "comparable_result_count": result["comparable_result_count"],
        "incomparable_result_count": result["incomparable_result_count"],
        "left_win_count": result["right_win_count"],
        "right_win_count": result["left_win_count"],
        "tie_count": result["tie_count"],
        "non_tie_count": result["non_tie_count"],
        "left_win_rate": result["right_win_rate"],
        "right_win_rate": result["left_win_rate"],
        "tie_rate": result["tie_rate"],
        "left_non_tie_win_rate": result["right_non_tie_win_rate"],
        "right_non_tie_win_rate": result["left_non_tie_win_rate"],
        "mean_delta_left_minus_right": reversed_mean_delta,
        "median_delta_left_minus_right": reversed_median_delta,
    }


def build_pairwise_metric_matrix(
    *,
    methods: list[Any],
    pairwise_results: list[dict[str, Any]],
    left_value_key: str,
    right_value_key: str,
) -> pd.DataFrame:
    """Build a square method-vs-method matrix from one pairwise metric."""
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=object)
    for method in methods:
        matrix.at[method, method] = None
    for result in pairwise_results:
        left_method = result["left_method"]
        right_method = result["right_method"]
        matrix.at[left_method, right_method] = normalize_scalar(result[left_value_key])
        matrix.at[right_method, left_method] = normalize_scalar(result[right_value_key])
    return matrix


def safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return one safe ratio or None when the denominator is zero."""
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def unique_preserve_order(values: list[str]) -> list[str]:
    """Return ordered unique strings."""
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_requested_method_pairs(
    value: Any,
    *,
    label: str,
) -> list[RequestedMethodPair]:
    """Normalize optional explicitly requested method pairs."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisError(f"Expected '{label}' to be a list of method pairs.")

    normalized_pairs: list[RequestedMethodPair] = []
    seen_pair_keys: set[tuple[str, str]] = set()
    for index, raw_pair in enumerate(value):
        pair = normalize_requested_method_pair(
            raw_pair,
            label=f"{label}[{index}]",
        )
        pair_key = build_requested_pair_dedup_key(pair)
        if pair_key in seen_pair_keys:
            raise AnalysisError(
                f"Duplicate method pair requested at '{label}[{index}]': "
                f"{pair.left_method!r} vs {pair.right_method!r}."
            )
        seen_pair_keys.add(pair_key)
        normalized_pairs.append(pair)
    return normalized_pairs


def normalize_requested_method_pair(
    value: Any,
    *,
    label: str,
) -> RequestedMethodPair:
    """Normalize one requested pair from list or mapping syntax."""
    if isinstance(value, list):
        if len(value) != 2:
            raise AnalysisError(
                f"Expected '{label}' to contain exactly two method names."
            )
        left_method = require_nonempty_string(value[0], label=f"{label}[0]")
        right_method = require_nonempty_string(value[1], label=f"{label}[1]")
    elif isinstance(value, Mapping):
        left_method = require_nonempty_string(
            value.get("left_method"),
            label=f"{label}.left_method",
        )
        right_method = require_nonempty_string(
            value.get("right_method"),
            label=f"{label}.right_method",
        )
    else:
        raise AnalysisError(
            f"Expected '{label}' to be either [left_method, right_method] or a mapping."
        )

    if left_method == right_method:
        raise AnalysisError(
            f"Expected '{label}' to reference two different methods, got {left_method!r}."
        )
    return RequestedMethodPair(
        left_method=left_method,
        right_method=right_method,
    )


def build_requested_pair_dedup_key(pair: RequestedMethodPair) -> tuple[str, str]:
    """Build an undirected deduplication key for one method pair."""
    return tuple(sorted((pair.left_method, pair.right_method), key=str.casefold))


def resolve_requested_pairs(
    requested_pairs: list[RequestedMethodPair],
    *,
    available_methods: list[Any],
) -> list[RequestedMethodPair]:
    """Validate requested pairs against the methods present after filtering."""
    if not requested_pairs:
        return []

    available_method_names = {str(method) for method in available_methods}
    missing_methods: list[str] = []
    for pair in requested_pairs:
        for method_name in (pair.left_method, pair.right_method):
            if method_name not in available_method_names:
                missing_methods.append(method_name)
    if missing_methods:
        missing_unique = unique_preserve_order(missing_methods)
        raise AnalysisError(
            "Requested method pairs reference methods that do not exist after filtering: "
            f"{missing_unique}. Available methods: {sorted(available_method_names)}."
        )
    return requested_pairs


def collect_requested_pair_methods(
    requested_pairs: list[RequestedMethodPair],
) -> list[str]:
    """Collect ordered unique methods referenced by the requested pair list."""
    methods: list[str] = []
    for pair in requested_pairs:
        methods.append(pair.left_method)
        methods.append(pair.right_method)
    return unique_preserve_order(methods)


def normalize_optional_string_list(value: Any, *, label: str) -> list[str]:
    """Normalize an optional list of strings."""
    if value is None:
        return []
    return require_string_list(value, label=label)


def require_nonnegative_float(value: Any, *, label: str) -> float:
    """Require one non-negative numeric value."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, bool):
        raise AnalysisError(f"Expected '{label}' to be a non-negative float, got bool.")
    if isinstance(normalized_value, (int, float)):
        numeric_value = float(normalized_value)
        if math.isnan(numeric_value) or numeric_value < 0.0:
            raise AnalysisError(f"Expected '{label}' to be a non-negative float.")
        return numeric_value
    raise AnalysisError(
        f"Expected '{label}' to be a non-negative float, got {type(normalized_value).__name__}."
    )


if __name__ == "__main__":
    main()
