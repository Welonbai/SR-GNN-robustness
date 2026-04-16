#!/usr/bin/env python3
"""Render one report-table bundle into a PNG slide image."""

from __future__ import annotations

import argparse
import json
import string
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SUPPORTED_OUTPUT_FORMATS = {"png"}
ALLOWED_ALIGNMENTS = {"left", "center", "right"}
COLUMN_LABEL_SEPARATOR = " | "
TABLE_AX_POSITION = [0.01, 0.04, 0.98, 0.76]
STUB_COLUMN_WIDTH_WEIGHT = 1.35
LEAF_COLUMN_WIDTH_WEIGHT = 1.0
CELL_PADDING_FRACTION = 0.06


class AnalysisError(ValueError):
    """Raised when a render spec or report-table bundle is malformed."""


@dataclass(frozen=True)
class TitleSpec:
    """Title configuration for one rendered slide."""

    template: str
    align: str
    font_size: float
    color: str


@dataclass(frozen=True)
class FigureSpec:
    """Figure sizing and background configuration."""

    width: float
    height: float
    dpi: int
    background_color: str


@dataclass(frozen=True)
class TableSpec:
    """Table formatting configuration for the renderer."""

    font_size: float
    round_digits: int
    text_color: str
    show_grid: bool
    auto_shrink: bool
    wrap_text: bool
    cell_align: str
    display_alias: dict[str, str]
    value_alias: dict[str, dict[str, str]]


@dataclass(frozen=True)
class RenderSpec:
    """Validated render YAML content."""

    input_bundle_dir: Path | None
    style_name: str
    output_format: str
    title: TitleSpec
    figure: FigureSpec
    table: TableSpec


@dataclass(frozen=True)
class TableStructure:
    """Validated row/column hierarchy for one rendered table."""

    row_levels: list[str]
    col_levels: list[str]
    row_tuples: list[tuple[Any, ...]]
    column_tuples: list[tuple[Any, ...]]
    row_column_names: list[str]
    value_column_names: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render one report-table bundle into a PNG image.",
    )
    parser.add_argument(
        "--input-dir",
        help="Path to one view bundle directory containing table.csv and meta.json.",
    )
    parser.add_argument(
        "--bundle-dir",
        dest="input_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to one render YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the renderer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_file(args.config, label="render config")
        render_spec = parse_render_spec(load_yaml_mapping(config_path, label="render config"))
        bundle_dir = resolve_bundle_dir(cli_input_dir=args.input_dir, render_spec=render_spec)

        table_path = require_file(bundle_dir / "table.csv", label="bundle table")
        meta_path = require_file(bundle_dir / "meta.json", label="bundle metadata")
        table_dataframe = load_table_csv(table_path)
        meta_payload = load_json_mapping(meta_path, label="bundle metadata")
        row_column_names = extract_identifier_column_names(
            meta_payload=meta_payload,
            dataframe=table_dataframe,
        )
        table_structure = extract_table_structure(
            meta_payload=meta_payload,
            dataframe=table_dataframe,
            row_column_names=row_column_names,
        )
        title_text = resolve_title(template=render_spec.title.template, meta_payload=meta_payload)

        output_path = bundle_dir / "render.png"
        render_png(
            dataframe=table_dataframe,
            table_structure=table_structure,
            title_text=title_text,
            render_spec=render_spec,
            output_path=output_path,
        )

        print(
            f"Wrote '{output_path}' from bundle '{bundle_dir}' "
            f"using style '{render_spec.style_name}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_render_spec(payload: Mapping[str, Any]) -> RenderSpec:
    """Validate and normalize one render YAML spec."""
    input_bundle_dir = normalize_optional_directory_path(
        payload.get("input_dir"),
        label="input_dir",
    )
    style_name = require_nonempty_string(payload.get("style_name"), label="style_name")
    output_format = require_nonempty_string(payload.get("output_format"), label="output_format").lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise AnalysisError(
            f"Unsupported output_format '{output_format}'. Supported values: {sorted(SUPPORTED_OUTPUT_FORMATS)}."
        )

    title_payload = require_mapping(payload.get("title"), label="title")
    figure_payload = require_mapping(payload.get("figure"), label="figure")
    table_payload = require_mapping(payload.get("table"), label="table")

    title_spec = TitleSpec(
        template=require_nonempty_string(title_payload.get("template"), label="title.template"),
        align=require_alignment(title_payload.get("align"), label="title.align"),
        font_size=require_positive_float(title_payload.get("font_size"), label="title.font_size"),
        color=require_nonempty_string(title_payload.get("color"), label="title.color"),
    )
    figure_spec = FigureSpec(
        width=require_positive_float(figure_payload.get("width"), label="figure.width"),
        height=require_positive_float(figure_payload.get("height"), label="figure.height"),
        dpi=require_positive_int(figure_payload.get("dpi"), label="figure.dpi"),
        background_color=require_nonempty_string(
            figure_payload.get("background_color"),
            label="figure.background_color",
        ),
    )
    table_spec = TableSpec(
        font_size=require_positive_float(table_payload.get("font_size"), label="table.font_size"),
        round_digits=require_nonnegative_int(table_payload.get("round_digits"), label="table.round_digits"),
        text_color=require_nonempty_string(table_payload.get("text_color"), label="table.text_color"),
        show_grid=require_bool(table_payload.get("show_grid"), label="table.show_grid"),
        auto_shrink=require_bool(table_payload.get("auto_shrink"), label="table.auto_shrink"),
        wrap_text=require_bool(table_payload.get("wrap_text"), label="table.wrap_text"),
        cell_align=require_alignment(table_payload.get("cell_align"), label="table.cell_align"),
        display_alias=normalize_string_mapping(
            table_payload.get("display_alias", {}),
            label="table.display_alias",
        ),
        value_alias=normalize_value_alias_mapping(
            table_payload.get("value_alias", {}),
            label="table.value_alias",
        ),
    )

    return RenderSpec(
        input_bundle_dir=input_bundle_dir,
        style_name=style_name,
        output_format=output_format,
        title=title_spec,
        figure=figure_spec,
        table=table_spec,
    )


def render_png(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    title_text: str,
    render_spec: RenderSpec,
    output_path: Path,
) -> None:
    """Render one report table into a PNG image."""
    validate_display_alias_columns(
        dataframe=dataframe,
        display_alias=render_spec.table.display_alias,
    )
    display_dataframe = format_dataframe_for_display(
        dataframe,
        identifier_columns=set(table_structure.row_column_names),
        round_digits=render_spec.table.round_digits,
        value_alias=render_spec.table.value_alias,
    )

    fig, ax = plt.subplots(
        figsize=(render_spec.figure.width, render_spec.figure.height),
        dpi=render_spec.figure.dpi,
    )
    fig.patch.set_facecolor(render_spec.figure.background_color)
    ax.set_facecolor(render_spec.figure.background_color)
    ax.axis("off")
    ax.set_position(TABLE_AX_POSITION)

    draw_structured_table(
        ax=ax,
        dataframe=display_dataframe,
        table_structure=table_structure,
        render_spec=render_spec,
    )

    fig.suptitle(
        title_text,
        x=title_alignment_x(render_spec.title.align),
        y=0.95,
        ha=render_spec.title.align,
        va="center",
        fontsize=render_spec.title.font_size,
        color=render_spec.title.color,
    )
    fig.savefig(
        output_path,
        dpi=render_spec.figure.dpi,
        facecolor=render_spec.figure.background_color,
    )
    plt.close(fig)


def resolve_bundle_dir(*, cli_input_dir: str | None, render_spec: RenderSpec) -> Path:
    """Resolve the input bundle directory from CLI override or render YAML."""
    if cli_input_dir is not None:
        bundle_dir = resolve_existing_directory(cli_input_dir, label="input directory")
    elif render_spec.input_bundle_dir is not None:
        bundle_dir = render_spec.input_bundle_dir
    else:
        raise AnalysisError("The render config must contain 'input_dir' or the CLI must provide '--input-dir'.")

    ensure_path_within(bundle_dir, RESULTS_ROOT, label="bundle directory")
    return bundle_dir


def draw_structured_table(
    *,
    ax: Any,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    render_spec: RenderSpec,
) -> None:
    """Draw one hierarchy-aware table with merged headers and grouped row labels."""
    header_row_count = len(table_structure.col_levels)
    data_row_count = len(table_structure.row_tuples)
    stub_column_count = len(table_structure.row_levels)
    leaf_column_count = len(table_structure.column_tuples)
    total_row_count = header_row_count + data_row_count

    if total_row_count <= 0:
        raise AnalysisError("Cannot render an empty table structure.")

    column_width_weights = ([STUB_COLUMN_WIDTH_WEIGHT] * stub_column_count) + (
        [LEAF_COLUMN_WIDTH_WEIGHT] * leaf_column_count
    )
    column_boundaries = build_boundaries(column_width_weights)

    ax.set_xlim(0.0, column_boundaries[-1])
    ax.set_ylim(float(total_row_count), 0.0)

    header_label_row_index = header_row_count - 1
    for header_row_index in range(header_row_count):
        for stub_column_index, row_level_name in enumerate(table_structure.row_levels):
            header_text = ""
            if header_row_index == header_label_row_index:
                header_text = render_spec.table.display_alias.get(row_level_name, row_level_name)
            draw_cell_block(
                ax=ax,
                x0=column_boundaries[stub_column_index],
                x1=column_boundaries[stub_column_index + 1],
                y0=float(header_row_index),
                y1=float(header_row_index + 1),
                text=header_text,
                font_weight="bold",
                render_spec=render_spec,
                total_table_width=column_boundaries[-1],
                total_row_count=total_row_count,
            )

    for level_index in range(header_row_count):
        for start_index, end_index in iterate_hierarchy_spans(
            table_structure.column_tuples,
            level_index=level_index,
        ):
            draw_cell_block(
                ax=ax,
                x0=column_boundaries[stub_column_count + start_index],
                x1=column_boundaries[stub_column_count + end_index],
                y0=float(level_index),
                y1=float(level_index + 1),
                text=resolve_column_header_label(
                    column_tuple=table_structure.column_tuples[start_index],
                    value_column_name=table_structure.value_column_names[start_index],
                    level_index=level_index,
                    level_count=header_row_count,
                    display_alias=render_spec.table.display_alias,
                ),
                font_weight="bold",
                render_spec=render_spec,
                total_table_width=column_boundaries[-1],
                total_row_count=total_row_count,
            )

    for level_index in range(stub_column_count):
        for start_index, end_index in iterate_hierarchy_spans(
            table_structure.row_tuples,
            level_index=level_index,
        ):
            draw_cell_block(
                ax=ax,
                x0=column_boundaries[level_index],
                x1=column_boundaries[level_index + 1],
                y0=float(header_row_count + start_index),
                y1=float(header_row_count + end_index),
                text=stringify_header_value(
                    dataframe.iloc[start_index][table_structure.row_column_names[level_index]]
                ),
                font_weight="normal",
                render_spec=render_spec,
                total_table_width=column_boundaries[-1],
                total_row_count=total_row_count,
            )

    for row_index in range(data_row_count):
        for leaf_column_index, value_column_name in enumerate(table_structure.value_column_names):
            draw_cell_block(
                ax=ax,
                x0=column_boundaries[stub_column_count + leaf_column_index],
                x1=column_boundaries[stub_column_count + leaf_column_index + 1],
                y0=float(header_row_count + row_index),
                y1=float(header_row_count + row_index + 1),
                text=str(dataframe.iloc[row_index][value_column_name]),
                font_weight="normal",
                render_spec=render_spec,
                total_table_width=column_boundaries[-1],
                total_row_count=total_row_count,
            )


def draw_cell_block(
    *,
    ax: Any,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    text: str,
    font_weight: str,
    render_spec: RenderSpec,
    total_table_width: float,
    total_row_count: int,
) -> None:
    """Draw one rectangular cell or merged block and its text."""
    edge_color = "black" if render_spec.table.show_grid else render_spec.figure.background_color
    line_width = 0.5 if render_spec.table.show_grid else 0.0

    rectangle = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=render_spec.figure.background_color,
        edgecolor=edge_color,
        linewidth=line_width,
    )
    ax.add_patch(rectangle)

    if not text:
        return

    text_artist = ax.text(
        resolve_text_x(x0=x0, x1=x1, align=render_spec.table.cell_align),
        (y0 + y1) / 2.0,
        text,
        ha=render_spec.table.cell_align,
        va="center",
        color=render_spec.table.text_color,
        fontsize=resolve_font_size(
            text=text,
            cell_width=x1 - x0,
            cell_height=y1 - y0,
            total_table_width=total_table_width,
            total_row_count=total_row_count,
            render_spec=render_spec,
        ),
        fontweight=font_weight,
        wrap=render_spec.table.wrap_text,
        clip_on=True,
    )
    text_artist.set_clip_path(rectangle)


def build_boundaries(width_weights: list[float]) -> list[float]:
    """Convert width weights into cumulative x boundaries."""
    boundaries = [0.0]
    for weight in width_weights:
        boundaries.append(boundaries[-1] + float(weight))
    return boundaries


def iterate_hierarchy_spans(keys: list[tuple[Any, ...]], *, level_index: int) -> list[tuple[int, int]]:
    """Group consecutive hierarchy entries that share the same prefix through one level."""
    spans: list[tuple[int, int]] = []
    start_index = 0
    while start_index < len(keys):
        prefix = keys[start_index][: level_index + 1]
        end_index = start_index + 1
        while end_index < len(keys) and keys[end_index][: level_index + 1] == prefix:
            end_index += 1
        spans.append((start_index, end_index))
        start_index = end_index
    return spans


def resolve_column_header_label(
    *,
    column_tuple: tuple[Any, ...],
    value_column_name: str,
    level_index: int,
    level_count: int,
    display_alias: Mapping[str, str],
) -> str:
    """Resolve the text for one column-header block."""
    if level_count == 1:
        return display_alias.get(value_column_name, stringify_header_value(column_tuple[0]))
    return stringify_header_value(column_tuple[level_index])


def stringify_header_value(value: Any) -> str:
    """Convert one structural value into renderable header/body text."""
    normalized_value = normalize_scalar(value)
    if normalized_value is None:
        return ""
    return str(normalized_value)


def resolve_text_x(*, x0: float, x1: float, align: str) -> float:
    """Resolve one text anchor position inside a cell rectangle."""
    padding = (x1 - x0) * CELL_PADDING_FRACTION
    if align == "left":
        return x0 + padding
    if align == "center":
        return (x0 + x1) / 2.0
    if align == "right":
        return x1 - padding
    raise AnalysisError(f"Unsupported alignment '{align}'.")


def resolve_font_size(
    *,
    text: str,
    cell_width: float,
    cell_height: float,
    total_table_width: float,
    total_row_count: int,
    render_spec: RenderSpec,
) -> float:
    """Return either the configured font size or a conservative shrunken size."""
    base_font_size = render_spec.table.font_size
    if not render_spec.table.auto_shrink:
        return base_font_size

    table_width_inches = render_spec.figure.width * TABLE_AX_POSITION[2]
    table_height_inches = render_spec.figure.height * TABLE_AX_POSITION[3]
    cell_width_inches = table_width_inches * (cell_width / total_table_width)
    cell_height_inches = table_height_inches * (cell_height / float(total_row_count))
    max_height_points = cell_height_inches * 72.0 * 0.55

    longest_line_length = max(len(line) for line in text.splitlines()) if text else 1
    estimated_char_width = 0.56 * max(longest_line_length, 1)
    max_width_points = (cell_width_inches * 72.0 * 0.9) / estimated_char_width
    return max(4.0, min(base_font_size, max_height_points, max_width_points))


def format_dataframe_for_display(
    dataframe: pd.DataFrame,
    *,
    identifier_columns: set[str],
    round_digits: int,
    value_alias: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Convert a dataframe into display strings for slide rendering."""
    validate_value_alias_columns(dataframe=dataframe, value_alias=value_alias)

    formatted_columns: dict[str, pd.Series] = {}
    for column_name in dataframe.columns:
        normalized_column_name = str(column_name)
        is_identifier_column = normalized_column_name in identifier_columns
        column_value_alias = value_alias.get(normalized_column_name, {})
        formatted_columns[normalized_column_name] = dataframe[column_name].map(
            lambda value: format_cell_value(
                value,
                is_identifier_column=is_identifier_column,
                round_digits=round_digits,
                value_alias=column_value_alias,
            )
        )
    return pd.DataFrame(formatted_columns)


def validate_display_alias_columns(
    *,
    dataframe: pd.DataFrame,
    display_alias: Mapping[str, str],
) -> None:
    """Require every display_alias key to exist in table.csv."""
    available_columns = [str(column) for column in dataframe.columns]
    missing_columns = sorted(column for column in display_alias if column not in available_columns)
    if missing_columns:
        raise AnalysisError(
            f"The render config display_alias keys {missing_columns} do not exist in table.csv "
            f"columns {available_columns}."
        )


def validate_value_alias_columns(
    *,
    dataframe: pd.DataFrame,
    value_alias: Mapping[str, Mapping[str, str]],
) -> None:
    """Require every value_alias column to exist in table.csv."""
    available_columns = [str(column) for column in dataframe.columns]
    missing_columns = sorted(column for column in value_alias if column not in available_columns)
    if missing_columns:
        raise AnalysisError(
            f"The render config value_alias keys {missing_columns} do not exist in table.csv "
            f"columns {available_columns}."
        )


def format_cell_value(
    value: Any,
    *,
    is_identifier_column: bool,
    round_digits: int,
    value_alias: Mapping[str, str],
) -> str:
    """Format one cell value for display."""
    if pd.isna(value):
        return ""

    normalized_value = normalize_scalar(value)
    raw_lookup_key = stringify_alias_lookup_value(normalized_value)
    if is_identifier_column:
        formatted_value = str(normalized_value)
    elif isinstance(normalized_value, bool):
        formatted_value = "True" if normalized_value else "False"
    elif isinstance(normalized_value, Integral):
        formatted_value = str(int(normalized_value))
    elif isinstance(normalized_value, Real):
        formatted_value = f"{float(normalized_value):.{round_digits}f}"
    else:
        formatted_value = str(normalized_value)

    return value_alias.get(raw_lookup_key, value_alias.get(formatted_value, formatted_value))


def extract_identifier_column_names(
    *,
    meta_payload: Mapping[str, Any],
    dataframe: pd.DataFrame,
) -> list[str]:
    """Read ordered identifier columns from meta.json rows and validate table.csv layout."""
    rows_value = meta_payload.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise AnalysisError("The bundle metadata must contain a non-empty 'rows' list for rendering.")

    ordered_columns: list[str] = []
    for index, raw_name in enumerate(rows_value):
        column_name = require_nonempty_string(raw_name, label=f"meta.json rows[{index}]")
        ordered_columns.append(column_name)

    actual_prefix = [str(column) for column in dataframe.columns[: len(ordered_columns)]]
    if actual_prefix != ordered_columns:
        raise AnalysisError(
            f"The bundle metadata rows {ordered_columns} do not match the leading table.csv columns "
            f"{actual_prefix}. Full columns: {list(dataframe.columns)}."
        )
    return ordered_columns


def extract_table_structure(
    *,
    meta_payload: Mapping[str, Any],
    dataframe: pd.DataFrame,
    row_column_names: list[str],
) -> TableStructure:
    """Load hierarchy metadata from meta.json with a flat fallback for old bundles."""
    row_levels = extract_optional_string_list(meta_payload.get("row_levels"), label="meta.json row_levels")
    if row_levels is None:
        row_levels = list(row_column_names)
    elif row_levels != row_column_names:
        raise AnalysisError(
            f"meta.json row_levels {row_levels} do not match table.csv identifier columns {row_column_names}."
        )

    value_column_names = [str(column) for column in dataframe.columns[len(row_column_names) :]]
    if not value_column_names:
        raise AnalysisError("The rendered table must contain at least one value column.")

    dataframe_row_tuples = extract_dataframe_row_tuples(
        dataframe=dataframe,
        row_column_names=row_column_names,
    )
    meta_row_tuples = extract_optional_structure_tuples(
        meta_payload.get("row_tuples"),
        expected_arity=len(row_levels),
        label="meta.json row_tuples",
    )
    if meta_row_tuples is None:
        row_tuples = dataframe_row_tuples
    else:
        if meta_row_tuples != dataframe_row_tuples:
            raise AnalysisError(
                "meta.json row_tuples do not match the actual identifier rows in table.csv."
            )
        row_tuples = meta_row_tuples

    raw_col_levels = meta_payload.get("col_levels")
    raw_column_tuples = meta_payload.get("column_tuples")
    if raw_col_levels is None and raw_column_tuples is None:
        col_levels = ["column"]
        column_tuples = [(column_name,) for column_name in value_column_names]
    elif raw_col_levels is None or raw_column_tuples is None:
        raise AnalysisError("meta.json must contain both 'col_levels' and 'column_tuples', or neither.")
    else:
        col_levels = require_string_list(raw_col_levels, label="meta.json col_levels")
        column_tuples = extract_structure_tuples(
            raw_column_tuples,
            expected_arity=len(col_levels),
            label="meta.json column_tuples",
        )
        if len(column_tuples) != len(value_column_names):
            raise AnalysisError(
                f"meta.json column_tuples has {len(column_tuples)} entries but table.csv has "
                f"{len(value_column_names)} value columns."
            )

        flattened_column_labels = [
            flatten_column_tuple(column_tuple) for column_tuple in column_tuples
        ]
        if flattened_column_labels != value_column_names:
            raise AnalysisError(
                "meta.json column_tuples do not match the flattened value columns in table.csv. "
                f"Expected {value_column_names}, got {flattened_column_labels}."
            )

    return TableStructure(
        row_levels=row_levels,
        col_levels=col_levels,
        row_tuples=row_tuples,
        column_tuples=column_tuples,
        row_column_names=row_column_names,
        value_column_names=value_column_names,
    )


def extract_optional_string_list(value: Any, *, label: str) -> list[str] | None:
    """Return a validated non-empty string list when present."""
    if value is None:
        return None
    return require_string_list(value, label=label)


def extract_optional_structure_tuples(
    value: Any,
    *,
    expected_arity: int,
    label: str,
) -> list[tuple[Any, ...]] | None:
    """Return structure tuples when present, otherwise None."""
    if value is None:
        return None
    return extract_structure_tuples(value, expected_arity=expected_arity, label=label)


def extract_structure_tuples(
    value: Any,
    *,
    expected_arity: int,
    label: str,
) -> list[tuple[Any, ...]]:
    """Validate one JSON list-of-tuples payload from meta.json."""
    if not isinstance(value, list):
        raise AnalysisError(f"Expected '{label}' to be a list, got {type(value).__name__}.")

    normalized_tuples: list[tuple[Any, ...]] = []
    for index, raw_item in enumerate(value):
        if not isinstance(raw_item, list):
            raise AnalysisError(
                f"Expected '{label}[{index}]' to be a list, got {type(raw_item).__name__}."
            )
        if len(raw_item) != expected_arity:
            raise AnalysisError(
                f"Expected '{label}[{index}]' to contain {expected_arity} items, got {len(raw_item)}."
            )
        normalized_tuples.append(tuple(normalize_scalar(item) for item in raw_item))
    return normalized_tuples


def extract_dataframe_row_tuples(
    *,
    dataframe: pd.DataFrame,
    row_column_names: list[str],
) -> list[tuple[Any, ...]]:
    """Read ordered row tuples directly from table.csv."""
    row_tuples: list[tuple[Any, ...]] = []
    for _, row in dataframe.iterrows():
        row_tuples.append(tuple(normalize_scalar(row[column_name]) for column_name in row_column_names))
    return row_tuples


def flatten_column_tuple(column_tuple: tuple[Any, ...]) -> str:
    """Flatten one structural column tuple using the builder's CSV label convention."""
    parts = [str(item) for item in column_tuple if str(item) != ""]
    if not parts:
        return "column"
    return COLUMN_LABEL_SEPARATOR.join(parts)


def resolve_title(*, template: str, meta_payload: Mapping[str, Any]) -> str:
    """Resolve the title template using meta.json context only."""
    context_payload = require_mapping(meta_payload.get("context"), label="meta.json context")
    render_context = build_render_context(context_payload)

    formatter = string.Formatter()
    required_fields: list[str] = []
    for _literal_text, field_name, _format_spec, _conversion in formatter.parse(template):
        if field_name is None:
            continue
        if not field_name or any(symbol in field_name for symbol in ".[]"):
            raise AnalysisError(
                f"Unsupported title template field '{field_name}'. Use simple keys from meta.json context only."
            )
        required_fields.append(field_name)

    missing_fields = sorted(field for field in required_fields if field not in render_context)
    if missing_fields:
        raise AnalysisError(
            f"Could not resolve title template '{template}'. Missing context keys: {missing_fields}. "
            f"Available context keys: {sorted(render_context.keys())}."
        )

    try:
        return template.format_map(render_context)
    except Exception as exc:  # pragma: no cover - defensive formatting error wrapping
        raise AnalysisError(
            f"Could not resolve title template '{template}': {exc}."
        ) from exc


def build_render_context(context_payload: Mapping[str, Any]) -> dict[str, str]:
    """Convert meta.json context values into title-template strings."""
    render_context: dict[str, str] = {}
    for key, value in context_payload.items():
        render_context[require_nonempty_string(key, label="meta.json context key")] = stringify_context_value(value)
    return render_context


def stringify_context_value(value: Any) -> str:
    """Convert one context value into a readable template string."""
    if isinstance(value, list):
        return ", ".join(stringify_context_value(item) for item in value)
    if isinstance(value, dict):
        raise AnalysisError("Nested objects are not supported in meta.json context for title rendering.")
    return str(normalize_scalar(value))


def title_alignment_x(align: str) -> float:
    """Map title alignment names onto matplotlib figure coordinates."""
    if align == "left":
        return 0.01
    if align == "center":
        return 0.5
    if align == "right":
        return 0.99
    raise AnalysisError(f"Unsupported alignment '{align}'.")


def load_table_csv(path: Path) -> pd.DataFrame:
    """Load one report table CSV and require at least one row and one column."""
    try:
        dataframe = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive pandas error wrapping
        raise AnalysisError(f"Could not read table CSV at '{path}': {exc}.") from exc

    if dataframe.empty:
        raise AnalysisError(f"The table CSV at '{path}' is empty.")
    if len(dataframe.columns) == 0:
        raise AnalysisError(f"The table CSV at '{path}' does not contain any columns.")
    return dataframe


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
    """Load a JSON file and require a top-level object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid JSON: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level JSON object.")
    return payload


def normalize_optional_directory_path(value: Any, *, label: str) -> Path | None:
    """Resolve one optional existing directory path."""
    if value is None:
        return None
    raw_path = require_nonempty_string(value, label=label)
    return resolve_existing_directory(raw_path, label=label)


def resolve_existing_file(raw_path: str, *, label: str) -> Path:
    """Resolve and validate one existing file path."""
    path = resolve_repo_path(raw_path)
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def resolve_existing_directory(raw_path: str, *, label: str) -> Path:
    """Resolve and validate one existing directory path."""
    path = resolve_repo_path(raw_path)
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_dir():
        raise AnalysisError(f"The {label} path is not a directory: '{path}'.")
    return path


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a path relative to the repository root when needed."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def require_file(path: Path, *, label: str) -> Path:
    """Require one file to exist."""
    if not path.is_file():
        raise AnalysisError(f"Missing required {label} file: '{path}'.")
    return path


def ensure_path_within(path: Path, root: Path, *, label: str) -> None:
    """Require a path to stay inside one repository subtree."""
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(
            f"The {label} must stay under '{root.resolve()}', got '{path}'."
        ) from exc


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Require a mapping value."""
    if not isinstance(value, Mapping):
        raise AnalysisError(f"Expected '{label}' to be a mapping, got {type(value).__name__}.")
    return value


def normalize_string_mapping(value: Any, *, label: str) -> dict[str, str]:
    """Normalize an optional mapping of strings to strings."""
    if value is None:
        return {}

    mapping = require_mapping(value, label=label)
    normalized: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = require_nonempty_string(raw_key, label=f"{label} key")
        normalized[key] = require_nonempty_string(raw_value, label=f"{label}[{key}]")
    return normalized


def normalize_value_alias_mapping(value: Any, *, label: str) -> dict[str, dict[str, str]]:
    """Normalize an optional nested mapping from column names to value aliases."""
    if value is None:
        return {}

    mapping = require_mapping(value, label=label)
    normalized: dict[str, dict[str, str]] = {}
    for raw_column_name, raw_alias_map in mapping.items():
        column_name = require_nonempty_string(raw_column_name, label=f"{label} column")
        alias_map = require_mapping(raw_alias_map, label=f"{label}[{column_name}]")

        normalized_alias_map: dict[str, str] = {}
        for raw_key, raw_value in alias_map.items():
            alias_key = stringify_alias_lookup_value(raw_key)
            normalized_alias_map[alias_key] = require_nonempty_string(
                raw_value,
                label=f"{label}[{column_name}][{alias_key}]",
            )
        normalized[column_name] = normalized_alias_map
    return normalized


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


def require_positive_float(value: Any, *, label: str) -> float:
    """Require a positive numeric value."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be numeric, got bool.")
    if isinstance(value, Real):
        numeric_value = float(value)
        if numeric_value <= 0:
            raise AnalysisError(f"Expected '{label}' to be greater than zero, got {numeric_value}.")
        return numeric_value
    raise AnalysisError(f"Expected '{label}' to be numeric, got {type(value).__name__}.")


def require_positive_int(value: Any, *, label: str) -> int:
    """Require a positive integer value."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be an integer, got bool.")
    if isinstance(value, Integral):
        integer_value = int(value)
        if integer_value <= 0:
            raise AnalysisError(f"Expected '{label}' to be greater than zero, got {integer_value}.")
        return integer_value
    raise AnalysisError(f"Expected '{label}' to be an integer, got {type(value).__name__}.")


def require_nonnegative_int(value: Any, *, label: str) -> int:
    """Require a non-negative integer value."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be an integer, got bool.")
    if isinstance(value, Integral):
        integer_value = int(value)
        if integer_value < 0:
            raise AnalysisError(f"Expected '{label}' to be non-negative, got {integer_value}.")
        return integer_value
    raise AnalysisError(f"Expected '{label}' to be an integer, got {type(value).__name__}.")


def require_bool(value: Any, *, label: str) -> bool:
    """Require a boolean value."""
    if not isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be a boolean, got {type(value).__name__}.")
    return value


def require_alignment(value: Any, *, label: str) -> str:
    """Require one supported alignment token."""
    alignment = require_nonempty_string(value, label=label).lower()
    if alignment not in ALLOWED_ALIGNMENTS:
        raise AnalysisError(
            f"Unsupported alignment '{alignment}' for '{label}'. Allowed values: {sorted(ALLOWED_ALIGNMENTS)}."
        )
    return alignment


def stringify_alias_lookup_value(value: Any) -> str:
    """Convert one scalar into a stable alias-lookup key."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, bool):
        return "True" if normalized_value else "False"
    if isinstance(normalized_value, Integral):
        return str(int(normalized_value))
    if isinstance(normalized_value, Real):
        return str(float(normalized_value))
    return str(normalized_value)


def normalize_scalar(value: Any) -> Any:
    """Convert pandas and numpy scalar types into plain Python primitives."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value


if __name__ == "__main__":
    main()
