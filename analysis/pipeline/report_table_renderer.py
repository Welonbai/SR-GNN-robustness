#!/usr/bin/env python3
"""Render one report-table bundle into a PNG slide image."""

from __future__ import annotations

import argparse
import json
import string
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SUPPORTED_OUTPUT_FORMATS = {"png"}
ALLOWED_ALIGNMENTS = {"left", "center", "right"}
ALLOWED_COMPARE_ALONG = {"rows"}
ALLOWED_BEST_VALUE_MODES = {"max"}
COLUMN_LABEL_SEPARATOR = " | "
TABLE_AX_POSITION = [0.01, 0.04, 0.98, 0.76]
STUB_COLUMN_WIDTH_WEIGHT = 1.35
LEAF_COLUMN_WIDTH_WEIGHT = 1.0
CELL_PADDING_FRACTION = 0.06
GROUP_SEPARATOR_LINE_WIDTH = 1.2
SECOND_BEST_UNDERLINE_LINE_WIDTH = 1.3
SECOND_BEST_UNDERLINE_Y_FRACTION = 0.76
SECOND_BEST_UNDERLINE_MIN_FRACTION = 0.42
SECOND_BEST_UNDERLINE_MAX_FRACTION = 0.8
SECOND_BEST_UNDERLINE_PER_CHAR_FRACTION = 0.1
METRIC_SCOPE_COLUMN = "metric_scope"
GROUND_TRUTH_SCOPE = "ground_truth"
ABSOLUTE_VALUE_DISPLAY_MODE = "absolute"
SIGNED_PERCENT_VALUE_DISPLAY_MODE = "signed_percent"
RELATIVE_HEATMAP_NEUTRAL_COLOR = "#FFFFFF"
RELATIVE_HEATMAP_POSITIVE_COLOR = "#2CA25F"
RELATIVE_HEATMAP_NEGATIVE_COLOR = "#DE2D26"


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
class DisplayAliasSpec:
    """Display-only aliases for header labels and structural values."""

    label_alias: dict[str, str]
    value_alias: dict[str, dict[str, str]]


@dataclass(frozen=True)
class BestValueBoldingSpec:
    """Configuration for emphasizing ranked values within each comparable slice."""

    compare_along: str
    mode: str
    partition_by_levels: list[str]
    underline_second_best: bool


@dataclass(frozen=True)
class TableSpec:
    """Table formatting configuration for the renderer."""

    font_size: float
    round_digits: int
    signed_percent_round_digits: int | None
    text_color: str
    show_grid: bool
    auto_shrink: bool
    wrap_text: bool
    cell_align: str
    display_alias: DisplayAliasSpec
    value_alias: dict[str, dict[str, str]]
    dimension_value_orders: dict[str, list[Any]]
    scope_colors: dict[str, dict[str, str]]
    best_value_bolding: BestValueBoldingSpec | None
    top_level_group_separators: bool


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


@dataclass(frozen=True)
class RankedValueCellHighlights:
    """Resolved cell coordinates for ranked-value emphasis within data cells."""

    best_value_cells: set[tuple[int, int]]
    second_best_value_cells: set[tuple[int, int]]


@dataclass(frozen=True)
class SignedPercentHeatmapScale:
    """Positive/negative magnitude scales for one GT signed-percent comparison group."""

    positive_abs_max: float | None
    negative_abs_max: float | None


@dataclass(frozen=True)
class DataCellPresentation:
    """Per-cell display modes and GT-relative heatmap scaling for one rendered table."""

    display_modes: list[list[str]]
    signed_percent_scales: list[list[SignedPercentHeatmapScale | None]]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render one report-table bundle into a PNG image.",
    )
    parser.add_argument(
        "--bundle-dir",
        help="Path to one view bundle directory containing table.csv and meta.json.",
    )
    parser.add_argument(
        "--input-dir",
        dest="bundle_dir",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bundle-parent-dir",
        help="Path to a parent directory whose direct child bundle directories will all be rendered.",
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
        render_mode, bundle_dirs = resolve_bundle_targets(
            cli_bundle_dir=args.bundle_dir,
            cli_bundle_parent_dir=args.bundle_parent_dir,
            render_spec=render_spec,
        )
        output_paths = [
            render_bundle(bundle_dir=bundle_dir, render_spec=render_spec) for bundle_dir in bundle_dirs
        ]

        if render_mode == "single":
            print(
                f"Wrote '{output_paths[0]}' from bundle '{bundle_dirs[0]}' "
                f"using style '{render_spec.style_name}'."
            )
        else:
            rendered_paths = ", ".join(str(path) for path in output_paths)
            print(
                f"Rendered {len(output_paths)} bundle(s) using style '{render_spec.style_name}': "
                f"{rendered_paths}"
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
        signed_percent_round_digits=normalize_optional_nonnegative_int(
            table_payload.get("signed_percent_round_digits"),
            label="table.signed_percent_round_digits",
        ),
        text_color=require_nonempty_string(table_payload.get("text_color"), label="table.text_color"),
        show_grid=require_bool(table_payload.get("show_grid"), label="table.show_grid"),
        auto_shrink=require_bool(table_payload.get("auto_shrink"), label="table.auto_shrink"),
        wrap_text=require_bool(table_payload.get("wrap_text"), label="table.wrap_text"),
        cell_align=require_alignment(table_payload.get("cell_align"), label="table.cell_align"),
        display_alias=normalize_display_alias_spec(
            table_payload.get("display_alias", {}),
            label="table.display_alias",
        ),
        value_alias=normalize_value_alias_mapping(
            table_payload.get("value_alias", {}),
            label="table.value_alias",
        ),
        dimension_value_orders=normalize_dimension_value_orders(
            table_payload.get("dimension_value_orders", {}),
            label="table.dimension_value_orders",
        ),
        scope_colors=normalize_dimension_value_string_mapping(
            table_payload.get("scope_colors", {}),
            label="table.scope_colors",
        ),
        best_value_bolding=normalize_best_value_bolding_spec(
            table_payload.get("best_value_bolding"),
            label="table.best_value_bolding",
        ),
        top_level_group_separators=require_bool(
            table_payload.get("top_level_group_separators", False),
            label="table.top_level_group_separators",
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
    data_cell_presentation: DataCellPresentation,
    title_text: str,
    render_spec: RenderSpec,
    output_path: Path,
) -> None:
    """Render one report table into a PNG image."""
    validate_display_alias_targets(
        table_structure=table_structure,
        display_alias=render_spec.table.display_alias,
    )
    validate_scope_color_targets(
        table_structure=table_structure,
        scope_colors=render_spec.table.scope_colors,
    )
    display_dataframe = format_dataframe_for_display(
        dataframe,
        identifier_columns=set(table_structure.row_column_names),
        round_digits=render_spec.table.round_digits,
        signed_percent_round_digits=render_spec.table.signed_percent_round_digits,
        value_alias=render_spec.table.value_alias,
        table_structure=table_structure,
        data_cell_presentation=data_cell_presentation,
    )
    ranked_value_highlights = resolve_ranked_value_highlights(
        dataframe=dataframe,
        table_structure=table_structure,
        best_value_bolding=render_spec.table.best_value_bolding,
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
        raw_dataframe=dataframe,
        dataframe=display_dataframe,
        table_structure=table_structure,
        data_cell_presentation=data_cell_presentation,
        best_value_cells=ranked_value_highlights.best_value_cells,
        second_best_value_cells=ranked_value_highlights.second_best_value_cells,
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


def resolve_bundle_targets(
    *,
    cli_bundle_dir: str | None,
    cli_bundle_parent_dir: str | None,
    render_spec: RenderSpec,
) -> tuple[str, list[Path]]:
    """Resolve either one bundle or a batch of direct child bundles."""
    if cli_bundle_dir is not None and cli_bundle_parent_dir is not None:
        raise AnalysisError("Provide exactly one of '--bundle-dir' or '--bundle-parent-dir', not both.")

    if cli_bundle_parent_dir is not None:
        bundle_parent_dir = resolve_existing_directory(
            cli_bundle_parent_dir,
            label="bundle parent directory",
        )
        ensure_path_within(bundle_parent_dir, RESULTS_ROOT, label="bundle parent directory")
        bundle_dirs = discover_bundle_dirs(bundle_parent_dir)
        if not bundle_dirs:
            raise AnalysisError(
                f"No valid bundle directories were found directly under '{bundle_parent_dir}'. "
                "Expected child directories containing both 'table.csv' and 'meta.json'."
            )
        return "batch", bundle_dirs

    if cli_bundle_dir is not None:
        bundle_dir = resolve_existing_directory(cli_bundle_dir, label="bundle directory")
        ensure_path_within(bundle_dir, RESULTS_ROOT, label="bundle directory")
        validate_bundle_dir(bundle_dir)
        return "single", [bundle_dir]

    if render_spec.input_bundle_dir is not None:
        ensure_path_within(render_spec.input_bundle_dir, RESULTS_ROOT, label="bundle directory")
        validate_bundle_dir(render_spec.input_bundle_dir)
        return "single", [render_spec.input_bundle_dir]

    raise AnalysisError(
        "Provide '--bundle-dir' or '--bundle-parent-dir'. "
        "Legacy fallback: the render config may also set 'input_dir' for single-bundle rendering."
    )


def discover_bundle_dirs(bundle_parent_dir: Path) -> list[Path]:
    """Return valid direct child bundle directories in deterministic order."""
    child_directories = sorted(
        (child_path for child_path in bundle_parent_dir.iterdir() if child_path.is_dir()),
        key=lambda child_path: child_path.name,
    )
    return [child_path for child_path in child_directories if is_valid_bundle_dir(child_path)]


def is_valid_bundle_dir(bundle_dir: Path) -> bool:
    """Return whether one directory looks like a renderable bundle."""
    return (bundle_dir / "table.csv").is_file() and (bundle_dir / "meta.json").is_file()


def validate_bundle_dir(bundle_dir: Path) -> None:
    """Require one directory to contain the files needed for rendering."""
    if not bundle_dir.is_dir():
        raise AnalysisError(f"The bundle directory path is not a directory: '{bundle_dir}'.")
    require_file(bundle_dir / "table.csv", label="bundle table")
    require_file(bundle_dir / "meta.json", label="bundle metadata")


def render_bundle(*, bundle_dir: Path, render_spec: RenderSpec) -> Path:
    """Render exactly one bundle directory using the shared rendering path."""
    try:
        validate_bundle_dir(bundle_dir)
        table_path = bundle_dir / "table.csv"
        meta_path = bundle_dir / "meta.json"
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
        table_dataframe, table_structure = apply_dimension_value_orders(
            dataframe=table_dataframe,
            table_structure=table_structure,
            dimension_value_orders=render_spec.table.dimension_value_orders,
        )
        data_cell_presentation = build_data_cell_presentation(
            dataframe=table_dataframe,
            table_structure=table_structure,
            meta_payload=meta_payload,
        )
        title_text = resolve_title(template=render_spec.title.template, meta_payload=meta_payload)

        output_path = bundle_dir / "render.png"
        render_png(
            dataframe=table_dataframe,
            table_structure=table_structure,
            data_cell_presentation=data_cell_presentation,
            title_text=title_text,
            render_spec=render_spec,
            output_path=output_path,
        )
        return output_path
    except AnalysisError as exc:
        raise AnalysisError(f"Could not render bundle '{bundle_dir}': {exc}") from exc


def draw_structured_table(
    *,
    ax: Any,
    raw_dataframe: pd.DataFrame,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    data_cell_presentation: DataCellPresentation,
    best_value_cells: set[tuple[int, int]],
    second_best_value_cells: set[tuple[int, int]],
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
    leaf_column_fill_colors = [
        resolve_leaf_axis_fill_color(
            axis_levels=table_structure.col_levels,
            axis_tuple=column_tuple,
            scope_colors=render_spec.table.scope_colors,
        )
        for column_tuple in table_structure.column_tuples
    ]

    ax.set_xlim(0.0, column_boundaries[-1])
    ax.set_ylim(float(total_row_count), 0.0)

    header_label_row_index = header_row_count - 1
    for header_row_index in range(header_row_count):
        for stub_column_index, row_level_name in enumerate(table_structure.row_levels):
            header_text = ""
            if header_row_index == header_label_row_index:
                header_text = resolve_label_alias(
                    label=row_level_name,
                    display_alias=render_spec.table.display_alias,
                )
            draw_cell_block(
                ax=ax,
                x0=column_boundaries[stub_column_index],
                x1=column_boundaries[stub_column_index + 1],
                y0=float(header_row_index),
                y1=float(header_row_index + 1),
                text=header_text,
                font_weight="bold",
                facecolor=None,
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
                    level_name=table_structure.col_levels[level_index],
                    column_tuple=table_structure.column_tuples[start_index],
                    value_column_name=table_structure.value_column_names[start_index],
                    level_index=level_index,
                    level_count=header_row_count,
                    display_alias=render_spec.table.display_alias,
                ),
                font_weight="bold",
                facecolor=resolve_dimension_fill_color(
                    dimension_name=table_structure.col_levels[level_index],
                    value=table_structure.column_tuples[start_index][level_index],
                    scope_colors=render_spec.table.scope_colors,
                ),
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
                text=resolve_dimension_value_alias(
                    dimension_name=table_structure.row_levels[level_index],
                    value=raw_dataframe.iloc[start_index][table_structure.row_column_names[level_index]],
                    display_alias=render_spec.table.display_alias,
                    fallback_value_alias=render_spec.table.value_alias.get(
                        table_structure.row_column_names[level_index],
                        {},
                    ),
                ),
                font_weight="normal",
                facecolor=resolve_dimension_fill_color(
                    dimension_name=table_structure.row_levels[level_index],
                    value=raw_dataframe.iloc[start_index][table_structure.row_column_names[level_index]],
                    scope_colors=render_spec.table.scope_colors,
                ),
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
                font_weight="bold" if (row_index, leaf_column_index) in best_value_cells else "normal",
                underline_text=(row_index, leaf_column_index) in second_best_value_cells,
                facecolor=resolve_data_cell_fill_color(
                    table_structure=table_structure,
                    row_tuple=table_structure.row_tuples[row_index],
                    column_tuple=table_structure.column_tuples[leaf_column_index],
                    raw_value=raw_dataframe.iloc[row_index][value_column_name],
                    display_mode=data_cell_presentation.display_modes[row_index][leaf_column_index],
                    signed_percent_heatmap_scale=data_cell_presentation.signed_percent_scales[row_index][leaf_column_index],
                    leaf_column_fill_color=leaf_column_fill_colors[leaf_column_index],
                    scope_colors=render_spec.table.scope_colors,
                ),
                render_spec=render_spec,
                total_table_width=column_boundaries[-1],
                total_row_count=total_row_count,
            )

    if render_spec.table.top_level_group_separators:
        draw_top_level_group_separators(
            ax=ax,
            column_boundaries=column_boundaries,
            table_structure=table_structure,
            header_row_count=header_row_count,
            total_row_count=total_row_count,
            render_spec=render_spec,
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
    facecolor: str | None,
    render_spec: RenderSpec,
    total_table_width: float,
    total_row_count: int,
    underline_text: bool = False,
) -> None:
    """Draw one rectangular cell or merged block and its text."""
    edge_color = "black" if render_spec.table.show_grid else render_spec.figure.background_color
    line_width = 0.5 if render_spec.table.show_grid else 0.0

    rectangle = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=facecolor or render_spec.figure.background_color,
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
    if underline_text:
        draw_cell_underline(
            ax=ax,
            rectangle=rectangle,
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            text=text,
            align=render_spec.table.cell_align,
            color=render_spec.table.text_color,
        )


def draw_cell_underline(
    *,
    ax: Any,
    rectangle: Rectangle,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    text: str,
    align: str,
    color: str,
) -> None:
    """Draw a visible underline under one cell's text without relying on font support."""
    stripped_text = text.strip()
    if not stripped_text:
        return

    cell_width = x1 - x0
    underline_width = cell_width * min(
        SECOND_BEST_UNDERLINE_MAX_FRACTION,
        max(
            SECOND_BEST_UNDERLINE_MIN_FRACTION,
            0.12 + (len(stripped_text) * SECOND_BEST_UNDERLINE_PER_CHAR_FRACTION),
        ),
    )
    text_anchor_x = resolve_text_x(x0=x0, x1=x1, align=align)
    padding = cell_width * CELL_PADDING_FRACTION
    if align == "left":
        line_start = text_anchor_x
        line_end = min(x1 - padding, line_start + underline_width)
    elif align == "center":
        line_start = text_anchor_x - (underline_width / 2.0)
        line_end = text_anchor_x + (underline_width / 2.0)
    elif align == "right":
        line_end = text_anchor_x
        line_start = max(x0 + padding, line_end - underline_width)
    else:
        raise AnalysisError(f"Unsupported alignment '{align}'.")

    underline_y = y0 + ((y1 - y0) * SECOND_BEST_UNDERLINE_Y_FRACTION)
    underline_artist = ax.plot(
        [line_start, line_end],
        [underline_y, underline_y],
        color=color,
        linewidth=SECOND_BEST_UNDERLINE_LINE_WIDTH,
        solid_capstyle="butt",
        clip_on=True,
    )[0]
    underline_artist.set_clip_path(rectangle)


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
    level_name: str,
    column_tuple: tuple[Any, ...],
    value_column_name: str,
    level_index: int,
    level_count: int,
    display_alias: DisplayAliasSpec,
) -> str:
    """Resolve the text for one column-header block."""
    if level_count == 1 and value_column_name in display_alias.label_alias:
        return display_alias.label_alias[value_column_name]
    return resolve_dimension_value_alias(
        dimension_name=level_name,
        value=column_tuple[level_index],
        display_alias=display_alias,
        fallback_value_alias={},
    )


def resolve_label_alias(*, label: str, display_alias: DisplayAliasSpec) -> str:
    """Resolve one display-only alias for a structural label name."""
    return display_alias.label_alias.get(label, label)


def resolve_dimension_value_alias(
    *,
    dimension_name: str,
    value: Any,
    display_alias: DisplayAliasSpec,
    fallback_value_alias: Mapping[str, str],
) -> str:
    """Resolve one display alias for a structural dimension value."""
    lookup_key = stringify_alias_lookup_value(value)
    dimension_aliases = display_alias.value_alias.get(dimension_name, {})
    if lookup_key in dimension_aliases:
        return dimension_aliases[lookup_key]
    if lookup_key in fallback_value_alias:
        return fallback_value_alias[lookup_key]
    stringified_value = stringify_header_value(value)
    if stringified_value in fallback_value_alias:
        return fallback_value_alias[stringified_value]
    return stringified_value


def resolve_dimension_fill_color(
    *,
    dimension_name: str,
    value: Any,
    scope_colors: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Return one configured semantic fill color for a dimension value."""
    lookup_key = stringify_alias_lookup_value(value)
    dimension_colors = scope_colors.get(dimension_name, {})
    return dimension_colors.get(lookup_key)


def resolve_leaf_axis_fill_color(
    *,
    axis_levels: list[str],
    axis_tuple: tuple[Any, ...],
    scope_colors: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Return the most specific configured fill color for one leaf axis tuple."""
    for dimension_name, value in reversed(list(zip(axis_levels, axis_tuple))):
        fill_color = resolve_dimension_fill_color(
            dimension_name=dimension_name,
            value=value,
            scope_colors=scope_colors,
        )
        if fill_color is not None:
            return fill_color
    return None


def resolve_data_cell_fill_color(
    *,
    table_structure: TableStructure,
    row_tuple: tuple[Any, ...],
    column_tuple: tuple[Any, ...],
    raw_value: Any,
    display_mode: str,
    signed_percent_heatmap_scale: SignedPercentHeatmapScale | None,
    leaf_column_fill_color: str | None,
    scope_colors: Mapping[str, Mapping[str, str]],
) -> str | None:
    """Resolve one semantic fill color for a data cell."""
    if display_mode == SIGNED_PERCENT_VALUE_DISPLAY_MODE:
        heatmap_color = resolve_signed_percent_heatmap_color(
            value=raw_value,
            signed_percent_heatmap_scale=signed_percent_heatmap_scale,
        )
        if heatmap_color is not None:
            return heatmap_color
    if leaf_column_fill_color is not None:
        return leaf_column_fill_color
    return resolve_leaf_axis_fill_color(
        axis_levels=table_structure.row_levels,
        axis_tuple=row_tuple,
        scope_colors=scope_colors,
    )


def resolve_signed_percent_heatmap_color(
    *,
    value: Any,
    signed_percent_heatmap_scale: SignedPercentHeatmapScale | None,
) -> str | None:
    """Map one signed GT-relative percentage onto a light-to-dark diverging color."""
    numeric_value = coerce_numeric_value(value)
    if numeric_value is None:
        return None
    if are_close(numeric_value, 0.0):
        return RELATIVE_HEATMAP_NEUTRAL_COLOR
    if signed_percent_heatmap_scale is None:
        return RELATIVE_HEATMAP_NEUTRAL_COLOR

    if numeric_value > 0.0:
        signed_magnitude_abs_max = signed_percent_heatmap_scale.positive_abs_max
        target_color = RELATIVE_HEATMAP_POSITIVE_COLOR
    else:
        signed_magnitude_abs_max = signed_percent_heatmap_scale.negative_abs_max
        target_color = RELATIVE_HEATMAP_NEGATIVE_COLOR

    if signed_magnitude_abs_max is None or are_close(signed_magnitude_abs_max, 0.0):
        return RELATIVE_HEATMAP_NEUTRAL_COLOR

    magnitude = min(abs(numeric_value) / signed_magnitude_abs_max, 1.0)
    return interpolate_hex_color(
        RELATIVE_HEATMAP_NEUTRAL_COLOR,
        target_color,
        magnitude,
    )


def interpolate_hex_color(start_color: str, end_color: str, blend: float) -> str:
    """Blend two colors in RGB space and return a hex string."""
    clamped_blend = max(0.0, min(1.0, float(blend)))
    start_rgb = mcolors.to_rgb(start_color)
    end_rgb = mcolors.to_rgb(end_color)
    blended_rgb = tuple(
        start_component + ((end_component - start_component) * clamped_blend)
        for start_component, end_component in zip(start_rgb, end_rgb, strict=True)
    )
    return mcolors.to_hex(blended_rgb)


def draw_top_level_group_separators(
    *,
    ax: Any,
    column_boundaries: list[float],
    table_structure: TableStructure,
    header_row_count: int,
    total_row_count: int,
    render_spec: RenderSpec,
) -> None:
    """Draw slightly stronger separators between top-level groups."""
    separator_color = "black" if render_spec.table.show_grid else render_spec.table.text_color
    stub_column_count = len(table_structure.row_levels)
    total_width = column_boundaries[-1]

    top_level_column_spans = iterate_hierarchy_spans(table_structure.column_tuples, level_index=0)
    for _, end_index in top_level_column_spans[:-1]:
        x_position = column_boundaries[stub_column_count + end_index]
        ax.plot(
            [x_position, x_position],
            [0.0, float(total_row_count)],
            color=separator_color,
            linewidth=GROUP_SEPARATOR_LINE_WIDTH,
            solid_capstyle="butt",
        )

    top_level_row_spans = iterate_hierarchy_spans(table_structure.row_tuples, level_index=0)
    for _, end_index in top_level_row_spans[:-1]:
        y_position = float(header_row_count + end_index)
        ax.plot(
            [0.0, total_width],
            [y_position, y_position],
            color=separator_color,
            linewidth=GROUP_SEPARATOR_LINE_WIDTH,
            solid_capstyle="butt",
        )


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
    table_structure: TableStructure,
    data_cell_presentation: DataCellPresentation,
    signed_percent_round_digits: int | None = None,
) -> pd.DataFrame:
    """Convert a dataframe into display strings for slide rendering."""
    validate_value_alias_columns(dataframe=dataframe, value_alias=value_alias)
    validate_data_cell_presentation(
        dataframe=dataframe,
        table_structure=table_structure,
        data_cell_presentation=data_cell_presentation,
    )

    formatted_columns: dict[str, pd.Series] = {}
    value_column_indexes = {
        column_name: index
        for index, column_name in enumerate(table_structure.value_column_names)
    }
    for column_name in dataframe.columns:
        normalized_column_name = str(column_name)
        is_identifier_column = normalized_column_name in identifier_columns
        column_value_alias = value_alias.get(normalized_column_name, {})
        if is_identifier_column:
            formatted_columns[normalized_column_name] = dataframe[column_name].map(
                lambda value: format_cell_value(
                    value,
                    is_identifier_column=True,
                    round_digits=round_digits,
                    signed_percent_round_digits=signed_percent_round_digits,
                    value_alias=column_value_alias,
                    display_mode=ABSOLUTE_VALUE_DISPLAY_MODE,
                )
            )
            continue

        leaf_column_index = value_column_indexes[normalized_column_name]
        formatted_columns[normalized_column_name] = pd.Series(
            [
                format_cell_value(
                    dataframe.iloc[row_index][normalized_column_name],
                    is_identifier_column=False,
                    round_digits=round_digits,
                    signed_percent_round_digits=signed_percent_round_digits,
                    value_alias=column_value_alias,
                    display_mode=data_cell_presentation.display_modes[row_index][leaf_column_index],
                )
                for row_index in range(len(dataframe))
            ],
            index=dataframe.index,
        )
    return pd.DataFrame(formatted_columns)


def validate_data_cell_presentation(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    data_cell_presentation: DataCellPresentation,
) -> None:
    """Require the resolved per-cell presentation matrix to match table.csv dimensions."""
    if len(data_cell_presentation.display_modes) != len(dataframe):
        raise AnalysisError(
            "Resolved data-cell presentation rows do not match the rendered dataframe row count."
        )
    if len(data_cell_presentation.signed_percent_scales) != len(dataframe):
        raise AnalysisError(
            "Resolved GT heatmap scale rows do not match the rendered dataframe row count."
        )
    expected_column_count = len(table_structure.value_column_names)
    for row_index, row_modes in enumerate(data_cell_presentation.display_modes):
        if len(row_modes) != expected_column_count:
            raise AnalysisError(
                "Resolved data-cell presentation columns do not match the rendered value-column "
                f"count at row {row_index}."
            )
        for display_mode in row_modes:
            validate_display_mode(display_mode)
    for row_index, row_scales in enumerate(data_cell_presentation.signed_percent_scales):
        if len(row_scales) != expected_column_count:
            raise AnalysisError(
                "Resolved GT heatmap scale columns do not match the rendered value-column "
                f"count at row {row_index}."
            )
 

def build_data_cell_presentation(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    meta_payload: Mapping[str, Any],
) -> DataCellPresentation:
    """Resolve one per-cell display-mode matrix plus GT-relative heatmap scaling."""
    display_modes: list[list[str]] = []
    signed_percent_scales: list[list[SignedPercentHeatmapScale | None]] = []
    positive_abs_max_by_group: dict[tuple[tuple[Any, ...], int], float] = {}
    negative_abs_max_by_group: dict[tuple[tuple[Any, ...], int], float] = {}

    for row_index, row_tuple in enumerate(table_structure.row_tuples):
        comparison_group_key = build_signed_percent_comparison_group_key(
            row_tuple=row_tuple,
            table_structure=table_structure,
        )
        for leaf_column_index, column_tuple in enumerate(table_structure.column_tuples):
            display_mode = resolve_data_cell_display_mode(
                table_structure=table_structure,
                row_tuple=row_tuple,
                column_tuple=column_tuple,
                meta_payload=meta_payload,
            )
            if display_mode != SIGNED_PERCENT_VALUE_DISPLAY_MODE:
                continue

            numeric_value = coerce_numeric_value(
                dataframe.iloc[row_index][table_structure.value_column_names[leaf_column_index]]
            )
            if numeric_value is None or are_close(numeric_value, 0.0):
                continue

            scale_group_key = (comparison_group_key, leaf_column_index)
            if numeric_value > 0.0:
                previous_positive_abs_max = positive_abs_max_by_group.get(scale_group_key, 0.0)
                positive_abs_max_by_group[scale_group_key] = max(
                    previous_positive_abs_max,
                    abs(numeric_value),
                )
            else:
                previous_negative_abs_max = negative_abs_max_by_group.get(scale_group_key, 0.0)
                negative_abs_max_by_group[scale_group_key] = max(
                    previous_negative_abs_max,
                    abs(numeric_value),
                )

    for row_index, row_tuple in enumerate(table_structure.row_tuples):
        row_display_modes: list[str] = []
        row_signed_percent_scales: list[SignedPercentHeatmapScale | None] = []
        comparison_group_key = build_signed_percent_comparison_group_key(
            row_tuple=row_tuple,
            table_structure=table_structure,
        )
        for leaf_column_index, column_tuple in enumerate(table_structure.column_tuples):
            display_mode = resolve_data_cell_display_mode(
                table_structure=table_structure,
                row_tuple=row_tuple,
                column_tuple=column_tuple,
                meta_payload=meta_payload,
            )
            row_display_modes.append(display_mode)
            if display_mode == SIGNED_PERCENT_VALUE_DISPLAY_MODE:
                scale_group_key = (comparison_group_key, leaf_column_index)
                row_signed_percent_scales.append(
                    SignedPercentHeatmapScale(
                        positive_abs_max=positive_abs_max_by_group.get(scale_group_key),
                        negative_abs_max=negative_abs_max_by_group.get(scale_group_key),
                    )
                )
            else:
                row_signed_percent_scales.append(None)
        display_modes.append(row_display_modes)
        signed_percent_scales.append(row_signed_percent_scales)

    return DataCellPresentation(
        display_modes=display_modes,
        signed_percent_scales=signed_percent_scales,
    )


def build_signed_percent_comparison_group_key(
    *,
    row_tuple: tuple[Any, ...],
    table_structure: TableStructure,
) -> tuple[Any, ...]:
    """Group GT heatmap scaling by sibling leaf rows within the same row hierarchy branch."""
    if len(table_structure.row_levels) <= 1:
        return tuple()
    return row_tuple[:-1]


def resolve_data_cell_display_mode(
    *,
    table_structure: TableStructure,
    row_tuple: tuple[Any, ...],
    column_tuple: tuple[Any, ...],
    meta_payload: Mapping[str, Any],
) -> str:
    """Resolve whether one data cell should render as an absolute value or signed percentage."""
    ground_truth_relative_to_clean_config = resolve_ground_truth_relative_to_clean_config(meta_payload)
    if ground_truth_relative_to_clean_config is None:
        return ABSOLUTE_VALUE_DISPLAY_MODE

    metric_scope = resolve_dimension_value_from_cell_or_meta(
        dimension_name=METRIC_SCOPE_COLUMN,
        table_structure=table_structure,
        row_tuple=row_tuple,
        column_tuple=column_tuple,
        meta_payload=meta_payload,
    )
    if metric_scope is None:
        raise AnalysisError(
            "The bundle enables ground_truth_relative_to_clean, but the renderer could not resolve "
            "metric_scope for at least one cell. Keep metric_scope in rows/cols/split/filters or "
            "ensure it is uniquely inferable from bundle metadata."
        )
    if stringify_alias_lookup_value(metric_scope) != GROUND_TRUTH_SCOPE:
        return ABSOLUTE_VALUE_DISPLAY_MODE

    attack_method = resolve_dimension_value_from_cell_or_meta(
        dimension_name="attack_method",
        table_structure=table_structure,
        row_tuple=row_tuple,
        column_tuple=column_tuple,
        meta_payload=meta_payload,
    )
    if attack_method is None:
        raise AnalysisError(
            "The bundle enables ground_truth_relative_to_clean, but the renderer could not resolve "
            "attack_method for at least one GT cell. Keep attack_method in rows/cols/split/filters "
            "or rebuild the view so clean GT raw values do not mix with attack deltas."
        )
    if (
        stringify_alias_lookup_value(attack_method)
        == ground_truth_relative_to_clean_config["baseline_attack_method"]
    ):
        return ABSOLUTE_VALUE_DISPLAY_MODE
    return SIGNED_PERCENT_VALUE_DISPLAY_MODE


def resolve_ground_truth_relative_to_clean_config(
    meta_payload: Mapping[str, Any],
) -> dict[str, str] | None:
    """Return validated GT-relative rendering config from bundle metadata when enabled."""
    raw_config = meta_payload.get("ground_truth_relative_to_clean")
    if not isinstance(raw_config, Mapping):
        return None
    raw_enabled = raw_config.get("enabled")
    if raw_enabled is not None and not bool(raw_enabled):
        return None

    baseline_attack_method = raw_config.get("baseline_attack_method")
    if not isinstance(baseline_attack_method, str) or not baseline_attack_method.strip():
        raise AnalysisError(
            "The bundle metadata ground_truth_relative_to_clean block must contain a non-empty "
            "'baseline_attack_method'."
        )
    return {"baseline_attack_method": baseline_attack_method.strip()}


def resolve_dimension_value_from_cell_or_meta(
    *,
    dimension_name: str,
    table_structure: TableStructure,
    row_tuple: tuple[Any, ...],
    column_tuple: tuple[Any, ...],
    meta_payload: Mapping[str, Any],
) -> Any | None:
    """Resolve one dimension value from row/column tuples or bundle metadata fallbacks."""
    if dimension_name in table_structure.row_levels:
        return row_tuple[table_structure.row_levels.index(dimension_name)]
    if dimension_name in table_structure.col_levels:
        return column_tuple[table_structure.col_levels.index(dimension_name)]

    for metadata_key in ("effective_filters", "split_values", "context", "filters"):
        metadata_value = meta_payload.get(metadata_key)
        if not isinstance(metadata_value, Mapping):
            continue
        if dimension_name in metadata_value:
            return metadata_value[dimension_name]
    return None


def validate_display_alias_targets(
    *,
    table_structure: TableStructure,
    display_alias: DisplayAliasSpec,
) -> None:
    """Require every display alias target to exist in the rendered structure."""
    allowed_label_targets = set(table_structure.row_levels)
    allowed_label_targets.update(table_structure.col_levels)
    allowed_label_targets.update(table_structure.row_column_names)
    allowed_label_targets.update(table_structure.value_column_names)
    missing_label_targets = sorted(
        label_key for label_key in display_alias.label_alias if label_key not in allowed_label_targets
    )
    if missing_label_targets:
        raise AnalysisError(
            f"The render config display_alias label targets {missing_label_targets} do not exist in the "
            "current table structure."
        )

    allowed_value_dimensions = set(table_structure.row_levels)
    allowed_value_dimensions.update(table_structure.col_levels)
    allowed_value_dimensions.update(table_structure.row_column_names)
    missing_value_dimensions = sorted(
        dimension_name
        for dimension_name in display_alias.value_alias
        if dimension_name not in allowed_value_dimensions
    )
    if missing_value_dimensions:
        raise AnalysisError(
            f"The render config display_alias value targets {missing_value_dimensions} do not exist in the "
            "current table structure."
        )


def validate_scope_color_targets(
    *,
    table_structure: TableStructure,
    scope_colors: Mapping[str, Mapping[str, str]],
) -> None:
    """Require every semantic-color target dimension to exist in the rendered structure."""
    allowed_dimensions = set(table_structure.row_levels)
    allowed_dimensions.update(table_structure.col_levels)
    missing_dimensions = sorted(
        dimension_name for dimension_name in scope_colors if dimension_name not in allowed_dimensions
    )
    if missing_dimensions:
        raise AnalysisError(
            f"The render config scope_colors targets {missing_dimensions} do not exist in the current "
            "table structure."
        )


def resolve_ranked_value_highlights(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    best_value_bolding: BestValueBoldingSpec | None,
) -> RankedValueCellHighlights:
    """Return the ranked data-cell highlights configured for one rendered table."""
    if best_value_bolding is None:
        return RankedValueCellHighlights(best_value_cells=set(), second_best_value_cells=set())

    if best_value_bolding.compare_along != "rows":
        raise AnalysisError(
            f"Unsupported best_value_bolding.compare_along '{best_value_bolding.compare_along}'."
        )
    if best_value_bolding.mode != "max":
        raise AnalysisError(f"Unsupported best_value_bolding.mode '{best_value_bolding.mode}'.")

    row_groups = resolve_partition_row_groups(
        table_structure=table_structure,
        partition_by_levels=best_value_bolding.partition_by_levels,
    )
    best_cells: set[tuple[int, int]] = set()
    second_best_cells: set[tuple[int, int]] = set()
    for leaf_column_index, value_column_name in enumerate(table_structure.value_column_names):
        numeric_values = pd.to_numeric(dataframe[value_column_name], errors="coerce")
        numeric_value_list = numeric_values.tolist()
        for row_group in row_groups:
            valid_group_entries = [
                (row_index, float(numeric_value_list[row_index]))
                for row_index in row_group
                if not pd.isna(numeric_value_list[row_index])
            ]
            if not valid_group_entries:
                continue

            distinct_ranked_values = resolve_distinct_ranked_values(
                [numeric_value for _, numeric_value in valid_group_entries]
            )
            best_value = distinct_ranked_values[0]
            second_best_value = distinct_ranked_values[1] if len(distinct_ranked_values) > 1 else None

            for row_index, numeric_value in valid_group_entries:
                if are_close(numeric_value, best_value):
                    best_cells.add((row_index, leaf_column_index))
                    continue
                if (
                    best_value_bolding.underline_second_best
                    and second_best_value is not None
                    and are_close(numeric_value, second_best_value)
                ):
                    second_best_cells.add((row_index, leaf_column_index))
    return RankedValueCellHighlights(
        best_value_cells=best_cells,
        second_best_value_cells=second_best_cells,
    )


def resolve_best_value_cells(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    best_value_bolding: BestValueBoldingSpec | None,
) -> set[tuple[int, int]]:
    """Return the set of data-cell coordinates that should be bolded."""
    return resolve_ranked_value_highlights(
        dataframe=dataframe,
        table_structure=table_structure,
        best_value_bolding=best_value_bolding,
    ).best_value_cells


def resolve_distinct_ranked_values(values: list[float]) -> list[float]:
    """Return descending distinct values with float-noise collapse."""
    distinct_values: list[float] = []
    for value in sorted(values, reverse=True):
        if distinct_values and are_close(value, distinct_values[-1]):
            continue
        distinct_values.append(value)
    return distinct_values


def resolve_partition_row_groups(
    *,
    table_structure: TableStructure,
    partition_by_levels: list[str],
) -> list[list[int]]:
    """Partition row indices by one or more row hierarchy levels while preserving display order."""
    if not partition_by_levels:
        return [list(range(len(table_structure.row_tuples)))]

    validate_partition_levels(
        partition_by_levels=partition_by_levels,
        available_levels=table_structure.row_levels,
        label="best_value_bolding.partition_by_levels",
    )
    level_indexes = [table_structure.row_levels.index(level_name) for level_name in partition_by_levels]

    grouped_row_indexes: dict[tuple[Any, ...], list[int]] = {}
    for row_index, row_tuple in enumerate(table_structure.row_tuples):
        group_key = tuple(row_tuple[level_index] for level_index in level_indexes)
        grouped_row_indexes.setdefault(group_key, []).append(row_index)
    return list(grouped_row_indexes.values())


def validate_partition_levels(
    *,
    partition_by_levels: list[str],
    available_levels: list[str],
    label: str,
) -> None:
    """Require partition levels to be unique and present on the compared axis."""
    seen_levels: set[str] = set()
    duplicate_levels: list[str] = []
    for level_name in partition_by_levels:
        if level_name in seen_levels and level_name not in duplicate_levels:
            duplicate_levels.append(level_name)
        seen_levels.add(level_name)
    if duplicate_levels:
        raise AnalysisError(
            f"Duplicate {label} values are not allowed: {sorted(duplicate_levels)}."
        )

    missing_levels = sorted(level_name for level_name in partition_by_levels if level_name not in available_levels)
    if missing_levels:
        raise AnalysisError(
            f"The render config {label} targets {missing_levels} do not exist on the current axis. "
            f"Available levels: {available_levels}."
        )


def are_close(left: float, right: float) -> bool:
    """Compare two floats with a small tolerance for formatting noise."""
    tolerance = 1e-12 * max(1.0, abs(left), abs(right))
    return abs(left - right) <= tolerance


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
    display_mode: str,
    signed_percent_round_digits: int | None = None,
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
        numeric_value = float(normalized_value)
        if are_close(numeric_value, 0.0):
            numeric_value = 0.0
        if display_mode == SIGNED_PERCENT_VALUE_DISPLAY_MODE:
            effective_signed_percent_round_digits = (
                round_digits
                if signed_percent_round_digits is None
                else signed_percent_round_digits
            )
            formatted_value = f"{numeric_value:+.{effective_signed_percent_round_digits}f}%"
        else:
            formatted_value = f"{numeric_value:.{round_digits}f}"
    else:
        formatted_value = str(normalized_value)

    return value_alias.get(raw_lookup_key, value_alias.get(formatted_value, formatted_value))


def validate_display_mode(display_mode: str) -> None:
    """Require one supported value-display mode."""
    if display_mode not in {ABSOLUTE_VALUE_DISPLAY_MODE, SIGNED_PERCENT_VALUE_DISPLAY_MODE}:
        raise AnalysisError(f"Unsupported data-cell display mode '{display_mode}'.")


def coerce_numeric_value(value: Any) -> float | None:
    """Convert one scalar cell value into a float when possible."""
    normalized_value = normalize_scalar(value)
    if normalized_value is None or pd.isna(normalized_value) or isinstance(normalized_value, bool):
        return None
    if isinstance(normalized_value, Real):
        return float(normalized_value)
    return None


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


def apply_dimension_value_orders(
    *,
    dataframe: pd.DataFrame,
    table_structure: TableStructure,
    dimension_value_orders: Mapping[str, list[Any]],
) -> tuple[pd.DataFrame, TableStructure]:
    """Apply render-time row/column value ordering without mutating the view bundle."""
    if not dimension_value_orders:
        return dataframe, table_structure

    validate_dimension_value_order_targets(
        table_structure=table_structure,
        dimension_value_orders=dimension_value_orders,
    )

    ordered_dataframe = dataframe.copy()
    ordered_table_structure = table_structure

    row_indexes = reorder_axis_indexes(
        axis_tuples=ordered_table_structure.row_tuples,
        axis_levels=ordered_table_structure.row_levels,
        dimension_value_orders=dimension_value_orders,
    )
    if row_indexes != list(range(len(ordered_table_structure.row_tuples))):
        ordered_dataframe = ordered_dataframe.iloc[row_indexes].reset_index(drop=True)
        ordered_table_structure = replace(
            ordered_table_structure,
            row_tuples=[ordered_table_structure.row_tuples[index] for index in row_indexes],
        )

    column_indexes = reorder_axis_indexes(
        axis_tuples=ordered_table_structure.column_tuples,
        axis_levels=ordered_table_structure.col_levels,
        dimension_value_orders=dimension_value_orders,
    )
    if column_indexes != list(range(len(ordered_table_structure.column_tuples))):
        ordered_value_column_names = [
            ordered_table_structure.value_column_names[index] for index in column_indexes
        ]
        ordered_dataframe = ordered_dataframe.loc[
            :,
            ordered_table_structure.row_column_names + ordered_value_column_names,
        ]
        ordered_table_structure = replace(
            ordered_table_structure,
            column_tuples=[ordered_table_structure.column_tuples[index] for index in column_indexes],
            value_column_names=ordered_value_column_names,
        )

    return ordered_dataframe, ordered_table_structure


def validate_dimension_value_order_targets(
    *,
    table_structure: TableStructure,
    dimension_value_orders: Mapping[str, list[Any]],
) -> None:
    """Require render-time ordering rules to target existing row/column dimensions."""
    allowed_dimensions = set(table_structure.row_levels)
    allowed_dimensions.update(table_structure.col_levels)
    missing_dimensions = sorted(
        dimension_name
        for dimension_name in dimension_value_orders
        if dimension_name not in allowed_dimensions
    )
    if missing_dimensions:
        raise AnalysisError(
            f"The render config dimension_value_orders targets {missing_dimensions} do not exist in "
            "the current table structure."
        )


def reorder_axis_indexes(
    *,
    axis_tuples: list[tuple[Any, ...]],
    axis_levels: list[str],
    dimension_value_orders: Mapping[str, list[Any]],
) -> list[int]:
    """Return render-ordered indexes for one axis while preserving hierarchy groups."""
    if not axis_tuples:
        return []
    return reorder_axis_indexes_for_level(
        candidate_indexes=list(range(len(axis_tuples))),
        axis_tuples=axis_tuples,
        axis_levels=axis_levels,
        dimension_value_orders=dimension_value_orders,
        level_index=0,
    )


def reorder_axis_indexes_for_level(
    *,
    candidate_indexes: list[int],
    axis_tuples: list[tuple[Any, ...]],
    axis_levels: list[str],
    dimension_value_orders: Mapping[str, list[Any]],
    level_index: int,
) -> list[int]:
    """Recursively reorder contiguous hierarchy groups for one axis level at a time."""
    if level_index >= len(axis_levels) or len(candidate_indexes) <= 1:
        return list(candidate_indexes)

    grouped_indexes = group_axis_indexes_by_level_value(
        candidate_indexes=candidate_indexes,
        axis_tuples=axis_tuples,
        level_index=level_index,
    )
    grouped_indexes = sort_axis_index_groups(
        grouped_indexes=grouped_indexes,
        dimension_name=axis_levels[level_index],
        dimension_value_orders=dimension_value_orders,
    )

    ordered_indexes: list[int] = []
    for _, child_indexes in grouped_indexes:
        ordered_indexes.extend(
            reorder_axis_indexes_for_level(
                candidate_indexes=child_indexes,
                axis_tuples=axis_tuples,
                axis_levels=axis_levels,
                dimension_value_orders=dimension_value_orders,
                level_index=level_index + 1,
            )
        )
    return ordered_indexes


def group_axis_indexes_by_level_value(
    *,
    candidate_indexes: list[int],
    axis_tuples: list[tuple[Any, ...]],
    level_index: int,
) -> list[tuple[Any, list[int]]]:
    """Group contiguous axis indexes by one hierarchy level value."""
    grouped_indexes: list[tuple[Any, list[int]]] = []
    for index in candidate_indexes:
        level_value = axis_tuples[index][level_index]
        if grouped_indexes and grouped_indexes[-1][0] == level_value:
            grouped_indexes[-1][1].append(index)
            continue
        grouped_indexes.append((level_value, [index]))
    return grouped_indexes


def sort_axis_index_groups(
    *,
    grouped_indexes: list[tuple[Any, list[int]]],
    dimension_name: str,
    dimension_value_orders: Mapping[str, list[Any]],
) -> list[tuple[Any, list[int]]]:
    """Sort sibling hierarchy groups by one optional render-time preferred value order."""
    preferred_values = dimension_value_orders.get(dimension_name)
    if not preferred_values:
        return grouped_indexes

    order_lookup = {
        stringify_alias_lookup_value(value): index for index, value in enumerate(preferred_values)
    }
    sortable_groups = list(enumerate(grouped_indexes))
    sortable_groups.sort(
        key=lambda item: build_dimension_group_sort_key(
            value=item[1][0],
            original_position=item[0],
            order_lookup=order_lookup,
        )
    )
    return [group for _, group in sortable_groups]


def build_dimension_group_sort_key(
    *,
    value: Any,
    original_position: int,
    order_lookup: Mapping[str, int],
) -> tuple[int, int]:
    """Sort specified values first and leave unspecified values in their existing relative order."""
    lookup_key = stringify_alias_lookup_value(value)
    if lookup_key in order_lookup:
        return (0, order_lookup[lookup_key])
    return (1, original_position)


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
    """Resolve the title template using merged meta.json title context."""
    render_context = build_title_render_context(meta_payload)

    formatter = string.Formatter()
    required_fields: list[str] = []
    for _literal_text, field_name, _format_spec, _conversion in formatter.parse(template):
        if field_name is None:
            continue
        if not field_name or any(symbol in field_name for symbol in ".[]"):
            raise AnalysisError(
                f"Unsupported title template field '{field_name}'. Use simple keys from merged meta.json title context only."
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


def build_title_render_context(meta_payload: Mapping[str, Any]) -> dict[str, str]:
    """Merge title context from meta.json context, effective_filters, and split_values."""
    render_context = build_render_context(
        require_mapping(meta_payload.get("context"), label="meta.json context")
    )
    merge_render_context(
        render_context,
        meta_payload.get("slice_context"),
        label="meta.json slice_context",
    )
    merge_render_context(
        render_context,
        meta_payload.get("effective_filters"),
        label="meta.json effective_filters",
    )
    merge_render_context(
        render_context,
        meta_payload.get("split_values"),
        label="meta.json split_values",
    )
    return render_context


def build_render_context(context_payload: Mapping[str, Any]) -> dict[str, str]:
    """Convert meta.json context values into title-template strings."""
    render_context: dict[str, str] = {}
    for key, value in context_payload.items():
        render_context[require_nonempty_string(key, label="meta.json context key")] = stringify_context_value(value)
    return render_context


def merge_render_context(
    render_context: dict[str, str],
    payload: Any,
    *,
    label: str,
) -> None:
    """Add missing title-template keys from one optional metadata mapping."""
    if payload is None:
        return
    for key, value in build_render_context(require_mapping(payload, label=label)).items():
        render_context.setdefault(key, value)


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


def normalize_display_alias_spec(value: Any, *, label: str) -> DisplayAliasSpec:
    """Normalize display aliases for structural labels and values."""
    if value is None:
        return DisplayAliasSpec(label_alias={}, value_alias={})

    mapping = require_mapping(value, label=label)
    label_alias: dict[str, str] = {}
    value_alias: dict[str, dict[str, str]] = {}

    for raw_key, raw_value in mapping.items():
        key = require_nonempty_string(raw_key, label=f"{label} key")
        if key == "labels":
            nested_labels = normalize_string_mapping(raw_value, label=f"{label}.labels")
            label_alias.update(nested_labels)
            continue
        if key == "values":
            nested_value_alias = normalize_dimension_value_string_mapping(
                raw_value,
                label=f"{label}.values",
            )
            for dimension_name, alias_map in nested_value_alias.items():
                value_alias.setdefault(dimension_name, {}).update(alias_map)
            continue
        if isinstance(raw_value, Mapping):
            value_alias[key] = normalize_one_alias_map(
                raw_value,
                label=f"{label}[{key}]",
            )
            continue
        label_alias[key] = require_nonempty_string(raw_value, label=f"{label}[{key}]")

    return DisplayAliasSpec(label_alias=label_alias, value_alias=value_alias)


def normalize_dimension_value_string_mapping(value: Any, *, label: str) -> dict[str, dict[str, str]]:
    """Normalize a mapping from dimension names to per-value strings."""
    if value is None:
        return {}

    mapping = require_mapping(value, label=label)
    normalized: dict[str, dict[str, str]] = {}
    for raw_dimension_name, raw_alias_map in mapping.items():
        dimension_name = require_nonempty_string(raw_dimension_name, label=f"{label} dimension")
        normalized[dimension_name] = normalize_one_alias_map(
            raw_alias_map,
            label=f"{label}[{dimension_name}]",
        )
    return normalized


def normalize_one_alias_map(value: Any, *, label: str) -> dict[str, str]:
    """Normalize one per-dimension alias map."""
    mapping = require_mapping(value, label=label)
    normalized: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        alias_key = stringify_alias_lookup_value(raw_key)
        normalized[alias_key] = require_nonempty_string(raw_value, label=f"{label}[{alias_key}]")
    return normalized


def normalize_best_value_bolding_spec(value: Any, *, label: str) -> BestValueBoldingSpec | None:
    """Normalize optional best-value bolding configuration."""
    if value is None:
        return None

    payload = require_mapping(value, label=label)
    compare_along = require_nonempty_string(
        payload.get("compare_along"),
        label=f"{label}.compare_along",
    ).lower()
    if compare_along not in ALLOWED_COMPARE_ALONG:
        raise AnalysisError(
            f"Unsupported {label}.compare_along '{compare_along}'. Allowed values: {sorted(ALLOWED_COMPARE_ALONG)}."
        )

    mode = require_nonempty_string(
        payload.get("mode", "max"),
        label=f"{label}.mode",
    ).lower()
    if mode not in ALLOWED_BEST_VALUE_MODES:
        raise AnalysisError(
            f"Unsupported {label}.mode '{mode}'. Allowed values: {sorted(ALLOWED_BEST_VALUE_MODES)}."
        )

    partition_by_levels = normalize_optional_string_list(
        payload.get("partition_by_levels"),
        label=f"{label}.partition_by_levels",
    )
    underline_second_best = normalize_second_best_underline_flag(payload=payload, label=label)

    return BestValueBoldingSpec(
        compare_along=compare_along,
        mode=mode,
        partition_by_levels=partition_by_levels,
        underline_second_best=underline_second_best,
    )


def normalize_second_best_underline_flag(*, payload: Mapping[str, Any], label: str) -> bool:
    """Normalize the second-best emphasis flag and accept the italic key as a fallback."""
    has_underline_flag = "underline_second_best" in payload
    has_italic_flag = "italicize_second_best" in payload
    if has_underline_flag and has_italic_flag:
        raise AnalysisError(
            f"Provide only one of {label}.underline_second_best or "
            f"{label}.italicize_second_best."
        )
    if has_underline_flag:
        return require_bool(
            payload.get("underline_second_best"),
            label=f"{label}.underline_second_best",
        )
    if has_italic_flag:
        return require_bool(
            payload.get("italicize_second_best"),
            label=f"{label}.italicize_second_best",
        )
    return False


def normalize_dimension_value_orders(value: Any, *, label: str) -> dict[str, list[Any]]:
    """Normalize optional render-time preferred value orders for row/column dimensions."""
    if value is None:
        return {}

    mapping = require_mapping(value, label=label)
    normalized: dict[str, list[Any]] = {}
    for raw_dimension_name, raw_values in mapping.items():
        dimension_name = require_nonempty_string(raw_dimension_name, label=f"{label} dimension")
        normalized[dimension_name] = normalize_scalar_order_list(
            raw_values,
            label=f"{label}[{dimension_name}]",
        )
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


def normalize_scalar_order_list(value: Any, *, label: str) -> list[Any]:
    """Normalize one ordered list of scalar values and reject duplicates."""
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"Expected '{label}' to be a non-empty list of scalar values.")

    normalized: list[Any] = []
    seen_keys: set[str] = set()
    duplicate_keys: list[str] = []
    for index, raw_item in enumerate(value):
        normalized_item = normalize_order_scalar(raw_item, label=f"{label}[{index}]")
        lookup_key = stringify_alias_lookup_value(normalized_item)
        if lookup_key in seen_keys and lookup_key not in duplicate_keys:
            duplicate_keys.append(lookup_key)
        seen_keys.add(lookup_key)
        normalized.append(normalized_item)

    if duplicate_keys:
        raise AnalysisError(f"Duplicate values are not allowed in '{label}': {sorted(duplicate_keys)}.")
    return normalized


def normalize_order_scalar(value: Any, *, label: str) -> Any:
    """Normalize one scalar that may appear in a preferred render order list."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, str):
        return require_nonempty_string(normalized_value, label=label)
    if isinstance(normalized_value, (bool, Integral, Real)):
        return normalized_value
    raise AnalysisError(
        f"Expected '{label}' to be a scalar string/number/bool, got {type(normalized_value).__name__}."
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


def normalize_optional_string_list(value: Any, *, label: str) -> list[str]:
    """Normalize an optional list of strings."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise AnalysisError(f"Expected '{label}' to be a list of strings.")

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


def normalize_optional_nonnegative_int(value: Any, *, label: str) -> int | None:
    """Normalize one optional non-negative integer value."""
    if value is None:
        return None
    return require_nonnegative_int(value, label=label)


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
