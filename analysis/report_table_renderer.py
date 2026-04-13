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
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "results"
SUPPORTED_OUTPUT_FORMATS = {"png"}
ALLOWED_ALIGNMENTS = {"left", "center", "right"}


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


@dataclass(frozen=True)
class RenderSpec:
    """Validated render YAML content."""

    style_name: str
    output_format: str
    title: TitleSpec
    figure: FigureSpec
    table: TableSpec


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 3 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Render one report-table bundle into a PNG image.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="Path to one view bundle directory containing table.csv and meta.json.",
    )
    parser.add_argument(
        "--spec",
        required=True,
        help="Path to one render YAML spec.",
    )
    return parser


def main() -> None:
    """Run the renderer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        bundle_dir = resolve_existing_directory(args.bundle_dir, label="bundle directory")
        ensure_path_within(bundle_dir, RESULTS_ROOT, label="bundle directory")

        table_path = require_file(bundle_dir / "table.csv", label="bundle table")
        meta_path = require_file(bundle_dir / "meta.json", label="bundle metadata")
        spec_path = resolve_existing_file(args.spec, label="render spec")

        render_spec = parse_render_spec(load_yaml_mapping(spec_path, label="render spec"))
        table_dataframe = load_table_csv(table_path)
        meta_payload = load_json_mapping(meta_path, label="bundle metadata")
        identifier_columns = extract_identifier_columns(meta_payload=meta_payload, dataframe=table_dataframe)
        title_text = resolve_title(template=render_spec.title.template, meta_payload=meta_payload)

        output_path = bundle_dir / "render.png"
        render_png(
            dataframe=table_dataframe,
            identifier_columns=identifier_columns,
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
    )

    return RenderSpec(
        style_name=style_name,
        output_format=output_format,
        title=title_spec,
        figure=figure_spec,
        table=table_spec,
    )


def render_png(
    *,
    dataframe: pd.DataFrame,
    identifier_columns: set[str],
    title_text: str,
    render_spec: RenderSpec,
    output_path: Path,
) -> None:
    """Render one report table into a PNG image."""
    display_dataframe = format_dataframe_for_display(
        dataframe,
        identifier_columns=identifier_columns,
        round_digits=render_spec.table.round_digits,
    )

    fig, ax = plt.subplots(
        figsize=(render_spec.figure.width, render_spec.figure.height),
        dpi=render_spec.figure.dpi,
    )
    fig.patch.set_facecolor(render_spec.figure.background_color)
    ax.set_facecolor(render_spec.figure.background_color)
    ax.axis("off")
    ax.set_position([0.01, 0.04, 0.98, 0.76])

    table = ax.table(
        cellText=display_dataframe.values.tolist(),
        colLabels=[str(column) for column in display_dataframe.columns],
        loc="center",
        cellLoc=render_spec.table.cell_align,
        colLoc=render_spec.table.cell_align,
        bbox=[0.0, 0.0, 1.0, 1.0],
    )

    if render_spec.table.auto_shrink:
        table.auto_set_font_size(True)
    else:
        table.auto_set_font_size(False)
        table.set_fontsize(render_spec.table.font_size)

    table.scale(1.0, 1.3)
    style_table_cells(
        table=table,
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


def style_table_cells(*, table: Any, render_spec: RenderSpec) -> None:
    """Apply consistent visual styling to every table cell."""
    edge_color = "black" if render_spec.table.show_grid else render_spec.figure.background_color
    line_width = 0.5 if render_spec.table.show_grid else 0.0

    for (row_index, _column_index), cell in table.get_celld().items():
        cell.set_facecolor(render_spec.figure.background_color)
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(line_width)

        text = cell.get_text()
        text.set_color(render_spec.table.text_color)
        text.set_wrap(render_spec.table.wrap_text)
        text.set_ha(render_spec.table.cell_align)
        text.set_va("center")
        if not render_spec.table.auto_shrink:
            text.set_fontsize(render_spec.table.font_size)
        if row_index == 0:
            text.set_weight("bold")


def format_dataframe_for_display(
    dataframe: pd.DataFrame,
    *,
    identifier_columns: set[str],
    round_digits: int,
) -> pd.DataFrame:
    """Convert a dataframe into display strings for slide rendering."""
    formatted_columns: dict[str, pd.Series] = {}
    for column_name in dataframe.columns:
        is_identifier_column = str(column_name) in identifier_columns
        formatted_columns[str(column_name)] = dataframe[column_name].map(
            lambda value: format_cell_value(
                value,
                is_identifier_column=is_identifier_column,
                round_digits=round_digits,
            )
        )
    return pd.DataFrame(formatted_columns)


def format_cell_value(value: Any, *, is_identifier_column: bool, round_digits: int) -> str:
    """Format one cell value for display."""
    if pd.isna(value):
        return ""

    normalized_value = normalize_scalar(value)
    if is_identifier_column:
        return str(normalized_value)
    if isinstance(normalized_value, bool):
        return "True" if normalized_value else "False"
    if isinstance(normalized_value, Integral):
        return str(int(normalized_value))
    if isinstance(normalized_value, Real):
        return f"{float(normalized_value):.{round_digits}f}"
    return str(normalized_value)


def extract_identifier_columns(*, meta_payload: Mapping[str, Any], dataframe: pd.DataFrame) -> set[str]:
    """Read identifier columns from meta.json rows and validate that they exist in table.csv."""
    rows_value = meta_payload.get("rows")
    if not isinstance(rows_value, list) or not rows_value:
        raise AnalysisError("The bundle metadata must contain a non-empty 'rows' list for rendering.")

    identifier_columns: set[str] = set()
    missing_columns: list[str] = []
    for index, raw_name in enumerate(rows_value):
        column_name = require_nonempty_string(raw_name, label=f"meta.json rows[{index}]")
        identifier_columns.add(column_name)
        if column_name not in dataframe.columns:
            missing_columns.append(column_name)

    if missing_columns:
        raise AnalysisError(
            f"The bundle metadata rows {missing_columns} do not exist in table.csv columns {list(dataframe.columns)}."
        )
    return identifier_columns


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


def require_nonempty_string(value: Any, *, label: str) -> str:
    """Require a non-empty string value."""
    if not isinstance(value, str):
        raise AnalysisError(f"Expected '{label}' to be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise AnalysisError(f"Expected '{label}' to be a non-empty string.")
    return stripped


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
