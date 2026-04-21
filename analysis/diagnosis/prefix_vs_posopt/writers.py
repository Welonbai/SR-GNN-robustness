#!/usr/bin/env python3
"""Writers for diagnosis tables, manifests, summaries, and Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> None:
    """Create one output directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe into JSON-safe records."""
    if dataframe.empty:
        return []
    return json.loads(dataframe.to_json(orient="records", double_precision=15))


def write_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write one dataframe to CSV."""
    ensure_directory(output_path.parent)
    dataframe.to_csv(output_path, index=False)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """Write one JSON file with stable formatting."""
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def dataframe_to_markdown(dataframe: pd.DataFrame, *, max_rows: int | None = None) -> str:
    """Render a small dataframe as a simple markdown table."""
    subset = dataframe if max_rows is None else dataframe.head(max_rows)
    if subset.empty:
        return "_No rows._"
    columns = [str(column) for column in subset.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for _, row in subset.iterrows():
        body_lines.append(
            "| "
            + " | ".join(_format_markdown_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join([header, separator, *body_lines])


def _format_markdown_value(value: Any) -> str:
    """Render one markdown table cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}"
    return str(value)


def write_report(markdown_text: str, output_path: Path) -> None:
    """Write one Markdown report."""
    ensure_directory(output_path.parent)
    output_path.write_text(markdown_text, encoding="utf-8")
