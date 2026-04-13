"""Helpers for generating lightweight inventory files for long tables."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import pandas as pd


DEFAULT_LISTED_COLUMNS_EXCLUDE = {"value"}


def build_inventory(
    dataframe: pd.DataFrame,
    *,
    exclude_unique_values_for: set[str] | None = None,
) -> dict[str, Any]:
    """Build a discovery-friendly inventory for one long-table dataframe."""
    excluded_columns = exclude_unique_values_for or DEFAULT_LISTED_COLUMNS_EXCLUDE
    unique_counts = {
        str(column): int(dataframe[column].nunique(dropna=True))
        for column in dataframe.columns
    }
    unique_values = {
        str(column): collect_unique_values(dataframe[column])
        for column in dataframe.columns
        if str(column) not in excluded_columns
    }

    return {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": [str(column) for column in dataframe.columns],
        "listed_unique_value_columns": [str(column) for column in dataframe.columns if str(column) not in excluded_columns],
        "unique_counts": unique_counts,
        "unique_values": unique_values,
    }


def collect_unique_values(series: pd.Series) -> list[Any]:
    """Collect sorted unique non-null values for one dataframe column."""
    normalized_values = [
        normalize_scalar(value)
        for value in series.dropna().unique().tolist()
    ]
    return sorted(normalized_values, key=inventory_sort_key)


def inventory_sort_key(value: Any) -> tuple[int, Any]:
    """Return a deterministic sort key across common scalar types."""
    normalized_value = normalize_scalar(value)
    if isinstance(normalized_value, bool):
        return (0, int(normalized_value))
    if isinstance(normalized_value, Integral):
        return (1, int(normalized_value))
    if isinstance(normalized_value, Real):
        return (2, float(normalized_value))
    return (3, str(normalized_value).casefold())


def normalize_scalar(value: Any) -> Any:
    """Convert pandas and numpy scalar types into plain Python primitives."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            return value
    return value
