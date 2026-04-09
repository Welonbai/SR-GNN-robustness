from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import time


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_path: Path
    delimiter: str
    parse_event_date: Callable[[dict[str, str]], float]
    extract_session_item: Callable[[dict[str, str]], tuple[str, str, int]]


def _parse_diginetica_event_date(row: dict[str, str]) -> float:
    return time.mktime(time.strptime(row["eventdate"], "%Y-%m-%d"))


def _extract_diginetica_session_item(row: dict[str, str]) -> tuple[str, str, int]:
    session_id = row.get("sessionId", row.get("session_id"))
    item_id = row.get("itemId", row.get("item_id"))
    sort_key = int(row.get("timeframe", "0"))
    return str(session_id), str(item_id), sort_key


def resolve_dataset_spec(dataset_name: str, datasets_root: Path) -> DatasetSpec:
    name = dataset_name.lower()
    if name == "diginetica":
        raw_path = datasets_root / "diginetica" / "train-item-views.csv"
        return DatasetSpec(
            name="diginetica",
            raw_path=raw_path,
            delimiter=";",
            parse_event_date=_parse_diginetica_event_date,
            extract_session_item=_extract_diginetica_session_item,
        )
    raise NotImplementedError(
        f"Dataset '{dataset_name}' is not supported yet. "
        "Add a DatasetSpec in resolve_dataset_spec()."
    )


__all__ = ["DatasetSpec", "resolve_dataset_spec"]
