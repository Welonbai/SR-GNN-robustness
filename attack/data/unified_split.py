from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import csv

from attack.common.config import Config
from attack.common.paths import canonical_split_paths
from attack.data.canonical_dataset import (
    CanonicalDataset,
    canonical_dataset_exists,
    load_canonical_dataset,
    save_canonical_dataset,
)
from attack.data.dataset_specs import resolve_dataset_spec


_SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class SplitConfig:
    min_item_count: int = 5
    min_session_len: int = 2
    valid_ratio: float = 0.1
    test_days: int = 7


def _split_key(split_config: SplitConfig) -> str:
    ratio_token = f"{split_config.valid_ratio:.4f}".rstrip("0").rstrip(".")
    ratio_token = ratio_token.replace(".", "p")
    return (
        f"unified_minitems{split_config.min_item_count}"
        f"_minsess{split_config.min_session_len}"
        f"_testdays{split_config.test_days}"
        f"_valid{ratio_token}"
    )


def _load_raw_sessions(
    spec,
) -> tuple[dict[str, list[tuple[str, int]]], dict[str, float]]:
    raw_path = spec.raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {raw_path}")

    sess_clicks: dict[str, list[tuple[str, int]]] = {}
    sess_date: dict[str, float] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=spec.delimiter)
        for row in reader:
            session_id, item_id, sort_key = spec.extract_session_item(row)
            event_date = spec.parse_event_date(row)
            sess_clicks.setdefault(session_id, []).append((item_id, sort_key))
            current_date = sess_date.get(session_id, 0.0)
            if event_date > current_date:
                sess_date[session_id] = event_date

    sessions: dict[str, list[str]] = {}
    for session_id, clicks in sess_clicks.items():
        ordered = sorted(clicks, key=lambda x: x[1])
        sessions[session_id] = [item for item, _ in ordered]
    return sessions, sess_date


def _filter_sessions(
    sessions: dict[str, list[str]],
    session_dates: dict[str, float],
    *,
    min_item_count: int,
    min_session_len: int,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    filtered_sessions = {
        sid: seq for sid, seq in sessions.items() if len(seq) >= min_session_len
    }
    filtered_dates = {
        sid: session_dates[sid] for sid in filtered_sessions if sid in session_dates
    }

    item_counts: dict[str, int] = {}
    for seq in filtered_sessions.values():
        for item in seq:
            item_counts[item] = item_counts.get(item, 0) + 1

    retained_sessions: dict[str, list[str]] = {}
    for sid, seq in filtered_sessions.items():
        kept = [item for item in seq if item_counts.get(item, 0) >= min_item_count]
        if len(kept) >= min_session_len:
            retained_sessions[sid] = kept

    retained_dates = {sid: filtered_dates[sid] for sid in retained_sessions}
    return retained_sessions, retained_dates


def _time_split_sessions(
    sessions: dict[str, list[str]],
    session_dates: dict[str, float],
    *,
    test_days: int,
) -> tuple[list[str], list[str], float]:
    if not session_dates:
        raise ValueError("No sessions available after filtering.")
    max_date = max(session_dates.values())
    split_date = max_date - test_days * _SECONDS_PER_DAY
    train_ids = [sid for sid, date in session_dates.items() if date < split_date]
    test_ids = [sid for sid, date in session_dates.items() if date > split_date]
    train_ids.sort(key=lambda sid: session_dates[sid])
    test_ids.sort(key=lambda sid: session_dates[sid])
    return train_ids, test_ids, split_date


def _map_sessions(
    session_ids: list[str],
    sessions: dict[str, list[str]],
    item_map: dict[str, int] | None = None,
) -> tuple[list[list[int]], dict[str, int]]:
    mapping = item_map or {}
    next_id = max(mapping.values(), default=0) + 1
    mapped_sessions: list[list[int]] = []
    for sid in session_ids:
        mapped_seq = []
        for item in sessions[sid]:
            if item in mapping:
                mapped_seq.append(mapping[item])
            elif item_map is None:
                mapping[item] = next_id
                mapped_seq.append(next_id)
                next_id += 1
        if len(mapped_seq) >= 2:
            mapped_sessions.append(mapped_seq)
    return mapped_sessions, mapping


def _split_train_valid(
    train_sessions: list[list[int]],
    *,
    valid_ratio: float,
) -> tuple[list[list[int]], list[list[int]]]:
    if not train_sessions:
        raise ValueError("No training sessions available for split.")
    valid_count = max(1, int(round(len(train_sessions) * valid_ratio)))
    if len(train_sessions) <= valid_count:
        valid_count = max(1, len(train_sessions) - 1)
    if valid_count == 0:
        raise ValueError("Training set too small to create a valid split.")
    train_sub = train_sessions[:-valid_count]
    valid = train_sessions[-valid_count:]
    return train_sub, valid


def default_split_config(dataset_name: str) -> SplitConfig:
    name = dataset_name.lower()
    if name.startswith("yoochoose"):
        return SplitConfig(test_days=1)
    return SplitConfig(test_days=7)


def build_canonical_dataset(
    config: Config,
    *,
    split_config: SplitConfig | None = None,
    dataset_root: Path | None = None,
) -> CanonicalDataset:
    split_config = split_config or default_split_config(config.data.dataset_name)
    dataset_root = dataset_root or Path("datasets")
    spec = resolve_dataset_spec(config.data.dataset_name, dataset_root)
    raw_path = spec.raw_path

    sessions, session_dates = _load_raw_sessions(spec)
    sessions, session_dates = _filter_sessions(
        sessions,
        session_dates,
        min_item_count=split_config.min_item_count,
        min_session_len=split_config.min_session_len,
    )
    train_ids, test_ids, split_date = _time_split_sessions(
        sessions,
        session_dates,
        test_days=split_config.test_days,
    )
    train_sessions, item_map = _map_sessions(train_ids, sessions, item_map=None)
    test_sessions, _ = _map_sessions(test_ids, sessions, item_map=item_map)
    train_sub, valid = _split_train_valid(train_sessions, valid_ratio=split_config.valid_ratio)

    metadata: dict[str, Any] = {
        "dataset_name": config.data.dataset_name,
        "split_protocol": config.data.split_protocol,
        "split_key": _split_key(split_config),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "filtering": {
            "min_item_count": split_config.min_item_count,
            "min_session_len": split_config.min_session_len,
        },
        "time_split": {
            "test_days": split_config.test_days,
            "split_date_epoch": split_date,
        },
        "valid_split": {
            "valid_ratio": split_config.valid_ratio,
        },
        "counts": {
            "train_sub": len(train_sub),
            "valid": len(valid),
            "test": len(test_sessions),
            "items": len(item_map),
        },
    }

    return CanonicalDataset(
        train_sub=train_sub,
        valid=valid,
        test=test_sessions,
        item_map=item_map,
        metadata=metadata,
    )


def ensure_canonical_dataset(
    config: Config,
    *,
    split_config: SplitConfig | None = None,
    dataset_root: Path | None = None,
    force_rebuild: bool = False,
) -> CanonicalDataset:
    split_config = split_config or default_split_config(config.data.dataset_name)
    split_key = _split_key(split_config)
    paths = canonical_split_paths(config, split_key=split_key)
    if not force_rebuild and canonical_dataset_exists(paths):
        return load_canonical_dataset(paths)

    dataset = build_canonical_dataset(
        config,
        split_config=split_config,
        dataset_root=dataset_root,
    )
    save_canonical_dataset(dataset, paths)
    return dataset


__all__ = [
    "SplitConfig",
    "default_split_config",
    "build_canonical_dataset",
    "ensure_canonical_dataset",
]
