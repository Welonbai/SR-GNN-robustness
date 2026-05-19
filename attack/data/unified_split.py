from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import csv

from attack.common.config import Config
from attack.common.paths import canonical_split_paths, split_key as canonical_paths_split_key
from attack.data.canonical_dataset import (
    CanonicalDataset,
    canonical_dataset_exists,
    load_canonical_dataset,
    save_canonical_dataset,
)
from attack.data.dataset_specs import resolve_dataset_spec


_SECONDS_PER_DAY = 86400
_YOOCHOOSE_VARIANTS = {
    "yoochoose": ("full", None),
    "yoochoose1_64": ("1_64", 1.0 / 64.0),
    "yoochoose1_4": ("1_4", 1.0 / 4.0),
}


@dataclass(frozen=True)
class SplitConfig:
    min_item_count: int = 5
    min_session_len: int = 2
    valid_ratio: float = 0.1
    test_days: int = 7


def _split_key(config: Config, split_config: SplitConfig) -> str:
    if split_config == split_config_from_config(config):
        return canonical_paths_split_key(config)
    ratio_token = f"{split_config.valid_ratio:.4f}".rstrip("0").rstrip(".")
    ratio_token = ratio_token.replace(".", "p")
    return (
        f"split_{config.data.dataset_name.lower()}"
        f"_{config.data.split_protocol}"
        f"_trainonly{int(bool(config.data.poison_train_only))}"
        f"_minitems{split_config.min_item_count}"
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
        if spec.has_header:
            reader = csv.DictReader(handle, delimiter=spec.delimiter)
        else:
            if not spec.fieldnames:
                raise ValueError(f"Dataset spec '{spec.name}' must define fieldnames.")
            reader = csv.DictReader(
                handle,
                fieldnames=spec.fieldnames,
                delimiter=spec.delimiter,
            )
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


def resolve_yoochoose_variant(dataset_name: str) -> tuple[str, float | None]:
    name = dataset_name.lower()
    if name not in _YOOCHOOSE_VARIANTS:
        raise ValueError(f"Dataset '{dataset_name}' is not a Yoochoose variant.")
    return _YOOCHOOSE_VARIANTS[name]


def _is_yoochoose_variant(dataset_name: str) -> bool:
    return dataset_name.lower() in _YOOCHOOSE_VARIANTS


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


def _expanded_pair_count(sessions: list[list[int]]) -> int:
    return sum(max(0, len(session) - 1) for session in sessions)


def _expanded_sample_session_ids(train_sessions: list[list[int]]) -> list[int]:
    ids: list[int] = []
    for session_idx, seq in enumerate(train_sessions):
        for _ in range(1, len(seq)):
            ids.append(session_idx)
    return ids


def _apply_recent_fraction_by_expanded_pairs(
    train_sessions: list[list[int]],
    *,
    fraction: float,
) -> list[list[int]]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("Yoochoose train tail fraction must be in (0, 1].")
    expanded_ids = _expanded_sample_session_ids(train_sessions)
    if not expanded_ids:
        return train_sessions
    keep_samples = max(1, int(len(expanded_ids) * fraction))
    start_session_idx = expanded_ids[-keep_samples]
    return train_sessions[start_session_idx:]


def _max_item_id(*session_groups: list[list[int]]) -> int:
    max_item = 0
    for sessions in session_groups:
        for session in sessions:
            if session:
                max_item = max(max_item, max(int(item) for item in session))
    return int(max_item)


def split_config_from_config(config: Config) -> SplitConfig:
    canonical_split = config.data.canonical_split
    return SplitConfig(
        min_item_count=int(canonical_split.min_item_count),
        min_session_len=int(canonical_split.min_session_len),
        valid_ratio=float(canonical_split.valid_ratio),
        test_days=int(canonical_split.test_days),
    )


def build_canonical_dataset(
    config: Config,
    *,
    split_config: SplitConfig | None = None,
    dataset_root: Path | None = None,
) -> CanonicalDataset:
    split_config = split_config or split_config_from_config(config)
    dataset_root = dataset_root or Path("datasets")
    spec = resolve_dataset_spec(config.data.dataset_name, dataset_root)
    raw_path = spec.raw_path

    sessions, session_dates = _load_raw_sessions(spec)
    raw_session_count = len(sessions)
    sessions, session_dates = _filter_sessions(
        sessions,
        session_dates,
        min_item_count=split_config.min_item_count,
        min_session_len=split_config.min_session_len,
    )
    filtered_session_count = len(sessions)
    train_ids, test_ids, split_date = _time_split_sessions(
        sessions,
        session_dates,
        test_days=split_config.test_days,
    )
    train_sessions, item_map = _map_sessions(train_ids, sessions, item_map=None)
    test_sessions, _ = _map_sessions(test_ids, sessions, item_map=item_map)
    train_sessions_before_variant = len(train_sessions)
    expanded_pairs_before_variant = _expanded_pair_count(train_sessions)

    source_dataset = spec.name
    variant = "full"
    train_tail_fraction: float | None = None
    if _is_yoochoose_variant(config.data.dataset_name):
        source_dataset = "yoochoose"
        variant, train_tail_fraction = resolve_yoochoose_variant(config.data.dataset_name)
        if train_tail_fraction is not None:
            train_sessions = _apply_recent_fraction_by_expanded_pairs(
                train_sessions,
                fraction=train_tail_fraction,
            )

    train_sessions_after_variant = len(train_sessions)
    expanded_pairs_after_variant = _expanded_pair_count(train_sessions)
    train_sub, valid = _split_train_valid(train_sessions, valid_ratio=split_config.valid_ratio)
    max_item_id = max(item_map.values(), default=0)

    metadata: dict[str, Any] = {
        "dataset_name": config.data.dataset_name,
        "source_dataset": source_dataset,
        "variant": variant,
        "train_tail_fraction": train_tail_fraction,
        "split_protocol": config.data.split_protocol,
        "split_key": _split_key(config, split_config),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "raw_path": str(raw_path),
        "raw_session_count": raw_session_count,
        "filtered_session_count": filtered_session_count,
        "train_sessions_before_variant": train_sessions_before_variant,
        "train_sessions_after_variant": train_sessions_after_variant,
        "expanded_pairs_before_variant": expanded_pairs_before_variant,
        "expanded_pairs_after_variant": expanded_pairs_after_variant,
        "max_item_id": max_item_id,
        "item_count": len(item_map),
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
            "max_item_id": max_item_id,
        },
        "counts_before_variant": {
            "train_sessions": train_sessions_before_variant,
            "train_pairs": expanded_pairs_before_variant,
        },
        "counts_after_variant": {
            "train_sessions": train_sessions_after_variant,
            "train_pairs": expanded_pairs_after_variant,
        },
        "observed": {
            "max_item_id": _max_item_id(train_sub, valid, test_sessions),
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
    split_config = split_config or split_config_from_config(config)
    split_key = _split_key(config, split_config)
    paths = canonical_split_paths(config, split_key=split_key)
    if not force_rebuild and canonical_dataset_exists(paths):
        print(f"[split] Found canonical dataset at {paths['canonical_dir']}")
        return load_canonical_dataset(paths)

    print(f"[split] Building canonical dataset for {config.data.dataset_name}")
    dataset = build_canonical_dataset(
        config,
        split_config=split_config,
        dataset_root=dataset_root,
    )
    save_canonical_dataset(dataset, paths)
    print(f"[split] Saved canonical dataset to {paths['canonical_dir']}")
    return dataset


__all__ = [
    "SplitConfig",
    "resolve_yoochoose_variant",
    "split_config_from_config",
    "build_canonical_dataset",
    "ensure_canonical_dataset",
]
