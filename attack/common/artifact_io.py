from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_poison_model(runner, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.save_model(path)


def load_poison_model(runner, path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    runner.load_model(path)
    return True


def save_fake_sessions(sessions: list[list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(sessions, handle)


def load_fake_sessions(path: str | Path) -> list[list[int]] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_target_info(
    path: str | Path,
    *,
    target_items: list[int],
    target_selection_mode: str,
    seed: int,
    bucket: str | None = None,
    count: int | None = None,
    explicit_list: list[int] | None = None,
) -> None:
    target_items = [int(item) for item in target_items]
    payload = {
        "target_items": target_items,
        "target_selection_mode": target_selection_mode,
        "seed": int(seed),
        "bucket": bucket,
        "count": int(count) if count is not None else None,
        "explicit_list": [int(item) for item in (explicit_list or [])],
    }
    if target_items:
        payload["target_item"] = int(target_items[0])
    save_json(payload, path)


def load_target_info(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("target_info.json must contain a JSON object.")
    return payload


def save_selected_targets(path: str | Path, target_items: list[int]) -> None:
    save_json({"target_items": [int(item) for item in target_items]}, path)


def load_selected_targets(path: str | Path) -> list[int] | None:
    payload = load_json(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("selected_targets.json must contain a JSON object.")
    target_items = payload.get("target_items")
    if not isinstance(target_items, list):
        raise ValueError("selected_targets.json is missing target_items.")
    return [int(item) for item in target_items]


def save_target_selection_meta(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(payload, path)


def load_target_selection_meta(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("target_selection_meta.json must contain a JSON object.")
    return payload


__all__ = [
    "save_json",
    "load_json",
    "save_poison_model",
    "load_poison_model",
    "save_fake_sessions",
    "load_fake_sessions",
    "save_target_info",
    "load_target_info",
    "save_selected_targets",
    "load_selected_targets",
    "save_target_selection_meta",
    "load_target_selection_meta",
]
