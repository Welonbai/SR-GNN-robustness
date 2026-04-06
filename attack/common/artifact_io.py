from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


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
    target_item: int,
    target_selection_mode: str,
    seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_item": int(target_item),
        "target_selection_mode": target_selection_mode,
        "seed": int(seed),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_target_info(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "save_poison_model",
    "load_poison_model",
    "save_fake_sessions",
    "load_fake_sessions",
    "save_target_info",
    "load_target_info",
]
