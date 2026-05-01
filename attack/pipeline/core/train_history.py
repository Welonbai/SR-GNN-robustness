from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import json


def save_train_history(
    path: str | Path,
    *,
    role: str,
    model: str,
    epochs: int,
    train_loss: Sequence[float | None],
    valid_loss: Sequence[float | None],
    notes: str | None = None,
    extra: Mapping[str, object] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "role": role,
        "model": model,
        "epochs": int(epochs),
        "train_loss": [None if v is None else float(v) for v in train_loss],
        "valid_loss": [None if v is None else float(v) for v in valid_loss],
    }
    if notes:
        payload["notes"] = notes
    if extra:
        collision_keys = sorted(set(payload).intersection(extra))
        if collision_keys:
            raise ValueError(
                "Train history extra fields must not overwrite base fields: "
                + ", ".join(collision_keys)
            )
        payload.update(dict(extra))
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


__all__ = ["save_train_history"]
