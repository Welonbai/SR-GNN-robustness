from __future__ import annotations

from typing import Sequence


def replace_item_at_position(
    session: Sequence[int],
    position: int,
    target_item: int,
) -> list[int]:
    if not session:
        raise ValueError("session must contain at least one item.")
    if position < 0 or position >= len(session):
        raise IndexError("position is outside the session range.")
    if int(target_item) <= 0:
        raise ValueError("target_item must be a positive item id.")

    updated = list(session)
    updated[int(position)] = int(target_item)
    return updated


__all__ = ["replace_item_at_position"]
