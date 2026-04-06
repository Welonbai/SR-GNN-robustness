from __future__ import annotations

from typing import Sequence

from .base_policy import InsertionPolicy


class AlwaysPositionZeroPolicy(InsertionPolicy):
    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        if not session:
            raise ValueError("Session must contain at least one item.")
        updated = list(session)
        updated[0] = int(target_item)
        return updated


__all__ = ["AlwaysPositionZeroPolicy"]
