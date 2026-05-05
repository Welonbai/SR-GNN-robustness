from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TailReplacementResult:
    session: list[int]
    replacement_position: int
    original_item: int
    was_noop: bool


class TailReplacementPolicy:
    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> TailReplacementResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        if len(session) == 1:
            raise ValueError(
                "Tail-Replacement-NZ requires session length >= 2; "
                "length-1 tail replacement would select position 0."
            )
        updated = [int(item) for item in session]
        replacement_position = len(updated) - 1
        original_item = int(updated[replacement_position])
        updated[replacement_position] = int(target_item)
        return TailReplacementResult(
            session=updated,
            replacement_position=int(replacement_position),
            original_item=original_item,
            was_noop=bool(original_item == int(target_item)),
        )


__all__ = ["TailReplacementPolicy", "TailReplacementResult"]
