from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


DEFAULT_CROP_POLICY = "crop_tail_non_target"


@dataclass(frozen=True)
class RandomInsertionThenCropResult:
    session: list[int]
    insertion_slot: int
    inserted_session: list[int]
    crop_position: int
    cropped_item: int
    final_target_positions: list[int]
    target_position_after_crop: int | None
    original_length: int
    inserted_length: int
    final_length: int
    pre_existing_target_count: int
    target_occurrence_count_after_crop: int


class RandomInsertionThenCropNonzeroWhenPossiblePolicy:
    def __init__(
        self,
        topk_ratio: float,
        rng: random.Random | None = None,
        crop_policy: str = DEFAULT_CROP_POLICY,
    ) -> None:
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be within (0, 1].")
        if crop_policy != DEFAULT_CROP_POLICY:
            raise ValueError(
                f"Unsupported crop_policy {crop_policy!r}; expected {DEFAULT_CROP_POLICY!r}."
            )
        self.topk_ratio = float(topk_ratio)
        self.rng = rng or random.Random()
        self.crop_policy = crop_policy

    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        return self.apply_with_metadata(session, target_item).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
    ) -> RandomInsertionThenCropResult:
        if not session:
            raise ValueError("Session must contain at least one item.")
        target = int(target_item)
        original = [int(item) for item in session]
        original_length = len(original)
        pre_existing_target_count = sum(1 for item in original if item == target)

        topk_count = max(1, int(math.ceil(original_length * self.topk_ratio)))
        max_slot = min(topk_count, original_length)
        insertion_slot = int(self.rng.randint(1, max_slot))

        inserted_session = list(original)
        inserted_session.insert(insertion_slot, target)
        crop_position = _last_non_target_position(inserted_session, target)
        if crop_position is None:
            raise ValueError(
                "Random-Insertion-Then-Crop-NZ cannot crop because the inserted "
                "session contains no non-target item."
            )
        cropped_item = int(inserted_session[crop_position])
        if cropped_item == target:
            raise ValueError("Random-Insertion-Then-Crop-NZ attempted to crop target item.")

        final_session = list(inserted_session)
        final_session.pop(crop_position)
        if crop_position < insertion_slot:
            target_position_after_crop = insertion_slot - 1
        else:
            target_position_after_crop = insertion_slot
        final_target_positions = [
            index
            for index, item in enumerate(final_session)
            if int(item) == target
        ]

        if len(final_session) != original_length:
            raise ValueError("Final session length must equal original session length.")
        if target not in final_session:
            raise ValueError("Final session must contain target_item.")
        if target_position_after_crop < 0 or target_position_after_crop >= len(final_session):
            raise ValueError("Inserted target position after crop is out of range.")
        if final_session[target_position_after_crop] != target:
            raise ValueError("Inserted target was not preserved through crop.")
        if not _remaining_original_order_is_preserved(
            original=original,
            final_session=final_session,
            target_position_after_crop=target_position_after_crop,
        ):
            raise ValueError("Remaining original items did not preserve relative order.")

        return RandomInsertionThenCropResult(
            session=final_session,
            insertion_slot=insertion_slot,
            inserted_session=inserted_session,
            crop_position=int(crop_position),
            cropped_item=cropped_item,
            final_target_positions=[int(position) for position in final_target_positions],
            target_position_after_crop=int(target_position_after_crop),
            original_length=int(original_length),
            inserted_length=int(len(inserted_session)),
            final_length=int(len(final_session)),
            pre_existing_target_count=int(pre_existing_target_count),
            target_occurrence_count_after_crop=int(len(final_target_positions)),
        )


def _last_non_target_position(session: Sequence[int], target_item: int) -> int | None:
    target = int(target_item)
    for position in range(len(session) - 1, -1, -1):
        if int(session[position]) != target:
            return int(position)
    return None


def _remaining_original_order_is_preserved(
    *,
    original: Sequence[int],
    final_session: Sequence[int],
    target_position_after_crop: int,
) -> bool:
    final_without_inserted_target = [
        int(item)
        for index, item in enumerate(final_session)
        if index != int(target_position_after_crop)
    ]
    original_index = 0
    for item in final_without_inserted_target:
        while original_index < len(original) and int(original[original_index]) != int(item):
            original_index += 1
        if original_index >= len(original):
            return False
        original_index += 1
    return True


__all__ = [
    "DEFAULT_CROP_POLICY",
    "RandomInsertionThenCropNonzeroWhenPossiblePolicy",
    "RandomInsertionThenCropResult",
]
