from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from attack.creat.candidates import (
    position_distribution,
    target_exposure_counts,
    target_label_pair_count,
)
from attack.data.poisoned_dataset_builder import expand_session_to_samples


@dataclass(frozen=True)
class CreatPoisonBuildResult:
    poisoned_sessions: list[list[int]]
    selected_positions: list[int]
    metadata: dict[str, object]


def build_creat_poisoned_sessions(
    *,
    adapter,
    masker,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    replacement_topk_ratio: float,
    nonzero_when_possible: bool,
    max_item_id: int,
) -> CreatPoisonBuildResult:
    target = int(target_item)
    if target <= 0:
        raise ValueError("target_item must be positive.")
    if target > int(max_item_id):
        raise ValueError("target_item exceeds dataset max item id.")

    source_sessions = [list(session) for session in template_sessions]
    if any(len(session) == 0 for session in source_sessions):
        raise ValueError("Template sessions must be non-empty.")
    if any(max(int(item) for item in session) > int(max_item_id) for session in source_sessions):
        raise ValueError("Template sessions contain item ids above dataset max item id.")

    poisoned: list[list[int]] = []
    selected_positions: list[int] = []
    masker.eval()
    with torch.no_grad():
        for session in source_sessions:
            session_rep = adapter.encode_session(session)
            item_embeddings = adapter.item_embeddings(session)
            valid_mask = adapter.valid_position_mask(
                session,
                target,
                replacement_topk_ratio,
                nonzero_when_possible=bool(nonzero_when_possible),
            )
            logits = masker(session_rep, item_embeddings, valid_mask)
            position = int(torch.argmax(logits).item())
            if not bool(valid_mask[position].item()):
                raise RuntimeError("CREAT masker selected an invalid position.")
            if (
                bool(nonzero_when_possible)
                and len(session) > 1
                and any(bool(item) for item in valid_mask[1:].tolist())
                and position == 0
            ):
                raise RuntimeError("CREAT selected position 0 despite a nonzero candidate.")
            updated = list(session)
            updated[position] = target
            if updated[position] != target:
                raise RuntimeError("CREAT target replacement failed.")
            poisoned.append(updated)
            selected_positions.append(position)

    if len(poisoned) != len(source_sessions):
        raise RuntimeError("CREAT output count does not match template count.")
    expanded_pair_count = sum(
        len(expand_session_to_samples(session)[0]) for session in poisoned
    )
    post_exposure = target_exposure_counts(poisoned, target_item=target)
    selected_replacement_target_pair_count = target_label_pair_count(selected_positions)
    metadata = {
        "effective_poisoned_copied_session_count": int(len(poisoned)),
        "expanded_poisoned_prefix_label_pair_count": int(expanded_pair_count),
        "selected_replacement_target_pair_count": int(
            selected_replacement_target_pair_count
        ),
        "expanded_target_label_pair_count": int(
            post_exposure["target_label_pair_count"]
        ),
        "target_label_poisoned_pair_count": int(selected_replacement_target_pair_count),
        "selected_position_distribution": position_distribution(selected_positions),
    }
    return CreatPoisonBuildResult(
        poisoned_sessions=poisoned,
        selected_positions=selected_positions,
        metadata=metadata,
    )


__all__ = ["CreatPoisonBuildResult", "build_creat_poisoned_sessions"]
