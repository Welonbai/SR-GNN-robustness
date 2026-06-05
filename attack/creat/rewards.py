from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CreatRewardComponents:
    attack_reward: float
    stealth_reward: float
    local_reward: float
    total_reward: float


def compute_reward_components(
    adapter,
    *,
    original_session: Sequence[int],
    polluted_session: Sequence[int],
    selected_position: int,
    target_item: int,
    stealth_weight: float,
    local_weight: float,
) -> CreatRewardComponents:
    position = int(selected_position)
    attack_reward = (
        0.0
        if position <= 0
        else float(adapter.target_score_for_prefix(original_session[:position], target_item))
    )
    with torch.no_grad():
        original_rep = adapter.encode_session(original_session)
        polluted_rep = adapter.encode_session(polluted_session)
        stealth_reward = -float(torch.linalg.vector_norm(original_rep - polluted_rep).item())
        local_reward = _local_compatibility_reward(
            adapter,
            original_session=original_session,
            selected_position=position,
            target_item=target_item,
        )
    total = (
        float(attack_reward)
        + float(stealth_weight) * float(stealth_reward)
        + float(local_weight) * float(local_reward)
    )
    return CreatRewardComponents(
        attack_reward=float(attack_reward),
        stealth_reward=float(stealth_reward),
        local_reward=float(local_reward),
        total_reward=float(total),
    )


def _local_compatibility_reward(
    adapter,
    *,
    original_session: Sequence[int],
    selected_position: int,
    target_item: int,
) -> float:
    if selected_position < 0:
        return 0.0
    neighbors = []
    if selected_position - 1 >= 0:
        neighbors.append(int(original_session[selected_position - 1]))
    if selected_position + 1 < len(original_session):
        neighbors.append(int(original_session[selected_position + 1]))
    if not neighbors:
        return 0.0
    target_embedding = adapter.target_embedding(target_item)
    neighbor_embeddings = adapter.item_embeddings(neighbors)
    target = target_embedding.view(1, -1).expand_as(neighbor_embeddings)
    similarities = F.cosine_similarity(target, neighbor_embeddings, dim=1)
    return float(similarities.mean().item())


__all__ = ["CreatRewardComponents", "compute_reward_components"]
