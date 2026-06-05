from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch

from attack.common.config import CreatAdditiveSBRConfig
from attack.creat.candidates import position_distribution
from attack.creat.masker import CreatMasker
from attack.creat.rewards import compute_reward_components


@dataclass(frozen=True)
class CreatTrainingResult:
    masker: CreatMasker
    history: dict[str, object]


class CreatAdditiveSBRTrainer:
    def __init__(
        self,
        *,
        adapter,
        config: CreatAdditiveSBRConfig,
        replacement_topk_ratio: float,
        seed: int,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.replacement_topk_ratio = float(replacement_topk_ratio)
        self.seed = int(seed)

    def train(
        self,
        *,
        target_item: int,
        template_sessions: Sequence[Sequence[int]],
    ) -> CreatTrainingResult:
        sessions = [list(session) for session in template_sessions]
        if not sessions:
            raise ValueError("CREAT training requires at least one effective template.")
        max_session_length = max(len(session) for session in sessions)
        masker = CreatMasker(
            session_dim=self.adapter.embedding_dim,
            item_dim=self.adapter.embedding_dim,
            hidden_dim=int(self.config.hidden_dim),
            position_embedding_dim=int(self.config.position_embedding_dim),
            max_session_length=max_session_length,
        )
        optimizer = torch.optim.Adam(masker.parameters(), lr=float(self.config.lr))
        rng = random.Random(self.seed + int(target_item))
        torch.manual_seed(self.seed + int(target_item))

        epoch_rows: list[dict[str, object]] = []
        for epoch in range(int(self.config.epochs)):
            order = list(range(len(sessions)))
            rng.shuffle(order)
            losses: list[float] = []
            attack_rewards: list[float] = []
            stealth_rewards: list[float] = []
            local_rewards: list[float] = []
            total_rewards: list[float] = []
            entropies: list[float] = []
            selected_positions: list[int] = []

            for start in range(0, len(order), int(self.config.batch_size)):
                batch_indices = order[start : start + int(self.config.batch_size)]
                batch_log_probs = []
                batch_rewards = []
                batch_entropies = []
                for index in batch_indices:
                    session = sessions[index]
                    session_rep = self.adapter.encode_session(session)
                    item_embeddings = self.adapter.item_embeddings(session)
                    valid_mask = self.adapter.valid_position_mask(
                        session,
                        int(target_item),
                        self.replacement_topk_ratio,
                        nonzero_when_possible=bool(self.config.nonzero_when_possible),
                    )
                    logits = masker(session_rep, item_embeddings, valid_mask)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = distribution.sample()
                    position = int(action.item())
                    polluted = list(session)
                    polluted[position] = int(target_item)
                    rewards = compute_reward_components(
                        self.adapter,
                        original_session=session,
                        polluted_session=polluted,
                        selected_position=position,
                        target_item=int(target_item),
                        stealth_weight=float(self.config.stealth_weight),
                        local_weight=float(self.config.local_weight),
                    )
                    entropy = distribution.entropy()
                    batch_log_probs.append(distribution.log_prob(action))
                    batch_rewards.append(
                        torch.as_tensor(
                            rewards.total_reward,
                            dtype=logits.dtype,
                            device=logits.device,
                        )
                    )
                    batch_entropies.append(entropy)

                    selected_positions.append(position)
                    attack_rewards.append(float(rewards.attack_reward))
                    stealth_rewards.append(float(rewards.stealth_reward))
                    local_rewards.append(float(rewards.local_reward))
                    total_rewards.append(float(rewards.total_reward))
                    entropies.append(float(entropy.detach().cpu().item()))

                if batch_log_probs:
                    optimizer.zero_grad()
                    reward_tensor = torch.stack(batch_rewards)
                    advantage = reward_tensor - reward_tensor.mean()
                    log_probs = torch.stack(batch_log_probs)
                    entropy_tensor = torch.stack(batch_entropies)
                    batch_loss = -(log_probs * advantage.detach()).mean()
                    batch_loss = batch_loss - float(self.config.entropy_weight) * entropy_tensor.mean()
                    batch_loss.backward()
                    optimizer.step()
                    losses.append(float(batch_loss.detach().cpu().item()))

            epoch_rows.append(
                {
                    "epoch": int(epoch + 1),
                    "loss": _mean(losses),
                    "attack_reward": _mean(attack_rewards),
                    "stealth_reward": _mean(stealth_rewards),
                    "local_reward": _mean(local_rewards),
                    "entropy": _mean(entropies),
                    "total_reward": _mean(total_rewards),
                    "average_selected_position": _mean(selected_positions),
                    "position_distribution": position_distribution(selected_positions),
                }
            )

        history = {
            "target_item": int(target_item),
            "config": self.config.__dict__,
            "epochs": epoch_rows,
        }
        return CreatTrainingResult(masker=masker, history=history)


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / float(len(values)))


__all__ = ["CreatAdditiveSBRTrainer", "CreatTrainingResult"]
