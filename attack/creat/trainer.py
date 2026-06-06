from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import torch

from attack.common.config import CreatAdditiveSBRConfig
from attack.creat.candidates import position_distribution
from attack.creat.diagnostics import position_collapse_summary
from attack.creat.masker import CreatMasker
from attack.creat.reward_table import CreatV2RewardTable, build_v2_reward_table
from attack.creat.rewards import compute_reward_components
from attack.creat.rewards_v2 import RAW_REWARD_COMPONENTS, compose_v2_reward


@dataclass(frozen=True)
class CreatTrainingResult:
    masker: CreatMasker
    history: dict[str, object]
    reward_table: CreatV2RewardTable | None = None


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
        if str(self.config.variant) == "v2":
            return self._train_v2(
                target_item=int(target_item),
                template_sessions=template_sessions,
            )
        return self._train_v1(
            target_item=int(target_item),
            template_sessions=template_sessions,
        )

    def _train_v1(
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
        total_epochs = int(self.config.epochs)
        train_started_at = _timestamp_utc()
        train_started_monotonic = time.monotonic()
        print(
            "[CREAT-Additive-SBR] "
            f"target={int(target_item)} masker training started at {train_started_at}; "
            f"epochs={total_epochs}, templates={len(sessions)}",
            flush=True,
        )
        for epoch in range(total_epochs):
            epoch_started_at = _timestamp_utc()
            epoch_started_monotonic = time.monotonic()
            print(
                "[CREAT-Additive-SBR] "
                f"target={int(target_item)} epoch {epoch + 1}/{total_epochs} started",
                flush=True,
            )
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

            epoch_completed_at = _timestamp_utc()
            epoch_elapsed_seconds = time.monotonic() - epoch_started_monotonic
            epoch_row = {
                "epoch": int(epoch + 1),
                "started_at": epoch_started_at,
                "completed_at": epoch_completed_at,
                "elapsed_seconds": round(float(epoch_elapsed_seconds), 3),
                "loss": _mean(losses),
                "attack_reward": _mean(attack_rewards),
                "stealth_reward": _mean(stealth_rewards),
                "local_reward": _mean(local_rewards),
                "entropy": _mean(entropies),
                "total_reward": _mean(total_rewards),
                "average_selected_position": _mean(selected_positions),
                "position_distribution": position_distribution(selected_positions),
            }
            epoch_rows.append(epoch_row)
            print(
                "[CREAT-Additive-SBR] "
                f"target={int(target_item)} epoch {epoch + 1}/{total_epochs} completed "
                f"in {epoch_row['elapsed_seconds']}s; "
                f"loss={_format_optional_float(epoch_row['loss'])}, "
                f"total_reward={_format_optional_float(epoch_row['total_reward'])}",
                flush=True,
            )

        train_completed_at = _timestamp_utc()
        train_elapsed_seconds = time.monotonic() - train_started_monotonic
        print(
            "[CREAT-Additive-SBR] "
            f"target={int(target_item)} masker training completed at {train_completed_at}; "
            f"elapsed_seconds={round(float(train_elapsed_seconds), 3)}",
            flush=True,
        )
        history = {
            "target_item": int(target_item),
            "config": self.config.__dict__,
            "started_at": train_started_at,
            "completed_at": train_completed_at,
            "elapsed_seconds": round(float(train_elapsed_seconds), 3),
            "epochs": epoch_rows,
        }
        return CreatTrainingResult(masker=masker, history=history)

    def _train_v2(
        self,
        *,
        target_item: int,
        template_sessions: Sequence[Sequence[int]],
    ) -> CreatTrainingResult:
        sessions = [list(session) for session in template_sessions]
        if not sessions:
            raise ValueError("CREAT v2 training requires at least one effective template.")
        masker = CreatMasker(
            session_dim=self.adapter.embedding_dim,
            item_dim=self.adapter.embedding_dim,
            hidden_dim=int(self.config.hidden_dim),
            position_embedding_dim=int(self.config.position_embedding_dim),
            max_session_length=max(len(session) for session in sessions),
        )
        optimizer = torch.optim.Adam(masker.parameters(), lr=float(self.config.lr))
        rng = random.Random(self.seed + int(target_item))
        torch.manual_seed(self.seed + int(target_item))
        reward_table = build_v2_reward_table(
            self.adapter,
            template_sessions=sessions,
            target_item=int(target_item),
            replacement_topk_ratio=self.replacement_topk_ratio,
            nonzero_when_possible=bool(self.config.nonzero_when_possible),
            local_window_size=int(self.config.local_window_size),
            dpp_score_mode=str(self.config.dpp_score_mode),
            dpp_eps=float(self.config.dpp_eps),
        )
        phases = (
            ("attack", int(self.config.attack_epochs)),
            ("consistency", int(self.config.consistency_epochs)),
        )
        total_epochs = sum(count for _phase, count in phases)
        train_started_at = _timestamp_utc()
        train_started_monotonic = time.monotonic()
        epoch_rows: list[dict[str, object]] = []
        phase_rows: list[dict[str, object]] = []
        global_epoch = 0
        print(
            "[CREAT-Additive-SBR v2] "
            f"target={int(target_item)} training started at {train_started_at}; "
            f"epochs={total_epochs}, templates={len(sessions)}",
            flush=True,
        )
        for phase, phase_epoch_count in phases:
            phase_history: list[dict[str, object]] = []
            for phase_epoch in range(1, phase_epoch_count + 1):
                global_epoch += 1
                row = self._train_v2_epoch(
                    masker=masker,
                    optimizer=optimizer,
                    reward_table=reward_table,
                    sessions=sessions,
                    target_item=int(target_item),
                    phase=phase,
                    phase_epoch=phase_epoch,
                    global_epoch=global_epoch,
                    rng=rng,
                )
                epoch_rows.append(row)
                phase_history.append(row)
                print(
                    "[CREAT-Additive-SBR v2] "
                    f"target={int(target_item)} phase={phase} "
                    f"epoch {phase_epoch}/{phase_epoch_count} completed "
                    f"in {row['elapsed_seconds']}s; "
                    f"loss={_format_optional_float(row['loss'])}, "
                    f"total_reward={_format_optional_float(row['total_reward'])}",
                    flush=True,
                )
            phase_rows.append({"phase": phase, "epochs": phase_history})
        train_completed_at = _timestamp_utc()
        train_elapsed_seconds = time.monotonic() - train_started_monotonic
        history = {
            "variant": "v2",
            "target_item": int(target_item),
            "config": self.config.__dict__,
            "started_at": train_started_at,
            "completed_at": train_completed_at,
            "elapsed_seconds": round(float(train_elapsed_seconds), 3),
            "final_policy_phase": "consistency",
            "final_policy_global_epoch": int(global_epoch),
            "candidate_reward_stats": reward_table.candidate_reward_stats,
            "reward_table_build_metadata": reward_table.build_metadata,
            "candidate_composed_reward_stats": reward_table.composed_reward_stats(
                pattern_reward_weight=float(self.config.pattern_reward_weight),
                dpp_reward_weight=float(self.config.dpp_reward_weight),
                global_consistency_weight=float(self.config.global_consistency_weight),
                local_consistency_weight=float(self.config.local_consistency_weight),
            ),
            "epochs": epoch_rows,
            "phases": phase_rows,
        }
        print(
            "[CREAT-Additive-SBR v2] "
            f"target={int(target_item)} training completed at {train_completed_at}; "
            f"elapsed_seconds={round(float(train_elapsed_seconds), 3)}",
            flush=True,
        )
        return CreatTrainingResult(masker=masker, history=history, reward_table=reward_table)

    def _train_v2_epoch(
        self,
        *,
        masker: CreatMasker,
        optimizer,
        reward_table: CreatV2RewardTable,
        sessions: Sequence[Sequence[int]],
        target_item: int,
        phase: str,
        phase_epoch: int,
        global_epoch: int,
        rng: random.Random,
    ) -> dict[str, object]:
        started_at = _timestamp_utc()
        started_monotonic = time.monotonic()
        order = list(range(len(sessions)))
        rng.shuffle(order)
        losses: list[float] = []
        entropies: list[float] = []
        rewards: list[float] = []
        advantages: list[float] = []
        selected_positions: list[int] = []
        component_values = {name: [] for name in RAW_REWARD_COMPONENTS}
        local_affected_count = 0
        local_skipped_count = 0
        for start in range(0, len(order), int(self.config.batch_size)):
            batch_indices = order[start : start + int(self.config.batch_size)]
            batch_log_probs = []
            batch_rewards = []
            batch_entropies = []
            for index in batch_indices:
                session = sessions[index]
                logits = masker(
                    self.adapter.encode_session(session),
                    self.adapter.item_embeddings(session),
                    self.adapter.valid_position_mask(
                        session,
                        int(target_item),
                        self.replacement_topk_ratio,
                        nonzero_when_possible=bool(self.config.nonzero_when_possible),
                    ),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                position = int(action.item())
                components = reward_table.get(index, position)
                total_reward = compose_v2_reward(
                    components,
                    phase=phase,
                    pattern_reward_weight=float(self.config.pattern_reward_weight),
                    dpp_reward_weight=float(self.config.dpp_reward_weight),
                    global_consistency_weight=float(self.config.global_consistency_weight),
                    local_consistency_weight=float(self.config.local_consistency_weight),
                )
                batch_log_probs.append(distribution.log_prob(action))
                batch_rewards.append(
                    torch.as_tensor(total_reward, dtype=logits.dtype, device=logits.device)
                )
                batch_entropies.append(distribution.entropy())
                selected_positions.append(position)
                rewards.append(float(total_reward))
                entropies.append(float(distribution.entropy().detach().cpu().item()))
                values = components.to_dict()
                for name in RAW_REWARD_COMPONENTS:
                    component_values[name].append(float(values[name]))
                local_affected_count += int(components.local_affected_kgram_count)
                local_skipped_count += int(components.local_skipped_count)
            if batch_log_probs:
                optimizer.zero_grad()
                reward_tensor = torch.stack(batch_rewards)
                advantage = reward_tensor - reward_tensor.mean()
                entropy_tensor = torch.stack(batch_entropies)
                loss = -(torch.stack(batch_log_probs) * advantage.detach()).mean()
                loss = loss - float(self.config.entropy_weight) * entropy_tensor.mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
                advantages.extend(float(value) for value in advantage.detach().cpu().tolist())
        row: dict[str, object] = {
            "phase": phase,
            "phase_epoch": int(phase_epoch),
            "global_epoch": int(global_epoch),
            "started_at": started_at,
            "completed_at": _timestamp_utc(),
            "elapsed_seconds": round(float(time.monotonic() - started_monotonic), 3),
            "loss": _mean(losses),
            "entropy": _mean(entropies),
            "total_reward": _mean(rewards),
            "regularized_objective": (
                None
                if not rewards
                else float(_mean(rewards) + float(self.config.entropy_weight) * float(_mean(entropies)))
            ),
            "advantage_mean": _mean(advantages),
            "advantage_std": _population_std(advantages),
            "advantage_abs_mean": _mean([abs(value) for value in advantages]),
            "average_selected_position": _mean(selected_positions),
            "position_distribution": position_distribution(selected_positions),
            "local_affected_kgram_count": int(local_affected_count),
            "local_skipped_count": int(local_skipped_count),
            **position_collapse_summary(selected_positions),
        }
        row.update({name: _mean(values) for name, values in component_values.items()})
        return row


def _mean(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    return float(sum(float(value) for value in values) / float(len(values)))


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


def _population_std(values: Sequence[float | int]) -> float | None:
    if not values:
        return None
    mean = float(_mean(values))
    return float(
        (sum((float(value) - mean) ** 2 for value in values) / float(len(values)))
        ** 0.5
    )


__all__ = ["CreatAdditiveSBRTrainer", "CreatTrainingResult"]
