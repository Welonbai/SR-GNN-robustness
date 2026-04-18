from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import pickle

import torch

from attack.common.config import Config
from attack.data.poisoned_dataset_builder import (
    build_poisoned_dataset,
    expand_session_to_samples,
)
from attack.inner_train.base import InnerTrainer
from attack.position_opt.artifacts import ensure_position_opt_artifact_dirs
from attack.position_opt.candidate_builder import build_candidate_positions
from attack.position_opt.objective import compute_position_opt_objective
from attack.position_opt.policy import PerSessionLogitPolicy
from attack.position_opt.poison_builder import replace_item_at_position
from attack.position_opt.selector import sample_position_reinforce, select_position_eval
from attack.position_opt.types import (
    CandidateMetadata,
    PositionOptArtifactPaths,
    PositionOptDefaults,
    SelectedPositionResult,
    TruncatedFineTuneConfig,
    resolve_position_opt_config,
)
from attack.surrogate.base import SurrogateBackend


@dataclass(frozen=True)
class _SessionCandidateState:
    original_session: list[int]
    metadata: CandidateMetadata
    candidate_sessions: list[list[int]]


class PositionOptMVPTrainer:
    """Joint-MVP trainer with one sampled poison set per outer step.

    Phase 2.5 switches the outer update to REINFORCE:
    - sample one categorical action per fake session
    - build one joint poisoned-session set
    - run one surrogate fine-tuning step from the clean checkpoint
    - evaluate once on real validation prefixes
    - update the per-session logits with a scalar reward and moving-average baseline
    """

    def __init__(
        self,
        surrogate_backend: SurrogateBackend,
        inner_trainer: InnerTrainer,
        *,
        clean_surrogate_checkpoint_path: str | Path,
        position_opt_config: PositionOptDefaults | Mapping[str, Any] | None = None,
    ) -> None:
        if str(clean_surrogate_checkpoint_path).strip() == "":
            raise ValueError("clean_surrogate_checkpoint_path must be provided explicitly.")
        checkpoint_path = Path(clean_surrogate_checkpoint_path)

        self.surrogate_backend = surrogate_backend
        self.inner_trainer = inner_trainer
        self.clean_surrogate_checkpoint_path = checkpoint_path
        self.position_opt_config = resolve_position_opt_config(position_opt_config)

        self.policy: PerSessionLogitPolicy | None = None
        self.training_history: list[dict[str, Any]] = []
        self._session_states: list[_SessionCandidateState] = []
        self._target_item: int | None = None
        self._trained_config: Config | None = None
        self._reward_baseline: float | None = None
        self._final_selected_positions: list[SelectedPositionResult] | None = None
        self._final_poisoned_sessions: list[list[int]] | None = None

    def train(
        self,
        fake_sessions: Sequence[Sequence[int]],
        target_item: int,
        shared_artifacts: object,
        config: Config,
    ) -> dict[str, Any]:
        normalized_fake_sessions = _normalize_fake_sessions(fake_sessions)
        target_id = int(target_item)
        if target_id <= 0:
            raise ValueError("target_item must be a positive item id.")

        clean_sessions, clean_labels = _resolve_clean_pairs(shared_artifacts)
        validation_sessions, validation_labels = _resolve_validation_pairs(shared_artifacts)
        validation_sessions, validation_labels = _select_validation_subset(
            validation_sessions,
            validation_labels,
            subset_size=self.position_opt_config.validation_subset_size,
        )

        self._target_item = target_id
        self._trained_config = config
        self._reward_baseline = None
        self._final_selected_positions = None
        self._final_poisoned_sessions = None
        self.training_history = []

        self._session_states = _build_session_states(
            normalized_fake_sessions,
            target_item=target_id,
            replacement_topk_ratio=config.attack.replacement_topk_ratio,
        )
        candidate_sizes = [len(state.metadata.positions) for state in self._session_states]
        self.policy = PerSessionLogitPolicy(
            candidate_sizes
        )
        candidate_avg = sum(candidate_sizes) / len(candidate_sizes)
        print(
            "[position-opt] "
            f"target={target_id} fake_sessions={len(normalized_fake_sessions)} "
            f"validation_prefixes={len(validation_sessions)} "
            f"candidate_sizes(min/avg/max)="
            f"{min(candidate_sizes)}/{candidate_avg:.2f}/{max(candidate_sizes)}"
        )
        print(
            "[position-opt] "
            f"outer_steps={int(self.position_opt_config.outer_steps)} "
            f"policy_lr={float(self.position_opt_config.policy_lr):g} "
            f"fine_tune_steps={int(self.position_opt_config.fine_tune_steps)} "
            f"checkpoint={self.clean_surrogate_checkpoint_path}"
        )

        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.position_opt_config.policy_lr),
        )
        fine_tune_config = TruncatedFineTuneConfig(
            steps=int(self.position_opt_config.fine_tune_steps),
            epochs=1,
        )

        clean_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            clean_gt_utility = self._precompute_clean_gt_utility(
                validation_sessions=validation_sessions,
                validation_labels=validation_labels,
            )

        for outer_step in range(int(self.position_opt_config.outer_steps)):
            optimizer.zero_grad()
            step_state = self._run_training_step(
                clean_sessions=clean_sessions,
                clean_labels=clean_labels,
                validation_sessions=validation_sessions,
                validation_labels=validation_labels,
                target_item=target_id,
                fine_tune_config=fine_tune_config,
                clean_gt_utility=clean_gt_utility,
            )
            step_state["policy_loss_tensor"].backward()
            optimizer.step()
            self._update_reward_baseline(step_state["reward"])
            baseline_value = step_state["baseline"]
            baseline_text = "None" if baseline_value is None else f"{float(baseline_value):.6f}"
            print(
                "[position-opt] "
                f"step {outer_step + 1}/{int(self.position_opt_config.outer_steps)} "
                f"reward={step_state['reward']:.6f} "
                f"baseline={baseline_text} "
                f"target={float(step_state['target_utility_tensor'].item()):.6f} "
                f"gt_penalty={float(step_state['gt_penalty_tensor'].item()):.6f} "
                f"entropy={step_state['mean_entropy']:.6f}"
            )
            self.training_history.append(
                {
                    "outer_step": int(outer_step),
                    "policy_update": "reinforce",
                    "outer_eval_source": "real_validation_sessions",
                    "validation_eval_count": int(len(validation_sessions)),
                    "policy_loss": float(step_state["policy_loss_tensor"].item()),
                    "reward": float(step_state["reward"]),
                    "baseline": (
                        None if step_state["baseline"] is None else float(step_state["baseline"])
                    ),
                    "advantage": float(step_state["advantage"]),
                    "joint_log_prob": float(step_state["joint_log_prob"]),
                    "mean_entropy": float(step_state["mean_entropy"]),
                    "target_utility": float(step_state["target_utility_tensor"].item()),
                    "gt_penalty": float(step_state["gt_penalty_tensor"].item()),
                    "gt_drop": float(step_state["gt_drop_tensor"].item()),
                    "clean_gt_utility": (
                        None
                        if step_state["clean_gt_utility"] is None
                        else float(step_state["clean_gt_utility"])
                    ),
                    "poisoned_gt_utility": (
                        None
                        if step_state["poisoned_gt_utility"] is None
                        else float(step_state["poisoned_gt_utility"])
                    ),
                    "selected_positions": [int(pos) for pos in step_state["selected_positions"]],
                    "selected_candidate_indices": [
                        int(idx) for idx in step_state["selected_candidate_indices"]
                    ],
                    "inner_train": step_state["inner_train_summary"],
                }
            )

        final_positions = self.export_final_selected_positions()
        final_sessions = self.export_final_poisoned_sessions()
        return {
            "training_history": list(self.training_history),
            "final_selected_positions": [asdict(result) for result in final_positions],
            "final_poisoned_session_count": int(len(final_sessions)),
            "target_item": target_id,
            "reward_baseline": self._reward_baseline,
        }

    def export_final_selected_positions(self) -> list[SelectedPositionResult]:
        if self.policy is None or not self._session_states:
            raise RuntimeError("train() must be called before exporting final selections.")

        results: list[SelectedPositionResult] = []
        for session_idx, session_state in enumerate(self._session_states):
            logits = self.policy.get_logits(session_idx)
            if self.position_opt_config.final_selection != "argmax":
                raise ValueError(
                    "Unsupported final_selection for the current position-opt MVP."
                )
            candidate_index = select_position_eval(logits)
            position = session_state.metadata.positions[candidate_index]
            results.append(
                SelectedPositionResult(
                    position=int(position),
                    candidate_index=int(candidate_index),
                    score=float(logits[candidate_index].detach().cpu().item()),
                )
            )

        self._final_selected_positions = results
        return list(results)

    def export_final_poisoned_sessions(self) -> list[list[int]]:
        if not self._session_states:
            raise RuntimeError("train() must be called before exporting final sessions.")

        selection_results = self._final_selected_positions or self.export_final_selected_positions()
        poisoned_sessions = [
            list(
                self._session_states[idx].candidate_sessions[
                    0 if result.candidate_index is None else result.candidate_index
                ]
            )
            for idx, result in enumerate(selection_results)
        ]
        self._final_poisoned_sessions = poisoned_sessions
        return [list(session) for session in poisoned_sessions]

    def save_artifacts(self, artifact_paths: PositionOptArtifactPaths) -> None:
        if self.policy is None:
            raise RuntimeError("train() must be called before saving artifacts.")

        paths = ensure_position_opt_artifact_dirs(artifact_paths)
        final_sessions = self.export_final_poisoned_sessions()
        final_positions = self.export_final_selected_positions()

        with paths.optimized_poisoned_sessions.open("wb") as handle:
            pickle.dump(final_sessions, handle)

        if paths.selected_positions is not None:
            payload = [asdict(result) for result in final_positions]
            with paths.selected_positions.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        if paths.training_history is not None:
            payload = {
                "target_item": self._target_item,
                "position_opt_config": asdict(self.position_opt_config),
                "policy_update": "reinforce",
                "reward_baseline_final": self._reward_baseline,
                "outer_eval_source": "real_validation_sessions",
                "training_history": self.training_history,
                "final_selected_positions": [asdict(result) for result in final_positions],
            }
            with paths.training_history.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        if paths.learned_logits is not None:
            torch.save(
                {
                    "target_item": self._target_item,
                    "logits": self.policy.export_logits(),
                    "candidate_positions": [
                        list(map(int, state.metadata.positions)) for state in self._session_states
                    ],
                },
                paths.learned_logits,
            )

    def _precompute_clean_gt_utility(
        self,
        *,
        validation_sessions: Sequence[Sequence[int]],
        validation_labels: Sequence[int],
    ) -> float:
        self.surrogate_backend.load_clean_checkpoint(self.clean_surrogate_checkpoint_path)
        clean_model = self.surrogate_backend.clone_clean_model()
        clean_result = self.surrogate_backend.score_gt(
            clean_model,
            validation_sessions,
            validation_labels,
        )
        return float(clean_result.mean)

    def _run_training_step(
        self,
        *,
        clean_sessions: Sequence[Sequence[int]],
        clean_labels: Sequence[int],
        validation_sessions: Sequence[Sequence[int]],
        validation_labels: Sequence[int],
        target_item: int,
        fine_tune_config: TruncatedFineTuneConfig,
        clean_gt_utility: float | None,
    ) -> dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")

        selected_candidate_indices: list[int] = []
        selected_positions: list[int] = []
        selected_poisoned_sessions: list[list[int]] = []
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []

        for session_idx, session_state in enumerate(self._session_states):
            logits = self.policy.get_logits(session_idx)
            candidate_index, log_prob, entropy = sample_position_reinforce(logits)
            selected_candidate_indices.append(int(candidate_index))
            selected_positions.append(int(session_state.metadata.positions[candidate_index]))
            selected_poisoned_sessions.append(list(session_state.candidate_sessions[candidate_index]))
            log_prob_terms.append(log_prob)
            entropy_terms.append(entropy)

        joint_log_prob = torch.stack(log_prob_terms).sum()
        mean_entropy = torch.stack(entropy_terms).mean()

        # Phase 2.5 uses one joint poison-set sample, one surrogate fine-tune, and
        # one validation-based reward per outer step. The inner trainer still sees
        # clean-train prefixes union the selected poisoned sessions.
        poisoned_train_data = build_poisoned_dataset(
            clean_sessions,
            clean_labels,
            selected_poisoned_sessions,
        )
        inner_result = self.inner_trainer.run(
            self.surrogate_backend,
            self.clean_surrogate_checkpoint_path,
            poisoned_train_data,
            config=fine_tune_config,
        )
        surrogate_model = inner_result.model

        target_result = self.surrogate_backend.score_target(
            surrogate_model,
            validation_sessions,
            target_item,
        )
        target_utility_tensor = joint_log_prob.new_tensor(float(target_result.mean))

        poisoned_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            poisoned_gt_result = self.surrogate_backend.score_gt(
                surrogate_model,
                validation_sessions,
                validation_labels,
            )
            poisoned_gt_utility = float(poisoned_gt_result.mean)

        objective = compute_position_opt_objective(
            target_utility_tensor,
            clean_gt_utility=clean_gt_utility,
            poisoned_gt_utility=poisoned_gt_utility,
            enable_gt_penalty=bool(self.position_opt_config.enable_gt_penalty),
            gt_penalty_weight=float(self.position_opt_config.gt_penalty_weight),
            gt_tolerance=float(self.position_opt_config.gt_tolerance),
        )

        reward = float(objective.reward.detach().item())
        baseline = self._reward_baseline
        advantage = reward if baseline is None else (reward - baseline)
        advantage_tensor = joint_log_prob.new_tensor(float(advantage))
        policy_loss_tensor = -(joint_log_prob * advantage_tensor)

        return {
            "policy_loss_tensor": policy_loss_tensor,
            "reward": reward,
            "baseline": baseline,
            "advantage": float(advantage),
            "joint_log_prob": float(joint_log_prob.detach().item()),
            "mean_entropy": float(mean_entropy.detach().item()),
            "target_utility_tensor": objective.target_utility,
            "gt_penalty_tensor": objective.gt_penalty,
            "gt_drop_tensor": objective.gt_drop,
            "clean_gt_utility": clean_gt_utility,
            "poisoned_gt_utility": poisoned_gt_utility,
            "selected_positions": selected_positions,
            "selected_candidate_indices": selected_candidate_indices,
            "inner_train_summary": _summarize_inner_history(inner_result.history),
        }

    def _update_reward_baseline(self, reward: float) -> None:
        momentum = float(self.position_opt_config.reward_baseline_momentum)
        if self._reward_baseline is None:
            self._reward_baseline = float(reward)
            return
        self._reward_baseline = (
            momentum * self._reward_baseline
            + (1.0 - momentum) * float(reward)
        )


def _build_session_states(
    fake_sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    replacement_topk_ratio: float,
) -> list[_SessionCandidateState]:
    session_states: list[_SessionCandidateState] = []
    for session in fake_sessions:
        session_list = list(session)
        candidate_positions = build_candidate_positions(session_list, replacement_topk_ratio)
        metadata = CandidateMetadata(
            session_length=len(session_list),
            replacement_topk_ratio=float(replacement_topk_ratio),
            positions=tuple(int(position) for position in candidate_positions),
        )
        candidate_sessions = [
            replace_item_at_position(session_list, position, target_item)
            for position in candidate_positions
        ]
        session_states.append(
            _SessionCandidateState(
                original_session=session_list,
                metadata=metadata,
                candidate_sessions=candidate_sessions,
            )
        )
    return session_states


def _normalize_fake_sessions(fake_sessions: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = [list(session) for session in fake_sessions]
    if not normalized:
        raise ValueError("fake_sessions must contain at least one session.")
    if any(len(session) == 0 for session in normalized):
        raise ValueError("fake_sessions must not contain empty sessions.")
    return normalized


def _resolve_clean_pairs(shared_artifacts: object) -> tuple[list[list[int]], list[int]]:
    clean_sessions = getattr(shared_artifacts, "clean_sessions", None)
    clean_labels = getattr(shared_artifacts, "clean_labels", None)
    if clean_sessions is None or clean_labels is None:
        raise ValueError(
            "shared_artifacts must expose clean_sessions and clean_labels for "
            "position optimization training."
        )
    normalized_sessions = [list(session) for session in clean_sessions]
    normalized_labels = [int(label) for label in clean_labels]
    if len(normalized_sessions) != len(normalized_labels):
        raise ValueError("shared_artifacts clean_sessions and clean_labels must align.")
    return normalized_sessions, normalized_labels


def _resolve_validation_pairs(shared_artifacts: object) -> tuple[list[list[int]], list[int]]:
    validation_sessions = getattr(shared_artifacts, "validation_sessions", None)
    validation_labels = getattr(shared_artifacts, "validation_labels", None)
    if validation_sessions is not None and validation_labels is not None:
        normalized_sessions = [list(session) for session in validation_sessions]
        normalized_labels = [int(label) for label in validation_labels]
        if len(normalized_sessions) != len(normalized_labels):
            raise ValueError("validation_sessions and validation_labels must align.")
        if not normalized_sessions:
            raise ValueError("validation_sessions must contain at least one prefix.")
        return normalized_sessions, normalized_labels

    canonical_dataset = getattr(shared_artifacts, "canonical_dataset", None)
    if canonical_dataset is None or getattr(canonical_dataset, "valid", None) is None:
        raise ValueError(
            "shared_artifacts must expose canonical_dataset.valid or explicit "
            "validation_sessions/validation_labels for position optimization."
        )

    validation_prefixes: list[list[int]] = []
    validation_next_items: list[int] = []
    for session in canonical_dataset.valid:
        prefixes, labels = expand_session_to_samples(session)
        validation_prefixes.extend(prefixes)
        validation_next_items.extend(int(label) for label in labels)

    if not validation_prefixes:
        raise ValueError("No validation prefixes could be derived from canonical_dataset.valid.")
    return validation_prefixes, validation_next_items


def _select_validation_subset(
    validation_sessions: Sequence[Sequence[int]],
    validation_labels: Sequence[int],
    *,
    subset_size: int | None,
) -> tuple[list[list[int]], list[int]]:
    # For MVP practicality, an optional validation subset uses the first N derived
    # validation prefixes deterministically. This keeps runs reproducible without
    # adding extra sampling/plumbing before Phase 3.
    sessions = [list(session) for session in validation_sessions]
    labels = [int(label) for label in validation_labels]
    if subset_size is None or subset_size >= len(sessions):
        return sessions, labels
    return sessions[:subset_size], labels[:subset_size]

def _summarize_inner_history(history: Mapping[str, Any] | None) -> dict[str, Any]:
    if history is None:
        return {}
    return {
        "steps": int(history.get("steps", 0)),
        "epochs": int(history.get("epochs", 0)),
        "avg_loss": (
            None if history.get("avg_loss") is None else float(history.get("avg_loss"))
        ),
    }


__all__ = ["PositionOptMVPTrainer"]
