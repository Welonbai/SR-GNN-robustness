from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import pickle

import torch

from attack.common.config import Config
from attack.common.seed import derive_seed, set_seed
from attack.data.poisoned_dataset_builder import (
    build_poisoned_dataset,
    expand_session_to_samples,
)
from attack.inner_train.base import InnerTrainer
from attack.position_opt.artifacts import ensure_position_opt_artifact_dirs
from attack.position_opt.candidate_builder import build_candidate_position_result
from attack.position_opt.feature_builder import (
    PolicySpecialItemIds,
    SessionCandidateFeatures,
    build_policy_special_item_ids,
    build_session_candidate_features,
    infer_max_item_id,
)
from attack.position_opt.objective import compute_position_opt_objective
from attack.position_opt.policy import (
    SharedContextualPositionPolicy,
    resolve_policy_feature_set_spec,
)
from attack.position_opt.poison_builder import replace_item_at_position
from attack.position_opt.selector import sample_position_reinforce, select_position_eval
from attack.position_opt.types import (
    CandidateMetadata,
    PositionOptConfig,
    PositionOptArtifactPaths,
    SelectedPositionResult,
    SurrogateScoreResult,
    TruncatedFineTuneConfig,
    resolve_position_opt_config,
)
from attack.surrogate.base import SurrogateBackend

_LOWK_TARGET_METRIC_KEYS = (
    "targeted_mrr@10",
    "targeted_recall@10",
    "targeted_recall@20",
)
_LOWK_TARGET_METRIC_WEIGHTS = {
    "targeted_mrr@10": 0.6,
    "targeted_recall@10": 0.3,
    "targeted_recall@20": 0.1,
}


@dataclass(frozen=True)
class _SessionCandidateState:
    original_session: list[int]
    metadata: CandidateMetadata
    features: SessionCandidateFeatures
    candidate_sessions: list[list[int]]


@dataclass(frozen=True)
class _ArgmaxSelectionSnapshot:
    selected_position_results: tuple[SelectedPositionResult, ...]
    selected_candidate_indices: tuple[int, ...]
    selected_positions: tuple[int, ...]
    poisoned_sessions: tuple[tuple[int, ...], ...]
    selected_pos0_pct: float
    selected_pos_le_1_pct: float
    selected_pos_le_2_pct: float


@dataclass(frozen=True)
class _DeterministicCheckpointState:
    outer_step: int
    reward: float
    target_utility: float
    poisoned_target_utility: float
    delta_target_utility: float | None
    gt_penalty: float
    gt_drop: float
    poisoned_gt_utility: float | None
    selected_position_results: tuple[SelectedPositionResult, ...]
    selected_candidate_indices: tuple[int, ...]
    selected_positions: tuple[int, ...]
    poisoned_sessions: tuple[tuple[int, ...], ...]
    selected_pos0_pct: float
    selected_pos_le_1_pct: float
    selected_pos_le_2_pct: float
    policy_state_dict: dict[str, torch.Tensor]


class PositionOptMVPTrainer:
    """Joint-MVP trainer with one sampled poison set per outer step."""

    def __init__(
        self,
        surrogate_backend: SurrogateBackend,
        inner_trainer: InnerTrainer,
        *,
        clean_surrogate_checkpoint_path: str | Path,
        position_opt_config: PositionOptConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if str(clean_surrogate_checkpoint_path).strip() == "":
            raise ValueError("clean_surrogate_checkpoint_path must be provided explicitly.")
        checkpoint_path = Path(clean_surrogate_checkpoint_path)

        self.surrogate_backend = surrogate_backend
        self.inner_trainer = inner_trainer
        self.clean_surrogate_checkpoint_path = checkpoint_path
        self.position_opt_config = resolve_position_opt_config(position_opt_config)
        self._policy_feature_set_spec = resolve_policy_feature_set_spec(
            self.position_opt_config.policy_feature_set
        )

        self.policy: SharedContextualPositionPolicy | None = None
        self.training_history: list[dict[str, Any]] = []
        self._session_states: list[_SessionCandidateState] = []
        self._policy_special_item_ids: PolicySpecialItemIds | None = None
        self._target_item: int | None = None
        self._trained_config: Config | None = None
        self._reward_baseline: float | None = None
        self._clean_target_utility: float | None = None
        self._clean_target_metrics: dict[str, float | None] = _empty_lowk_target_metrics()
        self._final_selected_positions: list[SelectedPositionResult] | None = None
        self._final_poisoned_sessions: list[list[int]] | None = None
        self._candidate_space_diagnostics: dict[str, Any] | None = None
        self._best_deterministic_checkpoint: _DeterministicCheckpointState | None = None
        self._last_deterministic_checkpoint: _DeterministicCheckpointState | None = None
        self._deterministic_eval_schedule: tuple[int, ...] = ()
        self._exported_policy_source: str | None = None

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
        self._clean_target_utility = None
        self._clean_target_metrics = _empty_lowk_target_metrics()
        self._final_selected_positions = None
        self._final_poisoned_sessions = None
        self._candidate_space_diagnostics = None
        self._best_deterministic_checkpoint = None
        self._last_deterministic_checkpoint = None
        self._deterministic_eval_schedule = ()
        self._exported_policy_source = None
        self.training_history = []

        self._session_states, self._policy_special_item_ids = _build_session_states(
            normalized_fake_sessions,
            target_item=target_id,
            replacement_topk_ratio=config.attack.replacement_topk_ratio,
            nonzero_action_when_possible=bool(
                self.position_opt_config.nonzero_action_when_possible
            ),
        )
        self._candidate_space_diagnostics = _build_candidate_space_diagnostics(
            self._session_states
        )
        if self._prefix_score_enabled():
            self._session_states = self._build_prefix_scored_session_states(target_item=target_id)
        candidate_sizes = [len(state.metadata.positions) for state in self._session_states]
        self.policy = SharedContextualPositionPolicy(
            num_item_embeddings=self._policy_special_item_ids.num_item_embeddings,
            embedding_dim=int(self.position_opt_config.policy_embedding_dim),
            hidden_dim=int(self.position_opt_config.policy_hidden_dim),
            policy_feature_set=str(self.position_opt_config.policy_feature_set),
        )
        total_outer_steps = int(self.position_opt_config.outer_steps)
        deterministic_eval_schedule = _resolve_deterministic_eval_schedule(
            total_outer_steps=total_outer_steps,
            deterministic_eval_every=int(self.position_opt_config.deterministic_eval_every),
            deterministic_eval_include_final=bool(
                self.position_opt_config.deterministic_eval_include_final
            ),
        )
        self._deterministic_eval_schedule = tuple(deterministic_eval_schedule)
        if (
            self.position_opt_config.final_policy_selection == "best_deterministic"
            and not deterministic_eval_schedule
        ):
            raise ValueError(
                "attack.position_opt.final_policy_selection='best_deterministic' "
                "requires at least one deterministic evaluation step. Set "
                "attack.position_opt.deterministic_eval_every > 0 with outer_steps > 0, "
                "or use attack.position_opt.final_policy_selection='last'."
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
            f"outer_steps={total_outer_steps} "
            f"policy_lr={float(self.position_opt_config.policy_lr):g} "
            f"fine_tune_steps={int(self.position_opt_config.fine_tune_steps)} "
            f"reward_mode={self.position_opt_config.reward_mode} "
            f"entropy_coef={float(self.position_opt_config.entropy_coef):g} "
            f"checkpoint={self.clean_surrogate_checkpoint_path} "
            f"policy_feature_set={self.policy.policy_feature_set} "
            f"active_item_features={list(self.policy.active_item_features)} "
            f"active_scalar_features={list(self.policy.active_scalar_features)} "
            f"policy_input_dim={int(self.policy.policy_input_dim)}"
        )
        print(
            "[position-opt] "
            f"deterministic_eval_every={int(self.position_opt_config.deterministic_eval_every)} "
            f"deterministic_eval_include_final="
            f"{bool(self.position_opt_config.deterministic_eval_include_final)} "
            f"final_policy_selection={self.position_opt_config.final_policy_selection} "
            f"deterministic_eval_steps={deterministic_eval_schedule}"
        )

        optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(self.position_opt_config.policy_lr),
        )
        fine_tune_config = TruncatedFineTuneConfig(
            steps=int(self.position_opt_config.fine_tune_steps),
            epochs=1,
        )

        # Compute the clean baseline only after the validation subset is finalized
        # so the clean and poisoned utilities are measured on the same prefixes.
        clean_target_result = self._precompute_clean_target_result(
            validation_sessions=validation_sessions,
            target_item=target_id,
        )
        self._clean_target_metrics = _extract_lowk_target_metrics(
            clean_target_result,
            required=(self.position_opt_config.reward_mode == "delta_lowk_rank_utility"),
        )
        self._clean_target_utility = _resolve_scored_target_utility(
            reward_mode=self.position_opt_config.reward_mode,
            target_result=clean_target_result,
        )
        clean_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            clean_gt_utility = self._precompute_clean_gt_utility(
                validation_sessions=validation_sessions,
                validation_labels=validation_labels,
            )
        print(
            "[position-opt] "
            f"clean_target_utility={float(self._clean_target_utility):.6f}"
        )

        for outer_step in range(total_outer_steps):
            optimizer.zero_grad()
            step_state = self._run_training_step(
                clean_sessions=clean_sessions,
                clean_labels=clean_labels,
                validation_sessions=validation_sessions,
                validation_labels=validation_labels,
                target_item=target_id,
                fine_tune_config=fine_tune_config,
                clean_target_utility=self._clean_target_utility,
                clean_gt_utility=clean_gt_utility,
                outer_step=outer_step,
            )
            step_state["policy_loss_tensor"].backward()
            optimizer.step()
            self._update_reward_baseline(step_state["reward"])
            current_outer_step = int(outer_step) + 1
            deterministic_checkpoint = None
            is_best_deterministic_checkpoint = False
            if current_outer_step in deterministic_eval_schedule:
                deterministic_checkpoint = self._run_deterministic_checkpoint_eval(
                    clean_sessions=clean_sessions,
                    clean_labels=clean_labels,
                    validation_sessions=validation_sessions,
                    validation_labels=validation_labels,
                    target_item=target_id,
                    fine_tune_config=fine_tune_config,
                    clean_target_utility=self._clean_target_utility,
                    clean_gt_utility=clean_gt_utility,
                    outer_step=current_outer_step,
                )
                self._last_deterministic_checkpoint = deterministic_checkpoint
                is_best_deterministic_checkpoint = self._update_best_deterministic_checkpoint(
                    deterministic_checkpoint
                )
            baseline_value = step_state["baseline"]
            baseline_text = "None" if baseline_value is None else f"{float(baseline_value):.6f}"
            delta_text = (
                "None"
                if step_state["delta_target_utility"] is None
                else f"{float(step_state['delta_target_utility']):.6f}"
            )
            print(
                "[position-opt] "
                f"step {current_outer_step}/{total_outer_steps} "
                f"reward={step_state['reward']:.6f} "
                f"baseline={baseline_text} "
                f"poisoned_target={float(step_state['target_utility_tensor'].item()):.6f} "
                f"delta_target={delta_text} "
                f"gt_penalty={float(step_state['gt_penalty_tensor'].item()):.6f} "
                f"entropy={step_state['mean_entropy']:.6f} "
                f"entropy_loss={float(step_state['entropy_loss_tensor'].item()):.6f}"
            )
            if deterministic_checkpoint is not None:
                print(
                    "[position-opt] "
                    f"deterministic_eval step {current_outer_step}/{total_outer_steps} "
                    f"reward={deterministic_checkpoint.reward:.6f} "
                    f"target_utility={deterministic_checkpoint.target_utility:.6f} "
                    f"gt_penalty={deterministic_checkpoint.gt_penalty:.6f} "
                    f"best={is_best_deterministic_checkpoint}"
                )
            best_deterministic_checkpoint = self._best_deterministic_checkpoint
            self.training_history.append(
                {
                    "outer_step": int(outer_step),
                    "policy_update": "reinforce",
                    "outer_eval_source": "real_validation_sessions",
                    "validation_eval_count": int(len(validation_sessions)),
                    "reward_mode": str(step_state["reward_mode"]),
                    "policy_loss": float(step_state["policy_loss_tensor"].item()),
                    "reinforce_loss": float(step_state["reinforce_loss_tensor"].item()),
                    "entropy_loss": float(step_state["entropy_loss_tensor"].item()),
                    "entropy_coef": float(self.position_opt_config.entropy_coef),
                    "reward": float(step_state["reward"]),
                    "baseline": (
                        None if step_state["baseline"] is None else float(step_state["baseline"])
                    ),
                    "advantage": float(step_state["advantage"]),
                    "joint_log_prob": float(step_state["joint_log_prob"]),
                    "mean_entropy": float(step_state["mean_entropy"]),
                    "joint_entropy": float(step_state["joint_entropy"]),
                    "target_utility": float(step_state["target_utility_tensor"].item()),
                    "poisoned_target_utility": float(step_state["poisoned_target_utility"]),
                    "clean_target_utility": (
                        None
                        if step_state["clean_target_utility"] is None
                        else float(step_state["clean_target_utility"])
                    ),
                    "delta_target_utility": (
                        None
                        if step_state["delta_target_utility"] is None
                        else float(step_state["delta_target_utility"])
                    ),
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
                    **_lowk_target_metric_history_fields(
                        step_state["clean_target_metrics"],
                        prefix="clean",
                    ),
                    **_lowk_target_metric_history_fields(
                        step_state["poisoned_target_metrics"],
                        prefix="poisoned",
                    ),
                    "selected_positions": [int(pos) for pos in step_state["selected_positions"]],
                    "selected_candidate_indices": [
                        int(idx) for idx in step_state["selected_candidate_indices"]
                    ],
                    "position_opt_step_seed": int(step_state["position_opt_step_seed"]),
                    "surrogate_train_step_seed": int(step_state["surrogate_train_step_seed"]),
                    "inner_train": step_state["inner_train_summary"],
                    "deterministic_eval_ran": bool(deterministic_checkpoint is not None),
                    "deterministic_reward": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.reward)
                    ),
                    "deterministic_target_utility": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.target_utility)
                    ),
                    "deterministic_poisoned_target_utility": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.poisoned_target_utility)
                    ),
                    "deterministic_delta_target_utility": (
                        None
                        if deterministic_checkpoint is None
                        or deterministic_checkpoint.delta_target_utility is None
                        else float(deterministic_checkpoint.delta_target_utility)
                    ),
                    "deterministic_gt_penalty": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.gt_penalty)
                    ),
                    "deterministic_gt_drop": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.gt_drop)
                    ),
                    "deterministic_poisoned_gt_utility": (
                        None
                        if deterministic_checkpoint is None
                        or deterministic_checkpoint.poisoned_gt_utility is None
                        else float(deterministic_checkpoint.poisoned_gt_utility)
                    ),
                    "deterministic_selected_pos0_pct": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.selected_pos0_pct)
                    ),
                    "deterministic_selected_pos_leq_1_pct": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.selected_pos_le_1_pct)
                    ),
                    "deterministic_selected_pos_leq_2_pct": (
                        None
                        if deterministic_checkpoint is None
                        else float(deterministic_checkpoint.selected_pos_le_2_pct)
                    ),
                    "is_best_deterministic_checkpoint": bool(
                        is_best_deterministic_checkpoint
                    ),
                    "best_deterministic_step_so_far": (
                        None
                        if best_deterministic_checkpoint is None
                        else int(best_deterministic_checkpoint.outer_step)
                    ),
                    "best_deterministic_reward_so_far": (
                        None
                        if best_deterministic_checkpoint is None
                        else float(best_deterministic_checkpoint.reward)
                    ),
                }
            )

        final_positions = self.export_final_selected_positions()
        final_sessions = self.export_final_poisoned_sessions()
        policy_input_metadata = self._policy_input_metadata()
        best_deterministic_checkpoint = self._best_deterministic_checkpoint
        last_deterministic_checkpoint = self._last_deterministic_checkpoint
        last_deterministic_reward = (
            None
            if last_deterministic_checkpoint is None
            or last_deterministic_checkpoint.outer_step != total_outer_steps
            else float(last_deterministic_checkpoint.reward)
        )
        best_minus_last_deterministic_reward = (
            None
            if best_deterministic_checkpoint is None or last_deterministic_reward is None
            else float(best_deterministic_checkpoint.reward - last_deterministic_reward)
        )
        return {
            "training_history": list(self.training_history),
            "final_selected_positions": [asdict(result) for result in final_positions],
            "final_poisoned_session_count": int(len(final_sessions)),
            "target_item": target_id,
            "nonzero_action_when_possible": bool(
                self.position_opt_config.nonzero_action_when_possible
            ),
            "candidate_space_diagnostics": self._candidate_space_diagnostics_payload(),
            "final_position_diagnostics": _build_final_position_diagnostics(final_positions),
            "policy_representation": "shared_contextual_mlp",
            "reward_baseline": self._reward_baseline,
            "reward_mode": str(self.position_opt_config.reward_mode),
            "clean_target_utility": self._clean_target_utility,
            "deterministic_eval_every": int(self.position_opt_config.deterministic_eval_every),
            "deterministic_eval_include_final": bool(
                self.position_opt_config.deterministic_eval_include_final
            ),
            "final_policy_selection": str(self.position_opt_config.final_policy_selection),
            "best_deterministic_step": (
                None
                if best_deterministic_checkpoint is None
                else int(best_deterministic_checkpoint.outer_step)
            ),
            "best_deterministic_reward": (
                None
                if best_deterministic_checkpoint is None
                else float(best_deterministic_checkpoint.reward)
            ),
            "best_deterministic_target_utility": (
                None
                if best_deterministic_checkpoint is None
                else float(best_deterministic_checkpoint.target_utility)
            ),
            "best_deterministic_gt_penalty": (
                None
                if best_deterministic_checkpoint is None
                else float(best_deterministic_checkpoint.gt_penalty)
            ),
            "last_deterministic_reward": last_deterministic_reward,
            "best_minus_last_deterministic_reward": best_minus_last_deterministic_reward,
            "exported_policy_source": str(self._resolve_final_policy_source()),
            "policy_scalar_feature_names": (
                None if self.policy is None else list(self.policy.scalar_feature_names)
            ),
            "policy_item_feature_names": policy_input_metadata["active_item_features"],
            "active_item_features": policy_input_metadata["active_item_features"],
            "active_scalar_features": policy_input_metadata["active_scalar_features"],
            "policy_input_dim": policy_input_metadata["policy_input_dim"],
            "policy_embedding_dim": policy_input_metadata["policy_embedding_dim"],
            "policy_hidden_dim": policy_input_metadata["policy_hidden_dim"],
            **_lowk_target_metric_history_fields(
                self._clean_target_metrics,
                prefix="clean",
            ),
            **self._prefix_score_metadata(),
        }

    def export_final_selected_positions(self) -> list[SelectedPositionResult]:
        if self.policy is None or not self._session_states:
            raise RuntimeError("train() must be called before exporting final selections.")
        if self._final_selected_positions is not None:
            return list(self._final_selected_positions)

        self._ensure_export_policy_selected()
        selection_snapshot = self._collect_argmax_selection_snapshot()
        self._final_selected_positions = list(selection_snapshot.selected_position_results)
        return list(self._final_selected_positions)

    def export_final_poisoned_sessions(self) -> list[list[int]]:
        if not self._session_states:
            raise RuntimeError("train() must be called before exporting final sessions.")
        if self._final_poisoned_sessions is not None:
            return [list(session) for session in self._final_poisoned_sessions]

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
        final_positions = self.export_final_selected_positions()
        final_sessions = self.export_final_poisoned_sessions()
        exported_policy_source = self._resolve_final_policy_source()
        best_deterministic_checkpoint = self._best_deterministic_checkpoint
        last_deterministic_reward = self._last_deterministic_reward()
        best_minus_last_deterministic_reward = self._best_minus_last_deterministic_reward()

        with paths.optimized_poisoned_sessions.open("wb") as handle:
            pickle.dump(final_sessions, handle)

        if paths.selected_positions is not None:
            payload = [asdict(result) for result in final_positions]
            with paths.selected_positions.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        if paths.training_history is not None:
            policy_input_metadata = self._policy_input_metadata()
            payload = {
                "target_item": self._target_item,
                "position_opt_config": asdict(self.position_opt_config),
                "resolved_seeds": (
                    None
                    if self._trained_config is None
                    else asdict(self._trained_config.seeds)
                ),
                "policy_representation": "shared_contextual_mlp",
                "policy_update": "reinforce",
                "nonzero_action_when_possible": bool(
                    self.position_opt_config.nonzero_action_when_possible
                ),
                "candidate_space_diagnostics": self._candidate_space_diagnostics_payload(),
                "final_position_diagnostics": _build_final_position_diagnostics(final_positions),
                "reward_baseline_final": self._reward_baseline,
                "reward_mode": str(self.position_opt_config.reward_mode),
                "deterministic_eval_every": int(self.position_opt_config.deterministic_eval_every),
                "deterministic_eval_include_final": bool(
                    self.position_opt_config.deterministic_eval_include_final
                ),
                "final_policy_selection": str(self.position_opt_config.final_policy_selection),
                "best_deterministic_step": (
                    None
                    if best_deterministic_checkpoint is None
                    else int(best_deterministic_checkpoint.outer_step)
                ),
                "best_deterministic_reward": (
                    None
                    if best_deterministic_checkpoint is None
                    else float(best_deterministic_checkpoint.reward)
                ),
                "best_deterministic_target_utility": (
                    None
                    if best_deterministic_checkpoint is None
                    else float(best_deterministic_checkpoint.target_utility)
                ),
                "best_deterministic_gt_penalty": (
                    None
                    if best_deterministic_checkpoint is None
                    else float(best_deterministic_checkpoint.gt_penalty)
                ),
                "last_deterministic_reward": last_deterministic_reward,
                "best_minus_last_deterministic_reward": best_minus_last_deterministic_reward,
                "exported_policy_source": str(exported_policy_source),
                "policy_scalar_feature_names": (
                    None if self.policy is None else list(self.policy.scalar_feature_names)
                ),
                "policy_item_feature_names": policy_input_metadata["active_item_features"],
                "active_item_features": policy_input_metadata["active_item_features"],
                "active_scalar_features": policy_input_metadata["active_scalar_features"],
                "policy_input_dim": policy_input_metadata["policy_input_dim"],
                "policy_embedding_dim": policy_input_metadata["policy_embedding_dim"],
                "policy_hidden_dim": policy_input_metadata["policy_hidden_dim"],
                "clean_target_utility": self._clean_target_utility,
                **_lowk_target_metric_history_fields(
                    self._clean_target_metrics,
                    prefix="clean",
                ),
                "outer_eval_source": "real_validation_sessions",
                **self._prefix_score_metadata(),
                "training_history": self.training_history,
                "final_selected_positions": [asdict(result) for result in final_positions],
            }
            with paths.training_history.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)

        if paths.learned_logits is not None:
            torch.save(self._build_learned_logits_payload(), paths.learned_logits)

    def _precompute_clean_target_result(
        self,
        *,
        validation_sessions: Sequence[Sequence[int]],
        target_item: int,
    ) -> SurrogateScoreResult:
        self.surrogate_backend.load_clean_checkpoint(self.clean_surrogate_checkpoint_path)
        clean_model = self.surrogate_backend.clone_clean_model()
        # score_target() owns the evaluation path and applies eval/no_grad before
        # scoring, so the clean baseline stays free of training-side effects.
        clean_result = self.surrogate_backend.score_target(
            clean_model,
            validation_sessions,
            target_item,
        )
        return clean_result

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
        clean_target_utility: float | None,
        clean_gt_utility: float | None,
        outer_step: int,
    ) -> dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        if self._trained_config is None:
            raise RuntimeError("train() must be called with a Config before running steps.")

        selected_candidate_indices: list[int] = []
        selected_positions: list[int] = []
        selected_poisoned_sessions: list[list[int]] = []
        log_prob_terms: list[torch.Tensor] = []
        entropy_terms: list[torch.Tensor] = []
        position_opt_step_seed = derive_seed(
            self._trained_config.seeds.position_opt_seed,
            "position_opt",
            int(target_item),
            int(outer_step),
        )
        set_seed(position_opt_step_seed)

        for session_state in self._session_states:
            logits = self._score_session_candidates(session_state)
            candidate_index, log_prob, entropy = sample_position_reinforce(logits)
            selected_candidate_indices.append(int(candidate_index))
            selected_positions.append(int(session_state.metadata.positions[candidate_index]))
            selected_poisoned_sessions.append(list(session_state.candidate_sessions[candidate_index]))
            log_prob_terms.append(log_prob)
            entropy_terms.append(entropy)

        joint_log_prob = torch.stack(log_prob_terms).sum()
        mean_entropy = torch.stack(entropy_terms).mean()
        joint_entropy = torch.stack(entropy_terms).sum()

        # Phase 2.5 uses one joint poison-set sample, one surrogate fine-tune, and
        # one validation-based reward per outer step. The inner trainer still sees
        # clean-train prefixes union the selected poisoned sessions.
        poisoned_train_data = build_poisoned_dataset(
            clean_sessions,
            clean_labels,
            selected_poisoned_sessions,
        )
        surrogate_train_step_seed = derive_seed(
            self._trained_config.seeds.surrogate_train_seed,
            "surrogate_train",
            int(target_item),
            int(outer_step),
        )
        inner_result = self.inner_trainer.run(
            self.surrogate_backend,
            self.clean_surrogate_checkpoint_path,
            poisoned_train_data,
            config=fine_tune_config,
            seed=surrogate_train_step_seed,
        )
        surrogate_model = inner_result.model

        target_result = self.surrogate_backend.score_target(
            surrogate_model,
            validation_sessions,
            target_item,
        )
        poisoned_target_metrics = _extract_lowk_target_metrics(
            target_result,
            required=(self.position_opt_config.reward_mode == "delta_lowk_rank_utility"),
        )
        poisoned_target_utility = _resolve_scored_target_utility(
            reward_mode=self.position_opt_config.reward_mode,
            target_result=target_result,
        )
        target_utility_tensor = joint_log_prob.new_tensor(poisoned_target_utility)
        reward_target_utility = _resolve_reward_target_utility(
            reward_mode=self.position_opt_config.reward_mode,
            poisoned_target_utility=poisoned_target_utility,
            clean_target_utility=clean_target_utility,
        )
        reward_target_utility_tensor = joint_log_prob.new_tensor(float(reward_target_utility))
        delta_target_utility = (
            None
            if clean_target_utility is None
            else float(poisoned_target_utility - float(clean_target_utility))
        )

        poisoned_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            poisoned_gt_result = self.surrogate_backend.score_gt(
                surrogate_model,
                validation_sessions,
                validation_labels,
            )
            poisoned_gt_utility = float(poisoned_gt_result.mean)

        objective = compute_position_opt_objective(
            reward_target_utility_tensor,
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
        policy_loss_tensor, reinforce_loss_tensor, entropy_loss_tensor = _build_policy_loss(
            joint_log_prob=joint_log_prob,
            advantage_tensor=advantage_tensor,
            joint_entropy=joint_entropy,
            entropy_coef=float(self.position_opt_config.entropy_coef),
        )

        return {
            "policy_loss_tensor": policy_loss_tensor,
            "reinforce_loss_tensor": reinforce_loss_tensor,
            "entropy_loss_tensor": entropy_loss_tensor,
            "reward": reward,
            "baseline": baseline,
            "advantage": float(advantage),
            "joint_log_prob": float(joint_log_prob.detach().item()),
            "mean_entropy": float(mean_entropy.detach().item()),
            "joint_entropy": float(joint_entropy.detach().item()),
            "reward_mode": str(self.position_opt_config.reward_mode),
            "target_utility_tensor": target_utility_tensor,
            "poisoned_target_utility": poisoned_target_utility,
            "delta_target_utility": delta_target_utility,
            "gt_penalty_tensor": objective.gt_penalty,
            "gt_drop_tensor": objective.gt_drop,
            "clean_target_utility": clean_target_utility,
            "clean_target_metrics": dict(self._clean_target_metrics),
            "clean_gt_utility": clean_gt_utility,
            "poisoned_gt_utility": poisoned_gt_utility,
            "poisoned_target_metrics": poisoned_target_metrics,
            "selected_positions": selected_positions,
            "selected_candidate_indices": selected_candidate_indices,
            "position_opt_step_seed": int(position_opt_step_seed),
            "surrogate_train_step_seed": int(surrogate_train_step_seed),
            "inner_train_summary": _summarize_inner_history(inner_result.history),
        }

    def _run_deterministic_checkpoint_eval(
        self,
        *,
        clean_sessions: Sequence[Sequence[int]],
        clean_labels: Sequence[int],
        validation_sessions: Sequence[Sequence[int]],
        validation_labels: Sequence[int],
        target_item: int,
        fine_tune_config: TruncatedFineTuneConfig,
        clean_target_utility: float | None,
        clean_gt_utility: float | None,
        outer_step: int,
    ) -> _DeterministicCheckpointState:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        if self._trained_config is None:
            raise RuntimeError("train() must be called with a Config before running steps.")

        selection_snapshot = self._collect_argmax_selection_snapshot()
        poisoned_train_data = build_poisoned_dataset(
            clean_sessions,
            clean_labels,
            [list(session) for session in selection_snapshot.poisoned_sessions],
        )
        surrogate_train_step_seed = derive_seed(
            self._trained_config.seeds.surrogate_train_seed,
            "surrogate_train_deterministic_eval",
            int(target_item),
        )
        inner_result = self.inner_trainer.run(
            self.surrogate_backend,
            self.clean_surrogate_checkpoint_path,
            poisoned_train_data,
            config=fine_tune_config,
            seed=surrogate_train_step_seed,
        )
        surrogate_model = inner_result.model
        target_result = self.surrogate_backend.score_target(
            surrogate_model,
            validation_sessions,
            target_item,
        )
        poisoned_target_utility = _resolve_scored_target_utility(
            reward_mode=self.position_opt_config.reward_mode,
            target_result=target_result,
        )
        target_utility = _resolve_reward_target_utility(
            reward_mode=self.position_opt_config.reward_mode,
            poisoned_target_utility=poisoned_target_utility,
            clean_target_utility=clean_target_utility,
        )
        delta_target_utility = (
            None
            if clean_target_utility is None
            else float(poisoned_target_utility - float(clean_target_utility))
        )

        poisoned_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            poisoned_gt_result = self.surrogate_backend.score_gt(
                surrogate_model,
                validation_sessions,
                validation_labels,
            )
            poisoned_gt_utility = float(poisoned_gt_result.mean)

        objective = compute_position_opt_objective(
            float(target_utility),
            clean_gt_utility=clean_gt_utility,
            poisoned_gt_utility=poisoned_gt_utility,
            enable_gt_penalty=bool(self.position_opt_config.enable_gt_penalty),
            gt_penalty_weight=float(self.position_opt_config.gt_penalty_weight),
            gt_tolerance=float(self.position_opt_config.gt_tolerance),
        )
        return _DeterministicCheckpointState(
            outer_step=int(outer_step),
            reward=float(objective.reward.detach().item()),
            target_utility=float(target_utility),
            poisoned_target_utility=float(poisoned_target_utility),
            delta_target_utility=delta_target_utility,
            gt_penalty=float(objective.gt_penalty.detach().item()),
            gt_drop=float(objective.gt_drop.detach().item()),
            poisoned_gt_utility=(
                None if poisoned_gt_utility is None else float(poisoned_gt_utility)
            ),
            selected_position_results=selection_snapshot.selected_position_results,
            selected_candidate_indices=selection_snapshot.selected_candidate_indices,
            selected_positions=selection_snapshot.selected_positions,
            poisoned_sessions=selection_snapshot.poisoned_sessions,
            selected_pos0_pct=float(selection_snapshot.selected_pos0_pct),
            selected_pos_le_1_pct=float(selection_snapshot.selected_pos_le_1_pct),
            selected_pos_le_2_pct=float(selection_snapshot.selected_pos_le_2_pct),
            policy_state_dict=self._capture_policy_state_dict(),
        )

    def _update_best_deterministic_checkpoint(
        self,
        checkpoint_state: _DeterministicCheckpointState,
    ) -> bool:
        best_checkpoint = self._best_deterministic_checkpoint
        if (
            best_checkpoint is None
            or float(checkpoint_state.reward) > float(best_checkpoint.reward)
        ):
            self._best_deterministic_checkpoint = checkpoint_state
            return True
        return False

    def _collect_argmax_selection_snapshot(self) -> _ArgmaxSelectionSnapshot:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        if self.position_opt_config.final_selection != "argmax":
            raise ValueError(
                "Unsupported final_selection for the current position-opt MVP."
            )

        was_training = bool(self.policy.training)
        self.policy.eval()
        results: list[SelectedPositionResult] = []
        selected_candidate_indices: list[int] = []
        selected_positions: list[int] = []
        poisoned_sessions: list[tuple[int, ...]] = []
        try:
            with torch.no_grad():
                for session_state in self._session_states:
                    logits = self._score_session_candidates(session_state)
                    candidate_index = select_position_eval(logits)
                    position = int(session_state.metadata.positions[candidate_index])
                    results.append(
                        SelectedPositionResult(
                            position=position,
                            candidate_index=int(candidate_index),
                            score=float(logits[candidate_index].detach().cpu().item()),
                        )
                    )
                    selected_candidate_indices.append(int(candidate_index))
                    selected_positions.append(position)
                    poisoned_sessions.append(
                        tuple(session_state.candidate_sessions[candidate_index])
                    )
        finally:
            if was_training:
                self.policy.train()

        return _ArgmaxSelectionSnapshot(
            selected_position_results=tuple(results),
            selected_candidate_indices=tuple(selected_candidate_indices),
            selected_positions=tuple(selected_positions),
            poisoned_sessions=tuple(poisoned_sessions),
            selected_pos0_pct=_selected_position_pct(selected_positions, max_position=0),
            selected_pos_le_1_pct=_selected_position_pct(selected_positions, max_position=1),
            selected_pos_le_2_pct=_selected_position_pct(selected_positions, max_position=2),
        )

    def _capture_policy_state_dict(self) -> dict[str, torch.Tensor]:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        return {
            key: value.detach().cpu().clone()
            for key, value in self.policy.state_dict().items()
        }

    def _ensure_export_policy_selected(self) -> None:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        desired_source = self._resolve_final_policy_source()
        if self._exported_policy_source == desired_source:
            return
        if desired_source == "best_deterministic":
            best_checkpoint = self._best_deterministic_checkpoint
            if best_checkpoint is None:
                raise ValueError(
                    "attack.position_opt.final_policy_selection='best_deterministic' "
                    "requires at least one deterministic checkpoint evaluation."
                )
            self.policy.load_state_dict(best_checkpoint.policy_state_dict)
        self._final_selected_positions = None
        self._final_poisoned_sessions = None
        self._exported_policy_source = desired_source

    def _resolve_final_policy_source(self) -> str:
        if self.position_opt_config.final_policy_selection == "last":
            return "last"
        if self.position_opt_config.final_policy_selection == "best_deterministic":
            if self._best_deterministic_checkpoint is None:
                raise ValueError(
                    "attack.position_opt.final_policy_selection='best_deterministic' "
                    "requires at least one deterministic checkpoint evaluation."
                )
            return "best_deterministic"
        raise ValueError(
            "Unsupported final_policy_selection: "
            f"{self.position_opt_config.final_policy_selection!r}"
        )

    def _last_deterministic_reward(self) -> float | None:
        last_checkpoint = self._last_deterministic_checkpoint
        if (
            last_checkpoint is None
            or int(last_checkpoint.outer_step) != int(self.position_opt_config.outer_steps)
        ):
            return None
        return float(last_checkpoint.reward)

    def _best_minus_last_deterministic_reward(self) -> float | None:
        best_checkpoint = self._best_deterministic_checkpoint
        last_reward = self._last_deterministic_reward()
        if best_checkpoint is None or last_reward is None:
            return None
        return float(best_checkpoint.reward - last_reward)

    def _update_reward_baseline(self, reward: float) -> None:
        momentum = float(self.position_opt_config.reward_baseline_momentum)
        if self._reward_baseline is None:
            self._reward_baseline = float(reward)
            return
        self._reward_baseline = (
            momentum * self._reward_baseline
            + (1.0 - momentum) * float(reward)
        )

    def _score_session_candidates(self, session_state: _SessionCandidateState) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        return self.policy.score_candidates(session_state.features.tensors)

    def _prefix_score_enabled(self) -> bool:
        return bool(self._policy_feature_set_spec.requires_prefix_features)

    def _prefix_score_metadata(self) -> dict[str, Any]:
        enabled = self._prefix_score_enabled()
        return {
            "policy_feature_set": str(self.position_opt_config.policy_feature_set),
            "prefix_score_enabled": bool(enabled),
            "prefix_score_type": ("probability" if enabled else None),
            "pos0_prefix_handling": (
                "score_zero_has_prefix_false" if enabled else None
            ),
        }

    def _policy_input_metadata(self) -> dict[str, Any]:
        if self.policy is None:
            return {
                "active_item_features": None,
                "active_scalar_features": None,
                "policy_input_dim": None,
                "policy_embedding_dim": int(self.position_opt_config.policy_embedding_dim),
                "policy_hidden_dim": int(self.position_opt_config.policy_hidden_dim),
            }
        return self.policy.input_metadata()

    def _candidate_space_diagnostics_payload(self) -> dict[str, Any] | None:
        if self._candidate_space_diagnostics is None:
            return None
        return dict(self._candidate_space_diagnostics)

    def _build_prefix_scored_session_states(
        self,
        *,
        target_item: int,
    ) -> list[_SessionCandidateState]:
        if self._policy_special_item_ids is None:
            raise RuntimeError("Policy special-item ids are not initialized.")

        clean_model: object | None = None
        if self._policy_feature_set_spec.requires_prefix_scores:
            self.surrogate_backend.load_clean_checkpoint(self.clean_surrogate_checkpoint_path)
            clean_model = self.surrogate_backend.clone_clean_model()
        prefix_scored_states: list[_SessionCandidateState] = []
        for session_state in self._session_states:
            prefix_scores, has_prefixes = self._resolve_session_prefix_features(
                clean_model,
                session_state=session_state,
                target_item=target_item,
            )
            features = build_session_candidate_features(
                session_state.original_session,
                session_state.metadata.positions,
                target_item=target_item,
                special_item_ids=self._policy_special_item_ids,
                prefix_scores=prefix_scores,
                has_prefixes=has_prefixes,
            )
            prefix_scored_states.append(
                _SessionCandidateState(
                    original_session=list(session_state.original_session),
                    metadata=session_state.metadata,
                    features=features,
                    candidate_sessions=[list(session) for session in session_state.candidate_sessions],
                )
            )
        return prefix_scored_states

    def _resolve_session_prefix_features(
        self,
        clean_model: object | None,
        *,
        session_state: _SessionCandidateState,
        target_item: int,
    ) -> tuple[list[float], list[bool]]:
        positions = [int(position) for position in session_state.metadata.positions]
        prefix_scores = [0.0] * len(positions)
        has_prefixes = [False] * len(positions)
        nonzero_prefixes: list[list[int]] = []
        nonzero_candidate_indices: list[int] = []

        for candidate_index, position in enumerate(positions):
            if position <= 0:
                continue
            nonzero_prefixes.append(list(session_state.original_session[:position]))
            nonzero_candidate_indices.append(int(candidate_index))
            has_prefixes[candidate_index] = True

        if not nonzero_prefixes or not self._policy_feature_set_spec.requires_prefix_scores:
            return prefix_scores, has_prefixes
        if clean_model is None:
            raise RuntimeError("clean_model is required when prefix scores are active.")

        target_result = self.surrogate_backend.score_target(
            clean_model,
            nonzero_prefixes,
            target_item,
        )
        if len(target_result.values) != len(nonzero_candidate_indices):
            raise RuntimeError("Prefix score count does not match scored candidate positions.")
        for candidate_index, prefix_score in zip(
            nonzero_candidate_indices,
            target_result.values,
        ):
            prefix_scores[candidate_index] = float(prefix_score)
        return prefix_scores, has_prefixes

    def _build_learned_logits_payload(self) -> dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Policy is not initialized.")
        if self._policy_special_item_ids is None:
            raise RuntimeError("Policy special-item ids are not initialized.")

        final_selected_positions = self.export_final_selected_positions()
        exported_policy_source = self._resolve_final_policy_source()
        best_deterministic_checkpoint = self._best_deterministic_checkpoint
        last_deterministic_reward = self._last_deterministic_reward()
        best_minus_last_deterministic_reward = self._best_minus_last_deterministic_reward()
        sessions_payload: list[dict[str, Any]] = []
        with torch.no_grad():
            for session_idx, session_state in enumerate(self._session_states):
                logits = self._score_session_candidates(session_state).detach().cpu().clone()
                sessions_payload.append(
                    {
                        "session_index": int(session_idx),
                        "session_length": int(session_state.metadata.session_length),
                        "candidate_positions_before_mask": list(
                            map(int, session_state.metadata.positions_before_mask)
                        ),
                        "candidate_positions": list(map(int, session_state.metadata.positions)),
                        "nonzero_action_when_possible": bool(
                            session_state.metadata.nonzero_action_when_possible
                        ),
                        "pos0_removed": bool(session_state.metadata.pos0_removed),
                        "forced_single_candidate": bool(
                            session_state.metadata.forced_single_candidate
                        ),
                        "fallback_to_pos0_only": bool(
                            session_state.metadata.fallback_to_pos0_only
                        ),
                        "candidate_logits": logits,
                        "candidate_feature_metadata": [
                            asdict(row) for row in session_state.features.metadata
                        ],
                    }
                )

        policy_input_metadata = self._policy_input_metadata()
        return {
            "target_item": self._target_item,
            "policy_representation": "shared_contextual_mlp",
            "exported_policy_source": str(exported_policy_source),
            "nonzero_action_when_possible": bool(
                self.position_opt_config.nonzero_action_when_possible
            ),
            "candidate_space_diagnostics": self._candidate_space_diagnostics_payload(),
            "final_position_diagnostics": _build_final_position_diagnostics(
                final_selected_positions
            ),
            "deterministic_eval_every": int(self.position_opt_config.deterministic_eval_every),
            "deterministic_eval_include_final": bool(
                self.position_opt_config.deterministic_eval_include_final
            ),
            "final_policy_selection": str(self.position_opt_config.final_policy_selection),
            "best_deterministic_step": (
                None
                if best_deterministic_checkpoint is None
                else int(best_deterministic_checkpoint.outer_step)
            ),
            "best_deterministic_reward": (
                None
                if best_deterministic_checkpoint is None
                else float(best_deterministic_checkpoint.reward)
            ),
            "last_deterministic_reward": last_deterministic_reward,
            "best_minus_last_deterministic_reward": best_minus_last_deterministic_reward,
            "exported_selected_positions": [
                asdict(result) for result in final_selected_positions
            ],
            "policy_config": {
                "policy_embedding_dim": policy_input_metadata["policy_embedding_dim"],
                "policy_hidden_dim": policy_input_metadata["policy_hidden_dim"],
                "policy_input_dim": policy_input_metadata["policy_input_dim"],
                "num_item_embeddings": int(self.policy.num_item_embeddings),
                "policy_feature_set": str(self.position_opt_config.policy_feature_set),
                "active_item_features": policy_input_metadata["active_item_features"],
                "active_scalar_features": policy_input_metadata["active_scalar_features"],
                "item_feature_names": policy_input_metadata["active_item_features"],
                "scalar_feature_names": list(self.policy.scalar_feature_names),
            },
            "special_item_ids": self._policy_special_item_ids.to_payload(),
            **self._prefix_score_metadata(),
            "sessions": sessions_payload,
            "policy_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.policy.state_dict().items()
            },
        }


def _build_session_states(
    fake_sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    replacement_topk_ratio: float,
    nonzero_action_when_possible: bool,
) -> tuple[list[_SessionCandidateState], PolicySpecialItemIds]:
    special_item_ids = build_policy_special_item_ids(
        infer_max_item_id(fake_sessions, target_item=target_item)
    )
    session_states: list[_SessionCandidateState] = []
    for session in fake_sessions:
        session_list = list(session)
        candidate_build_result = build_candidate_position_result(
            session_list,
            replacement_topk_ratio,
            nonzero_action_when_possible=nonzero_action_when_possible,
        )
        candidate_positions = list(candidate_build_result.positions)
        metadata = CandidateMetadata(
            session_length=len(session_list),
            replacement_topk_ratio=float(replacement_topk_ratio),
            positions_before_mask=tuple(
                int(position) for position in candidate_build_result.positions_before_mask
            ),
            positions=tuple(int(position) for position in candidate_positions),
            nonzero_action_when_possible=bool(
                candidate_build_result.nonzero_action_when_possible
            ),
            pos0_removed=bool(candidate_build_result.pos0_removed),
            forced_single_candidate=bool(candidate_build_result.forced_single_candidate),
            fallback_to_pos0_only=bool(candidate_build_result.fallback_to_pos0_only),
        )
        features = build_session_candidate_features(
            session_list,
            candidate_positions,
            target_item=target_item,
            special_item_ids=special_item_ids,
        )
        candidate_sessions = [
            replace_item_at_position(session_list, position, target_item)
            for position in candidate_positions
        ]
        session_states.append(
            _SessionCandidateState(
                original_session=session_list,
                metadata=metadata,
                features=features,
                candidate_sessions=candidate_sessions,
            )
        )
    return session_states, special_item_ids


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


def _resolve_deterministic_eval_schedule(
    *,
    total_outer_steps: int,
    deterministic_eval_every: int,
    deterministic_eval_include_final: bool,
) -> list[int]:
    if total_outer_steps < 0:
        raise ValueError("total_outer_steps must be non-negative.")
    if deterministic_eval_every < 0:
        raise ValueError("deterministic_eval_every must be non-negative.")
    if deterministic_eval_every == 0 or total_outer_steps == 0:
        return []

    schedule = list(
        range(
            int(deterministic_eval_every),
            int(total_outer_steps) + 1,
            int(deterministic_eval_every),
        )
    )
    if deterministic_eval_include_final and (
        not schedule or schedule[-1] != int(total_outer_steps)
    ):
        schedule.append(int(total_outer_steps))
    return schedule


def _selected_position_pct(
    selected_positions: Sequence[int],
    *,
    max_position: int,
) -> float:
    if not selected_positions:
        return 0.0
    match_count = sum(1 for position in selected_positions if int(position) <= int(max_position))
    return float(match_count) / float(len(selected_positions)) * 100.0


def _build_candidate_space_diagnostics(
    session_states: Sequence[_SessionCandidateState],
) -> dict[str, Any]:
    total_session_count = int(len(session_states))
    if total_session_count == 0:
        return {
            "total_session_count": 0,
            "pos0_removed_session_count": 0,
            "pos0_removed_pct": 0.0,
            "forced_single_candidate_count": 0,
            "forced_single_candidate_pct": 0.0,
            "fallback_to_pos0_only_count": 0,
            "fallback_to_pos0_only_pct": 0.0,
            "mean_candidate_count_before_mask": 0.0,
            "mean_candidate_count_after_mask": 0.0,
            "min_candidate_count_after_mask": 0,
            "max_candidate_count_after_mask": 0,
        }

    before_counts = [
        int(len(session_state.metadata.positions_before_mask))
        for session_state in session_states
    ]
    after_counts = [
        int(len(session_state.metadata.positions))
        for session_state in session_states
    ]
    pos0_removed_session_count = sum(
        1 for session_state in session_states if bool(session_state.metadata.pos0_removed)
    )
    forced_single_candidate_count = sum(
        1
        for session_state in session_states
        if bool(session_state.metadata.forced_single_candidate)
    )
    fallback_to_pos0_only_count = sum(
        1
        for session_state in session_states
        if bool(session_state.metadata.fallback_to_pos0_only)
    )
    return {
        "total_session_count": total_session_count,
        "pos0_removed_session_count": int(pos0_removed_session_count),
        "pos0_removed_pct": (
            float(pos0_removed_session_count) / float(total_session_count) * 100.0
        ),
        "forced_single_candidate_count": int(forced_single_candidate_count),
        "forced_single_candidate_pct": (
            float(forced_single_candidate_count) / float(total_session_count) * 100.0
        ),
        "fallback_to_pos0_only_count": int(fallback_to_pos0_only_count),
        "fallback_to_pos0_only_pct": (
            float(fallback_to_pos0_only_count) / float(total_session_count) * 100.0
        ),
        "mean_candidate_count_before_mask": (
            float(sum(before_counts)) / float(total_session_count)
        ),
        "mean_candidate_count_after_mask": (
            float(sum(after_counts)) / float(total_session_count)
        ),
        "min_candidate_count_after_mask": int(min(after_counts)),
        "max_candidate_count_after_mask": int(max(after_counts)),
    }


def _build_final_position_diagnostics(
    selected_positions: Sequence[SelectedPositionResult] | Sequence[int],
) -> dict[str, Any]:
    normalized_positions: list[int] = []
    for value in selected_positions:
        if isinstance(value, SelectedPositionResult):
            normalized_positions.append(int(value.position))
        else:
            normalized_positions.append(int(value))

    total_count = int(len(normalized_positions))
    if total_count == 0:
        return {
            "final_pos0_pct": 0.0,
            "final_pos1_pct": 0.0,
            "final_pos_leq_2_pct": 0.0,
            "dominant_position": None,
            "top5_positions": [],
        }

    counts = Counter(normalized_positions)
    top5_positions = [
        {
            "position": int(position),
            "count": int(count),
            "pct": float(count) / float(total_count) * 100.0,
        }
        for position, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), int(item[0])),
        )[:5]
    ]
    dominant_position = (
        None
        if not top5_positions
        else int(top5_positions[0]["position"])
    )
    return {
        "final_pos0_pct": _selected_position_pct(normalized_positions, max_position=0),
        "final_pos1_pct": (
            float(counts.get(1, 0)) / float(total_count) * 100.0
        ),
        "final_pos_leq_2_pct": _selected_position_pct(normalized_positions, max_position=2),
        "dominant_position": dominant_position,
        "top5_positions": top5_positions,
    }


def _build_policy_loss(
    *,
    joint_log_prob: torch.Tensor,
    advantage_tensor: torch.Tensor,
    joint_entropy: torch.Tensor,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(entropy_coef) < 0.0:
        raise ValueError("entropy_coef must be non-negative.")
    reinforce_loss = -(joint_log_prob * advantage_tensor)
    entropy_loss = -joint_log_prob.new_tensor(float(entropy_coef)) * joint_entropy
    return reinforce_loss + entropy_loss, reinforce_loss, entropy_loss


def _resolve_reward_target_utility(
    *,
    reward_mode: str,
    poisoned_target_utility: float,
    clean_target_utility: float | None,
) -> float:
    if reward_mode == "poisoned_target_utility":
        return float(poisoned_target_utility)
    if reward_mode in {"delta_target_utility", "delta_lowk_rank_utility"}:
        if clean_target_utility is None:
            raise ValueError(
                f"clean_target_utility is required when reward_mode={reward_mode!r}."
            )
        return float(poisoned_target_utility) - float(clean_target_utility)
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")


def _resolve_scored_target_utility(
    *,
    reward_mode: str,
    target_result: SurrogateScoreResult,
) -> float:
    if reward_mode in {"poisoned_target_utility", "delta_target_utility"}:
        return float(target_result.mean)
    if reward_mode == "delta_lowk_rank_utility":
        return _compute_lowk_target_utility(target_result.metrics)
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")


def _compute_lowk_target_utility(metrics: Mapping[str, float] | None) -> float:
    resolved_metrics = _coerce_lowk_target_metrics(metrics, required=True)
    return float(
        sum(
            _LOWK_TARGET_METRIC_WEIGHTS[metric_key] * float(resolved_metrics[metric_key])
            for metric_key in _LOWK_TARGET_METRIC_KEYS
        )
    )


def _extract_lowk_target_metrics(
    target_result: SurrogateScoreResult,
    *,
    required: bool,
) -> dict[str, float | None]:
    return _coerce_lowk_target_metrics(target_result.metrics, required=required)


def _coerce_lowk_target_metrics(
    metrics: Mapping[str, float] | None,
    *,
    required: bool,
) -> dict[str, float | None]:
    normalized = _empty_lowk_target_metrics()
    source = {} if metrics is None else dict(metrics)
    missing_keys: list[str] = []
    for metric_key in _LOWK_TARGET_METRIC_KEYS:
        value = source.get(metric_key)
        if value is None:
            if required:
                missing_keys.append(metric_key)
            continue
        normalized[metric_key] = float(value)
    if missing_keys:
        raise ValueError(
            "Missing low-k target metrics: " + ", ".join(sorted(missing_keys))
        )
    return normalized


def _empty_lowk_target_metrics() -> dict[str, float | None]:
    return {metric_key: None for metric_key in _LOWK_TARGET_METRIC_KEYS}


def _lowk_target_metric_history_fields(
    metrics: Mapping[str, float | None] | None,
    *,
    prefix: str,
) -> dict[str, float | None]:
    normalized = _coerce_lowk_target_metrics(metrics, required=False)
    return {
        f"{prefix}_{metric_key.replace('@', '_at_')}": (
            None if normalized[metric_key] is None else float(normalized[metric_key])
        )
        for metric_key in _LOWK_TARGET_METRIC_KEYS
    }


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
