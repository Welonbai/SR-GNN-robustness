from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.config import Config, PositionOptConfig, RankBucketCEMConfig
from attack.common.seed import derive_seed, set_seed
from attack.data.poisoned_dataset_builder import (
    build_poisoned_dataset,
    expand_session_to_samples,
)
from attack.inner_train.base import InnerTrainer
from attack.position_opt.objective import compute_position_opt_objective
from attack.position_opt.poison_builder import replace_item_at_position
from attack.position_opt.types import (
    InnerTrainResult,
    SurrogateScoreResult,
    TruncatedFineTuneConfig,
)
from attack.surrogate.base import SurrogateBackend

from .artifacts import (
    RankBucketCEMArtifactPaths,
    ensure_rank_bucket_cem_artifact_dirs,
    save_availability_summary,
    save_cem_best_policy,
    save_cem_state_history,
    save_final_position_summary,
    save_optimized_poisoned_sessions,
    write_cem_trace_jsonl,
    write_selected_positions_jsonl,
)
from .availability import (
    RankCandidateSessionState,
    build_availability_summary,
    build_rank_candidate_states,
)
from .optimizer import (
    CEMCandidateResult,
    CEMState,
    initialize_cem_state,
    sample_cem_candidates,
    update_cem_state,
)
from .rank_policy import (
    RankBucketPolicy,
    RankBucketSelectionRecord,
    build_rank_position_summary,
    sample_positions_from_rank_policy,
)


RANK_BUCKET_CEM_METHOD_NAME = "rank_bucket_cem"
RANK_BUCKET_CEM_IMPLEMENTATION_TAG = "rank_bucket_cem_v1"
_SUPPORTED_REWARD_MODES = frozenset(("poisoned_target_utility", "delta_target_utility"))
_LOG_PREFIX = "[rank-bucket-cem]"
_TRACE_OPTIONAL_METRIC_KEYS = (
    "clean_gt_utility",
    "poisoned_gt_utility",
    "gt_penalty",
    "gt_drop",
    "poison_balance_enabled",
    "poison_balance_mode",
    "requested_poison_ratio_in_batch",
    "configured_clean_batch_size",
    "configured_poison_batch_size",
    "effective_poison_ratio_seen",
    "configured_fine_tune_steps",
    "actual_optimizer_steps",
    "batch_size",
    "clean_pool_size",
    "poison_pool_size",
    "clean_examples_seen",
    "poison_examples_seen",
    "unique_poison_prefix_examples_seen",
    "unique_poison_source_sessions_seen",
    "sampling_strategy",
    "sampling_wrapped",
    "fine_tune_seconds",
    "score_target_seconds",
    "candidate_total_seconds",
)


class RankBucketCEMTrainer:
    def __init__(
        self,
        surrogate_backend: SurrogateBackend,
        inner_trainer: InnerTrainer,
        *,
        clean_surrogate_checkpoint_path: str | Path,
        position_opt_config: PositionOptConfig,
        rank_bucket_cem_config: RankBucketCEMConfig,
    ) -> None:
        checkpoint_path = Path(clean_surrogate_checkpoint_path)
        if not str(checkpoint_path).strip():
            raise ValueError("clean_surrogate_checkpoint_path must be provided explicitly.")

        self.surrogate_backend = surrogate_backend
        self.inner_trainer = inner_trainer
        self.clean_surrogate_checkpoint_path = checkpoint_path
        self.position_opt_config = position_opt_config
        self.rank_bucket_cem_config = rank_bucket_cem_config

        self._trained_config: Config | None = None
        self._target_item: int | None = None
        self._replacement_topk_ratio: float | None = None
        self._rank_candidate_states: list[RankCandidateSessionState] = []
        self._availability_summary: dict[str, Any] | None = None
        self._cem_trace_rows: list[dict[str, Any]] = []
        self._cem_state_history: list[dict[str, Any]] = []
        self._best_candidate_result: CEMCandidateResult | None = None
        self._best_selection_records: list[RankBucketSelectionRecord] | None = None
        self._best_poisoned_sessions: list[list[int]] | None = None
        self._best_selection_seed: int | None = None
        self._final_position_summary: dict[str, Any] | None = None
        self._clean_target_result_mean: float | None = None
        self._clean_target_metrics: dict[str, float] = {}
        self._clean_reward_baseline: float | None = None
        self._shared_surrogate_train_seed: int | None = None
        self._validation_subset_strategy: str = "full_validation_set"
        self._validation_subset_seed: int | None = None
        self._validation_subset_effective_size: int | None = None
        self._artifact_paths: RankBucketCEMArtifactPaths | None = None

    def train(
        self,
        fake_sessions: Sequence[Sequence[int]],
        target_item: int,
        shared_artifacts: object,
        config: Config,
        *,
        artifact_paths: RankBucketCEMArtifactPaths | None = None,
    ) -> dict[str, Any]:
        normalized_fake_sessions = _normalize_fake_sessions(fake_sessions)
        target_id = int(target_item)
        if target_id <= 0:
            raise ValueError("target_item must be a positive item id.")
        _validate_reward_mode(self.position_opt_config.reward_mode)

        clean_sessions, clean_labels = _resolve_clean_pairs(shared_artifacts)
        validation_sessions, validation_labels = _resolve_validation_pairs(shared_artifacts)
        validation_total_size = len(validation_sessions)
        validation_subset_seed = (
            None
            if self.position_opt_config.validation_subset_size is None
            else derive_seed(
                config.seeds.position_opt_seed,
                "rank_bucket_cem_validation_subset",
                target_id,
            )
        )
        (
            validation_sessions,
            validation_labels,
            validation_subset_metadata,
        ) = _select_validation_subset(
            validation_sessions,
            validation_labels,
            subset_size=self.position_opt_config.validation_subset_size,
            subset_seed=validation_subset_seed,
        )

        paths = None
        if artifact_paths is not None:
            paths = ensure_rank_bucket_cem_artifact_dirs(artifact_paths)
        self._artifact_paths = paths

        self._trained_config = config
        self._target_item = target_id
        self._replacement_topk_ratio = float(config.attack.replacement_topk_ratio)
        self._validation_subset_strategy = str(validation_subset_metadata["strategy"])
        self._validation_subset_seed = validation_subset_metadata["seed"]
        self._validation_subset_effective_size = int(validation_subset_metadata["selected_count"])
        self._rank_candidate_states = build_rank_candidate_states(
            normalized_fake_sessions,
            config.attack.replacement_topk_ratio,
            bool(self.position_opt_config.nonzero_action_when_possible),
        )
        self._availability_summary = build_availability_summary(self._rank_candidate_states)
        _print_cem_start(
            target_item=target_id,
            config=self.rank_bucket_cem_config,
            position_opt_config=self.position_opt_config,
            validation_subset_effective_size=self._validation_subset_effective_size,
            validation_subset_total_size=validation_total_size,
            reward_metric_name=self._selected_reward_metric_name(),
        )
        _print_availability_summary(
            target_item=target_id,
            summary=self._availability_summary,
            nonzero_action_when_possible=bool(
                self.position_opt_config.nonzero_action_when_possible
            ),
            replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
        )
        self._cem_trace_rows = []
        self._cem_state_history = []
        self._best_candidate_result = None
        self._best_selection_records = None
        self._best_poisoned_sessions = None
        self._best_selection_seed = None
        self._final_position_summary = None
        self._clean_target_result_mean = None
        self._clean_target_metrics = {}
        self._clean_reward_baseline = None
        self._shared_surrogate_train_seed = derive_seed(
            config.seeds.surrogate_train_seed,
            "rank_bucket_cem_surrogate_train",
            target_id,
        )

        if paths is not None:
            save_availability_summary(paths.availability_summary, self._availability_summary)

        fine_tune_config = TruncatedFineTuneConfig(
            steps=int(self.position_opt_config.fine_tune_steps),
            epochs=1,
        )
        clean_target_result = self._precompute_clean_target_result(
            validation_sessions=validation_sessions,
            target_item=target_id,
        )
        self._clean_target_result_mean = float(clean_target_result.mean)
        self._clean_target_metrics = _coerce_target_metrics(clean_target_result.metrics)
        self._clean_reward_baseline = _resolve_reward_value(
            target_result=clean_target_result,
            reward_metric=self.rank_bucket_cem_config.reward_metric,
        )
        clean_gt_utility = None
        if self.position_opt_config.enable_gt_penalty:
            clean_gt_utility = self._precompute_clean_gt_utility(
                validation_sessions=validation_sessions,
                validation_labels=validation_labels,
            )

        state = initialize_cem_state(self.rank_bucket_cem_config.initial_std)
        self._cem_state_history.append(
            self._state_history_entry(
                iteration=-1,
                state=state,
                best_reward_so_far=None,
                elite_count=None,
            )
        )
        if paths is not None:
            save_cem_state_history(paths.cem_state_history, self._cem_state_history_payload())

        candidate_rng = random.Random(
            derive_seed(
                config.seeds.position_opt_seed,
                "rank_bucket_cem_candidate_sampling",
                target_id,
            )
        )

        for iteration in range(int(self.rank_bucket_cem_config.iterations)):
            candidate_results: list[CEMCandidateResult] = []
            candidates = sample_cem_candidates(
                state,
                iteration=iteration,
                population_size=int(self.rank_bucket_cem_config.population_size),
                rng=candidate_rng,
            )
            for candidate in candidates:
                policy = RankBucketPolicy(
                    pi_g2=tuple(float(value) for value in candidate.pi_g2),
                    pi_g3=tuple(float(value) for value in candidate.pi_g3),
                )
                selection_seed = derive_seed(
                    config.seeds.position_opt_seed,
                    "rank_bucket_cem_position_sampling",
                    target_id,
                    int(iteration),
                    int(candidate.candidate_id),
                )
                selected_positions, selection_records = sample_positions_from_rank_policy(
                    self._rank_candidate_states,
                    policy,
                    target_id,
                    random.Random(selection_seed),
                )
                poisoned_fake_sessions = [
                    replace_item_at_position(state_row.original_session, position, target_id)
                    for state_row, position in zip(
                        self._rank_candidate_states,
                        selected_positions,
                    )
                ]
                candidate_start_time = time.perf_counter()
                poison_balance_enabled = bool(
                    self.rank_bucket_cem_config.surrogate_eval_poison_balance.enabled
                )
                poisoned_train_data = (
                    None
                    if poison_balance_enabled
                    else build_poisoned_dataset(
                        clean_sessions,
                        clean_labels,
                        poisoned_fake_sessions,
                    )
                )
                fine_tune_start_time = time.perf_counter()
                if poison_balance_enabled:
                    inner_result = self._run_fixed_ratio_surrogate_fine_tune(
                        clean_sessions=clean_sessions,
                        clean_labels=clean_labels,
                        poisoned_fake_sessions=poisoned_fake_sessions,
                        fine_tune_config=fine_tune_config,
                        seed=int(self._shared_surrogate_train_seed),
                    )
                else:
                    if poisoned_train_data is None:
                        raise RuntimeError("Normal CEM fine-tune missing poisoned_train_data.")
                    inner_result = self.inner_trainer.run(
                        self.surrogate_backend,
                        self.clean_surrogate_checkpoint_path,
                        poisoned_train_data,
                        config=fine_tune_config,
                        seed=self._shared_surrogate_train_seed,
                    )
                fine_tune_seconds = time.perf_counter() - fine_tune_start_time
                surrogate_model = inner_result.model
                score_target_start_time = time.perf_counter()
                target_result = self.surrogate_backend.score_target(
                    surrogate_model,
                    validation_sessions,
                    target_id,
                )
                score_target_seconds = time.perf_counter() - score_target_start_time
                target_metrics = _coerce_target_metrics(target_result.metrics)
                reward_value = _resolve_reward_value(
                    target_result=target_result,
                    reward_metric=self.rank_bucket_cem_config.reward_metric,
                )
                reward_target_utility = _resolve_reward_target_utility(
                    reward_mode=self.position_opt_config.reward_mode,
                    reward_value=reward_value,
                    clean_reward_baseline=self._clean_reward_baseline,
                )

                poisoned_gt_utility = None
                if self.position_opt_config.enable_gt_penalty:
                    poisoned_gt_result = self.surrogate_backend.score_gt(
                        surrogate_model,
                        validation_sessions,
                        validation_labels,
                    )
                    poisoned_gt_utility = float(poisoned_gt_result.mean)
                candidate_total_seconds = time.perf_counter() - candidate_start_time

                objective = compute_position_opt_objective(
                    float(reward_target_utility),
                    clean_gt_utility=clean_gt_utility,
                    poisoned_gt_utility=poisoned_gt_utility,
                    enable_gt_penalty=bool(self.position_opt_config.enable_gt_penalty),
                    gt_penalty_weight=float(self.position_opt_config.gt_penalty_weight),
                    gt_tolerance=float(self.position_opt_config.gt_tolerance),
                )
                reward = float(objective.reward.detach().item())
                position_summary = build_rank_position_summary(selection_records)
                candidate_metrics = {
                    "reward_mode": str(self.position_opt_config.reward_mode),
                    "reward_metric": self.rank_bucket_cem_config.reward_metric,
                    "target_result_mean": float(target_result.mean),
                    "target_metrics": dict(target_metrics),
                    "selection_seed": int(selection_seed),
                    "surrogate_train_seed": int(self._shared_surrogate_train_seed),
                }
                candidate_metrics.update(
                    _candidate_fine_tune_metadata(
                        history=inner_result.history,
                        poison_balance_enabled=poison_balance_enabled,
                        configured_fine_tune_steps=int(
                            self.position_opt_config.fine_tune_steps
                        ),
                        fine_tune_seconds=fine_tune_seconds,
                        score_target_seconds=score_target_seconds,
                        candidate_total_seconds=candidate_total_seconds,
                    )
                )
                if clean_gt_utility is not None or poisoned_gt_utility is not None:
                    candidate_metrics.update(
                        {
                            "clean_gt_utility": (
                                None
                                if clean_gt_utility is None
                                else float(clean_gt_utility)
                            ),
                            "poisoned_gt_utility": (
                                None
                                if poisoned_gt_utility is None
                                else float(poisoned_gt_utility)
                            ),
                            "gt_penalty": float(objective.gt_penalty.detach().item()),
                            "gt_drop": float(objective.gt_drop.detach().item()),
                        }
                    )
                candidate_result = CEMCandidateResult(
                    candidate=candidate,
                    reward=float(reward),
                    metrics=candidate_metrics,
                )
                candidate_results.append(candidate_result)
                self._cem_trace_rows.append(
                    self._trace_row(
                        target_item=target_id,
                        candidate_result=candidate_result,
                        selection_records=selection_records,
                        position_summary=position_summary,
                    )
                )
                if (
                    self._best_candidate_result is None
                    or float(reward) > float(self._best_candidate_result.reward)
                ):
                    self._best_candidate_result = candidate_result
                    self._best_selection_records = list(selection_records)
                    self._best_poisoned_sessions = [
                        list(session) for session in poisoned_fake_sessions
                    ]
                    self._best_selection_seed = int(selection_seed)
                    self._final_position_summary = dict(position_summary)

            if paths is not None:
                write_cem_trace_jsonl(paths.cem_trace, self._cem_trace_rows)

            best_iter_reward = max(float(result.reward) for result in candidate_results)
            elite_count = _elite_count(
                len(candidate_results),
                float(self.rank_bucket_cem_config.elite_ratio),
            )
            state = update_cem_state(
                state,
                candidate_results,
                elite_ratio=float(self.rank_bucket_cem_config.elite_ratio),
                smoothing=float(self.rank_bucket_cem_config.smoothing),
                min_std=float(self.rank_bucket_cem_config.min_std),
            )
            self._cem_state_history.append(
                self._state_history_entry(
                    iteration=iteration,
                    state=state,
                    best_reward_so_far=(
                        None
                        if self._best_candidate_result is None
                        else float(self._best_candidate_result.reward)
                    ),
                    elite_count=elite_count,
                )
            )
            if paths is not None:
                save_cem_state_history(paths.cem_state_history, self._cem_state_history_payload())
            _print_iteration_summary(
                target_item=target_id,
                iteration=iteration,
                total_iterations=int(self.rank_bucket_cem_config.iterations),
                best_iter_reward=best_iter_reward,
                best_so_far=(
                    None
                    if self._best_candidate_result is None
                    else float(self._best_candidate_result.reward)
                ),
                elite_count=elite_count,
                population_size=len(candidate_results),
                state=state,
            )

        if self._best_candidate_result is None:
            raise RuntimeError("RankBucket-CEM did not evaluate any candidates.")
        if self._best_selection_records is None or self._best_poisoned_sessions is None:
            raise RuntimeError("RankBucket-CEM best candidate state is incomplete.")
        if self._final_position_summary is None:
            self._final_position_summary = build_rank_position_summary(
                self._best_selection_records
            )

        return {
            "target_item": int(target_id),
            "method_name": RANK_BUCKET_CEM_METHOD_NAME,
            "method_version": RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
            "availability_summary": dict(self._availability_summary or {}),
            "final_position_summary": dict(self._final_position_summary),
            "best_reward": float(self._best_candidate_result.reward),
            "best_iteration": int(self._best_candidate_result.candidate.iteration),
            "best_candidate_id": int(self._best_candidate_result.candidate.candidate_id),
            "best_selection_seed": self._best_selection_seed,
            "reward_mode": str(self.position_opt_config.reward_mode),
            "reward_metric": self.rank_bucket_cem_config.reward_metric,
            "selected_reward_metric_name": self._selected_reward_metric_name(),
            "clean_target_result_mean": self._clean_target_result_mean,
            "clean_target_metrics": dict(self._clean_target_metrics),
            "clean_reward_baseline": self._clean_reward_baseline,
            "validation_subset_size": (
                None
                if self.position_opt_config.validation_subset_size is None
                else int(self.position_opt_config.validation_subset_size)
            ),
            "validation_subset_effective_size": self._validation_subset_effective_size,
            "validation_subset_strategy": self._validation_subset_strategy,
            "validation_subset_seed": self._validation_subset_seed,
            "shared_surrogate_train_seed": self._shared_surrogate_train_seed,
            "cem_trace_row_count": int(len(self._cem_trace_rows)),
            "cem_state_history_length": int(len(self._cem_state_history)),
            "replay_metadata": (
                self._replay_metadata()
                if self.rank_bucket_cem_config.save_replay_metadata
                else None
            ),
        }

    def export_final_selection_records(self) -> list[RankBucketSelectionRecord]:
        if self._best_selection_records is None:
            raise RuntimeError("train() must be called before exporting final selections.")
        return list(self._best_selection_records)

    def export_final_poisoned_sessions(self) -> list[list[int]]:
        if self._best_poisoned_sessions is None:
            raise RuntimeError("train() must be called before exporting final sessions.")
        return [list(session) for session in self._best_poisoned_sessions]

    def _run_fixed_ratio_surrogate_fine_tune(
        self,
        *,
        clean_sessions: Sequence[Sequence[int]],
        clean_labels: Sequence[int],
        poisoned_fake_sessions: Sequence[Sequence[int]],
        fine_tune_config: TruncatedFineTuneConfig,
        seed: int,
    ) -> InnerTrainResult:
        fine_tune_fixed_ratio = getattr(
            self.surrogate_backend,
            "fine_tune_fixed_ratio",
            None,
        )
        if fine_tune_fixed_ratio is None:
            raise TypeError(
                "RankBucket-CEM fixed-ratio poison balance currently requires a "
                "surrogate backend with fine_tune_fixed_ratio()."
            )
        (
            poison_sessions,
            poison_labels,
            poison_source_session_ids,
        ) = _expand_poisoned_fake_sessions_with_source(poisoned_fake_sessions)
        if not clean_sessions:
            raise ValueError(
                "RankBucket-CEM fixed-ratio surrogate fine-tune requires a non-empty "
                "clean pool."
            )
        if not poison_sessions:
            raise ValueError(
                "RankBucket-CEM fixed-ratio surrogate fine-tune requires a non-empty "
                "poison pool."
            )
        set_seed(int(seed))
        self.surrogate_backend.load_clean_checkpoint(self.clean_surrogate_checkpoint_path)
        model = self.surrogate_backend.clone_clean_model()
        history = fine_tune_fixed_ratio(
            model,
            clean_sessions=clean_sessions,
            clean_labels=clean_labels,
            poison_sessions=poison_sessions,
            poison_labels=poison_labels,
            poison_source_session_ids=poison_source_session_ids,
            fine_tune_config=fine_tune_config,
            poison_ratio_in_batch=float(
                self.rank_bucket_cem_config.surrogate_eval_poison_balance.poison_ratio_in_batch
            ),
            seed=int(seed),
        )
        return InnerTrainResult(model=model, history=history)

    def save_artifacts(
        self,
        artifact_paths: RankBucketCEMArtifactPaths | None = None,
    ) -> None:
        paths = artifact_paths or self._artifact_paths
        if paths is None:
            raise ValueError("artifact_paths must be provided before saving artifacts.")
        if self._best_candidate_result is None:
            raise RuntimeError("train() must be called before saving artifacts.")
        if self._availability_summary is None or self._final_position_summary is None:
            raise RuntimeError("Trainer artifacts are incomplete.")

        paths = ensure_rank_bucket_cem_artifact_dirs(paths)
        self._artifact_paths = paths
        if self.rank_bucket_cem_config.save_optimized_poisoned_sessions:
            save_optimized_poisoned_sessions(
                paths.optimized_poisoned_sessions,
                self.export_final_poisoned_sessions(),
            )
        else:
            _remove_if_exists(paths.optimized_poisoned_sessions)
        save_availability_summary(paths.availability_summary, self._availability_summary)
        write_cem_trace_jsonl(paths.cem_trace, self._cem_trace_rows)
        save_cem_state_history(paths.cem_state_history, self._cem_state_history_payload())
        save_cem_best_policy(paths.cem_best_policy, self._best_policy_payload())
        if self.rank_bucket_cem_config.save_final_selected_positions:
            write_selected_positions_jsonl(
                paths.final_selected_positions,
                self.export_final_selection_records(),
            )
        else:
            _remove_if_exists(paths.final_selected_positions)
        save_final_position_summary(paths.final_position_summary, self._final_position_summary)

    def _precompute_clean_target_result(
        self,
        *,
        validation_sessions: Sequence[Sequence[int]],
        target_item: int,
    ) -> SurrogateScoreResult:
        self.surrogate_backend.load_clean_checkpoint(self.clean_surrogate_checkpoint_path)
        clean_model = self.surrogate_backend.clone_clean_model()
        return self.surrogate_backend.score_target(
            clean_model,
            validation_sessions,
            target_item,
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

    def _trace_row(
        self,
        *,
        target_item: int,
        candidate_result: CEMCandidateResult,
        selection_records: Sequence[RankBucketSelectionRecord],
        position_summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        candidate = candidate_result.candidate
        row = {
            "target_item": int(target_item),
            "iteration": int(candidate.iteration),
            "candidate_id": int(candidate.candidate_id),
            "logits_g2": [float(value) for value in candidate.logits_g2],
            "pi_g2": [float(value) for value in candidate.pi_g2],
            "logits_g3": [float(value) for value in candidate.logits_g3],
            "pi_g3": [float(value) for value in candidate.pi_g3],
            "reward": float(candidate_result.reward),
            "target_result_mean": float(candidate_result.metrics["target_result_mean"]),
            "target_metrics": dict(candidate_result.metrics["target_metrics"]),
            "position_summary": dict(position_summary),
            "selection_seed": int(candidate_result.metrics["selection_seed"]),
            "surrogate_train_seed": int(candidate_result.metrics["surrogate_train_seed"]),
        }
        for optional_key in _TRACE_OPTIONAL_METRIC_KEYS:
            if optional_key in candidate_result.metrics:
                row[optional_key] = candidate_result.metrics[optional_key]
        if self.rank_bucket_cem_config.save_candidate_selected_positions:
            row["selected_positions"] = [
                int(record.selected_position) for record in selection_records
            ]
        return row

    def _state_history_entry(
        self,
        *,
        iteration: int,
        state: CEMState,
        best_reward_so_far: float | None,
        elite_count: int | None,
    ) -> dict[str, Any]:
        return {
            "iteration": int(iteration),
            "mean_g2": [float(value) for value in state.mean_g2],
            "std_g2": [float(value) for value in state.std_g2],
            "mean_g3": [float(value) for value in state.mean_g3],
            "std_g3": [float(value) for value in state.std_g3],
            "best_reward_so_far": (
                None if best_reward_so_far is None else float(best_reward_so_far)
            ),
            "elite_count": None if elite_count is None else int(elite_count),
        }

    def _cem_state_history_payload(self) -> dict[str, Any]:
        return {
            "target_item": self._target_item,
            "method_name": RANK_BUCKET_CEM_METHOD_NAME,
            "method_version": RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
            "rank_bucket_cem_config": asdict(self.rank_bucket_cem_config),
            "resolved_seeds": (
                None
                if self._trained_config is None
                else asdict(self._trained_config.seeds)
            ),
            "validation_subset_strategy": self._validation_subset_strategy,
            "validation_subset_seed": self._validation_subset_seed,
            "shared_surrogate_train_seed": self._shared_surrogate_train_seed,
            "history": list(self._cem_state_history),
        }

    def _best_policy_payload(self) -> dict[str, Any]:
        if self._best_candidate_result is None:
            raise RuntimeError("Best candidate result is not available.")
        candidate = self._best_candidate_result.candidate
        payload = {
            "target_item": int(self._target_item or 0),
            "method_name": RANK_BUCKET_CEM_METHOD_NAME,
            "method_version": RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
            "best_iteration": int(candidate.iteration),
            "best_candidate_id": int(candidate.candidate_id),
            "best_reward": float(self._best_candidate_result.reward),
            "pi_g2": {
                "rank1": float(candidate.pi_g2[0]),
                "rank2": float(candidate.pi_g2[1]),
            },
            "pi_g3": {
                "rank1": float(candidate.pi_g3[0]),
                "rank2": float(candidate.pi_g3[1]),
                "tail": float(candidate.pi_g3[2]),
            },
            "reward_mode": str(self.position_opt_config.reward_mode),
            "reward_metric": self.rank_bucket_cem_config.reward_metric,
            "selected_reward_metric_name": self._selected_reward_metric_name(),
            "cem_hyperparameters": _cem_hyperparameters_payload(self.rank_bucket_cem_config),
            "artifact_flags": _artifact_flag_payload(self.rank_bucket_cem_config),
        }
        if self.rank_bucket_cem_config.save_replay_metadata:
            payload["replay_metadata"] = self._replay_metadata()
        return payload

    def _replay_metadata(self) -> dict[str, Any]:
        if self._trained_config is None:
            raise RuntimeError("train() must be called before replay metadata is available.")
        if self._best_candidate_result is None:
            raise RuntimeError("Best candidate result is not available.")
        return {
            "target_item": int(self._target_item or 0),
            "method_name": RANK_BUCKET_CEM_METHOD_NAME,
            "method_version": RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
            "replacement_topk_ratio": float(self._replacement_topk_ratio or 0.0),
            "nonzero_action_when_possible": bool(
                self.position_opt_config.nonzero_action_when_possible
            ),
            "candidate_construction": {
                "replacement_topk_ratio": float(self._replacement_topk_ratio or 0.0),
                "nonzero_action_when_possible": bool(
                    self.position_opt_config.nonzero_action_when_possible
                ),
                "candidate_positions_sorted": True,
                "availability_groups": {
                    "G1": "rank1_only",
                    "G2": "rank1_rank2",
                    "G3": "rank1_rank2_tail",
                },
            },
            "pi_g2": {
                "rank1": float(self._best_candidate_result.candidate.pi_g2[0]),
                "rank2": float(self._best_candidate_result.candidate.pi_g2[1]),
            },
            "pi_g3": {
                "rank1": float(self._best_candidate_result.candidate.pi_g3[0]),
                "rank2": float(self._best_candidate_result.candidate.pi_g3[1]),
                "tail": float(self._best_candidate_result.candidate.pi_g3[2]),
            },
            "final_selection_seed": (
                None if self._best_selection_seed is None else int(self._best_selection_seed)
            ),
            "position_opt_seed": int(self._trained_config.seeds.position_opt_seed),
            "surrogate_train_seed": (
                None
                if self._shared_surrogate_train_seed is None
                else int(self._shared_surrogate_train_seed)
            ),
            "cem_hyperparameters": _cem_hyperparameters_payload(self.rank_bucket_cem_config),
            "reward_mode": str(self.position_opt_config.reward_mode),
            "reward_metric": self.rank_bucket_cem_config.reward_metric,
            "selected_reward_metric_name": self._selected_reward_metric_name(),
            "surrogate_eval_poison_balance": asdict(
                self.rank_bucket_cem_config.surrogate_eval_poison_balance
            ),
            "validation_subset_strategy": self._validation_subset_strategy,
            "validation_subset_seed": self._validation_subset_seed,
        }

    def _selected_reward_metric_name(self) -> str:
        base_name = (
            "target_result.mean"
            if self.rank_bucket_cem_config.reward_metric is None
            else str(self.rank_bucket_cem_config.reward_metric)
        )
        if self.position_opt_config.reward_mode == "poisoned_target_utility":
            return base_name
        return f"delta({base_name})"


def _normalize_fake_sessions(
    fake_sessions: Sequence[Sequence[int]],
) -> list[list[int]]:
    normalized = [list(session) for session in fake_sessions]
    if not normalized:
        raise ValueError("fake_sessions must contain at least one session.")
    if any(len(session) == 0 for session in normalized):
        raise ValueError("fake_sessions must not contain empty sessions.")
    return normalized


def _expand_poisoned_fake_sessions_with_source(
    poisoned_fake_sessions: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[int], list[int]]:
    poison_sessions: list[list[int]] = []
    poison_labels: list[int] = []
    poison_source_session_ids: list[int] = []
    for source_session_id, session in enumerate(poisoned_fake_sessions):
        prefixes, labels = expand_session_to_samples(session)
        poison_sessions.extend(prefixes)
        poison_labels.extend(int(label) for label in labels)
        poison_source_session_ids.extend([int(source_session_id)] * len(prefixes))
    return poison_sessions, poison_labels, poison_source_session_ids


def _resolve_clean_pairs(shared_artifacts: object) -> tuple[list[list[int]], list[int]]:
    clean_sessions = getattr(shared_artifacts, "clean_sessions", None)
    clean_labels = getattr(shared_artifacts, "clean_labels", None)
    if clean_sessions is None or clean_labels is None:
        raise ValueError(
            "shared_artifacts must expose clean_sessions and clean_labels for "
            "RankBucket-CEM training."
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
            "validation_sessions/validation_labels for RankBucket-CEM."
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
    subset_seed: int | None,
) -> tuple[list[list[int]], list[int], dict[str, Any]]:
    sessions = [list(session) for session in validation_sessions]
    labels = [int(label) for label in validation_labels]
    total = len(sessions)
    if subset_size is None or subset_size >= total:
        return sessions, labels, {
            "strategy": "full_validation_set",
            "seed": None,
            "selected_count": int(total),
        }

    if subset_seed is None:
        subset = list(range(int(subset_size)))
        strategy = "first_n_subset"
    else:
        rng = random.Random(int(subset_seed))
        indices = list(range(total))
        rng.shuffle(indices)
        subset = sorted(indices[: int(subset_size)])
        strategy = "deterministic_random_subset"
    return (
        [sessions[index] for index in subset],
        [labels[index] for index in subset],
        {
            "strategy": strategy,
            "seed": None if subset_seed is None else int(subset_seed),
            "selected_count": int(len(subset)),
        },
    )


def _validate_reward_mode(reward_mode: str) -> None:
    if str(reward_mode) not in _SUPPORTED_REWARD_MODES:
        raise ValueError(
            "RankBucket-CEM supports only position_opt.reward_mode values "
            "'poisoned_target_utility' and 'delta_target_utility'. "
            f"Received {reward_mode!r}."
        )


def _resolve_reward_value(
    *,
    target_result: SurrogateScoreResult,
    reward_metric: str | None,
) -> float:
    if reward_metric is None:
        return float(target_result.mean)
    metrics = _coerce_target_metrics(target_result.metrics)
    if reward_metric not in metrics:
        available_keys = sorted(metrics)
        raise ValueError(
            "rank_bucket_cem.reward_metric "
            f"{reward_metric!r} was not found in target_result.metrics. "
            f"Available metric keys: {available_keys}."
        )
    return float(metrics[reward_metric])


def _candidate_fine_tune_metadata(
    *,
    history: Mapping[str, Any] | None,
    poison_balance_enabled: bool,
    configured_fine_tune_steps: int,
    fine_tune_seconds: float,
    score_target_seconds: float,
    candidate_total_seconds: float,
) -> dict[str, Any]:
    history_map = dict(history or {})
    actual_steps = history_map.get("steps")
    if poison_balance_enabled:
        poison_balance = history_map.get("poison_balance")
        if not isinstance(poison_balance, Mapping):
            poison_balance = {}
        payload = {
            key: poison_balance.get(key)
            for key in (
                "poison_balance_enabled",
                "poison_balance_mode",
                "requested_poison_ratio_in_batch",
                "configured_clean_batch_size",
                "configured_poison_batch_size",
                "effective_poison_ratio_seen",
                "configured_fine_tune_steps",
                "actual_optimizer_steps",
                "batch_size",
                "clean_pool_size",
                "poison_pool_size",
                "clean_examples_seen",
                "poison_examples_seen",
                "unique_poison_prefix_examples_seen",
                "unique_poison_source_sessions_seen",
                "sampling_strategy",
                "sampling_wrapped",
            )
        }
        payload["poison_balance_enabled"] = bool(
            payload.get("poison_balance_enabled", True)
        )
        if payload.get("configured_fine_tune_steps") is None:
            payload["configured_fine_tune_steps"] = int(configured_fine_tune_steps)
        if payload.get("actual_optimizer_steps") is None and actual_steps is not None:
            payload["actual_optimizer_steps"] = int(actual_steps)
    else:
        payload = {
            "poison_balance_enabled": False,
            "poison_balance_mode": None,
            "requested_poison_ratio_in_batch": None,
            "configured_clean_batch_size": None,
            "configured_poison_batch_size": None,
            "effective_poison_ratio_seen": None,
            "configured_fine_tune_steps": int(configured_fine_tune_steps),
            "actual_optimizer_steps": (
                None if actual_steps is None else int(actual_steps)
            ),
            "batch_size": None,
            "clean_pool_size": None,
            "poison_pool_size": None,
            "clean_examples_seen": None,
            "poison_examples_seen": None,
            "unique_poison_prefix_examples_seen": None,
            "unique_poison_source_sessions_seen": None,
            "sampling_strategy": None,
            "sampling_wrapped": None,
        }
    payload.update(
        {
            "fine_tune_seconds": float(fine_tune_seconds),
            "score_target_seconds": float(score_target_seconds),
            "candidate_total_seconds": float(candidate_total_seconds),
        }
    )
    return payload


def _resolve_reward_target_utility(
    *,
    reward_mode: str,
    reward_value: float,
    clean_reward_baseline: float | None,
) -> float:
    if reward_mode == "poisoned_target_utility":
        return float(reward_value)
    if reward_mode == "delta_target_utility":
        if clean_reward_baseline is None:
            raise ValueError(
                "clean reward baseline is required when reward_mode='delta_target_utility'."
            )
        return float(reward_value) - float(clean_reward_baseline)
    raise ValueError(f"Unsupported reward_mode: {reward_mode!r}")


def _coerce_target_metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    if metrics is None:
        return {}
    return {str(key): float(value) for key, value in dict(metrics).items()}


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


def _elite_count(result_count: int, elite_ratio: float) -> int:
    if result_count <= 0:
        raise ValueError("result_count must be positive.")
    return max(1, int((float(result_count) * float(elite_ratio)) + 0.999999999))


def _cem_hyperparameters_payload(
    config: RankBucketCEMConfig,
) -> dict[str, Any]:
    return {
        "iterations": int(config.iterations),
        "population_size": int(config.population_size),
        "elite_ratio": float(config.elite_ratio),
        "initial_std": float(config.initial_std),
        "min_std": float(config.min_std),
        "smoothing": float(config.smoothing),
        "reward_metric": config.reward_metric,
        "surrogate_eval_poison_balance": asdict(config.surrogate_eval_poison_balance),
    }


def _artifact_flag_payload(
    config: RankBucketCEMConfig,
) -> dict[str, bool]:
    return {
        "save_candidate_selected_positions": bool(config.save_candidate_selected_positions),
        "save_final_selected_positions": bool(config.save_final_selected_positions),
        "save_optimized_poisoned_sessions": bool(config.save_optimized_poisoned_sessions),
        "save_replay_metadata": bool(config.save_replay_metadata),
    }


def _remove_if_exists(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    path.unlink()


def _print_cem_start(
    *,
    target_item: int,
    config: RankBucketCEMConfig,
    position_opt_config: PositionOptConfig,
    validation_subset_effective_size: int | None,
    validation_subset_total_size: int,
    reward_metric_name: str,
) -> None:
    validation_label = (
        "full"
        if validation_subset_effective_size is None
        or int(validation_subset_effective_size) >= int(validation_subset_total_size)
        else f"{int(validation_subset_effective_size)}/{int(validation_subset_total_size)}"
    )
    print(
        f"{_LOG_PREFIX} target={int(target_item)} start CEM: "
        f"iterations={int(config.iterations)} "
        f"population_size={int(config.population_size)} "
        f"elite_ratio={float(config.elite_ratio):g} "
        f"fine_tune_steps={int(position_opt_config.fine_tune_steps)} "
        f"validation_subset={validation_label} "
        f"reward={reward_metric_name} "
        f"mode={position_opt_config.reward_mode}"
    )


def _print_availability_summary(
    *,
    target_item: int,
    summary: Mapping[str, Any],
    nonzero_action_when_possible: bool,
    replacement_topk_ratio: float,
) -> None:
    print(
        f"{_LOG_PREFIX} target={int(target_item)} candidates: "
        f"fake_sessions={int(summary.get('total_fake_sessions', 0))} "
        f"G1={_count_pct(summary, 'G1')} "
        f"G2={_count_pct(summary, 'G2')} "
        f"G3={_count_pct(summary, 'G3')} "
        f"nonzero_action={bool(nonzero_action_when_possible)} "
        f"replacement_topk_ratio={float(replacement_topk_ratio):g}"
    )


def _print_iteration_summary(
    *,
    target_item: int,
    iteration: int,
    total_iterations: int,
    best_iter_reward: float,
    best_so_far: float | None,
    elite_count: int,
    population_size: int,
    state: CEMState,
) -> None:
    pi_g2 = _softmax_values(state.mean_g2)
    pi_g3 = _softmax_values(state.mean_g3)
    print(
        f"{_LOG_PREFIX} target={int(target_item)} "
        f"iter={int(iteration) + 1}/{int(total_iterations)} "
        f"best_iter_reward={float(best_iter_reward):.6g} "
        f"best_so_far={_format_optional_reward(best_so_far)} "
        f"elite={int(elite_count)}/{int(population_size)} "
        f"pi_g2=(rank1={pi_g2[0]:.2f}, rank2={pi_g2[1]:.2f}) "
        f"pi_g3=(rank1={pi_g3[0]:.2f}, rank2={pi_g3[1]:.2f}, tail={pi_g3[2]:.2f})"
    )


def _count_pct(summary: Mapping[str, Any], prefix: str) -> str:
    count = int(summary.get(f"{prefix}_count", 0))
    pct = float(summary.get(f"{prefix}_pct", 0.0))
    return f"{count}({pct:.2f}%)"


def _format_optional_reward(value: float | None) -> str:
    return "none" if value is None else f"{float(value):.6g}"


def _softmax_values(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("values must not be empty.")
    max_value = max(float(value) for value in values)
    exp_values = [math.exp(float(value) - max_value) for value in values]
    total = sum(exp_values)
    return [float(value / total) for value in exp_values]


__all__ = [
    "RANK_BUCKET_CEM_IMPLEMENTATION_TAG",
    "RANK_BUCKET_CEM_METHOD_NAME",
    "RankBucketCEMTrainer",
    "_resolve_reward_value",
]
