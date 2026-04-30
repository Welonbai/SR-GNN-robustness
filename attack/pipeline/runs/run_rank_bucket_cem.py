from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import pickle
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import load_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.truncated_finetune import TruncatedFineTuneInnerTrainer
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts, prepare_shared_attack_artifacts
from attack.pipeline.core.position_stats import save_position_stats
from attack.position_opt import (
    position_opt_identity_payload,
    resolve_clean_surrogate_checkpoint_path,
    resolve_position_opt_config,
)
from attack.position_opt.cem import (
    RankBucketCEMTrainer,
    build_rank_bucket_cem_artifact_paths,
    build_rank_bucket_cem_attack_identity_context,
    ensure_rank_bucket_cem_artifact_dirs,
    rank_bucket_cem_identity_payload,
    resolve_rank_bucket_cem_config,
)
from attack.position_opt.cem.artifacts import save_run_metadata
from attack.surrogate.srgnn_backend import SRGNNBackend


DEFAULT_RANK_BUCKET_CEM_CONFIG_PATH = (
    "attack/configs/diginetica_attack_rank_bucket_cem.yaml"
)
_LOG_PREFIX = "[rank-bucket-cem]"


def run_rank_bucket_cem(
    config: Config,
    *,
    clean_surrogate_checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    position_opt_config: Mapping[str, Any] | None = None,
    rank_bucket_cem_config: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    _validate_rank_bucket_cem_run_config(config)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)

    resolved_position_opt_config = resolve_position_opt_config(
        config.attack.position_opt,
        position_opt_config,
    )
    resolved_rank_bucket_cem_config = resolve_rank_bucket_cem_config(
        config.attack.rank_bucket_cem,
        rank_bucket_cem_config,
    )
    _validate_rank_bucket_cem_resolved_config(resolved_position_opt_config)
    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        override=(
            str(clean_surrogate_checkpoint_path)
            if clean_surrogate_checkpoint_path is not None
            else resolved_position_opt_config.clean_surrogate_checkpoint
        ),
    ).resolve()
    if not clean_checkpoint.exists():
        raise FileNotFoundError(f"Clean surrogate checkpoint not found: {clean_checkpoint}")

    attack_identity_context = build_rank_bucket_cem_attack_identity_context(
        position_opt_config=position_opt_identity_payload(resolved_position_opt_config),
        rank_bucket_cem_config=rank_bucket_cem_identity_payload(
            resolved_rank_bucket_cem_config
        ),
        clean_surrogate_checkpoint=clean_checkpoint,
        runtime_seeds={
            "position_opt_seed": int(config.seeds.position_opt_seed),
            "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        },
    )

    print(
        f"{_LOG_PREFIX} "
        f"loaded {len(shared.template_sessions)} shared fake sessions from "
        f"{shared.shared_paths['fake_sessions']}"
    )
    print(
        f"{_LOG_PREFIX} "
        f"clean_surrogate_checkpoint={clean_checkpoint} "
        f"iterations={int(resolved_rank_bucket_cem_config.iterations)} "
        f"population_schedule="
        f"{list(resolved_rank_bucket_cem_config.effective_population_schedule)} "
        f"candidate_count={int(resolved_rank_bucket_cem_config.candidate_count)} "
        f"elite_ratio={float(resolved_rank_bucket_cem_config.elite_ratio):g} "
        f"nonzero_action_when_possible="
        f"{bool(resolved_position_opt_config.nonzero_action_when_possible)}"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        artifact_paths = ensure_rank_bucket_cem_artifact_dirs(
            build_rank_bucket_cem_artifact_paths(
                config,
                run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
                target_item=target_item,
                clean_checkpoint_override=clean_checkpoint,
                attack_identity_context=attack_identity_context,
            )
        )
        cached_output = _load_cached_rank_bucket_cem_target_output(
            artifact_paths=artifact_paths,
            shared=shared,
            target_item=target_item,
            rank_bucket_cem_config=resolved_rank_bucket_cem_config,
        )
        if cached_output is not None:
            print(
                f"{_LOG_PREFIX} "
                f"target={int(target_item)} reusing CEM artifacts from "
                f"{artifact_paths.base_dir}"
            )
            return cached_output

        surrogate_backend = SRGNNBackend(config, base_dir=Path.cwd())
        inner_trainer = TruncatedFineTuneInnerTrainer()
        trainer = RankBucketCEMTrainer(
            surrogate_backend,
            inner_trainer,
            clean_surrogate_checkpoint_path=artifact_paths.clean_surrogate_checkpoint,
            position_opt_config=resolved_position_opt_config,
            rank_bucket_cem_config=resolved_rank_bucket_cem_config,
        )
        trainer_result = trainer.train(
            shared.template_sessions,
            target_item,
            shared,
            config,
            artifact_paths=artifact_paths,
        )
        trainer.save_artifacts(artifact_paths)
        final_selection_records = trainer.export_final_selection_records()
        optimized_poisoned_sessions = trainer.export_final_poisoned_sessions()

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            optimized_poisoned_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=[int(record.selected_position) for record in final_selection_records],
            run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
            target_item=int(target_item),
        )
        _save_rank_bucket_cem_run_metadata(
            artifact_paths=artifact_paths,
            shared=shared,
            target_item=target_item,
            trainer=trainer,
            trainer_result=trainer_result,
            config=config,
            clean_checkpoint=clean_checkpoint,
        )
        print(
            f"{_LOG_PREFIX} "
            f"target={int(target_item)} done "
            f"best_reward={float(trainer_result['best_reward']):.6g} "
            f"final_selection_reward="
            f"{trainer_result.get('final_selection_reward_name')}:"
            f"{trainer_result.get('final_selection_reward_value')} "
            f"best_iter={int(trainer_result['best_iteration'])} "
            f"best_candidate={int(trainer_result['best_candidate_id'])} "
            f"final_pos={_format_final_position_summary(trainer_result)} "
            f"artifacts={artifact_paths.base_dir}"
        )

        metadata = {
            "position_opt_method": POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
            "position_stats_path": str(position_stats_path),
            "position_opt_clean_surrogate_checkpoint": str(
                artifact_paths.clean_surrogate_checkpoint
            ),
            "position_opt_optimized_poisoned_sessions_path": (
                str(artifact_paths.optimized_poisoned_sessions)
                if resolved_rank_bucket_cem_config.save_optimized_poisoned_sessions
                else None
            ),
            "rank_bucket_cem_artifact_dir": str(artifact_paths.base_dir),
            "rank_bucket_cem_availability_summary_path": str(
                artifact_paths.availability_summary
            ),
            "rank_bucket_cem_trace_path": str(artifact_paths.cem_trace),
            "rank_bucket_cem_state_history_path": str(artifact_paths.cem_state_history),
            "rank_bucket_cem_best_policy_path": str(artifact_paths.cem_best_policy),
            "rank_bucket_cem_final_selected_positions_path": (
                str(artifact_paths.final_selected_positions)
                if resolved_rank_bucket_cem_config.save_final_selected_positions
                else None
            ),
            "rank_bucket_cem_final_position_summary_path": str(
                artifact_paths.final_position_summary
            ),
            "rank_bucket_cem_run_metadata_path": (
                None
                if artifact_paths.run_metadata is None
                else str(artifact_paths.run_metadata)
            ),
            "rank_bucket_cem_best_reward": trainer_result.get("best_reward"),
            "rank_bucket_cem_best_reward_name": trainer_result.get("best_reward_name"),
            "rank_bucket_cem_best_iteration_reward": trainer_result.get(
                "best_iteration_reward"
            ),
            "rank_bucket_cem_final_selection_reward_name": trainer_result.get(
                "final_selection_reward_name"
            ),
            "rank_bucket_cem_final_selection_reward_value": trainer_result.get(
                "final_selection_reward_value"
            ),
            "rank_bucket_cem_reward_mode": trainer_result.get("reward_mode"),
            "rank_bucket_cem_reward_metric": trainer_result.get("reward_metric"),
            "rank_bucket_cem_selected_reward_metric_name": trainer_result.get(
                "selected_reward_metric_name"
            ),
            "rank_bucket_cem_replay_metadata": trainer_result.get("replay_metadata"),
            "rank_bucket_cem_availability_summary": trainer_result.get(
                "availability_summary"
            ),
            "rank_bucket_cem_final_position_summary": trainer_result.get(
                "final_position_summary"
            ),
            "rank_bucket_cem_clean_target_result_mean": trainer_result.get(
                "clean_target_result_mean"
            ),
            "rank_bucket_cem_clean_target_metrics": trainer_result.get(
                "clean_target_metrics"
            ),
        }
        return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)

    summary = run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )
    print(f"{_LOG_PREFIX} Final victim evaluation completed.")
    return summary


def _validate_rank_bucket_cem_run_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("RankBucket-CEM run expects data.poison_train_only to be true.")
    if config.attack.position_opt is None:
        raise ValueError(
            "RankBucket-CEM run requires attack.position_opt for clean-surrogate and "
            "fine-tune settings."
        )
    if config.attack.rank_bucket_cem is None:
        raise ValueError(
            "RankBucket-CEM run requires attack.rank_bucket_cem to be configured."
        )
    if not bool(config.attack.position_opt.nonzero_action_when_possible):
        raise ValueError(
            "RankBucket-CEM currently requires "
            "attack.position_opt.nonzero_action_when_possible == true."
        )


def _validate_rank_bucket_cem_resolved_config(position_opt_config) -> None:
    if str(position_opt_config.reward_mode) not in {
        "poisoned_target_utility",
        "delta_target_utility",
    }:
        raise ValueError(
            "RankBucket-CEM supports only attack.position_opt.reward_mode values "
            "'poisoned_target_utility' and 'delta_target_utility'."
        )


def _load_cached_rank_bucket_cem_target_output(
    *,
    artifact_paths,
    shared: SharedAttackArtifacts,
    target_item: int,
    rank_bucket_cem_config,
) -> TargetPoisonOutput | None:
    if not bool(rank_bucket_cem_config.save_optimized_poisoned_sessions):
        return None
    required_paths = (
        artifact_paths.optimized_poisoned_sessions,
        artifact_paths.availability_summary,
        artifact_paths.cem_trace,
        artifact_paths.cem_state_history,
        artifact_paths.cem_best_policy,
        artifact_paths.final_position_summary,
        artifact_paths.run_metadata,
    )
    if any(path is None or not Path(path).exists() for path in required_paths):
        return None

    with Path(artifact_paths.optimized_poisoned_sessions).open("rb") as handle:
        optimized_poisoned_sessions = pickle.load(handle)
    if not isinstance(optimized_poisoned_sessions, list):
        raise ValueError(
            "Cached RankBucket-CEM optimized_poisoned_sessions.pkl must contain a list."
        )

    poisoned = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        optimized_poisoned_sessions,
    )
    run_metadata = load_json(artifact_paths.run_metadata) or {}
    if not isinstance(run_metadata, Mapping):
        raise ValueError("Cached RankBucket-CEM run_metadata.json must be a JSON object.")
    trainer_result = run_metadata.get("trainer_result", {})
    if not isinstance(trainer_result, Mapping):
        trainer_result = {}

    metadata = _rank_bucket_cem_target_metadata_from_result(
        artifact_paths=artifact_paths,
        target_item=target_item,
        trainer_result=trainer_result,
        save_optimized_poisoned_sessions=bool(
            rank_bucket_cem_config.save_optimized_poisoned_sessions
        ),
        save_final_selected_positions=bool(
            rank_bucket_cem_config.save_final_selected_positions
        ),
    )
    metadata["rank_bucket_cem_reused_artifacts"] = True
    return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)


def _rank_bucket_cem_target_metadata_from_result(
    *,
    artifact_paths,
    target_item: int,
    trainer_result: Mapping[str, Any],
    save_optimized_poisoned_sessions: bool,
    save_final_selected_positions: bool,
) -> dict[str, Any]:
    position_stats_path = artifact_paths.base_dir.parent.parent / "position_stats.json"
    return {
        "position_opt_method": POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        "position_stats_path": str(position_stats_path),
        "position_opt_clean_surrogate_checkpoint": str(
            artifact_paths.clean_surrogate_checkpoint
        ),
        "position_opt_optimized_poisoned_sessions_path": (
            str(artifact_paths.optimized_poisoned_sessions)
            if save_optimized_poisoned_sessions
            else None
        ),
        "rank_bucket_cem_artifact_dir": str(artifact_paths.base_dir),
        "rank_bucket_cem_availability_summary_path": str(
            artifact_paths.availability_summary
        ),
        "rank_bucket_cem_trace_path": str(artifact_paths.cem_trace),
        "rank_bucket_cem_state_history_path": str(artifact_paths.cem_state_history),
        "rank_bucket_cem_best_policy_path": str(artifact_paths.cem_best_policy),
        "rank_bucket_cem_final_selected_positions_path": (
            str(artifact_paths.final_selected_positions)
            if save_final_selected_positions
            else None
        ),
        "rank_bucket_cem_final_position_summary_path": str(
            artifact_paths.final_position_summary
        ),
        "rank_bucket_cem_run_metadata_path": (
            None
            if artifact_paths.run_metadata is None
            else str(artifact_paths.run_metadata)
        ),
        "rank_bucket_cem_best_reward": trainer_result.get("best_reward"),
        "rank_bucket_cem_best_reward_name": trainer_result.get("best_reward_name"),
        "rank_bucket_cem_best_iteration_reward": trainer_result.get(
            "best_iteration_reward"
        ),
        "rank_bucket_cem_final_selection_reward_name": trainer_result.get(
            "final_selection_reward_name"
        ),
        "rank_bucket_cem_final_selection_reward_value": trainer_result.get(
            "final_selection_reward_value"
        ),
        "rank_bucket_cem_reward_mode": trainer_result.get("reward_mode"),
        "rank_bucket_cem_reward_metric": trainer_result.get("reward_metric"),
        "rank_bucket_cem_selected_reward_metric_name": trainer_result.get(
            "selected_reward_metric_name"
        ),
        "rank_bucket_cem_replay_metadata": trainer_result.get("replay_metadata"),
        "rank_bucket_cem_availability_summary": trainer_result.get(
            "availability_summary"
        ),
        "rank_bucket_cem_final_position_summary": trainer_result.get(
            "final_position_summary"
        ),
        "rank_bucket_cem_clean_target_result_mean": trainer_result.get(
            "clean_target_result_mean"
        ),
        "rank_bucket_cem_clean_target_metrics": trainer_result.get(
            "clean_target_metrics"
        ),
    }


def _format_final_position_summary(trainer_result: Mapping[str, Any]) -> str:
    summary = trainer_result.get("final_position_summary")
    if not isinstance(summary, Mapping):
        return "(unavailable)"
    return (
        "("
        f"rank1={float(summary.get('rank1_pct', 0.0)):.1f}%, "
        f"rank2={float(summary.get('rank2_pct', 0.0)):.1f}%, "
        f"tail={float(summary.get('tail_pct', 0.0)):.1f}%, "
        f"pos0={float(summary.get('pos0_pct', 0.0)):.1f}%"
        ")"
    )


def _save_rank_bucket_cem_run_metadata(
    *,
    artifact_paths,
    shared: SharedAttackArtifacts,
    target_item: int,
    trainer: RankBucketCEMTrainer,
    trainer_result: Mapping[str, Any],
    config: Config,
    clean_checkpoint: Path,
) -> None:
    if artifact_paths.run_metadata is None:
        return
    rank_bucket_cem_config_payload = asdict(trainer.rank_bucket_cem_config)
    effective_population_schedule = [
        int(value)
        for value in trainer.rank_bucket_cem_config.effective_population_schedule
    ]
    rank_bucket_cem_config_payload["effective_population_schedule"] = (
        effective_population_schedule
    )
    rank_bucket_cem_config_payload["candidate_count"] = int(
        sum(effective_population_schedule)
    )
    payload = {
        "target_item": int(target_item),
        "run_type": POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        "position_opt_method": POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        "clean_surrogate_checkpoint": str(clean_checkpoint),
        "shared_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
        "shared_attack_dir": str(shared.shared_paths["attack_shared_dir"]),
        "shared_target_dir": str(shared.shared_paths["target_shared_dir"]),
        "fake_session_count": int(len(shared.template_sessions)),
        "clean_train_prefix_count": int(len(shared.clean_sessions)),
        "validation_session_count": int(len(shared.canonical_dataset.valid)),
        "resolved_seeds": asdict(config.seeds),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "nonzero_action_when_possible": bool(
            trainer.position_opt_config.nonzero_action_when_possible
        ),
        "position_opt_config": asdict(trainer.position_opt_config),
        "rank_bucket_cem_config": rank_bucket_cem_config_payload,
        "best_reward_name": trainer_result.get("best_reward_name"),
        "best_iteration_reward": trainer_result.get("best_iteration_reward"),
        "final_selection_reward_name": trainer_result.get("final_selection_reward_name"),
        "final_selection_reward_value": trainer_result.get("final_selection_reward_value"),
        "replay_metadata": trainer_result.get("replay_metadata"),
        "validation_subset_strategy": trainer_result.get("validation_subset_strategy"),
        "validation_subset_seed": trainer_result.get("validation_subset_seed"),
        "trainer_result": dict(trainer_result),
        "artifact_paths": {
            "base_dir": str(artifact_paths.base_dir),
            "optimized_poisoned_sessions": (
                str(artifact_paths.optimized_poisoned_sessions)
                if trainer.rank_bucket_cem_config.save_optimized_poisoned_sessions
                else None
            ),
            "availability_summary": str(artifact_paths.availability_summary),
            "cem_trace": str(artifact_paths.cem_trace),
            "cem_state_history": str(artifact_paths.cem_state_history),
            "cem_best_policy": str(artifact_paths.cem_best_policy),
            "final_selected_positions": (
                str(artifact_paths.final_selected_positions)
                if trainer.rank_bucket_cem_config.save_final_selected_positions
                else None
            ),
            "final_position_summary": str(artifact_paths.final_position_summary),
        },
    }
    save_run_metadata(artifact_paths.run_metadata, payload)


def _resolve_position_opt_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if hasattr(args, "fine_tune_steps"):
        overrides["fine_tune_steps"] = int(args.fine_tune_steps)
    if hasattr(args, "validation_subset_size"):
        overrides["validation_subset_size"] = args.validation_subset_size
    if hasattr(args, "reward_mode"):
        overrides["reward_mode"] = str(args.reward_mode)
    if hasattr(args, "enable_gt_penalty"):
        overrides["enable_gt_penalty"] = bool(args.enable_gt_penalty)
    if hasattr(args, "gt_penalty_weight"):
        overrides["gt_penalty_weight"] = float(args.gt_penalty_weight)
    if hasattr(args, "gt_tolerance"):
        overrides["gt_tolerance"] = float(args.gt_tolerance)
    return overrides


def _resolve_rank_bucket_cem_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if hasattr(args, "cem_iterations"):
        overrides["iterations"] = int(args.cem_iterations)
    if hasattr(args, "cem_population_size"):
        overrides["population_size"] = int(args.cem_population_size)
    if hasattr(args, "cem_elite_ratio"):
        overrides["elite_ratio"] = float(args.cem_elite_ratio)
    if hasattr(args, "cem_initial_std"):
        overrides["initial_std"] = float(args.cem_initial_std)
    if hasattr(args, "cem_min_std"):
        overrides["min_std"] = float(args.cem_min_std)
    if hasattr(args, "cem_smoothing"):
        overrides["smoothing"] = float(args.cem_smoothing)
    if hasattr(args, "cem_reward_metric"):
        overrides["reward_metric"] = args.cem_reward_metric
    return overrides


def _parse_optional_int(value: str) -> int | None:
    stripped = value.strip()
    if stripped.lower() == "none":
        return None
    return int(stripped)


def _parse_optional_text(value: str) -> str | None:
    stripped = value.strip()
    if stripped.lower() == "none":
        return None
    return stripped


def _apply_single_target_override(config: Config, target_item: int | None) -> Config:
    if target_item is None:
        return config
    return replace(
        config,
        targets=replace(
            config.targets,
            mode="explicit_list",
            explicit_list=(int(target_item),),
            count=1,
            reuse_saved_targets=False,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_RANK_BUCKET_CEM_CONFIG_PATH,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--clean-surrogate-checkpoint",
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.clean_surrogate_checkpoint.",
    )
    parser.add_argument(
        "--target-item",
        type=int,
        default=None,
        help="Optional single-target override for a smoke test.",
    )
    parser.add_argument(
        "--fine-tune-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.fine_tune_steps.",
    )
    parser.add_argument(
        "--validation-subset-size",
        type=_parse_optional_int,
        default=argparse.SUPPRESS,
        help=(
            "Optional CLI override for attack.position_opt.validation_subset_size. "
            "Use an integer or 'none'."
        ),
    )
    parser.add_argument(
        "--reward-mode",
        choices=[
            "poisoned_target_utility",
            "delta_target_utility",
        ],
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.reward_mode.",
    )
    parser.add_argument(
        "--enable-gt-penalty",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.enable_gt_penalty.",
    )
    parser.add_argument(
        "--gt-penalty-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.gt_penalty_weight.",
    )
    parser.add_argument(
        "--gt-tolerance",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.gt_tolerance.",
    )
    parser.add_argument(
        "--cem-iterations",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.iterations.",
    )
    parser.add_argument(
        "--cem-population-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.population_size.",
    )
    parser.add_argument(
        "--cem-elite-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.elite_ratio.",
    )
    parser.add_argument(
        "--cem-initial-std",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.initial_std.",
    )
    parser.add_argument(
        "--cem-min-std",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.min_std.",
    )
    parser.add_argument(
        "--cem-smoothing",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.rank_bucket_cem.smoothing.",
    )
    parser.add_argument(
        "--cem-reward-metric",
        type=_parse_optional_text,
        default=argparse.SUPPRESS,
        help=(
            "Optional CLI override for attack.rank_bucket_cem.reward_metric. "
            "Use a metric name or 'none'."
        ),
    )
    args = parser.parse_args()

    config = _apply_single_target_override(load_config(args.config), args.target_item)
    run_rank_bucket_cem(
        config,
        clean_surrogate_checkpoint_path=(
            args.clean_surrogate_checkpoint
            if hasattr(args, "clean_surrogate_checkpoint")
            else None
        ),
        config_path=args.config,
        position_opt_config=_resolve_position_opt_overrides(args),
        rank_bucket_cem_config=_resolve_rank_bucket_cem_overrides(args),
    )


if __name__ == "__main__":
    main()
