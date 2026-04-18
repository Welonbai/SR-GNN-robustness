from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import build_position_opt_attack_identity_context, target_dir
from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.truncated_finetune import TruncatedFineTuneInnerTrainer
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts, prepare_shared_attack_artifacts
from attack.position_opt import (
    POSITION_OPT_RUN_TYPE,
    PositionOptConfig,
    PositionOptArtifactPaths,
    PositionOptMVPTrainer,
    build_candidate_positions,
    build_position_opt_artifact_paths,
    ensure_position_opt_artifact_dirs,
    position_opt_identity_payload,
    resolve_clean_surrogate_checkpoint_path,
    resolve_position_opt_config,
)
from attack.surrogate.srgnn_backend import SRGNNBackend


def run_position_opt_mvp(
    config: Config,
    *,
    clean_surrogate_checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    position_opt_config: PositionOptConfig | Mapping[str, Any] | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Position-opt MVP expects data.poison_train_only to be true.")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)

    cli_overrides = _coerce_position_opt_overrides(position_opt_config)
    if clean_surrogate_checkpoint_path is not None:
        cli_overrides["clean_surrogate_checkpoint"] = str(clean_surrogate_checkpoint_path)
    resolved_position_opt_config = resolve_position_opt_config(
        config.attack.position_opt,
        cli_overrides or None,
    )
    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=POSITION_OPT_RUN_TYPE,
        override=resolved_position_opt_config.clean_surrogate_checkpoint,
    ).resolve()
    if not clean_checkpoint.exists():
        raise FileNotFoundError(
            "Clean surrogate checkpoint not found: "
            f"{clean_checkpoint}"
        )
    resolved_position_opt_config = replace(
        resolved_position_opt_config,
        clean_surrogate_checkpoint=str(clean_checkpoint),
    )
    attack_identity_context = build_position_opt_attack_identity_context(
        position_opt_config=position_opt_identity_payload(resolved_position_opt_config),
        clean_surrogate_checkpoint=clean_checkpoint,
    )
    candidate_summary = _candidate_size_summary(
        shared.template_sessions,
        replacement_topk_ratio=config.attack.replacement_topk_ratio,
    )
    print(
        "[position-opt] "
        f"loaded {len(shared.template_sessions)} shared fake sessions from "
        f"{shared.shared_paths['fake_sessions']}"
    )
    print(
        "[position-opt] "
        f"candidate_sizes(min/avg/max)="
        f"{candidate_summary['min']}/{candidate_summary['avg']:.2f}/{candidate_summary['max']} "
        f"clean_surrogate_checkpoint={clean_checkpoint}"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        artifact_paths = ensure_position_opt_artifact_dirs(
            build_position_opt_artifact_paths(
                config,
                run_type=POSITION_OPT_RUN_TYPE,
                target_item=target_item,
                clean_checkpoint_override=clean_checkpoint,
                attack_identity_context=attack_identity_context,
            )
        )
        surrogate_backend = SRGNNBackend(config, base_dir=Path.cwd())
        inner_trainer = TruncatedFineTuneInnerTrainer()
        trainer = PositionOptMVPTrainer(
            surrogate_backend,
            inner_trainer,
            clean_surrogate_checkpoint_path=artifact_paths.clean_surrogate_checkpoint,
            position_opt_config=resolved_position_opt_config,
        )

        trainer_result = trainer.train(
            shared.template_sessions,
            target_item,
            shared,
            config,
        )
        trainer.save_artifacts(artifact_paths)
        final_position_results = trainer.export_final_selected_positions()
        optimized_poisoned_sessions = trainer.export_final_poisoned_sessions()
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            optimized_poisoned_sessions,
        )
        target_root = target_dir(
            config,
            int(target_item),
            run_type=POSITION_OPT_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=[int(result.position) for result in final_position_results],
            run_type=POSITION_OPT_RUN_TYPE,
            target_item=int(target_item),
        )
        _save_position_opt_run_metadata(
            artifact_paths=artifact_paths,
            shared=shared,
            target_item=target_item,
            trainer=trainer,
            trainer_result=trainer_result,
            config=config,
            clean_checkpoint=clean_checkpoint,
        )
        print(
            "[position-opt] "
            f"target={int(target_item)} optimized_poison_sessions="
            f"{len(optimized_poisoned_sessions)} artifacts={artifact_paths.base_dir}"
        )

        metadata = {
            "position_stats_path": str(position_stats_path),
            "position_opt_artifact_dir": str(artifact_paths.base_dir),
            "position_opt_clean_surrogate_checkpoint": str(
                artifact_paths.clean_surrogate_checkpoint
            ),
            "position_opt_optimized_poisoned_sessions_path": str(
                artifact_paths.optimized_poisoned_sessions
            ),
            "position_opt_selected_positions_path": (
                None
                if artifact_paths.selected_positions is None
                else str(artifact_paths.selected_positions)
            ),
            "position_opt_training_history_path": (
                None
                if artifact_paths.training_history is None
                else str(artifact_paths.training_history)
            ),
            "position_opt_learned_logits_path": (
                None
                if artifact_paths.learned_logits is None
                else str(artifact_paths.learned_logits)
            ),
            "position_opt_run_metadata_path": (
                None if artifact_paths.run_metadata is None else str(artifact_paths.run_metadata)
            ),
            "position_opt_final_poisoned_session_count": int(len(optimized_poisoned_sessions)),
            "position_opt_policy_update": "reinforce",
            "position_opt_outer_eval_source": "real_validation_sessions",
            "position_opt_validation_subset_size": (
                None
                if trainer.position_opt_config.validation_subset_size is None
                else int(trainer.position_opt_config.validation_subset_size)
            ),
        }
        return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)

    summary = run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POSITION_OPT_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )
    print("[position-opt] Final victim evaluation completed.")
    return summary


def _candidate_size_summary(
    fake_sessions: list[list[int]],
    *,
    replacement_topk_ratio: float,
) -> dict[str, float]:
    sizes = [
        len(build_candidate_positions(list(session), replacement_topk_ratio))
        for session in fake_sessions
    ]
    if not sizes:
        return {"min": 0.0, "avg": 0.0, "max": 0.0}
    return {
        "min": float(min(sizes)),
        "avg": float(sum(sizes) / len(sizes)),
        "max": float(max(sizes)),
    }


def _save_position_opt_run_metadata(
    *,
    artifact_paths: PositionOptArtifactPaths,
    shared: SharedAttackArtifacts,
    target_item: int,
    trainer: PositionOptMVPTrainer,
    trainer_result: dict[str, Any],
    config: Config,
    clean_checkpoint: Path,
) -> None:
    if artifact_paths.run_metadata is None:
        return
    payload = {
        "target_item": int(target_item),
        "run_type": POSITION_OPT_RUN_TYPE,
        "clean_surrogate_checkpoint": str(clean_checkpoint),
        "shared_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
        "shared_attack_dir": str(shared.shared_paths["attack_shared_dir"]),
        "shared_target_dir": str(shared.shared_paths["target_shared_dir"]),
        "fake_session_count": int(len(shared.template_sessions)),
        "clean_train_prefix_count": int(len(shared.clean_sessions)),
        "validation_session_count": int(len(shared.canonical_dataset.valid)),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "position_opt_config": asdict(trainer.position_opt_config),
        "trainer_result": trainer_result,
        "artifact_paths": {
            "base_dir": str(artifact_paths.base_dir),
            "optimized_poisoned_sessions": str(artifact_paths.optimized_poisoned_sessions),
            "selected_positions": (
                None
                if artifact_paths.selected_positions is None
                else str(artifact_paths.selected_positions)
            ),
            "training_history": (
                None
                if artifact_paths.training_history is None
                else str(artifact_paths.training_history)
            ),
            "learned_logits": (
                None
                if artifact_paths.learned_logits is None
                else str(artifact_paths.learned_logits)
            ),
        },
        "config_resolution_order": [
            "cli_overrides",
            "attack.position_opt",
            "code_defaults",
        ],
    }
    save_json(payload, artifact_paths.run_metadata)


def _resolve_position_opt_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if hasattr(args, "outer_steps"):
        overrides["outer_steps"] = int(args.outer_steps)
    if hasattr(args, "policy_lr"):
        overrides["policy_lr"] = float(args.policy_lr)
    if hasattr(args, "fine_tune_steps"):
        overrides["fine_tune_steps"] = int(args.fine_tune_steps)
    if hasattr(args, "validation_subset_size"):
        overrides["validation_subset_size"] = args.validation_subset_size
    if hasattr(args, "reward_baseline_momentum"):
        overrides["reward_baseline_momentum"] = float(args.reward_baseline_momentum)
    if hasattr(args, "enable_gt_penalty"):
        overrides["enable_gt_penalty"] = bool(args.enable_gt_penalty)
    if hasattr(args, "gt_penalty_weight"):
        overrides["gt_penalty_weight"] = float(args.gt_penalty_weight)
    if hasattr(args, "gt_tolerance"):
        overrides["gt_tolerance"] = float(args.gt_tolerance)
    if hasattr(args, "final_selection"):
        overrides["final_selection"] = str(args.final_selection)
    return overrides


def _coerce_position_opt_overrides(
    value: PositionOptConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, PositionOptConfig):
        return asdict(value)
    return dict(value)


def _parse_optional_int(value: str) -> int | None:
    stripped = value.strip()
    if stripped.lower() == "none":
        return None
    return int(stripped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/diginetica_attack_position_optimization_reward.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--clean-surrogate-checkpoint",
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.clean_surrogate_checkpoint.",
    )
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.outer_steps.",
    )
    parser.add_argument(
        "--policy-lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.policy_lr.",
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
        "--reward-baseline-momentum",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.reward_baseline_momentum.",
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
        "--final-selection",
        choices=["argmax"],
        default=argparse.SUPPRESS,
        help="Optional CLI override for attack.position_opt.final_selection.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_position_opt_mvp(
        config,
        clean_surrogate_checkpoint_path=(
            args.clean_surrogate_checkpoint
            if hasattr(args, "clean_surrogate_checkpoint")
            else None
        ),
        config_path=args.config,
        position_opt_config=_resolve_position_opt_overrides(args),
    )


if __name__ == "__main__":
    main()
