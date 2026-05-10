from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import (
    Config,
    PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1,
    PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED,
    PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE,
    PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX,
    PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH,
    PTS_PREFIX_RANGE_INTERNAL,
    PTS_PREFIX_SAMPLER_UNIFORM,
    PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
    PTSConstructionConfig,
    load_config,
)
from attack.common.paths import PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE, target_dir
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.srgnn_full_retrain_validation_best import (
    SRGNNFullRetrainValidationBestInnerTrainer,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import (
    SharedAttackArtifacts,
    prepare_shared_attack_artifacts,
)
from attack.position_opt.cem.trainer import (
    _candidate_checkpoint_metadata,
    _coerce_target_metrics,
    _lowk_reward_metric_payload,
    _resolve_validation_pairs,
)
from attack.pts.artifacts import write_pts_cem_artifacts
from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSCEMInitConfig,
    PTSCEMSamplerConfig,
    PTSCEMUpdateConfig,
    PTSGroupedCEMTrainer,
)
from attack.pts.grouping import SuffixLengthBucket
from attack.pts.specs import (
    PTSConstructionSpec,
    get_default_pts_v1_specs,
    lookup_spec_by_name,
)
from attack.surrogate.srgnn_backend import SRGNNBackend
from pytorch_code.utils import Data


DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_ratio1_srgnn_partial4.yaml"
)
_LOG_PREFIX = "[pts-construction-cem]"


def run_pts_construction_grouped_cem(
    config: Config,
    *,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_pts_construction_run_config(config)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)
    pts_config = _require_pts_config(config)
    specs = _build_pts_specs_from_config(pts_config)
    suffix_length_buckets = _build_suffix_length_buckets_from_config(pts_config)
    cem_config = _build_pts_cem_config_from_config(config)
    attack_identity_context = build_pts_construction_attack_identity_context(config)

    print(
        f"{_LOG_PREFIX} loaded {len(shared.template_sessions)} shared fake sessions "
        f"from {shared.shared_paths['fake_sessions']}"
    )
    print(
        f"{_LOG_PREFIX} method={pts_config.method} "
        f"iterations={int(cem_config.iterations)} "
        f"population_schedule={cem_config.population_schedule or cem_config.population_size} "
        f"actions={list(pts_config.actions.enabled)}"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        evaluator_context = _build_candidate_evaluator_context(config, shared)
        trainer = PTSGroupedCEMTrainer(
            cem_config=cem_config,
            specs=specs,
            suffix_length_buckets=suffix_length_buckets,
            disable_consume_one_when_suffix_len_leq_1=(
                pts_config.actions.dynamic_masks.disable_consume_one_when_suffix_len_leq_1
            ),
            generation_topk=int(pts_config.generation.topk),
            generation_rng_tag="pts_generated_suffix",
        )

        def evaluator_fn(
            *,
            candidate_sessions: list[list[int]],
            candidate_session_records: list[dict[str, object]],
            candidate_summary: dict[str, object],
            iteration: int,
            candidate_id: int,
            candidate_seed: int,
            policy,
        ) -> PTSCEMEvaluationResult:
            del candidate_session_records, candidate_summary, policy
            return _evaluate_candidate_retrain_validation_reward(
                config=config,
                evaluator_context=evaluator_context,
                candidate_sessions=candidate_sessions,
                target_item=int(target_item),
                iteration=int(iteration),
                candidate_id=int(candidate_id),
                candidate_seed=int(candidate_seed),
            )

        result = trainer.train(
            template_sessions=shared.template_sessions,
            target_item=int(target_item),
            poison_runner=shared.poison_runner,
            evaluator_fn=evaluator_fn,
        )

        pts_artifact_dir = (
            target_dir(
                config,
                int(target_item),
                run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                attack_identity_context=attack_identity_context,
            )
            / "pts_construction_cem"
        )
        artifact_paths = write_pts_cem_artifacts(
            result=result,
            output_dir=pts_artifact_dir,
            save_top_candidate_sessions=(
                bool(pts_config.artifacts.save_top_candidate_sessions)
                or bool(pts_config.artifacts.save_best_sessions)
            ),
            save_per_session_records=bool(pts_config.artifacts.save_per_session_records),
        )

        final_sessions = result.best_candidate.final_sessions
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            final_sessions,
        )
        metadata = _target_metadata(
            config=config,
            pts_config=pts_config,
            cem_config=cem_config,
            artifact_dir=pts_artifact_dir,
            artifact_paths=artifact_paths,
            best_candidate=result.best_candidate,
        )
        print(
            f"{_LOG_PREFIX} target={int(target_item)} done "
            f"best_reward={float(result.best_candidate.reward):.6g} "
            f"best_iter={int(result.best_candidate.iteration)} "
            f"best_candidate={int(result.best_candidate.candidate_id)} "
            f"artifacts={pts_artifact_dir}"
        )
        return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )


def _require_pts_config(config: Config) -> PTSConstructionConfig:
    pts_config = config.attack.pts_construction
    if pts_config is None:
        raise ValueError("PTS-CEM runner requires attack.pts_construction.")
    return pts_config


def _validate_pts_construction_run_config(config: Config) -> None:
    if not bool(config.data.poison_train_only):
        raise ValueError("PTS-CEM runner requires data.poison_train_only == true.")
    pts_config = _require_pts_config(config)
    if not bool(pts_config.enabled):
        raise ValueError("PTS-CEM runner requires attack.pts_construction.enabled == true.")
    if pts_config.method != PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1:
        raise ValueError("PTS-CEM runner supports only method='grouped_cem_v1'.")
    if (
        pts_config.prefix_selector.range != PTS_PREFIX_RANGE_INTERNAL
        or pts_config.prefix_selector.sampler != PTS_PREFIX_SAMPLER_UNIFORM
    ):
        raise ValueError("PTS-CEM Phase 3 supports only internal/uniform prefix selection.")
    if pts_config.grouping.mode != PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH:
        raise ValueError("PTS-CEM Phase 3 supports only residual_suffix_length grouping.")
    _build_pts_specs_from_config(pts_config)
    _build_suffix_length_buckets_from_config(pts_config)
    if (
        pts_config.generation.length_policy
        != PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX
    ):
        raise ValueError(
            "PTS-CEM Phase 3 supports only generation.length_policy="
            "'same_as_residual_suffix'."
        )
    if pts_config.reward.target_summary != PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20:
        raise ValueError(
            "PTS-CEM Phase 3 supports only reward.target_summary="
            "'raw_lowk_mrr_recall_10_20'."
        )
    if bool(pts_config.reward.enable_gt_penalty):
        raise NotImplementedError("PTS-CEM GT penalty is not implemented in Phase 3.")
    if bool(pts_config.reward.enable_length_penalty):
        raise NotImplementedError("PTS-CEM length penalty is not implemented in Phase 3.")
    if pts_config.final_selection.mode != PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE:
        raise ValueError(
            "PTS-CEM Phase 3 supports only final_selection.mode="
            "'global_best_candidate'."
        )
    if not bool(pts_config.artifacts.save_cem_trace):
        raise ValueError("PTS-CEM runner requires artifacts.save_cem_trace == true.")
    if not bool(pts_config.artifacts.save_best_policy):
        raise ValueError("PTS-CEM runner requires artifacts.save_best_policy == true.")
    if not bool(pts_config.artifacts.save_final_policy):
        raise ValueError("PTS-CEM runner requires artifacts.save_final_policy == true.")
    if bool(pts_config.artifacts.save_candidate_sessions):
        raise ValueError("PTS-CEM runner does not support artifacts.save_candidate_sessions.")
    _resolve_pts_cem_base_seed(config)
    _srgnn_candidate_train_config(config)


def _build_pts_specs_from_config(
    pts_config: PTSConstructionConfig,
) -> tuple[PTSConstructionSpec, ...]:
    default_specs = get_default_pts_v1_specs()
    return tuple(
        lookup_spec_by_name(default_specs, action_name)
        for action_name in pts_config.actions.enabled
    )


def _build_suffix_length_buckets_from_config(
    pts_config: PTSConstructionConfig,
) -> tuple[SuffixLengthBucket, ...]:
    return tuple(
        SuffixLengthBucket(
            name=bucket.name,
            min_len=int(bucket.min),
            max_len=(None if bucket.max is None else int(bucket.max)),
        )
        for bucket in pts_config.grouping.buckets
    )


def _build_pts_cem_config_from_config(config: Config) -> PTSCEMConfig:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
    return PTSCEMConfig(
        iterations=int(cem.iterations),
        population_schedule=(
            None
            if cem.population_schedule is None
            else [int(value) for value in cem.population_schedule]
        ),
        population_size=None if cem.population_size is None else int(cem.population_size),
        elite_ratio=float(cem.elite_ratio),
        sampler=PTSCEMSamplerConfig(
            type=cem.sampler.type,
            concentration_scale=float(cem.sampler.concentration_scale),
        ),
        update=PTSCEMUpdateConfig(
            smoothing=float(cem.update.smoothing),
            min_probability=float(cem.update.min_probability),
            max_probability=float(cem.update.max_probability),
        ),
        init=PTSCEMInitConfig(mode=cem.init.mode),
        base_seed=_resolve_pts_cem_base_seed(config),
        candidate_seed_stride=int(cem.candidate_seed_stride),
        save_top_k_candidates=int(cem.save_top_k_candidates),
    )


def _resolve_pts_cem_base_seed(config: Config) -> int:
    seed_source = _require_pts_config(config).cem.seed_source
    if seed_source == PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED:
        return int(config.seeds.position_opt_seed)
    raise ValueError(
        "PTS-CEM Phase 3 supports only cem.seed_source='position_opt_seed'."
    )


def build_pts_construction_attack_identity_context(config: Config) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    return {
        "pts_construction": {
            "method": pts_config.method,
            "runtime_seeds": {
                "position_opt_seed": int(config.seeds.position_opt_seed),
                "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
            },
        }
    }


def _build_candidate_evaluator_context(
    config: Config,
    shared: SharedAttackArtifacts,
) -> dict[str, object]:
    train_config = _srgnn_candidate_train_config(config)
    validation_sessions, validation_labels = _resolve_validation_pairs(shared)
    return {
        "backend": SRGNNBackend(config, base_dir=Path.cwd(), train_config=train_config),
        "inner_trainer": SRGNNFullRetrainValidationBestInnerTrainer(
            train_config=train_config,
            max_epochs=int(train_config["epochs"]),
            patience=int(train_config["patience"]),
            log_prefix="[pts-cem:candidate-retrain]",
        ),
        "validation_sessions": validation_sessions,
        "validation_labels": validation_labels,
        "validation_eval_data": Data((validation_sessions, validation_labels), shuffle=False),
        "train_config": train_config,
        "shared": shared,
    }


def _evaluate_candidate_retrain_validation_reward(
    *,
    config: Config,
    evaluator_context: Mapping[str, object],
    candidate_sessions: Sequence[Sequence[int]],
    target_item: int,
    iteration: int,
    candidate_id: int,
    candidate_seed: int,
) -> PTSCEMEvaluationResult:
    shared = evaluator_context["shared"]
    if not isinstance(shared, SharedAttackArtifacts):
        raise TypeError("PTS-CEM evaluator context has invalid shared artifacts.")
    backend = evaluator_context["backend"]
    inner_trainer = evaluator_context["inner_trainer"]
    validation_sessions = evaluator_context["validation_sessions"]
    validation_eval_data = evaluator_context["validation_eval_data"]
    if not isinstance(backend, SRGNNBackend):
        raise TypeError("PTS-CEM evaluator context has invalid SRGNNBackend.")
    if not isinstance(inner_trainer, SRGNNFullRetrainValidationBestInnerTrainer):
        raise TypeError("PTS-CEM evaluator context has invalid inner trainer.")

    candidate_start = time.perf_counter()
    poisoned_train = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        candidate_sessions,
    )
    retrain_start = time.perf_counter()
    inner_result = inner_trainer.run(
        backend,
        None,
        poisoned_train,
        config=None,
        eval_data=validation_eval_data,
        seed=int(config.seeds.surrogate_train_seed),
    )
    retrain_seconds = time.perf_counter() - retrain_start

    score_start = time.perf_counter()
    target_result = backend.score_target(
        inner_result.model,
        validation_sessions,
        int(target_item),
    )
    score_target_seconds = time.perf_counter() - score_start
    metrics = _coerce_target_metrics(target_result.metrics)
    lowk_payload = _lowk_reward_metric_payload(metrics)
    reward = float(lowk_payload["absolute_raw_family_lowk_reward"])
    candidate_total_seconds = time.perf_counter() - candidate_start

    reward_metrics = {
        **metrics,
        **{
            key: float(value)
            for key, value in lowk_payload.items()
            if isinstance(value, (int, float))
        },
        PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20: reward,
    }
    metadata: dict[str, object] = {
        "reward_name": PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
        "candidate_retrain_validation_reward": reward,
        "candidate_retrain_seed": int(config.seeds.surrogate_train_seed),
        "candidate_seed": int(candidate_seed),
        "candidate_retrain_validation_prefix_count": int(len(validation_sessions)),
        "candidate_retrain_epochs": int(_srgnn_candidate_train_config(config)["epochs"]),
        "iteration": int(iteration),
        "candidate_id": int(candidate_id),
        "candidate_total_seconds": float(candidate_total_seconds),
        "candidate_retrain_seconds": float(retrain_seconds),
        "score_target_seconds": float(score_target_seconds),
    }
    metadata.update(_candidate_checkpoint_metadata(inner_result.history))
    return PTSCEMEvaluationResult(
        reward=reward,
        reward_metrics=reward_metrics,
        metadata=metadata,
    )


def _srgnn_candidate_train_config(config: Config) -> dict[str, Any]:
    victim_params = config.victims.params.get("srgnn")
    if not isinstance(victim_params, Mapping):
        raise ValueError("PTS-CEM Phase 3 requires victims.params.srgnn.")
    train_config = victim_params.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError("PTS-CEM Phase 3 requires victims.params.srgnn.train.")
    return dict(train_config)


def _target_metadata(
    *,
    config: Config,
    pts_config: PTSConstructionConfig,
    cem_config: PTSCEMConfig,
    artifact_dir: Path,
    artifact_paths: Mapping[str, str],
    best_candidate,
) -> dict[str, object]:
    rank1_sessions = artifact_paths.get("top_candidate_rank_1_sessions")
    rank1_metadata = artifact_paths.get("top_candidate_rank_1_metadata")
    return {
        "pts_cem_trace_path": artifact_paths.get("pts_cem_trace"),
        "pts_policy_history_path": artifact_paths.get("pts_policy_history"),
        "pts_best_policy_path": artifact_paths.get("pts_best_policy"),
        "pts_final_policy_path": artifact_paths.get("pts_final_policy"),
        "pts_top_candidates_path": artifact_paths.get("pts_top_candidates"),
        "pts_top_candidate_policies_path": artifact_paths.get(
            "pts_top_candidate_policies"
        ),
        "pts_artifact_dir": str(artifact_dir),
        "pts_best_candidate_iteration": int(best_candidate.iteration),
        "pts_best_candidate_id": int(best_candidate.candidate_id),
        "pts_best_candidate_seed": int(best_candidate.candidate_seed),
        "pts_best_candidate_reward": float(best_candidate.reward),
        "pts_best_candidate_reward_metrics": dict(best_candidate.reward_metrics),
        "pts_best_candidate_sessions_path": rank1_sessions,
        "pts_best_candidate_metadata_path": rank1_metadata,
        "pts_final_selection_mode": pts_config.final_selection.mode,
        "pts_construction_method": pts_config.method,
        "pts_population_schedule": (
            list(cem_config.population_schedule)
            if cem_config.population_schedule is not None
            else None
        ),
        "pts_population_size": cem_config.population_size,
        "pts_actions_enabled": list(pts_config.actions.enabled),
        "pts_grouping_mode": pts_config.grouping.mode,
        "pts_candidate_retrain_seed": int(config.seeds.surrogate_train_seed),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Grouped PTS-CEM construction through the attack pipeline."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH,
        help="Path to a YAML config.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    run_pts_construction_grouped_cem(config, config_path=config_path)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH",
    "build_pts_construction_attack_identity_context",
    "main",
    "run_pts_construction_grouped_cem",
    "_build_pts_cem_config_from_config",
    "_build_pts_specs_from_config",
    "_build_suffix_length_buckets_from_config",
    "_resolve_pts_cem_base_seed",
    "_validate_pts_construction_run_config",
]
