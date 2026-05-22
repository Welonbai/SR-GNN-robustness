from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import (
    Config,
    PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
    PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM,
    PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1,
    PTS_CEM_SAMPLER_DIRICHLET,
    PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED,
    PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST,
    PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST,
    PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE,
    PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX,
    PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH,
    PTS_PREFIX_RANGE_INTERNAL,
    PTS_PREFIX_SAMPLER_UNIFORM,
    PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
    PTSConstructionConfig,
    load_config,
)
from attack.common.artifact_io import load_fake_sessions, load_json, save_json
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    attack_key,
    poison_model_key_payload,
    run_group_key,
    shared_attack_artifact_key_payload,
    shared_root,
    target_dir,
    target_cohort_key,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.srgnn_full_retrain_fixed_last import (
    SRGNNFullRetrainFixedLastInnerTrainer,
)
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
from attack.pipeline.core.victim_execution import victim_effective_train_seed
from attack.position_opt.cem.trainer import (
    _candidate_checkpoint_metadata as _rank_bucket_candidate_checkpoint_metadata,
    _coerce_target_metrics,
    _lowk_reward_metric_payload,
    _resolve_validation_pairs,
)
from attack.pts.artifacts import write_pts_cem_artifacts
from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSCEMInitConfig,
    PTSCEMResamplingConfig,
    PTSCEMSamplerConfig,
    PTSCEMUpdateConfig,
    PTSGroupedCEMTrainer,
)
from attack.pts.continuous_cem import (
    PTSContinuousBetaCEMConfig,
    PTSContinuousBetaCEMTrainer,
)
from attack.pts.continuous_init_selection import (
    build_continuous_mlp_initial_sample_plan,
    continuous_mlp_init_cache_key,
    continuous_mlp_init_identity_payload,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_NORMALIZED_SAMPLER,
    CONTINUOUS_BETA_SHARED_PREFIX_TAG,
)
from attack.pts.direct_action_cem import (
    DIRECT_ACTION_MLP_CEM_METHOD,
    PTSDirectActionMLPCEMConfig,
    PTSDirectActionMLPCEMTrainer,
)
from attack.pts.direct_action_executor import (
    DIRECT_ACTION_FORMAL_GENERATION_TAG,
    DIRECT_ACTION_FORMAL_PREFIX_TAG,
    DIRECT_ACTION_FORMAL_SAMPLE_TAG,
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
    "diginetica_valbest_attack_ptscem_internal_sample.yaml"
)
_LOG_PREFIX = "[pts-construction-cem]"
_PTS_CONSTRUCTION_ARTIFACT_DIR_NAME = "pts_construction_cem"
_PTS_CONSTRUCTION_COMPLETE_MARKER = "pts_construction_complete.json"
_PTS_CONSTRUCTION_SHARED_COMPLETE_MARKER = "pts_cem_shared_complete.json"
PTS_CEM_SHARED_CACHE_SCHEMA_VERSION = "pts_cem_shared_construction_cache_v1"
PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION = "pts_cem_local_artifact_v2"
PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE = "victim_effective_seed"
PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME = "srgnn"
_EPOCH_REWARD_DIAGNOSTICS_CACHE_WARNING = (
    "Epoch reward diagnostics requested, but reused PTS-CEM cache does not "
    "contain diagnostics. Use --force-recompute-pts-cem or delete the old "
    "cache to regenerate diagnostics."
)
_SURROGATE_RETRAIN_IDENTITY_NOTE = (
    "Surrogate retrain checkpoint protocol is intentionally excluded from "
    "PTS-CEM cache identity."
)
_SURROGATE_RETRAIN_REUSE_NOTE = (
    "fixed_last is intentionally excluded from cache identity; reused artifact "
    "may have been generated with validation_best."
)


@dataclass(frozen=True)
class CachedPTSBestCandidate:
    sessions: list[list[int]]
    metadata: dict[str, object]
    sessions_path: Path
    metadata_path: Path
    top_candidates_path: Path | None
    complete_marker_path: Path | None
    cache_mode: str
    cache_marker_missing: bool
    shared_pts_cem_cache_key: str | None = None
    shared_cache_path: Path | None = None
    reused_shared_pts_cem: bool = False
    local_materialized_from_shared: bool = False


def run_pts_construction_grouped_cem(
    config: Config,
    *,
    config_path: str | Path | None = None,
    force_recompute_pts_cem: bool = False,
) -> dict[str, object]:
    _validate_pts_construction_run_config(config)

    run_type = _pts_construction_run_type(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=run_type,
        require_poison_runner=True,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)
    pts_config = _require_pts_config(config)
    specs = (
        _build_pts_specs_from_config(pts_config)
        if pts_config.method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1
        else ()
    )
    suffix_length_buckets = (
        _build_suffix_length_buckets_from_config(pts_config)
        if pts_config.method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1
        else ()
    )
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
        f"{_pts_construction_log_method_detail(pts_config)}"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        pts_artifact_dir = _pts_construction_artifact_dir(
            config,
            int(target_item),
            attack_identity_context=attack_identity_context,
        )
        shared_cache_identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=int(target_item),
            fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_path=shared.shared_paths.get("poison_model"),
        )
        shared_cache_key = pts_cem_shared_cache_key(shared_cache_identity)
        shared_cache_dir = pts_cem_shared_cache_dir(config, shared_cache_key)
        cache_identity = _current_pts_construction_cache_identity(
            config,
            attack_identity_context=attack_identity_context,
            target_item=int(target_item),
        )
        if force_recompute_pts_cem:
            if pts_artifact_dir.exists():
                print(
                    f"{_LOG_PREFIX} Existing PTS-CEM artifact folder ignored "
                    "because force recompute was requested."
                )
                _reset_pts_artifact_dir_for_force(
                    artifact_dir=pts_artifact_dir,
                    config=config,
                    target_item=int(target_item),
                    attack_identity_context=attack_identity_context,
                )
        else:
            cached = _try_load_cached_pts_best_candidate(
                artifact_dir=pts_artifact_dir,
                target_item=int(target_item),
                current_identity=cache_identity,
                current_shared_cache_key=shared_cache_key,
            )
            if cached is not None:
                _warn_if_reused_cache_missing_epoch_diagnostics(config, cached)
                print(
                    f"{_LOG_PREFIX} target={int(target_item)} reuse CEM cache"
                )
                poisoned = build_poisoned_dataset(
                    shared.clean_sessions,
                    shared.clean_labels,
                    cached.sessions,
                )
                metadata = _target_metadata_from_cache(
                    config=config,
                    pts_config=pts_config,
                    cem_config=cem_config,
                    artifact_dir=pts_artifact_dir,
                    target_item=int(target_item),
                    cached=cached,
                )
                return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)
            shared_cached = _try_load_shared_pts_cem_cache(
                shared_cache_dir=shared_cache_dir,
                target_item=int(target_item),
                shared_cache_key=shared_cache_key,
                shared_cache_identity=shared_cache_identity,
            )
            if shared_cached is not None:
                print(
                    f"{_LOG_PREFIX} target={int(target_item)} reuse shared CEM cache"
                )
                if (
                    _pts_cem_surrogate_retrain_protocol(config)
                    == PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST
                ):
                    print(
                        f"{_LOG_PREFIX} Reusing existing PTS-CEM shared cache. "
                        "Requested surrogate retrain protocol is fixed_last, "
                        "but this option is identity-neutral and does not "
                        "invalidate existing validation_best artifacts."
                    )
                cached = _materialize_shared_pts_cem_cache(
                    config=config,
                    target_item=int(target_item),
                    local_artifact_dir=pts_artifact_dir,
                    shared_cache_dir=shared_cache_dir,
                    shared_cached=shared_cached,
                    shared_cache_key=shared_cache_key,
                    attack_identity_context=attack_identity_context,
                    current_identity=cache_identity,
                )
                _warn_if_reused_cache_missing_epoch_diagnostics(config, cached)
                poisoned = build_poisoned_dataset(
                    shared.clean_sessions,
                    shared.clean_labels,
                    cached.sessions,
                )
                metadata = _target_metadata_from_cache(
                    config=config,
                    pts_config=pts_config,
                    cem_config=cem_config,
                    artifact_dir=pts_artifact_dir,
                    target_item=int(target_item),
                    cached=cached,
                )
                return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)
            print(
                f"{_LOG_PREFIX} target={int(target_item)} run CEM"
            )

        evaluator_context = _build_candidate_evaluator_context(config, shared)
        if pts_config.method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1:
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
        elif pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
            continuous_config = _build_continuous_beta_cem_config(pts_config)
            init_selection = build_continuous_mlp_initial_sample_plan(
                config=config,
                cem_config=cem_config,
                continuous_config=continuous_config,
                template_sessions=shared.template_sessions,
                generation_topk=int(pts_config.generation.topk),
            )
            trainer = PTSContinuousBetaCEMTrainer(
                cem_config=cem_config,
                continuous_config=continuous_config,
                generation_topk=int(pts_config.generation.topk),
                generation_rng_tag="pts_generated_suffix",
                shared_prefix_rng_tag=CONTINUOUS_BETA_SHARED_PREFIX_TAG,
                initial_sample_plan=init_selection.selected_sample_plan,
                seed_scope="target_independent",
            )
        elif pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
            trainer = PTSDirectActionMLPCEMTrainer(
                cem_config=cem_config,
                direct_action_config=_build_direct_action_mlp_cem_config(pts_config),
                generation_topk=int(pts_config.generation.topk),
                generation_rng_tag=DIRECT_ACTION_FORMAL_GENERATION_TAG,
                action_sampling_tag=DIRECT_ACTION_FORMAL_SAMPLE_TAG,
                shared_prefix_rng_tag=DIRECT_ACTION_FORMAL_PREFIX_TAG,
            )
        else:
            raise ValueError(f"Unsupported PTS-CEM method {pts_config.method!r}.")

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
                population_size=_pts_cem_population_size(cem_config, int(iteration)),
                candidate_id=int(candidate_id),
                candidate_seed=int(candidate_seed),
            )

        result = trainer.train(
            template_sessions=shared.template_sessions,
            target_item=int(target_item),
            poison_runner=shared.poison_runner,
            evaluator_fn=evaluator_fn,
        )

        artifact_paths = write_pts_cem_artifacts(
            result=result,
            output_dir=pts_artifact_dir,
            save_top_candidate_sessions=bool(
                pts_config.artifacts.save_top_candidate_sessions
            ),
            save_per_session_records=bool(pts_config.artifacts.save_per_session_records),
            write_candidate_epoch_metrics=bool(
                pts_config.cem.epoch_reward_diagnostics.write_candidate_epoch_metrics
            ),
            write_epoch_reward_ranking_summary=(
                bool(pts_config.cem.epoch_reward_diagnostics.enabled)
                and bool(pts_config.cem.epoch_reward_diagnostics.write_ranking_summary)
            ),
        )
        _write_shared_pts_cem_cache(
            config=config,
            target_item=int(target_item),
            local_artifact_dir=pts_artifact_dir,
            artifact_paths=artifact_paths,
            best_candidate=result.best_candidate,
            shared_cache_dir=shared_cache_dir,
            shared_cache_key=shared_cache_key,
            shared_cache_identity=shared_cache_identity,
            attack_identity_context=attack_identity_context,
        )
        print(f"{_LOG_PREFIX} target={int(target_item)} wrote shared CEM cache")
        complete_marker_path = _write_pts_construction_complete_marker(
            config=config,
            target_item=int(target_item),
            artifact_dir=pts_artifact_dir,
            artifact_paths=artifact_paths,
            best_candidate=result.best_candidate,
            attack_identity_context=attack_identity_context,
            shared_cache_key=shared_cache_key,
            shared_cache_path=shared_cache_dir,
            reused_shared_pts_cem=False,
            local_materialized_from_shared=False,
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
            target_item=int(target_item),
            complete_marker_path=complete_marker_path,
            shared_cache_key=shared_cache_key,
            shared_cache_path=shared_cache_dir,
            reused_shared_pts_cem=False,
            local_materialized_from_shared=False,
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
        run_type=run_type,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )


def _require_pts_config(config: Config) -> PTSConstructionConfig:
    pts_config = config.attack.pts_construction
    if pts_config is None:
        raise ValueError("PTS-CEM runner requires attack.pts_construction.")
    return pts_config


def _pts_construction_run_type(config: Config) -> str:
    pts_config = _require_pts_config(config)
    if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        return PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE
    return PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE


def _pts_construction_log_method_detail(pts_config: PTSConstructionConfig) -> str:
    if pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        return "continuous_policy=suffix_length_mlp_h2"
    if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        return (
            "direct_action_policy=direct_action_mlp_h2 "
            f"length_feature={pts_config.direct_action_policy.length_feature}"
        )
    return f"actions={list(pts_config.actions.enabled)}"


def _validate_pts_construction_run_config(config: Config) -> None:
    if not bool(config.data.poison_train_only):
        raise ValueError("PTS-CEM runner requires data.poison_train_only == true.")
    pts_config = _require_pts_config(config)
    if not bool(pts_config.enabled):
        raise ValueError("PTS-CEM runner requires attack.pts_construction.enabled == true.")
    if pts_config.method not in {
        PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1,
        PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
        PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM,
    }:
        raise ValueError(
            "PTS-CEM runner supports method='grouped_cem_v1', "
            "method='continuous_mlp_cem', or method='direct_action_mlp_cem'."
        )
    if (
        pts_config.prefix_selector.range != PTS_PREFIX_RANGE_INTERNAL
        or pts_config.prefix_selector.sampler != PTS_PREFIX_SAMPLER_UNIFORM
    ):
        raise ValueError("PTS-CEM Phase 3 supports only internal/uniform prefix selection.")
    if pts_config.method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1:
        if pts_config.cem.sampler.type != PTS_CEM_SAMPLER_DIRICHLET:
            raise ValueError("Grouped PTS-CEM requires cem.sampler.type='dirichlet'.")
        if pts_config.grouping.mode != PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH:
            raise ValueError("PTS-CEM Phase 3 supports only residual_suffix_length grouping.")
        _build_pts_specs_from_config(pts_config)
        _build_suffix_length_buckets_from_config(pts_config)
    elif pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        if pts_config.cem.sampler.type != "gaussian":
            raise ValueError("Continuous MLP-CEM requires cem.sampler.type='gaussian'.")
        _build_continuous_beta_cem_config(pts_config)
    else:
        if pts_config.cem.sampler.type != "gaussian":
            raise ValueError("Direct-action MLP-CEM requires cem.sampler.type='gaussian'.")
        if pts_config.cem.update.mode != "elite_centered_empirical_gaussian":
            raise ValueError(
                "Direct-action MLP-CEM requires "
                "cem.update.mode='elite_centered_empirical_gaussian'."
            )
        _build_direct_action_mlp_cem_config(pts_config)
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
    if not bool(pts_config.artifacts.save_top_candidate_sessions):
        raise ValueError(
            "PTS-CEM runner requires artifacts.save_top_candidate_sessions == true "
            "because victim append reuse depends on top_candidates/rank_1/sessions.json."
        )
    _resolve_pts_cem_base_seed(config)
    _srgnn_candidate_train_config(config)
    _validate_pts_epoch_reward_diagnostics_config(config)


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
        init=PTSCEMInitConfig(
            mode=cem.init.mode,
            mandatory_enabled=bool(cem.init.mandatory_enabled),
            extreme_count=int(cem.init.extreme_count),
            moderate_count=int(cem.init.moderate_count),
            balanced_count=int(cem.init.balanced_count),
            extreme_pool_size=int(cem.init.extreme_pool_size),
            moderate_pool_size=int(cem.init.moderate_pool_size),
            extreme_alpha=float(cem.init.extreme_alpha),
            moderate_alpha=float(cem.init.moderate_alpha),
            distance=cem.init.distance,
        ),
        resampling=PTSCEMResamplingConfig(
            mode=cem.resampling.mode,
            local_concentration_scale=float(
                cem.resampling.local_concentration_scale
            ),
        ),
        base_seed=_resolve_pts_cem_base_seed(config),
        candidate_seed_stride=int(cem.candidate_seed_stride),
        save_top_k_candidates=int(cem.save_top_k_candidates),
    )


def _build_continuous_beta_cem_config(
    pts_config: PTSConstructionConfig,
) -> PTSContinuousBetaCEMConfig:
    continuous = pts_config.continuous_policy
    if not bool(continuous.deterministic_sampling):
        raise ValueError(
            "Continuous MLP-CEM requires continuous_policy.deterministic_sampling=true."
        )
    return PTSContinuousBetaCEMConfig(
        parameterization=continuous.internal_parameterization,
        parameter_bounds=(
            float(continuous.parameter_bounds.min),
            float(continuous.parameter_bounds.max),
        ),
        initial_std=float(pts_config.cem.init.soft_extreme_initial_std),
        min_std=float(pts_config.cem.update.min_std),
        smoothing_epsilon=float(continuous.smoothing_epsilon),
        deterministic_sampling=bool(continuous.deterministic_sampling),
        initialization_mode=pts_config.cem.init.mode,
        gaussian_fill=True,
    )


def _build_direct_action_mlp_cem_config(
    pts_config: PTSConstructionConfig,
) -> PTSDirectActionMLPCEMConfig:
    policy = pts_config.direct_action_policy
    return PTSDirectActionMLPCEMConfig(
        length_feature_mode=policy.length_feature,
        elite_min_std=float(pts_config.cem.update.elite_min_std),
    )


def _pts_cem_population_size(cem_config: PTSCEMConfig, iteration: int) -> int:
    if cem_config.population_schedule is not None:
        return int(cem_config.population_schedule[int(iteration)])
    if cem_config.population_size is None:
        raise ValueError("PTS-CEM population_size is required without population_schedule.")
    return int(cem_config.population_size)


def _resolve_pts_cem_base_seed(config: Config) -> int:
    seed_source = _require_pts_config(config).cem.seed_source
    if seed_source == PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED:
        return int(config.seeds.position_opt_seed)
    raise ValueError(
        "PTS-CEM Phase 3 supports only cem.seed_source='position_opt_seed'."
    )


def resolve_pts_cem_surrogate_effective_seed(
    config: Config,
    *,
    target_item: int,
) -> int:
    return victim_effective_train_seed(
        config,
        victim_name=PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME,
        run_type=_pts_construction_run_type(config),
        target_item=int(target_item),
    )


def pts_cem_surrogate_seed_alignment_metadata(
    config: Config,
    *,
    target_item: int,
) -> dict[str, object]:
    resolved_seed = resolve_pts_cem_surrogate_effective_seed(
        config,
        target_item=int(target_item),
    )
    return {
        "target_item": int(target_item),
        "pts_cem_surrogate_seed_alignment_mode": (
            PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE
        ),
        "pts_cem_surrogate_seed_alignment_target_victim_name": (
            PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME
        ),
        "configured_surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        "configured_victim_train_seed": int(config.seeds.victim_train_seed),
        "resolved_surrogate_effective_seed": int(resolved_seed),
        "resolved_victim_effective_seed": int(resolved_seed),
        "surrogate_victim_seed_aligned": True,
    }


def pts_cem_surrogate_retrain_metadata(config: Config) -> dict[str, object]:
    surrogate_retrain = _require_pts_config(config).cem.surrogate_retrain
    return {
        "pts_cem_surrogate_retrain_checkpoint_protocol": (
            surrogate_retrain.checkpoint_protocol
        ),
        "pts_cem_surrogate_retrain_validation_enabled": bool(
            surrogate_retrain.validation_enabled
        ),
        "pts_cem_surrogate_retrain_reward_checkpoint": (
            surrogate_retrain.reward_checkpoint
        ),
        "pts_cem_surrogate_retrain_identity_neutral": True,
        "pts_cem_surrogate_retrain_identity_note": _SURROGATE_RETRAIN_IDENTITY_NOTE,
    }


def _pts_cem_surrogate_retrain_reuse_metadata(
    config: Config,
    cached: CachedPTSBestCandidate,
) -> dict[str, object]:
    requested = _require_pts_config(config).cem.surrogate_retrain.checkpoint_protocol
    if requested != PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST:
        return {}
    if not bool(cached.reused_shared_pts_cem):
        return {}
    return {
        "requested_surrogate_retrain_checkpoint_protocol": (
            PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST
        ),
        "reused_shared_pts_cem": True,
        "reused_artifact_surrogate_retrain_protocol": (
            _infer_reused_artifact_surrogate_retrain_protocol(cached.metadata)
        ),
        "pts_cem_surrogate_retrain_identity_neutral": True,
        "note": _SURROGATE_RETRAIN_REUSE_NOTE,
    }


def _pts_cem_surrogate_retrain_marker_reuse_metadata(
    config: Config,
    *,
    reused_shared_pts_cem: bool,
) -> dict[str, object]:
    requested = _require_pts_config(config).cem.surrogate_retrain.checkpoint_protocol
    if requested != PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST or not bool(reused_shared_pts_cem):
        return {}
    return {
        "requested_surrogate_retrain_checkpoint_protocol": (
            PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST
        ),
        "reused_artifact_surrogate_retrain_protocol": "unknown",
        "note": _SURROGATE_RETRAIN_REUSE_NOTE,
    }


def _infer_reused_artifact_surrogate_retrain_protocol(
    metadata: Mapping[str, object],
) -> str:
    protocol = metadata.get("pts_cem_surrogate_retrain_checkpoint_protocol")
    if protocol is None:
        evaluator_metadata = metadata.get("evaluator_metadata")
        if isinstance(evaluator_metadata, Mapping):
            protocol = evaluator_metadata.get(
                "pts_cem_surrogate_retrain_checkpoint_protocol"
            )
    if protocol is not None:
        return str(protocol)
    evaluator_metadata = metadata.get("evaluator_metadata")
    if isinstance(evaluator_metadata, Mapping):
        if (
            evaluator_metadata.get("selected_checkpoint_metric") is not None
            or evaluator_metadata.get("best_valid_mrr20") is not None
            or evaluator_metadata.get("valid_ground_truth_mrr@20") is not None
        ):
            return PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST
    return "unknown"


def _pts_cem_surrogate_retrain_protocol(config: Config) -> str:
    return str(_require_pts_config(config).cem.surrogate_retrain.checkpoint_protocol)


def _pts_cem_seed_alignment_identity(config: Config) -> dict[str, object]:
    identity: dict[str, object] = {
        "pts_cem_surrogate_seed_alignment_mode": (
            PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE
        ),
        "pts_cem_surrogate_seed_alignment_target_victim_name": (
            PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME
        ),
        "configured_surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        "configured_victim_train_seed": int(config.seeds.victim_train_seed),
    }
    if config.targets.mode == "explicit_list" and len(config.targets.explicit_list) == 1:
        target_item = int(config.targets.explicit_list[0])
        identity.update(
            pts_cem_surrogate_seed_alignment_metadata(
                config,
                target_item=target_item,
            )
        )
    return identity


def build_pts_construction_attack_identity_context(config: Config) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
    resolved_cem_base_seed = _resolve_pts_cem_base_seed(config)
    if pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        return {
            "pts_construction": {
                **_continuous_pts_construction_identity_payload(
                    config,
                    target_item=None,
                ),
                "runtime_seeds": {
                    "position_opt_seed": int(config.seeds.position_opt_seed),
                    "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
                    "victim_train_seed": int(config.seeds.victim_train_seed),
                    "resolved_cem_base_seed": int(resolved_cem_base_seed),
                    "surrogate_seed_alignment": _pts_cem_seed_alignment_identity(config),
                },
            }
        }
    if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        return {
            "pts_construction": {
                **_direct_action_pts_construction_identity_payload(
                    config,
                    target_item=None,
                    fake_sessions_path=None,
                ),
                "runtime_seeds": {
                    "position_opt_seed": int(config.seeds.position_opt_seed),
                    "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
                    "victim_train_seed": int(config.seeds.victim_train_seed),
                    "resolved_cem_base_seed": int(resolved_cem_base_seed),
                    "surrogate_seed_alignment": _pts_cem_seed_alignment_identity(config),
                },
            }
        }
    return {
        "pts_construction": {
            "method": pts_config.method,
            "prefix_selector": {
                "range": pts_config.prefix_selector.range,
                "sampler": pts_config.prefix_selector.sampler,
            },
            "grouping": {
                "mode": pts_config.grouping.mode,
                "buckets": [
                    {
                        "name": bucket.name,
                        "min": int(bucket.min),
                        "max": None if bucket.max is None else int(bucket.max),
                    }
                    for bucket in pts_config.grouping.buckets
                ],
            },
            "actions": {
                "enabled": list(pts_config.actions.enabled),
                "dynamic_masks": {
                    "disable_consume_one_when_suffix_len_leq_1": bool(
                        pts_config.actions.dynamic_masks.disable_consume_one_when_suffix_len_leq_1
                    ),
                },
            },
            "generation": {
                "topk": int(pts_config.generation.topk),
                "length_policy": pts_config.generation.length_policy,
            },
            "cem": {
                "iterations": int(cem.iterations),
                "population_schedule": (
                    None
                    if cem.population_schedule is None
                    else [int(value) for value in cem.population_schedule]
                ),
                "population_size": (
                    None if cem.population_size is None else int(cem.population_size)
                ),
                "elite_ratio": float(cem.elite_ratio),
                "sampler": {
                    "type": cem.sampler.type,
                    "concentration_scale": float(cem.sampler.concentration_scale),
                },
                "update": {
                    "smoothing": float(cem.update.smoothing),
                    "min_probability": float(cem.update.min_probability),
                    "max_probability": float(cem.update.max_probability),
                },
                "init": {
                    "mode": cem.init.mode,
                    "mandatory_enabled": bool(cem.init.mandatory_enabled),
                    "extreme_count": int(cem.init.extreme_count),
                    "moderate_count": int(cem.init.moderate_count),
                    "balanced_count": int(cem.init.balanced_count),
                    "extreme_pool_size": int(cem.init.extreme_pool_size),
                    "moderate_pool_size": int(cem.init.moderate_pool_size),
                    "extreme_alpha": float(cem.init.extreme_alpha),
                    "moderate_alpha": float(cem.init.moderate_alpha),
                    "distance": cem.init.distance,
                },
                "resampling": {
                    "mode": cem.resampling.mode,
                    "local_concentration_scale": float(
                        cem.resampling.local_concentration_scale
                    ),
                },
                "seed_source": cem.seed_source,
                "cem_base_seed": int(resolved_cem_base_seed),
                "resolved_cem_base_seed": int(resolved_cem_base_seed),
                "candidate_seed_stride": int(cem.candidate_seed_stride),
            },
            "final_selection": {
                "mode": pts_config.final_selection.mode,
            },
            "runtime_seeds": {
                "position_opt_seed": int(config.seeds.position_opt_seed),
                "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
                "victim_train_seed": int(config.seeds.victim_train_seed),
                "resolved_cem_base_seed": int(resolved_cem_base_seed),
                "surrogate_seed_alignment": _pts_cem_seed_alignment_identity(config),
            },
        }
    }


def _continuous_pts_construction_identity_payload(
    config: Config,
    *,
    target_item: int | None,
    initialization_identity: Mapping[str, object] | None = None,
    initialization_cache_key: str | None = None,
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
    continuous = pts_config.continuous_policy
    payload: dict[str, object] = {
        "method": pts_config.method,
        "prefix_selector": {
            "range": pts_config.prefix_selector.range,
            "sampler": pts_config.prefix_selector.sampler,
        },
        "generation": {
            "topk": int(pts_config.generation.topk),
            "length_policy": pts_config.generation.length_policy,
            "generation_rng_tag": "pts_generated_suffix",
        },
        "continuous_policy": {
            "parameterization": continuous.parameterization,
            "hidden_size": int(continuous.hidden_size),
            "consume_distribution": continuous.consume_distribution,
            "source_policy": continuous.source_policy,
            "parameter_bounds": {
                "min": float(continuous.parameter_bounds.min),
                "max": float(continuous.parameter_bounds.max),
            },
            "smoothing_epsilon": float(continuous.smoothing_epsilon),
            "deterministic_sampling": bool(continuous.deterministic_sampling),
        },
        "init": {
            "mode": pts_config.cem.init.mode,
            "soft_extreme_pool_size": int(pts_config.cem.init.soft_extreme_pool_size),
            "moderate_pool_size": int(pts_config.cem.init.moderate_pool_size),
            "soft_extreme_select_size": int(pts_config.cem.init.soft_extreme_select_size),
            "moderate_select_size": int(pts_config.cem.init.moderate_select_size),
            "soft_extreme_initial_std": float(
                pts_config.cem.init.soft_extreme_initial_std
            ),
            "moderate_initial_std": float(pts_config.cem.init.moderate_initial_std),
            "q_grid_size": int(pts_config.cem.init.q_grid_size),
            "behavior_distance": pts_config.cem.init.behavior_distance,
            "init_materialize_generated_suffix": bool(
                pts_config.cem.init.init_materialize_generated_suffix
            ),
        },
        "shared_prefix_assignment": {
            "mode": "internal_uniform_target_independent_v1",
            "seed_scope": "target_independent",
            "seed_source": cem.seed_source,
            "resolved_seed": int(_resolve_pts_cem_base_seed(config)),
            "rng_tag": CONTINUOUS_BETA_SHARED_PREFIX_TAG,
        },
        "cem": {
            "iterations": int(cem.iterations),
            "population_schedule": (
                None
                if cem.population_schedule is None
                else [int(value) for value in cem.population_schedule]
            ),
            "population_size": (
                None if cem.population_size is None else int(cem.population_size)
            ),
            "elite_ratio": float(cem.elite_ratio),
            "sampler": {
                "type": CONTINUOUS_BETA_NORMALIZED_SAMPLER,
            },
            "update": {
                "smoothing": float(cem.update.smoothing),
            },
            "seed_source": cem.seed_source,
            "cem_base_seed": int(_resolve_pts_cem_base_seed(config)),
            "resolved_cem_base_seed": int(_resolve_pts_cem_base_seed(config)),
            "candidate_seed_stride": int(cem.candidate_seed_stride),
        },
        "sampling": {
            "beta_seed_fields": [
                "base_seed",
                "target_item",
                "candidate_key",
                "fake_session_index",
                "sampling_tag",
            ],
            "source_seed_fields": [
                "base_seed",
                "target_item",
                "candidate_key",
                "fake_session_index",
                "sampling_tag",
            ],
        },
        "final_selection": {
            "mode": pts_config.final_selection.mode,
        },
    }
    if initialization_identity is not None and initialization_cache_key is not None:
        payload["initialization"] = {
            "cache_key": str(initialization_cache_key),
            "identity_version": str(initialization_identity.get("identity_version", "")),
            "target_independent": True,
            "materialize_generated_suffix": False,
        }
    if target_item is not None:
        payload["target_item"] = int(target_item)
    return payload


def _direct_action_pts_construction_identity_payload(
    config: Config,
    *,
    target_item: int | None,
    fake_sessions_path: Path | None,
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
    policy = pts_config.direct_action_policy
    payload: dict[str, object] = {
        "method": pts_config.method,
        "prefix_selector": {
            "range": pts_config.prefix_selector.range,
            "sampler": pts_config.prefix_selector.sampler,
        },
        "shared_prefix_assignment": {
            "mode": "internal_uniform_target_independent_v1",
            "seed_scope": "target_independent",
            "seed_source": cem.seed_source,
            "resolved_seed": int(_resolve_pts_cem_base_seed(config)),
            "rng_tag": DIRECT_ACTION_FORMAL_PREFIX_TAG,
        },
        "generation": {
            "topk": int(pts_config.generation.topk),
            "length_policy": pts_config.generation.length_policy,
            "generation_rng_tag": DIRECT_ACTION_FORMAL_GENERATION_TAG,
            "generation_seed_fields": [
                "base_seed",
                "target_item",
                "iteration",
                "candidate_key",
                "fake_session_index",
                "consume_count",
                "generated_length",
                "formal_generation_tag",
            ],
        },
        "sampling": {
            "action_rng_tag": DIRECT_ACTION_FORMAL_SAMPLE_TAG,
            "action_seed_fields": [
                "base_seed",
                "target_item",
                "iteration",
                "candidate_key",
                "fake_session_index",
                "formal_action_sampling_tag",
            ],
        },
        "direct_action_policy": {
            "parameterization": policy.parameterization,
            "length_feature": policy.length_feature,
        },
        "cem_init": {
            "mode": "standard_normal",
            "parameter_space": "standardized_policy_parameter_space",
        },
        "cem": {
            "iterations": int(cem.iterations),
            "population_schedule": (
                None
                if cem.population_schedule is None
                else [int(value) for value in cem.population_schedule]
            ),
            "population_size": (
                None if cem.population_size is None else int(cem.population_size)
            ),
            "elite_ratio": float(cem.elite_ratio),
            "sampler": {
                "type": "gaussian",
            },
            "update": {
                "mode": "elite_centered_empirical_gaussian",
                "elite_min_std": float(cem.update.elite_min_std),
                "std_ddof": 0,
            },
            "seed_source": cem.seed_source,
            "cem_base_seed": int(_resolve_pts_cem_base_seed(config)),
            "resolved_cem_base_seed": int(_resolve_pts_cem_base_seed(config)),
            "candidate_seed_stride": int(cem.candidate_seed_stride),
        },
        "valid_actions": {
            "families": ["keep(k)", "generate(k)", "stop"],
            "k_range": "0..m-1",
            "generate_m_allowed": False,
            "keep_m_allowed": False,
            "stop_is_only_length_0_suffix_action": True,
        },
        "final_selection": {
            "mode": pts_config.final_selection.mode,
        },
    }
    if fake_sessions_path is not None:
        payload["fake_sessions"] = {
            "artifact_identity": _file_sha1_identity(fake_sessions_path),
        }
    if target_item is not None:
        payload["target_item"] = int(target_item)
    return payload


def _pts_construction_artifact_dir(
    config: Config,
    target_item: int,
    *,
    attack_identity_context: Mapping[str, Any] | None,
) -> Path:
    return (
        target_dir(
            config,
            int(target_item),
            run_type=_pts_construction_run_type(config),
            attack_identity_context=attack_identity_context,
        )
        / _PTS_CONSTRUCTION_ARTIFACT_DIR_NAME
    )


def pts_cem_shared_cache_dir(config: Config, shared_pts_cem_cache_key: str) -> Path:
    return (
        shared_root(config)
        / _PTS_CONSTRUCTION_ARTIFACT_DIR_NAME
        / str(shared_pts_cem_cache_key)
    )


def pts_cem_shared_cache_key(identity_payload: Mapping[str, object]) -> str:
    return f"pts_cem_shared_{_hash_json_payload(identity_payload)}"


def build_pts_cem_shared_cache_identity(
    config: Config,
    *,
    target_item: int,
    fake_sessions_path: Path,
    poison_model_path: Path | None = None,
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    if pts_config.method in {
        PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
        PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM,
    }:
        if load_fake_sessions(fake_sessions_path) is None:
            raise FileNotFoundError(
                f"{pts_config.method} construction identity requires readable fake sessions: "
                f"{fake_sessions_path}"
            )
    split_config = config.data.canonical_split
    poison_checkpoint_identity = (
        _file_sha1_identity(poison_model_path)
        if poison_model_path is not None and Path(poison_model_path).exists()
        else None
    )
    return {
        "schema_version": PTS_CEM_SHARED_CACHE_SCHEMA_VERSION,
        "artifact_schema_version": PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION,
        "run_type": _pts_construction_run_type(config),
        "dataset": {
            "dataset_name": config.data.dataset_name,
            "split_protocol": config.data.split_protocol,
            "poison_train_only": bool(config.data.poison_train_only),
            "canonical_split": {
                "min_item_count": int(split_config.min_item_count),
                "min_session_len": int(split_config.min_session_len),
                "valid_ratio": float(split_config.valid_ratio),
                "test_days": int(split_config.test_days),
            },
        },
        "target_item": int(target_item),
        "fake_sessions": {
            "artifact_identity": _file_sha1_identity(fake_sessions_path),
            "generation_identity": shared_attack_artifact_key_payload(
                config,
                run_type=_pts_construction_run_type(config),
            ),
        },
        "poison_model": {
            "key_payload": poison_model_key_payload(config),
            "checkpoint_identity": poison_checkpoint_identity,
        },
        "pts_construction": _pts_construction_shared_identity_payload(
            config,
            target_item=int(target_item),
            fake_sessions_path=fake_sessions_path,
        ),
        "surrogate_reward": {
            "surrogate_model": "srgnn",
            "surrogate_train_params": _srgnn_candidate_train_config(config),
            "reward": {
                "target_summary": pts_config.reward.target_summary,
                "enable_gt_penalty": bool(pts_config.reward.enable_gt_penalty),
                "gt_penalty_weight": float(pts_config.reward.gt_penalty_weight),
                "enable_length_penalty": bool(pts_config.reward.enable_length_penalty),
                "length_penalty_weight": float(
                    pts_config.reward.length_penalty_weight
                ),
            },
            "evaluation": {
                "topk": [int(value) for value in config.evaluation.topk],
                "targeted_metrics": list(config.evaluation.targeted_metrics),
                "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
            },
        },
        "seed_alignment": pts_cem_surrogate_seed_alignment_metadata(
            config,
            target_item=int(target_item),
        ),
    }


def _pts_construction_shared_identity_payload(
    config: Config,
    *,
    target_item: int,
    fake_sessions_path: Path | None = None,
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
    if pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        if fake_sessions_path is None:
            raise ValueError(
                "continuous_mlp_cem construction identity requires fake_sessions_path."
            )
        template_sessions = load_fake_sessions(fake_sessions_path)
        if template_sessions is None:
            raise FileNotFoundError(
                "continuous_mlp_cem construction identity requires readable fake sessions: "
                f"{fake_sessions_path}"
            )
        init_identity = continuous_mlp_init_identity_payload(
            config=config,
            template_sessions=template_sessions,
        )
        init_cache_key = continuous_mlp_init_cache_key(init_identity)
        return _continuous_pts_construction_identity_payload(
            config,
            target_item=int(target_item),
            initialization_identity=init_identity,
            initialization_cache_key=init_cache_key,
        )
    if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        if fake_sessions_path is None:
            raise ValueError(
                "direct_action_mlp_cem construction identity requires fake_sessions_path."
            )
        if load_fake_sessions(fake_sessions_path) is None:
            raise FileNotFoundError(
                "direct_action_mlp_cem construction identity requires readable fake sessions: "
                f"{fake_sessions_path}"
            )
        return _direct_action_pts_construction_identity_payload(
            config,
            target_item=int(target_item),
            fake_sessions_path=fake_sessions_path,
        )
    return {
        "target_item": int(target_item),
        "method": pts_config.method,
        "prefix_selector": {
            "range": pts_config.prefix_selector.range,
            "sampler": pts_config.prefix_selector.sampler,
        },
        "grouping": {
            "mode": pts_config.grouping.mode,
            "buckets": [
                {
                    "name": bucket.name,
                    "min": int(bucket.min),
                    "max": None if bucket.max is None else int(bucket.max),
                }
                for bucket in pts_config.grouping.buckets
            ],
        },
        "actions": {
            "enabled": list(pts_config.actions.enabled),
            "dynamic_masks": {
                "disable_consume_one_when_suffix_len_leq_1": bool(
                    pts_config.actions.dynamic_masks.disable_consume_one_when_suffix_len_leq_1
                ),
            },
        },
        "generation": {
            "topk": int(pts_config.generation.topk),
            "length_policy": pts_config.generation.length_policy,
        },
        "cem": {
            "iterations": int(cem.iterations),
            "population_schedule": (
                None
                if cem.population_schedule is None
                else [int(value) for value in cem.population_schedule]
            ),
            "population_size": (
                None if cem.population_size is None else int(cem.population_size)
            ),
            "elite_ratio": float(cem.elite_ratio),
            "sampler": {
                "type": cem.sampler.type,
                "concentration_scale": float(cem.sampler.concentration_scale),
            },
            "update": {
                "smoothing": float(cem.update.smoothing),
                "min_probability": float(cem.update.min_probability),
                "max_probability": float(cem.update.max_probability),
            },
            "init": {
                "mode": cem.init.mode,
                "mandatory_enabled": bool(cem.init.mandatory_enabled),
                "extreme_count": int(cem.init.extreme_count),
                "moderate_count": int(cem.init.moderate_count),
                "balanced_count": int(cem.init.balanced_count),
                "extreme_pool_size": int(cem.init.extreme_pool_size),
                "moderate_pool_size": int(cem.init.moderate_pool_size),
                "extreme_alpha": float(cem.init.extreme_alpha),
                "moderate_alpha": float(cem.init.moderate_alpha),
                "distance": cem.init.distance,
            },
            "resampling": {
                "mode": cem.resampling.mode,
                "local_concentration_scale": float(
                    cem.resampling.local_concentration_scale
                ),
            },
            "seed_source": cem.seed_source,
            "resolved_cem_base_seed": int(_resolve_pts_cem_base_seed(config)),
            "position_opt_seed": int(config.seeds.position_opt_seed),
            "candidate_seed_stride": int(cem.candidate_seed_stride),
        },
        "final_selection": {
            "mode": pts_config.final_selection.mode,
        },
    }


def _file_sha1_identity(path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing PTS-CEM artifact: {file_path}")
    digest = hashlib.sha1()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "type": "file_sha1",
        "sha1": digest.hexdigest(),
        "bytes": int(file_path.stat().st_size),
    }


def _stable_json_payload(payload: object) -> str:
    return json.dumps(
        _to_jsonable_cache_payload(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_json_payload(payload: object) -> str:
    return hashlib.sha1(_stable_json_payload(payload).encode("utf-8")).hexdigest()[:10]


def _current_pts_construction_cache_identity(
    config: Config,
    *,
    attack_identity_context: Mapping[str, Any] | None,
    target_item: int | None = None,
) -> dict[str, object]:
    payload = {
        "run_type": _pts_construction_run_type(config),
        "attack_key": attack_key(
            config,
            run_type=_pts_construction_run_type(config),
            attack_identity_context=attack_identity_context,
        ),
        "run_group_key": run_group_key(
            config,
            run_type=_pts_construction_run_type(config),
            attack_identity_context=attack_identity_context,
        ),
        "experiment_name": config.experiment.name,
        "dataset_name": config.data.dataset_name,
        "split_protocol": config.data.split_protocol,
    }
    if target_item is not None:
        payload.update(
            pts_cem_surrogate_seed_alignment_metadata(
                config,
                target_item=int(target_item),
            )
        )
    return payload


def _load_json_sessions(path: Path) -> list[list[int]]:
    payload = load_json(path)
    if payload is None:
        raise FileNotFoundError(f"PTS-CEM sessions file does not exist: {path}")
    if not isinstance(payload, list):
        raise ValueError(f"PTS-CEM sessions file must contain a JSON list: {path}")
    sessions: list[list[int]] = []
    for row_index, row in enumerate(payload):
        if not isinstance(row, list):
            raise ValueError(
                "PTS-CEM sessions file must contain a list of session lists: "
                f"{path}, row={row_index}"
            )
        session: list[int] = []
        for item_index, item in enumerate(row):
            session.append(
                _coerce_json_session_item(
                    item,
                    path=path,
                    row_index=row_index,
                    item_index=item_index,
                )
            )
        sessions.append(session)
    return sessions


def _coerce_json_session_item(
    value: object,
    *,
    path: Path,
    row_index: int,
    item_index: int,
) -> int:
    if isinstance(value, bool):
        raise ValueError(
            "PTS-CEM sessions item must be int-like, not bool: "
            f"{path}, row={row_index}, item={item_index}"
        )
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                "PTS-CEM sessions item must be int-like: "
                f"{path}, row={row_index}, item={item_index}, value={value!r}"
            )
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "PTS-CEM sessions item must be int-like: "
                f"{path}, row={row_index}, item={item_index}, value={value!r}"
            )
        try:
            return int(stripped)
        except ValueError as exc:
            raise ValueError(
                "PTS-CEM sessions item must be int-like: "
                f"{path}, row={row_index}, item={item_index}, value={value!r}"
            ) from exc
    raise ValueError(
        "PTS-CEM sessions item must be int-like: "
        f"{path}, row={row_index}, item={item_index}, value={value!r}"
    )


def _load_json_dict(path: Path) -> dict[str, object]:
    payload = load_json(path)
    if payload is None:
        raise FileNotFoundError(f"PTS-CEM JSON file does not exist: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"PTS-CEM JSON file must contain an object: {path}")
    return dict(payload)


def _try_load_cached_pts_best_candidate(
    *,
    artifact_dir: Path,
    target_item: int,
    current_identity: Mapping[str, object] | None = None,
    current_shared_cache_key: str | None = None,
) -> CachedPTSBestCandidate | None:
    root = Path(artifact_dir)
    marker_path = root / _PTS_CONSTRUCTION_COMPLETE_MARKER
    if marker_path.exists():
        try:
            return _load_marker_cached_pts_best_candidate(
                artifact_dir=root,
                marker_path=marker_path,
                target_item=int(target_item),
                current_identity=current_identity,
                current_shared_cache_key=current_shared_cache_key,
            )
        except ValueError as exc:
            if current_shared_cache_key is not None:
                _raise_incompatible_local_pts_cache(root, exc)
            raise

    if current_shared_cache_key is not None and root.exists() and any(root.iterdir()):
        _raise_incompatible_local_pts_cache(
            root,
            ValueError("local marker is missing but the artifact folder is not empty"),
        )

    sessions_path = _rank1_sessions_path(root)
    metadata_path = _rank1_metadata_path(root)
    top_candidates_path = root / "pts_top_candidates.json"
    legacy_paths = (sessions_path, metadata_path, top_candidates_path)
    existing_paths = [path for path in legacy_paths if path.exists()]
    if not existing_paths:
        return None
    if current_shared_cache_key is not None:
        _raise_incompatible_local_pts_cache(
            root,
            ValueError("local marker is missing but PTS-CEM artifact files exist"),
        )
    if len(existing_paths) != len(legacy_paths):
        missing = [str(path) for path in legacy_paths if not path.exists()]
        raise ValueError(
            "Incomplete legacy PTS-CEM cache; missing required files: "
            + ", ".join(missing)
        )

    sessions = _load_json_sessions(sessions_path)
    metadata = _load_json_dict(metadata_path)
    _load_json_dict(top_candidates_path)
    _validate_cached_candidate_metadata_target(
        metadata,
        target_item=int(target_item),
        label=str(metadata_path),
    )
    return CachedPTSBestCandidate(
        sessions=sessions,
        metadata=metadata,
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        top_candidates_path=top_candidates_path,
        complete_marker_path=None,
        cache_mode="legacy_top_candidate_files",
        cache_marker_missing=True,
    )


def _load_marker_cached_pts_best_candidate(
    *,
    artifact_dir: Path,
    marker_path: Path,
    target_item: int,
    current_identity: Mapping[str, object] | None,
    current_shared_cache_key: str | None = None,
) -> CachedPTSBestCandidate:
    marker = _load_json_dict(marker_path)
    if marker.get("status") != "completed":
        raise ValueError(
            f"PTS-CEM cache marker status must be 'completed': {marker_path}"
        )
    expected_run_type = str(
        (current_identity or {}).get("run_type", PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE)
    )
    if marker.get("run_type") != expected_run_type:
        raise ValueError(
            "PTS-CEM cache marker run_type mismatch: "
            f"{marker.get('run_type')!r}"
        )
    saved_target = _coerce_marker_int(
        marker.get("target_item"),
        label=f"{marker_path}: target_item",
    )
    if saved_target != int(target_item):
        raise ValueError(
            "PTS-CEM cache marker target_item mismatch: "
            f"expected {int(target_item)}, found {saved_target}."
        )
    _validate_marker_identity(
        marker,
        current_identity=current_identity,
        marker_path=marker_path,
    )
    _validate_local_marker_shared_cache(
        marker,
        current_shared_cache_key=current_shared_cache_key,
        marker_path=marker_path,
    )

    best_candidate = marker.get("best_candidate")
    if not isinstance(best_candidate, Mapping):
        raise ValueError(
            f"PTS-CEM cache marker is missing best_candidate object: {marker_path}"
        )
    sessions_path = _resolve_artifact_relative_path(
        artifact_dir,
        best_candidate.get("sessions_path"),
        label=f"{marker_path}: best_candidate.sessions_path",
    )
    metadata_path = _resolve_artifact_relative_path(
        artifact_dir,
        best_candidate.get("metadata_path"),
        label=f"{marker_path}: best_candidate.metadata_path",
    )
    sessions = _load_json_sessions(sessions_path)
    metadata = _load_json_dict(metadata_path)
    _validate_cached_candidate_metadata_target(
        metadata,
        target_item=int(target_item),
        label=str(metadata_path),
    )
    merged_metadata = dict(metadata)
    for key in (
        "rank",
        "iteration",
        "candidate_id",
        "candidate_seed",
        "reward",
        "reward_metrics",
    ):
        if key not in merged_metadata and key in best_candidate:
            merged_metadata[key] = best_candidate[key]
    top_candidates_path = artifact_dir / "pts_top_candidates.json"
    return CachedPTSBestCandidate(
        sessions=sessions,
        metadata=merged_metadata,
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        top_candidates_path=(
            top_candidates_path if top_candidates_path.exists() else None
        ),
        complete_marker_path=marker_path,
        cache_mode="complete_marker",
        cache_marker_missing=False,
        shared_pts_cem_cache_key=(
            str(marker["shared_pts_cem_cache_key"])
            if marker.get("shared_pts_cem_cache_key") is not None
            else None
        ),
        shared_cache_path=(
            Path(str(marker["shared_cache_path"]))
            if marker.get("shared_cache_path") is not None
            else None
        ),
        reused_shared_pts_cem=bool(marker.get("reused_shared_pts_cem", False)),
        local_materialized_from_shared=bool(
            marker.get("local_materialized_from_shared", False)
        ),
    )


def _validate_local_marker_shared_cache(
    marker: Mapping[str, object],
    *,
    current_shared_cache_key: str | None,
    marker_path: Path,
) -> None:
    if current_shared_cache_key is None:
        return
    schema = marker.get("local_artifact_schema_version")
    if schema != PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "PTS-CEM local marker schema mismatch: "
            f"expected {PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION!r}, found {schema!r}."
        )
    saved_key = marker.get("shared_pts_cem_cache_key")
    if str(saved_key) != str(current_shared_cache_key):
        raise ValueError(
            "PTS-CEM local marker shared cache key mismatch: "
            f"expected {current_shared_cache_key!r}, found {saved_key!r}."
        )


def _raise_incompatible_local_pts_cache(
    artifact_dir: Path,
    reason: Exception,
) -> None:
    raise ValueError(
        "Existing local PTS-CEM artifact folder is incompatible or incomplete: "
        f"{artifact_dir}. Remove this pts_construction_cem folder or rerun with "
        f"--force-recompute-pts-cem. Reason: {reason}"
    ) from reason


def _validate_marker_identity(
    marker: Mapping[str, object],
    *,
    current_identity: Mapping[str, object] | None,
    marker_path: Path,
) -> None:
    if current_identity is None:
        return
    saved_identity = marker.get("identity")
    if saved_identity is None:
        return
    if not isinstance(saved_identity, Mapping):
        raise ValueError(f"PTS-CEM cache marker identity must be an object: {marker_path}")
    for key in ("attack_key", "run_group_key"):
        saved_value = saved_identity.get(key)
        current_value = current_identity.get(key)
        if saved_value is not None and current_value is not None:
            if str(saved_value) != str(current_value):
                raise ValueError(
                    f"PTS-CEM cache marker identity mismatch for {key}: "
                    f"expected {current_value!r}, found {saved_value!r}."
                )


def _validate_cached_candidate_metadata_target(
    metadata: Mapping[str, object],
    *,
    target_item: int,
    label: str,
) -> None:
    if "target_item" not in metadata:
        return
    saved_target = _coerce_marker_int(
        metadata.get("target_item"),
        label=f"{label}: target_item",
    )
    if saved_target != int(target_item):
        raise ValueError(
            "PTS-CEM cached best-candidate metadata target_item mismatch: "
            f"expected {int(target_item)}, found {saved_target}."
        )


def _warn_if_reused_cache_missing_epoch_diagnostics(
    config: Config,
    cached: CachedPTSBestCandidate,
) -> None:
    diagnostics = _require_pts_config(config).cem.epoch_reward_diagnostics
    if not bool(diagnostics.enabled):
        return
    if _cached_metadata_has_requested_epoch_diagnostics(config, cached.metadata):
        return
    print(f"{_LOG_PREFIX} WARNING: {_EPOCH_REWARD_DIAGNOSTICS_CACHE_WARNING}")


def _cached_metadata_has_requested_epoch_diagnostics(
    config: Config,
    metadata: Mapping[str, object],
) -> bool:
    diagnostics_config = _require_pts_config(config).cem.epoch_reward_diagnostics
    diagnostics = metadata.get("epoch_reward_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    rewards_by_epoch = diagnostics.get("rewards_by_epoch")
    if not isinstance(rewards_by_epoch, Mapping):
        return False
    required_epochs = {int(epoch) for epoch in diagnostics_config.epochs}
    if bool(diagnostics_config.include_final_epoch):
        required_epochs.add(_resolved_pts_candidate_retrain_epochs(config))
    return all(str(epoch) in rewards_by_epoch for epoch in required_epochs)


def _coerce_marker_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def _resolve_artifact_relative_path(
    artifact_dir: Path,
    raw_path: object,
    *,
    label: str,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(artifact_dir) / path


def _rank1_sessions_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / "top_candidates" / "rank_1" / "sessions.json"


def _rank1_metadata_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / "top_candidates" / "rank_1" / "metadata.json"


def _rank1_policy_path(artifact_dir: Path) -> Path:
    return Path(artifact_dir) / "top_candidates" / "rank_1" / "policy.json"


def _has_pts_construction_cache_files(artifact_dir: Path) -> bool:
    root = Path(artifact_dir)
    return any(
        path.exists()
        for path in (
            root / _PTS_CONSTRUCTION_COMPLETE_MARKER,
            _rank1_sessions_path(root),
            _rank1_metadata_path(root),
            root / "pts_top_candidates.json",
        )
    )


def _reset_pts_artifact_dir_for_force(
    *,
    artifact_dir: Path,
    config: Config,
    target_item: int,
    attack_identity_context: Mapping[str, Any] | None,
) -> None:
    root = Path(artifact_dir)
    expected = _pts_construction_artifact_dir(
        config,
        int(target_item),
        attack_identity_context=attack_identity_context,
    )
    _remove_directory_inside(root, expected.parent)


def _remove_directory_inside(path: Path, allowed_parent: Path) -> None:
    target = Path(path).resolve()
    parent = Path(allowed_parent).resolve()
    try:
        target.relative_to(parent)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to remove directory outside expected parent: {target}"
        ) from exc
    if target.exists():
        shutil.rmtree(target)


def _try_load_shared_pts_cem_cache(
    *,
    shared_cache_dir: Path,
    target_item: int,
    shared_cache_key: str,
    shared_cache_identity: Mapping[str, object],
) -> CachedPTSBestCandidate | None:
    marker_path = Path(shared_cache_dir) / _PTS_CONSTRUCTION_SHARED_COMPLETE_MARKER
    if not marker_path.exists():
        return None
    try:
        marker = _load_json_dict(marker_path)
        _validate_shared_pts_cem_marker(
            marker,
            shared_cache_dir=Path(shared_cache_dir),
            target_item=int(target_item),
            shared_cache_key=shared_cache_key,
            shared_cache_identity=shared_cache_identity,
        )
        return _load_shared_marker_cached_pts_best_candidate(
            artifact_dir=Path(shared_cache_dir),
            marker_path=marker_path,
            marker=marker,
            target_item=int(target_item),
            shared_cache_key=shared_cache_key,
        )
    except (FileNotFoundError, ValueError):
        return None


def _validate_shared_pts_cem_marker(
    marker: Mapping[str, object],
    *,
    shared_cache_dir: Path,
    target_item: int,
    shared_cache_key: str,
    shared_cache_identity: Mapping[str, object],
) -> None:
    if marker.get("schema_version") != PTS_CEM_SHARED_CACHE_SCHEMA_VERSION:
        raise ValueError("PTS-CEM shared cache marker schema mismatch.")
    if marker.get("status") != "completed":
        raise ValueError("PTS-CEM shared cache marker is not completed.")
    if str(marker.get("shared_pts_cem_cache_key")) != str(shared_cache_key):
        raise ValueError("PTS-CEM shared cache key mismatch.")
    saved_target = _coerce_marker_int(
        marker.get("target_item"),
        label=f"{shared_cache_dir}: target_item",
    )
    if saved_target != int(target_item):
        raise ValueError("PTS-CEM shared cache target_item mismatch.")
    identity = marker.get("construction_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("PTS-CEM shared cache marker is missing identity.")
    if _stable_json_payload(identity) != _stable_json_payload(shared_cache_identity):
        raise ValueError("PTS-CEM shared cache identity mismatch.")
    required_files = marker.get("required_artifact_files")
    if not isinstance(required_files, list):
        raise ValueError("PTS-CEM shared cache marker missing required files.")
    for rel_path in required_files:
        if not isinstance(rel_path, str) or not rel_path.strip():
            raise ValueError("PTS-CEM shared cache required path is invalid.")
        if not (Path(shared_cache_dir) / rel_path).exists():
            raise ValueError(
                f"PTS-CEM shared cache required file is missing: {rel_path}"
            )


def _load_shared_marker_cached_pts_best_candidate(
    *,
    artifact_dir: Path,
    marker_path: Path,
    marker: Mapping[str, object],
    target_item: int,
    shared_cache_key: str,
) -> CachedPTSBestCandidate:
    best_candidate = marker.get("best_candidate")
    if not isinstance(best_candidate, Mapping):
        raise ValueError(
            f"PTS-CEM shared cache marker is missing best_candidate: {marker_path}"
        )
    sessions_path = _resolve_artifact_relative_path(
        artifact_dir,
        best_candidate.get("sessions_path"),
        label=f"{marker_path}: best_candidate.sessions_path",
    )
    metadata_path = _resolve_artifact_relative_path(
        artifact_dir,
        best_candidate.get("metadata_path"),
        label=f"{marker_path}: best_candidate.metadata_path",
    )
    sessions = _load_json_sessions(sessions_path)
    metadata = _load_json_dict(metadata_path)
    _validate_cached_candidate_metadata_target(
        metadata,
        target_item=int(target_item),
        label=str(metadata_path),
    )
    merged_metadata = dict(metadata)
    for key in (
        "rank",
        "iteration",
        "candidate_id",
        "candidate_seed",
        "reward",
        "reward_metrics",
    ):
        if key not in merged_metadata and key in best_candidate:
            merged_metadata[key] = best_candidate[key]
    top_candidates_path = artifact_dir / "pts_top_candidates.json"
    return CachedPTSBestCandidate(
        sessions=sessions,
        metadata=merged_metadata,
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        top_candidates_path=(
            top_candidates_path if top_candidates_path.exists() else None
        ),
        complete_marker_path=marker_path,
        cache_mode="shared_complete_marker",
        cache_marker_missing=False,
        shared_pts_cem_cache_key=str(shared_cache_key),
        shared_cache_path=Path(artifact_dir),
        reused_shared_pts_cem=True,
        local_materialized_from_shared=False,
    )


def _write_pts_construction_complete_marker(
    *,
    config: Config,
    target_item: int,
    artifact_dir: Path,
    artifact_paths: Mapping[str, str],
    best_candidate,
    attack_identity_context: Mapping[str, Any] | None,
    shared_cache_key: str | None = None,
    shared_cache_path: Path | None = None,
    reused_shared_pts_cem: bool = False,
    local_materialized_from_shared: bool = False,
) -> Path:
    root = Path(artifact_dir)
    sessions_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_sessions",
    )
    metadata_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_metadata",
    )
    policy_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_policy",
    )
    marker_path = root / _PTS_CONSTRUCTION_COMPLETE_MARKER
    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
    payload = {
        "schema_version": "pts_construction_cache_v1",
        "local_artifact_schema_version": PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION,
        "status": "completed",
        "run_type": _pts_construction_run_type(config),
        "run_group_key": run_group_key(
            config,
            run_type=_pts_construction_run_type(config),
            attack_identity_context=attack_identity_context,
        ),
        "target_cohort_key": target_cohort_key(config),
        "target_item": int(target_item),
        "cache_mode": (
            "shared_pts_cem_materialized"
            if bool(local_materialized_from_shared)
            else "fresh_cem"
        ),
        "shared_pts_cem_cache_key": shared_cache_key,
        "shared_cache_path": (
            None if shared_cache_path is None else str(shared_cache_path)
        ),
        "reused_shared_pts_cem": bool(reused_shared_pts_cem),
        "local_materialized_from_shared": bool(local_materialized_from_shared),
        **pts_cem_surrogate_retrain_metadata(config),
        **_pts_cem_surrogate_retrain_marker_reuse_metadata(
            config,
            reused_shared_pts_cem=bool(reused_shared_pts_cem),
        ),
        **seed_alignment,
        "identity": _current_pts_construction_cache_identity(
            config,
            attack_identity_context=attack_identity_context,
            target_item=int(target_item),
        ),
        "best_candidate": {
            "rank": 1,
            "iteration": _candidate_marker_int(best_candidate, "iteration"),
            "candidate_id": _candidate_marker_int(best_candidate, "candidate_id"),
            "candidate_seed": _candidate_marker_int(best_candidate, "candidate_seed"),
            "reward": _candidate_marker_float(best_candidate, "reward"),
            "reward_metrics": _candidate_marker_mapping(
                best_candidate,
                "reward_metrics",
            ),
            "sessions_path": _relative_to_artifact_dir(root, sessions_path),
            "metadata_path": _relative_to_artifact_dir(root, metadata_path),
            "policy_path": _relative_to_artifact_dir(root, policy_path),
        },
    }
    save_json(_to_jsonable_cache_payload(payload), marker_path)
    return marker_path


def _candidate_marker_value(candidate, key: str) -> object:
    if isinstance(candidate, Mapping):
        return candidate.get(key)
    return getattr(candidate, key)


def _candidate_marker_int(candidate, key: str) -> int:
    return _coerce_marker_int(
        _candidate_marker_value(candidate, key),
        label=f"best_candidate.{key}",
    )


def _candidate_marker_float(candidate, key: str) -> float:
    value = _candidate_marker_value(candidate, key)
    if value is None:
        raise ValueError(f"best_candidate.{key} is required.")
    return float(value)


def _candidate_marker_mapping(candidate, key: str) -> dict[str, object]:
    value = _candidate_marker_value(candidate, key)
    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def _materialize_shared_pts_cem_cache(
    *,
    config: Config,
    target_item: int,
    local_artifact_dir: Path,
    shared_cache_dir: Path,
    shared_cached: CachedPTSBestCandidate,
    shared_cache_key: str,
    attack_identity_context: Mapping[str, Any] | None,
    current_identity: Mapping[str, object],
) -> CachedPTSBestCandidate:
    local_root = Path(local_artifact_dir)
    if local_root.exists() and any(local_root.iterdir()):
        _raise_incompatible_local_pts_cache(
            local_root,
            ValueError("local folder is not empty before shared materialization"),
        )
    _copy_pts_cem_artifact_tree(
        source_dir=Path(shared_cache_dir),
        destination_dir=local_root,
        exclude_markers=True,
    )
    _prune_optional_pts_cem_artifacts_for_config(
        config=config,
        artifact_dir=local_root,
    )
    _rewrite_top_candidate_paths(local_root)
    required_paths = _required_pts_cem_relative_artifact_paths()
    _verify_relative_files_exist(local_root, required_paths)
    artifact_paths = _existing_pts_artifact_paths(local_root)
    marker_path = _write_pts_construction_complete_marker(
        config=config,
        target_item=int(target_item),
        artifact_dir=local_root,
        artifact_paths=artifact_paths,
        best_candidate=shared_cached.metadata,
        attack_identity_context=attack_identity_context,
        shared_cache_key=shared_cache_key,
        shared_cache_path=Path(shared_cache_dir),
        reused_shared_pts_cem=True,
        local_materialized_from_shared=True,
    )
    cached = _try_load_cached_pts_best_candidate(
        artifact_dir=local_root,
        target_item=int(target_item),
        current_identity=current_identity,
        current_shared_cache_key=shared_cache_key,
    )
    if cached is None:
        raise ValueError(f"Materialized PTS-CEM cache was not loadable: {marker_path}")
    return cached


def _prune_optional_pts_cem_artifacts_for_config(
    *,
    config: Config,
    artifact_dir: Path,
) -> None:
    pts_config = _require_pts_config(config)
    root = Path(artifact_dir)
    top_k = int(pts_config.cem.save_top_k_candidates)
    top_candidates_dir = root / "top_candidates"
    if top_candidates_dir.exists():
        for rank_dir in top_candidates_dir.glob("rank_*"):
            if not rank_dir.is_dir():
                continue
            try:
                rank = int(rank_dir.name.removeprefix("rank_"))
            except ValueError:
                continue
            if rank > top_k:
                shutil.rmtree(rank_dir)
                continue
            if not bool(pts_config.artifacts.save_per_session_records):
                session_records_path = rank_dir / "session_records.jsonl"
                if session_records_path.exists():
                    session_records_path.unlink()

    _prune_top_candidate_summary_json(
        root / "pts_top_candidates.json",
        top_k=top_k,
        save_per_session_records=bool(pts_config.artifacts.save_per_session_records),
    )
    _prune_top_candidate_summary_json(
        root / "pts_top_candidate_policies.json",
        top_k=top_k,
        save_per_session_records=bool(pts_config.artifacts.save_per_session_records),
    )


def _prune_top_candidate_summary_json(
    path: Path,
    *,
    top_k: int,
    save_per_session_records: bool,
) -> None:
    if not path.exists():
        return
    payload = _load_json_dict(path)
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        pruned: list[object] = []
        for row in candidates:
            if not isinstance(row, Mapping):
                pruned.append(row)
                continue
            try:
                rank = int(row.get("rank", len(pruned) + 1))
            except (TypeError, ValueError):
                rank = len(pruned) + 1
            if rank > top_k:
                continue
            row = dict(row)
            if not save_per_session_records and "session_records_path" in row:
                row["session_records_path"] = None
            pruned.append(row)
        payload["candidates"] = pruned
        if "top_k" in payload:
            payload["top_k"] = min(int(top_k), len(pruned))
    save_json(_to_jsonable_cache_payload(payload), path)


def _write_shared_pts_cem_cache(
    *,
    config: Config,
    target_item: int,
    local_artifact_dir: Path,
    artifact_paths: Mapping[str, str],
    best_candidate,
    shared_cache_dir: Path,
    shared_cache_key: str,
    shared_cache_identity: Mapping[str, object],
    attack_identity_context: Mapping[str, Any] | None,
) -> Path:
    shared_root_dir = Path(shared_cache_dir)
    _remove_directory_inside(
        shared_root_dir,
        shared_root(config) / _PTS_CONSTRUCTION_ARTIFACT_DIR_NAME,
    )
    _copy_pts_cem_artifact_tree(
        source_dir=Path(local_artifact_dir),
        destination_dir=shared_root_dir,
        exclude_markers=True,
    )
    _rewrite_top_candidate_paths(shared_root_dir)
    required_paths = _required_pts_cem_relative_artifact_paths()
    _verify_relative_files_exist(shared_root_dir, required_paths)
    completeness = _artifact_completeness_metadata(
        config=config,
        artifact_dir=shared_root_dir,
    )
    marker_path = shared_root_dir / _PTS_CONSTRUCTION_SHARED_COMPLETE_MARKER
    sessions_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_sessions",
    )
    metadata_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_metadata",
    )
    policy_path = _required_artifact_path(
        artifact_paths,
        "top_candidate_rank_1_policy",
    )
    payload = {
        "schema_version": PTS_CEM_SHARED_CACHE_SCHEMA_VERSION,
        "status": "completed",
        "run_type": _pts_construction_run_type(config),
        "shared_pts_cem_cache_key": str(shared_cache_key),
        "target_item": int(target_item),
        "construction_identity": dict(shared_cache_identity),
        "fake_sessions_artifact_identity": (
            dict(shared_cache_identity)
            .get("fake_sessions", {})
            .get("artifact_identity")
            if isinstance(dict(shared_cache_identity).get("fake_sessions"), Mapping)
            else None
        ),
        "seed_alignment": pts_cem_surrogate_seed_alignment_metadata(
            config,
            target_item=int(target_item),
        ),
        "required_artifact_files": list(required_paths),
        "artifact_completeness": completeness,
        **pts_cem_surrogate_retrain_metadata(config),
        "created_from_experiment": config.experiment.name,
        "created_from_run_group": run_group_key(
            config,
            run_type=_pts_construction_run_type(config),
            attack_identity_context=attack_identity_context,
        ),
        "best_candidate": {
            "rank": 1,
            "iteration": _candidate_marker_int(best_candidate, "iteration"),
            "candidate_id": _candidate_marker_int(best_candidate, "candidate_id"),
            "candidate_seed": _candidate_marker_int(best_candidate, "candidate_seed"),
            "reward": _candidate_marker_float(best_candidate, "reward"),
            "reward_metrics": _candidate_marker_mapping(
                best_candidate,
                "reward_metrics",
            ),
            "sessions_path": _relative_to_artifact_dir(
                Path(local_artifact_dir),
                sessions_path,
            ),
            "metadata_path": _relative_to_artifact_dir(
                Path(local_artifact_dir),
                metadata_path,
            ),
            "policy_path": _relative_to_artifact_dir(
                Path(local_artifact_dir),
                policy_path,
            ),
        },
    }
    save_json(_to_jsonable_cache_payload(payload), marker_path)
    return marker_path


def _copy_pts_cem_artifact_tree(
    *,
    source_dir: Path,
    destination_dir: Path,
    exclude_markers: bool,
) -> None:
    source_root = Path(source_dir)
    dest_root = Path(destination_dir)
    dest_root.mkdir(parents=True, exist_ok=True)
    marker_names = {
        _PTS_CONSTRUCTION_COMPLETE_MARKER,
        _PTS_CONSTRUCTION_SHARED_COMPLETE_MARKER,
    }
    for child in source_root.iterdir():
        if exclude_markers and child.name in marker_names:
            continue
        dest_child = dest_root / child.name
        if child.is_dir():
            if dest_child.exists():
                shutil.rmtree(dest_child)
            shutil.copytree(child, dest_child)
        else:
            shutil.copy2(child, dest_child)


def _required_pts_cem_relative_artifact_paths() -> list[str]:
    return [
        "pts_cem_trace.jsonl",
        "pts_policy_history.json",
        "pts_best_policy.json",
        "pts_final_policy.json",
        "pts_top_candidates.json",
        "pts_top_candidate_policies.json",
        "top_candidates/rank_1/policy.json",
        "top_candidates/rank_1/sessions.json",
        "top_candidates/rank_1/metadata.json",
    ]


def _verify_relative_files_exist(root: Path, relative_paths: Sequence[str]) -> None:
    missing = [
        str(Path(root) / rel_path)
        for rel_path in relative_paths
        if not (Path(root) / rel_path).exists()
    ]
    if missing:
        raise ValueError(
            "PTS-CEM artifact tree is missing required files: "
            + ", ".join(missing)
        )


def _rewrite_top_candidate_paths(artifact_dir: Path) -> None:
    root = Path(artifact_dir)
    path = root / "pts_top_candidates.json"
    if not path.exists():
        return
    payload = _load_json_dict(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return
    for row in candidates:
        if not isinstance(row, dict):
            continue
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            continue
        rank_dir = root / "top_candidates" / f"rank_{rank}"
        row["policy_path"] = str(rank_dir / "policy.json")
        row["sessions_path"] = (
            str(rank_dir / "sessions.json")
            if (rank_dir / "sessions.json").exists()
            else None
        )
        row["session_records_path"] = (
            str(rank_dir / "session_records.jsonl")
            if (rank_dir / "session_records.jsonl").exists()
            else None
        )
        row["metadata_path"] = str(rank_dir / "metadata.json")
    save_json(_to_jsonable_cache_payload(payload), path)


def _artifact_completeness_metadata(
    *,
    config: Config,
    artifact_dir: Path,
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    root = Path(artifact_dir)
    top_candidates = _load_json_dict(root / "pts_top_candidates.json")
    rows = top_candidates.get("candidates", [])
    available_ranks: list[int] = []
    candidate_keys_with_sessions: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                rank = int(row.get("rank"))
            except (TypeError, ValueError):
                continue
            available_ranks.append(rank)
            sessions_path = root / "top_candidates" / f"rank_{rank}" / "sessions.json"
            candidate_key = row.get("candidate_key")
            if sessions_path.exists() and candidate_key is not None:
                candidate_keys_with_sessions.append(str(candidate_key))
    return {
        "save_top_k_candidates": int(pts_config.cem.save_top_k_candidates),
        "save_candidate_sessions": bool(pts_config.artifacts.save_candidate_sessions),
        "save_best_sessions": bool(pts_config.artifacts.save_best_sessions),
        "save_top_candidate_sessions": bool(
            pts_config.artifacts.save_top_candidate_sessions
        ),
        "available_candidate_keys_with_sessions": candidate_keys_with_sessions,
        "available_top_candidate_ranks": sorted(set(available_ranks)),
        "best_sessions_path": "top_candidates/rank_1/sessions.json",
    }


def _required_artifact_path(
    artifact_paths: Mapping[str, str],
    key: str,
) -> Path:
    value = artifact_paths.get(key)
    if not value:
        raise ValueError(f"PTS-CEM artifact writer did not return required path {key!r}.")
    return Path(value)


def _relative_to_artifact_dir(artifact_dir: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(artifact_dir).resolve()).as_posix()
    except ValueError:
        return str(path)


def _to_jsonable_cache_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable_cache_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable_cache_payload(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _to_jsonable_cache_payload(value.item())
        except Exception:
            pass
    return str(value)


def _existing_pts_artifact_paths(artifact_dir: Path) -> dict[str, str]:
    root = Path(artifact_dir)
    candidates = {
        "pts_cem_trace": root / "pts_cem_trace.jsonl",
        "pts_policy_history": root / "pts_policy_history.json",
        "pts_best_policy": root / "pts_best_policy.json",
        "pts_final_policy": root / "pts_final_policy.json",
        "pts_top_candidates": root / "pts_top_candidates.json",
        "pts_top_candidate_policies": root / "pts_top_candidate_policies.json",
        "pts_epoch_reward_ranking_summary_json": (
            root / "pts_epoch_reward_ranking_summary.json"
        ),
        "pts_epoch_reward_ranking_summary_csv": (
            root / "pts_epoch_reward_ranking_summary.csv"
        ),
        "top_candidate_rank_1_policy": _rank1_policy_path(root),
        "top_candidate_rank_1_sessions": _rank1_sessions_path(root),
        "top_candidate_rank_1_metadata": _rank1_metadata_path(root),
        "pts_construction_complete_marker": root / _PTS_CONSTRUCTION_COMPLETE_MARKER,
    }
    return {key: str(path) for key, path in candidates.items() if path.exists()}


def _build_candidate_evaluator_context(
    config: Config,
    shared: SharedAttackArtifacts,
) -> dict[str, object]:
    train_config = _srgnn_candidate_train_config(config)
    validation_sessions, validation_labels = _resolve_validation_pairs(shared)
    protocol = _pts_cem_surrogate_retrain_protocol(config)
    retrain_epochs = int(train_config["epochs"])
    if protocol == PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST:
        print(
            f"{_LOG_PREFIX} PTS-CEM surrogate validation_best mode: using "
            "per-epoch validation metrics and best checkpoint selection."
        )
        inner_trainer = SRGNNFullRetrainValidationBestInnerTrainer(
            train_config=train_config,
            max_epochs=retrain_epochs,
            patience=int(train_config["patience"]),
            log_prefix="[pts-cem:candidate-retrain]",
            log_epochs=False,
        )
        validation_eval_data = Data((validation_sessions, validation_labels), shuffle=False)
    elif protocol == PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST:
        print(
            f"{_LOG_PREFIX} PTS-CEM surrogate fixed_last mode: training for "
            f"{retrain_epochs} epochs, skipping per-epoch validation-best "
            "checkpointing, using last model for reward."
        )
        inner_trainer = SRGNNFullRetrainFixedLastInnerTrainer(
            train_config=train_config,
            max_epochs=retrain_epochs,
            log_prefix="[pts-cem:candidate-retrain-fixed-last]",
            log_epochs=False,
        )
        validation_eval_data = None
    else:
        raise ValueError(f"Unsupported PTS-CEM surrogate retrain protocol: {protocol!r}")
    return {
        "backend": SRGNNBackend(config, base_dir=Path.cwd(), train_config=train_config),
        "inner_trainer": inner_trainer,
        "validation_sessions": validation_sessions,
        "validation_labels": validation_labels,
        "validation_eval_data": validation_eval_data,
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
    population_size: int,
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
    if not isinstance(
        inner_trainer,
        (
            SRGNNFullRetrainValidationBestInnerTrainer,
            SRGNNFullRetrainFixedLastInnerTrainer,
        ),
    ):
        raise TypeError("PTS-CEM evaluator context has invalid inner trainer.")

    pts_config = _require_pts_config(config)
    diagnostics_config = pts_config.cem.epoch_reward_diagnostics
    diagnostics_enabled = bool(diagnostics_config.enabled)
    diagnostic_epochs = {int(epoch) for epoch in diagnostics_config.epochs}
    epoch_rewards: dict[int, dict[str, object]] = {}

    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
    surrogate_effective_seed = int(seed_alignment["resolved_surrogate_effective_seed"])
    candidate_start = time.perf_counter()
    poisoned_train = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        candidate_sessions,
    )
    print(
        f"{_LOG_PREFIX} target={int(target_item)} "
        f"iter={int(iteration) + 1}/{int(pts_config.cem.iterations)} "
        f"population={int(candidate_id) + 1}/{int(population_size)} "
        f"seed={int(candidate_seed)}"
    )

    def epoch_callback(model: object, row: Mapping[str, Any]) -> None:
        epoch = int(row["epoch"])
        if epoch not in diagnostic_epochs:
            return
        reward, reward_metrics, score_seconds = _score_candidate_target_reward(
            backend=backend,
            model=model,
            validation_sessions=validation_sessions,
            target_item=int(target_item),
        )
        epoch_payload = _epoch_reward_payload_from_metrics(
            reward=reward,
            reward_metrics=reward_metrics,
        )
        epoch_payload.update(
            {
                "epoch": int(epoch),
                "reward_source": "epoch_reward_diagnostic",
                "checkpoint_mode": "current_epoch",
                "epoch_diagnostic_checkpoint_mode": "current_epoch",
                "score_target_seconds": float(score_seconds),
            }
        )
        if row.get("valid_ground_truth_mrr@20") is not None:
            epoch_payload["ground_truth_mrr@20"] = float(
                row["valid_ground_truth_mrr@20"]
            )
        if row.get("valid_ground_truth_recall@20") is not None:
            epoch_payload["ground_truth_recall@20"] = float(
                row["valid_ground_truth_recall@20"]
            )
        epoch_rewards[int(epoch)] = epoch_payload

    retrain_start = time.perf_counter()
    inner_result = inner_trainer.run(
        backend,
        None,
        poisoned_train,
        config=None,
        eval_data=validation_eval_data,
        seed=surrogate_effective_seed,
        epoch_callback=epoch_callback if diagnostics_enabled else None,
    )
    retrain_seconds = time.perf_counter() - retrain_start

    reward, reward_metrics, score_target_seconds = _score_candidate_target_reward(
        backend=backend,
        model=inner_result.model,
        validation_sessions=validation_sessions,
        target_item=int(target_item),
    )
    candidate_total_seconds = time.perf_counter() - candidate_start
    checkpoint_metadata = _candidate_checkpoint_metadata(inner_result.history)
    metadata: dict[str, object] = {
        "reward_name": PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
        "candidate_retrain_validation_reward": reward,
        "candidate_retrain_seed": surrogate_effective_seed,
        **seed_alignment,
        **pts_cem_surrogate_retrain_metadata(config),
        "candidate_seed": int(candidate_seed),
        "candidate_retrain_validation_prefix_count": int(len(validation_sessions)),
        "candidate_retrain_epochs": _resolved_pts_candidate_retrain_epochs(config),
        "iteration": int(iteration),
        "candidate_id": int(candidate_id),
        "candidate_total_seconds": float(candidate_total_seconds),
        "candidate_retrain_seconds": float(retrain_seconds),
        "score_target_seconds": float(score_target_seconds),
    }
    metadata.update(checkpoint_metadata)
    epoch_reward_diagnostics = None
    if diagnostics_enabled:
        _validate_collected_epoch_diagnostics(
            requested_epochs=diagnostic_epochs,
            collected_epochs=set(epoch_rewards),
            history=inner_result.history,
        )
        training_budget_epoch = _resolved_pts_candidate_retrain_epochs(config)
        selected_checkpoint_epoch = checkpoint_metadata.get("selected_checkpoint_epoch")
        if bool(diagnostics_config.include_final_epoch) and training_budget_epoch not in epoch_rewards:
            final_payload = _epoch_reward_payload_from_metrics(
                reward=reward,
                reward_metrics=reward_metrics,
            )
            final_payload.update(
                {
                    "epoch": int(training_budget_epoch),
                    "reward_source": "official_reward",
                    "checkpoint_mode": "final_partial_retrain_protocol",
                    "official_reward_source": "final_partial_retrain_protocol",
                    "training_budget_epoch": int(training_budget_epoch),
                    "selected_checkpoint_epoch": selected_checkpoint_epoch,
                    "selected_checkpoint_protocol": checkpoint_metadata.get(
                        "selected_checkpoint_protocol"
                    ),
                    "selected_checkpoint_source": checkpoint_metadata.get(
                        "selected_checkpoint_source"
                    ),
                    "selected_checkpoint_metric": checkpoint_metadata.get(
                        "selected_checkpoint_metric"
                    ),
                    "official_reward_checkpoint_epoch": checkpoint_metadata.get(
                        "official_reward_checkpoint_epoch",
                        selected_checkpoint_epoch,
                    ),
                    "score_target_seconds": float(score_target_seconds),
                }
            )
            if checkpoint_metadata.get("valid_ground_truth_mrr@20") is not None:
                final_payload["ground_truth_mrr@20"] = float(
                    checkpoint_metadata["valid_ground_truth_mrr@20"]
                )
            if checkpoint_metadata.get("valid_ground_truth_recall@20") is not None:
                final_payload["ground_truth_recall@20"] = float(
                    checkpoint_metadata["valid_ground_truth_recall@20"]
                )
            epoch_rewards[int(training_budget_epoch)] = final_payload
        epoch_reward_diagnostics = {
            "enabled": True,
            "reward_name": PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
            "diagnostic_epochs": sorted(int(epoch) for epoch in diagnostic_epochs),
            "include_final_epoch": bool(diagnostics_config.include_final_epoch),
            "write_candidate_epoch_metrics": bool(
                diagnostics_config.write_candidate_epoch_metrics
            ),
            "write_ranking_summary": bool(diagnostics_config.write_ranking_summary),
            "official_reward_source": "final_partial_retrain_protocol",
            "training_budget_epoch": int(training_budget_epoch),
            "selected_checkpoint_epoch": selected_checkpoint_epoch,
            "selected_checkpoint_protocol": checkpoint_metadata.get(
                "selected_checkpoint_protocol"
            ),
            "selected_checkpoint_source": checkpoint_metadata.get(
                "selected_checkpoint_source"
            ),
            "selected_checkpoint_metric": checkpoint_metadata.get(
                "selected_checkpoint_metric"
            ),
            "official_reward_checkpoint_epoch": checkpoint_metadata.get(
                "official_reward_checkpoint_epoch",
                selected_checkpoint_epoch,
            ),
            "epoch_diagnostic_checkpoint_mode": "current_epoch",
            "target_item": int(target_item),
            "surrogate_effective_seed": surrogate_effective_seed,
            "surrogate_victim_seed_aligned": True,
            "rewards_by_epoch": {
                str(epoch): epoch_rewards[epoch]
                for epoch in sorted(epoch_rewards)
            },
        }
        metadata["epoch_reward_diagnostics_enabled"] = True
        metadata["diagnostic_epochs"] = sorted(int(epoch) for epoch in diagnostic_epochs)
        metadata["official_reward_source"] = "final_partial_retrain_protocol"
        metadata["training_budget_epoch"] = int(training_budget_epoch)
        metadata["epoch_diagnostic_checkpoint_mode"] = "current_epoch"
        metadata["include_final_epoch"] = bool(diagnostics_config.include_final_epoch)
        metadata["write_ranking_summary"] = bool(diagnostics_config.write_ranking_summary)
    return PTSCEMEvaluationResult(
        reward=reward,
        reward_metrics=reward_metrics,
        metadata=metadata,
        epoch_reward_diagnostics=epoch_reward_diagnostics,
    )


def _score_candidate_target_reward(
    *,
    backend: SRGNNBackend,
    model: object,
    validation_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> tuple[float, dict[str, float], float]:
    score_start = time.perf_counter()
    target_result = backend.score_target(
        model,
        validation_sessions,
        int(target_item),
    )
    score_target_seconds = time.perf_counter() - score_start
    metrics = _coerce_target_metrics(target_result.metrics)
    lowk_payload = _lowk_reward_metric_payload(metrics)
    reward = float(lowk_payload["absolute_raw_family_lowk_reward"])
    reward_metrics = {
        **metrics,
        **{
            key: float(value)
            for key, value in lowk_payload.items()
            if isinstance(value, (int, float))
        },
        PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20: reward,
    }
    return reward, reward_metrics, float(score_target_seconds)


def _epoch_reward_payload_from_metrics(
    *,
    reward: float,
    reward_metrics: Mapping[str, float],
) -> dict[str, object]:
    payload: dict[str, object] = {
        key: float(value)
        for key, value in reward_metrics.items()
    }
    payload["target_summary_value"] = float(reward)
    payload[PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20] = float(reward)
    return payload


def _validate_collected_epoch_diagnostics(
    *,
    requested_epochs: set[int],
    collected_epochs: set[int],
    history: Mapping[str, Any] | None,
) -> None:
    missing = sorted(set(requested_epochs) - set(collected_epochs))
    if not missing:
        return
    stopped_epoch = None
    if isinstance(history, Mapping) and history.get("stopped_epoch") is not None:
        stopped_epoch = int(history["stopped_epoch"])
    detail = (
        "" if stopped_epoch is None else f" Retrain stopped at epoch {stopped_epoch}."
    )
    raise RuntimeError(
        "PTS-CEM epoch reward diagnostics were requested for epochs "
        f"{missing}, but those epochs were not evaluated.{detail}"
    )


def _candidate_checkpoint_metadata(
    history: Mapping[str, Any] | None,
) -> dict[str, Any]:
    history_map = dict(history or {})
    checkpoint_protocol = history_map.get("checkpoint_protocol")
    if checkpoint_protocol == PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST:
        selected_epoch = history_map.get("selected_checkpoint_epoch")
        official_epoch = history_map.get("official_reward_checkpoint_epoch")
        return {
            "selected_checkpoint_epoch": (
                None if selected_epoch is None else int(selected_epoch)
            ),
            "selected_checkpoint_protocol": PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST,
            "selected_checkpoint_source": "last_epoch",
            "selected_checkpoint_metric": None,
            "validation_best_metrics_recorded": False,
            "official_reward_checkpoint_epoch": (
                None if official_epoch is None else int(official_epoch)
            ),
        }
    return _rank_bucket_candidate_checkpoint_metadata(history)


def _srgnn_candidate_train_config(config: Config) -> dict[str, Any]:
    victim_params = config.victims.params.get("srgnn")
    if not isinstance(victim_params, Mapping):
        raise ValueError("PTS-CEM Phase 3 requires victims.params.srgnn.")
    train_config = victim_params.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError("PTS-CEM Phase 3 requires victims.params.srgnn.train.")
    return dict(train_config)


def _resolved_pts_candidate_retrain_epochs(config: Config) -> int:
    train_config = _srgnn_candidate_train_config(config)
    epochs = train_config.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("victims.params.srgnn.train.epochs must be an integer.")
    if int(epochs) <= 0:
        raise ValueError("victims.params.srgnn.train.epochs must be positive.")
    return int(epochs)


def _validate_pts_epoch_reward_diagnostics_config(config: Config) -> None:
    diagnostics = _require_pts_config(config).cem.epoch_reward_diagnostics
    if not bool(diagnostics.enabled):
        return
    retrain_epochs = _resolved_pts_candidate_retrain_epochs(config)
    if bool(diagnostics.include_final_epoch):
        reserved = [
            int(epoch)
            for epoch in diagnostics.epochs
            if int(epoch) >= int(retrain_epochs)
        ]
        if reserved:
            raise ValueError(
                "attack.pts_construction.cem.epoch_reward_diagnostics.epochs "
                "must be less than resolved PTS-CEM surrogate candidate retrain "
                "epochs when include_final_epoch=true, because the final epoch "
                "slot is reserved for the official final_partial_retrain_protocol "
                f"reward; retrain_epochs={retrain_epochs}, invalid epochs: {reserved}."
            )
    invalid = [
        int(epoch)
        for epoch in diagnostics.epochs
        if int(epoch) > int(retrain_epochs)
    ]
    if invalid:
        raise ValueError(
            "attack.pts_construction.cem.epoch_reward_diagnostics.epochs "
            "must be <= resolved PTS-CEM surrogate candidate retrain epochs "
            f"({retrain_epochs}); invalid epochs: {invalid}."
        )


def _epoch_reward_diagnostics_metadata_payload(
    config: Config,
    *,
    artifact_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    diagnostics = _require_pts_config(config).cem.epoch_reward_diagnostics
    payload: dict[str, object] = {
        "epoch_reward_diagnostics_enabled": bool(diagnostics.enabled),
        "diagnostic_epochs": [int(epoch) for epoch in diagnostics.epochs],
        "include_final_epoch": bool(diagnostics.include_final_epoch),
        "write_candidate_epoch_metrics": bool(
            diagnostics.write_candidate_epoch_metrics
        ),
        "write_ranking_summary": bool(diagnostics.write_ranking_summary),
    }
    if bool(diagnostics.enabled):
        payload.update(
            {
                "official_reward_source": "final_partial_retrain_protocol",
                "training_budget_epoch": _resolved_pts_candidate_retrain_epochs(config),
                "epoch_diagnostic_checkpoint_mode": "current_epoch",
            }
        )
    if artifact_paths is not None:
        payload["pts_epoch_reward_ranking_summary_json_path"] = artifact_paths.get(
            "pts_epoch_reward_ranking_summary_json"
        )
        payload["pts_epoch_reward_ranking_summary_csv_path"] = artifact_paths.get(
            "pts_epoch_reward_ranking_summary_csv"
        )
    return payload


def _pts_method_metadata_payload(
    pts_config: PTSConstructionConfig,
) -> dict[str, object]:
    if pts_config.method == PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        continuous = pts_config.continuous_policy
        return {
            "pts_actions_enabled": [],
            "pts_grouping_mode": None,
            "pts_continuous_policy": {
                "parameterization": continuous.parameterization,
                "hidden_size": int(continuous.hidden_size),
                "consume_distribution": continuous.consume_distribution,
                "source_policy": continuous.source_policy,
                "parameter_bounds": {
                    "min": float(continuous.parameter_bounds.min),
                    "max": float(continuous.parameter_bounds.max),
                },
                "smoothing_epsilon": float(continuous.smoothing_epsilon),
                "deterministic_sampling": bool(continuous.deterministic_sampling),
            },
        }
    if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        policy = pts_config.direct_action_policy
        return {
            "pts_actions_enabled": [],
            "pts_grouping_mode": None,
            "pts_direct_action_policy": {
                "parameterization": policy.parameterization,
                "length_feature": policy.length_feature,
                "cem_init": {
                    "mode": "standard_normal",
                    "parameter_space": "standardized_policy_parameter_space",
                },
            },
        }
    return {
        "pts_actions_enabled": list(pts_config.actions.enabled),
        "pts_grouping_mode": pts_config.grouping.mode,
    }


def _target_metadata(
    *,
    config: Config,
    pts_config: PTSConstructionConfig,
    cem_config: PTSCEMConfig,
    artifact_dir: Path,
    artifact_paths: Mapping[str, str],
    best_candidate,
    target_item: int,
    complete_marker_path: Path,
    shared_cache_key: str | None = None,
    shared_cache_path: Path | None = None,
    reused_shared_pts_cem: bool = False,
    local_materialized_from_shared: bool = False,
) -> dict[str, object]:
    rank1_sessions = artifact_paths.get("top_candidate_rank_1_sessions")
    rank1_metadata = artifact_paths.get("top_candidate_rank_1_metadata")
    selected_sessions_sha1 = _selected_sessions_sha1(
        sessions_path=rank1_sessions,
        sessions=best_candidate.final_sessions,
    )
    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
    method_metadata = _pts_method_metadata_payload(pts_config)
    return {
        "pts_cem_trace_path": artifact_paths.get("pts_cem_trace"),
        "pts_policy_history_path": artifact_paths.get("pts_policy_history"),
        "pts_best_policy_path": artifact_paths.get("pts_best_policy"),
        "pts_final_policy_path": artifact_paths.get("pts_final_policy"),
        "pts_top_candidates_path": artifact_paths.get("pts_top_candidates"),
        "pts_top_candidate_policies_path": artifact_paths.get(
            "pts_top_candidate_policies"
        ),
        "pts_epoch_reward_ranking_summary_json_path": artifact_paths.get(
            "pts_epoch_reward_ranking_summary_json"
        ),
        "pts_epoch_reward_ranking_summary_csv_path": artifact_paths.get(
            "pts_epoch_reward_ranking_summary_csv"
        ),
        "pts_artifact_dir": str(artifact_dir),
        "pts_best_candidate_iteration": int(best_candidate.iteration),
        "pts_best_candidate_id": int(best_candidate.candidate_id),
        "pts_best_candidate_seed": int(best_candidate.candidate_seed),
        "pts_best_candidate_reward": float(best_candidate.reward),
        "pts_best_candidate_reward_metrics": dict(best_candidate.reward_metrics),
        "pts_best_candidate_sessions_path": rank1_sessions,
        "pts_best_candidate_metadata_path": rank1_metadata,
        "selected_pts_cem_sessions_sha1": selected_sessions_sha1,
        "source_candidate_rank": 1,
        "source_candidate_key": str(best_candidate.candidate_key),
        "pts_final_selection_mode": pts_config.final_selection.mode,
        "pts_construction_method": pts_config.method,
        "pts_population_schedule": (
            list(cem_config.population_schedule)
            if cem_config.population_schedule is not None
            else None
        ),
        "pts_population_size": cem_config.population_size,
        **method_metadata,
        "pts_reward_target_summary": pts_config.reward.target_summary,
        "pts_candidate_retrain_seed": int(
            seed_alignment["resolved_surrogate_effective_seed"]
        ),
        **_epoch_reward_diagnostics_metadata_payload(
            config,
            artifact_paths=artifact_paths,
        ),
        **pts_cem_surrogate_retrain_metadata(config),
        **seed_alignment,
        "pts_cem_reused": False,
        "pts_cem_cache_mode": "fresh_cem",
        "pts_cem_cache_marker_missing": False,
        "shared_pts_cem_cache_key": shared_cache_key,
        "shared_pts_cem_cache_path": (
            None if shared_cache_path is None else str(shared_cache_path)
        ),
        "reused_shared_pts_cem": bool(reused_shared_pts_cem),
        "local_materialized_from_shared": bool(local_materialized_from_shared),
        "pts_construction_complete_marker_path": str(complete_marker_path),
    }


def _target_metadata_from_cache(
    *,
    config: Config,
    pts_config: PTSConstructionConfig,
    cem_config: PTSCEMConfig,
    artifact_dir: Path,
    target_item: int,
    cached: CachedPTSBestCandidate,
) -> dict[str, object]:
    artifact_paths = _existing_pts_artifact_paths(artifact_dir)
    selected_sessions_sha1 = _selected_sessions_sha1(
        sessions_path=cached.sessions_path,
        sessions=cached.sessions,
    )
    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
    method_metadata = _pts_method_metadata_payload(pts_config)
    payload: dict[str, object] = {
        "pts_cem_trace_path": artifact_paths.get("pts_cem_trace"),
        "pts_policy_history_path": artifact_paths.get("pts_policy_history"),
        "pts_best_policy_path": artifact_paths.get("pts_best_policy"),
        "pts_final_policy_path": artifact_paths.get("pts_final_policy"),
        "pts_top_candidates_path": artifact_paths.get("pts_top_candidates"),
        "pts_top_candidate_policies_path": artifact_paths.get(
            "pts_top_candidate_policies"
        ),
        "pts_epoch_reward_ranking_summary_json_path": artifact_paths.get(
            "pts_epoch_reward_ranking_summary_json"
        ),
        "pts_epoch_reward_ranking_summary_csv_path": artifact_paths.get(
            "pts_epoch_reward_ranking_summary_csv"
        ),
        "pts_artifact_dir": str(artifact_dir),
        "pts_best_candidate_sessions_path": str(cached.sessions_path),
        "pts_best_candidate_metadata_path": str(cached.metadata_path),
        "selected_pts_cem_sessions_sha1": selected_sessions_sha1,
        "source_candidate_rank": int(cached.metadata.get("rank", 1)),
        "source_candidate_key": (
            None
            if cached.metadata.get("candidate_key") is None
            else str(cached.metadata["candidate_key"])
        ),
        "pts_final_selection_mode": pts_config.final_selection.mode,
        "pts_construction_method": pts_config.method,
        "pts_population_schedule": (
            list(cem_config.population_schedule)
            if cem_config.population_schedule is not None
            else None
        ),
        "pts_population_size": cem_config.population_size,
        **method_metadata,
        "pts_reward_target_summary": pts_config.reward.target_summary,
        "pts_candidate_retrain_seed": int(
            seed_alignment["resolved_surrogate_effective_seed"]
        ),
        **_epoch_reward_diagnostics_metadata_payload(
            config,
            artifact_paths=artifact_paths,
        ),
        **pts_cem_surrogate_retrain_metadata(config),
        **_pts_cem_surrogate_retrain_reuse_metadata(config, cached),
        **seed_alignment,
        "pts_cem_reused": True,
        "pts_cem_cache_mode": cached.cache_mode,
        "pts_cem_cache_marker_missing": bool(cached.cache_marker_missing),
        "shared_pts_cem_cache_key": cached.shared_pts_cem_cache_key,
        "shared_pts_cem_cache_path": (
            None if cached.shared_cache_path is None else str(cached.shared_cache_path)
        ),
        "reused_shared_pts_cem": bool(cached.reused_shared_pts_cem),
        "local_materialized_from_shared": bool(
            cached.local_materialized_from_shared
        ),
        "pts_reused_candidate_rank": 1,
        "pts_reused_sessions_path": str(cached.sessions_path),
        "pts_reused_metadata_path": str(cached.metadata_path),
    }
    if cached.complete_marker_path is not None:
        payload["pts_construction_complete_marker_path"] = str(
            cached.complete_marker_path
        )
    _copy_cached_best_candidate_fields(payload, cached.metadata)
    return payload


def _copy_cached_best_candidate_fields(
    payload: dict[str, object],
    metadata: Mapping[str, object],
) -> None:
    int_fields = {
        "iteration": "pts_best_candidate_iteration",
        "candidate_id": "pts_best_candidate_id",
        "candidate_seed": "pts_best_candidate_seed",
    }
    for source_key, target_key in int_fields.items():
        if source_key in metadata and metadata[source_key] is not None:
            payload[target_key] = _coerce_marker_int(
                metadata[source_key],
                label=f"cached metadata {source_key}",
            )
    if "reward" in metadata and metadata["reward"] is not None:
        payload["pts_best_candidate_reward"] = float(metadata["reward"])
    if "candidate_key" in metadata and metadata["candidate_key"] is not None:
        payload["source_candidate_key"] = str(metadata["candidate_key"])
    reward_metrics = metadata.get("reward_metrics")
    if isinstance(reward_metrics, Mapping):
        payload["pts_best_candidate_reward_metrics"] = dict(reward_metrics)
    epoch_diagnostics = metadata.get("epoch_reward_diagnostics")
    if isinstance(epoch_diagnostics, Mapping):
        payload["pts_best_candidate_epoch_reward_diagnostics"] = dict(
            epoch_diagnostics
        )


def _selected_sessions_sha1(
    *,
    sessions_path: str | Path | None,
    sessions: Sequence[Sequence[int]],
) -> str:
    if sessions_path is not None and Path(sessions_path).exists():
        return str(_file_sha1_identity(sessions_path)["sha1"])
    payload = [[int(item) for item in session] for session in sessions]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run PTS-CEM construction through the attack pipeline."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH,
        help="Path to a YAML config.",
    )
    parser.add_argument(
        "--force-recompute-pts-cem",
        action="store_true",
        help="Ignore existing target-level PTS-CEM best-candidate cache and rerun CEM.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    run_pts_construction_grouped_cem(
        config,
        config_path=config_path,
        force_recompute_pts_cem=bool(args.force_recompute_pts_cem),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH",
    "PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION",
    "PTS_CEM_SHARED_CACHE_SCHEMA_VERSION",
    "PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE",
    "PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME",
    "build_pts_cem_shared_cache_identity",
    "build_pts_construction_attack_identity_context",
    "main",
    "pts_cem_surrogate_seed_alignment_metadata",
    "pts_cem_shared_cache_dir",
    "pts_cem_shared_cache_key",
    "resolve_pts_cem_surrogate_effective_seed",
    "run_pts_construction_grouped_cem",
    "_build_pts_cem_config_from_config",
    "_build_pts_specs_from_config",
    "_build_suffix_length_buckets_from_config",
    "_load_json_dict",
    "_load_json_sessions",
    "_materialize_shared_pts_cem_cache",
    "_resolve_pts_cem_base_seed",
    "_try_load_shared_pts_cem_cache",
    "_try_load_cached_pts_best_candidate",
    "_validate_pts_construction_run_config",
    "_write_pts_construction_complete_marker",
]
