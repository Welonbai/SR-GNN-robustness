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
from attack.common.artifact_io import load_json, save_json
from attack.common.paths import (
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
    PTSCEMResamplingConfig,
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
    "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)
_LOG_PREFIX = "[pts-construction-cem]"
_PTS_CONSTRUCTION_ARTIFACT_DIR_NAME = "pts_construction_cem"
_PTS_CONSTRUCTION_COMPLETE_MARKER = "pts_construction_complete.json"
_PTS_CONSTRUCTION_SHARED_COMPLETE_MARKER = "pts_cem_shared_complete.json"
PTS_CEM_SHARED_CACHE_SCHEMA_VERSION = "pts_cem_shared_construction_cache_v1"
PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION = "pts_cem_local_artifact_v2"
PTS_CEM_SURROGATE_SEED_ALIGNMENT_MODE = "victim_effective_seed"
PTS_CEM_SURROGATE_SEED_ALIGNMENT_TARGET_VICTIM_NAME = "srgnn"


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
                print(
                    f"{_LOG_PREFIX} Reusing cached PTS-CEM best candidate for "
                    f"target {int(target_item)}; skipping CEM."
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
                    f"{_LOG_PREFIX} Reusing shared PTS-CEM construction cache "
                    f"for target {int(target_item)}"
                )
                print(f"{_LOG_PREFIX} shared_pts_cem_cache_key = {shared_cache_key}")
                print(f"{_LOG_PREFIX} shared_cache_path = {shared_cache_dir}")
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
                print(
                    f"{_LOG_PREFIX} Materialized shared PTS-CEM cache into "
                    "local run_group target folder."
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
            print(
                f"{_LOG_PREFIX} No compatible shared PTS-CEM construction cache "
                f"found for target {int(target_item)}; running CEM."
            )

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

        artifact_paths = write_pts_cem_artifacts(
            result=result,
            output_dir=pts_artifact_dir,
            save_top_candidate_sessions=bool(
                pts_config.artifacts.save_top_candidate_sessions
            ),
            save_per_session_records=bool(pts_config.artifacts.save_per_session_records),
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
        print(f"{_LOG_PREFIX} Wrote shared PTS-CEM construction cache:")
        print(f"{_LOG_PREFIX} shared_pts_cem_cache_key = {shared_cache_key}")
        print(f"{_LOG_PREFIX} shared_cache_path = {shared_cache_dir}")
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
    if not bool(pts_config.artifacts.save_top_candidate_sessions):
        raise ValueError(
            "PTS-CEM runner requires artifacts.save_top_candidate_sessions == true "
            "because victim append reuse depends on top_candidates/rank_1/sessions.json."
        )
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
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
    split_config = config.data.canonical_split
    poison_checkpoint_identity = (
        _file_sha1_identity(poison_model_path)
        if poison_model_path is not None and Path(poison_model_path).exists()
        else None
    )
    return {
        "schema_version": PTS_CEM_SHARED_CACHE_SCHEMA_VERSION,
        "artifact_schema_version": PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION,
        "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
                run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            ),
        },
        "poison_model": {
            "key_payload": poison_model_key_payload(config),
            "checkpoint_identity": poison_checkpoint_identity,
        },
        "pts_construction": _pts_construction_shared_identity_payload(
            config,
            target_item=int(target_item),
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
) -> dict[str, object]:
    pts_config = _require_pts_config(config)
    cem = pts_config.cem
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
        "attack_key": attack_key(
            config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        ),
        "run_group_key": run_group_key(
            config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
    if marker.get("run_type") != PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE:
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
        "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        "run_group_key": run_group_key(
            config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
        "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
        "created_from_experiment": config.experiment.name,
        "created_from_run_group": run_group_key(
            config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
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
    retrain_start = time.perf_counter()
    inner_result = inner_trainer.run(
        backend,
        None,
        poisoned_train,
        config=None,
        eval_data=validation_eval_data,
        seed=surrogate_effective_seed,
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
        "candidate_retrain_seed": surrogate_effective_seed,
        **seed_alignment,
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
    target_item: int,
    complete_marker_path: Path,
    shared_cache_key: str | None = None,
    shared_cache_path: Path | None = None,
    reused_shared_pts_cem: bool = False,
    local_materialized_from_shared: bool = False,
) -> dict[str, object]:
    rank1_sessions = artifact_paths.get("top_candidate_rank_1_sessions")
    rank1_metadata = artifact_paths.get("top_candidate_rank_1_metadata")
    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
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
        "pts_candidate_retrain_seed": int(
            seed_alignment["resolved_surrogate_effective_seed"]
        ),
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
    seed_alignment = pts_cem_surrogate_seed_alignment_metadata(
        config,
        target_item=int(target_item),
    )
    payload: dict[str, object] = {
        "pts_cem_trace_path": artifact_paths.get("pts_cem_trace"),
        "pts_policy_history_path": artifact_paths.get("pts_policy_history"),
        "pts_best_policy_path": artifact_paths.get("pts_best_policy"),
        "pts_final_policy_path": artifact_paths.get("pts_final_policy"),
        "pts_top_candidates_path": artifact_paths.get("pts_top_candidates"),
        "pts_top_candidate_policies_path": artifact_paths.get(
            "pts_top_candidate_policies"
        ),
        "pts_artifact_dir": str(artifact_dir),
        "pts_best_candidate_sessions_path": str(cached.sessions_path),
        "pts_best_candidate_metadata_path": str(cached.metadata_path),
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
        "pts_candidate_retrain_seed": int(
            seed_alignment["resolved_surrogate_effective_seed"]
        ),
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
    reward_metrics = metadata.get("reward_metrics")
    if isinstance(reward_metrics, Mapping):
        payload["pts_best_candidate_reward_metrics"] = dict(reward_metrics)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Grouped PTS-CEM construction through the attack pipeline."
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
