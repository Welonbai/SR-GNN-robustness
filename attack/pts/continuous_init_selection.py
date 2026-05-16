from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import load_json, save_json
from attack.common.config import Config, PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING
from attack.common.paths import PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE, shared_root, split_key
from attack.pts.cem import PTSCEMConfig
from attack.pts.continuous_cem import (
    ContinuousCandidateSampleSpec,
    PTSContinuousBetaCEMConfig,
)
from attack.pts.continuous_executor import build_continuous_shared_session_contexts


@dataclass(frozen=True)
class ContinuousMLPInitialSelectionResult:
    cache_key: str
    cache_path: Path
    identity: dict[str, object]
    selected_sample_plan: list[ContinuousCandidateSampleSpec]
    selected_candidates: list[dict[str, object]]
    behavior_metrics: list[dict[str, object]]
    loaded_from_cache: bool = False


def build_continuous_mlp_initial_sample_plan(
    *,
    config: Config,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    template_sessions: Sequence[Sequence[int]],
    generation_topk: int,
    force_rebuild: bool = False,
) -> ContinuousMLPInitialSelectionResult:
    pts_config = config.attack.pts_construction
    if pts_config is None:
        raise ValueError("continuous MLP initialization requires pts_construction config.")
    if pts_config.cem.init.mode != PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING:
        raise ValueError(
            "continuous MLP initialization requires cem.init.mode="
            "'two_pool_behavior_curve_space_filling'."
        )
    identity = continuous_mlp_init_identity_payload(
        config=config,
        template_sessions=template_sessions,
    )
    cache_key = continuous_mlp_init_cache_key(identity)
    cache_path = continuous_mlp_init_cache_path(config, cache_key=cache_key)
    if not force_rebuild:
        cached = load_json(cache_path)
        if isinstance(cached, Mapping):
            return _selection_result_from_cache(cache_path, dict(cached), loaded=True)

    result = _build_uncached_selection(
        config=config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        template_sessions=template_sessions,
        generation_topk=int(generation_topk),
        cache_key=cache_key,
        cache_path=cache_path,
        identity=identity,
    )
    save_json(_cache_payload(result), cache_path)
    return result


def continuous_mlp_init_identity_payload(
    *,
    config: Config,
    template_sessions: Sequence[Sequence[int]],
) -> dict[str, object]:
    pts = config.attack.pts_construction
    if pts is None:
        raise ValueError("pts_construction config is required.")
    policy = pts.continuous_policy
    init = pts.cem.init
    return {
        "identity_version": "continuous_mlp_init_v1",
        "split_key": split_key(config),
        "dataset": config.data.dataset_name,
        "fake_sessions_hash": _hash_json(
            [[int(item) for item in session] for session in template_sessions]
        ),
        "prefix_assignment": {
            "mode": "internal_uniform_target_independent_v1",
            "seed_scope": "target_independent",
            "seed_source": pts.cem.seed_source,
            "resolved_seed": int(config.seeds.position_opt_seed),
        },
        "method": "continuous_mlp_cem",
        "continuous_policy": {
            "parameterization": policy.parameterization,
            "hidden_size": int(policy.hidden_size),
            "consume_distribution": policy.consume_distribution,
            "smoothing_epsilon": float(policy.smoothing_epsilon),
            "source_policy": policy.source_policy,
            "parameter_bounds": {
                "min": float(policy.parameter_bounds.min),
                "max": float(policy.parameter_bounds.max),
            },
            "deterministic_sampling": bool(policy.deterministic_sampling),
        },
        "init": {
            "mode": init.mode,
            "soft_extreme_pool_size": int(init.soft_extreme_pool_size),
            "moderate_pool_size": int(init.moderate_pool_size),
            "soft_extreme_select_size": int(init.soft_extreme_select_size),
            "moderate_select_size": int(init.moderate_select_size),
            "soft_extreme_initial_std": float(init.soft_extreme_initial_std),
            "moderate_initial_std": float(init.moderate_initial_std),
            "q_grid_size": int(init.q_grid_size),
            "behavior_distance": init.behavior_distance,
            "rounding_mode": "half_up",
            "candidate_seed_stride": int(pts.cem.candidate_seed_stride),
            "init_materialize_generated_suffix": False,
        },
    }


def continuous_mlp_init_cache_key(identity: Mapping[str, object]) -> str:
    return f"continuous_mlp_init_{_hash_json(identity)}"


def continuous_mlp_init_cache_path(config: Config, *, cache_key: str) -> Path:
    return (
        shared_root(config)
        / PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
        / "continuous_mlp_initialization"
        / str(cache_key)
        / "init_cache.json"
    )


def _build_uncached_selection(
    *,
    config: Config,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    template_sessions: Sequence[Sequence[int]],
    generation_topk: int,
    cache_key: str,
    cache_path: Path,
    identity: dict[str, object],
) -> ContinuousMLPInitialSelectionResult:
    # Import the current diagnostic helpers lazily to avoid a pipeline-level
    # import cycle.  They are the single implementation used by both diagnostic
    # reports and formal CEM initialization until this module fully owns them.
    from attack.pipeline.runs.run_pts_continuous_init_diagnostic import (
        BehaviorAwareSelectionConfig,
        _build_behavior_candidate_pool,
        _behavior_curve_metrics_row,
        _with_behavior_curve_profile,
        select_behavior_curve_two_pool_candidates,
    )

    pts = config.attack.pts_construction
    if pts is None:
        raise ValueError("pts_construction config is required.")
    init = pts.cem.init
    behavior_config = BehaviorAwareSelectionConfig(
        enabled=True,
        mode="two_pool_behavior_curve_space_filling",
        soft_extreme_pool_size=int(init.soft_extreme_pool_size),
        moderate_pool_size=int(init.moderate_pool_size),
        soft_extreme_select_size=int(init.soft_extreme_select_size),
        moderate_select_size=int(init.moderate_select_size),
        soft_extreme_std=float(init.soft_extreme_initial_std),
        moderate_std=float(init.moderate_initial_std),
        q_grid_size=int(init.q_grid_size),
        q_grid_min=0.05,
        q_grid_max=0.95,
        q_kernel_bandwidth=0.10,
        distance=str(init.behavior_distance),
    )
    session_contexts = build_continuous_shared_session_contexts(
        template_sessions=template_sessions,
        target_item=0,
        base_seed=int(cem_config.base_seed),
        seed_scope="target_independent",
    )
    soft_pool = _build_behavior_candidate_pool(
        behavior_config=behavior_config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=_PTSGenerationOnly(generation_topk),
        session_contexts=session_contexts,
        target_item=0,
        pool_size=int(init.soft_extreme_pool_size),
        source_pool="soft_extreme",
        key_prefix="soft_extreme_pool_cand",
        initial_std=float(init.soft_extreme_initial_std),
        seed_offset=0,
    )
    moderate_pool = _build_behavior_candidate_pool(
        behavior_config=behavior_config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=_PTSGenerationOnly(generation_topk),
        session_contexts=session_contexts,
        target_item=0,
        pool_size=int(init.moderate_pool_size),
        source_pool="moderate",
        key_prefix="moderate_pool_cand",
        initial_std=float(init.moderate_initial_std),
        seed_offset=100000,
    )
    all_pool = [
        _with_behavior_curve_profile(
            candidate,
            q_grid_size=int(init.q_grid_size),
            q_grid_min=0.05,
            q_grid_max=0.95,
            q_kernel_bandwidth=0.10,
        )
        for candidate in [*soft_pool, *moderate_pool]
    ]
    selected_pool, _fallbacks = select_behavior_curve_two_pool_candidates(
        soft_extreme_pool=[
            candidate for candidate in all_pool if candidate["source_pool"] == "soft_extreme"
        ],
        moderate_pool=[
            candidate for candidate in all_pool if candidate["source_pool"] == "moderate"
        ],
        soft_extreme_select_size=int(init.soft_extreme_select_size),
        moderate_select_size=int(init.moderate_select_size),
        distance=str(init.behavior_distance),
    )
    selected_candidates: list[dict[str, object]] = []
    sample_plan: list[ContinuousCandidateSampleSpec] = []
    for rank, candidate in enumerate(selected_pool):
        info = candidate["candidate_info"]
        candidate_key = f"iter0_cand{rank}"
        vector = [float(value) for value in info["parameter_vector"]]
        selected_candidates.append(
            {
                "candidate_key": candidate_key,
                "pool_origin": str(candidate.get("source_pool", "")),
                "pool_candidate_key": str(candidate["pool_candidate_key"]),
                "selection_stage": str(candidate.get("selection_stage", "")),
                "selection_reason": str(candidate.get("selection_reason", "")),
                "parameter_vector": vector,
                "sample_origin": "continuous_mlp_two_pool_behavior_curve",
            }
        )
        sample_plan.append(
            ContinuousCandidateSampleSpec(
                vector=vector,
                sample_origin="continuous_mlp_two_pool_behavior_curve",
                sample_metadata={
                    "candidate_key": candidate_key,
                    "pool_origin": str(candidate.get("source_pool", "")),
                    "pool_candidate_key": str(candidate["pool_candidate_key"]),
                    "selection_stage": str(candidate.get("selection_stage", "")),
                    "selection_reason": str(candidate.get("selection_reason", "")),
                    "init_materialize_generated_suffix": False,
                },
            )
        )
    selected_source = {
        str(item["pool_candidate_key"]): int(index)
        for index, item in enumerate(selected_candidates)
    }
    behavior_metrics = [
        _behavior_curve_metrics_row(
            candidate,
            selected_entry=(
                selected_source[str(candidate["pool_candidate_key"])],
                candidate,
            )
            if str(candidate["pool_candidate_key"]) in selected_source
            else None,
        )
        for candidate in all_pool
    ]
    return ContinuousMLPInitialSelectionResult(
        cache_key=cache_key,
        cache_path=cache_path,
        identity=identity,
        selected_sample_plan=sample_plan,
        selected_candidates=selected_candidates,
        behavior_metrics=behavior_metrics,
        loaded_from_cache=False,
    )


@dataclass(frozen=True)
class _PTSGenerationOnly:
    topk: int

    @property
    def generation(self) -> "_PTSGenerationOnly":
        return self


def _selection_result_from_cache(
    cache_path: Path,
    payload: dict[str, Any],
    *,
    loaded: bool,
) -> ContinuousMLPInitialSelectionResult:
    selected = [dict(item) for item in payload["selected_candidates"]]
    sample_plan = [
        ContinuousCandidateSampleSpec(
            vector=[float(value) for value in item["parameter_vector"]],
            sample_origin=str(item.get("sample_origin", "continuous_mlp_init_cache")),
            sample_metadata={
                "candidate_key": str(item["candidate_key"]),
                "pool_origin": str(item.get("pool_origin", "")),
                "pool_candidate_key": str(item.get("pool_candidate_key", "")),
                "init_materialize_generated_suffix": False,
            },
        )
        for item in selected
    ]
    return ContinuousMLPInitialSelectionResult(
        cache_key=str(payload["cache_key"]),
        cache_path=cache_path,
        identity=dict(payload["identity"]),
        selected_sample_plan=sample_plan,
        selected_candidates=selected,
        behavior_metrics=[dict(item) for item in payload.get("behavior_metrics", [])],
        loaded_from_cache=loaded,
    )


def _cache_payload(result: ContinuousMLPInitialSelectionResult) -> dict[str, object]:
    return {
        "cache_key": result.cache_key,
        "identity": result.identity,
        "init_materialize_generated_suffix": False,
        "selected_candidates": result.selected_candidates,
        "behavior_metrics": result.behavior_metrics,
    }


def _hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ContinuousMLPInitialSelectionResult",
    "build_continuous_mlp_initial_sample_plan",
    "continuous_mlp_init_cache_key",
    "continuous_mlp_init_cache_path",
    "continuous_mlp_init_identity_payload",
]
