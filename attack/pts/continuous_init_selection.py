from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import load_json, save_json
from attack.common.config import (
    Config,
    PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
    PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED,
    PTS_CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1,
)
from attack.common.paths import shared_root, split_key
from attack.pts.cem import PTSCEMConfig
from attack.pts.continuous_cem import (
    ContinuousCandidateSampleSpec,
    PTSContinuousBetaCEMConfig,
    build_continuous_beta_initial_sample_plan,
)
from attack.pts.continuous_executor import (
    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
    CONTINUOUS_ACTION_KEEP_FULL_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
    CONTINUOUS_ACTION_STOP,
    PTSContinuousSessionContext,
    apply_pts_continuous_beta_construction_batch,
    build_continuous_shared_session_contexts,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_ALL_PARAMETER_NAMES,
    ContinuousBetaPolicy,
)


CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE = "continuous_mlp_cem_initialization"
BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING = (
    "two_pool_behavior_curve_space_filling"
)
ACTION_COLUMNS = (
    CONTINUOUS_ACTION_KEEP_FULL_SUFFIX,
    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
    CONTINUOUS_ACTION_STOP,
)


@dataclass(frozen=True)
class ContinuousMLPInitialSelectionResult:
    cache_key: str
    cache_path: Path
    identity: dict[str, object]
    selected_sample_plan: list[ContinuousCandidateSampleSpec]
    selected_candidates: list[dict[str, object]]
    behavior_metrics: list[dict[str, object]]
    loaded_from_cache: bool = False


@dataclass(frozen=True)
class BehaviorCurveSelectionConfig:
    enabled: bool = True
    mode: str = BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING
    pool_size: int = 256
    select_size: int | None = None
    distance: str = "l1"
    min_behavior_distance: float = 1e-9
    soft_extreme_pool_size: int = 512
    moderate_pool_size: int = 512
    soft_extreme_select_size: int = 5
    moderate_select_size: int = 11
    soft_extreme_std: float = 1.25
    moderate_std: float = 0.80
    q_grid_size: int = 19
    q_grid_min: float = 0.05
    q_grid_max: float = 0.95
    q_kernel_bandwidth: float = 0.10

    def resolved_select_size(self, default_size: int) -> int:
        value = int(default_size if self.select_size is None else self.select_size)
        if value <= 0:
            raise ValueError("behavior_select_size must be positive.")
        return value

    def validate(self, *, default_select_size: int) -> None:
        mode = str(self.mode).strip().lower()
        if mode != BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING:
            raise ValueError(f"Unsupported behavior_selection_mode: {self.mode}")
        if int(self.pool_size) <= 0:
            raise ValueError("behavior_pool_size must be positive.")
        self.resolved_select_size(default_select_size)
        if str(self.distance).strip().lower() != "l1":
            raise ValueError("behavior_distance currently supports only 'l1'.")
        if int(self.soft_extreme_pool_size) <= 0:
            raise ValueError("soft_extreme_pool_size must be positive.")
        if int(self.moderate_pool_size) <= 0:
            raise ValueError("moderate_pool_size must be positive.")
        if int(self.soft_extreme_select_size) < 0:
            raise ValueError("soft_extreme_select_size must be non-negative.")
        if int(self.moderate_select_size) < 0:
            raise ValueError("moderate_select_size must be non-negative.")
        if int(self.soft_extreme_select_size) + int(self.moderate_select_size) <= 0:
            raise ValueError("two-pool select sizes must request at least one candidate.")
        if float(self.soft_extreme_std) <= 0.0:
            raise ValueError("soft_extreme_std must be positive.")
        if float(self.moderate_std) <= 0.0:
            raise ValueError("moderate_std must be positive.")
        if int(self.q_grid_size) <= 1:
            raise ValueError("q_grid_size must be greater than 1.")
        if not (0.0 <= float(self.q_grid_min) < float(self.q_grid_max) <= 1.0):
            raise ValueError("q grid bounds must satisfy 0 <= min < max <= 1.")
        if float(self.q_kernel_bandwidth) <= 0.0:
            raise ValueError("q_kernel_bandwidth must be positive.")


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
            "resolved_init_seed": int(resolve_continuous_mlp_init_seed(config)),
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
        / CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE
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
    pts = config.attack.pts_construction
    if pts is None:
        raise ValueError("pts_construction config is required.")
    init = pts.cem.init
    behavior_config = BehaviorCurveSelectionConfig(
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
        init_target_placeholder=0,
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
        init_target_placeholder=0,
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


def resolve_continuous_mlp_init_seed(config: Config) -> int:
    pts = config.attack.pts_construction
    if pts is None:
        raise ValueError("pts_construction config is required.")
    # Mirrors formal PTS-CEM's current base-seed resolver. Keep this explicit
    # here so init cache identity changes if seed_source support broadens.
    if pts.cem.seed_source == PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED:
        return int(config.seeds.position_opt_seed)
    raise ValueError(
        "continuous_mlp_cem initialization supports only "
        "cem.seed_source='position_opt_seed'."
    )


def _build_behavior_candidate_pool(
    *,
    behavior_config: BehaviorCurveSelectionConfig,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    init_target_placeholder: int = 0,
    pool_size: int | None = None,
    source_pool: str = "behavior",
    key_prefix: str = "pool_cand",
    initial_std: float | None = None,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    # Initialization selection is target-independent and never materializes
    # generated suffix item IDs. The executor still requires a target_item for
    # seed inputs and stop/generate plumbing, so use a fixed placeholder.
    effective_continuous_config = (
        continuous_config
        if initial_std is None
        else replace(continuous_config, initial_std=float(initial_std))
    )
    if (
        effective_continuous_config.initialization_mode
        == BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING
    ):
        effective_continuous_config = replace(
            effective_continuous_config,
            initialization_mode=PTS_CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1,
        )
    sample_plan = build_continuous_beta_initial_sample_plan(
        cem_config=cem_config,
        continuous_config=effective_continuous_config,
        population_size=int(behavior_config.pool_size if pool_size is None else pool_size),
    )
    pool: list[dict[str, Any]] = []
    for pool_id, sample_spec in enumerate(sample_plan):
        pool_key = f"{key_prefix}{pool_id}"
        seed = int(cem_config.base_seed) + int(seed_offset) + int(pool_id)
        policy = ContinuousBetaPolicy.from_vector(
            sample_spec.vector,
            parameter_bounds=effective_continuous_config.parameter_bounds,
            parameterization=effective_continuous_config.parameterization,
            smoothing_epsilon=float(effective_continuous_config.smoothing_epsilon),
        )
        construction_result = apply_pts_continuous_beta_construction_batch(
            session_contexts=session_contexts,
            target_item=int(init_target_placeholder),
            policy=policy,
            base_seed=int(cem_config.base_seed),
            candidate_key=pool_key,
            poison_runner=None,
            generation_topk=int(pts_config.generation.topk),
            generation_rng_base_seed=seed,
            generation_rng_tag="pts_generated_suffix",
            materialize_generated_suffix=False,
        )
        sample_metadata = dict(sample_spec.sample_metadata)
        sample_metadata["source_pool"] = str(source_pool)
        if initial_std is not None:
            sample_metadata["diagnostic_initial_std"] = float(initial_std)
        candidate_info = {
            "candidate_key": pool_key,
            "candidate_id": int(pool_id),
            "sample_origin": sample_spec.sample_origin,
            "prototype_name": str(sample_metadata.get("prototype_name", "")),
            "sample_metadata": sample_metadata,
            "policy": policy,
            "parameter_vector": policy.to_vector(),
            "parameter_names": list(policy.to_dict()["parameter_names"]),
        }
        records = [dict(record) for record in construction_result.per_session_records]
        summary = _candidate_summary_row(candidate_info, records)
        behavior_vector = build_behavior_vector(records)
        dominant_action_family, dominant_action_ratio = _dominant_action_family(summary)
        behavior_stats = behavior_statistics(summary, behavior_vector)
        pool.append(
            {
                "pool_candidate_key": pool_key,
                "pool_candidate_id": int(pool_id),
                "source_pool": str(source_pool),
                "candidate_info": candidate_info,
                "records": records,
                "summary": summary,
                "behavior_vector": behavior_vector,
                "behavior_stats": behavior_stats,
                "dominant_action_family": dominant_action_family,
                "dominant_action_ratio": float(dominant_action_ratio),
            }
        )
    return pool


def _candidate_summary_row(
    candidate_info: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row = _candidate_base_row(candidate_info)
    row.update(_parameter_row(candidate_info))
    row.update(
        {
            "num_sessions": int(len(records)),
            "residual_suffix_len_mean": _mean(_field_ints(records, "residual_suffix_length")),
            "residual_suffix_len_min": _min(_field_ints(records, "residual_suffix_length")),
            "residual_suffix_len_max": _max(_field_ints(records, "residual_suffix_length")),
            "q_suffix_mean": _mean(_field_floats(records, "suffix_length_percentile")),
            "q_suffix_min": _min(_field_floats(records, "suffix_length_percentile")),
            "q_suffix_max": _max(_field_floats(records, "suffix_length_percentile")),
            "rho_mean": _mean(_field_floats(records, "consume_ratio")),
            "rho_std": _std(_field_floats(records, "consume_ratio")),
            "rho_min": _min(_field_floats(records, "consume_ratio")),
            "rho_max": _max(_field_floats(records, "consume_ratio")),
            "consume_count_mean": _mean(_field_ints(records, "consume_count")),
            "consume_count_min": _min(_field_ints(records, "consume_count")),
            "consume_count_max": _max(_field_ints(records, "consume_count")),
        }
    )
    row.update(_behavior_ratio_fields(records))
    row.update(_action_ratio_fields(records))
    return row


def _candidate_base_row(candidate_info: Mapping[str, object]) -> dict[str, object]:
    row = {
        "candidate_key": str(candidate_info["candidate_key"]),
        "candidate_id": int(candidate_info["candidate_id"]),
        "sample_origin": str(candidate_info["sample_origin"]),
        "prototype_name": str(candidate_info.get("prototype_name", "")),
    }
    policy = candidate_info.get("policy")
    if isinstance(policy, ContinuousBetaPolicy):
        row["smoothing_epsilon"] = float(policy.smoothing_epsilon)
        row["consume_smoothing_epsilon"] = float(policy.smoothing_epsilon)
        row["source_probability_floor"] = float(policy.smoothing_epsilon)
    return row


def _parameter_row(candidate_info: Mapping[str, object]) -> dict[str, object]:
    vector = [float(value) for value in candidate_info["parameter_vector"]]
    raw_names = candidate_info.get("parameter_names")
    if raw_names is None:
        policy = candidate_info.get("policy")
        if not isinstance(policy, ContinuousBetaPolicy):
            raise KeyError("candidate_info must include parameter_names or policy.")
        raw_names = policy.to_dict()["parameter_names"]
    names = [str(name) for name in raw_names]
    row = {name: "" for name in CONTINUOUS_BETA_ALL_PARAMETER_NAMES}
    row.update({name: float(vector[index]) for index, name in enumerate(names)})
    return {str(name): value for name, value in row.items()}


def build_behavior_vector(records: Sequence[Mapping[str, object]]) -> list[float]:
    groups = [
        list(records),
        [record for record in records if int(record["residual_suffix_length"]) == 1],
        [record for record in records if int(record["residual_suffix_length"]) == 2],
        [record for record in records if int(record["residual_suffix_length"]) >= 3],
    ]
    vector: list[float] = []
    for group_records in groups:
        ratios = _action_ratio_fields(group_records)
        vector.extend(float(ratios[f"{action_name}_ratio"]) for action_name in ACTION_COLUMNS)
    return vector


def behavior_statistics(
    summary: Mapping[str, object],
    behavior_vector: Sequence[float],
) -> dict[str, float]:
    overall = [float(summary[f"{action_name}_ratio"]) for action_name in ACTION_COLUMNS]
    vector = [float(value) for value in behavior_vector]
    return {
        "max_action_ratio_overall": max(overall) if overall else 0.0,
        "entropy_overall": _normalized_entropy(overall),
        "max_action_ratio_behavior_vector": max(vector) if vector else 0.0,
        "entropy_behavior_vector": _normalized_entropy(vector),
    }


def build_behavior_curve_profile(
    records: Sequence[Mapping[str, object]],
    *,
    q_grid_size: int = 19,
    q_grid_min: float = 0.05,
    q_grid_max: float = 0.95,
    q_kernel_bandwidth: float = 0.10,
) -> dict[str, object]:
    q_grid = _q_grid(
        q_grid_size=int(q_grid_size),
        q_grid_min=float(q_grid_min),
        q_grid_max=float(q_grid_max),
    )
    distributions = [
        _weighted_action_distribution(records, q0=float(q0), bandwidth=float(q_kernel_bandwidth))
        for q0 in q_grid
    ]
    vector = [float(probability) for distribution in distributions for probability in distribution]
    entropies = [_normalized_entropy(distribution) for distribution in distributions]
    max_probs = [max(float(probability) for probability in distribution) for distribution in distributions]
    adjacent_distances = [
        _l1_distance(distributions[index - 1], distributions[index])
        for index in range(1, len(distributions))
    ]
    mean_entropy = _mean(entropies)
    mean_max_prob = _mean(max_probs)
    return {
        "q_grid": q_grid,
        "q_kernel_bandwidth": float(q_kernel_bandwidth),
        "behavior_curve_vector": vector,
        "mean_entropy_over_q": float(mean_entropy),
        "min_entropy_over_q": _min(entropies),
        "mean_max_action_prob_over_q": float(mean_max_prob),
        "max_action_prob_over_q": _max(max_probs),
        "q_variation": _mean(adjacent_distances),
        "collapse_score": float(mean_max_prob) - float(mean_entropy),
    }


def select_behavior_curve_two_pool_candidates(
    *,
    soft_extreme_pool: Sequence[Mapping[str, Any]],
    moderate_pool: Sequence[Mapping[str, Any]],
    soft_extreme_select_size: int,
    moderate_select_size: int,
    distance: str = "l1",
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    if str(distance).strip().lower() != "l1":
        raise ValueError("behavior curve distance currently supports only 'l1'.")
    fallback_records: list[dict[str, object]] = []
    soft_filtered = _filter_behavior_curve_pool(
        soft_extreme_pool,
        drop_top_collapse_fraction=0.05,
        drop_bottom_entropy_fraction=0.0,
        drop_bottom_variation_fraction=0.0,
    )
    selected_soft = _select_behavior_curve_maximin(
        soft_filtered,
        count=int(soft_extreme_select_size),
        initial_selected=[],
        selection_stage="soft_extreme",
        selection_reason="soft_extreme_curve_maximin",
    )
    if len(selected_soft) < int(soft_extreme_select_size):
        fallback_records.append(
            {
                "stage": "soft_extreme",
                "reason": "insufficient_filtered_pool",
                "requested": int(soft_extreme_select_size),
                "selected": int(len(selected_soft)),
            }
        )
        selected_soft = _fill_behavior_curve_selection(
            selected_soft,
            soft_extreme_pool,
            count=int(soft_extreme_select_size),
            initial_selected=[],
            selection_stage="soft_extreme",
            selection_reason="soft_extreme_unfiltered_fallback",
        )
    moderate_filtered = _filter_behavior_curve_pool(
        moderate_pool,
        drop_top_collapse_fraction=0.10,
        drop_bottom_entropy_fraction=0.10,
        drop_bottom_variation_fraction=0.10,
    )
    selected_moderate = _select_behavior_curve_maximin(
        moderate_filtered,
        count=int(moderate_select_size),
        initial_selected=selected_soft,
        selection_stage="moderate",
        selection_reason="moderate_curve_maximin",
    )
    if len(selected_moderate) < int(moderate_select_size):
        fallback_records.append(
            {
                "stage": "moderate",
                "reason": "insufficient_filtered_pool",
                "requested": int(moderate_select_size),
                "selected": int(len(selected_moderate)),
            }
        )
        selected_moderate = _fill_behavior_curve_selection(
            selected_moderate,
            moderate_pool,
            count=int(moderate_select_size),
            initial_selected=selected_soft,
            selection_stage="moderate",
            selection_reason="moderate_unfiltered_fallback",
        )
    return selected_soft + selected_moderate, fallback_records


def _with_behavior_curve_profile(
    candidate: Mapping[str, Any],
    *,
    q_grid_size: int,
    q_grid_min: float,
    q_grid_max: float,
    q_kernel_bandwidth: float,
) -> dict[str, Any]:
    item = dict(candidate)
    if "behavior_curve_profile" not in item:
        item["behavior_curve_profile"] = build_behavior_curve_profile(
            item["records"],
            q_grid_size=int(q_grid_size),
            q_grid_min=float(q_grid_min),
            q_grid_max=float(q_grid_max),
            q_kernel_bandwidth=float(q_kernel_bandwidth),
        )
    return item


def _behavior_curve_metrics_row(
    candidate: Mapping[str, Any],
    *,
    selected_entry: tuple[int, Mapping[str, Any]] | None,
    force_source_pool: str | None = None,
) -> dict[str, object]:
    info = candidate["candidate_info"]
    profile = candidate["behavior_curve_profile"]
    policy = info["policy"]
    if not isinstance(policy, ContinuousBetaPolicy):
        raise TypeError("candidate_info['policy'] must be a ContinuousBetaPolicy.")
    selected_rank = "" if selected_entry is None else int(selected_entry[0])
    parameter_vector = [float(value) for value in info["parameter_vector"]]
    return {
        "candidate_key": str(candidate["pool_candidate_key"]),
        "source_pool": str(force_source_pool or candidate.get("source_pool", "")),
        "selected": bool(selected_entry is not None),
        "selected_rank": selected_rank,
        "selection_stage": str(candidate.get("selection_stage", "")),
        "sample_origin": str(info["sample_origin"]),
        "prototype_name": str(info.get("prototype_name", "")),
        "parameterization": str(policy.parameterization),
        "parameter_count": int(len(parameter_vector)),
        "mean_entropy_over_q": float(profile["mean_entropy_over_q"]),
        "min_entropy_over_q": float(profile["min_entropy_over_q"]),
        "mean_max_action_prob_over_q": float(profile["mean_max_action_prob_over_q"]),
        "max_action_prob_over_q": float(profile["max_action_prob_over_q"]),
        "q_variation": float(profile["q_variation"]),
        "collapse_score": float(profile["collapse_score"]),
        "behavior_curve_vector_json": json.dumps(
            [float(value) for value in profile["behavior_curve_vector"]]
        ),
        "parameter_vector_json": json.dumps(parameter_vector),
    }


def _filter_behavior_curve_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    drop_top_collapse_fraction: float,
    drop_bottom_entropy_fraction: float,
    drop_bottom_variation_fraction: float,
) -> list[dict[str, Any]]:
    candidates = [dict(candidate) for candidate in pool]
    if len(candidates) <= 1:
        return candidates
    collapse_cutoff = _quantile(
        [float(candidate["behavior_curve_profile"]["collapse_score"]) for candidate in candidates],
        1.0 - float(drop_top_collapse_fraction),
    )
    entropy_cutoff = _quantile(
        [float(candidate["behavior_curve_profile"]["mean_entropy_over_q"]) for candidate in candidates],
        float(drop_bottom_entropy_fraction),
    )
    variation_cutoff = _quantile(
        [float(candidate["behavior_curve_profile"]["q_variation"]) for candidate in candidates],
        float(drop_bottom_variation_fraction),
    )
    filtered = [
        candidate
        for candidate in candidates
        if float(candidate["behavior_curve_profile"]["collapse_score"]) <= float(collapse_cutoff)
        and float(candidate["behavior_curve_profile"]["mean_entropy_over_q"]) >= float(entropy_cutoff)
        and float(candidate["behavior_curve_profile"]["q_variation"]) >= float(variation_cutoff)
    ]
    return filtered if filtered else candidates


def _select_behavior_curve_maximin(
    pool: Sequence[Mapping[str, Any]],
    *,
    count: int,
    initial_selected: Sequence[Mapping[str, Any]],
    selection_stage: str,
    selection_reason: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_keys = {_candidate_key(candidate) for candidate in initial_selected}
    local_keys: set[str] = set()
    while len(selected) < int(count):
        remaining = [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
            and _candidate_key(candidate) not in local_keys
        ]
        if not remaining:
            break
        if initial_selected or selected:
            reference = list(initial_selected) + selected
            best = max(
                remaining,
                key=lambda candidate: (
                    _min_behavior_curve_distance(candidate, reference),
                    _curve_profile_metric(candidate, "q_variation"),
                    _curve_profile_metric(candidate, "mean_entropy_over_q"),
                    -_candidate_pool_id(candidate),
                ),
            )
        else:
            centroid = _behavior_curve_centroid(remaining)
            best = max(
                remaining,
                key=lambda candidate: (
                    _l1_distance(_behavior_curve_vector(candidate), centroid),
                    _curve_profile_metric(candidate, "q_variation"),
                    -_candidate_pool_id(candidate),
                ),
            )
        item = dict(best)
        item["selection_stage"] = str(selection_stage)
        item["selection_stratum"] = str(selection_stage)
        item["selection_reason"] = str(selection_reason)
        selected.append(item)
        local_keys.add(_candidate_key(best))
    return selected


def _fill_behavior_curve_selection(
    selected: Sequence[Mapping[str, Any]],
    pool: Sequence[Mapping[str, Any]],
    *,
    count: int,
    initial_selected: Sequence[Mapping[str, Any]],
    selection_stage: str,
    selection_reason: str,
) -> list[dict[str, Any]]:
    filled = [dict(candidate) for candidate in selected]
    already = list(initial_selected) + filled
    selected_keys = {_candidate_key(candidate) for candidate in already}
    while len(filled) < int(count):
        remaining = [candidate for candidate in pool if _candidate_key(candidate) not in selected_keys]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda candidate: (
                _min_behavior_curve_distance(candidate, already),
                _curve_profile_metric(candidate, "q_variation"),
                _curve_profile_metric(candidate, "mean_entropy_over_q"),
                -_candidate_pool_id(candidate),
            ),
        )
        item = dict(best)
        item["selection_stage"] = str(selection_stage)
        item["selection_stratum"] = str(selection_stage)
        item["selection_reason"] = str(selection_reason)
        filled.append(item)
        already.append(item)
        selected_keys.add(_candidate_key(item))
    return filled


def _behavior_curve_vector(candidate: Mapping[str, Any]) -> list[float]:
    return [float(value) for value in candidate["behavior_curve_profile"]["behavior_curve_vector"]]


def _behavior_curve_centroid(candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    vectors = [_behavior_curve_vector(candidate) for candidate in candidates]
    if not vectors:
        return []
    width = len(vectors[0])
    return [float(sum(vector[index] for vector in vectors)) / float(len(vectors)) for index in range(width)]


def _min_behavior_curve_distance(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> float:
    if not selected:
        return float("inf")
    vector = _behavior_curve_vector(candidate)
    return min(_l1_distance(vector, _behavior_curve_vector(item)) for item in selected)


def _curve_profile_metric(candidate: Mapping[str, Any], field_name: str) -> float:
    return float(candidate["behavior_curve_profile"][field_name])


def _q_grid(*, q_grid_size: int, q_grid_min: float, q_grid_max: float) -> list[float]:
    if int(q_grid_size) <= 1:
        raise ValueError("q_grid_size must be greater than 1.")
    step = (float(q_grid_max) - float(q_grid_min)) / float(int(q_grid_size) - 1)
    return [float(q_grid_min) + step * index for index in range(int(q_grid_size))]


def _weighted_action_distribution(
    records: Sequence[Mapping[str, object]],
    *,
    q0: float,
    bandwidth: float,
) -> list[float]:
    weights = {action_name: 0.0 for action_name in ACTION_COLUMNS}
    total = 0.0
    for record in records:
        q_value = float(record["suffix_length_percentile"])
        weight = math.exp(-0.5 * ((q_value - float(q0)) / float(bandwidth)) ** 2)
        weights[str(record["action"])] += float(weight)
        total += float(weight)
    if total <= 1e-12:
        nearest = min(
            records,
            key=lambda record: abs(float(record["suffix_length_percentile"]) - float(q0)),
        )
        weights = {action_name: 0.0 for action_name in ACTION_COLUMNS}
        weights[str(nearest["action"])] = 1.0
        total = 1.0
    return [float(weights[action_name]) / float(total) for action_name in ACTION_COLUMNS]


def _action_ratio_fields(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    total = int(len(records))
    counts = Counter(str(record["action"]) for record in records)
    return {f"{action_name}_ratio": _ratio(int(counts[action_name]), total) for action_name in ACTION_COLUMNS}


def _behavior_ratio_fields(records: Sequence[Mapping[str, object]]) -> dict[str, float]:
    total = int(len(records))
    non_stop_records = [record for record in records if str(record["continuation_source"]) != "stop"]
    generated_non_stop = [record for record in non_stop_records if str(record["continuation_source"]) == "generate"]
    return {
        "consume_0_ratio": _ratio(sum(1 for record in records if int(record["consume_count"]) == 0), total),
        "consume_partial_ratio": _ratio(
            sum(
                1
                for record in records
                if 0 < int(record["consume_count"]) < int(record["residual_suffix_length"])
            ),
            total,
        ),
        "consume_all_ratio": _ratio(
            sum(
                1
                for record in records
                if int(record["consume_count"]) == int(record["residual_suffix_length"])
            ),
            total,
        ),
        "non_stop_ratio": _ratio(len(non_stop_records), total),
        "stop_ratio": _ratio(total - len(non_stop_records), total),
        "generate_ratio_non_stop": _ratio(len(generated_non_stop), len(non_stop_records)),
    }


def _field_floats(records: Sequence[Mapping[str, object]], field_name: str) -> list[float]:
    return [float(record[field_name]) for record in records]


def _field_ints(records: Sequence[Mapping[str, object]], field_name: str) -> list[int]:
    return [int(record[field_name]) for record in records]


def _dominant_action_family(summary: Mapping[str, object]) -> tuple[str, float]:
    action_ratios = [(action_name, float(summary[f"{action_name}_ratio"])) for action_name in ACTION_COLUMNS]
    return max(action_ratios, key=lambda item: (item[1], item[0]))


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate["pool_candidate_key"])


def _candidate_pool_id(candidate: Mapping[str, Any]) -> int:
    return int(candidate.get("pool_candidate_id", 0))


def _quantile(values: Sequence[float | int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    fraction = min(1.0, max(0.0, float(fraction)))
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)


def _normalized_entropy(values: Sequence[float | int]) -> float:
    positives = [float(value) for value in values if float(value) > 0.0]
    total = float(sum(positives))
    if total <= 0.0 or len(values) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in positives)
    return float(entropy / math.log(float(len(values))))


def _mean(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(sum(float(value) for value in values)) / float(len(values))


def _std(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    center = _mean(values)
    return float((sum((float(value) - center) ** 2.0 for value in values) / float(len(values))) ** 0.5)


def _min(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(min(float(value) for value in values))


def _max(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(max(float(value) for value in values))


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
    "BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING",
    "BehaviorCurveSelectionConfig",
    "CONTINUOUS_MLP_INITIALIZATION_RUN_TYPE",
    "ContinuousMLPInitialSelectionResult",
    "build_behavior_curve_profile",
    "build_continuous_mlp_initial_sample_plan",
    "continuous_mlp_init_cache_key",
    "continuous_mlp_init_cache_path",
    "continuous_mlp_init_identity_payload",
    "resolve_continuous_mlp_init_seed",
    "select_behavior_curve_two_pool_candidates",
]
