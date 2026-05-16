from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import load_fake_sessions, save_json
from attack.common.config import (
    Config,
    PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
    PTS_CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1,
    load_config,
)
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.pipeline.core.pipeline_utils import (
    ensure_target_registry_prefix,
    requested_target_prefix,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _build_continuous_beta_cem_config,
    _build_pts_cem_config_from_config,
    _pts_construction_artifact_dir,
    _require_pts_config,
    _validate_pts_construction_run_config,
    build_pts_construction_attack_identity_context,
)
from attack.pts.cem import PTSCEMConfig, candidate_key
from attack.pts.continuous_cem import (
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
    compute_half_up_consume_count,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_ALL_PARAMETER_NAMES,
    CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    ContinuousBetaPolicy,
)


ACTION_COLUMNS = (
    CONTINUOUS_ACTION_KEEP_FULL_SUFFIX,
    CONTINUOUS_ACTION_GENERATE_FULL_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
    CONTINUOUS_ACTION_STOP,
)

BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN = "behavior_space_greedy_maximin"
BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1 = (
    "behavior_stratified_space_filling_v1"
)
BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1 = (
    "behavior_curve_two_pool_space_filling_v1"
)
BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING = (
    "two_pool_behavior_curve_space_filling"
)

CANDIDATE_DISTRIBUTION_COLUMNS = (
    "candidate_key",
    "candidate_id",
    "sample_origin",
    "prototype_name",
    "smoothing_epsilon",
    "consume_smoothing_epsilon",
    "source_probability_floor",
    *CONTINUOUS_BETA_ALL_PARAMETER_NAMES,
    "num_sessions",
    "residual_suffix_len_mean",
    "residual_suffix_len_min",
    "residual_suffix_len_max",
    "q_suffix_mean",
    "q_suffix_min",
    "q_suffix_max",
    "rho_mean",
    "rho_std",
    "rho_min",
    "rho_max",
    "consume_count_mean",
    "consume_count_min",
    "consume_count_max",
    "consume_0_ratio",
    "consume_partial_ratio",
    "consume_all_ratio",
    "non_stop_ratio",
    "stop_ratio",
    "generate_ratio_non_stop",
    "continuous_keep_full_suffix_ratio",
    "continuous_generate_full_suffix_ratio",
    "continuous_partial_keep_suffix_ratio",
    "continuous_partial_generate_suffix_ratio",
    "continuous_stop_ratio",
)

BY_SUFFIX_COLUMNS = (
    "candidate_key",
    "candidate_id",
    "sample_origin",
    "prototype_name",
    "smoothing_epsilon",
    "consume_smoothing_epsilon",
    "source_probability_floor",
    "residual_suffix_len",
    "num_sessions",
    "q_suffix",
    "rho_mean",
    "rho_std",
    "rho_min",
    "rho_max",
    "consume_count_mean",
    "consume_0_ratio",
    "consume_partial_ratio",
    "consume_all_ratio",
    "generate_ratio_non_stop",
    "continuous_keep_full_suffix_ratio",
    "continuous_generate_full_suffix_ratio",
    "continuous_partial_keep_suffix_ratio",
    "continuous_partial_generate_suffix_ratio",
    "continuous_stop_ratio",
)

BEHAVIOR_SELECTED_METADATA_COLUMNS = (
    "selected_rank",
    "source_pool_candidate_key",
    "selection_stratum",
    "selection_reason",
)

BEHAVIOR_SELECTED_DISTRIBUTION_COLUMNS = (
    *CANDIDATE_DISTRIBUTION_COLUMNS[:2],
    *BEHAVIOR_SELECTED_METADATA_COLUMNS,
    "max_action_ratio_overall",
    "entropy_overall",
    *CANDIDATE_DISTRIBUTION_COLUMNS[2:],
)

BEHAVIOR_SELECTED_BY_SUFFIX_COLUMNS = (
    *BY_SUFFIX_COLUMNS[:2],
    *BEHAVIOR_SELECTED_METADATA_COLUMNS,
    *BY_SUFFIX_COLUMNS[2:],
)

OVERALL_SUFFIX_COLUMNS = (
    "residual_suffix_len",
    "num_sessions",
    "ratio",
    "q_suffix",
)

ROUNDING_VARIANT_COLUMNS = (
    "candidate_key",
    "rounding_mode",
    "residual_suffix_len",
    "num_sessions",
    "consume_0_ratio",
    "consume_partial_ratio",
    "consume_all_ratio",
    "consume_count_mean",
)

BEHAVIOR_POOL_SUMMARY_COLUMNS = (
    "pool_candidate_key",
    "selected",
    "selected_rank",
    "selected_candidate_key",
    "sample_origin",
    "prototype_name",
    "parameterization",
    "parameter_count",
    "parameter_vector_json",
    "dominant_action_family",
    "dominant_action_ratio",
    "max_action_ratio_overall",
    "entropy_overall",
    "max_action_ratio_behavior_vector",
    "entropy_behavior_vector",
    "behavior_selection_pool",
    "behavior_vector_json",
    "stop_ratio",
    "full_suffix_ratio",
    "partial_ratio",
    "generate_ratio_non_stop",
    "continuous_keep_full_suffix_ratio",
    "continuous_generate_full_suffix_ratio",
    "continuous_partial_keep_suffix_ratio",
    "continuous_partial_generate_suffix_ratio",
    "continuous_stop_ratio",
)

BEHAVIOR_CURVE_METRICS_COLUMNS = (
    "candidate_key",
    "source_pool",
    "selected",
    "selected_rank",
    "selection_stage",
    "sample_origin",
    "prototype_name",
    "parameterization",
    "parameter_count",
    "mean_entropy_over_q",
    "min_entropy_over_q",
    "mean_max_action_prob_over_q",
    "max_action_prob_over_q",
    "q_variation",
    "collapse_score",
    "behavior_curve_vector_json",
    "parameter_vector_json",
)


@dataclass(frozen=True)
class ContinuousInitDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]


@dataclass(frozen=True)
class BehaviorAwareSelectionConfig:
    enabled: bool = False
    mode: str = BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN
    pool_size: int = 256
    select_size: int | None = None
    distance: str = "l1"
    min_behavior_distance: float = 1e-9
    extreme_count: int = 6
    moderate_count: int = 9
    balanced_count: int = 1
    extreme_max_action_ratio_min: float = 0.70
    extreme_max_action_ratio_max: float = 0.90
    moderate_max_action_ratio_min: float = 0.35
    moderate_max_action_ratio_max: float = 0.70
    reject_max_action_ratio_above: float = 0.95
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
        if mode not in {
            BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN,
            BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1,
            BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
            BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
        }:
            raise ValueError(f"Unsupported behavior_selection_mode: {self.mode}")
        if int(self.pool_size) <= 0:
            raise ValueError("behavior_pool_size must be positive.")
        select_size = self.resolved_select_size(default_select_size)
        if str(self.distance).strip().lower() != "l1":
            raise ValueError("behavior_distance currently supports only 'l1'.")
        if float(self.min_behavior_distance) < 0.0:
            raise ValueError("behavior_min_distance must be non-negative.")
        if mode in {
            BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
            BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
        }:
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
            return
        if mode != BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1:
            return
        for field_name in ("extreme_count", "moderate_count", "balanced_count"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        requested = (
            int(self.extreme_count)
            + int(self.moderate_count)
            + int(self.balanced_count)
        )
        if requested > select_size:
            raise ValueError(
                "behavior_extreme_count + behavior_moderate_count + "
                "behavior_balanced_count must not exceed behavior_select_size."
            )
        if not (
            0.0
            <= float(self.moderate_max_action_ratio_min)
            <= float(self.moderate_max_action_ratio_max)
            <= float(self.extreme_max_action_ratio_max)
            <= 1.0
        ):
            raise ValueError("behavior max-ratio thresholds must be ordered in [0, 1].")
        if not (
            0.0
            <= float(self.extreme_max_action_ratio_min)
            <= float(self.extreme_max_action_ratio_max)
            <= 1.0
        ):
            raise ValueError("behavior extreme max-ratio thresholds must be in [0, 1].")
        if float(self.reject_max_action_ratio_above) < 0.0:
            raise ValueError("behavior_reject_max_ratio_above must be non-negative.")


def run_continuous_beta_init_diagnostic(
    *,
    config: Config,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    max_candidates: int | None = None,
    sample_sessions: int = 200,
    include_rounding_variants: bool = False,
    template_sessions: Sequence[Sequence[int]] | None = None,
    target_item: int | None = None,
    behavior_aware_select: bool = False,
    behavior_selection_mode: str = BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN,
    behavior_pool_size: int = 256,
    behavior_select_size: int | None = None,
    behavior_distance: str = "l1",
    behavior_min_distance: float = 1e-9,
    behavior_max_stop_ratio: float | None = None,
    behavior_max_per_dominant_family: int | None = None,
    behavior_min_partial_candidates: int | None = None,
    behavior_min_generate_candidates: int | None = None,
    behavior_extreme_count: int = 6,
    behavior_moderate_count: int = 9,
    behavior_balanced_count: int = 1,
    behavior_extreme_max_ratio_min: float = 0.70,
    behavior_extreme_max_ratio_max: float = 0.90,
    behavior_moderate_max_ratio_min: float = 0.35,
    behavior_moderate_max_ratio_max: float = 0.70,
    behavior_reject_max_ratio_above: float = 0.95,
    soft_extreme_pool_size: int = 512,
    moderate_pool_size: int = 512,
    soft_extreme_select_size: int = 5,
    moderate_select_size: int = 11,
    soft_extreme_std: float = 1.25,
    moderate_std: float = 0.80,
    q_grid_size: int = 19,
    q_grid_min: float = 0.05,
    q_grid_max: float = 0.95,
    q_kernel_bandwidth: float = 0.10,
) -> ContinuousInitDiagnosticResult:
    _validate_continuous_config(config)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    if template_sessions is None:
        loaded = load_fake_sessions(shared_paths["fake_sessions"])
        if loaded is None:
            raise FileNotFoundError(
                "Continuous init diagnostic requires existing shared "
                f"fake sessions and will not generate them: {shared_paths['fake_sessions']}"
            )
        template_sessions = loaded
    templates = [[int(item) for item in session] for session in template_sessions]
    if not templates:
        raise ValueError("Diagnostic requires at least one template fake session.")

    resolved_target_item = (
        int(target_item)
        if target_item is not None
        else _resolve_first_target_item(config, shared_paths=shared_paths)
    )
    pts_config = _require_pts_config(config)
    cem_config = _build_pts_cem_config_from_config(config)
    continuous_config = _build_continuous_beta_cem_config(pts_config)
    first_population_size = _first_population_size(cem_config)
    candidate_limit = (
        first_population_size
        if max_candidates is None
        else min(int(max_candidates), first_population_size)
    )
    if candidate_limit <= 0:
        raise ValueError("max_candidates must be positive.")
    behavior_config = BehaviorAwareSelectionConfig(
        enabled=bool(behavior_aware_select),
        mode=str(behavior_selection_mode),
        pool_size=int(behavior_pool_size),
        select_size=behavior_select_size,
        distance=str(behavior_distance),
        min_behavior_distance=float(behavior_min_distance),
        extreme_count=int(behavior_extreme_count),
        moderate_count=int(behavior_moderate_count),
        balanced_count=int(behavior_balanced_count),
        extreme_max_action_ratio_min=float(behavior_extreme_max_ratio_min),
        extreme_max_action_ratio_max=float(behavior_extreme_max_ratio_max),
        moderate_max_action_ratio_min=float(behavior_moderate_max_ratio_min),
        moderate_max_action_ratio_max=float(behavior_moderate_max_ratio_max),
        reject_max_action_ratio_above=float(behavior_reject_max_ratio_above),
        soft_extreme_pool_size=int(soft_extreme_pool_size),
        moderate_pool_size=int(moderate_pool_size),
        soft_extreme_select_size=int(soft_extreme_select_size),
        moderate_select_size=int(moderate_select_size),
        soft_extreme_std=float(soft_extreme_std),
        moderate_std=float(moderate_std),
        q_grid_size=int(q_grid_size),
        q_grid_min=float(q_grid_min),
        q_grid_max=float(q_grid_max),
        q_kernel_bandwidth=float(q_kernel_bandwidth),
    )
    if behavior_config.enabled:
        behavior_config.validate(default_select_size=first_population_size)

    session_contexts = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=resolved_target_item,
        base_seed=int(cem_config.base_seed),
        prefix_rng_tag=CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    )
    if pts_config.cem.init.mode == BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING:
        from attack.pts.continuous_init_selection import (
            build_continuous_mlp_initial_sample_plan,
        )

        init_selection = build_continuous_mlp_initial_sample_plan(
            config=config,
            cem_config=cem_config,
            continuous_config=continuous_config,
            template_sessions=templates,
            generation_topk=int(pts_config.generation.topk),
        )
        sample_plan = init_selection.selected_sample_plan[:candidate_limit]
    else:
        sample_plan = build_continuous_beta_initial_sample_plan(
            cem_config=cem_config,
            continuous_config=continuous_config,
            population_size=first_population_size,
        )[:candidate_limit]

    candidate_rows: list[dict[str, object]] = []
    by_suffix_rows: list[dict[str, object]] = []
    rounding_rows: list[dict[str, object]] = []
    session_sample_rows: list[dict[str, object]] = []
    initial_candidates: list[dict[str, object]] = []

    for candidate_id, sample_spec in enumerate(sample_plan):
        key = candidate_key(0, candidate_id)
        seed = int(cem_config.base_seed) + int(candidate_id)
        policy = ContinuousBetaPolicy.from_vector(
            sample_spec.vector,
            parameter_bounds=continuous_config.parameter_bounds,
            parameterization=continuous_config.parameterization,
            smoothing_epsilon=float(continuous_config.smoothing_epsilon),
        )
        construction_result = apply_pts_continuous_beta_construction_batch(
            session_contexts=session_contexts,
            target_item=resolved_target_item,
            policy=policy,
            base_seed=int(cem_config.base_seed),
            candidate_key=key,
            poison_runner=None,
            generation_topk=int(pts_config.generation.topk),
            generation_rng_base_seed=seed,
            generation_rng_tag="pts_generated_suffix",
            materialize_generated_suffix=False,
        )
        sample_metadata = dict(sample_spec.sample_metadata)
        candidate_info = {
            "candidate_key": key,
            "candidate_id": int(candidate_id),
            "sample_origin": sample_spec.sample_origin,
            "prototype_name": str(sample_metadata.get("prototype_name", "")),
            "sample_metadata": sample_metadata,
            "policy": policy,
            "parameter_vector": policy.to_vector(),
            "parameter_names": list(policy.to_dict()["parameter_names"]),
        }
        initial_candidates.append(_initial_candidate_payload(candidate_info))
        records = [dict(record) for record in construction_result.per_session_records]
        candidate_rows.append(_candidate_summary_row(candidate_info, records))
        by_suffix_rows.extend(_by_suffix_summary_rows(candidate_info, records))
        if include_rounding_variants:
            rounding_rows.extend(_rounding_variant_rows(candidate_info, records))
        for record in records:
            if len(session_sample_rows) >= int(sample_sessions):
                break
            session_sample_rows.append(_session_sample_row(key, record))

    if output_dir is None:
        attack_identity_context = build_pts_construction_attack_identity_context(config)
        output_path = (
            _pts_construction_artifact_dir(
                config,
                resolved_target_item,
                attack_identity_context=attack_identity_context,
            )
            / "continuous_init_diagnostic"
        )
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    diagnostic_config_path = output_path / "diagnostic_config.json"
    save_json(
        _diagnostic_config_payload(
            config=config,
            config_path=config_path,
            target_item=resolved_target_item,
            cem_config=cem_config,
            continuous_config=continuous_config,
            population_size=first_population_size,
            candidate_count=len(sample_plan),
        ),
        diagnostic_config_path,
    )
    paths["diagnostic_config"] = str(diagnostic_config_path)

    initial_candidates_path = output_path / "initial_candidates.json"
    save_json(initial_candidates, initial_candidates_path)
    paths["initial_candidates"] = str(initial_candidates_path)

    candidate_summary_path = output_path / "candidate_distribution_summary.csv"
    _write_csv(candidate_summary_path, CANDIDATE_DISTRIBUTION_COLUMNS, candidate_rows)
    paths["candidate_distribution_summary"] = str(candidate_summary_path)

    by_suffix_path = output_path / "candidate_by_suffix_len_summary.csv"
    _write_csv(by_suffix_path, BY_SUFFIX_COLUMNS, by_suffix_rows)
    paths["candidate_by_suffix_len_summary"] = str(by_suffix_path)

    overall_suffix_path = output_path / "overall_suffix_context_summary.csv"
    _write_csv(
        overall_suffix_path,
        OVERALL_SUFFIX_COLUMNS,
        _overall_suffix_context_rows(session_contexts),
    )
    paths["overall_suffix_context_summary"] = str(overall_suffix_path)

    if include_rounding_variants:
        rounding_path = output_path / "rounding_variant_summary.csv"
        _write_csv(rounding_path, ROUNDING_VARIANT_COLUMNS, rounding_rows)
        paths["rounding_variant_summary"] = str(rounding_path)

    session_samples_path = output_path / "session_samples.jsonl"
    _write_jsonl(session_samples_path, session_sample_rows)
    paths["session_samples"] = str(session_samples_path)

    if behavior_config.enabled:
        _write_behavior_aware_artifacts(
            output_path=output_path,
            paths=paths,
            behavior_config=behavior_config,
            default_select_size=first_population_size,
            cem_config=cem_config,
            continuous_config=continuous_config,
            pts_config=pts_config,
            session_contexts=session_contexts,
            target_item=resolved_target_item,
        )

    print(f"[continuous-init-diagnostic] wrote {output_path}")
    return ContinuousInitDiagnosticResult(output_dir=output_path, paths=paths)


def run_continuous_init_diagnostic(
    **kwargs: Any,
) -> ContinuousInitDiagnosticResult:
    return run_continuous_beta_init_diagnostic(**kwargs)


def _write_behavior_aware_artifacts(
    *,
    output_path: Path,
    paths: dict[str, str],
    behavior_config: BehaviorAwareSelectionConfig,
    default_select_size: int,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
) -> None:
    if (
        str(behavior_config.mode).strip().lower()
        in {
            BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
            BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
        }
    ):
        _write_behavior_curve_two_pool_artifacts(
            output_path=output_path,
            paths=paths,
            behavior_config=behavior_config,
            cem_config=cem_config,
            continuous_config=continuous_config,
            pts_config=pts_config,
            session_contexts=session_contexts,
            target_item=int(target_item),
        )
        return
    pool = _build_behavior_candidate_pool(
        behavior_config=behavior_config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=pts_config,
        session_contexts=session_contexts,
        target_item=int(target_item),
    )
    select_size = min(
        behavior_config.resolved_select_size(default_select_size),
        len(pool),
    )
    if select_size < behavior_config.resolved_select_size(default_select_size):
        print(
            "[continuous-beta-init-diagnostic] warning: behavior_select_size "
            "exceeds pool size; selecting all available candidates."
        )
    fallback_records: list[dict[str, object]] = []
    if (
        str(behavior_config.mode).strip().lower()
        == BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1
    ):
        selected = select_behavior_stratified_space_filling_candidates(
            pool,
            select_size=select_size,
            distance=str(behavior_config.distance),
            extreme_count=int(behavior_config.extreme_count),
            moderate_count=int(behavior_config.moderate_count),
            balanced_count=int(behavior_config.balanced_count),
            extreme_max_action_ratio_min=float(
                behavior_config.extreme_max_action_ratio_min
            ),
            extreme_max_action_ratio_max=float(
                behavior_config.extreme_max_action_ratio_max
            ),
            moderate_max_action_ratio_min=float(
                behavior_config.moderate_max_action_ratio_min
            ),
            moderate_max_action_ratio_max=float(
                behavior_config.moderate_max_action_ratio_max
            ),
            reject_max_action_ratio_above=float(
                behavior_config.reject_max_action_ratio_above
            ),
            fallback_records=fallback_records,
        )
    else:
        selected = select_behavior_aware_candidates(
            pool,
            select_size=select_size,
            distance=str(behavior_config.distance),
            min_behavior_distance=float(behavior_config.min_behavior_distance),
        )
    selected = _attach_selected_min_distances(selected)
    selected_by_pool_key = {
        str(candidate["pool_candidate_key"]): (rank, candidate)
        for rank, candidate in enumerate(selected)
    }
    balanced_pool_keys = {
        str(candidate["pool_candidate_key"])
        for candidate in selected
        if str(candidate.get("selection_stratum", "")) == "balanced"
    }

    pool_summary_path = output_path / "behavior_pool_summary.csv"
    _write_csv(
        pool_summary_path,
        BEHAVIOR_POOL_SUMMARY_COLUMNS,
        [
            _behavior_pool_summary_row(
                candidate,
                selected_entry=selected_by_pool_key.get(
                    str(candidate["pool_candidate_key"])
                ),
                behavior_config=behavior_config,
                balanced_pool_keys=balanced_pool_keys,
            )
            for candidate in pool
        ],
    )
    paths["behavior_pool_summary"] = str(pool_summary_path)

    selected_payload = [
        _behavior_selected_candidate_payload(candidate, selected_rank=rank)
        for rank, candidate in enumerate(selected)
    ]
    selected_candidates_path = output_path / "behavior_selected_candidates.json"
    save_json(selected_payload, selected_candidates_path)
    paths["behavior_selected_candidates"] = str(selected_candidates_path)

    selected_candidate_rows: list[dict[str, object]] = []
    selected_by_suffix_rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(selected):
        selected_info = _selected_candidate_info(candidate, selected_rank=rank)
        records = candidate["records"]
        selected_candidate_rows.append(_candidate_summary_row(selected_info, records))
        selected_by_suffix_rows.extend(_by_suffix_summary_rows(selected_info, records))

    selected_summary_path = output_path / "behavior_selected_distribution_summary.csv"
    _write_csv(
        selected_summary_path,
        BEHAVIOR_SELECTED_DISTRIBUTION_COLUMNS,
        selected_candidate_rows,
    )
    paths["behavior_selected_distribution_summary"] = str(selected_summary_path)

    selected_by_suffix_path = output_path / "behavior_selected_by_suffix_len_summary.csv"
    _write_csv(
        selected_by_suffix_path,
        BEHAVIOR_SELECTED_BY_SUFFIX_COLUMNS,
        selected_by_suffix_rows,
    )
    paths["behavior_selected_by_suffix_len_summary"] = str(selected_by_suffix_path)

    selection_config_path = output_path / "behavior_selection_config.json"
    save_json(
        _behavior_selection_config_payload(
            behavior_config,
            default_select_size=default_select_size,
            selected_count=len(selected),
            fallbacks_used=fallback_records,
        ),
        selection_config_path,
    )
    paths["behavior_selection_config"] = str(selection_config_path)


def _write_behavior_curve_two_pool_artifacts(
    *,
    output_path: Path,
    paths: dict[str, str],
    behavior_config: BehaviorAwareSelectionConfig,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
) -> None:
    soft_pool = _build_behavior_candidate_pool(
        behavior_config=behavior_config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=pts_config,
        session_contexts=session_contexts,
        target_item=int(target_item),
        pool_size=int(behavior_config.soft_extreme_pool_size),
        source_pool="soft_extreme",
        key_prefix="soft_extreme_pool_cand",
        initial_std=float(behavior_config.soft_extreme_std),
        seed_offset=0,
    )
    moderate_pool = _build_behavior_candidate_pool(
        behavior_config=behavior_config,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=pts_config,
        session_contexts=session_contexts,
        target_item=int(target_item),
        pool_size=int(behavior_config.moderate_pool_size),
        source_pool="moderate",
        key_prefix="moderate_pool_cand",
        initial_std=float(behavior_config.moderate_std),
        seed_offset=100000,
    )
    pool = [
        _with_behavior_curve_profile(
            candidate,
            q_grid_size=int(behavior_config.q_grid_size),
            q_grid_min=float(behavior_config.q_grid_min),
            q_grid_max=float(behavior_config.q_grid_max),
            q_kernel_bandwidth=float(behavior_config.q_kernel_bandwidth),
        )
        for candidate in [*soft_pool, *moderate_pool]
    ]
    soft_pool = [
        candidate for candidate in pool if str(candidate["source_pool"]) == "soft_extreme"
    ]
    moderate_pool = [
        candidate for candidate in pool if str(candidate["source_pool"]) == "moderate"
    ]
    selected_pool_candidates, fallback_records = select_behavior_curve_two_pool_candidates(
        soft_extreme_pool=soft_pool,
        moderate_pool=moderate_pool,
        soft_extreme_select_size=int(behavior_config.soft_extreme_select_size),
        moderate_select_size=int(behavior_config.moderate_select_size),
        distance=str(behavior_config.distance),
    )
    selected = _reapply_selected_behavior_candidates(
        selected_pool_candidates,
        cem_config=cem_config,
        continuous_config=continuous_config,
        pts_config=pts_config,
        session_contexts=session_contexts,
        target_item=int(target_item),
    )
    selected = _attach_selected_min_distances(selected)
    selected_source_by_key = {
        str(candidate["source_pool_candidate_key"]): (rank, candidate)
        for rank, candidate in enumerate(selected)
    }

    pool_summary_path = output_path / "behavior_pool_summary.csv"
    _write_csv(
        pool_summary_path,
        BEHAVIOR_POOL_SUMMARY_COLUMNS,
        [
            _behavior_pool_summary_row(
                candidate,
                selected_entry=selected_source_by_key.get(
                    str(candidate["pool_candidate_key"])
                ),
                behavior_config=behavior_config,
                balanced_pool_keys=set(),
            )
            for candidate in pool
        ],
    )
    paths["behavior_pool_summary"] = str(pool_summary_path)

    selected_candidates_path = output_path / "behavior_selected_candidates.json"
    save_json(
        [
            _behavior_selected_candidate_payload(candidate, selected_rank=rank)
            for rank, candidate in enumerate(selected)
        ],
        selected_candidates_path,
    )
    paths["behavior_selected_candidates"] = str(selected_candidates_path)

    selected_candidate_rows: list[dict[str, object]] = []
    selected_by_suffix_rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(selected):
        selected_info = _selected_candidate_info(candidate, selected_rank=rank)
        records = candidate["records"]
        selected_candidate_rows.append(_candidate_summary_row(selected_info, records))
        selected_by_suffix_rows.extend(_by_suffix_summary_rows(selected_info, records))

    selected_summary_path = output_path / "behavior_selected_distribution_summary.csv"
    _write_csv(
        selected_summary_path,
        BEHAVIOR_SELECTED_DISTRIBUTION_COLUMNS,
        selected_candidate_rows,
    )
    paths["behavior_selected_distribution_summary"] = str(selected_summary_path)

    selected_by_suffix_path = output_path / "behavior_selected_by_suffix_len_summary.csv"
    _write_csv(
        selected_by_suffix_path,
        BEHAVIOR_SELECTED_BY_SUFFIX_COLUMNS,
        selected_by_suffix_rows,
    )
    paths["behavior_selected_by_suffix_len_summary"] = str(selected_by_suffix_path)

    curve_metrics_path = output_path / "behavior_curve_metrics.csv"
    _write_csv(
        curve_metrics_path,
        BEHAVIOR_CURVE_METRICS_COLUMNS,
        [
            _behavior_curve_metrics_row(
                candidate,
                selected_entry=selected_source_by_key.get(
                    str(candidate["pool_candidate_key"])
                ),
            )
            for candidate in pool
        ]
        + [
            _behavior_curve_metrics_row(
                candidate,
                selected_entry=(rank, candidate),
                force_source_pool="selected",
            )
            for rank, candidate in enumerate(selected)
        ],
    )
    paths["behavior_curve_metrics"] = str(curve_metrics_path)

    selection_config_path = output_path / "behavior_selection_config.json"
    save_json(
        _behavior_selection_config_payload(
            behavior_config,
            default_select_size=int(
                behavior_config.soft_extreme_select_size
                + behavior_config.moderate_select_size
            ),
            selected_count=len(selected),
            fallbacks_used=fallback_records,
        ),
        selection_config_path,
    )
    paths["behavior_selection_config"] = str(selection_config_path)


def _build_behavior_candidate_pool(
    *,
    behavior_config: BehaviorAwareSelectionConfig,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
    pool_size: int | None = None,
    source_pool: str = "behavior",
    key_prefix: str = "pool_cand",
    initial_std: float | None = None,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
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
            target_item=int(target_item),
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


def _validate_continuous_config(config: Config) -> None:
    pts_config = _require_pts_config(config)
    if pts_config.method != PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM:
        raise ValueError(
            "Continuous init diagnostic requires "
            "attack.pts_construction.method='continuous_mlp_cem'."
        )
    _validate_pts_construction_run_config(config)


def _resolve_first_target_item(
    config: Config,
    *,
    shared_paths: Mapping[str, Path],
) -> int:
    if config.targets.mode == "explicit_list":
        if not config.targets.explicit_list:
            raise ValueError("targets.explicit_list must not be empty.")
        return int(config.targets.explicit_list[0])
    canonical_dataset = ensure_canonical_dataset(config)
    stats = compute_session_stats(canonical_dataset.train_sub)
    target_registry = ensure_target_registry_prefix(
        stats,
        config,
        shared_paths=dict(shared_paths),
    )
    target_items = requested_target_prefix(config, target_registry=target_registry)
    if not target_items:
        raise ValueError("No target items resolved for diagnostic.")
    return int(target_items[0])


def _first_population_size(cem_config: PTSCEMConfig) -> int:
    if cem_config.population_schedule is not None:
        return int(cem_config.population_schedule[0])
    if cem_config.population_size is None:
        raise ValueError("population_size is required without population_schedule.")
    return int(cem_config.population_size)


def _diagnostic_config_payload(
    *,
    config: Config,
    config_path: str | Path | None,
    target_item: int,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    population_size: int,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": None if config_path is None else str(config_path),
        "target_item": int(target_item),
        "dataset": config.data.dataset_name,
        "method": PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
        "parameterization": continuous_config.parameterization,
        "continuous_policy": {
            "parameterization": "suffix_length_mlp",
            "hidden_size": 2,
            "consume_distribution": "beta",
            "smoothing_epsilon": float(continuous_config.smoothing_epsilon),
            "source_policy": "q_and_rho_logistic",
            "deterministic_sampling": bool(continuous_config.deterministic_sampling),
        },
        "population_size": int(population_size),
        "candidate_count": int(candidate_count),
        "initialization_mode": continuous_config.initialization_mode,
        "parameter_bounds": {
            "min": float(continuous_config.parameter_bounds[0]),
            "max": float(continuous_config.parameter_bounds[1]),
        },
        "initial_std": float(continuous_config.initial_std),
        "min_std": float(continuous_config.min_std),
        "smoothing_epsilon": float(continuous_config.smoothing_epsilon),
        "consume_smoothing": "beta_uniform_mixture",
        "source_probability_floor": float(continuous_config.smoothing_epsilon),
        "cem_base_seed": int(cem_config.base_seed),
        "candidate_seed_stride": int(cem_config.candidate_seed_stride),
        "shared_prefix_assignment_tag": CONTINUOUS_BETA_SHARED_PREFIX_TAG,
        "rounding_mode": "half_up",
        "materialize_generated_suffix": False,
        "init_materialize_generated_suffix": False,
    }


def _initial_candidate_payload(candidate_info: Mapping[str, object]) -> dict[str, object]:
    policy = candidate_info["policy"]
    if not isinstance(policy, ContinuousBetaPolicy):
        raise TypeError("candidate_info['policy'] must be a ContinuousBetaPolicy.")
    return {
        "candidate_key": str(candidate_info["candidate_key"]),
        "candidate_id": int(candidate_info["candidate_id"]),
        "sample_origin": str(candidate_info["sample_origin"]),
        "sample_metadata": dict(candidate_info["sample_metadata"]),
        "policy": policy.to_dict(),
        "parameter_vector": [float(value) for value in policy.to_vector()],
    }


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


def _by_suffix_summary_rows(
    candidate_info: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["residual_suffix_length"])].append(record)
    rows: list[dict[str, object]] = []
    for suffix_len, group_records in sorted(grouped.items()):
        row = _candidate_base_row(candidate_info)
        row.update(
            {
                "residual_suffix_len": int(suffix_len),
                "num_sessions": int(len(group_records)),
                "q_suffix": _mean(
                    _field_floats(group_records, "suffix_length_percentile")
                ),
                "rho_mean": _mean(_field_floats(group_records, "consume_ratio")),
                "rho_std": _std(_field_floats(group_records, "consume_ratio")),
                "rho_min": _min(_field_floats(group_records, "consume_ratio")),
                "rho_max": _max(_field_floats(group_records, "consume_ratio")),
                "consume_count_mean": _mean(_field_ints(group_records, "consume_count")),
            }
        )
        row.update(_behavior_ratio_fields(group_records))
        row.update(_action_ratio_fields(group_records))
        rows.append(row)
    return rows


def _overall_suffix_context_rows(
    session_contexts: Sequence[PTSContinuousSessionContext],
) -> list[dict[str, object]]:
    total = int(len(session_contexts))
    grouped: dict[int, list[PTSContinuousSessionContext]] = defaultdict(list)
    for context in session_contexts:
        grouped[int(context.residual_suffix_length)].append(context)
    return [
        {
            "residual_suffix_len": int(suffix_len),
            "num_sessions": int(len(contexts)),
            "ratio": _ratio(int(len(contexts)), total),
            "q_suffix": _mean(
                [float(context.suffix_length_percentile) for context in contexts]
            ),
        }
        for suffix_len, contexts in sorted(grouped.items())
    ]


def build_behavior_vector(records: Sequence[Mapping[str, object]]) -> list[float]:
    groups = [
        list(records),
        [
            record
            for record in records
            if int(record["residual_suffix_length"]) == 1
        ],
        [
            record
            for record in records
            if int(record["residual_suffix_length"]) == 2
        ],
        [
            record
            for record in records
            if int(record["residual_suffix_length"]) >= 3
        ],
    ]
    vector: list[float] = []
    for group_records in groups:
        ratios = _action_ratio_fields(group_records)
        vector.extend(
            float(ratios[f"{action_name}_ratio"])
            for action_name in ACTION_COLUMNS
        )
    return vector


def behavior_statistics(
    summary: Mapping[str, object],
    behavior_vector: Sequence[float],
) -> dict[str, float]:
    overall = [
        float(summary[f"{action_name}_ratio"])
        for action_name in ACTION_COLUMNS
    ]
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
        _weighted_action_distribution(
            records,
            q0=float(q0),
            bandwidth=float(q_kernel_bandwidth),
        )
        for q0 in q_grid
    ]
    vector = [
        float(probability)
        for distribution in distributions
        for probability in distribution
    ]
    entropies = [_normalized_entropy(distribution) for distribution in distributions]
    max_probs = [
        max(float(probability) for probability in distribution)
        for distribution in distributions
    ]
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


def select_behavior_aware_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    select_size: int,
    distance: str = "l1",
    min_behavior_distance: float = 1e-9,
    max_stop_ratio: float | None = None,
    max_per_dominant_family: int | None = None,
    min_partial_candidates: int | None = None,
    min_generate_candidates: int | None = None,
) -> list[Mapping[str, Any]]:
    if str(distance).strip().lower() != "l1":
        raise ValueError("behavior_distance currently supports only 'l1'.")
    if int(select_size) <= 0:
        raise ValueError("select_size must be positive.")
    if float(min_behavior_distance) < 0.0:
        raise ValueError("min_behavior_distance must be non-negative.")
    pool = list(candidates)
    if not pool:
        return []
    selected: list[Mapping[str, Any]] = [_most_balanced_candidate(pool)]
    _fill_behavior_selection(
        pool,
        selected,
        select_size=int(select_size),
        min_behavior_distance=float(min_behavior_distance),
        enforce_min_distance=True,
    )
    _fill_behavior_selection(
        pool,
        selected,
        select_size=int(select_size),
        min_behavior_distance=float(min_behavior_distance),
        enforce_min_distance=False,
    )
    return selected[: int(select_size)]


def select_behavior_stratified_space_filling_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    select_size: int,
    distance: str = "l1",
    extreme_count: int = 6,
    moderate_count: int = 9,
    balanced_count: int = 1,
    extreme_max_action_ratio_min: float = 0.70,
    extreme_max_action_ratio_max: float = 0.90,
    moderate_max_action_ratio_min: float = 0.35,
    moderate_max_action_ratio_max: float = 0.70,
    reject_max_action_ratio_above: float = 0.95,
    fallback_records: list[dict[str, object]] | None = None,
) -> list[Mapping[str, Any]]:
    if str(distance).strip().lower() != "l1":
        raise ValueError("behavior_distance currently supports only 'l1'.")
    if int(select_size) <= 0:
        raise ValueError("select_size must be positive.")
    for name, value in (
        ("extreme_count", extreme_count),
        ("moderate_count", moderate_count),
        ("balanced_count", balanced_count),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be non-negative.")
    if int(extreme_count) + int(moderate_count) + int(balanced_count) > int(select_size):
        raise ValueError("requested stratum counts must not exceed select_size.")

    pool = [_with_behavior_stats(candidate) for candidate in candidates]
    if not pool:
        return []

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    fallbacks = fallback_records if fallback_records is not None else []

    noncollapsed = [
        candidate
        for candidate in pool
        if _candidate_max_action_ratio(candidate) <= float(reject_max_action_ratio_above)
    ]
    eligible_for_balance = noncollapsed if noncollapsed else pool
    if not noncollapsed:
        fallbacks.append(
            {
                "stage": "balanced",
                "reason": "no_noncollapsed_candidates",
                "added": 0,
            }
        )

    balanced = _highest_entropy_candidates(
        eligible_for_balance,
        count=int(balanced_count),
        selected_keys=selected_keys,
    )
    _append_selected(
        selected,
        selected_keys,
        balanced,
        stratum="balanced",
        reason="highest_entropy",
    )

    extreme_pool = [
        candidate
        for candidate in noncollapsed
        if _candidate_key(candidate) not in selected_keys
        and float(extreme_max_action_ratio_min)
        <= _candidate_max_action_ratio(candidate)
        <= float(extreme_max_action_ratio_max)
    ]
    extreme = _maximin_candidates(
        extreme_pool,
        count=int(extreme_count),
        selected_keys=selected_keys,
    )
    _append_selected(
        selected,
        selected_keys,
        extreme,
        stratum="extreme",
        reason="extreme_maximin",
    )
    if len(extreme) < int(extreme_count):
        fallbacks.append(
            {
                "stage": "extreme",
                "reason": "insufficient_extreme_pool",
                "requested": int(extreme_count),
                "selected": int(len(extreme)),
            }
        )

    moderate_pool = [
        candidate
        for candidate in noncollapsed
        if _candidate_key(candidate) not in selected_keys
        and float(moderate_max_action_ratio_min)
        <= _candidate_max_action_ratio(candidate)
        < float(moderate_max_action_ratio_max)
    ]
    moderate = _maximin_candidates(
        moderate_pool,
        count=int(moderate_count),
        selected_keys=selected_keys,
    )
    _append_selected(
        selected,
        selected_keys,
        moderate,
        stratum="moderate",
        reason="moderate_maximin",
    )
    if len(moderate) < int(moderate_count):
        fallbacks.append(
            {
                "stage": "moderate",
                "reason": "insufficient_moderate_pool",
                "requested": int(moderate_count),
                "selected": int(len(moderate)),
            }
        )

    _fill_stratified_fallback(
        selected,
        selected_keys,
        [
            candidate
            for candidate in noncollapsed
            if (
                float(moderate_max_action_ratio_min)
                <= _candidate_max_action_ratio(candidate)
                < float(moderate_max_action_ratio_max)
            )
            or (
                float(extreme_max_action_ratio_min)
                <= _candidate_max_action_ratio(candidate)
                <= float(extreme_max_action_ratio_max)
            )
        ],
        select_size=int(select_size),
        stratum="fallback",
        reason="fallback_other_stratum",
        fallbacks=fallbacks,
    )
    _fill_stratified_fallback(
        selected,
        selected_keys,
        noncollapsed,
        select_size=int(select_size),
        stratum="fallback",
        reason="fallback_noncollapsed",
        fallbacks=fallbacks,
    )
    _fill_stratified_fallback(
        selected,
        selected_keys,
        pool,
        select_size=int(select_size),
        stratum="fallback",
        reason="fallback_full_pool",
        fallbacks=fallbacks,
    )
    return selected[: int(select_size)]


def _fill_behavior_selection(
    pool: Sequence[Mapping[str, Any]],
    selected: list[Mapping[str, Any]],
    *,
    select_size: int,
    min_behavior_distance: float,
    enforce_min_distance: bool,
) -> None:
    selected_keys = {str(item["pool_candidate_key"]) for item in selected}
    while len(selected) < int(select_size):
        remaining = [
            candidate
            for candidate in pool
            if str(candidate["pool_candidate_key"]) not in selected_keys
        ]
        if not remaining:
            return
        best = max(
            remaining,
            key=lambda candidate: (
                _min_behavior_distance(candidate, selected),
                -int(candidate.get("pool_candidate_id", 0)),
            ),
        )
        if enforce_min_distance and _min_behavior_distance(
            best,
            selected,
        ) < float(min_behavior_distance):
            return
        selected.append(best)
        selected_keys.add(str(best["pool_candidate_key"]))


def _with_behavior_stats(candidate: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(candidate)
    if "behavior_stats" not in item:
        item["behavior_stats"] = behavior_statistics(
            item["summary"],
            item["behavior_vector"],
        )
    return item


def _candidate_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate["pool_candidate_key"])


def _candidate_pool_id(candidate: Mapping[str, Any]) -> int:
    return int(candidate.get("pool_candidate_id", 0))


def _candidate_max_action_ratio(candidate: Mapping[str, Any]) -> float:
    return float(candidate["behavior_stats"]["max_action_ratio_overall"])


def _candidate_entropy(candidate: Mapping[str, Any]) -> float:
    return float(candidate["behavior_stats"]["entropy_overall"])


def _highest_entropy_candidates(
    pool: Sequence[Mapping[str, Any]],
    *,
    count: int,
    selected_keys: set[str],
) -> list[Mapping[str, Any]]:
    if int(count) <= 0:
        return []
    ordered = sorted(
        [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
        ],
        key=lambda candidate: (
            -_candidate_entropy(candidate),
            _candidate_max_action_ratio(candidate),
            _candidate_pool_id(candidate),
        ),
    )
    return ordered[: int(count)]


def _maximin_candidates(
    pool: Sequence[Mapping[str, Any]],
    *,
    count: int,
    selected_keys: set[str],
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    local_selected_keys: set[str] = set()
    while len(selected) < int(count):
        remaining = [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
            and _candidate_key(candidate) not in local_selected_keys
        ]
        if not remaining:
            break
        if not selected:
            best = _highest_entropy_candidates(
                remaining,
                count=1,
                selected_keys=set(),
            )[0]
        else:
            best = max(
                remaining,
                key=lambda candidate: (
                    _min_behavior_distance(candidate, selected),
                    _candidate_entropy(candidate),
                    -_candidate_max_action_ratio(candidate),
                    -_candidate_pool_id(candidate),
                ),
            )
        selected.append(best)
        local_selected_keys.add(_candidate_key(best))
    return selected


def _append_selected(
    selected: list[dict[str, Any]],
    selected_keys: set[str],
    candidates: Sequence[Mapping[str, Any]],
    *,
    stratum: str,
    reason: str,
) -> None:
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in selected_keys:
            continue
        item = dict(candidate)
        item["selection_stratum"] = str(stratum)
        item["selection_reason"] = str(reason)
        selected.append(item)
        selected_keys.add(key)


def _fill_stratified_fallback(
    selected: list[dict[str, Any]],
    selected_keys: set[str],
    pool: Sequence[Mapping[str, Any]],
    *,
    select_size: int,
    stratum: str,
    reason: str,
    fallbacks: list[dict[str, object]],
) -> None:
    if len(selected) >= int(select_size):
        return
    before = len(selected)
    while len(selected) < int(select_size):
        remaining = [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
        ]
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda candidate: (
                _min_behavior_distance(candidate, selected),
                _candidate_entropy(candidate),
                -_candidate_max_action_ratio(candidate),
                -_candidate_pool_id(candidate),
            ),
        )
        _append_selected(
            selected,
            selected_keys,
            [best],
            stratum=stratum,
            reason=reason,
        )
    added = len(selected) - before
    if added:
        fallbacks.append(
            {
                "stage": reason,
                "added": int(added),
            }
        )


def _q_grid(
    *,
    q_grid_size: int,
    q_grid_min: float,
    q_grid_max: float,
) -> list[float]:
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
        [
            float(candidate["behavior_curve_profile"]["collapse_score"])
            for candidate in candidates
        ],
        1.0 - float(drop_top_collapse_fraction),
    )
    entropy_cutoff = _quantile(
        [
            float(candidate["behavior_curve_profile"]["mean_entropy_over_q"])
            for candidate in candidates
        ],
        float(drop_bottom_entropy_fraction),
    )
    variation_cutoff = _quantile(
        [
            float(candidate["behavior_curve_profile"]["q_variation"])
            for candidate in candidates
        ],
        float(drop_bottom_variation_fraction),
    )
    filtered = [
        candidate
        for candidate in candidates
        if float(candidate["behavior_curve_profile"]["collapse_score"])
        <= float(collapse_cutoff)
        and float(candidate["behavior_curve_profile"]["mean_entropy_over_q"])
        >= float(entropy_cutoff)
        and float(candidate["behavior_curve_profile"]["q_variation"])
        >= float(variation_cutoff)
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
        remaining = [
            candidate
            for candidate in pool
            if _candidate_key(candidate) not in selected_keys
        ]
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
    return [
        float(value)
        for value in candidate["behavior_curve_profile"]["behavior_curve_vector"]
    ]


def _behavior_curve_centroid(candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    vectors = [_behavior_curve_vector(candidate) for candidate in candidates]
    if not vectors:
        return []
    width = len(vectors[0])
    return [
        float(sum(vector[index] for vector in vectors)) / float(len(vectors))
        for index in range(width)
    ]


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


def _quantile(values: Sequence[float | int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    fraction = min(1.0, max(0.0, float(fraction)))
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _min_behavior_distance(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> float:
    if not selected:
        return float("inf")
    return min(
        _l1_distance(candidate["behavior_vector"], item["behavior_vector"])
        for item in selected
    )


def _l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return float(
        sum(abs(float(a) - float(b)) for a, b in zip(left, right))
    )


def _most_balanced_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    target = [0.2 for _ in ACTION_COLUMNS] + [0.0 for _ in range(15)]
    return min(
        candidates,
        key=lambda candidate: (
            _l1_distance(candidate["behavior_vector"], target),
            int(candidate.get("pool_candidate_id", 0)),
        ),
    )


def _dominant_action_family(
    summary: Mapping[str, object],
) -> tuple[str, float]:
    action_ratios = [
        (action_name, float(summary[f"{action_name}_ratio"]))
        for action_name in ACTION_COLUMNS
    ]
    return max(action_ratios, key=lambda item: (item[1], item[0]))


def _behavior_pool_summary_row(
    candidate: Mapping[str, Any],
    *,
    selected_entry: tuple[int, Mapping[str, Any]] | None,
    behavior_config: BehaviorAwareSelectionConfig,
    balanced_pool_keys: set[str],
) -> dict[str, object]:
    summary = candidate["summary"]
    info = candidate["candidate_info"]
    selected_rank = "" if selected_entry is None else int(selected_entry[0])
    selected_key = (
        ""
        if selected_entry is None
        else str(selected_entry[1]["pool_candidate_key"])
    )
    vector = [float(value) for value in candidate["behavior_vector"]]
    parameter_vector = [float(value) for value in info["parameter_vector"]]
    stats = _with_behavior_stats(candidate)["behavior_stats"]
    return {
        "pool_candidate_key": str(candidate["pool_candidate_key"]),
        "selected": bool(selected_entry is not None),
        "selected_rank": selected_rank,
        "selected_candidate_key": selected_key,
        "sample_origin": str(info["sample_origin"]),
        "prototype_name": str(info.get("prototype_name", "")),
        "parameterization": str(info["policy"].parameterization),
        "parameter_count": int(len(parameter_vector)),
        "parameter_vector_json": json.dumps(parameter_vector),
        "dominant_action_family": str(candidate["dominant_action_family"]),
        "dominant_action_ratio": float(candidate["dominant_action_ratio"]),
        "max_action_ratio_overall": float(stats["max_action_ratio_overall"]),
        "entropy_overall": float(stats["entropy_overall"]),
        "max_action_ratio_behavior_vector": float(
            stats["max_action_ratio_behavior_vector"]
        ),
        "entropy_behavior_vector": float(stats["entropy_behavior_vector"]),
        "behavior_selection_pool": _behavior_selection_pool_label(
            candidate,
            behavior_config=behavior_config,
            balanced_pool_keys=balanced_pool_keys,
        ),
        "behavior_vector_json": json.dumps(vector),
        "stop_ratio": float(summary["continuous_stop_ratio"]),
        "full_suffix_ratio": float(summary["continuous_keep_full_suffix_ratio"])
        + float(summary["continuous_generate_full_suffix_ratio"]),
        "partial_ratio": float(summary["continuous_partial_keep_suffix_ratio"])
        + float(summary["continuous_partial_generate_suffix_ratio"]),
        "generate_ratio_non_stop": float(summary["generate_ratio_non_stop"]),
        "continuous_keep_full_suffix_ratio": float(
            summary["continuous_keep_full_suffix_ratio"]
        ),
        "continuous_generate_full_suffix_ratio": float(
            summary["continuous_generate_full_suffix_ratio"]
        ),
        "continuous_partial_keep_suffix_ratio": float(
            summary["continuous_partial_keep_suffix_ratio"]
        ),
        "continuous_partial_generate_suffix_ratio": float(
            summary["continuous_partial_generate_suffix_ratio"]
        ),
        "continuous_stop_ratio": float(summary["continuous_stop_ratio"]),
    }


def _behavior_selection_pool_label(
    candidate: Mapping[str, Any],
    *,
    behavior_config: BehaviorAwareSelectionConfig,
    balanced_pool_keys: set[str],
) -> str:
    if (
        str(behavior_config.mode).strip().lower()
        != BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1
    ):
        return "unclassified"
    key = _candidate_key(candidate)
    if key in balanced_pool_keys:
        return "balanced_candidate"
    max_ratio = _candidate_max_action_ratio(_with_behavior_stats(candidate))
    if max_ratio > float(behavior_config.reject_max_action_ratio_above):
        return "rejected_overcollapsed"
    if (
        float(behavior_config.extreme_max_action_ratio_min)
        <= max_ratio
        <= float(behavior_config.extreme_max_action_ratio_max)
    ):
        return "extreme"
    if (
        float(behavior_config.moderate_max_action_ratio_min)
        <= max_ratio
        < float(behavior_config.moderate_max_action_ratio_max)
    ):
        return "moderate"
    return "fallback_only"


def _reapply_selected_behavior_candidates(
    selected_pool_candidates: Sequence[Mapping[str, Any]],
    *,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
) -> list[dict[str, Any]]:
    reapplied: list[dict[str, Any]] = []
    for selected_rank, candidate in enumerate(selected_pool_candidates):
        info = candidate["candidate_info"]
        policy = info["policy"]
        if not isinstance(policy, ContinuousBetaPolicy):
            raise TypeError("candidate_info['policy'] must be a ContinuousBetaPolicy.")
        selected_key = f"selected_cand{selected_rank}"
        seed = int(cem_config.base_seed) + 200000 + int(selected_rank)
        construction_result = apply_pts_continuous_beta_construction_batch(
            session_contexts=session_contexts,
            target_item=int(target_item),
            policy=policy,
            base_seed=int(cem_config.base_seed),
            candidate_key=selected_key,
            poison_runner=None,
            generation_topk=int(pts_config.generation.topk),
            generation_rng_base_seed=seed,
            generation_rng_tag="pts_generated_suffix",
            materialize_generated_suffix=False,
        )
        records = [dict(record) for record in construction_result.per_session_records]
        candidate_info = dict(info)
        candidate_info["candidate_key"] = selected_key
        candidate_info["candidate_id"] = int(selected_rank)
        sample_metadata = dict(candidate_info.get("sample_metadata", {}))
        sample_metadata["source_pool_candidate_key"] = str(candidate["pool_candidate_key"])
        sample_metadata["selected_rank"] = int(selected_rank)
        sample_metadata["selection_stage"] = str(candidate.get("selection_stage", ""))
        sample_metadata["selection_reason"] = str(candidate.get("selection_reason", ""))
        candidate_info["sample_metadata"] = sample_metadata
        summary = _candidate_summary_row(candidate_info, records)
        behavior_vector = build_behavior_vector(records)
        dominant_action_family, dominant_action_ratio = _dominant_action_family(summary)
        behavior_stats = behavior_statistics(summary, behavior_vector)
        profile = build_behavior_curve_profile(
            records,
            q_grid_size=len(candidate["behavior_curve_profile"]["q_grid"]),
            q_grid_min=float(candidate["behavior_curve_profile"]["q_grid"][0]),
            q_grid_max=float(candidate["behavior_curve_profile"]["q_grid"][-1]),
            q_kernel_bandwidth=float(
                candidate["behavior_curve_profile"].get("q_kernel_bandwidth", 0.10)
            ),
        )
        reapplied.append(
            {
                "pool_candidate_key": selected_key,
                "pool_candidate_id": int(selected_rank),
                "source_pool_candidate_key": str(candidate["pool_candidate_key"]),
                "source_pool": str(candidate.get("source_pool", "")),
                "candidate_info": candidate_info,
                "records": records,
                "summary": summary,
                "behavior_vector": behavior_vector,
                "behavior_stats": behavior_stats,
                "behavior_curve_profile": profile,
                "dominant_action_family": dominant_action_family,
                "dominant_action_ratio": float(dominant_action_ratio),
                "selection_stage": str(candidate.get("selection_stage", "")),
                "selection_stratum": str(candidate.get("selection_stratum", "")),
                "selection_reason": str(candidate.get("selection_reason", "")),
            }
        )
    return reapplied


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


def _behavior_selected_candidate_payload(
    candidate: Mapping[str, Any],
    *,
    selected_rank: int,
) -> dict[str, object]:
    info = candidate["candidate_info"]
    policy = info["policy"]
    if not isinstance(policy, ContinuousBetaPolicy):
        raise TypeError("candidate_info['policy'] must be a ContinuousBetaPolicy.")
    stats = _with_behavior_stats(candidate)["behavior_stats"]
    return {
        "selected_rank": int(selected_rank),
        "candidate_key": str(candidate["pool_candidate_key"]),
        "pool_candidate_key": str(candidate["pool_candidate_key"]),
        "source_pool_candidate_key": str(
            candidate.get("source_pool_candidate_key", candidate["pool_candidate_key"])
        ),
        "source_pool": str(candidate.get("source_pool", "")),
        "selection_stage": str(candidate.get("selection_stage", "")),
        "selection_stratum": str(candidate.get("selection_stratum", "")),
        "selection_reason": str(candidate.get("selection_reason", "")),
        "max_action_ratio_overall": float(stats["max_action_ratio_overall"]),
        "entropy_overall": float(stats["entropy_overall"]),
        "sample_origin": str(info["sample_origin"]),
        "prototype_name": str(info.get("prototype_name", "")),
        "parameterization": str(policy.parameterization),
        "policy": policy.to_dict(),
        "parameter_vector": [float(value) for value in info["parameter_vector"]],
        "dominant_action_family": str(candidate["dominant_action_family"]),
        "behavior_vector": [float(value) for value in candidate["behavior_vector"]],
        "min_distance_to_previous_selected": float(
            candidate.get("min_distance_to_previous_selected", 0.0)
        ),
    }


def _selected_candidate_info(
    candidate: Mapping[str, Any],
    *,
    selected_rank: int,
) -> dict[str, object]:
    info = dict(candidate["candidate_info"])
    info["candidate_key"] = str(candidate["pool_candidate_key"])
    info["candidate_id"] = int(candidate.get("pool_candidate_id", selected_rank))
    info["selected_rank"] = int(selected_rank)
    info["source_pool_candidate_key"] = str(
        candidate.get("source_pool_candidate_key", candidate["pool_candidate_key"])
    )
    info["selection_stratum"] = str(candidate.get("selection_stratum", ""))
    info["selection_reason"] = str(candidate.get("selection_reason", ""))
    stats = _with_behavior_stats(candidate)["behavior_stats"]
    info["max_action_ratio_overall"] = float(stats["max_action_ratio_overall"])
    info["entropy_overall"] = float(stats["entropy_overall"])
    sample_metadata = dict(info.get("sample_metadata", {}))
    sample_metadata["pool_candidate_key"] = str(candidate["pool_candidate_key"])
    sample_metadata["selected_rank"] = int(selected_rank)
    sample_metadata["selection_stratum"] = str(candidate.get("selection_stratum", ""))
    sample_metadata["selection_reason"] = str(candidate.get("selection_reason", ""))
    info["sample_metadata"] = sample_metadata
    return info


def _behavior_selection_config_payload(
    behavior_config: BehaviorAwareSelectionConfig,
    *,
    default_select_size: int,
    selected_count: int,
    fallbacks_used: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    mode = str(behavior_config.mode).strip().lower()
    return {
        "behavior_aware_select": True,
        "mode": mode,
        "selection_mode": mode,
        "behavior_pool_size": int(behavior_config.pool_size),
        "behavior_select_size": int(
            behavior_config.resolved_select_size(default_select_size)
        ),
        "behavior_selected_count": int(selected_count),
        "selection_method": mode,
        "behavior_vector": (
            "overall_plus_suffix_1_suffix_2_suffix_3plus_action_ratios"
        ),
        "distance": str(behavior_config.distance),
        "behavior_min_distance": float(behavior_config.min_behavior_distance),
        "extreme_count": int(behavior_config.extreme_count),
        "moderate_count": int(behavior_config.moderate_count),
        "balanced_count": int(behavior_config.balanced_count),
        "extreme_max_action_ratio_min": float(
            behavior_config.extreme_max_action_ratio_min
        ),
        "extreme_max_action_ratio_max": float(
            behavior_config.extreme_max_action_ratio_max
        ),
        "moderate_max_action_ratio_min": float(
            behavior_config.moderate_max_action_ratio_min
        ),
        "moderate_max_action_ratio_max": float(
            behavior_config.moderate_max_action_ratio_max
        ),
        "reject_max_action_ratio_above": float(
            behavior_config.reject_max_action_ratio_above
        ),
        "soft_extreme_pool_size": int(behavior_config.soft_extreme_pool_size),
        "moderate_pool_size": int(behavior_config.moderate_pool_size),
        "soft_extreme_select_size": int(behavior_config.soft_extreme_select_size),
        "moderate_select_size": int(behavior_config.moderate_select_size),
        "soft_extreme_std": float(behavior_config.soft_extreme_std),
        "moderate_std": float(behavior_config.moderate_std),
        "q_grid_size": int(behavior_config.q_grid_size),
        "q_grid_min": float(behavior_config.q_grid_min),
        "q_grid_max": float(behavior_config.q_grid_max),
        "q_kernel_bandwidth": float(behavior_config.q_kernel_bandwidth),
        "filtering": {
            "soft_extreme": "drop top 5% by collapse_score",
            "moderate": (
                "drop top 10% by collapse_score, bottom 10% by "
                "mean_entropy_over_q, and bottom 10% by q_variation"
            ),
        },
        "fallbacks_used": [dict(item) for item in (fallbacks_used or [])],
        "uses_action_specific_quotas": False,
        "uses_action_specific_caps": False,
        "candidate_key_policy": "preserve_pool_candidate_key_with_selected_rank",
    }


def _attach_selected_min_distances(
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for candidate in selected:
        item = dict(candidate)
        item["min_distance_to_previous_selected"] = (
            0.0 if not attached else _min_behavior_distance(candidate, attached)
        )
        attached.append(item)
    return attached


def _rounding_variant_rows(
    candidate_info: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[int(record["residual_suffix_length"])].append(record)
    rows: list[dict[str, object]] = []
    for suffix_len, group_records in sorted(grouped.items()):
        for mode in ("floor", "half_up", "ceil"):
            consume_counts = [
                _rounding_variant_count(
                    float(record["consume_ratio"]),
                    int(record["residual_suffix_length"]),
                    mode=mode,
                )
                for record in group_records
            ]
            rows.append(
                {
                    "candidate_key": str(candidate_info["candidate_key"]),
                    "rounding_mode": mode,
                    "residual_suffix_len": int(suffix_len),
                    "num_sessions": int(len(group_records)),
                    "consume_0_ratio": _ratio(
                        sum(1 for count in consume_counts if int(count) == 0),
                        len(consume_counts),
                    ),
                    "consume_partial_ratio": _ratio(
                        sum(1 for count in consume_counts if 0 < int(count) < suffix_len),
                        len(consume_counts),
                    ),
                    "consume_all_ratio": _ratio(
                        sum(1 for count in consume_counts if int(count) == suffix_len),
                        len(consume_counts),
                    ),
                    "consume_count_mean": _mean(consume_counts),
                }
            )
    return rows


def _rounding_variant_count(rho: float, suffix_len: int, *, mode: str) -> int:
    m = int(suffix_len)
    if mode == "floor":
        count = int(math.floor(float(rho) * float(m)))
    elif mode == "half_up":
        return compute_half_up_consume_count(float(rho), m)
    elif mode == "ceil":
        count = int(math.ceil(float(rho) * float(m)))
    else:
        raise ValueError(f"Unsupported rounding mode: {mode}")
    return min(max(count, 0), m)


def _session_sample_row(
    candidate_key_value: str,
    record: Mapping[str, object],
) -> dict[str, object]:
    residual_len = int(record["residual_suffix_length"])
    consume_count = int(record["consume_count"])
    return {
        "candidate_key": str(candidate_key_value),
        "fake_session_index": int(record["fake_session_index"]),
        "template_length": int(record["template_length"]),
        "anchor_position": int(record["anchor_position"]),
        "prefix_length": int(record["prefix_length"]),
        "residual_suffix_length": residual_len,
        "suffix_length_percentile": float(record["suffix_length_percentile"]),
        "alpha": float(record["beta_alpha"]),
        "beta": float(record["beta_beta"]),
        "rho": float(record["consume_ratio"]),
        "consume_count": consume_count,
        "remaining_length": int(residual_len - consume_count),
        "source_generate_probability": record.get("source_generate_probability"),
        "continuation_source": str(record["continuation_source"]),
        "action": str(record["action"]),
    }


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
    for field_name in (
        "selected_rank",
        "source_pool_candidate_key",
        "selection_stratum",
        "selection_reason",
        "max_action_ratio_overall",
        "entropy_overall",
    ):
        if field_name in candidate_info:
            row[field_name] = candidate_info[field_name]
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
    row.update(
        {
            name: float(vector[index])
            for index, name in enumerate(names)
        }
    )
    return {
        str(name): value
        for name, value in row.items()
    }


def _behavior_ratio_fields(
    records: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    total = int(len(records))
    non_stop_records = [
        record for record in records if str(record["continuation_source"]) != "stop"
    ]
    generated_non_stop = [
        record
        for record in non_stop_records
        if str(record["continuation_source"]) == "generate"
    ]
    return {
        "consume_0_ratio": _ratio(
            sum(1 for record in records if int(record["consume_count"]) == 0),
            total,
        ),
        "consume_partial_ratio": _ratio(
            sum(
                1
                for record in records
                if 0
                < int(record["consume_count"])
                < int(record["residual_suffix_length"])
            ),
            total,
        ),
        "consume_all_ratio": _ratio(
            sum(
                1
                for record in records
                if int(record["consume_count"])
                == int(record["residual_suffix_length"])
            ),
            total,
        ),
        "non_stop_ratio": _ratio(len(non_stop_records), total),
        "stop_ratio": _ratio(total - len(non_stop_records), total),
        "generate_ratio_non_stop": _ratio(
            len(generated_non_stop),
            len(non_stop_records),
        ),
    }


def _action_ratio_fields(
    records: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    total = int(len(records))
    counts = Counter(str(record["action"]) for record in records)
    return {
        f"{action_name}_ratio": _ratio(int(counts[action_name]), total)
        for action_name in ACTION_COLUMNS
    }


def _field_floats(
    records: Sequence[Mapping[str, object]],
    field_name: str,
) -> list[float]:
    return [float(record[field_name]) for record in records]


def _field_ints(
    records: Sequence[Mapping[str, object]],
    field_name: str,
) -> list[int]:
    return [int(record[field_name]) for record in records]


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
    return float(
        (
            sum((float(value) - center) ** 2.0 for value in values)
            / float(len(values))
        )
        ** 0.5
    )


def _min(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(min(float(value) for value in values))


def _max(values: Sequence[float | int]) -> float:
    return 0.0 if not values else float(max(float(value) for value in values))


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect continuous parameter CEM iteration-0 initialization.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--sample-sessions", type=int, default=200)
    parser.add_argument("--include-rounding-variants", action="store_true")
    parser.add_argument("--behavior-aware-select", action="store_true")
    parser.add_argument(
        "--behavior-selection-mode",
        default=BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN,
        choices=[
            BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN,
            BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1,
            BEHAVIOR_SELECTION_MODE_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
            BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1,
        ],
    )
    parser.add_argument("--behavior-pool-size", type=int, default=256)
    parser.add_argument("--behavior-select-size", type=int, default=None)
    parser.add_argument("--behavior-distance", default="l1")
    parser.add_argument("--behavior-min-distance", type=float, default=1e-9)
    parser.add_argument("--behavior-max-stop-ratio", type=float, default=None)
    parser.add_argument("--behavior-max-per-dominant-family", type=int, default=None)
    parser.add_argument("--behavior-min-partial-candidates", type=int, default=None)
    parser.add_argument("--behavior-min-generate-candidates", type=int, default=None)
    parser.add_argument("--behavior-extreme-count", type=int, default=6)
    parser.add_argument("--behavior-moderate-count", type=int, default=9)
    parser.add_argument("--behavior-balanced-count", type=int, default=1)
    parser.add_argument("--behavior-extreme-max-ratio-min", type=float, default=0.70)
    parser.add_argument("--behavior-extreme-max-ratio-max", type=float, default=0.90)
    parser.add_argument("--behavior-moderate-max-ratio-min", type=float, default=0.35)
    parser.add_argument("--behavior-moderate-max-ratio-max", type=float, default=0.70)
    parser.add_argument("--behavior-reject-max-ratio-above", type=float, default=0.95)
    parser.add_argument("--soft-extreme-pool-size", type=int, default=512)
    parser.add_argument("--moderate-pool-size", type=int, default=512)
    parser.add_argument("--soft-extreme-select-size", type=int, default=5)
    parser.add_argument("--moderate-select-size", type=int, default=11)
    parser.add_argument("--soft-extreme-std", type=float, default=1.25)
    parser.add_argument("--moderate-std", type=float, default=0.80)
    parser.add_argument("--q-grid-size", type=int, default=19)
    parser.add_argument("--q-grid-min", type=float, default=0.05)
    parser.add_argument("--q-grid-max", type=float, default=0.95)
    parser.add_argument("--q-kernel-bandwidth", type=float, default=0.10)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_continuous_beta_init_diagnostic(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        sample_sessions=args.sample_sessions,
        include_rounding_variants=bool(args.include_rounding_variants),
        behavior_aware_select=bool(args.behavior_aware_select),
        behavior_selection_mode=str(args.behavior_selection_mode),
        behavior_pool_size=int(args.behavior_pool_size),
        behavior_select_size=args.behavior_select_size,
        behavior_distance=str(args.behavior_distance),
        behavior_min_distance=float(args.behavior_min_distance),
        behavior_max_stop_ratio=args.behavior_max_stop_ratio,
        behavior_max_per_dominant_family=args.behavior_max_per_dominant_family,
        behavior_min_partial_candidates=args.behavior_min_partial_candidates,
        behavior_min_generate_candidates=args.behavior_min_generate_candidates,
        behavior_extreme_count=int(args.behavior_extreme_count),
        behavior_moderate_count=int(args.behavior_moderate_count),
        behavior_balanced_count=int(args.behavior_balanced_count),
        behavior_extreme_max_ratio_min=float(args.behavior_extreme_max_ratio_min),
        behavior_extreme_max_ratio_max=float(args.behavior_extreme_max_ratio_max),
        behavior_moderate_max_ratio_min=float(args.behavior_moderate_max_ratio_min),
        behavior_moderate_max_ratio_max=float(args.behavior_moderate_max_ratio_max),
        behavior_reject_max_ratio_above=float(args.behavior_reject_max_ratio_above),
        soft_extreme_pool_size=int(args.soft_extreme_pool_size),
        moderate_pool_size=int(args.moderate_pool_size),
        soft_extreme_select_size=int(args.soft_extreme_select_size),
        moderate_select_size=int(args.moderate_select_size),
        soft_extreme_std=float(args.soft_extreme_std),
        moderate_std=float(args.moderate_std),
        q_grid_size=int(args.q_grid_size),
        q_grid_min=float(args.q_grid_min),
        q_grid_max=float(args.q_grid_max),
        q_kernel_bandwidth=float(args.q_kernel_bandwidth),
    )
    print(f"[continuous-init-diagnostic] output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEHAVIOR_POOL_SUMMARY_COLUMNS",
    "BEHAVIOR_SELECTED_BY_SUFFIX_COLUMNS",
    "BEHAVIOR_SELECTED_DISTRIBUTION_COLUMNS",
    "BEHAVIOR_SELECTED_METADATA_COLUMNS",
    "BEHAVIOR_CURVE_METRICS_COLUMNS",
    "BEHAVIOR_SELECTION_MODE_CURVE_TWO_POOL_SPACE_FILLING_V1",
    "BEHAVIOR_SELECTION_MODE_GREEDY_MAXIMIN",
    "BEHAVIOR_SELECTION_MODE_STRATIFIED_SPACE_FILLING_V1",
    "BehaviorAwareSelectionConfig",
    "ContinuousInitDiagnosticResult",
    "build_behavior_curve_profile",
    "behavior_statistics",
    "build_behavior_vector",
    "run_continuous_init_diagnostic",
    "run_continuous_beta_init_diagnostic",
    "select_behavior_aware_candidates",
    "select_behavior_curve_two_pool_candidates",
    "select_behavior_stratified_space_filling_candidates",
    "main",
]
