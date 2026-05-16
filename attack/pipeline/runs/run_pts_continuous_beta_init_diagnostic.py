from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import load_fake_sessions, save_json
from attack.common.config import (
    Config,
    PTS_CONSTRUCTION_METHOD_CONTINUOUS_BETA_CEM_V1,
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

CANDIDATE_DISTRIBUTION_COLUMNS = (
    "candidate_key",
    "candidate_id",
    "sample_origin",
    "prototype_name",
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


@dataclass(frozen=True)
class ContinuousInitDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]


@dataclass(frozen=True)
class BehaviorAwareSelectionConfig:
    enabled: bool = False
    pool_size: int = 256
    select_size: int | None = None
    distance: str = "l1"
    max_stop_ratio: float = 0.90
    max_per_dominant_family: int = 3
    min_partial_candidates: int = 2
    min_generate_candidates: int = 2

    def resolved_select_size(self, default_size: int) -> int:
        value = int(default_size if self.select_size is None else self.select_size)
        if value <= 0:
            raise ValueError("behavior_select_size must be positive.")
        return value

    def validate(self, *, default_select_size: int) -> None:
        if int(self.pool_size) <= 0:
            raise ValueError("behavior_pool_size must be positive.")
        self.resolved_select_size(default_select_size)
        if str(self.distance).strip().lower() != "l1":
            raise ValueError("behavior_distance currently supports only 'l1'.")
        if not 0.0 <= float(self.max_stop_ratio) <= 1.0:
            raise ValueError("behavior_max_stop_ratio must be in [0, 1].")
        if int(self.max_per_dominant_family) <= 0:
            raise ValueError("behavior_max_per_dominant_family must be positive.")
        if int(self.min_partial_candidates) < 0:
            raise ValueError("behavior_min_partial_candidates must be non-negative.")
        if int(self.min_generate_candidates) < 0:
            raise ValueError("behavior_min_generate_candidates must be non-negative.")


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
    behavior_pool_size: int = 256,
    behavior_select_size: int | None = None,
    behavior_distance: str = "l1",
    behavior_max_stop_ratio: float = 0.90,
    behavior_max_per_dominant_family: int = 3,
    behavior_min_partial_candidates: int = 2,
    behavior_min_generate_candidates: int = 2,
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
                "Continuous beta init diagnostic requires existing shared "
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
        pool_size=int(behavior_pool_size),
        select_size=behavior_select_size,
        distance=str(behavior_distance),
        max_stop_ratio=float(behavior_max_stop_ratio),
        max_per_dominant_family=int(behavior_max_per_dominant_family),
        min_partial_candidates=int(behavior_min_partial_candidates),
        min_generate_candidates=int(behavior_min_generate_candidates),
    )
    if behavior_config.enabled:
        behavior_config.validate(default_select_size=first_population_size)

    session_contexts = build_continuous_shared_session_contexts(
        template_sessions=templates,
        target_item=resolved_target_item,
        base_seed=int(cem_config.base_seed),
        prefix_rng_tag=CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    )
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

    print(f"[continuous-beta-init-diagnostic] wrote {output_path}")
    return ContinuousInitDiagnosticResult(output_dir=output_path, paths=paths)


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
    selected = select_behavior_aware_candidates(
        pool,
        select_size=select_size,
        distance=str(behavior_config.distance),
        max_stop_ratio=float(behavior_config.max_stop_ratio),
        max_per_dominant_family=int(behavior_config.max_per_dominant_family),
        min_partial_candidates=int(behavior_config.min_partial_candidates),
        min_generate_candidates=int(behavior_config.min_generate_candidates),
    )
    selected_by_pool_key = {
        str(candidate["pool_candidate_key"]): (rank, candidate)
        for rank, candidate in enumerate(selected)
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
        CANDIDATE_DISTRIBUTION_COLUMNS,
        selected_candidate_rows,
    )
    paths["behavior_selected_distribution_summary"] = str(selected_summary_path)

    selected_by_suffix_path = output_path / "behavior_selected_by_suffix_len_summary.csv"
    _write_csv(
        selected_by_suffix_path,
        BY_SUFFIX_COLUMNS,
        selected_by_suffix_rows,
    )
    paths["behavior_selected_by_suffix_len_summary"] = str(selected_by_suffix_path)

    selection_config_path = output_path / "behavior_selection_config.json"
    save_json(
        _behavior_selection_config_payload(
            behavior_config,
            default_select_size=default_select_size,
            selected_count=len(selected),
        ),
        selection_config_path,
    )
    paths["behavior_selection_config"] = str(selection_config_path)

    _warn_if_behavior_soft_targets_missed(
        selected,
        behavior_config=behavior_config,
    )


def _build_behavior_candidate_pool(
    *,
    behavior_config: BehaviorAwareSelectionConfig,
    cem_config: PTSCEMConfig,
    continuous_config: PTSContinuousBetaCEMConfig,
    pts_config: Any,
    session_contexts: Sequence[PTSContinuousSessionContext],
    target_item: int,
) -> list[dict[str, Any]]:
    sample_plan = build_continuous_beta_initial_sample_plan(
        cem_config=cem_config,
        continuous_config=continuous_config,
        population_size=int(behavior_config.pool_size),
    )
    pool: list[dict[str, Any]] = []
    for pool_id, sample_spec in enumerate(sample_plan):
        pool_key = f"pool_cand{pool_id}"
        seed = int(cem_config.base_seed) + int(pool_id)
        policy = ContinuousBetaPolicy.from_vector(
            sample_spec.vector,
            parameter_bounds=continuous_config.parameter_bounds,
            parameterization=continuous_config.parameterization,
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
        pool.append(
            {
                "pool_candidate_key": pool_key,
                "pool_candidate_id": int(pool_id),
                "candidate_info": candidate_info,
                "records": records,
                "summary": summary,
                "behavior_vector": behavior_vector,
                "dominant_action_family": dominant_action_family,
                "dominant_action_ratio": float(dominant_action_ratio),
            }
        )
    return pool


def _validate_continuous_config(config: Config) -> None:
    pts_config = _require_pts_config(config)
    if pts_config.method != PTS_CONSTRUCTION_METHOD_CONTINUOUS_BETA_CEM_V1:
        raise ValueError(
            "Continuous beta init diagnostic requires "
            "attack.pts_construction.method='continuous_beta_cem_v1'."
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
        "method": PTS_CONSTRUCTION_METHOD_CONTINUOUS_BETA_CEM_V1,
        "parameterization": continuous_config.parameterization,
        "population_size": int(population_size),
        "candidate_count": int(candidate_count),
        "initialization_mode": continuous_config.initialization_mode,
        "parameter_bounds": {
            "min": float(continuous_config.parameter_bounds[0]),
            "max": float(continuous_config.parameter_bounds[1]),
        },
        "initial_std": float(continuous_config.initial_std),
        "min_std": float(continuous_config.min_std),
        "cem_base_seed": int(cem_config.base_seed),
        "candidate_seed_stride": int(cem_config.candidate_seed_stride),
        "shared_prefix_assignment_tag": CONTINUOUS_BETA_SHARED_PREFIX_TAG,
        "rounding_mode": "half_up",
        "materialize_generated_suffix": False,
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


def select_behavior_aware_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    select_size: int,
    distance: str = "l1",
    max_stop_ratio: float = 0.90,
    max_per_dominant_family: int = 3,
    min_partial_candidates: int = 2,
    min_generate_candidates: int = 2,
) -> list[Mapping[str, Any]]:
    if str(distance).strip().lower() != "l1":
        raise ValueError("behavior_distance currently supports only 'l1'.")
    if int(select_size) <= 0:
        raise ValueError("select_size must be positive.")
    pool = list(candidates)
    if not pool:
        return []
    selected: list[Mapping[str, Any]] = []

    for candidate in pool:
        if len(selected) >= int(select_size):
            break
        if not _is_behavior_covering_candidate(candidate):
            continue
        if not _passes_behavior_selection_caps(
            candidate,
            selected,
            max_stop_ratio=float(max_stop_ratio),
            max_per_dominant_family=int(max_per_dominant_family),
        ):
            continue
        if _min_behavior_distance(candidate, selected) <= 1e-12:
            continue
        selected.append(candidate)

    if not selected:
        selected.append(_most_balanced_candidate(pool))

    _fill_behavior_selection(
        pool,
        selected,
        select_size=int(select_size),
        strict_caps=True,
        max_stop_ratio=float(max_stop_ratio),
        max_per_dominant_family=int(max_per_dominant_family),
        min_partial_candidates=int(min_partial_candidates),
        min_generate_candidates=int(min_generate_candidates),
    )
    _fill_behavior_selection(
        pool,
        selected,
        select_size=int(select_size),
        strict_caps=False,
        max_stop_ratio=float(max_stop_ratio),
        max_per_dominant_family=int(max_per_dominant_family),
        min_partial_candidates=int(min_partial_candidates),
        min_generate_candidates=int(min_generate_candidates),
    )
    return selected[: int(select_size)]


def _fill_behavior_selection(
    pool: Sequence[Mapping[str, Any]],
    selected: list[Mapping[str, Any]],
    *,
    select_size: int,
    strict_caps: bool,
    max_stop_ratio: float,
    max_per_dominant_family: int,
    min_partial_candidates: int,
    min_generate_candidates: int,
) -> None:
    while len(selected) < int(select_size):
        remaining = [
            candidate
            for candidate in pool
            if str(candidate["pool_candidate_key"])
            not in {str(item["pool_candidate_key"]) for item in selected}
        ]
        if strict_caps:
            remaining = [
                candidate
                for candidate in remaining
                if _passes_behavior_selection_caps(
                    candidate,
                    selected,
                    max_stop_ratio=float(max_stop_ratio),
                    max_per_dominant_family=int(max_per_dominant_family),
                )
            ]
        if not remaining:
            return
        best = max(
            remaining,
            key=lambda candidate: (
                _behavior_selection_score(
                    candidate,
                    selected,
                    min_partial_candidates=int(min_partial_candidates),
                    min_generate_candidates=int(min_generate_candidates),
                ),
                -int(candidate.get("pool_candidate_id", 0)),
            ),
        )
        if strict_caps and _min_behavior_distance(best, selected) <= 1e-12:
            return
        selected.append(best)


def _behavior_selection_score(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    min_partial_candidates: int,
    min_generate_candidates: int,
) -> float:
    score = _min_behavior_distance(candidate, selected)
    if _partial_candidate_count(selected) < int(min_partial_candidates) and _is_partial_rich(
        candidate
    ):
        score += 0.25
    if _generate_candidate_count(selected) < int(min_generate_candidates) and _is_generate_rich(
        candidate
    ):
        score += 0.25
    if _is_mixed_behavior(candidate):
        score += 0.05
    return float(score)


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


def _passes_behavior_selection_caps(
    candidate: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    *,
    max_stop_ratio: float,
    max_per_dominant_family: int,
) -> bool:
    dominant_family = str(candidate["dominant_action_family"])
    dominant_count = sum(
        1
        for item in selected
        if str(item["dominant_action_family"]) == dominant_family
    )
    if dominant_count >= int(max_per_dominant_family):
        return False
    if (
        dominant_family == CONTINUOUS_ACTION_STOP
        and float(candidate["summary"]["continuous_stop_ratio"]) > float(max_stop_ratio)
    ):
        stop_heavy_count = sum(
            1
            for item in selected
            if str(item["dominant_action_family"]) == CONTINUOUS_ACTION_STOP
            and float(item["summary"]["continuous_stop_ratio"]) > float(max_stop_ratio)
        )
        if stop_heavy_count >= 1:
            return False
    return True


def _is_behavior_covering_candidate(candidate: Mapping[str, Any]) -> bool:
    return str(candidate["candidate_info"]["sample_origin"]) == (
        "continuous_beta_behavior_covering"
    )


def _is_partial_rich(candidate: Mapping[str, Any]) -> bool:
    summary = candidate["summary"]
    return (
        float(summary["continuous_partial_keep_suffix_ratio"])
        + float(summary["continuous_partial_generate_suffix_ratio"])
    ) >= 0.20


def _is_generate_rich(candidate: Mapping[str, Any]) -> bool:
    summary = candidate["summary"]
    return (
        float(summary["continuous_generate_full_suffix_ratio"])
        + float(summary["continuous_partial_generate_suffix_ratio"])
    ) >= 0.20 or float(summary["generate_ratio_non_stop"]) >= 0.20


def _is_mixed_behavior(candidate: Mapping[str, Any]) -> bool:
    summary = candidate["summary"]
    active = sum(
        1
        for action_name in ACTION_COLUMNS
        if float(summary[f"{action_name}_ratio"]) >= 0.10
    )
    return active >= 2


def _partial_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for candidate in candidates if _is_partial_rich(candidate))


def _generate_candidate_count(candidates: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for candidate in candidates if _is_generate_rich(candidate))


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
) -> dict[str, object]:
    summary = candidate["summary"]
    info = candidate["candidate_info"]
    selected_rank = "" if selected_entry is None else int(selected_entry[0])
    selected_key = "" if selected_entry is None else f"selected_cand{selected_entry[0]}"
    vector = [float(value) for value in candidate["behavior_vector"]]
    parameter_vector = [float(value) for value in info["parameter_vector"]]
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


def _behavior_selected_candidate_payload(
    candidate: Mapping[str, Any],
    *,
    selected_rank: int,
) -> dict[str, object]:
    info = candidate["candidate_info"]
    policy = info["policy"]
    if not isinstance(policy, ContinuousBetaPolicy):
        raise TypeError("candidate_info['policy'] must be a ContinuousBetaPolicy.")
    return {
        "selected_candidate_key": f"selected_cand{int(selected_rank)}",
        "pool_candidate_key": str(candidate["pool_candidate_key"]),
        "selected_rank": int(selected_rank),
        "sample_origin": str(info["sample_origin"]),
        "prototype_name": str(info.get("prototype_name", "")),
        "policy": policy.to_dict(),
        "parameter_vector": [float(value) for value in info["parameter_vector"]],
        "dominant_action_family": str(candidate["dominant_action_family"]),
        "behavior_vector": [float(value) for value in candidate["behavior_vector"]],
    }


def _selected_candidate_info(
    candidate: Mapping[str, Any],
    *,
    selected_rank: int,
) -> dict[str, object]:
    info = dict(candidate["candidate_info"])
    info["candidate_key"] = f"selected_cand{int(selected_rank)}"
    info["candidate_id"] = int(selected_rank)
    sample_metadata = dict(info.get("sample_metadata", {}))
    sample_metadata["pool_candidate_key"] = str(candidate["pool_candidate_key"])
    sample_metadata["selected_rank"] = int(selected_rank)
    info["sample_metadata"] = sample_metadata
    return info


def _behavior_selection_config_payload(
    behavior_config: BehaviorAwareSelectionConfig,
    *,
    default_select_size: int,
    selected_count: int,
) -> dict[str, object]:
    return {
        "behavior_aware_select": True,
        "behavior_pool_size": int(behavior_config.pool_size),
        "behavior_select_size": int(
            behavior_config.resolved_select_size(default_select_size)
        ),
        "behavior_selected_count": int(selected_count),
        "behavior_distance": str(behavior_config.distance),
        "behavior_max_stop_ratio": float(behavior_config.max_stop_ratio),
        "behavior_max_per_dominant_family": int(
            behavior_config.max_per_dominant_family
        ),
        "behavior_min_partial_candidates": int(
            behavior_config.min_partial_candidates
        ),
        "behavior_min_generate_candidates": int(
            behavior_config.min_generate_candidates
        ),
    }


def _warn_if_behavior_soft_targets_missed(
    selected: Sequence[Mapping[str, Any]],
    *,
    behavior_config: BehaviorAwareSelectionConfig,
) -> None:
    partial_count = _partial_candidate_count(selected)
    generate_count = _generate_candidate_count(selected)
    if partial_count < int(behavior_config.min_partial_candidates):
        print(
            "[continuous-beta-init-diagnostic] warning: behavior-aware selection "
            f"found only {partial_count} partial-rich candidates."
        )
    if generate_count < int(behavior_config.min_generate_candidates):
        print(
            "[continuous-beta-init-diagnostic] warning: behavior-aware selection "
            f"found only {generate_count} generate-rich candidates."
        )


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
    return {
        "candidate_key": str(candidate_info["candidate_key"]),
        "candidate_id": int(candidate_info["candidate_id"]),
        "sample_origin": str(candidate_info["sample_origin"]),
        "prototype_name": str(candidate_info.get("prototype_name", "")),
    }


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
        description="Inspect continuous_beta_cem_v1 iteration-0 initialization.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--sample-sessions", type=int, default=200)
    parser.add_argument("--include-rounding-variants", action="store_true")
    parser.add_argument("--behavior-aware-select", action="store_true")
    parser.add_argument("--behavior-pool-size", type=int, default=256)
    parser.add_argument("--behavior-select-size", type=int, default=None)
    parser.add_argument("--behavior-distance", default="l1")
    parser.add_argument("--behavior-max-stop-ratio", type=float, default=0.90)
    parser.add_argument("--behavior-max-per-dominant-family", type=int, default=3)
    parser.add_argument("--behavior-min-partial-candidates", type=int, default=2)
    parser.add_argument("--behavior-min-generate-candidates", type=int, default=2)
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
        behavior_pool_size=int(args.behavior_pool_size),
        behavior_select_size=args.behavior_select_size,
        behavior_distance=str(args.behavior_distance),
        behavior_max_stop_ratio=float(args.behavior_max_stop_ratio),
        behavior_max_per_dominant_family=int(args.behavior_max_per_dominant_family),
        behavior_min_partial_candidates=int(args.behavior_min_partial_candidates),
        behavior_min_generate_candidates=int(args.behavior_min_generate_candidates),
    )
    print(f"[continuous-beta-init-diagnostic] output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BEHAVIOR_POOL_SUMMARY_COLUMNS",
    "BehaviorAwareSelectionConfig",
    "ContinuousInitDiagnosticResult",
    "build_behavior_vector",
    "run_continuous_beta_init_diagnostic",
    "select_behavior_aware_candidates",
    "main",
]
