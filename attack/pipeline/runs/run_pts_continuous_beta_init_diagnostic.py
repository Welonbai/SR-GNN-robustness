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
    CONTINUOUS_BETA_PARAMETER_NAMES,
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
    *CONTINUOUS_BETA_PARAMETER_NAMES,
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


@dataclass(frozen=True)
class ContinuousInitDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]


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

    print(f"[continuous-beta-init-diagnostic] wrote {output_path}")
    return ContinuousInitDiagnosticResult(output_dir=output_path, paths=paths)


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


def _parameter_row(candidate_info: Mapping[str, object]) -> dict[str, float]:
    vector = [float(value) for value in candidate_info["parameter_vector"]]
    return {
        name: float(vector[index])
        for index, name in enumerate(CONTINUOUS_BETA_PARAMETER_NAMES)
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
    args = parser.parse_args(argv)

    config = load_config(args.config)
    result = run_continuous_beta_init_diagnostic(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        sample_sessions=args.sample_sessions,
        include_rounding_variants=bool(args.include_rounding_variants),
    )
    print(f"[continuous-beta-init-diagnostic] output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ContinuousInitDiagnosticResult",
    "run_continuous_beta_init_diagnostic",
    "main",
]
