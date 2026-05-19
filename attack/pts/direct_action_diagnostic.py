from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import Config
from attack.common.paths import shared_root
from attack.pts.cem import candidate_key as cem_candidate_key
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_FAMILIES,
    DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX,
    DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX,
    DIRECT_ACTION_FAMILY_STOP,
    DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_POLICY_LINEAR_LENGTH,
    DIRECT_ACTION_POLICY_MLP_H2,
    DirectAction,
    action_probability_payload,
    deterministic_direct_action_seed,
    direct_action_consume_ratio,
    direct_action_entropy,
    direct_action_family_probabilities,
    direct_action_generated_length,
    direct_action_length_feature,
    direct_action_policy_payload,
    enumerate_valid_direct_actions,
    map_direct_action_to_family,
    normalize_direct_action_length_feature_mode,
    normalize_direct_action_policy_variant,
    parameter_count_for_policy,
    parameter_names_for_policy,
    sample_direct_action_categorical,
    sample_theta,
    score_direct_action,
    stable_softmax,
    uniform_family_baseline,
)
from attack.pts.prefix_selector import select_internal_uniform_anchor


DIRECT_ACTION_SAMPLE_TAG = "direct_action_sample"
DIRECT_ACTION_PREFIX_TAG = "direct_action_shared_prefix"
DIRECT_ACTION_OUTPUT_SCHEMA_VERSION = "direct_action_init_diagnostic_v1"

FAMILY_COLUMN_PREFIXES = {
    DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX: "keep_full",
    DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX: "generate_full",
    DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX: "partial_keep",
    DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX: "partial_generate",
    DIRECT_ACTION_FAMILY_STOP: "stop",
}

OVERALL_COLUMNS = (
    "candidate_key",
    "policy_variant",
    "initial_std",
    "num_sessions",
    "expected_keep_full_ratio",
    "expected_generate_full_ratio",
    "expected_partial_keep_ratio",
    "expected_partial_generate_ratio",
    "expected_stop_ratio",
    "sampled_keep_full_ratio",
    "sampled_generate_full_ratio",
    "sampled_partial_keep_ratio",
    "sampled_partial_generate_ratio",
    "sampled_stop_ratio",
    "expected_consume_ratio_mean",
    "sampled_consume_ratio_mean",
    "entropy_mean",
    "max_action_probability_mean",
    "max_action_probability_p90",
)

BY_SUFFIX_LEN_COLUMNS = (
    "candidate_key",
    "policy_variant",
    "initial_std",
    "residual_suffix_len",
    "session_count",
    "expected_keep_full_ratio",
    "expected_generate_full_ratio",
    "expected_partial_keep_ratio",
    "expected_partial_generate_ratio",
    "expected_stop_ratio",
    "sampled_keep_full_ratio",
    "sampled_generate_full_ratio",
    "sampled_partial_keep_ratio",
    "sampled_partial_generate_ratio",
    "sampled_stop_ratio",
    "expected_consume_ratio_mean",
    "sampled_consume_ratio_mean",
    "entropy_mean",
    "max_action_probability_mean",
)

BY_SUFFIX_GROUP_COLUMNS = (
    "candidate_key",
    "policy_variant",
    "initial_std",
    "suffix_group",
    "session_count",
    "expected_keep_full_ratio",
    "expected_generate_full_ratio",
    "expected_partial_keep_ratio",
    "expected_partial_generate_ratio",
    "expected_stop_ratio",
    "sampled_keep_full_ratio",
    "sampled_generate_full_ratio",
    "sampled_partial_keep_ratio",
    "sampled_partial_generate_ratio",
    "sampled_stop_ratio",
    "expected_consume_ratio_mean",
    "sampled_consume_ratio_mean",
    "entropy_mean",
    "max_action_probability_mean",
)

UNIFORM_BASELINE_COLUMNS = (
    "residual_suffix_len",
    "valid_action_count",
    "uniform_keep_full_ratio",
    "uniform_generate_full_ratio",
    "uniform_partial_keep_ratio",
    "uniform_partial_generate_ratio",
    "uniform_stop_ratio",
)

BIAS_COLUMNS = (
    "candidate_key",
    "policy_variant",
    "initial_std",
    "residual_suffix_len",
    "keep_full_minus_uniform",
    "generate_full_minus_uniform",
    "partial_keep_minus_uniform",
    "partial_generate_minus_uniform",
    "stop_minus_uniform",
)

PAIRWISE_COLUMNS = (
    "policy_variant",
    "initial_std",
    "candidate_key_a",
    "candidate_key_b",
    "overall_family_l1",
    "by_suffix_len_l1",
)

ELITE_SELECT_MODE_STOP_HEAVY = "stop_heavy"
ELITE_SELECT_MODE_GENERATE_ORIENTED = "generate_oriented"
ELITE_SELECT_MODE_PARTIAL_GENERATE_ORIENTED = "partial_generate_oriented"
ELITE_SELECT_MODE_KEEP_ORIENTED = "keep_oriented"
ELITE_SELECT_MODE_MIXED = "mixed"
ELITE_SELECT_MODE_DIVERSE = "diverse"
ELITE_SELECT_MODES = (
    ELITE_SELECT_MODE_STOP_HEAVY,
    ELITE_SELECT_MODE_GENERATE_ORIENTED,
    ELITE_SELECT_MODE_PARTIAL_GENERATE_ORIENTED,
    ELITE_SELECT_MODE_KEEP_ORIENTED,
    ELITE_SELECT_MODE_MIXED,
    ELITE_SELECT_MODE_DIVERSE,
)

ELITE_SELECTION_COLUMNS = (
    "policy_variant",
    "initial_std",
    "elite_select_mode",
    "elite_rank",
    "candidate_key",
    "selection_score",
    "expected_keep_full_ratio",
    "expected_generate_full_ratio",
    "expected_partial_keep_ratio",
    "expected_partial_generate_ratio",
    "expected_stop_ratio",
    "entropy_mean",
    "max_action_probability_mean",
)

ELITE_RESAMPLED_OVERALL_COLUMNS = (
    *OVERALL_COLUMNS,
    "source_initial_std",
    "elite_select_mode",
    "resample_index",
    "param_l2_to_elite_mean",
    "param_mean_abs_to_elite_mean",
    "param_avg_l2_to_elite_thetas",
    "behavior_l1_to_elite_mean_overall",
    "behavior_l1_to_nearest_elite_overall",
    "behavior_l1_to_initial_population_mean_overall",
    "behavior_l1_to_elite_mean_by_len",
    "behavior_l1_to_nearest_elite_by_len",
    "behavior_l1_to_initial_population_mean_by_len",
)

ELITE_RESAMPLED_BY_SUFFIX_LEN_COLUMNS = (
    *BY_SUFFIX_LEN_COLUMNS,
    "elite_select_mode",
    "resample_index",
)

ELITE_RESAMPLED_BY_SUFFIX_GROUP_COLUMNS = (
    *BY_SUFFIX_GROUP_COLUMNS,
    "elite_select_mode",
    "resample_index",
)


@dataclass(frozen=True)
class DirectActionSessionContext:
    fake_session_index: int
    template_session: list[int]
    anchor_position: int
    prefix: list[int]
    residual_suffix: list[int]

    @property
    def residual_suffix_len(self) -> int:
        return int(len(self.residual_suffix))


@dataclass(frozen=True)
class DirectActionInitDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]


def build_direct_action_session_contexts(
    *,
    template_sessions: Sequence[Sequence[int]],
    base_seed: int,
    prefix_rng_tag: str = DIRECT_ACTION_PREFIX_TAG,
    prefix_seed_scope: str = "target_independent",
) -> tuple[DirectActionSessionContext, ...]:
    if str(prefix_seed_scope).strip().lower() != "target_independent":
        raise ValueError("direct-action diagnostic supports only target_independent prefix scope.")
    contexts: list[DirectActionSessionContext] = []
    for index, session in enumerate(template_sessions):
        template = [int(item) for item in session]
        if len(template) < 2:
            raise ValueError(
                "Direct-action diagnostic requires fake sessions with length >= 2 "
                "so prefix and residual_suffix are both non-empty."
            )
        seed = _stable_seed(
            int(base_seed),
            str(prefix_rng_tag),
            str(prefix_seed_scope),
            int(index),
        )
        anchor_position = select_internal_uniform_anchor(
            len(template),
            rng=random.Random(seed),
        )
        prefix = template[:anchor_position]
        residual_suffix = template[anchor_position:]
        if not prefix or not residual_suffix:
            raise ValueError(
                "Internal prefix selection must produce non-empty prefix and residual_suffix."
            )
        contexts.append(
            DirectActionSessionContext(
                fake_session_index=int(index),
                template_session=template,
                anchor_position=int(anchor_position),
                prefix=prefix,
                residual_suffix=residual_suffix,
            )
        )
    if not contexts:
        raise ValueError("Diagnostic requires at least one fake session.")
    return tuple(contexts)


def run_direct_action_init_diagnostic(
    *,
    config: Config,
    fake_sessions: Sequence[Sequence[int]],
    fake_sessions_path: str | Path | None,
    config_path: str | Path | None = None,
    policy_variant: str = DIRECT_ACTION_POLICY_MLP_H2,
    length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    initial_stds: Sequence[float] = (0.5, 1.0, 1.5),
    num_candidates: int = 16,
    sample_sessions: int = 200,
    output_dir: str | Path | None = None,
    seed: int | None = None,
    prefix_seed_scope: str = "target_independent",
    include_elite_centered_diagnostic: bool = False,
    elite_select_mode: str = ELITE_SELECT_MODE_DIVERSE,
    elite_count: int = 4,
    elite_resample_count: int = 8,
    elite_min_std: float = 0.25,
    elite_std_scale: float = 1.0,
    elite_centered_seed: int | None = None,
) -> DirectActionInitDiagnosticResult:
    policy = normalize_direct_action_policy_variant(policy_variant)
    length_feature = normalize_direct_action_length_feature_mode(length_feature_mode)
    stds = [float(value) for value in initial_stds]
    if not stds:
        raise ValueError("initial_stds must not be empty.")
    if any(value < 0.0 for value in stds):
        raise ValueError("initial_stds must be non-negative.")
    candidate_count = int(num_candidates)
    if candidate_count <= 0:
        raise ValueError("num_candidates must be positive.")
    sample_limit = int(sample_sessions)
    if sample_limit < 0:
        raise ValueError("sample_sessions must be non-negative.")
    elite_mode = normalize_elite_select_mode(elite_select_mode)
    if int(elite_count) <= 0:
        raise ValueError("elite_count must be positive.")
    if int(elite_resample_count) <= 0:
        raise ValueError("elite_resample_count must be positive.")
    if float(elite_min_std) < 0.0:
        raise ValueError("elite_min_std must be non-negative.")
    if float(elite_std_scale) < 0.0:
        raise ValueError("elite_std_scale must be non-negative.")
    base_seed = int(config.seeds.position_opt_seed if seed is None else seed)
    resolved_elite_seed = (
        _stable_seed(base_seed, "direct_action_elite_centered")
        if elite_centered_seed is None
        else int(elite_centered_seed)
    )
    templates = [[int(item) for item in session] for session in fake_sessions]
    contexts = build_direct_action_session_contexts(
        template_sessions=templates,
        base_seed=base_seed,
        prefix_rng_tag=DIRECT_ACTION_PREFIX_TAG,
        prefix_seed_scope=prefix_seed_scope,
    )
    max_residual_suffix_len = max(int(context.residual_suffix_len) for context in contexts)
    residual_suffix_lengths = [int(context.residual_suffix_len) for context in contexts]
    mean_residual_suffix_len = _mean(residual_suffix_lengths)
    std_residual_suffix_len = _std(residual_suffix_lengths)

    fake_identity = _fake_sessions_identity(fake_sessions_path, templates)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(
            config=config,
            fake_sessions_identity=fake_identity,
            policy_variant=policy,
            length_feature_mode=length_feature,
            initial_stds=stds,
            num_candidates=candidate_count,
            seed=base_seed,
            prefix_seed_scope=prefix_seed_scope,
            include_elite_centered_diagnostic=bool(include_elite_centered_diagnostic),
            elite_select_mode=elite_mode,
            elite_count=int(elite_count),
            elite_resample_count=int(elite_resample_count),
            elite_min_std=float(elite_min_std),
            elite_std_scale=float(elite_std_scale),
            elite_centered_seed=resolved_elite_seed,
        )
    )
    output_path.mkdir(parents=True, exist_ok=True)

    initial_candidates: list[dict[str, object]] = []
    all_records_by_candidate: dict[str, list[dict[str, object]]] = {}
    candidate_metadata: dict[str, dict[str, object]] = {}
    session_sample_rows: list[dict[str, object]] = []

    for std_index, initial_std in enumerate(stds):
        for candidate_id in range(candidate_count):
            key = _direct_action_candidate_key(
                initial_std=initial_std,
                candidate_id=candidate_id,
            )
            theta_seed = deterministic_direct_action_seed(
                base_seed=base_seed,
                policy_variant=policy,
                initial_std=initial_std,
                candidate_key=key,
                session_index=candidate_id,
                tag="direct_action_theta",
            )
            theta = sample_theta(
                policy_variant=policy,
                initial_std=initial_std,
                seed=theta_seed,
            )
            metadata = {
                "candidate_key": key,
                "candidate_id": int(candidate_id),
                "std_index": int(std_index),
                "policy_variant": policy,
                "initial_std": float(initial_std),
                "parameter_vector": [float(value) for value in theta],
                "parameter_names": list(parameter_names_for_policy(policy)),
                "parameter_count": int(parameter_count_for_policy(policy)),
            }
            candidate_metadata[key] = metadata
            initial_candidates.append(
                {
                    **metadata,
                    "policy": direct_action_policy_payload(
                        policy_variant=policy,
                        theta=theta,
                    ),
                }
            )
            records = [
                _evaluate_candidate_session(
                    context=context,
                    policy_variant=policy,
                    length_feature_mode=length_feature,
                    max_residual_suffix_len=max_residual_suffix_len,
                    mean_residual_suffix_len=mean_residual_suffix_len,
                    std_residual_suffix_len=std_residual_suffix_len,
                    initial_std=initial_std,
                    candidate_key=key,
                    theta=theta,
                    base_seed=base_seed,
                )
                for context in contexts
            ]
            all_records_by_candidate[key] = records
            for record in records[:sample_limit]:
                session_sample_rows.append(_session_sample_row(record))

    overall_rows = [
        _summary_row(candidate_metadata[key], records, include_p90=True)
        for key, records in all_records_by_candidate.items()
    ]
    by_suffix_len_rows = _grouped_summary_rows(
        candidate_metadata=candidate_metadata,
        records_by_candidate=all_records_by_candidate,
        group_field="residual_suffix_len",
        output_field="residual_suffix_len",
    )
    by_suffix_group_rows = _grouped_summary_rows(
        candidate_metadata=candidate_metadata,
        records_by_candidate=all_records_by_candidate,
        group_field="suffix_group",
        output_field="suffix_group",
    )
    uniform_rows = _uniform_baseline_rows(contexts)
    bias_rows = _bias_vs_uniform_rows(by_suffix_len_rows)
    pairwise_rows = _pairwise_behavior_distance_rows(
        candidate_metadata=candidate_metadata,
        overall_rows=overall_rows,
        by_suffix_len_rows=by_suffix_len_rows,
    )

    paths: dict[str, str] = {}
    diagnostic_config_path = output_path / "diagnostic_config.json"
    save_json(
        _diagnostic_config_payload(
            config=config,
            config_path=config_path,
            fake_sessions_path=fake_sessions_path,
            fake_sessions_identity=fake_identity,
            policy_variant=policy,
            length_feature_mode=length_feature,
            max_residual_suffix_len=max_residual_suffix_len,
            mean_residual_suffix_len=mean_residual_suffix_len,
            std_residual_suffix_len=std_residual_suffix_len,
            initial_stds=stds,
            num_candidates=candidate_count,
            sample_sessions=sample_limit,
            seed=base_seed,
            fake_session_count=len(contexts),
            prefix_seed_scope=prefix_seed_scope,
        ),
        diagnostic_config_path,
    )
    paths["diagnostic_config"] = str(diagnostic_config_path)

    initial_candidates_path = output_path / "initial_candidates.json"
    save_json(initial_candidates, initial_candidates_path)
    paths["initial_candidates"] = str(initial_candidates_path)

    paths["candidate_overall_summary"] = str(output_path / "candidate_overall_summary.csv")
    _write_csv(output_path / "candidate_overall_summary.csv", OVERALL_COLUMNS, overall_rows)
    paths["candidate_by_suffix_len_summary"] = str(
        output_path / "candidate_by_suffix_len_summary.csv"
    )
    _write_csv(
        output_path / "candidate_by_suffix_len_summary.csv",
        BY_SUFFIX_LEN_COLUMNS,
        by_suffix_len_rows,
    )
    paths["candidate_by_suffix_group_summary"] = str(
        output_path / "candidate_by_suffix_group_summary.csv"
    )
    _write_csv(
        output_path / "candidate_by_suffix_group_summary.csv",
        BY_SUFFIX_GROUP_COLUMNS,
        by_suffix_group_rows,
    )
    paths["uniform_baseline_by_suffix_len"] = str(
        output_path / "uniform_baseline_by_suffix_len.csv"
    )
    _write_csv(
        output_path / "uniform_baseline_by_suffix_len.csv",
        UNIFORM_BASELINE_COLUMNS,
        uniform_rows,
    )
    paths["bias_vs_uniform_summary"] = str(output_path / "bias_vs_uniform_summary.csv")
    _write_csv(output_path / "bias_vs_uniform_summary.csv", BIAS_COLUMNS, bias_rows)
    paths["pairwise_behavior_distance"] = str(
        output_path / "pairwise_behavior_distance.csv"
    )
    _write_csv(
        output_path / "pairwise_behavior_distance.csv",
        PAIRWISE_COLUMNS,
        pairwise_rows,
    )
    paths["session_samples"] = str(output_path / "session_samples.jsonl")
    _write_jsonl(output_path / "session_samples.jsonl", session_sample_rows)

    if bool(include_elite_centered_diagnostic):
        elite_paths = _write_elite_centered_artifacts(
            output_path=output_path,
            config=config,
            policy_variant=policy,
            length_feature_mode=length_feature,
            initial_stds=stds,
            sample_sessions=sample_limit,
            elite_select_mode=elite_mode,
            elite_count=int(elite_count),
            elite_resample_count=int(elite_resample_count),
            elite_min_std=float(elite_min_std),
            elite_std_scale=float(elite_std_scale),
            elite_centered_seed=resolved_elite_seed,
            contexts=contexts,
            max_residual_suffix_len=max_residual_suffix_len,
            mean_residual_suffix_len=mean_residual_suffix_len,
            std_residual_suffix_len=std_residual_suffix_len,
            base_seed=base_seed,
            candidate_metadata=candidate_metadata,
            initial_candidates=initial_candidates,
            initial_records_by_candidate=all_records_by_candidate,
            overall_rows=overall_rows,
            by_suffix_len_rows=by_suffix_len_rows,
            by_suffix_group_rows=by_suffix_group_rows,
        )
        paths.update(elite_paths)

    return DirectActionInitDiagnosticResult(output_dir=output_path, paths=paths)


def normalize_elite_select_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in ELITE_SELECT_MODES:
        raise ValueError(f"Unsupported elite_select_mode: {mode}")
    return value


def select_behavior_elites(
    *,
    candidate_keys: Sequence[str],
    elite_select_mode: str,
    elite_count: int,
    overall_rows: Sequence[Mapping[str, object]],
    by_suffix_len_rows: Sequence[Mapping[str, object]] | None = None,
    by_suffix_group_rows: Sequence[Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    mode = normalize_elite_select_mode(elite_select_mode)
    requested = int(elite_count)
    if requested <= 0:
        raise ValueError("elite_count must be positive.")
    keys = [str(key) for key in candidate_keys]
    if not keys:
        raise ValueError("candidate_keys must not be empty.")
    overall_by_key = {str(row["candidate_key"]): row for row in overall_rows}
    missing = [key for key in keys if key not in overall_by_key]
    if missing:
        raise KeyError(f"Missing overall summary rows for: {', '.join(missing)}")
    if mode == ELITE_SELECT_MODE_DIVERSE:
        return _select_diverse_behavior_elites(
            candidate_keys=keys,
            elite_count=requested,
            overall_by_key=overall_by_key,
            by_suffix_len_rows=by_suffix_len_rows or (),
        )

    scored = [
        {
            "candidate_key": key,
            "selection_score": _elite_selection_score(
                key,
                mode=mode,
                overall_by_key=overall_by_key,
                by_suffix_group_rows=by_suffix_group_rows or (),
            ),
        }
        for key in keys
    ]
    scored.sort(key=lambda item: (-float(item["selection_score"]), str(item["candidate_key"])))
    return [
        {**item, "elite_rank": int(index)}
        for index, item in enumerate(scored[:requested])
    ]


def compute_elite_gaussian(
    elite_theta_vectors: Sequence[Sequence[float]],
    *,
    elite_min_std: float,
    elite_std_scale: float,
) -> dict[str, list[float]]:
    vectors = [[float(value) for value in vector] for vector in elite_theta_vectors]
    if not vectors:
        raise ValueError("elite_theta_vectors must not be empty.")
    width = len(vectors[0])
    if width <= 0:
        raise ValueError("elite theta vectors must not be empty.")
    if any(len(vector) != width for vector in vectors):
        raise ValueError("elite theta vectors must have matching lengths.")
    min_std = float(elite_min_std)
    std_scale = float(elite_std_scale)
    if min_std < 0.0:
        raise ValueError("elite_min_std must be non-negative.")
    if std_scale < 0.0:
        raise ValueError("elite_std_scale must be non-negative.")
    means = [
        _mean(vector[index] for vector in vectors)
        for index in range(width)
    ]
    stds = [
        _std(vector[index] for vector in vectors)
        for index in range(width)
    ]
    resample_stds = [
        max(float(std) * std_scale, min_std)
        for std in stds
    ]
    return {
        "elite_mean": means,
        "elite_std": stds,
        "resample_std": resample_stds,
    }


def sample_elite_centered_candidates(
    *,
    policy_variant: str,
    initial_std: float,
    elite_select_mode: str,
    elite_resample_count: int,
    elite_mean: Sequence[float],
    resample_std: Sequence[float],
    seed: int,
) -> list[dict[str, object]]:
    policy = normalize_direct_action_policy_variant(policy_variant)
    mode = normalize_elite_select_mode(elite_select_mode)
    count = int(elite_resample_count)
    if count <= 0:
        raise ValueError("elite_resample_count must be positive.")
    means = [float(value) for value in elite_mean]
    stds = [float(value) for value in resample_std]
    if len(means) != len(stds):
        raise ValueError("elite_mean and resample_std must have matching lengths.")
    expected_count = parameter_count_for_policy(policy)
    if len(means) != expected_count:
        raise ValueError(f"{policy} requires {expected_count} parameters.")
    rows: list[dict[str, object]] = []
    names = list(parameter_names_for_policy(policy))
    for index in range(count):
        key = _elite_resampled_candidate_key(
            elite_select_mode=mode,
            initial_std=float(initial_std),
            resample_index=index,
        )
        rng = random.Random(_stable_seed(seed, policy, mode, initial_std, index))
        vector = [
            float(rng.gauss(float(mean), float(std)))
            for mean, std in zip(means, stds)
        ]
        rows.append(
            {
                "candidate_key": key,
                "source_policy_variant": policy,
                "policy_variant": policy,
                "initial_std": float(initial_std),
                "source_initial_std": float(initial_std),
                "elite_select_mode": mode,
                "resample_index": int(index),
                "parameter_vector": vector,
                "parameter_names": names,
                "parameter_count": int(expected_count),
            }
        )
    return rows


def _elite_selection_score(
    candidate_key: str,
    *,
    mode: str,
    overall_by_key: Mapping[str, Mapping[str, object]],
    by_suffix_group_rows: Sequence[Mapping[str, object]],
) -> float:
    row = overall_by_key[str(candidate_key)]
    if mode == ELITE_SELECT_MODE_STOP_HEAVY:
        return float(row["expected_stop_ratio"])
    if mode == ELITE_SELECT_MODE_GENERATE_ORIENTED:
        return float(row["expected_generate_full_ratio"]) + float(
            row["expected_partial_generate_ratio"]
        )
    if mode == ELITE_SELECT_MODE_PARTIAL_GENERATE_ORIENTED:
        group_rows = [
            item
            for item in by_suffix_group_rows
            if str(item["candidate_key"]) == str(candidate_key)
            and str(item["suffix_group"]) in {"suffix_2", "suffix_3plus"}
        ]
        if group_rows:
            total = sum(int(item["session_count"]) for item in group_rows)
            if total > 0:
                return sum(
                    float(item["expected_partial_generate_ratio"])
                    * float(int(item["session_count"]))
                    for item in group_rows
                ) / float(total)
        return float(row["expected_partial_generate_ratio"])
    if mode == ELITE_SELECT_MODE_KEEP_ORIENTED:
        return float(row["expected_keep_full_ratio"]) + float(
            row["expected_partial_keep_ratio"]
        )
    if mode == ELITE_SELECT_MODE_MIXED:
        return float(row["entropy_mean"]) - float(row["max_action_probability_mean"])
    raise ValueError(f"Unsupported score mode for ranking: {mode}")


def _select_diverse_behavior_elites(
    *,
    candidate_keys: Sequence[str],
    elite_count: int,
    overall_by_key: Mapping[str, Mapping[str, object]],
    by_suffix_len_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    keys = [str(key) for key in candidate_keys]
    if not by_suffix_len_rows:
        vectors = {key: _overall_family_vector(overall_by_key[key]) for key in keys}
    else:
        suffix_lengths = sorted(
            {
                int(row["residual_suffix_len"])
                for row in by_suffix_len_rows
                if str(row["candidate_key"]) in set(keys)
            }
        )
        lookup = {
            (str(row["candidate_key"]), int(row["residual_suffix_len"])): row
            for row in by_suffix_len_rows
            if str(row["candidate_key"]) in set(keys)
        }
        vectors = {
            key: _by_suffix_behavior_vector(key, suffix_lengths, lookup)
            for key in keys
        }
    mean_vector = _mean_vector(list(vectors.values()))
    first_key = min(keys, key=lambda key: (_l1(vectors[key], mean_vector), key))
    selected = [
        {
            "candidate_key": first_key,
            "selection_score": -_l1(vectors[first_key], mean_vector),
            "elite_rank": 0,
        }
    ]
    selected_keys = {first_key}
    while len(selected) < min(int(elite_count), len(keys)):
        best_key = max(
            (key for key in keys if key not in selected_keys),
            key=lambda key: (
                min(_l1(vectors[key], vectors[item]) for item in selected_keys),
                key,
            ),
        )
        score = min(_l1(vectors[best_key], vectors[item]) for item in selected_keys)
        selected.append(
            {
                "candidate_key": best_key,
                "selection_score": float(score),
                "elite_rank": int(len(selected)),
            }
        )
        selected_keys.add(best_key)
    return selected


def _elite_resampled_overall_row(
    candidate_info: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    distance_context: Mapping[str, object],
    suffix_lengths: Sequence[int],
    by_suffix_len_rows: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    row = _summary_row(candidate_info, records, include_p90=True)
    theta = [float(value) for value in candidate_info["parameter_vector"]]
    elite_mean_theta = [float(value) for value in distance_context["elite_mean_theta"]]
    elite_thetas = [
        [float(value) for value in vector]
        for vector in distance_context["elite_thetas"]
    ]
    overall_vector = _overall_family_vector(row)
    row.update(
        {
            "source_initial_std": float(candidate_info["source_initial_std"]),
            "elite_select_mode": str(candidate_info["elite_select_mode"]),
            "resample_index": int(candidate_info["resample_index"]),
            "param_l2_to_elite_mean": _euclidean(theta, elite_mean_theta),
            "param_mean_abs_to_elite_mean": _mean(
                abs(left - right)
                for left, right in zip(theta, elite_mean_theta)
            ),
            "param_avg_l2_to_elite_thetas": _mean(
                _euclidean(theta, elite_theta)
                for elite_theta in elite_thetas
            ),
            "behavior_l1_to_elite_mean_overall": _l1(
                overall_vector,
                distance_context["elite_mean_overall"],
            ),
            "behavior_l1_to_nearest_elite_overall": min(
                _l1(overall_vector, elite_vector)
                for elite_vector in distance_context["elite_overall_vectors"]
            ),
            "behavior_l1_to_initial_population_mean_overall": _l1(
                overall_vector,
                distance_context["population_mean_overall"],
            ),
            "behavior_l1_to_elite_mean_by_len": 0.0,
            "behavior_l1_to_nearest_elite_by_len": 0.0,
            "behavior_l1_to_initial_population_mean_by_len": 0.0,
        }
    )
    return row


def _attach_resampled_metadata(
    rows: Sequence[Mapping[str, object]],
    metadata_by_key: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    attached: list[dict[str, object]] = []
    for row in rows:
        metadata = metadata_by_key[str(row["candidate_key"])]
        attached.append(
            {
                **dict(row),
                "elite_select_mode": str(metadata["elite_select_mode"]),
                "resample_index": int(metadata["resample_index"]),
            }
        )
    return attached


def _write_elite_centered_artifacts(
    *,
    output_path: Path,
    config: Config,
    policy_variant: str,
    length_feature_mode: str,
    initial_stds: Sequence[float],
    sample_sessions: int,
    elite_select_mode: str,
    elite_count: int,
    elite_resample_count: int,
    elite_min_std: float,
    elite_std_scale: float,
    elite_centered_seed: int,
    contexts: Sequence[DirectActionSessionContext],
    max_residual_suffix_len: int,
    mean_residual_suffix_len: float,
    std_residual_suffix_len: float,
    base_seed: int,
    candidate_metadata: Mapping[str, Mapping[str, object]],
    initial_candidates: Sequence[Mapping[str, object]],
    initial_records_by_candidate: Mapping[str, Sequence[Mapping[str, object]]],
    overall_rows: Sequence[Mapping[str, object]],
    by_suffix_len_rows: Sequence[Mapping[str, object]],
    by_suffix_group_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    mode = normalize_elite_select_mode(elite_select_mode)
    policy = normalize_direct_action_policy_variant(policy_variant)
    candidate_payload_by_key = {
        str(candidate["candidate_key"]): candidate
        for candidate in initial_candidates
    }
    overall_by_key = {str(row["candidate_key"]): row for row in overall_rows}
    by_len_lookup = {
        (str(row["candidate_key"]), int(row["residual_suffix_len"])): row
        for row in by_suffix_len_rows
    }
    suffix_lengths = sorted({int(row["residual_suffix_len"]) for row in by_suffix_len_rows})

    elite_selection_rows: list[dict[str, object]] = []
    distribution_parameters: list[dict[str, object]] = []
    resampled_candidates: list[dict[str, object]] = []
    resampled_records_by_candidate: dict[str, list[dict[str, object]]] = {}
    resampled_metadata: dict[str, dict[str, object]] = {}
    resampled_sample_rows: list[dict[str, object]] = []
    distance_context_by_key: dict[str, dict[str, object]] = {}

    for initial_std in initial_stds:
        group_keys = [
            key
            for key, metadata in candidate_metadata.items()
            if float(metadata["initial_std"]) == float(initial_std)
        ]
        if len(group_keys) < int(elite_count):
            raise ValueError(
                "elite_count must not exceed initialized candidate count per initial_std."
            )
        selected = select_behavior_elites(
            candidate_keys=group_keys,
            elite_select_mode=mode,
            elite_count=int(elite_count),
            overall_rows=overall_rows,
            by_suffix_len_rows=by_suffix_len_rows,
            by_suffix_group_rows=by_suffix_group_rows,
        )
        elite_keys = [str(item["candidate_key"]) for item in selected]
        for item in selected:
            row = dict(overall_by_key[str(item["candidate_key"])])
            elite_selection_rows.append(
                {
                    "policy_variant": policy,
                    "initial_std": float(initial_std),
                    "elite_select_mode": mode,
                    "elite_rank": int(item["elite_rank"]),
                    "candidate_key": str(item["candidate_key"]),
                    "selection_score": float(item["selection_score"]),
                    "expected_keep_full_ratio": float(row["expected_keep_full_ratio"]),
                    "expected_generate_full_ratio": float(row["expected_generate_full_ratio"]),
                    "expected_partial_keep_ratio": float(row["expected_partial_keep_ratio"]),
                    "expected_partial_generate_ratio": float(
                        row["expected_partial_generate_ratio"]
                    ),
                    "expected_stop_ratio": float(row["expected_stop_ratio"]),
                    "entropy_mean": float(row["entropy_mean"]),
                    "max_action_probability_mean": float(row["max_action_probability_mean"]),
                }
            )
        elite_thetas = [
            [
                float(value)
                for value in candidate_payload_by_key[key]["parameter_vector"]
            ]
            for key in elite_keys
        ]
        gaussian = compute_elite_gaussian(
            elite_thetas,
            elite_min_std=float(elite_min_std),
            elite_std_scale=float(elite_std_scale),
        )
        distribution_parameters.append(
            {
                "policy_variant": policy,
                "initial_std": float(initial_std),
                "elite_select_mode": mode,
                "elite_candidate_keys": elite_keys,
                "parameter_names": list(parameter_names_for_policy(policy)),
                "elite_mean": gaussian["elite_mean"],
                "elite_std": gaussian["elite_std"],
                "resample_std": gaussian["resample_std"],
                "elite_min_std": float(elite_min_std),
                "elite_std_scale": float(elite_std_scale),
            }
        )
        sampled = sample_elite_centered_candidates(
            policy_variant=policy,
            initial_std=float(initial_std),
            elite_select_mode=mode,
            elite_resample_count=int(elite_resample_count),
            elite_mean=gaussian["elite_mean"],
            resample_std=gaussian["resample_std"],
            seed=int(elite_centered_seed),
        )
        resampled_candidates.extend(sampled)

        elite_overall_vectors = [_overall_family_vector(overall_by_key[key]) for key in elite_keys]
        elite_by_len_vectors = [
            _by_suffix_behavior_vector(key, suffix_lengths, by_len_lookup)
            for key in elite_keys
        ]
        population_overall_vectors = [
            _overall_family_vector(overall_by_key[key])
            for key in group_keys
        ]
        population_by_len_vectors = [
            _by_suffix_behavior_vector(key, suffix_lengths, by_len_lookup)
            for key in group_keys
        ]
        elite_mean_overall = _mean_vector(elite_overall_vectors)
        elite_mean_by_len = _mean_vector(elite_by_len_vectors)
        population_mean_overall = _mean_vector(population_overall_vectors)
        population_mean_by_len = _mean_vector(population_by_len_vectors)

        for candidate in sampled:
            key = str(candidate["candidate_key"])
            metadata = {
                "candidate_key": key,
                "candidate_id": int(candidate["resample_index"]),
                "policy_variant": policy,
                "initial_std": float(initial_std),
                "source_initial_std": float(initial_std),
                "elite_select_mode": mode,
                "resample_index": int(candidate["resample_index"]),
                "parameter_vector": [
                    float(value) for value in candidate["parameter_vector"]
                ],
                "parameter_names": list(candidate["parameter_names"]),
                "parameter_count": int(candidate["parameter_count"]),
            }
            resampled_metadata[key] = metadata
            records = [
                _evaluate_candidate_session(
                    context=context,
                    policy_variant=policy,
                    length_feature_mode=length_feature_mode,
                    max_residual_suffix_len=max_residual_suffix_len,
                    mean_residual_suffix_len=mean_residual_suffix_len,
                    std_residual_suffix_len=std_residual_suffix_len,
                    initial_std=float(initial_std),
                    candidate_key=key,
                    theta=metadata["parameter_vector"],
                    base_seed=base_seed,
                )
                for context in contexts
            ]
            resampled_records_by_candidate[key] = records
            for record in records[: int(sample_sessions)]:
                resampled_sample_rows.append(_session_sample_row(record))
            distance_context_by_key[key] = {
                "elite_thetas": elite_thetas,
                "elite_mean_theta": gaussian["elite_mean"],
                "elite_overall_vectors": elite_overall_vectors,
                "elite_by_len_vectors": elite_by_len_vectors,
                "elite_mean_overall": elite_mean_overall,
                "elite_mean_by_len": elite_mean_by_len,
                "population_mean_overall": population_mean_overall,
                "population_mean_by_len": population_mean_by_len,
            }

    resampled_overall_rows = [
        _elite_resampled_overall_row(
            resampled_metadata[key],
            records,
            distance_context=distance_context_by_key[key],
            suffix_lengths=suffix_lengths,
            by_suffix_len_rows=None,
        )
        for key, records in resampled_records_by_candidate.items()
    ]
    resampled_by_len_rows = _attach_resampled_metadata(
        _grouped_summary_rows(
            candidate_metadata=resampled_metadata,
            records_by_candidate=resampled_records_by_candidate,
            group_field="residual_suffix_len",
            output_field="residual_suffix_len",
        ),
        resampled_metadata,
    )
    resampled_by_group_rows = _attach_resampled_metadata(
        _grouped_summary_rows(
            candidate_metadata=resampled_metadata,
            records_by_candidate=resampled_records_by_candidate,
            group_field="suffix_group",
            output_field="suffix_group",
        ),
        resampled_metadata,
    )
    by_len_lookup_resampled = {
        (str(row["candidate_key"]), int(row["residual_suffix_len"])): row
        for row in resampled_by_len_rows
    }
    for row in resampled_overall_rows:
        key = str(row["candidate_key"])
        vector = _by_suffix_behavior_vector(key, suffix_lengths, by_len_lookup_resampled)
        context = distance_context_by_key[key]
        row["behavior_l1_to_elite_mean_by_len"] = _l1(
            vector,
            context["elite_mean_by_len"],
        )
        row["behavior_l1_to_nearest_elite_by_len"] = min(
            _l1(vector, elite_vector)
            for elite_vector in context["elite_by_len_vectors"]
        )
        row["behavior_l1_to_initial_population_mean_by_len"] = _l1(
            vector,
            context["population_mean_by_len"],
        )

    pairwise_rows = _pairwise_behavior_distance_rows(
        candidate_metadata=resampled_metadata,
        overall_rows=resampled_overall_rows,
        by_suffix_len_rows=resampled_by_len_rows,
    )
    pairwise_rows = [
        {
            **row,
            "elite_select_mode": str(resampled_metadata[str(row["candidate_key_a"])]["elite_select_mode"]),
        }
        for row in pairwise_rows
    ]

    elite_config_path = output_path / "elite_centered_config.json"
    save_json(
        {
            "include_elite_centered_diagnostic": True,
            "elite_select_mode": mode,
            "elite_count": int(elite_count),
            "elite_resample_count": int(elite_resample_count),
            "elite_min_std": float(elite_min_std),
            "elite_std_scale": float(elite_std_scale),
            "elite_centered_seed": int(elite_centered_seed),
            "policy_variant": policy,
            "initial_stds": [float(value) for value in initial_stds],
            "note": (
                "This elite-centered diagnostic uses behavior-selected pseudo-elites. "
                "It is only intended to validate numerical behavior of diagonal "
                "Gaussian resampling. It does not evaluate attack reward and must "
                "not be interpreted as proof of attack effectiveness."
            ),
            "surrogate_retraining": False,
            "victim_training": False,
            "formal_cem_connected": False,
        },
        elite_config_path,
    )
    paths["elite_centered_config"] = str(elite_config_path)

    selection_path = output_path / "elite_selection_summary.csv"
    _write_csv(selection_path, ELITE_SELECTION_COLUMNS, elite_selection_rows)
    paths["elite_selection_summary"] = str(selection_path)

    params_path = output_path / "elite_distribution_parameters.json"
    save_json(distribution_parameters, params_path)
    paths["elite_distribution_parameters"] = str(params_path)

    candidates_path = output_path / "elite_resampled_candidates.json"
    save_json(resampled_candidates, candidates_path)
    paths["elite_resampled_candidates"] = str(candidates_path)

    overall_path = output_path / "elite_resampled_overall_summary.csv"
    _write_csv(overall_path, ELITE_RESAMPLED_OVERALL_COLUMNS, resampled_overall_rows)
    paths["elite_resampled_overall_summary"] = str(overall_path)

    by_len_path = output_path / "elite_resampled_by_suffix_len_summary.csv"
    _write_csv(by_len_path, ELITE_RESAMPLED_BY_SUFFIX_LEN_COLUMNS, resampled_by_len_rows)
    paths["elite_resampled_by_suffix_len_summary"] = str(by_len_path)

    by_group_path = output_path / "elite_resampled_by_suffix_group_summary.csv"
    _write_csv(
        by_group_path,
        ELITE_RESAMPLED_BY_SUFFIX_GROUP_COLUMNS,
        resampled_by_group_rows,
    )
    paths["elite_resampled_by_suffix_group_summary"] = str(by_group_path)

    pairwise_path = output_path / "elite_resampling_pairwise_distance.csv"
    _write_csv(
        pairwise_path,
        (
            "policy_variant",
            "initial_std",
            "elite_select_mode",
            "candidate_key_a",
            "candidate_key_b",
            "overall_family_l1",
            "by_suffix_len_l1",
        ),
        pairwise_rows,
    )
    paths["elite_resampling_pairwise_distance"] = str(pairwise_path)

    samples_path = output_path / "elite_resampled_session_samples.jsonl"
    _write_jsonl(samples_path, resampled_sample_rows)
    paths["elite_resampled_session_samples"] = str(samples_path)
    return paths


def _evaluate_candidate_session(
    *,
    context: DirectActionSessionContext,
    policy_variant: str,
    length_feature_mode: str,
    max_residual_suffix_len: int,
    mean_residual_suffix_len: float,
    std_residual_suffix_len: float,
    initial_std: float,
    candidate_key: str,
    theta: Sequence[float],
    base_seed: int,
) -> dict[str, object]:
    m = int(context.residual_suffix_len)
    actions = enumerate_valid_direct_actions(m)
    scores = [
        score_direct_action(
            policy_variant=policy_variant,
            theta=theta,
            action=action,
            residual_suffix_len=m,
            length_feature_mode=length_feature_mode,
            max_residual_suffix_len=max_residual_suffix_len,
            mean_residual_suffix_len=mean_residual_suffix_len,
            std_residual_suffix_len=std_residual_suffix_len,
        )
        for action in actions
    ]
    probabilities = stable_softmax(scores)
    sample_seed = deterministic_direct_action_seed(
        base_seed=int(base_seed),
        policy_variant=policy_variant,
        initial_std=float(initial_std),
        candidate_key=str(candidate_key),
        session_index=int(context.fake_session_index),
        tag=DIRECT_ACTION_SAMPLE_TAG,
    )
    sampled_action = sample_direct_action_categorical(
        actions=actions,
        probabilities=probabilities,
        seed=sample_seed,
    )
    expected_family = direct_action_family_probabilities(
        actions=actions,
        probabilities=probabilities,
        residual_suffix_len=m,
    )
    sampled_family = map_direct_action_to_family(sampled_action, m)
    expected_consume_ratio = sum(
        float(probability) * direct_action_consume_ratio(action, m)
        for action, probability in zip(actions, probabilities)
    )
    sampled_consume_ratio = direct_action_consume_ratio(sampled_action, m)
    return {
        "candidate_key": str(candidate_key),
        "policy_variant": normalize_direct_action_policy_variant(policy_variant),
        "initial_std": float(initial_std),
        "fake_session_index": int(context.fake_session_index),
        "template_session": [int(item) for item in context.template_session],
        "template_length": int(len(context.template_session)),
        "anchor_position": int(context.anchor_position),
        "prefix": [int(item) for item in context.prefix],
        "prefix_length": int(len(context.prefix)),
        "residual_suffix": [int(item) for item in context.residual_suffix],
        "residual_suffix_len": m,
        "suffix_group": _suffix_group(m),
        "length_feature_mode": normalize_direct_action_length_feature_mode(
            length_feature_mode
        ),
        "length_feature": direct_action_length_feature(
            m,
            mode=length_feature_mode,
            max_residual_suffix_len=max_residual_suffix_len,
            mean_residual_suffix_len=mean_residual_suffix_len,
            std_residual_suffix_len=std_residual_suffix_len,
        ),
        "max_residual_suffix_len": int(max_residual_suffix_len),
        "mean_residual_suffix_len": float(mean_residual_suffix_len),
        "std_residual_suffix_len": float(std_residual_suffix_len),
        "valid_action_count": int(len(actions)),
        "valid_actions": [action.to_dict() for action in actions],
        "action_scores": [float(score) for score in scores],
        "action_probabilities": [float(probability) for probability in probabilities],
        "action_probability_details": action_probability_payload(
            actions=actions,
            scores=scores,
            probabilities=probabilities,
            residual_suffix_len=m,
        ),
        "sampled_action": sampled_action.to_dict(),
        "sampled_action_family": sampled_family,
        "sampled_generated_length": direct_action_generated_length(sampled_action, m),
        "expected_family_probabilities": expected_family,
        "expected_consume_ratio": float(expected_consume_ratio),
        "sampled_consume_ratio": float(sampled_consume_ratio),
        "entropy": direct_action_entropy(probabilities),
        "max_action_probability": float(max(probabilities)),
    }


def _diagnostic_config_payload(
    *,
    config: Config,
    config_path: str | Path | None,
    fake_sessions_path: str | Path | None,
    fake_sessions_identity: Mapping[str, object],
    policy_variant: str,
    length_feature_mode: str,
    max_residual_suffix_len: int,
    mean_residual_suffix_len: float,
    std_residual_suffix_len: float,
    initial_stds: Sequence[float],
    num_candidates: int,
    sample_sessions: int,
    seed: int,
    fake_session_count: int,
    prefix_seed_scope: str,
) -> dict[str, object]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_schema_version": DIRECT_ACTION_OUTPUT_SCHEMA_VERSION,
        "config_path": None if config_path is None else str(config_path),
        "dataset": config.data.dataset_name,
        "policy_variant": normalize_direct_action_policy_variant(policy_variant),
        "length_feature": normalize_direct_action_length_feature_mode(
            length_feature_mode
        ),
        "length_feature_definition": (
            "l = log(1 + m)"
            if normalize_direct_action_length_feature_mode(length_feature_mode)
            == DIRECT_ACTION_LENGTH_FEATURE_LOG1P
            else (
                "l = m"
                if normalize_direct_action_length_feature_mode(length_feature_mode)
                == DIRECT_ACTION_LENGTH_FEATURE_RAW_M
                else (
                    "l = (m - mean_m) / std_m"
                    if normalize_direct_action_length_feature_mode(length_feature_mode)
                    == DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
                    else "l = m / max_m"
                )
            )
        ),
        "max_residual_suffix_len": int(max_residual_suffix_len),
        "mean_residual_suffix_len": float(mean_residual_suffix_len),
        "std_residual_suffix_len": float(std_residual_suffix_len),
        "input_features": ["is_keep", "is_generate", "is_stop", "r", "l"],
        "parameter_count": parameter_count_for_policy(policy_variant),
        "parameter_names": list(parameter_names_for_policy(policy_variant)),
        "initial_stds": [float(value) for value in initial_stds],
        "num_candidates": int(num_candidates),
        "sample_sessions_per_candidate": int(sample_sessions),
        "seed": int(seed),
        "fake_session_count": int(fake_session_count),
        "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
        "fake_sessions_identity": dict(fake_sessions_identity),
        "prefix_selector": {
            "range": "internal",
            "sampler": "uniform",
            "seed_scope": str(prefix_seed_scope),
            "rng_tag": DIRECT_ACTION_PREFIX_TAG,
        },
        "target_independent": True,
        "materialize_generated_suffix": False,
        "formal_cem_connected": False,
        "surrogate_retraining": False,
        "victim_training": False,
    }


def _summary_row(
    candidate_info: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    *,
    include_p90: bool,
) -> dict[str, object]:
    total = int(len(records))
    row: dict[str, object] = {
        "candidate_key": str(candidate_info["candidate_key"]),
        "policy_variant": str(candidate_info["policy_variant"]),
        "initial_std": float(candidate_info["initial_std"]),
        "num_sessions": total,
        "session_count": total,
        "expected_consume_ratio_mean": _mean(
            float(record["expected_consume_ratio"]) for record in records
        ),
        "sampled_consume_ratio_mean": _mean(
            float(record["sampled_consume_ratio"]) for record in records
        ),
        "entropy_mean": _mean(float(record["entropy"]) for record in records),
        "max_action_probability_mean": _mean(
            float(record["max_action_probability"]) for record in records
        ),
    }
    if include_p90:
        row["max_action_probability_p90"] = _percentile(
            [float(record["max_action_probability"]) for record in records],
            0.90,
        )
    for family in DIRECT_ACTION_FAMILIES:
        prefix = FAMILY_COLUMN_PREFIXES[family]
        row[f"expected_{prefix}_ratio"] = _mean(
            float(record["expected_family_probabilities"][family])
            for record in records
        )
        row[f"sampled_{prefix}_ratio"] = _ratio(
            sum(1 for record in records if str(record["sampled_action_family"]) == family),
            total,
        )
    return row


def _grouped_summary_rows(
    *,
    candidate_metadata: Mapping[str, Mapping[str, object]],
    records_by_candidate: Mapping[str, Sequence[Mapping[str, object]]],
    group_field: str,
    output_field: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, records in records_by_candidate.items():
        groups: dict[object, list[Mapping[str, object]]] = defaultdict(list)
        for record in records:
            groups[record[group_field]].append(record)
        for group_value, group_records in sorted(groups.items(), key=lambda item: item[0]):
            row = _summary_row(
                candidate_metadata[key],
                group_records,
                include_p90=False,
            )
            row[output_field] = group_value
            rows.append(row)
    return rows


def _uniform_baseline_rows(
    contexts: Sequence[DirectActionSessionContext],
) -> list[dict[str, object]]:
    suffix_lengths = sorted({int(context.residual_suffix_len) for context in contexts})
    rows: list[dict[str, object]] = []
    for m in suffix_lengths:
        baseline = uniform_family_baseline(m)
        rows.append(
            {
                "residual_suffix_len": int(m),
                "valid_action_count": int(2 * m + 1),
                "uniform_keep_full_ratio": baseline[
                    DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX
                ],
                "uniform_generate_full_ratio": baseline[
                    DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX
                ],
                "uniform_partial_keep_ratio": baseline[
                    DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX
                ],
                "uniform_partial_generate_ratio": baseline[
                    DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX
                ],
                "uniform_stop_ratio": baseline[DIRECT_ACTION_FAMILY_STOP],
            }
        )
    return rows


def _bias_vs_uniform_rows(
    by_suffix_len_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in by_suffix_len_rows:
        m = int(row["residual_suffix_len"])
        baseline = uniform_family_baseline(m)
        rows.append(
            {
                "candidate_key": str(row["candidate_key"]),
                "policy_variant": str(row["policy_variant"]),
                "initial_std": float(row["initial_std"]),
                "residual_suffix_len": m,
                "keep_full_minus_uniform": float(row["expected_keep_full_ratio"])
                - baseline[DIRECT_ACTION_FAMILY_KEEP_FULL_SUFFIX],
                "generate_full_minus_uniform": float(row["expected_generate_full_ratio"])
                - baseline[DIRECT_ACTION_FAMILY_GENERATE_FULL_SUFFIX],
                "partial_keep_minus_uniform": float(row["expected_partial_keep_ratio"])
                - baseline[DIRECT_ACTION_FAMILY_PARTIAL_KEEP_SUFFIX],
                "partial_generate_minus_uniform": float(
                    row["expected_partial_generate_ratio"]
                )
                - baseline[DIRECT_ACTION_FAMILY_PARTIAL_GENERATE_SUFFIX],
                "stop_minus_uniform": float(row["expected_stop_ratio"])
                - baseline[DIRECT_ACTION_FAMILY_STOP],
            }
        )
    return rows


def _pairwise_behavior_distance_rows(
    *,
    candidate_metadata: Mapping[str, Mapping[str, object]],
    overall_rows: Sequence[Mapping[str, object]],
    by_suffix_len_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    overall_by_key = {str(row["candidate_key"]): row for row in overall_rows}
    suffix_lengths = sorted(
        {int(row["residual_suffix_len"]) for row in by_suffix_len_rows}
    )
    by_suffix_lookup = {
        (str(row["candidate_key"]), int(row["residual_suffix_len"])): row
        for row in by_suffix_len_rows
    }
    groups: dict[tuple[str, float], list[str]] = defaultdict(list)
    for key, metadata in candidate_metadata.items():
        groups[(str(metadata["policy_variant"]), float(metadata["initial_std"]))].append(
            str(key)
        )

    rows: list[dict[str, object]] = []
    for (policy_variant, initial_std), keys in sorted(groups.items()):
        ordered_keys = sorted(keys)
        for left_index, key_a in enumerate(ordered_keys):
            for key_b in ordered_keys[left_index + 1 :]:
                rows.append(
                    {
                        "policy_variant": policy_variant,
                        "initial_std": float(initial_std),
                        "candidate_key_a": key_a,
                        "candidate_key_b": key_b,
                        "overall_family_l1": _l1(
                            _overall_family_vector(overall_by_key[key_a]),
                            _overall_family_vector(overall_by_key[key_b]),
                        ),
                        "by_suffix_len_l1": _l1(
                            _by_suffix_behavior_vector(
                                key_a,
                                suffix_lengths,
                                by_suffix_lookup,
                            ),
                            _by_suffix_behavior_vector(
                                key_b,
                                suffix_lengths,
                                by_suffix_lookup,
                            ),
                        ),
                    }
                )
    return rows


def _overall_family_vector(row: Mapping[str, object]) -> list[float]:
    return [
        float(row["expected_keep_full_ratio"]),
        float(row["expected_generate_full_ratio"]),
        float(row["expected_partial_keep_ratio"]),
        float(row["expected_partial_generate_ratio"]),
        float(row["expected_stop_ratio"]),
    ]


def _by_suffix_behavior_vector(
    candidate_key: str,
    suffix_lengths: Sequence[int],
    by_suffix_lookup: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[float]:
    values: list[float] = []
    for suffix_len in suffix_lengths:
        values.extend(_overall_family_vector(by_suffix_lookup[(candidate_key, suffix_len)]))
    return values


def _session_sample_row(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "candidate_key": str(record["candidate_key"]),
        "policy_variant": str(record["policy_variant"]),
        "initial_std": float(record["initial_std"]),
        "session_index": int(record["fake_session_index"]),
        "residual_suffix_len": int(record["residual_suffix_len"]),
        "length_feature_mode": str(record["length_feature_mode"]),
        "length_feature": float(record["length_feature"]),
        "max_residual_suffix_len": int(record["max_residual_suffix_len"]),
        "mean_residual_suffix_len": float(record["mean_residual_suffix_len"]),
        "std_residual_suffix_len": float(record["std_residual_suffix_len"]),
        "valid_actions": record["valid_actions"],
        "action_scores": record["action_scores"],
        "action_probabilities": record["action_probabilities"],
        "action_probability_details": record["action_probability_details"],
        "sampled_action": record["sampled_action"],
        "sampled_action_family": str(record["sampled_action_family"]),
        "sampled_consume_ratio": float(record["sampled_consume_ratio"]),
        "sampled_generated_length": int(record["sampled_generated_length"]),
        "expected_consume_ratio": float(record["expected_consume_ratio"]),
        "expected_family_probabilities": record["expected_family_probabilities"],
    }


def _fake_sessions_identity(
    fake_sessions_path: str | Path | None,
    fake_sessions: Sequence[Sequence[int]],
) -> dict[str, object]:
    if fake_sessions_path is not None:
        path = Path(fake_sessions_path)
        if path.exists():
            digest = hashlib.sha1()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
            return {
                "type": "file_sha1",
                "path": str(path),
                "sha1": digest.hexdigest(),
            }
    payload = json.dumps(
        [[int(item) for item in session] for session in fake_sessions],
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"type": "content_sha1", "sha1": hashlib.sha1(payload.encode()).hexdigest()}


def _default_output_dir(
    *,
    config: Config,
    fake_sessions_identity: Mapping[str, object],
    policy_variant: str,
    length_feature_mode: str,
    initial_stds: Sequence[float],
    num_candidates: int,
    seed: int,
    prefix_seed_scope: str,
    include_elite_centered_diagnostic: bool = False,
    elite_select_mode: str = ELITE_SELECT_MODE_DIVERSE,
    elite_count: int = 4,
    elite_resample_count: int = 8,
    elite_min_std: float = 0.25,
    elite_std_scale: float = 1.0,
    elite_centered_seed: int | None = None,
) -> Path:
    payload = {
        "fake_sessions_identity": dict(fake_sessions_identity),
        "policy_variant": normalize_direct_action_policy_variant(policy_variant),
        "length_feature": normalize_direct_action_length_feature_mode(
            length_feature_mode
        ),
        "initial_stds": [float(value) for value in initial_stds],
        "num_candidates": int(num_candidates),
        "seed": int(seed),
        "prefix_selector": {
            "range": "internal",
            "sampler": "uniform",
            "seed_scope": str(prefix_seed_scope),
            "rng_tag": DIRECT_ACTION_PREFIX_TAG,
        },
    }
    if bool(include_elite_centered_diagnostic):
        payload["elite_centered_diagnostic"] = {
            "elite_select_mode": normalize_elite_select_mode(elite_select_mode),
            "elite_count": int(elite_count),
            "elite_resample_count": int(elite_resample_count),
            "elite_min_std": float(elite_min_std),
            "elite_std_scale": float(elite_std_scale),
            "elite_centered_seed": None
            if elite_centered_seed is None
            else int(elite_centered_seed),
        }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return shared_root(config) / "direct_action_init_diagnostics" / f"diag_{digest}"


def _direct_action_candidate_key(*, initial_std: float, candidate_id: int) -> str:
    std_token = f"{float(initial_std):g}".replace(".", "p").replace("-", "m")
    return f"std{std_token}_{cem_candidate_key(0, int(candidate_id))}"


def _elite_resampled_candidate_key(
    *,
    elite_select_mode: str,
    initial_std: float,
    resample_index: int,
) -> str:
    mode = normalize_elite_select_mode(elite_select_mode)
    std_token = f"{float(initial_std):g}".replace(".", "p").replace("-", "m")
    return f"elite_{mode}_std{std_token}_resamp{int(resample_index)}"


def _suffix_group(residual_suffix_len: int) -> str:
    m = int(residual_suffix_len)
    if m == 1:
        return "suffix_1"
    if m == 2:
        return "suffix_2"
    return "suffix_3plus"


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16], 16)


def _mean(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    return float(sum(materialized) / float(len(materialized)))


def _std(values: Sequence[float] | Any) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return 0.0
    center = _mean(materialized)
    return float(
        (
            sum((value - center) ** 2.0 for value in materialized)
            / float(len(materialized))
        )
        ** 0.5
    )


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(float(percentile) * float(len(ordered))) - 1)
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


def _l1(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compute L1 distance for vectors with different lengths.")
    return float(sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _euclidean(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compute L2 distance for vectors with different lengths.")
    return float(sum((float(a) - float(b)) ** 2.0 for a, b in zip(left, right)) ** 0.5)


def _mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    materialized = [[float(value) for value in vector] for vector in vectors]
    if not materialized:
        raise ValueError("Cannot compute mean vector from an empty sequence.")
    width = len(materialized[0])
    if any(len(vector) != width for vector in materialized):
        raise ValueError("Mean vector inputs must have matching lengths.")
    return [
        _mean(vector[index] for vector in materialized)
        for index in range(width)
    ]


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


__all__ = [
    "BIAS_COLUMNS",
    "BY_SUFFIX_GROUP_COLUMNS",
    "BY_SUFFIX_LEN_COLUMNS",
    "DIRECT_ACTION_OUTPUT_SCHEMA_VERSION",
    "DIRECT_ACTION_PREFIX_TAG",
    "DIRECT_ACTION_SAMPLE_TAG",
    "ELITE_SELECT_MODE_DIVERSE",
    "ELITE_SELECT_MODE_GENERATE_ORIENTED",
    "ELITE_SELECT_MODE_KEEP_ORIENTED",
    "ELITE_SELECT_MODE_MIXED",
    "ELITE_SELECT_MODE_PARTIAL_GENERATE_ORIENTED",
    "ELITE_SELECT_MODE_STOP_HEAVY",
    "ELITE_SELECT_MODES",
    "OVERALL_COLUMNS",
    "PAIRWISE_COLUMNS",
    "UNIFORM_BASELINE_COLUMNS",
    "DirectActionInitDiagnosticResult",
    "DirectActionSessionContext",
    "build_direct_action_session_contexts",
    "compute_elite_gaussian",
    "normalize_elite_select_mode",
    "run_direct_action_init_diagnostic",
    "sample_elite_centered_candidates",
    "select_behavior_elites",
]
