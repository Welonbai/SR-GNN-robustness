from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import (
    load_fake_sessions,
    load_selected_targets,
    load_target_info,
    load_target_registry,
    save_json,
)
from attack.common.config import (
    Config,
    PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM,
    load_config,
)
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    dataset_name,
    run_group_key,
    runs_root,
    shared_artifact_paths,
    shared_attack_artifact_key,
)
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.pipeline.core.pipeline_utils import (
    build_clean_pairs,
    fake_session_count_from_ratio,
    load_or_init_target_registry,
    requested_target_prefix,
)


SOURCE_TYPE = "train_template_clean_exact_length_matched"
RAW_SESSION_REPRESENTATION = "canonical_dataset.train_sub raw sessions"
DENOMINATOR_REPRESENTATION = "expanded prefix-label pairs"

TEMPLATE_JSONL = "sampled_train_template_sessions.jsonl"
SUMMARY_JSON = "train_template_source_summary.json"
LENGTH_CSV = "length_distribution_comparison.csv"
TARGET_CSV = "target_pre_existing_stats.csv"
TARGET_COLUMNS = (
    "target_item",
    "template_sessions_containing_target_count",
    "template_sessions_containing_target_ratio",
    "total_target_occurrences_in_templates",
)

EXACT_MODE = "exact_without_replacement"
FALLBACK_MODE = "nearest_length_fallback"
REPLACEMENT_MODE = "replacement"


@dataclass(frozen=True)
class TrainTemplateDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]
    summary: dict[str, Any]


def allocate_exact_length_quotas(
    length_counts: Mapping[int, int],
    n_fake: int,
) -> dict[int, int]:
    if int(n_fake) < 0:
        raise ValueError("n_fake must be non-negative.")
    counts = {int(length): int(count) for length, count in length_counts.items() if int(count) > 0}
    if not counts:
        if int(n_fake) == 0:
            return {}
        raise ValueError("length_counts must contain at least one positive count.")
    total = int(sum(counts.values()))
    raw = {
        length: (float(n_fake) * float(count) / float(total))
        for length, count in counts.items()
    }
    quotas = {length: int(math.floor(value)) for length, value in raw.items()}
    remaining = int(n_fake) - int(sum(quotas.values()))
    if remaining < 0:
        raise RuntimeError("Largest-remainder quota allocation over-assigned samples.")
    remainders = sorted(
        counts,
        key=lambda length: (-(raw[length] - math.floor(raw[length])), int(length)),
    )
    for length in remainders[:remaining]:
        quotas[int(length)] += 1
    if sum(quotas.values()) != int(n_fake):
        raise RuntimeError("Length quotas do not sum to n_fake.")
    return {int(length): int(quotas[length]) for length in sorted(quotas)}


def sample_train_templates_clean_exact_length_matched(
    train_raw_sessions: Sequence[Sequence[int]],
    n_fake: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, Any], list[dict[str, Any]]]:
    sessions = validate_train_sub_raw_sessions(train_raw_sessions)
    if int(n_fake) < 0:
        raise ValueError("n_fake must be non-negative.")
    if int(n_fake) == 0:
        return [], _empty_sampling_metadata(len(sessions)), []
    if not sessions:
        raise ValueError("train_raw_sessions must contain at least one session.")

    rng = random.Random(int(seed))
    groups: dict[int, list[int]] = defaultdict(list)
    for index, session in enumerate(sessions):
        groups[len(session)].append(int(index))
    length_counts = {length: len(indices) for length, indices in groups.items()}
    quotas = allocate_exact_length_quotas(length_counts, int(n_fake))

    selected_source_indices: set[int] = set()
    sampled_sessions: list[list[int]] = []
    rows: list[dict[str, Any]] = []
    fallback_count = 0
    replacement_count = 0
    shortage_details: list[dict[str, Any]] = []

    for quota_length in sorted(quotas):
        quota = int(quotas[quota_length])
        if quota <= 0:
            continue
        exact_pool = [
            index for index in groups[quota_length]
            if index not in selected_source_indices
        ]
        exact_take = min(quota, len(exact_pool))
        for source_index in _sample_indices(rng, exact_pool, exact_take):
            _append_sample(
                sampled_sessions,
                rows,
                sessions,
                selected_source_indices,
                source_index=source_index,
                quota_length=quota_length,
                sampling_mode=EXACT_MODE,
            )

        deficit = quota - exact_take
        if deficit <= 0:
            continue

        detail: dict[str, Any] = {
            "quota_length": int(quota_length),
            "quota": int(quota),
            "exact_available_without_replacement": int(len(exact_pool)),
            "initial_deficit": int(deficit),
            "fallback_filled": 0,
            "replacement_filled": 0,
        }

        for fallback_length in _nearest_lengths(quota_length, groups.keys(), include_exact=False):
            if deficit <= 0:
                break
            fallback_pool = [
                index for index in groups[fallback_length]
                if index not in selected_source_indices
            ]
            take = min(deficit, len(fallback_pool))
            for source_index in _sample_indices(rng, fallback_pool, take):
                _append_sample(
                    sampled_sessions,
                    rows,
                    sessions,
                    selected_source_indices,
                    source_index=source_index,
                    quota_length=quota_length,
                    sampling_mode=FALLBACK_MODE,
                )
            deficit -= take
            fallback_count += take
            detail["fallback_filled"] = int(detail["fallback_filled"]) + int(take)

        for replacement_length in _nearest_lengths(quota_length, groups.keys(), include_exact=True):
            while deficit > 0 and groups[replacement_length]:
                source_index = int(rng.choice(groups[replacement_length]))
                _append_sample(
                    sampled_sessions,
                    rows,
                    sessions,
                    selected_source_indices,
                    source_index=source_index,
                    quota_length=quota_length,
                    sampling_mode=REPLACEMENT_MODE,
                    mark_selected=False,
                )
                deficit -= 1
                replacement_count += 1
                detail["replacement_filled"] = int(detail["replacement_filled"]) + 1
            if deficit <= 0:
                break
        if deficit > 0:
            raise RuntimeError("Replacement sampling failed to fill requested n_fake.")
        shortage_details.append(detail)

    if len(sampled_sessions) != int(n_fake):
        raise RuntimeError(
            f"Sampled template count mismatch: {len(sampled_sessions)} != {int(n_fake)}"
        )

    duplicate_metadata = _duplicate_metadata(rows, sampled_sessions)
    metadata = {
        "n_fake": int(n_fake),
        "sampling_pool_size": int(len(sessions)),
        "sampled_template_count": int(len(sampled_sessions)),
        "length_quota_by_length": _length_keyed_ints(quotas),
        "sampled_count_by_length": _length_distribution(sampled_sessions),
        "fallback_nearest_length_count": int(fallback_count),
        "replacement_sample_count": int(replacement_count),
        "shortage_by_quota_length": shortage_details,
        **duplicate_metadata,
    }
    return sampled_sessions, metadata, rows


def run_train_template_source_diagnostic(
    *,
    config: Config,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainTemplateDiagnosticResult:
    canonical_dataset = ensure_canonical_dataset(config)
    train_raw_sessions = validate_train_sub_raw_sessions(canonical_dataset.train_sub)
    clean_pairs = build_clean_pairs(canonical_dataset)
    denominator_count = int(len(clean_pairs[0]))
    n_fake = fake_session_count_from_ratio(float(config.attack.size), denominator_count)

    shared_run_type = _reference_generated_run_type(config)
    shared_paths = shared_artifact_paths(config, run_type=shared_run_type)
    stats = compute_session_stats(train_raw_sessions)
    targets, target_source = _resolve_target_items_without_surprise_initialization(
        config,
        stats=stats,
        shared_paths=shared_paths,
    )

    sampled_templates, sampling_metadata, sample_rows = (
        sample_train_templates_clean_exact_length_matched(
            train_raw_sessions,
            n_fake=int(n_fake),
            seed=int(config.seeds.fake_session_seed),
        )
    )
    target_stats = target_pre_existing_stats(sampled_templates, targets)
    generated_cache = _try_load_generated_cache_comparison(
        config,
        run_type=shared_run_type,
        shared_paths=shared_paths,
    )
    generated_sessions_for_csv = generated_cache.get("sessions")
    generated_cache_summary = dict(generated_cache)
    generated_cache_summary.pop("sessions", None)
    warnings = _sampling_warnings(sampling_metadata)

    summary = {
        "source_type": SOURCE_TYPE,
        "reference_split": "train_sub",
        "target_filtering": "none",
        "raw_session_source": "canonical_dataset.train_sub",
        "raw_session_representation": RAW_SESSION_REPRESENTATION,
        "source_pool_representation": RAW_SESSION_REPRESENTATION,
        "config_path": None if config_path is None else str(config_path),
        "dataset": dataset_name(config),
        "experiment_name": config.experiment.name,
        "target_registry_mode": target_source["target_registry_mode"],
        "target_list_source": target_source["target_list_source"],
        "target_items": [int(item) for item in targets],
        "denominator_source": "build_clean_pairs(canonical_dataset)[0]",
        "denominator_representation": DENOMINATOR_REPRESENTATION,
        "denominator_count": int(denominator_count),
        "attack_size": float(config.attack.size),
        "computed_n_fake": int(n_fake),
        "n_fake": int(n_fake),
        "sampling_pool_size": int(len(train_raw_sessions)),
        "sampled_template_count": int(len(sampled_templates)),
        "replacement": bool(int(sampling_metadata["replacement_sample_count"]) > 0),
        "warnings": warnings,
        "sampling": sampling_metadata,
        "length_stats": {
            "clean_train_sub": length_stats(train_raw_sessions),
            "sampled_templates": length_stats(sampled_templates),
        },
        "length_distribution_distance": {
            "sampled_vs_clean_js": jensen_shannon_divergence(
                _length_count_by_int(train_raw_sessions),
                _length_count_by_int(sampled_templates),
                log_base=2,
            ),
            "sampled_vs_clean_js_log_base": 2,
            "sampled_vs_clean_ks": ks_statistic(
                [len(session) for session in train_raw_sessions],
                [len(session) for session in sampled_templates],
            ),
        },
        "target_pre_existing_stats": target_stats,
        "generated_fake_cache": generated_cache_summary,
        "reference_generated_run_type": shared_run_type,
        "reference_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=shared_run_type,
        ),
        "reference_run_group_key": run_group_key(config, run_type=shared_run_type),
    }

    output_path = (
        Path(output_dir)
        if output_dir is not None
        else runs_root(config) / "train_template_source_diagnostic"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    template_rows_path = output_path / TEMPLATE_JSONL
    _write_jsonl(template_rows_path, sample_rows)
    paths["sampled_train_template_sessions"] = str(template_rows_path)

    summary_path = output_path / SUMMARY_JSON
    save_json(summary, summary_path)
    paths["summary"] = str(summary_path)

    length_csv_path = output_path / LENGTH_CSV
    _write_csv(
        length_csv_path,
        _length_distribution_comparison_rows(
            train_raw_sessions,
            sampled_templates,
            generated_sessions=generated_sessions_for_csv,
        ),
    )
    paths["length_distribution_comparison"] = str(length_csv_path)

    target_csv_path = output_path / TARGET_CSV
    _write_csv(target_csv_path, target_stats, fieldnames=TARGET_COLUMNS)
    paths["target_pre_existing_stats"] = str(target_csv_path)

    print(f"[train-template-source-diagnostic] wrote {output_path}")
    return TrainTemplateDiagnosticResult(
        output_dir=output_path,
        paths=paths,
        summary=summary,
    )


def validate_train_sub_raw_sessions(
    train_sub: Sequence[Sequence[int]],
) -> list[list[int]]:
    if _looks_like_sessions_labels_pair(train_sub):
        raise ValueError(
            "train_sub appears to be an expanded (sessions, labels) pair, not raw sessions."
        )
    if not isinstance(train_sub, Sequence) or isinstance(train_sub, (str, bytes)):
        raise TypeError("train_sub must be a sequence of raw sessions.")
    sessions: list[list[int]] = []
    for index, raw_session in enumerate(train_sub):
        if not isinstance(raw_session, Sequence) or isinstance(raw_session, (str, bytes)):
            raise TypeError(f"train_sub[{index}] must be a sequence of item IDs.")
        if _looks_like_expanded_record(raw_session):
            raise ValueError(
                f"train_sub[{index}] appears to be an expanded (prefix, label) record."
            )
        if len(raw_session) == 0:
            raise ValueError(f"train_sub[{index}] is empty.")
        session: list[int] = []
        for item in raw_session:
            if isinstance(item, bool) or not isinstance(item, Integral):
                raise TypeError(
                    f"train_sub[{index}] contains non-integer item ID: {item!r}"
                )
            session.append(int(item))
        sessions.append(session)
    return sessions


def length_stats(sessions: Sequence[Sequence[int]]) -> dict[str, Any]:
    lengths = np.asarray([len(session) for session in sessions], dtype=np.float64)
    if lengths.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "ratio_len_le_2": 0.0,
            "ratio_len_le_3": 0.0,
            "length_count_by_length": {},
        }
    return {
        "count": int(lengths.size),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "p25": float(np.percentile(lengths, 25)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "ratio_len_le_2": float(np.mean(lengths <= 2)),
        "ratio_len_le_3": float(np.mean(lengths <= 3)),
        "length_count_by_length": _length_distribution(sessions),
    }


def jensen_shannon_divergence(
    left_counts: Mapping[int, int],
    right_counts: Mapping[int, int],
    *,
    log_base: int | float = 2,
) -> float:
    support = sorted(set(int(key) for key in left_counts) | set(int(key) for key in right_counts))
    if not support:
        return 0.0
    left_total = float(sum(int(left_counts.get(key, 0)) for key in support))
    right_total = float(sum(int(right_counts.get(key, 0)) for key in support))
    if left_total <= 0.0 or right_total <= 0.0:
        raise ValueError("Both distributions must have positive total mass.")
    p = np.asarray([float(left_counts.get(key, 0)) / left_total for key in support])
    q = np.asarray([float(right_counts.get(key, 0)) / right_total for key in support])
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m, log_base) + 0.5 * _kl_divergence(q, m, log_base))


def ks_statistic(left_samples: Sequence[int], right_samples: Sequence[int]) -> float:
    if not left_samples and not right_samples:
        return 0.0
    if not left_samples or not right_samples:
        raise ValueError("Both samples must be non-empty for KS statistic.")
    left = np.sort(np.asarray([int(value) for value in left_samples], dtype=np.float64))
    right = np.sort(np.asarray([int(value) for value in right_samples], dtype=np.float64))
    support = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, support, side="right") / float(left.size)
    right_cdf = np.searchsorted(right, support, side="right") / float(right.size)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def target_pre_existing_stats(
    sampled_templates: Sequence[Sequence[int]],
    target_items: Sequence[int],
) -> list[dict[str, Any]]:
    total = int(len(sampled_templates))
    rows: list[dict[str, Any]] = []
    for target_item in target_items:
        target = int(target_item)
        containing_count = int(
            sum(1 for session in sampled_templates if target in {int(item) for item in session})
        )
        total_occurrences = int(
            sum(sum(1 for item in session if int(item) == target) for session in sampled_templates)
        )
        rows.append(
            {
                "target_item": target,
                "template_sessions_containing_target_count": containing_count,
                "template_sessions_containing_target_ratio": (
                    0.0 if total <= 0 else float(containing_count) / float(total)
                ),
                "total_target_occurrences_in_templates": total_occurrences,
            }
        )
    return rows


def _resolve_target_items_without_surprise_initialization(
    config: Config,
    *,
    stats,
    shared_paths: Mapping[str, Path],
) -> tuple[list[int], dict[str, str]]:
    registry = load_target_registry(shared_paths["target_registry"])
    if registry is not None:
        return requested_target_prefix(config, target_registry=registry), {
            "target_registry_mode": "existing_registry",
            "target_list_source": str(shared_paths["target_registry"]),
        }

    selected_targets = load_selected_targets(shared_paths["selected_targets"])
    if selected_targets is not None:
        return [int(item) for item in selected_targets], {
            "target_registry_mode": "legacy_saved_targets",
            "target_list_source": str(shared_paths["selected_targets"]),
        }

    legacy_target_info = load_target_info(shared_paths["target_info"])
    if legacy_target_info is not None and isinstance(legacy_target_info.get("target_items"), list):
        return [int(item) for item in legacy_target_info["target_items"]], {
            "target_registry_mode": "legacy_saved_targets",
            "target_list_source": str(shared_paths["target_info"]),
        }

    if bool(config.targets.reuse_saved_targets):
        raise FileNotFoundError(
            "targets.reuse_saved_targets=true, but no existing target registry or "
            "saved target artifact was found. Refusing to initialize targets in "
            "diagnostic mode."
        )

    registry = load_or_init_target_registry(
        stats,
        config,
        shared_paths=dict(shared_paths),
    )
    return requested_target_prefix(config, target_registry=registry), {
        "target_registry_mode": "initialized_registry",
        "target_list_source": str(shared_paths["target_registry"]),
    }


def _try_load_generated_cache_comparison(
    config: Config,
    *,
    run_type: str,
    shared_paths: Mapping[str, Path],
) -> dict[str, Any]:
    fake_sessions_path = Path(shared_paths["fake_sessions"])
    payload: dict[str, Any] = {
        "loaded": False,
        "path": str(fake_sessions_path),
        "run_type": str(run_type),
        "shared_fake_sessions_key": shared_attack_artifact_key(config, run_type=run_type),
        "identity_confirmation": "path derived from current config and reference run type",
    }
    if not fake_sessions_path.exists():
        payload["reason"] = "cache file does not exist for derived shared identity"
        return payload
    try:
        sessions = load_fake_sessions(fake_sessions_path)
    except Exception as exc:  # pragma: no cover - defensive artifact handling
        payload["reason"] = f"failed to load cache: {exc}"
        return payload
    if sessions is None:
        payload["reason"] = "cache loader returned no sessions"
        return payload
    normalized = validate_train_sub_raw_sessions(sessions)
    payload.update(
        {
            "loaded": True,
            "reason": None,
            "fake_session_count": int(len(normalized)),
            "sessions": normalized,
        }
    )
    return payload


def _reference_generated_run_type(config: Config) -> str:
    pts_config = config.attack.pts_construction
    if pts_config is not None:
        if pts_config.method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
            return PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE
        return PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
    return "random_nonzero_when_possible"


def _append_sample(
    sampled_sessions: list[list[int]],
    rows: list[dict[str, Any]],
    source_sessions: Sequence[Sequence[int]],
    selected_source_indices: set[int],
    *,
    source_index: int,
    quota_length: int,
    sampling_mode: str,
    mark_selected: bool = True,
) -> None:
    session = [int(item) for item in source_sessions[int(source_index)]]
    if mark_selected:
        selected_source_indices.add(int(source_index))
    sampled_sessions.append(session)
    rows.append(
        {
            "template_id": int(len(rows)),
            "source_session_index": int(source_index),
            "session": session,
            "length": int(len(session)),
            "quota_length": int(quota_length),
            "sampled_from_length": int(len(session)),
            "sampling_mode": str(sampling_mode),
        }
    )


def _sample_indices(rng: random.Random, candidates: Sequence[int], count: int) -> list[int]:
    if int(count) <= 0:
        return []
    if int(count) > len(candidates):
        raise ValueError("Cannot sample without replacement beyond candidate count.")
    return [int(index) for index in rng.sample(list(candidates), int(count))]


def _nearest_lengths(
    quota_length: int,
    lengths: Sequence[int],
    *,
    include_exact: bool,
) -> list[int]:
    return [
        int(length)
        for length in sorted(
            {int(length) for length in lengths if include_exact or int(length) != int(quota_length)},
            key=lambda length: (
                abs(int(length) - int(quota_length)),
                0 if int(length) < int(quota_length) else 1,
                int(length),
            ),
        )
    ]


def _duplicate_metadata(
    rows: Sequence[Mapping[str, Any]],
    sampled_sessions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    total = int(len(rows))
    record_counts = Counter(int(row["source_session_index"]) for row in rows)
    content_counts = Counter(tuple(int(item) for item in session) for session in sampled_sessions)
    record_duplicate_count = int(sum(count - 1 for count in record_counts.values() if count > 1))
    content_duplicate_count = int(sum(count - 1 for count in content_counts.values() if count > 1))
    return {
        "record_duplicate_count": record_duplicate_count,
        "record_duplicate_ratio": 0.0 if total <= 0 else float(record_duplicate_count) / float(total),
        "content_duplicate_count": content_duplicate_count,
        "content_duplicate_ratio": 0.0 if total <= 0 else float(content_duplicate_count) / float(total),
        "duplicate_template_count": content_duplicate_count,
        "duplicate_template_ratio": 0.0 if total <= 0 else float(content_duplicate_count) / float(total),
    }


def _sampling_warnings(sampling_metadata: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    fallback_count = int(sampling_metadata.get("fallback_nearest_length_count", 0))
    replacement_count = int(sampling_metadata.get("replacement_sample_count", 0))
    shortage_lengths = [
        int(item["quota_length"])
        for item in sampling_metadata.get("shortage_by_quota_length", [])
        if isinstance(item, Mapping)
    ]
    if fallback_count > 0:
        warnings.append(
            "nearest-length fallback was used; shortage quota lengths="
            + ",".join(str(length) for length in shortage_lengths)
        )
    if replacement_count > 0:
        warnings.append(
            "replacement sampling was used; sampled templates contain record-level duplicates"
        )
    return warnings


def _length_distribution_comparison_rows(
    clean_train_sessions: Sequence[Sequence[int]],
    sampled_templates: Sequence[Sequence[int]],
    *,
    generated_sessions: object | None,
) -> list[dict[str, Any]]:
    clean_counts = _length_count_by_int(clean_train_sessions)
    sampled_counts = _length_count_by_int(sampled_templates)
    generated_counts = (
        _length_count_by_int(generated_sessions)
        if isinstance(generated_sessions, Sequence)
        else None
    )
    support = sorted(set(clean_counts) | set(sampled_counts) | (set(generated_counts or {})))
    clean_total = sum(clean_counts.values())
    sampled_total = sum(sampled_counts.values())
    generated_total = sum(generated_counts.values()) if generated_counts is not None else 0
    rows: list[dict[str, Any]] = []
    for length in support:
        row: dict[str, Any] = {
            "length": int(length),
            "clean_train_count": int(clean_counts.get(length, 0)),
            "clean_train_ratio": _ratio(clean_counts.get(length, 0), clean_total),
            "sampled_template_count": int(sampled_counts.get(length, 0)),
            "sampled_template_ratio": _ratio(sampled_counts.get(length, 0), sampled_total),
        }
        if generated_counts is not None:
            row["generated_fake_count"] = int(generated_counts.get(length, 0))
            row["generated_fake_ratio"] = _ratio(generated_counts.get(length, 0), generated_total)
        rows.append(row)
    return rows


def _length_count_by_int(sessions: Sequence[Sequence[int]]) -> dict[int, int]:
    return {int(length): int(count) for length, count in Counter(len(session) for session in sessions).items()}


def _length_distribution(sessions: Sequence[Sequence[int]]) -> dict[str, int]:
    return {
        str(length): int(count)
        for length, count in sorted(_length_count_by_int(sessions).items())
    }


def _length_keyed_ints(values: Mapping[int, int]) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in sorted(values.items())}


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)


def _kl_divergence(p: np.ndarray, q: np.ndarray, log_base: int | float) -> float:
    mask = p > 0
    values = p[mask] * np.log(p[mask] / q[mask])
    if float(log_base) != math.e:
        values = values / math.log(float(log_base))
    return float(np.sum(values))


def _looks_like_sessions_labels_pair(value: object) -> bool:
    if not isinstance(value, tuple) or len(value) != 2:
        return False
    sessions, labels = value
    if not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes)):
        return False
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return False
    return _looks_like_sequence_of_sessions(sessions) and all(
        isinstance(label, Integral) and not isinstance(label, bool)
        for label in labels[: min(len(labels), 5)]
    )


def _looks_like_sequence_of_sessions(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        return False
    sample = value[0]
    return isinstance(sample, Sequence) and not isinstance(sample, (str, bytes))


def _looks_like_expanded_record(value: Sequence[object]) -> bool:
    return (
        len(value) == 2
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
        and isinstance(value[1], Integral)
        and not isinstance(value[1], bool)
    )


def _empty_sampling_metadata(pool_size: int) -> dict[str, Any]:
    return {
        "n_fake": 0,
        "sampling_pool_size": int(pool_size),
        "sampled_template_count": 0,
        "length_quota_by_length": {},
        "sampled_count_by_length": {},
        "fallback_nearest_length_count": 0,
        "replacement_sample_count": 0,
        "shortage_by_quota_length": [],
        "record_duplicate_count": 0,
        "record_duplicate_ratio": 0.0,
        "content_duplicate_count": 0,
        "content_duplicate_ratio": 0.0,
        "duplicate_template_count": 0,
        "duplicate_template_ratio": 0.0,
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_fieldnames: list[str] = []
    if fieldnames is not None:
        resolved_fieldnames = [str(field) for field in fieldnames]
    else:
        for row in rows:
            for key in row:
                if key not in resolved_fieldnames:
                    resolved_fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in resolved_fieldnames})


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample train_sub raw sessions as clean exact-length-matched fake templates.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    run_train_template_source_diagnostic(
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DENOMINATOR_REPRESENTATION",
    "RAW_SESSION_REPRESENTATION",
    "SOURCE_TYPE",
    "TrainTemplateDiagnosticResult",
    "allocate_exact_length_quotas",
    "jensen_shannon_divergence",
    "ks_statistic",
    "length_stats",
    "main",
    "run_train_template_source_diagnostic",
    "sample_train_templates_clean_exact_length_matched",
    "target_pre_existing_stats",
    "validate_train_sub_raw_sessions",
]
