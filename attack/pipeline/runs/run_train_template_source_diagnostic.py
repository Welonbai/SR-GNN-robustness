from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from attack.fake_session_sources.train_template_source import (
    DENOMINATOR_REPRESENTATION,
    RAW_SESSION_REPRESENTATION,
    SOURCE_TYPE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    allocate_exact_length_quotas,
    jensen_shannon_divergence,
    ks_statistic,
    length_count_by_int,
    length_distribution_comparison_rows,
    length_stats,
    sample_train_templates_clean_exact_length_matched,
    target_pre_existing_stats,
    validate_train_sub_raw_sessions,
)
from attack.pipeline.core.pipeline_utils import (
    build_clean_pairs,
    fake_session_count_from_ratio,
    load_or_init_target_registry,
    requested_target_prefix,
)


SOURCE_TYPE = SOURCE_TYPE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED

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

@dataclass(frozen=True)
class TrainTemplateDiagnosticResult:
    output_dir: Path
    paths: dict[str, str]
    summary: dict[str, Any]


def run_train_template_source_diagnostic(
    *,
    config: Config,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> TrainTemplateDiagnosticResult:
    canonical_dataset = ensure_canonical_dataset(config)
    train_raw_sessions = validate_train_sub_raw_sessions(canonical_dataset.train_sub)
    clean_prefixes, clean_labels = build_clean_pairs(canonical_dataset)
    denominator_count = int(len(clean_prefixes))
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
    warnings = [str(item) for item in sampling_metadata.get("warnings", [])]

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
                length_count_by_int(train_raw_sessions),
                length_count_by_int(sampled_templates),
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
        length_distribution_comparison_rows(
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
