#!/usr/bin/env python3
"""Read-only replacement-position diagnostics for SBR targeted poisoning runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.utils.run_bundle_loader import RunBundle, RunBundleLoaderError, load_run_bundle  # noqa: E402
from attack.position_opt.candidate_builder import build_candidate_position_result  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "diagnostics" / "replacement_position_space"
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    run_root: Path
    config_path: Path
    artifact_format: str
    nonzero_when_possible: bool | None


@dataclass
class LoadedMethod:
    spec: MethodSpec
    bundle: RunBundle | None
    load_error: str | None


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="random_nz_ratio1",
        label="Random-NZ@1.0",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_random_nonzero_when_possible_ratio1"
        / "run_group_8679b974a1",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_random_nonzero_when_possible_ratio1.yaml",
        artifact_format="position_stats.json + random_nonzero_position_metadata.json (aggregate only)",
        nonzero_when_possible=True,
    ),
    MethodSpec(
        key="prefix_nz_ratio1",
        label="Prefix-NZ@1.0",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_prefix_nonzero_when_possible_ratio1"
        / "run_group_122a28bd27",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_prefix_nonzero_when_possible_ratio1.yaml",
        artifact_format="position_stats.json + prefix_nonzero_when_possible_metadata.pkl (per-session)",
        nonzero_when_possible=True,
    ),
    MethodSpec(
        key="prefix_nz_ratio02",
        label="Prefix-NZ@0.2",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_prefix_nonzero_when_possible"
        / "run_group_14818d6dd6",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_prefix_nonzero_when_possible.yaml",
        artifact_format="position_stats.json + prefix_nonzero_when_possible_metadata.pkl (per-session)",
        nonzero_when_possible=True,
    ),
    MethodSpec(
        key="posopt_mvp_ratio1",
        label="PosOptMVP@1.0",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_position_optimization_reward_mvp_ratio1"
        / "run_group_3becc51c46",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_position_optimization_reward.yaml",
        artifact_format="position_stats.json + position_opt/selected_positions.json + position_opt/run_metadata.json",
        nonzero_when_possible=False,
    ),
    MethodSpec(
        key="shared_policy_ratio1",
        label="SharedPolicy@1.0",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_position_opt_shared_policy_ratio1"
        / "run_group_c1835ab73f",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_position_opt_shared_policy.yaml",
        artifact_format="position_stats.json + position_opt/selected_positions.json + position_opt/run_metadata.json",
        nonzero_when_possible=False,
    ),
    MethodSpec(
        key="shared_policy_nz_ratio1",
        label="SharedPolicy-NZ@1.0",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_position_opt_shared_policy_nonzero"
        / "run_group_0bce31ef52",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_position_opt_shared_policy_nonzero.yaml",
        artifact_format="position_stats.json + position_opt/selected_positions.json + position_opt/run_metadata.json",
        nonzero_when_possible=True,
    ),
    MethodSpec(
        key="dpsbr_ratio02",
        label="DPSBR original random@0.2",
        run_root=REPO_ROOT
        / "outputs"
        / "runs"
        / "diginetica"
        / "attack_dpsbr"
        / "run_group_7db577fb2e",
        config_path=REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_attack_dpsbr.yaml",
        artifact_format="position_stats.json + dpsbr_position_metadata.json (aggregate only)",
        nonzero_when_possible=False,
    ),
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a read-only replacement-position diagnostic report from existing artifacts."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for markdown/JSON/CSV diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_methods = [load_method(spec) for spec in METHOD_SPECS]
    available_methods = [method for method in loaded_methods if method.bundle is not None]
    if not available_methods:
        parser.exit(status=2, message="No configured run bundles were available.\n")

    canonical = build_canonical_section(available_methods)
    fake_session_section = build_fake_session_section(available_methods, canonical)
    method_overview = build_method_overview_table(available_methods)
    position_distribution = build_position_distribution_table(
        available_methods,
        fake_session_section["shared_fake_sessions"],
    )
    position_artifact_sources = build_position_artifact_sources_table(
        available_methods
    )
    random_nz_status = build_random_nz_status_section(available_methods)
    bucket_coverage = build_bucket_coverage_table(fake_session_section["shared_fake_sessions"])
    fairness = build_fairness_section(
        available_methods,
        canonical,
        fake_session_section,
    )
    candidate_section = build_candidate_section(fake_session_section["shared_fake_sessions"])

    summary_payload = {
        "canonical": canonical["json"],
        "fake_sessions": fake_session_section["json"],
        "candidate_space_ratio1_nonzero": candidate_section["json"],
        "method_overview": dataframe_to_records(method_overview),
        "position_distribution": dataframe_to_records(position_distribution),
        "position_artifact_sources": dataframe_to_records(position_artifact_sources),
        "random_nz_status": random_nz_status["json"],
        "bucket_coverage": dataframe_to_records(bucket_coverage),
        "fairness": fairness["json"],
        "method_load_status": [
            {
                "method": method.spec.label,
                "run_root": repo_relative(method.spec.run_root),
                "available": method.bundle is not None,
                "load_error": method.load_error,
            }
            for method in loaded_methods
        ],
    }

    write_json(summary_payload, output_dir / "summary.json")
    write_dataframe(method_overview, output_dir / "method_overview.csv")
    write_dataframe(position_distribution, output_dir / "position_distribution.csv")
    write_dataframe(position_artifact_sources, output_dir / "position_artifact_sources.csv")
    write_dataframe(bucket_coverage, output_dir / "bucket_coverage.csv")
    write_dataframe(random_nz_status["cells"], output_dir / "random_nz_status.csv")

    markdown = build_markdown_report(
        loaded_methods=loaded_methods,
        canonical=canonical,
        fake_session_section=fake_session_section,
        candidate_section=candidate_section,
        method_overview=method_overview,
        position_distribution=position_distribution,
        position_artifact_sources=position_artifact_sources,
        random_nz_status=random_nz_status,
        bucket_coverage=bucket_coverage,
        fairness=fairness,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote diagnostics to {output_dir}")
    return 0


def load_method(spec: MethodSpec) -> LoadedMethod:
    try:
        bundle = load_run_bundle(
            run_root=spec.run_root,
            method_key=spec.key,
            label=spec.label,
            dataset_hint="diginetica",
        )
    except (RunBundleLoaderError, FileNotFoundError, ValueError) as exc:
        return LoadedMethod(spec=spec, bundle=None, load_error=str(exc))
    return LoadedMethod(spec=spec, bundle=bundle, load_error=None)


def build_canonical_section(available_methods: list[LoadedMethod]) -> dict[str, Any]:
    split_keys = sorted(
        {
            require_nested_string(
                method.bundle.key_payloads,
                ("stable_run_group", "split_identity", "key"),
            )
            for method in available_methods
            if method.bundle is not None
        }
    )
    representative = available_methods[0].bundle
    assert representative is not None
    metadata_path = find_bundle_path(representative, "metadata.json", parent_fragment="canonical")
    train_sub_path = find_bundle_path(representative, "train_sub.pkl", parent_fragment="canonical")
    valid_path = find_bundle_path(representative, "valid.pkl", parent_fragment="canonical")
    test_path = find_bundle_path(representative, "test.pkl", parent_fragment="canonical")
    item_map_path = find_bundle_path(representative, "item_map.pkl", parent_fragment="canonical")

    metadata = load_json(metadata_path)
    train_sub = load_pickle(train_sub_path)
    valid = load_pickle(valid_path)
    test = load_pickle(test_path)
    item_map = load_pickle(item_map_path)

    train_sub_interactions = sum(len(session) for session in train_sub)
    valid_interactions = sum(len(session) for session in valid)
    test_interactions = sum(len(session) for session in test)
    full_train_sessions = len(train_sub) + len(valid)
    full_train_interactions = train_sub_interactions + valid_interactions

    counts_df = pd.DataFrame(
        [
            {"split": "train_sub", "sessions": len(train_sub), "interactions": train_sub_interactions},
            {"split": "valid", "sessions": len(valid), "interactions": valid_interactions},
            {"split": "test", "sessions": len(test), "interactions": test_interactions},
            {
                "split": "train_before_valid_holdout",
                "sessions": full_train_sessions,
                "interactions": full_train_interactions,
            },
        ]
    )
    artifact_df = pd.DataFrame(
        [
            {"artifact": "canonical metadata", "path": repo_relative(metadata_path)},
            {"artifact": "canonical train_sub", "path": repo_relative(train_sub_path)},
            {"artifact": "canonical valid", "path": repo_relative(valid_path)},
            {"artifact": "canonical test", "path": repo_relative(test_path)},
            {"artifact": "canonical item_map", "path": repo_relative(item_map_path)},
        ]
    )
    core = {
        "dataset_name": metadata.get("dataset_name"),
        "split_protocol": metadata.get("split_protocol"),
        "split_keys": split_keys,
        "min_session_len": require_nested_value(metadata, ("filtering", "min_session_len")),
        "min_item_count": require_nested_value(metadata, ("filtering", "min_item_count")),
        "valid_ratio": require_nested_value(metadata, ("valid_split", "valid_ratio")),
        "test_days": require_nested_value(metadata, ("time_split", "test_days")),
        "train_valid_test_seed": None,
        "train_valid_test_seed_note": (
            "No explicit split seed exists in the current pipeline. "
            "The split is deterministic: time-based test split plus trailing valid holdout."
        ),
        "train_sub_sessions": len(train_sub),
        "valid_sessions": len(valid),
        "test_sessions": len(test),
        "train_sub_interactions": train_sub_interactions,
        "valid_interactions": valid_interactions,
        "test_interactions": test_interactions,
        "full_train_sessions_before_valid_holdout": full_train_sessions,
        "full_train_interactions_before_valid_holdout": full_train_interactions,
        "items": len(item_map),
        "config_fields": [
            "data.dataset_name",
            "data.split_protocol",
            "data.poison_train_only",
            "data.canonical_split.min_session_len",
            "data.canonical_split.min_item_count",
            "data.canonical_split.valid_ratio",
            "data.canonical_split.test_days",
        ],
        "config_paths": sorted(
            {repo_relative(method.spec.config_path) for method in available_methods}
        ),
        "code_paths": [
            "attack/data/unified_split.py::build_canonical_dataset",
            "attack/data/unified_split.py::_time_split_sessions",
            "attack/data/unified_split.py::_split_train_valid",
            "attack/data/canonical_dataset.py::load_canonical_dataset",
        ],
        "artifact_paths": [repo_relative(metadata_path), repo_relative(train_sub_path), repo_relative(valid_path), repo_relative(test_path), repo_relative(item_map_path)],
    }
    return {
        "json": core,
        "counts_df": counts_df,
        "artifact_df": artifact_df,
    }


def build_fake_session_section(
    available_methods: list[LoadedMethod],
    canonical: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fake_session_paths: dict[str, Path] = {}
    fake_session_keys: dict[str, str] = {}
    for method in available_methods:
        assert method.bundle is not None
        bundle = method.bundle
        shared_key = require_nested_string(
            bundle.key_payloads,
            (
                "stable_run_group",
                "attack_identity",
                "shared_attack_artifact_identity",
                "key",
            ),
        )
        shared_payload = require_nested_mapping(
            bundle.key_payloads,
            (
                "stable_run_group",
                "attack_identity",
                "shared_attack_artifact_identity",
                "payload",
            ),
        )
        fake_path = find_bundle_path(bundle, "fake_sessions.pkl", parent_fragment="attack")
        fake_session_paths[method.spec.key] = fake_path
        fake_session_keys[method.spec.key] = shared_key
        cohort_payload = require_nested_mapping(
            bundle.key_payloads,
            ("stable_run_group", "target_cohort_identity", "payload"),
        )
        rows.append(
            {
                "method": method.spec.label,
                "run_group_key": bundle.run_group_key,
                "shared_attack_artifact_key": shared_key,
                "shared_fake_sessions_path": repo_relative(fake_path),
                "split_key": shared_payload.get("split_key"),
                "fake_session_seed": shared_payload.get("fake_session_seed"),
                "attack_size": require_nested_value(shared_payload, ("attack_generation", "size")),
                "fake_session_generation_topk": require_nested_value(
                    shared_payload, ("attack_generation", "fake_session_generation_topk")
                ),
                "target_mode": cohort_payload.get("mode"),
                "target_selection_seed": cohort_payload.get("target_selection_seed"),
                "target_items": ", ".join(str(item) for item in bundle.target_items),
            }
        )

    unique_fake_paths = sorted({path.resolve() for path in fake_session_paths.values()})
    unique_shared_keys = sorted(set(fake_session_keys.values()))
    shared_fake_sessions_path = unique_fake_paths[0]
    shared_fake_sessions = normalize_sessions(load_pickle(shared_fake_sessions_path))
    shared_attack_snapshot_path = shared_fake_sessions_path.parent / "config.yaml"
    shared_attack_snapshot = load_yaml(shared_attack_snapshot_path) if shared_attack_snapshot_path.is_file() else {}

    train_sub_path = resolve_repo_path(canonical["json"]["artifact_paths"][1])
    train_sub = normalize_sessions(load_pickle(train_sub_path))
    clean_train_prefix_count = sum(max(len(session) - 1, 0) for session in train_sub)
    fake_session_count = len(shared_fake_sessions)
    attack_size = float(rows[0]["attack_size"])
    expected_fake_session_count = max(1, int(round(clean_train_prefix_count * attack_size)))

    summary_df = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    summary = {
        "attack_size": attack_size,
        "clean_train_prefix_count": clean_train_prefix_count,
        "fake_session_count": fake_session_count,
        "expected_fake_session_count_via_rounding": expected_fake_session_count,
        "fake_session_seed": rows[0]["fake_session_seed"],
        "fake_session_generation_topk": rows[0]["fake_session_generation_topk"],
        "shared_fake_session_paths": [repo_relative(path) for path in unique_fake_paths],
        "shared_attack_artifact_keys": unique_shared_keys,
        "generated_per_target": False,
        "shared_across_targets": True,
        "shared_across_methods": len(unique_fake_paths) == 1,
        "fixed_before_replacement_selection": True,
        "shared_generation_identity_code_path": "attack/common/paths.py::shared_attack_artifact_key_payload",
        "generation_code_paths": [
            "attack/pipeline/core/pipeline_utils.py::prepare_shared_attack_artifacts",
            "attack/pipeline/core/pipeline_utils.py::_fake_session_count",
            "attack/generation/fake_session_parameter_sampler.py::FakeSessionParameterSampler",
            "attack/generation/fake_session_generator.py::FakeSessionGenerator",
            "attack/pipeline/runs/run_random_nonzero.py::run_random_nonzero",
            "attack/pipeline/runs/run_prefix_nonzero_when_possible.py::run_prefix_nonzero_when_possible",
            "attack/pipeline/runs/run_dp_sbr_baseline.py::run_dp_sbr_baseline",
        ],
        "shared_attack_snapshot_path": (
            repo_relative(shared_attack_snapshot_path) if shared_attack_snapshot_path.is_file() else None
        ),
        "shared_attack_snapshot_replacement_topk_ratio": require_optional_nested_value(
            shared_attack_snapshot, ("attack", "replacement_topk_ratio")
        ),
        "shared_attack_snapshot_note": (
            "The shared attack directory snapshot config is not authoritative for replacement_topk_ratio reuse. "
            "The shared fake-session identity ignores replacement_topk_ratio, so a ratio=0.2 snapshot can back ratio=1.0 methods."
        ),
    }

    length_hist_df = histogram_dataframe([len(session) for session in shared_fake_sessions], label="length")
    length_summary = summarize_numeric_distribution([len(session) for session in shared_fake_sessions])
    length_summary.update(
        {
            "pct_length_eq_2": fraction([len(session) == 2 for session in shared_fake_sessions]) * 100.0,
            "pct_length_le_3": fraction([len(session) <= 3 for session in shared_fake_sessions]) * 100.0,
            "pct_length_le_5": fraction([len(session) <= 5 for session in shared_fake_sessions]) * 100.0,
            "pct_length_le_10": fraction([len(session) <= 10 for session in shared_fake_sessions]) * 100.0,
        }
    )
    summary["length_summary"] = length_summary

    return {
        "json": summary,
        "summary_df": summary_df,
        "shared_fake_sessions": shared_fake_sessions,
        "length_summary_df": pd.DataFrame([length_summary]),
        "length_hist_df": length_hist_df,
    }


def build_candidate_section(fake_sessions: list[list[int]]) -> dict[str, Any]:
    results = [
        build_candidate_position_result(
            session,
            1.0,
            nonzero_action_when_possible=True,
        )
        for session in fake_sessions
    ]
    before_counts = [len(result.positions_before_mask) for result in results]
    after_counts = [len(result.positions) for result in results]
    fallback_count = sum(1 for result in results if result.fallback_to_pos0_only)
    pos0_removed_count = sum(1 for result in results if result.pos0_removed)
    no_nonzero_before_fallback_count = sum(
        1
        for result in results
        if all(position == 0 for position in result.positions_before_mask)
    )

    summary = summarize_numeric_distribution(after_counts)
    summary.update(
        {
            "replacement_topk_ratio": 1.0,
            "nonzero_action_when_possible": True,
            "pos0_removed_when_nonzero_available": True,
            "fallback_to_pos0_only_count": fallback_count,
            "pos0_removed_session_count": pos0_removed_count,
            "base_candidate_count_mean": float(np.mean(before_counts)),
            "base_candidate_count_std": float(np.std(before_counts, ddof=0)),
            "nonzero_candidate_count_eq_1_pct": fraction([count == 1 for count in after_counts]) * 100.0,
            "nonzero_candidate_count_le_2_pct": fraction([count <= 2 for count in after_counts]) * 100.0,
            "nonzero_candidate_count_le_3_pct": fraction([count <= 3 for count in after_counts]) * 100.0,
            "nonzero_candidate_count_ge_5_pct": fraction([count >= 5 for count in after_counts]) * 100.0,
            "no_valid_nonzero_candidate_before_fallback_count": no_nonzero_before_fallback_count,
            "any_session_without_valid_nonzero_after_current_logic": False,
        }
    )
    hist_df = histogram_dataframe(after_counts, label="nonzero_candidate_count")
    return {
        "json": summary,
        "hist_df": hist_df,
        "summary_df": pd.DataFrame([summary]),
    }


def build_method_overview_table(available_methods: list[LoadedMethod]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in available_methods:
        assert method.bundle is not None
        bundle = method.bundle
        cohort_payload = require_nested_mapping(
            bundle.key_payloads,
            ("stable_run_group", "target_cohort_identity", "payload"),
        )
        shared_key = require_nested_string(
            bundle.key_payloads,
            (
                "stable_run_group",
                "attack_identity",
                "shared_attack_artifact_identity",
                "key",
            ),
        )
        fake_path = find_bundle_path(bundle, "fake_sessions.pkl", parent_fragment="attack")
        clean_surrogate_path = find_clean_surrogate_checkpoint(bundle)
        rows.append(
            {
                "method": method.spec.label,
                "run_group_key": bundle.run_group_key,
                "run_root": repo_relative(bundle.run_root),
                "config_path": repo_relative(method.spec.config_path),
                "split_key": require_nested_string(
                    bundle.key_payloads,
                    ("stable_run_group", "split_identity", "key"),
                ),
                "target_cohort_key": bundle.target_cohort_key,
                "target_mode": cohort_payload.get("mode"),
                "target_bucket": cohort_payload.get("bucket"),
                "target_selection_seed": cohort_payload.get("target_selection_seed"),
                "target_items": ", ".join(str(item) for item in bundle.target_items),
                "victims": ", ".join(bundle.victims),
                "attack_size": bundle.attack_size,
                "replacement_topk_ratio": bundle.replacement_topk_ratio,
                "nonzero_action_when_possible": (
                    method.spec.nonzero_when_possible
                    if method.spec.nonzero_when_possible is not None
                    else bundle.nonzero_action_when_possible
                ),
                "shared_attack_artifact_key": shared_key,
                "shared_fake_sessions_path": repo_relative(fake_path),
                "clean_surrogate_checkpoint": (
                    None if clean_surrogate_path is None else repo_relative(clean_surrogate_path)
                ),
                "seeds_json": stable_json(bundle.seeds),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def build_position_distribution_table(
    available_methods: list[LoadedMethod],
    fake_sessions: list[list[int]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in available_methods:
        assert method.bundle is not None
        bundle = method.bundle
        shared_fake_path = find_bundle_path(bundle, "fake_sessions.pkl", parent_fragment="attack")
        shared_fake_sessions = normalize_sessions(load_pickle(shared_fake_path))
        fallback_count = None
        if method.spec.nonzero_when_possible:
            fallback_count = sum(
                1
                for session in shared_fake_sessions
                if build_candidate_position_result(
                    session,
                    float(bundle.replacement_topk_ratio),
                    nonzero_action_when_possible=True,
                ).fallback_to_pos0_only
            )
        for target_item, target_artifacts in sorted(bundle.target_artifacts.items()):
            if target_artifacts.position_stats_path is None or not target_artifacts.position_stats_path.is_file():
                continue
            payload = load_json(target_artifacts.position_stats_path)
            overall_counts = parse_int_mapping(
                require_nested_mapping(payload, ("overall", "counts"))
            )
            total_sessions = int(payload["total_sessions"])
            positions = expand_counts(overall_counts)
            normalized_positions = normalized_positions_from_stats_payload(payload)
            row = {
                "method": method.spec.label,
                "target_item": int(target_item),
                "total_fake_sessions": total_sessions,
                "pos0_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if pos == 0), total_sessions),
                "pos1_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if pos == 1), total_sessions),
                "pos2_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if pos == 2), total_sessions),
                "pos3_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if pos == 3), total_sessions),
                "pos4_pos5_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if 4 <= pos <= 5), total_sessions),
                "pos6plus_pct": ratio_pct(sum(count for pos, count in overall_counts.items() if pos >= 6), total_sessions),
                "mean_absolute_position": float(np.mean(positions)),
                "median_absolute_position": float(np.median(positions)),
                "unique_selected_positions": len(overall_counts),
                "normalized_position_mean": float(np.mean(normalized_positions)) if normalized_positions else None,
                "normalized_position_median": float(np.median(normalized_positions)) if normalized_positions else None,
                "fallback_count": fallback_count,
                "position_artifact_format": method.spec.artifact_format,
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["method", "target_item"]).reset_index(drop=True)


def build_position_artifact_sources_table(
    available_methods: list[LoadedMethod],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in available_methods:
        assert method.bundle is not None
        bundle = method.bundle
        for target_item, target_artifacts in sorted(bundle.target_artifacts.items()):
            if target_artifacts.position_stats_path is None:
                continue
            source_paths = [repo_relative(target_artifacts.position_stats_path)]
            if "prefix_nonzero_when_possible_metadata.pkl" in method.spec.artifact_format:
                if target_artifacts.prefix_metadata_path is not None:
                    source_paths.append(repo_relative(target_artifacts.prefix_metadata_path))
            elif "random_nonzero_position_metadata.json" in method.spec.artifact_format:
                if target_artifacts.random_nonzero_metadata_path is not None:
                    source_paths.append(repo_relative(target_artifacts.random_nonzero_metadata_path))
            elif "dpsbr_position_metadata.json" in method.spec.artifact_format:
                target_path = target_artifacts.target_dir / "dpsbr_position_metadata.json"
                if target_path.is_file():
                    source_paths.append(repo_relative(target_path))
            elif target_artifacts.selected_positions_path is not None:
                source_paths.append(repo_relative(target_artifacts.selected_positions_path))
                if target_artifacts.run_metadata_path is not None:
                    source_paths.append(repo_relative(target_artifacts.run_metadata_path))
            rows.append(
                {
                    "method": method.spec.label,
                    "target_item": int(target_item),
                    "artifact_format": method.spec.artifact_format,
                    "position_sources": "; ".join(source_paths),
                }
            )
    return pd.DataFrame(rows).sort_values(["method", "target_item"]).reset_index(drop=True)


def build_random_nz_status_section(
    available_methods: list[LoadedMethod],
) -> dict[str, Any]:
    random_method = next(
        method for method in available_methods if method.spec.key == "random_nz_ratio1"
    )
    assert random_method.bundle is not None
    bundle = random_method.bundle
    cells = require_nested_mapping(bundle.run_coverage, ("cells",))

    cell_rows: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for target_item in bundle.target_items:
        target_cells = require_nested_mapping(cells, (str(target_item),))
        for victim_name in bundle.victims:
            payload = require_nested_mapping(target_cells, (victim_name,))
            status = payload.get("status")
            artifacts = require_nested_mapping(payload, ("artifacts",))
            metrics_path = artifacts.get("metrics")
            metrics_payload = None
            if isinstance(metrics_path, str) and metrics_path.strip():
                metrics_payload = load_json(resolve_repo_path(metrics_path))
            metric_values = (
                require_nested_mapping(metrics_payload, ("metrics",))
                if isinstance(metrics_payload, dict) and isinstance(metrics_payload.get("metrics"), dict)
                else {}
            )
            row = {
                "target_item": int(target_item),
                "victim": victim_name,
                "status": status,
                "attempt_count": payload.get("attempt_count"),
                "completed_at": payload.get("completed_at"),
                "targeted_mrr@30": metric_values.get("targeted_mrr@30"),
                "targeted_recall@30": metric_values.get("targeted_recall@30"),
                "metrics_path": metrics_path,
            }
            cell_rows.append(row)
            if status == "completed":
                completed.append({"target_item": int(target_item), "victim": victim_name})
            else:
                missing.append({"target_item": int(target_item), "victim": victim_name, "status": status})

    multi_seed_append_possible = False
    multi_seed_note = (
        "Not under the current run-group identity. Changing fake_session_seed changes the final attack "
        "identity and run_group_key; changing victim_train_seed changes victim_prediction_key and "
        "run_coverage compatibility checks reject that mismatch. Multi-seed runs need separate run groups "
        "or an additional seed dimension in the storage model."
    )

    return {
        "json": {
            "run_root": repo_relative(bundle.run_root),
            "run_group_key": bundle.run_group_key,
            "target_items": list(bundle.target_items),
            "victims": list(bundle.victims),
            "completed_cells": completed,
            "missing_cells": missing,
            "completed_cell_count": len(completed),
            "expected_cell_count": len(bundle.target_items) * len(bundle.victims),
            "seed_values": bundle.seeds,
            "is_single_seed": True,
            "supports_multi_seed_append_under_current_architecture": multi_seed_append_possible,
            "multi_seed_append_note": multi_seed_note,
            "status_sources": [
                repo_relative(bundle.run_coverage_path),
                repo_relative(bundle.summary_current_path),
            ],
        },
        "cells": pd.DataFrame(cell_rows).sort_values(["target_item", "victim"]).reset_index(drop=True),
    }


def build_bucket_coverage_table(fake_sessions: list[list[int]]) -> pd.DataFrame:
    mode_rows: list[dict[str, Any]] = []
    for mode_name in (
        "UniformNonzero",
        "FirstNonzero",
        "AbsPos2",
        "AbsPos3Plus",
        "NormalizedEarly",
        "NormalizedMiddle",
        "NormalizedLate",
    ):
        per_session_candidates: list[list[int]] = []
        base_nonzero_candidates: list[list[int]] = []
        for session in fake_sessions:
            result = build_candidate_position_result(
                session,
                1.0,
                nonzero_action_when_possible=True,
            )
            nonzero_candidates = list(result.positions)
            base_nonzero_candidates.append(nonzero_candidates)
            session_length = len(session)
            denom = session_length - 1
            if mode_name == "UniformNonzero":
                filtered = list(nonzero_candidates)
            elif mode_name == "FirstNonzero":
                filtered = [position for position in nonzero_candidates if position == 1]
            elif mode_name == "AbsPos2":
                filtered = [position for position in nonzero_candidates if position == 2]
            elif mode_name == "AbsPos3Plus":
                filtered = [position for position in nonzero_candidates if position >= 3]
            elif mode_name == "NormalizedEarly":
                filtered = [
                    position
                    for position in nonzero_candidates
                    if 0.0 < float(position) / float(denom) <= (1.0 / 3.0)
                ]
            elif mode_name == "NormalizedMiddle":
                filtered = [
                    position
                    for position in nonzero_candidates
                    if (1.0 / 3.0) < float(position) / float(denom) <= (2.0 / 3.0)
                ]
            elif mode_name == "NormalizedLate":
                filtered = [
                    position
                    for position in nonzero_candidates
                    if (2.0 / 3.0) < float(position) / float(denom) <= 1.0
                ]
            else:  # pragma: no cover
                raise ValueError(f"Unhandled mode: {mode_name}")
            per_session_candidates.append(filtered)
        non_empty_mask = [bool(candidates) for candidates in per_session_candidates]
        total_sessions = len(per_session_candidates)
        fallback_ratio = ratio_pct(sum(not flag for flag in non_empty_mask), total_sessions)
        avg_candidate_count_all = float(
            np.mean([len(candidates) for candidates in per_session_candidates])
        )
        avg_candidate_count_non_empty = float(
            np.mean([len(candidates) for candidates in per_session_candidates if candidates])
        ) if any(non_empty_mask) else 0.0
        note = bucket_mode_note(
            mode_name=mode_name,
            fake_sessions=fake_sessions,
            per_session_candidates=per_session_candidates,
        )
        mode_rows.append(
            {
                "mode": mode_name,
                "non_empty_session_pct": ratio_pct(sum(non_empty_mask), total_sessions),
                "avg_candidate_count_all_sessions": avg_candidate_count_all,
                "avg_candidate_count_non_empty_sessions": avg_candidate_count_non_empty,
                "fallback_to_uniform_nonzero_pct": fallback_ratio,
                "note": note,
            }
        )
    return pd.DataFrame(mode_rows)


def build_fairness_section(
    available_methods: list[LoadedMethod],
    canonical: dict[str, Any],
    fake_session_section: dict[str, Any],
) -> dict[str, Any]:
    overview = build_method_overview_table(available_methods)
    sampled_methods = {
        method.spec.label: method
        for method in available_methods
        if method.bundle is not None and method.bundle.target_cohort_key == "target_cohort_8be070ab82"
    }
    target_registry_path = None
    registry_payload = None
    legacy_selected_targets_path = None
    legacy_selected_targets_payload = None
    legacy_mismatch_note = None

    representative_sampled = sampled_methods.get("Prefix-NZ@1.0")
    if representative_sampled is not None and representative_sampled.bundle is not None:
        bundle = representative_sampled.bundle
        target_registry_path = find_bundle_path(bundle, "target_registry.json", parent_fragment="target_cohorts")
        registry_payload = load_json(target_registry_path)
        selected_targets_path = find_bundle_path(bundle, "selected_targets.json", parent_fragment="targets")
        if selected_targets_path.is_file():
            legacy_selected_targets_path = selected_targets_path
            legacy_selected_targets_payload = load_json(selected_targets_path)
            registry_prefix = [int(item) for item in bundle.target_items]
            legacy_targets = [int(item) for item in legacy_selected_targets_payload.get("target_items", [])]
            if legacy_targets != registry_prefix:
                legacy_mismatch_note = (
                    "Legacy selected_targets.json does not match the authoritative appendable target cohort. "
                    "Use target_cohort target_registry.json instead."
                )

    checks = [
        {
            "check": "same target item list",
            "status": "partial",
            "detail": (
                "Prefix-NZ@1.0, SharedPolicy@1.0, and SharedPolicy-NZ@1.0 use the same 3-target prefix "
                "[11103, 39588, 5334]. PosOptMVP@1.0 extends the same cohort to 6 targets. "
                "Random-NZ@1.0 is a separate explicit 2-target run [5334, 11103]."
            ),
        },
        {
            "check": "same target_selection_seed",
            "status": "partial",
            "detail": (
                "Sampled-cohort methods use target_selection_seed=20260405. Random-NZ@1.0 uses an explicit list, "
                "so target_selection_seed is not part of its target_cohort identity."
            ),
        },
        {
            "check": "same fake sessions",
            "status": "yes",
            "detail": (
                "All inspected methods point at the same shared fake-session pool "
                f"{fake_session_section['json']['shared_attack_artifact_keys'][0]} "
                f"at {fake_session_section['json']['shared_fake_session_paths'][0]}."
            ),
        },
        {
            "check": "same victim models",
            "status": "yes",
            "detail": "All inspected runs enable srgnn, miasrec, and tron.",
        },
        {
            "check": "same dataset split",
            "status": "yes",
            "detail": (
                "All inspected runs use split_diginetica_unified_trainonly1_minitems5_minsess2_testdays7_valid0p1."
            ),
        },
        {
            "check": "same poison size",
            "status": "yes",
            "detail": "All inspected runs use attack.size=0.01.",
        },
        {
            "check": "same replacement_topk_ratio",
            "status": "no",
            "detail": (
                "Random-NZ@1.0, Prefix-NZ@1.0, PosOptMVP@1.0, SharedPolicy@1.0, and SharedPolicy-NZ@1.0 use 1.0. "
                "Prefix-NZ@0.2 and DPSBR original random@0.2 use 0.2."
            ),
        },
        {
            "check": "same nonzero_action_when_possible semantics",
            "status": "no",
            "detail": (
                "Random-NZ, Prefix-NZ, and SharedPolicy-NZ remove pos0 when a nonzero candidate exists. "
                "PosOptMVP@1.0 and SharedPolicy@1.0 allow pos0."
            ),
        },
        {
            "check": "same clean surrogate checkpoint for position-opt methods",
            "status": "yes",
            "detail": (
                "PosOptMVP@1.0, SharedPolicy@1.0, and SharedPolicy-NZ@1.0 all reference "
                "outputs/surrogates/diginetica/clean_srgnn_surrogate_from_attack_a7fd31f6af.pt in run_metadata.json."
            ),
        },
        {
            "check": "same victim prediction artifacts across methods",
            "status": "no",
            "detail": (
                "Each attack method has method-specific victim_prediction_key values keyed by attack identity. "
                "This is expected; attacked victim predictions are not shared across methods."
            ),
        },
    ]

    incompatibilities = [
        "Random-NZ@1.0 is not a like-for-like comparator against the sampled 3-target ratio=1.0 runs because it uses an explicit 2-target cohort.",
        "PosOptMVP@1.0 materialized a longer 6-target prefix of the same appendable cohort, so cross-method comparisons should be restricted to the shared prefix [11103, 39588, 5334].",
        "SharedPolicy@1.0 versus SharedPolicy-NZ@1.0 changes the action space by allowing or forbidding pos0, so differences are not attributable solely to policy learning.",
        "Prefix-NZ@0.2 and DPSBR original random@0.2 share the ratio=0.2 setting, but only Prefix-NZ removes pos0 when nonzero positions exist.",
    ]
    if legacy_mismatch_note is not None:
        incompatibilities.append(
            f"{legacy_mismatch_note} Legacy path: {repo_relative(legacy_selected_targets_path)}; authoritative path: {repo_relative(target_registry_path)}."
        )

    payload = {
        "checks": checks,
        "incompatibilities": incompatibilities,
        "target_registry_path": None if target_registry_path is None else repo_relative(target_registry_path),
        "legacy_selected_targets_path": (
            None if legacy_selected_targets_path is None else repo_relative(legacy_selected_targets_path)
        ),
        "legacy_selected_targets": legacy_selected_targets_payload,
        "target_registry": registry_payload,
    }
    return {
        "json": payload,
        "checks_df": pd.DataFrame(checks),
        "overview_df": overview,
    }


def build_markdown_report(
    *,
    loaded_methods: list[LoadedMethod],
    canonical: dict[str, Any],
    fake_session_section: dict[str, Any],
    candidate_section: dict[str, Any],
    method_overview: pd.DataFrame,
    position_distribution: pd.DataFrame,
    position_artifact_sources: pd.DataFrame,
    random_nz_status: dict[str, Any],
    bucket_coverage: pd.DataFrame,
    fairness: dict[str, Any],
    output_dir: Path,
) -> str:
    unavailable_methods = [
        method for method in loaded_methods if method.bundle is None
    ]
    lines: list[str] = [
        "# Replacement Position Space Diagnostics",
        "",
        "This report is read-only. It uses existing configs, JSON, YAML, and pickle artifacts only.",
        "",
        "## 1. Canonical dataset / split setting",
        "",
        f"- Dataset name: `{canonical['json']['dataset_name']}`",
        f"- min_session_len: `{canonical['json']['min_session_len']}`",
        f"- min_item_count: `{canonical['json']['min_item_count']}`",
        f"- valid_ratio: `{canonical['json']['valid_ratio']}`",
        f"- test_days: `{canonical['json']['test_days']}`",
        f"- Split seed: unknown / not used. {canonical['json']['train_valid_test_seed_note']}",
        f"- Items: `{canonical['json']['items']}`",
        "",
        dataframe_to_markdown(canonical["counts_df"]),
        "",
        "Relevant config fields:",
        "",
        *(f"- `{field}`" for field in canonical["json"]["config_fields"]),
        "",
        "Relevant config files:",
        "",
        *(f"- `{path}`" for path in canonical["json"]["config_paths"]),
        "",
        "Relevant artifacts:",
        "",
        dataframe_to_markdown(canonical["artifact_df"]),
        "",
        "Relevant code paths:",
        "",
        *(f"- `{path}`" for path in canonical["json"]["code_paths"]),
        "",
        "## 2. Fake session generation setting",
        "",
        f"- attack.size / poison size: `{fake_session_section['json']['attack_size']}`",
        f"- clean_train_prefix_count used for poisoning ratio: `{fake_session_section['json']['clean_train_prefix_count']}`",
        f"- fake session count: `{fake_session_section['json']['fake_session_count']}`",
        f"- fake_session_seed: `{fake_session_section['json']['fake_session_seed']}`",
        f"- fake_session_generation_topk: `{fake_session_section['json']['fake_session_generation_topk']}`",
        f"- generated per target item: `{fake_session_section['json']['generated_per_target']}`",
        f"- shared across targets: `{fake_session_section['json']['shared_across_targets']}`",
        f"- shared across inspected methods: `{fake_session_section['json']['shared_across_methods']}`",
        f"- fixed before replacement-position selection: `{fake_session_section['json']['fixed_before_replacement_selection']}`",
        "",
        dataframe_to_markdown(fake_session_section["summary_df"]),
        "",
        f"- Shared fake-session snapshot config path: `{fake_session_section['json']['shared_attack_snapshot_path']}`",
        f"- Snapshot replacement_topk_ratio stored there: `{fake_session_section['json']['shared_attack_snapshot_replacement_topk_ratio']}`",
        f"- Note: {fake_session_section['json']['shared_attack_snapshot_note']}",
        "",
        "Relevant code paths:",
        "",
        *(f"- `{path}`" for path in fake_session_section["json"]["generation_code_paths"]),
        "",
        "## 3. Fake session length distribution",
        "",
        "- All inspected methods in this report use the same shared fake-session pool, so there is one template length distribution.",
        f"- Evidence: shared_attack_artifact_keys = `{', '.join(fake_session_section['json']['shared_attack_artifact_keys'])}`",
        "",
        dataframe_to_markdown(fake_session_section["length_summary_df"]),
        "",
        dataframe_to_markdown(fake_session_section["length_hist_df"]),
        "",
        "## 4. Replacement candidate distribution under replacement_topk_ratio = 1.0 and nonzero action space",
        "",
        f"- replacement_topk_ratio analyzed here: `{candidate_section['json']['replacement_topk_ratio']}`",
        f"- pos0 removed when nonzero_action_when_possible=true: `{candidate_section['json']['pos0_removed_when_nonzero_available']}`",
        f"- fallback_to_pos0_only_count: `{candidate_section['json']['fallback_to_pos0_only_count']}`",
        f"- pos0_removed_session_count: `{candidate_section['json']['pos0_removed_session_count']}`",
        f"- no valid nonzero candidates before fallback: `{candidate_section['json']['no_valid_nonzero_candidate_before_fallback_count']}`",
        f"- any session without a valid candidate after current logic: `{candidate_section['json']['any_session_without_valid_nonzero_after_current_logic']}`",
        "",
        dataframe_to_markdown(candidate_section["summary_df"]),
        "",
        dataframe_to_markdown(candidate_section["hist_df"]),
        "",
        "Candidate construction source:",
        "",
        "- `attack/position_opt/candidate_builder.py::build_candidate_position_result`",
        "- `attack/position_opt/candidate_builder.py::filter_candidate_positions_nonzero_when_possible`",
        "",
        "## 5. Existing method position distributions",
        "",
        "Artifact formats by method:",
        "",
        *(f"- `{method.spec.label}`: {method.spec.artifact_format}" for method in loaded_methods if method.bundle is not None),
        "",
        dataframe_to_markdown(position_distribution),
        "",
        "Position artifact source paths:",
        "",
        dataframe_to_markdown(position_artifact_sources),
        "",
        "## 6. Current Random-NZ run status",
        "",
        f"- Run root: `{random_nz_status['json']['run_root']}`",
        f"- Run group key: `{random_nz_status['json']['run_group_key']}`",
        f"- Completed cells: `{random_nz_status['json']['completed_cell_count']}` / `{random_nz_status['json']['expected_cell_count']}`",
        f"- Missing cells within the requested cohort: `{len(random_nz_status['json']['missing_cells'])}`",
        f"- Seeds used: `{stable_json(random_nz_status['json']['seed_values'])}`",
        f"- Single-seed run: `{random_nz_status['json']['is_single_seed']}`",
        f"- Future multi-seed append under current architecture: `{random_nz_status['json']['supports_multi_seed_append_under_current_architecture']}`",
        f"- Note: {random_nz_status['json']['multi_seed_append_note']}",
        "",
        dataframe_to_markdown(random_nz_status["cells"]),
        "",
        "## 7. Preliminary bucket coverage diagnostics only",
        "",
        dataframe_to_markdown(bucket_coverage),
        "",
        "## 8. Fairness / comparability checks",
        "",
        dataframe_to_markdown(fairness["checks_df"]),
        "",
        "Method overview:",
        "",
        dataframe_to_markdown(method_overview),
        "",
        "Flagged incompatibilities:",
        "",
        *(f"- {item}" for item in fairness["json"]["incompatibilities"]),
        "",
        "## 9. Deliverables",
        "",
        f"- Markdown report: `{repo_relative(output_dir / 'report.md')}`",
        f"- Structured JSON summary: `{repo_relative(output_dir / 'summary.json')}`",
        f"- Method overview CSV: `{repo_relative(output_dir / 'method_overview.csv')}`",
        f"- Position distribution CSV: `{repo_relative(output_dir / 'position_distribution.csv')}`",
        f"- Position artifact source CSV: `{repo_relative(output_dir / 'position_artifact_sources.csv')}`",
        f"- Bucket coverage CSV: `{repo_relative(output_dir / 'bucket_coverage.csv')}`",
        f"- Random-NZ status CSV: `{repo_relative(output_dir / 'random_nz_status.csv')}`",
        f"- Standalone script: `{repo_relative(Path(__file__).resolve())}`",
        "",
        "Commands:",
        "",
        f"- `python analysis/diagnose_replacement_position_space.py --output-dir {repo_relative(output_dir)}`",
        "- `python analysis/diagnose_replacement_position_space.py --help`",
        "",
    ]
    if unavailable_methods:
        lines.extend(
            [
                "## Unavailable configured methods",
                "",
                *(
                    f"- `{method.spec.label}`: {method.load_error}"
                    for method in unavailable_methods
                ),
                "",
            ]
        )
    return "\n".join(lines)


def find_bundle_path(bundle: RunBundle, filename: str, *, parent_fragment: str | None = None) -> Path:
    candidates = [
        path
        for path in bundle.shared_artifact_paths.values()
        if path.name == filename and (parent_fragment is None or parent_fragment in path.as_posix())
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if filename == "fake_sessions.pkl":
            direct = REPO_ROOT / bundle.artifact_manifest["shared_artifacts"]["attack"]["fake_sessions"]
            if direct.is_file():
                return direct.resolve()
        raise FileNotFoundError(
            f"Could not locate {filename!r} inside shared artifacts for {bundle.label}."
        )
    return sorted(candidates)[0]


def find_bundle_dir(bundle: RunBundle, dirname: str, *, parent_fragment: str | None = None) -> Path:
    candidates = [
        path
        for path in bundle.shared_artifact_paths.values()
        if path.name == dirname and path.is_dir() and (parent_fragment is None or parent_fragment in path.as_posix())
    ]
    if not candidates:
        shared_root = REPO_ROOT / bundle.artifact_manifest["shared_artifacts"]["target_cohort"]["shared_dir"]
        sibling = shared_root.parent.parent / dirname
        if sibling.is_dir():
            return sibling.resolve()
        raise FileNotFoundError(
            f"Could not locate directory {dirname!r} inside shared artifacts for {bundle.label}."
        )
    return sorted(candidates)[0]


def find_clean_surrogate_checkpoint(bundle: RunBundle) -> Path | None:
    for target_item, target_artifacts in sorted(bundle.target_artifacts.items()):
        if target_artifacts.run_metadata_path is None:
            continue
        payload = load_json(target_artifacts.run_metadata_path)
        checkpoint = payload.get("clean_surrogate_checkpoint")
        if isinstance(checkpoint, str) and checkpoint.strip():
            return Path(checkpoint).resolve()
    return None


def normalize_sessions(payload: Any) -> list[list[int]]:
    if not isinstance(payload, list):
        raise ValueError("Expected a list of sessions.")
    sessions: list[list[int]] = []
    for index, session in enumerate(payload):
        if not isinstance(session, list):
            raise ValueError(f"Session {index} is not a list.")
        sessions.append([int(item) for item in session])
    return sessions


def load_json(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_pickle(path: str | Path) -> Any:
    resolved = Path(path).resolve()
    with resolved.open("rb") as handle:
        return pickle.load(handle)


def resolve_repo_path(raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def require_nested_mapping(payload: Any, path: tuple[str, ...]) -> dict[str, Any]:
    value = require_nested_value(payload, path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected {'.'.join(path)} to be a mapping.")
    return value


def require_nested_string(payload: Any, path: tuple[str, ...]) -> str:
    value = require_nested_value(payload, path)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected {'.'.join(path)} to be a non-empty string.")
    return value.strip()


def require_nested_value(payload: Any, path: tuple[str, ...]) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing {'.'.join(path)}")
        value = value[key]
    return value


def require_optional_nested_value(payload: Any, path: tuple[str, ...]) -> Any:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def parse_int_mapping(payload: dict[str, Any]) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for raw_key, raw_value in payload.items():
        parsed[int(raw_key)] = int(raw_value)
    return dict(sorted(parsed.items()))


def expand_counts(counts: dict[int, int]) -> list[int]:
    output: list[int] = []
    for value, count in sorted(counts.items()):
        output.extend([int(value)] * int(count))
    return output


def normalized_positions_from_stats_payload(payload: dict[str, Any]) -> list[float]:
    by_session_length = require_nested_mapping(payload, ("by_session_length",))
    values: list[float] = []
    for raw_length, raw_entry in sorted(by_session_length.items(), key=lambda item: int(item[0])):
        session_length = int(raw_length)
        if session_length <= 1:
            raise ValueError(
                f"Cannot compute normalized positions for session length {session_length}."
            )
        position_counts = parse_int_mapping(
            require_nested_mapping(raw_entry, ("position_counts",))
        )
        for position, count in position_counts.items():
            normalized = float(position) / float(session_length - 1)
            values.extend([normalized] * int(count))
    total_sessions = int(payload["total_sessions"])
    if len(values) != total_sessions:
        raise ValueError(
            f"Normalized position reconstruction mismatch: {len(values)} != {total_sessions}."
        )
    return values


def summarize_numeric_distribution(values: list[int] | list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def histogram_dataframe(values: list[int], *, label: str) -> pd.DataFrame:
    counts = Counter(int(value) for value in values)
    total = sum(counts.values())
    rows = [
        {
            label: int(value),
            "count": int(count),
            "pct": ratio_pct(int(count), total),
        }
        for value, count in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def fraction(mask: list[bool]) -> float:
    if not mask:
        return 0.0
    return float(sum(bool(value) for value in mask)) / float(len(mask))


def ratio_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator) * 100.0


def stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []
    return json.loads(dataframe.to_json(orient="records", double_precision=15))


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "_No rows._"
    columns = [str(column) for column in dataframe.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in dataframe.iterrows():
        lines.append(
            "| "
            + " | ".join(format_markdown_value(row[column]) for column in columns)
            + " |"
        )
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}"
    return str(value)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_dataframe(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def bucket_mode_note(
    *,
    mode_name: str,
    fake_sessions: list[list[int]],
    per_session_candidates: list[list[int]],
) -> str:
    total_sessions = len(fake_sessions)
    non_empty = sum(1 for candidates in per_session_candidates if candidates)
    if mode_name == "UniformNonzero":
        return "Always non-empty here because all fake sessions have length >= 2."
    if mode_name == "FirstNonzero":
        return "Equivalent to pos1 only; non-empty whenever the session length is at least 2."
    if mode_name == "AbsPos2":
        empty_sessions = total_sessions - non_empty
        return f"Empty for sessions that expose only pos1 (mainly length-2 sessions); empty count={empty_sessions}."
    if mode_name == "AbsPos3Plus":
        empty_sessions = total_sessions - non_empty
        return f"Empty for sessions whose nonzero action space is limited to pos1-pos2; empty count={empty_sessions}."
    if mode_name.startswith("Normalized"):
        return (
            "Defined over p/(L-1); bucket support depends on session length, so very short sessions reduce coverage."
        )
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
