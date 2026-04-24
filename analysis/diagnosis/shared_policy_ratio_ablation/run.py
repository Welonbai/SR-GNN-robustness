#!/usr/bin/env python3
"""Extract a Shared Policy replacement_topk_ratio ablation report from run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.diagnosis.prefix_vs_posopt.loaders import (  # noqa: E402
    DiagnosisError,
    REPO_ROOT,
    normalize_fake_sessions,
    normalize_posopt_selected_positions,
    normalize_prefix_selected_positions,
    repo_relative,
)
from analysis.diagnosis.prefix_vs_posopt.writers import (  # noqa: E402
    dataframe_to_markdown,
    dataframe_to_records,
    ensure_directory,
    write_dataframe,
    write_json,
    write_report,
)
from analysis.utils.position_collapse_summary import summarize_position_collapse_file  # noqa: E402
from analysis.utils.position_stats_summary import summarize_position_stats_file  # noqa: E402
from attack.position_opt.candidate_builder import build_candidate_positions  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis" / "diagnosis_outputs"
METHOD_ORDER = (
    "clean",
    "prefix_nz",
    "posopt_mvp",
    "shared_policy_ratio1",
    "shared_policy_ratio05",
)
POSITION_METHOD_ORDER = (
    "prefix_nz",
    "posopt_mvp",
    "shared_policy_ratio1",
    "shared_policy_ratio05",
)
TRAINING_METHOD_ORDER = (
    "shared_policy_ratio1",
    "shared_policy_ratio05",
)
FINAL_METRIC_SPECS = (
    ("target_recall@10", "targeted_recall@10"),
    ("target_recall@20", "targeted_recall@20"),
    ("target_recall@30", "targeted_recall@30"),
    ("target_mrr@10", "targeted_mrr@10"),
    ("target_mrr@20", "targeted_mrr@20"),
    ("target_mrr@30", "targeted_mrr@30"),
    ("gt_recall@10", "ground_truth_recall@10"),
    ("gt_recall@20", "ground_truth_recall@20"),
    ("gt_recall@30", "ground_truth_recall@30"),
    ("gt_mrr@10", "ground_truth_mrr@10"),
    ("gt_mrr@20", "ground_truth_mrr@20"),
    ("gt_mrr@30", "ground_truth_mrr@30"),
)
TARGET_METRIC_COLUMNS = tuple(name for name, _ in FINAL_METRIC_SPECS[:6])
GT_METRIC_COLUMNS = tuple(name for name, _ in FINAL_METRIC_SPECS[6:])
TRAINING_OUTPUT_COLUMNS = [
    "method",
    "target_item",
    "outer_step",
    "reward",
    "baseline",
    "advantage",
    "mean_entropy",
    "policy_loss",
    "target_utility",
    "avg_selected_position",
    "median_selected_position",
    "fraction_pos0",
    "fraction_pos<=1",
    "fraction_pos<=2",
    "gt_drop",
    "gt_penalty",
]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    attack_method: str
    run_root: Path
    summary_current: Path
    replacement_topk_ratio: float | None = None


@dataclass(frozen=True)
class ManifestSpec:
    path: Path
    report_id: str
    dataset: str
    victim_model: str
    required_targets: tuple[int, ...]
    optional_targets: tuple[int, ...]
    shared_fake_sessions: Path
    methods: dict[str, MethodSpec]


@dataclass(frozen=True)
class MethodRuntime:
    spec: MethodSpec
    summary_payload: dict[str, Any]


@dataclass(frozen=True)
class PositionArtifacts:
    method_key: str
    target_item: int
    position_stats_path: Path
    selected_positions_path: Path
    position_summary: dict[str, Any]
    selected_positions: list[dict[str, Any]]
    training_history_path: Path | None = None
    training_summary: dict[str, Any] | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a Shared Policy replacement_topk_ratio ablation report from run artifacts.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a diagnosis manifest YAML file.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder where diagnosis outputs will be written.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        output_dir = run_report(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
        )
        print(f"Wrote shared-policy ratio ablation outputs to '{output_dir}'.")
    except DiagnosisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def run_report(*, config_path: Path, output_root: Path) -> Path:
    manifest = load_manifest(config_path)
    runtimes = {
        key: MethodRuntime(
            spec=spec,
            summary_payload=_load_json_mapping(spec.summary_current, label=f"{key} summary_current"),
        )
        for key, spec in manifest.methods.items()
    }
    fake_sessions = _load_shared_fake_sessions(manifest.shared_fake_sessions)
    selected_targets = _select_targets(manifest, runtimes)
    output_dir = output_root.resolve() / manifest.report_id
    ensure_directory(output_dir)

    final_metrics = build_final_metrics_table(
        manifest=manifest,
        runtimes=runtimes,
        selected_targets=selected_targets,
    )
    delta_metrics = build_delta_table(final_metrics)
    position_artifacts = load_position_artifacts(
        manifest=manifest,
        runtimes=runtimes,
        selected_targets=selected_targets,
    )
    final_position_summary = build_final_position_summary(
        manifest=manifest,
        position_artifacts=position_artifacts,
        selected_targets=selected_targets,
    )
    candidate_space = build_candidate_space_summary(
        fake_sessions=fake_sessions,
        selected_targets=selected_targets,
        replacement_topk_ratio=_require_shared_ratio05(manifest),
    )
    ratio1_exclusion = build_ratio1_exclusion_summary(
        manifest=manifest,
        fake_sessions=fake_sessions,
        position_artifacts=position_artifacts,
        final_metrics=final_metrics,
        selected_targets=selected_targets,
    )
    training_dynamics = build_training_dynamics_table(
        manifest=manifest,
        position_artifacts=position_artifacts,
        selected_targets=selected_targets,
    )
    textual_summary = build_textual_summary(
        final_metrics=final_metrics,
        final_position_summary=final_position_summary,
        ratio1_exclusion=ratio1_exclusion,
        selected_targets=selected_targets,
    )

    _write_outputs(
        output_dir=output_dir,
        manifest=manifest,
        runtimes=runtimes,
        selected_targets=selected_targets,
        final_metrics=final_metrics,
        delta_metrics=delta_metrics,
        final_position_summary=final_position_summary,
        candidate_space=candidate_space,
        ratio1_exclusion=ratio1_exclusion,
        training_dynamics=training_dynamics,
        textual_summary=textual_summary,
    )
    return output_dir


def load_manifest(path: Path) -> ManifestSpec:
    resolved_path = _resolve_existing_file(path, label="diagnosis manifest")
    payload = _load_yaml_mapping(resolved_path)
    methods_payload = _require_mapping(payload.get("methods"), "methods")
    methods: dict[str, MethodSpec] = {}
    for key, raw_method in methods_payload.items():
        method_mapping = _require_mapping(raw_method, f"methods.{key}")
        methods[str(key)] = MethodSpec(
            key=str(key),
            label=_require_string(method_mapping.get("label"), f"methods.{key}.label"),
            attack_method=_require_string(
                method_mapping.get("attack_method"),
                f"methods.{key}.attack_method",
            ),
            run_root=_resolve_repo_path(
                _require_string(method_mapping.get("run_root"), f"methods.{key}.run_root"),
                label=f"methods.{key}.run_root",
            ),
            summary_current=_resolve_repo_path(
                _require_string(
                    method_mapping.get("summary_current"),
                    f"methods.{key}.summary_current",
                ),
                label=f"methods.{key}.summary_current",
            ),
            replacement_topk_ratio=(
                None
                if method_mapping.get("replacement_topk_ratio") is None
                else float(method_mapping["replacement_topk_ratio"])
            ),
        )
    if "shared_policy_ratio05" not in methods:
        raise DiagnosisError("Manifest methods must include 'shared_policy_ratio05'.")
    targets_payload = _require_mapping(payload.get("targets"), "targets")
    return ManifestSpec(
        path=resolved_path,
        report_id=_require_string(payload.get("report_id"), "report_id"),
        dataset=_require_string(payload.get("dataset"), "dataset"),
        victim_model=_require_string(payload.get("victim_model"), "victim_model"),
        required_targets=_parse_int_list(
            targets_payload.get("required"),
            "targets.required",
        ),
        optional_targets=_parse_int_list(
            targets_payload.get("optional_if_available"),
            "targets.optional_if_available",
            required=False,
        ),
        shared_fake_sessions=_resolve_repo_path(
            _require_string(
                _require_mapping(payload.get("shared_artifacts"), "shared_artifacts").get(
                    "fake_sessions"
                ),
                "shared_artifacts.fake_sessions",
            ),
            label="shared_artifacts.fake_sessions",
        ),
        methods=methods,
    )


def build_final_metrics_table(
    *,
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key in METHOD_ORDER:
            spec = manifest.methods.get(method_key)
            if spec is None:
                continue
            metrics = _lookup_metrics(
                summary_payload=runtimes[method_key].summary_payload,
                target_item=target_item,
                victim_model=manifest.victim_model,
            )
            row: dict[str, Any] = {
                "target_item": int(target_item),
                "method_key": method_key,
                "method": spec.label,
            }
            for output_key, metric_key in FINAL_METRIC_SPECS:
                row[output_key] = None if metrics is None else _optional_float(metrics.get(metric_key))
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_item", "method"]).reset_index(drop=True)


def build_delta_table(final_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [name for name, _ in FINAL_METRIC_SPECS]
    if final_metrics.empty:
        return pd.DataFrame(columns=["target_item", "comparison", *metric_columns])
    lookup = {
        (int(row.target_item), str(row.method_key)): row
        for _, row in final_metrics.iterrows()
    }
    comparison_specs = (
        ("shared_policy_ratio1", "Shared ratio=0.5 minus Shared ratio=1.0"),
        ("prefix_nz", "Shared ratio=0.5 minus Prefix-NZ"),
        ("posopt_mvp", "Shared ratio=0.5 minus PosOptMVP"),
    )
    rows: list[dict[str, Any]] = []
    for target_item in sorted({int(value) for value in final_metrics["target_item"].tolist()}):
        baseline = lookup.get((target_item, "shared_policy_ratio05"))
        for comparison_key, label in comparison_specs:
            comparison = lookup.get((target_item, comparison_key))
            row: dict[str, Any] = {
                "target_item": int(target_item),
                "comparison": label,
            }
            for metric_column in metric_columns:
                row[metric_column] = _subtract_optional(
                    None if baseline is None else baseline.get(metric_column),
                    None if comparison is None else comparison.get(metric_column),
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_item", "comparison"]).reset_index(drop=True)


def load_position_artifacts(
    *,
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
    selected_targets: tuple[int, ...],
) -> dict[tuple[str, int], PositionArtifacts]:
    artifacts: dict[tuple[str, int], PositionArtifacts] = {}
    for target_item in selected_targets:
        for method_key in POSITION_METHOD_ORDER:
            spec = manifest.methods.get(method_key)
            if spec is None:
                continue
            if not _summary_has_target(runtimes[method_key].summary_payload, target_item):
                continue
            artifacts[(method_key, target_item)] = _load_position_artifacts_for_method(
                method_key=method_key,
                spec=spec,
                target_item=target_item,
            )
    return artifacts


def build_final_position_summary(
    *,
    manifest: ManifestSpec,
    position_artifacts: dict[tuple[str, int], PositionArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key in POSITION_METHOD_ORDER:
            spec = manifest.methods.get(method_key)
            if spec is None:
                continue
            artifacts = position_artifacts.get((method_key, target_item))
            summary = None if artifacts is None else artifacts.position_summary
            max_share = None if summary is None else summary.get("max_position_share")
            cumulative = {} if summary is None else dict(summary.get("cumulative_ratios", {}))
            row = {
                "target_item": int(target_item),
                "method_key": method_key,
                "method": spec.label,
                "total": None if summary is None else int(summary["total_sessions"]),
                "unique_positions": (
                    None if summary is None else int(summary["unique_selected_position_count"])
                ),
                "dominant_position": (
                    None if not isinstance(max_share, dict) else int(max_share["position"])
                ),
                "dominant_pct": (
                    None if not isinstance(max_share, dict) else float(max_share["ratio_pct"])
                ),
                "pos0_pct": _extract_ratio_pct(cumulative, "<=0"),
                "pos<=1_pct": _extract_ratio_pct(cumulative, "<=1"),
                "pos<=2_pct": _extract_ratio_pct(cumulative, "<=2"),
                "pos<=5_pct": _extract_ratio_pct(cumulative, "<=5"),
                "top5_positions": (
                    None
                    if summary is None
                    else _format_top_position_rows(summary.get("top_positions", []), pct_key="ratio_pct")
                ),
            }
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target_item", "method"]).reset_index(drop=True)


def build_candidate_space_summary(
    *,
    fake_sessions: list[list[int]],
    selected_targets: tuple[int, ...],
    replacement_topk_ratio: float,
) -> pd.DataFrame:
    session_lengths = [len(session) for session in fake_sessions]
    candidate_positions_ratio05 = [
        build_candidate_positions(session, replacement_topk_ratio) for session in fake_sessions
    ]
    candidate_positions_ratio1 = [build_candidate_positions(session, 1.0) for session in fake_sessions]
    candidate_counts = [len(positions) for positions in candidate_positions_ratio05]
    max_positions = [max(positions) for positions in candidate_positions_ratio05]
    coverage_counter: Counter[int] = Counter()
    for positions in candidate_positions_ratio05:
        coverage_counter.update(int(position) for position in positions)
    total_sessions = len(fake_sessions)
    coverage_summary = _format_counter_pct_summary(
        coverage_counter,
        denominator=total_sessions,
        top_n=10,
    )
    rows = []
    for target_item in selected_targets:
        rows.append(
            {
                "target_item": int(target_item),
                "average_session_length": float(sum(session_lengths) / total_sessions),
                "average_candidate_count": float(sum(candidate_counts) / total_sessions),
                "min_candidate_count": int(min(candidate_counts)),
                "max_candidate_count": int(max(candidate_counts)),
                "average_max_candidate_position": float(sum(max_positions) / total_sessions),
                "candidate_position_coverage_summary": coverage_summary,
                "candidate_count_eq_1_pct": float(
                    sum(1 for count in candidate_counts if count == 1) / total_sessions * 100.0
                ),
                "pos0_only_pct": float(
                    sum(1 for positions in candidate_positions_ratio05 if positions == [0])
                    / total_sessions
                    * 100.0
                ),
                "positions_beyond_ratio05_exist_under_ratio1_pct": float(
                    sum(
                        1
                        for positions_ratio05, positions_ratio1 in zip(
                            candidate_positions_ratio05,
                            candidate_positions_ratio1,
                        )
                        if len(positions_ratio1) > len(positions_ratio05)
                    )
                    / total_sessions
                    * 100.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item"]).reset_index(drop=True)


def build_ratio1_exclusion_summary(
    *,
    manifest: ManifestSpec,
    fake_sessions: list[list[int]],
    position_artifacts: dict[tuple[str, int], PositionArtifacts],
    final_metrics: pd.DataFrame,
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    ratio05 = _require_shared_ratio05(manifest)
    rows: list[dict[str, Any]] = []
    metrics_lookup = {
        (int(row.target_item), str(row.method_key)): {
            metric_name: _optional_float(row[metric_name]) for metric_name, _ in FINAL_METRIC_SPECS
        }
        for _, row in final_metrics.iterrows()
    }
    candidate_counts_ratio05 = [len(build_candidate_positions(session, ratio05)) for session in fake_sessions]
    for target_item in selected_targets:
        ratio1_artifacts = position_artifacts.get(("shared_policy_ratio1", target_item))
        if ratio1_artifacts is None:
            continue
        selected_positions = [int(record["position"]) for record in ratio1_artifacts.selected_positions]
        if len(selected_positions) != len(candidate_counts_ratio05):
            raise DiagnosisError(
                f"Selected-position count mismatch for target {target_item}: "
                f"{len(selected_positions)} selections vs {len(candidate_counts_ratio05)} fake sessions."
            )
        excluded_positions = [
            position
            for position, candidate_count in zip(selected_positions, candidate_counts_ratio05)
            if position >= candidate_count
        ]
        excluded_counter = Counter(excluded_positions)
        ratio05_metrics = metrics_lookup.get((target_item, "shared_policy_ratio05"))
        ratio1_metrics = metrics_lookup.get((target_item, "shared_policy_ratio1"))
        rows.append(
            {
                "target_item": int(target_item),
                "ratio1_selected_outside_ratio05_pct": float(
                    len(excluded_positions) / len(selected_positions) * 100.0
                ),
                "excluded_selection_count": int(len(excluded_positions)),
                "average_excluded_selected_position": (
                    None
                    if not excluded_positions
                    else float(sum(excluded_positions) / len(excluded_positions))
                ),
                "excluded_position_frequency": _format_counter_pct_summary(
                    excluded_counter,
                    denominator=max(1, len(excluded_positions)),
                    top_n=5,
                ),
                "likely_ratio05_excludes_useful_positions": _assess_exclusion_usefulness(
                    excluded_rate=(
                        0.0
                        if not selected_positions
                        else float(len(excluded_positions) / len(selected_positions))
                    ),
                    ratio05_metrics=ratio05_metrics,
                    ratio1_metrics=ratio1_metrics,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item"]).reset_index(drop=True)


def build_training_dynamics_table(
    *,
    manifest: ManifestSpec,
    position_artifacts: dict[tuple[str, int], PositionArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key in TRAINING_METHOD_ORDER:
            spec = manifest.methods.get(method_key)
            if spec is None:
                continue
            artifacts = position_artifacts.get((method_key, target_item))
            if artifacts is None or artifacts.training_summary is None:
                continue
            for step in artifacts.training_summary.get("step_summaries", []):
                rows.append(
                    {
                        "method_key": method_key,
                        "method": spec.label,
                        "target_item": int(target_item),
                        "outer_step": int(step["outer_step"]),
                        "reward": _optional_float(step.get("reward")),
                        "baseline": _optional_float(step.get("baseline")),
                        "advantage": _optional_float(step.get("advantage")),
                        "mean_entropy": _optional_float(step.get("mean_entropy")),
                        "policy_loss": _optional_float(step.get("policy_loss")),
                        "target_utility": _optional_float(step.get("target_utility")),
                        "avg_selected_position": _optional_float(
                            step.get("average_selected_position")
                        ),
                        "median_selected_position": _optional_float(
                            step.get("median_selected_position")
                        ),
                        "fraction_pos0": _optional_float(step.get("fraction_pos0")),
                        "fraction_pos<=1": _optional_float(step.get("fraction_pos_le_1")),
                        "fraction_pos<=2": _optional_float(step.get("fraction_pos_le_2")),
                        "gt_drop": _optional_float(step.get("gt_drop")),
                        "gt_penalty": _optional_float(step.get("gt_penalty")),
                    }
                )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(columns=["method_key", *TRAINING_OUTPUT_COLUMNS])
    return dataframe.sort_values(["target_item", "method", "outer_step"]).reset_index(drop=True)


def build_textual_summary(
    *,
    final_metrics: pd.DataFrame,
    final_position_summary: pd.DataFrame,
    ratio1_exclusion: pd.DataFrame,
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    metrics_lookup = {
        (int(row.target_item), str(row.method_key)): {
            metric_name: _optional_float(row[metric_name]) for metric_name, _ in FINAL_METRIC_SPECS
        }
        for _, row in final_metrics.iterrows()
    }
    position_lookup = {
        (int(row.target_item), str(row.method_key)): row for _, row in final_position_summary.iterrows()
    }
    exclusion_lookup = {int(row.target_item): row for _, row in ratio1_exclusion.iterrows()}
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        ratio05_metrics = metrics_lookup.get((target_item, "shared_policy_ratio05"))
        ratio1_metrics = metrics_lookup.get((target_item, "shared_policy_ratio1"))
        prefix_metrics = metrics_lookup.get((target_item, "prefix_nz"))
        mvp_metrics = metrics_lookup.get((target_item, "posopt_mvp"))
        ratio05_position = position_lookup.get((target_item, "shared_policy_ratio05"))
        ratio1_position = position_lookup.get((target_item, "shared_policy_ratio1"))
        exclusion_row = exclusion_lookup.get(target_item)
        rows.append(
            {
                "target_item": int(target_item),
                "1_target_vs_ratio1": _summarize_metric_delta(
                    ratio05_metrics,
                    ratio1_metrics,
                    TARGET_METRIC_COLUMNS,
                ),
                "2_gt_vs_ratio1": _summarize_metric_delta(
                    ratio05_metrics,
                    ratio1_metrics,
                    GT_METRIC_COLUMNS,
                ),
                "3_position_behavior": _summarize_position_behavior(
                    ratio05_position=ratio05_position,
                    ratio1_position=ratio1_position,
                ),
                "4_useful_positions_excluded": (
                    "No direct evidence available."
                    if exclusion_row is None
                    else str(exclusion_row["likely_ratio05_excludes_useful_positions"])
                ),
                "5_closest_method": _closest_method_label(
                    ratio05_metrics=ratio05_metrics,
                    candidates={
                        "Prefix-NZ": prefix_metrics,
                        "PosOptMVP": mvp_metrics,
                        "Shared Policy ratio=1.0": ratio1_metrics,
                    },
                ),
                "6_verdict": _overall_verdict(
                    ratio05_metrics=ratio05_metrics,
                    ratio1_metrics=ratio1_metrics,
                    exclusion_row=exclusion_row,
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item"]).reset_index(drop=True)


def _write_outputs(
    *,
    output_dir: Path,
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
    selected_targets: tuple[int, ...],
    final_metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    final_position_summary: pd.DataFrame,
    candidate_space: pd.DataFrame,
    ratio1_exclusion: pd.DataFrame,
    training_dynamics: pd.DataFrame,
    textual_summary: pd.DataFrame,
) -> None:
    write_dataframe(final_metrics, output_dir / "final_metrics.csv")
    write_dataframe(delta_metrics, output_dir / "delta_metrics.csv")
    write_dataframe(final_position_summary, output_dir / "final_position_summary.csv")
    write_dataframe(candidate_space, output_dir / "candidate_space_ratio05.csv")
    write_dataframe(ratio1_exclusion, output_dir / "ratio1_exclusion_vs_ratio05.csv")
    write_dataframe(training_dynamics, output_dir / "training_dynamics.csv")
    write_dataframe(textual_summary, output_dir / "target_text_summary.csv")
    write_json(
        {
            "report_id": manifest.report_id,
            "config_path": repo_relative(manifest.path),
            "dataset": manifest.dataset,
            "victim_model": manifest.victim_model,
            "selected_targets": list(selected_targets),
            "methods": {
                key: {
                    "label": runtime.spec.label,
                    "attack_method": runtime.spec.attack_method,
                    "run_root": repo_relative(runtime.spec.run_root),
                    "summary_current": repo_relative(runtime.spec.summary_current),
                    "available_targets": sorted(
                        int(target_key)
                        for target_key in _require_mapping(
                            runtime.summary_payload.get("targets"),
                            f"{key} summary_current.targets",
                        ).keys()
                    ),
                }
                for key, runtime in runtimes.items()
            },
            "tables": {
                "final_metrics": dataframe_to_records(final_metrics),
                "delta_metrics": dataframe_to_records(delta_metrics),
                "final_position_summary": dataframe_to_records(final_position_summary),
                "candidate_space_ratio05": dataframe_to_records(candidate_space),
                "ratio1_exclusion_vs_ratio05": dataframe_to_records(ratio1_exclusion),
                "training_dynamics": dataframe_to_records(training_dynamics),
                "target_text_summary": dataframe_to_records(textual_summary),
            },
        },
        output_dir / "report_data.json",
    )
    write_json(
        {
            "report_id": manifest.report_id,
            "config_path": repo_relative(manifest.path),
            "dataset": manifest.dataset,
            "victim_model": manifest.victim_model,
            "required_targets": list(manifest.required_targets),
            "optional_targets": list(manifest.optional_targets),
            "selected_targets": list(selected_targets),
            "shared_fake_sessions": repo_relative(manifest.shared_fake_sessions),
            "methods": {
                key: {
                    "label": spec.label,
                    "attack_method": spec.attack_method,
                    "run_root": repo_relative(spec.run_root),
                    "summary_current": repo_relative(spec.summary_current),
                    "replacement_topk_ratio": spec.replacement_topk_ratio,
                }
                for key, spec in manifest.methods.items()
            },
        },
        output_dir / "manifest_resolved.json",
    )
    write_report(
        build_markdown_report(
            manifest=manifest,
            selected_targets=selected_targets,
            final_metrics=final_metrics,
            delta_metrics=delta_metrics,
            final_position_summary=final_position_summary,
            candidate_space=candidate_space,
            ratio1_exclusion=ratio1_exclusion,
            training_dynamics=training_dynamics,
            textual_summary=textual_summary,
        ),
        output_dir / "report.md",
    )


def build_markdown_report(
    *,
    manifest: ManifestSpec,
    selected_targets: tuple[int, ...],
    final_metrics: pd.DataFrame,
    delta_metrics: pd.DataFrame,
    final_position_summary: pd.DataFrame,
    candidate_space: pd.DataFrame,
    ratio1_exclusion: pd.DataFrame,
    training_dynamics: pd.DataFrame,
    textual_summary: pd.DataFrame,
) -> str:
    lines = [
        f"# {manifest.report_id}",
        "",
        "## Scope",
        f"- Dataset: `{manifest.dataset}`",
        f"- Victim: `{manifest.victim_model}`",
        f"- Selected targets: {', '.join(str(target) for target in selected_targets)}",
        "- Blank cells indicate unavailable method-target artifacts.",
        "- Authoritative final metrics come from `summary_current.json`.",
        "- Candidate-space summaries are computed from the shared fake-session pool and current `replacement_topk_ratio` semantics.",
        "",
        "## A. Final Metrics Comparison",
        dataframe_to_markdown(final_metrics.drop(columns=["method_key"])),
        "",
        "## B. Delta Table",
        dataframe_to_markdown(delta_metrics),
        "",
        "## C. Final Position Summary",
        dataframe_to_markdown(final_position_summary.drop(columns=["method_key"])),
        "",
        "## D. Candidate Space Summary for ratio=0.5",
        dataframe_to_markdown(candidate_space),
        "",
        "## E. Compare Selected Positions Against ratio=1.0",
        dataframe_to_markdown(ratio1_exclusion),
        "",
        "## F. Training Dynamics",
        "",
    ]
    if training_dynamics.empty:
        lines.append("_No Shared Policy training dynamics were available._")
    else:
        for target_item in selected_targets:
            for method_key in TRAINING_METHOD_ORDER:
                subset = training_dynamics[
                    (training_dynamics["target_item"] == int(target_item))
                    & (training_dynamics["method_key"] == method_key)
                ]
                if subset.empty:
                    continue
                lines.append(f"### {subset.iloc[0]['method']} / target {target_item}")
                lines.append("")
                lines.append(
                    dataframe_to_markdown(
                        subset[
                            [
                                "outer_step",
                                "reward",
                                "baseline",
                                "advantage",
                                "mean_entropy",
                                "policy_loss",
                                "target_utility",
                                "avg_selected_position",
                                "median_selected_position",
                                "fraction_pos0",
                                "fraction_pos<=1",
                                "fraction_pos<=2",
                                "gt_drop",
                                "gt_penalty",
                            ]
                        ]
                    )
                )
                lines.append("")
    lines.extend(
        [
            "## G. Short Textual Summary",
            dataframe_to_markdown(textual_summary),
            "",
        ]
    )
    return "\n".join(lines)


def _load_position_artifacts_for_method(
    *,
    method_key: str,
    spec: MethodSpec,
    target_item: int,
) -> PositionArtifacts:
    target_dir = spec.run_root / "targets" / str(target_item)
    position_stats_path = _resolve_existing_file(
        target_dir / "position_stats.json",
        label=f"{method_key} position_stats for target {target_item}",
    )
    position_summary = summarize_position_stats_file(position_stats_path, top_n=5)
    if method_key == "prefix_nz":
        selected_positions_path = _resolve_existing_file(
            target_dir / "prefix_nonzero_when_possible_metadata.pkl",
            label=f"{method_key} selected_positions for target {target_item}",
        )
        raw_selected_positions = _load_pickle_file(
            selected_positions_path,
            label=f"{method_key} selected_positions for target {target_item}",
        )
        selected_positions = normalize_prefix_selected_positions(
            raw_selected_positions,
            path=selected_positions_path,
        )
        return PositionArtifacts(
            method_key=method_key,
            target_item=int(target_item),
            position_stats_path=position_stats_path,
            selected_positions_path=selected_positions_path,
            position_summary=position_summary,
            selected_positions=selected_positions,
        )

    selected_positions_path = _resolve_existing_file(
        target_dir / "position_opt" / "selected_positions.json",
        label=f"{method_key} selected_positions for target {target_item}",
    )
    selected_positions = normalize_posopt_selected_positions(
        _load_json_sequence(
            selected_positions_path,
            label=f"{method_key} selected_positions for target {target_item}",
        ),
        path=selected_positions_path,
    )
    training_history_path = target_dir / "position_opt" / "training_history.json"
    training_summary = None
    resolved_training_history_path = None
    if training_history_path.is_file():
        resolved_training_history_path = training_history_path.resolve()
        training_summary = summarize_position_collapse_file(training_history_path, top_n=5)
    return PositionArtifacts(
        method_key=method_key,
        target_item=int(target_item),
        position_stats_path=position_stats_path,
        selected_positions_path=selected_positions_path,
        position_summary=position_summary,
        selected_positions=selected_positions,
        training_history_path=resolved_training_history_path,
        training_summary=training_summary,
    )


def _load_shared_fake_sessions(path: Path) -> list[list[int]]:
    payload = _load_pickle_file(path, label="shared fake sessions")
    return normalize_fake_sessions(payload, path=path)


def _lookup_metrics(
    *,
    summary_payload: dict[str, Any],
    target_item: int,
    victim_model: str,
) -> dict[str, Any] | None:
    targets_payload = _require_mapping(summary_payload.get("targets"), "summary_current.targets")
    target_payload = targets_payload.get(str(target_item))
    if not isinstance(target_payload, dict):
        return None
    victims_payload = _require_mapping(target_payload.get("victims"), "target.victims")
    victim_payload = victims_payload.get(victim_model)
    if not isinstance(victim_payload, dict):
        return None
    metrics = victim_payload.get("metrics")
    return metrics if isinstance(metrics, dict) else None


def _summary_has_target(summary_payload: dict[str, Any], target_item: int) -> bool:
    targets_payload = _require_mapping(summary_payload.get("targets"), "summary_current.targets")
    return str(target_item) in targets_payload


def _select_targets(
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
) -> tuple[int, ...]:
    ratio05_targets = {
        int(target_key)
        for target_key in _require_mapping(
            runtimes["shared_policy_ratio05"].summary_payload.get("targets"),
            "shared_policy_ratio05 summary_current.targets",
        ).keys()
    }
    selected: list[int] = []
    for target_item in manifest.required_targets:
        if int(target_item) not in ratio05_targets:
            raise DiagnosisError(
                f"Required target {target_item} is missing from shared_policy_ratio05 summary_current."
            )
        selected.append(int(target_item))
    for target_item in manifest.optional_targets:
        if int(target_item) in ratio05_targets:
            selected.append(int(target_item))
    return tuple(selected)


def _closest_method_label(
    *,
    ratio05_metrics: dict[str, float | None] | None,
    candidates: dict[str, dict[str, float | None] | None],
) -> str:
    if ratio05_metrics is None:
        return "No ratio=0.5 metrics available."
    distances = []
    for label, metrics in candidates.items():
        distance = _metric_distance(ratio05_metrics, metrics)
        if distance is not None:
            distances.append((distance, label))
    if not distances:
        return "No comparison method available."
    distances.sort(key=lambda item: (item[0], item[1]))
    return distances[0][1]


def _overall_verdict(
    *,
    ratio05_metrics: dict[str, float | None] | None,
    ratio1_metrics: dict[str, float | None] | None,
    exclusion_row: pd.Series | None,
) -> str:
    target_summary = _summarize_metric_delta(
        ratio05_metrics,
        ratio1_metrics,
        TARGET_METRIC_COLUMNS,
    )
    gt_summary = _summarize_metric_delta(
        ratio05_metrics,
        ratio1_metrics,
        GT_METRIC_COLUMNS,
    )
    exclusion_text = (
        None
        if exclusion_row is None
        else str(exclusion_row["likely_ratio05_excludes_useful_positions"])
    )
    if target_summary.startswith("Improved") and not gt_summary.startswith("Hurt"):
        return "Promising"
    if target_summary.startswith("Hurt") and (
        gt_summary.startswith("Hurt") or exclusion_text == "Likely yes."
    ):
        return "Harmful"
    return "Inconclusive"


def _summarize_metric_delta(
    ratio05_metrics: dict[str, float | None] | None,
    comparison_metrics: dict[str, float | None] | None,
    metric_columns: tuple[str, ...],
) -> str:
    deltas = _collect_metric_deltas(ratio05_metrics, comparison_metrics, metric_columns)
    if not deltas:
        return "No overlapping metrics."
    positive = sum(delta > 0 for delta in deltas)
    negative = sum(delta < 0 for delta in deltas)
    mean_delta = sum(deltas) / len(deltas)
    if positive >= max(4, negative + 2):
        return f"Improved overall (mean delta {mean_delta:.6f})."
    if negative >= max(4, positive + 2):
        return f"Hurt overall (mean delta {mean_delta:.6f})."
    return f"Mixed / near-flat (mean delta {mean_delta:.6f})."


def _summarize_position_behavior(
    *,
    ratio05_position: pd.Series | None,
    ratio1_position: pd.Series | None,
) -> str:
    if ratio05_position is None or ratio1_position is None:
        return "No aligned position summaries available."
    dominant_delta = _subtract_optional(
        ratio05_position.get("dominant_pct"),
        ratio1_position.get("dominant_pct"),
    )
    unique_delta = _subtract_optional(
        ratio05_position.get("unique_positions"),
        ratio1_position.get("unique_positions"),
    )
    pos0_delta = _subtract_optional(
        ratio05_position.get("pos0_pct"),
        ratio1_position.get("pos0_pct"),
    )
    if (
        dominant_delta is not None
        and dominant_delta >= 5.0
        and unique_delta is not None
        and unique_delta <= -1.0
    ):
        return f"More extreme / collapsed (dominant_pct +{dominant_delta:.2f}, pos0_pct {_signed(pos0_delta)})."
    if (
        dominant_delta is not None
        and dominant_delta <= -5.0
        and unique_delta is not None
        and unique_delta >= 1.0
    ):
        return f"More stable / dispersed (dominant_pct {dominant_delta:.2f}, pos0_pct {_signed(pos0_delta)})."
    return f"Broadly similar (dominant_pct {_signed(dominant_delta)}, pos0_pct {_signed(pos0_delta)})."


def _assess_exclusion_usefulness(
    *,
    excluded_rate: float,
    ratio05_metrics: dict[str, float | None] | None,
    ratio1_metrics: dict[str, float | None] | None,
) -> str:
    if excluded_rate <= 0.0:
        return "No."
    target_deltas = _collect_metric_deltas(
        ratio05_metrics,
        ratio1_metrics,
        TARGET_METRIC_COLUMNS,
    )
    if target_deltas and sum(delta < 0 for delta in target_deltas) >= 4:
        return "Likely yes."
    if target_deltas and sum(delta > 0 for delta in target_deltas) >= 4:
        return "Not obviously."
    return "Possibly."


def _metric_distance(
    left: dict[str, float | None] | None,
    right: dict[str, float | None] | None,
) -> float | None:
    if left is None or right is None:
        return None
    distance = 0.0
    overlap = 0
    for metric_name, _ in FINAL_METRIC_SPECS:
        left_value = left.get(metric_name)
        right_value = right.get(metric_name)
        if left_value is None or right_value is None:
            continue
        distance += abs(float(left_value) - float(right_value))
        overlap += 1
    if overlap == 0:
        return None
    return float(distance)


def _collect_metric_deltas(
    left: dict[str, float | None] | None,
    right: dict[str, float | None] | None,
    metric_columns: tuple[str, ...],
) -> list[float]:
    if left is None or right is None:
        return []
    deltas: list[float] = []
    for metric_name in metric_columns:
        left_value = left.get(metric_name)
        right_value = right.get(metric_name)
        if left_value is None or right_value is None:
            continue
        deltas.append(float(left_value) - float(right_value))
    return deltas


def _format_top_position_rows(rows: list[dict[str, Any]], *, pct_key: str) -> str | None:
    if not rows:
        return None
    return ", ".join(
        f"{int(row['position'])}:{float(row[pct_key]):.2f}%"
        for row in rows[:5]
        if isinstance(row, dict)
    )


def _format_counter_pct_summary(
    counter: Counter[int],
    *,
    denominator: int,
    top_n: int,
) -> str:
    if denominator <= 0 or not counter:
        return "none"
    pairs = sorted(counter.items(), key=lambda item: (item[0]))
    rendered = [
        f"{int(position)}:{float(count / denominator * 100.0):.2f}%"
        for position, count in pairs[:top_n]
    ]
    if len(pairs) > top_n:
        rendered.append("...")
    return ", ".join(rendered)


def _extract_ratio_pct(cumulative: dict[str, Any], key: str) -> float | None:
    payload = cumulative.get(key)
    if not isinstance(payload, dict):
        return None
    return _optional_float(payload.get("ratio_pct"))


def _require_shared_ratio05(manifest: ManifestSpec) -> float:
    value = manifest.methods["shared_policy_ratio05"].replacement_topk_ratio
    if value is None:
        raise DiagnosisError("Manifest shared_policy_ratio05 is missing replacement_topk_ratio.")
    return float(value)


def _subtract_optional(left: Any, right: Any) -> float | None:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _signed(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.2f}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise DiagnosisError(f"Diagnosis manifest must be a mapping: {path}")
    return payload


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_existing_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiagnosisError(f"Invalid JSON in {label}: {resolved}") from exc
    if not isinstance(payload, dict):
        raise DiagnosisError(f"{label} must contain a JSON object: {resolved}")
    return payload


def _load_json_sequence(path: Path, *, label: str) -> list[Any]:
    resolved = _resolve_existing_file(path, label=label)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiagnosisError(f"Invalid JSON in {label}: {resolved}") from exc
    if not isinstance(payload, list):
        raise DiagnosisError(f"{label} must contain a JSON list: {resolved}")
    return payload


def _load_pickle_file(path: Path, *, label: str) -> Any:
    resolved = _resolve_existing_file(path, label=label)
    with resolved.open("rb") as handle:
        return pickle.load(handle)


def _resolve_repo_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    if not resolved.exists():
        raise DiagnosisError(f"Missing required {label}: {resolved}")
    return resolved.resolve()


def _resolve_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise DiagnosisError(f"Missing required {label}: {resolved}")
    return resolved


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosisError(f"Expected {label} to be a mapping.")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosisError(f"Expected {label} to be a non-empty string.")
    return value.strip()


def _parse_int_list(value: Any, label: str, *, required: bool = True) -> tuple[int, ...]:
    if value is None and not required:
        return tuple()
    if not isinstance(value, list):
        raise DiagnosisError(f"Expected {label} to be a list.")
    parsed: list[int] = []
    for item in value:
        try:
            parsed.append(int(item))
        except (TypeError, ValueError) as exc:
            raise DiagnosisError(f"Expected {label} to contain only integers.") from exc
    return tuple(parsed)


if __name__ == "__main__":
    main()
