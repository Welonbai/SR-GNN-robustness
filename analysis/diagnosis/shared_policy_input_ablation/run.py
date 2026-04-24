#!/usr/bin/env python3
"""Extract a Shared Policy input-ablation report from run artifacts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.diagnosis.prefix_vs_posopt.loaders import (  # noqa: E402
    DiagnosisError,
    REPO_ROOT,
    ensure_existing_file,
    load_json_file,
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


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis" / "diagnosis_outputs"
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
TRAINING_MARKDOWN_FULL_ROW_LIMIT = 120
TRAINING_MARKDOWN_PREVIEW_ROWS = 5
POSITION_OPT_SHARED_POLICY_METHOD = "position_opt_shared_policy"
POSITION_OPT_MVP_METHOD = "position_opt_mvp"
CORE_VERIFICATION_FIELDS = (
    "policy_feature_set",
    "active_item_features",
    "active_scalar_features",
    "policy_input_dim",
    "policy_embedding_dim",
    "policy_hidden_dim",
    "prefix_score_enabled",
)


@dataclass(frozen=True)
class VerificationExpectation:
    policy_feature_set: str | None = None
    active_item_features: tuple[str, ...] | None = None
    active_scalar_features: tuple[str, ...] | None = None
    prefix_score_enabled: bool | None = None


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    attack_method: str
    run_root: Path
    summary_current: Path
    expected: VerificationExpectation | None = None

    @property
    def is_shared_policy(self) -> bool:
        return self.attack_method == POSITION_OPT_SHARED_POLICY_METHOD


@dataclass(frozen=True)
class ManifestSpec:
    path: Path
    report_id: str
    dataset: str
    victim_model: str
    reference_method: str | None
    required_targets: tuple[int, ...]
    optional_targets: tuple[int, ...]
    methods: dict[str, MethodSpec]

    @property
    def method_order(self) -> tuple[str, ...]:
        return tuple(self.methods.keys())


@dataclass(frozen=True)
class MethodRuntime:
    spec: MethodSpec
    summary_payload: dict[str, Any]


@dataclass(frozen=True)
class MethodArtifacts:
    method_key: str
    target_item: int
    position_stats_path: Path | None = None
    position_summary: dict[str, Any] | None = None
    training_history_path: Path | None = None
    training_history_payload: dict[str, Any] | None = None
    training_summary: dict[str, Any] | None = None
    run_metadata_path: Path | None = None
    run_metadata: dict[str, Any] | None = None
    learned_logits_path: Path | None = None
    learned_logits: dict[str, Any] | None = None


@dataclass(frozen=True)
class VerificationObservation:
    source: str
    fields: dict[str, Any]
    missing_core_fields: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a Shared Policy input-ablation report from run artifacts.",
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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = run_report(
            config_path=Path(args.config),
            output_root=Path(args.output_root),
        )
        print(f"Wrote shared-policy input-ablation outputs to '{output_dir}'.")
    except DiagnosisError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    return 0


def run_report(*, config_path: Path, output_root: Path) -> Path:
    manifest = load_manifest(config_path)
    runtimes = {
        key: MethodRuntime(
            spec=spec,
            summary_payload=load_json_file(spec.summary_current, label=f"{key} summary_current"),
        )
        for key, spec in manifest.methods.items()
    }
    selected_targets = _select_targets(manifest, runtimes)
    artifacts = load_method_artifacts(manifest=manifest, selected_targets=selected_targets)

    final_metrics = build_final_metrics_table(
        manifest=manifest,
        runtimes=runtimes,
        selected_targets=selected_targets,
    )
    delta_vs_reference = build_delta_vs_reference_table(
        manifest=manifest,
        final_metrics=final_metrics,
        selected_targets=selected_targets,
    )
    final_position_summary = build_final_position_summary(
        manifest=manifest,
        artifacts=artifacts,
        selected_targets=selected_targets,
    )
    training_dynamics = build_training_dynamics_table(
        manifest=manifest,
        artifacts=artifacts,
        selected_targets=selected_targets,
    )
    training_final_step_summary = build_training_final_step_summary(
        manifest=manifest,
        artifacts=artifacts,
        selected_targets=selected_targets,
    )
    verification_summary = build_verification_summary_table(
        manifest=manifest,
        artifacts=artifacts,
        selected_targets=selected_targets,
    )

    output_dir = output_root.resolve() / manifest.report_id
    ensure_directory(output_dir)
    _write_outputs(
        output_dir=output_dir,
        manifest=manifest,
        runtimes=runtimes,
        selected_targets=selected_targets,
        final_metrics=final_metrics,
        delta_vs_reference=delta_vs_reference,
        final_position_summary=final_position_summary,
        training_dynamics=training_dynamics,
        training_final_step_summary=training_final_step_summary,
        verification_summary=verification_summary,
    )
    return output_dir


def load_manifest(path: Path) -> ManifestSpec:
    resolved_path = ensure_existing_file(path, label="diagnosis manifest")
    payload = _load_yaml_mapping(resolved_path)
    methods_payload = _require_mapping(payload.get("methods"), "methods")
    methods: dict[str, MethodSpec] = {}
    for key, raw_method in methods_payload.items():
        method_mapping = _require_mapping(raw_method, f"methods.{key}")
        attack_method = _require_string(method_mapping.get("attack_method"), f"methods.{key}.attack_method")
        expected = _parse_expected_verification(
            method_mapping,
            label=f"methods.{key}",
            attack_method=attack_method,
        )
        methods[str(key)] = MethodSpec(
            key=str(key),
            label=_require_string(method_mapping.get("label"), f"methods.{key}.label"),
            attack_method=attack_method,
            run_root=_resolve_repo_path(
                _require_string(method_mapping.get("run_root"), f"methods.{key}.run_root"),
                label=f"methods.{key}.run_root",
            ),
            summary_current=_resolve_repo_path(
                _require_string(method_mapping.get("summary_current"), f"methods.{key}.summary_current"),
                label=f"methods.{key}.summary_current",
            ),
            expected=expected,
        )
    reference_method = payload.get("reference_method")
    if reference_method is not None:
        reference_method = _require_string(reference_method, "reference_method")
        if reference_method not in methods:
            raise DiagnosisError(
                f"reference_method '{reference_method}' is not present in methods."
            )
    targets_payload = _require_mapping(payload.get("targets"), "targets")
    return ManifestSpec(
        path=resolved_path.resolve(),
        report_id=_require_string(payload.get("report_id"), "report_id"),
        dataset=_require_string(payload.get("dataset"), "dataset"),
        victim_model=_require_string(payload.get("victim_model"), "victim_model"),
        reference_method=reference_method,
        required_targets=_parse_int_list(targets_payload.get("required"), "targets.required"),
        optional_targets=_parse_int_list(
            targets_payload.get("optional_if_available"),
            "targets.optional_if_available",
            required=False,
        ),
        methods=methods,
    )


def load_method_artifacts(
    *,
    manifest: ManifestSpec,
    selected_targets: tuple[int, ...],
) -> dict[tuple[str, int], MethodArtifacts]:
    artifacts: dict[tuple[str, int], MethodArtifacts] = {}
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            artifacts[(method_key, target_item)] = _load_artifacts_for_method_target(
                method_key=method_key,
                spec=spec,
                target_item=target_item,
            )
    return artifacts


def build_final_metrics_table(
    *,
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            metrics = _lookup_metrics(
                summary_payload=runtimes[method_key].summary_payload,
                target_item=target_item,
                victim_model=manifest.victim_model,
            )
            row: dict[str, Any] = {
                "target_item": int(target_item),
                "method_key": method_key,
                "method": spec.label,
                "_method_order": method_order[method_key],
            }
            for output_key, metric_key in FINAL_METRIC_SPECS:
                row[output_key] = None if metrics is None else _optional_float(metrics.get(metric_key))
            rows.append(row)
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(columns=["target_item", "method_key", "method", *[name for name, _ in FINAL_METRIC_SPECS]])
    dataframe = dataframe.sort_values(["target_item", "_method_order"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_delta_vs_reference_table(
    *,
    manifest: ManifestSpec,
    final_metrics: pd.DataFrame,
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    columns = [
        "target_item",
        "method_key",
        "method",
        "reference_method_key",
        "reference_method",
        *[name for name, _ in FINAL_METRIC_SPECS],
    ]
    if manifest.reference_method is None:
        return pd.DataFrame(columns=columns)
    lookup = {
        (int(row.target_item), str(row.method_key)): row
        for _, row in final_metrics.iterrows()
    }
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    reference_key = manifest.reference_method
    reference_label = manifest.methods[reference_key].label
    for target_item in selected_targets:
        reference_row = lookup.get((int(target_item), reference_key))
        for method_key, spec in manifest.methods.items():
            if method_key == reference_key:
                continue
            method_row = lookup.get((int(target_item), method_key))
            row: dict[str, Any] = {
                "target_item": int(target_item),
                "method_key": method_key,
                "method": spec.label,
                "reference_method_key": reference_key,
                "reference_method": reference_label,
                "_method_order": method_order[method_key],
            }
            for metric_name, _ in FINAL_METRIC_SPECS:
                row[metric_name] = _subtract_optional(
                    None if method_row is None else method_row.get(metric_name),
                    None if reference_row is None else reference_row.get(metric_name),
                )
            rows.append(row)
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(columns=columns)
    dataframe = dataframe.sort_values(["target_item", "_method_order"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_final_position_summary(
    *,
    manifest: ManifestSpec,
    artifacts: dict[tuple[str, int], MethodArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            artifact = artifacts[(method_key, target_item)]
            summary = artifact.position_summary
            cumulative = {} if summary is None else dict(summary.get("cumulative_ratios", {}))
            max_share = None if summary is None else summary.get("max_position_share")
            rows.append(
                {
                    "target_item": int(target_item),
                    "method_key": method_key,
                    "method": spec.label,
                    "_method_order": method_order[method_key],
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
            )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "target_item",
                "method_key",
                "method",
                "total",
                "unique_positions",
                "dominant_position",
                "dominant_pct",
                "pos0_pct",
                "pos<=1_pct",
                "pos<=2_pct",
                "pos<=5_pct",
                "top5_positions",
            ]
        )
    dataframe = dataframe.sort_values(["target_item", "_method_order"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_training_dynamics_table(
    *,
    manifest: ManifestSpec,
    artifacts: dict[tuple[str, int], MethodArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            artifact = artifacts[(method_key, target_item)]
            if artifact.training_summary is None:
                continue
            for step in artifact.training_summary.get("step_summaries", []):
                rows.append(
                    {
                        "method_key": method_key,
                        "method": spec.label,
                        "target_item": int(target_item),
                        "_method_order": method_order[method_key],
                        "outer_step": int(step["outer_step"]),
                        "reward": _optional_float(step.get("reward")),
                        "baseline": _optional_float(step.get("baseline")),
                        "advantage": _optional_float(step.get("advantage")),
                        "mean_entropy": _optional_float(step.get("mean_entropy")),
                        "policy_loss": _optional_float(step.get("policy_loss")),
                        "target_utility": _optional_float(step.get("target_utility")),
                        "avg_selected_position": _optional_float(step.get("average_selected_position")),
                        "median_selected_position": _optional_float(step.get("median_selected_position")),
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
    dataframe = dataframe.sort_values(["target_item", "_method_order", "outer_step"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_training_final_step_summary(
    *,
    manifest: ManifestSpec,
    artifacts: dict[tuple[str, int], MethodArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            artifact = artifacts[(method_key, target_item)]
            if artifact.training_summary is None:
                continue
            step_rows = artifact.training_summary.get("step_summaries", [])
            if not step_rows:
                continue
            final_row = step_rows[-1]
            final_dominant = artifact.training_summary.get("final_sampled_dominant")
            rows.append(
                {
                    "method_key": method_key,
                    "method": spec.label,
                    "target_item": int(target_item),
                    "_method_order": method_order[method_key],
                    "outer_step": int(final_row["outer_step"]),
                    "reward": _optional_float(final_row.get("reward")),
                    "baseline": _optional_float(final_row.get("baseline")),
                    "advantage": _optional_float(final_row.get("advantage")),
                    "mean_entropy": _optional_float(final_row.get("mean_entropy")),
                    "policy_loss": _optional_float(final_row.get("policy_loss")),
                    "target_utility": _optional_float(final_row.get("target_utility")),
                    "avg_selected_position": _optional_float(final_row.get("average_selected_position")),
                    "median_selected_position": _optional_float(final_row.get("median_selected_position")),
                    "fraction_pos0": _optional_float(final_row.get("fraction_pos0")),
                    "fraction_pos<=1": _optional_float(final_row.get("fraction_pos_le_1")),
                    "fraction_pos<=2": _optional_float(final_row.get("fraction_pos_le_2")),
                    "gt_drop": _optional_float(final_row.get("gt_drop")),
                    "gt_penalty": _optional_float(final_row.get("gt_penalty")),
                    "entropy_drop": _optional_float(artifact.training_summary.get("entropy_drop")),
                    "final_dominant_position": (
                        None if not isinstance(final_dominant, dict) else int(final_dominant["dominant_position"])
                    ),
                    "final_dominant_share_pct": (
                        None
                        if not isinstance(final_dominant, dict)
                        else _optional_float(final_dominant.get("dominant_share_pct"))
                    ),
                    "final_unique_positions": (
                        None if not isinstance(final_dominant, dict) else int(final_dominant["unique_positions"])
                    ),
                }
            )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "method_key",
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
                "entropy_drop",
                "final_dominant_position",
                "final_dominant_share_pct",
                "final_unique_positions",
            ]
        )
    dataframe = dataframe.sort_values(["target_item", "_method_order"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_verification_summary_table(
    *,
    manifest: ManifestSpec,
    artifacts: dict[tuple[str, int], MethodArtifacts],
    selected_targets: tuple[int, ...],
) -> pd.DataFrame:
    method_order = _method_order_lookup(manifest)
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            if not spec.is_shared_policy:
                continue
            artifact = artifacts[(method_key, target_item)]
            run_metadata_obs = _extract_run_metadata_observation(artifact.run_metadata)
            training_history_obs = _extract_training_history_observation(artifact.training_history_payload)
            learned_logits_obs = _extract_learned_logits_observation(artifact.learned_logits)
            merged_fields = _merge_observation_fields(
                run_metadata_obs,
                training_history_obs,
                learned_logits_obs,
            )
            expected = spec.expected
            expected_policy_feature_set = None if expected is None else expected.policy_feature_set
            expected_active_item_features = (
                None if expected is None or expected.active_item_features is None else list(expected.active_item_features)
            )
            expected_active_scalar_features = (
                None if expected is None or expected.active_scalar_features is None else list(expected.active_scalar_features)
            )
            expected_prefix_score_enabled = None if expected is None else expected.prefix_score_enabled
            expected_input_dim_from_features = _expected_input_dim_from_features(
                expected_active_item_features=expected_active_item_features,
                expected_active_scalar_features=expected_active_scalar_features,
                policy_embedding_dim=merged_fields.get("policy_embedding_dim"),
            )
            artifact_consistency_status = _artifact_consistency_status(
                run_metadata_obs,
                training_history_obs,
                learned_logits_obs,
            )
            feature_set_matches_expected = _match_optional_string(
                merged_fields.get("policy_feature_set"),
                expected_policy_feature_set,
            )
            active_features_match_expected = _match_optional_feature_lists(
                merged_fields.get("active_item_features"),
                merged_fields.get("active_scalar_features"),
                expected_active_item_features,
                expected_active_scalar_features,
            )
            input_dim_matches_expected = _match_optional_int(
                merged_fields.get("policy_input_dim"),
                expected_input_dim_from_features,
            )
            prefix_flag_matches_expected = _match_optional_bool(
                merged_fields.get("prefix_score_enabled"),
                expected_prefix_score_enabled,
            )
            verification_status = _verification_status(
                run_metadata_present=artifact.run_metadata is not None,
                artifact_consistency_status=artifact_consistency_status,
                feature_set_matches_expected=feature_set_matches_expected,
                active_features_match_expected=active_features_match_expected,
                input_dim_matches_expected=input_dim_matches_expected,
                prefix_flag_matches_expected=prefix_flag_matches_expected,
            )
            rows.append(
                {
                    "target_item": int(target_item),
                    "method_key": method_key,
                    "method": spec.label,
                    "_method_order": method_order[method_key],
                    "policy_feature_set": merged_fields.get("policy_feature_set"),
                    "active_item_features": merged_fields.get("active_item_features"),
                    "active_scalar_features": merged_fields.get("active_scalar_features"),
                    "policy_input_dim": merged_fields.get("policy_input_dim"),
                    "policy_embedding_dim": merged_fields.get("policy_embedding_dim"),
                    "policy_hidden_dim": merged_fields.get("policy_hidden_dim"),
                    "prefix_score_enabled": merged_fields.get("prefix_score_enabled"),
                    "run_metadata_present": bool(artifact.run_metadata is not None),
                    "training_history_present": bool(artifact.training_history_payload is not None),
                    "learned_logits_present": bool(artifact.learned_logits is not None),
                    "expected_policy_feature_set": expected_policy_feature_set,
                    "expected_active_item_features": expected_active_item_features,
                    "expected_active_scalar_features": expected_active_scalar_features,
                    "expected_input_dim_from_features": expected_input_dim_from_features,
                    "feature_set_matches_expected": feature_set_matches_expected,
                    "active_features_match_expected": active_features_match_expected,
                    "input_dim_matches_expected": input_dim_matches_expected,
                    "prefix_flag_matches_expected": prefix_flag_matches_expected,
                    "artifact_consistency_status": artifact_consistency_status,
                    "verification_status": verification_status,
                }
            )
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "target_item",
                "method_key",
                "method",
                "policy_feature_set",
                "active_item_features",
                "active_scalar_features",
                "policy_input_dim",
                "policy_embedding_dim",
                "policy_hidden_dim",
                "prefix_score_enabled",
                "run_metadata_present",
                "training_history_present",
                "learned_logits_present",
                "expected_policy_feature_set",
                "expected_active_item_features",
                "expected_active_scalar_features",
                "expected_input_dim_from_features",
                "feature_set_matches_expected",
                "active_features_match_expected",
                "input_dim_matches_expected",
                "prefix_flag_matches_expected",
                "artifact_consistency_status",
                "verification_status",
            ]
        )
    dataframe = dataframe.sort_values(["target_item", "_method_order"]).reset_index(drop=True)
    return dataframe.drop(columns=["_method_order"])


def build_markdown_report(
    *,
    manifest: ManifestSpec,
    selected_targets: tuple[int, ...],
    final_metrics: pd.DataFrame,
    delta_vs_reference: pd.DataFrame,
    final_position_summary: pd.DataFrame,
    verification_summary: pd.DataFrame,
    training_final_step_summary: pd.DataFrame,
    training_dynamics: pd.DataFrame,
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
    ]
    if manifest.reference_method is not None:
        lines.append(f"- Reference method: `{manifest.methods[manifest.reference_method].label}`")
    lines.extend(
        [
            "",
            "## A. Final Metrics Comparison",
            dataframe_to_markdown(final_metrics.drop(columns=["method_key"], errors="ignore")),
            "",
            "## B. Delta vs Reference",
            dataframe_to_markdown(
                delta_vs_reference.drop(columns=["method_key", "reference_method_key"], errors="ignore")
            ),
            "",
            "## C. Final Position Summary",
            dataframe_to_markdown(final_position_summary.drop(columns=["method_key"], errors="ignore")),
            "",
            "## D. Verification Summary",
            dataframe_to_markdown(verification_summary.drop(columns=["method_key"], errors="ignore")),
            "",
            "## E. Training Final-Step Summary",
            dataframe_to_markdown(training_final_step_summary.drop(columns=["method_key"], errors="ignore")),
            "",
            "## F. Training Dynamics",
            "",
        ]
    )
    lines.extend(
        _render_training_dynamics_markdown(
            manifest=manifest,
            selected_targets=selected_targets,
            training_dynamics=training_dynamics,
        )
    )
    return "\n".join(lines)


def _write_outputs(
    *,
    output_dir: Path,
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
    selected_targets: tuple[int, ...],
    final_metrics: pd.DataFrame,
    delta_vs_reference: pd.DataFrame,
    final_position_summary: pd.DataFrame,
    training_dynamics: pd.DataFrame,
    training_final_step_summary: pd.DataFrame,
    verification_summary: pd.DataFrame,
) -> None:
    write_dataframe(final_metrics, output_dir / "final_metrics.csv")
    write_dataframe(delta_vs_reference, output_dir / "delta_vs_reference.csv")
    write_dataframe(final_position_summary, output_dir / "final_position_summary.csv")
    write_dataframe(training_dynamics, output_dir / "training_dynamics.csv")
    write_dataframe(training_final_step_summary, output_dir / "training_final_step_summary.csv")
    write_dataframe(verification_summary, output_dir / "verification_summary.csv")
    write_json(
        {
            "report_id": manifest.report_id,
            "config_path": repo_relative(manifest.path),
            "dataset": manifest.dataset,
            "victim_model": manifest.victim_model,
            "reference_method": manifest.reference_method,
            "selected_targets": list(selected_targets),
            "methods": {
                key: {
                    "label": runtime.spec.label,
                    "attack_method": runtime.spec.attack_method,
                    "run_root": repo_relative(runtime.spec.run_root),
                    "summary_current": repo_relative(runtime.spec.summary_current),
                    "expected_policy_feature_set": (
                        None if runtime.spec.expected is None else runtime.spec.expected.policy_feature_set
                    ),
                    "expected_active_item_features": (
                        None
                        if runtime.spec.expected is None or runtime.spec.expected.active_item_features is None
                        else list(runtime.spec.expected.active_item_features)
                    ),
                    "expected_active_scalar_features": (
                        None
                        if runtime.spec.expected is None or runtime.spec.expected.active_scalar_features is None
                        else list(runtime.spec.expected.active_scalar_features)
                    ),
                    "expected_prefix_score_enabled": (
                        None if runtime.spec.expected is None else runtime.spec.expected.prefix_score_enabled
                    ),
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
                "delta_vs_reference": dataframe_to_records(delta_vs_reference),
                "final_position_summary": dataframe_to_records(final_position_summary),
                "training_dynamics": dataframe_to_records(training_dynamics),
                "training_final_step_summary": dataframe_to_records(training_final_step_summary),
                "verification_summary": dataframe_to_records(verification_summary),
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
            "reference_method": manifest.reference_method,
            "required_targets": list(manifest.required_targets),
            "optional_targets": list(manifest.optional_targets),
            "selected_targets": list(selected_targets),
            "methods": {
                key: {
                    "label": spec.label,
                    "attack_method": spec.attack_method,
                    "run_root": repo_relative(spec.run_root),
                    "summary_current": repo_relative(spec.summary_current),
                    "expected_policy_feature_set": (
                        None if spec.expected is None else spec.expected.policy_feature_set
                    ),
                    "expected_active_item_features": (
                        None
                        if spec.expected is None or spec.expected.active_item_features is None
                        else list(spec.expected.active_item_features)
                    ),
                    "expected_active_scalar_features": (
                        None
                        if spec.expected is None or spec.expected.active_scalar_features is None
                        else list(spec.expected.active_scalar_features)
                    ),
                    "expected_prefix_score_enabled": (
                        None if spec.expected is None else spec.expected.prefix_score_enabled
                    ),
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
            delta_vs_reference=delta_vs_reference,
            final_position_summary=final_position_summary,
            verification_summary=verification_summary,
            training_final_step_summary=training_final_step_summary,
            training_dynamics=training_dynamics,
        ),
        output_dir / "report.md",
    )


def _render_training_dynamics_markdown(
    *,
    manifest: ManifestSpec,
    selected_targets: tuple[int, ...],
    training_dynamics: pd.DataFrame,
) -> list[str]:
    if training_dynamics.empty:
        return ["_No training dynamics were available._", ""]

    preview_only = len(training_dynamics) > TRAINING_MARKDOWN_FULL_ROW_LIMIT
    lines: list[str] = []
    if preview_only:
        lines.append(
            f"Full training dynamics are available in `training_dynamics.csv`; "
            f"the tables below show the first {TRAINING_MARKDOWN_PREVIEW_ROWS} rows per method-target pair."
        )
        lines.append("")
    for target_item in selected_targets:
        for method_key, spec in manifest.methods.items():
            subset = training_dynamics[
                (training_dynamics["target_item"] == int(target_item))
                & (training_dynamics["method_key"] == method_key)
            ]
            if subset.empty:
                continue
            lines.append(f"### {spec.label} / target {target_item}")
            lines.append("")
            rendered = subset[
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
            lines.append(
                dataframe_to_markdown(
                    rendered,
                    max_rows=(TRAINING_MARKDOWN_PREVIEW_ROWS if preview_only else None),
                )
            )
            lines.append("")
    return lines


def _load_artifacts_for_method_target(
    *,
    method_key: str,
    spec: MethodSpec,
    target_item: int,
) -> MethodArtifacts:
    target_dir = spec.run_root / "targets" / str(target_item)
    position_stats_path = target_dir / "position_stats.json"
    training_history_path = target_dir / "position_opt" / "training_history.json"
    run_metadata_path = target_dir / "position_opt" / "run_metadata.json"
    learned_logits_path = target_dir / "position_opt" / "learned_logits.pt"

    position_summary = None
    if position_stats_path.is_file():
        position_summary = summarize_position_stats_file(position_stats_path, top_n=5)

    training_history_payload = None
    training_summary = None
    if training_history_path.is_file():
        training_history_payload = load_json_file(
            training_history_path,
            label=f"{method_key} training_history for target {target_item}",
        )
        training_summary = summarize_position_collapse_file(training_history_path, top_n=5)

    run_metadata = None
    if run_metadata_path.is_file():
        run_metadata = load_json_file(
            run_metadata_path,
            label=f"{method_key} run_metadata for target {target_item}",
        )

    learned_logits = None
    if learned_logits_path.is_file():
        learned_logits = _load_torch_mapping(
            learned_logits_path,
            label=f"{method_key} learned_logits for target {target_item}",
        )

    return MethodArtifacts(
        method_key=method_key,
        target_item=int(target_item),
        position_stats_path=(position_stats_path.resolve() if position_stats_path.is_file() else None),
        position_summary=position_summary,
        training_history_path=(training_history_path.resolve() if training_history_path.is_file() else None),
        training_history_payload=training_history_payload,
        training_summary=training_summary,
        run_metadata_path=(run_metadata_path.resolve() if run_metadata_path.is_file() else None),
        run_metadata=run_metadata,
        learned_logits_path=(learned_logits_path.resolve() if learned_logits_path.is_file() else None),
        learned_logits=learned_logits,
    )


def _extract_run_metadata_observation(payload: dict[str, Any] | None) -> VerificationObservation | None:
    if payload is None:
        return None
    trainer_result = _optional_mapping(payload.get("trainer_result"))
    position_opt_config = _optional_mapping(payload.get("position_opt_config"))
    fields = {
        "policy_feature_set": _normalize_optional_string(
            _first_not_none(
                payload.get("policy_feature_set"),
                None if trainer_result is None else trainer_result.get("policy_feature_set"),
                None if position_opt_config is None else position_opt_config.get("policy_feature_set"),
            )
        ),
        "active_item_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_item_features"),
                payload.get("policy_item_feature_names"),
                None if trainer_result is None else trainer_result.get("active_item_features"),
                None if trainer_result is None else trainer_result.get("policy_item_feature_names"),
            )
        ),
        "active_scalar_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_scalar_features"),
                payload.get("policy_scalar_feature_names"),
                None if trainer_result is None else trainer_result.get("active_scalar_features"),
                None if trainer_result is None else trainer_result.get("policy_scalar_feature_names"),
            )
        ),
        "policy_input_dim": _normalize_optional_int(
            _first_not_none(
                payload.get("policy_input_dim"),
                None if trainer_result is None else trainer_result.get("policy_input_dim"),
            )
        ),
        "policy_embedding_dim": _normalize_optional_int(
            _first_not_none(
                payload.get("policy_embedding_dim"),
                None if trainer_result is None else trainer_result.get("policy_embedding_dim"),
                None if position_opt_config is None else position_opt_config.get("policy_embedding_dim"),
            )
        ),
        "policy_hidden_dim": _normalize_optional_int(
            _first_not_none(
                payload.get("policy_hidden_dim"),
                None if trainer_result is None else trainer_result.get("policy_hidden_dim"),
                None if position_opt_config is None else position_opt_config.get("policy_hidden_dim"),
            )
        ),
        "prefix_score_enabled": _normalize_optional_bool(
            _first_not_none(
                payload.get("prefix_score_enabled"),
                None if trainer_result is None else trainer_result.get("prefix_score_enabled"),
            )
        ),
    }
    return VerificationObservation(
        source="run_metadata",
        fields=fields,
        missing_core_fields=_missing_core_fields(fields),
    )


def _extract_training_history_observation(payload: dict[str, Any] | None) -> VerificationObservation | None:
    if payload is None:
        return None
    position_opt_config = _optional_mapping(payload.get("position_opt_config"))
    fields = {
        "policy_feature_set": _normalize_optional_string(
            _first_not_none(
                payload.get("policy_feature_set"),
                None if position_opt_config is None else position_opt_config.get("policy_feature_set"),
            )
        ),
        "active_item_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_item_features"),
                payload.get("policy_item_feature_names"),
            )
        ),
        "active_scalar_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_scalar_features"),
                payload.get("policy_scalar_feature_names"),
            )
        ),
        "policy_input_dim": _normalize_optional_int(payload.get("policy_input_dim")),
        "policy_embedding_dim": _normalize_optional_int(
            _first_not_none(
                payload.get("policy_embedding_dim"),
                None if position_opt_config is None else position_opt_config.get("policy_embedding_dim"),
            )
        ),
        "policy_hidden_dim": _normalize_optional_int(
            _first_not_none(
                payload.get("policy_hidden_dim"),
                None if position_opt_config is None else position_opt_config.get("policy_hidden_dim"),
            )
        ),
        "prefix_score_enabled": _normalize_optional_bool(payload.get("prefix_score_enabled")),
    }
    return VerificationObservation(
        source="training_history",
        fields=fields,
        missing_core_fields=_missing_core_fields(fields),
    )


def _extract_learned_logits_observation(payload: dict[str, Any] | None) -> VerificationObservation | None:
    if payload is None:
        return None
    policy_config = _optional_mapping(payload.get("policy_config"))
    fields = {
        "policy_feature_set": _normalize_optional_string(
            _first_not_none(
                payload.get("policy_feature_set"),
                None if policy_config is None else policy_config.get("policy_feature_set"),
            )
        ),
        "active_item_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_item_features"),
                payload.get("policy_item_feature_names"),
                None if policy_config is None else policy_config.get("active_item_features"),
                None if policy_config is None else policy_config.get("item_feature_names"),
            )
        ),
        "active_scalar_features": _normalize_optional_string_list(
            _first_not_none(
                payload.get("active_scalar_features"),
                payload.get("policy_scalar_feature_names"),
                None if policy_config is None else policy_config.get("active_scalar_features"),
                None if policy_config is None else policy_config.get("scalar_feature_names"),
            )
        ),
        "policy_input_dim": _normalize_optional_int(
            None if policy_config is None else policy_config.get("policy_input_dim")
        ),
        "policy_embedding_dim": _normalize_optional_int(
            _first_not_none(
                None if policy_config is None else policy_config.get("policy_embedding_dim"),
                None if policy_config is None else policy_config.get("embedding_dim"),
            )
        ),
        "policy_hidden_dim": _normalize_optional_int(
            _first_not_none(
                None if policy_config is None else policy_config.get("policy_hidden_dim"),
                None if policy_config is None else policy_config.get("hidden_dim"),
            )
        ),
        "prefix_score_enabled": _normalize_optional_bool(payload.get("prefix_score_enabled")),
    }
    return VerificationObservation(
        source="learned_logits",
        fields=fields,
        missing_core_fields=_missing_core_fields(fields),
    )


def _merge_observation_fields(*observations: VerificationObservation | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for field_name in CORE_VERIFICATION_FIELDS:
        merged[field_name] = _first_not_none(
            *[
                None if observation is None else observation.fields.get(field_name)
                for observation in observations
            ]
        )
    return merged


def _artifact_consistency_status(
    *observations: VerificationObservation | None,
) -> str:
    present = [observation for observation in observations if observation is not None]
    if not present:
        return "legacy_missing_fields"
    comparable = [
        observation
        for observation in present
        if not observation.missing_core_fields
    ]
    if len(comparable) >= 2:
        baseline = comparable[0].fields
        for observation in comparable[1:]:
            if any(
                observation.fields.get(field_name) != baseline.get(field_name)
                for field_name in CORE_VERIFICATION_FIELDS
            ):
                return "mismatch"
    if any(observation.missing_core_fields for observation in present):
        return "legacy_missing_fields"
    return "ok"


def _verification_status(
    *,
    run_metadata_present: bool,
    artifact_consistency_status: str,
    feature_set_matches_expected: bool | None,
    active_features_match_expected: bool | None,
    input_dim_matches_expected: bool | None,
    prefix_flag_matches_expected: bool | None,
) -> str:
    if not run_metadata_present:
        return "missing_artifacts"
    if artifact_consistency_status == "mismatch":
        return "mismatch"
    match_flags = (
        feature_set_matches_expected,
        active_features_match_expected,
        input_dim_matches_expected,
        prefix_flag_matches_expected,
    )
    if any(flag is False for flag in match_flags):
        return "mismatch"
    if artifact_consistency_status == "legacy_missing_fields":
        return "legacy_missing_fields"
    return "ok"


def _load_torch_mapping(path: Path, *, label: str) -> dict[str, Any]:
    resolved = ensure_existing_file(path, label=label)
    try:
        payload = torch.load(resolved, map_location="cpu")
    except Exception as exc:  # pragma: no cover - torch raises several exception types here
        raise DiagnosisError(f"Unable to load {label}: {resolved}") from exc
    if not isinstance(payload, dict):
        raise DiagnosisError(f"{label} must contain a mapping payload: {resolved}")
    return payload


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


def _select_targets(
    manifest: ManifestSpec,
    runtimes: dict[str, MethodRuntime],
) -> tuple[int, ...]:
    available_targets = {
        key: {
            int(target_key)
            for target_key in _require_mapping(
                runtime.summary_payload.get("targets"),
                f"{key} summary_current.targets",
            ).keys()
        }
        for key, runtime in runtimes.items()
    }
    if manifest.reference_method is not None:
        gating_targets = available_targets[manifest.reference_method]
        missing_required = [
            int(target_item)
            for target_item in manifest.required_targets
            if int(target_item) not in gating_targets
        ]
        if missing_required:
            raise DiagnosisError(
                "Required targets are missing from the reference method summary_current: "
                + ", ".join(str(target) for target in missing_required)
            )
        selected = list(manifest.required_targets)
        for target_item in manifest.optional_targets:
            if int(target_item) in gating_targets and int(target_item) not in selected:
                selected.append(int(target_item))
        return tuple(selected)

    union_targets: set[int] = set()
    for target_set in available_targets.values():
        union_targets.update(target_set)
    missing_required = [
        int(target_item)
        for target_item in manifest.required_targets
        if int(target_item) not in union_targets
    ]
    if missing_required:
        raise DiagnosisError(
            "Required targets are missing from all methods: "
            + ", ".join(str(target) for target in missing_required)
        )
    selected = list(manifest.required_targets)
    for target_item in manifest.optional_targets:
        if int(target_item) in union_targets and int(target_item) not in selected:
            selected.append(int(target_item))
    return tuple(selected)


def _parse_expected_verification(
    method_mapping: dict[str, Any],
    *,
    label: str,
    attack_method: str,
) -> VerificationExpectation | None:
    expected_keys = (
        "expected_policy_feature_set",
        "expected_active_item_features",
        "expected_active_scalar_features",
        "expected_prefix_score_enabled",
    )
    if not any(key in method_mapping for key in expected_keys):
        return None
    if attack_method != POSITION_OPT_SHARED_POLICY_METHOD:
        raise DiagnosisError(
            f"{label} defines shared-policy verification expectations but attack_method is "
            f"'{attack_method}', not '{POSITION_OPT_SHARED_POLICY_METHOD}'."
        )
    return VerificationExpectation(
        policy_feature_set=(
            None
            if method_mapping.get("expected_policy_feature_set") is None
            else _require_string(
                method_mapping.get("expected_policy_feature_set"),
                f"{label}.expected_policy_feature_set",
            )
        ),
        active_item_features=_parse_string_list(
            method_mapping.get("expected_active_item_features"),
            f"{label}.expected_active_item_features",
            required=False,
        ),
        active_scalar_features=_parse_string_list(
            method_mapping.get("expected_active_scalar_features"),
            f"{label}.expected_active_scalar_features",
            required=False,
        ),
        prefix_score_enabled=_parse_optional_bool(
            method_mapping.get("expected_prefix_score_enabled"),
            f"{label}.expected_prefix_score_enabled",
        ),
    )


def _method_order_lookup(manifest: ManifestSpec) -> dict[str, int]:
    return {method_key: index for index, method_key in enumerate(manifest.method_order)}


def _expected_input_dim_from_features(
    *,
    expected_active_item_features: list[str] | None,
    expected_active_scalar_features: list[str] | None,
    policy_embedding_dim: Any,
) -> int | None:
    if expected_active_item_features is None or expected_active_scalar_features is None:
        return None
    if not expected_active_item_features:
        return int(len(expected_active_scalar_features))
    normalized_embedding_dim = _normalize_optional_int(policy_embedding_dim)
    if normalized_embedding_dim is None:
        return None
    return int(len(expected_active_item_features) * normalized_embedding_dim + len(expected_active_scalar_features))


def _match_optional_string(actual: Any, expected: str | None) -> bool | None:
    if expected is None:
        return None
    normalized_actual = _normalize_optional_string(actual)
    if normalized_actual is None:
        return None
    return normalized_actual == expected


def _match_optional_feature_lists(
    actual_items: Any,
    actual_scalars: Any,
    expected_items: list[str] | None,
    expected_scalars: list[str] | None,
) -> bool | None:
    if expected_items is None or expected_scalars is None:
        return None
    normalized_actual_items = _normalize_optional_string_list(actual_items)
    normalized_actual_scalars = _normalize_optional_string_list(actual_scalars)
    if normalized_actual_items is None or normalized_actual_scalars is None:
        return None
    return (
        normalized_actual_items == expected_items
        and normalized_actual_scalars == expected_scalars
    )


def _match_optional_int(actual: Any, expected: int | None) -> bool | None:
    if expected is None:
        return None
    normalized_actual = _normalize_optional_int(actual)
    if normalized_actual is None:
        return None
    return normalized_actual == int(expected)


def _match_optional_bool(actual: Any, expected: bool | None) -> bool | None:
    if expected is None:
        return None
    normalized_actual = _normalize_optional_bool(actual)
    if normalized_actual is None:
        return None
    return normalized_actual == bool(expected)


def _missing_core_fields(fields: dict[str, Any]) -> bool:
    return any(fields.get(field_name) is None for field_name in CORE_VERIFICATION_FIELDS)


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    return None if not stripped else stripped


def _normalize_optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[str] = []
    for item in value:
        item_value = _normalize_optional_string(item)
        if item_value is None:
            return None
        normalized.append(item_value)
    return normalized


def _normalize_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _format_top_position_rows(rows: list[dict[str, Any]], *, pct_key: str) -> str | None:
    if not rows:
        return None
    return ", ".join(
        f"{int(row['position'])}:{float(row[pct_key]):.2f}%"
        for row in rows[:5]
        if isinstance(row, dict)
    )


def _extract_ratio_pct(cumulative: dict[str, Any], key: str) -> float | None:
    payload = cumulative.get(key)
    if not isinstance(payload, dict):
        return None
    return _optional_float(payload.get("ratio_pct"))


def _subtract_optional(left: Any, right: Any) -> float | None:
    left_value = _optional_float(left)
    right_value = _optional_float(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


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


def _resolve_repo_path(raw_path: str, *, label: str) -> Path:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else (REPO_ROOT / path)
    if not resolved.exists():
        raise DiagnosisError(f"Missing required {label}: {resolved}")
    return resolved.resolve()


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


def _parse_string_list(value: Any, label: str, *, required: bool = True) -> tuple[str, ...] | None:
    if value is None and not required:
        return None
    if not isinstance(value, list):
        raise DiagnosisError(f"Expected {label} to be a list.")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise DiagnosisError(f"Expected {label} to contain only non-empty strings.")
        parsed.append(item.strip())
    return tuple(parsed)


def _parse_optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    parsed = _normalize_optional_bool(value)
    if parsed is None:
        raise DiagnosisError(f"Expected {label} to be a boolean.")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
