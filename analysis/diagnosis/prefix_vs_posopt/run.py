#!/usr/bin/env python3
"""CLI entry point for Prefix vs PosOptMVP diagnosis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.diagnosis.prefix_vs_posopt.compute import (
    MetricPair,
    build_context_join_dataframe,
    build_metrics_comparison_dataframe,
    build_per_session_join_dataframe,
    build_top_context_table,
    build_training_dynamics_dataframe,
    summarize_context,
    summarize_positions,
    summarize_training_dynamics,
)
from analysis.diagnosis.prefix_vs_posopt.loaders import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POSOPT_RUN_GROUP,
    DEFAULT_PREFIX_RUN_GROUP,
    DEFAULT_SHARED_FAKE_SESSIONS,
    DiagnosisError,
    REPO_ROOT,
    load_posopt_target_artifacts,
    load_prefix_target_artifacts,
    load_shared_artifacts,
    load_victim_run_artifacts,
    repo_relative,
)
from analysis.diagnosis.prefix_vs_posopt.plots import (
    plot_normalized_position_histogram,
    plot_position_histogram,
    plot_targeted_metrics_overview,
    plot_training_curve,
)
from analysis.diagnosis.prefix_vs_posopt.writers import (
    dataframe_to_markdown,
    dataframe_to_records,
    ensure_directory,
    write_dataframe,
    write_json,
    write_report,
)


@dataclass(frozen=True)
class DiagnosisCase:
    """One deterministic diagnosis slice configuration."""

    case_id: str
    description: str
    output_dirname: str
    kind: str
    targets: tuple[int, ...]
    victims: tuple[str, ...]
    failure_target: int | None = None
    success_target: int | None = None


CASES: dict[str, DiagnosisCase] = {
    "tron_pair": DiagnosisCase(
        case_id="tron_pair",
        description=(
            "Same-victim case pair on TRON: failure case target 5334 where Prefix beats PosOptMVP, "
            "and success case target 11103 where PosOptMVP beats Prefix."
        ),
        output_dirname="tron_case_pair_5334_vs_11103",
        kind="same_victim_pair",
        targets=(5334, 11103),
        victims=("tron",),
        failure_target=5334,
        success_target=11103,
    ),
    "target5334_cross_victim": DiagnosisCase(
        case_id="target5334_cross_victim",
        description=(
            "Cross-victim slice for target 5334 on TRON, MIARec, and SR-GNN, using the shared "
            "Prefix and PosOptMVP target-level artifacts."
        ),
        output_dirname="target_5334_cross_victim",
        kind="cross_victim_target",
        targets=(5334,),
        victims=("tron", "miasrec", "srgnn"),
        failure_target=5334,
    ),
    "miasrec_pair": DiagnosisCase(
        case_id="miasrec_pair",
        description=(
            "Same-victim case pair on MIARec: failure case target 5334 where Prefix beats PosOptMVP, "
            "and success case target 39588 where PosOptMVP beats Prefix."
        ),
        output_dirname="miasrec_case_pair_5334_vs_39588",
        kind="same_victim_pair",
        targets=(5334, 39588),
        victims=("miasrec",),
        failure_target=5334,
        success_target=39588,
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the diagnosis CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run a deterministic Prefix vs PosOptMVP diagnosis slice.",
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=sorted(CASES),
        help="Diagnosis slice to generate.",
    )
    parser.add_argument(
        "--shared-fake-sessions",
        default=str(DEFAULT_SHARED_FAKE_SESSIONS),
        help="Path to the shared fake-session pool pickle.",
    )
    parser.add_argument(
        "--posopt-root",
        default=str(DEFAULT_POSOPT_RUN_GROUP),
        help="Path to the PosOpt run-group root.",
    )
    parser.add_argument(
        "--prefix-root",
        default=str(DEFAULT_PREFIX_RUN_GROUP),
        help="Path to the Prefix run-group root.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root folder for generated diagnosis outputs.",
    )
    parser.add_argument(
        "--top-context-items",
        type=int,
        default=20,
        help="How many items to keep in each context frequency table.",
    )
    return parser


def main() -> None:
    """Run the diagnosis CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()
    case = CASES[args.case]
    try:
        output_dir = run_case(
            case,
            shared_fake_sessions_path=Path(args.shared_fake_sessions),
            posopt_root=Path(args.posopt_root),
            prefix_root=Path(args.prefix_root),
            output_root=Path(args.output_root),
            top_context_items=args.top_context_items,
        )
        print(f"Wrote Prefix vs PosOptMVP diagnosis outputs to '{output_dir}'.")
    except DiagnosisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def run_case(
    case: DiagnosisCase,
    *,
    shared_fake_sessions_path: Path,
    posopt_root: Path,
    prefix_root: Path,
    output_root: Path,
    top_context_items: int,
) -> Path:
    """Run one configured diagnosis case."""
    shared = load_shared_artifacts(shared_fake_sessions_path)
    output_dir = output_root.resolve() / case.output_dirname
    plots_dir = output_dir / "plots"
    ensure_directory(output_dir)
    ensure_directory(plots_dir)

    prefix_targets = {
        target_item: load_prefix_target_artifacts(prefix_root, target_item=target_item)
        for target_item in case.targets
    }
    posopt_targets = {
        target_item: load_posopt_target_artifacts(posopt_root, target_item=target_item)
        for target_item in case.targets
    }

    metric_pairs: list[MetricPair] = []
    for target_item in case.targets:
        for victim_model in case.victims:
            metric_pairs.append(
                MetricPair(
                    prefix=load_victim_run_artifacts(
                        prefix_root,
                        method="prefix_nonzero_when_possible",
                        target_item=target_item,
                        victim_model=victim_model,
                    ),
                    posopt=load_victim_run_artifacts(
                        posopt_root,
                        method="position_opt_mvp",
                        target_item=target_item,
                        victim_model=victim_model,
                    ),
                )
            )

    metrics_comparison = build_metrics_comparison_dataframe(metric_pairs, case_id=case.case_id)

    shared_across_victims = len(case.victims) > 1
    victim_scope = case.victims[0] if len(case.victims) == 1 else None
    per_session_join_frames = []
    context_join_frames = []
    training_dynamics_frames = []
    for target_item in case.targets:
        per_session_join_frames.append(
            build_per_session_join_dataframe(
                shared,
                prefix_targets[target_item],
                posopt_targets[target_item],
                case_id=case.case_id,
                shared_across_victims=shared_across_victims,
                victim_scope=victim_scope,
            )
        )
        context_join_frames.append(
            build_context_join_dataframe(
                shared,
                prefix_targets[target_item],
                posopt_targets[target_item],
                case_id=case.case_id,
                shared_across_victims=shared_across_victims,
                victim_scope=victim_scope,
            )
        )
        training_dynamics_frames.append(
            build_training_dynamics_dataframe(shared, posopt_targets[target_item], case_id=case.case_id)
        )

    per_session_join = pd.concat(per_session_join_frames, ignore_index=True)
    context_join = pd.concat(context_join_frames, ignore_index=True)
    training_dynamics = pd.concat(training_dynamics_frames, ignore_index=True)
    position_summary = summarize_positions(
        per_session_join,
        shared_across_victims=shared_across_victims,
        victim_scope=victim_scope,
    )
    context_summary = summarize_context(
        context_join,
        shared_across_victims=shared_across_victims,
        victim_scope=victim_scope,
    )
    training_summary = summarize_training_dynamics(training_dynamics)

    top_replaced_items = build_top_context_table(
        context_join,
        prefix_column="prefix_original_item",
        posopt_column="posopt_original_item",
        label="replaced_original_item",
        top_n=top_context_items,
    )
    top_left_neighbors = build_top_context_table(
        context_join,
        prefix_column="prefix_left_neighbor",
        posopt_column="posopt_left_neighbor",
        label="left_neighbor",
        top_n=top_context_items,
    )
    top_right_neighbors = build_top_context_table(
        context_join,
        prefix_column="prefix_right_neighbor",
        posopt_column="posopt_right_neighbor",
        label="right_neighbor",
        top_n=top_context_items,
    )

    metrics_csv = output_dir / "metrics_comparison.csv"
    position_csv = output_dir / "position_comparison.csv"
    per_session_csv = output_dir / "per_session_join.csv"
    per_session_context_csv = output_dir / "per_session_context_join.csv"
    context_summary_csv = output_dir / "context_comparison.csv"
    top_replaced_csv = output_dir / "top_replaced_original_items.csv"
    top_left_csv = output_dir / "top_left_neighbors.csv"
    top_right_csv = output_dir / "top_right_neighbors.csv"
    training_csv = output_dir / "posopt_training_dynamics.csv"

    write_dataframe(metrics_comparison, metrics_csv)
    write_dataframe(position_summary, position_csv)
    write_dataframe(per_session_join, per_session_csv)
    write_dataframe(context_join, per_session_context_csv)
    write_dataframe(context_summary, context_summary_csv)
    write_dataframe(top_replaced_items, top_replaced_csv)
    write_dataframe(top_left_neighbors, top_left_csv)
    write_dataframe(top_right_neighbors, top_right_csv)
    write_dataframe(training_dynamics, training_csv)

    generated_plot_paths: list[Path] = []
    metrics_plot = plots_dir / "targeted_metrics_overview.png"
    plot_targeted_metrics_overview(metrics_comparison, metrics_plot, title=case.description)
    generated_plot_paths.append(metrics_plot)

    for target_item in case.targets:
        position_plot = plots_dir / f"final_position_histogram_target_{target_item}.png"
        plot_position_histogram(
            per_session_join,
            position_plot,
            target_item=target_item,
            title=f"Final selected positions for target {target_item}",
        )
        generated_plot_paths.append(position_plot)

        normalized_plot = plots_dir / f"normalized_position_histogram_target_{target_item}.png"
        plot_normalized_position_histogram(
            per_session_join,
            normalized_plot,
            target_item=target_item,
            title=f"Normalized selected positions for target {target_item}",
        )
        generated_plot_paths.append(normalized_plot)

        reward_plot = plots_dir / f"posopt_reward_curve_target_{target_item}.png"
        plot_training_curve(
            training_dynamics,
            reward_plot,
            target_item=target_item,
            value_column="reward",
            ylabel="Reward",
            title=f"PosOpt reward curve for target {target_item}",
        )
        generated_plot_paths.append(reward_plot)

        baseline_plot = plots_dir / f"posopt_baseline_curve_target_{target_item}.png"
        plot_training_curve(
            training_dynamics,
            baseline_plot,
            target_item=target_item,
            value_column="baseline",
            ylabel="Baseline",
            title=f"PosOpt baseline curve for target {target_item}",
        )
        generated_plot_paths.append(baseline_plot)

        advantage_plot = plots_dir / f"posopt_advantage_curve_target_{target_item}.png"
        plot_training_curve(
            training_dynamics,
            advantage_plot,
            target_item=target_item,
            value_column="advantage",
            ylabel="Advantage",
            title=f"PosOpt advantage curve for target {target_item}",
        )
        generated_plot_paths.append(advantage_plot)

        entropy_plot = plots_dir / f"posopt_mean_entropy_curve_target_{target_item}.png"
        plot_training_curve(
            training_dynamics,
            entropy_plot,
            target_item=target_item,
            value_column="mean_entropy",
            ylabel="Mean entropy",
            title=f"PosOpt mean entropy curve for target {target_item}",
        )
        generated_plot_paths.append(entropy_plot)

        average_position_plot = plots_dir / f"posopt_average_selected_position_curve_target_{target_item}.png"
        plot_training_curve(
            training_dynamics,
            average_position_plot,
            target_item=target_item,
            value_column="average_selected_position",
            ylabel="Average selected position",
            title=f"PosOpt average selected position for target {target_item}",
        )
        generated_plot_paths.append(average_position_plot)

    summary_payload = build_summary_payload(
        case,
        shared_fake_sessions_path=shared.fake_sessions_path,
        posopt_root=posopt_root,
        prefix_root=prefix_root,
        metrics_comparison=metrics_comparison,
        position_summary=position_summary,
        context_summary=context_summary,
        training_summary=training_summary,
        output_dir=output_dir,
        plot_paths=generated_plot_paths,
        table_paths={
            "metrics_comparison": metrics_csv,
            "position_comparison": position_csv,
            "per_session_join": per_session_csv,
            "per_session_context_join": per_session_context_csv,
            "context_comparison": context_summary_csv,
            "top_replaced_original_items": top_replaced_csv,
            "top_left_neighbors": top_left_csv,
            "top_right_neighbors": top_right_csv,
            "posopt_training_dynamics": training_csv,
        },
    )
    summary_path = output_dir / "summary.json"
    write_json(summary_payload, summary_path)

    manifest_payload = build_manifest_payload(
        case,
        shared_fake_sessions_path=shared.fake_sessions_path,
        prefix_root=prefix_root,
        posopt_root=posopt_root,
        prefix_targets=prefix_targets,
        posopt_targets=posopt_targets,
        metric_pairs=metric_pairs,
        output_dir=output_dir,
        generated_plot_paths=generated_plot_paths,
        generated_table_paths=[
            metrics_csv,
            position_csv,
            per_session_csv,
            per_session_context_csv,
            context_summary_csv,
            top_replaced_csv,
            top_left_csv,
            top_right_csv,
            training_csv,
            summary_path,
        ],
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_payload, manifest_path)

    report_path = output_dir / "report.md"
    write_report(
        build_report(
            case,
            summary_payload=summary_payload,
            manifest_payload=manifest_payload,
            metrics_comparison=metrics_comparison,
            position_summary=position_summary,
            context_summary=context_summary,
            training_summary=training_summary,
            top_replaced_items=top_replaced_items,
        ),
        report_path,
    )
    return output_dir


def build_summary_payload(
    case: DiagnosisCase,
    *,
    shared_fake_sessions_path: Path,
    posopt_root: Path,
    prefix_root: Path,
    metrics_comparison: pd.DataFrame,
    position_summary: pd.DataFrame,
    context_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    output_dir: Path,
    plot_paths: list[Path],
    table_paths: dict[str, Path],
) -> dict[str, Any]:
    """Build one machine-readable diagnosis summary payload."""
    metrics_highlights = metrics_comparison[
        (metrics_comparison["metric_scope"] == "targeted")
        & (metrics_comparison["metric_name"].isin(["recall", "mrr"]))
        & (metrics_comparison["k"] == 10)
    ].copy()
    return {
        "case": {
            "case_id": case.case_id,
            "description": case.description,
            "kind": case.kind,
            "targets": list(case.targets),
            "victims": list(case.victims),
            "failure_target": case.failure_target,
            "success_target": case.success_target,
        },
        "input_roots": {
            "shared_fake_sessions_path": repo_relative(shared_fake_sessions_path),
            "prefix_run_group_root": repo_relative(prefix_root.resolve()),
            "posopt_run_group_root": repo_relative(posopt_root.resolve()),
        },
        "metric_highlights": dataframe_to_records(metrics_highlights),
        "position_summary": dataframe_to_records(position_summary),
        "context_summary": dataframe_to_records(context_summary),
        "training_summary": dataframe_to_records(training_summary),
        "tables": {name: repo_relative(path) for name, path in table_paths.items()},
        "plots": [repo_relative(path) for path in plot_paths],
        "output_dir": repo_relative(output_dir),
    }


def build_manifest_payload(
    case: DiagnosisCase,
    *,
    shared_fake_sessions_path: Path,
    prefix_root: Path,
    posopt_root: Path,
    prefix_targets: dict[int, Any],
    posopt_targets: dict[int, Any],
    metric_pairs: list[MetricPair],
    output_dir: Path,
    generated_plot_paths: list[Path],
    generated_table_paths: list[Path],
) -> dict[str, Any]:
    """Build one manifest that records exact inputs and outputs."""
    target_inputs = []
    for target_item in case.targets:
        prefix_target = prefix_targets[target_item]
        posopt_target = posopt_targets[target_item]
        target_inputs.append(
            {
                "target_item": target_item,
                "prefix_artifacts": {
                    "target_dir": repo_relative(prefix_target.target_dir),
                    "position_stats": repo_relative(prefix_target.position_stats_path),
                    "selected_positions": repo_relative(prefix_target.selected_positions_path),
                },
                "posopt_artifacts": {
                    "target_dir": repo_relative(posopt_target.target_dir),
                    "position_stats": repo_relative(posopt_target.position_stats_path),
                    "selected_positions": repo_relative(posopt_target.selected_positions_path),
                    "training_history": repo_relative(posopt_target.training_history_path),
                    "run_metadata": repo_relative(posopt_target.run_metadata_path),
                    "optimized_poisoned_sessions": repo_relative(posopt_target.optimized_poisoned_sessions_path),
                },
            }
        )
    victim_inputs = []
    for pair in metric_pairs:
        victim_inputs.append(
            {
                "target_item": pair.prefix.target_item,
                "victim_model": pair.prefix.victim_model,
                "prefix_run_dir": repo_relative(pair.prefix.run_dir),
                "prefix_metrics": repo_relative(pair.prefix.metrics_path),
                "prefix_train_history": repo_relative(pair.prefix.train_history_path),
                "prefix_resolved_config": repo_relative(pair.prefix.resolved_config_path),
                "posopt_run_dir": repo_relative(pair.posopt.run_dir),
                "posopt_metrics": repo_relative(pair.posopt.metrics_path),
                "posopt_train_history": repo_relative(pair.posopt.train_history_path),
                "posopt_resolved_config": repo_relative(pair.posopt.resolved_config_path),
            }
        )
    return {
        "case": {
            "case_id": case.case_id,
            "description": case.description,
            "kind": case.kind,
            "targets": list(case.targets),
            "victims": list(case.victims),
        },
        "input_roots": {
            "repo_root": repo_relative(REPO_ROOT),
            "shared_fake_sessions_path": repo_relative(shared_fake_sessions_path),
            "prefix_run_group_root": repo_relative(prefix_root.resolve()),
            "posopt_run_group_root": repo_relative(posopt_root.resolve()),
        },
        "target_level_inputs": target_inputs,
        "victim_level_inputs": victim_inputs,
        "output_dir": repo_relative(output_dir),
        "generated_tables": [repo_relative(path) for path in generated_table_paths],
        "generated_plots": [repo_relative(path) for path in generated_plot_paths],
    }


def build_report(
    case: DiagnosisCase,
    *,
    summary_payload: dict[str, Any],
    manifest_payload: dict[str, Any],
    metrics_comparison: pd.DataFrame,
    position_summary: pd.DataFrame,
    context_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    top_replaced_items: pd.DataFrame,
) -> str:
    """Render one evidence-focused Markdown report."""
    highlights = metrics_comparison[
        (metrics_comparison["metric_scope"] == "targeted")
        & (metrics_comparison["metric_name"].isin(["recall", "mrr"]))
        & (metrics_comparison["k"] == 10)
    ][
        ["target_item", "victim_model", "metric_key", "prefix_value", "posopt_value", "posopt_minus_prefix"]
    ]
    top_context_preview = top_replaced_items[
        ["target_item", "method", "rank", "item_id", "count", "ratio_among_non_null"]
    ].head(12)

    hypotheses = build_candidate_hypotheses(
        case,
        metrics_comparison=metrics_comparison,
        position_summary=position_summary,
        context_summary=context_summary,
        training_summary=training_summary,
    )

    artifact_lines = [
        f"- Shared fake-session pool: `{manifest_payload['input_roots']['shared_fake_sessions_path']}`",
        f"- Prefix run group: `{manifest_payload['input_roots']['prefix_run_group_root']}`",
        f"- PosOpt run group: `{manifest_payload['input_roots']['posopt_run_group_root']}`",
    ]
    for target_entry in manifest_payload["target_level_inputs"]:
        artifact_lines.append(f"- Target {target_entry['target_item']} Prefix artifacts:")
        artifact_lines.append(f"  - `{target_entry['prefix_artifacts']['position_stats']}`")
        artifact_lines.append(f"  - `{target_entry['prefix_artifacts']['selected_positions']}`")
        artifact_lines.append(f"- Target {target_entry['target_item']} PosOpt artifacts:")
        artifact_lines.append(f"  - `{target_entry['posopt_artifacts']['position_stats']}`")
        artifact_lines.append(f"  - `{target_entry['posopt_artifacts']['selected_positions']}`")
        artifact_lines.append(f"  - `{target_entry['posopt_artifacts']['training_history']}`")
        artifact_lines.append(f"  - `{target_entry['posopt_artifacts']['run_metadata']}`")
        artifact_lines.append(f"  - `{target_entry['posopt_artifacts']['optimized_poisoned_sessions']}`")

    lines = [
        f"# Prefix vs PosOptMVP Diagnosis: {case.case_id}",
        "",
        "## What Was Analyzed",
        "",
        case.description,
        "",
        "## Run Paths And Artifacts",
        "",
        *artifact_lines,
        "",
        "## High-Level Metric Comparison",
        "",
        dataframe_to_markdown(highlights),
        "",
        "Generated chart:",
        f"- `plots/targeted_metrics_overview.png`",
        "",
        "## High-Level Position Comparison",
        "",
        dataframe_to_markdown(
            position_summary[
                [
                    "target_item",
                    "method",
                    "mean_selected_position",
                    "median_selected_position",
                    "fraction_position0",
                    "fraction_top10pct",
                    "fraction_top20pct",
                ]
            ]
        ),
        "",
        "Position plots:",
        *[
            f"- `plots/final_position_histogram_target_{target_item}.png`"
            for target_item in case.targets
        ],
        *[
            f"- `plots/normalized_position_histogram_target_{target_item}.png`"
            for target_item in case.targets
        ],
        "",
        "## High-Level PosOpt Training Dynamics",
        "",
        dataframe_to_markdown(training_summary),
        "",
        "Training plots:",
        *[
            f"- `plots/posopt_reward_curve_target_{target_item}.png`"
            for target_item in case.targets
        ],
        *[
            f"- `plots/posopt_baseline_curve_target_{target_item}.png`"
            for target_item in case.targets
        ],
        *[
            f"- `plots/posopt_advantage_curve_target_{target_item}.png`"
            for target_item in case.targets
        ],
        *[
            f"- `plots/posopt_mean_entropy_curve_target_{target_item}.png`"
            for target_item in case.targets
        ],
        *[
            f"- `plots/posopt_average_selected_position_curve_target_{target_item}.png`"
            for target_item in case.targets
        ],
        "",
        "## High-Level Context Comparison",
        "",
        dataframe_to_markdown(context_summary),
        "",
        "Preview of top replaced original items:",
        "",
        dataframe_to_markdown(top_context_preview),
        "",
        "## Candidate Hypotheses For Why Prefix Beats PosOpt Here",
        "",
        *[f"- {line}" for line in hypotheses],
        "",
        "## Output Files",
        "",
        *[
            f"- `{relative_path}`"
            for relative_path in list(summary_payload["tables"].values()) + summary_payload["plots"]
        ],
    ]
    return "\n".join(lines) + "\n"


def build_candidate_hypotheses(
    case: DiagnosisCase,
    *,
    metrics_comparison: pd.DataFrame,
    position_summary: pd.DataFrame,
    context_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
) -> list[str]:
    """Generate short, evidence-based candidate hypotheses."""
    hypotheses: list[str] = []
    recall10 = metrics_comparison[
        (metrics_comparison["metric_key"] == "targeted_recall@10")
    ].copy()
    position_lookup = {
        (int(row["target_item"]), str(row["method"])): row
        for _, row in position_summary.iterrows()
    }
    context_lookup = {
        int(row["target_item"]): row
        for _, row in context_summary.iterrows()
    }
    training_lookup = {
        int(row["target_item"]): row
        for _, row in training_summary.iterrows()
    }

    for _, row in recall10.sort_values(["target_item", "victim_model"]).iterrows():
        target_item = int(row["target_item"])
        victim_model = str(row["victim_model"])
        delta = float(row["posopt_minus_prefix"])
        prefix_pos = position_lookup[(target_item, "prefix_nonzero_when_possible")]
        posopt_pos = position_lookup[(target_item, "position_opt_mvp")]
        context_row = context_lookup[target_item]
        training_row = training_lookup[target_item]
        if delta < 0:
            if float(prefix_pos["fraction_position0"]) - float(posopt_pos["fraction_position0"]) >= 0.15:
                hypotheses.append(
                    f"For target {target_item} on {victim_model}, Prefix wins while using a much stronger "
                    f"position-0 bias ({prefix_pos['fraction_position0']:.3f} vs {posopt_pos['fraction_position0']:.3f}). "
                    "That supports an early-position-bias explanation more than a broad target-quality explanation."
                )
            if float(context_row["fraction_same_selected_position"]) <= 0.40:
                hypotheses.append(
                    f"For target {target_item} on {victim_model}, Prefix and PosOpt match on the exact selected "
                    f"position only {context_row['fraction_same_selected_position']:.3f} of the time, so different "
                    "target-context compatibility is a plausible contributor."
                )
            if (
                float(training_row["entropy_drop"]) <= 0.05
                or float(training_row["final_max_position_share"]) <= 0.35
            ):
                hypotheses.append(
                    f"For target {target_item}, the PosOpt policy stays relatively diffuse "
                    f"(entropy drop {training_row['entropy_drop']:.3f}, final max-position share "
                    f"{training_row['final_max_position_share']:.3f}), which is consistent with a policy "
                    "training instability or weak concentration explanation."
                )
        else:
            hypotheses.append(
                f"For target {target_item} on {victim_model}, PosOpt improves targeted Recall@10 over Prefix by "
                f"{delta:.3f}. This serves as a useful control case showing that broader position search can help "
                "when the target benefits from a less front-loaded replacement pattern."
            )

    if case.kind == "cross_victim_target":
        target_item = case.targets[0]
        cross_victim = recall10.loc[recall10["target_item"] == target_item].sort_values("victim_model")
        if len(cross_victim) >= 2:
            best = cross_victim.loc[cross_victim["posopt_minus_prefix"].idxmax()]
            worst = cross_victim.loc[cross_victim["posopt_minus_prefix"].idxmin()]
            hypotheses.append(
                f"For shared target {target_item}, Prefix and PosOpt use the same target-level poisoned-session pool "
                f"across victims, yet the PosOpt-minus-Prefix Recall@10 delta ranges from {worst['posopt_minus_prefix']:.3f} "
                f"({worst['victim_model']}) to {best['posopt_minus_prefix']:.3f} ({best['victim_model']}). "
                "That points to victim-model sensitivity rather than per-victim differences in final positions."
            )
    return hypotheses


if __name__ == "__main__":
    main()
