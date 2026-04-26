#!/usr/bin/env python3
"""Collect a slice-specific analysis bundle from completed run-group artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.utils.run_bundle_loader import RunBundle, load_run_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "analysis" / "diagnosis_output" / "random_nz_slice_bundle_5334_11103"
)
DATASET = "diginetica"
TARGETS = [5334, 11103]
VICTIMS = ["srgnn", "tron"]
KS = [5, 10, 15, 20, 25, 30, 40, 50]
METRIC_KEY_PATTERN = re.compile(
    r"^(?P<scope>targeted|ground_truth)_(?P<metric>precision|recall|mrr|ndcg)@(?P<k>\d+)$"
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    method_label: str
    run_root: str
    attack_method_hint: str | None
    required: bool
    requested_ratio: float | None = None


METHOD_SPECS = [
    MethodSpec(
        key="clean",
        method_label="Clean",
        run_root="outputs/runs/diginetica/clean_run_no_attack/run_group_e0caef2757",
        attack_method_hint="clean",
        required=True,
    ),
    MethodSpec(
        key="dpsbr",
        method_label="DPSBR random",
        run_root="outputs/runs/diginetica/attack_dpsbr/run_group_7db577fb2e",
        attack_method_hint="dpsbr",
        required=False,
        requested_ratio=0.2,
    ),
    MethodSpec(
        key="prefix_r02",
        method_label="Prefix-NZ@0.2",
        run_root="outputs/runs/diginetica/attack_prefix_nonzero_when_possible/run_group_14818d6dd6",
        attack_method_hint="prefix_nonzero_when_possible",
        required=True,
        requested_ratio=0.2,
    ),
    MethodSpec(
        key="prefix_r1",
        method_label="Prefix-NZ@1.0",
        run_root="outputs/runs/diginetica/attack_prefix_nonzero_when_possible_ratio1/run_group_122a28bd27",
        attack_method_hint="prefix_nonzero_when_possible",
        required=True,
        requested_ratio=1.0,
    ),
    MethodSpec(
        key="shared_policy_r1",
        method_label="SharedPolicy@1.0",
        run_root="outputs/runs/diginetica/attack_position_opt_shared_policy_ratio1/run_group_c1835ab73f",
        attack_method_hint="position_opt_shared_policy",
        required=True,
        requested_ratio=1.0,
    ),
    MethodSpec(
        key="shared_policy_nz_r1",
        method_label="SharedPolicy-NZ@1.0",
        run_root="outputs/runs/diginetica/attack_position_opt_shared_policy_nonzero/run_group_0bce31ef52",
        attack_method_hint="position_opt_shared_policy",
        required=True,
        requested_ratio=1.0,
    ),
    MethodSpec(
        key="random_nz_r1",
        method_label="Random-NZ@1.0",
        run_root="outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_27360b4a79",
        attack_method_hint="random_nonzero_when_possible",
        required=True,
        requested_ratio=1.0,
    ),
    MethodSpec(
        key="mvp_r1",
        method_label="MVP@1.0",
        run_root="outputs/runs/diginetica/attack_position_optimization_reward_mvp_ratio1/run_group_3becc51c46",
        attack_method_hint="position_opt_mvp",
        required=False,
        requested_ratio=1.0,
    ),
]

METRIC_SCOPE_ORDER = {"targeted": 0, "ground_truth": 1}
METRIC_NAME_ORDER = {"recall": 0, "mrr": 1, "ndcg": 2, "precision": 3}
VICTIM_ORDER = {name: index for index, name in enumerate(VICTIMS)}
TARGET_ORDER = {target: index for index, target in enumerate(TARGETS)}
METHOD_ORDER = {spec.key: index for index, spec in enumerate(METHOD_SPECS)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect a completed-run analysis bundle for the Random-NZ slice."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory that will receive the markdown and CSV outputs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_methods, skipped_methods = load_methods()
    metrics_df = build_metrics_long_table(loaded_methods)
    position_df = build_position_statistics_table(loaded_methods)
    wide_df = build_comparison_wide_table(metrics_df, loaded_methods)

    metrics_path = output_dir / "random_nz_metrics_long.csv"
    position_path = output_dir / "random_nz_position_statistics.csv"
    wide_path = output_dir / "random_nz_comparison_wide.csv"
    summary_path = output_dir / "random_nz_slice_summary.md"

    metrics_df.to_csv(metrics_path, index=False)
    position_df.to_csv(position_path, index=False)
    wide_df.to_csv(wide_path, index=False)
    summary_path.write_text(
        build_summary_markdown(
            loaded_methods=loaded_methods,
            skipped_methods=skipped_methods,
            position_df=position_df,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    print(f"Wrote analysis bundle to {output_dir}")
    print(f"- {metrics_path.name}")
    print(f"- {position_path.name}")
    print(f"- {wide_path.name}")
    print(f"- {summary_path.name}")


def load_methods() -> tuple[list[tuple[MethodSpec, RunBundle]], list[dict[str, Any]]]:
    loaded: list[tuple[MethodSpec, RunBundle]] = []
    skipped: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        try:
            bundle = load_run_bundle(
                run_root=spec.run_root,
                method_key=spec.key,
                label=spec.method_label,
                attack_method_hint=spec.attack_method_hint,
                dataset_hint=DATASET,
            )
            validate_bundle_slice(bundle, spec)
            loaded.append((spec, bundle))
        except Exception as exc:  # noqa: BLE001
            if spec.required:
                raise
            skipped.append(
                {
                    "method_key": spec.key,
                    "method_label": spec.method_label,
                    "run_root": spec.run_root,
                    "reason": str(exc),
                }
            )
    return loaded, skipped


def validate_bundle_slice(bundle: RunBundle, spec: MethodSpec) -> None:
    missing_targets = [target for target in TARGETS if target not in bundle.target_items]
    missing_victims = [victim for victim in VICTIMS if victim not in bundle.victims]
    if missing_targets:
        raise ValueError(
            f"{spec.key} is missing requested targets {missing_targets} in run_root {bundle.run_root}."
        )
    if missing_victims:
        raise ValueError(
            f"{spec.key} is missing requested victims {missing_victims} in run_root {bundle.run_root}."
        )
    for target in TARGETS:
        target_payload = bundle.summary_current.get("targets", {}).get(str(target))
        if not isinstance(target_payload, Mapping):
            raise ValueError(f"{spec.key} summary_current is missing target {target}.")
        victims_payload = target_payload.get("victims", {})
        if not isinstance(victims_payload, Mapping):
            raise ValueError(f"{spec.key} summary_current target {target} has no victims mapping.")
        for victim in VICTIMS:
            victim_payload = victims_payload.get(victim)
            if not isinstance(victim_payload, Mapping):
                raise ValueError(
                    f"{spec.key} summary_current target {target} is missing victim {victim}."
                )
            if not bool(victim_payload.get("metrics_available")):
                raise ValueError(
                    f"{spec.key} summary_current target {target} victim {victim} has no metrics."
                )


def build_metrics_long_table(loaded_methods: list[tuple[MethodSpec, RunBundle]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec, bundle in loaded_methods:
        method_context = method_context_row(spec, bundle)
        row_count = 0
        for target in TARGETS:
            victims_payload = bundle.summary_current["targets"][str(target)]["victims"]
            for victim in VICTIMS:
                metrics_payload = victims_payload[victim]["metrics"]
                for metric_key, value in metrics_payload.items():
                    match = METRIC_KEY_PATTERN.match(str(metric_key))
                    if not match:
                        continue
                    k_value = int(match.group("k"))
                    if k_value not in KS:
                        continue
                    row_count += 1
                    rows.append(
                        {
                            **method_context,
                            "dataset": DATASET,
                            "target_item": int(target),
                            "victim_model": victim,
                            "metric_scope": match.group("scope"),
                            "metric_name": match.group("metric"),
                            "metric_key": str(metric_key),
                            "k": k_value,
                            "value": float(value),
                        }
                    )
        expected_rows = len(TARGETS) * len(VICTIMS) * 2 * 4 * len(KS)
        if row_count != expected_rows:
            raise ValueError(
                f"{spec.key} produced {row_count} metric rows; expected {expected_rows}."
            )
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(
        by=[
            "target_item",
            "victim_model",
            "metric_scope",
            "metric_name",
            "k",
            "method_key",
        ],
        key=lambda column: sort_key_for_metrics(column),
        kind="mergesort",
    ).reset_index(drop=True)
    return dataframe


def build_comparison_wide_table(
    metrics_df: pd.DataFrame,
    loaded_methods: list[tuple[MethodSpec, RunBundle]],
) -> pd.DataFrame:
    ordered_labels = [spec.method_label for spec, _ in loaded_methods]
    pivot = metrics_df.pivot_table(
        index=["dataset", "target_item", "victim_model", "metric_scope", "metric_name", "k"],
        columns="method_label",
        values="value",
        aggfunc="first",
    )
    pivot = pivot.reset_index()
    for label in ordered_labels:
        if label not in pivot.columns:
            pivot[label] = pd.NA
    pivot = pivot[
        ["dataset", "target_item", "victim_model", "metric_scope", "metric_name", "k", *ordered_labels]
    ]
    pivot = pivot.sort_values(
        by=["target_item", "victim_model", "metric_scope", "metric_name", "k"],
        key=lambda column: sort_key_for_metrics(column),
        kind="mergesort",
    ).reset_index(drop=True)
    return pivot


def build_position_statistics_table(
    loaded_methods: list[tuple[MethodSpec, RunBundle]],
) -> pd.DataFrame:
    requested_keys = {"random_nz_r1", "prefix_r1", "shared_policy_nz_r1"}
    rows: list[dict[str, Any]] = []
    for spec, bundle in loaded_methods:
        if spec.key not in requested_keys:
            continue
        method_context = method_context_row(spec, bundle)
        for target in TARGETS:
            target_artifacts = bundle.target_artifacts[target]
            if target_artifacts.position_stats_path is None:
                raise ValueError(f"{spec.key} target {target} is missing position_stats.json.")
            stats_payload = load_json_object(target_artifacts.position_stats_path)
            counts = parse_position_counts(stats_payload["overall"]["counts"])
            total_sessions = int(stats_payload["total_sessions"])
            run_metadata_payload = load_optional_json(
                None
                if target_artifacts.position_opt_dir is None
                else target_artifacts.position_opt_dir / "run_metadata.json"
            )
            candidate_diagnostics = extract_candidate_space_diagnostics(run_metadata_payload)
            final_position_diagnostics = extract_final_position_diagnostics(run_metadata_payload)
            extra_random_payload = (
                load_optional_json(target_artifacts.random_nonzero_metadata_path)
                if target_artifacts.random_nonzero_metadata_path is not None
                else None
            )
            rows.append(
                {
                    **method_context,
                    "dataset": DATASET,
                    "target_item": int(target),
                    "total_fake_sessions": total_sessions,
                    "selected_position_histogram_json": json.dumps(
                        {str(position): count for position, count in counts.items()},
                        sort_keys=True,
                    ),
                    "pos0_pct": percent_at_position(counts, total_sessions, 0),
                    "pos1_pct": percent_at_position(counts, total_sessions, 1),
                    "pos2_pct": percent_at_position(counts, total_sessions, 2),
                    "pos3_pct": percent_at_position(counts, total_sessions, 3),
                    "pos4_pct": percent_at_position(counts, total_sessions, 4),
                    "pos5_pct": percent_at_position(counts, total_sessions, 5),
                    "pos_ge_6_pct": percent_for_positions(
                        counts,
                        total_sessions,
                        [position for position in counts if position >= 6],
                    ),
                    "pos_le_1_pct": percent_for_positions(
                        counts,
                        total_sessions,
                        [position for position in counts if position <= 1],
                    ),
                    "pos_le_2_pct": percent_for_positions(
                        counts,
                        total_sessions,
                        [position for position in counts if position <= 2],
                    ),
                    "pos_le_5_pct": percent_for_positions(
                        counts,
                        total_sessions,
                        [position for position in counts if position <= 5],
                    ),
                    "unique_selected_positions": len(counts),
                    "mean_selected_position": weighted_mean(counts),
                    "median_selected_position": weighted_median(counts),
                    "diagnostics_source": diagnostics_source_name(spec, run_metadata_payload),
                    "pos0_removed_session_count": lookup_nested(
                        candidate_diagnostics, "pos0_removed_session_count"
                    ),
                    "fallback_to_pos0_only_count": lookup_nested(
                        candidate_diagnostics, "fallback_to_pos0_only_count"
                    ),
                    "forced_single_candidate_count": lookup_nested(
                        candidate_diagnostics, "forced_single_candidate_count"
                    ),
                    "final_pos0_pct": lookup_nested(final_position_diagnostics, "final_pos0_pct"),
                    "min_candidate_count_after_mask": lookup_nested(
                        candidate_diagnostics, "min_candidate_count_after_mask"
                    ),
                    "mean_candidate_count_after_mask": lookup_nested(
                        candidate_diagnostics, "mean_candidate_count_after_mask"
                    ),
                    "max_candidate_count_after_mask": lookup_nested(
                        candidate_diagnostics, "max_candidate_count_after_mask"
                    ),
                    "random_seed": resolve_random_seed(bundle),
                    "attack_seed": resolve_random_seed(bundle),
                    "replacement_seed": None,
                    "fake_session_seed": bundle.seeds.get("fake_session_seed"),
                    "target_selection_seed": bundle.seeds.get("target_selection_seed"),
                    "position_opt_seed": bundle.seeds.get("position_opt_seed"),
                    "surrogate_train_seed": bundle.seeds.get("surrogate_train_seed"),
                    "victim_train_seed": bundle.seeds.get("victim_train_seed"),
                    "random_nonzero_metadata_total": (
                        extra_random_payload.get("total")
                        if isinstance(extra_random_payload, Mapping)
                        else None
                    ),
                    "notes": position_notes(spec, bundle, run_metadata_payload),
                }
            )
    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values(
        by=["target_item", "method_key"],
        key=lambda column: sort_key_for_positions(column),
        kind="mergesort",
    ).reset_index(drop=True)
    return dataframe


def method_context_row(spec: MethodSpec, bundle: RunBundle) -> dict[str, Any]:
    actual_ratio = None if spec.key == "clean" else bundle.replacement_topk_ratio
    requested_ratio = spec.requested_ratio
    ratio_mismatch = (
        requested_ratio is not None
        and actual_ratio is not None
        and not math.isclose(float(requested_ratio), float(actual_ratio), rel_tol=0.0, abs_tol=1e-12)
    )
    effective_nonzero = resolve_effective_nonzero_flag(spec, bundle)
    actual_label = resolve_actual_method_label(spec, actual_ratio)
    return {
        "method_key": spec.key,
        "method_label": spec.method_label,
        "actual_method_label": actual_label,
        "attack_method": bundle.attack_method,
        "run_group_key": bundle.run_group_key,
        "run_root": str(bundle.run_root),
        "requested_replacement_topk_ratio": requested_ratio,
        "actual_replacement_topk_ratio": actual_ratio,
        "method_label_ratio_mismatch": ratio_mismatch,
        "nonzero_action_when_possible": bundle.nonzero_action_when_possible,
        "effective_nonzero_when_possible": effective_nonzero,
        "fake_session_seed": bundle.seeds.get("fake_session_seed"),
        "target_selection_seed": bundle.seeds.get("target_selection_seed"),
        "position_opt_seed": bundle.seeds.get("position_opt_seed"),
        "surrogate_train_seed": bundle.seeds.get("surrogate_train_seed"),
        "victim_train_seed": bundle.seeds.get("victim_train_seed"),
        "seed_mode": "single_seed",
        "slice_origin": (
            "exact_slice_bundle"
            if sorted(bundle.target_items) == sorted(TARGETS) and sorted(bundle.victims) == sorted(VICTIMS)
            else "subset_from_larger_bundle"
        ),
    }


def resolve_effective_nonzero_flag(spec: MethodSpec, bundle: RunBundle) -> bool | None:
    if bundle.nonzero_action_when_possible is not None:
        return bool(bundle.nonzero_action_when_possible)
    if spec.key in {"prefix_r02", "prefix_r1", "random_nz_r1"}:
        return True
    if spec.key == "clean":
        return None
    return False


def resolve_actual_method_label(spec: MethodSpec, actual_ratio: float | None) -> str:
    if spec.key == "clean":
        return "Clean"
    if spec.key == "dpsbr":
        return "DPSBR random"
    if actual_ratio is None:
        return spec.method_label
    if spec.key.startswith("prefix_"):
        return f"Prefix-NZ@{actual_ratio:.1f}"
    if spec.key == "shared_policy_r1":
        return f"SharedPolicy@{actual_ratio:.1f}"
    if spec.key == "shared_policy_nz_r1":
        return f"SharedPolicy-NZ@{actual_ratio:.1f}"
    if spec.key == "random_nz_r1":
        return f"Random-NZ@{actual_ratio:.1f}"
    if spec.key == "mvp_r1":
        return f"MVP@{actual_ratio:.1f}"
    return spec.method_label


def load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}.")
    return payload


def load_optional_json(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved_path = Path(path)
    if not resolved_path.is_file():
        return None
    return load_json_object(resolved_path)


def parse_position_counts(payload: Mapping[str, Any]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for raw_position, raw_count in payload.items():
        counts[int(raw_position)] = int(raw_count)
    return dict(sorted(counts.items()))


def weighted_mean(counts: Mapping[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    return sum(float(position) * float(count) for position, count in counts.items()) / float(total)


def weighted_median(counts: Mapping[int, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    midpoint = (total + 1) / 2.0
    cumulative = 0
    for position, count in sorted(counts.items()):
        cumulative += int(count)
        if cumulative >= midpoint:
            return float(position)
    return float(max(counts))


def percent_at_position(counts: Mapping[int, int], total: int, position: int) -> float:
    return 100.0 * float(counts.get(position, 0)) / float(total)


def percent_for_positions(counts: Mapping[int, int], total: int, positions: list[int]) -> float:
    return 100.0 * float(sum(int(counts.get(position, 0)) for position in positions)) / float(total)


def extract_candidate_space_diagnostics(
    run_metadata_payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(run_metadata_payload, Mapping):
        return None
    payload = run_metadata_payload.get("candidate_space_diagnostics")
    if isinstance(payload, Mapping):
        return payload
    trainer_result = run_metadata_payload.get("trainer_result")
    if isinstance(trainer_result, Mapping):
        trainer_payload = trainer_result.get("candidate_space_diagnostics")
        if isinstance(trainer_payload, Mapping):
            return trainer_payload
    return None


def extract_final_position_diagnostics(
    run_metadata_payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(run_metadata_payload, Mapping):
        return None
    payload = run_metadata_payload.get("final_position_diagnostics")
    if isinstance(payload, Mapping):
        return payload
    trainer_result = run_metadata_payload.get("trainer_result")
    if isinstance(trainer_result, Mapping):
        trainer_payload = trainer_result.get("final_position_diagnostics")
        if isinstance(trainer_payload, Mapping):
            return trainer_payload
    return None


def lookup_nested(payload: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(payload, Mapping):
        return None
    return payload.get(key)


def resolve_random_seed(bundle: RunBundle) -> Any:
    return bundle.seeds.get("fake_session_seed")


def diagnostics_source_name(
    spec: MethodSpec,
    run_metadata_payload: Mapping[str, Any] | None,
) -> str:
    if run_metadata_payload is not None:
        return "position_opt_run_metadata"
    if spec.key == "random_nz_r1":
        return "random_nonzero_position_metadata_plus_position_stats"
    return "position_stats_only"


def position_notes(
    spec: MethodSpec,
    bundle: RunBundle,
    run_metadata_payload: Mapping[str, Any] | None,
) -> str:
    notes: list[str] = []
    if spec.key == "random_nz_r1":
        actual_ratio = bundle.replacement_topk_ratio
        if actual_ratio is not None and not math.isclose(actual_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
            notes.append(
                f"requested label {spec.method_label} points to actual replacement_topk_ratio={actual_ratio}"
            )
        notes.append(
            "candidate-space diagnostics are not recorded for random_nonzero_when_possible artifacts"
        )
    if run_metadata_payload is None and spec.key != "random_nz_r1":
        notes.append("no position_opt run_metadata.json available")
    return "; ".join(notes)


def sort_key_for_metrics(column: pd.Series) -> pd.Series:
    if column.name == "target_item":
        return column.map(lambda value: TARGET_ORDER[int(value)])
    if column.name == "victim_model":
        return column.map(lambda value: VICTIM_ORDER[str(value)])
    if column.name == "metric_scope":
        return column.map(lambda value: METRIC_SCOPE_ORDER[str(value)])
    if column.name == "metric_name":
        return column.map(lambda value: METRIC_NAME_ORDER[str(value)])
    if column.name == "method_key":
        return column.map(lambda value: METHOD_ORDER[str(value)])
    return column


def sort_key_for_positions(column: pd.Series) -> pd.Series:
    if column.name == "target_item":
        return column.map(lambda value: TARGET_ORDER[int(value)])
    if column.name == "method_key":
        return column.map(lambda value: METHOD_ORDER[str(value)])
    return column


def build_summary_markdown(
    *,
    loaded_methods: list[tuple[MethodSpec, RunBundle]],
    skipped_methods: list[dict[str, Any]],
    position_df: pd.DataFrame,
    output_dir: Path,
) -> str:
    lines: list[str] = [
        "# Random-NZ Slice Summary",
        "",
        "## Scope",
        f"- output_dir: `{repo_relative(output_dir)}`",
        f"- dataset: `{DATASET}`",
        f"- target_items: `{TARGETS}`",
        f"- victim_models: `{VICTIMS}`",
        f"- collected_at_source_date: `2026-04-26`",
        "",
        "## Method Inventory",
        "| method | run_group_key | run_root | attack_method | replacement_topk_ratio | nonzero_action_when_possible | random_seed | seed_mode | notes |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- | --- |",
    ]

    for spec, bundle in loaded_methods:
        actual_ratio = None if spec.key == "clean" else bundle.replacement_topk_ratio
        effective_nonzero = resolve_effective_nonzero_flag(spec, bundle)
        notes: list[str] = []
        if spec.requested_ratio is not None and actual_ratio is not None:
            if not math.isclose(spec.requested_ratio, actual_ratio, rel_tol=0.0, abs_tol=1e-12):
                notes.append(
                    f"requested label {spec.method_label} but artifacts resolve to ratio={actual_ratio}"
                )
        if sorted(bundle.target_items) != sorted(TARGETS) or sorted(bundle.victims) != sorted(VICTIMS):
            notes.append("slice extracted from a larger completed run group")
        lines.append(
            "| "
            + " | ".join(
                [
                    spec.method_label,
                    str(bundle.run_group_key),
                    f"`{repo_relative(bundle.run_root)}`",
                    str(bundle.attack_method),
                    "" if actual_ratio is None else f"{actual_ratio}",
                    "null" if effective_nonzero is None else str(bool(effective_nonzero)).lower(),
                    str(bundle.seeds.get("fake_session_seed")),
                    "single-seed",
                    "; ".join(notes) if notes else "",
                ]
            )
            + " |"
        )

    if skipped_methods:
        lines.extend(["", "## Skipped Optional Methods"])
        for skipped in skipped_methods:
            lines.append(
                f"- `{skipped['method_label']}` skipped: {skipped['reason']} "
                f"(`{skipped['run_root']}`)"
            )

    random_bundle = next(bundle for spec, bundle in loaded_methods if spec.key == "random_nz_r1")
    random_ratio = random_bundle.replacement_topk_ratio
    random_target_rows = position_df[position_df["method_key"] == "random_nz_r1"].copy()
    random_observed_pos0 = None
    if not random_target_rows.empty:
        random_observed_pos0 = float(random_target_rows["pos0_pct"].mean())
    lines.extend(
        [
            "",
            "## Random-NZ Run",
            f"- run_root: `{repo_relative(random_bundle.run_root)}`",
            f"- run_group_key: `{random_bundle.run_group_key}`",
            f"- dataset: `{DATASET}`",
            f"- target_items: `{TARGETS}`",
            f"- victim_models: `{VICTIMS}`",
            f"- replacement_topk_ratio: `{random_ratio}`",
            "- nonzero_action_when_possible: `method semantics only; no explicit position_opt flag for this baseline`",
            f"- random_seed: `{random_bundle.seeds.get('fake_session_seed')}`",
            "- seed_mode: `single-seed`",
        ]
    )
    if random_ratio is not None and not math.isclose(random_ratio, 1.0, rel_tol=0.0, abs_tol=1e-12):
        lines.append(
            "- note: the provided `Random-NZ@1.0` run root actually resolves to "
            f"`replacement_topk_ratio={random_ratio}` from `resolved_config.json`."
        )
    if random_observed_pos0 is not None:
        lines.append(
            f"- observed mean pos0 percentage from `position_stats.json`: `{random_observed_pos0:.4f}%`"
        )
    lines.append(
        "- sanity check (`final_pos0_pct == 0` and `fallback_to_pos0_only_count == 0`): "
        "`not available for Random-NZ artifacts; these diagnostics are only recorded for position-opt runs here`"
    )

    shared_policy_rows = position_df[position_df["method_key"] == "shared_policy_nz_r1"].copy()
    if not shared_policy_rows.empty:
        all_zero_pos0 = bool((shared_policy_rows["final_pos0_pct"].fillna(float("nan")) == 0.0).all())
        all_zero_fallback = bool(
            (shared_policy_rows["fallback_to_pos0_only_count"].fillna(float("nan")) == 0.0).all()
        )
        lines.extend(
            [
                "",
                "## SharedPolicy-NZ Reference Sanity",
                f"- final_pos0_pct is 0 for all selected targets: `{str(all_zero_pos0).lower()}`",
                f"- fallback_to_pos0_only_count is 0 for all selected targets: `{str(all_zero_fallback).lower()}`",
            ]
        )

    return "\n".join(lines) + "\n"


def repo_relative(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
