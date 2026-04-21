#!/usr/bin/env python3
"""Core diagnosis computations for Prefix vs PosOptMVP."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .loaders import DiagnosisError, SharedArtifacts, TargetMethodArtifacts, VictimRunArtifacts


COMPARISON_METRICS: tuple[str, ...] = (
    "targeted_recall@10",
    "targeted_recall@20",
    "targeted_recall@30",
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_mrr@30",
    "targeted_ndcg@10",
    "targeted_ndcg@20",
    "targeted_ndcg@30",
    "ground_truth_recall@10",
    "ground_truth_recall@20",
    "ground_truth_recall@30",
    "ground_truth_mrr@10",
    "ground_truth_mrr@20",
    "ground_truth_mrr@30",
    "ground_truth_ndcg@10",
    "ground_truth_ndcg@20",
    "ground_truth_ndcg@30",
)


@dataclass(frozen=True)
class MetricPair:
    """One Prefix/PosOpt victim-run pair for metric comparison."""

    prefix: VictimRunArtifacts
    posopt: VictimRunArtifacts


def normalized_position(position: int, session_length: int) -> float:
    """Compute the requested normalized position statistic."""
    if session_length <= 0:
        raise DiagnosisError(f"Encountered non-positive session length: {session_length}.")
    return float(position) / float(session_length)


def serialize_session(session: list[int]) -> str:
    """Serialize one session deterministically for CSV output."""
    return json.dumps(session, separators=(",", ":"))


def replace_at_position(session: list[int], position: int, target_item: int) -> list[int]:
    """Replace one item inside a session."""
    if position < 0 or position >= len(session):
        raise DiagnosisError(
            f"Replacement position {position} is out of bounds for session length {len(session)}."
        )
    replaced = list(session)
    replaced[position] = target_item
    return replaced


def build_metrics_comparison_dataframe(metric_pairs: list[MetricPair], *, case_id: str) -> pd.DataFrame:
    """Build one long metrics comparison table."""
    rows: list[dict[str, Any]] = []
    for pair in metric_pairs:
        prefix_metrics = pair.prefix.metrics.get("metrics", {})
        posopt_metrics = pair.posopt.metrics.get("metrics", {})
        if not isinstance(prefix_metrics, dict) or not isinstance(posopt_metrics, dict):
            raise DiagnosisError("metrics.json is missing the expected 'metrics' object.")
        for metric_key in COMPARISON_METRICS:
            if metric_key.startswith("ground_truth_"):
                scope = "ground_truth"
                suffix = metric_key[len("ground_truth_") :]
            elif metric_key.startswith("targeted_"):
                scope = "targeted"
                suffix = metric_key[len("targeted_") :]
            else:
                raise DiagnosisError(f"Unsupported metric key format: '{metric_key}'.")
            metric_name, k_string = suffix.split("@", maxsplit=1)
            prefix_value = prefix_metrics.get(metric_key)
            posopt_value = posopt_metrics.get(metric_key)
            rows.append(
                {
                    "case_id": case_id,
                    "target_item": pair.prefix.target_item,
                    "victim_model": pair.prefix.victim_model,
                    "metric_key": metric_key,
                    "metric_scope": scope,
                    "metric_name": metric_name,
                    "k": int(k_string),
                    "prefix_method": pair.prefix.method,
                    "posopt_method": pair.posopt.method,
                    "prefix_value": prefix_value,
                    "posopt_value": posopt_value,
                    "posopt_minus_prefix": (
                        None
                        if prefix_value is None or posopt_value is None
                        else float(posopt_value) - float(prefix_value)
                    ),
                    "prefix_run_path": pair.prefix.run_dir.as_posix(),
                    "posopt_run_path": pair.posopt.run_dir.as_posix(),
                }
            )
    dataframe = pd.DataFrame(rows)
    sort_columns = ["target_item", "victim_model", "metric_scope", "metric_name", "k"]
    return dataframe.sort_values(sort_columns).reset_index(drop=True)


def validate_aligned_lengths(
    shared: SharedArtifacts,
    prefix_target: TargetMethodArtifacts,
    posopt_target: TargetMethodArtifacts,
) -> None:
    """Check that all per-session target-level artifacts align on session index."""
    expected_length = len(shared.fake_sessions)
    observed_lengths = {
        "shared fake sessions": expected_length,
        "Prefix selected positions": len(prefix_target.selected_positions),
        "PosOpt selected positions": len(posopt_target.selected_positions),
    }
    if posopt_target.optimized_poisoned_sessions is not None:
        observed_lengths["PosOpt optimized poisoned sessions"] = len(posopt_target.optimized_poisoned_sessions)
    unique_lengths = set(observed_lengths.values())
    if len(unique_lengths) != 1:
        rendered = ", ".join(f"{label}={length}" for label, length in observed_lengths.items())
        raise DiagnosisError(
            f"Per-session artifact lengths do not align for target {prefix_target.target_item}: {rendered}."
        )


def build_per_session_join_dataframe(
    shared: SharedArtifacts,
    prefix_target: TargetMethodArtifacts,
    posopt_target: TargetMethodArtifacts,
    *,
    case_id: str,
    shared_across_victims: bool,
    victim_scope: str | None,
) -> pd.DataFrame:
    """Join Prefix and PosOpt per-session selected positions on the shared fake-session index."""
    validate_aligned_lengths(shared, prefix_target, posopt_target)
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(shared.fake_sessions):
        session_length = len(session)
        prefix_record = prefix_target.selected_positions[session_index]
        posopt_record = posopt_target.selected_positions[session_index]
        prefix_position = int(prefix_record["position"])
        posopt_position = int(posopt_record["position"])
        if prefix_position < 0 or prefix_position >= session_length:
            raise DiagnosisError(
                f"Prefix position {prefix_position} is out of bounds for session {session_index} "
                f"(length={session_length}) for target {prefix_target.target_item}."
            )
        if posopt_position < 0 or posopt_position >= session_length:
            raise DiagnosisError(
                f"PosOpt position {posopt_position} is out of bounds for session {session_index} "
                f"(length={session_length}) for target {posopt_target.target_item}."
            )
        rows.append(
            {
                "case_id": case_id,
                "target_item": prefix_target.target_item,
                "shared_across_victims": shared_across_victims,
                "victim_scope": victim_scope,
                "session_index": session_index,
                "session_length": session_length,
                "prefix_selected_position": prefix_position,
                "posopt_selected_position": posopt_position,
                "normalized_prefix_position": normalized_position(prefix_position, session_length),
                "normalized_posopt_position": normalized_position(posopt_position, session_length),
                "position_delta_posopt_minus_prefix": posopt_position - prefix_position,
                "prefix_score": prefix_record.get("score"),
                "posopt_score": posopt_record.get("score"),
                "posopt_candidate_index": posopt_record.get("candidate_index"),
                "same_selected_position": prefix_position == posopt_position,
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item", "session_index"]).reset_index(drop=True)


def summarize_positions(
    per_session_join: pd.DataFrame,
    *,
    shared_across_victims: bool,
    victim_scope: str | None,
) -> pd.DataFrame:
    """Summarize final selected positions for each target and method."""
    rows: list[dict[str, Any]] = []
    for target_item, group in per_session_join.groupby("target_item", sort=True):
        for method, position_column, normalized_column in (
            ("prefix_nonzero_when_possible", "prefix_selected_position", "normalized_prefix_position"),
            ("position_opt_mvp", "posopt_selected_position", "normalized_posopt_position"),
        ):
            positions = group[position_column].astype(float)
            normalized = group[normalized_column].astype(float)
            rows.append(
                {
                    "target_item": int(target_item),
                    "shared_across_victims": shared_across_victims,
                    "victim_scope": victim_scope,
                    "method": method,
                    "session_count": int(len(group)),
                    "mean_selected_position": float(positions.mean()),
                    "median_selected_position": float(positions.median()),
                    "mean_normalized_position": float(normalized.mean()),
                    "median_normalized_position": float(normalized.median()),
                    "fraction_position0": float((positions == 0).mean()),
                    "fraction_top10pct": float((normalized <= 0.10).mean()),
                    "fraction_top20pct": float((normalized <= 0.20).mean()),
                    "min_selected_position": int(positions.min()),
                    "max_selected_position": int(positions.max()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_item", "method"]).reset_index(drop=True)


def build_context_join_dataframe(
    shared: SharedArtifacts,
    prefix_target: TargetMethodArtifacts,
    posopt_target: TargetMethodArtifacts,
    *,
    case_id: str,
    shared_across_victims: bool,
    victim_scope: str | None,
) -> pd.DataFrame:
    """Recover local context around the replaced item for both methods."""
    validate_aligned_lengths(shared, prefix_target, posopt_target)
    if posopt_target.optimized_poisoned_sessions is None:
        raise DiagnosisError(
            f"PosOpt target {posopt_target.target_item} is missing optimized poisoned sessions."
        )
    rows: list[dict[str, Any]] = []
    mismatched_indices: list[int] = []
    for session_index, session in enumerate(shared.fake_sessions):
        prefix_position = int(prefix_target.selected_positions[session_index]["position"])
        posopt_position = int(posopt_target.selected_positions[session_index]["position"])
        prefix_poisoned = replace_at_position(session, prefix_position, prefix_target.target_item)
        posopt_reconstructed = replace_at_position(session, posopt_position, posopt_target.target_item)
        posopt_artifact = list(posopt_target.optimized_poisoned_sessions[session_index])
        if posopt_reconstructed != posopt_artifact:
            mismatched_indices.append(session_index)
        prefix_original_item = session[prefix_position]
        posopt_original_item = session[posopt_position]
        rows.append(
            {
                "case_id": case_id,
                "target_item": prefix_target.target_item,
                "shared_across_victims": shared_across_victims,
                "victim_scope": victim_scope,
                "session_index": session_index,
                "session_length": len(session),
                "original_fake_session": serialize_session(session),
                "target_item_value": prefix_target.target_item,
                "prefix_selected_position": prefix_position,
                "prefix_original_item": prefix_original_item,
                "prefix_left_neighbor": None if prefix_position == 0 else session[prefix_position - 1],
                "prefix_right_neighbor": (
                    None if prefix_position + 1 >= len(session) else session[prefix_position + 1]
                ),
                "prefix_poisoned_session": serialize_session(prefix_poisoned),
                "prefix_score": prefix_target.selected_positions[session_index].get("score"),
                "posopt_selected_position": posopt_position,
                "posopt_original_item": posopt_original_item,
                "posopt_left_neighbor": None if posopt_position == 0 else session[posopt_position - 1],
                "posopt_right_neighbor": (
                    None if posopt_position + 1 >= len(session) else session[posopt_position + 1]
                ),
                "posopt_poisoned_session": serialize_session(posopt_artifact),
                "posopt_candidate_index": posopt_target.selected_positions[session_index].get("candidate_index"),
                "posopt_score": posopt_target.selected_positions[session_index].get("score"),
                "posopt_reconstruction_matches_artifact": posopt_reconstructed == posopt_artifact,
                "same_selected_position": prefix_position == posopt_position,
                "same_replaced_original_item": prefix_original_item == posopt_original_item,
                "same_left_neighbor": (
                    (None if prefix_position == 0 else session[prefix_position - 1])
                    == (None if posopt_position == 0 else session[posopt_position - 1])
                ),
                "same_right_neighbor": (
                    (None if prefix_position + 1 >= len(session) else session[prefix_position + 1])
                    == (None if posopt_position + 1 >= len(session) else session[posopt_position + 1])
                ),
            }
        )
    if mismatched_indices:
        preview = ", ".join(str(index) for index in mismatched_indices[:10])
        raise DiagnosisError(
            f"PosOpt optimized poisoned sessions do not match reconstructed replacement sessions for "
            f"target {posopt_target.target_item}. Example session indices: {preview}."
        )
    return pd.DataFrame(rows).sort_values(["target_item", "session_index"]).reset_index(drop=True)


def summarize_context(
    context_join: pd.DataFrame,
    *,
    shared_across_victims: bool,
    victim_scope: str | None,
) -> pd.DataFrame:
    """Summarize whether Prefix and PosOpt choose similar or different local contexts."""
    rows: list[dict[str, Any]] = []
    for target_item, group in context_join.groupby("target_item", sort=True):
        rows.append(
            {
                "target_item": int(target_item),
                "shared_across_victims": shared_across_victims,
                "victim_scope": victim_scope,
                "session_count": int(len(group)),
                "fraction_same_selected_position": float(group["same_selected_position"].mean()),
                "fraction_same_replaced_original_item": float(group["same_replaced_original_item"].mean()),
                "fraction_same_left_neighbor": float(group["same_left_neighbor"].mean()),
                "fraction_same_right_neighbor": float(group["same_right_neighbor"].mean()),
                "fraction_prefix_earlier": float(
                    (group["prefix_selected_position"] < group["posopt_selected_position"]).mean()
                ),
                "fraction_posopt_earlier": float(
                    (group["posopt_selected_position"] < group["prefix_selected_position"]).mean()
                ),
                "mean_abs_position_delta": float(
                    (group["posopt_selected_position"] - group["prefix_selected_position"]).abs().mean()
                ),
                "validated_posopt_reconstruction_fraction": float(
                    group["posopt_reconstruction_matches_artifact"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item"]).reset_index(drop=True)


def build_top_context_table(
    context_join: pd.DataFrame,
    *,
    prefix_column: str,
    posopt_column: str,
    label: str,
    top_n: int,
) -> pd.DataFrame:
    """Build one top-items frequency table for a context role."""
    rows: list[dict[str, Any]] = []
    for target_item, group in context_join.groupby("target_item", sort=True):
        for method, column in (
            ("prefix_nonzero_when_possible", prefix_column),
            ("position_opt_mvp", posopt_column),
        ):
            series = group[column].dropna()
            counts = Counter(int(value) for value in series.tolist())
            non_null_total = int(sum(counts.values()))
            if non_null_total == 0:
                continue
            for rank, (item_id, count) in enumerate(
                sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n],
                start=1,
            ):
                rows.append(
                    {
                        "target_item": int(target_item),
                        "method": method,
                        "context_role": label,
                        "rank": rank,
                        "item_id": item_id,
                        "count": int(count),
                        "ratio_among_non_null": float(count / non_null_total),
                        "non_null_total": non_null_total,
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_item", "method", "rank"]).reset_index(drop=True)


def build_training_dynamics_dataframe(
    shared: SharedArtifacts,
    posopt_target: TargetMethodArtifacts,
    *,
    case_id: str,
) -> pd.DataFrame:
    """Derive step-wise PosOpt training dynamics summaries."""
    if posopt_target.training_history is None:
        raise DiagnosisError(f"PosOpt target {posopt_target.target_item} is missing training history.")
    steps = posopt_target.training_history.get("training_history")
    if not isinstance(steps, list):
        raise DiagnosisError(
            f"PosOpt training history for target {posopt_target.target_item} is missing 'training_history'."
        )
    session_lengths = [len(session) for session in shared.fake_sessions]
    rows: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            raise DiagnosisError(
                f"PosOpt training history for target {posopt_target.target_item} contains a non-dict step."
            )
        selected_positions = step.get("selected_positions")
        selected_candidate_indices = step.get("selected_candidate_indices")
        if not isinstance(selected_positions, list) or not isinstance(selected_candidate_indices, list):
            raise DiagnosisError(
                f"PosOpt training history for target {posopt_target.target_item} is missing selected positions "
                "or candidate indices."
            )
        if len(selected_positions) != len(session_lengths):
            raise DiagnosisError(
                f"PosOpt training step {step.get('outer_step')} for target {posopt_target.target_item} has "
                f"{len(selected_positions)} selected positions, expected {len(session_lengths)}."
            )
        positions = np.asarray(selected_positions, dtype=float)
        normalized = np.asarray(
            [normalized_position(int(position), session_length) for position, session_length in zip(selected_positions, session_lengths)],
            dtype=float,
        )
        position_counts = Counter(int(position) for position in selected_positions)
        max_position_share = max(position_counts.values()) / len(selected_positions)
        candidate_counts = Counter(int(index) for index in selected_candidate_indices)
        max_candidate_share = (
            max(candidate_counts.values()) / len(selected_candidate_indices)
            if selected_candidate_indices
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "target_item": posopt_target.target_item,
                "outer_step": int(step["outer_step"]),
                "reward": step.get("reward"),
                "baseline": step.get("baseline"),
                "advantage": step.get("advantage"),
                "mean_entropy": step.get("mean_entropy"),
                "policy_loss": step.get("policy_loss"),
                "target_utility": step.get("target_utility"),
                "gt_drop": step.get("gt_drop"),
                "gt_penalty": step.get("gt_penalty"),
                "average_selected_position": float(np.mean(positions)),
                "median_selected_position": float(np.median(positions)),
                "average_normalized_position": float(np.mean(normalized)),
                "fraction_position0": float(np.mean(positions == 0)),
                "fraction_top10pct": float(np.mean(normalized <= 0.10)),
                "fraction_top20pct": float(np.mean(normalized <= 0.20)),
                "unique_selected_positions": int(len(position_counts)),
                "max_position_share": float(max_position_share),
                "unique_selected_candidate_indices": int(len(candidate_counts)),
                "max_candidate_index_share": None if max_candidate_share is None else float(max_candidate_share),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item", "outer_step"]).reset_index(drop=True)


def summarize_training_dynamics(training_dynamics: pd.DataFrame) -> pd.DataFrame:
    """Build one compact target-level PosOpt training summary."""
    rows: list[dict[str, Any]] = []
    for target_item, group in training_dynamics.groupby("target_item", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        peak = group.loc[group["reward"].astype(float).idxmax()]
        rows.append(
            {
                "target_item": int(target_item),
                "step_count": int(len(group)),
                "initial_reward": float(first["reward"]),
                "final_reward": float(last["reward"]),
                "peak_reward": float(peak["reward"]),
                "peak_reward_step": int(peak["outer_step"]),
                "initial_mean_entropy": float(first["mean_entropy"]),
                "final_mean_entropy": float(last["mean_entropy"]),
                "entropy_drop": float(first["mean_entropy"]) - float(last["mean_entropy"]),
                "initial_average_selected_position": float(first["average_selected_position"]),
                "final_average_selected_position": float(last["average_selected_position"]),
                "final_fraction_position0": float(last["fraction_position0"]),
                "final_fraction_top10pct": float(last["fraction_top10pct"]),
                "final_fraction_top20pct": float(last["fraction_top20pct"]),
                "final_max_position_share": float(last["max_position_share"]),
                "final_unique_selected_positions": int(last["unique_selected_positions"]),
                "final_max_candidate_index_share": (
                    None
                    if pd.isna(last["max_candidate_index_share"])
                    else float(last["max_candidate_index_share"])
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["target_item"]).reset_index(drop=True)
