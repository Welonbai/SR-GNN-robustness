#!/usr/bin/env python3
"""Matplotlib plot helpers for Prefix vs PosOptMVP diagnosis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_targeted_metrics_overview(metrics_comparison: pd.DataFrame, output_path: Path, *, title: str) -> None:
    """Plot targeted Recall/MRR comparisons for Prefix vs PosOpt."""
    subset = metrics_comparison[
        (metrics_comparison["metric_scope"] == "targeted")
        & (metrics_comparison["metric_name"].isin(["recall", "mrr"]))
    ].copy()
    subset["label"] = (
        "T"
        + subset["target_item"].astype(str)
        + " / "
        + subset["victim_model"].astype(str)
        + "\n"
        + subset["metric_name"].str.upper()
        + "@"
        + subset["k"].astype(str)
    )
    subset = subset.sort_values(["target_item", "victim_model", "metric_name", "k"]).reset_index(drop=True)
    x = np.arange(len(subset))
    width = 0.38

    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(max(12, len(subset) * 0.7), 6.5))
    ax.bar(x - width / 2, subset["prefix_value"], width=width, label="Prefix", color="#E07A5F")
    ax.bar(x + width / 2, subset["posopt_value"], width=width, label="PosOptMVP", color="#3D405B")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["label"], rotation=45, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_position_histogram(
    per_session_join: pd.DataFrame,
    output_path: Path,
    *,
    target_item: int,
    title: str,
) -> None:
    """Plot a discrete final-position distribution comparison."""
    subset = per_session_join.loc[per_session_join["target_item"] == target_item].copy()
    prefix_counts = subset["prefix_selected_position"].value_counts(normalize=True).sort_index()
    posopt_counts = subset["posopt_selected_position"].value_counts(normalize=True).sort_index()
    positions = sorted(set(prefix_counts.index.tolist()) | set(posopt_counts.index.tolist()))
    prefix_values = [float(prefix_counts.get(position, 0.0)) for position in positions]
    posopt_values = [float(posopt_counts.get(position, 0.0)) for position in positions]
    x = np.arange(len(positions))
    width = 0.38

    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(max(10, len(positions) * 0.45), 5.5))
    ax.bar(x - width / 2, prefix_values, width=width, label="Prefix", color="#E07A5F")
    ax.bar(x + width / 2, posopt_values, width=width, label="PosOptMVP", color="#3D405B")
    ax.set_xticks(x)
    ax.set_xticklabels([str(position) for position in positions], rotation=0)
    ax.set_xlabel("Selected replacement position")
    ax.set_ylabel("Fraction of sessions")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_normalized_position_histogram(
    per_session_join: pd.DataFrame,
    output_path: Path,
    *,
    target_item: int,
    title: str,
) -> None:
    """Plot normalized selected-position distributions."""
    subset = per_session_join.loc[per_session_join["target_item"] == target_item].copy()
    bins = np.linspace(0.0, 1.0, 21)
    weights = np.full(len(subset), 1.0 / len(subset))

    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.hist(
        subset["normalized_prefix_position"],
        bins=bins,
        weights=weights,
        alpha=0.6,
        label="Prefix",
        color="#E07A5F",
    )
    ax.hist(
        subset["normalized_posopt_position"],
        bins=bins,
        weights=weights,
        alpha=0.6,
        label="PosOptMVP",
        color="#3D405B",
    )
    ax.set_xlabel("Normalized position (position / session_length)")
    ax.set_ylabel("Fraction of sessions")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_training_curve(
    training_dynamics: pd.DataFrame,
    output_path: Path,
    *,
    target_item: int,
    value_column: str,
    ylabel: str,
    title: str,
) -> None:
    """Plot one PosOpt training curve for one target."""
    subset = training_dynamics.loc[training_dynamics["target_item"] == target_item].copy()
    ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        subset["outer_step"],
        subset[value_column],
        marker="o",
        linewidth=1.8,
        markersize=4,
        color="#3D405B",
    )
    ax.set_xlabel("Outer step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
