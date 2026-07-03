#!/usr/bin/env python3
"""Plot Yoochoose unpopular Target Recall@K curves for all six victims in one figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/comparisons/formal_6method6victim_all"
INPUT = BASE / "merged_long_table.csv"
OUTPUT = BASE / "k_line_compare"
OUTPUT_IMAGE = OUTPUT / "yoochoose_unpopular_all_victims_target_recall_k_lines.png"
OUTPUT_MANIFEST = OUTPUT / "yoochoose_unpopular_all_victims_manifest.json"

DATASET = "yoochoose1_64"
TARGET_TYPE = "unpopular"
KS = [10, 20, 30, 40, 50]
VICTIMS = {
    "srgnn": "SRGNN",
    "miasrec": "MiaSRec",
    "tron": "TRON",
    "mdhg": "MDHG",
    "freqrec": "FreqRec",
    "wearec": "WEARec",
}
METHODS = {
    "random_nz": "Random-NZ",
    "poisoning_ssl_sbr": "Poisoning-SSL-NZ",
    "creat": "CREAT-NZ",
    "generated_direct_cem": "DA-CEM-G",
    "copy_direct_cem": "DA-CEM-C",
}
COLORS = {
    "Random-NZ": "#4C78A8",
    "Poisoning-SSL-NZ": "#7A5195",
    "CREAT-NZ": "#72B7B2",
    "DA-CEM-G": "#F2A541",
    "DA-CEM-C": "#C00000",
}
MARKERS = {
    "Random-NZ": "o",
    "Poisoning-SSL-NZ": "s",
    "CREAT-NZ": "^",
    "DA-CEM-G": "D",
    "DA-CEM-C": "P",
}


def build_curves(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one mean Recall@K curve per victim/method."""
    selected = dataframe[
        dataframe["dataset"].eq(DATASET)
        & dataframe["target_type"].eq(TARGET_TYPE)
        & dataframe["victim_model"].isin(VICTIMS)
        & dataframe["attack_method"].isin(METHODS)
        & dataframe["metric"].eq("recall")
        & pd.to_numeric(dataframe["k"], errors="coerce").isin(KS)
    ].copy()
    if selected.empty:
        raise ValueError("No rows selected for Yoochoose unpopular Target Recall@K plot.")

    selected["k"] = pd.to_numeric(selected["k"], errors="raise").astype(int)
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    selected["method"] = selected["attack_method"].map(METHODS)
    curves = (
        selected.groupby(["victim_model", "method", "k"], sort=True)["value"]
        .mean()
        .reset_index()
    )

    expected_rows = len(VICTIMS) * len(METHODS) * len(KS)
    if len(curves) != expected_rows:
        raise ValueError(f"Expected {expected_rows} curve rows, got {len(curves)}.")
    for victim in VICTIMS:
        for method in METHODS.values():
            line = curves[curves["victim_model"].eq(victim) & curves["method"].eq(method)]
            if line.sort_values("k")["k"].tolist() != KS:
                raise ValueError(f"Incomplete K values for {victim}/{method}.")
    return curves


def plot_curves(curves: pd.DataFrame) -> Path:
    """Render the six-victim panel figure."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 8.8), sharex=True, sharey=True)

    for ax, (victim, victim_label) in zip(axes.flat, VICTIMS.items(), strict=True):
        panel = curves[curves["victim_model"].eq(victim)]
        for method in METHODS.values():
            line = panel[panel["method"].eq(method)].sort_values("k")
            ax.plot(
                line["k"],
                line["value"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2.0,
                markersize=6.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=method,
            )
        ax.set_title(victim_label, fontsize=13, pad=8)
        ax.set_xticks(KS)
        ax.grid(True, linestyle="--", linewidth=0.6, color="#D5D5D5", alpha=0.8)

    y_min, y_max = float(curves["value"].min()), float(curves["value"].max())
    padding = max((y_max - y_min) * 0.1, 0.02)
    for ax in axes.flat:
        ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))
    for ax in axes[-1, :]:
        ax.set_xlabel("K")
    for ax in axes[:, 0]:
        ax.set_ylabel("Target Recall@K")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2,
            markeredgecolor="white",
            label=method,
        )
        for method in METHODS.values()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=5, frameon=False)
    fig.suptitle("Yoochoose 1/64 Unpopular Target Recall@K Across Victim RS Models", fontsize=17, y=0.975)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.89, bottom=0.13, hspace=0.28, wspace=0.13)
    fig.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUTPUT_IMAGE


def main() -> None:
    """Build and render the plot."""
    dataframe = pd.read_csv(INPUT, low_memory=False)
    curves = build_curves(dataframe)
    output_path = plot_curves(curves)
    OUTPUT_MANIFEST.write_text(
        json.dumps(
            {
                "input": str(INPUT),
                "output": str(output_path),
                "dataset": DATASET,
                "target_type": TARGET_TYPE,
                "metric": "targeted recall",
                "k_values": KS,
                "victims": VICTIMS,
                "attack_methods": METHODS,
                "colors": COLORS,
                "curve_rows": int(len(curves)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"curve rows={len(curves)}")
    print(output_path)
    print(OUTPUT_MANIFEST)


if __name__ == "__main__":
    main()
