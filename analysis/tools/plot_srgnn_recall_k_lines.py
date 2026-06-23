#!/usr/bin/env python3
"""Plot Target Recall@K curves for five attacks in four dataset/cohort panels, one figure per victim."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/comparisons/formal_6method6victim_all"
INPUT = BASE / "merged_long_table.csv"
OUTPUT = BASE / "k_line_compare_srgnn"
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
    "generated_direct_cem": "Generated CEM",
    "copy_direct_cem": "Copy CEM",
}
COLORS = dict(zip(METHODS.values(), ["#4C78A8", "#E45756", "#72B7B2", "#F2A541", "#7A5195"]))
MARKERS = dict(zip(METHODS.values(), ["o", "s", "^", "D", "P"]))


def plot_victim(df: pd.DataFrame, victim: str, victim_label: str) -> Path:
    selected = df[
        df["victim_model"].eq(victim)
        & df["attack_method"].isin(METHODS)
        & df["metric"].eq("recall")
        & pd.to_numeric(df["k"], errors="coerce").isin(KS)
    ].copy()
    selected["k"] = pd.to_numeric(selected["k"], errors="raise").astype(int)
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    selected["method"] = selected["attack_method"].map(METHODS)
    curves = selected.groupby(["dataset", "target_type", "method", "k"], sort=True)["value"].mean().reset_index()

    panels = [("diginetica", "popular"), ("diginetica", "unpopular"),
              ("yoochoose1_64", "popular"), ("yoochoose1_64", "unpopular")]
    dataset_names = {"diginetica": "Diginetica", "yoochoose1_64": "Yoochoose 1/64"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.3), sharex=True, sharey=True)
    for ax, (dataset, cohort) in zip(axes.flat, panels, strict=True):
        panel = curves[curves.dataset.eq(dataset) & curves.target_type.eq(cohort)]
        if panel.empty: raise ValueError(f"Missing panel {dataset}/{cohort}")
        for method in METHODS.values():
            line = panel[panel.method.eq(method)].sort_values("k")
            if line.k.tolist() != KS: raise ValueError(f"Incomplete K values for {dataset}/{cohort}/{method}")
            ax.plot(line.k, line.value, color=COLORS[method], marker=MARKERS[method],
                    linewidth=2.0, markersize=6.5, markeredgecolor="white", markeredgewidth=.6)
        ax.set_title(f"{dataset_names[dataset]} / {cohort.capitalize()}", fontsize=13, pad=8)
        ax.set_xticks(KS); ax.grid(True, linestyle="--", linewidth=.6, color="#D5D5D5", alpha=.8)
    y_min, y_max = float(curves.value.min()), float(curves.value.max())
    padding = max((y_max - y_min) * .1, .02)
    for ax in axes.flat: ax.set_ylim(max(0, y_min - padding), min(1, y_max + padding))
    for ax in axes[-1]: ax.set_xlabel("K")
    for ax in axes[:, 0]: ax.set_ylabel("Target Recall@K")
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], linewidth=2,
                      markeredgecolor="white", label=m) for m in METHODS.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=5, frameon=False)
    fig.suptitle(f"{victim_label} Target Recall@K Across Attack Methods", fontsize=17, y=.975)
    fig.subplots_adjust(left=.085, right=.985, top=.91, bottom=.13, hspace=.24, wspace=.12)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{victim}_target_recall_k_lines.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"{victim}: selected rows={len(selected)}; aggregated curve rows={len(curves)}; output={path}")
    return path


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    outputs = [plot_victim(df, victim, victim_label) for victim, victim_label in VICTIMS.items()]
    print("Output files:")
    for path in outputs:
        print(path)


if __name__ == "__main__": main()
