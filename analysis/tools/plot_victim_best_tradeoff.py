#!/usr/bin/env python3
"""Create one Recall/MRR best-rate plot per victim on a normalized 0–24 scale."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/comparisons/formal_6method6victim_all"
INPUT = BASE / "merged_long_table.csv"
OUTPUT = BASE / "tradeoff_recall_mrr_vs_clean"
METHODS = {"random_nz": "Random-NZ", "poisoning_ssl_sbr": "Poisoning-SSL-NZ",
           "creat": "CREAT-NZ", "generated_direct_cem": "Generated CEM",
           "copy_direct_cem": "Copy CEM"}
COLORS = dict(zip(METHODS.values(), ["#4C78A8", "#E45756", "#72B7B2", "#F2A541", "#7A5195"]))
SIZES = dict(zip(METHODS.values(), [210, 170, 130, 90, 55]))


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def main() -> None:
    df = pd.read_csv(INPUT, low_memory=False)
    required = {"dataset", "target_type", "victim_model", "target_item", "attack_method", "metric", "k", "value"}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"Missing required columns: {missing}")
    selected = df[df.attack_method.isin(METHODS) & df.metric.isin(["recall", "mrr"])
                  & pd.to_numeric(df.k, errors="coerce").eq(20)].copy()
    selected["value"] = pd.to_numeric(selected.value, errors="raise")
    selected["method"] = selected.attack_method.map(METHODS)
    units = selected.groupby(["dataset", "target_type", "victim_model", "metric", "method"], sort=True).value.mean().reset_index()
    maxima = units.groupby(["dataset", "target_type", "victim_model", "metric"]).value.transform("max")
    units["is_best"] = units.value.sub(maxima).abs().le(1e-12)
    counts = units.groupby(["victim_model", "method", "metric"], sort=True).is_best.agg(
        best_count="sum", n_units="size").reset_index()
    counts["best_score"] = counts.best_count / counts.n_units * 24.0
    points = counts.pivot(index=["victim_model", "method"], columns="metric", values="best_score").reset_index()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    handles = [Line2D([0], [0], marker="o", ls="none", color=COLORS[name], label=name, markersize=8)
               for name in METHODS.values()]
    for victim in sorted(points.victim_model.unique()):
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
        victim_points = points[points.victim_model.eq(victim)].set_index("method")
        for method in METHODS.values():
            row = victim_points.loc[method]
            ax.scatter(row.recall, row.mrr, s=SIZES[method], color=COLORS[method],
                       edgecolor="white", lw=.9, zorder=3, clip_on=False)
        ax.set(xlim=(0, 24), ylim=(0, 24), xlabel="Target Recall@20 best score (0–24)",
               ylabel="Target MRR@20 best score (0–24)")
        ax.set_xticks(range(0, 25, 4)); ax.set_yticks(range(0, 25, 4))
        ax.grid(True, ls="--", lw=.6, color="#D6D6D6", alpha=.8)
        ax.set_title(f"Recall–MRR Best Trade-off — {victim}", fontsize=14, pad=12)
        ax.legend(handles=handles, title="Attack method", loc="upper left", frameon=False)
        fig.tight_layout(); fig.savefig(OUTPUT / f"best_tradeoff_{slug(victim)}.png", dpi=300,
                                        bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Detected columns: {list(df.columns)}")
    print(f"Selected rows: {len(selected)}; victims: {sorted(points.victim_model.unique())}")
    for victim in sorted(points.victim_model.unique()): print(OUTPUT / f"best_tradeoff_{slug(victim)}.png")


if __name__ == "__main__": main()
