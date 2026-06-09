from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = (
    REPO_ROOT
    / "analysis"
    / "diagnosis_outputs"
    / "diginetica_copy_cem_tron_source_diagnostic"
)
TRAIN_SUB_PATH = (
    REPO_ROOT
    / "outputs"
    / "shared"
    / "diginetica"
    / "canonical"
    / "split_diginetica_unified_trainonly1_minitems5_minsess2_testdays7_valid0p1"
    / "train_sub.pkl"
)

RUNS = {
    ("popular", "copy"): {
        "run_dir": "outputs/runs/diginetica/ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train/run_group_57527fe97f",
        "long_table": "results/runs/diginetica_copy_direct_cem_popular_source_methods_sample10/long_table.csv",
    },
    ("popular", "generated"): {
        "run_dir": "outputs/runs/diginetica/diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample/run_group_1edddc067f",
        "long_table": "results/runs/diginetica_generated_direct_cem_popular_source_methods_sample10/long_table.csv",
    },
    ("unpopular", "copy"): {
        "run_dir": "outputs/runs/diginetica/ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train/run_group_e9d162312a",
        "long_table": "results/runs/diginetica_copy_direct_cem_unpopular_source_methods_sample10/long_table.csv",
    },
    ("unpopular", "generated"): {
        "run_dir": "outputs/runs/diginetica/diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_unpopular_sample10_fixed_epoch/run_group_f0d23205ae",
        "long_table": "results/runs/diginetica_generated_direct_cem_unpopular_source_methods_sample10/long_table.csv",
    },
}


def _repo_path(path: str | Path) -> Path:
    return (REPO_ROOT / path).resolve()


def _load_metrics(path: Path) -> dict[tuple[str, int, str, int], float]:
    values: dict[tuple[str, int, str, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[
                (
                    str(row["victim_model"]),
                    int(row["target_item"]),
                    str(row["metric"]),
                    int(row["k"]),
                )
            ] = float(row["value"])
    return values


def _clean_reference_sets() -> tuple[set[tuple[int, int]], set[tuple[int, ...]]]:
    with TRAIN_SUB_PATH.open("rb") as handle:
        train_sub = pickle.load(handle)
    bigrams: set[tuple[int, int]] = set()
    prefixes: set[tuple[int, ...]] = set()
    for raw_session in train_sub:
        session = tuple(int(item) for item in raw_session)
        for index in range(1, len(session)):
            bigrams.add((session[index - 1], session[index]))
            prefixes.add(session[: index + 1])
    return bigrams, prefixes


def _target_row(
    *,
    bucket: str,
    source: str,
    target_dir: Path,
    metrics: dict[tuple[str, int, str, int], float],
    clean_bigrams: set[tuple[int, int]],
    clean_prefixes: set[tuple[int, ...]],
) -> dict[str, object]:
    target = int(target_dir.name)
    candidate_dir = target_dir / "pts_construction_cem" / "top_candidates" / "rank_1"
    with (candidate_dir / "sessions.json").open(encoding="utf-8") as handle:
        sessions = [[int(item) for item in session] for session in json.load(handle)]
    with (candidate_dir / "metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    tron_transition_count = 0
    original_transition_count = 0
    target_positive_count = 0
    weighted_clean_bigram_count = 0
    weighted_non_target_count = 0
    weighted_non_target_clean_count = 0
    weighted_pre_target_count = 0
    weighted_pre_target_clean_count = 0
    expanded_prefix_count = 0
    exact_clean_prefix_count = 0
    target_positions: list[tuple[int, int]] = []
    target_predecessors: Counter[int] = Counter()

    for session in sessions:
        length = len(session)
        original_transition_count += length - 1
        tron_transition_count += length * (length - 1) // 2
        first_target = next(
            (index for index, item in enumerate(session) if item == target),
            length,
        )

        for end in range(2, length + 1):
            expanded_prefix_count += 1
            exact_clean_prefix_count += int(tuple(session[:end]) in clean_prefixes)

        for index in range(1, length):
            weight = length - index
            item = session[index]
            is_clean = (session[index - 1], item) in clean_bigrams
            weighted_clean_bigram_count += weight * int(is_clean)
            if item != target:
                weighted_non_target_count += weight
                weighted_non_target_clean_count += weight * int(is_clean)
            if index < first_target:
                weighted_pre_target_count += weight
                weighted_pre_target_clean_count += weight * int(is_clean)

        for index, item in enumerate(session):
            if item != target:
                continue
            target_positions.append((index, length))
            if index >= 1:
                target_positive_count += length - index
                target_predecessors[session[index - 1]] += 1

    final_loss_path = target_dir / "victims" / "tron" / "train_history.json"
    with final_loss_path.open(encoding="utf-8") as handle:
        train_history = json.load(handle)
    train_losses = [float(value) for value in train_history["train_loss"]]
    construction = metadata["construction_summary"]

    return {
        "bucket": bucket,
        "source": source,
        "target_item": target,
        "tron_targeted_recall@20": metrics[("tron", target, "recall", 20)],
        "srgnn_targeted_recall@20": metrics[("srgnn", target, "recall", 20)],
        "miasrec_targeted_recall@20": metrics[("miasrec", target, "recall", 20)],
        "surrogate_reward": float(metadata["reward"]),
        "tron_epochs": int(train_history["epochs"]),
        "tron_final_train_loss": train_losses[-1],
        "fake_session_count": len(sessions),
        "mean_session_length": mean(len(session) for session in sessions),
        "tron_transition_duplication_factor": (
            tron_transition_count / original_transition_count
        ),
        "target_positive_weight": target_positive_count / tron_transition_count,
        "target_tail_ratio": (
            sum(index == length - 1 for index, length in target_positions)
            / len(sessions)
        ),
        "target_internal_ratio": (
            sum(0 < index < length - 1 for index, length in target_positions)
            / len(sessions)
        ),
        "weighted_clean_bigram_overlap": (
            weighted_clean_bigram_count / tron_transition_count
        ),
        "weighted_non_target_clean_bigram_overlap": (
            weighted_non_target_clean_count / weighted_non_target_count
        ),
        "weighted_pre_target_clean_bigram_overlap": (
            weighted_pre_target_clean_count / weighted_pre_target_count
            if weighted_pre_target_count
            else 0.0
        ),
        "exact_clean_prefix_ratio": exact_clean_prefix_count / expanded_prefix_count,
        "target_predecessor_unique_count": len(target_predecessors),
        "generated_source_ratio": float(
            construction["continuous"]["generated_source_ratio"]
        ),
    }


def _mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    numeric_columns = [
        "tron_targeted_recall@20",
        "srgnn_targeted_recall@20",
        "miasrec_targeted_recall@20",
        "surrogate_reward",
        "tron_epochs",
        "tron_final_train_loss",
        "mean_session_length",
        "tron_transition_duplication_factor",
        "target_positive_weight",
        "target_tail_ratio",
        "target_internal_ratio",
        "weighted_clean_bigram_overlap",
        "weighted_non_target_clean_bigram_overlap",
        "weighted_pre_target_clean_bigram_overlap",
        "exact_clean_prefix_ratio",
        "target_predecessor_unique_count",
        "generated_source_ratio",
    ]
    summaries: list[dict[str, object]] = []
    for bucket, source in RUNS:
        selected = [
            row
            for row in rows
            if row["bucket"] == bucket and row["source"] == source
        ]
        summary: dict[str, object] = {
            "bucket": bucket,
            "source": source,
            "target_count": len(selected),
        }
        for column in numeric_columns:
            summary[column] = mean(float(row[column]) for row in selected)
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _report(summary: list[dict[str, object]]) -> str:
    lookup = {(row["bucket"], row["source"]): row for row in summary}
    lines = [
        "# Diginetica copy-CEM vs generated-CEM: why TRON transfers poorly",
        "",
        "## Result",
        "",
        "The copy-CEM TRON gap is reproducible in the fairer unpopular comparison, so it is not only a popular-run epoch artifact. The main source-specific mechanism is TRON's training-data semantics: the pipeline expands every fake session into all prefix-label pairs, then the TRON exporter reconstructs each pair as a sequence, and TRON again trains every next-item transition in that sequence. A length-L fake session therefore contributes L(L-1)/2 transition losses instead of L-1.",
        "",
        "Copy templates preserve clean train prefixes and transitions. Under TRON's second expansion, those already-clean transitions are repeatedly reinforced, while generated templates contribute mostly novel transitions. The target-positive weight and target-position distributions are nearly the same, so target placement is not the primary explanation.",
        "",
        "## Mean metrics over 10 targets",
        "",
        "| Bucket | Source | TRON R@20 | SR-GNN R@20 | MiaSRec R@20 | Final TRON loss | Clean bigram overlap | Non-target clean overlap | Pre-target clean overlap | Exact clean prefixes | Target-positive weight |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for bucket in ("popular", "unpopular"):
        for source in ("copy", "generated"):
            row = lookup[(bucket, source)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        bucket,
                        source,
                        f"{float(row['tron_targeted_recall@20']):.3f}",
                        f"{float(row['srgnn_targeted_recall@20']):.3f}",
                        f"{float(row['miasrec_targeted_recall@20']):.3f}",
                        f"{float(row['tron_final_train_loss']):.3f}",
                        _pct(row["weighted_clean_bigram_overlap"]),
                        _pct(row["weighted_non_target_clean_bigram_overlap"]),
                        _pct(row["weighted_pre_target_clean_bigram_overlap"]),
                        _pct(row["exact_clean_prefix_ratio"]),
                        _pct(row["target_positive_weight"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Unpopular is the clean source ablation: both copy and generated TRON runs use the same seed, fixed-epoch protocol, and 7 epochs. TRON targeted Recall@20 is still lower for copy (0.347 vs 0.486), while copy is slightly better on SR-GNN and MiaSRec.",
            "- Copy's pre-target transition overlap with clean train is 100%; generated is about 21%. Copy's non-target clean-transition overlap is about 82%; generated is about 18%.",
            "- Copy reaches much lower TRON train loss, consistent with training on many repeated/easy clean transitions rather than stronger target transfer.",
            "- CEM is optimized with an SR-GNN surrogate. Copy has equal or higher surrogate reward, but that reward does not model TRON's repeated transition weighting.",
            "- Popular is additionally confounded: copy uses TRON epoch 3 and fixed-last SR-GNN surrogate training, while generated uses TRON epoch 4 and validation-best surrogate training. Do not attribute the full popular gap to source alone.",
            "",
            "## Recommended confirmation experiment",
            "",
            "Export TRON from raw clean and raw fake sequences exactly once, instead of passing already-expanded prefix-label pairs through `_pairs_to_sequences`. Re-run the unpopular copy/generated pair with the same seed and 7 epochs. If the diagnosis is correct, copy TRON performance should move substantially closer to generated, and the copy train-loss advantage should shrink.",
            "",
            "A second controlled experiment is to keep the current exporter but reweight each reconstructed prefix so every original fake-session transition has total weight one. This isolates transition duplication from source novelty.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    clean_bigrams, clean_prefixes = _clean_reference_sets()
    rows: list[dict[str, object]] = []
    for (bucket, source), config in RUNS.items():
        run_dir = _repo_path(config["run_dir"])
        metrics = _load_metrics(_repo_path(config["long_table"]))
        for target_dir in sorted(
            (run_dir / "targets").iterdir(), key=lambda path: int(path.name)
        ):
            candidate_path = (
                target_dir
                / "pts_construction_cem"
                / "top_candidates"
                / "rank_1"
                / "sessions.json"
            )
            if candidate_path.exists():
                rows.append(
                    _target_row(
                        bucket=bucket,
                        source=source,
                        target_dir=target_dir,
                        metrics=metrics,
                        clean_bigrams=clean_bigrams,
                        clean_prefixes=clean_prefixes,
                    )
                )

    summary = _mean_rows(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUTPUT_DIR / "per_target.csv", rows)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "report.md").write_text(_report(summary), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
