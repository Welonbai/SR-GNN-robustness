from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import random
import time
from collections import Counter
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import Config, load_config
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.evaluator import evaluate_targeted_metrics


TARGET_ITEM = 14514
MAX_EPOCHS = 8
REPORT_CHECKPOINTS = (1, 2, 4, 8)
LOWK_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)
FLOAT_TIE_EPSILON = 1.0e-12
CSV_FIELDS = (
    "target_item",
    "candidate_type",
    "epoch",
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
    "raw_lowk",
    "train_time_seconds",
    "wall_time_seconds",
    "actual_train_epochs",
    "seed",
    "notes",
)

DEFAULT_CONFIG = Path(
    "attack/configs/diginetica_attack_rank_bucket_cem_lowk_ft6500_srgnn_targets11103_14514_5418.yaml"
)
DEFAULT_OUTPUT_DIR = Path("outputs/diagnostics/partial_retrain_alignment_14514")
DEFAULT_SHARED_ATTACK_DIR = Path("outputs/shared/diginetica/attack/attack_shared_1c4345bfa3")
DEFAULT_CEM_DIR = Path(
    "outputs/runs/diginetica/"
    "attack_rank_bucket_cem_lowk_ft6500_srgnn_targets11103_14514_5418/"
    "run_group_ca917142fb/targets/14514"
)
DEFAULT_RANDOM_NZ_DIR = Path(
    "outputs/runs/diginetica/"
    "attack_random_nonzero_when_possible_ratio1_srgnn_sample5/"
    "run_group_720516397a/targets/14514"
)
_TRAINING_DEPS: dict[str, Any] | None = None


def _derive_seed(base_seed: int, *components: object) -> int:
    normalized_base = int(base_seed)
    if not components:
        return normalized_base
    payload = json.dumps(
        [normalized_base, *components],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(derived % (2**31 - 1))


def _load_training_deps() -> dict[str, Any]:
    global _TRAINING_DEPS
    if _TRAINING_DEPS is not None:
        return _TRAINING_DEPS
    try:
        import numpy as np
        import torch

        from attack.models.victim.srgnn_runner import SRGNNVictimRunner
        from pytorch_code.model import forward as srg_forward
        from pytorch_code.model import trans_to_cuda
        from pytorch_code.utils import Data
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        raise RuntimeError(
            "Unable to import the SR-GNN training dependencies (NumPy/PyTorch). "
            "Your traceback shows system Python loading packages from the local "
            "`robustness` environment. Run with the venv interpreter explicitly, for example: "
            "`.\\robustness\\Scripts\\python.exe -m attack.tools.partial_retrain_alignment`. "
            "Also check `where python` and "
            "`python -c \"import sys; print(sys.executable); import numpy; print(numpy.__file__)\"`."
        ) from exc
    _TRAINING_DEPS = {
        "Data": Data,
        "SRGNNVictimRunner": SRGNNVictimRunner,
        "np": np,
        "srg_forward": srg_forward,
        "torch": torch,
        "trans_to_cuda": trans_to_cuda,
    }
    return _TRAINING_DEPS


def _set_seed(seed: int) -> None:
    deps = _load_training_deps()
    np = deps["np"]
    torch = deps["torch"]
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_srgnn_opt_from_train_config(train_config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=int(train_config["batch_size"]),
        hiddenSize=int(train_config["hidden_size"]),
        epoch=int(train_config["epochs"]),
        lr=float(train_config["lr"]),
        lr_dc=float(train_config["lr_dc"]),
        lr_dc_step=int(train_config["lr_dc_step"]),
        l2=float(train_config["l2"]),
        step=int(train_config["step"]),
        patience=int(train_config["patience"]),
        nonhybrid=bool(train_config["nonhybrid"]),
    )


def raw_lowk(metrics: Mapping[str, float | int | None]) -> float:
    missing = [key for key in LOWK_KEYS if metrics.get(key) is None]
    if missing:
        raise ValueError(f"Missing low-k metrics: {', '.join(missing)}")
    return float(sum(float(metrics[key]) for key in LOWK_KEYS) / float(len(LOWK_KEYS)))


def winner_by_raw_lowk(cem_raw_lowk: float, random_raw_lowk: float) -> str:
    if abs(float(cem_raw_lowk) - float(random_raw_lowk)) <= FLOAT_TIE_EPSILON:
        return "tie"
    if float(cem_raw_lowk) > float(random_raw_lowk):
        return "cem_best"
    if float(random_raw_lowk) > float(cem_raw_lowk):
        return "random_nz"
    return "tie"


def metric_win_counts(
    cem_metrics: Mapping[str, float | int | None],
    random_metrics: Mapping[str, float | int | None],
) -> tuple[int, int, str]:
    cem_wins = 0
    random_wins = 0
    for key in LOWK_KEYS:
        cem_value = float(cem_metrics[key])
        random_value = float(random_metrics[key])
        if cem_value > random_value:
            cem_wins += 1
        elif random_value > cem_value:
            random_wins += 1
    if cem_wins > random_wins:
        winner = "cem_best"
    elif random_wins > cem_wins:
        winner = "random_nz"
    else:
        winner = "tie"
    return cem_wins, random_wins, winner


def build_comparison_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_epoch: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        epoch = int(row["epoch"])
        candidate_type = str(row["candidate_type"])
        by_epoch.setdefault(epoch, {})[candidate_type] = row

    comparison_rows: list[dict[str, Any]] = []
    for epoch in sorted(by_epoch):
        epoch_rows = by_epoch[epoch]
        if "cem_best" not in epoch_rows or "random_nz" not in epoch_rows:
            raise ValueError(f"Missing candidate row for epoch {epoch}.")
        cem = epoch_rows["cem_best"]
        random_row = epoch_rows["random_nz"]
        cem_raw = float(cem["raw_lowk"])
        random_raw = float(random_row["raw_lowk"])
        cem_wins, random_wins, metric_winner = metric_win_counts(cem, random_row)
        comparison_rows.append(
            {
                "target_item": int(cem["target_item"]),
                "epoch": int(epoch),
                "cem_raw_lowk": cem_raw,
                "random_raw_lowk": random_raw,
                "delta_raw_lowk": cem_raw - random_raw,
                "cem_wins_count_4": int(cem_wins),
                "random_wins_count_4": int(random_wins),
                "winner_by_raw_lowk": winner_by_raw_lowk(cem_raw, random_raw),
                "winner_by_metric_count": metric_winner,
            }
        )
    return comparison_rows


def first_aligned_epoch(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    final_winner: str,
) -> int | None:
    for row in sorted(comparison_rows, key=lambda item: int(item["epoch"])):
        if row["winner_by_raw_lowk"] == final_winner:
            return int(row["epoch"])
    return None


def first_stably_aligned_epoch(
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    final_winner: str,
) -> int | None:
    sorted_rows = sorted(comparison_rows, key=lambda item: int(item["epoch"]))
    for index, row in enumerate(sorted_rows):
        if row["winner_by_raw_lowk"] != final_winner:
            continue
        if all(tail["winner_by_raw_lowk"] == final_winner for tail in sorted_rows[index:]):
            return int(row["epoch"])
    return None


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_clean_train_pairs(shared_attack_dir: Path) -> tuple[list[list[int]], list[int]]:
    sessions, labels = load_pickle(shared_attack_dir / "export" / "train.txt")
    return [list(session) for session in sessions], [int(label) for label in labels]


def load_validation_data(shared_attack_dir: Path) -> Data:
    Data = _load_training_deps()["Data"]
    valid_sessions, valid_labels = load_pickle(shared_attack_dir / "export" / "valid.txt")
    return Data((valid_sessions, valid_labels), shuffle=False)


def replay_random_nz_candidate(
    template_sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
    fake_session_seed: int,
    replacement_topk_ratio: float,
) -> tuple[list[list[int]], dict[str, int]]:
    rng = random.Random(int(fake_session_seed))
    policy = RandomNonzeroWhenPossiblePolicy(float(replacement_topk_ratio), rng=rng)
    fake_sessions: list[list[int]] = []
    counts: Counter[int] = Counter()
    for session in template_sessions:
        result = policy.apply_with_metadata(session, int(target_item))
        fake_sessions.append(result.session)
        counts[int(result.position)] += 1
    return fake_sessions, {str(position): int(count) for position, count in sorted(counts.items())}


def assert_random_nz_histogram_matches(
    actual_counts: Mapping[str, int],
    metadata_path: str | Path,
) -> None:
    payload = load_json(metadata_path)
    expected_counts = {str(key): int(value) for key, value in payload["counts"].items()}
    normalized_actual = {str(key): int(value) for key, value in actual_counts.items()}
    if normalized_actual != expected_counts:
        raise ValueError(
            "Random-NZ replay position histogram does not match existing metadata. "
            f"actual={normalized_actual} expected={expected_counts}"
        )


def validate_cem_metadata(run_metadata: Mapping[str, Any], *, target_item: int) -> dict[str, Any]:
    if int(run_metadata.get("target_item")) != int(target_item):
        raise ValueError(
            f"CEM run_metadata target_item mismatch: {run_metadata.get('target_item')} != {target_item}"
        )
    replay = run_metadata.get("replay_metadata")
    if not isinstance(replay, Mapping):
        raise ValueError("CEM run_metadata is missing replay_metadata.")
    if replay.get("final_selected_global_candidate_id") is None:
        raise ValueError("CEM replay_metadata is missing final_selected_global_candidate_id.")
    return {
        "final_selected_global_candidate_id": int(replay["final_selected_global_candidate_id"]),
        "final_selected_iteration": int(replay["final_selected_iteration"]),
        "final_selected_candidate_id": int(replay["final_selected_candidate_id"]),
        "final_selection_reward_name": str(replay["final_selection_reward_name"]),
        "final_selection_reward_value": float(replay["final_selection_reward_value"]),
    }


def final_lowk_context(cem_metrics_path: Path, random_metrics_path: Path) -> dict[str, Any]:
    cem_metrics = load_json(cem_metrics_path)["metrics"]
    random_metrics = load_json(random_metrics_path)["metrics"]
    cem_lowk = _lowk_metric_payload(cem_metrics)
    random_lowk = _lowk_metric_payload(random_metrics)
    cem_raw = raw_lowk(cem_lowk)
    random_raw = raw_lowk(random_lowk)
    cem_wins, random_wins, metric_winner = metric_win_counts(cem_lowk, random_lowk)
    return {
        "cem_best": {**cem_lowk, "raw_lowk": cem_raw},
        "random_nz": {**random_lowk, "raw_lowk": random_raw},
        "delta_raw_lowk": cem_raw - random_raw,
        "winner_by_raw_lowk": winner_by_raw_lowk(cem_raw, random_raw),
        "cem_wins_count_4": cem_wins,
        "random_wins_count_4": random_wins,
        "winner_by_metric_count": metric_winner,
    }


def _lowk_metric_payload(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in LOWK_KEYS}


def train_candidate_partial_retrain(
    *,
    config: Config,
    candidate_type: str,
    target_item: int,
    clean_sessions: Sequence[Sequence[int]],
    clean_labels: Sequence[int],
    fake_sessions: Sequence[Sequence[int]],
    validation_data: Data,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deps = _load_training_deps()
    Data = deps["Data"]
    SRGNNVictimRunner = deps["SRGNNVictimRunner"]
    print(
        f"[partial-retrain] candidate={candidate_type} "
        f"target={int(target_item)} preparing fresh SR-GNN seed={int(seed)}",
        flush=True,
    )
    _set_seed(int(seed))
    train_config = dict(config.victims.params["srgnn"]["train"])
    runner = SRGNNVictimRunner(config)
    runner.build_model(_build_srgnn_opt_from_train_config(train_config))
    poisoned = build_poisoned_dataset(clean_sessions, clean_labels, fake_sessions)
    train_data = Data((poisoned.sessions, poisoned.labels), shuffle=True)
    print(
        f"[partial-retrain] candidate={candidate_type} clean_prefixes={poisoned.clean_count} "
        f"fake_sessions={poisoned.fake_count} poisoned_prefixes={len(poisoned.sessions)}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    candidate_start = time.perf_counter()
    cumulative_train_seconds = 0.0
    for epoch in range(1, int(max_epochs) + 1):
        print(
            f"[partial-retrain] candidate={candidate_type} epoch={epoch}/{int(max_epochs)} "
            "training...",
            flush=True,
        )
        epoch_train_start = time.perf_counter()
        train_loss = _train_one_epoch(runner, train_data)
        cumulative_train_seconds += time.perf_counter() - epoch_train_start
        print(
            f"[partial-retrain] candidate={candidate_type} epoch={epoch}/{int(max_epochs)} "
            "evaluating full validation...",
            flush=True,
        )
        metrics = evaluate_candidate_epoch(
            runner,
            validation_data,
            target_item=target_item,
        )
        wall_time_seconds = time.perf_counter() - candidate_start
        rows.append(
            {
                "target_item": int(target_item),
                "candidate_type": str(candidate_type),
                "epoch": int(epoch),
                **metrics,
                "raw_lowk": raw_lowk(metrics),
                "train_time_seconds": float(cumulative_train_seconds),
                "wall_time_seconds": float(wall_time_seconds),
                "actual_train_epochs": int(epoch),
                "seed": int(seed),
                "notes": f"partial_from_scratch_full_validation train_loss={train_loss:.6g}",
            }
        )
        print(
            f"[partial-retrain] candidate={candidate_type} epoch={epoch}/{int(max_epochs)} "
            f"done raw_lowk={rows[-1]['raw_lowk']:.9f} "
            f"mrr@10={rows[-1]['targeted_mrr@10']:.9f} "
            f"mrr@20={rows[-1]['targeted_mrr@20']:.9f} "
            f"recall@10={rows[-1]['targeted_recall@10']:.9f} "
            f"recall@20={rows[-1]['targeted_recall@20']:.9f} "
            f"train_s={cumulative_train_seconds:.1f} wall_s={wall_time_seconds:.1f}",
            flush=True,
        )

    metadata = {
        "candidate_type": str(candidate_type),
        "clean_count": int(poisoned.clean_count),
        "fake_session_count": int(poisoned.fake_count),
        "poisoned_prefix_count": int(len(poisoned.sessions)),
        "seed": int(seed),
        "max_epochs": int(max_epochs),
    }
    return rows, metadata


def _train_one_epoch(runner: SRGNNVictimRunner, train_data: Data) -> float:
    deps = _load_training_deps()
    srg_forward = deps["srg_forward"]
    torch = deps["torch"]
    trans_to_cuda = deps["trans_to_cuda"]
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    model = runner.model
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch_indices in train_data.generate_batch(model.batch_size):
        model.optimizer.zero_grad()
        targets, scores = srg_forward(model, batch_indices, train_data)
        targets_tensor = trans_to_cuda(torch.as_tensor(targets, dtype=torch.long))
        loss = model.loss_function(scores, targets_tensor - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
    model.scheduler.step()
    avg_loss = total_loss / float(max(1, batch_count))
    runner.train_loss_history.append(float(avg_loss))
    return float(avg_loss)


def evaluate_candidate_epoch(
    runner: SRGNNVictimRunner,
    validation_data: Data,
    *,
    target_item: int,
) -> dict[str, float]:
    rankings = runner.predict_topk(validation_data, topk=20)
    metrics, available = evaluate_targeted_metrics(
        rankings,
        target_item=int(target_item),
        metrics=("mrr", "recall"),
        topk=(10, 20),
    )
    if not available:
        raise RuntimeError("Validation targeted metrics are unavailable.")
    return {key: float(metrics[key]) for key in LOWK_KEYS}


def write_outputs(
    output_dir: str | Path,
    *,
    target_item: int,
    rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    final_context: Mapping[str, Any],
    interpretation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "partial_retrain_alignment_14514.csv"
    json_path = output_path / "partial_retrain_alignment_14514.json"
    md_path = output_path / "partial_retrain_alignment_14514.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})

    payload = {
        "target_item": int(target_item),
        "rows": list(rows),
        "comparison_by_epoch": list(comparison_rows),
        "summary_checkpoints": [
            row for row in comparison_rows if int(row["epoch"]) in REPORT_CHECKPOINTS
        ],
        "final_victim_context": dict(final_context),
        "interpretation": dict(interpretation),
        "metadata": dict(metadata),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    md_path.write_text(
        render_markdown_report(
            target_item=target_item,
            comparison_rows=comparison_rows,
            final_context=final_context,
            interpretation=interpretation,
            metadata=metadata,
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def render_markdown_report(
    *,
    target_item: int,
    comparison_rows: Sequence[Mapping[str, Any]],
    final_context: Mapping[str, Any],
    interpretation: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    lines = [
        f"# Partial Retrain Alignment Diagnostic: target {int(target_item)}",
        "",
        "## Context",
        "- Warm-start ft6500 surrogate previously judged CEM_best > Random-NZ.",
        (
            f"- Final SR-GNN victim for target {int(target_item)} judged "
            f"`{final_context['winner_by_raw_lowk']}` by raw_lowk."
        ),
        "- This diagnostic tests whether partial from-scratch retrain closes that gap.",
        "",
        "## Final SR-GNN Victim Test-Set Low-K Context",
        _final_context_table(final_context),
        "",
        "## Partial Retrain Comparison By Epoch",
        _comparison_table(comparison_rows),
        "",
        "## Checkpoint Summary",
        _comparison_table(
            [row for row in comparison_rows if int(row["epoch"]) in REPORT_CHECKPOINTS]
        ),
        "",
        "## Interpretation",
        f"- final_winner_by_raw_lowk: `{interpretation['final_winner_by_raw_lowk']}`",
        f"- first_aligned_epoch: `{interpretation['first_aligned_epoch']}`",
        f"- first_stably_aligned_epoch: `{interpretation['first_stably_aligned_epoch']}`",
        f"- diagnostic_result: {interpretation['diagnostic_result']}",
        "",
        "## Sanity Checks",
        f"- clean_count: `{metadata['candidate_metadata']['cem_best']['clean_count']}`",
        f"- fake_session_count: `{metadata['candidate_metadata']['cem_best']['fake_session_count']}`",
        f"- random_nz_histogram_match: `{metadata['random_nz_histogram_match']}`",
        f"- seed: `{metadata['seed']}`",
    ]
    return "\n".join(lines) + "\n"


def _final_context_table(final_context: Mapping[str, Any]) -> str:
    rows = []
    for candidate in ("cem_best", "random_nz"):
        metrics = final_context[candidate]
        rows.append(
            "| {candidate} | {mrr10:.9f} | {mrr20:.9f} | {rec10:.9f} | {rec20:.9f} | {raw:.9f} |".format(
                candidate=candidate,
                mrr10=float(metrics["targeted_mrr@10"]),
                mrr20=float(metrics["targeted_mrr@20"]),
                rec10=float(metrics["targeted_recall@10"]),
                rec20=float(metrics["targeted_recall@20"]),
                raw=float(metrics["raw_lowk"]),
            )
        )
    return "\n".join(
        [
            "| candidate | MRR@10 | MRR@20 | Recall@10 | Recall@20 | raw_lowk |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
        ]
    )


def _comparison_table(comparison_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| epoch | cem_raw_lowk | random_raw_lowk | delta_raw_lowk | raw winner | metric winner |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison_rows:
        lines.append(
            "| {epoch} | {cem:.9f} | {random:.9f} | {delta:.9f} | {raw_winner} | {metric_winner} |".format(
                epoch=int(row["epoch"]),
                cem=float(row["cem_raw_lowk"]),
                random=float(row["random_raw_lowk"]),
                delta=float(row["delta_raw_lowk"]),
                raw_winner=row["winner_by_raw_lowk"],
                metric_winner=row["winner_by_metric_count"],
            )
        )
    return "\n".join(lines)


def build_interpretation(
    comparison_rows: Sequence[Mapping[str, Any]],
    final_context: Mapping[str, Any],
) -> dict[str, Any]:
    final_winner = str(final_context["winner_by_raw_lowk"])
    first_epoch = first_aligned_epoch(comparison_rows, final_winner=final_winner)
    stable_epoch = first_stably_aligned_epoch(comparison_rows, final_winner=final_winner)
    last_winner = str(sorted(comparison_rows, key=lambda row: int(row["epoch"]))[-1]["winner_by_raw_lowk"])
    if stable_epoch is not None and final_winner == "random_nz":
        result = (
            "Partial from-scratch retrain is stably aligned with final victim direction "
            "for the observed epoch range."
        )
    elif last_winner != final_winner:
        result = (
            "Short from-scratch retrain does not fix the observed surrogate-victim "
            "misalignment at epoch 8; check validation/test mismatch or longer training."
        )
    else:
        result = (
            "Partial retrain direction changes over epochs; use first_aligned_epoch and "
            "first_stably_aligned_epoch instead of a single transient epoch."
        )
    return {
        "final_winner_by_raw_lowk": final_winner,
        "first_aligned_epoch": first_epoch,
        "first_stably_aligned_epoch": stable_epoch,
        "diagnostic_result": result,
    }


def run_diagnostic(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    shared_attack_dir: str | Path = DEFAULT_SHARED_ATTACK_DIR,
    cem_target_dir: str | Path = DEFAULT_CEM_DIR,
    random_nz_target_dir: str | Path = DEFAULT_RANDOM_NZ_DIR,
    target_item: int = TARGET_ITEM,
    max_epochs: int = MAX_EPOCHS,
) -> dict[str, Any]:
    print(
        f"[partial-retrain] starting target={int(target_item)} max_epochs={int(max_epochs)}",
        flush=True,
    )
    config = load_config(config_path)
    shared_dir = Path(shared_attack_dir)
    cem_dir = Path(cem_target_dir)
    random_dir = Path(random_nz_target_dir)

    print("[partial-retrain] loading CEM_best artifacts...", flush=True)
    run_metadata = load_json(cem_dir / "position_opt" / "cem" / "run_metadata.json")
    cem_selection = validate_cem_metadata(run_metadata, target_item=target_item)
    cem_fake_sessions = load_pickle(
        cem_dir / "position_opt" / "cem" / "optimized_poisoned_sessions.pkl"
    )
    cem_fake_sessions = [list(session) for session in cem_fake_sessions]

    print("[partial-retrain] replaying Random-NZ and checking histogram...", flush=True)
    template_sessions = load_pickle(shared_dir / "fake_sessions.pkl")
    random_fake_sessions, random_counts = replay_random_nz_candidate(
        template_sessions,
        target_item=target_item,
        fake_session_seed=int(config.seeds.fake_session_seed),
        replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
    )
    assert_random_nz_histogram_matches(
        random_counts,
        random_dir / "random_nonzero_position_metadata.json",
    )
    print(
        "[partial-retrain] Random-NZ histogram matched existing metadata.",
        flush=True,
    )

    print("[partial-retrain] loading clean train pairs and validation set...", flush=True)
    clean_sessions, clean_labels = load_clean_train_pairs(shared_dir)
    validation_data = load_validation_data(shared_dir)
    seed = _derive_seed(
        int(config.seeds.surrogate_train_seed),
        "partial_retrain_diagnostic",
        int(target_item),
    )

    all_rows: list[dict[str, Any]] = []
    candidate_metadata: dict[str, Any] = {}
    for candidate_type, fake_sessions in (
        ("cem_best", cem_fake_sessions),
        ("random_nz", random_fake_sessions),
    ):
        rows, metadata = train_candidate_partial_retrain(
            config=config,
            candidate_type=candidate_type,
            target_item=target_item,
            clean_sessions=clean_sessions,
            clean_labels=clean_labels,
            fake_sessions=fake_sessions,
            validation_data=validation_data,
            seed=seed,
            max_epochs=max_epochs,
        )
        all_rows.extend(rows)
        candidate_metadata[candidate_type] = metadata

    print("[partial-retrain] building comparison tables and reports...", flush=True)
    comparison_rows = build_comparison_table(all_rows)
    final_context = final_lowk_context(
        cem_dir / "victims" / "srgnn" / "metrics.json",
        random_dir / "victims" / "srgnn" / "metrics.json",
    )
    interpretation = build_interpretation(comparison_rows, final_context)
    metadata = {
        "config_path": str(config_path),
        "shared_attack_dir": str(shared_dir),
        "cem_target_dir": str(cem_dir),
        "random_nz_target_dir": str(random_dir),
        "cem_selection": cem_selection,
        "seed": int(seed),
        "max_epochs": int(max_epochs),
        "random_nz_histogram_match": True,
        "candidate_metadata": candidate_metadata,
    }
    output_paths = write_outputs(
        output_dir,
        target_item=target_item,
        rows=all_rows,
        comparison_rows=comparison_rows,
        final_context=final_context,
        interpretation=interpretation,
        metadata=metadata,
    )
    print("[partial-retrain] completed. Outputs:", flush=True)
    for label, path in output_paths.items():
        print(f"[partial-retrain]   {label}: {path}", flush=True)
    return {
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "interpretation": interpretation,
        "comparison_by_epoch": comparison_rows,
        "final_victim_context": final_context,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc partial from-scratch retrain diagnostic for target 14514."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shared-attack-dir", default=str(DEFAULT_SHARED_ATTACK_DIR))
    parser.add_argument("--cem-target-dir", default=str(DEFAULT_CEM_DIR))
    parser.add_argument("--random-nz-target-dir", default=str(DEFAULT_RANDOM_NZ_DIR))
    parser.add_argument("--target-item", type=int, default=TARGET_ITEM)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    args = parser.parse_args()

    result = run_diagnostic(
        config_path=args.config,
        output_dir=args.output_dir,
        shared_attack_dir=args.shared_attack_dir,
        cem_target_dir=args.cem_target_dir,
        random_nz_target_dir=args.random_nz_target_dir,
        target_item=int(args.target_item),
        max_epochs=int(args.max_epochs),
    )
    print(json.dumps(result["output_paths"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
