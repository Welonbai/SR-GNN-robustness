"""Diagnose CEM-vs-Random alignment using stored CEM rewards and Random rescore.

This is intentionally a diagnostic utility, not part of the attack pipeline.
It reuses CEM-best rewards already stored in cem_trace.jsonl and only runs the
expensive surrogate fine-tune for Random-NZ under the same CEM evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import load_config
from attack.common.paths import POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.truncated_finetune import TruncatedFineTuneInnerTrainer
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.position_opt import (
    resolve_clean_surrogate_checkpoint_path,
    resolve_position_opt_config,
)
from attack.position_opt.cem import resolve_rank_bucket_cem_config
from attack.position_opt.cem.trainer import (
    _resolve_validation_pairs,
    _select_validation_subset,
)
from attack.position_opt.types import TruncatedFineTuneConfig
from attack.surrogate.srgnn_backend import SRGNNBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "attack/configs/diginetica_attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5.yaml"
)
DEFAULT_CEM_RUN_ROOT = (
    REPO_ROOT
    / "outputs/runs/diginetica/attack_rank_bucket_cem_mixed_ft2500_srgnn_sample5/run_group_e4d7034404"
)
DEFAULT_RANDOM_RUN_ROOT = (
    REPO_ROOT
    / "outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1_srgnn_sample5/run_group_720516397a"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/diagnosis_outputs/diginetica_cem_mixed_ft2500_sample5_vs_random_nz_surrogate_reward"
)
SELECTED_TARGET_METRICS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_mrr@30",
    "targeted_recall@10",
    "targeted_recall@20",
    "targeted_recall@30",
)
SELECTED_TARGET_METRICS_10_20 = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)
SURROGATE_METRICS = (
    "targeted_mrr@10",
    "targeted_mrr@30",
    "targeted_recall@10",
    "targeted_recall@20",
    "targeted_recall@30",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = _repo_path(args.config)
    cem_run_root = _repo_path(args.cem_run_root)
    random_run_root = _repo_path(args.random_nz_run_root)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    position_opt_config = resolve_position_opt_config(config.attack.position_opt)
    rank_bucket_cem_config = resolve_rank_bucket_cem_config(config.attack.rank_bucket_cem)
    if position_opt_config.enable_gt_penalty:
        raise ValueError("This diagnostic expects enable_gt_penalty=false.")

    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        override=position_opt_config.clean_surrogate_checkpoint,
    ).resolve()
    if not clean_checkpoint.is_file():
        raise FileNotFoundError(clean_checkpoint)

    random_summary = _load_json(random_run_root / "summary_current.json")
    cem_summary = _load_json(cem_run_root / "summary_current.json")
    target_items = _resolve_targets(random_summary, cem_summary, args.target_items)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        require_poison_runner=False,
        config_path=None,
    )
    validation_sessions, validation_labels = _resolve_validation_pairs(shared)
    fine_tune_config = TruncatedFineTuneConfig(
        steps=int(position_opt_config.fine_tune_steps),
        epochs=1,
    )

    reward_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    final_rows = _final_metric_rows(
        random_summary=random_summary,
        cem_summary=cem_summary,
        target_items=target_items,
    )

    started = time.perf_counter()
    for index, target_item in enumerate(target_items, start=1):
        target_started = time.perf_counter()
        print(f"[alignment] target {target_item} ({index}/{len(target_items)}) random rescore start", flush=True)

        validation_subset_seed = (
            None
            if position_opt_config.validation_subset_size is None
            else derive_seed(
                config.seeds.position_opt_seed,
                "rank_bucket_cem_validation_subset",
                int(target_item),
            )
        )
        selected_validation_sessions, _selected_validation_labels, subset_meta = _select_validation_subset(
            validation_sessions,
            validation_labels,
            subset_size=position_opt_config.validation_subset_size,
            subset_seed=validation_subset_seed,
        )

        clean_reward_value = _score_clean_reward(
            config=config,
            clean_checkpoint=clean_checkpoint,
            validation_sessions=selected_validation_sessions,
            target_item=int(target_item),
            reward_metric=rank_bucket_cem_config.reward_metric,
        )
        surrogate_train_seed = derive_seed(
            config.seeds.surrogate_train_seed,
            "rank_bucket_cem_surrogate_train",
            int(target_item),
        )

        cem_row = _stored_cem_best_row(
            cem_run_root=cem_run_root,
            target_item=int(target_item),
            clean_reward_value=clean_reward_value,
            reward_mode=position_opt_config.reward_mode,
            reward_metric=rank_bucket_cem_config.reward_metric,
            fine_tune_steps=int(position_opt_config.fine_tune_steps),
            surrogate_train_seed=surrogate_train_seed,
            validation_subset_metadata=subset_meta,
        )
        random_sessions = _build_random_nz_sessions(
            config,
            shared.template_sessions,
            target_item=int(target_item),
        )
        random_row = _score_random_row(
            config=config,
            clean_checkpoint=clean_checkpoint,
            clean_sessions=shared.clean_sessions,
            clean_labels=shared.clean_labels,
            poisoned_fake_sessions=random_sessions,
            validation_sessions=selected_validation_sessions,
            target_item=int(target_item),
            clean_reward_value=clean_reward_value,
            reward_mode=position_opt_config.reward_mode,
            reward_metric=rank_bucket_cem_config.reward_metric,
            fine_tune_config=fine_tune_config,
            surrogate_train_seed=surrogate_train_seed,
            validation_subset_metadata=subset_meta,
        )
        reward_rows.extend([cem_row, random_row])

        pairwise = _alignment_pairwise_row(
            target_item=int(target_item),
            cem_row=cem_row,
            random_row=random_row,
            final_rows=final_rows,
        )
        pairwise["random_rescore_seconds"] = round(time.perf_counter() - target_started, 3)
        pairwise_rows.append(pairwise)

        _write_outputs(
            output_dir=output_dir,
            reward_rows=reward_rows,
            pairwise_rows=pairwise_rows,
            final_rows=final_rows,
            config_path=config_path,
            cem_run_root=cem_run_root,
            random_run_root=random_run_root,
            clean_checkpoint=clean_checkpoint,
            target_items=target_items,
            elapsed_seconds=time.perf_counter() - started,
            complete=False,
        )
        print(
            "[alignment] target "
            f"{target_item} done surrogate_delta={pairwise['surrogate_delta']:.10f} "
            f"final_wins_6={pairwise['final_target_wins_6']}/6 "
            f"case={pairwise['alignment_case']}",
            flush=True,
        )

    _write_outputs(
        output_dir=output_dir,
        reward_rows=reward_rows,
        pairwise_rows=pairwise_rows,
        final_rows=final_rows,
        config_path=config_path,
        cem_run_root=cem_run_root,
        random_run_root=random_run_root,
        clean_checkpoint=clean_checkpoint,
        target_items=target_items,
        elapsed_seconds=time.perf_counter() - started,
        complete=True,
    )
    print(f"[alignment] wrote diagnosis to {output_dir}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cem-run-root", default=str(DEFAULT_CEM_RUN_ROOT))
    parser.add_argument("--random-nz-run-root", default=str(DEFAULT_RANDOM_RUN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-items", nargs="*", type=int)
    return parser


def _score_clean_reward(
    *,
    config,
    clean_checkpoint: Path,
    validation_sessions,
    target_item: int,
    reward_metric: str | None,
) -> float:
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    backend.load_clean_checkpoint(clean_checkpoint)
    result = backend.score_target(
        backend.clone_clean_model(),
        validation_sessions,
        int(target_item),
    )
    return _resolve_reward_value(result.mean, result.metrics, reward_metric=reward_metric)


def _score_random_row(
    *,
    config,
    clean_checkpoint: Path,
    clean_sessions,
    clean_labels,
    poisoned_fake_sessions,
    validation_sessions,
    target_item: int,
    clean_reward_value: float,
    reward_mode: str,
    reward_metric: str | None,
    fine_tune_config: TruncatedFineTuneConfig,
    surrogate_train_seed: int,
    validation_subset_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    poisoned_dataset = build_poisoned_dataset(
        [list(session) for session in clean_sessions],
        [int(label) for label in clean_labels],
        [list(session) for session in poisoned_fake_sessions],
    )
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    inner_result = TruncatedFineTuneInnerTrainer().run(
        backend,
        clean_checkpoint,
        poisoned_dataset,
        config=fine_tune_config,
        seed=int(surrogate_train_seed),
    )
    score_result = backend.score_target(
        inner_result.model,
        validation_sessions,
        int(target_item),
    )
    reward_value = _resolve_reward_value(
        score_result.mean,
        score_result.metrics,
        reward_metric=reward_metric,
    )
    objective_reward = _objective_reward(
        reward_mode=reward_mode,
        reward_value=reward_value,
        clean_reward_value=clean_reward_value,
    )
    return _reward_row(
        target_item=target_item,
        method="random_nz_ratio1",
        source="rescored",
        clean_reward_value=clean_reward_value,
        reward_value=reward_value,
        objective_reward=objective_reward,
        metrics=score_result.metrics,
        reward_mode=reward_mode,
        reward_metric=reward_metric,
        fine_tune_steps=int(fine_tune_config.steps),
        surrogate_train_seed=surrogate_train_seed,
        validation_subset_metadata=validation_subset_metadata,
        cem_iteration=None,
        cem_candidate_id=None,
    )


def _stored_cem_best_row(
    *,
    cem_run_root: Path,
    target_item: int,
    clean_reward_value: float,
    reward_mode: str,
    reward_metric: str | None,
    fine_tune_steps: int,
    surrogate_train_seed: int,
    validation_subset_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    best = _best_cem_trace_row(cem_run_root=cem_run_root, target_item=target_item)
    target_metrics = dict(best.get("target_metrics") or {})
    reward_value = (
        float(best["target_result_mean"])
        if reward_metric is None
        else float(target_metrics[reward_metric])
    )
    return _reward_row(
        target_item=target_item,
        method="cem_best_stored",
        source="cem_trace",
        clean_reward_value=clean_reward_value,
        reward_value=reward_value,
        objective_reward=float(best["reward"]),
        metrics=target_metrics,
        reward_mode=reward_mode,
        reward_metric=reward_metric,
        fine_tune_steps=fine_tune_steps,
        surrogate_train_seed=surrogate_train_seed,
        validation_subset_metadata=validation_subset_metadata,
        cem_iteration=int(best["iteration"]),
        cem_candidate_id=int(best["candidate_id"]),
    )


def _reward_row(
    *,
    target_item: int,
    method: str,
    source: str,
    clean_reward_value: float,
    reward_value: float,
    objective_reward: float,
    metrics: Mapping[str, Any],
    reward_mode: str,
    reward_metric: str | None,
    fine_tune_steps: int,
    surrogate_train_seed: int,
    validation_subset_metadata: Mapping[str, Any],
    cem_iteration: int | None,
    cem_candidate_id: int | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "target_item": int(target_item),
        "method": method,
        "source": source,
        "reward_mode": str(reward_mode),
        "reward_metric": "target_result.mean" if reward_metric is None else str(reward_metric),
        "clean_reward_value": float(clean_reward_value),
        "surrogate_reward_value": float(reward_value),
        "objective_reward": float(objective_reward),
        "delta_reward_value_vs_clean": float(reward_value) - float(clean_reward_value),
        "fine_tune_steps": int(fine_tune_steps),
        "surrogate_train_seed": int(surrogate_train_seed),
        "validation_subset_strategy": validation_subset_metadata.get("strategy"),
        "validation_subset_seed": validation_subset_metadata.get("seed"),
        "validation_subset_count": validation_subset_metadata.get("selected_count"),
        "cem_iteration": cem_iteration,
        "cem_candidate_id": cem_candidate_id,
    }
    for key in SURROGATE_METRICS:
        row[key] = metrics.get(key)
    return row


def _alignment_pairwise_row(
    *,
    target_item: int,
    cem_row: Mapping[str, Any],
    random_row: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    final_for_target = [
        row for row in final_rows if int(row["target_item"]) == int(target_item)
    ]
    final_metrics = {
        str(row["metric_key"]): row
        for row in final_for_target
        if row["metric_key"] in SELECTED_TARGET_METRICS
    }
    wins_6 = sum(1 for row in final_metrics.values() if float(row["cem_minus_random_nz"]) > 0.0)
    wins_4 = sum(
        1
        for key, row in final_metrics.items()
        if key in SELECTED_TARGET_METRICS_10_20
        and float(row["cem_minus_random_nz"]) > 0.0
    )
    surrogate_delta = float(cem_row["objective_reward"]) - float(random_row["objective_reward"])
    surrogate_judgement = "CEM > Random" if surrogate_delta > 0.0 else "CEM < Random"
    if wins_6 >= 4:
        final_judgement = "CEM > Random"
    elif wins_6 <= 2:
        final_judgement = "CEM < Random"
    else:
        final_judgement = "mixed"
    if surrogate_judgement == "CEM > Random" and final_judgement == "CEM > Random":
        case = "reward_align"
        interpretation = "reward align, CEM also wins final target metrics"
    elif surrogate_judgement == "CEM > Random" and final_judgement == "CEM < Random":
        case = "reward_final_misalignment"
        interpretation = "surrogate prefers CEM but final victim metrics prefer Random"
    elif surrogate_judgement == "CEM < Random" and final_judgement == "CEM < Random":
        case = "search_or_reference_gap"
        interpretation = "surrogate also prefers Random; CEM selection did not compare against Random"
    elif surrogate_judgement == "CEM < Random" and final_judgement == "CEM > Random":
        case = "surrogate_too_conservative"
        interpretation = "surrogate prefers Random but final victim metrics prefer CEM"
    else:
        case = "mixed_final"
        interpretation = "final selected metrics are split"
    row = {
        "target_item": int(target_item),
        "cem_surrogate_reward": float(cem_row["objective_reward"]),
        "random_surrogate_reward": float(random_row["objective_reward"]),
        "surrogate_delta": surrogate_delta,
        "surrogate_judgement": surrogate_judgement,
        "final_target_wins_6": wins_6,
        "final_target_wins_4_at10_20": wins_4,
        "final_judgement": final_judgement,
        "alignment_case": case,
        "interpretation": interpretation,
    }
    for key in SELECTED_TARGET_METRICS:
        metric_row = final_metrics.get(key)
        if metric_row is not None:
            row[f"final_delta_{key}"] = float(metric_row["cem_minus_random_nz"])
    for key in SURROGATE_METRICS:
        row[f"surrogate_delta_{key}"] = _optional_delta(cem_row.get(key), random_row.get(key))
    return row


def _final_metric_rows(
    *,
    random_summary: Mapping[str, Any],
    cem_summary: Mapping[str, Any],
    target_items: Sequence[int],
) -> list[dict[str, Any]]:
    random_targets = _summary_targets(random_summary)
    cem_targets = _summary_targets(cem_summary)
    rows: list[dict[str, Any]] = []
    for target_item in target_items:
        target_key = str(int(target_item))
        random_victims = random_targets[target_key]["victims"]
        cem_victims = cem_targets[target_key]["victims"]
        for victim in sorted(set(random_victims) & set(cem_victims)):
            random_metrics = random_victims[victim]["metrics"]
            cem_metrics = cem_victims[victim]["metrics"]
            for metric_key in sorted(set(random_metrics) & set(cem_metrics)):
                if "@" not in metric_key:
                    continue
                random_value = float(random_metrics[metric_key])
                cem_value = float(cem_metrics[metric_key])
                rows.append(
                    {
                        "target_item": int(target_item),
                        "victim_model": victim,
                        "metric_key": metric_key,
                        "random_nz_ratio1": random_value,
                        "cem": cem_value,
                        "cem_minus_random_nz": cem_value - random_value,
                    }
                )
    return rows


def _write_outputs(
    *,
    output_dir: Path,
    reward_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    config_path: Path,
    cem_run_root: Path,
    random_run_root: Path,
    clean_checkpoint: Path,
    target_items: Sequence[int],
    elapsed_seconds: float,
    complete: bool,
) -> None:
    _write_csv(output_dir / "surrogate_reward_comparison.csv", reward_rows)
    _write_csv(output_dir / "surrogate_final_alignment.csv", pairwise_rows)
    _write_csv(output_dir / "final_metric_overlap.csv", final_rows)
    (output_dir / "alignment_report.md").write_text(
        _render_report(
            pairwise_rows=pairwise_rows,
            complete=complete,
            elapsed_seconds=elapsed_seconds,
            config_path=config_path,
            cem_run_root=cem_run_root,
            random_run_root=random_run_root,
            clean_checkpoint=clean_checkpoint,
        ),
        encoding="utf-8",
    )
    manifest = {
        "complete": bool(complete),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "config": _repo_relative(config_path),
        "cem_run_root": _repo_relative(cem_run_root),
        "random_nz_run_root": _repo_relative(random_run_root),
        "clean_surrogate_checkpoint": _repo_relative(clean_checkpoint),
        "target_items": [int(item) for item in target_items],
        "completed_target_items": [int(row["target_item"]) for row in pairwise_rows],
        "surrogate_reward_comparison": _repo_relative(output_dir / "surrogate_reward_comparison.csv"),
        "surrogate_final_alignment": _repo_relative(output_dir / "surrogate_final_alignment.csv"),
        "final_metric_overlap": _repo_relative(output_dir / "final_metric_overlap.csv"),
        "report": _repo_relative(output_dir / "alignment_report.md"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _render_report(
    *,
    pairwise_rows: Sequence[Mapping[str, Any]],
    complete: bool,
    elapsed_seconds: float,
    config_path: Path,
    cem_run_root: Path,
    random_run_root: Path,
    clean_checkpoint: Path,
) -> str:
    lines = [
        "# CEM ft2500 Sample5 Surrogate-Final Alignment",
        "",
        f"- status: {'complete' if complete else 'partial'}",
        f"- elapsed_seconds: {elapsed_seconds:.1f}",
        f"- config: `{_repo_relative(config_path)}`",
        f"- CEM run: `{_repo_relative(cem_run_root)}`",
        f"- Random-NZ run: `{_repo_relative(random_run_root)}`",
        f"- clean surrogate: `{_repo_relative(clean_checkpoint)}`",
        "- CEM reward source: stored best row in `cem_trace.jsonl`",
        "- Random reward source: rescored with the same normal mixed ft2500 surrogate evaluator",
        "",
        "## Four-Case Table",
        "",
        "| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |",
        "|---:|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in pairwise_rows:
        lines.append(
            f"| {row['target_item']} | "
            f"{float(row['cem_surrogate_reward']):.10f} | "
            f"{float(row['random_surrogate_reward']):.10f} | "
            f"{float(row['surrogate_delta']):.10f} | "
            f"{row['surrogate_judgement']} | "
            f"{row['final_target_wins_6']} | "
            f"{row['final_target_wins_4_at10_20']} | "
            f"{row['final_judgement']} | "
            f"{row['alignment_case']} |"
        )
    if pairwise_rows:
        counts: dict[str, int] = {}
        for row in pairwise_rows:
            counts[str(row["alignment_case"])] = counts.get(str(row["alignment_case"]), 0) + 1
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Case counts: `{json.dumps(counts, sort_keys=True)}`",
                (
                    "- `reward_final_misalignment` means CEM surrogate reward is higher "
                    "than Random-NZ, but final SRGNN target metrics are worse."
                ),
                (
                    "- `search_or_reference_gap` means the surrogate evaluator itself "
                    "prefers Random-NZ; CEM did not include Random-NZ as a candidate/reference."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `surrogate_reward_comparison.csv`",
            "- `surrogate_final_alignment.csv`",
            "- `final_metric_overlap.csv`",
            "- `manifest.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_random_nz_sessions(config, template_sessions, *, target_item: int) -> list[list[int]]:
    rng = random.Random(int(config.seeds.fake_session_seed))
    policy = RandomNonzeroWhenPossiblePolicy(
        float(config.attack.replacement_topk_ratio),
        rng=rng,
    )
    return [
        policy.apply_with_metadata(session, int(target_item)).session
        for session in template_sessions
    ]


def _best_cem_trace_row(*, cem_run_root: Path, target_item: int) -> dict[str, Any]:
    path = cem_run_root / "targets" / str(int(target_item)) / "position_opt" / "cem" / "cem_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty CEM trace: {path}")
    return dict(max(rows, key=lambda row: float(row["reward"])))


def _resolve_targets(
    random_summary: Mapping[str, Any],
    cem_summary: Mapping[str, Any],
    requested_targets: Sequence[int] | None,
) -> list[int]:
    overlap = sorted(
        set(_summary_targets(random_summary)) & set(_summary_targets(cem_summary)),
        key=lambda item: int(item),
    )
    if requested_targets:
        requested = [str(int(item)) for item in requested_targets]
        missing = sorted(set(requested) - set(overlap), key=lambda item: int(item))
        if missing:
            raise ValueError(f"requested targets not found in overlap: {missing}")
        overlap = requested
    return [int(item) for item in overlap]


def _summary_targets(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw = summary.get("targets")
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(int(v["target_item"])): v for v in raw}
    raise ValueError("summary_current.json missing targets")


def _resolve_reward_value(mean: float, metrics: Mapping[str, Any], *, reward_metric: str | None) -> float:
    if reward_metric is None:
        return float(mean)
    return float(metrics[reward_metric])


def _objective_reward(*, reward_mode: str, reward_value: float, clean_reward_value: float) -> float:
    if reward_mode == "poisoned_target_utility":
        return float(reward_value)
    if reward_mode == "delta_target_utility":
        return float(reward_value) - float(clean_reward_value)
    raise ValueError(f"unsupported reward_mode: {reward_mode}")


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
