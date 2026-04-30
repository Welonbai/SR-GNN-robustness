"""Rescore target 5418 CEM-best and Random-NZ with ft10000 surrogate eval."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import load_config
from attack.common.paths import POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.truncated_finetune import TruncatedFineTuneInnerTrainer
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.evaluator import (
    evaluate_ground_truth_metrics,
    evaluate_targeted_metrics,
)
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
from attack.surrogate.srgnn_backend import SRGNNBackend, SRGNNModelHandle
from pytorch_code.model import forward as srg_forward
from pytorch_code.model import trans_to_cpu
from pytorch_code.utils import Data


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
    / "analysis/diagnosis_outputs/diginetica_cem_mixed_ft10000_target5418_vs_random_nz_surrogate_reward"
)
DEFAULT_TARGET_ITEM = 5418
DEFAULT_FINE_TUNE_STEPS = 10000
REWARD_MODE_TARGET_MEAN = "target_mean"
REWARD_MODE_BALANCED_MRR_RECALL_10_20 = "balanced_mrr_recall_10_20"
REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20 = "paired_relative_mrr_recall_10_20"
TOPK = (10, 20, 30)
METRICS = ("mrr", "recall")
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
PAIRWISE_RELATIVE_REWARD_METRICS = SELECTED_TARGET_METRICS_10_20


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config_path = _repo_path(args.config)
    cem_run_root = _repo_path(args.cem_run_root)
    random_run_root = _repo_path(args.random_nz_run_root)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_item = int(args.target_item)
    fine_tune_steps = int(args.fine_tune_steps)
    reward_mode = str(args.reward_mode)

    config = load_config(config_path)
    position_opt_config = resolve_position_opt_config(config.attack.position_opt)
    rank_bucket_cem_config = resolve_rank_bucket_cem_config(config.attack.rank_bucket_cem)
    if rank_bucket_cem_config.reward_metric is not None:
        raise ValueError("This diagnostic expects reward_metric=null / target_result.mean.")

    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        override=position_opt_config.clean_surrogate_checkpoint,
    ).resolve()
    if not clean_checkpoint.is_file():
        raise FileNotFoundError(clean_checkpoint)

    random_summary = _load_json(random_run_root / "summary_current.json")
    cem_summary = _load_json(cem_run_root / "summary_current.json")
    _assert_target_available(random_summary, cem_summary, target_item=target_item)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        require_poison_runner=False,
        config_path=None,
    )
    validation_sessions, validation_labels = _resolve_validation_pairs(shared)
    validation_subset_seed = (
        None
        if position_opt_config.validation_subset_size is None
        else derive_seed(
            config.seeds.position_opt_seed,
            "rank_bucket_cem_validation_subset",
            target_item,
        )
    )
    selected_validation_sessions, selected_validation_labels, subset_meta = _select_validation_subset(
        validation_sessions,
        validation_labels,
        subset_size=position_opt_config.validation_subset_size,
        subset_seed=validation_subset_seed,
    )
    surrogate_train_seed = derive_seed(
        config.seeds.surrogate_train_seed,
        "rank_bucket_cem_surrogate_train",
        target_item,
    )
    fine_tune_config = TruncatedFineTuneConfig(steps=fine_tune_steps, epochs=1)

    final_rows = _final_metric_rows(
        random_summary=random_summary,
        cem_summary=cem_summary,
        target_item=target_item,
    )
    best_trace = _best_cem_trace_row(cem_run_root=cem_run_root, target_item=target_item)
    clean_score = _score_clean(
        config=config,
        clean_checkpoint=clean_checkpoint,
        validation_sessions=selected_validation_sessions,
        validation_labels=selected_validation_labels,
        target_item=target_item,
    )

    candidate_specs = [
        {
            "method": f"cem_best_rescored_ft{fine_tune_steps}",
            "source": "optimized_poisoned_sessions.pkl",
            "poisoned_fake_sessions": _load_cem_optimized_sessions(
                cem_run_root=cem_run_root,
                target_item=target_item,
            ),
            "cem_iteration": int(best_trace["iteration"]),
            "cem_candidate_id": int(best_trace["candidate_id"]),
            "stored_ft2500_reward": float(best_trace["reward"]),
        },
        {
            "method": f"random_nz_ratio1_rescored_ft{fine_tune_steps}",
            "source": "reconstructed_random_nz",
            "poisoned_fake_sessions": _build_random_nz_sessions(
                config,
                shared.template_sessions,
                target_item=target_item,
            ),
            "cem_iteration": None,
            "cem_candidate_id": None,
            "stored_ft2500_reward": None,
        },
    ]

    reward_rows: list[dict[str, Any]] = _load_existing_reward_rows(
        output_dir / "surrogate_reward_comparison.csv",
        reward_mode=reward_mode,
    )
    completed_methods = {str(row.get("method")) for row in reward_rows}
    started = time.perf_counter()
    for index, spec in enumerate(candidate_specs, start=1):
        if str(spec["method"]) in completed_methods:
            print(
                f"[ft{fine_tune_steps}-alignment] target={target_item} candidate="
                f"{spec['method']} ({index}/{len(candidate_specs)}) skip existing",
                flush=True,
            )
            continue
        print(
            f"[ft{fine_tune_steps}-alignment] target={target_item} candidate="
            f"{spec['method']} ({index}/{len(candidate_specs)}) start",
            flush=True,
        )
        row = _score_candidate(
            config=config,
            clean_checkpoint=clean_checkpoint,
            clean_sessions=shared.clean_sessions,
            clean_labels=shared.clean_labels,
            validation_sessions=selected_validation_sessions,
            validation_labels=selected_validation_labels,
            target_item=target_item,
            fine_tune_config=fine_tune_config,
            surrogate_train_seed=surrogate_train_seed,
            clean_score=clean_score,
            validation_subset_metadata=subset_meta,
            reward_mode=reward_mode,
            **spec,
        )
        reward_rows.append(row)
        pairwise_rows = _pairwise_rows_if_ready(
            reward_rows=reward_rows,
            final_rows=final_rows,
            target_item=target_item,
            reward_mode=reward_mode,
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
            target_item=target_item,
            fine_tune_steps=fine_tune_steps,
            reward_mode=reward_mode,
            elapsed_seconds=time.perf_counter() - started,
            complete=len(reward_rows) == len(candidate_specs),
        )
        print(
            f"[ft{fine_tune_steps}-alignment] target={target_item} candidate={spec['method']} "
            f"done objective_reward={row['objective_reward']:.10f} "
            f"target_mean={row['target_result_mean']:.10f} "
            f"gt_mean={row['gt_result_mean']:.10f} "
            f"total_seconds={row['candidate_total_seconds']:.1f}",
            flush=True,
        )

    pairwise_rows = _pairwise_rows_if_ready(
        reward_rows=reward_rows,
        final_rows=final_rows,
        target_item=target_item,
        reward_mode=reward_mode,
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
        target_item=target_item,
        fine_tune_steps=fine_tune_steps,
        reward_mode=reward_mode,
        elapsed_seconds=time.perf_counter() - started,
        complete=len(reward_rows) == len(candidate_specs),
    )
    print(f"[ft{fine_tune_steps}-alignment] wrote diagnosis to {output_dir}", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cem-run-root", default=str(DEFAULT_CEM_RUN_ROOT))
    parser.add_argument("--random-nz-run-root", default=str(DEFAULT_RANDOM_RUN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-item", type=int, default=DEFAULT_TARGET_ITEM)
    parser.add_argument("--fine-tune-steps", type=int, default=DEFAULT_FINE_TUNE_STEPS)
    parser.add_argument(
        "--reward-mode",
        choices=(
            REWARD_MODE_TARGET_MEAN,
            REWARD_MODE_BALANCED_MRR_RECALL_10_20,
            REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20,
        ),
        default=REWARD_MODE_TARGET_MEAN,
        help=(
            "Surrogate reward used for CEM-vs-Random comparison. "
            "balanced_mrr_recall_10_20 gives MRR@10/20 and Recall@10/20 "
            "equal family weight. paired_relative_mrr_recall_10_20 normalizes "
            "each selected metric by max(CEM, Random) and averages the "
            "normalized metrics."
        ),
    )
    return parser


def _score_clean(
    *,
    config,
    clean_checkpoint: Path,
    validation_sessions,
    validation_labels,
    target_item: int,
) -> dict[str, Any]:
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    backend.load_clean_checkpoint(clean_checkpoint)
    model = backend.clone_clean_model()
    return _score_model(
        model,
        validation_sessions=validation_sessions,
        validation_labels=validation_labels,
        target_item=target_item,
    )


def _score_candidate(
    *,
    config,
    clean_checkpoint: Path,
    clean_sessions,
    clean_labels,
    poisoned_fake_sessions,
    validation_sessions,
    validation_labels,
    target_item: int,
    fine_tune_config: TruncatedFineTuneConfig,
    surrogate_train_seed: int,
    clean_score: Mapping[str, Any],
    validation_subset_metadata: Mapping[str, Any],
    reward_mode: str,
    method: str,
    source: str,
    cem_iteration: int | None,
    cem_candidate_id: int | None,
    stored_ft2500_reward: float | None,
) -> dict[str, Any]:
    candidate_started = time.perf_counter()
    poisoned_dataset = build_poisoned_dataset(
        [list(session) for session in clean_sessions],
        [int(label) for label in clean_labels],
        [list(session) for session in poisoned_fake_sessions],
    )
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    fine_tune_started = time.perf_counter()
    inner_result = TruncatedFineTuneInnerTrainer().run(
        backend,
        clean_checkpoint,
        poisoned_dataset,
        config=fine_tune_config,
        seed=int(surrogate_train_seed),
    )
    inner_history = dict(inner_result.history or {})
    fine_tune_seconds = time.perf_counter() - fine_tune_started
    score_started = time.perf_counter()
    score = _score_model(
        inner_result.model,
        validation_sessions=validation_sessions,
        validation_labels=validation_labels,
        target_item=target_item,
    )
    score_seconds = time.perf_counter() - score_started
    target_mean = float(score["target_result_mean"])
    reward_parts = _compute_reward_parts(
        score["metrics"],
        reward_mode=reward_mode,
        target_mean=target_mean,
    )
    clean_reward_parts = _compute_reward_parts(
        clean_score["metrics"],
        reward_mode=reward_mode,
        target_mean=float(clean_score["target_result_mean"]),
    )
    row: dict[str, Any] = {
        "target_item": int(target_item),
        "method": method,
        "source": source,
        "reward_mode": reward_mode,
        "fine_tune_steps": int(fine_tune_config.steps),
        "actual_optimizer_steps": inner_history.get("steps"),
        "actual_epochs": inner_history.get("epochs"),
        "fine_tune_avg_loss": inner_history.get("avg_loss"),
        "surrogate_train_seed": int(surrogate_train_seed),
        "validation_subset_strategy": validation_subset_metadata.get("strategy"),
        "validation_subset_seed": validation_subset_metadata.get("seed"),
        "validation_subset_count": validation_subset_metadata.get("selected_count"),
        "target_result_mean": target_mean,
        "objective_reward": reward_parts["objective_reward"],
        "reward_mrr_part_10_20": reward_parts["mrr_part_10_20"],
        "reward_recall_part_10_20": reward_parts["recall_part_10_20"],
        "reward_raw_metric_avg_10_20": reward_parts["raw_metric_avg_10_20"],
        "clean_objective_reward": clean_reward_parts["objective_reward"],
        "clean_target_result_mean": float(clean_score["target_result_mean"]),
        "delta_target_result_mean_vs_clean": target_mean
        - float(clean_score["target_result_mean"]),
        "gt_result_mean": float(score["gt_result_mean"]),
        "clean_gt_result_mean": float(clean_score["gt_result_mean"]),
        "delta_gt_result_mean_vs_clean": float(score["gt_result_mean"])
        - float(clean_score["gt_result_mean"]),
        "cem_iteration": cem_iteration,
        "cem_candidate_id": cem_candidate_id,
        "stored_ft2500_reward": stored_ft2500_reward,
        "fine_tune_seconds": round(fine_tune_seconds, 3),
        "score_seconds": round(score_seconds, 3),
        "candidate_total_seconds": round(time.perf_counter() - candidate_started, 3),
    }
    row.update(score["metrics"])
    return row


def _score_model(
    model: SRGNNModelHandle,
    *,
    validation_sessions,
    validation_labels,
    target_item: int,
) -> dict[str, Any]:
    if not isinstance(model, SRGNNModelHandle):
        raise TypeError("Expected SRGNNModelHandle.")
    sessions = [list(session) for session in validation_sessions]
    labels = [int(label) for label in validation_labels]
    if len(sessions) != len(labels):
        raise ValueError("validation sessions and labels must align.")
    data = Data((sessions, [1] * len(sessions)), shuffle=False)
    torch_model = model.model
    rankings: list[list[int]] = []
    target_values: list[float] = []
    gt_values: list[float] = []
    cursor = 0

    torch_model.eval()
    with torch.no_grad():
        for batch_indices in data.generate_batch(torch_model.batch_size):
            _, scores = srg_forward(torch_model, batch_indices, data)
            probabilities = torch.softmax(scores, dim=1)
            batch_size = probabilities.shape[0]
            batch_labels = labels[cursor : cursor + batch_size]
            target_tensor = torch.as_tensor(
                [int(target_item) - 1] * batch_size,
                dtype=torch.long,
                device=probabilities.device,
            )
            gt_tensor = torch.as_tensor(
                [int(label) - 1 for label in batch_labels],
                dtype=torch.long,
                device=probabilities.device,
            )
            target_scores = probabilities.gather(1, target_tensor.unsqueeze(1)).squeeze(1)
            gt_scores = probabilities.gather(1, gt_tensor.unsqueeze(1)).squeeze(1)
            target_values.extend(float(value) for value in trans_to_cpu(target_scores).tolist())
            gt_values.extend(float(value) for value in trans_to_cpu(gt_scores).tolist())
            topk = min(max(TOPK), scores.shape[1])
            topk_indices = trans_to_cpu(scores.topk(topk)[1]).detach().numpy()
            rankings.extend([int(item) + 1 for item in row.tolist()] for row in topk_indices)
            cursor += batch_size

    targeted, _ = evaluate_targeted_metrics(
        rankings,
        target_item=int(target_item),
        metrics=METRICS,
        topk=TOPK,
    )
    gt, _ = evaluate_ground_truth_metrics(
        rankings,
        labels=labels,
        metrics=METRICS,
        topk=TOPK,
    )
    metrics = {**targeted, **gt}
    return {
        "target_result_mean": _mean(target_values),
        "gt_result_mean": _mean(gt_values),
        "metrics": metrics,
    }


def _compute_reward_parts(
    metrics: Mapping[str, Any],
    *,
    reward_mode: str,
    target_mean: float,
) -> dict[str, float]:
    mrr_part = _mean(
        [
            float(metrics["targeted_mrr@10"]),
            float(metrics["targeted_mrr@20"]),
        ]
    )
    recall_part = _mean(
        [
            float(metrics["targeted_recall@10"]),
            float(metrics["targeted_recall@20"]),
        ]
    )
    raw_metric_avg = _mean(
        [
            float(metrics["targeted_mrr@10"]),
            float(metrics["targeted_mrr@20"]),
            float(metrics["targeted_recall@10"]),
            float(metrics["targeted_recall@20"]),
        ]
    )
    if reward_mode == REWARD_MODE_TARGET_MEAN:
        objective_reward = float(target_mean)
    elif reward_mode == REWARD_MODE_BALANCED_MRR_RECALL_10_20:
        # Give the MRR family and Recall family equal weight so recall's larger
        # numerical scale cannot dominate the composite by contributing two raw
        # large-valued terms.
        objective_reward = 0.5 * mrr_part + 0.5 * recall_part
    elif reward_mode == REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20:
        # The final paired-relative objective is computed only after both CEM
        # and Random rows are available. Keep this as an unnormalized companion
        # score for the per-candidate CSV.
        objective_reward = 0.5 * mrr_part + 0.5 * recall_part
    else:
        raise ValueError(f"Unsupported reward mode: {reward_mode!r}")
    return {
        "objective_reward": float(objective_reward),
        "mrr_part_10_20": float(mrr_part),
        "recall_part_10_20": float(recall_part),
        "raw_metric_avg_10_20": float(raw_metric_avg),
    }


def _pairwise_rows_if_ready(
    *,
    reward_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    target_item: int,
    reward_mode: str,
) -> list[dict[str, Any]]:
    if len(reward_rows) < 2:
        return []
    by_method = {str(row["method"]): row for row in reward_rows}
    cem = _find_single_method(by_method, "cem_best_rescored_ft")
    random_row = _find_single_method(by_method, "random_nz_ratio1_rescored_ft")
    if cem is None or random_row is None:
        return []
    final_metrics = {
        str(row["metric_key"]): row
        for row in final_rows
        if row["metric_key"] in SELECTED_TARGET_METRICS
    }
    final_wins_6 = sum(
        1 for row in final_metrics.values() if float(row["cem_minus_random_nz"]) > 0.0
    )
    final_wins_4 = sum(
        1
        for key, row in final_metrics.items()
        if key in SELECTED_TARGET_METRICS_10_20
        and float(row["cem_minus_random_nz"]) > 0.0
    )
    if reward_mode == REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20:
        cem_reward, random_reward, paired_norm_fields = _paired_relative_rewards(
            cem,
            random_row,
        )
    else:
        cem_reward = float(cem["objective_reward"])
        random_reward = float(random_row["objective_reward"])
        paired_norm_fields = {}
    surrogate_delta = cem_reward - random_reward
    surrogate_judgement = "CEM > Random" if surrogate_delta > 0 else "CEM < Random"
    if final_wins_6 >= 4:
        final_judgement = "CEM > Random"
    elif final_wins_6 <= 2:
        final_judgement = "CEM < Random"
    else:
        final_judgement = "mixed"
    if surrogate_judgement == "CEM > Random" and final_judgement == "CEM > Random":
        case = "reward_align"
    elif surrogate_judgement == "CEM > Random" and final_judgement == "CEM < Random":
        case = "reward_final_misalignment"
    elif surrogate_judgement == "CEM < Random" and final_judgement == "CEM < Random":
        case = "search_or_reference_gap"
    elif surrogate_judgement == "CEM < Random" and final_judgement == "CEM > Random":
        case = "surrogate_too_conservative"
    else:
        case = "mixed_final"

    row: dict[str, Any] = {
        "target_item": int(target_item),
        "fine_tune_steps": int(cem["fine_tune_steps"]),
        "reward_mode": reward_mode,
        "cem_surrogate_reward": float(cem_reward),
        "random_surrogate_reward": float(random_reward),
        "surrogate_delta": surrogate_delta,
        "surrogate_judgement": surrogate_judgement,
        "final_target_wins_6": final_wins_6,
        "final_target_wins_4_at10_20": final_wins_4,
        "final_judgement": final_judgement,
        "alignment_case": case,
    }
    row.update(paired_norm_fields)
    for key in SELECTED_TARGET_METRICS:
        metric_row = final_metrics.get(key)
        if metric_row is not None:
            row[f"final_delta_{key}"] = float(metric_row["cem_minus_random_nz"])
        row[f"surrogate_delta_{key}"] = float(cem[key]) - float(random_row[key])
    for key in (
        "ground_truth_mrr@10",
        "ground_truth_mrr@20",
        "ground_truth_mrr@30",
        "ground_truth_recall@10",
        "ground_truth_recall@20",
        "ground_truth_recall@30",
    ):
        row[f"surrogate_delta_{key}"] = float(cem[key]) - float(random_row[key])
    return [row]


def _paired_relative_rewards(
    cem: Mapping[str, Any],
    random_row: Mapping[str, Any],
) -> tuple[float, float, dict[str, float]]:
    eps = 1.0e-12
    cem_norms: list[float] = []
    random_norms: list[float] = []
    fields: dict[str, float] = {}
    for key in PAIRWISE_RELATIVE_REWARD_METRICS:
        cem_value = float(cem[key])
        random_value = float(random_row[key])
        denominator = max(cem_value, random_value, eps)
        cem_norm = cem_value / denominator
        random_norm = random_value / denominator
        cem_norms.append(cem_norm)
        random_norms.append(random_norm)
        safe_key = key.replace("@", "_at_").replace("/", "_")
        fields[f"paired_denominator_{safe_key}"] = float(denominator)
        fields[f"cem_paired_norm_{safe_key}"] = float(cem_norm)
        fields[f"random_paired_norm_{safe_key}"] = float(random_norm)
    return float(_mean(cem_norms)), float(_mean(random_norms)), fields


def _find_single_method(
    by_method: Mapping[str, Mapping[str, Any]],
    prefix: str,
) -> Mapping[str, Any] | None:
    matches = [row for method, row in by_method.items() if method.startswith(prefix)]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple rows matched method prefix {prefix!r}.")
    return matches[0]


def _final_metric_rows(
    *,
    random_summary: Mapping[str, Any],
    cem_summary: Mapping[str, Any],
    target_item: int,
) -> list[dict[str, Any]]:
    random_target = _summary_targets(random_summary)[str(int(target_item))]
    cem_target = _summary_targets(cem_summary)[str(int(target_item))]
    rows: list[dict[str, Any]] = []
    random_victims = random_target["victims"]
    cem_victims = cem_target["victims"]
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


def _reward_rows_with_pairwise_objective(
    reward_rows: Sequence[Mapping[str, Any]],
    *,
    reward_mode: str,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in reward_rows]
    if reward_mode != REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20 or len(rows) < 2:
        return rows
    by_method = {str(row["method"]): row for row in rows}
    cem = _find_single_method(by_method, "cem_best_rescored_ft")
    random_row = _find_single_method(by_method, "random_nz_ratio1_rescored_ft")
    if cem is None or random_row is None:
        return rows
    cem_reward, random_reward, _ = _paired_relative_rewards(cem, random_row)
    for row, value in ((cem, cem_reward), (random_row, random_reward)):
        row["objective_reward"] = float(value)
        row["paired_relative_objective_reward"] = float(value)
        row["objective_reward_note"] = (
            "paired-relative normalized over CEM and Random using "
            "targeted_mrr@10,targeted_mrr@20,targeted_recall@10,targeted_recall@20"
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
    target_item: int,
    fine_tune_steps: int,
    reward_mode: str,
    elapsed_seconds: float,
    complete: bool,
) -> None:
    reward_rows_for_output = _reward_rows_with_pairwise_objective(
        reward_rows,
        reward_mode=reward_mode,
    )
    _write_csv(output_dir / "surrogate_reward_comparison.csv", reward_rows_for_output)
    alignment_name = _alignment_filename(
        fine_tune_steps=fine_tune_steps,
        reward_mode=reward_mode,
    )
    _write_csv(output_dir / alignment_name, pairwise_rows)
    _write_csv(output_dir / "final_metric_overlap.csv", final_rows)
    (output_dir / "alignment_report.md").write_text(
        _render_report(
            pairwise_rows=pairwise_rows,
            reward_rows=reward_rows,
            reward_rows_for_output=reward_rows_for_output,
            complete=complete,
            elapsed_seconds=elapsed_seconds,
            config_path=config_path,
            cem_run_root=cem_run_root,
            random_run_root=random_run_root,
            clean_checkpoint=clean_checkpoint,
            target_item=target_item,
            fine_tune_steps=fine_tune_steps,
            reward_mode=reward_mode,
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
        "target_item": int(target_item),
        "fine_tune_steps": int(fine_tune_steps),
        "reward_mode": reward_mode,
        "completed_methods": [str(row["method"]) for row in reward_rows_for_output],
        "surrogate_reward_comparison": _repo_relative(
            output_dir / "surrogate_reward_comparison.csv"
        ),
        "surrogate_final_alignment": _repo_relative(output_dir / alignment_name),
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
    reward_rows: Sequence[Mapping[str, Any]],
    reward_rows_for_output: Sequence[Mapping[str, Any]],
    complete: bool,
    elapsed_seconds: float,
    config_path: Path,
    cem_run_root: Path,
    random_run_root: Path,
    clean_checkpoint: Path,
    target_item: int,
    fine_tune_steps: int,
    reward_mode: str,
) -> str:
    lines = [
        f"# Target {int(target_item)} ft{int(fine_tune_steps)} Surrogate-Final Alignment",
        "",
        f"- status: {'complete' if complete else 'partial'}",
        f"- target_item: {int(target_item)}",
        f"- fine_tune_steps: {int(fine_tune_steps)}",
        f"- reward_mode: `{reward_mode}`",
        f"- elapsed_seconds: {elapsed_seconds:.1f}",
        f"- config: `{_repo_relative(config_path)}`",
        f"- CEM run: `{_repo_relative(cem_run_root)}`",
        f"- Random-NZ run: `{_repo_relative(random_run_root)}`",
        f"- clean surrogate: `{_repo_relative(clean_checkpoint)}`",
        f"- CEM reward source: ft{int(fine_tune_steps)} rescore of existing CEM best poisoned sessions",
        f"- Random reward source: ft{int(fine_tune_steps)} rescore of reconstructed Random-NZ sessions",
        (
            "- paired-relative reward: each of targeted_mrr@10, targeted_mrr@20, "
            "targeted_recall@10, targeted_recall@20 is divided by "
            "max(CEM, Random) before averaging"
            if reward_mode == REWARD_MODE_PAIRED_RELATIVE_MRR_RECALL_10_20
            else "- paired-relative reward: not used"
        ),
        "",
        "## Candidate Scores",
        "",
        "| method | objective_reward | mrr_part | recall_part | actual_steps | actual_epochs | target_mean | gt_mean | T_MRR10 | T_MRR20 | T_MRR30 | T_R10 | T_R20 | T_R30 | GT_MRR10 | GT_MRR20 | GT_MRR30 | GT_R10 | GT_R20 | GT_R30 | total_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in reward_rows_for_output:
        lines.append(
            f"| {row['method']} | "
            f"{float(row['objective_reward']):.10f} | "
            f"{float(row['reward_mrr_part_10_20']):.6f} | "
            f"{float(row['reward_recall_part_10_20']):.6f} | "
            f"{_format_optional_int(row.get('actual_optimizer_steps'))} | "
            f"{_format_optional_int(row.get('actual_epochs'))} | "
            f"{float(row['target_result_mean']):.10f} | "
            f"{float(row['gt_result_mean']):.10f} | "
            f"{float(row['targeted_mrr@10']):.6f} | "
            f"{float(row['targeted_mrr@20']):.6f} | "
            f"{float(row['targeted_mrr@30']):.6f} | "
            f"{float(row['targeted_recall@10']):.6f} | "
            f"{float(row['targeted_recall@20']):.6f} | "
            f"{float(row['targeted_recall@30']):.6f} | "
            f"{float(row['ground_truth_mrr@10']):.6f} | "
            f"{float(row['ground_truth_mrr@20']):.6f} | "
            f"{float(row['ground_truth_mrr@30']):.6f} | "
            f"{float(row['ground_truth_recall@10']):.6f} | "
            f"{float(row['ground_truth_recall@20']):.6f} | "
            f"{float(row['ground_truth_recall@30']):.6f} | "
            f"{float(row['candidate_total_seconds']):.1f} |"
        )
    lines.extend(["", "## Four-Case Result", ""])
    if pairwise_rows:
        row = pairwise_rows[0]
        lines.extend(
            [
                "| target | CEM reward | Random reward | surrogate delta | surrogate judgement | final wins 6 | final wins @10/@20 4 | final judgement | case |",
                "|---:|---:|---:|---:|---|---:|---:|---|---|",
                (
                    f"| {row['target_item']} | "
                    f"{float(row['cem_surrogate_reward']):.10f} | "
                    f"{float(row['random_surrogate_reward']):.10f} | "
                    f"{float(row['surrogate_delta']):.10f} | "
                    f"{row['surrogate_judgement']} | "
                    f"{row['final_target_wins_6']} | "
                    f"{row['final_target_wins_4_at10_20']} | "
                    f"{row['final_judgement']} | "
                    f"{row['alignment_case']} |"
                ),
            ]
        )
    else:
        lines.append("Pairwise comparison pending until both candidates finish.")
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `surrogate_reward_comparison.csv`",
            f"- `{_alignment_filename(fine_tune_steps=fine_tune_steps, reward_mode=reward_mode)}`",
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


def _load_cem_optimized_sessions(*, cem_run_root: Path, target_item: int) -> list[list[int]]:
    path = (
        cem_run_root
        / "targets"
        / str(int(target_item))
        / "position_opt"
        / "cem"
        / "optimized_poisoned_sessions.pkl"
    )
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return [list(map(int, session)) for session in payload]


def _best_cem_trace_row(*, cem_run_root: Path, target_item: int) -> dict[str, Any]:
    path = cem_run_root / "targets" / str(int(target_item)) / "position_opt" / "cem" / "cem_trace.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty CEM trace: {path}")
    return dict(max(rows, key=lambda row: float(row["reward"])))


def _assert_target_available(
    random_summary: Mapping[str, Any],
    cem_summary: Mapping[str, Any],
    *,
    target_item: int,
) -> None:
    target_key = str(int(target_item))
    if target_key not in _summary_targets(random_summary):
        raise ValueError(f"target {target_item} missing from Random-NZ summary")
    if target_key not in _summary_targets(cem_summary):
        raise ValueError(f"target {target_item} missing from CEM summary")


def _summary_targets(summary: Mapping[str, Any]) -> dict[str, Any]:
    raw = summary.get("targets")
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, list):
        return {str(int(v["target_item"])): v for v in raw}
    raise ValueError("summary_current.json missing targets")


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else float("nan")


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


def _load_existing_reward_rows(path: Path, *, reward_mode: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_reward_mode = row.get("reward_mode")
        if not row_reward_mode and reward_mode == REWARD_MODE_TARGET_MEAN:
            filtered.append(row)
        elif row_reward_mode == reward_mode:
            filtered.append(row)
    return filtered


def _alignment_filename(*, fine_tune_steps: int, reward_mode: str) -> str:
    if reward_mode == REWARD_MODE_TARGET_MEAN:
        return f"surrogate_final_alignment_ft{int(fine_tune_steps)}.csv"
    return f"surrogate_final_alignment_ft{int(fine_tune_steps)}_{_safe_label(reward_mode)}.csv"


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def _format_optional_int(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(int(value))


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
