"""Score CEM and Random-NZ poisoned sessions with the same CEM surrogate reward."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import Config, load_config
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
from attack.position_opt.types import SurrogateScoreResult, TruncatedFineTuneConfig
from attack.surrogate.srgnn_backend import SRGNNBackend


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "attack/configs/diginetica_attack_rank_bucket_cem.yaml"
DEFAULT_CEM_RUN_ROOT = (
    REPO_ROOT
    / "outputs/runs/diginetica/attack_rank_bucket_cem/run_group_cadb73910d"
)
DEFAULT_RANDOM_NZ_RUN_ROOT = (
    REPO_ROOT
    / "outputs/runs/diginetica/attack_random_nonzero_when_possible_ratio1/run_group_8679b974a1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "analysis/diagnosis_outputs/diginetica_cem_vs_random_nz_surrogate_reward"
)
SURROGATE_METRIC_KEYS = (
    "targeted_mrr@10",
    "targeted_recall@10",
    "targeted_recall@20",
)
FINAL_COMPACT_METRICS = {
    f"targeted_{metric}@{k}"
    for metric in ("recall", "mrr")
    for k in (10, 20, 30)
}


class SurrogateRewardDiagnosisError(ValueError):
    """Raised when the diagnostic inputs are inconsistent."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        output_dir = _repo_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = _repo_path(args.config)
        cem_run_root = _repo_path(args.cem_run_root)
        random_run_root = _repo_path(args.random_nz_run_root)
        config = load_config(config_path)

        random_summary = _load_json_mapping(
            random_run_root / "summary_current.json",
            label="random-nz summary_current.json",
        )
        cem_summary = _load_json_mapping(
            cem_run_root / "summary_current.json",
            label="CEM summary_current.json",
        )
        target_items = _resolve_overlap_targets(
            random_summary,
            cem_summary,
            requested_targets=args.target_items,
        )

        position_opt_config = resolve_position_opt_config(config.attack.position_opt)
        rank_bucket_cem_config = resolve_rank_bucket_cem_config(
            config.attack.rank_bucket_cem
        )
        if position_opt_config.enable_gt_penalty:
            raise SurrogateRewardDiagnosisError(
                "This diagnostic currently supports runs with enable_gt_penalty=false."
            )
        clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
            config,
            run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
            override=position_opt_config.clean_surrogate_checkpoint,
        ).resolve()
        if not clean_checkpoint.is_file():
            raise SurrogateRewardDiagnosisError(
                f"Clean surrogate checkpoint not found: {clean_checkpoint}"
            )

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
        for target_item in target_items:
            validation_subset_seed = (
                None
                if position_opt_config.validation_subset_size is None
                else derive_seed(
                    config.seeds.position_opt_seed,
                    "rank_bucket_cem_validation_subset",
                    int(target_item),
                )
            )
            selected_validation_sessions, selected_validation_labels, subset_meta = (
                _select_validation_subset(
                    validation_sessions,
                    validation_labels,
                    subset_size=position_opt_config.validation_subset_size,
                    subset_seed=validation_subset_seed,
                )
            )
            del selected_validation_labels

            clean_result = _score_clean_target(
                config=config,
                clean_checkpoint=clean_checkpoint,
                validation_sessions=selected_validation_sessions,
                target_item=int(target_item),
            )
            clean_reward_value = _resolve_reward_value(
                clean_result,
                reward_metric=rank_bucket_cem_config.reward_metric,
            )
            clean_objective_reward = _resolve_objective_reward(
                reward_mode=position_opt_config.reward_mode,
                reward_value=clean_reward_value,
                clean_reward_value=clean_reward_value,
            )
            surrogate_train_seed = derive_seed(
                config.seeds.surrogate_train_seed,
                "rank_bucket_cem_surrogate_train",
                int(target_item),
            )

            cem_stored_row = _stored_cem_best_row(
                cem_run_root=cem_run_root,
                target_item=int(target_item),
                clean_reward_value=clean_reward_value,
                clean_objective_reward=clean_objective_reward,
                reward_mode=position_opt_config.reward_mode,
                reward_metric=rank_bucket_cem_config.reward_metric,
                fine_tune_steps=int(position_opt_config.fine_tune_steps),
                surrogate_train_seed=surrogate_train_seed,
                validation_subset_metadata=subset_meta,
            )
            reward_rows.append(cem_stored_row)

            method_sessions = {
                "cem_best_rescored": _load_cem_optimized_sessions(
                    cem_run_root=cem_run_root,
                    target_item=int(target_item),
                ),
                "random_nz_ratio1": _build_random_nz_sessions(
                    config,
                    shared.template_sessions,
                    target_item=int(target_item),
                ),
            }
            for method, poisoned_fake_sessions in method_sessions.items():
                score_result = _score_poisoned_sessions(
                    config=config,
                    clean_checkpoint=clean_checkpoint,
                    clean_sessions=shared.clean_sessions,
                    clean_labels=shared.clean_labels,
                    poisoned_fake_sessions=poisoned_fake_sessions,
                    validation_sessions=selected_validation_sessions,
                    target_item=int(target_item),
                    fine_tune_config=fine_tune_config,
                    surrogate_train_seed=surrogate_train_seed,
                )
                reward_value = _resolve_reward_value(
                    score_result,
                    reward_metric=rank_bucket_cem_config.reward_metric,
                )
                objective_reward = _resolve_objective_reward(
                    reward_mode=position_opt_config.reward_mode,
                    reward_value=reward_value,
                    clean_reward_value=clean_reward_value,
                )
                reward_rows.append(
                    _score_row(
                        target_item=int(target_item),
                        method=method,
                        source="rescored",
                        clean_reward_value=clean_reward_value,
                        clean_objective_reward=clean_objective_reward,
                        reward_value=reward_value,
                        objective_reward=objective_reward,
                        score_result=score_result,
                        reward_mode=position_opt_config.reward_mode,
                        reward_metric=rank_bucket_cem_config.reward_metric,
                        fine_tune_steps=int(position_opt_config.fine_tune_steps),
                        surrogate_train_seed=surrogate_train_seed,
                        validation_subset_metadata=subset_meta,
                    )
                )

            pairwise_rows.append(_build_pairwise_row(target_item, reward_rows))

        final_metric_rows = _build_final_metric_rows(
            random_summary=random_summary,
            cem_summary=cem_summary,
            target_items=target_items,
        )
        _write_csv(output_dir / "surrogate_reward_comparison.csv", reward_rows)
        _write_csv(output_dir / "surrogate_reward_pairwise.csv", pairwise_rows)
        _write_csv(output_dir / "final_metric_overlap.csv", final_metric_rows)
        report = _render_report(
            config_path=config_path,
            cem_run_root=cem_run_root,
            random_run_root=random_run_root,
            clean_checkpoint=clean_checkpoint,
            target_items=target_items,
            reward_rows=reward_rows,
            pairwise_rows=pairwise_rows,
            final_metric_rows=final_metric_rows,
        )
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        manifest = {
            "config": _repo_relative(config_path),
            "cem_run_root": _repo_relative(cem_run_root),
            "random_nz_run_root": _repo_relative(random_run_root),
            "clean_surrogate_checkpoint": _repo_relative(clean_checkpoint),
            "target_items": [int(item) for item in target_items],
            "surrogate_reward_comparison": _repo_relative(
                output_dir / "surrogate_reward_comparison.csv"
            ),
            "surrogate_reward_pairwise": _repo_relative(
                output_dir / "surrogate_reward_pairwise.csv"
            ),
            "final_metric_overlap": _repo_relative(output_dir / "final_metric_overlap.csv"),
            "report": _repo_relative(output_dir / "report.md"),
            "surrogate_reward_rows": len(reward_rows),
            "pairwise_rows": len(pairwise_rows),
            "final_metric_rows": len(final_metric_rows),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote surrogate reward diagnosis to {output_dir}")
        print(f"Targets: {', '.join(str(item) for item in target_items)}")
    except SurrogateRewardDiagnosisError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare CEM and Random-NZ with the same CEM surrogate reward."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cem-run-root", default=str(DEFAULT_CEM_RUN_ROOT))
    parser.add_argument("--random-nz-run-root", default=str(DEFAULT_RANDOM_NZ_RUN_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--target-items",
        nargs="*",
        type=int,
        help="Optional explicit target subset. Defaults to CEM/random-nz overlap.",
    )
    return parser


def _score_clean_target(
    *,
    config: Config,
    clean_checkpoint: Path,
    validation_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> SurrogateScoreResult:
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    backend.load_clean_checkpoint(clean_checkpoint)
    model = backend.clone_clean_model()
    return backend.score_target(model, validation_sessions, int(target_item))


def _score_poisoned_sessions(
    *,
    config: Config,
    clean_checkpoint: Path,
    clean_sessions: Sequence[Sequence[int]],
    clean_labels: Sequence[int],
    poisoned_fake_sessions: Sequence[Sequence[int]],
    validation_sessions: Sequence[Sequence[int]],
    target_item: int,
    fine_tune_config: TruncatedFineTuneConfig,
    surrogate_train_seed: int,
) -> SurrogateScoreResult:
    poisoned_dataset = build_poisoned_dataset(
        [list(session) for session in clean_sessions],
        [int(label) for label in clean_labels],
        [list(session) for session in poisoned_fake_sessions],
    )
    backend = SRGNNBackend(config, base_dir=REPO_ROOT)
    inner_trainer = TruncatedFineTuneInnerTrainer()
    inner_result = inner_trainer.run(
        backend,
        clean_checkpoint,
        poisoned_dataset,
        config=fine_tune_config,
        seed=int(surrogate_train_seed),
    )
    return backend.score_target(inner_result.model, validation_sessions, int(target_item))


def _stored_cem_best_row(
    *,
    cem_run_root: Path,
    target_item: int,
    clean_reward_value: float,
    clean_objective_reward: float,
    reward_mode: str,
    reward_metric: str | None,
    fine_tune_steps: int,
    surrogate_train_seed: int,
    validation_subset_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    best = _best_cem_trace_row(cem_run_root=cem_run_root, target_item=target_item)
    reward_value = (
        float(best["target_result_mean"])
        if reward_metric is None
        else float(_require_mapping(best.get("target_metrics"), "target_metrics")[reward_metric])
    )
    objective_reward = float(best["reward"])
    return _score_row(
        target_item=target_item,
        method="cem_best_stored",
        source="cem_trace",
        clean_reward_value=clean_reward_value,
        clean_objective_reward=clean_objective_reward,
        reward_value=reward_value,
        objective_reward=objective_reward,
        score_result=None,
        reward_mode=reward_mode,
        reward_metric=reward_metric,
        fine_tune_steps=fine_tune_steps,
        surrogate_train_seed=surrogate_train_seed,
        validation_subset_metadata=validation_subset_metadata,
        trace_row=best,
    )


def _score_row(
    *,
    target_item: int,
    method: str,
    source: str,
    clean_reward_value: float,
    clean_objective_reward: float,
    reward_value: float,
    objective_reward: float,
    score_result: SurrogateScoreResult | None,
    reward_mode: str,
    reward_metric: str | None,
    fine_tune_steps: int,
    surrogate_train_seed: int,
    validation_subset_metadata: Mapping[str, Any],
    trace_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = dict(score_result.metrics or {}) if score_result is not None else {}
    if trace_row is not None:
        metrics.update(dict(_require_mapping(trace_row.get("target_metrics"), "target_metrics")))
    row: dict[str, Any] = {
        "target_item": int(target_item),
        "method": method,
        "source": source,
        "reward_mode": str(reward_mode),
        "reward_metric": "target_result.mean" if reward_metric is None else str(reward_metric),
        "clean_reward_value": float(clean_reward_value),
        "clean_objective_reward": float(clean_objective_reward),
        "surrogate_reward_value": float(reward_value),
        "objective_reward": float(objective_reward),
        "delta_reward_value_vs_clean": float(reward_value) - float(clean_reward_value),
        "delta_objective_reward_vs_clean": float(objective_reward)
        - float(clean_objective_reward),
        "fine_tune_steps": int(fine_tune_steps),
        "surrogate_train_seed": int(surrogate_train_seed),
        "validation_subset_strategy": validation_subset_metadata.get("strategy"),
        "validation_subset_seed": validation_subset_metadata.get("seed"),
        "validation_subset_count": validation_subset_metadata.get("selected_count"),
    }
    for metric_key in SURROGATE_METRIC_KEYS:
        row[metric_key] = metrics.get(metric_key)
    if trace_row is not None:
        row["cem_iteration"] = int(trace_row["iteration"])
        row["cem_candidate_id"] = int(trace_row["candidate_id"])
    else:
        row["cem_iteration"] = None
        row["cem_candidate_id"] = None
    return row


def _build_pairwise_row(
    target_item: int,
    all_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = {
        str(row["method"]): row
        for row in all_rows
        if int(row["target_item"]) == int(target_item)
    }
    random_row = _require_mapping(rows.get("random_nz_ratio1"), "random_nz_ratio1 row")
    cem_stored = _require_mapping(rows.get("cem_best_stored"), "cem_best_stored row")
    cem_rescored = _require_mapping(rows.get("cem_best_rescored"), "cem_best_rescored row")
    output: dict[str, Any] = {
        "target_item": int(target_item),
        "random_nz_reward": random_row["objective_reward"],
        "cem_best_stored_reward": cem_stored["objective_reward"],
        "cem_best_rescored_reward": cem_rescored["objective_reward"],
        "cem_stored_minus_random_nz_reward": float(cem_stored["objective_reward"])
        - float(random_row["objective_reward"]),
        "cem_rescored_minus_random_nz_reward": float(cem_rescored["objective_reward"])
        - float(random_row["objective_reward"]),
        "cem_stored_minus_rescored_reward": float(cem_stored["objective_reward"])
        - float(cem_rescored["objective_reward"]),
    }
    for metric_key in SURROGATE_METRIC_KEYS:
        output[f"random_nz_{metric_key}"] = random_row.get(metric_key)
        output[f"cem_stored_{metric_key}"] = cem_stored.get(metric_key)
        output[f"cem_rescored_{metric_key}"] = cem_rescored.get(metric_key)
        output[f"cem_stored_minus_random_nz_{metric_key}"] = _optional_delta(
            cem_stored.get(metric_key),
            random_row.get(metric_key),
        )
        output[f"cem_rescored_minus_random_nz_{metric_key}"] = _optional_delta(
            cem_rescored.get(metric_key),
            random_row.get(metric_key),
        )
    return output


def _build_final_metric_rows(
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
        random_victims = _require_mapping(
            random_targets[target_key].get("victims"),
            f"random targets[{target_key}].victims",
        )
        cem_victims = _require_mapping(
            cem_targets[target_key].get("victims"),
            f"CEM targets[{target_key}].victims",
        )
        for victim in sorted(set(random_victims) & set(cem_victims)):
            random_metrics = _require_mapping(
                _require_mapping(random_victims[victim], f"random victim {victim}").get("metrics"),
                f"random victim {victim}.metrics",
            )
            cem_metrics = _require_mapping(
                _require_mapping(cem_victims[victim], f"CEM victim {victim}").get("metrics"),
                f"CEM victim {victim}.metrics",
            )
            for metric_key in sorted(set(random_metrics) & set(cem_metrics)):
                if "@" not in metric_key:
                    continue
                scope_metric, raw_k = metric_key.rsplit("@", 1)
                if "_" not in scope_metric:
                    continue
                scope, metric_name = scope_metric.split("_", 1)
                random_value = float(random_metrics[metric_key])
                cem_value = float(cem_metrics[metric_key])
                rows.append(
                    {
                        "target_item": int(target_item),
                        "victim_model": victim,
                        "metric_scope": scope,
                        "metric_name": metric_name,
                        "k": int(raw_k),
                        "metric_key": metric_key,
                        "random_nz_ratio1": random_value,
                        "cem": cem_value,
                        "cem_minus_random_nz": cem_value - random_value,
                        "relative_delta_pct": (
                            None
                            if random_value == 0.0
                            else (cem_value - random_value) / random_value * 100.0
                        ),
                    }
                )
    return rows


def _build_random_nz_sessions(
    config: Config,
    template_sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
) -> list[list[int]]:
    rng = random.Random(int(config.seeds.fake_session_seed))
    policy = RandomNonzeroWhenPossiblePolicy(
        float(config.attack.replacement_topk_ratio),
        rng=rng,
    )
    return [
        policy.apply_with_metadata(session, int(target_item)).session
        for session in template_sessions
    ]


def _load_cem_optimized_sessions(
    *,
    cem_run_root: Path,
    target_item: int,
) -> list[list[int]]:
    path = (
        cem_run_root
        / "targets"
        / str(int(target_item))
        / "position_opt"
        / "cem"
        / "optimized_poisoned_sessions.pkl"
    )
    if not path.is_file():
        raise SurrogateRewardDiagnosisError(f"Missing CEM optimized sessions: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list):
        raise SurrogateRewardDiagnosisError(f"CEM optimized sessions must be a list: {path}")
    return [list(map(int, session)) for session in payload]


def _best_cem_trace_row(
    *,
    cem_run_root: Path,
    target_item: int,
) -> dict[str, Any]:
    path = (
        cem_run_root
        / "targets"
        / str(int(target_item))
        / "position_opt"
        / "cem"
        / "cem_trace.jsonl"
    )
    if not path.is_file():
        raise SurrogateRewardDiagnosisError(f"Missing CEM trace: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise SurrogateRewardDiagnosisError(f"CEM trace is empty: {path}")
    best = max(rows, key=lambda row: float(row["reward"]))
    return dict(best)


def _resolve_overlap_targets(
    random_summary: Mapping[str, Any],
    cem_summary: Mapping[str, Any],
    *,
    requested_targets: Sequence[int] | None,
) -> list[int]:
    random_targets = _summary_targets(random_summary)
    cem_targets = _summary_targets(cem_summary)
    overlap = sorted(set(random_targets) & set(cem_targets), key=lambda item: int(item))
    if requested_targets:
        requested = [str(int(item)) for item in requested_targets]
        missing = sorted(set(requested) - set(overlap), key=lambda item: int(item))
        if missing:
            raise SurrogateRewardDiagnosisError(
                f"Requested targets are not in the CEM/random-nz overlap: {missing}"
            )
        overlap = requested
    if not overlap:
        raise SurrogateRewardDiagnosisError("No overlapping targets found.")
    return [int(item) for item in overlap]


def _summary_targets(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_targets = summary.get("targets")
    if isinstance(raw_targets, Mapping):
        return {str(key): _require_mapping(value, f"targets[{key}]") for key, value in raw_targets.items()}
    if isinstance(raw_targets, list):
        output: dict[str, Mapping[str, Any]] = {}
        for index, value in enumerate(raw_targets):
            entry = _require_mapping(value, f"targets[{index}]")
            output[str(int(entry["target_item"]))] = entry
        return output
    raise SurrogateRewardDiagnosisError("summary_current.json is missing targets.")


def _resolve_reward_value(
    result: SurrogateScoreResult,
    *,
    reward_metric: str | None,
) -> float:
    if reward_metric is None:
        return float(result.mean)
    metrics = result.metrics or {}
    if reward_metric not in metrics:
        raise SurrogateRewardDiagnosisError(
            f"Reward metric {reward_metric!r} not found in surrogate metrics: {sorted(metrics)}"
        )
    return float(metrics[reward_metric])


def _resolve_objective_reward(
    *,
    reward_mode: str,
    reward_value: float,
    clean_reward_value: float,
) -> float:
    if reward_mode == "poisoned_target_utility":
        return float(reward_value)
    if reward_mode == "delta_target_utility":
        return float(reward_value) - float(clean_reward_value)
    raise SurrogateRewardDiagnosisError(f"Unsupported reward_mode: {reward_mode!r}")


def _render_report(
    *,
    config_path: Path,
    cem_run_root: Path,
    random_run_root: Path,
    clean_checkpoint: Path,
    target_items: Sequence[int],
    reward_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    final_metric_rows: Sequence[Mapping[str, Any]],
) -> str:
    wins = sum(
        1
        for row in pairwise_rows
        if float(row["cem_rescored_minus_random_nz_reward"]) > 0.0
    )
    final_compact = [
        row
        for row in final_metric_rows
        if row["metric_key"] in FINAL_COMPACT_METRICS
        and str(row["metric_scope"]) == "targeted"
    ]
    final_wins = sum(
        1 for row in final_compact if float(row["cem_minus_random_nz"]) > 0.0
    )
    lines = [
        "# CEM vs Random-NZ Surrogate Reward Diagnosis",
        "",
        f"- config: `{_repo_relative(config_path)}`",
        f"- CEM run: `{_repo_relative(cem_run_root)}`",
        f"- Random-NZ run: `{_repo_relative(random_run_root)}`",
        f"- clean surrogate: `{_repo_relative(clean_checkpoint)}`",
        f"- targets: {', '.join(str(item) for item in target_items)}",
        "",
        "## Surrogate Reward Pairwise",
        "",
        "| target | random_reward | cem_rescored_reward | cem-random | stored-rescored |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise_rows:
        lines.append(
            f"| {row['target_item']} | "
            f"{float(row['random_nz_reward']):.10f} | "
            f"{float(row['cem_best_rescored_reward']):.10f} | "
            f"{float(row['cem_rescored_minus_random_nz_reward']):.10f} | "
            f"{float(row['cem_stored_minus_rescored_reward']):.10f} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            (
                f"- CEM rescored surrogate reward beats Random-NZ on "
                f"{wins}/{len(pairwise_rows)} target(s)."
            ),
            (
                f"- On final targeted recall/mrr@10/20/30 across victims, CEM beats "
                f"Random-NZ on {final_wins}/{len(final_compact)} cell(s)."
            ),
            "- If CEM reward wins here but final metrics do not, the main issue is reward/proxy misalignment.",
            "- If Random-NZ reward wins here, the main issue is CEM search budget or policy parameterization.",
            "",
            "## Output Files",
            "",
            "- `surrogate_reward_comparison.csv`",
            "- `surrogate_reward_pairwise.csv`",
            "- `final_metric_overlap.csv`",
            "- `manifest.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise SurrogateRewardDiagnosisError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SurrogateRewardDiagnosisError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    return _require_mapping(payload, label)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SurrogateRewardDiagnosisError(f"{label} must be a JSON object.")
    return value


def _optional_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else (REPO_ROOT / path_obj)


def _repo_relative(path: str | Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path_obj)


if __name__ == "__main__":
    raise SystemExit(main())
