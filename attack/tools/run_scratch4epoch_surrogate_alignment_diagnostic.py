from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import load_json, save_json
from attack.common.config import (
    RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR,
    Config,
    load_config,
)
from attack.common.paths import POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset, expand_session_to_samples
from attack.inner_train.srgnn_full_retrain_validation_best import (
    SRGNNFullRetrainValidationBestInnerTrainer,
)
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.runs.run_rank_bucket_cem import run_rank_bucket_cem
from attack.surrogate.srgnn_backend import SRGNNBackend
from pytorch_code.utils import Data


TARGET_ITEM = 11103
LOWK_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)
FLOAT_TIE_EPSILON = 1.0e-12
DEFAULT_CONFIG = Path(
    "attack/configs/"
    "diginetica_valbest_attack_rank_bucket_cem_scratch4epoch_surrogate_tiny_11103.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/diagnostics/"
    "attack_rank_bucket_cem_scratch4epoch_surrogate_tiny_valbest_11103"
)
DEFAULT_RANDOM_VICTIM_METRICS = Path(
    "outputs/runs/diginetica/"
    "valbest_attack_random_nonzero_when_possible_ratio1_srgnn_sample3/"
    "run_group_7b8bde5ee4/targets/11103/victims/srgnn/metrics.json"
)
DEFAULT_RANDOM_POSITION_METADATA = Path(
    "outputs/runs/diginetica/"
    "valbest_attack_random_nonzero_when_possible_ratio1_srgnn_sample3/"
    "run_group_7b8bde5ee4/targets/11103/random_nonzero_position_metadata.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or assemble the SR-GNN scratch-4-epoch surrogate alignment diagnostic "
            "for tiny RankBucket-CEM vs Random-NZ."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Diagnostic CEM config.")
    parser.add_argument("--target-item", type=int, default=TARGET_ITEM)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for diagnostic report artifacts.",
    )
    parser.add_argument(
        "--random-victim-metrics",
        default=str(DEFAULT_RANDOM_VICTIM_METRICS),
        help="Existing new-valbest Random-NZ victim metrics.json.",
    )
    parser.add_argument(
        "--random-position-metadata",
        default=None,
        help=(
            "Existing Random-NZ random_nonzero_position_metadata.json. "
            "Defaults to the target directory next to --random-victim-metrics."
        ),
    )
    parser.add_argument(
        "--cem-metrics-path",
        default=None,
        help="Optional existing CEM victim metrics.json. Required with --skip-cem-run.",
    )
    parser.add_argument(
        "--cem-trace-path",
        default=None,
        help="Optional existing CEM cem_trace.jsonl. Defaults to path in CEM metrics.",
    )
    parser.add_argument(
        "--skip-cem-run",
        action="store_true",
        help="Do not call run_rank_bucket_cem; use --cem-metrics-path instead.",
    )
    parser.add_argument(
        "--random-surrogate-path",
        default=None,
        help="Optional cached random_nz_surrogate_scratch4epoch.json to reuse.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    target_item = int(args.target_item)
    _validate_diagnostic_config(config, target_item=target_item)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random_victim_metrics_path = Path(args.random_victim_metrics)
    random_position_metadata_path = _resolve_random_position_metadata_path(
        args.random_position_metadata,
        random_victim_metrics_path=random_victim_metrics_path,
    )

    cem_summary = None
    if not bool(args.skip_cem_run):
        cem_summary = run_rank_bucket_cem(config, config_path=config_path)

    cem_metrics_path = _resolve_cem_metrics_path(
        args.cem_metrics_path,
        summary=cem_summary,
        target_item=target_item,
    )
    cem_metrics_payload = _load_json_object(cem_metrics_path)
    safety = _reuse_safety_checks(
        config,
        cem_metrics_payload=cem_metrics_payload,
        cem_metrics_path=cem_metrics_path,
    )
    _fail_if_safety_checks_failed(safety)
    cem_trace_path = _resolve_cem_trace_path(
        args.cem_trace_path,
        cem_metrics_payload,
        cem_metrics_path=cem_metrics_path,
    )
    cem_trace_rows = [_normalize_cem_trace_row(row) for row in _read_jsonl(cem_trace_path)]
    cem_best_surrogate = _select_cem_best_surrogate(cem_trace_rows)

    if args.random_surrogate_path:
        random_surrogate = _load_json_object(Path(args.random_surrogate_path))
        _validate_cached_random_surrogate_replay(
            random_surrogate,
            expected_surrogate_train_seed=_target_level_surrogate_train_seed(
                config,
                target_item=target_item,
            ),
            expected_random_position_metadata_path=random_position_metadata_path,
        )
    else:
        random_surrogate = evaluate_random_nz_surrogate(
            config,
            target_item=target_item,
            random_position_metadata_path=random_position_metadata_path,
        )
    _assert_surrogate_train_seeds_match(cem_best_surrogate, random_surrogate)

    cem_victim = _victim_payload(cem_metrics_path)
    random_victim = _victim_payload(random_victim_metrics_path)
    comparison = _build_comparison(
        cem_best_surrogate=cem_best_surrogate,
        random_surrogate=random_surrogate,
        cem_victim=cem_victim,
        random_victim=random_victim,
    )

    _write_jsonl(output_dir / "cem_candidate_trace.jsonl", cem_trace_rows)
    save_json(cem_best_surrogate, output_dir / "cem_selected_surrogate.json")
    save_json(random_surrogate, output_dir / "random_nz_surrogate_scratch4epoch.json")
    save_json(
        {
            "cem_victim": cem_victim,
            "random_nz_victim": random_victim,
        },
        output_dir / "victim_comparison.json",
    )
    save_json(comparison, output_dir / "alignment_summary.json")
    manifest = {
        "target_item": int(target_item),
        "config_path": str(config_path),
        "cem_metrics_path": str(cem_metrics_path),
        "cem_trace_path": str(cem_trace_path),
        "random_victim_metrics_path": str(random_victim_metrics_path),
        "random_position_metadata_path": str(random_position_metadata_path),
        "output_dir": str(output_dir),
        "warning": _diagnostic_warning(),
        "reuse_safety_checks": safety,
        "artifacts": {
            "cem_candidate_trace": str(output_dir / "cem_candidate_trace.jsonl"),
            "cem_selected_surrogate": str(output_dir / "cem_selected_surrogate.json"),
            "random_nz_surrogate": str(output_dir / "random_nz_surrogate_scratch4epoch.json"),
            "victim_comparison": str(output_dir / "victim_comparison.json"),
            "alignment_summary": str(output_dir / "alignment_summary.json"),
            "report": str(output_dir / "report.md"),
        },
    }
    save_json(manifest, output_dir / "manifest.json")
    (output_dir / "report.md").write_text(
        _render_report(
            target_item=target_item,
            cem_best_surrogate=cem_best_surrogate,
            random_surrogate=random_surrogate,
            cem_victim=cem_victim,
            random_victim=random_victim,
            comparison=comparison,
            safety=safety,
        ),
        encoding="utf-8",
    )
    print(f"Wrote diagnostic artifacts to {output_dir}")


def evaluate_random_nz_surrogate(
    config: Config,
    *,
    target_item: int,
    random_position_metadata_path: Path,
) -> dict[str, Any]:
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        require_poison_runner=False,
        config_path=None,
    )
    rng = random.Random(int(config.seeds.fake_session_seed))
    policy = RandomNonzeroWhenPossiblePolicy(
        float(config.attack.replacement_topk_ratio),
        rng=rng,
    )
    selected_positions: list[int] = []
    random_fake_sessions: list[list[int]] = []
    for session in shared.template_sessions:
        result = policy.apply_with_metadata(session, int(target_item))
        random_fake_sessions.append(list(result.session))
        selected_positions.append(int(result.position))
    random_position_replay_check = _assert_random_position_replay_matches(
        selected_positions,
        random_position_metadata_path=random_position_metadata_path,
    )

    poisoned = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        random_fake_sessions,
    )
    validation_sessions, validation_labels = _validation_pairs(shared.canonical_dataset.valid)
    validation_data = Data((validation_sessions, validation_labels), shuffle=False)

    backend = SRGNNBackend(config, base_dir=Path.cwd())
    inner_trainer = SRGNNFullRetrainValidationBestInnerTrainer(
        train_config=backend.train_config,
        max_epochs=config.attack.rank_bucket_cem.surrogate_evaluator.max_epochs,
        patience=config.attack.rank_bucket_cem.surrogate_evaluator.patience,
        log_prefix="[diagnostic:random-nz-surrogate-full-retrain]",
    )
    surrogate_train_seed = _target_level_surrogate_train_seed(
        config,
        target_item=target_item,
    )
    start = time.perf_counter()
    result = inner_trainer.run(
        backend,
        None,
        poisoned,
        config=None,
        eval_data=validation_data,
        seed=surrogate_train_seed,
    )
    train_seconds = time.perf_counter() - start
    score_start = time.perf_counter()
    target_result = backend.score_target(result.model, validation_sessions, int(target_item))
    score_seconds = time.perf_counter() - score_start
    metrics = _metric_subset(_coerce_metrics(target_result.metrics))
    history = dict(result.history or {})
    return {
        "candidate_type": "random_nz",
        "target_item": int(target_item),
        "surrogate_evaluator_mode": RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR,
        "surrogate_train_seed": int(surrogate_train_seed),
        "selected_checkpoint_epoch": _optional_int(history.get("selected_checkpoint_epoch")),
        "valid_ground_truth_mrr@20": _optional_float(history.get("best_valid_mrr20")),
        "valid_ground_truth_recall@20": _optional_float(
            history.get("best_valid_recall20_at_best_mrr20_epoch")
        ),
        "best_valid_mrr20": _optional_float(history.get("best_valid_mrr20")),
        "best_valid_recall20": _optional_float(history.get("best_valid_recall20")),
        "stopped_epoch": _optional_int(history.get("stopped_epoch")),
        "stop_reason": history.get("stop_reason"),
        **metrics,
        "raw_lowk": _raw_lowk(metrics),
        "normalized_reward": None,
        "selected_as_iteration_elite": None,
        "selected_as_global_best": None,
        "train_seconds": float(train_seconds),
        "score_target_seconds": float(score_seconds),
        "fake_session_count": int(len(random_fake_sessions)),
        "selected_position_summary": _position_summary(selected_positions),
        "random_position_replay_check": random_position_replay_check,
    }


def _validate_diagnostic_config(config: Config, *, target_item: int) -> None:
    if config.attack.position_opt is None:
        raise ValueError("Diagnostic config requires attack.position_opt.")
    if config.attack.rank_bucket_cem is None:
        raise ValueError("Diagnostic config requires attack.rank_bucket_cem.")
    if tuple(config.targets.explicit_list) != (int(target_item),):
        raise ValueError(
            "Diagnostic config must target exactly the requested target item."
        )
    if config.attack.position_opt.clean_surrogate_checkpoint is not None:
        raise ValueError("Full-retrain diagnostic config must not set clean_surrogate_checkpoint.")
    if str(config.attack.position_opt.reward_mode) != "poisoned_target_utility":
        raise ValueError("Full-retrain diagnostic requires poisoned_target_utility reward mode.")
    if bool(config.attack.position_opt.enable_gt_penalty):
        raise ValueError("Full-retrain diagnostic does not support GT penalty.")
    if (
        str(config.attack.rank_bucket_cem.surrogate_evaluator.mode)
        != RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR
    ):
        raise ValueError("Diagnostic config must use full_retrain_validation_best surrogate mode.")


def _target_level_surrogate_train_seed(config: Config, *, target_item: int) -> int:
    return derive_seed(
        int(config.seeds.surrogate_train_seed),
        "rank_bucket_cem_surrogate_train",
        int(target_item),
    )


def _resolve_cem_metrics_path(
    explicit_path: str | None,
    *,
    summary: Mapping[str, Any] | None,
    target_item: int,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    if summary is None:
        raise ValueError("--cem-metrics-path is required when --skip-cem-run is used.")
    target_payload = summary.get("targets", {}).get(str(int(target_item)))
    if not isinstance(target_payload, Mapping):
        raise ValueError(f"CEM summary does not contain target {int(target_item)}.")
    victims = target_payload.get("victims", {})
    if not isinstance(victims, Mapping) or "srgnn" not in victims:
        raise ValueError("CEM summary does not contain srgnn victim output.")
    metrics_path = victims["srgnn"].get("metrics_path")
    if not isinstance(metrics_path, str) or not metrics_path.strip():
        raise ValueError("CEM summary srgnn victim output is missing metrics_path.")
    return Path(metrics_path)


def _resolve_random_position_metadata_path(
    explicit_path: str | None,
    *,
    random_victim_metrics_path: Path,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
    else:
        target_root = random_victim_metrics_path.parent.parent.parent
        path = target_root / "random_nonzero_position_metadata.json"
        if not path.exists() and DEFAULT_RANDOM_POSITION_METADATA.exists():
            path = DEFAULT_RANDOM_POSITION_METADATA
    if not path.exists():
        raise FileNotFoundError(
            "Random-NZ position metadata was not found. Provide "
            "--random-position-metadata pointing to random_nonzero_position_metadata.json. "
            f"Resolved path: {path}"
        )
    return path


def _resolve_cem_trace_path(
    explicit_path: str | None,
    cem_metrics_payload: Mapping[str, Any],
    *,
    cem_metrics_path: Path | None = None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    trace_path = cem_metrics_payload.get("rank_bucket_cem_trace_path")
    if isinstance(trace_path, str) and trace_path.strip():
        path = Path(trace_path)
        if path.exists():
            return path
        raise FileNotFoundError(
            "CEM metrics payload contains rank_bucket_cem_trace_path, but the file "
            f"does not exist: {path}"
        )
    if cem_metrics_path is not None:
        fallback = cem_metrics_path.parent.parent.parent / "position_opt" / "cem" / "cem_trace.jsonl"
        if fallback.exists():
            return fallback
        raise ValueError(
            "CEM metrics payload is missing rank_bucket_cem_trace_path and fallback "
            "cem_trace.jsonl was not found. "
            f"metrics_path={cem_metrics_path}; fallback={fallback}; "
            f"available_top_level_keys={sorted(str(key) for key in cem_metrics_payload.keys())}"
        )
    raise ValueError(
        "CEM metrics payload is missing rank_bucket_cem_trace_path. "
        f"available_top_level_keys={sorted(str(key) for key in cem_metrics_payload.keys())}"
    )


def _normalize_cem_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    target_metrics = _coerce_metrics(row.get("target_metrics"))
    metrics = _metric_subset({**target_metrics, **_coerce_metrics(row)})
    raw_lowk = row.get("absolute_raw_family_lowk_reward")
    if raw_lowk is None:
        raw_lowk = _raw_lowk(metrics)
    normalized_reward = row.get("global_normalized_lowk_reward")
    if normalized_reward is None:
        normalized_reward = row.get("iteration_normalized_lowk_reward")
    return {
        "target_item": int(row["target_item"]),
        "iteration": int(row["iteration"]),
        "candidate_id_in_iteration": int(row.get("candidate_id_in_iteration", row["candidate_id"])),
        "global_candidate_id": int(row.get("global_candidate_id", -1)),
        "surrogate_evaluator_mode": row.get("surrogate_evaluator_mode"),
        "surrogate_train_seed": _optional_int(row.get("surrogate_train_seed")),
        "selected_checkpoint_epoch": _optional_int(row.get("selected_checkpoint_epoch")),
        "valid_ground_truth_mrr@20": _optional_float(row.get("valid_ground_truth_mrr@20")),
        "valid_ground_truth_recall@20": _optional_float(row.get("valid_ground_truth_recall@20")),
        "best_valid_mrr20": _optional_float(row.get("best_valid_mrr20")),
        "best_valid_recall20": _optional_float(row.get("best_valid_recall20")),
        "stopped_epoch": _optional_int(row.get("stopped_epoch")),
        "stop_reason": row.get("stop_reason"),
        **metrics,
        "raw_lowk": float(raw_lowk),
        "normalized_reward": _optional_float(normalized_reward),
        "iteration_normalized_lowk_reward": _optional_float(
            row.get("iteration_normalized_lowk_reward")
        ),
        "global_normalized_lowk_reward": _optional_float(
            row.get("global_normalized_lowk_reward")
        ),
        "selected_as_iteration_elite": bool(row.get("selected_as_iteration_elite", False)),
        "selected_as_global_best": bool(row.get("selected_as_global_best", False)),
        "selected_positions": row.get("selected_positions"),
    }


def _select_cem_best_surrogate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("CEM trace is empty.")
    selected = [row for row in rows if bool(row.get("selected_as_global_best"))]
    if selected:
        return dict(selected[0])
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row.get("normalized_reward") or row.get("raw_lowk") or 0.0),
                -int(row.get("iteration", 0)),
                -int(row.get("candidate_id_in_iteration", 0)),
            ),
        )
    )


def _victim_payload(metrics_path: Path) -> dict[str, Any]:
    metrics_payload = _load_json_object(metrics_path)
    metrics_source = metrics_payload.get("metrics", metrics_payload)
    metrics = _coerce_metrics(metrics_source)
    selected_metrics = _metric_subset(metrics)
    train_history_path = metrics_path.parent / "train_history.json"
    train_history = (
        _load_json_object(train_history_path)
        if train_history_path.exists()
        else {}
    )
    return {
        "candidate_type": (
            "cem_selected"
            if str(metrics_payload.get("run_type")) == POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE
            else "random_nz"
        ),
        "target_item": int(metrics_payload.get("target_item", 0)),
        "metrics_path": str(metrics_path),
        "train_history_path": str(train_history_path) if train_history_path.exists() else None,
        **selected_metrics,
        "raw_lowk": _raw_lowk(selected_metrics),
        "ground_truth_mrr@20": _optional_float(metrics.get("ground_truth_mrr@20")),
        "ground_truth_recall@20": _optional_float(metrics.get("ground_truth_recall@20")),
        "selected_checkpoint_epoch": _optional_int(
            train_history.get("selected_checkpoint_epoch")
            or train_history.get("best_valid_mrr20_epoch")
        ),
        "best_valid_mrr20": _optional_float(train_history.get("best_valid_mrr20")),
        "best_valid_recall20": _optional_float(train_history.get("best_valid_recall20")),
        "stopped_epoch": _optional_int(train_history.get("stopped_epoch")),
        "stop_reason": train_history.get("stop_reason"),
    }


def _assert_random_position_replay_matches(
    selected_positions: Sequence[int],
    *,
    random_position_metadata_path: Path,
) -> dict[str, Any]:
    expected = _load_random_position_metadata(random_position_metadata_path)
    replay = _position_summary(selected_positions)
    replay_counts = {
        str(int(position)): int(count)
        for position, count in replay["counts"].items()
    }
    expected_counts = dict(expected["counts"])
    total_matches = int(replay["total"]) == int(expected["total"])
    counts_match = replay_counts == expected_counts
    check = {
        "checks_passed": bool(total_matches and counts_match),
        "metadata_path": str(random_position_metadata_path),
        "expected_total": int(expected["total"]),
        "replay_total": int(replay["total"]),
        "expected_counts": expected_counts,
        "replay_counts": replay_counts,
    }
    if not check["checks_passed"]:
        check["count_differences"] = _count_differences(
            expected_counts,
            replay_counts,
        )
        raise RuntimeError(
            "Random-NZ surrogate replay does not match the existing Random-NZ victim "
            "position metadata. This would compare Random victim positions against a "
            "different Random surrogate candidate. "
            f"check={check}"
        )
    return check


def _load_random_position_metadata(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    counts_raw = payload.get("counts")
    if not isinstance(counts_raw, Mapping):
        raise ValueError(
            "Random-NZ position metadata must contain a 'counts' object. "
            f"path={path}; available_keys={sorted(str(key) for key in payload.keys())}"
        )
    counts = {str(int(key)): int(value) for key, value in counts_raw.items()}
    total = int(payload.get("total", sum(counts.values())))
    if total != sum(counts.values()):
        raise ValueError(
            "Random-NZ position metadata total does not equal sum(counts). "
            f"path={path}; total={total}; sum_counts={sum(counts.values())}"
        )
    return {
        "path": str(path),
        "total": int(total),
        "counts": counts,
    }


def _count_differences(
    expected_counts: Mapping[str, int],
    replay_counts: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    keys = sorted(
        set(expected_counts) | set(replay_counts),
        key=lambda value: (0, int(value)) if str(value).lstrip("-").isdigit() else (1, str(value)),
    )
    return {
        str(key): {
            "expected": int(expected_counts.get(str(key), 0)),
            "replay": int(replay_counts.get(str(key), 0)),
        }
        for key in keys
        if int(expected_counts.get(str(key), 0)) != int(replay_counts.get(str(key), 0))
    }


def _validate_cached_random_surrogate_replay(
    random_surrogate: Mapping[str, Any],
    *,
    expected_surrogate_train_seed: int,
    expected_random_position_metadata_path: Path,
) -> None:
    check = random_surrogate.get("random_position_replay_check")
    if not isinstance(check, Mapping) or not bool(check.get("checks_passed")):
        raise RuntimeError(
            "Cached Random-NZ surrogate artifact is missing a successful "
            "random_position_replay_check. Recompute it without --random-surrogate-path "
            "or provide a cache produced by the updated diagnostic script."
        )
    if random_surrogate.get("surrogate_train_seed") is None:
        raise RuntimeError(
            "Cached Random-NZ surrogate artifact is missing surrogate_train_seed; "
            "recompute it with the updated diagnostic script."
        )
    cached_seed = int(random_surrogate["surrogate_train_seed"])
    if cached_seed != int(expected_surrogate_train_seed):
        raise RuntimeError(
            "Cached Random-NZ surrogate artifact used a different surrogate_train_seed "
            "than the CEM target-level scratch surrogate seed. "
            f"cached={cached_seed}; expected={int(expected_surrogate_train_seed)}"
        )
    metadata_path = check.get("metadata_path")
    if not isinstance(metadata_path, str) or not metadata_path.strip():
        raise RuntimeError(
            "Cached Random-NZ surrogate artifact replay check is missing metadata_path."
        )
    if Path(metadata_path).resolve() != expected_random_position_metadata_path.resolve():
        raise RuntimeError(
            "Cached Random-NZ surrogate artifact was checked against a different "
            "Random-NZ position metadata file. "
            f"cached={metadata_path}; expected={expected_random_position_metadata_path}"
        )


def _build_comparison(
    *,
    cem_best_surrogate: Mapping[str, Any],
    random_surrogate: Mapping[str, Any],
    cem_victim: Mapping[str, Any],
    random_victim: Mapping[str, Any],
) -> dict[str, Any]:
    surrogate_winner = _winner(
        float(cem_best_surrogate["raw_lowk"]),
        float(random_surrogate["raw_lowk"]),
    )
    victim_winner = _winner(
        float(cem_victim["raw_lowk"]),
        float(random_victim["raw_lowk"]),
    )
    return {
        "surrogate_winner_by_raw_lowk": surrogate_winner,
        "victim_winner_by_raw_lowk": victim_winner,
        "alignment_case": _alignment_case(surrogate_winner, victim_winner),
        "cem_surrogate_train_seed": cem_best_surrogate.get("surrogate_train_seed"),
        "random_surrogate_train_seed": random_surrogate.get("surrogate_train_seed"),
        "surrogate_train_seed_matched": (
            cem_best_surrogate.get("surrogate_train_seed")
            == random_surrogate.get("surrogate_train_seed")
        ),
        "surrogate_delta_raw_lowk_cem_minus_random": (
            float(cem_best_surrogate["raw_lowk"]) - float(random_surrogate["raw_lowk"])
        ),
        "victim_delta_raw_lowk_cem_minus_random": (
            float(cem_victim["raw_lowk"]) - float(random_victim["raw_lowk"])
        ),
        "warning": _diagnostic_warning(),
    }


def _winner(cem_value: float, random_value: float) -> str:
    if abs(float(cem_value) - float(random_value)) <= FLOAT_TIE_EPSILON:
        return "tie"
    return "CEM" if float(cem_value) > float(random_value) else "Random-NZ"


def _alignment_case(surrogate_winner: str, victim_winner: str) -> str:
    if surrogate_winner == "CEM" and victim_winner == "CEM":
        return "reward_align"
    if surrogate_winner == "Random-NZ" and victim_winner == "Random-NZ":
        return "search_or_candidate_gap"
    if surrogate_winner == "CEM" and victim_winner == "Random-NZ":
        return "reward_final_misalignment"
    if surrogate_winner == "Random-NZ" and victim_winner == "CEM":
        return "surrogate_too_conservative"
    return "tie_or_inconclusive"


def _assert_surrogate_train_seeds_match(
    cem_best_surrogate: Mapping[str, Any],
    random_surrogate: Mapping[str, Any],
) -> None:
    cem_seed = cem_best_surrogate.get("surrogate_train_seed")
    random_seed = random_surrogate.get("surrogate_train_seed")
    if cem_seed is None or random_seed is None:
        raise RuntimeError(
            "Surrogate train seed is missing from CEM or Random-NZ surrogate metrics. "
            f"cem_seed={cem_seed}; random_seed={random_seed}"
        )
    if int(cem_seed) != int(random_seed):
        raise RuntimeError(
            "CEM and Random-NZ scratch surrogate seeds differ; alignment "
            "comparison would mix candidate effect with init/shuffle noise. "
            f"cem_seed={int(cem_seed)}; random_seed={int(random_seed)}"
        )


def _reuse_safety_checks(
    config: Config,
    *,
    cem_metrics_payload: Mapping[str, Any],
    cem_metrics_path: Path,
) -> dict[str, Any]:
    run_metadata_path = cem_metrics_payload.get("rank_bucket_cem_run_metadata_path")
    run_metadata = {}
    if isinstance(run_metadata_path, str) and run_metadata_path.strip() and Path(run_metadata_path).exists():
        run_metadata = _load_json_object(Path(run_metadata_path))
    rank_config = run_metadata.get("rank_bucket_cem_config", {})
    if not isinstance(rank_config, Mapping):
        rank_config = {}
    evaluator = rank_config.get("surrogate_evaluator", {})
    if not isinstance(evaluator, Mapping):
        evaluator = {}
    return {
        "config_surrogate_evaluator_mode": config.attack.rank_bucket_cem.surrogate_evaluator.mode,
        "config_position_opt_clean_surrogate_checkpoint": (
            config.attack.position_opt.clean_surrogate_checkpoint
        ),
        "cem_metrics_clean_surrogate_checkpoint": cem_metrics_payload.get(
            "position_opt_clean_surrogate_checkpoint"
        ),
        "cem_run_metadata_path": run_metadata_path,
        "cem_run_metadata_clean_surrogate_checkpoint": run_metadata.get(
            "clean_surrogate_checkpoint"
        ),
        "cem_run_metadata_surrogate_evaluator_mode": evaluator.get("mode"),
        "shared_fake_sessions_path": run_metadata.get("shared_fake_sessions_path"),
        "checks_passed": bool(
            config.attack.position_opt.clean_surrogate_checkpoint is None
            and cem_metrics_payload.get("position_opt_clean_surrogate_checkpoint") is None
            and run_metadata.get("clean_surrogate_checkpoint") is None
            and evaluator.get("mode") == RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR
            and Path(cem_metrics_path).exists()
        ),
    }


def _fail_if_safety_checks_failed(safety: Mapping[str, Any]) -> None:
    if bool(safety.get("checks_passed")):
        return
    raise RuntimeError(
        "Scratch-4-epoch diagnostic safety checks failed. This usually means an old "
        "warm-start/clean-checkpoint artifact was reused or CEM scratch surrogate mode "
        "did not materialize correctly. "
        f"safety={dict(safety)}"
    )


def _render_report(
    *,
    target_item: int,
    cem_best_surrogate: Mapping[str, Any],
    random_surrogate: Mapping[str, Any],
    cem_victim: Mapping[str, Any],
    random_victim: Mapping[str, Any],
    comparison: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> str:
    lines = [
        "# Scratch-4-Epoch SR-GNN Surrogate Alignment Diagnostic",
        "",
        f"Target item: `{int(target_item)}`",
        "",
        f"Warning: {_diagnostic_warning()}",
        "",
        "## Surrogate Validation",
        "",
        _markdown_row("CEM selected", cem_best_surrogate),
        _markdown_row("Random-NZ", random_surrogate),
        "",
        "## Victim Test",
        "",
        _markdown_row("CEM selected", cem_victim),
        _markdown_row("Random-NZ", random_victim),
        "",
        "## Alignment",
        "",
        f"- surrogate_winner_by_raw_lowk: `{comparison['surrogate_winner_by_raw_lowk']}`",
        f"- victim_winner_by_raw_lowk: `{comparison['victim_winner_by_raw_lowk']}`",
        f"- alignment_case: `{comparison['alignment_case']}`",
        f"- safety_checks_passed: `{safety['checks_passed']}`",
        "",
        "## Safety Checks",
        "",
    ]
    for key, value in safety.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _markdown_row(label: str, payload: Mapping[str, Any]) -> str:
    return (
        f"- {label}: raw_lowk={float(payload['raw_lowk']):.9f}, "
        f"mrr10={float(payload['targeted_mrr@10']):.9f}, "
        f"mrr20={float(payload['targeted_mrr@20']):.9f}, "
        f"recall10={float(payload['targeted_recall@10']):.9f}, "
        f"recall20={float(payload['targeted_recall@20']):.9f}, "
        f"selected_epoch={payload.get('selected_checkpoint_epoch')}, "
        f"surrogate_seed={payload.get('surrogate_train_seed')}"
    )


def _validation_pairs(valid_sessions: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for session in valid_sessions:
        session_prefixes, session_labels = expand_session_to_samples(session)
        prefixes.extend([list(prefix) for prefix in session_prefixes])
        labels.extend(int(label) for label in session_labels)
    if not prefixes:
        raise ValueError("Validation split produced no prefixes.")
    return prefixes, labels


def _metric_subset(metrics: Mapping[str, float]) -> dict[str, float]:
    missing = [key for key in LOWK_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"Missing low-k metrics: {missing}")
    return {key: float(metrics[key]) for key in LOWK_KEYS}


def _raw_lowk(metrics: Mapping[str, float | int | None]) -> float:
    return float(sum(float(metrics[key]) for key in LOWK_KEYS) / float(len(LOWK_KEYS)))


def _position_summary(positions: Sequence[int]) -> dict[str, Any]:
    total = int(len(positions))
    counts: dict[str, int] = {}
    for position in positions:
        key = str(int(position))
        counts[key] = int(counts.get(key, 0) + 1)
    return {
        "total": total,
        "counts": counts,
        "ratios": {
            key: (float(count) / float(total) if total else 0.0)
            for key, count in counts.items()
        },
    }


def _coerce_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (int, float)):
            result[str(key)] = float(item)
    return result


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _diagnostic_warning() -> str:
    return (
        "tiny population [4,2,2] is for transfer/alignment diagnostic only; "
        "it is not evidence of final CEM effectiveness."
    )


if __name__ == "__main__":
    main()
