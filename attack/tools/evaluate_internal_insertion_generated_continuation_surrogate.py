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

import torch

from attack.common.artifact_io import load_poison_model, save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.inner_train.srgnn_full_retrain_validation_best import (
    SRGNNFullRetrainValidationBestInnerTrainer,
)
from attack.insertion.internal_random_insertion_generated_suffix import (
    InternalRandomInsertionGeneratedContinuationPolicy,
)
from attack.models.poison.srgnn_poison_runner import SRGNNPoisonRunner
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics
from attack.pipeline.core.pipeline_utils import (
    build_srgnn_opt_from_train_config,
    prepare_shared_attack_artifacts,
)
from attack.pipeline.core.victim_execution import _victim_stage_seed
from attack.pipeline.runs.run_internal_random_insertion_generated_continuation import (
    build_internal_random_insertion_generated_continuation_metadata,
    _validate_internal_insertion_generated_continuation_sessions,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _candidate_checkpoint_metadata,
    _coerce_target_metrics,
    _lowk_reward_metric_payload,
    _resolve_validation_pairs,
    _srgnn_candidate_train_config,
    _pts_construction_artifact_dir,
    build_pts_construction_attack_identity_context,
)
from attack.surrogate.srgnn_backend import SRGNNBackend
from pytorch_code.model import forward as srg_forward
from pytorch_code.model import trans_to_cpu
from pytorch_code.utils import Data


DEFAULT_CONFIG = Path(
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_vertex_sf_a0a4_"
    "elite_centered_ratio1_srgnn_partial4_target5334.yaml"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/diagnostics/"
    "pts_cem_internal_insertion_generated_continuation_target5334"
)
DEFAULT_TARGET_ITEM = 5334
LOWK_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate internal insertion generated continuation with the PTS-CEM "
            "surrogate candidate-retrain reward protocol. This does not run CEM "
            "or final victim evaluation."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="PTS-CEM YAML config.")
    parser.add_argument("--target-item", type=int, default=DEFAULT_TARGET_ITEM)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--sessions-path",
        default=None,
        help=(
            "Optional existing heuristic sessions JSON. If omitted, sessions are "
            "generated from existing shared fake sessions and poison model."
        ),
    )
    parser.add_argument(
        "--known-final-victim-raw-lowk",
        type=float,
        default=0.146375,
        help="Reference final-victim raw_lowk to include in the comparison only.",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    target_item = int(args.target_item)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _validate_config(config, target_item=target_item)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    _require_existing_shared_artifacts(
        shared_paths,
        need_poison_model=args.sessions_path is None,
    )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        require_poison_runner=False,
        config_path=None,
    )

    if args.sessions_path is None:
        sessions, heuristic_metadata = _generate_heuristic_sessions(
            config,
            shared=shared,
            target_item=target_item,
        )
        sessions_path = output_dir / "internal_insertion_generated_continuation_sessions.json"
        save_json(sessions, sessions_path)
        save_json(
            heuristic_metadata,
            output_dir / "internal_insertion_generated_continuation_metadata.json",
        )
    else:
        sessions_path = Path(args.sessions_path)
        sessions = _load_sessions_json(sessions_path)
        heuristic_metadata = {
            "source": "loaded_sessions",
            "sessions_path": str(sessions_path),
            "fake_session_count": int(len(sessions)),
            "target_item": target_item,
        }

    evaluation = _evaluate_pts_surrogate_reward(
        config,
        shared=shared,
        candidate_sessions=sessions,
        target_item=target_item,
    )
    comparison = _comparison_payload(
        config,
        target_item=target_item,
        heuristic_raw_lowk=float(evaluation["raw_lowk_reward"]),
        known_final_victim_raw_lowk=float(args.known_final_victim_raw_lowk),
    )
    payload = {
        "heuristic_name": "internal_insertion_generated_continuation",
        "target_item": target_item,
        "config_path": str(config_path),
        "sessions_path": str(sessions_path),
        "output_dir": str(output_dir),
        "surrogate_effective_seed": int(config.seeds.surrogate_train_seed),
        "victim_effective_seed_reference": int(
            _victim_stage_seed(
                config,
                victim_name="srgnn",
                run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                target_item=target_item,
            )
        ),
        "note": (
            "Victim seed is reported only as reference. This diagnostic trains "
            "only the PTS-CEM surrogate evaluator."
        ),
        "heuristic_metadata": heuristic_metadata,
        "surrogate_evaluation": evaluation,
        "comparison": comparison,
    }
    result_path = output_dir / "surrogate_reward_summary.json"
    report_path = output_dir / "report.md"
    save_json(payload, result_path)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    print(f"heuristic=internal_insertion_generated_continuation")
    print(f"target_item={target_item}")
    print(f"surrogate_effective_seed={int(config.seeds.surrogate_train_seed)}")
    print(f"victim_effective_seed_reference={payload['victim_effective_seed_reference']}")
    print(f"targeted_mrr@10={evaluation['targeted_mrr@10']}")
    print(f"targeted_mrr@20={evaluation['targeted_mrr@20']}")
    print(f"targeted_recall@10={evaluation['targeted_recall@10']}")
    print(f"targeted_recall@20={evaluation['targeted_recall@20']}")
    print(f"raw_lowk_reward={evaluation['raw_lowk_reward']}")
    print(f"summary_path={result_path}")
    print(f"report_path={report_path}")


def _validate_config(config: Config, *, target_item: int) -> None:
    if not bool(config.data.poison_train_only):
        raise ValueError("Diagnostic requires data.poison_train_only=true.")
    if "srgnn" not in set(config.victims.enabled):
        raise ValueError("Diagnostic requires srgnn victim/surrogate config.")
    if config.attack.pts_construction is None:
        raise ValueError("Diagnostic expects a PTS-CEM config with attack.pts_construction.")
    if config.targets.mode == "explicit_list" and int(target_item) not in {
        int(item) for item in config.targets.explicit_list
    }:
        raise ValueError("target_item is not present in config.targets.explicit_list.")


def _require_existing_shared_artifacts(
    shared_paths: Mapping[str, Path],
    *,
    need_poison_model: bool,
) -> None:
    fake_sessions_path = Path(shared_paths["fake_sessions"])
    if not fake_sessions_path.exists():
        raise FileNotFoundError(
            "Shared fake sessions are required and will not be regenerated by this "
            f"diagnostic. Missing: {fake_sessions_path}"
        )
    if need_poison_model:
        poison_model_path = Path(shared_paths["poison_model"])
        if not poison_model_path.exists():
            raise FileNotFoundError(
                "Existing poison model checkpoint is required for generated "
                f"continuation. Missing: {poison_model_path}"
            )


def _generate_heuristic_sessions(
    config: Config,
    *,
    shared,
    target_item: int,
) -> tuple[list[list[int]], dict[str, object]]:
    poison_runner = _load_existing_poison_runner(config, shared.shared_paths["poison_model"])
    policy = InternalRandomInsertionGeneratedContinuationPolicy(
        topk_ratio=float(config.attack.replacement_topk_ratio),
        poison_runner=poison_runner,
        generation_topk=int(config.attack.fake_session_generation_topk),
        insertion_rng=random.Random(int(config.seeds.fake_session_seed)),
        generation_rng_base_seed=int(config.seeds.fake_session_seed),
    )
    results = [
        policy.apply_with_metadata(session, int(target_item), index)
        for index, session in enumerate(shared.template_sessions)
    ]
    sessions = [[int(item) for item in result.session] for result in results]
    max_item = max(shared.stats.item_counts)
    _validate_internal_insertion_generated_continuation_sessions(
        template_sessions=shared.template_sessions,
        final_sessions=sessions,
        results=results,
        target_item=int(target_item),
        max_item_id=max_item,
    )
    metadata = build_internal_random_insertion_generated_continuation_metadata(
        config=config,
        run_type="surrogate_diagnostic_only",
        operation="internal_insertion_generated_continuation",
        suffix_strategy="target_conditioned_generated_continuation",
        target_item=int(target_item),
        template_sessions=shared.template_sessions,
        insertion_results=results,
        clean_train_sessions=shared.canonical_dataset.train_sub,
        slot_stats_payload={"overall": {"count": int(len(results))}},
        template_fake_sessions_path=shared.shared_paths["fake_sessions"],
        poison_model_checkpoint_path=shared.shared_paths["poison_model"],
        generation_topk=int(config.attack.fake_session_generation_topk),
        generation_rng_base_seed=int(config.seeds.fake_session_seed),
    )
    return sessions, metadata


def _load_existing_poison_runner(config: Config, checkpoint_path: str | Path) -> SRGNNPoisonRunner:
    train_config = dict(config.attack.poison_model.params["train"])
    runner = SRGNNPoisonRunner(config)
    runner.build_model(build_srgnn_opt_from_train_config(train_config))
    if not load_poison_model(runner, checkpoint_path):
        raise FileNotFoundError(f"Poison model checkpoint not found: {checkpoint_path}")
    return runner


def _evaluate_pts_surrogate_reward(
    config: Config,
    *,
    shared,
    candidate_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> dict[str, object]:
    train_config = _srgnn_candidate_train_config(config)
    validation_sessions, validation_labels = _resolve_validation_pairs(shared)
    validation_data = Data((validation_sessions, validation_labels), shuffle=False)
    backend = SRGNNBackend(config, base_dir=Path.cwd(), train_config=train_config)
    inner_trainer = SRGNNFullRetrainValidationBestInnerTrainer(
        train_config=train_config,
        max_epochs=int(train_config["epochs"]),
        patience=int(train_config["patience"]),
        log_prefix="[diagnostic:pts-cem-surrogate-retrain]",
    )
    poisoned_train = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        candidate_sessions,
    )
    candidate_start = time.perf_counter()
    retrain_start = time.perf_counter()
    inner_result = inner_trainer.run(
        backend,
        None,
        poisoned_train,
        config=None,
        eval_data=validation_data,
        seed=int(config.seeds.surrogate_train_seed),
    )
    retrain_seconds = time.perf_counter() - retrain_start
    score_start = time.perf_counter()
    target_result = backend.score_target(
        inner_result.model,
        validation_sessions,
        int(target_item),
    )
    score_target_seconds = time.perf_counter() - score_start
    metrics = _coerce_target_metrics(target_result.metrics)
    lowk_payload = _lowk_reward_metric_payload(metrics)
    gt_metrics = _ground_truth_metrics(
        inner_result.model,
        validation_sessions=validation_sessions,
        validation_labels=validation_labels,
        topk=list(config.evaluation.topk),
        metrics=list(config.evaluation.ground_truth_metrics),
    )
    total_seconds = time.perf_counter() - candidate_start
    history = dict(inner_result.history or {})
    return {
        "surrogate_protocol": "pts_cem_candidate_retrain_validation_best",
        "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        "candidate_retrain_epochs": int(train_config["epochs"]),
        "candidate_retrain_validation_prefix_count": int(len(validation_sessions)),
        "targeted_mrr@10": float(metrics["targeted_mrr@10"]),
        "targeted_mrr@20": float(metrics["targeted_mrr@20"]),
        "targeted_recall@10": float(metrics["targeted_recall@10"]),
        "targeted_recall@20": float(metrics["targeted_recall@20"]),
        "targeted_mrr@30": float(metrics["targeted_mrr@30"]),
        "targeted_recall@30": float(metrics["targeted_recall@30"]),
        "raw_lowk_reward": float(lowk_payload["absolute_raw_family_lowk_reward"]),
        "reward_name": "raw_lowk_mrr_recall_10_20",
        "ground_truth_metrics": gt_metrics,
        "candidate_retrain_seconds": float(retrain_seconds),
        "score_target_seconds": float(score_target_seconds),
        "candidate_total_seconds": float(total_seconds),
        **_candidate_checkpoint_metadata(history),
    }


def _ground_truth_metrics(
    model_handle,
    *,
    validation_sessions: Sequence[Sequence[int]],
    validation_labels: Sequence[int],
    topk: Sequence[int],
    metrics: Sequence[str],
) -> dict[str, float | None]:
    max_topk = max(int(value) for value in topk)
    rankings = _predict_topk(model_handle, validation_sessions, topk=max_topk)
    computed, _ = evaluate_ground_truth_metrics(
        rankings,
        labels=validation_labels,
        metrics=metrics,
        topk=topk,
    )
    return {str(key): (None if value is None else float(value)) for key, value in computed.items()}


def _predict_topk(model_handle, sessions: Sequence[Sequence[int]], *, topk: int) -> list[list[int]]:
    data = Data(([[int(item) for item in session] for session in sessions], [1] * len(sessions)), shuffle=False)
    torch_model = model_handle.model
    rankings: list[list[int]] = []
    torch_model.eval()
    with torch.no_grad():
        for batch_indices in data.generate_batch(torch_model.batch_size):
            _, scores = srg_forward(torch_model, batch_indices, data)
            k = min(int(topk), int(scores.shape[1]))
            topk_indices = scores.topk(k)[1]
            topk_indices = trans_to_cpu(topk_indices).detach().numpy()
            rankings.extend(
                [[int(item) + 1 for item in row.tolist()] for row in topk_indices]
            )
    return rankings


def _comparison_payload(
    config: Config,
    *,
    target_item: int,
    heuristic_raw_lowk: float,
    known_final_victim_raw_lowk: float,
) -> dict[str, object]:
    source = {
        "iter0_cand1_c0_generate_near_vertex_surrogate_reward": 0.1238733173348644,
        "iter0_cand3_c1_generate_where_valid_surrogate_reward": 0.142318,
        "known_internal_insertion_generated_continuation_final_victim_raw_lowk": (
            float(known_final_victim_raw_lowk)
        ),
    }
    trace_payload = _load_reference_trace_rewards(config, target_item=target_item)
    if trace_payload:
        source.update(trace_payload)
    return {
        **source,
        "heuristic_surrogate_raw_lowk": float(heuristic_raw_lowk),
        "delta_vs_iter0_cand1_surrogate": float(
            heuristic_raw_lowk
            - float(source["iter0_cand1_c0_generate_near_vertex_surrogate_reward"])
        ),
        "delta_vs_iter0_cand3_surrogate": float(
            heuristic_raw_lowk
            - float(source["iter0_cand3_c1_generate_where_valid_surrogate_reward"])
        ),
    }


def _load_reference_trace_rewards(config: Config, *, target_item: int) -> dict[str, object]:
    try:
        artifact_dir = _pts_construction_artifact_dir(
            config,
            int(target_item),
            attack_identity_context=build_pts_construction_attack_identity_context(config),
        )
    except Exception:
        return {}
    trace_path = artifact_dir / "pts_cem_trace.jsonl"
    if not trace_path.exists():
        return {}
    wanted = {
        "iter0_cand1": "iter0_cand1_c0_generate_near_vertex",
        "iter0_cand3": "iter0_cand3_c1_generate_where_valid",
    }
    found: dict[str, object] = {"reference_trace_path": str(trace_path)}
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("candidate_key"))
            prefix = wanted.get(key)
            if prefix is None:
                continue
            found[f"{prefix}_surrogate_reward"] = float(row["reward"])
            found[f"{prefix}_candidate_seed"] = int(row["candidate_seed"])
            found[f"{prefix}_sample_origin"] = row.get("sample_origin")
            sample_metadata = row.get("sample_metadata")
            if isinstance(sample_metadata, Mapping):
                found[f"{prefix}_vertex_name"] = sample_metadata.get("vertex_name")
    return found


def _load_sessions_json(path: Path) -> list[list[int]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("sessions JSON must contain a list.")
    sessions: list[list[int]] = []
    for row in payload:
        if not isinstance(row, list):
            raise ValueError("sessions JSON rows must be lists.")
        sessions.append([int(item) for item in row])
    return sessions


def _render_report(payload: Mapping[str, Any]) -> str:
    evaluation = payload["surrogate_evaluation"]
    comparison = payload["comparison"]
    gt = evaluation.get("ground_truth_metrics", {})
    lines = [
        "# Internal Insertion Generated Continuation Surrogate Diagnostic",
        "",
        f"- heuristic: `{payload['heuristic_name']}`",
        f"- target_item: `{payload['target_item']}`",
        f"- surrogate_effective_seed: `{payload['surrogate_effective_seed']}`",
        f"- victim_effective_seed_reference: `{payload['victim_effective_seed_reference']}`",
        "",
        "## Surrogate Reward",
        "",
        f"- targeted_mrr@10: `{float(evaluation['targeted_mrr@10']):.12f}`",
        f"- targeted_mrr@20: `{float(evaluation['targeted_mrr@20']):.12f}`",
        f"- targeted_recall@10: `{float(evaluation['targeted_recall@10']):.12f}`",
        f"- targeted_recall@20: `{float(evaluation['targeted_recall@20']):.12f}`",
        f"- raw_lowk_reward: `{float(evaluation['raw_lowk_reward']):.12f}`",
        "",
        "## Comparison",
        "",
        (
            "- iter0_cand1 c0_generate near-vertex surrogate reward: "
            f"`{float(comparison['iter0_cand1_c0_generate_near_vertex_surrogate_reward']):.12f}`"
        ),
        (
            "- iter0_cand3 c1_generate_where_valid surrogate reward: "
            f"`{float(comparison['iter0_cand3_c1_generate_where_valid_surrogate_reward']):.12f}`"
        ),
        (
            "- known internal insertion generated continuation final victim raw_lowk: "
            f"`{float(comparison['known_internal_insertion_generated_continuation_final_victim_raw_lowk']):.12f}`"
        ),
        (
            "- delta_vs_iter0_cand1_surrogate: "
            f"`{float(comparison['delta_vs_iter0_cand1_surrogate']):.12f}`"
        ),
        (
            "- delta_vs_iter0_cand3_surrogate: "
            f"`{float(comparison['delta_vs_iter0_cand3_surrogate']):.12f}`"
        ),
    ]
    if isinstance(gt, Mapping) and gt:
        lines.extend(
            [
                "",
                "## Ground Truth Metrics",
                "",
            ]
        )
        for key in sorted(gt):
            value = gt[key]
            lines.append(f"- {key}: `{None if value is None else f'{float(value):.12f}'}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
