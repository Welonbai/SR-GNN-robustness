from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.tools.run_scratch4epoch_surrogate_alignment_diagnostic import (
    DEFAULT_RANDOM_POSITION_METADATA,
    DEFAULT_RANDOM_VICTIM_METRICS,
    _assert_surrogate_train_seeds_match,
    _build_comparison,
    _fail_if_safety_checks_failed,
    _load_json_object,
    _normalize_cem_trace_row,
    _read_jsonl,
    _resolve_cem_trace_path,
    _resolve_random_position_metadata_path,
    _reuse_safety_checks,
    _select_cem_best_surrogate,
    _target_level_surrogate_train_seed,
    _validate_cached_random_surrogate_replay,
    _validate_diagnostic_config,
    _victim_payload,
    _winner,
    _write_jsonl,
)


TARGET_ITEM = 11103
LOWK_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)
DEFAULT_CONFIG = Path(
    "attack/configs/"
    "diginetica_valbest_attack_rank_bucket_cem_scratch4epoch_valbest_surrogate_tailboosted_p12_11103.yaml"
)
DEFAULT_RANDOM_SURROGATE = Path(
    "outputs/diagnostics/"
    "attack_rank_bucket_cem_scratch4epoch_surrogate_tiny_valbest_11103/"
    "random_nz_surrogate_scratch4epoch.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/diagnostics/"
    "attack_rank_bucket_cem_scratch4epoch_valbest_surrogate_tailboosted_p12_11103"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a report-only diagnostic for the tail-boosted scratch4epoch "
            "RankBucket-CEM run. This reads existing artifacts and does not retrain."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target-item", type=int, default=TARGET_ITEM)
    parser.add_argument(
        "--cem-metrics-path",
        default=None,
        help=(
            "Existing tail-boosted CEM victim metrics.json. If omitted, the latest "
            "run_group under the config experiment name is used."
        ),
    )
    parser.add_argument("--cem-trace-path", default=None)
    parser.add_argument(
        "--random-victim-metrics",
        default=str(DEFAULT_RANDOM_VICTIM_METRICS),
    )
    parser.add_argument(
        "--random-surrogate-path",
        default=str(DEFAULT_RANDOM_SURROGATE),
        help="Existing Random-NZ scratch4 surrogate artifact from a prior diagnosis.",
    )
    parser.add_argument(
        "--random-position-metadata",
        default=str(DEFAULT_RANDOM_POSITION_METADATA),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    target_item = int(args.target_item)
    _validate_diagnostic_config(config, target_item=target_item)

    cem_metrics_path = _resolve_tailboosted_cem_metrics_path(
        config,
        explicit_path=args.cem_metrics_path,
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
    cem_dir = cem_trace_path.parent
    raw_trace_rows = _read_jsonl(cem_trace_path)
    cem_trace_rows = [_normalize_trace_row_with_position(row) for row in raw_trace_rows]
    cem_best_surrogate = _select_cem_best_surrogate(cem_trace_rows)

    random_position_metadata_path = _resolve_random_position_metadata_path(
        args.random_position_metadata,
        random_victim_metrics_path=Path(args.random_victim_metrics),
    )
    random_surrogate = _load_json_object(Path(args.random_surrogate_path))
    _validate_cached_random_surrogate_replay(
        random_surrogate,
        expected_surrogate_train_seed=_target_level_surrogate_train_seed(
            config,
            target_item=target_item,
        ),
        expected_random_position_metadata_path=random_position_metadata_path,
    )
    _assert_surrogate_train_seeds_match(cem_best_surrogate, random_surrogate)

    cem_victim = _victim_payload(cem_metrics_path)
    random_victim = _victim_payload(Path(args.random_victim_metrics))
    comparison = _build_comparison(
        cem_best_surrogate=cem_best_surrogate,
        random_surrogate=random_surrogate,
        cem_victim=cem_victim,
        random_victim=random_victim,
    )
    comparison["warning"] = _report_warning()

    best_policy = _load_optional_json_object(cem_dir / "cem_best_policy.json")
    final_position_summary = _load_optional_json_object(cem_dir / "final_position_summary.json")
    cem_state_history = _load_optional_json_object(cem_dir / "cem_state_history.json")
    availability_summary = _load_optional_json_object(cem_dir / "availability_summary.json")
    run_metadata = _load_optional_json_object(cem_dir / "run_metadata.json")

    candidate_trace_summary = _candidate_trace_summary(cem_trace_rows)
    position_distribution = _position_distribution_payload(
        cem_best_surrogate=cem_best_surrogate,
        final_position_summary=final_position_summary,
        best_policy=best_policy,
        random_surrogate=random_surrogate,
        availability_summary=availability_summary,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cem_candidate_trace_summary.jsonl", _compact_trace_rows(cem_trace_rows))
    save_json(cem_best_surrogate, output_dir / "cem_selected_surrogate.json")
    save_json(random_surrogate, output_dir / "random_nz_surrogate_scratch4epoch.json")
    save_json(candidate_trace_summary, output_dir / "candidate_trace_summary.json")
    save_json(position_distribution, output_dir / "position_distribution.json")
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
        "report_type": "tailboosted_cem_report_only",
        "config_path": str(config_path),
        "cem_metrics_path": str(cem_metrics_path),
        "cem_trace_path": str(cem_trace_path),
        "cem_artifact_dir": str(cem_dir),
        "random_victim_metrics_path": str(args.random_victim_metrics),
        "random_surrogate_path": str(args.random_surrogate_path),
        "random_position_metadata_path": str(random_position_metadata_path),
        "output_dir": str(output_dir),
        "warning": _report_warning(),
        "reuse_safety_checks": safety,
        "run_metadata_path": str(cem_dir / "run_metadata.json"),
        "artifacts": {
            "report": str(output_dir / "report.md"),
            "candidate_trace_summary": str(output_dir / "candidate_trace_summary.json"),
            "candidate_trace_rows": str(output_dir / "cem_candidate_trace_summary.jsonl"),
            "position_distribution": str(output_dir / "position_distribution.json"),
            "victim_comparison": str(output_dir / "victim_comparison.json"),
            "alignment_summary": str(output_dir / "alignment_summary.json"),
            "cem_selected_surrogate": str(output_dir / "cem_selected_surrogate.json"),
            "random_nz_surrogate": str(output_dir / "random_nz_surrogate_scratch4epoch.json"),
        },
    }
    save_json(manifest, output_dir / "manifest.json")
    (output_dir / "report.md").write_text(
        _render_tailboosted_report(
            target_item=target_item,
            config=config,
            cem_metrics_path=cem_metrics_path,
            cem_trace_path=cem_trace_path,
            cem_best_surrogate=cem_best_surrogate,
            random_surrogate=random_surrogate,
            cem_victim=cem_victim,
            random_victim=random_victim,
            comparison=comparison,
            candidate_trace_summary=candidate_trace_summary,
            position_distribution=position_distribution,
            best_policy=best_policy,
            cem_state_history=cem_state_history,
            run_metadata=run_metadata,
            safety=safety,
        ),
        encoding="utf-8",
    )
    print(f"Wrote tailboosted CEM report artifacts to {output_dir}")


def _resolve_tailboosted_cem_metrics_path(
    config: Config,
    *,
    explicit_path: str | None,
    target_item: int,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"CEM metrics path does not exist: {path}")
        return path
    run_root = (
        Path(config.artifacts.root)
        / str(config.artifacts.runs_dir)
        / str(config.data.dataset_name)
        / str(config.experiment.name)
    )
    matches = sorted(
        run_root.glob(f"run_group_*/targets/{int(target_item)}/victims/srgnn/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-discover tailboosted CEM metrics.json. Provide "
            f"--cem-metrics-path. Searched under: {run_root}"
        )
    return matches[0]


def _normalize_trace_row_with_position(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_cem_trace_row(row)
    for key in (
        "position_summary",
        "pi_g2",
        "pi_g3",
        "logits_g2",
        "logits_g3",
    ):
        if key in row:
            normalized[key] = row[key]
    return normalized


def _candidate_trace_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("iteration", 0)),
            int(row.get("candidate_id_in_iteration", 0)),
        ),
    )
    per_iteration = []
    for iteration in sorted({int(row["iteration"]) for row in sorted_rows}):
        group = [row for row in sorted_rows if int(row["iteration"]) == iteration]
        best_normalized = max(
            group,
            key=lambda row: (
                float(row.get("normalized_reward") or row.get("raw_lowk") or 0.0),
                -int(row.get("candidate_id_in_iteration", 0)),
            ),
        )
        best_raw = max(
            group,
            key=lambda row: (
                float(row.get("raw_lowk") or 0.0),
                -int(row.get("candidate_id_in_iteration", 0)),
            ),
        )
        per_iteration.append(
            {
                "iteration": int(iteration),
                "population": int(len(group)),
                "elite_count": int(
                    sum(1 for row in group if bool(row.get("selected_as_iteration_elite")))
                ),
                "elite_global_candidate_ids": [
                    int(row["global_candidate_id"])
                    for row in group
                    if bool(row.get("selected_as_iteration_elite"))
                ],
                "best_by_normalized_reward": _candidate_summary(best_normalized),
                "best_by_raw_lowk": _candidate_summary(best_raw),
            }
        )
    return {
        "candidate_count": int(len(sorted_rows)),
        "global_best": _candidate_summary(_select_cem_best_surrogate(sorted_rows)),
        "per_iteration": per_iteration,
        "top_by_raw_lowk": [
            _candidate_summary(row)
            for row in sorted(
                sorted_rows,
                key=lambda row: float(row.get("raw_lowk") or 0.0),
                reverse=True,
            )[:10]
        ],
        "top_by_global_normalized_lowk": [
            _candidate_summary(row)
            for row in sorted(
                sorted_rows,
                key=lambda row: float(row.get("global_normalized_lowk_reward") or -1.0),
                reverse=True,
            )[:10]
        ],
    }


def _candidate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    position_summary = row.get("position_summary")
    if not isinstance(position_summary, Mapping):
        position_summary = {}
    return {
        "iteration": int(row.get("iteration", -1)),
        "candidate_id_in_iteration": int(row.get("candidate_id_in_iteration", -1)),
        "global_candidate_id": int(row.get("global_candidate_id", -1)),
        "selected_as_iteration_elite": bool(row.get("selected_as_iteration_elite", False)),
        "selected_as_global_best": bool(row.get("selected_as_global_best", False)),
        "raw_lowk": _optional_float(row.get("raw_lowk")),
        "normalized_reward": _optional_float(row.get("normalized_reward")),
        "iteration_normalized_lowk_reward": _optional_float(
            row.get("iteration_normalized_lowk_reward")
        ),
        "global_normalized_lowk_reward": _optional_float(
            row.get("global_normalized_lowk_reward")
        ),
        "targeted_mrr@10": _optional_float(row.get("targeted_mrr@10")),
        "targeted_mrr@20": _optional_float(row.get("targeted_mrr@20")),
        "targeted_recall@10": _optional_float(row.get("targeted_recall@10")),
        "targeted_recall@20": _optional_float(row.get("targeted_recall@20")),
        "selected_checkpoint_epoch": _optional_int(row.get("selected_checkpoint_epoch")),
        "valid_ground_truth_mrr@20": _optional_float(row.get("valid_ground_truth_mrr@20")),
        "valid_ground_truth_recall@20": _optional_float(
            row.get("valid_ground_truth_recall@20")
        ),
        "position_summary": {
            key: position_summary.get(key)
            for key in (
                "pos1_pct",
                "pos2_pct",
                "pos3_pct",
                "pos4_pos5_pct",
                "pos6plus_pct",
                "rank1_pct",
                "rank2_pct",
                "tail_pct",
                "mean_absolute_position",
                "median_absolute_position",
                "unique_selected_position_count",
            )
            if key in position_summary
        },
    }


def _compact_trace_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_candidate_summary(row) for row in rows]


def _position_distribution_payload(
    *,
    cem_best_surrogate: Mapping[str, Any],
    final_position_summary: Mapping[str, Any],
    best_policy: Mapping[str, Any],
    random_surrogate: Mapping[str, Any],
    availability_summary: Mapping[str, Any],
) -> dict[str, Any]:
    random_selected = random_surrogate.get("selected_position_summary")
    if not isinstance(random_selected, Mapping):
        random_selected = {}
    return {
        "cem_best_candidate_position_summary": cem_best_surrogate.get("position_summary"),
        "cem_final_position_summary": dict(final_position_summary),
        "cem_best_policy": {
            "pi_g2": best_policy.get("pi_g2"),
            "pi_g3": best_policy.get("pi_g3"),
            "best_iteration": best_policy.get("best_iteration"),
            "best_candidate_id": best_policy.get("best_candidate_id"),
            "final_selected_global_candidate_id": best_policy.get(
                "final_selected_global_candidate_id"
            ),
        },
        "random_nz_position_summary": {
            **dict(random_selected),
            "absolute_position_bands": _absolute_position_band_summary(random_selected),
        },
        "availability_summary": dict(availability_summary),
    }


def _absolute_position_band_summary(summary: Mapping[str, Any]) -> dict[str, float | int]:
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        return {}
    normalized_counts = {int(position): int(count) for position, count in counts.items()}
    total = int(summary.get("total", sum(normalized_counts.values())))
    if total <= 0:
        return {
            "total": 0,
            "pos1_pct": 0.0,
            "pos2_pct": 0.0,
            "pos3_pct": 0.0,
            "pos4_pos5_pct": 0.0,
            "pos6plus_pct": 0.0,
        }
    return {
        "total": int(total),
        "pos1_pct": _pct(normalized_counts.get(1, 0), total),
        "pos2_pct": _pct(normalized_counts.get(2, 0), total),
        "pos3_pct": _pct(normalized_counts.get(3, 0), total),
        "pos4_pos5_pct": _pct(
            normalized_counts.get(4, 0) + normalized_counts.get(5, 0),
            total,
        ),
        "pos6plus_pct": _pct(
            sum(count for position, count in normalized_counts.items() if position >= 6),
            total,
        ),
    }


def _pct(count: int | float, total: int | float) -> float:
    return 100.0 * float(count) / float(total) if float(total) else 0.0


def _render_tailboosted_report(
    *,
    target_item: int,
    config: Config,
    cem_metrics_path: Path,
    cem_trace_path: Path,
    cem_best_surrogate: Mapping[str, Any],
    random_surrogate: Mapping[str, Any],
    cem_victim: Mapping[str, Any],
    random_victim: Mapping[str, Any],
    comparison: Mapping[str, Any],
    candidate_trace_summary: Mapping[str, Any],
    position_distribution: Mapping[str, Any],
    best_policy: Mapping[str, Any],
    cem_state_history: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> str:
    del run_metadata
    lines = [
        "# Tail-Boosted RankBucket-CEM Report",
        "",
        f"Target item: `{int(target_item)}`",
        f"Experiment: `{config.experiment.name}`",
        f"CEM metrics: `{cem_metrics_path}`",
        f"CEM trace: `{cem_trace_path}`",
        "",
        f"Warning: {_report_warning()}",
        "",
        "## CEM Setup",
        "",
    ]
    hyper = best_policy.get("cem_hyperparameters") if isinstance(best_policy, Mapping) else {}
    if not isinstance(hyper, Mapping):
        hyper = {}
    lines.extend(
        [
            f"- population_per_iteration: `{hyper.get('population_per_iteration')}`",
            f"- candidate_count: `{hyper.get('candidate_count')}`",
            f"- cem_init_mode: `{hyper.get('cem_init_mode')}`",
            f"- g2_initial_pi: `{hyper.get('g2_initial_pi')}`",
            f"- g3_initial_pi: `{hyper.get('g3_initial_pi')}`",
            f"- surrogate_evaluator: `{hyper.get('surrogate_evaluator')}`",
            f"- reward_metric: `{hyper.get('reward_metric')}`",
            "",
            "## Surrogate Validation",
            "",
            _metric_row("CEM selected", cem_best_surrogate),
            _metric_row("Random-NZ reused", random_surrogate),
            _delta_line("surrogate", cem_best_surrogate, random_surrogate),
            "",
            "## Victim Test",
            "",
            _metric_row("CEM selected", cem_victim),
            _metric_row("Random-NZ", random_victim),
            _delta_line("victim", cem_victim, random_victim),
            "",
            "## Alignment",
            "",
            f"- surrogate_winner_by_raw_lowk: `{comparison['surrogate_winner_by_raw_lowk']}`",
            f"- victim_winner_by_raw_lowk: `{comparison['victim_winner_by_raw_lowk']}`",
            f"- alignment_case: `{comparison['alignment_case']}`",
            f"- surrogate_delta_raw_lowk_cem_minus_random: `{comparison['surrogate_delta_raw_lowk_cem_minus_random']:.9f}`",
            f"- victim_delta_raw_lowk_cem_minus_random: `{comparison['victim_delta_raw_lowk_cem_minus_random']:.9f}`",
            "",
            "## Selected Candidate",
            "",
            f"- iteration: `{cem_best_surrogate.get('iteration')}`",
            f"- candidate_id_in_iteration: `{cem_best_surrogate.get('candidate_id_in_iteration')}`",
            f"- global_candidate_id: `{cem_best_surrogate.get('global_candidate_id')}`",
            f"- selected_checkpoint_epoch: `{cem_best_surrogate.get('selected_checkpoint_epoch')}`",
            f"- global_normalized_lowk_reward: `{cem_best_surrogate.get('global_normalized_lowk_reward')}`",
            f"- best_policy pi_g2: `{best_policy.get('pi_g2')}`",
            f"- best_policy pi_g3: `{best_policy.get('pi_g3')}`",
            "",
            "## Position Distribution",
            "",
            _position_row(
                "CEM final",
                position_distribution.get("cem_final_position_summary", {}),
            ),
            _position_row(
                "CEM selected candidate",
                position_distribution.get("cem_best_candidate_position_summary", {}),
            ),
            _position_row(
                "Random-NZ absolute bands",
                position_distribution.get("random_nz_position_summary", {}).get(
                    "absolute_position_bands",
                    {},
                ),
            ),
            "",
            "## Iteration Summary",
            "",
            "| iteration | population | elite | best global | best norm | best raw_lowk | pos1% | pos2% | tail% |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in candidate_trace_summary.get("per_iteration", []):
        best = row["best_by_normalized_reward"]
        pos = best.get("position_summary", {})
        lines.append(
            "| {it} | {pop} | {elite} | {gid} | {norm:.9f} | {raw:.9f} | {pos1:.2f} | {pos2:.2f} | {tail:.2f} |".format(
                it=int(row["iteration"]),
                pop=int(row["population"]),
                elite=int(row["elite_count"]),
                gid=int(best["global_candidate_id"]),
                norm=float(best.get("normalized_reward") or 0.0),
                raw=float(best.get("raw_lowk") or 0.0),
                pos1=float(pos.get("pos1_pct") or 0.0),
                pos2=float(pos.get("pos2_pct") or 0.0),
                tail=float(pos.get("tail_pct") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## CEM State",
            "",
            f"- state_history_length: `{len(cem_state_history.get('history', [])) if isinstance(cem_state_history, Mapping) else 0}`",
            f"- shared_surrogate_train_seed: `{cem_state_history.get('shared_surrogate_train_seed') if isinstance(cem_state_history, Mapping) else None}`",
            "",
            "## Safety Checks",
            "",
        ]
    )
    for key, value in safety.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _metric_row(label: str, payload: Mapping[str, Any]) -> str:
    return (
        f"- {label}: raw_lowk={float(payload['raw_lowk']):.9f}, "
        f"mrr10={float(payload['targeted_mrr@10']):.9f}, "
        f"mrr20={float(payload['targeted_mrr@20']):.9f}, "
        f"recall10={float(payload['targeted_recall@10']):.9f}, "
        f"recall20={float(payload['targeted_recall@20']):.9f}, "
        f"selected_epoch={payload.get('selected_checkpoint_epoch')}"
    )


def _delta_line(label: str, cem: Mapping[str, Any], random_payload: Mapping[str, Any]) -> str:
    return (
        f"- {label}_delta_cem_minus_random: "
        f"raw_lowk={float(cem['raw_lowk']) - float(random_payload['raw_lowk']):.9f}, "
        f"mrr10={float(cem['targeted_mrr@10']) - float(random_payload['targeted_mrr@10']):.9f}, "
        f"mrr20={float(cem['targeted_mrr@20']) - float(random_payload['targeted_mrr@20']):.9f}, "
        f"recall10={float(cem['targeted_recall@10']) - float(random_payload['targeted_recall@10']):.9f}, "
        f"recall20={float(cem['targeted_recall@20']) - float(random_payload['targeted_recall@20']):.9f}"
    )


def _position_row(label: str, payload: Mapping[str, Any]) -> str:
    return (
        f"- {label}: pos1={float(payload.get('pos1_pct') or 0.0):.2f}%, "
        f"pos2={float(payload.get('pos2_pct') or 0.0):.2f}%, "
        f"pos3={float(payload.get('pos3_pct') or 0.0):.2f}%, "
        f"pos4-5={float(payload.get('pos4_pos5_pct') or 0.0):.2f}%, "
        f"pos6+={float(payload.get('pos6plus_pct') or 0.0):.2f}%, "
        f"tail={float(payload.get('tail_pct') or 0.0):.2f}%"
    )


def _load_optional_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _load_json_object(path)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _report_warning() -> str:
    return (
        "report-only assembly from existing artifacts; Random-NZ surrogate metrics "
        "are reused from the prior scratch4 diagnosis and no model is retrained."
    )


__all__ = [
    "_absolute_position_band_summary",
    "_candidate_trace_summary",
    "_resolve_tailboosted_cem_metrics_path",
]


if __name__ == "__main__":
    main()
