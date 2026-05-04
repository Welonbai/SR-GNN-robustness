from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.position_stats import save_position_stats
from attack.position_opt.poison_builder import replace_item_at_position


DEFAULT_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_rank_bucket_cem_scratch4epoch_valbest_surrogate_tailboosted_p12_11103.yaml"
)
_LOG_PREFIX = "[rank-bucket-cem-candidate-replay]"


def run_rank_bucket_cem_candidate_replay(
    config: Config,
    *,
    target_item: int,
    global_candidate_id: int,
    source_cem_trace_path: str | Path | None = None,
    source_cem_metrics_path: str | Path | None = None,
    experiment_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """Replay one sampled RankBucket-CEM candidate and run configured victims.

    This runner does not rerun CEM or surrogate scoring. It reads a candidate's
    saved selected_positions from an existing cem_trace.jsonl, reconstructs the
    poisoned train data, then delegates victim training/evaluation to the normal
    pipeline.
    """

    target_item = int(target_item)
    global_candidate_id = int(global_candidate_id)
    source_trace_path = _resolve_source_cem_trace_path(
        config,
        explicit_trace_path=source_cem_trace_path,
        explicit_metrics_path=source_cem_metrics_path,
        target_item=target_item,
    )
    source_row = _load_cem_candidate_row(
        source_trace_path,
        target_item=target_item,
        global_candidate_id=global_candidate_id,
    )
    config = _with_replay_overrides(
        config,
        target_item=target_item,
        global_candidate_id=global_candidate_id,
        experiment_name=experiment_name,
    )
    _validate_replay_config(config)
    selected_positions = _candidate_selected_positions(source_row)
    selected_positions_sha1 = _sha1_json(selected_positions)
    source_trace_sha1 = _sha1_file(source_trace_path)

    attack_identity_context = _candidate_replay_attack_identity_context(
        source_trace_sha1=source_trace_sha1,
        source_row=source_row,
        target_item=target_item,
        global_candidate_id=global_candidate_id,
        selected_positions_sha1=selected_positions_sha1,
    )

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    _validate_source_fake_sessions_match(
        source_trace_path,
        shared_fake_sessions_path=shared.shared_paths["fake_sessions"],
    )
    _validate_selected_positions(
        selected_positions,
        template_sessions=shared.template_sessions,
        target_item=target_item,
        source_trace_path=source_trace_path,
        global_candidate_id=global_candidate_id,
    )

    context = RunContext.from_shared(shared)
    print(
        f"{_LOG_PREFIX} target={target_item} gid={global_candidate_id} "
        f"source_trace={source_trace_path} "
        f"selected_positions={len(selected_positions)} "
        f"positions_sha1={selected_positions_sha1}"
    )

    def build_poisoned(requested_target_item: int) -> TargetPoisonOutput:
        if int(requested_target_item) != target_item:
            raise ValueError(
                "Candidate replay config requested an unexpected target item: "
                f"{requested_target_item}; expected {target_item}."
            )

        poisoned_fake_sessions = [
            replace_item_at_position(session, position, target_item)
            for session, position in zip(shared.template_sessions, selected_positions)
        ]
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            poisoned_fake_sessions,
        )

        replay_root = (
            target_dir(
                config,
                target_item,
                run_type=POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
                attack_identity_context=attack_identity_context,
            )
            / "position_opt"
            / "cem_candidate_replay"
        )
        replay_root.mkdir(parents=True, exist_ok=True)
        optimized_poisoned_sessions_path = replay_root / "optimized_poisoned_sessions.pkl"
        with optimized_poisoned_sessions_path.open("wb") as handle:
            pickle.dump(poisoned_fake_sessions, handle)

        selected_positions_path = replay_root / "selected_positions.json"
        save_json(
            [int(position) for position in selected_positions],
            selected_positions_path,
        )
        selected_positions_jsonl_path = replay_root / "selected_positions.jsonl"
        _write_jsonl(
            selected_positions_jsonl_path,
            [
                {
                    "fake_session_index": int(index),
                    "selected_position": int(position),
                    "target_item": int(target_item),
                    "source": "rank_bucket_cem_candidate_replay",
                }
                for index, position in enumerate(selected_positions)
            ],
        )
        position_stats_path = save_position_stats(
            replay_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=selected_positions,
            run_type=POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
            target_item=target_item,
            note=(
                "Replay of selected_positions from an existing RankBucket-CEM "
                "candidate; CEM and surrogate scoring were not rerun."
            ),
        )
        replay_metadata = _candidate_replay_metadata(
            source_trace_path=source_trace_path,
            source_trace_sha1=source_trace_sha1,
            source_row=source_row,
            target_item=target_item,
            global_candidate_id=global_candidate_id,
            selected_positions=selected_positions,
            selected_positions_sha1=selected_positions_sha1,
            optimized_poisoned_sessions_path=optimized_poisoned_sessions_path,
            selected_positions_path=selected_positions_path,
            selected_positions_jsonl_path=selected_positions_jsonl_path,
            position_stats_path=position_stats_path,
            shared_fake_sessions_path=shared.shared_paths["fake_sessions"],
        )
        replay_metadata_path = replay_root / "replay_metadata.json"
        save_json(replay_metadata, replay_metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "position_opt_method": POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
                "rank_bucket_cem_candidate_replay": replay_metadata,
                "rank_bucket_cem_candidate_replay_metadata_path": str(replay_metadata_path),
                "rank_bucket_cem_candidate_replay_source_trace_path": str(source_trace_path),
                "rank_bucket_cem_candidate_replay_source_trace_sha1": source_trace_sha1,
                "rank_bucket_cem_candidate_replay_global_candidate_id": global_candidate_id,
                "rank_bucket_cem_candidate_replay_selected_positions_sha1": (
                    selected_positions_sha1
                ),
                "position_opt_optimized_poisoned_sessions_path": str(
                    optimized_poisoned_sessions_path
                ),
                "position_opt_selected_positions_path": str(selected_positions_path),
                "position_opt_selected_positions_jsonl_path": str(
                    selected_positions_jsonl_path
                ),
                "position_stats_path": str(position_stats_path),
                "source_surrogate_raw_lowk": _optional_float(
                    source_row.get("absolute_raw_family_lowk_reward")
                    or source_row.get("raw_lowk")
                ),
                "source_surrogate_global_normalized_lowk_reward": _optional_float(
                    source_row.get("global_normalized_lowk_reward")
                ),
            },
        )

    summary = run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )
    print(f"{_LOG_PREFIX} victim replay completed.")
    return summary


def _with_replay_overrides(
    config: Config,
    *,
    target_item: int,
    global_candidate_id: int,
    experiment_name: str | None,
) -> Config:
    if experiment_name is None or not str(experiment_name).strip():
        experiment_name = (
            f"{config.experiment.name}_candidate_replay_gid{int(global_candidate_id)}"
        )
    return replace(
        config,
        experiment=replace(config.experiment, name=str(experiment_name)),
        targets=replace(
            config.targets,
            mode="explicit_list",
            explicit_list=(int(target_item),),
            count=1,
        ),
    )


def _validate_replay_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("RankBucket-CEM candidate replay expects data.poison_train_only=true.")
    if "srgnn" not in config.victims.enabled:
        raise ValueError("RankBucket-CEM candidate replay currently expects srgnn victim enabled.")


def _resolve_source_cem_trace_path(
    config: Config,
    *,
    explicit_trace_path: str | Path | None,
    explicit_metrics_path: str | Path | None,
    target_item: int,
) -> Path:
    if explicit_trace_path is not None:
        path = Path(explicit_trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Source CEM trace path does not exist: {path}")
        return path

    if explicit_metrics_path is not None:
        metrics_path = Path(explicit_metrics_path)
        if not metrics_path.exists():
            raise FileNotFoundError(f"Source CEM metrics path does not exist: {metrics_path}")
        metrics_payload = _load_json_object(metrics_path)
        trace_path = metrics_payload.get("rank_bucket_cem_trace_path")
        if isinstance(trace_path, str) and trace_path.strip():
            path = Path(trace_path)
            if not path.exists():
                raise FileNotFoundError(
                    "Source metrics rank_bucket_cem_trace_path does not exist: "
                    f"{path}"
                )
            return path
        fallback = metrics_path.parent.parent.parent / "position_opt" / "cem" / "cem_trace.jsonl"
        if fallback.exists():
            return fallback
        raise ValueError(
            "Source CEM metrics did not contain rank_bucket_cem_trace_path and "
            f"fallback trace was not found: {fallback}"
        )

    run_root = (
        Path(config.artifacts.root)
        / str(config.artifacts.runs_dir)
        / str(config.data.dataset_name)
        / str(config.experiment.name)
    )
    matches = sorted(
        run_root.glob(f"run_group_*/targets/{int(target_item)}/position_opt/cem/cem_trace.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            "Could not auto-discover source cem_trace.jsonl. Provide "
            f"--source-cem-trace. Searched under: {run_root}"
        )
    return matches[0]


def _load_cem_candidate_row(
    source_trace_path: Path,
    *,
    target_item: int,
    global_candidate_id: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    with source_trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if int(row.get("target_item", target_item)) != int(target_item):
                continue
            if int(row.get("global_candidate_id", -1)) == int(global_candidate_id):
                row["_source_trace_line_number"] = int(line_number)
                matches.append(row)
    if not matches:
        raise ValueError(
            "Could not find candidate in source CEM trace: "
            f"target_item={int(target_item)} global_candidate_id={int(global_candidate_id)} "
            f"path={source_trace_path}"
        )
    if len(matches) > 1:
        raise ValueError(
            "Source CEM trace contains duplicate rows for candidate: "
            f"target_item={int(target_item)} global_candidate_id={int(global_candidate_id)} "
            f"path={source_trace_path}"
        )
    return matches[0]


def _candidate_selected_positions(row: Mapping[str, Any]) -> list[int]:
    raw_positions = row.get("selected_positions")
    if not isinstance(raw_positions, list):
        raise ValueError(
            "CEM candidate row does not contain selected_positions. The source run "
            "must have rank_bucket_cem.save_candidate_selected_positions=true. "
            f"available_keys={sorted(str(key) for key in row.keys())}"
        )
    return [int(position) for position in raw_positions]


def _validate_selected_positions(
    selected_positions: Sequence[int],
    *,
    template_sessions: Sequence[Sequence[int]],
    target_item: int,
    source_trace_path: Path,
    global_candidate_id: int,
) -> None:
    if len(selected_positions) != len(template_sessions):
        raise ValueError(
            "CEM candidate selected_positions length does not match shared fake sessions. "
            f"target_item={int(target_item)} gid={int(global_candidate_id)} "
            f"positions={len(selected_positions)} fake_sessions={len(template_sessions)} "
            f"source_trace={source_trace_path}"
        )
    invalid: list[dict[str, int]] = []
    for index, (session, position) in enumerate(zip(template_sessions, selected_positions)):
        position = int(position)
        if position < 0 or position >= len(session):
            invalid.append(
                {
                    "fake_session_index": int(index),
                    "position": int(position),
                    "session_length": int(len(session)),
                }
            )
            if len(invalid) >= 5:
                break
    if invalid:
        raise ValueError(
            "CEM candidate selected_positions contain invalid positions. "
            f"examples={invalid}"
        )


def _validate_source_fake_sessions_match(
    source_trace_path: Path,
    *,
    shared_fake_sessions_path: Path,
) -> None:
    run_metadata_path = source_trace_path.parent / "run_metadata.json"
    if not run_metadata_path.exists():
        return
    run_metadata = _load_json_object(run_metadata_path)
    source_shared = run_metadata.get("shared_fake_sessions_path")
    if not isinstance(source_shared, str) or not source_shared.strip():
        return
    source_path = Path(source_shared)
    current_path = Path(shared_fake_sessions_path)
    if source_path.exists() and current_path.exists():
        source_identity = str(source_path.resolve())
        current_identity = str(current_path.resolve())
    else:
        source_identity = str(source_path)
        current_identity = str(current_path)
    if source_identity != current_identity:
        raise RuntimeError(
            "Source CEM trace was produced with a different fake_sessions.pkl. "
            "Replay would not reconstruct the original candidate. "
            f"source={source_shared}; current={shared_fake_sessions_path}"
        )


def _candidate_replay_attack_identity_context(
    *,
    source_trace_sha1: str,
    source_row: Mapping[str, Any],
    target_item: int,
    global_candidate_id: int,
    selected_positions_sha1: str,
) -> dict[str, Any]:
    return {
        "position_opt": {
            "candidate_replay": {
                "mode": "rank_bucket_cem_candidate_selected_positions",
                "source_trace_sha1": str(source_trace_sha1),
                "source_trace_line_number": _optional_int(
                    source_row.get("_source_trace_line_number")
                ),
                "target_item": int(target_item),
                "global_candidate_id": int(global_candidate_id),
                "iteration": _optional_int(source_row.get("iteration")),
                "candidate_id_in_iteration": _optional_int(
                    source_row.get("candidate_id_in_iteration")
                    or source_row.get("candidate_id")
                ),
                "selected_positions_sha1": str(selected_positions_sha1),
            },
            "clean_surrogate": {
                "type": "not_used",
                "reason": "candidate_replay_does_not_run_surrogate",
            },
        }
    }


def _candidate_replay_metadata(
    *,
    source_trace_path: Path,
    source_trace_sha1: str,
    source_row: Mapping[str, Any],
    target_item: int,
    global_candidate_id: int,
    selected_positions: Sequence[int],
    selected_positions_sha1: str,
    optimized_poisoned_sessions_path: Path,
    selected_positions_path: Path,
    selected_positions_jsonl_path: Path,
    position_stats_path: Path,
    shared_fake_sessions_path: Path,
) -> dict[str, Any]:
    return {
        "mode": "rank_bucket_cem_candidate_selected_positions",
        "target_item": int(target_item),
        "global_candidate_id": int(global_candidate_id),
        "source_cem_trace_path": str(source_trace_path),
        "source_cem_trace_sha1": str(source_trace_sha1),
        "source_trace_line_number": _optional_int(
            source_row.get("_source_trace_line_number")
        ),
        "source_iteration": _optional_int(source_row.get("iteration")),
        "source_candidate_id_in_iteration": _optional_int(
            source_row.get("candidate_id_in_iteration") or source_row.get("candidate_id")
        ),
        "source_selected_as_iteration_elite": bool(
            source_row.get("selected_as_iteration_elite", False)
        ),
        "source_selected_as_global_best": bool(
            source_row.get("selected_as_global_best", False)
        ),
        "source_surrogate": _source_surrogate_payload(source_row),
        "source_position_summary": source_row.get("position_summary"),
        "selected_positions_count": int(len(selected_positions)),
        "selected_positions_sha1": str(selected_positions_sha1),
        "optimized_poisoned_sessions_path": str(optimized_poisoned_sessions_path),
        "selected_positions_path": str(selected_positions_path),
        "selected_positions_jsonl_path": str(selected_positions_jsonl_path),
        "position_stats_path": str(position_stats_path),
        "shared_fake_sessions_path": str(shared_fake_sessions_path),
        "warning": (
            "Replay-only victim run from existing CEM candidate selected_positions; "
            "CEM and surrogate evaluation were not rerun."
        ),
    }


def _source_surrogate_payload(source_row: Mapping[str, Any]) -> dict[str, Any]:
    target_metrics = source_row.get("target_metrics")
    if not isinstance(target_metrics, Mapping):
        target_metrics = {}
    return {
        "surrogate_evaluator_mode": source_row.get("surrogate_evaluator_mode"),
        "surrogate_train_seed": _optional_int(source_row.get("surrogate_train_seed")),
        "selected_checkpoint_epoch": _optional_int(
            source_row.get("selected_checkpoint_epoch")
        ),
        "stopped_epoch": _optional_int(source_row.get("stopped_epoch")),
        "stop_reason": source_row.get("stop_reason"),
        "valid_ground_truth_mrr@20": _optional_float(
            source_row.get("valid_ground_truth_mrr@20")
        ),
        "valid_ground_truth_recall@20": _optional_float(
            source_row.get("valid_ground_truth_recall@20")
        ),
        "targeted_mrr@10": _optional_float(
            source_row.get("targeted_mrr@10") or target_metrics.get("targeted_mrr@10")
        ),
        "targeted_mrr@20": _optional_float(
            source_row.get("targeted_mrr@20") or target_metrics.get("targeted_mrr@20")
        ),
        "targeted_recall@10": _optional_float(
            source_row.get("targeted_recall@10")
            or target_metrics.get("targeted_recall@10")
        ),
        "targeted_recall@20": _optional_float(
            source_row.get("targeted_recall@20")
            or target_metrics.get("targeted_recall@20")
        ),
        "raw_lowk": _optional_float(
            source_row.get("absolute_raw_family_lowk_reward")
            or source_row.get("raw_lowk")
        ),
        "global_normalized_lowk_reward": _optional_float(
            source_row.get("global_normalized_lowk_reward")
        ),
        "iteration_normalized_lowk_reward": _optional_float(
            source_row.get("iteration_normalized_lowk_reward")
        ),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")
    return output_path


def _sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one saved RankBucket-CEM candidate selected_positions and run "
            "the configured victim evaluation."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--target-item", type=int, required=True)
    parser.add_argument("--global-candidate-id", type=int, required=True)
    parser.add_argument("--source-cem-trace", default=None)
    parser.add_argument("--source-cem-metrics", default=None)
    parser.add_argument(
        "--experiment-name",
        default=None,
        help=(
            "Output experiment name. Defaults to "
            "<config experiment>_candidate_replay_gid<global_candidate_id>."
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    run_rank_bucket_cem_candidate_replay(
        config,
        target_item=int(args.target_item),
        global_candidate_id=int(args.global_candidate_id),
        source_cem_trace_path=args.source_cem_trace,
        source_cem_metrics_path=args.source_cem_metrics,
        experiment_name=args.experiment_name,
        config_path=config_path,
    )


__all__ = [
    "POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE",
    "_candidate_selected_positions",
    "_load_cem_candidate_row",
    "_sha1_json",
    "_validate_selected_positions",
    "run_rank_bucket_cem_candidate_replay",
]


if __name__ == "__main__":
    main()
