from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import Config
from attack.common.paths import (
    POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    shared_attack_dir,
    target_dir,
)
from attack.position_opt.artifacts import resolve_clean_surrogate_checkpoint_path

from .rank_policy import RankBucketSelectionRecord


@dataclass(frozen=True)
class RankBucketCEMArtifactPaths:
    base_dir: Path
    clean_surrogate_checkpoint: Path
    optimized_poisoned_sessions: Path
    availability_summary: Path
    cem_trace: Path
    cem_state_history: Path
    cem_best_policy: Path
    final_selected_positions: Path
    final_position_summary: Path
    run_metadata: Path | None = None


def build_rank_bucket_cem_artifact_paths(
    config: Config,
    *,
    run_type: str = POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    target_item: int | None = None,
    clean_checkpoint_override: str | Path | None = None,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> RankBucketCEMArtifactPaths:
    if target_item is None:
        base_dir = shared_attack_dir(config, run_type=run_type) / "position_opt" / "cem"
    else:
        base_dir = (
            target_dir(
                config,
                target_item,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            )
            / "position_opt"
            / "cem"
        )
    return RankBucketCEMArtifactPaths(
        base_dir=base_dir,
        clean_surrogate_checkpoint=resolve_clean_surrogate_checkpoint_path(
            config,
            run_type=run_type,
            override=clean_checkpoint_override,
        ),
        optimized_poisoned_sessions=base_dir / "optimized_poisoned_sessions.pkl",
        availability_summary=base_dir / "availability_summary.json",
        cem_trace=base_dir / "cem_trace.jsonl",
        cem_state_history=base_dir / "cem_state_history.json",
        cem_best_policy=base_dir / "cem_best_policy.json",
        final_selected_positions=base_dir / "final_selected_positions.jsonl",
        final_position_summary=base_dir / "final_position_summary.json",
        run_metadata=base_dir / "run_metadata.json",
    )


def ensure_rank_bucket_cem_artifact_dirs(
    paths: RankBucketCEMArtifactPaths,
) -> RankBucketCEMArtifactPaths:
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        paths.optimized_poisoned_sessions,
        paths.availability_summary,
        paths.cem_trace,
        paths.cem_state_history,
        paths.cem_best_policy,
        paths.final_selected_positions,
        paths.final_position_summary,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    if paths.run_metadata is not None:
        paths.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    return paths


def save_availability_summary(path: str | Path, payload: Mapping[str, Any]) -> Path:
    save_json(dict(payload), path)
    return Path(path)


def save_cem_state_history(path: str | Path, payload: Mapping[str, Any]) -> Path:
    save_json(dict(payload), path)
    return Path(path)


def save_cem_best_policy(path: str | Path, payload: Mapping[str, Any]) -> Path:
    save_json(dict(payload), path)
    return Path(path)


def save_final_position_summary(path: str | Path, payload: Mapping[str, Any]) -> Path:
    save_json(dict(payload), path)
    return Path(path)


def save_run_metadata(path: str | Path, payload: Mapping[str, Any]) -> Path:
    save_json(dict(payload), path)
    return Path(path)


def save_optimized_poisoned_sessions(
    path: str | Path,
    sessions: Sequence[Sequence[int]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump([list(session) for session in sessions], handle)
    return output_path


def write_selected_positions_jsonl(
    path: str | Path,
    records: Sequence[RankBucketSelectionRecord],
) -> Path:
    return _write_jsonl(
        path,
        [selection_record_to_jsonable(record) for record in records],
    )


def write_cem_trace_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    return _write_jsonl(path, rows)


def selection_record_to_jsonable(
    record: RankBucketSelectionRecord,
) -> dict[str, object]:
    return {
        "fake_session_index": int(record.fake_session_index),
        "session_length": int(record.session_length),
        "candidate_count": int(record.candidate_count),
        "availability_group": str(record.availability_group),
        "candidate_positions": [int(position) for position in record.candidate_positions],
        "selected_position": int(record.selected_position),
        "selected_rank": str(record.selected_rank),
        "selected_rank_index": (
            None
            if record.selected_rank_index is None
            else int(record.selected_rank_index)
        ),
        "policy_probability": float(record.policy_probability),
        "target_item": int(record.target_item),
    }


def _write_jsonl(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")
    return output_path


__all__ = [
    "POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE",
    "RankBucketCEMArtifactPaths",
    "build_rank_bucket_cem_artifact_paths",
    "ensure_rank_bucket_cem_artifact_dirs",
    "save_availability_summary",
    "save_cem_best_policy",
    "save_cem_state_history",
    "save_final_position_summary",
    "save_optimized_poisoned_sessions",
    "save_run_metadata",
    "selection_record_to_jsonable",
    "write_cem_trace_jsonl",
    "write_selected_positions_jsonl",
]
