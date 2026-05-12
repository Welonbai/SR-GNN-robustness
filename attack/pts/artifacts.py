from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.pts.cem import PTSCEMCandidateResult, PTSCEMResult
from attack.pts.policy import GroupActionPolicy


def write_pts_cem_artifacts(
    *,
    result: PTSCEMResult,
    output_dir: Path,
    save_top_candidate_sessions: bool = True,
    save_per_session_records: bool = True,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    trace_path = root / "pts_cem_trace.jsonl"
    policy_history_path = root / "pts_policy_history.json"
    best_policy_path = root / "pts_best_policy.json"
    final_policy_path = root / "pts_final_policy.json"
    top_candidates_path = root / "pts_top_candidates.json"
    top_candidate_policies_path = root / "pts_top_candidate_policies.json"

    all_candidates = _all_candidates(result)
    _write_jsonl(trace_path, [_candidate_trace_row(candidate) for candidate in all_candidates])
    save_json(_to_jsonable(result.policy_history), policy_history_path)
    save_json(_best_policy_payload(result.best_candidate), best_policy_path)
    save_json(_policy_payload(result.final_policy), final_policy_path)

    artifact_paths: dict[str, str] = {
        "pts_cem_trace": str(trace_path),
        "pts_policy_history": str(policy_history_path),
        "pts_best_policy": str(best_policy_path),
        "pts_final_policy": str(final_policy_path),
        "pts_top_candidates": str(top_candidates_path),
        "pts_top_candidate_policies": str(top_candidate_policies_path),
    }

    top_rows: list[dict[str, Any]] = []
    top_policy_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(result.top_candidates, start=1):
        rank_dir = root / "top_candidates" / f"rank_{rank}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        policy_path = rank_dir / "policy.json"
        sessions_path = rank_dir / "sessions.json"
        session_records_path = rank_dir / "session_records.jsonl"
        metadata_path = rank_dir / "metadata.json"

        save_json(_policy_payload(candidate.policy), policy_path)
        if save_top_candidate_sessions:
            save_json(_to_jsonable(candidate.final_sessions), sessions_path)
        if save_per_session_records:
            _write_jsonl(session_records_path, candidate.per_session_records)
        metadata = _top_candidate_metadata(candidate, rank=rank)
        save_json(metadata, metadata_path)

        row = {
            **metadata,
            "policy_path": str(policy_path),
            "sessions_path": str(sessions_path) if save_top_candidate_sessions else None,
            "session_records_path": (
                str(session_records_path) if save_per_session_records else None
            ),
            "metadata_path": str(metadata_path),
        }
        top_rows.append(row)
        top_policy_rows.append(
            {
                "rank": int(rank),
                "candidate_key": str(candidate.candidate_key),
                "iteration": int(candidate.iteration),
                "candidate_id": int(candidate.candidate_id),
                "reward": float(candidate.reward),
                "selected_as_global_best": bool(candidate.selected_as_global_best),
                "policy": candidate.policy.to_dict(),
                **_candidate_sampling_payload(candidate),
            }
        )
        artifact_paths[f"top_candidate_rank_{rank}_policy"] = str(policy_path)
        if save_top_candidate_sessions:
            artifact_paths[f"top_candidate_rank_{rank}_sessions"] = str(sessions_path)
        if save_per_session_records:
            artifact_paths[f"top_candidate_rank_{rank}_session_records"] = str(
                session_records_path
            )
        artifact_paths[f"top_candidate_rank_{rank}_metadata"] = str(metadata_path)

    save_json(_to_jsonable({"candidates": top_rows}), top_candidates_path)
    save_json(
        _to_jsonable(
            {
                "top_k": int(len(result.top_candidates)),
                "candidates": top_policy_rows,
            }
        ),
        top_candidate_policies_path,
    )
    return artifact_paths


def _all_candidates(result: PTSCEMResult) -> list[PTSCEMCandidateResult]:
    candidates: list[PTSCEMCandidateResult] = []
    for iteration_result in result.iteration_results:
        candidates.extend(iteration_result.candidates)
    return candidates


def _candidate_trace_row(candidate: PTSCEMCandidateResult) -> dict[str, Any]:
    return _to_jsonable(
        {
            "candidate_key": candidate.candidate_key,
            "iteration": int(candidate.iteration),
            "candidate_id": int(candidate.candidate_id),
            "candidate_seed": int(candidate.candidate_seed),
            "reward": float(candidate.reward),
            "reward_metrics": dict(candidate.reward_metrics),
            "evaluator_metadata": dict(candidate.evaluator_metadata),
            "policy": candidate.policy.to_dict(),
            "construction_summary": dict(candidate.construction_summary),
            "selected_as_elite": bool(candidate.selected_as_elite),
            "selected_as_global_best": bool(candidate.selected_as_global_best),
            **_candidate_sampling_payload(candidate),
        }
    )


def _best_policy_payload(candidate: PTSCEMCandidateResult) -> dict[str, Any]:
    return _to_jsonable(
        {
            "candidate_key": candidate.candidate_key,
            "iteration": int(candidate.iteration),
            "candidate_id": int(candidate.candidate_id),
            "candidate_seed": int(candidate.candidate_seed),
            "reward": float(candidate.reward),
            "reward_metrics": dict(candidate.reward_metrics),
            "selected_as_global_best": bool(candidate.selected_as_global_best),
            **_candidate_sampling_payload(candidate),
            "policy": candidate.policy.to_dict(),
        }
    )


def _policy_payload(policy: GroupActionPolicy) -> dict[str, Any]:
    return _to_jsonable(policy.to_dict())


def _top_candidate_metadata(
    candidate: PTSCEMCandidateResult,
    *,
    rank: int,
) -> dict[str, Any]:
    return _to_jsonable(
        {
            "rank": int(rank),
            "candidate_key": candidate.candidate_key,
            "iteration": int(candidate.iteration),
            "candidate_id": int(candidate.candidate_id),
            "candidate_seed": int(candidate.candidate_seed),
            "reward": float(candidate.reward),
            "reward_metrics": dict(candidate.reward_metrics),
            "evaluator_metadata": dict(candidate.evaluator_metadata),
            "construction_summary": dict(candidate.construction_summary),
            "selected_as_elite": bool(candidate.selected_as_elite),
            "selected_as_global_best": bool(candidate.selected_as_global_best),
            **_candidate_sampling_payload(candidate),
            "policy": candidate.policy.to_dict(),
        }
    )


def _candidate_sampling_payload(candidate: PTSCEMCandidateResult) -> dict[str, Any]:
    sample_metadata = dict(candidate.sample_metadata)
    payload = {
        "sample_origin": candidate.sample_origin,
        "sample_metadata": sample_metadata,
        "parent_iteration": candidate.parent_iteration,
        "parent_candidate_id": candidate.parent_candidate_id,
        "parent_candidate_key": candidate.parent_candidate_key,
        "parent_reward": candidate.parent_reward,
        "parent_rank_among_elites": candidate.parent_rank_among_elites,
    }
    for key in (
        "init_mode",
        "vertex_name",
        "pool_index",
        "distance_to_uniform",
        "min_distance_to_previous_selected",
        "fixed_policy",
        "concentration_scale",
        "local_concentration_scale",
        "sampled_policy_projection_enabled",
        "sampled_policy_min_probability",
        "sampled_policy_max_probability",
    ):
        if key in sample_metadata:
            payload[key] = sample_metadata[key]
    return payload


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_to_jsonable(dict(row)), sort_keys=True))
            handle.write("\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, GroupActionPolicy):
        return _to_jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


__all__ = ["write_pts_cem_artifacts"]
