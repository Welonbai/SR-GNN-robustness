from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.pts.cem import PTSCEMCandidateResult, PTSCEMResult


def write_pts_cem_artifacts(
    *,
    result: PTSCEMResult,
    output_dir: Path,
    save_top_candidate_sessions: bool = True,
    save_per_session_records: bool = True,
    write_candidate_epoch_metrics: bool = True,
    write_epoch_reward_ranking_summary: bool = False,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    trace_path = root / "pts_cem_trace.jsonl"
    policy_history_path = root / "pts_policy_history.json"
    best_policy_path = root / "pts_best_policy.json"
    final_policy_path = root / "pts_final_policy.json"
    top_candidates_path = root / "pts_top_candidates.json"
    top_candidate_policies_path = root / "pts_top_candidate_policies.json"
    ranking_summary_json_path = root / "pts_epoch_reward_ranking_summary.json"
    ranking_summary_csv_path = root / "pts_epoch_reward_ranking_summary.csv"

    all_candidates = _all_candidates(result)
    _write_jsonl(
        trace_path,
        [
            _candidate_trace_row(
                candidate,
                write_candidate_epoch_metrics=write_candidate_epoch_metrics,
            )
            for candidate in all_candidates
        ],
    )
    save_json(_to_jsonable(result.policy_history), policy_history_path)
    save_json(
        _best_policy_payload(
            result.best_candidate,
            write_candidate_epoch_metrics=write_candidate_epoch_metrics,
        ),
        best_policy_path,
    )
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
        metadata = _top_candidate_metadata(
            candidate,
            rank=rank,
            write_candidate_epoch_metrics=write_candidate_epoch_metrics,
        )
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
                **_candidate_seed_alignment_payload(candidate),
                **_candidate_surrogate_retrain_payload(candidate),
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
    if write_epoch_reward_ranking_summary and _result_has_epoch_diagnostics(result):
        summary = build_epoch_reward_ranking_summary(result)
        save_json(_to_jsonable(summary), ranking_summary_json_path)
        _write_ranking_summary_csv(ranking_summary_csv_path, summary)
        artifact_paths["pts_epoch_reward_ranking_summary_json"] = str(
            ranking_summary_json_path
        )
        artifact_paths["pts_epoch_reward_ranking_summary_csv"] = str(
            ranking_summary_csv_path
        )
    return artifact_paths


def _all_candidates(result: PTSCEMResult) -> list[PTSCEMCandidateResult]:
    candidates: list[PTSCEMCandidateResult] = []
    for iteration_result in result.iteration_results:
        candidates.extend(iteration_result.candidates)
    return candidates


def _candidate_trace_row(
    candidate: PTSCEMCandidateResult,
    *,
    write_candidate_epoch_metrics: bool,
) -> dict[str, Any]:
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
            **_candidate_epoch_reward_payload(
                candidate,
                write_candidate_epoch_metrics=write_candidate_epoch_metrics,
            ),
            **_candidate_seed_alignment_payload(candidate),
            **_candidate_surrogate_retrain_payload(candidate),
            **_candidate_sampling_payload(candidate),
        }
    )


def _best_policy_payload(
    candidate: PTSCEMCandidateResult,
    *,
    write_candidate_epoch_metrics: bool,
) -> dict[str, Any]:
    return _to_jsonable(
        {
            "candidate_key": candidate.candidate_key,
            "iteration": int(candidate.iteration),
            "candidate_id": int(candidate.candidate_id),
            "candidate_seed": int(candidate.candidate_seed),
            "reward": float(candidate.reward),
            "reward_metrics": dict(candidate.reward_metrics),
            "selected_as_global_best": bool(candidate.selected_as_global_best),
            **_candidate_epoch_reward_payload(
                candidate,
                write_candidate_epoch_metrics=write_candidate_epoch_metrics,
            ),
            **_candidate_seed_alignment_payload(candidate),
            **_candidate_surrogate_retrain_payload(candidate),
            **_candidate_sampling_payload(candidate),
            "policy": candidate.policy.to_dict(),
        }
    )


def _policy_payload(policy: Any) -> dict[str, Any]:
    return _to_jsonable(policy.to_dict())


def _top_candidate_metadata(
    candidate: PTSCEMCandidateResult,
    *,
    rank: int,
    write_candidate_epoch_metrics: bool,
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
            **_candidate_epoch_reward_payload(
                candidate,
                write_candidate_epoch_metrics=write_candidate_epoch_metrics,
            ),
            **_candidate_seed_alignment_payload(candidate),
            **_candidate_surrogate_retrain_payload(candidate),
            **_candidate_sampling_payload(candidate),
            "policy": candidate.policy.to_dict(),
        }
    )


def _candidate_epoch_reward_payload(
    candidate: PTSCEMCandidateResult,
    *,
    write_candidate_epoch_metrics: bool,
) -> dict[str, Any]:
    if not write_candidate_epoch_metrics:
        return {}
    diagnostics = candidate.epoch_reward_diagnostics
    if diagnostics is None:
        return {}
    return {"epoch_reward_diagnostics": dict(diagnostics)}


_SEED_ALIGNMENT_METADATA_KEYS = (
    "target_item",
    "pts_cem_surrogate_seed_alignment_mode",
    "pts_cem_surrogate_seed_alignment_target_victim_name",
    "configured_surrogate_train_seed",
    "configured_victim_train_seed",
    "resolved_surrogate_effective_seed",
    "resolved_victim_effective_seed",
    "surrogate_victim_seed_aligned",
)

_SURROGATE_RETRAIN_METADATA_KEYS = (
    "pts_cem_surrogate_retrain_checkpoint_protocol",
    "pts_cem_surrogate_retrain_validation_enabled",
    "pts_cem_surrogate_retrain_reward_checkpoint",
    "pts_cem_surrogate_retrain_identity_neutral",
    "pts_cem_surrogate_retrain_identity_note",
    "selected_checkpoint_epoch",
    "selected_checkpoint_protocol",
    "selected_checkpoint_source",
    "selected_checkpoint_metric",
    "validation_best_metrics_recorded",
    "official_reward_checkpoint_epoch",
)


def _candidate_seed_alignment_payload(
    candidate: PTSCEMCandidateResult,
) -> dict[str, Any]:
    metadata = dict(candidate.evaluator_metadata)
    return {
        key: metadata[key]
        for key in _SEED_ALIGNMENT_METADATA_KEYS
        if key in metadata
    }


def _candidate_surrogate_retrain_payload(
    candidate: PTSCEMCandidateResult,
) -> dict[str, Any]:
    metadata = dict(candidate.evaluator_metadata)
    return {
        key: metadata[key]
        for key in _SURROGATE_RETRAIN_METADATA_KEYS
        if key in metadata
    }


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
        "method",
        "parameterization",
        "parameter_names",
        "parameter_vector",
        "theta",
        "policy_vector",
        "length_feature",
        "cem_init",
        "cem_update",
        "direct_action_policy_payload",
        "direct_action_context_stats",
        "direct_action_action_summary",
        "search_distribution_mean",
        "search_distribution_std",
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
    if hasattr(value, "to_dict") and callable(value.to_dict):
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


def _result_has_epoch_diagnostics(result: PTSCEMResult) -> bool:
    return any(
        candidate.epoch_reward_diagnostics is not None
        for candidate in _all_candidates(result)
    )


def build_epoch_reward_ranking_summary(result: PTSCEMResult) -> dict[str, Any]:
    all_candidates = _all_candidates(result)
    first_diagnostics = next(
        (
            candidate.epoch_reward_diagnostics
            for candidate in all_candidates
            if candidate.epoch_reward_diagnostics is not None
        ),
        None,
    )
    if first_diagnostics is None:
        return {"enabled": False, "by_iteration": {}, "global": {}}
    diagnostic_epochs = [
        int(epoch)
        for epoch in first_diagnostics.get("diagnostic_epochs", [])
        if _is_int_like(epoch)
    ]
    if not diagnostic_epochs:
        diagnostic_epochs = _available_non_official_epochs(first_diagnostics)
    payload: dict[str, Any] = {
        "enabled": bool(first_diagnostics.get("enabled", True)),
        "official_reward_source": first_diagnostics.get("official_reward_source"),
        "training_budget_epoch": first_diagnostics.get("training_budget_epoch"),
        "selected_checkpoint_epoch_note": (
            "selected_checkpoint_epoch is per candidate and appears in candidate "
            "epoch_reward_diagnostics."
        ),
        "epoch_diagnostic_checkpoint_mode": first_diagnostics.get(
            "epoch_diagnostic_checkpoint_mode"
        ),
        "diagnostic_epochs": diagnostic_epochs,
        "by_iteration": {},
        "global": _ranking_scope_summary(
            all_candidates,
            diagnostic_epochs=diagnostic_epochs,
            elite_k=sum(int(iteration.elite_count) for iteration in result.iteration_results),
        ),
    }
    for iteration in result.iteration_results:
        payload["by_iteration"][str(iteration.iteration)] = _ranking_scope_summary(
            iteration.candidates,
            diagnostic_epochs=diagnostic_epochs,
            elite_k=int(iteration.elite_count),
            population_size=int(iteration.population_size),
        )
    return payload


def _available_non_official_epochs(diagnostics: Mapping[str, Any]) -> list[int]:
    rewards_by_epoch = diagnostics.get("rewards_by_epoch", {})
    training_budget_epoch = diagnostics.get("training_budget_epoch")
    if not isinstance(rewards_by_epoch, Mapping):
        return []
    epochs: list[int] = []
    for raw_epoch in rewards_by_epoch:
        if not _is_int_like(raw_epoch):
            continue
        epoch = int(raw_epoch)
        if training_budget_epoch is not None and epoch == int(training_budget_epoch):
            continue
        epochs.append(epoch)
    return sorted(epochs)


def _ranking_scope_summary(
    candidates: Sequence[PTSCEMCandidateResult],
    *,
    diagnostic_epochs: Sequence[int],
    elite_k: int,
    population_size: int | None = None,
) -> dict[str, Any]:
    official_ranking = _rank_by_score(
        candidates,
        score_fn=lambda candidate: float(candidate.reward),
    )
    official_ranking_keys = [candidate.candidate_key for candidate in official_ranking]
    official_elite_keys = official_ranking_keys[: int(elite_k)]
    summary: dict[str, Any] = {
        "population_size": int(population_size if population_size is not None else len(candidates)),
        "elite_k": int(elite_k),
        "candidate_count": int(len(candidates)),
        "official_ranking_candidate_keys": official_ranking_keys,
        "official_elite_candidate_keys": official_elite_keys,
        "official_best_candidate_key": (
            None if not official_ranking else official_ranking[0].candidate_key
        ),
    }
    official_scores = {
        candidate.candidate_key: float(candidate.reward)
        for candidate in candidates
    }
    official_ranks = _rank_positions(official_ranking)
    for epoch in diagnostic_epochs:
        epoch_scores = {
            candidate.candidate_key: score
            for candidate in candidates
            for score in [_epoch_reward_value(candidate, int(epoch))]
            if score is not None
        }
        epoch_candidates = [
            candidate
            for candidate in candidates
            if candidate.candidate_key in epoch_scores
        ]
        epoch_ranking = _rank_by_score(
            epoch_candidates,
            score_fn=lambda candidate, scores=epoch_scores: float(
                scores[candidate.candidate_key]
            ),
        )
        epoch_ranking_keys = [candidate.candidate_key for candidate in epoch_ranking]
        epoch_elite_keys = epoch_ranking_keys[: int(elite_k)]
        epoch_best_key = None if not epoch_ranking else epoch_ranking[0].candidate_key
        official_best_key = summary["official_best_candidate_key"]
        epoch_ranks = _rank_positions(epoch_ranking)
        common_keys = sorted(set(official_scores) & set(epoch_scores))
        overlap = set(official_elite_keys) & set(epoch_elite_keys)
        summary[f"epoch_{int(epoch)}"] = {
            "epoch": int(epoch),
            "candidate_count_with_epoch_reward": int(len(epoch_scores)),
            "spearman_vs_official": _spearman_correlation(
                {key: official_scores[key] for key in common_keys},
                {key: epoch_scores[key] for key in common_keys},
            ),
            "kendall_tau_vs_official": _kendall_tau(
                {key: official_scores[key] for key in common_keys},
                {key: epoch_scores[key] for key in common_keys},
            ),
            "top1_match": bool(
                epoch_best_key is not None and epoch_best_key == official_best_key
            ),
            "elite_overlap_count": int(len(overlap)),
            "elite_overlap_ratio": (
                None if int(elite_k) <= 0 else float(len(overlap)) / float(elite_k)
            ),
            "official_elite_candidate_keys": official_elite_keys,
            "epoch_elite_candidate_keys": epoch_elite_keys,
            "epoch_best_candidate_key": epoch_best_key,
            "official_best_candidate_key": official_best_key,
            "epoch_best_candidate_keys": (
                [] if epoch_best_key is None else [epoch_best_key]
            ),
            "official_best_candidate_keys": (
                [] if official_best_key is None else [official_best_key]
            ),
            "epoch_best_official_rank": (
                None if epoch_best_key is None else official_ranks.get(epoch_best_key)
            ),
            "official_best_epoch_rank": (
                None if official_best_key is None else epoch_ranks.get(official_best_key)
            ),
            "epoch_ranking_candidate_keys": epoch_ranking_keys,
            "official_ranking_candidate_keys": official_ranking_keys,
        }
    return summary


def _epoch_reward_value(
    candidate: PTSCEMCandidateResult,
    epoch: int,
) -> float | None:
    diagnostics = candidate.epoch_reward_diagnostics
    if not isinstance(diagnostics, Mapping):
        return None
    rewards_by_epoch = diagnostics.get("rewards_by_epoch")
    if not isinstance(rewards_by_epoch, Mapping):
        return None
    epoch_payload = rewards_by_epoch.get(str(int(epoch)))
    if not isinstance(epoch_payload, Mapping):
        return None
    value = epoch_payload.get("target_summary_value")
    if value is None:
        target_summary = diagnostics.get("reward_name")
        if target_summary is not None:
            value = epoch_payload.get(str(target_summary))
    return None if value is None else float(value)


def _rank_by_score(
    candidates: Sequence[PTSCEMCandidateResult],
    *,
    score_fn,
) -> list[PTSCEMCandidateResult]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(score_fn(candidate)),
            int(candidate.iteration),
            int(candidate.candidate_id),
        ),
    )


def _rank_positions(
    ranked_candidates: Sequence[PTSCEMCandidateResult],
) -> dict[str, int]:
    return {
        candidate.candidate_key: int(index)
        for index, candidate in enumerate(ranked_candidates, start=1)
    }


def _spearman_correlation(
    official_scores: Mapping[str, float],
    epoch_scores: Mapping[str, float],
) -> float | None:
    keys = sorted(set(official_scores) & set(epoch_scores))
    if len(keys) < 2:
        return None
    official_ranks = _average_score_ranks(
        {key: float(official_scores[key]) for key in keys}
    )
    epoch_ranks = _average_score_ranks(
        {key: float(epoch_scores[key]) for key in keys}
    )
    return _pearson(
        [official_ranks[key] for key in keys],
        [epoch_ranks[key] for key in keys],
    )


def _average_score_ranks(scores: Mapping[str, float]) -> dict[str, float]:
    sorted_items = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(sorted_items):
        end = index + 1
        score = float(sorted_items[index][1])
        while end < len(sorted_items) and float(sorted_items[end][1]) == score:
            end += 1
        average_rank = (float(index + 1) + float(end)) / 2.0
        for item_index in range(index, end):
            ranks[sorted_items[item_index][0]] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / float(len(left))
    right_mean = sum(right) / float(len(right))
    numerator = sum(
        (float(left_value) - left_mean) * (float(right_value) - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_den = math.sqrt(
        sum((float(left_value) - left_mean) ** 2 for left_value in left)
    )
    right_den = math.sqrt(
        sum((float(right_value) - right_mean) ** 2 for right_value in right)
    )
    denominator = left_den * right_den
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def _kendall_tau(
    official_scores: Mapping[str, float],
    epoch_scores: Mapping[str, float],
) -> float | None:
    keys = sorted(set(official_scores) & set(epoch_scores))
    if len(keys) < 2:
        return None
    concordant = 0
    discordant = 0
    for left_index in range(len(keys)):
        for right_index in range(left_index + 1, len(keys)):
            left_key = keys[left_index]
            right_key = keys[right_index]
            official_delta = float(official_scores[left_key]) - float(
                official_scores[right_key]
            )
            epoch_delta = float(epoch_scores[left_key]) - float(
                epoch_scores[right_key]
            )
            product = official_delta * epoch_delta
            if product > 0.0:
                concordant += 1
            elif product < 0.0:
                discordant += 1
    denominator = concordant + discordant
    if denominator == 0:
        return None
    return float(concordant - discordant) / float(denominator)


def _write_ranking_summary_csv(
    path: Path,
    summary: Mapping[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    global_summary = summary.get("global")
    if isinstance(global_summary, Mapping):
        rows.extend(_ranking_summary_csv_rows("global", None, global_summary))
    by_iteration = summary.get("by_iteration")
    if isinstance(by_iteration, Mapping):
        for iteration, iteration_summary in sorted(
            by_iteration.items(),
            key=lambda item: int(item[0]),
        ):
            if isinstance(iteration_summary, Mapping):
                rows.extend(
                    _ranking_summary_csv_rows(
                        "iteration",
                        int(iteration),
                        iteration_summary,
                    )
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scope",
        "iteration",
        "epoch",
        "population_size",
        "elite_k",
        "candidate_count_with_epoch_reward",
        "spearman_vs_official",
        "kendall_tau_vs_official",
        "top1_match",
        "elite_overlap_count",
        "elite_overlap_ratio",
        "epoch_best_candidate_key",
        "official_best_candidate_key",
        "epoch_best_official_rank",
        "official_best_epoch_rank",
        "official_elite_candidate_keys",
        "epoch_elite_candidate_keys",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ranking_summary_csv_rows(
    scope: str,
    iteration: int | None,
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if not str(key).startswith("epoch_") or not isinstance(value, Mapping):
            continue
        rows.append(
            {
                "scope": scope,
                "iteration": "" if iteration is None else int(iteration),
                "epoch": value.get("epoch"),
                "population_size": payload.get("population_size"),
                "elite_k": payload.get("elite_k"),
                "candidate_count_with_epoch_reward": value.get(
                    "candidate_count_with_epoch_reward"
                ),
                "spearman_vs_official": value.get("spearman_vs_official"),
                "kendall_tau_vs_official": value.get("kendall_tau_vs_official"),
                "top1_match": value.get("top1_match"),
                "elite_overlap_count": value.get("elite_overlap_count"),
                "elite_overlap_ratio": value.get("elite_overlap_ratio"),
                "epoch_best_candidate_key": value.get("epoch_best_candidate_key"),
                "official_best_candidate_key": value.get("official_best_candidate_key"),
                "epoch_best_official_rank": value.get("epoch_best_official_rank"),
                "official_best_epoch_rank": value.get("official_best_epoch_rank"),
                "official_elite_candidate_keys": "|".join(
                    str(item) for item in value.get("official_elite_candidate_keys", [])
                ),
                "epoch_elite_candidate_keys": "|".join(
                    str(item) for item in value.get("epoch_elite_candidate_keys", [])
                ),
            }
        )
    return rows


def _is_int_like(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ["build_epoch_reward_ranking_summary", "write_pts_cem_artifacts"]
