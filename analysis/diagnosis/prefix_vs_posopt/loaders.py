#!/usr/bin/env python3
"""Artifact loaders for Prefix vs PosOptMVP diagnosis."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHARED_FAKE_SESSIONS = (
    REPO_ROOT
    / "outputs"
    / "shared"
    / "diginetica"
    / "attack"
    / "attack_shared_1c4345bfa3"
    / "fake_sessions.pkl"
)
DEFAULT_POSOPT_RUN_GROUP = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "diginetica"
    / "attack_position_optimization_reward_mvp_ratio1"
    / "run_group_3becc51c46"
)
DEFAULT_PREFIX_RUN_GROUP = (
    REPO_ROOT
    / "outputs"
    / "runs"
    / "diginetica"
    / "attack_prefix_nonzero_when_possible"
    / "run_group_14818d6dd6"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "analysis" / "diagnosis_outputs" / "prefix_vs_posopt"


class DiagnosisError(ValueError):
    """Raised when a required diagnosis input is missing or malformed."""


@dataclass(frozen=True)
class SharedArtifacts:
    """Shared fake-session pool used across attack runs."""

    fake_sessions_path: Path
    fake_sessions: list[list[int]]


@dataclass(frozen=True)
class TargetMethodArtifacts:
    """Target-level method artifacts needed for diagnosis."""

    method: str
    target_item: int
    target_dir: Path
    position_stats_path: Path
    position_stats: dict[str, Any]
    selected_positions_path: Path
    selected_positions: list[dict[str, Any]]
    training_history_path: Path | None = None
    training_history: dict[str, Any] | None = None
    run_metadata_path: Path | None = None
    run_metadata: dict[str, Any] | None = None
    optimized_poisoned_sessions_path: Path | None = None
    optimized_poisoned_sessions: list[list[int]] | None = None


@dataclass(frozen=True)
class VictimRunArtifacts:
    """Victim-level run artifacts used for metric comparison."""

    method: str
    target_item: int
    victim_model: str
    run_dir: Path
    metrics_path: Path
    metrics: dict[str, Any]
    predictions_path: Path | None
    train_history_path: Path
    resolved_config_path: Path


def ensure_existing_file(path: Path, *, label: str) -> Path:
    """Resolve and validate a required file path."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise DiagnosisError(f"Missing required {label}: '{path}'.")
    return resolved


def load_json_file(path: Path, *, label: str) -> dict[str, Any]:
    """Load one JSON file into a mapping."""
    resolved = ensure_existing_file(path, label=label)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DiagnosisError(f"Expected {label} '{resolved}' to contain a JSON object.")
    return payload


def load_pickle_file(path: Path, *, label: str) -> Any:
    """Load one pickle artifact."""
    resolved = ensure_existing_file(path, label=label)
    with resolved.open("rb") as handle:
        return pickle.load(handle)


def repo_relative(path: Path) -> str:
    """Return a stable repo-relative path string when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def optional_float(value: Any) -> float | None:
    """Normalize optional numeric values into Python floats."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosisError(f"Expected a numeric score value, received '{value!r}'.") from exc
    if math.isnan(numeric):
        return None
    return numeric


def require_int(value: Any, *, label: str) -> int:
    """Normalize one integer value."""
    if isinstance(value, bool):
        raise DiagnosisError(f"Expected {label} to be an integer, received bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DiagnosisError(f"Expected {label} to be an integer, received '{value!r}'.") from exc


def normalize_fake_sessions(payload: Any, *, path: Path) -> list[list[int]]:
    """Validate the shared fake-session artifact shape."""
    if not isinstance(payload, list):
        raise DiagnosisError(
            f"Expected shared fake sessions '{path}' to contain a list, found {type(payload).__name__}."
        )
    normalized: list[list[int]] = []
    for index, session in enumerate(payload):
        if not isinstance(session, list):
            raise DiagnosisError(
                f"Expected fake session {index} in '{path}' to be a list, found {type(session).__name__}."
            )
        normalized.append([require_int(item, label=f"fake_sessions[{index}] item") for item in session])
    return normalized


def normalize_prefix_selected_positions(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    """Validate and normalize Prefix per-session position metadata."""
    if not isinstance(payload, list):
        raise DiagnosisError(
            f"Expected Prefix metadata '{path}' to contain a list, found {type(payload).__name__}."
        )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise DiagnosisError(
                f"Expected Prefix metadata item {index} in '{path}' to be a dict, found {type(record).__name__}."
            )
        if "position" not in record:
            raise DiagnosisError(f"Prefix metadata item {index} in '{path}' is missing 'position'.")
        normalized.append(
            {
                "position": require_int(record["position"], label=f"prefix position at index {index}"),
                "score": optional_float(record.get("target_score")),
                "candidate_index": None,
            }
        )
    return normalized


def normalize_posopt_selected_positions(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    """Validate and normalize PosOpt per-session selected positions."""
    if not isinstance(payload, list):
        raise DiagnosisError(
            f"Expected PosOpt selected positions '{path}' to contain a list, found {type(payload).__name__}."
        )
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise DiagnosisError(
                f"Expected PosOpt selected position {index} in '{path}' to be a dict, found {type(record).__name__}."
            )
        if "position" not in record:
            raise DiagnosisError(f"PosOpt selected position {index} in '{path}' is missing 'position'.")
        normalized.append(
            {
                "position": require_int(record["position"], label=f"posopt position at index {index}"),
                "score": optional_float(record.get("score")),
                "candidate_index": (
                    None
                    if record.get("candidate_index") is None
                    else require_int(
                        record["candidate_index"],
                        label=f"posopt candidate_index at index {index}",
                    )
                ),
            }
        )
    return normalized


def load_shared_artifacts(fake_sessions_path: Path) -> SharedArtifacts:
    """Load the shared fake-session pool."""
    resolved = ensure_existing_file(fake_sessions_path, label="shared fake-session pool")
    fake_sessions = normalize_fake_sessions(load_pickle_file(resolved, label="shared fake-session pool"), path=resolved)
    return SharedArtifacts(fake_sessions_path=resolved, fake_sessions=fake_sessions)


def load_prefix_target_artifacts(prefix_root: Path, *, target_item: int) -> TargetMethodArtifacts:
    """Load Prefix target-level artifacts for one target item."""
    root = prefix_root.resolve()
    target_dir = root / "targets" / str(target_item)
    position_stats_path = ensure_existing_file(
        target_dir / "position_stats.json",
        label=f"Prefix position_stats for target {target_item}",
    )
    metadata_path = ensure_existing_file(
        target_dir / "prefix_nonzero_when_possible_metadata.pkl",
        label=f"Prefix metadata for target {target_item}",
    )

    position_stats = load_json_file(position_stats_path, label=f"Prefix position_stats for target {target_item}")
    metadata = normalize_prefix_selected_positions(
        load_pickle_file(metadata_path, label=f"Prefix metadata for target {target_item}"),
        path=metadata_path,
    )
    return TargetMethodArtifacts(
        method="prefix_nonzero_when_possible",
        target_item=target_item,
        target_dir=target_dir.resolve(),
        position_stats_path=position_stats_path,
        position_stats=position_stats,
        selected_positions_path=metadata_path,
        selected_positions=metadata,
    )


def load_posopt_target_artifacts(posopt_root: Path, *, target_item: int) -> TargetMethodArtifacts:
    """Load PosOpt target-level artifacts for one target item."""
    root = posopt_root.resolve()
    target_dir = root / "targets" / str(target_item)
    position_stats_path = ensure_existing_file(
        target_dir / "position_stats.json",
        label=f"PosOpt position_stats for target {target_item}",
    )
    selected_positions_path = ensure_existing_file(
        target_dir / "position_opt" / "selected_positions.json",
        label=f"PosOpt selected_positions for target {target_item}",
    )
    training_history_path = ensure_existing_file(
        target_dir / "position_opt" / "training_history.json",
        label=f"PosOpt training_history for target {target_item}",
    )
    run_metadata_path = ensure_existing_file(
        target_dir / "position_opt" / "run_metadata.json",
        label=f"PosOpt run_metadata for target {target_item}",
    )
    optimized_poisoned_sessions_path = ensure_existing_file(
        target_dir / "position_opt" / "optimized_poisoned_sessions.pkl",
        label=f"PosOpt optimized_poisoned_sessions for target {target_item}",
    )

    position_stats = load_json_file(position_stats_path, label=f"PosOpt position_stats for target {target_item}")
    selected_positions = normalize_posopt_selected_positions(
        json.loads(selected_positions_path.read_text(encoding="utf-8")),
        path=selected_positions_path,
    )
    training_history = load_json_file(
        training_history_path,
        label=f"PosOpt training_history for target {target_item}",
    )
    run_metadata = load_json_file(run_metadata_path, label=f"PosOpt run_metadata for target {target_item}")
    optimized_poisoned_sessions = normalize_fake_sessions(
        load_pickle_file(
            optimized_poisoned_sessions_path,
            label=f"PosOpt optimized_poisoned_sessions for target {target_item}",
        ),
        path=optimized_poisoned_sessions_path,
    )
    return TargetMethodArtifacts(
        method="position_opt_mvp",
        target_item=target_item,
        target_dir=target_dir.resolve(),
        position_stats_path=position_stats_path,
        position_stats=position_stats,
        selected_positions_path=selected_positions_path,
        selected_positions=selected_positions,
        training_history_path=training_history_path,
        training_history=training_history,
        run_metadata_path=run_metadata_path,
        run_metadata=run_metadata,
        optimized_poisoned_sessions_path=optimized_poisoned_sessions_path,
        optimized_poisoned_sessions=optimized_poisoned_sessions,
    )


def load_victim_run_artifacts(
    run_group_root: Path,
    *,
    method: str,
    target_item: int,
    victim_model: str,
) -> VictimRunArtifacts:
    """Load victim-level run artifacts for one method-target-victim cell."""
    root = run_group_root.resolve()
    run_dir = root / "targets" / str(target_item) / "victims" / victim_model
    metrics_path = ensure_existing_file(
        run_dir / "metrics.json",
        label=f"{method} metrics for target {target_item} victim {victim_model}",
    )
    train_history_path = ensure_existing_file(
        run_dir / "train_history.json",
        label=f"{method} victim train_history for target {target_item} victim {victim_model}",
    )
    resolved_config_path = ensure_existing_file(
        run_dir / "resolved_config.json",
        label=f"{method} resolved_config for target {target_item} victim {victim_model}",
    )
    metrics = load_json_file(metrics_path, label=f"{method} metrics for target {target_item} victim {victim_model}")
    predictions_path: Path | None = None
    raw_predictions_path = metrics.get("predictions_path")
    if isinstance(raw_predictions_path, str) and raw_predictions_path.strip():
        candidate = REPO_ROOT / Path(raw_predictions_path)
        if candidate.is_file():
            predictions_path = candidate.resolve()
    else:
        candidate = run_dir / "predictions.json"
        if candidate.is_file():
            predictions_path = candidate.resolve()
    return VictimRunArtifacts(
        method=method,
        target_item=target_item,
        victim_model=victim_model,
        run_dir=run_dir.resolve(),
        metrics_path=metrics_path,
        metrics=metrics,
        predictions_path=predictions_path,
        train_history_path=train_history_path,
        resolved_config_path=resolved_config_path,
    )
