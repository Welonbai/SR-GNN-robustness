from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_json_object(payload: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _validate_required_fields(
    payload: Mapping[str, Any],
    *,
    label: str,
    required_fields: tuple[str, ...],
) -> dict[str, Any]:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{label} is missing required fields: {', '.join(missing)}")
    return dict(payload)


def save_poison_model(runner, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner.save_model(path)


def load_poison_model(runner, path: str | Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    runner.load_model(path)
    return True


def save_fake_sessions(sessions: list[list[int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(sessions, handle)


def load_fake_sessions(path: str | Path) -> list[list[int]] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_target_info(
    path: str | Path,
    *,
    target_items: list[int],
    target_selection_mode: str,
    seed: int,
    bucket: str | None = None,
    count: int | None = None,
    explicit_list: list[int] | None = None,
) -> None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    target_items = [int(item) for item in target_items]
    payload = {
        "target_items": target_items,
        "target_selection_mode": target_selection_mode,
        "seed": int(seed),
        "bucket": bucket,
        "count": int(count) if count is not None else None,
        "explicit_list": [int(item) for item in (explicit_list or [])],
    }
    if target_items:
        payload["target_item"] = int(target_items[0])
    save_json(payload, path)


def load_target_info(path: str | Path) -> dict[str, Any] | None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    payload = load_json(path)
    if payload is None:
        return None
    return _require_json_object(payload, label="target_info.json")


def save_selected_targets(path: str | Path, target_items: list[int]) -> None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    save_json({"target_items": [int(item) for item in target_items]}, path)


def load_selected_targets(path: str | Path) -> list[int] | None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    payload = load_json(path)
    if payload is None:
        return None
    payload = _require_json_object(payload, label="selected_targets.json")
    target_items = payload.get("target_items")
    if not isinstance(target_items, list):
        raise ValueError("selected_targets.json is missing target_items.")
    return [int(item) for item in target_items]


def save_target_selection_meta(path: str | Path, payload: dict[str, Any]) -> None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    save_json(payload, path)


def load_target_selection_meta(path: str | Path) -> dict[str, Any] | None:
    """Legacy-only helper for batch-era target-selection artifacts."""
    payload = load_json(path)
    if payload is None:
        return None
    return _require_json_object(payload, label="target_selection_meta.json")


_TARGET_REGISTRY_FIELDS = (
    "target_cohort_key",
    "split_key",
    "selection_policy_version",
    "mode",
    "bucket",
    "seed",
    "explicit_list",
    "candidate_pool_hash",
    "candidate_pool_size",
    "ordered_targets",
    "current_count",
    "created_at",
    "updated_at",
)
_RUN_COVERAGE_FIELDS = (
    "run_group_key",
    "target_cohort_key",
    "targets_order",
    "victims",
    "cells",
    "created_at",
    "updated_at",
)
_EXECUTION_LOG_FIELDS = (
    "run_group_key",
    "target_cohort_key",
    "executions",
    "created_at",
    "updated_at",
)
_SUMMARY_CURRENT_FIELDS = (
    "run_group_key",
    "target_cohort_key",
    "run_type",
    "targets",
    "created_at",
    "updated_at",
)


def save_target_registry(payload: Mapping[str, Any], path: str | Path) -> None:
    save_json(
        _validate_required_fields(
            _require_json_object(payload, label="target_registry.json"),
            label="target_registry.json",
            required_fields=_TARGET_REGISTRY_FIELDS,
        ),
        path,
    )


def load_target_registry(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    return _validate_required_fields(
        _require_json_object(payload, label="target_registry.json"),
        label="target_registry.json",
        required_fields=_TARGET_REGISTRY_FIELDS,
    )


def save_run_coverage(payload: Mapping[str, Any], path: str | Path) -> None:
    save_json(
        _validate_required_fields(
            _require_json_object(payload, label="run_coverage.json"),
            label="run_coverage.json",
            required_fields=_RUN_COVERAGE_FIELDS,
        ),
        path,
    )


def load_run_coverage(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    return _validate_required_fields(
        _require_json_object(payload, label="run_coverage.json"),
        label="run_coverage.json",
        required_fields=_RUN_COVERAGE_FIELDS,
    )


def save_execution_log(payload: Mapping[str, Any], path: str | Path) -> None:
    save_json(
        _validate_required_fields(
            _require_json_object(payload, label="execution_log.json"),
            label="execution_log.json",
            required_fields=_EXECUTION_LOG_FIELDS,
        ),
        path,
    )


def load_execution_log(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    return _validate_required_fields(
        _require_json_object(payload, label="execution_log.json"),
        label="execution_log.json",
        required_fields=_EXECUTION_LOG_FIELDS,
    )


def save_summary_current(payload: Mapping[str, Any], path: str | Path) -> None:
    save_json(
        _validate_required_fields(
            _require_json_object(payload, label="summary_current.json"),
            label="summary_current.json",
            required_fields=_SUMMARY_CURRENT_FIELDS,
        ),
        path,
    )


def load_summary_current(path: str | Path) -> dict[str, Any] | None:
    payload = load_json(path)
    if payload is None:
        return None
    return _validate_required_fields(
        _require_json_object(payload, label="summary_current.json"),
        label="summary_current.json",
        required_fields=_SUMMARY_CURRENT_FIELDS,
    )


__all__ = [
    "load_execution_log",
    "load_fake_sessions",
    "load_json",
    "load_poison_model",
    "load_run_coverage",
    "load_selected_targets",
    "load_summary_current",
    "load_target_info",
    "load_target_registry",
    "load_target_selection_meta",
    "save_execution_log",
    "save_fake_sessions",
    "save_json",
    "save_poison_model",
    "save_run_coverage",
    "save_selected_targets",
    "save_summary_current",
    "save_target_info",
    "save_target_registry",
    "save_target_selection_meta",
]
