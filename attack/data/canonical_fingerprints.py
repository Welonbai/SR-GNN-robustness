from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping


CANONICAL_FINGERPRINT_SEMANTICS = "canonical_exported_rows_sha256_v1"
ITEM_VOCABULARY_FINGERPRINT_SEMANTICS = "canonical_dense_item_map_sha256_v1"
_PYTHON_RUNTIME_CACHE_SUFFIXES = (".pyc", ".pyo")


def fingerprint_exported_jsonl(path: str | Path) -> str:
    """Hash validated exported rows using platform-independent canonical bytes."""
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("r", encoding="utf-8-sig", newline=None) as handle:
        expected_id = 0
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Canonical JSONL row {line_number} must not be blank.")
            payload = json.loads(line)
            row = _canonical_row(payload, line_number=line_number)
            if row["example_id"] != expected_id:
                raise ValueError("Canonical JSONL example_id values must be contiguous.")
            expected_id += 1
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )
            digest.update(b"\n")
    if expected_id == 0:
        raise ValueError("Canonical JSONL file must not be empty.")
    return digest.hexdigest()


def load_exported_canonical_labels(path: str | Path) -> list[int]:
    """Load labels from the authoritative exported rows without regenerating data."""
    source = Path(path)
    labels: list[int] = []
    with source.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Canonical JSONL row {line_number} must not be blank.")
            row = _canonical_row(json.loads(line), line_number=line_number)
            if row["example_id"] != len(labels):
                raise ValueError("Canonical JSONL example_id values must be contiguous.")
            labels.append(row["label"])
    if not labels:
        raise ValueError("Canonical JSONL file must not be empty.")
    return labels


def fingerprint_item_vocabulary(item_map: Mapping[Any, Any]) -> str:
    if not item_map:
        raise ValueError("Canonical item_map must not be empty.")
    rows: list[dict[str, Any]] = []
    canonical_ids: set[int] = set()
    for source_id, canonical_id_raw in item_map.items():
        if type(canonical_id_raw) is not int:
            raise TypeError("Canonical item_map values must be Python integers.")
        canonical_id = int(canonical_id_raw)
        canonical_ids.add(canonical_id)
        rows.append(
            {
                "canonical_id": canonical_id,
                "source_item": normalize_source_item_id(source_id),
            }
        )
    if canonical_ids != set(range(1, len(rows) + 1)):
        raise ValueError("Canonical item_map values must be dense unique IDs from 1.")
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["canonical_id"]):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_source_item_id(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        raise TypeError("Boolean source item IDs are not supported.")
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is part of the benchmark stack.
        np = None
    if np is not None and isinstance(value, np.bool_):
        raise TypeError("Boolean source item IDs are not supported.")
    if type(value) is int or (
        np is not None and isinstance(value, np.integer)
    ):
        return {"type": "int", "value": int(value)}
    if type(value) is str or (
        np is not None and isinstance(value, np.str_)
    ):
        return {"type": "str", "value": str(value)}
    if isinstance(value, float) or (
        np is not None and isinstance(value, np.floating)
    ):
        if not math.isfinite(value):
            raise TypeError("Non-finite source item IDs are not supported.")
        raise TypeError("Floating-point source item IDs are not supported.")
    raise TypeError(f"Unsupported source item ID type: {type(value).__name__}.")


def file_provenance(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError(f"Required artifact is empty: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "size": int(size), "sha256": digest.hexdigest()}


def resolve_wearec_repository_provenance(
    parent_root: str | Path,
    wearec_root: str | Path,
    *,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    parent = Path(parent_root).resolve()
    wearec = Path(wearec_root).resolve()

    def git(cwd: Path, *args: str) -> str:
        result = command_runner(
            ["git", *args],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.rstrip("\r\n")

    parent_commit = git(parent, "rev-parse", "HEAD")
    gitlink_commit = git(parent, "rev-parse", "HEAD:third_party/wearec")
    wearec_commit = git(wearec, "rev-parse", "HEAD")
    parent_status = git(
        parent,
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--ignore-submodules=dirty",
    )
    wearec_status = git(wearec, "status", "--porcelain", "--untracked-files=no")
    parent_dirty = _partition_parent_repository_status(parent_status)
    wearec_dirty = _partition_python_runtime_cache_status(wearec_status)
    if parent_dirty["blocking"]:
        raise RuntimeError(
            "Parent tracked worktree has non-runtime-cache dirty files for "
            "cacheable WEARec execution: "
            + _format_dirty_paths(parent_dirty["blocking"])
        )
    if gitlink_commit != wearec_commit:
        raise RuntimeError("Committed WEARec gitlink does not match checked-out submodule HEAD.")
    if wearec_dirty["blocking"]:
        raise RuntimeError(
            "WEARec tracked worktree has non-runtime-cache dirty files for "
            "cacheable execution: "
            + _format_dirty_paths(wearec_dirty["blocking"])
        )
    return {
        "parent_repository_commit": parent_commit,
        "parent_tracked_worktree_clean": True,
        "parent_ignored_runtime_cache_dirty_paths": parent_dirty["ignored"],
        "wearec_gitlink_commit": gitlink_commit,
        "wearec_submodule_commit": wearec_commit,
        "wearec_tracked_worktree_clean": True,
        "wearec_ignored_runtime_cache_dirty_paths": wearec_dirty["ignored"],
    }


def _partition_python_runtime_cache_status(status: str) -> dict[str, list[str]]:
    ignored: list[str] = []
    blocking: list[str] = []
    for line in status.splitlines():
        if not line.strip():
            continue
        paths = _git_status_line_paths(line)
        if paths and all(_is_python_runtime_cache_path(path) for path in paths):
            ignored.extend(paths)
        else:
            blocking.extend(paths or [line.strip()])
    return {
        "ignored": _dedupe_preserving_order(ignored),
        "blocking": _dedupe_preserving_order(blocking),
    }


def _partition_parent_repository_status(status: str) -> dict[str, list[str]]:
    partitioned = _partition_python_runtime_cache_status(status)
    ignored = list(partitioned["ignored"])
    blocking: list[str] = []
    for line in status.splitlines():
        if (
            _is_unrelated_submodule_dirty_status_line(line)
            or _is_analysis_only_dirty_status_line(line)
        ):
            ignored.extend(_git_status_line_paths(line))
            continue
    ignored_set = set(ignored)
    for path in partitioned["blocking"]:
        if path not in ignored_set:
            blocking.append(path)
    return {
        "ignored": _dedupe_preserving_order(ignored),
        "blocking": _dedupe_preserving_order(blocking),
    }


def _is_analysis_only_dirty_status_line(line: str) -> bool:
    """Ignore parent analysis artifacts that cannot affect victim execution."""
    paths = _git_status_line_paths(line)
    return bool(paths) and all(
        path.replace("\\", "/").startswith("analysis/") for path in paths
    )


def _is_unrelated_submodule_dirty_status_line(line: str) -> bool:
    if len(line) < 3:
        return False
    paths = _git_status_line_paths(line)
    if len(paths) != 1:
        return False
    path = paths[0].replace("\\", "/")
    if not path.startswith("third_party/") or path == "third_party/wearec":
        return False
    return "/" not in path.removeprefix("third_party/")


def _git_status_line_paths(line: str) -> list[str]:
    if len(line) >= 3 and line[2] == " ":
        payload = line[3:]
    elif len(line) >= 2 and line[1] == " ":
        payload = line[2:]
    else:
        payload = line[3:] if len(line) >= 3 else line
    payload = payload.strip()
    if not payload:
        return []
    if " -> " in payload:
        return [_clean_git_status_path(path) for path in payload.split(" -> ", 1)]
    return [_clean_git_status_path(payload)]


def _clean_git_status_path(path: str) -> str:
    cleaned = path.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    return cleaned


def _is_python_runtime_cache_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False
    parts = [part for part in normalized.split("/") if part]
    if "__pycache__" in parts:
        return True
    return normalized.lower().endswith(_PYTHON_RUNTIME_CACHE_SUFFIXES)


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _format_dirty_paths(paths: list[str], *, limit: int = 5) -> str:
    shown = paths[:limit]
    suffix = "" if len(paths) <= limit else f" ... (+{len(paths) - limit} more)"
    return ", ".join(shown) + suffix


def _canonical_row(payload: Any, *, line_number: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "example_id",
        "input_prefix",
        "label",
    }:
        raise ValueError(f"Canonical JSONL row {line_number} has an invalid schema.")
    example_id = payload["example_id"]
    label = payload["label"]
    prefix = payload["input_prefix"]
    if type(example_id) is not int or example_id < 0:
        raise ValueError("Canonical example_id must be a non-negative integer.")
    if type(label) is not int or label <= 0:
        raise ValueError("Canonical label must be a positive integer.")
    if not isinstance(prefix, list) or not prefix:
        raise ValueError("Canonical input_prefix must be a non-empty list.")
    if any(type(item) is not int or item <= 0 for item in prefix):
        raise ValueError("Canonical input_prefix items must be positive integers.")
    return {
        "example_id": int(example_id),
        "input_prefix": [int(item) for item in prefix],
        "label": int(label),
    }


__all__ = [
    "CANONICAL_FINGERPRINT_SEMANTICS",
    "ITEM_VOCABULARY_FINGERPRINT_SEMANTICS",
    "file_provenance",
    "fingerprint_exported_jsonl",
    "fingerprint_item_vocabulary",
    "load_exported_canonical_labels",
    "normalize_source_item_id",
    "resolve_wearec_repository_provenance",
]
