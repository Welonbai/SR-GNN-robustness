from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


EXCLUDED_SURROGATE_TRAIN_KEYS = ("batch_size", "eval_batch_size", "train_batch_size")
SURROGATE_BATCH_IDENTITY_NOTE = (
    "Surrogate train batch-size parameters are intentionally excluded from "
    "PTS-CEM shared cache identity."
)
LOCAL_MARKER_NAME = "pts_construction_complete.json"
SHARED_MARKER_NAME = "pts_cem_shared_complete.json"


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _stable_json(payload: object) -> str:
    return json.dumps(
        _to_jsonable(payload),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _shared_key(identity: Mapping[str, object]) -> str:
    digest = hashlib.sha1(_stable_json(identity).encode("utf-8")).hexdigest()[:10]
    return f"pts_cem_shared_{digest}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


def _batch_neutral_identity(identity: Mapping[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(dict(identity))
    surrogate_reward = normalized.get("surrogate_reward")
    if isinstance(surrogate_reward, dict):
        params = surrogate_reward.get("surrogate_train_params")
        if isinstance(params, dict):
            for key in EXCLUDED_SURROGATE_TRAIN_KEYS:
                params.pop(key, None)
        surrogate_reward["surrogate_train_identity_excluded_params"] = sorted(
            EXCLUDED_SURROGATE_TRAIN_KEYS
        )
        surrogate_reward["surrogate_train_batch_identity_note"] = (
            SURROGATE_BATCH_IDENTITY_NOTE
        )
    return normalized


def _iter_shared_markers(root: Path, dataset: str | None) -> list[Path]:
    shared_root = root / "shared"
    if dataset:
        return sorted(
            (shared_root / dataset / "pts_construction_cem").glob(
                "pts_cem_shared_*/" + SHARED_MARKER_NAME
            )
        )
    return sorted(
        shared_root.glob("*/pts_construction_cem/pts_cem_shared_*/" + SHARED_MARKER_NAME)
    )


def _copy_shared_cache(
    marker_path: Path,
    *,
    new_key: str,
    new_identity: Mapping[str, object],
    apply: bool,
) -> tuple[str, str] | None:
    source_dir = marker_path.parent
    old_key = source_dir.name
    if old_key == new_key:
        return None
    destination_dir = source_dir.parent / new_key
    if destination_dir.exists():
        return old_key, new_key
    if not apply:
        return old_key, new_key

    shutil.copytree(source_dir, destination_dir)
    copied_marker_path = destination_dir / SHARED_MARKER_NAME
    marker = _load_json(copied_marker_path)
    marker["shared_pts_cem_cache_key"] = new_key
    marker["construction_identity"] = dict(new_identity)
    marker["migrated_from_shared_pts_cem_cache_key"] = old_key
    marker["migration_note"] = (
        "Migrated to batch-size-neutral surrogate train cache identity."
    )
    _write_json(copied_marker_path, marker)
    return old_key, new_key


def _migrate_run_markers(root: Path, key_map: Mapping[str, str], *, apply: bool) -> int:
    changed = 0
    runs_root = root / "runs"
    for marker_path in sorted(runs_root.glob("**/pts_construction_cem/" + LOCAL_MARKER_NAME)):
        marker = _load_json(marker_path)
        old_key = marker.get("shared_pts_cem_cache_key")
        if not isinstance(old_key, str) or old_key not in key_map:
            continue
        new_key = key_map[old_key]
        if old_key == new_key:
            continue
        changed += 1
        if not apply:
            continue
        marker["shared_pts_cem_cache_key"] = new_key
        shared_path = marker.get("shared_cache_path")
        if isinstance(shared_path, str):
            marker["shared_cache_path"] = shared_path.replace(old_key, new_key)
        marker["migrated_from_shared_pts_cem_cache_key"] = old_key
        marker["migration_note"] = (
            "Migrated to batch-size-neutral surrogate train cache identity."
        )
        _write_json(marker_path, marker)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy existing PTS-CEM shared caches to the batch-size-neutral "
            "surrogate train cache key."
        )
    )
    parser.add_argument("--root", default="outputs", help="Artifact root directory.")
    parser.add_argument("--dataset", default=None, help="Optional dataset name filter.")
    parser.add_argument("--apply", action="store_true", help="Actually copy/update files.")
    parser.add_argument(
        "--migrate-run-markers",
        action="store_true",
        help="Also update local run pts_construction_complete.json markers.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    key_map: dict[str, str] = {}
    for marker_path in _iter_shared_markers(root, args.dataset):
        marker = _load_json(marker_path)
        identity = marker.get("construction_identity")
        if not isinstance(identity, Mapping):
            continue
        new_identity = _batch_neutral_identity(identity)
        new_key = _shared_key(new_identity)
        migrated = _copy_shared_cache(
            marker_path,
            new_key=new_key,
            new_identity=new_identity,
            apply=bool(args.apply),
        )
        if migrated is None:
            continue
        old_key, new_key = migrated
        key_map[old_key] = new_key
        action = "migrate" if args.apply else "would migrate"
        print(f"{action}: {old_key} -> {new_key}")

    if args.migrate_run_markers and key_map:
        count = _migrate_run_markers(root, key_map, apply=bool(args.apply))
        action = "updated" if args.apply else "would update"
        print(f"{action} {count} local run marker(s)")

    if not args.apply:
        print("dry run only; pass --apply to write migrated cache copies.")


if __name__ == "__main__":
    main()
