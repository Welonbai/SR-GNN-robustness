from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import shared_artifact_paths
from attack.pipeline.runs.run_pts_construction_cem import (
    build_pts_cem_shared_cache_identity,
    build_pts_construction_attack_identity_context,
    pts_cem_shared_cache_dir,
    pts_cem_shared_cache_key,
    _existing_pts_artifact_paths,
    _pts_construction_run_type,
    _try_load_cached_pts_best_candidate,
    _try_load_shared_pts_cem_cache,
    _write_shared_pts_cem_cache,
)


def _resolve_source_artifact_dir(source: Path, target_item: int) -> Path:
    source = source.resolve()
    if source.name == "pts_construction_cem":
        return source

    matches = sorted(source.glob(f"run_group_*/targets/{int(target_item)}/pts_construction_cem"))
    if not matches:
        matches = sorted(source.glob(f"targets/{int(target_item)}/pts_construction_cem"))
    if not matches:
        raise FileNotFoundError(
            "Could not find source PTS-CEM artifact directory under "
            f"{source} for target {int(target_item)}."
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple source PTS-CEM artifact directories matched; pass the exact "
            "pts_construction_cem directory. Matches: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _validate_shared_cache_available(
    *,
    shared_cache_dir: Path,
    target_item: int,
    shared_cache_key: str,
    shared_cache_identity: dict[str, object],
) -> bool:
    return (
        _try_load_shared_pts_cem_cache(
            shared_cache_dir=shared_cache_dir,
            target_item=int(target_item),
            shared_cache_key=shared_cache_key,
            shared_cache_identity=shared_cache_identity,
        )
        is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Import a legacy/local PTS-CEM target artifact into the current "
            "shared PTS-CEM cache namespace so sampled runs can reuse it."
        )
    )
    parser.add_argument("--config", required=True, help="Current sampled PTS-CEM config.")
    parser.add_argument("--target-item", type=int, required=True)
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Legacy explicit run root, run_group root, or exact "
            "targets/<item>/pts_construction_cem directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate inputs without writing the shared cache.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    target_item = int(args.target_item)
    source_artifact_dir = _resolve_source_artifact_dir(Path(args.source), target_item)
    run_type = _pts_construction_run_type(config)
    attack_identity_context = build_pts_construction_attack_identity_context(config)

    cached = _try_load_cached_pts_best_candidate(
        artifact_dir=source_artifact_dir,
        target_item=target_item,
        current_identity={"run_type": run_type},
        current_shared_cache_key=None,
    )
    if cached is None:
        raise ValueError(f"Source PTS-CEM artifact is not loadable: {source_artifact_dir}")

    shared_paths = shared_artifact_paths(
        config,
        run_type=run_type,
    )
    fake_sessions_path = shared_paths["fake_sessions"]
    poison_model_path = shared_paths["poison_model"]
    if not fake_sessions_path.exists():
        raise FileNotFoundError(
            "Current config shared fake sessions do not exist: "
            f"{fake_sessions_path}"
        )

    shared_cache_identity = build_pts_cem_shared_cache_identity(
        config,
        target_item=target_item,
        fake_sessions_path=fake_sessions_path,
        poison_model_path=poison_model_path,
    )
    shared_cache_key = pts_cem_shared_cache_key(shared_cache_identity)
    shared_cache_dir = pts_cem_shared_cache_dir(config, shared_cache_key)

    print(f"target_item={target_item}")
    print(f"source_artifact_dir={source_artifact_dir}")
    print(f"shared_pts_cem_cache_key={shared_cache_key}")
    print(f"shared_cache_dir={shared_cache_dir}")

    if _validate_shared_cache_available(
        shared_cache_dir=shared_cache_dir,
        target_item=target_item,
        shared_cache_key=shared_cache_key,
        shared_cache_identity=shared_cache_identity,
    ):
        print("status=already_available")
        return 0

    if shared_cache_dir.exists() and any(shared_cache_dir.iterdir()):
        raise ValueError(
            "Computed shared cache directory exists but is not a valid matching "
            f"cache: {shared_cache_dir}"
        )

    artifact_paths = _existing_pts_artifact_paths(source_artifact_dir)
    if args.dry_run:
        print("status=dry_run_valid")
        return 0

    marker_path = _write_shared_pts_cem_cache(
        config=config,
        target_item=target_item,
        local_artifact_dir=source_artifact_dir,
        artifact_paths=artifact_paths,
        best_candidate=cached.metadata,
        shared_cache_dir=shared_cache_dir,
        shared_cache_key=shared_cache_key,
        shared_cache_identity=shared_cache_identity,
        attack_identity_context=attack_identity_context,
    )

    if not _validate_shared_cache_available(
        shared_cache_dir=shared_cache_dir,
        target_item=target_item,
        shared_cache_key=shared_cache_key,
        shared_cache_identity=shared_cache_identity,
    ):
        raise ValueError(f"Imported shared cache is not loadable: {marker_path}")

    print(f"marker_path={marker_path}")
    print("status=imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
