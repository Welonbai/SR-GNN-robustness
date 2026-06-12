from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import load_run_coverage, save_run_coverage
from attack.pipeline.core.pipeline_utils import (
    _default_cell_coverage_entry,
    _timestamp_utc,
    sync_run_coverage_materialized_prefix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_VICTIMS = {"tron", "mdhg", "freqrec"}
DERIVED_ARTIFACT_NAMES = (
    "summary_current.json",
    "artifact_manifest.json",
    "execution_log.json",
    "progress.json",
)


class InvalidationError(ValueError):
    """Raised when victim-cell invalidation cannot be performed safely."""


@dataclass(frozen=True)
class InvalidationPlan:
    run_dir: Path
    coverage_path: Path
    coverage: dict[str, Any]
    victim: str
    target_ids: tuple[str, ...]
    completed_target_ids: tuple[str, ...]
    non_completed_target_ids: tuple[str, ...]
    artifact_dirs: tuple[Path, ...]
    existing_artifact_dirs: tuple[Path, ...]
    shared_artifact_dirs: tuple[Path, ...]
    existing_shared_artifact_dirs: tuple[Path, ...]
    old_victim_prediction_key: str | None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely invalidate run-local external victim cells while preserving attack, "
            "CEM, fake-session, and shared victim prediction caches."
        )
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Explicit run-group directory. Repeat this option to process multiple runs.",
    )
    parser.add_argument(
        "--victim",
        required=True,
        help="Victim to invalidate. Supported victims: tron, mdhg, freqrec.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report planned changes without modifying files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        if not args.dry_run:
            preflight_results = invalidate_victim_cells(
                args.run_dir,
                victim=args.victim,
                dry_run=True,
            )
            for result in preflight_results:
                print(format_invalidation_result(result, mode="preflight"))
        results = invalidate_victim_cells(
            args.run_dir,
            victim=args.victim,
            dry_run=args.dry_run,
        )
    except InvalidationError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    for result in results:
        print(format_invalidation_result(result))


def invalidate_victim_cells(
    run_dirs: Sequence[str | Path],
    *,
    victim: str,
    dry_run: bool = False,
    allowed_roots: Sequence[str | Path] | None = None,
) -> list[dict[str, Any]]:
    if victim not in SUPPORTED_VICTIMS:
        raise InvalidationError(
            f"Unsupported victim '{victim}'. Supported victims: {sorted(SUPPORTED_VICTIMS)}."
        )
    if not run_dirs:
        raise InvalidationError("At least one explicit --run-dir must be provided.")

    roots = _resolve_allowed_roots(allowed_roots)
    plans = [
        inspect_invalidation_plan(run_dir, victim=victim, allowed_roots=roots)
        for run_dir in run_dirs
    ]
    if dry_run:
        return [_result_from_plan(plan, dry_run=True) for plan in plans]

    results: list[dict[str, Any]] = []
    for plan in plans:
        _apply_invalidation_plan(plan)
        results.append(_result_from_plan(plan, dry_run=False))
    return results


def inspect_invalidation_plan(
    run_dir: str | Path,
    *,
    victim: str,
    allowed_roots: Sequence[str | Path] | None = None,
) -> InvalidationPlan:
    if victim not in SUPPORTED_VICTIMS:
        raise InvalidationError(
            f"Unsupported victim '{victim}'. Supported victims: {sorted(SUPPORTED_VICTIMS)}."
        )
    raw_path = str(run_dir)
    if any(token in raw_path for token in ("*", "?", "[", "]")):
        raise InvalidationError("Glob patterns are not supported; pass explicit run directories.")

    roots = _resolve_allowed_roots(allowed_roots)
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    _validate_path_within_allowed_roots(resolved_run_dir, roots)
    if not resolved_run_dir.is_dir():
        raise InvalidationError(f"Run directory does not exist: {resolved_run_dir}")

    coverage_path = resolved_run_dir / "run_coverage.json"
    if not coverage_path.is_file():
        raise InvalidationError(f"Missing run_coverage.json: {coverage_path}")
    try:
        coverage = load_run_coverage(coverage_path)
    except (OSError, ValueError) as exc:
        raise InvalidationError(f"Invalid run_coverage.json at {coverage_path}: {exc}") from exc
    if coverage is None:
        raise InvalidationError(f"run_coverage.json is empty: {coverage_path}")

    victims = coverage.get("victims")
    cells = coverage.get("cells")
    targets_order = coverage.get("targets_order")
    run_group_key = coverage.get("run_group_key")
    if not isinstance(victims, dict) or not isinstance(cells, dict):
        raise InvalidationError("run_coverage.json must contain object-valued victims and cells.")
    if not isinstance(targets_order, list) or not isinstance(run_group_key, str) or not run_group_key:
        raise InvalidationError(
            "run_coverage.json does not look like a run group: "
            "targets_order and run_group_key are required."
        )
    victim_entry = victims.get(victim)
    if not isinstance(victim_entry, dict):
        raise InvalidationError(
            f"run_coverage.json does not contain an object-valued victims.{victim} entry."
        )

    target_ids: list[str] = []
    completed_target_ids: list[str] = []
    non_completed_target_ids: list[str] = []
    artifact_dirs: list[Path] = []
    for target_id, target_cells in cells.items():
        if not isinstance(target_cells, Mapping):
            raise InvalidationError(f"run_coverage.json cells[{target_id}] must be an object.")
        victim_cell = target_cells.get(victim)
        if victim_cell is None:
            continue
        if not isinstance(victim_cell, Mapping):
            raise InvalidationError(
                f"run_coverage.json cells[{target_id}][{victim}] must be an object."
            )
        target_key = str(target_id)
        if (
            target_key in ("", ".", "..")
            or Path(target_key).name != target_key
            or "/" in target_key
            or "\\" in target_key
        ):
            raise InvalidationError(
                f"Unsafe target id in run_coverage.json cells: {target_key!r}"
            )
        target_ids.append(target_key)
        if victim_cell.get("status") == "completed":
            completed_target_ids.append(target_key)
        else:
            non_completed_target_ids.append(target_key)
        artifact_dir = resolved_run_dir / "targets" / target_key / "victims" / victim
        resolved_artifact_dir = artifact_dir.resolve()
        if not resolved_artifact_dir.is_relative_to(resolved_run_dir):
            raise InvalidationError(
                "Refusing to delete a victim artifact path that resolves outside the "
                f"run directory: {artifact_dir}"
            )
        if artifact_dir.exists() and not artifact_dir.is_dir():
            raise InvalidationError(
                f"Expected victim artifact directory but found a non-directory: {artifact_dir}"
            )
        artifact_dirs.append(artifact_dir)

    if not target_ids:
        raise InvalidationError(
            f"run_coverage.json contains no target cells for victim '{victim}'."
        )

    old_key = victim_entry.get("victim_prediction_key")
    shared_artifact_dirs: list[Path] = []
    if victim == "freqrec":
        manifest_path = resolved_run_dir / "artifact_manifest.json"
        if not manifest_path.is_file():
            raise InvalidationError(
                "FreqRec shared-cache invalidation requires an exact artifact_manifest.json; "
                f"leaving cache untouched: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidationError(
                f"Cannot resolve exact FreqRec shared-cache paths from {manifest_path}: {exc}"
            ) from exc
        manifest_victims = manifest.get("victims") if isinstance(manifest, dict) else None
        if not isinstance(manifest_victims, Mapping):
            raise InvalidationError(
                "FreqRec artifact manifest does not contain object-valued victims."
            )
        for target_id in target_ids:
            target_manifest = manifest_victims.get(target_id)
            victim_manifest = (
                target_manifest.get(victim)
                if isinstance(target_manifest, Mapping)
                else None
            )
            shared_manifest = (
                victim_manifest.get("shared")
                if isinstance(victim_manifest, Mapping)
                else None
            )
            shared_value = (
                shared_manifest.get("shared_dir")
                if isinstance(shared_manifest, Mapping)
                else None
            )
            if not isinstance(shared_value, str) or not shared_value.strip():
                raise InvalidationError(
                    "Cannot derive exact FreqRec shared-cache path for "
                    f"target {target_id}; leaving all artifacts untouched."
                )
            shared_path = Path(shared_value)
            if not shared_path.is_absolute():
                shared_path = REPO_ROOT / shared_path
            shared_path = shared_path.resolve()
            _validate_path_within_allowed_roots(shared_path, roots)
            parts_lower = [part.lower() for part in shared_path.parts]
            if (
                "victim_predictions" not in parts_lower
                or "freqrec" not in parts_lower
                or shared_path.name not in {"shared", target_id}
            ):
                raise InvalidationError(
                    f"Refusing unsafe FreqRec shared-cache path: {shared_path}"
                )
            if shared_path.exists() and not shared_path.is_dir():
                raise InvalidationError(
                    f"Expected FreqRec shared cache directory: {shared_path}"
                )
            shared_artifact_dirs.append(shared_path)
        shared_artifact_dirs = list(dict.fromkeys(shared_artifact_dirs))
    return InvalidationPlan(
        run_dir=resolved_run_dir,
        coverage_path=coverage_path,
        coverage=coverage,
        victim=victim,
        target_ids=tuple(target_ids),
        completed_target_ids=tuple(completed_target_ids),
        non_completed_target_ids=tuple(non_completed_target_ids),
        artifact_dirs=tuple(artifact_dirs),
        existing_artifact_dirs=tuple(path for path in artifact_dirs if path.is_dir()),
        shared_artifact_dirs=tuple(shared_artifact_dirs),
        existing_shared_artifact_dirs=tuple(
            path for path in shared_artifact_dirs if path.is_dir()
        ),
        old_victim_prediction_key=str(old_key) if old_key is not None else None,
    )


def format_invalidation_result(
    result: Mapping[str, Any],
    *,
    mode: str | None = None,
) -> str:
    if mode is None:
        mode = "dry-run" if result["dry_run"] else "invalidated"
    artifact_dirs = result["artifact_dirs_to_delete"]
    artifact_text = ", ".join(artifact_dirs) if artifact_dirs else "(none present)"
    return "\n".join(
        (
            f"[{mode}] run_dir={result['run_dir']}",
            f"  victim={result['victim']}",
            (
                f"  cells={result['cell_count']} "
                f"completed_to_reset={result['completed_cell_count']} "
                f"non_completed_to_reset={result['non_completed_cell_count']}"
            ),
            f"  target_ids={', '.join(result['target_ids'])}",
            f"  local_victim_artifact_dirs_to_delete={artifact_text}",
            (
                "  shared_victim_artifact_dirs_to_delete="
                + (
                    ", ".join(result["shared_artifact_dirs_to_delete"])
                    if result["shared_artifact_dirs_to_delete"]
                    else "(none present)"
                )
            ),
            (
                f"  run_coverage_updates=cells[*].{result['victim']} -> requested; "
                f"victims.{result['victim']} current key removed; materialized prefix recomputed"
            ),
            (
                "  derived_artifacts_unchanged=summary_current.json, artifact_manifest.json, "
                "execution_log.json, progress.json "
                f"(may temporarily contain stale {result['victim']} entries)"
            ),
            (
                "  untouched=CEM caches, fake-session caches, SR-GNN/MiaSRec outputs, "
                "all shared victim prediction caches"
            ),
        )
    )


def _apply_invalidation_plan(plan: InvalidationPlan) -> None:
    now = _timestamp_utc()
    cells = plan.coverage["cells"]
    for target_id in plan.target_ids:
        cells[target_id][plan.victim] = _default_cell_coverage_entry(now)

    victim_entry = plan.coverage["victims"][plan.victim]
    old_key = victim_entry.pop("victim_prediction_key", None)
    if old_key is not None:
        victim_entry["previous_victim_prediction_key"] = old_key
    victim_entry["status"] = "requested"
    victim_entry["last_requested_at"] = now
    plan.coverage["updated_at"] = now
    sync_run_coverage_materialized_prefix(plan.coverage)

    save_run_coverage(plan.coverage, plan.coverage_path)
    for artifact_dir in plan.artifact_dirs:
        if artifact_dir.is_dir():
            shutil.rmtree(artifact_dir)
    for artifact_dir in plan.shared_artifact_dirs:
        if artifact_dir.is_dir():
            shutil.rmtree(artifact_dir)


def _result_from_plan(plan: InvalidationPlan, *, dry_run: bool) -> dict[str, Any]:
    return {
        "run_dir": str(plan.run_dir),
        "victim": plan.victim,
        "dry_run": dry_run,
        "cell_count": len(plan.target_ids),
        "completed_cell_count": len(plan.completed_target_ids),
        "non_completed_cell_count": len(plan.non_completed_target_ids),
        "target_ids": list(plan.target_ids),
        "artifact_dirs_to_delete": [str(path) for path in plan.existing_artifact_dirs],
        "shared_artifact_dirs_to_delete": [
            str(path) for path in plan.existing_shared_artifact_dirs
        ],
        "coverage_path": str(plan.coverage_path),
        "old_victim_prediction_key": plan.old_victim_prediction_key,
        "derived_artifacts_unchanged": list(DERIVED_ARTIFACT_NAMES),
    }


def _resolve_allowed_roots(
    allowed_roots: Sequence[str | Path] | None,
) -> tuple[Path, ...]:
    raw_roots = allowed_roots if allowed_roots is not None else (REPO_ROOT,)
    roots = tuple(Path(root).expanduser().resolve() for root in raw_roots)
    if not roots:
        raise InvalidationError("At least one allowed root is required.")
    return roots


def _validate_path_within_allowed_roots(path: Path, allowed_roots: Sequence[Path]) -> None:
    if any(path == root or path.is_relative_to(root) for root in allowed_roots):
        return
    roots = ", ".join(str(root) for root in allowed_roots)
    raise InvalidationError(
        f"Refusing to operate outside the allowed repository/output area: {path}. "
        f"Allowed roots: {roots}"
    )


if __name__ == "__main__":
    main()
