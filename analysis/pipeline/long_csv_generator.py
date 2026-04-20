#!/usr/bin/env python3
"""Generate a slice-aware canonical long-table CSV from appendable run-group artifacts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.utils.inventory_utils import build_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RESULTS_ROOT = REPO_ROOT / "results"
SLICE_POLICIES = (
    "largest_complete_prefix",
    "intersection_complete",
    "all_available",
)
CANONICAL_COLUMNS = [
    "run_id",
    "dataset",
    "attack_method",
    "victim_model",
    "target_item",
    "target_type",
    "attack_size",
    "poison_model",
    "fake_session_generation_topk",
    "replacement_topk_ratio",
    "metric",
    "k",
    "value",
]

METRIC_KEY_PATTERN = re.compile(r"^(?:targeted_)?(?P<metric>[A-Za-z0-9_]+)@(?P<k>\d+)$")
UNSAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
REPEATED_UNDERSCORE_PATTERN = re.compile(r"_+")


class AnalysisError(ValueError):
    """Raised when an input cannot be converted into the canonical schema."""


@dataclass(frozen=True)
class RunMetadata:
    """Stable run-level metadata copied onto every long-table row."""

    dataset: str
    attack_method: str
    target_type: str
    attack_size: float
    poison_model: str
    fake_session_generation_topk: int
    replacement_topk_ratio: float


@dataclass(frozen=True)
class SourceArtifacts:
    """Resolved appendable run-group source artifacts."""

    summary_current_path: Path
    run_coverage_path: Path
    target_registry_path: Path
    resolved_config_path: Path
    artifact_manifest_path: Path


@dataclass(frozen=True)
class SliceSelection:
    """Coverage-aware selected target slice."""

    slice_policy: str
    requested_victims: list[str]
    requested_victims_source: str
    requested_target_count: int | None
    considered_targets: list[int | str]
    selected_targets: list[int | str]
    excluded_targets: list[int | str]
    excluded_incomplete_cells: list[dict[str, Any]]
    fairness_safe: bool


@dataclass(frozen=True)
class LongCsvDefaultsSpec:
    """Optional shared defaults applied to every batch job."""

    output_name: str | None
    slice_policy: str | None
    requested_victims: list[str] | None
    requested_target_count: int | None


@dataclass(frozen=True)
class LongCsvJobSpec:
    """One validated long-table generation job from config."""

    summary_path: Path
    output_name: str | None
    slice_policy: str | None
    requested_victims: list[str] | None
    requested_victims_source_override: str | None
    requested_target_count: int | None


@dataclass(frozen=True)
class LongCsvBatchSpec:
    """Validated batch config content for one long-table generation run."""

    config_path: Path
    jobs: list[LongCsvJobSpec]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 7 CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate one or more slice-aware canonical long_table.csv bundles from a YAML config."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a long_csv YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_file(args.config, label="long_csv config")
        spec = parse_long_csv_batch_spec(
            load_yaml_mapping(config_path, label="long_csv config"),
            source_config_path=config_path,
        )
        results = build_long_table_bundles(spec)
        rendered_outputs = ", ".join(result["output_name"] for result in results)
        print(f"Wrote {len(results)} long-table bundle(s): {rendered_outputs}")
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def generate_long_table_bundle(
    *,
    summary_path: str | Path,
    output_name: str | None,
    slice_policy: str | None,
    requested_victims: list[str] | None,
    requested_target_count: int | None,
) -> dict[str, object]:
    """Generate one slice-aware long-table bundle and return its key paths."""
    return _generate_long_table_bundle(
        summary_path=summary_path,
        output_name=output_name,
        slice_policy=slice_policy,
        requested_victims=requested_victims,
        requested_target_count=requested_target_count,
        requested_victims_source_override=None,
    )


def build_long_table_bundles(spec: LongCsvBatchSpec) -> list[dict[str, object]]:
    """Generate one or more slice-aware long-table bundles from a validated config."""
    results: list[dict[str, object]] = []
    for job in spec.jobs:
        results.append(
            _generate_long_table_bundle(
                summary_path=job.summary_path,
                output_name=job.output_name,
                slice_policy=job.slice_policy,
                requested_victims=job.requested_victims,
                requested_target_count=job.requested_target_count,
                requested_victims_source_override=job.requested_victims_source_override,
            )
        )
    return results


def _generate_long_table_bundle(
    *,
    summary_path: str | Path,
    output_name: str | None,
    slice_policy: str | None,
    requested_victims: list[str] | None,
    requested_target_count: int | None,
    requested_victims_source_override: str | None,
) -> dict[str, object]:
    """Generate one slice-aware long-table bundle with an optional victim-source override."""
    requested_summary_path = resolve_existing_path(str(summary_path), label="summary JSON")
    ensure_path_within(requested_summary_path, OUTPUTS_ROOT, label="summary JSON")
    source_paths = resolve_source_artifacts(requested_summary_path)

    summary_payload = load_json_mapping(source_paths.summary_current_path, label="summary_current JSON")
    run_coverage_payload = load_json_mapping(source_paths.run_coverage_path, label="run_coverage JSON")
    target_registry_payload = load_json_mapping(source_paths.target_registry_path, label="target_registry JSON")
    resolved_config_payload = load_json_mapping(
        source_paths.resolved_config_path,
        label="resolved_config JSON",
    )

    validate_source_alignment(
        summary_payload=summary_payload,
        run_coverage_payload=run_coverage_payload,
        target_registry_payload=target_registry_payload,
    )

    metadata = extract_run_metadata(summary_payload, resolved_config_payload)
    resolved_requested_victims, requested_victims_source = resolve_requested_victims(
        run_coverage_payload,
        requested_victims=requested_victims,
        requested_victims_source_override=requested_victims_source_override,
    )
    resolved_slice_policy = slice_policy or "largest_complete_prefix"
    slice_selection = resolve_slice(
        run_coverage_payload,
        target_registry_payload,
        requested_victims=resolved_requested_victims,
        requested_victims_source=requested_victims_source,
        slice_policy=resolved_slice_policy,
        requested_target_count=requested_target_count,
    )

    if not slice_selection.selected_targets:
        raise AnalysisError(
            "The requested slice is empty. "
            f"slice_policy={slice_selection.slice_policy}, "
            f"requested_victims={slice_selection.requested_victims}, "
            f"requested_target_count={slice_selection.requested_target_count}."
        )

    resolved_output_name = resolve_output_name(
        output_name,
        summary_path=source_paths.summary_current_path,
        slice_selection=slice_selection,
    )
    outdir = RESULTS_ROOT / "runs" / resolved_output_name
    rows = extract_rows(
        summary_payload,
        metadata=metadata,
        run_id=resolved_output_name,
        selected_targets=slice_selection.selected_targets,
        requested_victims=slice_selection.requested_victims,
        run_coverage_payload=run_coverage_payload,
        slice_policy=slice_selection.slice_policy,
    )
    if not rows:
        raise AnalysisError(
            f"No metric rows were extracted from '{source_paths.summary_current_path}' "
            "for the selected slice."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    long_table_path = outdir / "long_table.csv"
    inventory_path = outdir / "inventory.json"
    source_resolved_config_path = outdir / "source_resolved_config.json"
    manifest_path = outdir / "manifest.json"
    slice_manifest_path = outdir / "slice_manifest.json"

    dataframe = sort_long_table(
        pd.DataFrame(rows, columns=CANONICAL_COLUMNS),
        target_order=slice_selection.selected_targets,
    )
    dataframe.to_csv(long_table_path, index=False)
    write_json(inventory_path, build_inventory(dataframe))
    shutil.copyfile(source_paths.resolved_config_path, source_resolved_config_path)

    slice_manifest = build_slice_manifest(
        summary_payload=summary_payload,
        run_coverage_payload=run_coverage_payload,
        target_registry_payload=target_registry_payload,
        source_paths=source_paths,
        slice_selection=slice_selection,
    )
    write_json(slice_manifest_path, slice_manifest)

    manifest = {
        "canonical_columns": CANONICAL_COLUMNS,
        "run_id": resolved_output_name,
        "source_summary_current_path": to_repo_relative(source_paths.summary_current_path),
        "source_run_coverage_path": to_repo_relative(source_paths.run_coverage_path),
        "source_target_registry_path": to_repo_relative(source_paths.target_registry_path),
        "source_resolved_config_path": to_repo_relative(source_paths.resolved_config_path),
        "source_artifact_manifest_path": to_repo_relative(source_paths.artifact_manifest_path),
        "output_dir": to_repo_relative(outdir),
        "generated_files": [
            "inventory.json",
            "long_table.csv",
            "manifest.json",
            "slice_manifest.json",
            "source_resolved_config.json",
        ],
        "row_count": len(rows),
        "slice": {
            "slice_policy": slice_selection.slice_policy,
            "requested_victims": list(slice_selection.requested_victims),
            "requested_target_count": slice_selection.requested_target_count,
            "selected_target_count": len(slice_selection.selected_targets),
            "fairness_safe": bool(slice_selection.fairness_safe),
            "slice_manifest_path": "slice_manifest.json",
        },
        "generation_timestamp": utc_now_iso(),
    }
    write_json(manifest_path, manifest)

    return {
        "output_name": resolved_output_name,
        "output_dir": outdir,
        "row_count": len(rows),
        "long_table_path": long_table_path,
        "slice_manifest_path": slice_manifest_path,
        "manifest_path": manifest_path,
    }


def parse_long_csv_batch_spec(
    payload: Mapping[str, Any],
    *,
    source_config_path: Path,
) -> LongCsvBatchSpec:
    """Validate and normalize one long_csv YAML config."""
    defaults = parse_long_csv_defaults_spec(payload.get("defaults"), label="defaults")
    jobs_value = payload.get("jobs")
    if not isinstance(jobs_value, list) or not jobs_value:
        raise AnalysisError("Expected 'jobs' to be a non-empty list of job mappings.")

    jobs: list[LongCsvJobSpec] = []
    for index, raw_job in enumerate(jobs_value):
        jobs.append(
            parse_long_csv_job_spec(
                require_mapping(raw_job, label=f"jobs[{index}]"),
                label=f"jobs[{index}]",
                defaults=defaults,
            )
        )

    return LongCsvBatchSpec(config_path=source_config_path, jobs=jobs)


def parse_long_csv_defaults_spec(value: Any, *, label: str) -> LongCsvDefaultsSpec:
    """Normalize optional shared defaults from a batch config."""
    if value is None:
        return LongCsvDefaultsSpec(
            output_name=None,
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )

    payload = require_mapping(value, label=label)
    return LongCsvDefaultsSpec(
        output_name=normalize_optional_output_name(
            payload.get("output_name"),
            label=f"{label}.output_name",
        ),
        slice_policy=normalize_optional_slice_policy(
            payload.get("slice_policy"),
            label=f"{label}.slice_policy",
        ),
        requested_victims=normalize_optional_requested_victims(
            payload.get("requested_victims"),
            label=f"{label}.requested_victims",
        ),
        requested_target_count=normalize_optional_requested_target_count(
            payload.get("requested_target_count"),
            label=f"{label}.requested_target_count",
        ),
    )


def parse_long_csv_job_spec(
    payload: Mapping[str, Any],
    *,
    label: str,
    defaults: LongCsvDefaultsSpec,
) -> LongCsvJobSpec:
    """Normalize one long_csv batch job after applying shared defaults."""
    summary_path = resolve_config_summary_path(
        require_nonempty_string(payload.get("summary"), label=f"{label}.summary"),
        label=f"{label}.summary",
    )
    raw_output_name = config_value_or_default(
        payload,
        key="output_name",
        default=defaults.output_name,
    )
    raw_slice_policy = config_value_or_default(
        payload,
        key="slice_policy",
        default=defaults.slice_policy,
    )
    raw_requested_victims = config_value_or_default(
        payload,
        key="requested_victims",
        default=defaults.requested_victims,
    )
    raw_requested_target_count = config_value_or_default(
        payload,
        key="requested_target_count",
        default=defaults.requested_target_count,
    )

    requested_victims = normalize_optional_requested_victims(
        raw_requested_victims,
        label=f"{label}.requested_victims",
    )
    return LongCsvJobSpec(
        summary_path=summary_path,
        output_name=normalize_optional_output_name(
            raw_output_name,
            label=f"{label}.output_name",
        ),
        slice_policy=normalize_optional_slice_policy(
            raw_slice_policy,
            label=f"{label}.slice_policy",
        ),
        requested_victims=requested_victims,
        requested_victims_source_override="config" if requested_victims is not None else None,
        requested_target_count=normalize_optional_requested_target_count(
            raw_requested_target_count,
            label=f"{label}.requested_target_count",
        ),
    )


def config_value_or_default(
    payload: Mapping[str, Any],
    *,
    key: str,
    default: Any,
) -> Any:
    """Return one config value, allowing explicit null to clear a shared default."""
    if key in payload:
        return payload[key]
    return default


def load_yaml_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a YAML file and require a top-level mapping."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid YAML: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level mapping.")
    return payload


def resolve_existing_file(raw_path: str, *, label: str) -> Path:
    """Resolve and validate one existing file path relative to the repo when needed."""
    path = resolve_repo_path(raw_path)
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve one path relative to the repository root when needed."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_config_summary_path(raw_path: str, *, label: str) -> Path:
    """Resolve one configured summary path and require it to stay under outputs/."""
    path = resolve_existing_file(raw_path, label=label)
    ensure_path_within(path, OUTPUTS_ROOT, label=label)
    return path


def resolve_existing_path(raw_path: str, *, label: str) -> Path:
    """Resolve one existing file path."""
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def ensure_path_within(path: Path, root: Path, *, label: str) -> None:
    """Require a path to stay inside one repository subtree."""
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(
            f"The {label} must stay under '{root.resolve()}', got '{path}'."
        ) from exc


def resolve_source_artifacts(summary_path: Path) -> SourceArtifacts:
    """Resolve the authoritative appendable run-group source files from one summary path."""
    run_root = summary_path.parent
    summary_current_path = run_root / "summary_current.json"
    if not summary_current_path.is_file():
        raise AnalysisError(
            f"Expected sibling summary_current.json next to '{summary_path}', but none was found."
        )

    run_coverage_path = run_root / "run_coverage.json"
    if not run_coverage_path.is_file():
        raise AnalysisError(
            f"Expected sibling run_coverage.json next to '{summary_path}', but none was found."
        )

    resolved_config_path = run_root / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise AnalysisError(
            f"Expected sibling resolved_config.json next to '{summary_path}', but none was found."
        )

    artifact_manifest_path = run_root / "artifact_manifest.json"
    if not artifact_manifest_path.is_file():
        raise AnalysisError(
            f"Expected sibling artifact_manifest.json next to '{summary_path}', but none was found."
        )
    artifact_manifest_payload = load_json_mapping(
        artifact_manifest_path,
        label="artifact_manifest JSON",
    )
    target_registry_path = resolve_target_registry_path(
        artifact_manifest_payload,
        artifact_manifest_path=artifact_manifest_path,
    )

    return SourceArtifacts(
        summary_current_path=summary_current_path,
        run_coverage_path=run_coverage_path,
        target_registry_path=target_registry_path,
        resolved_config_path=resolved_config_path,
        artifact_manifest_path=artifact_manifest_path,
    )


def resolve_target_registry_path(
    artifact_manifest_payload: Mapping[str, Any],
    *,
    artifact_manifest_path: Path,
) -> Path:
    """Resolve target_registry.json from artifact_manifest.json."""
    shared_artifacts = require_mapping(
        artifact_manifest_payload.get("shared_artifacts"),
        label="artifact_manifest.shared_artifacts",
    )
    target_cohort_payload = require_mapping(
        shared_artifacts.get("target_cohort"),
        label="artifact_manifest.shared_artifacts.target_cohort",
    )
    target_registry_raw = require_nonempty_string(
        target_cohort_payload.get("target_registry"),
        label="artifact_manifest.shared_artifacts.target_cohort.target_registry",
    )
    target_registry_path = repo_path_from_manifest(target_registry_raw)
    if not target_registry_path.is_file():
        raise AnalysisError(
            "The target_registry.json referenced by artifact_manifest.json does not exist: "
            f"'{target_registry_path}'."
        )
    ensure_path_within(target_registry_path, OUTPUTS_ROOT, label="target_registry JSON")
    return target_registry_path


def repo_path_from_manifest(raw_path: str) -> Path:
    """Resolve one manifest path relative to the repository root when needed."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON file and require a top-level object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The {label} at '{path}' is not valid JSON: {exc}.") from exc

    if not isinstance(payload, Mapping):
        raise AnalysisError(f"The {label} at '{path}' must contain a top-level JSON object.")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write formatted JSON with a stable layout."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_repo_relative(path: Path) -> str:
    """Convert an in-repo path to a portable manifest string."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_output_name(
    raw_output_name: str | None,
    *,
    summary_path: Path,
    slice_selection: SliceSelection,
) -> str:
    """Resolve the final output folder name under results/runs/."""
    if raw_output_name is None:
        return derive_output_name(summary_path, slice_selection=slice_selection)

    candidate = raw_output_name.strip()
    if not candidate:
        raise AnalysisError("The provided output_name is empty.")
    if "\\" in candidate or "/" in candidate:
        raise AnalysisError("The output_name value must be a single folder name, not a path.")

    parts = [sanitize_component(part) for part in re.split(r"__+", candidate)]
    normalized = "__".join(part for part in parts if part)
    if not normalized:
        raise AnalysisError(
            f"The provided output_name '{raw_output_name}' does not contain any usable characters."
        )
    return normalized


def derive_output_name(summary_path: Path, *, slice_selection: SliceSelection) -> str:
    """Build a readable output folder name from the summary path and slice identity."""
    try:
        relative_parts = list(summary_path.resolve().relative_to(OUTPUTS_ROOT).parts)
    except ValueError:
        relative_parts = [summary_path.stem]

    if relative_parts and relative_parts[0] == "runs":
        relative_parts = relative_parts[1:]
    if relative_parts:
        relative_parts[-1] = summary_path.stem
    sanitized_parts = [sanitize_component(part) for part in relative_parts if sanitize_component(part)]
    if not sanitized_parts:
        sanitized_parts = [sanitize_component(summary_path.stem)]

    victim_token = "_".join(sanitize_component(victim_name) for victim_name in slice_selection.requested_victims)
    if slice_selection.requested_target_count is None:
        target_count_token = "materialized"
    else:
        target_count_token = f"count_{slice_selection.requested_target_count}"
    sanitized_parts.extend(
        [
            f"slice_{sanitize_component(slice_selection.slice_policy)}",
            f"victims_{victim_token or 'none'}",
            target_count_token,
        ]
    )
    return "__".join(part for part in sanitized_parts if part)


def sanitize_component(raw_value: str) -> str:
    """Make one path-derived component filesystem-safe without losing readability."""
    sanitized = UNSAFE_COMPONENT_PATTERN.sub("_", raw_value.strip())
    sanitized = REPEATED_UNDERSCORE_PATTERN.sub("_", sanitized)
    sanitized = sanitized.strip("._-")
    return sanitized or "item"


def sort_long_table(dataframe: pd.DataFrame, *, target_order: list[int | str]) -> pd.DataFrame:
    """Return a deterministically ordered long table that preserves the selected target order."""
    target_position = {
        stringify_target_item(target_item): index
        for index, target_item in enumerate(target_order)
    }
    target_item_order = dataframe["target_item"].map(build_target_item_sort_key)
    sorted_dataframe = (
        dataframe.assign(
            _target_order=dataframe["target_item"].map(
                lambda item: target_position.get(stringify_target_item(item), len(target_position) + 1)
            ),
            _target_item_type_order=target_item_order.map(lambda item: item[0]),
            _target_item_int_order=target_item_order.map(lambda item: item[1]),
            _target_item_str_order=target_item_order.map(lambda item: item[2]),
        )
        .sort_values(
            by=[
                "_target_order",
                "_target_item_type_order",
                "_target_item_int_order",
                "_target_item_str_order",
                "victim_model",
                "metric",
                "k",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_target_order",
                "_target_item_type_order",
                "_target_item_int_order",
                "_target_item_str_order",
            ]
        )
        .reset_index(drop=True)
    )
    return sorted_dataframe


def build_target_item_sort_key(value: Any) -> tuple[int, int, str]:
    """Build a deterministic sort key for int-or-string target identifiers."""
    if isinstance(value, int):
        return (0, value, "")
    if isinstance(value, str):
        return (1, 0, value.casefold())
    raise AnalysisError(
        f"Expected 'target_item' to be int or str while sorting, got {type(value).__name__}."
    )


def validate_source_alignment(
    *,
    summary_payload: Mapping[str, Any],
    run_coverage_payload: Mapping[str, Any],
    target_registry_payload: Mapping[str, Any],
) -> None:
    """Validate that the appendable source artifacts describe the same run group and cohort."""
    summary_target_cohort_key = optional_nonempty_string(summary_payload.get("target_cohort_key"))
    coverage_target_cohort_key = require_nonempty_string(
        run_coverage_payload.get("target_cohort_key"),
        label="run_coverage.target_cohort_key",
    )
    registry_target_cohort_key = require_nonempty_string(
        target_registry_payload.get("target_cohort_key"),
        label="target_registry.target_cohort_key",
    )
    if summary_target_cohort_key and summary_target_cohort_key != coverage_target_cohort_key:
        raise AnalysisError(
            "summary_current.json and run_coverage.json disagree on target_cohort_key."
        )
    if coverage_target_cohort_key != registry_target_cohort_key:
        raise AnalysisError(
            "run_coverage.json and target_registry.json disagree on target_cohort_key."
        )

    summary_run_group_key = optional_nonempty_string(summary_payload.get("run_group_key"))
    coverage_run_group_key = require_nonempty_string(
        run_coverage_payload.get("run_group_key"),
        label="run_coverage.run_group_key",
    )
    if summary_run_group_key and summary_run_group_key != coverage_run_group_key:
        raise AnalysisError(
            "summary_current.json and run_coverage.json disagree on run_group_key."
        )


def resolve_requested_victims(
    run_coverage_payload: Mapping[str, Any],
    *,
    requested_victims: list[str] | None,
    requested_victims_source_override: str | None = None,
) -> tuple[list[str], str]:
    """Resolve requested victims from config/CLI input or run-group state."""
    victims_payload = require_mapping(
        run_coverage_payload.get("victims"),
        label="run_coverage.victims",
    )
    available_victims = [str(victim_name) for victim_name in victims_payload.keys()]
    if not available_victims:
        raise AnalysisError("run_coverage.json does not contain any victims.")

    if requested_victims is None:
        return available_victims, "run_coverage.victims"

    normalized: list[str] = []
    seen: set[str] = set()
    for victim_name in requested_victims:
        normalized_name = require_nonempty_string(victim_name, label="requested victim")
        if normalized_name in seen:
            raise AnalysisError(f"Duplicate requested victim '{normalized_name}'.")
        if normalized_name not in victims_payload:
            raise AnalysisError(
                f"Requested victim '{normalized_name}' does not exist in run_coverage.json."
            )
        seen.add(normalized_name)
        normalized.append(normalized_name)
    if not normalized:
        raise AnalysisError("The requested victim set must not be empty.")
    return normalized, requested_victims_source_override or "cli"


def resolve_slice(
    run_coverage_payload: Mapping[str, Any],
    target_registry_payload: Mapping[str, Any],
    *,
    requested_victims: list[str],
    requested_victims_source: str,
    slice_policy: str,
    requested_target_count: int | None,
) -> SliceSelection:
    """Resolve one explicit target slice from coverage plus target registry."""
    if slice_policy not in SLICE_POLICIES:
        raise AnalysisError(
            f"Unsupported slice_policy '{slice_policy}'. Expected one of {list(SLICE_POLICIES)}."
        )

    considered_targets = resolve_considered_targets(
        target_registry_payload,
        requested_target_count=requested_target_count,
    )
    if slice_policy == "largest_complete_prefix":
        selected_targets, excluded_targets = select_largest_complete_prefix(
            run_coverage_payload,
            target_order=considered_targets,
            requested_victims=requested_victims,
        )
        fairness_safe = True
    elif slice_policy == "intersection_complete":
        selected_targets, excluded_targets = select_intersection_complete(
            run_coverage_payload,
            target_order=considered_targets,
            requested_victims=requested_victims,
        )
        fairness_safe = True
    else:
        selected_targets, excluded_targets = select_all_available(
            run_coverage_payload,
            target_order=considered_targets,
            requested_victims=requested_victims,
        )
        fairness_safe = False

    return SliceSelection(
        slice_policy=slice_policy,
        requested_victims=list(requested_victims),
        requested_victims_source=requested_victims_source,
        requested_target_count=requested_target_count,
        considered_targets=considered_targets,
        selected_targets=selected_targets,
        excluded_targets=excluded_targets,
        excluded_incomplete_cells=collect_excluded_incomplete_cells(
            run_coverage_payload,
            target_order=considered_targets,
            requested_victims=requested_victims,
            selected_targets=selected_targets,
        ),
        fairness_safe=fairness_safe,
    )


def resolve_considered_targets(
    target_registry_payload: Mapping[str, Any],
    *,
    requested_target_count: int | None,
) -> list[int | str]:
    """Resolve the materialized ordered target prefix, optionally capped for analysis."""
    ordered_targets_raw = target_registry_payload.get("ordered_targets")
    if not isinstance(ordered_targets_raw, list):
        raise AnalysisError("target_registry.json is missing ordered_targets.")
    ordered_targets = [
        require_target_item(item, label="target_registry.ordered_targets[*]")
        for item in ordered_targets_raw
    ]
    current_count = require_int(
        target_registry_payload.get("current_count"),
        label="target_registry.current_count",
    )
    if current_count < 0 or current_count > len(ordered_targets):
        raise AnalysisError("target_registry.json has an invalid current_count.")

    considered_targets = ordered_targets[:current_count]
    if requested_target_count is None:
        return considered_targets
    if requested_target_count < 0:
        raise AnalysisError("requested_target_count must be non-negative.")
    if requested_target_count > len(considered_targets):
        raise AnalysisError(
            "requested_target_count exceeds the currently materialized target prefix: "
            f"{requested_target_count} > {len(considered_targets)}."
        )
    return considered_targets[:requested_target_count]


def select_largest_complete_prefix(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_order: list[int | str],
    requested_victims: list[str],
) -> tuple[list[int | str], list[int | str]]:
    """Select the largest ordered prefix completed for every requested victim."""
    selected_targets: list[int | str] = []
    for target_item in target_order:
        if is_target_completed_for_victims(
            run_coverage_payload,
            target_item=target_item,
            requested_victims=requested_victims,
        ):
            selected_targets.append(target_item)
            continue
        break
    excluded_targets = target_order[len(selected_targets) :]
    return selected_targets, excluded_targets


def select_intersection_complete(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_order: list[int | str],
    requested_victims: list[str],
) -> tuple[list[int | str], list[int | str]]:
    """Select every target completed for every requested victim while preserving registry order."""
    selected_targets: list[int | str] = []
    excluded_targets: list[int | str] = []
    for target_item in target_order:
        if is_target_completed_for_victims(
            run_coverage_payload,
            target_item=target_item,
            requested_victims=requested_victims,
        ):
            selected_targets.append(target_item)
        else:
            excluded_targets.append(target_item)
    return selected_targets, excluded_targets


def select_all_available(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_order: list[int | str],
    requested_victims: list[str],
) -> tuple[list[int | str], list[int | str]]:
    """Select every target with at least one completed requested-victim cell."""
    selected_targets: list[int | str] = []
    excluded_targets: list[int | str] = []
    for target_item in target_order:
        if has_any_completed_requested_victim(
            run_coverage_payload,
            target_item=target_item,
            requested_victims=requested_victims,
        ):
            selected_targets.append(target_item)
        else:
            excluded_targets.append(target_item)
    return selected_targets, excluded_targets


def is_target_completed_for_victims(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_item: int | str,
    requested_victims: list[str],
) -> bool:
    """Return whether every requested victim is completed for one target."""
    return all(
        coverage_cell_status(
            run_coverage_payload,
            target_item=target_item,
            victim_name=victim_name,
        )
        == "completed"
        for victim_name in requested_victims
    )


def has_any_completed_requested_victim(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_item: int | str,
    requested_victims: list[str],
) -> bool:
    """Return whether any requested victim is completed for one target."""
    return any(
        coverage_cell_status(
            run_coverage_payload,
            target_item=target_item,
            victim_name=victim_name,
        )
        == "completed"
        for victim_name in requested_victims
    )


def collect_excluded_incomplete_cells(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_order: list[int | str],
    requested_victims: list[str],
    selected_targets: list[int | str],
) -> list[dict[str, Any]]:
    """Collect debug detail for requested cells excluded because they are not completed."""
    selected_target_keys = {stringify_target_item(target_item) for target_item in selected_targets}
    excluded_cells: list[dict[str, Any]] = []
    for target_item in target_order:
        target_key = stringify_target_item(target_item)
        for victim_name in requested_victims:
            status = coverage_cell_status(
                run_coverage_payload,
                target_item=target_item,
                victim_name=victim_name,
            )
            if status == "completed" and target_key in selected_target_keys:
                continue
            if status == "completed":
                continue
            excluded_cells.append(
                {
                    "target_item": target_item,
                    "victim_name": victim_name,
                    "status": status,
                }
            )
    return excluded_cells


def coverage_cell_status(
    run_coverage_payload: Mapping[str, Any],
    *,
    target_item: int | str,
    victim_name: str,
) -> str:
    """Read one coverage status for one target/victim cell."""
    cells_payload = require_mapping(run_coverage_payload.get("cells"), label="run_coverage.cells")
    target_cells = require_mapping(
        cells_payload.get(stringify_target_item(target_item)),
        label=f"run_coverage.cells[{stringify_target_item(target_item)}]",
    )
    cell_payload = require_mapping(
        target_cells.get(victim_name),
        label=f"run_coverage.cells[{stringify_target_item(target_item)}][{victim_name}]",
    )
    return require_nonempty_string(
        cell_payload.get("status"),
        label=f"run_coverage.cells[{stringify_target_item(target_item)}][{victim_name}].status",
    )


def stringify_target_item(target_item: Any) -> str:
    """Convert a scalar target item into the coverage/summary object key shape."""
    normalized = require_target_item(target_item, label="target item")
    return str(normalized)


def build_slice_manifest(
    *,
    summary_payload: Mapping[str, Any],
    run_coverage_payload: Mapping[str, Any],
    target_registry_payload: Mapping[str, Any],
    source_paths: SourceArtifacts,
    slice_selection: SliceSelection,
) -> dict[str, Any]:
    """Build the explicit slice sidecar manifest."""
    return {
        "source_run_group_key": optional_nonempty_string(summary_payload.get("run_group_key"))
        or require_nonempty_string(
            run_coverage_payload.get("run_group_key"),
            label="run_coverage.run_group_key",
        ),
        "target_cohort_key": require_nonempty_string(
            target_registry_payload.get("target_cohort_key"),
            label="target_registry.target_cohort_key",
        ),
        "source_summary_current_path": to_repo_relative(source_paths.summary_current_path),
        "source_run_coverage_path": to_repo_relative(source_paths.run_coverage_path),
        "source_target_registry_path": to_repo_relative(source_paths.target_registry_path),
        "source_resolved_config_path": to_repo_relative(source_paths.resolved_config_path),
        "source_artifact_manifest_path": to_repo_relative(source_paths.artifact_manifest_path),
        "slice_policy": slice_selection.slice_policy,
        "requested_victims": list(slice_selection.requested_victims),
        "requested_victims_source": slice_selection.requested_victims_source,
        "requested_target_count": slice_selection.requested_target_count,
        "selected_targets": list(slice_selection.selected_targets),
        "selected_target_count": len(slice_selection.selected_targets),
        "excluded_targets": list(slice_selection.excluded_targets),
        "excluded_incomplete_cells": list(slice_selection.excluded_incomplete_cells),
        "fairness_safe": bool(slice_selection.fairness_safe),
        "generation_timestamp": utc_now_iso(),
    }


def extract_run_metadata(
    summary_payload: Mapping[str, Any],
    resolved_config_payload: Mapping[str, Any],
) -> RunMetadata:
    """Collect required run-level metadata from the summary and resolved config."""
    summary_run_type = optional_nonempty_string(summary_payload.get("run_type"))
    fallback_run_type = require_nonempty_string(
        get_nested_value(resolved_config_payload, ("derived", "run_type")),
        label="resolved_config.derived.run_type",
    )

    return RunMetadata(
        dataset=require_nonempty_string(
            get_nested_value(resolved_config_payload, ("result_config", "data", "dataset_name")),
            label="resolved_config.result_config.data.dataset_name",
        ),
        attack_method=summary_run_type or fallback_run_type,
        target_type=require_nonempty_string(
            get_nested_value(resolved_config_payload, ("result_config", "targets", "bucket")),
            label="resolved_config.result_config.targets.bucket",
        ),
        attack_size=require_float(
            get_nested_value(resolved_config_payload, ("result_config", "attack", "size")),
            label="resolved_config.result_config.attack.size",
        ),
        poison_model=require_nonempty_string(
            get_nested_value(resolved_config_payload, ("result_config", "attack", "poison_model", "name")),
            label="resolved_config.result_config.attack.poison_model.name",
        ),
        fake_session_generation_topk=require_int(
            get_nested_value(
                resolved_config_payload,
                ("result_config", "attack", "fake_session_generation_topk"),
            ),
            label="resolved_config.result_config.attack.fake_session_generation_topk",
        ),
        replacement_topk_ratio=require_float(
            get_nested_value(resolved_config_payload, ("result_config", "attack", "replacement_topk_ratio")),
            label="resolved_config.result_config.attack.replacement_topk_ratio",
        ),
    )


def get_nested_value(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """Read one required nested value from a JSON object."""
    current: Any = payload
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, Mapping):
            dotted = ".".join(traversed[:-1]) or "<root>"
            raise AnalysisError(
                f"Expected '{dotted}' to be a JSON object while reading '{'.'.join(path)}'."
            )
        if key not in current:
            raise AnalysisError(f"Missing required field '{'.'.join(path)}'.")
        current = current[key]
    return current


def optional_nonempty_string(value: Any) -> str | None:
    """Return a stripped string when present, otherwise None."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnalysisError(f"Expected a string value, got {type(value).__name__}.")
    stripped = value.strip()
    return stripped or None


def require_nonempty_string(value: Any, *, label: str) -> str:
    """Require a non-empty string."""
    if not isinstance(value, str):
        raise AnalysisError(f"Expected '{label}' to be a string, got {type(value).__name__}.")
    stripped = value.strip()
    if not stripped:
        raise AnalysisError(f"Expected '{label}' to be a non-empty string.")
    return stripped


def normalize_optional_output_name(value: Any, *, label: str) -> str | None:
    """Normalize one optional output_name from config."""
    if value is None:
        return None

    output_name = require_nonempty_string(value, label=label)
    if "\\" in output_name or "/" in output_name:
        raise AnalysisError(f"The '{label}' value must be a single folder name, not a path.")
    return output_name


def normalize_optional_slice_policy(value: Any, *, label: str) -> str | None:
    """Normalize one optional slice policy token."""
    if value is None:
        return None

    slice_policy = require_nonempty_string(value, label=label)
    if slice_policy not in SLICE_POLICIES:
        raise AnalysisError(
            f"Unsupported '{label}' value '{slice_policy}'. Expected one of {list(SLICE_POLICIES)}."
        )
    return slice_policy


def normalize_optional_requested_victims(value: Any, *, label: str) -> list[str] | None:
    """Normalize one optional requested_victims list from config."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"Expected '{label}' to be a non-empty list of strings.")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, raw_victim_name in enumerate(value):
        victim_name = require_nonempty_string(raw_victim_name, label=f"{label}[{index}]")
        if victim_name in seen:
            raise AnalysisError(f"Duplicate requested victim '{victim_name}' in '{label}'.")
        seen.add(victim_name)
        normalized.append(victim_name)
    return normalized


def normalize_optional_requested_target_count(value: Any, *, label: str) -> int | None:
    """Normalize one optional requested_target_count value from config."""
    if value is None:
        return None

    target_count = require_int(value, label=label)
    if target_count < 0:
        raise AnalysisError(f"Expected '{label}' to be non-negative.")
    return target_count


def require_float(value: Any, *, label: str) -> float:
    """Require a numeric value and coerce it to float."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be numeric, got bool.")
    if isinstance(value, (int, float)):
        return float(value)
    raise AnalysisError(f"Expected '{label}' to be numeric, got {type(value).__name__}.")


def require_int(value: Any, *, label: str) -> int:
    """Require an integer value."""
    if isinstance(value, bool):
        raise AnalysisError(f"Expected '{label}' to be an integer, got bool.")
    if isinstance(value, int):
        return value
    raise AnalysisError(f"Expected '{label}' to be an integer, got {type(value).__name__}.")


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Require a JSON object-like mapping."""
    if not isinstance(value, Mapping):
        raise AnalysisError(f"Expected '{label}' to be a JSON object.")
    return value


def require_target_item(value: Any, *, label: str) -> str | int:
    """Require a scalar target item identifier as int or non-empty string."""
    if isinstance(value, bool) or value is None:
        raise AnalysisError(f"Expected '{label}' to be a scalar target identifier.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise AnalysisError(f"Expected '{label}' to be a non-empty target identifier.")
        return value.strip()
    raise AnalysisError(
        f"Expected '{label}' to be a scalar target identifier, got {type(value).__name__}."
    )


def extract_rows(
    summary_payload: Mapping[str, Any],
    *,
    metadata: RunMetadata,
    run_id: str,
    selected_targets: list[int | str],
    requested_victims: list[str],
    run_coverage_payload: Mapping[str, Any],
    slice_policy: str,
) -> list[dict[str, Any]]:
    """Flatten summary_current.json into canonical rows for one selected slice."""
    if "targets" not in summary_payload:
        raise AnalysisError("The summary JSON is missing the top-level 'targets' field.")

    target_lookup = build_summary_target_lookup(summary_payload["targets"])
    rows: list[dict[str, Any]] = []
    for target_item in selected_targets:
        target_key = stringify_target_item(target_item)
        target_payload = target_lookup.get(target_key)
        if target_payload is None:
            raise AnalysisError(
                f"summary_current.json is missing selected target '{target_item}' for slice_policy "
                f"'{slice_policy}'. Coverage and registry are authoritative for slice resolution, "
                "so the summary snapshot must contain the selected completed cells."
            )
        victims = require_mapping(
            target_payload.get("victims"),
            label=f"summary.targets[{target_item}].victims",
        )
        for victim_name in requested_victims:
            if coverage_cell_status(
                run_coverage_payload,
                target_item=target_item,
                victim_name=victim_name,
            ) != "completed":
                continue
            victim_payload = require_mapping(
                victims.get(victim_name),
                label=f"summary.targets[{target_item}].victims[{victim_name}]",
            )
            metrics = require_mapping(
                victim_payload.get("metrics"),
                label=f"summary.targets[{target_item}].victims[{victim_name}].metrics",
            )
            for metric_key, metric_value in metrics.items():
                metric, k_value = parse_metric_key(metric_key)
                rows.append(
                    {
                        "run_id": run_id,
                        "dataset": metadata.dataset,
                        "attack_method": metadata.attack_method,
                        "victim_model": victim_name,
                        "target_item": target_item,
                        "target_type": metadata.target_type,
                        "attack_size": metadata.attack_size,
                        "poison_model": metadata.poison_model,
                        "fake_session_generation_topk": metadata.fake_session_generation_topk,
                        "replacement_topk_ratio": metadata.replacement_topk_ratio,
                        "metric": metric,
                        "k": k_value,
                        "value": require_float(
                            metric_value,
                            label=(
                                f"metric value for '{metric_key}' on victim '{victim_name}' "
                                f"and target '{target_item}'"
                            ),
                        ),
                    }
                )
    return rows


def build_summary_target_lookup(node: Any) -> dict[str, Mapping[str, Any]]:
    """Build a target lookup from summary_current.targets in any supported layout."""
    lookup: dict[str, Mapping[str, Any]] = {}
    for target_payload in iter_target_payloads(node):
        target_item = require_target_item(
            target_payload.get("target_item"),
            label="summary.targets[*].target_item",
        )
        lookup[stringify_target_item(target_item)] = target_payload
    if not lookup:
        raise AnalysisError("The summary JSON does not contain any target payloads.")
    return lookup


def iter_target_payloads(node: Any) -> Iterator[Mapping[str, Any]]:
    """Yield target payload objects from list/dict summary layouts."""
    if isinstance(node, Mapping):
        has_target_item = "target_item" in node
        has_victims = "victims" in node
        if has_target_item or has_victims:
            if not (has_target_item and has_victims):
                raise AnalysisError(
                    "Encountered a partial target payload; expected both 'target_item' and 'victims'."
                )
            yield node
            return

        for child in node.values():
            yield from iter_target_payloads(child)
        return

    if isinstance(node, list):
        for child in node:
            yield from iter_target_payloads(child)
        return

    raise AnalysisError(
        "The summary 'targets' field must be a list or object containing target payloads."
    )


def parse_metric_key(metric_key: Any) -> tuple[str, int]:
    """Split a metric key into canonical metric name and k."""
    if not isinstance(metric_key, str):
        raise AnalysisError(f"Metric keys must be strings, got {type(metric_key).__name__}.")

    match = METRIC_KEY_PATTERN.fullmatch(metric_key.strip())
    if match is None:
        raise AnalysisError(
            f"Unsupported metric key '{metric_key}'. Expected a form like 'targeted_precision@20'."
        )

    return match.group("metric").lower(), int(match.group("k"))


if __name__ == "__main__":
    main()
