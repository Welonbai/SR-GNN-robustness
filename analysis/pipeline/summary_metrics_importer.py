#!/usr/bin/env python3
"""Import metrics-only summary_current.json files into canonical run bundles."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.pipeline.view_table_builder import (
    AnalysisError,
    ensure_path_within,
    load_yaml_mapping,
    normalize_for_json,
    require_nonempty_string,
    require_string_list,
    resolve_repo_path,
    to_repo_relative,
    utc_now_iso,
    write_json,
)
from analysis.utils.inventory_utils import build_inventory


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
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
METRIC_KEY_PATTERN = re.compile(r"^(?P<metric>[A-Za-z0-9_]+)@(?P<k>\d+)$")


@dataclass(frozen=True)
class SummaryMetricsJobSpec:
    """One metrics-only import job."""

    output_name: str
    summaries: list[Path]
    dataset: str
    target_type: str
    attack_method: str
    attack_size: float
    poison_model: str
    fake_session_generation_topk: int
    replacement_topk_ratio: float
    requested_victims: list[str]
    selected_targets: list[int | str]


@dataclass(frozen=True)
class SummaryMetricsImportSpec:
    """Validated import config."""

    config_path: Path
    jobs: list[SummaryMetricsJobSpec]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the importer CLI parser."""
    parser = argparse.ArgumentParser(
        description="Import metrics-only summary_current.json files into results/runs bundles.",
    )
    parser.add_argument("--config", required=True, help="Path to summary-metrics import YAML.")
    parser.add_argument("--spec", dest="config", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    """Run the importer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_file(args.config, label="summary-metrics import config")
        spec = parse_summary_metrics_import_spec(
            load_yaml_mapping(config_path, label="summary-metrics import config"),
            source_config_path=config_path,
        )
        results = import_summary_metrics_bundles(spec)
        print(f"Wrote {len(results)} summary-metrics run bundle(s):")
        for result in results:
            print(f"- {result['output_name']}: {result['row_count']} rows")
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_summary_metrics_import_spec(
    payload: Mapping[str, Any],
    *,
    source_config_path: Path,
) -> SummaryMetricsImportSpec:
    """Validate and normalize one import YAML."""
    defaults = payload.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise AnalysisError("Expected 'defaults' to be a mapping when provided.")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise AnalysisError("Expected 'jobs' to be a non-empty list.")

    jobs = [
        parse_summary_metrics_job(
            raw_job,
            defaults=defaults,
            label=f"jobs[{index}]",
        )
        for index, raw_job in enumerate(raw_jobs)
    ]
    return SummaryMetricsImportSpec(config_path=source_config_path, jobs=jobs)


def parse_summary_metrics_job(
    value: Any,
    *,
    defaults: Mapping[str, Any],
    label: str,
) -> SummaryMetricsJobSpec:
    """Validate one import job."""
    if not isinstance(value, Mapping):
        raise AnalysisError(f"Expected '{label}' to be a mapping.")

    def field(name: str, default: Any = None) -> Any:
        if name in value:
            return value[name]
        if name in defaults:
            return defaults[name]
        return default

    summaries = [
        resolve_existing_file(str(path), label=f"{label}.summaries")
        for path in require_string_list(field("summaries"), label=f"{label}.summaries")
    ]
    selected_targets = require_scalar_list(field("selected_targets"), label=f"{label}.selected_targets")
    requested_victims = require_string_list(field("requested_victims"), label=f"{label}.requested_victims")

    return SummaryMetricsJobSpec(
        output_name=require_nonempty_string(field("output_name"), label=f"{label}.output_name"),
        summaries=summaries,
        dataset=require_nonempty_string(field("dataset"), label=f"{label}.dataset"),
        target_type=require_nonempty_string(field("target_type"), label=f"{label}.target_type"),
        attack_method=require_nonempty_string(field("attack_method"), label=f"{label}.attack_method"),
        attack_size=require_float(field("attack_size"), label=f"{label}.attack_size"),
        poison_model=require_nonempty_string(field("poison_model"), label=f"{label}.poison_model"),
        fake_session_generation_topk=require_int(
            field("fake_session_generation_topk"),
            label=f"{label}.fake_session_generation_topk",
        ),
        replacement_topk_ratio=require_float(
            field("replacement_topk_ratio"),
            label=f"{label}.replacement_topk_ratio",
        ),
        requested_victims=requested_victims,
        selected_targets=selected_targets,
    )


def import_summary_metrics_bundles(spec: SummaryMetricsImportSpec) -> list[dict[str, Any]]:
    """Import all configured summary-metrics bundles."""
    return [import_summary_metrics_bundle(job=job, source_config_path=spec.config_path) for job in spec.jobs]


def import_summary_metrics_bundle(
    *,
    job: SummaryMetricsJobSpec,
    source_config_path: Path,
) -> dict[str, Any]:
    """Import one metrics-only summary bundle into results/runs."""
    rows = build_long_rows(job)
    validate_complete_coverage(rows, job)

    output_dir = RUNS_ROOT / job.output_name
    ensure_path_within(output_dir, RUNS_ROOT, label="summary-metrics output")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    long_table_path = output_dir / "long_table.csv"
    inventory_path = output_dir / "inventory.json"
    manifest_path = output_dir / "manifest.json"
    slice_manifest_path = output_dir / "slice_manifest.json"
    dataframe.to_csv(long_table_path, index=False)
    write_json(inventory_path, build_inventory(dataframe))

    slice_manifest = {
        "source_summary_current_paths": [path_to_display(path) for path in job.summaries],
        "source_run_group_keys": collect_summary_field(job.summaries, "run_group_key"),
        "target_cohort_keys": collect_summary_field(job.summaries, "target_cohort_key"),
        "slice_policy": "largest_complete_prefix",
        "requested_victims": job.requested_victims,
        "requested_victims_source": "config",
        "requested_target_count": len(job.selected_targets),
        "selected_targets": normalize_for_json(job.selected_targets),
        "selected_target_count": len(job.selected_targets),
        "excluded_targets": [],
        "excluded_incomplete_cells": [],
        "fairness_safe": True,
        "generation_timestamp": utc_now_iso(),
    }
    write_json(slice_manifest_path, slice_manifest)

    manifest = {
        "run_id": job.output_name,
        "source_config_path": to_repo_relative(source_config_path),
        "source_summary_current_paths": [path_to_display(path) for path in job.summaries],
        "canonical_columns": CANONICAL_COLUMNS,
        "row_count": int(len(dataframe)),
        "generated_files": [
            "inventory.json",
            "long_table.csv",
            "manifest.json",
            "slice_manifest.json",
        ],
        "slice": slice_manifest,
        "metadata": {
            "dataset": job.dataset,
            "target_type": job.target_type,
            "attack_method": job.attack_method,
            "attack_size": job.attack_size,
            "poison_model": job.poison_model,
            "fake_session_generation_topk": job.fake_session_generation_topk,
            "replacement_topk_ratio": job.replacement_topk_ratio,
        },
        "generation_timestamp": utc_now_iso(),
    }
    write_json(manifest_path, manifest)
    return {
        "output_name": job.output_name,
        "output_dir": output_dir,
        "row_count": int(len(dataframe)),
    }


def build_long_rows(job: SummaryMetricsJobSpec) -> list[dict[str, Any]]:
    """Flatten configured summaries into canonical long-table rows."""
    target_payloads: dict[str, Mapping[str, Any]] = {}
    for summary_path in job.summaries:
        summary = load_json_file(summary_path, label=f"summary {summary_path}")
        validate_summary_victims(summary, job, summary_path=summary_path)
        for raw_target, payload in require_mapping(summary.get("targets"), label=f"{summary_path}.targets").items():
            target_key = str(raw_target)
            if target_key in target_payloads:
                raise AnalysisError(
                    f"Duplicate target '{target_key}' across summaries for output '{job.output_name}'."
                )
            target_payloads[target_key] = require_mapping(
                payload,
                label=f"{summary_path}.targets[{target_key}]",
            )

    selected_keys = [str(target) for target in job.selected_targets]
    missing_targets = [target for target in selected_keys if target not in target_payloads]
    if missing_targets:
        raise AnalysisError(
            f"Missing selected targets for output '{job.output_name}': {missing_targets}."
        )

    rows: list[dict[str, Any]] = []
    for target in job.selected_targets:
        target_key = str(target)
        target_payload = target_payloads[target_key]
        victim_payloads = require_mapping(
            target_payload.get("victims"),
            label=f"{job.output_name}.targets[{target_key}].victims",
        )
        for victim_model in job.requested_victims:
            victim_payload = require_mapping(
                victim_payloads.get(victim_model),
                label=f"{job.output_name}.targets[{target_key}].victims[{victim_model}]",
            )
            if not bool(victim_payload.get("metrics_available")):
                raise AnalysisError(
                    f"Metrics are not available for output '{job.output_name}', "
                    f"target '{target_key}', victim '{victim_model}'."
                )
            metrics = require_mapping(
                victim_payload.get("metrics"),
                label=f"{job.output_name}.targets[{target_key}].victims[{victim_model}].metrics",
            )
            for metric_key, raw_value in metrics.items():
                metric, k_value = parse_metric_key(str(metric_key))
                rows.append(
                    {
                        "run_id": job.output_name,
                        "dataset": job.dataset,
                        "attack_method": job.attack_method,
                        "victim_model": victim_model,
                        "target_item": target,
                        "target_type": job.target_type,
                        "attack_size": job.attack_size,
                        "poison_model": job.poison_model,
                        "fake_session_generation_topk": job.fake_session_generation_topk,
                        "replacement_topk_ratio": job.replacement_topk_ratio,
                        "metric": metric,
                        "k": k_value,
                        "value": require_float(raw_value, label=f"metric {metric_key}"),
                    }
                )
    return rows


def parse_metric_key(metric_key: str) -> tuple[str, int]:
    """Convert summary metric keys into canonical long-table metric/k values."""
    match = METRIC_KEY_PATTERN.match(metric_key)
    if match is None:
        raise AnalysisError(f"Unsupported metric key '{metric_key}'. Expected '<metric>@<k>'.")
    metric = match.group("metric")
    if metric.startswith("targeted_"):
        metric = metric[len("targeted_") :]
    return metric, int(match.group("k"))


def validate_complete_coverage(rows: list[dict[str, Any]], job: SummaryMetricsJobSpec) -> None:
    """Require every selected target/victim pair to have at least one metric row."""
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise AnalysisError(f"No rows were imported for output '{job.output_name}'.")
    observed = {
        (str(row["target_item"]), str(row["victim_model"]))
        for row in rows
    }
    missing = [
        (str(target), victim)
        for target in job.selected_targets
        for victim in job.requested_victims
        if (str(target), victim) not in observed
    ]
    if missing:
        raise AnalysisError(f"Incomplete imported coverage for '{job.output_name}': {missing[:10]}.")


def validate_summary_victims(
    summary: Mapping[str, Any],
    job: SummaryMetricsJobSpec,
    *,
    summary_path: Path,
) -> None:
    """Check summary-level victim declarations when present."""
    raw_victims = summary.get("victims")
    if raw_victims is None:
        return
    if list(raw_victims) != job.requested_victims:
        if sorted(str(value) for value in raw_victims) != sorted(job.requested_victims):
            raise AnalysisError(
                f"Summary '{summary_path}' victims {raw_victims} do not match requested "
                f"victims {job.requested_victims}."
            )


def collect_summary_field(paths: list[Path], field_name: str) -> list[Any]:
    """Collect a scalar field from all summaries."""
    values: list[Any] = []
    for path in paths:
        summary = load_json_file(path, label=f"summary {path}")
        if field_name in summary:
            values.append(summary[field_name])
    return normalize_for_json(values)


def path_to_display(path: Path) -> str:
    """Display repo-relative paths when possible, otherwise absolute paths."""
    try:
        return to_repo_relative(path)
    except ValueError:
        return str(path.resolve())


def load_json_file(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"The {label} is not valid JSON: {exc}.") from exc
    return require_mapping(payload, label=label)


def resolve_existing_file(raw_path: str, *, label: str) -> Path:
    """Resolve and require one existing file."""
    path = resolve_repo_path(raw_path)
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    """Require a mapping."""
    if not isinstance(value, Mapping):
        raise AnalysisError(f"Expected '{label}' to be a mapping, got {type(value).__name__}.")
    return value


def require_scalar_list(value: Any, *, label: str) -> list[int | str]:
    """Require a non-empty scalar list."""
    if not isinstance(value, list) or not value:
        raise AnalysisError(f"Expected '{label}' to be a non-empty list.")
    normalized: list[int | str] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, str)):
            raise AnalysisError(f"Expected '{label}[{index}]' to be an int or string.")
        normalized.append(item)
    return normalized


def require_int(value: Any, *, label: str) -> int:
    """Require an integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise AnalysisError(f"Expected '{label}' to be an integer.")
    return int(value)


def require_float(value: Any, *, label: str) -> float:
    """Require a numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalysisError(f"Expected '{label}' to be numeric.")
    return float(value)


if __name__ == "__main__":
    main()
