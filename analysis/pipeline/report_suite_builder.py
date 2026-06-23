#!/usr/bin/env python3
"""Run a configured report-building suite in dependency order."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.pipeline.rank_summary_builder import (
    build_rank_summary_bundle,
    parse_rank_summary_spec,
)
from analysis.pipeline.report_table_renderer import parse_render_spec, render_bundle
from analysis.pipeline.view_table_builder import (
    AnalysisError,
    build_view_bundles,
    ensure_path_within,
    load_yaml_mapping,
    require_mapping,
    require_nonempty_string,
    resolve_existing_path,
    resolve_repo_path,
)
from analysis.pipeline.view_table_builder import parse_view_spec


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
STEP_TYPES = {"view", "rank_summary", "render"}


@dataclass(frozen=True)
class SuiteStep:
    """One report-suite step."""

    name: str
    step_type: str
    config_path: Path
    bundle_dir: Path | None


@dataclass(frozen=True)
class ReportSuiteSpec:
    """Validated report-suite config."""

    suite_name: str
    source_spec_path: Path
    steps: list[SuiteStep]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the report-suite CLI parser."""
    parser = argparse.ArgumentParser(
        description="Build and render a configured report suite.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a report-suite YAML config.",
    )
    parser.add_argument(
        "--spec",
        dest="config",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    """Run the report-suite CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        config_path = resolve_existing_path(args.config, label="report-suite config")
        spec = parse_report_suite_spec(
            load_yaml_mapping(config_path, label="report-suite config"),
            source_spec_path=config_path,
        )
        results = run_report_suite(spec)
        print(f"Completed report suite '{spec.suite_name}' with {len(results)} step(s).")
        for result in results:
            print(f"- {result['name']}: {result['output']}")
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def parse_report_suite_spec(
    payload: Mapping[str, Any],
    *,
    source_spec_path: Path,
) -> ReportSuiteSpec:
    """Validate and normalize one report-suite YAML config."""
    suite_name = require_nonempty_string(payload.get("suite_name"), label="suite_name")
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise AnalysisError("Expected 'steps' to be a non-empty list.")

    steps: list[SuiteStep] = []
    for index, raw_step in enumerate(raw_steps):
        step_payload = require_mapping(raw_step, label=f"steps[{index}]")
        step_name = require_nonempty_string(step_payload.get("name"), label=f"steps[{index}].name")
        step_type = require_nonempty_string(step_payload.get("type"), label=f"steps[{index}].type").lower()
        if step_type not in STEP_TYPES:
            raise AnalysisError(
                f"Unsupported steps[{index}].type '{step_type}'. Allowed values: {sorted(STEP_TYPES)}."
            )
        config_path = resolve_existing_path(
            require_nonempty_string(step_payload.get("config"), label=f"steps[{index}].config"),
            label=f"steps[{index}].config",
        )
        bundle_dir = None
        if step_type == "render":
            bundle_dir = resolve_repo_path(
                require_nonempty_string(
                    step_payload.get("bundle_dir"),
                    label=f"steps[{index}].bundle_dir",
                )
            )
            ensure_path_within(bundle_dir, RESULTS_ROOT, label=f"steps[{index}].bundle_dir")
        elif step_payload.get("bundle_dir") is not None:
            raise AnalysisError(f"steps[{index}].bundle_dir is only supported for render steps.")

        steps.append(
            SuiteStep(
                name=step_name,
                step_type=step_type,
                config_path=config_path,
                bundle_dir=bundle_dir,
            )
        )
    return ReportSuiteSpec(
        suite_name=suite_name,
        source_spec_path=source_spec_path,
        steps=steps,
    )


def run_report_suite(spec: ReportSuiteSpec) -> list[dict[str, Any]]:
    """Execute all suite steps in order and return concise outputs."""
    results: list[dict[str, Any]] = []
    for step in spec.steps:
        if step.step_type == "view":
            view_spec = parse_view_spec(
                load_yaml_mapping(step.config_path, label=f"view config for step '{step.name}'"),
                source_spec_path=step.config_path,
            )
            bundle_dirs = build_view_bundles(view_spec)
            results.append(
                {
                    "name": step.name,
                    "type": step.step_type,
                    "output": [str(path) for path in bundle_dirs],
                }
            )
            continue

        if step.step_type == "rank_summary":
            rank_spec = parse_rank_summary_spec(
                load_yaml_mapping(
                    step.config_path,
                    label=f"rank-summary config for step '{step.name}'",
                ),
                source_spec_path=step.config_path,
            )
            result = build_rank_summary_bundle(rank_spec)
            results.append(
                {
                    "name": step.name,
                    "type": step.step_type,
                    "output": str(result["table_path"]),
                }
            )
            continue

        if step.bundle_dir is None:
            raise AnalysisError(f"Render step '{step.name}' is missing bundle_dir.")
        render_spec = parse_render_spec(
            load_yaml_mapping(step.config_path, label=f"render config for step '{step.name}'")
        )
        output_path = render_bundle(bundle_dir=step.bundle_dir, render_spec=render_spec)
        results.append(
            {
                "name": step.name,
                "type": step.step_type,
                "output": str(output_path),
            }
        )
    return results


if __name__ == "__main__":
    main()
