"""Export compact RL dynamics and final metrics for GPT-assisted analysis."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from analysis.utils.position_collapse_summary import (
    PositionCollapseSummaryError,
    summarize_position_collapse_file,
)


DEFAULT_K_VALUES = (10, 20, 30)
METRIC_KEYS = tuple(
    f"{scope}_{metric}@{k}"
    for scope in ("targeted", "ground_truth")
    for metric in ("mrr", "recall")
    for k in DEFAULT_K_VALUES
)
DYNAMICS_COLUMNS = [
    "method",
    "target_item",
    "outer_step",
    "reward",
    "baseline",
    "advantage",
    "mean_entropy",
    "policy_loss",
    "target_utility",
    "gt_drop",
    "gt_penalty",
    "average_selected_position",
    "median_selected_position",
    "fraction_pos0",
    "fraction_pos_le_1",
    "fraction_pos_le_2",
    "dominant_position",
    "dominant_share_pct",
    "unique_positions",
    "top_positions",
]
FINAL_METRIC_COLUMNS = ["method", "target_item", "victim", *METRIC_KEYS]


class AttackImprovementExportError(ValueError):
    """Raised when an attack-improvement report cannot be exported."""


def build_attack_improvement_payload(
    *,
    shared_run_root: str | Path,
    mvp_run_root: str | Path,
    prefix_run_root: str | Path,
    target_items: Sequence[str | int],
    victims: Sequence[str],
) -> dict[str, Any]:
    """Collect compact dynamics and metric rows for a GPT-readable report."""
    parsed_targets = _parse_target_items(target_items)
    parsed_victims = _parse_victims(victims)
    run_roots = {
        "Shared": _require_run_root(shared_run_root, "shared_run_root"),
        "MVP": _require_run_root(mvp_run_root, "mvp_run_root"),
        "Prefix": _require_run_root(prefix_run_root, "prefix_run_root"),
    }

    return {
        "target_items": parsed_targets,
        "victims": parsed_victims,
        "shared_training_dynamics": collect_training_dynamics_rows(
            method="Shared",
            run_root=run_roots["Shared"],
            target_items=parsed_targets,
        ),
        "mvp_training_dynamics": collect_training_dynamics_rows(
            method="MVP",
            run_root=run_roots["MVP"],
            target_items=parsed_targets,
        ),
        "final_metrics": collect_final_metric_rows(
            method_run_roots=[
                ("Prefix", run_roots["Prefix"]),
                ("MVP", run_roots["MVP"]),
                ("Shared", run_roots["Shared"]),
            ],
            target_items=parsed_targets,
            victims=parsed_victims,
        ),
    }


def collect_training_dynamics_rows(
    *,
    method: str,
    run_root: str | Path,
    target_items: Sequence[int],
) -> list[dict[str, Any]]:
    """Collect per-step position-optimizer dynamics for one method."""
    root = _require_run_root(run_root, f"{method} run root")
    rows: list[dict[str, Any]] = []
    for target_item in target_items:
        history_path = root / "targets" / str(target_item) / "position_opt" / "training_history.json"
        if not history_path.is_file():
            raise AttackImprovementExportError(
                f"Missing training_history.json for method={method}, target_item={target_item}: {history_path}"
            )
        try:
            summary = summarize_position_collapse_file(history_path)
        except PositionCollapseSummaryError as exc:
            raise AttackImprovementExportError(
                f"Could not summarize training history for method={method}, "
                f"target_item={target_item}: {exc}"
            ) from exc

        for step in _require_rows(summary.get("step_summaries"), "step_summaries"):
            row = {
                "method": method,
                "target_item": int(target_item),
                "outer_step": int(step["outer_step"]),
                "reward": step.get("reward"),
                "baseline": step.get("baseline"),
                "advantage": step.get("advantage"),
                "mean_entropy": step.get("mean_entropy"),
                "policy_loss": step.get("policy_loss"),
                "target_utility": step.get("target_utility"),
                "gt_drop": step.get("gt_drop"),
                "gt_penalty": step.get("gt_penalty"),
                "average_selected_position": step.get("average_selected_position"),
                "median_selected_position": step.get("median_selected_position"),
                "fraction_pos0": step.get("fraction_pos0"),
                "fraction_pos_le_1": step.get("fraction_pos_le_1"),
                "fraction_pos_le_2": step.get("fraction_pos_le_2"),
                "dominant_position": step.get("dominant_position"),
                "dominant_share_pct": step.get("dominant_share_pct"),
                "unique_positions": step.get("unique_positions"),
                "top_positions": _format_top_positions(step.get("top_positions")),
            }
            rows.append(row)
    return rows


def collect_final_metric_rows(
    *,
    method_run_roots: Sequence[tuple[str, Path]],
    target_items: Sequence[int],
    victims: Sequence[str],
) -> list[dict[str, Any]]:
    """Collect final targeted and ground-truth metrics from summary_current.json files."""
    rows: list[dict[str, Any]] = []
    for method, run_root in method_run_roots:
        summary_path = run_root / "summary_current.json"
        summary_payload = _load_json_mapping(summary_path, f"{method} summary_current.json")
        target_lookup = _target_payload_lookup(summary_payload.get("targets"))
        for target_item in target_items:
            target_payload = target_lookup.get(str(target_item))
            if target_payload is None:
                raise AttackImprovementExportError(
                    f"Missing target_item={target_item} in {method} summary_current.json: {summary_path}"
                )
            victim_payloads = _require_mapping(
                target_payload.get("victims"),
                f"{method}.targets[{target_item}].victims",
            )
            for victim in victims:
                victim_payload = _require_mapping(
                    victim_payloads.get(victim),
                    f"{method}.targets[{target_item}].victims[{victim}]",
                )
                metrics = _require_mapping(
                    victim_payload.get("metrics"),
                    f"{method}.targets[{target_item}].victims[{victim}].metrics",
                )
                row: dict[str, Any] = {
                    "method": method,
                    "target_item": int(target_item),
                    "victim": victim,
                }
                for metric_key in METRIC_KEYS:
                    if metric_key not in metrics:
                        raise AttackImprovementExportError(
                            f"Missing metric '{metric_key}' for method={method}, "
                            f"target_item={target_item}, victim={victim}."
                        )
                    row[metric_key] = _require_float(
                        metrics[metric_key],
                        f"{method}.targets[{target_item}].victims[{victim}].metrics.{metric_key}",
                    )
                rows.append(row)
    return rows


def render_markdown_report(payload: Mapping[str, Any]) -> str:
    """Render the collected payload as a single GPT-readable Markdown report."""
    target_items = ", ".join(str(target) for target in payload.get("target_items", []))
    victims = ", ".join(str(victim) for victim in payload.get("victims", []))
    lines = [
        "# Attack Improvement Analysis Export",
        "",
        "## Notes",
        f"- Target items: {target_items}",
        f"- Victims for final metrics: {victims}",
        "- Per-step dynamics are from the position optimizer trained with the SR-GNN surrogate.",
        "- Per-step dynamics are not victim-specific TRON/SRGNN/MiaSRec traces.",
        "- Victim-specific information in this report is limited to final metrics.",
        "- If gt_penalty is disabled in the run config, gt_drop and gt_penalty are expected to be 0.0.",
        "",
        "## Shared Policy Per-Step Training Dynamics",
        "```csv",
        _rows_to_csv(DYNAMICS_COLUMNS, _require_rows(payload.get("shared_training_dynamics"), "shared_training_dynamics")).rstrip(),
        "```",
        "",
        "## MVP Per-Step Training Dynamics",
        "```csv",
        _rows_to_csv(DYNAMICS_COLUMNS, _require_rows(payload.get("mvp_training_dynamics"), "mvp_training_dynamics")).rstrip(),
        "```",
        "",
        "## Final Metrics Wide Table",
        "```csv",
        _rows_to_csv(FINAL_METRIC_COLUMNS, _require_rows(payload.get("final_metrics"), "final_metrics")).rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report(path: str | Path, report: str) -> Path:
    """Write a Markdown report to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the attack-improvement report CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export compact RL dynamics and final metrics as a GPT-readable Markdown report."
    )
    parser.add_argument("--shared-run-root", required=True, help="Shared-policy run-group root.")
    parser.add_argument("--mvp-run-root", required=True, help="Position-opt MVP run-group root.")
    parser.add_argument("--prefix-run-root", required=True, help="Prefix baseline run-group root.")
    parser.add_argument("--target-items", nargs="+", required=True, help="Target items to export.")
    parser.add_argument("--victims", nargs="+", required=True, help="Victim models for final metrics.")
    parser.add_argument("--output-md", help="Optional Markdown output path. Prints to console when omitted.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_attack_improvement_payload(
            shared_run_root=args.shared_run_root,
            mvp_run_root=args.mvp_run_root,
            prefix_run_root=args.prefix_run_root,
            target_items=args.target_items,
            victims=args.victims,
        )
        report = render_markdown_report(payload)
        if args.output_md:
            output_path = write_markdown_report(args.output_md, report)
            print(f"Wrote attack improvement report: {output_path}")
        else:
            print(report)
    except AttackImprovementExportError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def _rows_to_csv(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return buffer.getvalue()


def _format_top_positions(raw_top_positions: Any) -> str:
    if raw_top_positions is None:
        return ""
    rows = _require_rows(raw_top_positions, "top_positions")
    parts = []
    for row in rows:
        parts.append(
            f"p{int(row['position'])}={int(row['count'])}({float(row['share_pct']):.4f}%)"
        )
    return "; ".join(parts)


def _target_payload_lookup(node: Any) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for target_payload in _iter_target_payloads(node):
        target_item = target_payload.get("target_item")
        if target_item is None:
            raise AttackImprovementExportError("Target payload is missing target_item.")
        lookup[str(int(target_item))] = target_payload
    if not lookup:
        raise AttackImprovementExportError("summary_current.json does not contain any target payloads.")
    return lookup


def _iter_target_payloads(node: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        has_target_item = "target_item" in node
        has_victims = "victims" in node
        if has_target_item or has_victims:
            if not (has_target_item and has_victims):
                raise AttackImprovementExportError(
                    "Encountered a partial target payload; expected both target_item and victims."
                )
            yield node
            return
        for child in node.values():
            yield from _iter_target_payloads(child)
        return
    if isinstance(node, list):
        for child in node:
            yield from _iter_target_payloads(child)
        return
    raise AttackImprovementExportError("summary_current.targets must contain target payload objects.")


def _load_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise AttackImprovementExportError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AttackImprovementExportError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    return _require_mapping(payload, label)


def _require_run_root(path: str | Path, label: str) -> Path:
    run_root = Path(path)
    if not run_root.is_dir():
        raise AttackImprovementExportError(f"{label} is not a directory: {run_root}")
    return run_root


def _parse_target_items(values: Sequence[str | int]) -> list[int]:
    if not values:
        raise AttackImprovementExportError("At least one target item is required.")
    parsed: list[int] = []
    for value in values:
        try:
            target_item = int(value)
        except (TypeError, ValueError) as exc:
            raise AttackImprovementExportError(f"target_item must be an integer: {value}") from exc
        parsed.append(target_item)
    return parsed


def _parse_victims(values: Sequence[str]) -> list[str]:
    victims = [str(value).strip() for value in values if str(value).strip()]
    if not victims:
        raise AttackImprovementExportError("At least one victim is required.")
    return victims


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttackImprovementExportError(f"{label} must be a JSON object.")
    return value


def _require_rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise AttackImprovementExportError(f"{label} must be a list.")
    if not all(isinstance(row, Mapping) for row in value):
        raise AttackImprovementExportError(f"{label} must contain objects.")
    return value


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AttackImprovementExportError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AttackImprovementExportError(f"{label} must be numeric.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
