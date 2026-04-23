"""Summarize entropy and dominant-position collapse from training_history.json."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_SHARE_THRESHOLDS_PCT = (50.0, 70.0, 90.0)
DEFAULT_TOP_N = 5


class PositionCollapseSummaryError(ValueError):
    """Raised when a training_history artifact cannot be summarized."""


def resolve_training_history_path(
    *,
    training_history: str | Path | None = None,
    run_root: str | Path | None = None,
    target_item: str | int | None = None,
) -> Path:
    """Resolve training_history.json from either direct or run-root input."""
    has_direct_path = training_history is not None
    has_run_root_input = run_root is not None or target_item is not None
    if has_direct_path and has_run_root_input:
        raise PositionCollapseSummaryError(
            "Use either --training-history or --run-root with --target-item, not both."
        )
    if has_direct_path:
        path = Path(training_history)
    else:
        if run_root is None or target_item is None:
            raise PositionCollapseSummaryError(
                "Provide --training-history, or provide both --run-root and --target-item."
            )
        path = Path(run_root) / "targets" / str(target_item) / "position_opt" / "training_history.json"
    if not path.is_file():
        raise PositionCollapseSummaryError(f"training_history.json not found: {path}")
    return path


def load_training_history(path: str | Path) -> dict[str, Any]:
    """Load a training_history.json file as a mapping."""
    history_path = Path(path)
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PositionCollapseSummaryError(
            f"Invalid JSON in training_history file '{history_path}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PositionCollapseSummaryError("training_history payload must be a JSON object.")
    return payload


def build_position_collapse_summary(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    share_thresholds_pct: Sequence[float] = DEFAULT_SHARE_THRESHOLDS_PCT,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Build a compact collapse summary from a position-opt training history payload."""
    if top_n <= 0:
        raise PositionCollapseSummaryError("top_n must be positive.")
    thresholds = _parse_thresholds(share_thresholds_pct)
    raw_steps = _require_sequence(payload.get("training_history"), "training_history")
    if not raw_steps:
        raise PositionCollapseSummaryError("training_history must contain at least one step.")

    step_rows = [_summarize_step(raw_step, index=index, top_n=top_n) for index, raw_step in enumerate(raw_steps)]
    initial_row = step_rows[0]
    final_row = step_rows[-1]
    initial_entropy = float(initial_row["mean_entropy"])
    final_entropy = float(final_row["mean_entropy"])
    for row in step_rows:
        row["entropy_drop_from_initial"] = initial_entropy - float(row["mean_entropy"])

    min_entropy_row = min(step_rows, key=lambda row: (float(row["mean_entropy"]), int(row["outer_step"])))
    max_entropy_row = max(step_rows, key=lambda row: (float(row["mean_entropy"]), -int(row["outer_step"])))
    max_dominant_row = max(
        step_rows,
        key=lambda row: (float(row["dominant_share_pct"]), -int(row["outer_step"])),
    )

    summary: dict[str, Any] = {
        "source_path": None if source_path is None else str(source_path),
        "target_item": payload.get("target_item"),
        "policy_representation": payload.get("policy_representation"),
        "steps": int(len(step_rows)),
        "sample_count": int(initial_row["sample_count"]),
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "entropy_drop": initial_entropy - final_entropy,
        "entropy_drop_pct": _ratio_pct(initial_entropy - final_entropy, initial_entropy),
        "min_entropy": _step_ref(min_entropy_row, include_dominant=False),
        "max_entropy": _step_ref(max_entropy_row, include_dominant=False),
        "initial_sampled_dominant": _step_ref(initial_row),
        "final_sampled_dominant": _step_ref(final_row),
        "max_sampled_dominant": _step_ref(max_dominant_row),
        "share_threshold_crossings": _threshold_crossings(step_rows, thresholds),
        "step_summaries": step_rows,
        "final_argmax_distribution": _final_argmax_distribution(payload.get("final_selected_positions")),
    }
    return summary


def summarize_position_collapse_file(
    path: str | Path,
    *,
    share_thresholds_pct: Sequence[float] = DEFAULT_SHARE_THRESHOLDS_PCT,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Load and summarize one position-opt training_history.json file."""
    history_path = Path(path)
    return build_position_collapse_summary(
        load_training_history(history_path),
        source_path=history_path,
        share_thresholds_pct=share_thresholds_pct,
        top_n=top_n,
    )


def render_console_summary(summary: Mapping[str, Any]) -> str:
    """Render a compact human-readable collapse summary."""
    lines = [
        "Position Collapse Summary",
        f"source_path: {summary.get('source_path')}",
        f"target_item: {summary.get('target_item')}",
        f"policy_representation: {summary.get('policy_representation')}",
        f"steps: {summary.get('steps')}",
        f"sample_count: {summary.get('sample_count')}",
        (
            "entropy: "
            f"initial={_format_float(summary.get('initial_entropy'))} "
            f"final={_format_float(summary.get('final_entropy'))} "
            f"drop={_format_float(summary.get('entropy_drop'))} "
            f"drop_pct={_format_pct(summary.get('entropy_drop_pct'))}"
        ),
    ]

    lines.append("")
    lines.append("Sampled dominant positions:")
    for label, key in (
        ("initial", "initial_sampled_dominant"),
        ("final", "final_sampled_dominant"),
        ("max", "max_sampled_dominant"),
    ):
        step = summary.get(key)
        if isinstance(step, Mapping):
            lines.append(
                f"  {label}: step={step.get('outer_step')} "
                f"position={step.get('dominant_position')} "
                f"share_pct={_format_pct(step.get('dominant_share_pct'))} "
                f"entropy={_format_float(step.get('mean_entropy'))}"
            )

    lines.append("")
    lines.append("Dominant-share threshold crossings:")
    crossings = summary.get("share_threshold_crossings", {})
    if isinstance(crossings, Mapping):
        for key, step in crossings.items():
            if isinstance(step, Mapping):
                lines.append(
                    f"  {key}: step={step.get('outer_step')} "
                    f"position={step.get('dominant_position')} "
                    f"share_pct={_format_pct(step.get('dominant_share_pct'))} "
                    f"entropy={_format_float(step.get('mean_entropy'))}"
                )
            else:
                lines.append(f"  {key}: never")

    final_distribution = summary.get("final_argmax_distribution")
    if isinstance(final_distribution, Mapping):
        dominant = final_distribution.get("dominant")
        lines.append("")
        lines.append("Final argmax selected-position distribution:")
        lines.append(
            f"  total={final_distribution.get('total')} "
            f"unique_positions={final_distribution.get('unique_positions')}"
        )
        if isinstance(dominant, Mapping):
            lines.append(
                f"  dominant: position={dominant.get('position')} "
                f"count={dominant.get('count')} "
                f"share_pct={_format_pct(dominant.get('share_pct'))}"
            )

    lines.append("")
    lines.append("Per-step sampled summary:")
    lines.append(
        "  step  entropy   reward       avg_pos  med_pos  dom_pos  "
        "dom_share_pct  frac_pos0  frac_pos<=1  frac_pos<=2  unique"
    )
    for row in _require_rows(summary.get("step_summaries"), "step_summaries"):
        lines.append(
            f"  {int(row['outer_step']):>4}  "
            f"{float(row['mean_entropy']):>7.4f}  "
            f"{_format_float(row.get('reward'), width=10)}  "
            f"{float(row['average_selected_position']):>7.3f}  "
            f"{float(row['median_selected_position']):>7.3f}  "
            f"{int(row['dominant_position']):>7}  "
            f"{float(row['dominant_share_pct']):>13.2f}  "
            f"{float(row['fraction_pos0']):>9.4f}  "
            f"{float(row['fraction_pos_le_1']):>11.4f}  "
            f"{float(row['fraction_pos_le_2']):>11.4f}  "
            f"{int(row['unique_positions']):>6}"
        )
    return "\n".join(lines)


def write_summary_json(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write the compact collapse summary JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_step_csv(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write per-step entropy and dominant-position metrics to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _require_rows(summary.get("step_summaries"), "step_summaries")
    fieldnames = [
        "outer_step",
        "reward",
        "baseline",
        "advantage",
        "mean_entropy",
        "policy_loss",
        "target_utility",
        "gt_drop",
        "gt_penalty",
        "entropy_drop_from_initial",
        "average_selected_position",
        "median_selected_position",
        "fraction_pos0",
        "fraction_pos_le_1",
        "fraction_pos_le_2",
        "sample_count",
        "unique_positions",
        "dominant_position",
        "dominant_count",
        "dominant_share_pct",
        "pos0_count",
        "pos0_share_pct",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the position-collapse summary CLI parser."""
    parser = argparse.ArgumentParser(
        description="Summarize entropy and dominant-position collapse from training_history.json."
    )
    parser.add_argument("--training-history", help="Direct path to a training_history.json file.")
    parser.add_argument(
        "--run-root",
        help="Run-group root containing targets/<target_item>/position_opt/training_history.json.",
    )
    parser.add_argument("--target-item", help="Target item used with --run-root.")
    parser.add_argument("--output-json", help="Optional compact summary JSON output path.")
    parser.add_argument("--output-csv", help="Optional per-step summary CSV output path.")
    parser.add_argument(
        "--threshold-pct",
        type=float,
        action="append",
        dest="thresholds_pct",
        help=(
            "Dominant-share threshold percentage. Can be repeated. "
            f"Default: {', '.join(str(value) for value in DEFAULT_SHARE_THRESHOLDS_PCT)}."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of most frequent sampled positions to store per step. Default: {DEFAULT_TOP_N}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        history_path = resolve_training_history_path(
            training_history=args.training_history,
            run_root=args.run_root,
            target_item=args.target_item,
        )
        thresholds = (
            DEFAULT_SHARE_THRESHOLDS_PCT
            if args.thresholds_pct is None
            else tuple(float(value) for value in args.thresholds_pct)
        )
        summary = summarize_position_collapse_file(
            history_path,
            share_thresholds_pct=thresholds,
            top_n=int(args.top_n),
        )
        print(render_console_summary(summary))
        if args.output_json:
            write_summary_json(args.output_json, summary)
        if args.output_csv:
            write_step_csv(args.output_csv, summary)
    except PositionCollapseSummaryError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def _summarize_step(raw_step: Any, *, index: int, top_n: int) -> dict[str, Any]:
    step = _require_mapping(raw_step, f"training_history[{index}]")
    outer_step = _require_int(step.get("outer_step"), f"training_history[{index}].outer_step")
    mean_entropy = _require_float(step.get("mean_entropy"), f"training_history[{index}].mean_entropy")
    selected_positions = _parse_position_sequence(
        step.get("selected_positions"),
        f"training_history[{index}].selected_positions",
    )
    counts = Counter(selected_positions)
    sample_count = len(selected_positions)
    dominant_position, dominant_count = _dominant_count(counts)
    pos0_count = int(counts.get(0, 0))
    pos_le_1_count = sum(1 for position in selected_positions if position <= 1)
    pos_le_2_count = sum(1 for position in selected_positions if position <= 2)
    top_positions = _distribution_rows(counts, sample_count)[:top_n]

    return {
        "outer_step": outer_step,
        "mean_entropy": mean_entropy,
        "reward": _optional_float(step.get("reward")),
        "baseline": _optional_float(step.get("baseline")),
        "advantage": _optional_float(step.get("advantage")),
        "policy_loss": _optional_float(step.get("policy_loss")),
        "target_utility": _optional_float(step.get("target_utility")),
        "gt_drop": _optional_float(step.get("gt_drop")),
        "gt_penalty": _optional_float(step.get("gt_penalty")),
        "sample_count": int(sample_count),
        "unique_positions": int(len(counts)),
        "average_selected_position": float(sum(selected_positions)) / float(sample_count),
        "median_selected_position": float(median(selected_positions)),
        "dominant_position": int(dominant_position),
        "dominant_count": int(dominant_count),
        "dominant_share_pct": _ratio_pct(dominant_count, sample_count),
        "pos0_count": pos0_count,
        "pos0_share_pct": _ratio_pct(pos0_count, sample_count),
        "fraction_pos0": _ratio(pos0_count, sample_count),
        "fraction_pos_le_1": _ratio(pos_le_1_count, sample_count),
        "fraction_pos_le_2": _ratio(pos_le_2_count, sample_count),
        "top_positions": top_positions,
    }


def _final_argmax_distribution(raw_final_positions: Any) -> dict[str, Any] | None:
    if raw_final_positions is None:
        return None
    raw_positions = _require_sequence(raw_final_positions, "final_selected_positions")
    if not raw_positions:
        return {
            "total": 0,
            "unique_positions": 0,
            "dominant": None,
            "positions": [],
        }

    positions: list[int] = []
    for index, raw_entry in enumerate(raw_positions):
        entry = _require_mapping(raw_entry, f"final_selected_positions[{index}]")
        positions.append(_require_position(entry.get("position"), f"final_selected_positions[{index}].position"))
    counts = Counter(positions)
    total = len(positions)
    dominant_position, dominant_count = _dominant_count(counts)
    rows = _distribution_rows(counts, total)
    return {
        "total": int(total),
        "unique_positions": int(len(counts)),
        "dominant": {
            "position": int(dominant_position),
            "count": int(dominant_count),
            "share_pct": _ratio_pct(dominant_count, total),
        },
        "positions": rows,
    }


def _threshold_crossings(
    rows: Sequence[Mapping[str, Any]],
    thresholds_pct: Sequence[float],
) -> dict[str, dict[str, Any] | None]:
    output: dict[str, dict[str, Any] | None] = {}
    for threshold in thresholds_pct:
        crossing = next(
            (row for row in rows if float(row["dominant_share_pct"]) >= float(threshold)),
            None,
        )
        output[f">={threshold:g}%"] = None if crossing is None else _step_ref(crossing)
    return output


def _step_ref(row: Mapping[str, Any], *, include_dominant: bool = True) -> dict[str, Any]:
    output: dict[str, Any] = {
        "outer_step": int(row["outer_step"]),
        "mean_entropy": float(row["mean_entropy"]),
    }
    if include_dominant:
        output.update(
            {
                "dominant_position": int(row["dominant_position"]),
                "dominant_count": int(row["dominant_count"]),
                "dominant_share_pct": float(row["dominant_share_pct"]),
                "pos0_share_pct": float(row["pos0_share_pct"]),
                "unique_positions": int(row["unique_positions"]),
            }
        )
    return output


def _distribution_rows(counts: Counter[int], total: int) -> list[dict[str, int | float]]:
    if total <= 0:
        raise PositionCollapseSummaryError("position distribution total must be positive.")
    return [
        {
            "position": int(position),
            "count": int(count),
            "share_pct": _ratio_pct(count, total),
        }
        for position, count in sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))
    ]


def _dominant_count(counts: Counter[int]) -> tuple[int, int]:
    if not counts:
        raise PositionCollapseSummaryError("selected_positions must contain at least one position.")
    position, count = sorted(counts.items(), key=lambda item: (-int(item[1]), int(item[0])))[0]
    return int(position), int(count)


def _parse_thresholds(values: Sequence[float]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        threshold = _require_float(value, "threshold_pct")
        if threshold <= 0.0 or threshold > 100.0:
            raise PositionCollapseSummaryError(
                f"threshold_pct must be in the range (0, 100]; got {threshold}."
            )
        parsed.append(threshold)
    if not parsed:
        raise PositionCollapseSummaryError("At least one threshold_pct is required.")
    return tuple(sorted(set(parsed)))


def _parse_position_sequence(value: Any, label: str) -> list[int]:
    sequence = _require_sequence(value, label)
    if not sequence:
        raise PositionCollapseSummaryError(f"{label} must contain at least one position.")
    return [_require_position(raw_position, f"{label}[{index}]") for index, raw_position in enumerate(sequence)]


def _require_position(value: Any, label: str) -> int:
    position = _require_int(value, label)
    if position < 0:
        raise PositionCollapseSummaryError(f"{label} must be non-negative; got {position}.")
    return position


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PositionCollapseSummaryError(f"training_history is missing required object: {label}.")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PositionCollapseSummaryError(f"training_history is missing required array: {label}.")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PositionCollapseSummaryError(f"{label} must be an integer.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PositionCollapseSummaryError(f"{label} must be an integer.") from exc


def _require_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PositionCollapseSummaryError(f"{label} must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PositionCollapseSummaryError(f"{label} must be numeric.") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _require_float(value, "optional numeric field")


def _ratio_pct(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        raise PositionCollapseSummaryError("Cannot compute percentage with zero denominator.")
    return float(numerator) / float(denominator) * 100.0


def _ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        raise PositionCollapseSummaryError("Cannot compute ratio with zero denominator.")
    return float(numerator) / float(denominator)


def _require_rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise PositionCollapseSummaryError(f"summary field '{label}' must be a list.")
    if not all(isinstance(row, Mapping) for row in value):
        raise PositionCollapseSummaryError(f"summary field '{label}' must contain objects.")
    return value


def _format_float(value: Any, width: int | None = None) -> str:
    if value is None:
        formatted = "None"
    else:
        try:
            formatted = f"{float(value):.6f}"
        except (TypeError, ValueError):
            formatted = "None"
    if width is None:
        return formatted
    return f"{formatted:>{width}}"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "None"


if __name__ == "__main__":
    raise SystemExit(main())
