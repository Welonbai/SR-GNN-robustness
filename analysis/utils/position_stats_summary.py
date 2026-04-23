"""Summarize one target-level position_stats.json artifact."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_CUMULATIVE_THRESHOLDS = (0, 1, 2, 5)
DEFAULT_TOP_N = 5


class PositionStatsSummaryError(ValueError):
    """Raised when a position_stats artifact cannot be summarized."""


def resolve_position_stats_path(
    *,
    position_stats: str | Path | None = None,
    run_root: str | Path | None = None,
    target_item: str | int | None = None,
) -> Path:
    """Resolve the position_stats.json path from either direct or run-root input."""
    has_direct_path = position_stats is not None
    has_run_root_input = run_root is not None or target_item is not None
    if has_direct_path and has_run_root_input:
        raise PositionStatsSummaryError(
            "Use either --position-stats or --run-root with --target-item, not both."
        )
    if has_direct_path:
        path = Path(position_stats)
    else:
        if run_root is None or target_item is None:
            raise PositionStatsSummaryError(
                "Provide --position-stats, or provide both --run-root and --target-item."
            )
        path = Path(run_root) / "targets" / str(target_item) / "position_stats.json"
    if not path.is_file():
        raise PositionStatsSummaryError(f"position_stats.json not found: {path}")
    return path


def load_position_stats(path: str | Path) -> dict[str, Any]:
    """Load a position_stats.json file as a mapping."""
    stats_path = Path(path)
    try:
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PositionStatsSummaryError(
            f"Invalid JSON in position_stats file '{stats_path}': {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PositionStatsSummaryError("position_stats payload must be a JSON object.")
    return payload


def build_position_stats_summary(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    include_session_lengths: bool = False,
    cumulative_thresholds: Sequence[int] = DEFAULT_CUMULATIVE_THRESHOLDS,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Build a compact summary from a position_stats payload."""
    if top_n <= 0:
        raise PositionStatsSummaryError("top_n must be positive.")
    total_sessions = _require_positive_int(payload.get("total_sessions"), "total_sessions")
    overall = _require_mapping(payload.get("overall"), "overall")
    counts = _parse_int_mapping(_require_mapping(overall.get("counts"), "overall.counts"), "overall.counts")
    ratios = _parse_float_mapping(
        _require_mapping(overall.get("ratios"), "overall.ratios"),
        "overall.ratios",
    )
    _validate_position_keys(counts, ratios)
    count_total = sum(counts.values())
    if count_total != total_sessions:
        raise PositionStatsSummaryError(
            f"overall.counts sum ({count_total}) does not match total_sessions ({total_sessions})."
        )

    rows = _position_distribution_rows(counts, ratios, total_sessions)
    top_positions = sorted(rows, key=lambda row: (-int(row["count"]), int(row["position"])))[:top_n]
    max_position_share = top_positions[0] if top_positions else None

    summary: dict[str, Any] = {
        "source_path": None if source_path is None else str(source_path),
        "run_type": _optional_string(payload.get("run_type")),
        "target_item": payload.get("target_item"),
        "total_sessions": total_sessions,
        "unique_selected_position_count": int(len(rows)),
        "max_position_share": max_position_share,
        "cumulative_ratios": _cumulative_ratios(
            counts,
            total_sessions=total_sessions,
            thresholds=cumulative_thresholds,
        ),
        "top_positions": top_positions,
        "positions": rows,
    }
    if include_session_lengths:
        summary["by_session_length"] = _session_length_summary(payload)
    return summary


def summarize_position_stats_file(
    path: str | Path,
    *,
    include_session_lengths: bool = False,
    cumulative_thresholds: Sequence[int] = DEFAULT_CUMULATIVE_THRESHOLDS,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Load and summarize one position_stats.json file."""
    stats_path = Path(path)
    return build_position_stats_summary(
        load_position_stats(stats_path),
        source_path=stats_path,
        include_session_lengths=include_session_lengths,
        cumulative_thresholds=cumulative_thresholds,
        top_n=top_n,
    )


def render_console_summary(summary: Mapping[str, Any], *, include_session_lengths: bool = False) -> str:
    """Render a compact human-readable summary."""
    lines = [
        "Position Stats Summary",
        f"source_path: {summary.get('source_path')}",
        f"run_type: {summary.get('run_type')}",
        f"target_item: {summary.get('target_item')}",
        f"total_sessions: {summary.get('total_sessions')}",
        f"unique_selected_position_count: {summary.get('unique_selected_position_count')}",
    ]
    max_share = summary.get("max_position_share")
    if isinstance(max_share, Mapping):
        lines.append(
            "max_position_share: "
            f"position={max_share.get('position')} "
            f"count={max_share.get('count')} "
            f"ratio_pct={_format_pct(max_share.get('ratio_pct'))}"
        )

    lines.append("")
    lines.append("Cumulative ratios:")
    cumulative_ratios = summary.get("cumulative_ratios", {})
    if isinstance(cumulative_ratios, Mapping):
        for key, payload in cumulative_ratios.items():
            if isinstance(payload, Mapping):
                lines.append(
                    f"  {key}: count={payload.get('count')} "
                    f"ratio_pct={_format_pct(payload.get('ratio_pct'))}"
                )

    lines.append("")
    lines.append("Position distribution:")
    lines.append("  position  count  ratio_pct")
    for row in _require_rows(summary.get("positions"), "positions"):
        lines.append(
            f"  {int(row['position']):>8}  {int(row['count']):>5}  "
            f"{float(row['ratio_pct']):>9.4f}"
        )

    lines.append("")
    lines.append("Top positions by count:")
    for row in _require_rows(summary.get("top_positions"), "top_positions"):
        lines.append(
            f"  position={int(row['position'])} count={int(row['count'])} "
            f"ratio_pct={float(row['ratio_pct']):.4f}"
        )

    if include_session_lengths and isinstance(summary.get("by_session_length"), list):
        lines.append("")
        lines.append("By session length:")
        lines.append("  length  sessions  dominant_position  dominant_ratio_pct")
        for row in _require_rows(summary.get("by_session_length"), "by_session_length"):
            lines.append(
                f"  {int(row['session_length']):>6}  {int(row['session_count']):>8}  "
                f"{int(row['dominant_position']):>17}  "
                f"{float(row['dominant_ratio_pct']):>18.4f}"
            )
    return "\n".join(lines)


def write_summary_json(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write the compact summary JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def write_distribution_csv(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write the overall position distribution CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _require_rows(summary.get("positions"), "positions")
    fieldnames = ["position", "count", "ratio", "ratio_pct", "cumulative_count", "cumulative_ratio_pct"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        cumulative_count = 0
        total_sessions = int(summary["total_sessions"])
        for row in rows:
            cumulative_count += int(row["count"])
            writer.writerow(
                {
                    "position": int(row["position"]),
                    "count": int(row["count"]),
                    "ratio": float(row["ratio"]),
                    "ratio_pct": float(row["ratio_pct"]),
                    "cumulative_count": int(cumulative_count),
                    "cumulative_ratio_pct": float(cumulative_count) / float(total_sessions) * 100.0,
                }
            )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the position-stats summary CLI parser."""
    parser = argparse.ArgumentParser(description="Summarize one target-level position_stats.json artifact.")
    parser.add_argument("--position-stats", help="Direct path to a position_stats.json file.")
    parser.add_argument("--run-root", help="Run-group root containing targets/<target_item>/position_stats.json.")
    parser.add_argument("--target-item", help="Target item used with --run-root.")
    parser.add_argument("--output-json", help="Optional compact summary JSON output path.")
    parser.add_argument("--output-csv", help="Optional position distribution CSV output path.")
    parser.add_argument(
        "--include-session-lengths",
        action="store_true",
        help="Include compact by_session_length summaries in console and JSON output.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of most frequent positions to list. Default: {DEFAULT_TOP_N}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        stats_path = resolve_position_stats_path(
            position_stats=args.position_stats,
            run_root=args.run_root,
            target_item=args.target_item,
        )
        summary = summarize_position_stats_file(
            stats_path,
            include_session_lengths=bool(args.include_session_lengths),
            top_n=int(args.top_n),
        )
        print(render_console_summary(summary, include_session_lengths=bool(args.include_session_lengths)))
        if args.output_json:
            write_summary_json(args.output_json, summary)
        if args.output_csv:
            write_distribution_csv(args.output_csv, summary)
    except PositionStatsSummaryError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PositionStatsSummaryError(f"position_stats is missing required object: {label}.")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise PositionStatsSummaryError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PositionStatsSummaryError(f"{label} must be a positive integer.") from exc
    if parsed <= 0:
        raise PositionStatsSummaryError(f"{label} must be positive; got {parsed}.")
    return parsed


def _parse_int_mapping(value: Mapping[str, Any], label: str) -> dict[int, int]:
    parsed: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        try:
            key = int(raw_key)
            count = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise PositionStatsSummaryError(f"{label} must map integer positions to integer counts.") from exc
        if key < 0:
            raise PositionStatsSummaryError(f"{label} contains a negative position: {key}.")
        if count < 0:
            raise PositionStatsSummaryError(f"{label} contains a negative count for position {key}.")
        parsed[key] = count
    if not parsed:
        raise PositionStatsSummaryError(f"{label} must contain at least one position.")
    return dict(sorted(parsed.items()))


def _parse_float_mapping(value: Mapping[str, Any], label: str) -> dict[int, float]:
    parsed: dict[int, float] = {}
    for raw_key, raw_value in value.items():
        try:
            key = int(raw_key)
            ratio = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise PositionStatsSummaryError(f"{label} must map integer positions to numeric ratios.") from exc
        if key < 0:
            raise PositionStatsSummaryError(f"{label} contains a negative position: {key}.")
        if ratio < 0:
            raise PositionStatsSummaryError(f"{label} contains a negative ratio for position {key}.")
        parsed[key] = ratio
    if not parsed:
        raise PositionStatsSummaryError(f"{label} must contain at least one position.")
    return dict(sorted(parsed.items()))


def _validate_position_keys(counts: Mapping[int, int], ratios: Mapping[int, float]) -> None:
    if set(counts) != set(ratios):
        missing_ratios = sorted(set(counts) - set(ratios))
        missing_counts = sorted(set(ratios) - set(counts))
        raise PositionStatsSummaryError(
            "overall.counts and overall.ratios must contain the same positions. "
            f"Missing ratios: {missing_ratios}; missing counts: {missing_counts}."
        )


def _position_distribution_rows(
    counts: Mapping[int, int],
    ratios: Mapping[int, float],
    total_sessions: int,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for position in sorted(counts):
        ratio = float(ratios[position])
        expected_ratio = float(counts[position]) / float(total_sessions)
        if abs(ratio - expected_ratio) > 1.0e-9:
            raise PositionStatsSummaryError(
                f"overall.ratios[{position}]={ratio} does not match count/total={expected_ratio}."
            )
        rows.append(
            {
                "position": int(position),
                "count": int(counts[position]),
                "ratio": ratio,
                "ratio_pct": ratio * 100.0,
            }
        )
    return rows


def _cumulative_ratios(
    counts: Mapping[int, int],
    *,
    total_sessions: int,
    thresholds: Sequence[int],
) -> dict[str, dict[str, int | float]]:
    output: dict[str, dict[str, int | float]] = {}
    for threshold in thresholds:
        parsed_threshold = int(threshold)
        count = sum(position_count for position, position_count in counts.items() if int(position) <= parsed_threshold)
        ratio = float(count) / float(total_sessions)
        output[f"<={parsed_threshold}"] = {
            "count": int(count),
            "ratio": ratio,
            "ratio_pct": ratio * 100.0,
        }
    return output


def _session_length_summary(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_session_length = _require_mapping(payload.get("by_session_length"), "by_session_length")
    rows: list[dict[str, Any]] = []
    for raw_length, raw_entry in sorted(by_session_length.items(), key=lambda item: int(item[0])):
        session_length = int(raw_length)
        entry = _require_mapping(raw_entry, f"by_session_length.{session_length}")
        session_count = _require_positive_int(
            entry.get("session_count"),
            f"by_session_length.{session_length}.session_count",
        )
        counts = _parse_int_mapping(
            _require_mapping(entry.get("position_counts"), f"by_session_length.{session_length}.position_counts"),
            f"by_session_length.{session_length}.position_counts",
        )
        ratios = _parse_float_mapping(
            _require_mapping(entry.get("position_ratios"), f"by_session_length.{session_length}.position_ratios"),
            f"by_session_length.{session_length}.position_ratios",
        )
        _validate_position_keys(counts, ratios)
        if sum(counts.values()) != session_count:
            raise PositionStatsSummaryError(
                f"by_session_length.{session_length}.position_counts sum does not match session_count."
            )
        distribution = _position_distribution_rows(counts, ratios, session_count)
        dominant = sorted(distribution, key=lambda row: (-int(row["count"]), int(row["position"])))[0]
        rows.append(
            {
                "session_length": int(session_length),
                "session_count": int(session_count),
                "unique_selected_position_count": int(len(distribution)),
                "dominant_position": int(dominant["position"]),
                "dominant_count": int(dominant["count"]),
                "dominant_ratio_pct": float(dominant["ratio_pct"]),
                "position_distribution": distribution,
            }
        )
    return rows


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise PositionStatsSummaryError(f"summary field '{label}' must be a list.")
    if not all(isinstance(row, Mapping) for row in value):
        raise PositionStatsSummaryError(f"summary field '{label}' must contain objects.")
    return value


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "None"


if __name__ == "__main__":
    raise SystemExit(main())
