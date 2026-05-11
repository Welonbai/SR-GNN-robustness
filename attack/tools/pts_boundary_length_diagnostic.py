from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import sys
from typing import Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import load_fake_sessions
from attack.common.config import Config, load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)


DEFAULT_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_elite_centered_ratio1_"
    "srgnn_partial4_target5334.yaml"
)
DEFAULT_OUTPUT_DIR = "outputs/diagnostics/pts_boundary_length_diagnostic"
LENGTH_BINS = [
    ("L=2", 2, 2),
    ("L=3", 3, 3),
    ("L=4", 4, 4),
    ("L=5", 5, 5),
    ("L=6", 6, 6),
    ("L=7", 7, 7),
    ("L=8", 8, 8),
    ("L=9", 9, 9),
    ("L>=10", 10, None),
]
INTERNAL_GROUPS = ("suffix_1", "suffix_2", "suffix_3plus")
NONZERO_GROUPS = ("suffix_0", "suffix_1", "suffix_2", "suffix_3plus")


@dataclass(frozen=True)
class SelectedFakeSessions:
    path: Path
    expected_path: Path
    discovered_paths: list[Path]
    selection_reason: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return (_repo_root() / path_obj).resolve()


def _repo_relative(path: str | Path) -> str:
    path_obj = Path(path).resolve()
    try:
        return path_obj.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(path_obj)


def _format_float(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _format_percent(value: float) -> str:
    return f"{float(value):.2f}%"


def _percent(part: float, total: float) -> float:
    if total == 0:
        return 0.0
    return 100.0 * float(part) / float(total)


def _bin_label(length: int) -> str:
    for label, lower, upper in LENGTH_BINS:
        if length >= lower and (upper is None or length <= upper):
            return label
    raise ValueError(f"Length {length} is below the supported diagnostic bins.")


def _empty_bin_counter() -> dict[str, float]:
    return {label: 0.0 for label, _, _ in LENGTH_BINS}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    def stringify(value: object) -> str:
        if isinstance(value, float):
            return _format_float(value)
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(stringify(value) for value in row) + " |")
    return "\n".join(lines)


def _discover_fake_session_paths() -> list[Path]:
    output_root = _repo_root() / "outputs"
    if not output_root.exists():
        return []
    return sorted(path.resolve() for path in output_root.rglob("fake_sessions.pkl"))


def _expected_fake_sessions_path(config: Config) -> Path:
    paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    return _repo_path(paths["fake_sessions"])


def _select_fake_sessions(
    config: Config,
    *,
    explicit_path: str | Path | None,
) -> SelectedFakeSessions:
    expected_path = _expected_fake_sessions_path(config)
    discovered_paths = _discover_fake_session_paths()
    if explicit_path is not None:
        selected = _repo_path(explicit_path)
        reason = "explicit --fake-sessions-path"
    elif expected_path.exists():
        selected = expected_path
        reason = "config-resolved PTS-CEM shared fake-session artifact"
    elif len(discovered_paths) == 1:
        selected = discovered_paths[0]
        reason = "only discovered fake-session artifact"
    else:
        discovered = "\n".join(f"  - {_repo_relative(path)}" for path in discovered_paths)
        raise FileNotFoundError(
            "Could not find the config-resolved fake-session artifact:\n"
            f"  {_repo_relative(expected_path)}\n"
            "Discovered candidates:\n"
            f"{discovered or '  <none>'}\n"
            "Pass --fake-sessions-path to select one explicitly."
        )
    if not selected.exists():
        raise FileNotFoundError(f"Selected fake sessions file does not exist: {selected}")
    return SelectedFakeSessions(
        path=selected,
        expected_path=expected_path,
        discovered_paths=discovered_paths,
        selection_reason=reason,
    )


def _load_lengths(fake_sessions_path: Path) -> list[int]:
    sessions = load_fake_sessions(fake_sessions_path)
    if sessions is None:
        raise FileNotFoundError(f"Fake sessions not found: {fake_sessions_path}")
    lengths = [len(session) for session in sessions]
    invalid_count = sum(1 for length in lengths if length < 2)
    if invalid_count:
        raise ValueError(
            "PTS boundary diagnostics require templates with length >= 2; "
            f"found {invalid_count} shorter templates in {fake_sessions_path}."
        )
    return lengths


def _template_length_rows(lengths: Sequence[int]) -> list[dict[str, object]]:
    counts = Counter(lengths)
    total = len(lengths)
    rows: list[dict[str, object]] = []
    for label, lower, upper in LENGTH_BINS:
        count = sum(
            value
            for length, value in counts.items()
            if length >= lower and (upper is None or length <= upper)
        )
        rows.append(
            {
                "bin": label,
                "lower_inclusive": lower,
                "upper_inclusive": "" if upper is None else upper,
                "count": int(count),
                "percent": _percent(count, total),
            }
        )
    return rows


def _internal_group_for_residual(residual_len: int) -> str:
    if residual_len == 1:
        return "suffix_1"
    if residual_len == 2:
        return "suffix_2"
    if residual_len >= 3:
        return "suffix_3plus"
    raise ValueError(f"Internal boundary residual length must be >= 1, got {residual_len}.")


def _nonzero_group_for_residual(residual_len: int) -> str:
    if residual_len == 0:
        return "suffix_0"
    if residual_len == 1:
        return "suffix_1"
    if residual_len == 2:
        return "suffix_2"
    if residual_len >= 3:
        return "suffix_3plus"
    raise ValueError(f"Residual length must be non-negative, got {residual_len}.")


def _internal_boundary_distribution(lengths: Sequence[int]) -> dict[str, float]:
    expected = {group: 0.0 for group in INTERNAL_GROUPS}
    for length in lengths:
        weight = 1.0 / float(length - 1)
        for boundary in range(1, length):
            residual_len = length - boundary
            expected[_internal_group_for_residual(residual_len)] += weight
    return expected


def _nonzero_boundary_distribution(lengths: Sequence[int]) -> dict[str, float]:
    expected = {group: 0.0 for group in NONZERO_GROUPS}
    for length in lengths:
        weight = 1.0 / float(length)
        for boundary in range(1, length + 1):
            residual_len = length - boundary
            expected[_nonzero_group_for_residual(residual_len)] += weight
    return expected


def _boundary_distribution_rows(
    lengths: Sequence[int],
    internal_counts: Mapping[str, float],
    nonzero_counts: Mapping[str, float],
) -> list[dict[str, object]]:
    total = float(len(lengths))
    rows: list[dict[str, object]] = []
    for group in NONZERO_GROUPS:
        internal_count = internal_counts.get(group)
        internal_percent = (
            None if internal_count is None else _percent(internal_count, total)
        )
        nonzero_count = nonzero_counts[group]
        nonzero_percent = _percent(nonzero_count, total)
        rows.append(
            {
                "group": group,
                "internal_expected_count": "" if internal_count is None else internal_count,
                "internal_expected_percent": "" if internal_percent is None else internal_percent,
                "nonzero_expected_count": nonzero_count,
                "nonzero_expected_percent": nonzero_percent,
                "delta_percentage_points": (
                    "" if internal_percent is None else nonzero_percent - internal_percent
                ),
                "note": (
                    "not present in internal-boundary mode"
                    if internal_count is None
                    else ""
                ),
            }
        )
    return rows


def _short_session_rows(lengths: Sequence[int]) -> list[dict[str, object]]:
    total = len(lengths)
    counts = Counter(lengths)
    rows: list[dict[str, object]] = []
    for length in (2, 3, 4):
        count = int(counts.get(length, 0))
        row: dict[str, object] = {
            "template_length": length,
            "template_count": count,
            "template_percent": _percent(count, total),
            "possible_nonzero_groups": ", ".join(
                _nonzero_group_for_residual(length - boundary)
                for boundary in range(1, length + 1)
            ),
        }
        contributions = {group: 0.0 for group in NONZERO_GROUPS}
        if count:
            weight = float(count) / float(length)
            for boundary in range(1, length + 1):
                residual_len = length - boundary
                contributions[_nonzero_group_for_residual(residual_len)] += weight
        for group in NONZERO_GROUPS:
            row[f"{group}_expected_count"] = contributions[group]
            row[f"{group}_expected_percent_of_all_templates"] = _percent(
                contributions[group],
                total,
            )
        rows.append(row)
    return rows


def _final_length_for_vertex(vertex: str, *, length: int, residual_len: int) -> int | None:
    group = _nonzero_group_for_residual(residual_len)
    boundary = length - residual_len
    if vertex == "c0_preserve":
        if group == "suffix_0":
            return None
        return length + 1
    if vertex == "c0_generate":
        if group == "suffix_0":
            return None
        return length + 1
    if vertex == "c1_preserve":
        if group in {"suffix_0", "suffix_1"}:
            return boundary + 1
        return length
    if vertex == "c1_generate":
        if group in {"suffix_0", "suffix_1"}:
            return boundary + 1
        return length
    if vertex == "stop":
        return boundary + 1
    raise ValueError(f"Unsupported vertex: {vertex}")


def _vertex_action_mapping(vertex: str) -> str:
    if vertex == "c0_preserve":
        return "A0 where valid; suffix_0 excluded"
    if vertex == "c0_generate":
        return "A1 where valid; suffix_0 excluded"
    if vertex == "c1_preserve":
        return "suffix_0/suffix_1 -> A4; suffix_2/suffix_3plus -> A2"
    if vertex == "c1_generate":
        return "suffix_0/suffix_1 -> A4; suffix_2/suffix_3plus -> A3"
    if vertex == "stop":
        return "A4 for all groups"
    raise ValueError(f"Unsupported vertex: {vertex}")


def _expected_final_length_rows(lengths: Sequence[int]) -> list[dict[str, object]]:
    template_mean = mean(lengths)
    total_sessions = float(len(lengths))
    rows: list[dict[str, object]] = []
    for vertex in ("c0_preserve", "c0_generate", "c1_preserve", "c1_generate", "stop"):
        final_hist = _empty_bin_counter()
        included_weight = 0.0
        final_length_sum = 0.0
        for length in lengths:
            weight = 1.0 / float(length)
            for boundary in range(1, length + 1):
                residual_len = length - boundary
                final_length = _final_length_for_vertex(
                    vertex,
                    length=length,
                    residual_len=residual_len,
                )
                if final_length is None:
                    continue
                included_weight += weight
                final_length_sum += weight * float(final_length)
                final_hist[_bin_label(final_length)] += weight

        if included_weight == 0:
            mean_final = 0.0
            shift = 0.0
        else:
            mean_final = final_length_sum / included_weight
            shift = mean_final - template_mean

        for label, lower, upper in LENGTH_BINS:
            bin_count = final_hist[label]
            rows.append(
                {
                    "vertex": vertex,
                    "action_mapping": _vertex_action_mapping(vertex),
                    "included_expected_boundary_count": included_weight,
                    "included_expected_boundary_ratio": included_weight / total_sessions,
                    "template_mean_length": template_mean,
                    "expected_mean_final_length": mean_final,
                    "expected_length_shift_from_template_mean": shift,
                    "bin": label,
                    "lower_inclusive": lower,
                    "upper_inclusive": "" if upper is None else upper,
                    "expected_count": bin_count,
                    "percent_within_included_boundaries": _percent(
                        bin_count,
                        included_weight,
                    ),
                }
            )
    return rows


def _candidate_rows(selection: SelectedFakeSessions) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in selection.discovered_paths:
        rows.append(
            {
                "path": _repo_relative(path),
                "size_bytes": path.stat().st_size,
                "selected": path.resolve() == selection.path.resolve(),
            }
        )
    return rows


def _build_report(
    *,
    config_path: Path,
    output_dir: Path,
    selection: SelectedFakeSessions,
    lengths: Sequence[int],
    length_rows: Sequence[Mapping[str, object]],
    boundary_rows: Sequence[Mapping[str, object]],
    short_rows: Sequence[Mapping[str, object]],
    final_length_rows: Sequence[Mapping[str, object]],
) -> str:
    total = len(lengths)
    template_mean = mean(lengths)
    template_median = median(lengths)
    internal_tail_boundary_ratio = 0.0
    suffix0_row = next(row for row in boundary_rows if row["group"] == "suffix_0")
    nonzero_tail_boundary_ratio = float(suffix0_row["nonzero_expected_percent"]) / 100.0

    candidate_table_rows = [
        [
            "`" + row["path"] + "`",
            row["size_bytes"],
            "yes" if row["selected"] else "no",
        ]
        for row in _candidate_rows(selection)
    ]
    length_table_rows = [
        [
            row["bin"],
            row["count"],
            _format_percent(float(row["percent"])),
        ]
        for row in length_rows
    ]
    boundary_table_rows = [
        [
            row["group"],
            (
                "N/A"
                if row["internal_expected_percent"] == ""
                else _format_percent(float(row["internal_expected_percent"]))
            ),
            _format_percent(float(row["nonzero_expected_percent"])),
            (
                "N/A"
                if row["delta_percentage_points"] == ""
                else _format_float(float(row["delta_percentage_points"]), digits=2)
            ),
            (
                "N/A"
                if row["internal_expected_count"] == ""
                else _format_float(float(row["internal_expected_count"]), digits=3)
            ),
            _format_float(float(row["nonzero_expected_count"]), digits=3),
        ]
        for row in boundary_rows
    ]
    short_table_rows = [
        [
            f"L={row['template_length']}",
            row["template_count"],
            _format_percent(float(row["template_percent"])),
            _format_float(float(row["suffix_0_expected_count"]), digits=3),
            _format_float(float(row["suffix_1_expected_count"]), digits=3),
            _format_float(float(row["suffix_2_expected_count"]), digits=3),
            _format_float(float(row["suffix_3plus_expected_count"]), digits=3),
        ]
        for row in short_rows
    ]
    final_summary: dict[str, Mapping[str, object]] = {}
    final_hist_by_vertex: dict[str, dict[str, float]] = {}
    for row in final_length_rows:
        vertex = str(row["vertex"])
        final_summary.setdefault(vertex, row)
        final_hist_by_vertex.setdefault(vertex, {})[str(row["bin"])] = float(
            row["percent_within_included_boundaries"]
        )
    final_summary_rows = [
        [
            vertex,
            _format_percent(float(row["included_expected_boundary_ratio"]) * 100.0),
            _format_float(float(row["expected_mean_final_length"]), digits=4),
            _format_float(
                float(row["expected_length_shift_from_template_mean"]),
                digits=4,
            ),
        ]
        for vertex, row in final_summary.items()
    ]
    final_hist_rows = [
        [vertex]
        + [
            _format_percent(final_hist_by_vertex[vertex].get(label, 0.0))
            for label, _, _ in LENGTH_BINS
        ]
        for vertex in final_hist_by_vertex
    ]

    report = [
        "# PTS Boundary Length Diagnostic",
        "",
        "This diagnostic uses the fixed fake session templates only. It does not "
        "claim or imply that switching from internal boundary to nonzero boundary "
        "changes the template length distribution.",
        "",
        "## Inputs",
        "",
        f"- Config: `{_repo_relative(config_path)}`",
        f"- Expected PTS-CEM fake sessions path: `{_repo_relative(selection.expected_path)}`",
        f"- Selected fake sessions path: `{_repo_relative(selection.path)}`",
        f"- Selection reason: {selection.selection_reason}",
        f"- Output directory: `{_repo_relative(output_dir)}`",
        "",
        "### Discovered fake-session artifacts",
        "",
        _markdown_table(["path", "size_bytes", "selected"], candidate_table_rows),
        "",
        "## Template Length Distribution",
        "",
        f"- Total fake sessions: {total}",
        f"- Min length: {min(lengths)}",
        f"- Max length: {max(lengths)}",
        f"- Mean length: {_format_float(template_mean, digits=4)}",
        f"- Median length: {_format_float(float(template_median), digits=4)}",
        "",
        _markdown_table(["bin", "count", "percent"], length_table_rows),
        "",
        "## Boundary Residual Group Distribution",
        "",
        "Internal boundary samples `b in {1, ..., L-1}` and cannot produce "
        "`suffix_0`. Nonzero boundary samples `b in {1, ..., L}` and introduces "
        "`suffix_0` when `b = L`.",
        "",
        _markdown_table(
            [
                "group",
                "internal expected %",
                "nonzero expected %",
                "delta pp",
                "internal expected count",
                "nonzero expected count",
            ],
            boundary_table_rows,
        ),
        "",
        "## Short-Session Contribution Under Nonzero Boundary",
        "",
        _markdown_table(
            [
                "length",
                "template count",
                "template %",
                "suffix_0 count",
                "suffix_1 count",
                "suffix_2 count",
                "suffix_3plus count",
            ],
            short_table_rows,
        ),
        "",
        "L=2 templates can only produce `suffix_0` and `suffix_1` under nonzero "
        "boundary sampling, so a large L=2 mass directly increases tail-append "
        "boundary exposure and one-item residual exposure.",
        "",
        "## Boundary-Level Tail Ratio",
        "",
        "- Internal boundary tail_append_boundary_ratio: "
        f"{_format_percent(internal_tail_boundary_ratio * 100.0)}",
        "- Nonzero boundary tail_append_boundary_ratio: "
        f"{_format_percent(nonzero_tail_boundary_ratio * 100.0)}",
        "",
        "These are boundary-level ratios before action sampling. They are not the "
        "final target_tail_ratio after stop/truncate actions.",
        "",
        "## Optional Expected Final Length By Vertex",
        "",
        "For `c1_preserve` and `c1_generate`, this uses the canonical where-valid "
        "interpretation: `suffix_0/suffix_1 -> A4`, and "
        "`suffix_2/suffix_3plus -> A2/A3`. For `c0_preserve` and `c0_generate`, "
        "`suffix_0` is excluded rather than forced to a fallback.",
        "",
        _markdown_table(
            [
                "vertex",
                "included boundary %",
                "expected mean final length",
                "shift from template mean",
            ],
            final_summary_rows,
        ),
        "",
        _markdown_table(["vertex"] + [label for label, _, _ in LENGTH_BINS], final_hist_rows),
        "",
        "## Output Files",
        "",
        "- `report.md`",
        "- `template_length_histogram.csv`",
        "- `boundary_group_expected_distribution.csv`",
        "- `short_session_contribution.csv`",
        "- `optional_expected_final_length_by_vertex.csv`",
        "",
    ]
    return "\n".join(report)


def run_diagnostic(
    *,
    config_path: str | Path,
    fake_sessions_path: str | Path | None,
    output_dir: str | Path,
) -> str:
    resolved_config_path = _repo_path(config_path)
    config = load_config(resolved_config_path)
    selected = _select_fake_sessions(config, explicit_path=fake_sessions_path)
    lengths = _load_lengths(selected.path)

    length_rows = _template_length_rows(lengths)
    internal_counts = _internal_boundary_distribution(lengths)
    nonzero_counts = _nonzero_boundary_distribution(lengths)
    boundary_rows = _boundary_distribution_rows(lengths, internal_counts, nonzero_counts)
    short_rows = _short_session_rows(lengths)
    final_length_rows = _expected_final_length_rows(lengths)

    resolved_output_dir = _repo_path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        resolved_output_dir / "template_length_histogram.csv",
        length_rows,
        ["bin", "lower_inclusive", "upper_inclusive", "count", "percent"],
    )
    _write_csv(
        resolved_output_dir / "boundary_group_expected_distribution.csv",
        boundary_rows,
        [
            "group",
            "internal_expected_count",
            "internal_expected_percent",
            "nonzero_expected_count",
            "nonzero_expected_percent",
            "delta_percentage_points",
            "note",
        ],
    )
    _write_csv(
        resolved_output_dir / "short_session_contribution.csv",
        short_rows,
        [
            "template_length",
            "template_count",
            "template_percent",
            "possible_nonzero_groups",
            "suffix_0_expected_count",
            "suffix_0_expected_percent_of_all_templates",
            "suffix_1_expected_count",
            "suffix_1_expected_percent_of_all_templates",
            "suffix_2_expected_count",
            "suffix_2_expected_percent_of_all_templates",
            "suffix_3plus_expected_count",
            "suffix_3plus_expected_percent_of_all_templates",
        ],
    )
    _write_csv(
        resolved_output_dir / "optional_expected_final_length_by_vertex.csv",
        final_length_rows,
        [
            "vertex",
            "action_mapping",
            "included_expected_boundary_count",
            "included_expected_boundary_ratio",
            "template_mean_length",
            "expected_mean_final_length",
            "expected_length_shift_from_template_mean",
            "bin",
            "lower_inclusive",
            "upper_inclusive",
            "expected_count",
            "percent_within_included_boundaries",
        ],
    )

    report = _build_report(
        config_path=resolved_config_path,
        output_dir=resolved_output_dir,
        selection=selected,
        lengths=lengths,
        length_rows=length_rows,
        boundary_rows=boundary_rows,
        short_rows=short_rows,
        final_length_rows=final_length_rows,
    )
    report_path = resolved_output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose fake-template length and expected residual suffix "
            "distribution for internal vs nonzero PTS boundaries."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="PTS-CEM config used to resolve the shared fake-session artifact.",
    )
    parser.add_argument(
        "--fake-sessions-path",
        default=None,
        help="Optional explicit fake_sessions.pkl path.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where report.md and CSV outputs are written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_diagnostic(
        config_path=args.config,
        fake_sessions_path=args.fake_sessions_path,
        output_dir=args.output_dir,
    )
    print(report)


if __name__ == "__main__":
    main()
