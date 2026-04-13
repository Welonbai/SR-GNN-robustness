#!/usr/bin/env python3
"""Generate a canonical long-table CSV from one run summary."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"
RESULTS_ROOT = REPO_ROOT / "results"
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Phase 1 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Convert one summary JSON plus sibling resolved_config.json into a canonical long_table.csv.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Path to one summary JSON under outputs/.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory under results/ for this run bundle.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional readable analysis identifier. If omitted, it is derived from the summary path.",
    )
    return parser


def main() -> None:
    """Run the CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        summary_path = resolve_existing_path(args.summary, label="summary JSON")
        outdir = resolve_output_path(args.outdir)
        ensure_path_within(summary_path, OUTPUTS_ROOT, label="summary JSON")
        ensure_path_within(outdir, RESULTS_ROOT, label="output directory")

        resolved_config_path = summary_path.parent / "resolved_config.json"
        if not resolved_config_path.is_file():
            raise AnalysisError(
                f"Expected sibling resolved_config.json next to '{summary_path}', but none was found."
            )

        summary_payload = load_json_mapping(summary_path, label="summary JSON")
        resolved_config_payload = load_json_mapping(resolved_config_path, label="resolved_config JSON")

        run_id = normalize_run_id(args.run_id) if args.run_id else derive_run_id(summary_path)
        warn_on_run_id_outdir_mismatch(run_id=run_id, outdir=outdir)
        metadata = extract_run_metadata(summary_payload, resolved_config_payload)
        rows = extract_rows(summary_payload, metadata=metadata, run_id=run_id)
        if not rows:
            raise AnalysisError(f"No metric rows were extracted from '{summary_path}'.")

        outdir.mkdir(parents=True, exist_ok=True)
        long_table_path = outdir / "long_table.csv"
        source_resolved_config_path = outdir / "source_resolved_config.json"
        manifest_path = outdir / "manifest.json"

        dataframe = sort_long_table(pd.DataFrame(rows, columns=CANONICAL_COLUMNS))
        dataframe.to_csv(long_table_path, index=False)
        shutil.copyfile(resolved_config_path, source_resolved_config_path)

        manifest = {
            "canonical_columns": CANONICAL_COLUMNS,
            "run_id": run_id,
            "source_summary_path": to_repo_relative(summary_path),
            "source_resolved_config_path": to_repo_relative(resolved_config_path),
            "output_dir": to_repo_relative(outdir),
            "generated_files": [
                "long_table.csv",
                "manifest.json",
                "source_resolved_config.json",
            ],
            "row_count": len(rows),
            "generation_timestamp": utc_now_iso(),
        }
        write_json(manifest_path, manifest)

        print(
            f"Wrote {len(rows)} canonical rows to '{long_table_path}' "
            f"for run_id '{run_id}'."
        )
    except AnalysisError as exc:
        raise SystemExit(f"Error: {exc}") from exc


def resolve_existing_path(raw_path: str, *, label: str) -> Path:
    """Resolve an existing path provided on the CLI."""
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise AnalysisError(f"The {label} path does not exist: '{path}'.")
    if not path.is_file():
        raise AnalysisError(f"The {label} path is not a file: '{path}'.")
    return path


def resolve_output_path(raw_path: str) -> Path:
    """Resolve an output directory path provided on the CLI."""
    return Path(raw_path).expanduser().resolve()


def ensure_path_within(path: Path, root: Path, *, label: str) -> None:
    """Require a path to stay inside one repository subtree."""
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(
            f"The {label} must stay under '{root.resolve()}', got '{path}'."
        ) from exc


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
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def normalize_run_id(raw_run_id: str) -> str:
    """Normalize a user-provided run identifier into a safe deterministic form."""
    candidate = raw_run_id.strip()
    if not candidate:
        raise AnalysisError("The provided --run-id is empty.")

    pieces = [sanitize_component(part) for part in re.split(r"(?:[\\/]+|__+)", candidate)]
    normalized = "__".join(piece for piece in pieces if piece)
    if not normalized:
        raise AnalysisError(
            f"The provided --run-id '{raw_run_id}' does not contain any usable characters."
        )
    return normalized


def warn_on_run_id_outdir_mismatch(*, run_id: str, outdir: Path) -> None:
    """Print a non-fatal warning when the output directory name does not match the run_id."""
    if outdir.name == run_id:
        return

    print(
        (
            f"Warning: run_id '{run_id}' does not match output directory name "
            f"'{outdir.name}'. Writing files to '{outdir}'."
        ),
        file=sys.stderr,
    )


def derive_run_id(summary_path: Path) -> str:
    """Build a readable run_id from the summary path under outputs/."""
    try:
        relative_parts = list(summary_path.resolve().relative_to(OUTPUTS_ROOT).parts)
    except ValueError:
        return sanitize_component(summary_path.stem)

    if relative_parts and relative_parts[0] == "runs":
        relative_parts = relative_parts[1:]
    if not relative_parts:
        return sanitize_component(summary_path.stem)

    relative_parts[-1] = summary_path.stem
    sanitized_parts = [sanitize_component(part) for part in relative_parts if sanitize_component(part)]
    if not sanitized_parts:
        raise AnalysisError(f"Could not derive a readable run_id from '{summary_path}'.")
    return "__".join(sanitized_parts)


def sanitize_component(raw_value: str) -> str:
    """Make one path-derived component filesystem-safe without losing readability."""
    sanitized = UNSAFE_COMPONENT_PATTERN.sub("_", raw_value.strip())
    sanitized = REPEATED_UNDERSCORE_PATTERN.sub("_", sanitized)
    sanitized = sanitized.strip("._-")
    return sanitized or "item"


def sort_long_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministically ordered long table."""
    target_item_order = dataframe["target_item"].map(build_target_item_sort_key)
    sorted_dataframe = (
        dataframe.assign(
            _target_item_type_order=target_item_order.map(lambda item: item[0]),
            _target_item_int_order=target_item_order.map(lambda item: item[1]),
            _target_item_str_order=target_item_order.map(lambda item: item[2]),
        )
        .sort_values(
            by=[
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


def require_target_item(value: Any, *, label: str) -> str | int:
    """Require a scalar target item identifier as int or non-empty string."""
    if isinstance(value, bool) or value is None:
        raise AnalysisError(f"Expected '{label}' to be a scalar target identifier.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if isinstance(value, str) and not value.strip():
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
) -> list[dict[str, Any]]:
    """Flatten the summary JSON into canonical row dictionaries."""
    if "targets" not in summary_payload:
        raise AnalysisError("The summary JSON is missing the top-level 'targets' field.")

    target_payloads = list(iter_target_payloads(summary_payload["targets"]))
    if not target_payloads:
        raise AnalysisError("The summary JSON does not contain any target payloads.")

    rows: list[dict[str, Any]] = []
    for target_payload in target_payloads:
        target_item = require_target_item(
            target_payload.get("target_item"),
            label="summary.targets[*].target_item",
        )
        victims = target_payload.get("victims")
        if not isinstance(victims, Mapping) or not victims:
            raise AnalysisError(
                f"Target '{target_item}' does not contain a non-empty 'victims' object."
            )

        for victim_model, victim_payload in victims.items():
            victim_name = require_nonempty_string(
                victim_model,
                label=f"summary.targets[{target_item}].victims key",
            )
            if not isinstance(victim_payload, Mapping):
                raise AnalysisError(
                    f"Victim '{victim_name}' for target '{target_item}' must be a JSON object."
                )

            metrics = victim_payload.get("metrics")
            if not isinstance(metrics, Mapping) or not metrics:
                raise AnalysisError(
                    f"Victim '{victim_name}' for target '{target_item}' is missing a non-empty 'metrics' object."
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
                            label=f"metric value for '{metric_key}' on victim '{victim_name}' and target '{target_item}'",
                        ),
                    }
                )
    return rows


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
