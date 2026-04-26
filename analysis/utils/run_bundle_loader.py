"""Load one or more diagnosis run bundles from run-root based YAML manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class RunBundleLoaderError(ValueError):
    """Raised when a diagnosis manifest or run bundle cannot be loaded."""


@dataclass(frozen=True)
class DiagnosisMethodSpec:
    """One method entry from a run-bundle manifest."""

    key: str
    label: str
    run_root: Path
    attack_method: str | None = None
    summary_current: Path | None = None


@dataclass(frozen=True)
class RunBundleManifest:
    """Parsed run-bundle manifest."""

    path: Path
    report_id: str
    dataset: str
    output_dir: Path | None
    expected_targets: tuple[int, ...]
    expected_victims: tuple[str, ...]
    methods: dict[str, DiagnosisMethodSpec]
    notes: dict[str, Any]


@dataclass(frozen=True)
class VictimArtifactPaths:
    """Victim-level artifact paths for one target item."""

    run_dir: Path
    metrics_path: Path | None
    predictions_path: Path | None
    train_history_path: Path | None
    resolved_config_path: Path | None
    config_snapshot_path: Path | None


@dataclass(frozen=True)
class TargetArtifactPaths:
    """Target-level artifact paths for one run bundle."""

    target_dir: Path
    position_stats_path: Path | None
    prefix_metadata_path: Path | None
    random_nonzero_metadata_path: Path | None
    position_opt_dir: Path | None
    selected_positions_path: Path | None
    training_history_path: Path | None
    run_metadata_path: Path | None
    learned_logits_path: Path | None
    optimized_poisoned_sessions_path: Path | None
    victims: dict[str, VictimArtifactPaths]


@dataclass(frozen=True)
class RunBundle:
    """Resolved run bundle rooted at one appendable run-group directory."""

    method_key: str
    label: str
    run_root: Path
    run_group_key: str | None
    dataset: str | None
    run_type: str | None
    target_cohort_key: str | None
    target_items: tuple[int, ...]
    victims: tuple[str, ...]
    attack_method: str | None
    replacement_topk_ratio: float | None
    nonzero_action_when_possible: bool | None
    policy_feature_set: str | None
    reward_mode: str | None
    final_policy_selection: str | None
    deterministic_eval_every: int | None
    deterministic_eval_include_final: bool | None
    attack_size: float | None
    seeds: dict[str, Any]
    summary_current_path: Path
    resolved_config_path: Path
    artifact_manifest_path: Path
    key_payloads_path: Path
    run_coverage_path: Path
    execution_log_path: Path | None
    progress_path: Path | None
    legacy_summary_path: Path | None
    target_artifacts: dict[int, TargetArtifactPaths]
    shared_artifact_paths: dict[str, Path]
    summary_current: dict[str, Any]
    resolved_config: dict[str, Any]
    artifact_manifest: dict[str, Any]
    key_payloads: dict[str, Any]
    run_coverage: dict[str, Any]


def resolve_repo_path(raw_path: str | Path, *, label: str) -> Path:
    """Resolve a repo-relative or absolute path and require that it exists."""
    candidate = Path(raw_path)
    resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    resolved = resolved.resolve()
    if not resolved.exists():
        raise RunBundleLoaderError(f"Missing {label}: '{raw_path}'.")
    return resolved


def load_run_bundle_manifest(path: str | Path) -> RunBundleManifest:
    """Load one YAML manifest that points at completed run roots."""
    manifest_path = resolve_repo_path(path, label="run-bundle manifest")
    payload = _load_yaml_mapping(manifest_path)
    methods_payload = _require_mapping(payload.get("methods"), "methods")
    methods: dict[str, DiagnosisMethodSpec] = {}
    for key, raw_method in methods_payload.items():
        method_payload = _require_mapping(raw_method, f"methods.{key}")
        run_root = resolve_repo_path(
            _require_string(method_payload.get("run_root"), f"methods.{key}.run_root"),
            label=f"methods.{key}.run_root",
        )
        summary_current = method_payload.get("summary_current")
        resolved_summary = None
        if summary_current is not None:
            resolved_summary = resolve_repo_path(
                _require_string(summary_current, f"methods.{key}.summary_current"),
                label=f"methods.{key}.summary_current",
            )
            expected_summary = (run_root / "summary_current.json").resolve()
            if resolved_summary != expected_summary:
                raise RunBundleLoaderError(
                    f"methods.{key}.summary_current must match '{expected_summary}'."
                )
        methods[str(key)] = DiagnosisMethodSpec(
            key=str(key),
            label=_require_string(method_payload.get("label"), f"methods.{key}.label"),
            run_root=run_root,
            attack_method=_optional_string(method_payload.get("attack_method")),
            summary_current=resolved_summary,
        )
    output_dir = payload.get("output_dir")
    return RunBundleManifest(
        path=manifest_path,
        report_id=_require_string(payload.get("report_id"), "report_id"),
        dataset=_require_string(payload.get("dataset"), "dataset"),
        output_dir=(
            None
            if output_dir is None
            else resolve_parent_repo_path(
                _require_string(output_dir, "output_dir"),
                label="output_dir",
            )
        ),
        expected_targets=_parse_int_list(
            _require_mapping(payload.get("targets"), "targets").get("expected"),
            "targets.expected",
        ),
        expected_victims=_parse_string_list(
            _require_mapping(payload.get("victims"), "victims").get("expected"),
            "victims.expected",
        ),
        methods=methods,
        notes=_require_mapping(payload.get("notes", {}), "notes"),
    )


def load_bundles_from_manifest(path: str | Path) -> tuple[RunBundleManifest, dict[str, RunBundle]]:
    """Load one manifest plus every referenced run bundle."""
    manifest = load_run_bundle_manifest(path)
    bundles = {
        method_key: load_run_bundle(
            run_root=spec.run_root,
            method_key=method_key,
            label=spec.label,
            attack_method_hint=spec.attack_method,
            expected_targets=manifest.expected_targets,
            expected_victims=manifest.expected_victims,
            dataset_hint=manifest.dataset,
        )
        for method_key, spec in manifest.methods.items()
    }
    return manifest, bundles


def load_run_bundle(
    *,
    run_root: str | Path,
    method_key: str = "method",
    label: str | None = None,
    attack_method_hint: str | None = None,
    expected_targets: tuple[int, ...] = (),
    expected_victims: tuple[str, ...] = (),
    dataset_hint: str | None = None,
) -> RunBundle:
    """Resolve one run-root into a structured bundle with fixed artifact paths."""
    root = resolve_repo_path(run_root, label=f"{method_key}.run_root")
    summary_current_path = _require_existing_file(root / "summary_current.json", label="summary_current.json")
    resolved_config_path = _require_existing_file(root / "resolved_config.json", label="resolved_config.json")
    artifact_manifest_path = _require_existing_file(root / "artifact_manifest.json", label="artifact_manifest.json")
    key_payloads_path = _require_existing_file(root / "key_payloads.json", label="key_payloads.json")
    run_coverage_path = _require_existing_file(root / "run_coverage.json", label="run_coverage.json")

    summary_current = _load_json_mapping(summary_current_path, label="summary_current.json")
    resolved_config = _load_json_mapping(resolved_config_path, label="resolved_config.json")
    artifact_manifest = _load_json_mapping(artifact_manifest_path, label="artifact_manifest.json")
    key_payloads = _load_json_mapping(key_payloads_path, label="key_payloads.json")
    run_coverage = _load_json_mapping(run_coverage_path, label="run_coverage.json")

    target_items = tuple(_parse_int_list(run_coverage.get("targets_order"), "run_coverage.targets_order"))
    victims = tuple(_victim_names_from_run_coverage(run_coverage))
    if expected_targets and tuple(target_items[: len(expected_targets)]) != tuple(expected_targets):
        raise RunBundleLoaderError(
            f"{method_key}.run_root target prefix {target_items[: len(expected_targets)]} "
            f"does not match expected_targets {expected_targets}."
        )
    if expected_victims:
        missing_victims = [victim for victim in expected_victims if victim not in victims]
        if missing_victims:
            raise RunBundleLoaderError(
                f"{method_key}.run_root is missing expected victims: {', '.join(missing_victims)}."
            )

    result_config = _require_mapping(resolved_config.get("result_config", {}), "result_config")
    attack_config = _require_mapping(result_config.get("attack", {}), "result_config.attack")
    position_opt_config = _optional_mapping(attack_config.get("position_opt"))
    target_artifacts = {
        target_item: _target_artifact_paths(root, target_item=target_item, victims=victims)
        for target_item in target_items
    }
    shared_artifact_paths = _flatten_shared_artifact_paths(
        _optional_mapping(artifact_manifest.get("shared_artifacts"))
    )
    run_type = _optional_string(summary_current.get("run_type"))
    bundle_dataset = dataset_hint
    if bundle_dataset is None:
        split_payload = shared_artifact_paths.get("canonical_split.metadata")
        if split_payload is not None and split_payload.is_file():
            bundle_dataset = _optional_string(
                _load_json_mapping(split_payload, label="canonical metadata").get("dataset_name")
            )
    return RunBundle(
        method_key=method_key,
        label=label or method_key,
        run_root=root,
        run_group_key=_optional_string(summary_current.get("run_group_key")),
        dataset=bundle_dataset,
        run_type=run_type,
        target_cohort_key=_optional_string(summary_current.get("target_cohort_key")),
        target_items=target_items,
        victims=victims,
        attack_method=attack_method_hint or run_type,
        replacement_topk_ratio=_optional_float(attack_config.get("replacement_topk_ratio")),
        nonzero_action_when_possible=(
            None
            if position_opt_config is None
            else _optional_bool(position_opt_config.get("nonzero_action_when_possible"))
        ),
        policy_feature_set=(
            None if position_opt_config is None else _optional_string(position_opt_config.get("policy_feature_set"))
        ),
        reward_mode=(
            None if position_opt_config is None else _optional_string(position_opt_config.get("reward_mode"))
        ),
        final_policy_selection=(
            None
            if position_opt_config is None
            else _optional_string(position_opt_config.get("final_policy_selection"))
        ),
        deterministic_eval_every=(
            None if position_opt_config is None else _optional_int(position_opt_config.get("deterministic_eval_every"))
        ),
        deterministic_eval_include_final=(
            None
            if position_opt_config is None
            else _optional_bool(position_opt_config.get("deterministic_eval_include_final"))
        ),
        attack_size=_optional_float(attack_config.get("size")),
        seeds=_require_mapping(result_config.get("seeds", {}), "result_config.seeds"),
        summary_current_path=summary_current_path,
        resolved_config_path=resolved_config_path,
        artifact_manifest_path=artifact_manifest_path,
        key_payloads_path=key_payloads_path,
        run_coverage_path=run_coverage_path,
        execution_log_path=_optional_existing_file(root / "execution_log.json"),
        progress_path=_optional_existing_file(root / "progress.json"),
        legacy_summary_path=_optional_existing_file(root / f"summary_{run_type}.json") if run_type else None,
        target_artifacts=target_artifacts,
        shared_artifact_paths=shared_artifact_paths,
        summary_current=summary_current,
        resolved_config=resolved_config,
        artifact_manifest=artifact_manifest,
        key_payloads=key_payloads,
        run_coverage=run_coverage,
    )


def bundle_to_dict(bundle: RunBundle) -> dict[str, Any]:
    """Convert one run bundle into a JSON-safe dict."""
    return {
        "method_key": bundle.method_key,
        "label": bundle.label,
        "run_root": _repo_relative(bundle.run_root),
        "run_group_key": bundle.run_group_key,
        "dataset": bundle.dataset,
        "run_type": bundle.run_type,
        "target_cohort_key": bundle.target_cohort_key,
        "target_items": list(bundle.target_items),
        "victims": list(bundle.victims),
        "attack_method": bundle.attack_method,
        "replacement_topk_ratio": bundle.replacement_topk_ratio,
        "nonzero_action_when_possible": bundle.nonzero_action_when_possible,
        "policy_feature_set": bundle.policy_feature_set,
        "reward_mode": bundle.reward_mode,
        "final_policy_selection": bundle.final_policy_selection,
        "deterministic_eval_every": bundle.deterministic_eval_every,
        "deterministic_eval_include_final": bundle.deterministic_eval_include_final,
        "attack_size": bundle.attack_size,
        "seeds": bundle.seeds,
        "artifacts": {
            "summary_current": _repo_relative(bundle.summary_current_path),
            "resolved_config": _repo_relative(bundle.resolved_config_path),
            "artifact_manifest": _repo_relative(bundle.artifact_manifest_path),
            "key_payloads": _repo_relative(bundle.key_payloads_path),
            "run_coverage": _repo_relative(bundle.run_coverage_path),
            "execution_log": _optional_repo_relative(bundle.execution_log_path),
            "progress": _optional_repo_relative(bundle.progress_path),
            "legacy_summary": _optional_repo_relative(bundle.legacy_summary_path),
        },
        "shared_artifact_paths": {
            key: _repo_relative(path) for key, path in sorted(bundle.shared_artifact_paths.items())
        },
        "targets": {
            str(target_item): _target_artifacts_to_dict(paths)
            for target_item, paths in sorted(bundle.target_artifacts.items())
        },
    }


def manifest_to_dict(manifest: RunBundleManifest) -> dict[str, Any]:
    """Convert one manifest into a JSON-safe dict."""
    return {
        "path": _repo_relative(manifest.path),
        "report_id": manifest.report_id,
        "dataset": manifest.dataset,
        "output_dir": _optional_repo_relative(manifest.output_dir),
        "expected_targets": list(manifest.expected_targets),
        "expected_victims": list(manifest.expected_victims),
        "methods": {
            method_key: {
                "label": spec.label,
                "run_root": _repo_relative(spec.run_root),
                "attack_method": spec.attack_method,
                "summary_current": _optional_repo_relative(spec.summary_current),
            }
            for method_key, spec in manifest.methods.items()
        },
        "notes": manifest.notes,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the run-bundle loader CLI parser."""
    parser = argparse.ArgumentParser(
        description="Resolve a diagnosis YAML manifest into run-bundle metadata."
    )
    parser.add_argument("--config", required=True, help="Path to the run-bundle YAML manifest.")
    parser.add_argument("--output-json", help="Optional resolved JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        manifest, bundles = load_bundles_from_manifest(args.config)
        payload = {
            "manifest": manifest_to_dict(manifest),
            "bundles": {
                method_key: bundle_to_dict(bundle)
                for method_key, bundle in bundles.items()
            },
        }
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            output_path = resolve_parent_repo_path(args.output_json, label="output JSON")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered, encoding="utf-8")
            print(f"Wrote resolved run-bundle manifest: {output_path}")
        else:
            print(rendered, end="")
    except RunBundleLoaderError as exc:
        parser.exit(status=2, message=f"Error: {exc}\n")
    return 0


def _target_artifacts_to_dict(paths: TargetArtifactPaths) -> dict[str, Any]:
    return {
        "target_dir": _repo_relative(paths.target_dir),
        "position_stats": _optional_repo_relative(paths.position_stats_path),
        "prefix_metadata": _optional_repo_relative(paths.prefix_metadata_path),
        "random_nonzero_metadata": _optional_repo_relative(paths.random_nonzero_metadata_path),
        "position_opt_dir": _optional_repo_relative(paths.position_opt_dir),
        "selected_positions": _optional_repo_relative(paths.selected_positions_path),
        "training_history": _optional_repo_relative(paths.training_history_path),
        "run_metadata": _optional_repo_relative(paths.run_metadata_path),
        "learned_logits": _optional_repo_relative(paths.learned_logits_path),
        "optimized_poisoned_sessions": _optional_repo_relative(paths.optimized_poisoned_sessions_path),
        "victims": {
            victim_name: {
                "run_dir": _repo_relative(victim_paths.run_dir),
                "metrics": _optional_repo_relative(victim_paths.metrics_path),
                "predictions": _optional_repo_relative(victim_paths.predictions_path),
                "train_history": _optional_repo_relative(victim_paths.train_history_path),
                "resolved_config": _optional_repo_relative(victim_paths.resolved_config_path),
                "config_snapshot": _optional_repo_relative(victim_paths.config_snapshot_path),
            }
            for victim_name, victim_paths in sorted(paths.victims.items())
        },
    }


def _target_artifact_paths(
    run_root: Path,
    *,
    target_item: int,
    victims: tuple[str, ...],
) -> TargetArtifactPaths:
    target_dir = run_root / "targets" / str(target_item)
    victim_paths = {
        victim_name: _victim_artifact_paths(target_dir, victim_name=victim_name)
        for victim_name in victims
    }
    position_opt_dir = _optional_existing_dir(target_dir / "position_opt")
    return TargetArtifactPaths(
        target_dir=target_dir.resolve(),
        position_stats_path=_optional_existing_file(target_dir / "position_stats.json"),
        prefix_metadata_path=_optional_existing_file(target_dir / "prefix_nonzero_when_possible_metadata.pkl"),
        random_nonzero_metadata_path=_optional_existing_file(target_dir / "random_nonzero_position_metadata.json"),
        position_opt_dir=position_opt_dir,
        selected_positions_path=_optional_existing_file(target_dir / "position_opt" / "selected_positions.json"),
        training_history_path=_optional_existing_file(target_dir / "position_opt" / "training_history.json"),
        run_metadata_path=_optional_existing_file(target_dir / "position_opt" / "run_metadata.json"),
        learned_logits_path=_optional_existing_file(target_dir / "position_opt" / "learned_logits.pt"),
        optimized_poisoned_sessions_path=_optional_existing_file(
            target_dir / "position_opt" / "optimized_poisoned_sessions.pkl"
        ),
        victims=victim_paths,
    )


def _victim_artifact_paths(target_dir: Path, *, victim_name: str) -> VictimArtifactPaths:
    run_dir = (target_dir / "victims" / victim_name).resolve()
    return VictimArtifactPaths(
        run_dir=run_dir,
        metrics_path=_optional_existing_file(run_dir / "metrics.json"),
        predictions_path=_optional_existing_file(run_dir / "predictions.json"),
        train_history_path=_optional_existing_file(run_dir / "train_history.json"),
        resolved_config_path=_optional_existing_file(run_dir / "resolved_config.json"),
        config_snapshot_path=_optional_existing_file(run_dir / "config.yaml"),
    )


def _flatten_shared_artifact_paths(shared_artifacts: dict[str, Any] | None) -> dict[str, Path]:
    flattened: dict[str, Path] = {}
    if shared_artifacts is None:
        return flattened
    _collect_path_leaves(shared_artifacts, prefix="shared_artifacts", output=flattened)
    return flattened


def _collect_path_leaves(node: Any, *, prefix: str, output: dict[str, Path]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _collect_path_leaves(value, prefix=child_prefix, output=output)
        return
    if isinstance(node, list):
        return
    if not isinstance(node, str):
        return
    candidate = Path(node)
    resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    if resolved.exists():
        output[prefix.removeprefix("shared_artifacts.")] = resolved.resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RunBundleLoaderError(f"Invalid YAML in '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RunBundleLoaderError(f"Expected YAML mapping in '{path}'.")
    return payload


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RunBundleLoaderError(f"Invalid JSON in {label} '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise RunBundleLoaderError(f"Expected {label} '{path}' to contain a JSON object.")
    return payload


def _victim_names_from_run_coverage(run_coverage: dict[str, Any]) -> list[str]:
    victims = _require_mapping(run_coverage.get("victims"), "run_coverage.victims")
    return [str(name) for name in victims.keys()]


def _require_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RunBundleLoaderError(f"Missing {label}: '{path}'.")
    return resolved


def _optional_existing_file(path: Path) -> Path | None:
    resolved = path.resolve()
    return resolved if resolved.is_file() else None


def _optional_existing_dir(path: Path) -> Path | None:
    resolved = path.resolve()
    return resolved if resolved.is_dir() else None


def resolve_parent_repo_path(raw_path: str | Path, *, label: str) -> Path:
    """Resolve a repo-relative or absolute path without requiring it to exist."""
    candidate = Path(raw_path)
    resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
    return resolved.resolve()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _optional_repo_relative(path: Path | None) -> str | None:
    return None if path is None else _repo_relative(path)


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunBundleLoaderError(f"{label} must be a non-empty string.")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return _require_string(value, "string field")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RunBundleLoaderError(f"Expected numeric value, received '{value!r}'.") from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RunBundleLoaderError("Expected integer value, received bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RunBundleLoaderError(f"Expected integer value, received '{value!r}'.") from exc


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise RunBundleLoaderError(f"Expected bool value, received '{value!r}'.")
    return value


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunBundleLoaderError(f"{label} must be a mapping.")
    return value


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, "mapping field")


def _parse_int_list(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RunBundleLoaderError(f"{label} must be a list of integers.")
    parsed: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise RunBundleLoaderError(f"{label} cannot contain bool values.")
        try:
            parsed.append(int(item))
        except (TypeError, ValueError) as exc:
            raise RunBundleLoaderError(f"{label} contains a non-integer value: '{item!r}'.") from exc
    return tuple(parsed)


def _parse_string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise RunBundleLoaderError(f"{label} must be a list of strings.")
    return tuple(_require_string(item, f"{label}[]") for item in value)


if __name__ == "__main__":
    raise SystemExit(main())
