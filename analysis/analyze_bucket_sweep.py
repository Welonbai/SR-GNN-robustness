from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from attack.common.artifact_io import load_fake_sessions, load_json
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.position_opt.bucket_diagnostics import build_bucket_position_summary
from attack.position_opt.bucket_selector import BucketSessionSelectionRecord
from attack.position_opt.candidate_builder import build_candidate_position_result
from analysis.pipeline.view_table_builder import calculate_signed_percent_change
from analysis.utils.run_bundle_loader import (
    REPO_ROOT,
    RunBundle,
    load_run_bundle,
    load_run_bundle_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze bucket-sweep attack runs against the existing Random-NZ baseline."
    )
    parser.add_argument("--config", required=True, help="Path to the bucket-sweep manifest YAML.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    manifest = load_run_bundle_manifest(args.config)
    bundles = {
        method_key: load_run_bundle(
            run_root=spec.run_root,
            method_key=method_key,
            label=spec.label,
            attack_method_hint=spec.attack_method,
            dataset_hint=manifest.dataset,
        )
        for method_key, spec in manifest.methods.items()
    }

    clean_method_key = _require_note_string(manifest.notes, "clean_method_key")
    baseline_method_key = _require_note_string(manifest.notes, "baseline_method_key")
    bucket_method_keys = _require_note_str_list(manifest.notes, "bucket_method_keys")
    expected_bucket_target_cohort_key = _require_note_string(
        manifest.notes,
        "expected_bucket_target_cohort_key",
    )
    expected_bucket_target_prefix = _require_note_int_list(
        manifest.notes,
        "expected_bucket_target_prefix",
    )

    if clean_method_key not in bundles:
        raise ValueError(f"Manifest clean_method_key '{clean_method_key}' is not present.")
    if baseline_method_key not in bundles:
        raise ValueError(f"Manifest baseline_method_key '{baseline_method_key}' is not present.")

    output_dir = (
        manifest.output_dir
        if manifest.output_dir is not None
        else (REPO_ROOT / "analysis" / "diagnosis_outputs" / manifest.report_id).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_bundle = bundles[clean_method_key]
    baseline_bundle = bundles[baseline_method_key]
    clean_metric_lookup = _build_metric_lookup(clean_bundle)

    long_rows: list[dict[str, Any]] = []
    for method_key, bundle in bundles.items():
        if method_key == clean_method_key:
            continue
        long_rows.extend(
            _flatten_bundle_metrics(
                bundle,
                method_key=method_key,
                clean_metric_lookup=clean_metric_lookup,
            )
        )
    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = _apply_ground_truth_signed_change(long_df)
        long_df = long_df.sort_values(
            by=["method", "target_item", "victim_model", "metric_name", "K"]
        ).reset_index(drop=True)
    long_csv_path = output_dir / "long_metrics.csv"
    long_df.to_csv(long_csv_path, index=False)

    position_summary_rows: list[dict[str, Any]] = []
    for method_key, bundle in bundles.items():
        if method_key == clean_method_key:
            continue
        if method_key == baseline_method_key:
            position_summary_rows.extend(
                _build_replayed_random_nonzero_position_rows(
                    bundle,
                    method_key=method_key,
                )
            )
            continue
        position_summary_rows.extend(
            _load_bucket_position_summary_rows(
                bundle,
                method_key=method_key,
                expected_target_cohort_key=expected_bucket_target_cohort_key,
                expected_target_prefix=expected_bucket_target_prefix,
            )
        )
    position_summary_df = pd.DataFrame(position_summary_rows)
    if not position_summary_df.empty:
        position_summary_df = position_summary_df.sort_values(
            by=["method", "target_item"]
        ).reset_index(drop=True)
    position_summary_csv_path = output_dir / "position_summary.csv"
    position_summary_df.to_csv(position_summary_csv_path, index=False)

    k30_summary_df = _build_k30_summary(
        long_df=long_df,
        position_summary_df=position_summary_df,
    )
    k30_summary_csv_path = output_dir / "k30_summary.csv"
    k30_summary_df.to_csv(k30_summary_csv_path, index=False)

    compatibility_report = _build_compatibility_report(
        bundles=bundles,
        clean_method_key=clean_method_key,
        baseline_method_key=baseline_method_key,
        bucket_method_keys=bucket_method_keys,
        expected_bucket_target_cohort_key=expected_bucket_target_cohort_key,
        expected_bucket_target_prefix=expected_bucket_target_prefix,
    )
    compatibility_json_path = output_dir / "compatibility_report.json"
    compatibility_json_path.write_text(
        json.dumps(compatibility_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pairwise_df = _build_pairwise_vs_random_nz(
        long_df=long_df,
        compatibility_report=compatibility_report,
        baseline_method_key=baseline_method_key,
        bucket_method_keys=bucket_method_keys,
    )
    pairwise_csv_path = output_dir / "pairwise_vs_random_nz.csv"
    pairwise_df.to_csv(pairwise_csv_path, index=False)

    print(f"Wrote: {output_dir}")
    print(f"- {long_csv_path.relative_to(REPO_ROOT)}")
    print(f"- {k30_summary_csv_path.relative_to(REPO_ROOT)}")
    print(f"- {position_summary_csv_path.relative_to(REPO_ROOT)}")
    print(f"- {pairwise_csv_path.relative_to(REPO_ROOT)}")
    print(f"- {compatibility_json_path.relative_to(REPO_ROOT)}")
    return 0


def _flatten_bundle_metrics(
    bundle: RunBundle,
    *,
    method_key: str,
    clean_metric_lookup: Mapping[tuple[int, str, str], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selection_seed = _selection_seed_for_bundle(bundle)
    nonzero_action_when_possible = _resolve_nonzero_action_semantics(bundle)
    for target_key, target_payload in bundle.summary_current.get("targets", {}).items():
        target_item = int(target_key)
        victims_payload = target_payload.get("victims", {})
        if not isinstance(victims_payload, Mapping):
            continue
        for victim_name, victim_payload in victims_payload.items():
            metrics_payload = victim_payload.get("metrics", {})
            if not isinstance(metrics_payload, Mapping):
                continue
            for metric_key, metric_value in metrics_payload.items():
                if not isinstance(metric_key, str):
                    continue
                parsed = _parse_metric_key(metric_key)
                if parsed is None:
                    continue
                metric_name, k_value = parsed
                clean_metric_value = clean_metric_lookup.get(
                    (target_item, str(victim_name), metric_key)
                )
                rows.append(
                    {
                        "dataset": bundle.dataset,
                        "run_group": bundle.run_group_key,
                        "method": method_key,
                        "method_label": bundle.label,
                        "target_item": int(target_item),
                        "victim_model": str(victim_name),
                        "metric_key": metric_key,
                        "metric_name": metric_name,
                        "K": int(k_value),
                        "metric_value": float(metric_value),
                        "clean_metric_value": (
                            None
                            if clean_metric_value is None
                            else float(clean_metric_value)
                        ),
                        "signed_change_pct_vs_clean": None,
                        "seed": int(selection_seed) if selection_seed is not None else None,
                        "replacement_topk_ratio": bundle.replacement_topk_ratio,
                        "nonzero_action_when_possible": nonzero_action_when_possible,
                    }
                )
    return rows


def _apply_ground_truth_signed_change(long_df: pd.DataFrame) -> pd.DataFrame:
    result = long_df.copy()
    mask = (
        result["metric_name"].astype(str).str.startswith("ground_truth_")
        & result["clean_metric_value"].notna()
    )
    if mask.any():
        result.loc[mask, "signed_change_pct_vs_clean"] = calculate_signed_percent_change(
            current_values=result.loc[mask, "metric_value"],
            baseline_values=result.loc[mask, "clean_metric_value"],
            label="ground-truth signed change vs clean",
        )
    return result


def _build_metric_lookup(bundle: RunBundle) -> dict[tuple[int, str, str], float]:
    lookup: dict[tuple[int, str, str], float] = {}
    for target_key, target_payload in bundle.summary_current.get("targets", {}).items():
        target_item = int(target_key)
        victims_payload = target_payload.get("victims", {})
        if not isinstance(victims_payload, Mapping):
            continue
        for victim_name, victim_payload in victims_payload.items():
            metrics_payload = victim_payload.get("metrics", {})
            if not isinstance(metrics_payload, Mapping):
                continue
            for metric_key, metric_value in metrics_payload.items():
                if isinstance(metric_key, str):
                    lookup[(target_item, str(victim_name), metric_key)] = float(metric_value)
    return lookup


def _build_replayed_random_nonzero_position_rows(
    bundle: RunBundle,
    *,
    method_key: str,
) -> list[dict[str, Any]]:
    fake_sessions_path = _find_shared_artifact_path(bundle, "fake_sessions.pkl")
    fake_sessions = load_fake_sessions(fake_sessions_path)
    if fake_sessions is None:
        raise FileNotFoundError(f"Missing shared fake sessions for replay: {fake_sessions_path}")
    rows: list[dict[str, Any]] = []
    for target_item in bundle.target_items:
        records = _replay_random_nonzero_records(
            bundle,
            fake_sessions=fake_sessions,
            target_item=int(target_item),
        )
        replay_validation = _validate_random_nonzero_replay_against_position_stats(
            records,
            position_stats_path=bundle.target_artifacts[int(target_item)].position_stats_path,
            method_key=method_key,
            target_item=int(target_item),
        )
        summary = build_bucket_position_summary(
            records,
            method_name=method_key,
            target_item=int(target_item),
            seed=int(bundle.seeds["fake_session_seed"]),
            seed_source="fake_session_seed",
            replacement_topk_ratio=float(bundle.replacement_topk_ratio or 0.0),
            nonzero_action_when_possible=True,
        )
        for field_name in (
            "mode_candidate_count_min",
            "mode_candidate_count_max",
            "mode_candidate_count_mean",
            "mode_candidate_count_std",
            "mode_candidate_count_p25",
            "mode_candidate_count_p50",
            "mode_candidate_count_p75",
            "mode_candidate_count_p90",
            "mode_candidate_count_p95",
        ):
            summary[field_name] = None
        summary["position_summary_source"] = replay_validation["position_summary_source"]
        summary["reconstruction_mode"] = "legacy_random_nonzero_policy"
        summary["replay_validation"] = replay_validation
        summary["shared_fake_sessions_path"] = str(fake_sessions_path)
        rows.append(
            {
                "dataset": bundle.dataset,
                "run_group": bundle.run_group_key,
                "method": method_key,
                **_serialize_summary_payload(summary),
            }
        )
    return rows


def _validate_random_nonzero_replay_against_position_stats(
    records: Sequence[BucketSessionSelectionRecord],
    *,
    position_stats_path: Path | None,
    method_key: str,
    target_item: int,
) -> dict[str, Any]:
    replayed_counts = _position_counts_from_records(records)
    if position_stats_path is None:
        return {
            "validated_against_position_stats": False,
            "position_stats_path": None,
            "position_summary_source": "offline_exact_replay_unverified",
            "reason": "position_stats.json not available",
        }

    payload = load_json(position_stats_path)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Random-NZ replay validation failed for {method_key} target {target_item}: "
            f"invalid position_stats payload at '{position_stats_path}'."
        )
    total_sessions = payload.get("total_sessions")
    overall_payload = payload.get("overall", {})
    if not isinstance(overall_payload, Mapping):
        raise ValueError(
            f"Random-NZ replay validation failed for {method_key} target {target_item}: "
            f"position_stats overall section is invalid."
        )
    counts_payload = overall_payload.get("counts", {})
    ratios_payload = overall_payload.get("ratios", {})
    if not isinstance(counts_payload, Mapping) or not isinstance(ratios_payload, Mapping):
        raise ValueError(
            f"Random-NZ replay validation failed for {method_key} target {target_item}: "
            "position_stats counts/ratios are invalid."
        )
    expected_counts = {
        str(int(position)): int(count)
        for position, count in replayed_counts.items()
    }
    observed_counts = {
        str(key): int(value)
        for key, value in counts_payload.items()
    }
    if int(total_sessions) != len(records):
        raise ValueError(
            f"Random-NZ replay validation failed for {method_key} target {target_item}: "
            f"replayed total_sessions={len(records)} does not match "
            f"position_stats total_sessions={total_sessions}."
        )
    if observed_counts != expected_counts:
        raise ValueError(
            f"Random-NZ replay validation failed for {method_key} target {target_item}: "
            f"replayed position counts {expected_counts} do not match "
            f"position_stats counts {observed_counts}."
        )
    for position_key, count in expected_counts.items():
        expected_ratio = float(count) / float(len(records))
        observed_ratio = float(ratios_payload.get(position_key, 0.0))
        if abs(expected_ratio - observed_ratio) > 1e-12:
            raise ValueError(
                f"Random-NZ replay validation failed for {method_key} target {target_item}: "
                f"replayed ratio for position {position_key} ({expected_ratio}) does not "
                f"match position_stats ratio ({observed_ratio})."
            )
    return {
        "validated_against_position_stats": True,
        "position_stats_path": str(position_stats_path),
        "position_summary_source": "offline_exact_replay_verified",
        "reason": None,
    }


def _replay_random_nonzero_records(
    bundle: RunBundle,
    *,
    fake_sessions: Sequence[Sequence[int]],
    target_item: int,
) -> list[BucketSessionSelectionRecord]:
    rng = random.Random(int(bundle.seeds["fake_session_seed"]))
    policy = RandomNonzeroWhenPossiblePolicy(
        float(bundle.replacement_topk_ratio or 0.0),
        rng=rng,
    )
    records: list[BucketSessionSelectionRecord] = []
    for session_index, session in enumerate(fake_sessions):
        candidate_build_result = build_candidate_position_result(
            session,
            float(bundle.replacement_topk_ratio or 0.0),
            nonzero_action_when_possible=True,
        )
        result = policy.apply_with_metadata(session, target_item)
        records.append(
            BucketSessionSelectionRecord(
                fake_session_index=int(session_index),
                session_length=int(len(session)),
                target_item=int(target_item),
                base_candidate_positions=tuple(
                    int(position)
                    for position in candidate_build_result.positions_before_mask
                ),
                nonzero_candidate_positions=tuple(
                    int(position) for position in candidate_build_result.positions
                ),
                mode_candidate_positions=tuple(),
                selected_position=int(result.position),
                selected_mode="random_nonzero_when_possible_replay",
                fallback_used=False,
                fallback_reason=None,
                selection_source="legacy_random_nonzero_replay",
                candidate_count=int(len(candidate_build_result.positions)),
                mode_candidate_count=0,
                pos0_removed=bool(candidate_build_result.pos0_removed),
                fallback_to_pos0_only=bool(
                    candidate_build_result.fallback_to_pos0_only
                ),
            )
        )
    return records


def _load_bucket_position_summary_rows(
    bundle: RunBundle,
    *,
    method_key: str,
    expected_target_cohort_key: str,
    expected_target_prefix: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_prefix = [int(item) for item in expected_target_prefix]
    actual_prefix = [int(item) for item in bundle.target_items[: len(expected_prefix)]]
    if bundle.target_cohort_key != expected_target_cohort_key:
        raise ValueError(
            f"{method_key} target_cohort_key mismatch: "
            f"{bundle.target_cohort_key} != {expected_target_cohort_key}"
        )
    if actual_prefix != expected_prefix:
        raise ValueError(
            f"{method_key} target prefix mismatch: {actual_prefix} != {expected_prefix}"
        )
    for target_item, target_artifacts in bundle.target_artifacts.items():
        if target_artifacts.bucket_position_summary_path is None:
            continue
        payload = load_json(target_artifacts.bucket_position_summary_path)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Invalid position_summary.json for {method_key} target {target_item}."
            )
        rows.append(
            {
                "dataset": bundle.dataset,
                "run_group": bundle.run_group_key,
                "method": method_key,
                **_serialize_summary_payload(payload),
            }
        )
    return rows


def _build_k30_summary(
    *,
    long_df: pd.DataFrame,
    position_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    k30_df = long_df[long_df["K"] == 30].copy()
    if k30_df.empty:
        return pd.DataFrame()

    value_columns = [
        "targeted_mrr@30",
        "targeted_recall@30",
        "targeted_ndcg@30",
        "targeted_precision@30",
        "ground_truth_mrr@30",
        "ground_truth_recall@30",
    ]
    rows: list[dict[str, Any]] = []
    for (method, target_item, victim_model), group in k30_df.groupby(
        ["method", "target_item", "victim_model"],
        dropna=False,
        sort=True,
    ):
        row = {
            "method": method,
            "target_item": int(target_item),
            "victim_model": victim_model,
        }
        for metric_key in value_columns:
            metric_row = group[group["metric_key"] == metric_key]
            row[metric_key] = (
                None
                if metric_row.empty
                else float(metric_row.iloc[0]["metric_value"])
            )
        for metric_key in ("ground_truth_mrr@30", "ground_truth_recall@30"):
            metric_row = group[group["metric_key"] == metric_key]
            delta_column = f"{metric_key}_signed_change_pct_vs_clean"
            row[delta_column] = (
                None
                if metric_row.empty or pd.isna(metric_row.iloc[0]["signed_change_pct_vs_clean"])
                else float(metric_row.iloc[0]["signed_change_pct_vs_clean"])
            )
        position_row = _find_position_summary_row(
            position_summary_df,
            method=str(method),
            target_item=int(target_item),
        )
        for field_name in (
            "fallback_ratio",
            "pos1_pct",
            "pos2_pct",
            "pos3_pct",
            "pos4_pos5_pct",
            "pos6plus_pct",
            "mean_absolute_position",
            "median_absolute_position",
            "unique_selected_positions",
            "mean_normalized_position",
            "median_normalized_position",
        ):
            row[field_name] = (
                None
                if position_row is None
                else position_row.get(field_name)
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        by=["method", "target_item", "victim_model"]
    ).reset_index(drop=True)


def _build_compatibility_report(
    *,
    bundles: Mapping[str, RunBundle],
    clean_method_key: str,
    baseline_method_key: str,
    bucket_method_keys: Sequence[str],
    expected_bucket_target_cohort_key: str,
    expected_bucket_target_prefix: Sequence[int],
) -> dict[str, Any]:
    clean_bundle = bundles[clean_method_key]
    baseline_bundle = bundles[baseline_method_key]
    report = {
        "clean_method_key": clean_method_key,
        "baseline_method_key": baseline_method_key,
        "expected_bucket_target_cohort_key": expected_bucket_target_cohort_key,
        "expected_bucket_target_prefix": [int(item) for item in expected_bucket_target_prefix],
        "methods": {},
    }
    clean_reference = str(clean_bundle.run_root)
    baseline_info = _bundle_identity_info(
        baseline_bundle,
        method_key=baseline_method_key,
        clean_reference=clean_reference,
    )
    report["methods"][baseline_method_key] = baseline_info

    for method_key, bundle in bundles.items():
        if method_key == clean_method_key:
            continue
        info = _bundle_identity_info(
            bundle,
            method_key=method_key,
            clean_reference=clean_reference,
        )
        if method_key in bucket_method_keys:
            info["bucket_target_cohort_check"] = {
                "expected_target_cohort_key": expected_bucket_target_cohort_key,
                "actual_target_cohort_key": bundle.target_cohort_key,
                "expected_target_prefix": [int(item) for item in expected_bucket_target_prefix],
                "actual_target_prefix": [
                    int(item)
                    for item in bundle.target_items[: len(expected_bucket_target_prefix)]
                ],
                "passed": (
                    bundle.target_cohort_key == expected_bucket_target_cohort_key
                    and list(bundle.target_items[: len(expected_bucket_target_prefix)])
                    == [int(item) for item in expected_bucket_target_prefix]
                ),
            }
        if method_key != baseline_method_key:
            checks = _compatibility_checks_against_baseline(
                baseline_info=baseline_info,
                method_info=info,
            )
            info["checks_vs_random_nz"] = checks
            info["comparison_enabled_vs_random_nz"] = all(
                check["status"] == "yes" for check in checks
            )
            info["shared_targets_with_random_nz"] = sorted(
                set(info["target_items"]).intersection(baseline_info["target_items"])
            )
            info["shared_victims_with_random_nz"] = sorted(
                set(info["victims"]).intersection(baseline_info["victims"])
            )
        report["methods"][method_key] = info
    return report


def _bundle_identity_info(
    bundle: RunBundle,
    *,
    method_key: str,
    clean_reference: str,
) -> dict[str, Any]:
    split_key = _require_nested(
        bundle.key_payloads,
        ("stable_run_group", "split_identity", "key"),
    )
    shared_attack_artifact_key = _require_nested(
        bundle.key_payloads,
        ("stable_run_group", "attack_identity", "shared_attack_artifact_identity", "key"),
    )
    evaluation_topk = _require_nested(
        bundle.key_payloads,
        ("stable_run_group", "run_group_identity", "payload", "evaluation_schema", "topk"),
    )
    canonical_split_metadata_path = str(_find_shared_artifact_path(bundle, "metadata.json"))
    fake_sessions_path = _find_shared_artifact_path(bundle, "fake_sessions.pkl")
    return {
        "method": method_key,
        "run_root": str(bundle.run_root),
        "run_group": bundle.run_group_key,
        "run_type": bundle.run_type,
        "dataset": bundle.dataset,
        "split_key": split_key,
        "target_cohort_key": bundle.target_cohort_key,
        "target_items": [int(item) for item in bundle.target_items],
        "victims": [str(item) for item in bundle.victims],
        "shared_attack_artifact_key": shared_attack_artifact_key,
        "shared_fake_sessions_path": str(fake_sessions_path),
        "canonical_split_metadata_path": canonical_split_metadata_path,
        "attack_size": bundle.attack_size,
        "replacement_topk_ratio": bundle.replacement_topk_ratio,
        "nonzero_action_when_possible": _resolve_nonzero_action_semantics(bundle),
        "evaluation_topk": [int(item) for item in evaluation_topk],
        "clean_reference": clean_reference,
    }


def _compatibility_checks_against_baseline(
    *,
    baseline_info: Mapping[str, Any],
    method_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for field_name, label in (
        ("dataset", "same dataset"),
        ("split_key", "same canonical split"),
        ("canonical_split_metadata_path", "same train/valid/test split identity"),
        ("shared_attack_artifact_key", "same fake sessions"),
        ("shared_fake_sessions_path", "same fake session path"),
        ("attack_size", "same poison size"),
        ("replacement_topk_ratio", "same replacement_topk_ratio"),
        ("nonzero_action_when_possible", "same nonzero_action_when_possible"),
        ("evaluation_topk", "same metric K list"),
        ("clean_reference", "same clean baseline reference"),
    ):
        baseline_value = baseline_info.get(field_name)
        method_value = method_info.get(field_name)
        checks.append(
            {
                "check": label,
                "baseline_value": baseline_value,
                "method_value": method_value,
                "status": "yes" if baseline_value == method_value else "no",
            }
        )
    return checks


def _build_pairwise_vs_random_nz(
    *,
    long_df: pd.DataFrame,
    compatibility_report: Mapping[str, Any],
    baseline_method_key: str,
    bucket_method_keys: Sequence[str],
) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    targeted_df = long_df[
        long_df["metric_name"].astype(str).str.startswith("targeted_")
    ].copy()
    baseline_rows = targeted_df[targeted_df["method"] == baseline_method_key].copy()
    if baseline_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    methods_info = compatibility_report.get("methods", {})
    for bucket_method_key in bucket_method_keys:
        method_info = methods_info.get(bucket_method_key, {})
        comparison_enabled = bool(method_info.get("comparison_enabled_vs_random_nz"))
        bucket_rows = targeted_df[targeted_df["method"] == bucket_method_key].copy()
        merged = pd.merge(
            bucket_rows,
            baseline_rows,
            on=["target_item", "victim_model", "metric_name", "K"],
            how="inner",
            suffixes=("_bucket", "_baseline"),
        )
        if not comparison_enabled:
            rows.append(
                _pairwise_row(
                    bucket_method=bucket_method_key,
                    breakdown_type="overall",
                    breakdown_value="__all__",
                    total=0,
                    bucket_wins=0,
                    random_wins=0,
                    ties=0,
                    comparison_enabled=False,
                    reason="core compatibility checks failed",
                )
            )
            continue
        if merged.empty:
            rows.append(
                _pairwise_row(
                    bucket_method=bucket_method_key,
                    breakdown_type="overall",
                    breakdown_value="__all__",
                    total=0,
                    bucket_wins=0,
                    random_wins=0,
                    ties=0,
                    comparison_enabled=True,
                    reason="no shared completed targeted metric cells",
                )
            )
            continue

        rows.append(
            _pairwise_stats_for_group(
                merged,
                bucket_method=bucket_method_key,
                breakdown_type="overall",
                breakdown_value="__all__",
            )
        )
        for breakdown_column in ("victim_model", "target_item", "metric_name", "K"):
            for breakdown_value, group in merged.groupby(breakdown_column, dropna=False, sort=True):
                rows.append(
                    _pairwise_stats_for_group(
                        group,
                        bucket_method=bucket_method_key,
                        breakdown_type=breakdown_column,
                        breakdown_value=str(breakdown_value),
                    )
                )
    return pd.DataFrame(rows).sort_values(
        by=["bucket_method", "breakdown_type", "breakdown_value"]
    ).reset_index(drop=True)


def _pairwise_stats_for_group(
    merged: pd.DataFrame,
    *,
    bucket_method: str,
    breakdown_type: str,
    breakdown_value: str,
) -> dict[str, Any]:
    deltas = merged["metric_value_bucket"].astype(float) - merged["metric_value_baseline"].astype(float)
    tolerance = 1e-12
    bucket_wins = int((deltas > tolerance).sum())
    random_wins = int((deltas < -tolerance).sum())
    ties = int((deltas.abs() <= tolerance).sum())
    total = int(len(merged))
    return _pairwise_row(
        bucket_method=bucket_method,
        breakdown_type=breakdown_type,
        breakdown_value=breakdown_value,
        total=total,
        bucket_wins=bucket_wins,
        random_wins=random_wins,
        ties=ties,
        comparison_enabled=True,
        reason=None,
    )


def _pairwise_row(
    *,
    bucket_method: str,
    breakdown_type: str,
    breakdown_value: str,
    total: int,
    bucket_wins: int,
    random_wins: int,
    ties: int,
    comparison_enabled: bool,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "bucket_method": bucket_method,
        "breakdown_type": breakdown_type,
        "breakdown_value": breakdown_value,
        "comparison_enabled": bool(comparison_enabled),
        "reason": reason,
        "total_comparable_targeted_metric_cells": int(total),
        "bucket_wins": int(bucket_wins),
        "random_nz_wins": int(random_wins),
        "ties": int(ties),
        "bucket_win_rate": (
            None
            if total <= 0
            else float(bucket_wins) / float(total)
        ),
    }


def _find_position_summary_row(
    position_summary_df: pd.DataFrame,
    *,
    method: str,
    target_item: int,
) -> dict[str, Any] | None:
    if position_summary_df.empty:
        return None
    matches = position_summary_df[
        (position_summary_df["method"] == method)
        & (position_summary_df["target_item"] == target_item)
    ]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def _selection_seed_for_bundle(bundle: RunBundle) -> int | None:
    if bundle.run_type == "random_nonzero_when_possible":
        raw_seed = bundle.seeds.get("fake_session_seed")
        return None if raw_seed is None else int(raw_seed)
    raw_seed = bundle.seeds.get("position_opt_seed")
    return None if raw_seed is None else int(raw_seed)


def _resolve_nonzero_action_semantics(bundle: RunBundle) -> bool | None:
    if bundle.run_type in {
        "random_nonzero_when_possible",
        "prefix_nonzero_when_possible",
    }:
        return True
    return bundle.nonzero_action_when_possible


def _find_shared_artifact_path(bundle: RunBundle, filename: str) -> Path:
    matches = [
        path
        for path in bundle.shared_artifact_paths.values()
        if path.name == filename
    ]
    if not matches:
        raise FileNotFoundError(
            f"Run bundle {bundle.method_key} is missing shared artifact '{filename}'."
        )
    if len(matches) > 1:
        matches = sorted(set(path.resolve() for path in matches))
    return matches[0]


def _parse_metric_key(metric_key: str) -> tuple[str, int] | None:
    if "@" not in metric_key:
        return None
    metric_name, raw_k = metric_key.rsplit("@", 1)
    if not raw_k.isdigit():
        return None
    return metric_name, int(raw_k)


def _serialize_summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (list, dict)):
            row[key] = json.dumps(value, sort_keys=True)
        else:
            row[key] = value
    return row


def _position_counts_from_records(
    records: Sequence[BucketSessionSelectionRecord],
) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        position = int(record.selected_position)
        counts[position] = counts.get(position, 0) + 1
    return counts


def _require_note_string(notes: Mapping[str, Any], key: str) -> str:
    value = notes.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest notes.{key} must be a non-empty string.")
    return value.strip()


def _require_note_str_list(notes: Mapping[str, Any], key: str) -> list[str]:
    value = notes.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Manifest notes.{key} must be a list of strings.")
    return [item.strip() for item in value]


def _require_note_int_list(notes: Mapping[str, Any], key: str) -> list[int]:
    value = notes.get(key)
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Manifest notes.{key} must be a list of ints.")
    return [int(item) for item in value]


def _require_nested(payload: Mapping[str, Any], path: Iterable[str]) -> Any:
    current: Any = payload
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError("Missing nested key: " + ".".join(walked))
        current = current[key]
    return current


if __name__ == "__main__":
    raise SystemExit(main())
