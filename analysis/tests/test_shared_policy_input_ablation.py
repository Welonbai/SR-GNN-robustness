from __future__ import annotations

import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.diagnosis.shared_policy_input_ablation.run import load_manifest, run_report


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_shared_policy_input_ablation" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_torch(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _metrics(seed: float) -> dict[str, float]:
    return {
        "targeted_recall@10": seed + 0.01,
        "targeted_recall@20": seed + 0.02,
        "targeted_recall@30": seed + 0.03,
        "targeted_mrr@10": seed + 0.04,
        "targeted_mrr@20": seed + 0.05,
        "targeted_mrr@30": seed + 0.06,
        "ground_truth_recall@10": seed + 0.07,
        "ground_truth_recall@20": seed + 0.08,
        "ground_truth_recall@30": seed + 0.09,
        "ground_truth_mrr@10": seed + 0.10,
        "ground_truth_mrr@20": seed + 0.11,
        "ground_truth_mrr@30": seed + 0.12,
    }


def _summary_current_payload(target_metrics: dict[int, dict[str, float]]) -> dict[str, object]:
    return {
        "targets": {
            str(target_item): {
                "victims": {
                    "tron": {
                        "metrics": metrics,
                    }
                }
            }
            for target_item, metrics in target_metrics.items()
        }
    }


def _position_stats_payload(target_item: int, *, counts: dict[int, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "run_type": "position_opt_shared_policy",
        "target_item": int(target_item),
        "total_sessions": int(total),
        "overall": {
            "counts": {str(position): int(count) for position, count in counts.items()},
            "ratios": {
                str(position): float(count) / float(total)
                for position, count in counts.items()
            },
        },
    }


def _training_history_payload(
    target_item: int,
    *,
    policy_feature_set: str | None,
    active_item_features: list[str] | None,
    active_scalar_features: list[str] | None,
    policy_input_dim: int | None,
    policy_embedding_dim: int = 16,
    policy_hidden_dim: int = 32,
    prefix_score_enabled: bool | None = False,
    include_verification_fields: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_item": int(target_item),
        "policy_representation": "shared_contextual_mlp",
        "position_opt_config": {
            "policy_embedding_dim": int(policy_embedding_dim),
            "policy_hidden_dim": int(policy_hidden_dim),
        },
        "training_history": [
            {
                "outer_step": 0,
                "mean_entropy": 1.2,
                "reward": 0.01,
                "baseline": None,
                "advantage": 0.01,
                "policy_loss": 0.5,
                "target_utility": 0.001,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 1, 1, 2],
            },
            {
                "outer_step": 1,
                "mean_entropy": 0.8,
                "reward": 0.02,
                "baseline": 0.01,
                "advantage": 0.01,
                "policy_loss": 0.4,
                "target_utility": 0.002,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 0, 1, 2],
            },
        ],
        "final_selected_positions": [
            {"candidate_index": 0, "position": 0, "score": 1.0},
            {"candidate_index": 0, "position": 0, "score": 2.0},
            {"candidate_index": 1, "position": 1, "score": 3.0},
            {"candidate_index": 2, "position": 2, "score": 4.0},
        ],
    }
    if include_verification_fields:
        payload.update(
            {
                "policy_feature_set": policy_feature_set,
                "policy_item_feature_names": active_item_features,
                "policy_scalar_feature_names": active_scalar_features,
                "active_item_features": active_item_features,
                "active_scalar_features": active_scalar_features,
                "policy_input_dim": policy_input_dim,
                "policy_embedding_dim": int(policy_embedding_dim),
                "policy_hidden_dim": int(policy_hidden_dim),
                "prefix_score_enabled": prefix_score_enabled,
            }
        )
    return payload


def _run_metadata_payload(
    target_item: int,
    *,
    policy_feature_set: str | None,
    active_item_features: list[str] | None,
    active_scalar_features: list[str] | None,
    policy_input_dim: int | None,
    policy_embedding_dim: int = 16,
    policy_hidden_dim: int = 32,
    prefix_score_enabled: bool | None = False,
    include_verification_fields: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "target_item": int(target_item),
        "position_opt_config": {
            "policy_embedding_dim": int(policy_embedding_dim),
            "policy_hidden_dim": int(policy_hidden_dim),
        },
    }
    if include_verification_fields:
        payload.update(
            {
                "policy_feature_set": policy_feature_set,
                "policy_item_feature_names": active_item_features,
                "policy_scalar_feature_names": active_scalar_features,
                "active_item_features": active_item_features,
                "active_scalar_features": active_scalar_features,
                "policy_input_dim": policy_input_dim,
                "policy_embedding_dim": int(policy_embedding_dim),
                "policy_hidden_dim": int(policy_hidden_dim),
                "prefix_score_enabled": prefix_score_enabled,
            }
        )
    return payload


def _learned_logits_payload(
    *,
    policy_feature_set: str | None,
    active_item_features: list[str] | None,
    active_scalar_features: list[str] | None,
    policy_input_dim: int | None,
    policy_embedding_dim: int = 16,
    policy_hidden_dim: int = 32,
    prefix_score_enabled: bool | None = False,
    legacy: bool = False,
) -> dict[str, object]:
    if legacy:
        return {
            "policy_config": {
                "embedding_dim": int(policy_embedding_dim),
                "hidden_dim": int(policy_hidden_dim),
                "num_item_embeddings": 100,
            },
            "policy_representation": "shared_contextual_mlp",
            "sessions": [],
        }
    return {
        "policy_representation": "shared_contextual_mlp",
        "prefix_score_enabled": prefix_score_enabled,
        "policy_config": {
            "policy_embedding_dim": int(policy_embedding_dim),
            "policy_hidden_dim": int(policy_hidden_dim),
            "policy_input_dim": policy_input_dim,
            "policy_feature_set": policy_feature_set,
            "active_item_features": active_item_features,
            "active_scalar_features": active_scalar_features,
            "item_feature_names": active_item_features,
            "scalar_feature_names": active_scalar_features,
            "num_item_embeddings": 100,
        },
        "sessions": [],
    }


def _create_run_root(
    base_dir: Path,
    name: str,
    *,
    target_metrics: dict[int, dict[str, float]],
    position_stats_by_target: dict[int, dict[str, object]] | None = None,
    training_history_by_target: dict[int, dict[str, object]] | None = None,
    run_metadata_by_target: dict[int, dict[str, object]] | None = None,
    learned_logits_by_target: dict[int, dict[str, object]] | None = None,
) -> Path:
    run_root = base_dir / name
    run_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_root / "summary_current.json", _summary_current_payload(target_metrics))
    for target_item, payload in (position_stats_by_target or {}).items():
        _write_json(run_root / "targets" / str(target_item) / "position_stats.json", payload)
    for target_item, payload in (training_history_by_target or {}).items():
        _write_json(
            run_root / "targets" / str(target_item) / "position_opt" / "training_history.json",
            payload,
        )
    for target_item, payload in (run_metadata_by_target or {}).items():
        _write_json(
            run_root / "targets" / str(target_item) / "position_opt" / "run_metadata.json",
            payload,
        )
    for target_item, payload in (learned_logits_by_target or {}).items():
        _write_torch(
            run_root / "targets" / str(target_item) / "position_opt" / "learned_logits.pt",
            payload,
        )
    return run_root


def test_load_manifest_accepts_arbitrary_method_order_without_ratio_specific_keys() -> None:
    with _temp_test_dir() as temp_dir:
        clean_root = _create_run_root(
            temp_dir,
            "clean",
            target_metrics={11103: _metrics(0.1)},
        )
        shared_root = _create_run_root(
            temp_dir,
            "shared_local",
            target_metrics={11103: _metrics(0.2)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 3, 1: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
        )
        config_path = temp_dir / "manifest.yaml"
        manifest_payload = {
            "report_id": "unit_test_manifest_order",
            "dataset": "diginetica",
            "victim_model": "tron",
            "reference_method": "shared_policy_local_context",
            "targets": {
                "required": [11103],
                "optional_if_available": [],
            },
            "methods": {
                "shared_policy_local_context": {
                    "label": "Shared local_context",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(shared_root),
                    "summary_current": str(shared_root / "summary_current.json"),
                    "expected_policy_feature_set": "local_context",
                    "expected_active_item_features": [
                        "target_item",
                        "original_item",
                        "left_item",
                        "right_item",
                    ],
                    "expected_active_scalar_features": [
                        "position_index",
                        "normalized_position",
                        "session_length",
                    ],
                    "expected_prefix_score_enabled": False,
                },
                "clean": {
                    "label": "Clean",
                    "attack_method": "clean",
                    "run_root": str(clean_root),
                    "summary_current": str(clean_root / "summary_current.json"),
                },
            },
        }
        _write_yaml(config_path, manifest_payload)

        manifest = load_manifest(config_path)

        assert manifest.reference_method == "shared_policy_local_context"
        assert tuple(manifest.methods) == ("shared_policy_local_context", "clean")


def test_run_report_supports_mixed_methods_and_verification_outputs() -> None:
    with _temp_test_dir() as temp_dir:
        local_context_root = _create_run_root(
            temp_dir,
            "shared_local_context",
            target_metrics={11103: _metrics(0.20)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 4, 1: 1, 2: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
            learned_logits_by_target={
                11103: _learned_logits_payload(
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
        )
        clean_root = _create_run_root(
            temp_dir,
            "clean",
            target_metrics={11103: _metrics(0.10)},
        )
        prefix_root = _create_run_root(
            temp_dir,
            "prefix",
            target_metrics={},
        )
        posopt_mvp_root = _create_run_root(
            temp_dir,
            "posopt_mvp",
            target_metrics={11103: _metrics(0.15)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 2, 1: 2, 3: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="legacy_mvp",
                    active_item_features=["target_item"],
                    active_scalar_features=["position_index"],
                    policy_input_dim=17,
                )
            },
        )
        normalized_only_root = _create_run_root(
            temp_dir,
            "shared_normalized_only",
            target_metrics={11103: _metrics(0.18), 22222: _metrics(0.19)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 5, 1: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="normalized_position_only",
                    active_item_features=[],
                    active_scalar_features=["normalized_position"],
                    policy_input_dim=1,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set="normalized_position_only",
                    active_item_features=[],
                    active_scalar_features=["normalized_position"],
                    policy_input_dim=1,
                )
            },
        )
        bad_expectation_root = _create_run_root(
            temp_dir,
            "shared_bad_expectation",
            target_metrics={11103: _metrics(0.12)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={1: 3, 2: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="normalized_position_only",
                    active_item_features=[],
                    active_scalar_features=["normalized_position"],
                    policy_input_dim=1,
                    prefix_score_enabled=False,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set="target_normalized_position",
                    active_item_features=["target_item"],
                    active_scalar_features=["normalized_position"],
                    policy_input_dim=17,
                    prefix_score_enabled=True,
                )
            },
        )
        legacy_root = _create_run_root(
            temp_dir,
            "shared_legacy",
            target_metrics={11103: _metrics(0.11)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 1, 2: 2})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set=None,
                    active_item_features=None,
                    active_scalar_features=None,
                    policy_input_dim=None,
                    include_verification_fields=False,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set=None,
                    active_item_features=None,
                    active_scalar_features=None,
                    policy_input_dim=None,
                    include_verification_fields=False,
                )
            },
            learned_logits_by_target={
                11103: _learned_logits_payload(
                    policy_feature_set=None,
                    active_item_features=None,
                    active_scalar_features=None,
                    policy_input_dim=None,
                    legacy=True,
                )
            },
        )

        config_path = temp_dir / "manifest.yaml"
        manifest_payload = {
            "report_id": "unit_test_shared_policy_input_ablation",
            "dataset": "diginetica",
            "victim_model": "tron",
            "reference_method": "shared_policy_local_context",
            "targets": {
                "required": [11103],
                "optional_if_available": [22222],
            },
            "methods": {
                "shared_policy_local_context": {
                    "label": "Shared local_context",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(local_context_root),
                    "summary_current": str(local_context_root / "summary_current.json"),
                    "expected_policy_feature_set": "local_context",
                    "expected_active_item_features": [
                        "target_item",
                        "original_item",
                        "left_item",
                        "right_item",
                    ],
                    "expected_active_scalar_features": [
                        "position_index",
                        "normalized_position",
                        "session_length",
                    ],
                    "expected_prefix_score_enabled": False,
                },
                "clean": {
                    "label": "Clean",
                    "attack_method": "clean",
                    "run_root": str(clean_root),
                    "summary_current": str(clean_root / "summary_current.json"),
                },
                "prefix_nz": {
                    "label": "Prefix-NZ",
                    "attack_method": "prefix_nonzero_when_possible",
                    "run_root": str(prefix_root),
                    "summary_current": str(prefix_root / "summary_current.json"),
                },
                "posopt_mvp": {
                    "label": "PosOptMVP",
                    "attack_method": "position_opt_mvp",
                    "run_root": str(posopt_mvp_root),
                    "summary_current": str(posopt_mvp_root / "summary_current.json"),
                },
                "shared_policy_normalized_position_only": {
                    "label": "Shared normalized_position_only",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(normalized_only_root),
                    "summary_current": str(normalized_only_root / "summary_current.json"),
                    "expected_policy_feature_set": "normalized_position_only",
                    "expected_active_item_features": [],
                    "expected_active_scalar_features": ["normalized_position"],
                    "expected_prefix_score_enabled": False,
                },
                "shared_policy_bad_expectation": {
                    "label": "Shared bad expectation",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(bad_expectation_root),
                    "summary_current": str(bad_expectation_root / "summary_current.json"),
                    "expected_policy_feature_set": "normalized_position_only",
                    "expected_active_item_features": [],
                    "expected_active_scalar_features": ["normalized_position"],
                    "expected_prefix_score_enabled": False,
                },
                "shared_policy_legacy": {
                    "label": "Shared legacy",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(legacy_root),
                    "summary_current": str(legacy_root / "summary_current.json"),
                    "expected_policy_feature_set": "full_context_normalized_position",
                    "expected_active_item_features": [
                        "target_item",
                        "original_item",
                        "left_item",
                        "right_item",
                    ],
                    "expected_active_scalar_features": ["normalized_position"],
                    "expected_prefix_score_enabled": False,
                },
            },
        }
        _write_yaml(config_path, manifest_payload)

        output_dir = run_report(config_path=config_path, output_root=temp_dir / "outputs")

        final_metrics = pd.read_csv(output_dir / "final_metrics.csv")
        final_position_summary = pd.read_csv(output_dir / "final_position_summary.csv")
        training_dynamics = pd.read_csv(output_dir / "training_dynamics.csv")
        training_final_step_summary = pd.read_csv(output_dir / "training_final_step_summary.csv")
        verification_summary = pd.read_csv(output_dir / "verification_summary.csv")
        delta_vs_reference = pd.read_csv(output_dir / "delta_vs_reference.csv")
        report_data = json.loads((output_dir / "report_data.json").read_text(encoding="utf-8"))
        report_markdown = (output_dir / "report.md").read_text(encoding="utf-8")

        assert report_data["selected_targets"] == [11103]
        assert (output_dir / "manifest_resolved.json").is_file()

        method_order = final_metrics.loc[final_metrics["target_item"] == 11103, "method_key"].tolist()
        assert method_order == [
            "shared_policy_local_context",
            "clean",
            "prefix_nz",
            "posopt_mvp",
            "shared_policy_normalized_position_only",
            "shared_policy_bad_expectation",
            "shared_policy_legacy",
        ]
        prefix_row = final_metrics[final_metrics["method_key"] == "prefix_nz"].iloc[0]
        assert pd.isna(prefix_row["target_recall@10"])

        normalized_position_row = final_position_summary[
            final_position_summary["method_key"] == "shared_policy_normalized_position_only"
        ].iloc[0]
        assert normalized_position_row["dominant_position"] == 0
        assert normalized_position_row["top5_positions"].startswith("0:")

        assert set(training_dynamics["method_key"]) == {
            "shared_policy_local_context",
            "posopt_mvp",
            "shared_policy_normalized_position_only",
            "shared_policy_bad_expectation",
            "shared_policy_legacy",
        }
        assert set(training_final_step_summary["method_key"]) == set(training_dynamics["method_key"])

        local_context_verification = verification_summary[
            verification_summary["method_key"] == "shared_policy_local_context"
        ].iloc[0]
        assert local_context_verification["verification_status"] == "ok"
        assert local_context_verification["expected_input_dim_from_features"] == 67

        normalized_only_verification = verification_summary[
            verification_summary["method_key"] == "shared_policy_normalized_position_only"
        ].iloc[0]
        assert normalized_only_verification["verification_status"] == "ok"
        assert normalized_only_verification["expected_input_dim_from_features"] == 1
        assert bool(normalized_only_verification["learned_logits_present"]) is False
        assert normalized_only_verification["artifact_consistency_status"] == "ok"

        mismatch_verification = verification_summary[
            verification_summary["method_key"] == "shared_policy_bad_expectation"
        ].iloc[0]
        assert mismatch_verification["verification_status"] == "mismatch"
        assert mismatch_verification["artifact_consistency_status"] == "mismatch"
        assert bool(mismatch_verification["feature_set_matches_expected"]) is False
        assert bool(mismatch_verification["active_features_match_expected"]) is False
        assert bool(mismatch_verification["input_dim_matches_expected"]) is False
        assert bool(mismatch_verification["prefix_flag_matches_expected"]) is False

        legacy_verification = verification_summary[
            verification_summary["method_key"] == "shared_policy_legacy"
        ].iloc[0]
        assert legacy_verification["verification_status"] == "legacy_missing_fields"
        assert legacy_verification["artifact_consistency_status"] == "legacy_missing_fields"
        assert bool(legacy_verification["learned_logits_present"]) is True

        prefix_delta = delta_vs_reference[delta_vs_reference["method_key"] == "prefix_nz"].iloc[0]
        assert pd.isna(prefix_delta["target_recall@10"])

        assert "## A. Final Metrics Comparison" in report_markdown
        assert "## C. Final Position Summary" in report_markdown
        assert "## D. Verification Summary" in report_markdown
        assert "## E. Training Final-Step Summary" in report_markdown
        assert "## F. Training Dynamics" in report_markdown
        assert "ratio=0.5" not in report_markdown
        assert "Compare Selected Positions Against ratio=1.0" not in report_markdown


def test_run_report_treats_zero_session_clean_position_stats_as_missing_summary() -> None:
    with _temp_test_dir() as temp_dir:
        shared_root = _create_run_root(
            temp_dir,
            "shared_local_context",
            target_metrics={11103: _metrics(0.20)},
            position_stats_by_target={
                11103: _position_stats_payload(11103, counts={0: 4, 1: 1, 2: 1})
            },
            training_history_by_target={
                11103: _training_history_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
            run_metadata_by_target={
                11103: _run_metadata_payload(
                    11103,
                    policy_feature_set="local_context",
                    active_item_features=["target_item", "original_item", "left_item", "right_item"],
                    active_scalar_features=["position_index", "normalized_position", "session_length"],
                    policy_input_dim=67,
                )
            },
        )
        clean_root = _create_run_root(
            temp_dir,
            "clean",
            target_metrics={11103: _metrics(0.10)},
            position_stats_by_target={
                11103: {
                    "run_type": "clean",
                    "target_item": 11103,
                    "total_sessions": 0,
                    "overall": {
                        "counts": {},
                        "ratios": {},
                    },
                }
            },
        )

        config_path = temp_dir / "manifest.yaml"
        manifest_payload = {
            "report_id": "unit_test_shared_policy_input_ablation_zero_clean_positions",
            "dataset": "diginetica",
            "victim_model": "tron",
            "reference_method": "shared_policy_local_context",
            "targets": {
                "required": [11103],
                "optional_if_available": [],
            },
            "methods": {
                "shared_policy_local_context": {
                    "label": "Shared local_context",
                    "attack_method": "position_opt_shared_policy",
                    "run_root": str(shared_root),
                    "summary_current": str(shared_root / "summary_current.json"),
                    "expected_policy_feature_set": "local_context",
                    "expected_active_item_features": [
                        "target_item",
                        "original_item",
                        "left_item",
                        "right_item",
                    ],
                    "expected_active_scalar_features": [
                        "position_index",
                        "normalized_position",
                        "session_length",
                    ],
                    "expected_prefix_score_enabled": False,
                },
                "clean": {
                    "label": "Clean",
                    "attack_method": "clean",
                    "run_root": str(clean_root),
                    "summary_current": str(clean_root / "summary_current.json"),
                },
            },
        }
        _write_yaml(config_path, manifest_payload)

        output_dir = run_report(config_path=config_path, output_root=temp_dir / "outputs")
        final_position_summary = pd.read_csv(output_dir / "final_position_summary.csv")

        clean_row = final_position_summary[final_position_summary["method_key"] == "clean"].iloc[0]
        assert pd.isna(clean_row["total"])
        assert pd.isna(clean_row["dominant_position"])
