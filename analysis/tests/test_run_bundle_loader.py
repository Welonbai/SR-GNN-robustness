from __future__ import annotations

import json
import pickle
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.utils.run_bundle_loader import (
    RunBundleLoaderError,
    bundle_to_dict,
    load_bundles_from_manifest,
)


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_run_bundle_loader" / uuid4().hex
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


def _write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _summary_current_payload(target_items: list[int], victims: list[str]) -> dict[str, object]:
    return {
        "run_group_key": "run_group_test",
        "run_type": "position_opt_shared_policy",
        "target_cohort_key": "target_cohort_test",
        "target_items": target_items,
        "victims": victims,
        "targets": {
            str(target_item): {
                "target_item": target_item,
                "victims": {
                    victim: {
                        "metrics": {
                            "targeted_recall@10": 0.1,
                            "ground_truth_recall@10": 0.2,
                        }
                    }
                    for victim in victims
                },
            }
            for target_item in target_items
        },
    }


def _resolved_config_payload() -> dict[str, object]:
    return {
        "result_config": {
            "attack": {
                "replacement_topk_ratio": 1.0,
                "size": 0.01,
                "position_opt": {
                    "nonzero_action_when_possible": True,
                    "policy_feature_set": "local_context",
                    "reward_mode": "poisoned_target_utility",
                    "final_policy_selection": "last",
                    "deterministic_eval_every": 0,
                    "deterministic_eval_include_final": True,
                },
            },
            "seeds": {
                "target_selection_seed": 20260405,
                "fake_session_seed": 20260405,
            },
        }
    }


def _run_coverage_payload(target_items: list[int], victims: list[str]) -> dict[str, object]:
    return {
        "run_group_key": "run_group_test",
        "target_cohort_key": "target_cohort_test",
        "targets_order": target_items,
        "victims": {victim: {"status": "completed"} for victim in victims},
        "cells": {},
        "created_at": "2026-04-26T00:00:00+00:00",
        "updated_at": "2026-04-26T00:00:00+00:00",
    }


def _artifact_manifest_payload(
    *,
    canonical_dir: Path,
    target_registry_path: Path,
) -> dict[str, object]:
    return {
        "shared_artifacts": {
            "canonical_split": {
                "canonical_dir": str(canonical_dir),
                "metadata": str(canonical_dir / "metadata.json"),
                "item_map": str(canonical_dir / "item_map.pkl"),
                "train_sub": str(canonical_dir / "train_sub.pkl"),
                "valid": str(canonical_dir / "valid.pkl"),
                "test": str(canonical_dir / "test.pkl"),
            },
            "target_cohort": {
                "shared_dir": str(target_registry_path.parent),
                "target_registry": str(target_registry_path),
            },
        }
    }


def _key_payloads_payload() -> dict[str, object]:
    return {
        "stable_run_group": {
            "target_cohort_identity": {
                "key": "target_cohort_test",
                "payload": {
                    "bucket": "popular",
                    "mode": "sampled",
                    "split_key": "split_test",
                    "target_selection_seed": 20260405,
                },
            }
        }
    }


def _create_run_root(base_dir: Path, *, target_items: list[int], victims: list[str]) -> Path:
    run_root = base_dir / "outputs" / "runs" / "diginetica" / "attack_position_opt_shared_policy_nonzero" / "run_group_test"
    canonical_dir = base_dir / "outputs" / "shared" / "diginetica" / "canonical" / "split_test"
    target_registry_path = (
        base_dir
        / "outputs"
        / "shared"
        / "diginetica"
        / "target_cohorts"
        / "target_cohort_test"
        / "target_registry.json"
    )

    _write_json(run_root / "summary_current.json", _summary_current_payload(target_items, victims))
    _write_json(run_root / "resolved_config.json", _resolved_config_payload())
    _write_json(
        run_root / "artifact_manifest.json",
        _artifact_manifest_payload(canonical_dir=canonical_dir, target_registry_path=target_registry_path),
    )
    _write_json(run_root / "key_payloads.json", _key_payloads_payload())
    _write_json(run_root / "run_coverage.json", _run_coverage_payload(target_items, victims))
    _write_json(run_root / "execution_log.json", {"events": []})
    _write_json(run_root / "progress.json", {"status": "completed"})
    _write_json(run_root / "summary_position_opt_shared_policy.json", {"legacy": True})

    _write_json(canonical_dir / "metadata.json", {"dataset_name": "diginetica"})
    _write_pickle(canonical_dir / "item_map.pkl", {})
    _write_pickle(canonical_dir / "train_sub.pkl", [[1, 2, 3], [4, 5]])
    _write_pickle(canonical_dir / "valid.pkl", [[6, 7]])
    _write_pickle(canonical_dir / "test.pkl", [[8, 9]])
    _write_json(
        target_registry_path,
        {
            "target_cohort_key": "target_cohort_test",
            "split_key": "split_test",
            "selection_policy_version": "appendable_target_cohort_v1",
            "mode": "sampled",
            "bucket": "popular",
            "seed": 20260405,
            "explicit_list": None,
            "candidate_pool_hash": "hash",
            "candidate_pool_size": 4,
            "ordered_targets": target_items,
            "current_count": len(target_items),
            "created_at": "2026-04-26T00:00:00+00:00",
            "updated_at": "2026-04-26T00:00:00+00:00",
        },
    )

    target_dir = run_root / "targets" / str(target_items[0])
    _write_json(target_dir / "position_stats.json", {"total_sessions": 2})
    _write_json(target_dir / "position_summary.json", {"method_name": "bucket_abs_pos2"})
    _write_json(target_dir / "bucket_diagnostics.json", {"fallback_ratio": 12.5})
    target_dir.joinpath("selected_positions.jsonl").parent.mkdir(parents=True, exist_ok=True)
    target_dir.joinpath("selected_positions.jsonl").write_text(
        json.dumps({"selected_position": 2}) + "\n",
        encoding="utf-8",
    )
    _write_json(target_dir / "position_opt" / "training_history.json", {"training_history": []})
    _write_json(target_dir / "position_opt" / "run_metadata.json", {"target_item": target_items[0]})
    _write_json(target_dir / "position_opt" / "selected_positions.json", [{"position": 1}])
    (target_dir / "position_opt" / "learned_logits.pt").parent.mkdir(parents=True, exist_ok=True)
    (target_dir / "position_opt" / "learned_logits.pt").write_bytes(b"pt")
    _write_pickle(target_dir / "position_opt" / "optimized_poisoned_sessions.pkl", [[1, 2, 3]])
    _write_pickle(target_dir / "prefix_nonzero_when_possible_metadata.pkl", [{"position": 1}])
    _write_json(target_dir / "random_nonzero_position_metadata.json", {"position": 1})
    for victim in victims:
        victim_dir = target_dir / "victims" / victim
        _write_json(victim_dir / "metrics.json", {"targeted_recall@10": 0.1})
        _write_json(victim_dir / "predictions.json", {"rows": []})
        _write_json(victim_dir / "train_history.json", {"epochs": []})
        _write_json(victim_dir / "resolved_config.json", {"victim": victim})
        victim_dir.joinpath("config.yaml").parent.mkdir(parents=True, exist_ok=True)
        victim_dir.joinpath("config.yaml").write_text("name: test\n", encoding="utf-8")
    return run_root


def test_load_bundles_from_manifest_derives_artifacts_from_run_root() -> None:
    with _temp_test_dir() as temp_dir:
        target_items = [11103, 39588, 5334, 5418]
        victims = ["srgnn", "miasrec", "tron"]
        run_root = _create_run_root(temp_dir, target_items=target_items, victims=victims)
        manifest_path = temp_dir / "manifest.yaml"
        _write_yaml(
            manifest_path,
            {
                "report_id": "unit_test_bundle",
                "dataset": "diginetica",
                "output_dir": str(temp_dir / "analysis_output"),
                "targets": {"expected": [11103, 39588, 5334]},
                "victims": {"expected": ["miasrec", "srgnn"]},
                "methods": {
                    "sp_nz": {
                        "label": "SharedPolicy-local_context-NZ@1.0",
                        "attack_method": "position_opt_shared_policy",
                        "run_root": str(run_root),
                    }
                },
                "notes": {"usage": "test"},
            },
        )

        manifest, bundles = load_bundles_from_manifest(manifest_path)

        assert manifest.report_id == "unit_test_bundle"
        bundle = bundles["sp_nz"]
        assert bundle.run_group_key == "run_group_test"
        assert bundle.dataset == "diginetica"
        assert bundle.target_items == (11103, 39588, 5334, 5418)
        assert bundle.victims == ("miasrec", "srgnn", "tron")
        assert bundle.policy_feature_set == "local_context"
        assert bundle.nonzero_action_when_possible is True
        assert "canonical_split.train_sub" in bundle.shared_artifact_paths
        assert bundle.target_artifacts[11103].position_stats_path is not None
        assert bundle.target_artifacts[11103].bucket_selected_positions_path is not None
        assert bundle.target_artifacts[11103].bucket_position_summary_path is not None
        assert bundle.target_artifacts[11103].bucket_diagnostics_path is not None
        assert bundle.target_artifacts[11103].training_history_path is not None
        assert bundle.target_artifacts[11103].victims["srgnn"].metrics_path is not None

        payload = bundle_to_dict(bundle)
        assert payload["artifacts"]["summary_current"].endswith("summary_current.json")
        assert payload["targets"]["11103"]["position_opt_dir"].endswith("/position_opt")
        assert payload["targets"]["11103"]["bucket_selected_positions"].endswith(
            "/selected_positions.jsonl"
        )


def test_loader_rejects_expected_targets_that_do_not_match_prefix() -> None:
    with _temp_test_dir() as temp_dir:
        run_root = _create_run_root(
            temp_dir,
            target_items=[11103, 39588, 5334],
            victims=["srgnn", "miasrec", "tron"],
        )
        manifest_path = temp_dir / "manifest_bad.yaml"
        _write_yaml(
            manifest_path,
            {
                "report_id": "unit_test_bad_prefix",
                "dataset": "diginetica",
                "targets": {"expected": [39588, 11103]},
                "victims": {"expected": ["srgnn"]},
                "methods": {
                    "sp_nz": {
                        "label": "SharedPolicy-local_context-NZ@1.0",
                        "run_root": str(run_root),
                    }
                },
            },
        )

        with pytest.raises(RunBundleLoaderError, match="target prefix"):
            load_bundles_from_manifest(manifest_path)
