from __future__ import annotations

import json
import pickle
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.common.paths import run_group_key
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.position_stats import build_position_stats_payload
from analysis.analyze_bucket_sweep import (
    _build_pairwise_vs_random_nz,
    _build_replayed_random_nonzero_position_rows,
    _compatibility_checks_against_baseline,
)
from analysis.utils.run_bundle_loader import load_run_bundle


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_analyze_bucket_sweep" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _summary_current_payload(
    *,
    run_type: str,
    target_items: list[int],
    victims: list[str],
) -> dict[str, object]:
    return {
        "run_group_key": f"run_group_{run_type}",
        "run_type": run_type,
        "target_cohort_key": "target_cohort_test",
        "target_items": target_items,
        "victims": victims,
        "targets": {
            str(target_item): {
                "target_item": target_item,
                "victims": {
                    victim: {
                        "metrics": {
                            "targeted_mrr@30": 0.1 + (target_item % 10) * 0.001,
                            "ground_truth_mrr@30": 0.2,
                        }
                    }
                    for victim in victims
                },
            }
            for target_item in target_items
        },
    }


def _resolved_config_payload(*, replacement_topk_ratio: float) -> dict[str, object]:
    return {
        "result_config": {
            "attack": {
                "replacement_topk_ratio": replacement_topk_ratio,
                "size": 0.01,
                "position_opt": {
                    "nonzero_action_when_possible": True,
                },
            },
            "seeds": {
                "fake_session_seed": 20260405,
                "position_opt_seed": 20260405,
            },
        }
    }


def _artifact_manifest_payload(
    *,
    canonical_dir: Path,
    target_registry_path: Path,
    fake_sessions_path: Path,
) -> dict[str, object]:
    return {
        "shared_artifacts": {
            "canonical_split": {
                "canonical_dir": str(canonical_dir),
                "metadata": str(canonical_dir / "metadata.json"),
            },
            "attack": {
                "fake_sessions": str(fake_sessions_path),
            },
            "target_cohort": {
                "shared_dir": str(target_registry_path.parent),
                "target_registry": str(target_registry_path),
            },
        }
    }


def _key_payloads_payload(*, run_type: str, replacement_topk_ratio: float) -> dict[str, object]:
    return {
        "stable_run_group": {
            "split_identity": {"key": "split_test", "payload": {}},
            "run_group_identity": {
                "key": f"run_group_{run_type}",
                "payload": {
                    "evaluation_schema": {
                        "topk": [5, 10, 15, 20, 25, 30, 40, 50],
                    }
                },
            },
            "attack_identity": {
                "key": f"attack_{run_type}",
                "shared_attack_artifact_identity": {
                    "key": "attack_shared_test",
                    "payload": {},
                },
            },
            "target_cohort_identity": {
                "key": "target_cohort_test",
                "payload": {},
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


def _create_run_root(
    base_dir: Path,
    *,
    experiment_name: str,
    run_type: str,
    target_items: list[int],
    victims: list[str],
    replacement_topk_ratio: float = 1.0,
) -> Path:
    run_root = (
        base_dir
        / "outputs"
        / "runs"
        / "diginetica"
        / experiment_name
        / f"run_group_{run_type}"
    )
    canonical_dir = base_dir / "outputs" / "shared" / "diginetica" / "canonical" / "split_test"
    fake_sessions_path = (
        base_dir
        / "outputs"
        / "shared"
        / "diginetica"
        / "attack"
        / "attack_shared_test"
        / "fake_sessions.pkl"
    )
    target_registry_path = (
        base_dir
        / "outputs"
        / "shared"
        / "diginetica"
        / "target_cohorts"
        / "target_cohort_test"
        / "target_registry.json"
    )

    _write_json(
        run_root / "summary_current.json",
        _summary_current_payload(
            run_type=run_type,
            target_items=target_items,
            victims=victims,
        ),
    )
    _write_json(
        run_root / "resolved_config.json",
        _resolved_config_payload(replacement_topk_ratio=replacement_topk_ratio),
    )
    _write_json(
        run_root / "artifact_manifest.json",
        _artifact_manifest_payload(
            canonical_dir=canonical_dir,
            target_registry_path=target_registry_path,
            fake_sessions_path=fake_sessions_path,
        ),
    )
    _write_json(
        run_root / "key_payloads.json",
        _key_payloads_payload(
            run_type=run_type,
            replacement_topk_ratio=replacement_topk_ratio,
        ),
    )
    _write_json(run_root / "run_coverage.json", _run_coverage_payload(target_items, victims))
    _write_json(run_root / "execution_log.json", {"executions": []})
    _write_json(run_root / "progress.json", {"status": "completed"})

    _write_json(canonical_dir / "metadata.json", {"dataset_name": "diginetica"})
    fake_sessions = [[1, 2], [1, 2, 3], [1, 2, 3, 4, 5]]
    _write_pickle(fake_sessions_path, fake_sessions)
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
            "ordered_targets": [11103, 39588, 5334, 5418],
            "current_count": 4,
            "created_at": "2026-04-26T00:00:00+00:00",
            "updated_at": "2026-04-26T00:00:00+00:00",
        },
    )
    rng = random.Random(20260405)
    policy = RandomNonzeroWhenPossiblePolicy(replacement_topk_ratio, rng=rng)
    replayed_positions = [policy.apply_with_metadata(session, 5334).position for session in fake_sessions]
    position_stats_payload = build_position_stats_payload(
        sessions=fake_sessions,
        positions=replayed_positions,
        run_type=run_type,
        target_item=5334,
    )
    for target_item in target_items:
        _write_json(
            run_root / "targets" / str(target_item) / "position_stats.json",
            {
                **position_stats_payload,
                "target_item": int(target_item),
            },
        )
    return run_root


def test_replayed_random_nonzero_position_rows_mark_unavailable_bucket_only_fields() -> None:
    with _temp_test_dir() as temp_dir:
        run_root = _create_run_root(
            temp_dir,
            experiment_name="attack_random_nonzero_when_possible_ratio1",
            run_type="random_nonzero_when_possible",
            target_items=[5334, 11103],
            victims=["srgnn", "tron"],
        )
        bundle = load_run_bundle(
            run_root=run_root,
            method_key="random_nz",
            label="Random-NZ@1.0",
            dataset_hint="diginetica",
        )

        rows = _build_replayed_random_nonzero_position_rows(
            bundle,
            method_key="random_nz",
        )

    assert len(rows) == 2
    assert rows[0]["position_summary_source"] == "offline_exact_replay_verified"
    assert rows[0]["reconstruction_mode"] == "legacy_random_nonzero_policy"
    assert rows[0]["mode_candidate_count_mean"] is None
    assert rows[0]["seed_source"] == "fake_session_seed"
    assert json.loads(rows[0]["replay_validation"])["validated_against_position_stats"] is True


def test_pairwise_vs_random_nz_uses_only_shared_completed_intersection() -> None:
    long_df = pd.DataFrame(
        [
            {
                "method": "random_nz",
                "target_item": 5334,
                "victim_model": "srgnn",
                "metric_name": "targeted_mrr",
                "K": 30,
                "metric_value": 0.10,
            },
            {
                "method": "random_nz",
                "target_item": 11103,
                "victim_model": "srgnn",
                "metric_name": "targeted_mrr",
                "K": 30,
                "metric_value": 0.20,
            },
            {
                "method": "bucket_abs_pos2",
                "target_item": 5334,
                "victim_model": "srgnn",
                "metric_name": "targeted_mrr",
                "K": 30,
                "metric_value": 0.15,
            },
        ]
    )
    compatibility_report = {
        "methods": {
            "bucket_abs_pos2": {
                "comparison_enabled_vs_random_nz": True,
            }
        }
    }

    pairwise_df = _build_pairwise_vs_random_nz(
        long_df=long_df,
        compatibility_report=compatibility_report,
        baseline_method_key="random_nz",
        bucket_method_keys=["bucket_abs_pos2"],
    )

    overall = pairwise_df[
        (pairwise_df["bucket_method"] == "bucket_abs_pos2")
        & (pairwise_df["breakdown_type"] == "overall")
    ].iloc[0]
    assert overall["total_comparable_targeted_metric_cells"] == 1
    assert overall["bucket_wins"] == 1
    assert overall["random_nz_wins"] == 0


def test_compatibility_checks_detect_ratio_mismatch() -> None:
    checks = _compatibility_checks_against_baseline(
        baseline_info={
            "dataset": "diginetica",
            "split_key": "split_test",
            "canonical_split_metadata_path": "canonical_a",
            "shared_attack_artifact_key": "attack_shared_x",
            "shared_fake_sessions_path": "fake_a",
            "attack_size": 0.01,
            "replacement_topk_ratio": 1.0,
            "nonzero_action_when_possible": True,
            "evaluation_topk": [5, 10, 30],
            "clean_reference": "clean_a",
        },
        method_info={
            "dataset": "diginetica",
            "split_key": "split_test",
            "canonical_split_metadata_path": "canonical_a",
            "shared_attack_artifact_key": "attack_shared_x",
            "shared_fake_sessions_path": "fake_a",
            "attack_size": 0.01,
            "replacement_topk_ratio": 0.2,
            "nonzero_action_when_possible": True,
            "evaluation_topk": [5, 10, 30],
            "clean_reference": "clean_a",
        },
    )

    ratio_check = next(
        check for check in checks if check["check"] == "same replacement_topk_ratio"
    )
    assert ratio_check["status"] == "no"


def test_bucket_sweep_manifest_run_roots_match_computed_run_group_keys() -> None:
    manifest_path = (
        REPO_ROOT / "analysis" / "configs" / "diagnosis" / "diginetica_bucket_sweep.yaml"
    )
    config_path = (
        REPO_ROOT / "attack" / "configs" / "diginetica_attack_bucket_position_baselines_ratio1.yaml"
    )
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config = load_config(config_path)

    methods_payload = manifest_payload["methods"]
    experiment_name = config.experiment.name
    for method_name in (
        "bucket_first_nonzero",
        "bucket_abs_pos2",
        "bucket_nonfirst_nonzero",
        "bucket_abs_pos3plus",
    ):
        expected_run_group = run_group_key(config, run_type=method_name)
        expected_run_root = (
            f"outputs/runs/diginetica/{experiment_name}/{expected_run_group}"
        )
        assert methods_payload[method_name]["run_root"] == expected_run_root
