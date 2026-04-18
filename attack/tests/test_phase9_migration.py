from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from analysis.pipeline.long_csv_generator import generate_long_table_bundle
from attack.common.artifact_io import (
    load_execution_log,
    load_json,
    load_run_coverage,
    load_summary_current,
    load_target_registry,
    save_json,
)
from attack.common.config import load_config
from attack.common.paths import run_artifact_paths, run_metadata_paths, shared_artifact_paths
from attack.tools.migrate_legacy_runs import inspect_legacy_run, migrate_legacy_runs


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 9 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase9_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase9" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _legacy_string(path: Path) -> str:
    return _repo_relative(path).replace("/", "\\")


def _legacy_result_config(
    *,
    victims: tuple[str, ...],
    targets_mode: str,
    target_items: list[int],
    bucket: str,
) -> dict[str, object]:
    base = _base_config()
    payload = base.result_config_dict()
    payload["seeds"] = {
        "fake_session_seed": int(base.seeds.fake_session_seed),
        "target_selection_seed": int(base.seeds.target_selection_seed),
    }
    payload["victims"] = {
        "enabled": list(victims),
        "params": {victim: payload["victims"]["params"][victim] for victim in victims},
    }
    payload["targets"] = {
        "mode": targets_mode,
        "bucket": bucket,
        "count": len(target_items),
        "explicit_list": list(target_items) if targets_mode == "explicit_list" else [],
        "reuse_saved_targets": True,
    }
    payload["evaluation"] = {
        "topk": [10],
        "metrics": ["targeted_recall", "targeted_mrr"],
    }
    return payload


def _legacy_runtime_config(*, victims: tuple[str, ...]) -> dict[str, object]:
    base = _base_config()
    runtime_payload = base.runtime_config_dict()
    victims_runtime = runtime_payload.get("victims", {}).get("runtime", {})
    return {
        "victims": {
            "runtime": {victim: victims_runtime[victim] for victim in victims if victim in victims_runtime}
        }
    }


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_legacy_run(
    temp_root: Path,
    *,
    experiment_name: str,
    legacy_eval_key: str,
    run_type: str,
    targets_mode: str,
    target_items: list[int],
    victims: tuple[str, ...],
    missing_required_cells: set[tuple[int, str]] | None = None,
) -> Path:
    missing_required_cells = missing_required_cells or set()
    dataset_name = _base_config().data.dataset_name
    legacy_outputs_root = temp_root / "legacy_outputs"
    legacy_run_root = legacy_outputs_root / "runs" / dataset_name / experiment_name / legacy_eval_key
    target_selection_key = f"targets_{experiment_name}_{legacy_eval_key}"
    target_shared_root = legacy_outputs_root / "shared" / dataset_name / "targets" / target_selection_key
    target_shared_root.mkdir(parents=True, exist_ok=True)

    _write_text(target_shared_root / "config.yaml", "experiment: legacy\n")
    save_json({"target_items": list(target_items)}, target_shared_root / "selected_targets.json")
    save_json(
        {
            "target_selection_seed": 20260405,
            "bucket": "all",
            "count": len(target_items),
            "explicit_list": list(target_items) if targets_mode == "explicit_list" else [],
            "targets": {
                "mode": targets_mode,
                "bucket": "all",
                "count": len(target_items),
                "explicit_list": list(target_items) if targets_mode == "explicit_list" else [],
            },
        },
        target_shared_root / "target_selection_meta.json",
    )
    save_json(
        {
            "target_selection_mode": targets_mode,
            "seed": 20260405,
            "bucket": "all",
            "count": len(target_items),
            "explicit_list": list(target_items) if targets_mode == "explicit_list" else [],
            "target_items": list(target_items),
        },
        target_shared_root / "target_info.json",
    )

    result_config = _legacy_result_config(
        victims=victims,
        targets_mode=targets_mode,
        target_items=target_items,
        bucket="all",
    )
    runtime_config = _legacy_runtime_config(victims=victims)
    resolved_payload = {
        "result_config": result_config,
        "runtime_config": runtime_config,
        "derived": {
            "run_type": run_type,
            "split_key": "legacy_split_placeholder",
            "target_selection_key": target_selection_key,
            "attack_key": f"attack_{experiment_name}_{legacy_eval_key}",
            "evaluation_key": legacy_eval_key,
            "victim_prediction_keys": {
                victim: f"victim_{victim}_{experiment_name}_{legacy_eval_key}"
                for victim in victims
            },
        },
    }
    save_json(resolved_payload, legacy_run_root / "resolved_config.json")
    save_json(
        {
            "split_key_payload": {"legacy": True},
            "target_selection_key_payload": {"legacy": True},
            "attack_key_payload": {"legacy": True},
            "evaluation_key_payload": {"legacy": True},
            "victim_prediction_key_payloads": {victim: {"legacy": True} for victim in victims},
        },
        legacy_run_root / "key_payloads.json",
    )

    victims_manifest: dict[str, dict[str, object]] = {}
    summary_targets: dict[str, dict[str, object]] = {}
    generated_configs: dict[str, object] = {}
    for target_item in target_items:
        target_key = str(target_item)
        victims_manifest[target_key] = {}
        summary_targets[target_key] = {
            "target_item": int(target_item),
            "victims": {},
        }
        for victim_name in victims:
            run_dir = legacy_run_root / "targets" / target_key / "victims" / victim_name
            metrics_path = run_dir / "metrics.json"
            predictions_path = run_dir / "predictions.json"
            train_history_path = run_dir / "train_history.json"
            poisoned_train_path = run_dir / "poisoned_train.txt"
            resolved_config_path = run_dir / "resolved_config.json"
            config_snapshot_path = run_dir / "config.yaml"

            _write_text(config_snapshot_path, f"victim: {victim_name}\n")
            save_json({"victim_name": victim_name, "legacy": True}, resolved_config_path)
            save_json({"epochs": [1.0, 0.5]}, train_history_path)
            if victim_name == "srgnn":
                _write_text(poisoned_train_path, "1 2 3\n")

            save_json(
                {
                    "metrics": {
                        "targeted_recall@10": round(target_item / 1000.0, 6),
                        "targeted_mrr@10": round(target_item / 2000.0, 6),
                        "ground_truth_recall@10": 0.9,
                    },
                    "metrics_available": True,
                    "predictions_path": _repo_relative(predictions_path),
                },
                metrics_path,
            )
            if (target_item, victim_name) not in missing_required_cells:
                save_json({"rankings": [[target_item, 1, 2, 3]]}, predictions_path)

            victims_manifest[target_key][victim_name] = {
                "reused_predictions": True,
                "local": {
                    "run_dir": _legacy_string(run_dir),
                    "resolved_config": _legacy_string(resolved_config_path),
                    "config_snapshot": _legacy_string(config_snapshot_path),
                    "predictions": _legacy_string(predictions_path),
                    "metrics": _legacy_string(metrics_path),
                    "train_history": _legacy_string(train_history_path),
                    "poisoned_train": _legacy_string(poisoned_train_path),
                },
                "shared": {},
            }
            generated_configs[f"{target_key}:{victim_name}"] = {"legacy_extra": True}
            summary_targets[target_key]["victims"][victim_name] = {
                "metrics_path": _legacy_string(metrics_path),
                "predictions_path": _legacy_string(predictions_path),
                "metrics": {
                    "targeted_recall@10": round(target_item / 1000.0, 6),
                    "targeted_mrr@10": round(target_item / 2000.0, 6),
                    "ground_truth_recall@10": 0.9,
                },
                "metrics_available": True,
                "reused_predictions": True,
            }

    summary_payload = {
        "run_type": run_type,
        "target_items": list(target_items),
        "victims": list(victims),
        "fake_session_count": 0,
        "clean_session_count": 12,
        "training": {
            "poison_model": None if run_type == "clean" else {"name": "srgnn"},
            "victims": {
                victim: result_config["victims"]["params"][victim]["train"] for victim in victims
            },
        },
        "targets": summary_targets,
    }
    save_json(summary_payload, legacy_run_root / f"summary_{run_type}.json")

    artifact_manifest_payload = {
        "run_type": run_type,
        "canonical_split_artifact": {
            "metadata": _legacy_string(legacy_outputs_root / "shared" / dataset_name / "canonical" / "metadata.json")
        },
        "target_selection_artifact": {
            "shared_dir": _legacy_string(target_shared_root),
            "config_snapshot": _legacy_string(target_shared_root / "config.yaml"),
            "selected_targets": _legacy_string(target_shared_root / "selected_targets.json"),
            "target_selection_meta": _legacy_string(target_shared_root / "target_selection_meta.json"),
            "legacy_target_info": _legacy_string(target_shared_root / "target_info.json"),
        },
        "poison_artifact": None if run_type == "clean" else {"shared_dir": _legacy_string(legacy_outputs_root / "shared" / dataset_name / "attack" / "legacy")},
        "generated_configs": generated_configs,
        "victims": victims_manifest,
        "output_files": {
            "resolved_config": _legacy_string(legacy_run_root / "resolved_config.json"),
            "key_payloads": _legacy_string(legacy_run_root / "key_payloads.json"),
            "artifact_manifest": _legacy_string(legacy_run_root / "artifact_manifest.json"),
            "summary": _legacy_string(legacy_run_root / f"summary_{run_type}.json"),
        },
    }
    save_json(artifact_manifest_payload, legacy_run_root / "artifact_manifest.json")
    return legacy_run_root


def test_migrate_legacy_run_into_valid_new_run_group_and_analysis_bundle(monkeypatch) -> None:
    with _phase9_temp_root() as temp_root:
        legacy_run_root = _create_legacy_run(
            temp_root,
            experiment_name="phase9_clean",
            legacy_eval_key="eval_legacy_phase9",
            run_type="clean",
            targets_mode="sampled",
            target_items=[31, 11],
            victims=("srgnn", "miasrec"),
        )
        migrated_outputs_root = temp_root / "migrated_outputs"
        preview = inspect_legacy_run(legacy_run_root, artifacts_root_override=migrated_outputs_root)
        migrate_legacy_runs([legacy_run_root], artifacts_root_override=migrated_outputs_root)

        shared_paths = shared_artifact_paths(preview.config, run_type=preview.run_type)
        metadata_paths = run_metadata_paths(
            preview.config,
            run_type=preview.run_type,
            attack_identity_context=preview.attack_identity_context,
        )
        registry = load_target_registry(shared_paths["target_registry"])
        coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])
        summary_current = load_summary_current(metadata_paths["summary_current"])
        manifest = load_json(metadata_paths["artifact_manifest"])

        monkeypatch.setattr(
            "analysis.pipeline.long_csv_generator.RESULTS_ROOT",
            temp_root / "results",
        )
        analysis_result = generate_long_table_bundle(
            summary_path=metadata_paths["summary_current"],
            output_name="phase9_migrated_bundle",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )
        assert registry is not None
        assert registry["imported_from_legacy"] is True
        assert coverage is not None
        assert execution_log is not None
        assert summary_current is not None
        assert manifest is not None
        assert manifest["migration"]["imported_from_legacy"] is True
        assert set(coverage["victims"]) == {"srgnn", "miasrec"}
        for target_item in registry["ordered_targets"][: registry["current_count"]]:
            assert coverage["cells"][str(target_item)]["srgnn"]["status"] == "completed"
            assert coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
            migrated_artifacts = run_artifact_paths(
                preview.config,
                run_type="clean",
                target_id=target_item,
                victim_name="srgnn",
            )
            assert migrated_artifacts["metrics"].is_file()
            assert migrated_artifacts["predictions"].is_file()
        assert execution_log["executions"][0]["mode"] == "legacy_import"
        assert execution_log["executions"][0]["imported_from_legacy"] is True
        assert summary_current["imported_from_legacy"] is True
        assert analysis_result["row_count"] > 0
        assert Path(analysis_result["slice_manifest_path"]).is_file()


def test_target_registry_reconstruction_from_sampled_selected_targets_is_deterministic_and_honest() -> None:
    with _phase9_temp_root() as temp_root:
        legacy_run_root = _create_legacy_run(
            temp_root,
            experiment_name="phase9_sampled",
            legacy_eval_key="eval_legacy_sampled",
            run_type="clean",
            targets_mode="sampled",
            target_items=[31, 11, 21],
            victims=("srgnn",),
        )
        migrated_outputs_root = temp_root / "migrated_outputs"
        preview = inspect_legacy_run(legacy_run_root, artifacts_root_override=migrated_outputs_root)
        migrate_legacy_runs([legacy_run_root], artifacts_root_override=migrated_outputs_root)
        registry = load_target_registry(shared_artifact_paths(preview.config, run_type="clean")["target_registry"])

    assert registry is not None
    assert registry["ordered_targets"] == [11, 21, 31]
    assert registry["current_count"] == 3
    assert registry["migration"]["order_reconstructed"] is True
    assert registry["migration"]["reconstruction_mode"] == "legacy_selected_targets_only"


def test_target_registry_reconstruction_preserves_explicit_order() -> None:
    with _phase9_temp_root() as temp_root:
        legacy_run_root = _create_legacy_run(
            temp_root,
            experiment_name="phase9_explicit",
            legacy_eval_key="eval_legacy_explicit",
            run_type="clean",
            targets_mode="explicit_list",
            target_items=[42, 7, 19],
            victims=("srgnn",),
        )
        migrated_outputs_root = temp_root / "migrated_outputs"
        preview = inspect_legacy_run(legacy_run_root, artifacts_root_override=migrated_outputs_root)
        migrate_legacy_runs([legacy_run_root], artifacts_root_override=migrated_outputs_root)
        registry = load_target_registry(shared_artifact_paths(preview.config, run_type="clean")["target_registry"])

    assert registry is not None
    assert registry["ordered_targets"] == [42, 7, 19]
    assert registry["explicit_list"] == [42, 7, 19]
    assert registry["migration"]["order_reconstructed"] is False


def test_run_coverage_summary_current_and_execution_log_rebuild_conservatively_from_partial_cells() -> None:
    with _phase9_temp_root() as temp_root:
        legacy_run_root = _create_legacy_run(
            temp_root,
            experiment_name="phase9_partial",
            legacy_eval_key="eval_legacy_partial",
            run_type="clean",
            targets_mode="sampled",
            target_items=[31, 11],
            victims=("srgnn", "miasrec"),
            missing_required_cells={(31, "miasrec")},
        )
        migrated_outputs_root = temp_root / "migrated_outputs"
        preview = inspect_legacy_run(legacy_run_root, artifacts_root_override=migrated_outputs_root)
        migrate_legacy_runs([legacy_run_root], artifacts_root_override=migrated_outputs_root)
        metadata_paths = run_metadata_paths(preview.config, run_type="clean")
        coverage = load_run_coverage(metadata_paths["run_coverage"])
        execution_log = load_execution_log(metadata_paths["execution_log"])
        summary_current = load_summary_current(metadata_paths["summary_current"])

    assert coverage is not None
    assert execution_log is not None
    assert summary_current is not None
    assert coverage["cells"]["31"]["miasrec"]["status"] == "failed"
    assert coverage["cells"]["31"]["srgnn"]["status"] == "completed"
    assert "miasrec" not in summary_current["targets"]["31"]["victims"]
    assert len(execution_log["executions"][0]["failed_cells"]) == 1
    assert execution_log["executions"][0]["completed_cells"]


def test_multiple_legacy_runs_mapping_to_same_destination_are_rejected_clearly() -> None:
    with _phase9_temp_root() as temp_root:
        migrated_outputs_root = temp_root / "migrated_outputs"
        legacy_run_a = _create_legacy_run(
            temp_root,
            experiment_name="phase9_duplicate",
            legacy_eval_key="eval_legacy_dup_a",
            run_type="clean",
            targets_mode="sampled",
            target_items=[31, 11],
            victims=("srgnn",),
        )
        legacy_run_b = _create_legacy_run(
            temp_root,
            experiment_name="phase9_duplicate",
            legacy_eval_key="eval_legacy_dup_b",
            run_type="clean",
            targets_mode="sampled",
            target_items=[31, 11],
            victims=("srgnn",),
        )
        with pytest.raises(ValueError) as exc_info:
            migrate_legacy_runs(
                [legacy_run_a, legacy_run_b],
                artifacts_root_override=migrated_outputs_root,
            )

    assert "Multiple legacy inputs map to the same new run-group destination" in str(exc_info.value)
