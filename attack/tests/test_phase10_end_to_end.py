from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

import analysis.pipeline.compare_runs as compare_runs_module
import analysis.pipeline.long_csv_generator as long_csv_generator_module
import analysis.pipeline.report_table_renderer as report_table_renderer_module
import analysis.pipeline.view_table_builder as view_table_builder_module
from attack.common.artifact_io import (
    load_execution_log,
    load_json,
    load_run_coverage,
    load_summary_current,
    save_json,
)
from attack.common.config import load_config
from attack.common.paths import run_metadata_paths, shared_artifact_paths
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.orchestrator import RunContext, TargetPoisonOutput, run_targets_and_victims
from attack.pipeline.core.pipeline_utils import build_ordered_target_cohort
from attack.pipeline.core.victim_execution import VictimExecutionResult
from attack.tools.migrate_legacy_runs import inspect_legacy_run, migrate_legacy_runs


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 10 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase10_temp_roots():
    token = uuid4().hex
    outputs_root = REPO_ROOT / "outputs" / ".pytest_phase10" / token
    results_root = REPO_ROOT / "results" / ".pytest_phase10" / token
    outputs_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    try:
        yield outputs_root, results_root
    finally:
        shutil.rmtree(outputs_root, ignore_errors=True)
        shutil.rmtree(results_root, ignore_errors=True)


def _patch_analysis_results_roots(monkeypatch, *, results_root: Path) -> None:
    monkeypatch.setattr(long_csv_generator_module, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(compare_runs_module, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(compare_runs_module, "RUNS_ROOT", results_root / "runs")
    monkeypatch.setattr(compare_runs_module, "COMPARISONS_ROOT", results_root / "comparisons")
    monkeypatch.setattr(view_table_builder_module, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(report_table_renderer_module, "RESULTS_ROOT", results_root)


def _sample_stats():
    return compute_session_stats(
        [
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
            [40, 50, 60],
            [50, 60, 70],
        ]
    )


def _config_for_temp_root(
    temp_root: Path,
    *,
    experiment_name: str,
    count: int,
    victims: tuple[str, ...],
    targets_mode: str = "sampled",
    explicit_list: tuple[int, ...] = (),
):
    base = _base_config()
    targets = replace(
        base.targets,
        mode=targets_mode,
        bucket="all",
        count=count,
        explicit_list=explicit_list,
    )
    return replace(
        base,
        experiment=replace(base.experiment, name=experiment_name),
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=targets,
        victims=replace(base.victims, enabled=victims),
    )


def _minimal_context(config, *, run_type: str) -> RunContext:
    return RunContext(
        canonical_dataset=object(),
        stats=_sample_stats(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        export_paths={},
        shared_paths=shared_artifact_paths(config, run_type=run_type),
        fake_session_count=0,
    )


def _with_enabled_victims(config, victims: tuple[str, ...]):
    base = _base_config()
    params = {
        victim_name: dict(base.victims.params[victim_name])
        for victim_name in victims
    }
    runtime = None
    if base.victims.runtime is not None:
        runtime = {
            victim_name: dict(base.victims.runtime[victim_name])
            for victim_name in victims
            if victim_name in base.victims.runtime
        }
    return replace(
        config,
        victims=replace(config.victims, enabled=victims, params=params, runtime=runtime),
    )


def _expected_target_prefix(config) -> list[int]:
    cohort = build_ordered_target_cohort(_sample_stats(), config)
    return [int(item) for item in cohort["ordered_targets"][: int(config.targets.count)]]


def _install_fake_execution(
    monkeypatch,
    *,
    calls: list[tuple[int, str]],
    exception_by_call_index: dict[int, BaseException] | None = None,
    exception_by_cell: dict[tuple[int, str], BaseException] | None = None,
    omit_prediction_cells: set[tuple[int, str]] | None = None,
) -> None:
    exception_by_call_index = exception_by_call_index or {}
    exception_by_cell = exception_by_cell or {}
    omit_prediction_cells = omit_prediction_cells or set()

    def fake_victim_execution(*args, **kwargs):
        target_item = int(kwargs["target_item"])
        victim_name = str(kwargs["victim_name"])
        calls.append((target_item, victim_name))
        save_json(
            {"victim_name": victim_name, "phase10_fake": True},
            kwargs["artifacts"]["resolved_config"],
        )

        call_index = len(calls)
        raised = exception_by_call_index.get(call_index)
        if raised is None:
            raised = exception_by_cell.get((target_item, victim_name))
        if raised is not None:
            raise raised

        if (target_item, victim_name) not in omit_prediction_cells:
            save_json(
                {"rankings": [[target_item, 1, 2, 3]], "victim": victim_name},
                kwargs["predictions_path"],
            )
        return (
            VictimExecutionResult(
                predictions=[[target_item, 1, 2, 3]],
                predictions_path=kwargs["predictions_path"],
                extra={"phase10_fake": True},
                poisoned_train_path=None,
            ),
            False,
        )

    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator._maybe_reuse_or_execute_victim",
        fake_victim_execution,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.resolve_ground_truth_labels",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.evaluate_prediction_metrics",
        lambda *args, **kwargs: ({"targeted_recall@10": 1.0}, True),
    )


def _build_poisoned(target_item: int) -> TargetPoisonOutput:
    return TargetPoisonOutput(
        poisoned=PoisonedDataset(
            sessions=[[1, 2]],
            labels=[int(target_item)],
            clean_count=1,
            fake_count=0,
        ),
        metadata={"phase10_smoke": True},
    )


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            "runtime": {
                victim: victims_runtime[victim]
                for victim in victims
                if victim in victims_runtime
            }
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
) -> Path:
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
            "target_selection_seed": 20260419,
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
            "seed": 20260419,
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
            "metadata": _legacy_string(
                legacy_outputs_root / "shared" / dataset_name / "canonical" / "metadata.json"
            )
        },
        "target_selection_artifact": {
            "shared_dir": _legacy_string(target_shared_root),
            "config_snapshot": _legacy_string(target_shared_root / "config.yaml"),
            "selected_targets": _legacy_string(target_shared_root / "selected_targets.json"),
            "target_selection_meta": _legacy_string(
                target_shared_root / "target_selection_meta.json"
            ),
            "legacy_target_info": _legacy_string(target_shared_root / "target_info.json"),
        },
        "poison_artifact": None,
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


def _comparison_spec(*, results_root: Path, comparison_id: str, run_ids: list[str]) -> compare_runs_module.ComparisonSpec:
    return compare_runs_module.ComparisonSpec(
        comparison_id=comparison_id,
        run_ids=run_ids,
        output_dir=results_root / "comparisons" / comparison_id,
        slice_compatibility="strict",
    )


def _render_spec():
    return report_table_renderer_module.parse_render_spec(
        {
            "style_name": "phase10_end_to_end",
            "output_format": "png",
            "title": {
                "template": "Slice {slice_policy} N={selected_target_count} Victims {requested_victims}",
                "align": "left",
                "font_size": 14,
                "color": "black",
            },
            "figure": {
                "width": 8,
                "height": 4,
                "dpi": 80,
                "background_color": "white",
            },
            "table": {
                "font_size": 10,
                "round_digits": 4,
                "text_color": "black",
                "show_grid": True,
                "auto_shrink": False,
                "wrap_text": False,
                "cell_align": "center",
                "display_alias": {},
                "value_alias": {},
                "scope_colors": {},
                "top_level_group_separators": False,
            },
        }
    )


def test_native_appendable_workflow_end_to_end(monkeypatch) -> None:
    with _phase10_temp_roots() as (outputs_root, results_root):
        _patch_analysis_results_roots(monkeypatch, results_root=results_root)

        initial_config = _config_for_temp_root(
            outputs_root / "native_outputs",
            experiment_name="phase10_native_a",
            count=2,
            victims=("miasrec",),
        )
        combined_config = _config_for_temp_root(
            outputs_root / "native_outputs",
            experiment_name="phase10_native_a",
            count=4,
            victims=("miasrec", "tron"),
        )
        peer_config = _config_for_temp_root(
            outputs_root / "native_outputs",
            experiment_name="phase10_native_b",
            count=4,
            victims=("miasrec", "tron"),
        )
        context_a = _minimal_context(initial_config, run_type="clean")
        context_b = _minimal_context(peer_config, run_type="clean")
        target_prefix = _expected_target_prefix(combined_config)
        old_targets = [int(item) for item in target_prefix[:2]]
        new_targets = [int(item) for item in target_prefix[2:]]

        initial_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=initial_calls)
        run_targets_and_victims(
            initial_config,
            config_path=None,
            context=context_a,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        interrupted_calls: list[tuple[int, str]] = []
        _install_fake_execution(
            monkeypatch,
            calls=interrupted_calls,
            exception_by_cell={(new_targets[0], "tron"): KeyboardInterrupt("phase10 interrupt")},
        )
        with pytest.raises(KeyboardInterrupt):
            run_targets_and_victims(
                combined_config,
                config_path=None,
                context=context_a,
                run_type="clean",
                build_poisoned=_build_poisoned,
            )

        metadata_paths = run_metadata_paths(combined_config, run_type="clean")
        interrupted_coverage = load_run_coverage(metadata_paths["run_coverage"])
        interrupted_execution_log = load_execution_log(metadata_paths["execution_log"])

        retry_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=retry_calls)
        run_targets_and_victims(
            combined_config,
            config_path=None,
            context=context_a,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        save_json({"targets": {}, "snapshot": "corrupted"}, metadata_paths["summary_current"])
        save_json(
            {
                "is_authoritative": False,
                "status": "completed",
                "runs": [],
                "note": "phase10_corrupted_debug_snapshot",
            },
            metadata_paths["progress"],
        )

        noop_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=noop_calls)
        run_targets_and_victims(
            combined_config,
            config_path=None,
            context=context_a,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        peer_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=peer_calls)
        run_targets_and_victims(
            peer_config,
            config_path=None,
            context=context_b,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        final_coverage = load_run_coverage(metadata_paths["run_coverage"])
        final_execution_log = load_execution_log(metadata_paths["execution_log"])
        final_summary_current = load_summary_current(metadata_paths["summary_current"])
        final_progress = load_json(metadata_paths["progress"])

        native_bundle_a = long_csv_generator_module.generate_long_table_bundle(
            summary_path=metadata_paths["summary_current"],
            output_name="phase10_native_a_bundle",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )
        peer_metadata_paths = run_metadata_paths(peer_config, run_type="clean")
        native_bundle_b = long_csv_generator_module.generate_long_table_bundle(
            summary_path=peer_metadata_paths["summary_current"],
            output_name="phase10_native_b_bundle",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )

        comparison_result = compare_runs_module.build_comparison_bundle(
            _comparison_spec(
                results_root=results_root,
                comparison_id="phase10_native_comparison",
                run_ids=["phase10_native_a_bundle", "phase10_native_b_bundle"],
            )
        )
        comparison_manifest = _load_json(comparison_result["manifest_path"])

        view_spec = view_table_builder_module.parse_view_spec(
            {
                "input": str(comparison_result["merged_csv_path"]),
                "output": str(results_root / "views" / "phase10_native_view"),
                "name": "phase10_native_view",
                "filters": {"metric": "recall", "k": 10},
                "rows": ["victim_model"],
                "cols": ["run_id"],
                "value_col": "value",
                "agg": "mean",
                "auto_context": True,
                "require_unique_cells": False,
            },
            source_spec_path=REPO_ROOT / "docs" / "phase10_native_view.yaml",
        )
        bundle_dirs = view_table_builder_module.build_view_bundles(view_spec)
        render_path = report_table_renderer_module.render_bundle(
            bundle_dir=bundle_dirs[0],
            render_spec=_render_spec(),
        )
        view_meta = _load_json(bundle_dirs[0] / "meta.json")
        render_exists = render_path.is_file()
        render_size = render_path.stat().st_size if render_exists else 0

    assert initial_calls == [(old_targets[0], "miasrec"), (old_targets[1], "miasrec")]
    assert interrupted_calls == [
        (old_targets[0], "tron"),
        (old_targets[1], "tron"),
        (new_targets[0], "miasrec"),
        (new_targets[0], "tron"),
    ]
    assert interrupted_coverage is not None
    assert interrupted_coverage["cells"][str(old_targets[0])]["tron"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(old_targets[1])]["tron"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(new_targets[0])]["miasrec"]["status"] == "completed"
    assert interrupted_coverage["cells"][str(new_targets[0])]["tron"]["status"] == "failed"
    assert interrupted_coverage["cells"][str(new_targets[1])]["miasrec"]["status"] == "requested"
    assert interrupted_coverage["cells"][str(new_targets[1])]["tron"]["status"] == "requested"
    assert interrupted_execution_log is not None
    assert interrupted_execution_log["executions"][1]["status"] == "failed"

    assert retry_calls == [
        (new_targets[0], "tron"),
        (new_targets[1], "miasrec"),
        (new_targets[1], "tron"),
    ]
    assert noop_calls == []
    assert len(peer_calls) == 8

    assert final_coverage is not None
    for target_item in target_prefix:
        assert final_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
        assert final_coverage["cells"][str(target_item)]["tron"]["status"] == "completed"

    assert final_execution_log is not None
    assert [record["mode"] for record in final_execution_log["executions"]] == [
        "initial_population",
        "target_and_victim_append",
        "retry_incomplete_cells",
        "noop",
    ]
    assert final_execution_log["executions"][-1]["planned_cells"] == []
    assert final_execution_log["executions"][-1]["skipped_completed_cells"]

    assert final_summary_current is not None
    assert set(final_summary_current["targets"]) == {str(item) for item in target_prefix}
    for target_item in target_prefix:
        assert set(final_summary_current["targets"][str(target_item)]["victims"]) == {
            "miasrec",
            "tron",
        }

    assert final_progress["is_authoritative"] is False
    assert final_progress["authoritative_state"]["run_coverage"].endswith("run_coverage.json")
    assert final_progress["authoritative_state"]["execution_log"].endswith("execution_log.json")

    assert native_bundle_a["row_count"] > 0
    assert native_bundle_b["row_count"] > 0
    assert comparison_manifest["slice_compatibility"]["mode"] == "strict"
    assert comparison_manifest["slice_compatibility"]["compatible"] is True
    assert comparison_manifest["slice"]["selected_target_count"] == 4

    assert len(bundle_dirs) == 1
    assert view_meta["slice"]["slice_policy"] == "largest_complete_prefix"
    assert view_meta["slice_context"]["selected_target_count"] == 4
    assert view_meta["context"]["slice_policy"] == "largest_complete_prefix"
    assert view_meta["context"]["requested_victims"] == ["miasrec", "tron"]
    assert render_exists is True
    assert render_size > 0


def test_end_to_end_strict_comparison_rejects_incompatible_slice_metadata(monkeypatch) -> None:
    with _phase10_temp_roots() as (outputs_root, results_root):
        _patch_analysis_results_roots(monkeypatch, results_root=results_root)

        config = _config_for_temp_root(
            outputs_root / "native_outputs",
            experiment_name="phase10_incompatible",
            count=4,
            victims=("miasrec", "tron"),
        )
        context = _minimal_context(config, run_type="clean")
        calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=calls)
        run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        metadata_paths = run_metadata_paths(config, run_type="clean")
        long_csv_generator_module.generate_long_table_bundle(
            summary_path=metadata_paths["summary_current"],
            output_name="phase10_incompatible_full",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )
        long_csv_generator_module.generate_long_table_bundle(
            summary_path=metadata_paths["summary_current"],
            output_name="phase10_incompatible_prefix2",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=2,
        )

        with pytest.raises(ValueError) as exc_info:
            compare_runs_module.build_comparison_bundle(
                _comparison_spec(
                    results_root=results_root,
                    comparison_id="phase10_incompatible_comparison",
                    run_ids=["phase10_incompatible_full", "phase10_incompatible_prefix2"],
                )
            )

    assert "Strict slice compatibility failed" in str(exc_info.value)


def test_migrated_run_plus_native_append_behaves_like_first_class_run_group(monkeypatch) -> None:
    with _phase10_temp_roots() as (outputs_root, results_root):
        _patch_analysis_results_roots(monkeypatch, results_root=results_root)

        legacy_run_root = _create_legacy_run(
            outputs_root,
            experiment_name="phase10_legacy_import",
            legacy_eval_key="eval_legacy_phase10",
            run_type="clean",
            targets_mode="explicit_list",
            target_items=[42, 7],
            victims=("miasrec",),
        )
        migrated_outputs_root = outputs_root / "migrated_outputs"
        preview = inspect_legacy_run(legacy_run_root, artifacts_root_override=migrated_outputs_root)
        migrate_legacy_runs([legacy_run_root], artifacts_root_override=migrated_outputs_root)

        append_config = _with_enabled_victims(preview.config, ("miasrec", "tron"))
        append_context = _minimal_context(append_config, run_type=preview.run_type)
        append_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=append_calls)
        run_targets_and_victims(
            append_config,
            config_path=None,
            context=append_context,
            run_type=preview.run_type,
            build_poisoned=_build_poisoned,
        )

        migrated_metadata_paths = run_metadata_paths(append_config, run_type=preview.run_type)
        migrated_coverage = load_run_coverage(migrated_metadata_paths["run_coverage"])
        migrated_execution_log = load_execution_log(migrated_metadata_paths["execution_log"])
        migrated_summary_current = load_summary_current(migrated_metadata_paths["summary_current"])

        migrated_bundle = long_csv_generator_module.generate_long_table_bundle(
            summary_path=migrated_metadata_paths["summary_current"],
            output_name="phase10_migrated_bundle",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )

        native_peer_config = _config_for_temp_root(
            migrated_outputs_root,
            experiment_name="phase10_native_peer",
            count=2,
            victims=("miasrec", "tron"),
            targets_mode="explicit_list",
            explicit_list=(42, 7),
        )
        native_peer_context = _minimal_context(native_peer_config, run_type="clean")
        native_peer_calls: list[tuple[int, str]] = []
        _install_fake_execution(monkeypatch, calls=native_peer_calls)
        run_targets_and_victims(
            native_peer_config,
            config_path=None,
            context=native_peer_context,
            run_type="clean",
            build_poisoned=_build_poisoned,
        )

        native_peer_metadata_paths = run_metadata_paths(native_peer_config, run_type="clean")
        native_peer_bundle = long_csv_generator_module.generate_long_table_bundle(
            summary_path=native_peer_metadata_paths["summary_current"],
            output_name="phase10_native_peer_bundle",
            slice_policy=None,
            requested_victims=None,
            requested_target_count=None,
        )
        comparison_result = compare_runs_module.build_comparison_bundle(
            _comparison_spec(
                results_root=results_root,
                comparison_id="phase10_migrated_native_comparison",
                run_ids=["phase10_migrated_bundle", "phase10_native_peer_bundle"],
            )
        )
        comparison_manifest = _load_json(comparison_result["manifest_path"])

    assert append_calls == [(42, "tron"), (7, "tron")]
    assert migrated_coverage is not None
    assert migrated_execution_log is not None
    assert migrated_summary_current is not None
    for target_item in (42, 7):
        assert migrated_coverage["cells"][str(target_item)]["miasrec"]["status"] == "completed"
        assert migrated_coverage["cells"][str(target_item)]["tron"]["status"] == "completed"
        assert set(migrated_summary_current["targets"][str(target_item)]["victims"]) == {
            "miasrec",
            "tron",
        }
    assert migrated_execution_log["executions"][0]["mode"] == "legacy_import"
    assert migrated_execution_log["executions"][1]["mode"] == "victim_append"
    assert migrated_bundle["row_count"] > 0
    assert native_peer_bundle["row_count"] > 0
    assert len(native_peer_calls) == 4
    assert comparison_manifest["slice_compatibility"]["compatible"] is True
    assert comparison_manifest["slice"]["selected_targets"] == [42, 7]
