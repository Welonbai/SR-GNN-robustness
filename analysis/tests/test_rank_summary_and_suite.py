from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from analysis.pipeline import rank_summary_builder, report_suite_builder, view_table_builder


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def _patch_results_root(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(rank_summary_builder, "RESULTS_ROOT", root)
    monkeypatch.setattr(report_suite_builder, "RESULTS_ROOT", root)
    monkeypatch.setattr(view_table_builder, "RESULTS_ROOT", root)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _comparison_rows() -> list[dict[str, object]]:
    methods = ["clean", "random_nz", "generated_direct_cem", "copy_direct_cem", "creat", "poisoning_ssl_sbr"]
    values_by_victim = {
        "srgnn": {
            "clean": 0.10,
            "random_nz": 0.20,
            "generated_direct_cem": 0.60,
            "copy_direct_cem": 0.50,
            "creat": 0.40,
            "poisoning_ssl_sbr": 0.30,
        },
        "tron": {
            "clean": 0.10,
            "random_nz": 0.20,
            "generated_direct_cem": 0.60,
            "copy_direct_cem": 0.70,
            "creat": 0.50,
            "poisoning_ssl_sbr": 0.30,
        },
    }
    rows: list[dict[str, object]] = []
    for victim_model, values in values_by_victim.items():
        for method in methods:
            rows.append(
                {
                    "run_id": f"{method}_{victim_model}",
                    "dataset": "diginetica",
                    "attack_method": method,
                    "victim_model": victim_model,
                    "target_item": 111,
                    "target_type": "popular",
                    "attack_size": 0.01,
                    "poison_model": "srgnn",
                    "fake_session_generation_topk": 100,
                    "replacement_topk_ratio": 1.0,
                    "metric": "recall",
                    "k": 20,
                    "value": values[method],
                }
            )
    return rows


def _write_comparison_bundle(root: Path) -> Path:
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    csv_path = comparison_dir / "merged_long_table.csv"
    pd.DataFrame(_comparison_rows()).to_csv(csv_path, index=False)
    _write_json(
        comparison_dir / "manifest.json",
        {"comparison_id": "test", "source_run_ids": [], "source_csvs": []},
    )
    _write_json(
        comparison_dir / "slice_manifest.json",
        {
            "slice_policy": "largest_complete_prefix",
            "fairness_safe": True,
            "selected_target_count": 1,
            "selected_targets": [111],
            "requested_victims": ["srgnn", "tron"],
        },
    )
    return csv_path


def _rank_summary_config(input_csv: Path, output_dir: Path) -> dict[str, object]:
    return {
        "analysis_type": "method_rank_counts",
        "input": str(input_csv),
        "output": str(output_dir),
        "filters": {
            "attack_method": [
                "clean",
                "random_nz",
                "generated_direct_cem",
                "copy_direct_cem",
                "creat",
                "poisoning_ssl_sbr",
            ],
            "metric_name": "recall",
            "metric_scope": "targeted",
            "k": 20,
        },
        "method_rank_counts": {
            "higher_is_better": True,
            "tie_tolerance": 0.0,
            "rank_unit_columns": [
                "dataset",
                "target_type",
                "victim_model",
                "metric_name",
                "metric_scope",
                "k",
            ],
            "row_columns": ["metric_name", "k"],
            "counted_methods": ["generated_direct_cem", "copy_direct_cem"],
            "method_families": [
                {
                    "name": "cem_family",
                    "methods": ["generated_direct_cem", "copy_direct_cem"],
                }
            ],
            "thresholds": [
                {"rank": 1, "label": "best"},
                {"rank": 2, "label": "top2"},
            ],
        },
    }


def test_rank_summary_counts_individual_methods_and_family_hits(tmp_path, monkeypatch) -> None:
    root = tmp_path / "results"
    root.mkdir()
    _patch_results_root(monkeypatch, root)
    input_csv = _write_comparison_bundle(root)
    config_path = root / "rank_summary.yaml"
    output_dir = root / "best_top2"
    _write_yaml(config_path, _rank_summary_config(input_csv, output_dir))

    spec = rank_summary_builder.parse_rank_summary_spec(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        source_spec_path=config_path,
    )
    rank_summary_builder.build_rank_summary_bundle(spec)

    table = pd.read_csv(output_dir / "table.csv")
    assert table.to_dict(orient="records") == [
        {
            "metric_name": "recall",
            "k": 20,
            "generated_direct_cem | best": 1,
            "generated_direct_cem | top2": 2,
            "copy_direct_cem | best": 1,
            "copy_direct_cem | top2": 2,
            "cem_family | best": 2,
            "cem_family | top2": 2,
        }
    ]


def test_report_suite_runs_view_and_rank_summary_steps(tmp_path, monkeypatch) -> None:
    root = tmp_path / "results"
    root.mkdir()
    _patch_results_root(monkeypatch, root)
    input_csv = _write_comparison_bundle(root)
    view_config_path = root / "view.yaml"
    rank_config_path = root / "rank.yaml"
    suite_config_path = root / "suite.yaml"
    view_output_dir = root / "main"
    rank_output_dir = root / "best_top2"

    _write_yaml(
        view_config_path,
        {
            "name": "main",
            "input": str(input_csv),
            "output": str(view_output_dir),
            "filters": {"metric_name": "recall", "metric_scope": "targeted", "k": 20},
            "split_by": [],
            "rows": ["attack_method"],
            "cols": ["victim_model"],
            "value_col": "value",
            "agg": "mean",
            "auto_context": True,
            "require_unique_cells": False,
        },
    )
    _write_yaml(rank_config_path, _rank_summary_config(input_csv, rank_output_dir))
    _write_yaml(
        suite_config_path,
        {
            "suite_name": "test_suite",
            "steps": [
                {"name": "build_main", "type": "view", "config": str(view_config_path)},
                {"name": "build_rank", "type": "rank_summary", "config": str(rank_config_path)},
            ],
        },
    )

    suite_spec = report_suite_builder.parse_report_suite_spec(
        yaml.safe_load(suite_config_path.read_text(encoding="utf-8")),
        source_spec_path=suite_config_path,
    )
    results = report_suite_builder.run_report_suite(suite_spec)

    assert [result["name"] for result in results] == ["build_main", "build_rank"]
    assert (view_output_dir / "table.csv").is_file()
    assert (rank_output_dir / "table.csv").is_file()
