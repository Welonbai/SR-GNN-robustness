from __future__ import annotations

import csv
import io
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import pytest

from analysis.utils.attack_improvement_export import (
    AttackImprovementExportError,
    build_attack_improvement_payload,
    main,
    render_markdown_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = [5334, 11103]
VICTIMS = ["tron", "srgnn"]


@contextmanager
def _temp_test_dir():
    path = REPO_ROOT / "analysis" / "tests" / "_tmp_attack_improvement_export" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _training_history_payload(target_item: int) -> dict[str, object]:
    return {
        "target_item": target_item,
        "policy_representation": "shared_contextual_mlp",
        "training_history": [
            {
                "outer_step": 0,
                "reward": 0.01,
                "baseline": None,
                "advantage": 0.01,
                "mean_entropy": 1.2,
                "policy_loss": 0.5,
                "target_utility": 0.001,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 1, 2, 2],
            },
            {
                "outer_step": 1,
                "reward": 0.02,
                "baseline": 0.01,
                "advantage": 0.01,
                "mean_entropy": 1.0,
                "policy_loss": 0.4,
                "target_utility": 0.002,
                "gt_drop": 0.0,
                "gt_penalty": 0.0,
                "selected_positions": [0, 0, 1, 2],
            },
        ],
        "final_selected_positions": [
            {"candidate_index": 0, "position": 0, "score": 1.0},
            {"candidate_index": 1, "position": 1, "score": 2.0},
        ],
    }


def _metrics_payload(targets: list[int], victims: list[str]) -> dict[str, object]:
    target_payloads: dict[str, object] = {}
    for target_item in targets:
        victim_payloads: dict[str, object] = {}
        for victim in victims:
            metrics = {
                "targeted_mrr@10": 0.101,
                "targeted_mrr@20": 0.102,
                "targeted_mrr@30": 0.103,
                "targeted_recall@10": 0.201,
                "targeted_recall@20": 0.202,
                "targeted_recall@30": 0.203,
                "ground_truth_mrr@10": 0.301,
                "ground_truth_mrr@20": 0.302,
                "ground_truth_mrr@30": 0.303,
                "ground_truth_recall@10": 0.401,
                "ground_truth_recall@20": 0.402,
                "ground_truth_recall@30": 0.403,
                "targeted_ndcg@10": 0.999,
            }
            victim_payloads[victim] = {"metrics": metrics}
        target_payloads[str(target_item)] = {
            "target_item": target_item,
            "victims": victim_payloads,
        }
    return {"targets": target_payloads}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_run_root(root: Path, *, include_training_history: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "summary_current.json", _metrics_payload(TARGETS, VICTIMS))
    if include_training_history:
        for target_item in TARGETS:
            history_path = root / "targets" / str(target_item) / "position_opt" / "training_history.json"
            _write_json(history_path, _training_history_payload(target_item))


def _read_csv_block(report: str, heading: str) -> list[dict[str, str]]:
    section_start = report.index(heading)
    block_start = report.index("```csv", section_start) + len("```csv")
    block_end = report.index("```", block_start)
    return list(csv.DictReader(io.StringIO(report[block_start:block_end].strip())))


def test_build_attack_improvement_payload_exports_dynamics_and_final_metrics() -> None:
    with _temp_test_dir() as temp_dir:
        shared = temp_dir / "shared"
        mvp = temp_dir / "mvp"
        prefix = temp_dir / "prefix"
        _write_run_root(shared)
        _write_run_root(mvp)
        _write_run_root(prefix, include_training_history=False)

        payload = build_attack_improvement_payload(
            shared_run_root=shared,
            mvp_run_root=mvp,
            prefix_run_root=prefix,
            target_items=TARGETS,
            victims=VICTIMS,
        )

        assert len(payload["shared_training_dynamics"]) == len(TARGETS) * 2
        assert len(payload["mvp_training_dynamics"]) == len(TARGETS) * 2
        first_row = payload["shared_training_dynamics"][0]
        assert first_row["method"] == "Shared"
        assert first_row["target_item"] == 5334
        assert first_row["outer_step"] == 0
        assert first_row["average_selected_position"] == pytest.approx(1.25)
        assert first_row["median_selected_position"] == pytest.approx(1.5)
        assert first_row["fraction_pos0"] == 0.25
        assert first_row["fraction_pos_le_1"] == 0.5
        assert first_row["fraction_pos_le_2"] == 1.0
        assert first_row["top_positions"].startswith("p2=2(50.0000%)")

        assert len(payload["final_metrics"]) == 3 * len(TARGETS) * len(VICTIMS)
        metric_row = payload["final_metrics"][0]
        assert metric_row["method"] == "Prefix"
        assert metric_row["victim"] == "tron"
        assert metric_row["targeted_mrr@10"] == 0.101
        assert "targeted_ndcg@10" not in metric_row


def test_render_markdown_report_contains_expected_csv_sections() -> None:
    with _temp_test_dir() as temp_dir:
        shared = temp_dir / "shared"
        mvp = temp_dir / "mvp"
        prefix = temp_dir / "prefix"
        _write_run_root(shared)
        _write_run_root(mvp)
        _write_run_root(prefix, include_training_history=False)

        payload = build_attack_improvement_payload(
            shared_run_root=shared,
            mvp_run_root=mvp,
            prefix_run_root=prefix,
            target_items=TARGETS,
            victims=VICTIMS,
        )
        report = render_markdown_report(payload)

        assert "## Shared Policy Per-Step Training Dynamics" in report
        assert "## MVP Per-Step Training Dynamics" in report
        assert "## Final Metrics Wide Table" in report
        shared_rows = _read_csv_block(report, "## Shared Policy Per-Step Training Dynamics")
        metric_rows = _read_csv_block(report, "## Final Metrics Wide Table")
        assert shared_rows[0]["fraction_pos_le_2"] == "1.0"
        assert metric_rows[0]["ground_truth_recall@30"] == "0.403"


def test_missing_training_history_raises_clear_error() -> None:
    with _temp_test_dir() as temp_dir:
        shared = temp_dir / "shared"
        mvp = temp_dir / "mvp"
        prefix = temp_dir / "prefix"
        _write_run_root(shared, include_training_history=False)
        _write_run_root(mvp)
        _write_run_root(prefix, include_training_history=False)

        with pytest.raises(AttackImprovementExportError, match="Missing training_history.json"):
            build_attack_improvement_payload(
                shared_run_root=shared,
                mvp_run_root=mvp,
                prefix_run_root=prefix,
                target_items=TARGETS,
                victims=VICTIMS,
            )


def test_missing_metric_raises_clear_error() -> None:
    with _temp_test_dir() as temp_dir:
        shared = temp_dir / "shared"
        mvp = temp_dir / "mvp"
        prefix = temp_dir / "prefix"
        _write_run_root(shared)
        _write_run_root(mvp)
        _write_run_root(prefix, include_training_history=False)
        summary = _metrics_payload(TARGETS, VICTIMS)
        del summary["targets"]["5334"]["victims"]["tron"]["metrics"]["targeted_mrr@10"]
        _write_json(prefix / "summary_current.json", summary)

        with pytest.raises(AttackImprovementExportError, match="targeted_mrr@10"):
            build_attack_improvement_payload(
                shared_run_root=shared,
                mvp_run_root=mvp,
                prefix_run_root=prefix,
                target_items=TARGETS,
                victims=VICTIMS,
            )


def test_cli_smoke_writes_markdown_report(capsys: pytest.CaptureFixture[str]) -> None:
    with _temp_test_dir() as temp_dir:
        shared = temp_dir / "shared"
        mvp = temp_dir / "mvp"
        prefix = temp_dir / "prefix"
        output_md = temp_dir / "report.md"
        _write_run_root(shared)
        _write_run_root(mvp)
        _write_run_root(prefix, include_training_history=False)

        exit_code = main(
            [
                "--shared-run-root",
                str(shared),
                "--mvp-run-root",
                str(mvp),
                "--prefix-run-root",
                str(prefix),
                "--target-items",
                "5334",
                "11103",
                "--victims",
                "tron",
                "srgnn",
                "--output-md",
                str(output_md),
            ]
        )

        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Wrote attack improvement report" in captured.out
        report = output_md.read_text(encoding="utf-8")
        assert "# Attack Improvement Analysis Export" in report
        assert "targeted_mrr@30" in report
