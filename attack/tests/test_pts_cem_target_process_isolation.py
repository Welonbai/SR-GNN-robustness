from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from attack.common.config import load_config
from attack.pipeline.core.orchestrator import _limit_planned_cells_by_target
from attack.pipeline.runs import run_pts_construction_cem


REPO_ROOT = Path(__file__).resolve().parents[2]
MDHG_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_generated_popular_all_victims.yaml"
)


def test_target_limit_keeps_every_cell_for_first_incomplete_target() -> None:
    cells = [
        {"target_item": 11, "victim_name": "freqrec"},
        {"target_item": 11, "victim_name": "srgnn"},
        {"target_item": 22, "victim_name": "freqrec"},
        {"target_item": 22, "victim_name": "srgnn"},
    ]

    limited = _limit_planned_cells_by_target(cells, max_targets=1)

    assert limited == cells[:2]
    assert _limit_planned_cells_by_target(cells, max_targets=None) == cells
    with pytest.raises(ValueError, match="must be positive"):
        _limit_planned_cells_by_target(cells, max_targets=0)


def test_mdhg_direct_cem_requires_target_process_isolation() -> None:
    config = load_config(MDHG_CONFIG_PATH)

    assert run_pts_construction_cem._requires_target_process_isolation(config) is True


def test_formal_entrypoint_dispatches_mdhg_run_to_supervisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_supervisor(config, **kwargs) -> None:
        captured["config"] = config
        captured.update(kwargs)

    monkeypatch.setattr(
        run_pts_construction_cem,
        "_run_target_isolation_supervisor",
        fake_supervisor,
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "run_pts_construction_grouped_cem",
        lambda *args, **kwargs: pytest.fail("MDHG parent must not execute the worker body"),
    )

    run_pts_construction_cem.main(["--config", str(MDHG_CONFIG_PATH)])

    assert captured["config_path"] == MDHG_CONFIG_PATH
    assert captured["force_recompute_pts_cem"] is False


def test_isolated_worker_limits_execution_to_one_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        run_pts_construction_cem,
        "run_pts_construction_grouped_cem",
        fake_run,
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_run_target_isolation_supervisor",
        lambda *args, **kwargs: pytest.fail("worker must not recursively supervise"),
    )

    run_pts_construction_cem.main(
        ["--config", str(MDHG_CONFIG_PATH), "--target-isolation-worker"]
    )

    assert captured["max_targets_per_execution"] == 1
    assert captured["force_recompute_pts_cem"] is False


def test_supervisor_restarts_abnormal_worker_after_committed_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(MDHG_CONFIG_PATH)
    snapshots = iter([(34, 60), (40, 60), (60, 60)])
    returncodes = iter([-11, 0])
    commands: list[list[str]] = []

    monkeypatch.setattr(
        run_pts_construction_cem,
        "_coverage_completion_snapshot",
        lambda _config: next(snapshots),
    )

    def fake_run(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, next(returncodes))

    monkeypatch.setattr(run_pts_construction_cem.subprocess, "run", fake_run)

    run_pts_construction_cem._run_target_isolation_supervisor(
        config,
        config_path=MDHG_CONFIG_PATH,
        force_recompute_pts_cem=False,
    )

    assert len(commands) == 2
    assert all("--target-isolation-worker" in cmd for cmd in commands)
    assert all("--force-recompute-pts-cem" not in cmd for cmd in commands)


def test_supervisor_stops_after_two_failures_without_coverage_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(MDHG_CONFIG_PATH)
    snapshots = iter([(40, 60), (40, 60), (40, 60)])
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_coverage_completion_snapshot",
        lambda _config: next(snapshots),
    )
    monkeypatch.setattr(
        run_pts_construction_cem.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, -11),
    )

    with pytest.raises(RuntimeError, match="failed twice without committing"):
        run_pts_construction_cem._run_target_isolation_supervisor(
            config,
            config_path=MDHG_CONFIG_PATH,
            force_recompute_pts_cem=False,
        )
