from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import ArtifactsConfig, load_config
from attack.pipeline.runs import run_creat_additive_sbr_generate_only as runner


def test_generate_only_skips_victim_orchestrator(monkeypatch, tmp_path: Path) -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_create_copy_source_popular_sample.yaml"
    )
    config = replace(
        config,
        artifacts=ArtifactsConfig(
            root=str(tmp_path),
            shared_dir="shared",
            runs_dir="runs",
            cleanup_victim_intermediates=False,
        ),
    )
    shared = SimpleNamespace(
        stats=SimpleNamespace(),
        shared_paths={"target_shared_dir": tmp_path / "target_shared"},
    )
    monkeypatch.setattr(runner, "prepare_shared_attack_artifacts", lambda *a, **k: shared)
    monkeypatch.setattr(runner, "prepare_creat_artifacts", lambda shared: object())
    monkeypatch.setattr(runner, "ensure_target_registry_prefix", lambda *a, **k: {"ordered_targets": [11103], "current_count": 1})
    monkeypatch.setattr(runner, "requested_target_prefix", lambda *a, **k: [11103])
    monkeypatch.setattr(
        runner,
        "generate_creat_target",
        lambda **kwargs: SimpleNamespace(metadata={"target_item": 11103}),
    )
    summary = runner.run_creat_additive_sbr_generate_only(config)
    assert summary["victim_execution_skipped"] is True
    assert summary["target_items"] == [11103]
