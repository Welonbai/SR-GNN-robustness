from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_fake_sessions, load_json
from attack.common.config import (
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    FakeSessionSourceConfig,
    TrainTemplateSourceConfig,
    load_config,
)
from attack.common.paths import shared_artifact_paths
from attack.creat.candidates import sessions_sha1
from attack.data.canonical_dataset import CanonicalDataset
from attack.pipeline.core import pipeline_utils


CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"


def _train_template_config(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(tmp_path / "outputs")),
        attack=replace(
            base.attack,
            size=1.0,
            fake_session_source=FakeSessionSourceConfig(
                type=FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
                train_template=TrainTemplateSourceConfig(),
            ),
        ),
        targets=replace(base.targets, reuse_saved_targets=False, count=1),
        victims=replace(base.victims, enabled=("srgnn",)),
    )


def _train_template_config_without_distribution_csv(tmp_path: Path):
    config = _train_template_config(tmp_path)
    return replace(
        config,
        attack=replace(
            config.attack,
            fake_session_source=FakeSessionSourceConfig(
                type=FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
                train_template=TrainTemplateSourceConfig(
                    record_distribution_diagnostics=False
                ),
            ),
        ),
    )


def _toy_dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train_sub=[[1, 2], [3, 4, 5], [6, 7]],
        valid=[[1, 2]],
        test=[[3, 4]],
        item_map={},
        metadata={"fixture": True},
    )


def _patch_lightweight_shared_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_utils,
        "ensure_canonical_dataset",
        lambda config: _toy_dataset(),
    )
    monkeypatch.setattr(
        pipeline_utils,
        "_export_srg_nn_dataset",
        lambda *, dataset, export_dir: {
            "train": export_dir / "train.txt",
            "test": export_dir / "test.txt",
        },
    )


def test_train_template_shared_artifacts_write_fake_sessions_without_poison_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _train_template_config(tmp_path)
    _patch_lightweight_shared_setup(monkeypatch)

    def fail_poison_runner(*args, **kwargs):
        raise AssertionError("poison runner should not be prepared")

    monkeypatch.setattr(pipeline_utils, "_load_or_train_poison_runner", fail_poison_runner)

    artifacts = pipeline_utils.prepare_shared_attack_artifacts(
        config,
        run_type="attack",
        require_poison_runner=False,
    )

    assert artifacts.poison_runner is None
    assert artifacts.fake_session_count == len(artifacts.template_sessions)
    assert all(isinstance(session, list) for session in artifacts.template_sessions)
    assert all(all(isinstance(item, int) for item in session) for session in artifacts.template_sessions)

    shared_paths = shared_artifact_paths(config, run_type="attack")
    saved_sessions = load_fake_sessions(shared_paths["fake_sessions"])
    assert saved_sessions == artifacts.template_sessions

    summary = load_json(shared_paths["attack_shared_dir"] / "fake_session_source_summary.json")
    assert summary["fake_session_source"]["type"] == (
        FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    )
    assert summary["denominator_representation"] == "expanded_prefix_label_pairs"
    assert summary["poison_runner_prepared"] is False
    assert summary["poison_runner_reason"] == "not_required_by_downstream"
    assert summary["shared_identity_includes_poison_model"] is False
    assert summary["template_sessions_sha1"] == sessions_sha1(artifacts.template_sessions)
    assert summary["base_fake_session_source"] == (
        FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    )
    assert (
        shared_paths["attack_shared_dir"] / "fake_session_source_length_distribution.csv"
    ).exists()


def test_train_template_shared_artifacts_prepare_poison_runner_when_required(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _train_template_config(tmp_path)
    _patch_lightweight_shared_setup(monkeypatch)
    sentinel_runner = object()
    calls: list[str] = []

    def fake_poison_runner(*args, **kwargs):
        calls.append("called")
        return sentinel_runner

    monkeypatch.setattr(pipeline_utils, "_load_or_train_poison_runner", fake_poison_runner)

    artifacts = pipeline_utils.prepare_shared_attack_artifacts(
        config,
        run_type="attack",
        require_poison_runner=True,
    )

    assert artifacts.poison_runner is sentinel_runner
    assert calls == ["called"]
    shared_paths = shared_artifact_paths(
        config,
        run_type="attack",
        require_poison_runner=True,
    )
    summary = load_json(shared_paths["attack_shared_dir"] / "fake_session_source_summary.json")
    assert summary["poison_runner_prepared"] is True
    assert summary["poison_runner_reason"] == "required_by_downstream_generated_suffix"
    assert summary["shared_identity_includes_poison_model"] is True


def test_train_template_distribution_csv_respects_diagnostics_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _train_template_config_without_distribution_csv(tmp_path)
    _patch_lightweight_shared_setup(monkeypatch)

    def fail_poison_runner(*args, **kwargs):
        raise AssertionError("poison runner should not be prepared")

    monkeypatch.setattr(pipeline_utils, "_load_or_train_poison_runner", fail_poison_runner)

    pipeline_utils.prepare_shared_attack_artifacts(
        config,
        run_type="attack",
        require_poison_runner=False,
    )

    shared_paths = shared_artifact_paths(config, run_type="attack")
    assert (shared_paths["attack_shared_dir"] / "fake_session_source_summary.json").exists()
    assert not (
        shared_paths["attack_shared_dir"] / "fake_session_source_length_distribution.csv"
    ).exists()
