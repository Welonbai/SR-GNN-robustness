from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

from attack.common.artifact_io import load_json, save_json
from attack.common.config import load_config
from attack.common.paths import run_metadata_paths, shared_artifact_paths
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    _initial_artifact_manifest,
    _key_payloads,
    _resolved_config_payload,
    run_targets_and_victims,
)
from attack.pipeline.core.victim_execution import VictimExecutionResult


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 3 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@contextmanager
def _phase3_temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_phase3" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _sample_stats():
    return compute_session_stats(
        [
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
        ]
    )


def _config_for_temp_root(temp_root: Path, *, victims: tuple[str, ...] | None = None):
    base = _base_config()
    if victims is None:
        victims = base.victims.enabled
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        targets=replace(base.targets, mode="sampled", bucket="all", count=1),
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


def test_resolved_config_separates_stable_run_group_from_execution_request() -> None:
    config = _base_config()

    resolved_payload = _resolved_config_payload(config, run_type="clean")
    derived = resolved_payload["derived"]
    stable_run_group = derived["stable_run_group"]
    execution_request = derived["execution_request"]

    assert stable_run_group["container_model"] == "appendable_experiment_container"
    assert stable_run_group["split_identity"]["key"].startswith("split_")
    assert stable_run_group["target_cohort_identity"]["key"].startswith("target_cohort_")
    assert stable_run_group["run_group_identity"]["key"].startswith("run_group_")
    assert stable_run_group["attack_identity"]["key"].startswith("attack_")
    assert "shared_attack_artifact_identity" in stable_run_group["attack_identity"]
    assert set(stable_run_group["victim_prediction_identities"]) == set(config.victims.enabled)
    assert "legacy_identities" in stable_run_group
    assert "target_selection_identity" in stable_run_group["legacy_identities"]
    assert "evaluation_identity" in stable_run_group["legacy_identities"]

    assert execution_request["request_model"] == "append_invocation"
    assert execution_request["execution_semantics"] == "append_against_stable_run_group"
    assert execution_request["requested_victims"] == list(config.victims.enabled)
    assert execution_request["target_request"]["requested_target_count"] == config.targets.count
    assert execution_request["target_request"]["count_interpretation"] == "prefix_length"

    assert "split_identity" not in derived
    assert "target_cohort_identity" not in derived
    assert "run_group_identity" not in derived
    assert "attack_identity" not in derived
    assert "victim_prediction_identities" not in derived
    assert "legacy_identities" not in derived


def test_key_payloads_use_canonical_identity_objects_with_key_and_payload() -> None:
    config = _base_config()

    key_payloads = _key_payloads(config, run_type="clean")
    stable_run_group = key_payloads["stable_run_group"]

    for identity_name in ("split_identity", "target_cohort_identity", "run_group_identity"):
        identity = stable_run_group[identity_name]
        assert set(identity) == {"key", "payload"}

    attack_identity = stable_run_group["attack_identity"]
    assert {"key", "payload", "shared_attack_artifact_identity"} <= set(attack_identity)
    assert set(attack_identity["shared_attack_artifact_identity"]) == {"key", "payload"}
    for victim_identity in stable_run_group["victim_prediction_identities"].values():
        assert set(victim_identity) == {"key", "payload"}

    assert "legacy_identities" in stable_run_group
    assert "target_selection_identity" in stable_run_group["legacy_identities"]
    assert "evaluation_identity" in stable_run_group["legacy_identities"]
    assert "target_selection_identity" not in key_payloads
    assert "evaluation_identity" not in key_payloads


def test_artifact_manifest_is_reorganized_around_stable_run_group_metadata() -> None:
    with _phase3_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root)
        context = _minimal_context(config, run_type="clean")
        metadata_paths = run_metadata_paths(config, run_type="clean")

        manifest = _initial_artifact_manifest(
            config,
            context=context,
            run_type="clean",
            metadata_paths=metadata_paths,
        )

    assert set(manifest) >= {
        "run_type",
        "identities",
        "execution_request",
        "shared_artifacts",
        "run_group_artifacts",
    }
    assert manifest["identities"]["container_model"] == "appendable_experiment_container"
    assert "legacy_identities" in manifest["identities"]
    assert "target_selection_identity" not in manifest["identities"]
    assert "evaluation_identity" not in manifest["identities"]

    run_group_artifacts = manifest["run_group_artifacts"]
    assert run_group_artifacts["run_coverage"].endswith("run_coverage.json")
    assert run_group_artifacts["execution_log"].endswith("execution_log.json")
    assert run_group_artifacts["summary_current"].endswith("summary_current.json")
    assert run_group_artifacts["progress"].endswith("progress.json")
    assert run_group_artifacts["legacy_summary"].endswith("summary_clean.json")
    assert not Path(run_group_artifacts["run_root"]).is_absolute()
    assert not Path(run_group_artifacts["summary_current"]).is_absolute()

    shared_artifacts = manifest["shared_artifacts"]
    assert shared_artifacts["target_cohort"]["target_registry"].endswith("target_registry.json")
    assert not Path(shared_artifacts["target_cohort"]["target_registry"]).is_absolute()
    assert "legacy_target_selection" in shared_artifacts


def test_batch_execution_behavior_is_unchanged_apart_from_metadata_semantics(
    monkeypatch,
) -> None:
    with _phase3_temp_root() as temp_root:
        config = _config_for_temp_root(temp_root, victims=("miasrec",))
        context = _minimal_context(config, run_type="clean")

        def fake_build_poisoned(target_item: int) -> TargetPoisonOutput:
            return TargetPoisonOutput(
                poisoned=PoisonedDataset(
                    sessions=[[1, 2]],
                    labels=[int(target_item)],
                    clean_count=1,
                    fake_count=0,
                ),
                metadata={"phase3_smoke": True},
            )

        def fake_victim_execution(*args, **kwargs):
            save_json({"victim_name": "miasrec", "fake": True}, kwargs["artifacts"]["resolved_config"])
            save_json(
                {"rankings": [[101, 1, 2, 3]], "victim": "miasrec"},
                kwargs["predictions_path"],
            )
            return (
                VictimExecutionResult(
                    predictions=[[101, 1, 2, 3]],
                    predictions_path=kwargs["predictions_path"],
                    extra={"phase3_smoke": True},
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

        summary = run_targets_and_victims(
            config,
            config_path=None,
            context=context,
            run_type="clean",
            build_poisoned=fake_build_poisoned,
        )
        metadata_paths = run_metadata_paths(config, run_type="clean")
        resolved_payload = load_json(metadata_paths["resolved_config"])
        progress_payload = load_json(metadata_paths["progress"])
        manifest = load_json(metadata_paths["artifact_manifest"])
        summary_payload = load_json(metadata_paths["summary"])
        assert len(summary["target_items"]) == 1
        assert metadata_paths["progress"].exists()
        assert metadata_paths["summary"].exists()
        assert metadata_paths["resolved_config"].exists()
        assert metadata_paths["artifact_manifest"].exists()

        assert resolved_payload["derived"]["stable_run_group"]["container_model"] == (
            "appendable_experiment_container"
        )
        assert resolved_payload["derived"]["execution_request"]["requested_victims"] == ["miasrec"]
        assert progress_payload["status"] == "completed"
        target_key = next(iter(summary_payload["targets"]))
        assert summary_payload["targets"][target_key]["victims"]["miasrec"]["metrics"]["targeted_recall@10"] == 1.0
        assert manifest["run_group_artifacts"]["legacy_summary"] == _repo_relative(
            metadata_paths["summary"]
        )
        assert manifest["output_files"]["summary"] == _repo_relative(metadata_paths["summary"])
