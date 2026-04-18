from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

from attack.common.config import load_config
from attack.common.paths import (
    attack_key_payload,
    run_config_dir,
    run_group_dir,
    run_group_key,
    run_group_key_payload,
    run_metadata_paths,
    shared_artifact_paths,
    shared_attack_artifact_key,
    target_dir,
    target_cohort_key,
    target_cohort_key_payload,
    victim_dir,
)
from attack.common.artifact_io import save_json
from attack.pipeline.core.orchestrator import (
    _guard_phase1_run_group_reuse,
    _key_payloads,
    _resolved_config_payload,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"
)


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 1 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


def test_target_cohort_key_ignores_requested_count_for_sampled_targets() -> None:
    config = _base_config()
    smaller = replace(config, targets=replace(config.targets, count=3))
    larger = replace(config, targets=replace(config.targets, count=6))

    assert target_cohort_key(smaller) == target_cohort_key(larger)
    assert target_cohort_key_payload(smaller) == target_cohort_key_payload(larger)


def test_target_cohort_key_ignores_reuse_saved_targets_flag() -> None:
    config = _base_config()
    reuse = replace(config, targets=replace(config.targets, reuse_saved_targets=True))
    no_reuse = replace(config, targets=replace(config.targets, reuse_saved_targets=False))

    assert target_cohort_key(reuse) == target_cohort_key(no_reuse)
    assert target_cohort_key_payload(reuse) == target_cohort_key_payload(no_reuse)


def test_run_group_key_ignores_enabled_victim_set() -> None:
    config = _base_config()
    single_victim = replace(config, victims=replace(config.victims, enabled=("srgnn",)))
    full_victims = replace(
        config,
        victims=replace(config.victims, enabled=("srgnn", "miasrec", "tron")),
    )

    assert run_group_key(single_victim, run_type="clean") == run_group_key(
        full_victims,
        run_type="clean",
    )
    assert run_group_key_payload(single_victim, run_type="clean") == run_group_key_payload(
        full_victims,
        run_type="clean",
    )
    assert "final_attack_key" in run_group_key_payload(single_victim, run_type="clean")
    assert "attack_key" not in run_group_key_payload(single_victim, run_type="clean")


def test_run_group_key_changes_with_attack_or_evaluation_identity() -> None:
    config = _base_config()
    different_attack = replace(
        config,
        attack=replace(config.attack, replacement_topk_ratio=0.123),
    )
    different_eval = replace(
        config,
        evaluation=replace(config.evaluation, topk=(1,)),
    )

    assert run_group_key(config, run_type="attack") != run_group_key(
        different_attack,
        run_type="attack",
    )
    assert run_group_key(config, run_type="attack") != run_group_key(
        different_eval,
        run_type="attack",
    )


def test_attack_key_payload_no_longer_depends_on_target_selection_identity() -> None:
    payload = attack_key_payload(_base_config(), run_type="attack")

    assert "target_selection_key" not in payload


def test_shared_attack_artifact_key_stays_stable_across_phase1_identity_changes() -> None:
    config = _base_config()
    changed_targets = replace(config, targets=replace(config.targets, count=99))
    changed_victims = replace(config, victims=replace(config.victims, enabled=("srgnn",)))

    assert shared_attack_artifact_key(config, run_type="attack") == shared_attack_artifact_key(
        changed_targets,
        run_type="attack",
    )
    assert shared_attack_artifact_key(config, run_type="attack") == shared_attack_artifact_key(
        changed_victims,
        run_type="attack",
    )


def test_run_paths_use_run_group_identity_and_expose_new_phase1_artifacts() -> None:
    config = _base_config()
    run_root = run_group_dir(config, run_type="clean")

    assert run_config_dir(config, run_type="clean") == run_root

    metadata_paths = run_metadata_paths(config, run_type="clean")
    assert metadata_paths["run_root"].name == run_group_key(config, run_type="clean")
    assert metadata_paths["run_coverage"].name == "run_coverage.json"
    assert metadata_paths["execution_log"].name == "execution_log.json"
    assert metadata_paths["summary_current"].name == "summary_current.json"

    target_root = target_dir(config, 42, run_type="clean")
    victim_root = victim_dir(config, 42, run_type="clean", victim_name="srgnn")
    assert target_root.parent == run_root / "targets"
    assert target_root.is_relative_to(run_root)
    assert victim_root.parent == target_root / "victims"
    assert victim_root.is_relative_to(run_root)

    shared_paths = shared_artifact_paths(config, run_type="clean")
    assert shared_paths["target_registry"].parent == shared_paths["target_cohort_dir"]
    assert shared_paths["target_cohort_dir"].parent.name == "target_cohorts"
    assert shared_paths["target_shared_dir"].parent.name == "targets"


def test_canonical_identity_metadata_uses_explicit_key_payload_objects() -> None:
    config = _base_config()

    resolved_payload = _resolved_config_payload(config, run_type="clean")
    derived = resolved_payload["derived"]
    stable_run_group = derived["stable_run_group"]
    assert stable_run_group["split_identity"]["key"].startswith("split_")
    assert "payload" in stable_run_group["split_identity"]
    assert stable_run_group["target_cohort_identity"]["key"].startswith("target_cohort_")
    assert stable_run_group["run_group_identity"]["key"].startswith("run_group_")
    assert stable_run_group["attack_identity"]["key"].startswith("attack_")
    assert "shared_attack_artifact_identity" in stable_run_group["attack_identity"]
    assert set(stable_run_group["victim_prediction_identities"]) == set(config.victims.enabled)
    assert "split_key" not in derived
    assert "target_cohort_key" not in derived
    assert "run_group_key" not in derived
    assert "attack_key" not in derived
    assert "victim_prediction_keys" not in derived
    assert "legacy_identities" in stable_run_group
    assert "target_selection_identity" in stable_run_group["legacy_identities"]
    assert "evaluation_identity" in stable_run_group["legacy_identities"]

    key_payloads = _key_payloads(config, run_type="clean")
    stable_key_payloads = key_payloads["stable_run_group"]
    assert stable_key_payloads["split_identity"]["key"].startswith("split_")
    assert "payload" in stable_key_payloads["split_identity"]
    assert "split_key_payload" not in key_payloads
    assert "target_cohort_key_payload" not in key_payloads
    assert "run_group_key_payload" not in key_payloads
    assert "attack_key_payload" not in key_payloads
    assert "victim_prediction_key_payloads" not in key_payloads
    assert "legacy_identities" in stable_key_payloads


def test_phase1_guardrail_rejects_incompatible_existing_run_group_root() -> None:
    base_dir = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "tests"
        / "_tmp_phase1_guardrail"
        / uuid4().hex
    )
    try:
        base_dir.mkdir(parents=True, exist_ok=True)

        base_config = _base_config()
        config = replace(
            base_config,
            artifacts=replace(base_config.artifacts, root=str(base_dir)),
        )
        metadata_paths = run_metadata_paths(config, run_type="clean")
        metadata_paths["run_root"].mkdir(parents=True, exist_ok=True)
        existing_payload = _resolved_config_payload(config, run_type="clean")
        save_json(existing_payload, metadata_paths["resolved_config"])

        incompatible_config = replace(
            config,
            victims=replace(config.victims, enabled=("srgnn",)),
        )
        incompatible_paths = run_metadata_paths(incompatible_config, run_type="clean")

        try:
            _guard_phase1_run_group_reuse(
                incompatible_config,
                run_type="clean",
                metadata_paths=incompatible_paths,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError(
                "Expected Phase 1 run-group guardrail to reject incompatible reuse."
            )

        assert "Run-group root collision detected" in message
        assert "append semantics are not implemented until later phases" in message
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
