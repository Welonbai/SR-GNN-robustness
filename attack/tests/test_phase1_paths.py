from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

from attack.common.config import (
    FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED,
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    FakeSessionSourceConfig,
    TrainTemplateFallbackConfig,
    TrainTemplateSourceConfig,
    load_config,
)
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    attack_key_payload,
    run_config_dir,
    run_group_dir,
    run_group_key,
    run_group_key_payload,
    run_metadata_paths,
    shared_artifact_paths,
    shared_attack_artifact_key,
    shared_attack_artifact_key_payload,
    shared_attack_identity_requires_poison_runner,
    target_dir,
    target_cohort_key,
    target_cohort_key_payload,
    victim_prediction_key,
    victim_prediction_key_payload,
    victim_dir,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    build_pts_construction_attack_identity_context,
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
PTS_DIRECT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample.yaml"
)


def _base_config():
    assert CONFIG_PATH.is_file(), f"Missing Phase 1 test config at {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


def _pts_direct_config():
    assert PTS_DIRECT_CONFIG_PATH.is_file(), (
        f"Missing PTS direct CEM test config at {PTS_DIRECT_CONFIG_PATH}"
    )
    return load_config(PTS_DIRECT_CONFIG_PATH)


def _with_train_template_source(
    config,
    *,
    train_template: TrainTemplateSourceConfig | None = None,
):
    return replace(
        config,
        attack=replace(
            config.attack,
            fake_session_source=FakeSessionSourceConfig(
                type=FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
                train_template=train_template or TrainTemplateSourceConfig(),
            ),
        ),
    )


def _with_poison_epochs(config, epochs: int):
    poison_params = dict(config.attack.poison_model.params)
    poison_train = dict(poison_params["train"])
    poison_train["epochs"] = int(epochs)
    poison_params["train"] = poison_train
    return replace(
        config,
        attack=replace(
            config.attack,
            poison_model=replace(config.attack.poison_model, params=poison_params),
        ),
    )


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


def test_mdhg_data_semantics_only_changes_mdhg_victim_prediction_identity(monkeypatch) -> None:
    from attack.common import paths

    config = _base_config()
    mdhg_params = dict(config.victims.params)
    mdhg_params["mdhg"] = {
        "train": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 0.001,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    mdhg_config = replace(config, victims=replace(config.victims, params=mdhg_params))
    before_run_group = run_group_key(mdhg_config, run_type="clean")
    before_shared_attack = shared_attack_artifact_key(mdhg_config, run_type="clean")
    before_mdhg = victim_prediction_key(mdhg_config, "mdhg", run_type="clean")
    before_srgnn = victim_prediction_key(mdhg_config, "srgnn", run_type="clean")

    monkeypatch.setattr(paths, "MDHG_VICTIM_DATA_SEMANTICS", "changed_for_test")

    assert run_group_key(mdhg_config, run_type="clean") == before_run_group
    assert shared_attack_artifact_key(mdhg_config, run_type="clean") == before_shared_attack
    assert victim_prediction_key(mdhg_config, "mdhg", run_type="clean") != before_mdhg
    assert victim_prediction_key(mdhg_config, "srgnn", run_type="clean") == before_srgnn


def test_mdhg_runtime_diagnostics_do_not_change_identities() -> None:
    config = _base_config()
    params = dict(config.victims.params)
    params["mdhg"] = {
        "train": {
            "epochs": 20,
            "batch_size": 100,
            "lr": 0.001,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    runtime = dict(config.victims.runtime or {})
    runtime["mdhg"] = {
        "python_executable": "python",
        "repo_root": "third_party/mdhg",
        "working_dir": "third_party/mdhg",
        "device": {"use_gpu": True, "gpu_id": "0"},
    }
    base = replace(config, victims=replace(config.victims, params=params, runtime=runtime))
    diagnostic_runtime = dict(runtime)
    diagnostic_runtime["mdhg"] = {
        **runtime["mdhg"],
        "diagnostics": {
            "epoch_metrics": True,
            "per_epoch_predictions": True,
        },
    }
    diagnostic = replace(
        base,
        victims=replace(base.victims, runtime=diagnostic_runtime),
    )

    assert run_group_key(base, run_type="clean") == run_group_key(
        diagnostic, run_type="clean"
    )
    assert target_cohort_key(base) == target_cohort_key(diagnostic)
    assert shared_attack_artifact_key(base, run_type="clean") == shared_attack_artifact_key(
        diagnostic, run_type="clean"
    )
    for victim_name in ("srgnn", "miasrec", "tron", "mdhg"):
        assert victim_prediction_key(base, victim_name, run_type="clean") == (
            victim_prediction_key(diagnostic, victim_name, run_type="clean")
        )


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


def test_pts_direct_artifact_persistence_knobs_do_not_affect_identity() -> None:
    config = _pts_direct_config()
    pts = config.attack.pts_construction
    assert pts is not None
    run_type = "pts_construction_direct_action_mlp_cem"
    changed = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    save_top_k_candidates=3,
                ),
                artifacts=replace(
                    pts.artifacts,
                    save_per_session_records=not bool(
                        pts.artifacts.save_per_session_records
                    ),
                ),
            ),
        ),
    )

    assert attack_key_payload(config, run_type=run_type) == attack_key_payload(
        changed,
        run_type=run_type,
    )
    assert run_group_key(config, run_type=run_type) == run_group_key(
        changed,
        run_type=run_type,
    )
    assert victim_prediction_key(config, "tron", run_type=run_type) == victim_prediction_key(
        changed,
        "tron",
        run_type=run_type,
    )


def test_pts_direct_copy_source_run_group_excludes_runtime_poison_runner_requirement() -> None:
    config = _with_train_template_source(_pts_direct_config())
    run_type = PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE
    context = build_pts_construction_attack_identity_context(config)

    assert shared_attack_identity_requires_poison_runner(run_type) is False
    assert shared_attack_artifact_key_payload(config, run_type=run_type) == (
        shared_attack_artifact_key_payload(
            config,
            run_type=run_type,
            require_poison_runner=False,
        )
    )
    assert shared_attack_artifact_key_payload(
        config,
        run_type=run_type,
        require_poison_runner=True,
    ) != shared_attack_artifact_key_payload(config, run_type=run_type)
    assert run_group_key_payload(
        config,
        run_type=run_type,
        attack_identity_context=context,
    )["shared_attack_artifact_key"] == shared_attack_artifact_key(
        config,
        run_type=run_type,
    )


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


def test_poison_generated_fake_session_source_keeps_legacy_shared_identity() -> None:
    config = _base_config()
    explicit_poison = replace(
        config,
        attack=replace(
            config.attack,
            fake_session_source=FakeSessionSourceConfig(
                type=FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED
            ),
        ),
    )

    assert shared_attack_artifact_key_payload(config, run_type="attack") == (
        shared_attack_artifact_key_payload(explicit_poison, run_type="attack")
    )
    assert shared_attack_artifact_key(config, run_type="attack") == shared_attack_artifact_key(
        explicit_poison,
        run_type="attack",
    )
    assert shared_attack_artifact_key_payload(config, run_type="attack") == (
        shared_attack_artifact_key_payload(
            explicit_poison,
            run_type="attack",
            require_poison_runner=True,
        )
    )
    assert shared_attack_artifact_key(config, run_type="attack") == shared_attack_artifact_key(
        explicit_poison,
        run_type="attack",
        require_poison_runner=True,
    )
    assert attack_key_payload(config, run_type="attack") == attack_key_payload(
        explicit_poison,
        run_type="attack",
    )
    payload = shared_attack_artifact_key_payload(explicit_poison, run_type="attack")
    assert "fake_session_source" not in payload["attack_generation"]
    assert "fake_session_source" not in attack_key_payload(explicit_poison, run_type="attack")[
        "attack"
    ]


def test_train_template_fake_session_source_uses_source_aware_shared_identity() -> None:
    config = _base_config()
    train_template = _with_train_template_source(config)

    legacy_payload = shared_attack_artifact_key_payload(config, run_type="attack")
    train_payload = shared_attack_artifact_key_payload(train_template, run_type="attack")

    assert train_payload != legacy_payload
    assert shared_attack_artifact_key(train_template, run_type="attack") != (
        shared_attack_artifact_key(config, run_type="attack")
    )
    assert attack_key_payload(train_template, run_type="attack") != attack_key_payload(
        config,
        run_type="attack",
    )
    assert attack_key_payload(train_template, run_type="attack")["attack"][
        "fake_session_source"
    ]["type"] == FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    source_payload = train_payload["attack_generation"]["fake_session_source"]
    assert source_payload["type"] == FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    assert source_payload["length_matching_mode"] == "exact_largest_remainder"
    assert source_payload["target_filtering"] == "none"
    assert "poison_model" not in train_payload["attack_generation"]
    assert "fake_session_generation_topk" not in train_payload["attack_generation"]
    assert train_payload["attack_generation"]["shared_identity_includes_poison_model"] is False


def test_train_template_shared_identity_option_b_poison_model_dependency() -> None:
    config = _with_train_template_source(_base_config())
    changed_poison = _with_train_template_source(_with_poison_epochs(_base_config(), 30))

    no_runner_payload = shared_attack_artifact_key_payload(
        config,
        run_type="attack",
        require_poison_runner=False,
    )
    no_runner_changed_payload = shared_attack_artifact_key_payload(
        changed_poison,
        run_type="attack",
        require_poison_runner=False,
    )
    assert no_runner_payload == no_runner_changed_payload
    assert "poison_model" not in no_runner_payload["attack_generation"]

    with_runner_payload = shared_attack_artifact_key_payload(
        config,
        run_type="attack",
        require_poison_runner=True,
    )
    with_runner_changed_payload = shared_attack_artifact_key_payload(
        changed_poison,
        run_type="attack",
        require_poison_runner=True,
    )
    assert with_runner_payload != with_runner_changed_payload
    assert "poison_model" in with_runner_payload["attack_generation"]
    assert "fake_session_generation_topk" in with_runner_payload["attack_generation"]
    assert with_runner_payload["attack_generation"][
        "shared_identity_includes_poison_model"
    ] is True
    assert shared_attack_artifact_key(
        config,
        run_type="attack",
        require_poison_runner=True,
    ) != shared_attack_artifact_key(
        changed_poison,
        run_type="attack",
        require_poison_runner=True,
    )


def test_train_template_record_distribution_diagnostics_does_not_affect_shared_identity() -> None:
    config = _base_config()
    enabled = _with_train_template_source(
        config,
        train_template=TrainTemplateSourceConfig(record_distribution_diagnostics=True),
    )
    disabled = _with_train_template_source(
        config,
        train_template=TrainTemplateSourceConfig(record_distribution_diagnostics=False),
    )

    assert shared_attack_artifact_key_payload(enabled, run_type="attack") == (
        shared_attack_artifact_key_payload(disabled, run_type="attack")
    )
    assert shared_attack_artifact_key(enabled, run_type="attack") == shared_attack_artifact_key(
        disabled,
        run_type="attack",
    )
    assert shared_attack_artifact_key(
        enabled,
        run_type="attack",
        require_poison_runner=True,
    ) == shared_attack_artifact_key(
        disabled,
        run_type="attack",
        require_poison_runner=True,
    )


def test_train_template_shared_identity_changes_with_sampling_settings_seed_and_size() -> None:
    config = _with_train_template_source(_base_config())
    fallback_changed = _with_train_template_source(
        _base_config(),
        train_template=TrainTemplateSourceConfig(
            fallback=TrainTemplateFallbackConfig(nearest_length_redistribution=False)
        ),
    )
    seed_changed = replace(config, seeds=replace(config.seeds, fake_session_seed=123456))
    size_changed = replace(config, attack=replace(config.attack, size=config.attack.size + 0.001))

    base_key = shared_attack_artifact_key(config, run_type="attack")
    assert shared_attack_artifact_key(fallback_changed, run_type="attack") != base_key
    assert shared_attack_artifact_key(seed_changed, run_type="attack") != base_key
    assert shared_attack_artifact_key(size_changed, run_type="attack") != base_key


def test_shared_attack_artifact_key_records_srgnn_validation_best_protocol() -> None:
    config = _base_config()
    poison_params = dict(config.attack.poison_model.params)
    poison_train = dict(poison_params["train"])
    poison_train.update(
        {
            "epochs": 30,
            "checkpoint_protocol": "validation_best",
            "best_metric": "valid_ground_truth_mrr@20",
            "patience_metric": "recall20_or_mrr20",
        }
    )
    poison_params["train"] = poison_train
    validation_config = replace(
        config,
        attack=replace(
            config.attack,
            poison_model=replace(config.attack.poison_model, params=poison_params),
        ),
    )

    payload = shared_attack_artifact_key_payload(validation_config, run_type="attack")
    poison_identity = payload["attack_generation"]["poison_model"]

    assert poison_identity["poison_model_training_protocol"] == "validation_best"
    assert poison_identity["poison_model_best_metric"] == "valid_ground_truth_mrr@20"
    assert poison_identity["poison_model_patience_metric"] == "recall20_or_mrr20"
    assert poison_identity["poison_model_max_epochs"] == 30
    assert poison_identity["poison_model_patience"] == 10
    assert shared_attack_artifact_key(config, run_type="attack") != shared_attack_artifact_key(
        validation_config,
        run_type="attack",
    )


def test_victim_prediction_key_ignores_batch_size_fields() -> None:
    config = _base_config()
    changed_victim_params = {
        name: dict(params)
        for name, params in config.victims.params.items()
    }
    miasrec_params = dict(changed_victim_params["miasrec"])
    miasrec_train = dict(miasrec_params["train"])
    miasrec_train["train_batch_size"] = 64
    miasrec_train["eval_batch_size"] = 32
    miasrec_params["train"] = miasrec_train
    changed_victim_params["miasrec"] = miasrec_params

    changed_config = replace(
        config,
        victims=replace(config.victims, params=changed_victim_params),
    )

    base_payload = victim_prediction_key_payload(config, "miasrec", run_type="attack")
    changed_payload = victim_prediction_key_payload(
        changed_config,
        "miasrec",
        run_type="attack",
    )

    assert base_payload == changed_payload
    assert "train_batch_size" not in base_payload["victim_params"]["train"]
    assert "eval_batch_size" not in base_payload["victim_params"]["train"]
    assert victim_prediction_key(config, "miasrec", run_type="attack") == victim_prediction_key(
        changed_config,
        "miasrec",
        run_type="attack",
    )


def test_victim_prediction_key_records_srgnn_validation_best_protocol() -> None:
    config = _base_config()
    changed_victim_params = {
        name: dict(params)
        for name, params in config.victims.params.items()
    }
    srgnn_params = dict(changed_victim_params["srgnn"])
    srgnn_train = dict(srgnn_params["train"])
    srgnn_train.update(
        {
            "epochs": 30,
            "checkpoint_protocol": "validation_best",
            "best_metric": "valid_ground_truth_mrr@20",
            "patience_metric": "recall20_or_mrr20",
        }
    )
    srgnn_params["train"] = srgnn_train
    changed_victim_params["srgnn"] = srgnn_params
    validation_config = replace(
        config,
        victims=replace(config.victims, params=changed_victim_params),
    )

    payload = victim_prediction_key_payload(validation_config, "srgnn", run_type="attack")

    assert payload["victim_srgnn_training_protocol"] == "validation_best"
    assert payload["victim_srgnn_best_metric"] == "valid_ground_truth_mrr@20"
    assert payload["victim_srgnn_patience_metric"] == "recall20_or_mrr20"
    assert payload["victim_srgnn_max_epochs"] == 30
    assert payload["victim_srgnn_patience"] == 10
    assert victim_prediction_key(config, "srgnn", run_type="attack") != victim_prediction_key(
        validation_config,
        "srgnn",
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
