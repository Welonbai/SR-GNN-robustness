from __future__ import annotations

from pathlib import Path

import pytest

from attack.common.config import load_config
from attack.common.paths import (
    CREAT_ADDITIVE_SBR_RUN_TYPE,
    attack_key,
    poison_model_key,
    run_group_key,
    shared_attack_artifact_key,
    target_cohort_key,
)
from attack.pipeline.runs.run_pts_construction_cem import (
    _pts_construction_run_type,
    build_pts_construction_attack_identity_context,
)


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DIAGNOSTICS_CONFIG_DIR = CONFIG_DIR / "diagnostics"
FORMAL_CONFIG_PAIRS = (
    ("ssh_diginetica_valbest_clean_sample10", "ssh_diginetica_valbest_clean_sample10_mdhg_only.yaml"),
    (
        "ssh_diginetica_valbest_clean_unpopular_sample10_fixed_epoch",
        "ssh_diginetica_valbest_clean_unpopular_sample10_fixed_epoch_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10",
        "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_unpopular_sample10_fixed_epoch",
        "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_unpopular_sample10_fixed_epoch_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular",
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular",
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_unpopular_sample10_fixed_epoch_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train",
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train",
        "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_create_copy_source_popular_sample",
        "ssh_diginetica_valbest_attack_create_copy_source_popular_sample_mdhg_only.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_create_copy_source_unpopular_sample",
        "ssh_diginetica_valbest_attack_create_copy_source_unpopular_sample_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_clean_sample10_popular",
        "ssh_yoochoose1_64_valbest_clean_sample10_popular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_clean_sample10_unpopular",
        "ssh_yoochoose1_64_valbest_clean_sample10_unpopular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_popular",
        "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_popular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_unpopular",
        "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_unpopular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train_mdhg_only.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_create_copy_source_popular_sample",
        "ssh_yoochoose1_64_valbest_attack_create_copy_source_popular_sample_nonzero_dpp001_consistency03.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_create_copy_source_unpopular_sample",
        "ssh_yoochoose1_64_valbest_attack_create_copy_source_unpopular_sample_nonzero_dpp001_consistency03.yaml",
    ),
)

FREQREC_DIRECT_CEM_CONFIG_NAMES = {
    "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_freqrec_only.yaml",
    "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_freqrec_only.yaml",
    "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_generated_popular_freqrec_only.yaml",
    "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_generated_unpopular_freqrec_only.yaml",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_freqrec_only.yaml",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_freqrec_only.yaml",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_generated_popular_freqrec_only.yaml",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_generated_unpopular_freqrec_only.yaml",
}
FREQREC_DIRECT_CEM_SMOKE_CONFIG_NAMES = {
    "diginetica_smoke_ptscem_direct_freqrec_surrogate_copy_source.yaml",
    "diginetica_smoke_ptscem_direct_freqrec_generated.yaml",
    "yoochoose1_64_smoke_ptscem_direct_freqrec_surrogate_copy_source.yaml",
    "yoochoose1_64_smoke_ptscem_direct_freqrec_generated.yaml",
}
FREQREC_DIRECT_CEM_REMAINING5_CONFIG_PAIRS = {
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_freqrec_only.yaml",
        "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_remaining5_victims.yaml",
    ),
    (
        "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_freqrec_only.yaml",
        "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_remaining5_victims.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_freqrec_only.yaml",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_popular_remaining5_victims.yaml",
    ),
    (
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_freqrec_only.yaml",
        "ssh_yoochoose1_64_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source_unpopular_remaining5_victims.yaml",
    ),
}


def _run_type_and_context(config, config_name: str):
    if "_clean_" in config_name:
        return "clean", None
    if "random_nonzero" in config_name:
        return "random_nonzero_when_possible", None
    if "create_copy_source" in config_name:
        return CREAT_ADDITIVE_SBR_RUN_TYPE, None
    run_type = _pts_construction_run_type(config)
    return run_type, build_pts_construction_attack_identity_context(config)


def test_complete_formal_freqrec_matrix_exists() -> None:
    expected = {
        f"{stem}_freqrec_only.yaml" for stem, _source_name in FORMAL_CONFIG_PAIRS
    } | FREQREC_DIRECT_CEM_CONFIG_NAMES
    actual = {
        path.name for path in CONFIG_DIR.glob("ssh_*_freqrec_only.yaml")
    }
    assert actual == expected
    assert len(actual) == 28


@pytest.mark.parametrize(("stem", "source_name"), FORMAL_CONFIG_PAIRS)
def test_freqrec_formal_config_preserves_source_attack_identity(
    stem: str,
    source_name: str,
) -> None:
    source = load_config(CONFIG_DIR / source_name)
    freqrec = load_config(CONFIG_DIR / f"{stem}_freqrec_only.yaml")
    run_type, source_context = _run_type_and_context(source, stem)
    _, freqrec_context = _run_type_and_context(freqrec, stem)

    for field in (
        "experiment",
        "data",
        "seeds",
        "attack",
        "targets",
        "evaluation",
        "artifacts",
    ):
        assert getattr(freqrec, field) == getattr(source, field)

    assert source_context == freqrec_context
    assert poison_model_key(freqrec) == poison_model_key(source)
    assert target_cohort_key(freqrec) == target_cohort_key(source)
    assert shared_attack_artifact_key(
        freqrec, run_type=run_type
    ) == shared_attack_artifact_key(source, run_type=run_type)
    assert attack_key(
        freqrec,
        run_type=run_type,
        attack_identity_context=freqrec_context,
    ) == attack_key(
        source,
        run_type=run_type,
        attack_identity_context=source_context,
    )
    assert run_group_key(
        freqrec,
        run_type=run_type,
        attack_identity_context=freqrec_context,
    ) == run_group_key(
        source,
        run_type=run_type,
        attack_identity_context=source_context,
    )


@pytest.mark.parametrize(("stem", "_source_name"), FORMAL_CONFIG_PAIRS)
def test_freqrec_formal_config_uses_fixed_epoch_ssh_profile(
    stem: str,
    _source_name: str,
) -> None:
    config = load_config(CONFIG_DIR / f"{stem}_freqrec_only.yaml")
    train = config.victims.params["freqrec"]["train"]
    runtime = config.victims.runtime["freqrec"]

    assert config.victims.enabled == ("freqrec",)
    assert train["model_type"] == "freqrec"
    assert train["epochs"] == 18
    assert train["checkpoint_protocol"] == "fixed_epoch"
    assert train["metric_cutoffs"] == [5, 10, 20]
    assert train["fre"] == 1.0
    assert train["fourier_loss"] is True
    assert runtime["python_executable"] == (
        "/data2/welon/miniconda3/envs/robustness/bin/python"
    )
    assert runtime["repo_root"] == (
        "/data2/welon/SR-GNN-robustness/third_party/freqrec"
    )
    assert runtime["working_dir"] == runtime["repo_root"]
    assert runtime["device"]["use_gpu"] is True
    expected_gpu = "1" if "unpopular" in stem else "0"
    assert runtime["device"]["gpu_id"] == expected_gpu
    assert runtime["dataloader"]["num_workers"] == 0


@pytest.mark.parametrize("config_name", sorted(FREQREC_DIRECT_CEM_CONFIG_NAMES))
def test_freqrec_direct_cem_formal_configs_use_tmux_visible_gpu_zero(
    config_name: str,
) -> None:
    config = load_config(CONFIG_DIR / config_name)
    assert config.victims.runtime["freqrec"]["device"]["gpu_id"] == "0"


@pytest.mark.parametrize(
    "config_name",
    sorted(
        name
        for name in FREQREC_DIRECT_CEM_CONFIG_NAMES
        if "surrogate_copy_source" in name
    ),
)
def test_freqrec_direct_cem_copy_source_configs_use_freqrec_suffix_generator(
    config_name: str,
) -> None:
    config = load_config(CONFIG_DIR / config_name)
    assert (
        config.attack.fake_session_source.type
        == "train_template_clean_exact_length_matched"
    )
    assert config.attack.poison_model.name == "freqrec"
    assert config.attack.poison_model.params["train"]["epochs"] == 18
    assert (
        config.attack.pts_construction.cem.surrogate_model.name
        == "freqrec"
    )


@pytest.mark.parametrize(
    "config_name",
    sorted(name for name in FREQREC_DIRECT_CEM_CONFIG_NAMES if "generated" in name),
)
def test_freqrec_direct_cem_generated_configs_use_freqrec_generator_and_surrogate(
    config_name: str,
) -> None:
    config = load_config(CONFIG_DIR / config_name)
    assert config.attack.fake_session_source.type == "poison_model_generated"
    assert config.attack.poison_model.name == "freqrec"
    assert (
        config.attack.pts_construction.cem.surrogate_model.name
        == "freqrec"
    )


@pytest.mark.parametrize(
    ("base_name", "remaining_name"),
    sorted(FREQREC_DIRECT_CEM_REMAINING5_CONFIG_PAIRS),
)
def test_freqrec_direct_cem_remaining5_configs_only_change_final_victims(
    base_name: str,
    remaining_name: str,
) -> None:
    base = load_config(CONFIG_DIR / base_name)
    remaining = load_config(CONFIG_DIR / remaining_name)

    assert remaining.victims.enabled == ("srgnn", "miasrec", "tron", "mdhg", "wearec")
    assert set(remaining.victims.params) == set(remaining.victims.enabled)
    assert "freqrec" not in remaining.victims.params
    assert "freqrec" not in remaining.victims.runtime
    assert remaining.attack.poison_model.name == "freqrec"
    assert remaining.attack.pts_construction.cem.surrogate_model.name == "freqrec"
    for runtime in remaining.victims.runtime.values():
        device = runtime.get("device")
        if isinstance(device, dict):
            assert device.get("gpu_id") == "0"

    for field in (
        "data",
        "seeds",
        "attack",
        "targets",
        "evaluation",
        "artifacts",
    ):
        assert getattr(remaining, field) == getattr(base, field)

    run_type, base_context = _run_type_and_context(base, base_name)
    _, remaining_context = _run_type_and_context(remaining, remaining_name)
    assert remaining_context == base_context
    assert shared_attack_artifact_key(
        remaining,
        run_type=run_type,
    ) == shared_attack_artifact_key(base, run_type=run_type)
    assert attack_key(
        remaining,
        run_type=run_type,
        attack_identity_context=remaining_context,
    ) == attack_key(
        base,
        run_type=run_type,
        attack_identity_context=base_context,
    )
    assert run_group_key(
        remaining,
        run_type=run_type,
        attack_identity_context=remaining_context,
    ) == run_group_key(
        base,
        run_type=run_type,
        attack_identity_context=base_context,
    )


@pytest.mark.parametrize("config_name", sorted(FREQREC_DIRECT_CEM_SMOKE_CONFIG_NAMES))
def test_freqrec_direct_cem_smoke_configs_use_tmux_visible_gpu_zero(
    config_name: str,
) -> None:
    config = load_config(DIAGNOSTICS_CONFIG_DIR / config_name)
    assert config.victims.runtime["freqrec"]["device"]["gpu_id"] == "0"


@pytest.mark.parametrize(
    "config_name",
    sorted(
        name
        for name in FREQREC_DIRECT_CEM_SMOKE_CONFIG_NAMES
        if "surrogate_copy_source" in name
    ),
)
def test_freqrec_direct_cem_smoke_copy_source_uses_freqrec_suffix_generator(
    config_name: str,
) -> None:
    config = load_config(DIAGNOSTICS_CONFIG_DIR / config_name)
    assert (
        config.attack.fake_session_source.type
        == "train_template_clean_exact_length_matched"
    )
    assert config.attack.poison_model.name == "freqrec"
    assert config.attack.poison_model.params["train"]["epochs"] == 1
    assert (
        config.attack.pts_construction.cem.surrogate_model.name
        == "freqrec"
    )


@pytest.mark.parametrize(
    "config_name",
    sorted(name for name in FREQREC_DIRECT_CEM_SMOKE_CONFIG_NAMES if "generated" in name),
)
def test_freqrec_direct_cem_smoke_generated_uses_freqrec_generator_and_surrogate(
    config_name: str,
) -> None:
    config = load_config(DIAGNOSTICS_CONFIG_DIR / config_name)
    assert config.attack.fake_session_source.type == "poison_model_generated"
    assert config.attack.poison_model.name == "freqrec"
    assert (
        config.attack.pts_construction.cem.surrogate_model.name
        == "freqrec"
    )


@pytest.mark.parametrize("dataset", ("diginetica", "yoochoose1_64"))
@pytest.mark.parametrize("bucket", ("popular", "unpopular"))
def test_generate_source_and_copy_train_cem_identities_remain_distinct(
    dataset: str,
    bucket: str,
) -> None:
    prefix = (
        f"ssh_{dataset}_valbest_attack_ptscem_direct_guassian_mlp_internal_"
        f"sample_{bucket}"
    )
    generated = load_config(CONFIG_DIR / f"{prefix}_freqrec_only.yaml")
    copied = load_config(CONFIG_DIR / f"{prefix}_copy_train_freqrec_only.yaml")
    run_type = _pts_construction_run_type(generated)
    generated_context = build_pts_construction_attack_identity_context(generated)
    copied_context = build_pts_construction_attack_identity_context(copied)

    assert generated.attack.fake_session_source.type == "poison_model_generated"
    assert (
        copied.attack.fake_session_source.type
        == "train_template_clean_exact_length_matched"
    )
    assert shared_attack_artifact_key(
        generated, run_type=run_type
    ) != shared_attack_artifact_key(copied, run_type=run_type)
    assert attack_key(
        generated,
        run_type=run_type,
        attack_identity_context=generated_context,
    ) != attack_key(
        copied,
        run_type=run_type,
        attack_identity_context=copied_context,
    )
    assert run_group_key(
        generated,
        run_type=run_type,
        attack_identity_context=generated_context,
    ) != run_group_key(
        copied,
        run_type=run_type,
        attack_identity_context=copied_context,
    )
