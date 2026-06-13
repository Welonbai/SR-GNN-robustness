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
FORMAL_CONFIG_STEMS = (
    "ssh_diginetica_valbest_clean_sample10",
    "ssh_diginetica_valbest_clean_unpopular_sample10_fixed_epoch",
    "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10",
    "ssh_diginetica_valbest_attack_random_nonzero_when_possible_ratio1_unpopular_sample10_fixed_epoch",
    "ssh_diginetica_valbest_attack_create_copy_source_popular_sample",
    "ssh_diginetica_valbest_attack_create_copy_source_unpopular_sample",
    "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train",
    "ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train",
    "ssh_yoochoose1_64_valbest_clean_sample10_popular",
    "ssh_yoochoose1_64_valbest_clean_sample10_unpopular",
    "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_popular",
    "ssh_yoochoose1_64_valbest_attack_random_nonzero_when_possible_ratio1_sample_unpopular",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_popular_copy_train",
    "ssh_yoochoose1_64_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train",
)


def _run_type_and_context(config, config_name: str):
    if "_clean_" in config_name:
        return "clean", None
    if "random_nonzero" in config_name:
        return "random_nonzero_when_possible", None
    if "create_copy_source" in config_name:
        return CREAT_ADDITIVE_SBR_RUN_TYPE, None
    run_type = _pts_construction_run_type(config)
    return run_type, build_pts_construction_attack_identity_context(config)


@pytest.mark.parametrize("stem", FORMAL_CONFIG_STEMS)
def test_freqrec_formal_config_preserves_source_attack_identity(stem: str) -> None:
    source = load_config(CONFIG_DIR / f"{stem}_mdhg_only.yaml")
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


@pytest.mark.parametrize("stem", FORMAL_CONFIG_STEMS)
def test_freqrec_formal_config_uses_fixed_epoch_ssh_profile(stem: str) -> None:
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
    assert runtime["device"]["gpu_id"] in {"0", "1"}
    assert runtime["dataloader"]["num_workers"] == 0
