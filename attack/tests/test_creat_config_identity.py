from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import CreatAdditiveSBRConfig, load_config
from attack.common.paths import (
    CREAT_ADDITIVE_SBR_RUN_TYPE,
    INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    attack_key,
    run_group_key_payload,
    shared_attack_artifact_key,
    shared_attack_identity_requires_poison_runner,
)


def test_creat_config_parses_defaults_from_yaml() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    creat = config.attack.creat_additive_sbr
    assert creat is not None
    assert creat.enabled is True
    assert creat.max_attack_num == 1
    assert creat.attack_reward_mode == "score"
    assert creat.seed_source == "position_opt_seed"


def test_creat_config_rejects_unsupported_v1_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        CreatAdditiveSBRConfig(enabled=True, epochs=0)
    assert CreatAdditiveSBRConfig(enabled=False, epochs=0).epochs == 0
    with pytest.raises(ValueError, match="max_attack_num"):
        CreatAdditiveSBRConfig(max_attack_num=2)
    with pytest.raises(ValueError, match="attack_reward_mode"):
        CreatAdditiveSBRConfig(attack_reward_mode="rank")
    with pytest.raises(ValueError, match="seed_source"):
        CreatAdditiveSBRConfig(seed_source="fake_session_seed")


def test_creat_config_changes_final_attack_identity() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    assert config.attack.creat_additive_sbr is not None
    changed = replace(
        config,
        attack=replace(
            config.attack,
            creat_additive_sbr=replace(config.attack.creat_additive_sbr, epochs=11),
        ),
    )
    assert attack_key(config, run_type=CREAT_ADDITIVE_SBR_RUN_TYPE) != attack_key(
        changed,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
    )


def test_creat_shared_identity_uses_poison_runner_and_matches_copy_source_direct_cem() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    assert shared_attack_identity_requires_poison_runner(CREAT_ADDITIVE_SBR_RUN_TYPE)
    run_group_payload = run_group_key_payload(
        config,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
    )
    expected_creat_shared_key = shared_attack_artifact_key(
        config,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        require_poison_runner=True,
    )
    assert run_group_payload["shared_attack_artifact_key"] == expected_creat_shared_key
    assert expected_creat_shared_key == shared_attack_artifact_key(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
        require_poison_runner=True,
    )


def test_shared_identity_helper_covers_known_poison_runner_run_types() -> None:
    for run_type in (
        CREAT_ADDITIVE_SBR_RUN_TYPE,
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
        TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
        TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    ):
        assert shared_attack_identity_requires_poison_runner(run_type)
