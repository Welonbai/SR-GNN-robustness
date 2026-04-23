from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from attack.common.config import PositionOptConfig, load_config
from attack.common.paths import (
    POSITION_OPT_SHARED_POLICY_RUN_TYPE,
    attack_key,
    build_position_opt_attack_identity_context,
    run_group_key,
    shared_attack_artifact_key,
    victim_prediction_key,
)
from attack.pipeline.runs.run_position_opt_shared_policy import _resolve_position_opt_overrides
from attack.position_opt.trainer import _resolve_reward_target_utility
from attack.position_opt.types import position_opt_identity_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
DELTA_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_delta_reward.yaml"
)


def test_reward_mode_loads_from_yaml_and_cli_override() -> None:
    config = load_config(DELTA_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.reward_mode == "delta_target_utility"
    assert _resolve_position_opt_overrides(Namespace(reward_mode="poisoned_target_utility"))[
        "reward_mode"
    ] == "poisoned_target_utility"


def test_reward_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="reward_mode"):
        PositionOptConfig(reward_mode="poison-clean")


def test_delta_reward_utility_subtracts_clean_baseline() -> None:
    assert _resolve_reward_target_utility(
        reward_mode="poisoned_target_utility",
        poisoned_target_utility=0.4,
        clean_target_utility=0.1,
    ) == pytest.approx(0.4)
    assert _resolve_reward_target_utility(
        reward_mode="delta_target_utility",
        poisoned_target_utility=0.4,
        clean_target_utility=0.1,
    ) == pytest.approx(0.3)


def test_delta_reward_mode_requires_clean_baseline() -> None:
    with pytest.raises(ValueError, match="clean_target_utility"):
        _resolve_reward_target_utility(
            reward_mode="delta_target_utility",
            poisoned_target_utility=0.4,
            clean_target_utility=None,
        )


def test_reward_mode_splits_position_opt_and_victim_cache_but_not_generation_cache() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_delta_reward" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")

        baseline_config, baseline_context = _config_with_reward_mode(
            config,
            reward_mode="poisoned_target_utility",
            checkpoint_path=checkpoint_path,
        )
        delta_config, delta_context = _config_with_reward_mode(
            config,
            reward_mode="delta_target_utility",
            checkpoint_path=checkpoint_path,
        )

        assert attack_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != attack_key(
            delta_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=delta_context,
        )
        assert run_group_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != run_group_key(
            delta_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=delta_context,
        )
        assert victim_prediction_key(
            baseline_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != victim_prediction_key(
            delta_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=delta_context,
        )
        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            delta_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _config_with_reward_mode(config, *, reward_mode: str, checkpoint_path: Path):
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")
    position_opt = replace(
        config.attack.position_opt,
        clean_surrogate_checkpoint=str(checkpoint_path),
        reward_mode=str(reward_mode),
    )
    updated = replace(
        config,
        attack=replace(config.attack, position_opt=position_opt),
    )
    context = build_position_opt_attack_identity_context(
        position_opt_config=position_opt_identity_payload(position_opt),
        clean_surrogate_checkpoint=checkpoint_path,
        runtime_seeds={
            "position_opt_seed": int(updated.seeds.position_opt_seed),
            "surrogate_train_seed": int(updated.seeds.surrogate_train_seed),
        },
    )
    return updated, context
