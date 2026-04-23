from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import shutil
from uuid import uuid4

import pytest
import torch

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
from attack.position_opt.trainer import _build_policy_loss
from attack.position_opt.types import position_opt_identity_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
ENTROPY_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_entropy_regularization.yaml"
)


def test_entropy_coef_loads_from_yaml_and_cli_override() -> None:
    config = load_config(ENTROPY_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.entropy_coef == pytest.approx(3.0e-5)
    assert _resolve_position_opt_overrides(Namespace(entropy_coef=3.0e-5))[
        "entropy_coef"
    ] == pytest.approx(3.0e-5)


def test_entropy_coef_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="entropy_coef"):
        PositionOptConfig(entropy_coef=-1.0e-5)


def test_policy_loss_uses_sum_entropy_bonus() -> None:
    joint_log_prob = torch.tensor(-10.0)
    advantage = torch.tensor(0.5)
    joint_entropy = torch.tensor(4.0)

    total, reinforce_loss, entropy_loss = _build_policy_loss(
        joint_log_prob=joint_log_prob,
        advantage_tensor=advantage,
        joint_entropy=joint_entropy,
        entropy_coef=3.0e-5,
    )

    assert reinforce_loss.item() == pytest.approx(5.0)
    assert entropy_loss.item() == pytest.approx(-0.00012)
    assert total.item() == pytest.approx(4.99988)


def test_zero_entropy_coef_preserves_original_reinforce_loss() -> None:
    total, reinforce_loss, entropy_loss = _build_policy_loss(
        joint_log_prob=torch.tensor(-10.0),
        advantage_tensor=torch.tensor(0.5),
        joint_entropy=torch.tensor(4.0),
        entropy_coef=0.0,
    )

    assert entropy_loss.item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(reinforce_loss.item())


def test_entropy_coef_splits_position_opt_and_victim_cache_but_not_generation_cache() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_entropy_regularization" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")

        baseline_config, baseline_context = _config_with_entropy(
            config,
            entropy_coef=0.0,
            checkpoint_path=checkpoint_path,
        )
        regularized_config, regularized_context = _config_with_entropy(
            config,
            entropy_coef=3.0e-5,
            checkpoint_path=checkpoint_path,
        )

        assert attack_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != attack_key(
            regularized_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=regularized_context,
        )
        assert run_group_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != run_group_key(
            regularized_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=regularized_context,
        )
        assert victim_prediction_key(
            baseline_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != victim_prediction_key(
            regularized_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=regularized_context,
        )
        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            regularized_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _config_with_entropy(config, *, entropy_coef: float, checkpoint_path: Path):
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")
    position_opt = replace(
        config.attack.position_opt,
        clean_surrogate_checkpoint=str(checkpoint_path),
        entropy_coef=float(entropy_coef),
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
