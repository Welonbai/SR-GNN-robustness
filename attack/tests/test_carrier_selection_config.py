from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config, validate_config_mapping
from attack.common.paths import (
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    attack_key,
    poison_model_key,
    shared_artifact_paths,
    shared_attack_artifact_key,
)
from attack.pipeline.runs.run_target_aware_carrier_selection_nz import (
    _validate_tacs_nz_config,
)


TACS_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tacs_nz_pool3_final1_srgnn_target11103.yaml"
)
RANDOM_NZ_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_random_nonzero_when_possible_ratio1_srgnn_sample3.yaml"
)


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError("Expected YAML root to be a mapping.")
    return payload


def test_tacs_carrier_selection_config_parses() -> None:
    config = load_config(TACS_CONFIG_PATH)

    assert config.attack.carrier_selection is not None
    assert config.attack.carrier_selection.enabled is True
    assert config.attack.carrier_selection.candidate_pool_size == pytest.approx(0.03)
    assert config.attack.carrier_selection.final_attack_size == pytest.approx(config.attack.size)


def test_existing_random_nz_config_still_parses_without_carrier_selection() -> None:
    config = load_config(RANDOM_NZ_CONFIG_PATH)

    assert config.attack.carrier_selection is None


def test_candidate_pool_size_equal_final_attack_size_is_valid() -> None:
    payload = _load_yaml(TACS_CONFIG_PATH)
    carrier_selection = payload["attack"]["carrier_selection"]
    carrier_selection["candidate_pool_size"] = 0.01
    carrier_selection["final_attack_size"] = 0.01

    validate_config_mapping(payload)


def test_invalid_carrier_selection_ratios_raise_clear_errors() -> None:
    payload = _load_yaml(TACS_CONFIG_PATH)
    carrier_selection = payload["attack"]["carrier_selection"]
    carrier_selection["candidate_pool_size"] = 0.005
    carrier_selection["final_attack_size"] = 0.01
    with pytest.raises(ValueError, match="final_attack_size must be in"):
        validate_config_mapping(payload)

    payload = _load_yaml(TACS_CONFIG_PATH)
    payload["attack"]["carrier_selection"]["final_attack_size"] = 0.02
    with pytest.raises(ValueError, match="final_attack_size must equal attack.size"):
        validate_config_mapping(payload)


def test_tacs_runner_rejects_missing_or_disabled_carrier_selection() -> None:
    config = load_config(TACS_CONFIG_PATH)
    missing = replace(config, attack=replace(config.attack, carrier_selection=None))
    disabled = replace(
        config,
        attack=replace(
            config.attack,
            carrier_selection=replace(config.attack.carrier_selection, enabled=False),
        ),
    )

    with pytest.raises(ValueError, match="requires attack.carrier_selection"):
        _validate_tacs_nz_config(missing)
    with pytest.raises(ValueError, match="enabled == true"):
        _validate_tacs_nz_config(disabled)


def test_tacs_runner_requires_random_nz_ratio1_placement() -> None:
    config = load_config(TACS_CONFIG_PATH)
    changed = replace(
        config,
        attack=replace(config.attack, replacement_topk_ratio=0.5),
    )

    with pytest.raises(ValueError, match="isolates carrier selection under Random-NZ ratio1"):
        _validate_tacs_nz_config(changed)


def test_tacs_shared_identity_uses_pool_size_not_scorer_weights() -> None:
    config = load_config(TACS_CONFIG_PATH)
    carrier = config.attack.carrier_selection
    if carrier is None:
        raise AssertionError("Expected carrier selection config.")

    changed_weights = replace(
        config,
        attack=replace(
            config.attack,
            carrier_selection=replace(
                carrier,
                embedding_weight=0.9,
                cooccurrence_weight=0.1,
                transition_weight=0.0,
            ),
        ),
    )
    changed_pool = replace(
        config,
        attack=replace(
            config.attack,
            carrier_selection=replace(carrier, candidate_pool_size=0.04),
        ),
    )

    assert shared_attack_artifact_key(
        config,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) == shared_attack_artifact_key(
        changed_weights,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) != shared_attack_artifact_key(
        changed_pool,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    )
    assert attack_key(
        config,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) != attack_key(
        changed_weights,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    )


def test_tacs_and_random_nz_share_poison_model_but_not_fake_sessions() -> None:
    tacs_config = load_config(TACS_CONFIG_PATH)
    random_config = load_config(RANDOM_NZ_CONFIG_PATH)

    tacs_paths = shared_artifact_paths(
        tacs_config,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    )
    random_paths = shared_artifact_paths(
        random_config,
        run_type="random_nonzero_when_possible",
    )

    assert poison_model_key(tacs_config) == poison_model_key(random_config)
    assert tacs_paths["poison_model"] == random_paths["poison_model"]
    assert tacs_paths["fake_sessions"] != random_paths["fake_sessions"]
