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
    TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    attack_key,
    poison_model_key,
    shared_artifact_paths,
    shared_attack_artifact_key,
)
from attack.pipeline.runs.run_target_aware_carrier_selection_nz import (
    _validate_tacs_nz_config,
)
from attack.pipeline.runs.run_target_aware_carrier_local_position import (
    _validate_tacs_local_position_config,
)
from attack.pipeline.runs.run_target_aware_coverage_local_position import (
    _validate_coverage_local_position_config,
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
TACS_LOCAL_POSITION_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tacs_local_position_pool3_final1_srgnn_target11103.yaml"
)
TACS_LOCAL_POSITION_PARTIAL4_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_tacs_local_position_pool3_final1_srgnn_target11103_partial4.yaml"
)
COVERAGE_LOCAL_POSITION_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_coverage_local_position_pool3_final1_srgnn_target11103.yaml"
)
COVERAGE_LOCAL_POSITION_PARTIAL4_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_coverage_local_position_pool3_final1_srgnn_target11103_partial4.yaml"
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


def test_tacs_local_position_configs_parse() -> None:
    full = load_config(TACS_LOCAL_POSITION_CONFIG_PATH)
    partial4 = load_config(TACS_LOCAL_POSITION_PARTIAL4_CONFIG_PATH)

    assert full.attack.carrier_selection is not None
    assert full.attack.carrier_selection.scorer == "hybrid_local_position_compatibility"
    assert full.attack.carrier_selection.placement_mode == "best_local_position"
    assert full.attack.carrier_selection.operation == "replacement"
    assert full.attack.carrier_selection.candidate_positions == "nonzero"
    assert full.attack.carrier_selection.left_to_target_weight == pytest.approx(1.0)
    assert full.attack.carrier_selection.target_to_right_weight == pytest.approx(0.0)
    assert full.attack.carrier_selection.debug_save_all_session_records is False
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert partial4.victims.params["srgnn"]["train"]["epochs"] == 4


def test_coverage_local_position_configs_parse() -> None:
    full = load_config(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    partial4 = load_config(COVERAGE_LOCAL_POSITION_PARTIAL4_CONFIG_PATH)

    assert full.attack.carrier_selection is not None
    carrier = full.attack.carrier_selection
    assert carrier.scorer == "coverage_aware_local_position"
    assert carrier.placement_mode == "best_local_position"
    assert carrier.operation == "replacement"
    assert carrier.candidate_positions == "nonzero"
    assert carrier.coverage_prefix_source == "validation"
    assert carrier.vulnerable_rank_min == 20
    assert carrier.vulnerable_rank_max == 200
    assert carrier.max_vulnerable_prefixes == 5000
    assert carrier.prefix_representation == "mean_item_embedding"
    assert carrier.candidate_representation == "targetized_prefix_mean_embedding"
    assert carrier.top_m_coverage == 20
    assert carrier.rank_weighting == "inverse_log_rank"
    assert carrier.coverage_similarity == "cosine"
    assert carrier.debug_save_all_position_records is False
    assert full.victims.params["srgnn"]["train"]["epochs"] == 30
    assert full.victims.params["srgnn"]["train"]["patience"] == 10
    assert (
        full.victims.params["srgnn"]["train"]["checkpoint_protocol"]
        == "validation_best"
    )
    assert partial4.experiment.name.endswith("partial4")
    assert partial4.victims.params["srgnn"]["train"]["epochs"] == 4


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


def test_tacs_local_position_runner_rejects_missing_or_disabled_carrier_selection() -> None:
    config = load_config(TACS_LOCAL_POSITION_CONFIG_PATH)
    missing = replace(config, attack=replace(config.attack, carrier_selection=None))
    disabled = replace(
        config,
        attack=replace(
            config.attack,
            carrier_selection=replace(config.attack.carrier_selection, enabled=False),
        ),
    )

    with pytest.raises(ValueError, match="requires attack.carrier_selection"):
        _validate_tacs_local_position_config(missing)
    with pytest.raises(ValueError, match="enabled == true"):
        _validate_tacs_local_position_config(disabled)


def test_invalid_local_position_runner_config_raises_clear_errors() -> None:
    config = load_config(TACS_LOCAL_POSITION_CONFIG_PATH)
    carrier = config.attack.carrier_selection
    if carrier is None:
        raise AssertionError("Expected carrier selection config.")

    with pytest.raises(ValueError, match="scorer"):
        _validate_tacs_local_position_config(
            replace(
                config,
                attack=replace(
                    config.attack,
                    carrier_selection=replace(
                        carrier,
                        scorer="hybrid_target_session_compatibility",
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="placement_mode"):
        _validate_tacs_local_position_config(
            replace(
                config,
                attack=replace(config.attack, carrier_selection=replace(carrier, placement_mode=None)),
            )
        )
    with pytest.raises(ValueError, match="operation"):
        _validate_tacs_local_position_config(
            replace(
                config,
                attack=replace(config.attack, carrier_selection=replace(carrier, operation=None)),
            )
        )
    with pytest.raises(ValueError, match="candidate_positions"):
        _validate_tacs_local_position_config(
            replace(
                config,
                attack=replace(
                    config.attack,
                    carrier_selection=replace(carrier, candidate_positions=None),
                ),
            )
        )


def test_invalid_coverage_local_position_config_raises_clear_errors() -> None:
    config = load_config(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    carrier = config.attack.carrier_selection
    if carrier is None:
        raise AssertionError("Expected carrier selection config.")

    with pytest.raises(ValueError, match="scorer"):
        _validate_coverage_local_position_config(
            replace(
                config,
                attack=replace(
                    config.attack,
                    carrier_selection=replace(
                        carrier,
                        scorer="hybrid_local_position_compatibility",
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="placement_mode"):
        _validate_coverage_local_position_config(
            replace(
                config,
                attack=replace(config.attack, carrier_selection=replace(carrier, placement_mode=None)),
            )
        )
    with pytest.raises(ValueError, match="operation"):
        _validate_coverage_local_position_config(
            replace(
                config,
                attack=replace(config.attack, carrier_selection=replace(carrier, operation=None)),
            )
        )
    with pytest.raises(ValueError, match="candidate_positions"):
        _validate_coverage_local_position_config(
            replace(
                config,
                attack=replace(
                    config.attack,
                    carrier_selection=replace(carrier, candidate_positions=None),
                ),
            )
        )


def test_invalid_coverage_config_fields_raise_clear_errors() -> None:
    payload = _load_yaml(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    carrier_selection = payload["attack"]["carrier_selection"]

    carrier_selection["vulnerable_rank_min"] = 20
    carrier_selection["vulnerable_rank_max"] = 20
    with pytest.raises(ValueError, match="vulnerable_rank_max"):
        validate_config_mapping(payload)

    payload = _load_yaml(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    payload["attack"]["carrier_selection"]["top_m_coverage"] = 0
    with pytest.raises(ValueError, match="top_m_coverage"):
        validate_config_mapping(payload)

    payload = _load_yaml(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    payload["attack"]["carrier_selection"]["rank_weighting"] = "unsupported"
    with pytest.raises(ValueError, match="rank_weighting"):
        validate_config_mapping(payload)


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


def test_tacs_nz_and_local_position_share_candidate_pool_but_not_final_attack_key() -> None:
    tacs_nz = load_config(TACS_CONFIG_PATH)
    local = load_config(TACS_LOCAL_POSITION_CONFIG_PATH)
    local_carrier = local.attack.carrier_selection
    if local_carrier is None:
        raise AssertionError("Expected carrier selection config.")

    changed_local_weight = replace(
        local,
        attack=replace(
            local.attack,
            carrier_selection=replace(local_carrier, local_embedding_weight=0.7, local_transition_weight=0.3),
        ),
    )

    assert shared_attack_artifact_key(
        tacs_nz,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) == shared_attack_artifact_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    )
    assert shared_attack_artifact_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        changed_local_weight,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    )
    assert attack_key(
        tacs_nz,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) != attack_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    )
    assert attack_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    ) != attack_key(
        changed_local_weight,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    )


def test_coverage_local_position_shares_candidate_pool_but_has_distinct_final_key() -> None:
    tacs_nz = load_config(TACS_CONFIG_PATH)
    local = load_config(TACS_LOCAL_POSITION_CONFIG_PATH)
    coverage = load_config(COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    coverage_carrier = coverage.attack.carrier_selection
    if coverage_carrier is None:
        raise AssertionError("Expected carrier selection config.")

    changed_rank_range = replace(
        coverage,
        attack=replace(
            coverage.attack,
            carrier_selection=replace(
                coverage_carrier,
                vulnerable_rank_min=30,
                vulnerable_rank_max=220,
            ),
        ),
    )
    changed_top_m = replace(
        coverage,
        attack=replace(
            coverage.attack,
            carrier_selection=replace(coverage_carrier, top_m_coverage=10),
        ),
    )

    assert shared_attack_artifact_key(
        tacs_nz,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    ) == shared_attack_artifact_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    )
    assert shared_attack_artifact_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    )
    assert shared_attack_artifact_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    ) == shared_attack_artifact_key(
        changed_rank_range,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    )
    assert attack_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    ) != attack_key(
        tacs_nz,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    )
    assert attack_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    ) != attack_key(
        local,
        run_type=TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    )
    assert attack_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    ) != attack_key(
        changed_rank_range,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    )
    assert attack_key(
        coverage,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    ) != attack_key(
        changed_top_m,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
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
