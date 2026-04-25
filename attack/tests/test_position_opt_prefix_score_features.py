from __future__ import annotations

import json
import shutil
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
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
from attack.common.position_opt_policy_feature_sets import (
    POSITION_OPT_POLICY_FEATURE_SET_SPECS,
)
from attack.position_opt.feature_builder import (
    build_policy_special_item_ids,
    build_session_candidate_features,
)
from attack.position_opt.policy import SharedContextualPositionPolicy
from attack.position_opt.trainer import PositionOptMVPTrainer
from attack.position_opt.types import (
    PositionOptArtifactPaths,
    SurrogateScoreResult,
    position_opt_identity_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
PREFIX_SCORE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_prefix_score.yaml"
)
TARGET_ORIGINAL_POSITION_SCALAR_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_target_original_position_scalar_tron_focus.yaml"
)
_POLICY_EMBEDDING_DIM = 8
_FEATURE_SET_EXPECTATIONS = {
    "local_context": {
        "item_features": [
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ],
        "scalar_features": [
            "position_index",
            "normalized_position",
            "session_length",
        ],
    },
    "local_context_prefix_score_prob": {
        "item_features": [
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ],
        "scalar_features": [
            "position_index",
            "normalized_position",
            "session_length",
            "prefix_score",
            "has_prefix",
        ],
    },
    "normalized_position_only": {
        "item_features": [],
        "scalar_features": ["normalized_position"],
    },
    "target_normalized_position": {
        "item_features": ["target_item"],
        "scalar_features": ["normalized_position"],
    },
    "target_original_normalized_position": {
        "item_features": ["target_item", "original_item"],
        "scalar_features": ["normalized_position"],
    },
    "target_original_position_scalar": {
        "item_features": ["target_item", "original_item"],
        "scalar_features": [
            "position_index",
            "normalized_position",
            "session_length",
        ],
    },
    "full_context_normalized_position": {
        "item_features": [
            "target_item",
            "original_item",
            "left_item",
            "right_item",
        ],
        "scalar_features": ["normalized_position"],
    },
}


def test_policy_feature_set_specs_cover_all_supported_modes() -> None:
    assert set(POSITION_OPT_POLICY_FEATURE_SET_SPECS) == set(_FEATURE_SET_EXPECTATIONS)


def test_prefix_score_config_loads_from_yaml_and_identity_omits_default() -> None:
    config = load_config(PREFIX_SCORE_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.policy_feature_set == "local_context_prefix_score_prob"
    assert position_opt_identity_payload(PositionOptConfig()) == {
        "outer_steps": 30,
        "policy_lr": 0.05,
        "policy_embedding_dim": 16,
        "policy_hidden_dim": 32,
        "fine_tune_steps": 20,
        "validation_subset_size": None,
        "reward_baseline_momentum": 0.9,
        "reward_mode": "poisoned_target_utility",
        "entropy_coef": 0.0,
        "enable_gt_penalty": False,
        "gt_penalty_weight": 0.0,
        "gt_tolerance": 0.0,
        "final_selection": "argmax",
    }


def test_target_original_position_scalar_focus_config_loads_from_yaml() -> None:
    config = load_config(TARGET_ORIGINAL_POSITION_SCALAR_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert (
        config.attack.position_opt.policy_feature_set
        == "target_original_position_scalar"
    )
    assert list(config.victims.enabled) == ["tron"]
    assert config.targets.mode == "explicit_list"
    assert list(config.targets.explicit_list) == [5334, 11103]


@pytest.mark.parametrize("policy_feature_set", sorted(_FEATURE_SET_EXPECTATIONS))
def test_policy_feature_set_config_accepts_supported_modes(policy_feature_set: str) -> None:
    config = PositionOptConfig(policy_feature_set=policy_feature_set)

    assert config.policy_feature_set == policy_feature_set


def test_policy_feature_set_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="policy_feature_set"):
        PositionOptConfig(policy_feature_set="normalized_position_with_magic")


def test_build_session_candidate_features_records_prefix_score_and_has_prefix() -> None:
    features = build_session_candidate_features(
        [4, 5, 6],
        [0, 1, 2],
        target_item=9,
        special_item_ids=build_policy_special_item_ids(20),
        prefix_scores=[0.0, 0.125, 0.25],
        has_prefixes=[False, True, True],
    )

    assert features.metadata[0].position == 0
    assert features.metadata[0].prefix_score == pytest.approx(0.0)
    assert features.metadata[0].has_prefix is False
    assert features.metadata[1].prefix_score == pytest.approx(0.125)
    assert features.metadata[1].has_prefix is True
    assert features.metadata[2].prefix_score == pytest.approx(0.25)
    assert features.metadata[2].has_prefix is True
    assert features.tensors.prefix_scores.tolist() == pytest.approx([0.0, 0.125, 0.25])
    assert features.tensors.has_prefixes.tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_target_original_position_scalar_spec_disables_prefix_features() -> None:
    spec = POSITION_OPT_POLICY_FEATURE_SET_SPECS["target_original_position_scalar"]

    assert list(spec.item_features) == ["target_item", "original_item"]
    assert list(spec.scalar_features) == [
        "position_index",
        "normalized_position",
        "session_length",
    ]
    assert spec.requires_prefix_features is False


@pytest.mark.parametrize(
    "policy_feature_set,expected",
    sorted(_FEATURE_SET_EXPECTATIONS.items()),
)
def test_policy_feature_set_controls_item_and_scalar_inputs(
    policy_feature_set: str,
    expected: dict[str, list[str]],
) -> None:
    policy = SharedContextualPositionPolicy(
        num_item_embeddings=32,
        embedding_dim=_POLICY_EMBEDDING_DIM,
        hidden_dim=16,
        policy_feature_set=policy_feature_set,
    )

    expected_input_dim = (
        len(expected["item_features"]) * _POLICY_EMBEDDING_DIM
    ) + len(expected["scalar_features"])

    assert list(policy.active_item_features) == expected["item_features"]
    assert list(policy.active_scalar_features) == expected["scalar_features"]
    assert list(policy.item_feature_names) == expected["item_features"]
    assert list(policy.scalar_feature_names) == expected["scalar_features"]
    assert policy.policy_input_dim == expected_input_dim
    assert policy.scorer[0].in_features == expected_input_dim
    assert (policy.item_embedding is None) is (len(expected["item_features"]) == 0)
    assert policy.input_metadata() == {
        "policy_feature_set": policy_feature_set,
        "active_item_features": expected["item_features"],
        "active_scalar_features": expected["scalar_features"],
        "policy_input_dim": expected_input_dim,
        "policy_embedding_dim": _POLICY_EMBEDDING_DIM,
        "policy_hidden_dim": 16,
    }


@pytest.mark.parametrize("policy_feature_set", sorted(_FEATURE_SET_EXPECTATIONS))
def test_policy_feature_set_forward_pass_smoke(policy_feature_set: str) -> None:
    features = build_session_candidate_features(
        [4, 5, 6],
        [0, 1, 2],
        target_item=9,
        special_item_ids=build_policy_special_item_ids(20),
        prefix_scores=[0.0, 0.125, 0.25],
        has_prefixes=[False, True, True],
    )
    policy = SharedContextualPositionPolicy(
        num_item_embeddings=32,
        embedding_dim=_POLICY_EMBEDDING_DIM,
        hidden_dim=16,
        policy_feature_set=policy_feature_set,
    )

    logits = policy.score_candidates(features.tensors)

    assert tuple(logits.shape) == (3,)
    assert torch.isfinite(logits).all()


def test_trainer_prefix_features_cache_probabilities_and_skip_empty_prefixes() -> None:
    base_config = load_config(BASE_CONFIG_PATH)
    if base_config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")

    trainer = PositionOptMVPTrainer(
        _RecordingPrefixSurrogateBackend(),
        object(),
        clean_surrogate_checkpoint_path="unused-clean-surrogate.pt",
        position_opt_config=replace(
            base_config.attack.position_opt,
            outer_steps=0,
            policy_feature_set="local_context_prefix_score_prob",
        ),
    )

    trainer_result = trainer.train(
        fake_sessions=[[7, 8, 9], [4]],
        target_item=12,
        shared_artifacts=SimpleNamespace(
            clean_sessions=[[1, 2]],
            clean_labels=[3],
            validation_sessions=[[5, 6]],
            validation_labels=[7],
        ),
        config=base_config,
    )

    first_session_rows = trainer._session_states[0].features.metadata
    second_session_rows = trainer._session_states[1].features.metadata
    backend = trainer.surrogate_backend

    assert [row.prefix_score for row in first_session_rows] == pytest.approx([0.0, 0.1, 0.2])
    assert [row.has_prefix for row in first_session_rows] == [False, True, True]
    assert second_session_rows[0].prefix_score == pytest.approx(0.0)
    assert second_session_rows[0].has_prefix is False
    assert backend.load_clean_checkpoint_calls == 2
    assert backend.clone_clean_model_calls == 2
    assert not any(
        len(session) == 0
        for call in backend.score_target_calls
        for session in call["sessions"]
    )
    assert trainer_result["prefix_score_enabled"] is True
    assert trainer_result["policy_item_feature_names"] == [
        "target_item",
        "original_item",
        "left_item",
        "right_item",
    ]
    assert trainer_result["active_scalar_features"] == [
        "position_index",
        "normalized_position",
        "session_length",
        "prefix_score",
        "has_prefix",
    ]


@pytest.mark.parametrize(
    "policy_feature_set,expected_item_features,expected_scalar_features,expected_input_dim",
    [
        ("normalized_position_only", [], ["normalized_position"], 1),
        (
            "target_original_position_scalar",
            ["target_item", "original_item"],
            ["position_index", "normalized_position", "session_length"],
            (2 * _POLICY_EMBEDDING_DIM) + 3,
        ),
    ],
)
def test_trainer_skips_prefix_score_enrichment_when_feature_set_does_not_need_it(
    policy_feature_set: str,
    expected_item_features: list[str],
    expected_scalar_features: list[str],
    expected_input_dim: int,
) -> None:
    base_config = load_config(BASE_CONFIG_PATH)
    if base_config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")

    trainer = PositionOptMVPTrainer(
        _RecordingPrefixSurrogateBackend(),
        object(),
        clean_surrogate_checkpoint_path="unused-clean-surrogate.pt",
        position_opt_config=replace(
            base_config.attack.position_opt,
            outer_steps=0,
            policy_feature_set=policy_feature_set,
            policy_embedding_dim=_POLICY_EMBEDDING_DIM,
        ),
    )

    trainer_result = trainer.train(
        fake_sessions=[[7, 8, 9], [4]],
        target_item=12,
        shared_artifacts=SimpleNamespace(
            clean_sessions=[[1, 2]],
            clean_labels=[3],
            validation_sessions=[[5, 6]],
            validation_labels=[7],
        ),
        config=base_config,
    )

    backend = trainer.surrogate_backend

    assert backend.load_clean_checkpoint_calls == 1
    assert backend.clone_clean_model_calls == 1
    assert len(backend.score_target_calls) == 1
    assert backend.score_target_calls[0]["sessions"] == [[5, 6]]
    assert trainer_result["prefix_score_enabled"] is False
    assert trainer_result["active_item_features"] == expected_item_features
    assert trainer_result["active_scalar_features"] == expected_scalar_features
    assert trainer_result["policy_input_dim"] == expected_input_dim


def test_trainer_artifacts_log_active_features_and_policy_dims() -> None:
    config = load_config(BASE_CONFIG_PATH)
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")

    temp_root = REPO_ROOT / "outputs" / ".pytest_policy_feature_metadata" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        trainer = PositionOptMVPTrainer(
            _RecordingPrefixSurrogateBackend(),
            object(),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=replace(
                config.attack.position_opt,
                outer_steps=0,
                policy_feature_set="target_original_normalized_position",
                policy_embedding_dim=7,
                policy_hidden_dim=11,
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[[7, 8, 9], [4, 5]],
            target_item=12,
            shared_artifacts=SimpleNamespace(
                clean_sessions=[[1, 2]],
                clean_labels=[3],
                validation_sessions=[[5, 6]],
                validation_labels=[7],
            ),
            config=config,
        )
        artifact_paths = PositionOptArtifactPaths(
            base_dir=temp_root / "artifacts",
            clean_surrogate_checkpoint=temp_root / "artifacts" / "clean_surrogate.pt",
            optimized_poisoned_sessions=temp_root / "artifacts" / "optimized_poisoned_sessions.pkl",
            training_history=temp_root / "artifacts" / "training_history.json",
            learned_logits=temp_root / "artifacts" / "learned_logits.pt",
        )
        trainer.save_artifacts(artifact_paths)

        training_history_payload = json.loads(
            artifact_paths.training_history.read_text(encoding="utf-8")
        )
        learned_logits_payload = torch.load(artifact_paths.learned_logits, map_location="cpu")

        assert trainer_result["policy_item_feature_names"] == [
            "target_item",
            "original_item",
        ]
        assert trainer_result["active_scalar_features"] == ["normalized_position"]
        assert trainer_result["policy_input_dim"] == 15
        assert training_history_payload["policy_item_feature_names"] == [
            "target_item",
            "original_item",
        ]
        assert training_history_payload["active_scalar_features"] == ["normalized_position"]
        assert training_history_payload["policy_input_dim"] == 15
        assert training_history_payload["policy_embedding_dim"] == 7
        assert training_history_payload["policy_hidden_dim"] == 11
        assert learned_logits_payload["policy_config"]["active_item_features"] == [
            "target_item",
            "original_item",
        ]
        assert learned_logits_payload["policy_config"]["active_scalar_features"] == [
            "normalized_position"
        ]
        assert learned_logits_payload["policy_config"]["policy_input_dim"] == 15
        assert learned_logits_payload["policy_config"]["policy_embedding_dim"] == 7
        assert learned_logits_payload["policy_config"]["policy_hidden_dim"] == 11
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_policy_feature_set_splits_position_opt_and_victim_cache_but_not_generation_cache() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_prefix_score_features" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")
        resolved = {
            policy_feature_set: _config_with_policy_feature_set(
                config,
                policy_feature_set=policy_feature_set,
                checkpoint_path=checkpoint_path,
            )
            for policy_feature_set in _FEATURE_SET_EXPECTATIONS
        }

        for left_feature_set, right_feature_set in combinations(
            sorted(_FEATURE_SET_EXPECTATIONS),
            2,
        ):
            left_config, left_context = resolved[left_feature_set]
            right_config, right_context = resolved[right_feature_set]
            assert attack_key(
                left_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=left_context,
            ) != attack_key(
                right_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=right_context,
            )
            assert run_group_key(
                left_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=left_context,
            ) != run_group_key(
                right_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=right_context,
            )
            assert victim_prediction_key(
                left_config,
                "srgnn",
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=left_context,
            ) != victim_prediction_key(
                right_config,
                "srgnn",
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                attack_identity_context=right_context,
            )

        baseline_config, _ = resolved["local_context"]
        for policy_feature_set, (feature_config, _) in resolved.items():
            assert shared_attack_artifact_key(
                baseline_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            ) == shared_attack_artifact_key(
                feature_config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            ), policy_feature_set
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _config_with_policy_feature_set(config, *, policy_feature_set: str, checkpoint_path: Path):
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")
    position_opt = replace(
        config.attack.position_opt,
        clean_surrogate_checkpoint=str(checkpoint_path),
        policy_feature_set=str(policy_feature_set),
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


class _RecordingPrefixSurrogateBackend:
    def __init__(self) -> None:
        self.load_clean_checkpoint_calls = 0
        self.clone_clean_model_calls = 0
        self.score_target_calls: list[dict[str, object]] = []

    def load_clean_checkpoint(self, path) -> None:
        del path
        self.load_clean_checkpoint_calls += 1

    def clone_clean_model(self) -> str:
        self.clone_clean_model_calls += 1
        return "clean-model"

    def fine_tune(self, model, poisoned_train_data, **kwargs):
        del model, poisoned_train_data, kwargs
        raise AssertionError("fine_tune() should not be called when outer_steps=0.")

    def score_target(self, model, eval_sessions, target_item) -> SurrogateScoreResult:
        normalized_sessions = [list(session) for session in eval_sessions]
        self.score_target_calls.append(
            {
                "model": model,
                "sessions": normalized_sessions,
                "target_item": int(target_item),
            }
        )
        values = [float(len(session)) / 10.0 for session in normalized_sessions]
        return SurrogateScoreResult.from_values(
            values,
            metrics={
                "targeted_mrr@10": 0.0,
                "targeted_recall@10": 0.0,
                "targeted_recall@20": 0.0,
            },
        )

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        del model, eval_sessions, ground_truth_items
        raise AssertionError("score_gt() should not be called when GT penalty is disabled.")
