from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
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
from attack.position_opt.feature_builder import (
    build_policy_special_item_ids,
    build_session_candidate_features,
)
from attack.position_opt.policy import SharedContextualPositionPolicy
from attack.position_opt.trainer import PositionOptMVPTrainer
from attack.position_opt.types import SurrogateScoreResult, position_opt_identity_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
PREFIX_SCORE_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_prefix_score.yaml"
)


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


def test_shared_policy_prefix_feature_set_expands_scalar_input_dim() -> None:
    baseline_policy = SharedContextualPositionPolicy(
        num_item_embeddings=32,
        embedding_dim=8,
        hidden_dim=16,
        policy_feature_set="local_context",
    )
    prefix_policy = SharedContextualPositionPolicy(
        num_item_embeddings=32,
        embedding_dim=8,
        hidden_dim=16,
        policy_feature_set="local_context_prefix_score_prob",
    )

    assert list(baseline_policy.scalar_feature_names) == [
        "position_index",
        "normalized_position",
        "session_length",
    ]
    assert list(prefix_policy.scalar_feature_names) == [
        "position_index",
        "normalized_position",
        "session_length",
        "prefix_score",
        "has_prefix",
    ]
    assert baseline_policy.scorer[0].in_features == (8 * 4) + 3
    assert prefix_policy.scorer[0].in_features == (8 * 4) + 5


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

    trainer.train(
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


def test_policy_feature_set_splits_position_opt_and_victim_cache_but_not_generation_cache() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_prefix_score_features" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")

        baseline_config, baseline_context = _config_with_policy_feature_set(
            config,
            policy_feature_set="local_context",
            checkpoint_path=checkpoint_path,
        )
        prefix_config, prefix_context = _config_with_policy_feature_set(
            config,
            policy_feature_set="local_context_prefix_score_prob",
            checkpoint_path=checkpoint_path,
        )

        assert attack_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != attack_key(
            prefix_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=prefix_context,
        )
        assert run_group_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != run_group_key(
            prefix_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=prefix_context,
        )
        assert victim_prediction_key(
            baseline_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != victim_prediction_key(
            prefix_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=prefix_context,
        )
        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            prefix_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
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
