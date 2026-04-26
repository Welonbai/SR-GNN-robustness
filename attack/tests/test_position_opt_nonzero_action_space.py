from __future__ import annotations

import json
import shutil
from dataclasses import replace
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
from attack.position_opt.candidate_builder import (
    build_candidate_positions,
    filter_candidate_positions_nonzero_when_possible,
)
from attack.position_opt.trainer import PositionOptMVPTrainer
from attack.position_opt.types import (
    InnerTrainResult,
    PositionOptArtifactPaths,
    SurrogateScoreResult,
    position_opt_identity_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
NONZERO_SAMPLE3_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_nonzero.yaml"
)


def test_nonzero_action_space_config_defaults_and_yaml_loads() -> None:
    defaults = PositionOptConfig()

    assert defaults.nonzero_action_when_possible is False
    assert "nonzero_action_when_possible" not in position_opt_identity_payload(defaults)

    config = load_config(NONZERO_SAMPLE3_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.nonzero_action_when_possible is True
    assert config.attack.position_opt.policy_feature_set == "local_context"
    assert config.attack.position_opt.reward_mode == "poisoned_target_utility"
    assert config.attack.position_opt.entropy_coef == pytest.approx(0.0)
    assert config.attack.position_opt.deterministic_eval_every == 0
    assert config.attack.position_opt.deterministic_eval_include_final is True
    assert config.attack.position_opt.final_policy_selection == "last"
    assert config.targets.mode == "sampled"
    assert config.targets.count == 3
    assert list(config.victims.enabled) == ["srgnn", "miasrec", "tron"]


@pytest.mark.parametrize(
    ("base_candidates", "expected_candidates"),
    [
        ([0], [0]),
        ([0, 1], [1]),
        ([0, 1, 2], [1, 2]),
        ([1, 2], [1, 2]),
    ],
)
def test_nonzero_action_space_candidate_filtering(
    base_candidates: list[int],
    expected_candidates: list[int],
) -> None:
    assert list(
        filter_candidate_positions_nonzero_when_possible(
            base_candidates,
            enabled=True,
        )
    ) == expected_candidates
    assert list(
        filter_candidate_positions_nonzero_when_possible(
            base_candidates,
            enabled=False,
        )
    ) == base_candidates


def test_build_candidate_positions_applies_nonzero_filter_after_topk_logic() -> None:
    assert build_candidate_positions(
        [10],
        1.0,
        nonzero_action_when_possible=True,
    ) == [0]
    assert build_candidate_positions(
        [10, 11],
        1.0,
        nonzero_action_when_possible=True,
    ) == [1]
    assert build_candidate_positions(
        [10, 11, 12],
        1.0,
        nonzero_action_when_possible=True,
    ) == [1, 2]


def test_nonzero_action_space_is_used_consistently_for_sampling_and_argmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_nonzero_action_space" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        sampled_candidate_counts: list[int] = []
        eval_candidate_counts: list[int] = []

        def _fake_sample_position_reinforce(logits: torch.Tensor):
            sampled_candidate_counts.append(int(logits.numel()))
            log_probs = torch.log_softmax(logits, dim=0)
            entropy = -(torch.softmax(logits, dim=0) * log_probs).sum()
            return 0, log_probs[0], entropy

        def _fake_select_position_eval(logits: torch.Tensor):
            eval_candidate_counts.append(int(logits.numel()))
            return 0

        monkeypatch.setattr(
            "attack.position_opt.trainer.sample_position_reinforce",
            _fake_sample_position_reinforce,
        )
        monkeypatch.setattr(
            "attack.position_opt.trainer.select_position_eval",
            _fake_select_position_eval,
        )

        trainer = PositionOptMVPTrainer(
            _ResultLookupSurrogateBackend(
                clean_result=SurrogateScoreResult.from_values([0.0]),
                results_by_model={
                    "sampled-0": SurrogateScoreResult.from_values([0.2]),
                    "det-1": SurrogateScoreResult.from_values([0.1]),
                },
            ),
            _SequenceModelInnerTrainer(["sampled-0", "det-1"]),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=1,
                fine_tune_steps=1,
                policy_feature_set="normalized_position_only",
                nonzero_action_when_possible=True,
                deterministic_eval_every=1,
                deterministic_eval_include_final=True,
                final_policy_selection="last",
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[[7, 8, 9]],
            target_item=12,
            shared_artifacts=SimpleNamespace(
                clean_sessions=[[1, 2]],
                clean_labels=[3],
                validation_sessions=[[5, 6]],
                validation_labels=[7],
            ),
            config=config,
        )

        assert sampled_candidate_counts == [2]
        assert eval_candidate_counts == [2, 2]
        assert trainer.training_history[0]["selected_positions"] == [1]
        assert trainer_result["final_selected_positions"][0]["position"] == 1
        assert trainer.export_final_selected_positions()[0].position == 1
        assert trainer._last_deterministic_checkpoint is not None
        assert trainer._last_deterministic_checkpoint.selected_positions == (1,)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_nonzero_action_space_changes_runtime_identity_but_not_shared_generation_key() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_nonzero_action_identity" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")

        baseline_config, baseline_context = _config_with_nonzero_action_setting(
            config,
            checkpoint_path=checkpoint_path,
            nonzero_action_when_possible=False,
        )
        nonzero_config, nonzero_context = _config_with_nonzero_action_setting(
            config,
            checkpoint_path=checkpoint_path,
            nonzero_action_when_possible=True,
        )

        baseline_payload = position_opt_identity_payload(baseline_config.attack.position_opt)
        nonzero_payload = position_opt_identity_payload(nonzero_config.attack.position_opt)

        assert "nonzero_action_when_possible" not in baseline_payload
        assert nonzero_payload["nonzero_action_when_possible"] is True
        assert attack_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != attack_key(
            nonzero_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=nonzero_context,
        )
        assert run_group_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != run_group_key(
            nonzero_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=nonzero_context,
        )
        assert victim_prediction_key(
            baseline_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=baseline_context,
        ) != victim_prediction_key(
            nonzero_config,
            "srgnn",
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=nonzero_context,
        )
        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            nonzero_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_nonzero_action_space_metadata_contains_candidate_and_final_position_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_nonzero_action_metadata" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "attack.position_opt.trainer.select_position_eval",
            lambda logits: 0,
        )

        trainer = PositionOptMVPTrainer(
            _ResultLookupSurrogateBackend(
                clean_result=SurrogateScoreResult.from_values([0.0]),
                results_by_model={},
            ),
            object(),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=0,
                policy_feature_set="normalized_position_only",
                nonzero_action_when_possible=True,
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[
                [1],
                [1, 2],
                [1, 2, 3],
                [1, 2, 3, 4],
            ],
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

        expected_candidate_space_diagnostics = {
            "total_session_count": 4,
            "pos0_removed_session_count": 3,
            "pos0_removed_pct": 75.0,
            "forced_single_candidate_count": 2,
            "forced_single_candidate_pct": 50.0,
            "fallback_to_pos0_only_count": 1,
            "fallback_to_pos0_only_pct": 25.0,
            "mean_candidate_count_before_mask": 2.5,
            "mean_candidate_count_after_mask": 1.75,
            "min_candidate_count_after_mask": 1,
            "max_candidate_count_after_mask": 3,
        }
        expected_final_position_diagnostics = {
            "final_pos0_pct": 25.0,
            "final_pos1_pct": 75.0,
            "final_pos_leq_2_pct": 100.0,
            "dominant_position": 1,
            "top5_positions": [
                {"position": 1, "count": 3, "pct": 75.0},
                {"position": 0, "count": 1, "pct": 25.0},
            ],
        }

        assert trainer_result["nonzero_action_when_possible"] is True
        assert trainer_result["candidate_space_diagnostics"] == expected_candidate_space_diagnostics
        assert trainer_result["final_position_diagnostics"] == expected_final_position_diagnostics
        assert training_history_payload["nonzero_action_when_possible"] is True
        assert (
            training_history_payload["candidate_space_diagnostics"]
            == expected_candidate_space_diagnostics
        )
        assert (
            training_history_payload["final_position_diagnostics"]
            == expected_final_position_diagnostics
        )
        assert learned_logits_payload["nonzero_action_when_possible"] is True
        assert (
            learned_logits_payload["candidate_space_diagnostics"]
            == expected_candidate_space_diagnostics
        )
        assert (
            learned_logits_payload["final_position_diagnostics"]
            == expected_final_position_diagnostics
        )
        assert learned_logits_payload["sessions"][1]["candidate_positions_before_mask"] == [0, 1]
        assert learned_logits_payload["sessions"][1]["candidate_positions"] == [1]
        assert learned_logits_payload["sessions"][1]["pos0_removed"] is True
        assert learned_logits_payload["sessions"][1]["forced_single_candidate"] is True
        assert learned_logits_payload["sessions"][1]["fallback_to_pos0_only"] is False
        assert learned_logits_payload["sessions"][0]["fallback_to_pos0_only"] is True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _config_with_nonzero_action_setting(
    config,
    *,
    checkpoint_path: Path,
    nonzero_action_when_possible: bool,
):
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")
    position_opt = replace(
        config.attack.position_opt,
        clean_surrogate_checkpoint=str(checkpoint_path),
        nonzero_action_when_possible=bool(nonzero_action_when_possible),
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


class _SequenceModelInnerTrainer:
    def __init__(self, model_names: list[str]) -> None:
        self.model_names = list(model_names)
        self.run_calls = 0

    def run(self, surrogate_backend, clean_checkpoint_path, poisoned_train_data, **kwargs):
        del surrogate_backend, clean_checkpoint_path, poisoned_train_data, kwargs
        model = self.model_names[self.run_calls]
        self.run_calls += 1
        return InnerTrainResult(
            model=model,
            history={"steps": 1, "epochs": 1, "avg_loss": 0.25},
        )


class _ResultLookupSurrogateBackend:
    def __init__(
        self,
        *,
        clean_result: SurrogateScoreResult,
        results_by_model: dict[str, SurrogateScoreResult],
    ) -> None:
        self.clean_result = clean_result
        self.results_by_model = dict(results_by_model)

    def load_clean_checkpoint(self, path) -> None:
        del path

    def clone_clean_model(self) -> str:
        return "clean-model"

    def score_target(self, model, eval_sessions, target_item) -> SurrogateScoreResult:
        del eval_sessions, target_item
        if model == "clean-model":
            return self.clean_result
        return self.results_by_model[model]

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        del model, eval_sessions, ground_truth_items
        raise AssertionError("score_gt() should not be called when GT penalty is disabled.")
