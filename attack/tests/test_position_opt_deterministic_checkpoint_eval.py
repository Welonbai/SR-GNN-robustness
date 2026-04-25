from __future__ import annotations

import json
import shutil
from argparse import Namespace
from dataclasses import asdict, replace
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
from attack.pipeline.runs.run_position_opt_shared_policy import _resolve_position_opt_overrides
from attack.position_opt.trainer import (
    PositionOptMVPTrainer,
    _resolve_deterministic_eval_schedule,
)
from attack.position_opt.types import (
    InnerTrainResult,
    PositionOptArtifactPaths,
    SurrogateScoreResult,
    position_opt_identity_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = (
    REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
)
FOCUS_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_checkpoint_eval_tron_focus.yaml"
)


def test_deterministic_checkpoint_config_defaults_validation_and_cli_override() -> None:
    defaults = PositionOptConfig()

    assert defaults.deterministic_eval_every == 0
    assert defaults.deterministic_eval_include_final is True
    assert defaults.final_policy_selection == "last"
    assert PositionOptConfig(final_policy_selection="best_deterministic").final_policy_selection == (
        "best_deterministic"
    )
    assert _resolve_position_opt_overrides(
        Namespace(
            deterministic_eval_every=5,
            deterministic_eval_include_final=False,
            final_policy_selection="best_deterministic",
        )
    ) == {
        "deterministic_eval_every": 5,
        "deterministic_eval_include_final": False,
        "final_policy_selection": "best_deterministic",
    }

    with pytest.raises(ValueError, match="deterministic_eval_every"):
        PositionOptConfig(deterministic_eval_every=-1)
    with pytest.raises(ValueError, match="final_policy_selection"):
        PositionOptConfig(final_policy_selection="checkpoint_max")


def test_default_checkpoint_fields_are_pruned_from_identity_payload() -> None:
    payload = position_opt_identity_payload(PositionOptConfig())

    assert "deterministic_eval_every" not in payload
    assert "deterministic_eval_include_final" not in payload
    assert "final_policy_selection" not in payload


def test_checkpoint_eval_focus_yaml_loads_expected_fields() -> None:
    config = load_config(FOCUS_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.policy_feature_set == "local_context"
    assert config.attack.position_opt.reward_mode == "poisoned_target_utility"
    assert config.attack.position_opt.deterministic_eval_every == 5
    assert config.attack.position_opt.deterministic_eval_include_final is True
    assert config.attack.position_opt.final_policy_selection == "best_deterministic"
    assert list(config.victims.enabled) == ["tron"]
    assert config.targets.mode == "explicit_list"
    assert list(config.targets.explicit_list) == [5334, 11103]


def test_deterministic_eval_schedule_matches_required_steps() -> None:
    assert _resolve_deterministic_eval_schedule(
        total_outer_steps=30,
        deterministic_eval_every=5,
        deterministic_eval_include_final=True,
    ) == [5, 10, 15, 20, 25, 30]
    assert _resolve_deterministic_eval_schedule(
        total_outer_steps=32,
        deterministic_eval_every=5,
        deterministic_eval_include_final=True,
    ) == [5, 10, 15, 20, 25, 30, 32]
    assert _resolve_deterministic_eval_schedule(
        total_outer_steps=32,
        deterministic_eval_every=5,
        deterministic_eval_include_final=False,
    ) == [5, 10, 15, 20, 25, 30]
    assert _resolve_deterministic_eval_schedule(
        total_outer_steps=30,
        deterministic_eval_every=0,
        deterministic_eval_include_final=True,
    ) == []


def test_default_disabled_checkpoint_eval_preserves_last_step_export_path_and_cost() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_deterministic_checkpoint_eval" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        trainer = PositionOptMVPTrainer(
            _ResultLookupSurrogateBackend(
                clean_result=SurrogateScoreResult.from_values([0.05]),
                results_by_model={
                    "model-0": SurrogateScoreResult.from_values([0.20]),
                    "model-1": SurrogateScoreResult.from_values([0.30]),
                },
            ),
            _CountingInnerTrainer(),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=2,
                fine_tune_steps=1,
                deterministic_eval_every=0,
                final_policy_selection="last",
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[[1, 2, 3], [2, 3, 4]],
            target_item=99,
            shared_artifacts=SimpleNamespace(
                clean_sessions=[[1, 2], [3, 4]],
                clean_labels=[3, 5],
                validation_sessions=[[4, 5], [5, 6]],
                validation_labels=[7, 8],
            ),
            config=config,
        )

        assert trainer.inner_trainer.run_calls == 2
        assert trainer.surrogate_backend.poisoned_score_calls == 2
        assert trainer_result["best_deterministic_step"] is None
        assert trainer_result["best_deterministic_reward"] is None
        assert trainer_result["exported_policy_source"] == "last"
        assert [row["deterministic_eval_ran"] for row in trainer_result["training_history"]] == [
            False,
            False,
        ]
        assert all(row["deterministic_reward"] is None for row in trainer_result["training_history"])

        artifact_dir = temp_root / "artifacts"
        artifact_paths = PositionOptArtifactPaths(
            base_dir=artifact_dir,
            clean_surrogate_checkpoint=artifact_dir / "clean_surrogate.pt",
            optimized_poisoned_sessions=artifact_dir / "optimized_poisoned_sessions.pkl",
            training_history=artifact_dir / "training_history.json",
            learned_logits=artifact_dir / "learned_logits.pt",
        )
        trainer.save_artifacts(artifact_paths)

        training_history_payload = json.loads(
            artifact_paths.training_history.read_text(encoding="utf-8")
        )
        learned_logits_payload = torch.load(artifact_paths.learned_logits, map_location="cpu")
        assert training_history_payload["deterministic_eval_every"] == 0
        assert training_history_payload["final_policy_selection"] == "last"
        assert training_history_payload["best_deterministic_step"] is None
        assert training_history_payload["exported_policy_source"] == "last"
        assert learned_logits_payload["exported_policy_source"] == "last"
        assert learned_logits_payload["best_deterministic_step"] is None
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_best_deterministic_checkpoint_is_tracked_and_exported() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_deterministic_checkpoint_eval" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        trainer = PositionOptMVPTrainer(
            _ResultLookupSurrogateBackend(
                clean_result=SurrogateScoreResult.from_values([0.00]),
                results_by_model={
                    "sampled-0": SurrogateScoreResult.from_values([0.25]),
                    "det-1": SurrogateScoreResult.from_values([0.10]),
                    "sampled-1": SurrogateScoreResult.from_values([0.35]),
                    "det-2": SurrogateScoreResult.from_values([0.30]),
                    "sampled-2": SurrogateScoreResult.from_values([0.45]),
                    "det-3": SurrogateScoreResult.from_values([0.20]),
                },
            ),
            _SequenceModelInnerTrainer(
                ["sampled-0", "det-1", "sampled-1", "det-2", "sampled-2", "det-3"]
            ),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=3,
                fine_tune_steps=1,
                deterministic_eval_every=1,
                deterministic_eval_include_final=True,
                final_policy_selection="best_deterministic",
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[[1, 2, 3], [2, 3, 4]],
            target_item=99,
            shared_artifacts=SimpleNamespace(
                clean_sessions=[[1, 2], [3, 4]],
                clean_labels=[3, 5],
                validation_sessions=[[4, 5], [5, 6]],
                validation_labels=[7, 8],
            ),
            config=config,
        )

        best_checkpoint = trainer._best_deterministic_checkpoint
        assert best_checkpoint is not None
        assert trainer_result["best_deterministic_step"] == 2
        assert trainer_result["best_deterministic_reward"] == pytest.approx(0.3)
        assert trainer_result["last_deterministic_reward"] == pytest.approx(0.2)
        assert trainer_result["best_minus_last_deterministic_reward"] == pytest.approx(0.1)
        assert trainer_result["exported_policy_source"] == "best_deterministic"
        assert [row["deterministic_reward"] for row in trainer_result["training_history"]] == (
            pytest.approx([0.1, 0.3, 0.2])
        )
        assert [row["is_best_deterministic_checkpoint"] for row in trainer_result["training_history"]] == [
            True,
            True,
            False,
        ]
        assert [
            row["best_deterministic_step_so_far"] for row in trainer_result["training_history"]
        ] == [1, 2, 2]
        assert trainer_result["final_selected_positions"] == [
            asdict(result) for result in best_checkpoint.selected_position_results
        ]
        assert _state_dicts_equal(trainer.policy.state_dict(), best_checkpoint.policy_state_dict)

        artifact_dir = temp_root / "artifacts"
        artifact_paths = PositionOptArtifactPaths(
            base_dir=artifact_dir,
            clean_surrogate_checkpoint=artifact_dir / "clean_surrogate.pt",
            optimized_poisoned_sessions=artifact_dir / "optimized_poisoned_sessions.pkl",
            training_history=artifact_dir / "training_history.json",
            learned_logits=artifact_dir / "learned_logits.pt",
        )
        trainer.save_artifacts(artifact_paths)

        training_history_payload = json.loads(
            artifact_paths.training_history.read_text(encoding="utf-8")
        )
        learned_logits_payload = torch.load(artifact_paths.learned_logits, map_location="cpu")
        assert training_history_payload["deterministic_eval_every"] == 1
        assert training_history_payload["deterministic_eval_include_final"] is True
        assert training_history_payload["final_policy_selection"] == "best_deterministic"
        assert training_history_payload["best_deterministic_step"] == 2
        assert training_history_payload["best_deterministic_reward"] == pytest.approx(0.3)
        assert training_history_payload["last_deterministic_reward"] == pytest.approx(0.2)
        assert training_history_payload["exported_policy_source"] == "best_deterministic"
        assert training_history_payload["training_history"][1]["deterministic_eval_ran"] is True
        assert training_history_payload["training_history"][1]["deterministic_reward"] == pytest.approx(
            0.3
        )
        assert "deterministic_selected_pos0_pct" in training_history_payload["training_history"][1]
        assert learned_logits_payload["exported_policy_source"] == "best_deterministic"
        assert learned_logits_payload["deterministic_eval_include_final"] is True
        assert learned_logits_payload["final_policy_selection"] == "best_deterministic"
        assert learned_logits_payload["best_deterministic_step"] == 2
        assert learned_logits_payload["best_deterministic_reward"] == pytest.approx(0.3)
        assert learned_logits_payload["last_deterministic_reward"] == pytest.approx(0.2)
        assert learned_logits_payload["best_minus_last_deterministic_reward"] == pytest.approx(
            0.1
        )
        assert learned_logits_payload["deterministic_eval_every"] == 1
        assert learned_logits_payload["exported_selected_positions"] == [
            asdict(result) for result in best_checkpoint.selected_position_results
        ]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_deterministic_checkpoint_config_changes_runtime_identity_but_not_shared_generation_key() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_deterministic_checkpoint_eval" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = temp_root / "clean_surrogate.pt"
        checkpoint_path.write_bytes(b"fake clean surrogate checkpoint")

        baseline_config, baseline_context = _config_with_checkpoint_eval_fields(
            config,
            checkpoint_path=checkpoint_path,
            deterministic_eval_every=0,
            deterministic_eval_include_final=True,
            final_policy_selection="last",
        )
        eval_every_config, eval_every_context = _config_with_checkpoint_eval_fields(
            config,
            checkpoint_path=checkpoint_path,
            deterministic_eval_every=5,
            deterministic_eval_include_final=True,
            final_policy_selection="last",
        )
        include_final_config, include_final_context = _config_with_checkpoint_eval_fields(
            config,
            checkpoint_path=checkpoint_path,
            deterministic_eval_every=5,
            deterministic_eval_include_final=False,
            final_policy_selection="last",
        )
        best_policy_config, best_policy_context = _config_with_checkpoint_eval_fields(
            config,
            checkpoint_path=checkpoint_path,
            deterministic_eval_every=5,
            deterministic_eval_include_final=True,
            final_policy_selection="best_deterministic",
        )
        baseline_payload = position_opt_identity_payload(baseline_config.attack.position_opt)
        eval_every_payload = position_opt_identity_payload(eval_every_config.attack.position_opt)
        best_policy_payload = position_opt_identity_payload(best_policy_config.attack.position_opt)

        assert "deterministic_eval_every" not in baseline_payload
        assert "deterministic_eval_include_final" not in baseline_payload
        assert "final_policy_selection" not in baseline_payload
        assert eval_every_payload["deterministic_eval_every"] == 5
        assert eval_every_payload["deterministic_eval_include_final"] is True
        assert eval_every_payload["final_policy_selection"] == "last"
        assert best_policy_payload["deterministic_eval_every"] == 5
        assert best_policy_payload["deterministic_eval_include_final"] is True
        assert best_policy_payload["final_policy_selection"] == "best_deterministic"

        for left_config, left_context, right_config, right_context in (
            (baseline_config, baseline_context, eval_every_config, eval_every_context),
            (eval_every_config, eval_every_context, include_final_config, include_final_context),
            (eval_every_config, eval_every_context, best_policy_config, best_policy_context),
        ):
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

        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            eval_every_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
        assert shared_attack_artifact_key(
            eval_every_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            include_final_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
        assert shared_attack_artifact_key(
            eval_every_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            best_policy_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_deterministic_checkpoint_eval_reuses_fixed_seed_across_steps() -> None:
    config = load_config(BASE_CONFIG_PATH)
    temp_root = REPO_ROOT / "outputs" / ".pytest_deterministic_checkpoint_eval" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        inner_trainer = _SequenceModelInnerTrainer(
            ["sampled-0", "det-1", "sampled-1", "det-2"]
        )
        trainer = PositionOptMVPTrainer(
            _ResultLookupSurrogateBackend(
                clean_result=SurrogateScoreResult.from_values([0.00]),
                results_by_model={
                    "sampled-0": SurrogateScoreResult.from_values([0.25]),
                    "det-1": SurrogateScoreResult.from_values([0.10]),
                    "sampled-1": SurrogateScoreResult.from_values([0.35]),
                    "det-2": SurrogateScoreResult.from_values([0.30]),
                },
            ),
            inner_trainer,
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=2,
                fine_tune_steps=1,
                deterministic_eval_every=1,
                deterministic_eval_include_final=True,
                final_policy_selection="last",
            ),
        )

        trainer_result = trainer.train(
            fake_sessions=[[1, 2, 3], [2, 3, 4]],
            target_item=99,
            shared_artifacts=SimpleNamespace(
                clean_sessions=[[1, 2], [3, 4]],
                clean_labels=[3, 5],
                validation_sessions=[[4, 5], [5, 6]],
                validation_labels=[7, 8],
            ),
            config=config,
        )

        assert len(inner_trainer.received_seeds) == 4
        sampled_seed_step1, deterministic_seed_step1, sampled_seed_step2, deterministic_seed_step2 = (
            inner_trainer.received_seeds
        )
        assert sampled_seed_step1 != sampled_seed_step2
        assert deterministic_seed_step1 == deterministic_seed_step2
        sampled_training_seeds = [
            row["surrogate_train_step_seed"] for row in trainer_result["training_history"]
        ]
        assert sampled_training_seeds[0] != sampled_training_seeds[1]
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _config_with_checkpoint_eval_fields(
    config,
    *,
    checkpoint_path: Path,
    deterministic_eval_every: int,
    deterministic_eval_include_final: bool,
    final_policy_selection: str,
):
    if config.attack.position_opt is None:
        raise AssertionError("Shared-policy config must include attack.position_opt.")
    position_opt = replace(
        config.attack.position_opt,
        clean_surrogate_checkpoint=str(checkpoint_path),
        deterministic_eval_every=int(deterministic_eval_every),
        deterministic_eval_include_final=bool(deterministic_eval_include_final),
        final_policy_selection=str(final_policy_selection),
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


def _state_dicts_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key].detach().cpu(), right[key].detach().cpu()) for key in left)


class _CountingInnerTrainer:
    def __init__(self) -> None:
        self.run_calls = 0

    def run(self, surrogate_backend, clean_checkpoint_path, poisoned_train_data, **kwargs):
        del surrogate_backend, clean_checkpoint_path, poisoned_train_data, kwargs
        model = f"model-{self.run_calls}"
        self.run_calls += 1
        return InnerTrainResult(
            model=model,
            history={"steps": 1, "epochs": 1, "avg_loss": 0.25},
        )


class _SequenceModelInnerTrainer:
    def __init__(self, model_names: list[str]) -> None:
        self.model_names = list(model_names)
        self.run_calls = 0
        self.received_seeds: list[int | None] = []

    def run(self, surrogate_backend, clean_checkpoint_path, poisoned_train_data, **kwargs):
        self.received_seeds.append(kwargs.get("seed"))
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
        self.load_clean_checkpoint_calls = 0
        self.clone_clean_model_calls = 0
        self.clean_score_calls = 0
        self.poisoned_score_calls = 0

    def load_clean_checkpoint(self, path) -> None:
        del path
        self.load_clean_checkpoint_calls += 1

    def clone_clean_model(self) -> str:
        self.clone_clean_model_calls += 1
        return "clean-model"

    def fine_tune(self, model, poisoned_train_data, **kwargs):
        del model, poisoned_train_data, kwargs
        return {"steps": 1, "epochs": 1, "avg_loss": 0.25}

    def score_target(self, model, eval_sessions, target_item) -> SurrogateScoreResult:
        del eval_sessions, target_item
        if model == "clean-model":
            self.clean_score_calls += 1
            return self.clean_result
        self.poisoned_score_calls += 1
        return self.results_by_model[model]

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        del model, eval_sessions, ground_truth_items
        raise AssertionError("score_gt() should not be called when GT penalty is disabled.")
