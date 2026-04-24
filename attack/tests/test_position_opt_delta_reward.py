from __future__ import annotations

import json
from argparse import Namespace
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
from attack.pipeline.runs.run_position_opt_shared_policy import _resolve_position_opt_overrides
from attack.position_opt.trainer import PositionOptMVPTrainer, _resolve_reward_target_utility
from attack.position_opt.types import (
    InnerTrainResult,
    PositionOptArtifactPaths,
    SurrogateScoreResult,
    position_opt_identity_payload,
)
import attack.surrogate.srgnn_backend as srgnn_backend_module
from attack.surrogate.srgnn_backend import SRGNNBackend, SRGNNModelHandle


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_position_opt_shared_policy.yaml"
DELTA_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_delta_reward.yaml"
)
LOWK_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_attack_position_opt_shared_policy_delta_lowk_rank_reward.yaml"
)


def test_reward_mode_loads_from_yaml_and_cli_override() -> None:
    config = load_config(DELTA_CONFIG_PATH)
    lowk_config = load_config(LOWK_CONFIG_PATH)

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.reward_mode == "delta_target_utility"
    assert lowk_config.attack.position_opt is not None
    assert lowk_config.attack.position_opt.reward_mode == "delta_lowk_rank_utility"
    assert _resolve_position_opt_overrides(Namespace(reward_mode="poisoned_target_utility"))[
        "reward_mode"
    ] == "poisoned_target_utility"
    assert _resolve_position_opt_overrides(Namespace(reward_mode="delta_lowk_rank_utility"))[
        "reward_mode"
    ] == "delta_lowk_rank_utility"


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
    assert _resolve_reward_target_utility(
        reward_mode="delta_lowk_rank_utility",
        poisoned_target_utility=0.7,
        clean_target_utility=0.2,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "reward_mode",
    ["delta_target_utility", "delta_lowk_rank_utility"],
)
def test_delta_reward_mode_requires_clean_baseline(reward_mode: str) -> None:
    with pytest.raises(ValueError, match="clean_target_utility"):
        _resolve_reward_target_utility(
            reward_mode=reward_mode,
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
        lowk_config, lowk_context = _config_with_reward_mode(
            config,
            reward_mode="delta_lowk_rank_utility",
            checkpoint_path=checkpoint_path,
        )

        for left_config, left_context, right_config, right_context in (
            (baseline_config, baseline_context, delta_config, delta_context),
            (baseline_config, baseline_context, lowk_config, lowk_context),
            (delta_config, delta_context, lowk_config, lowk_context),
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
            delta_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
        assert shared_attack_artifact_key(
            baseline_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        ) == shared_attack_artifact_key(
            lowk_config,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_lowk_reward_reuses_clean_baseline_and_logs_metrics() -> None:
    config = load_config(BASE_CONFIG_PATH)
    clean_metrics = {
        "targeted_mrr@10": 0.25,
        "targeted_recall@10": 0.4,
        "targeted_recall@20": 0.6,
    }
    poisoned_metrics_step1 = {
        "targeted_mrr@10": 0.5,
        "targeted_recall@10": 0.7,
        "targeted_recall@20": 0.8,
    }
    poisoned_metrics_step2 = {
        "targeted_mrr@10": 0.4,
        "targeted_recall@10": 0.5,
        "targeted_recall@20": 0.9,
    }
    surrogate_backend = _StubSurrogateBackend(
        clean_result=SurrogateScoreResult.from_values([0.05], metrics=clean_metrics),
        poisoned_results=[
            SurrogateScoreResult.from_values([0.9], metrics=poisoned_metrics_step1),
            SurrogateScoreResult.from_values([0.8], metrics=poisoned_metrics_step2),
        ],
    )
    temp_root = REPO_ROOT / "outputs" / ".pytest_delta_reward" / uuid4().hex
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        trainer = PositionOptMVPTrainer(
            surrogate_backend,
            _StubInnerTrainer(),
            clean_surrogate_checkpoint_path=temp_root / "clean_surrogate.pt",
            position_opt_config=PositionOptConfig(
                outer_steps=2,
                reward_mode="delta_lowk_rank_utility",
                fine_tune_steps=1,
                reward_baseline_momentum=0.9,
            ),
        )
        shared_artifacts = SimpleNamespace(
            clean_sessions=[[1, 2, 3], [2, 3, 4]],
            clean_labels=[4, 5],
            validation_sessions=[[1, 2], [2, 3]],
            validation_labels=[4, 5],
        )

        trainer_result = trainer.train(
            fake_sessions=[[1, 2, 3], [2, 3, 4]],
            target_item=99,
            shared_artifacts=shared_artifacts,
            config=config,
        )

        clean_utility = _weighted_lowk_utility(clean_metrics)
        poisoned_utility_step1 = _weighted_lowk_utility(poisoned_metrics_step1)
        poisoned_utility_step2 = _weighted_lowk_utility(poisoned_metrics_step2)

        assert surrogate_backend.load_clean_checkpoint_calls == 1
        assert surrogate_backend.clone_clean_model_calls == 1
        assert surrogate_backend.clean_score_calls == 1
        assert surrogate_backend.poisoned_score_calls == 2

        assert trainer_result["clean_target_utility"] == pytest.approx(clean_utility)
        assert trainer_result["clean_targeted_mrr_at_10"] == pytest.approx(clean_metrics["targeted_mrr@10"])
        assert trainer_result["clean_targeted_recall_at_10"] == pytest.approx(
            clean_metrics["targeted_recall@10"]
        )
        assert trainer_result["clean_targeted_recall_at_20"] == pytest.approx(
            clean_metrics["targeted_recall@20"]
        )

        first_step = trainer_result["training_history"][0]
        second_step = trainer_result["training_history"][1]
        assert first_step["target_utility"] == pytest.approx(poisoned_utility_step1)
        assert first_step["poisoned_target_utility"] == pytest.approx(poisoned_utility_step1)
        assert first_step["delta_target_utility"] == pytest.approx(
            poisoned_utility_step1 - clean_utility
        )
        assert first_step["reward"] == pytest.approx(poisoned_utility_step1 - clean_utility)
        assert first_step["clean_targeted_mrr_at_10"] == pytest.approx(clean_metrics["targeted_mrr@10"])
        assert first_step["poisoned_targeted_mrr_at_10"] == pytest.approx(
            poisoned_metrics_step1["targeted_mrr@10"]
        )
        assert first_step["poisoned_targeted_recall_at_10"] == pytest.approx(
            poisoned_metrics_step1["targeted_recall@10"]
        )
        assert first_step["poisoned_targeted_recall_at_20"] == pytest.approx(
            poisoned_metrics_step1["targeted_recall@20"]
        )
        assert second_step["target_utility"] == pytest.approx(poisoned_utility_step2)
        assert second_step["poisoned_target_utility"] == pytest.approx(poisoned_utility_step2)
        assert second_step["delta_target_utility"] == pytest.approx(
            poisoned_utility_step2 - clean_utility
        )

        artifact_dir = temp_root / "position_opt"
        trainer.save_artifacts(
            PositionOptArtifactPaths(
                base_dir=artifact_dir,
                clean_surrogate_checkpoint=artifact_dir / "clean_surrogate.pt",
                optimized_poisoned_sessions=artifact_dir / "optimized_poisoned_sessions.pkl",
                training_history=artifact_dir / "training_history.json",
            )
        )
        training_history_payload = json.loads(
            (artifact_dir / "training_history.json").read_text(encoding="utf-8")
        )
        assert training_history_payload["clean_target_utility"] == pytest.approx(clean_utility)
        assert training_history_payload["clean_targeted_mrr_at_10"] == pytest.approx(
            clean_metrics["targeted_mrr@10"]
        )
        assert training_history_payload["clean_targeted_recall_at_10"] == pytest.approx(
            clean_metrics["targeted_recall@10"]
        )
        assert training_history_payload["clean_targeted_recall_at_20"] == pytest.approx(
            clean_metrics["targeted_recall@20"]
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def test_srgnn_score_target_returns_lowk_metric_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(BASE_CONFIG_PATH)
    backend = SRGNNBackend(config, n_node=8)
    scores = torch.tensor(
        [
            [0.1, 2.0, 0.5, -1.0],
            [1.5, 0.4, 0.8, 0.2],
        ],
        dtype=torch.float32,
    )

    def fake_forward(model, batch_indices, data):
        del model, data
        index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        return None, scores.index_select(0, index_tensor)

    monkeypatch.setattr(srgnn_backend_module, "srg_forward", fake_forward)
    handle = SRGNNModelHandle(runner=SimpleNamespace(model=_FakeSRGNNModel(batch_size=8)))

    result = backend.score_target(handle, [[1, 2], [2, 3]], target_item=2)

    expected_mean = float(torch.softmax(scores, dim=1)[:, 1].mean().item())
    assert result.mean == pytest.approx(expected_mean)
    assert result.metrics is not None
    assert result.metrics["targeted_mrr@10"] == pytest.approx((1.0 + (1.0 / 3.0)) / 2.0)
    assert result.metrics["targeted_recall@10"] == pytest.approx(1.0)
    assert result.metrics["targeted_recall@20"] == pytest.approx(1.0)


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


def _weighted_lowk_utility(metrics: dict[str, float]) -> float:
    return float(
        0.6 * metrics["targeted_mrr@10"]
        + 0.3 * metrics["targeted_recall@10"]
        + 0.1 * metrics["targeted_recall@20"]
    )


class _StubInnerTrainer:
    def __init__(self) -> None:
        self.run_calls = 0

    def run(self, surrogate_backend, clean_checkpoint_path, poisoned_train_data, **kwargs):
        del surrogate_backend, clean_checkpoint_path, poisoned_train_data, kwargs
        model = f"poisoned-model-{self.run_calls}"
        self.run_calls += 1
        return InnerTrainResult(
            model=model,
            history={"steps": 1, "epochs": 1, "avg_loss": 0.25},
        )


class _StubSurrogateBackend:
    def __init__(
        self,
        *,
        clean_result: SurrogateScoreResult,
        poisoned_results: list[SurrogateScoreResult],
    ) -> None:
        self.clean_result = clean_result
        self.poisoned_results = list(poisoned_results)
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
        return self.poisoned_results[self.poisoned_score_calls - 1]

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        del model, eval_sessions, ground_truth_items
        raise AssertionError("score_gt() should not be called when GT penalty is disabled.")


class _FakeSRGNNModel:
    def __init__(self, *, batch_size: int) -> None:
        self.batch_size = int(batch_size)

    def eval(self) -> None:
        return None
