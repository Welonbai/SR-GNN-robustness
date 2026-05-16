from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import types

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSCEMEpochRewardDiagnosticsRuntimeConfig,
    PTSCEMSurrogateRetrainRuntimeConfig,
    load_config,
)
from attack.common.artifact_io import save_json
from attack.inner_train.srgnn_full_retrain_fixed_last import (
    SRGNNFullRetrainFixedLastInnerTrainer,
)
from attack.models import srgnn_validation_training
from attack.models.srgnn_validation_training import train_srgnn_one_epoch
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs import run_pts_construction_cem as pts_runner
from attack.position_opt.types import InnerTrainResult
from attack.surrogate.srgnn_backend import SRGNNBackend


CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "tests"
    / "fixtures"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.2], device=device))
        self.optimizer = torch.optim.SGD([self.logits], lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1,
            gamma=0.5,
        )
        self.loss_function = torch.nn.CrossEntropyLoss()


class _TinyRunner:
    def __init__(self) -> None:
        self.model = _TinyModel()
        self.train_loss_history: list[float] = []


class _TinyData:
    def generate_batch(self, batch_size: int):
        assert batch_size == 2
        yield [0, 1]


def test_train_srgnn_one_epoch_steps_scheduler_once(monkeypatch) -> None:
    runner = _TinyRunner()

    def fake_forward(model, batch_indices, train_data):
        del train_data
        scores = model.logits.unsqueeze(0).repeat(len(batch_indices), 1)
        scores = srgnn_validation_training.trans_to_cuda(scores)
        targets = [1 for _ in batch_indices]
        return targets, scores

    monkeypatch.setattr(srgnn_validation_training, "srg_forward", fake_forward)

    loss = train_srgnn_one_epoch(runner, _TinyData())

    assert loss > 0.0
    assert runner.model.scheduler.last_epoch == 1
    assert runner.model.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert len(runner.train_loss_history) == 1


def test_fixed_last_inner_trainer_uses_shared_epoch_train_helper(monkeypatch) -> None:
    calls: list[int] = []

    def fake_train_one_epoch(runner, train_data):
        del train_data
        calls.append(len(calls) + 1)
        runner.train_loss_history.append(0.25)
        return 0.25

    class Backend:
        def build_fresh_model(self):
            return types.SimpleNamespace(
                runner=types.SimpleNamespace(
                    train_loss_history=[],
                    model=types.SimpleNamespace(
                        optimizer=types.SimpleNamespace(
                            param_groups=[{"lr": 0.001}],
                        ),
                    ),
                )
            )

    monkeypatch.setattr(
        "attack.inner_train.srgnn_full_retrain_fixed_last.train_srgnn_one_epoch",
        fake_train_one_epoch,
    )
    trainer = SRGNNFullRetrainFixedLastInnerTrainer(
        train_config={"epochs": 4},
        max_epochs=4,
        log_epochs=False,
    )

    result = trainer.run(
        Backend(),
        None,
        ([[1, 2], [2, 3]], [2, 3]),
        seed=123,
    )

    assert calls == [1, 2, 3, 4]
    assert result.history["checkpoint_protocol"] == "fixed_last"
    assert result.history["selected_checkpoint_epoch"] == 4
    assert result.history["selected_checkpoint_source"] == "last_epoch"
    assert result.history["selected_checkpoint_metric"] is None
    assert result.history["validation_best_metrics_recorded"] is False
    assert result.history["official_reward_checkpoint_epoch"] == 4


def test_fixed_last_candidate_reward_uses_last_model_after_training(monkeypatch) -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    config = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=PTSCEMSurrogateRetrainRuntimeConfig(
                        checkpoint_protocol="fixed_last",
                        validation_enabled=False,
                        reward_checkpoint="last",
                    ),
                ),
            ),
        ),
    )
    shared = SharedAttackArtifacts(
        stats=object(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        canonical_dataset=object(),
        export_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=0,
        shared_paths={},
    )
    backend = SRGNNBackend(config, n_node=10)
    trainer = SRGNNFullRetrainFixedLastInnerTrainer(
        train_config={"epochs": 4},
        max_epochs=4,
        log_epochs=False,
    )

    def fake_run(self, *args, **kwargs):
        return InnerTrainResult(
            model="last-model",
            history={
                "checkpoint_protocol": "fixed_last",
                "selected_checkpoint_epoch": 4,
                "official_reward_checkpoint_epoch": 4,
            },
        )

    trainer.run = types.MethodType(fake_run, trainer)
    score_calls: list[object] = []

    def fake_score_candidate_target_reward(**kwargs):
        score_calls.append(kwargs["model"])
        assert kwargs["model"] == "last-model"
        return 1.25, {"targeted_mrr@20": 1.25}, 0.01

    monkeypatch.setattr(
        pts_runner,
        "_score_candidate_target_reward",
        fake_score_candidate_target_reward,
    )

    result = pts_runner._evaluate_candidate_retrain_validation_reward(
        config=config,
        evaluator_context={
            "shared": shared,
            "backend": backend,
            "inner_trainer": trainer,
            "validation_sessions": [[1, 2]],
            "validation_eval_data": None,
        },
        candidate_sessions=[[9, 10]],
        target_item=5334,
        iteration=0,
        population_size=1,
        candidate_id=0,
        candidate_seed=123,
    )

    assert score_calls == ["last-model"]
    assert result.reward == pytest.approx(1.25)
    assert result.metadata["selected_checkpoint_epoch"] == 4
    assert result.metadata["selected_checkpoint_source"] == "last_epoch"
    assert result.metadata["validation_best_metrics_recorded"] is False
    assert result.metadata["official_reward_checkpoint_epoch"] == 4


def test_candidate_evaluator_context_selects_protocol_specific_trainer(monkeypatch) -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    fixed_config = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=PTSCEMSurrogateRetrainRuntimeConfig(
                        checkpoint_protocol="fixed_last",
                        validation_enabled=False,
                        reward_checkpoint="last",
                    ),
                ),
            ),
        ),
    )
    shared = SharedAttackArtifacts(
        stats=object(),
        clean_sessions=[],
        clean_labels=[],
        canonical_dataset=object(),
        export_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=0,
        shared_paths={},
    )
    monkeypatch.setattr(
        pts_runner,
        "_resolve_validation_pairs",
        lambda shared: ([[1, 2]], [3]),
    )

    default_context = pts_runner._build_candidate_evaluator_context(config, shared)
    fixed_context = pts_runner._build_candidate_evaluator_context(fixed_config, shared)

    assert isinstance(
        default_context["inner_trainer"],
        pts_runner.SRGNNFullRetrainValidationBestInnerTrainer,
    )
    assert isinstance(
        fixed_context["inner_trainer"],
        pts_runner.SRGNNFullRetrainFixedLastInnerTrainer,
    )
    assert fixed_context["validation_eval_data"] is None


def test_fixed_last_shared_cache_reuse_records_identity_neutral_note(tmp_path) -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    config = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=PTSCEMSurrogateRetrainRuntimeConfig(
                        checkpoint_protocol="fixed_last",
                        validation_enabled=False,
                        reward_checkpoint="last",
                    ),
                ),
            ),
        ),
    )
    sessions_path = tmp_path / "sessions.json"
    metadata_path = tmp_path / "metadata.json"
    save_json([[1, 2, 5334]], sessions_path)
    save_json({}, metadata_path)
    cached = pts_runner.CachedPTSBestCandidate(
        sessions=[[1, 2, 5334]],
        metadata={
            "rank": 1,
            "candidate_key": "iter0_cand0",
            "evaluator_metadata": {
                "selected_checkpoint_metric": "valid_ground_truth_mrr@20",
            },
        },
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        top_candidates_path=None,
        complete_marker_path=None,
        cache_mode="shared_complete_marker",
        cache_marker_missing=False,
        shared_pts_cem_cache_key="pts_cem_shared_same",
        shared_cache_path=tmp_path,
        reused_shared_pts_cem=True,
        local_materialized_from_shared=True,
    )

    payload = pts_runner._target_metadata_from_cache(
        config=config,
        pts_config=pts,
        cem_config=pts_runner._build_pts_cem_config_from_config(config),
        artifact_dir=tmp_path,
        target_item=5334,
        cached=cached,
    )

    assert payload["requested_surrogate_retrain_checkpoint_protocol"] == "fixed_last"
    assert payload["reused_shared_pts_cem"] is True
    assert payload["reused_artifact_surrogate_retrain_protocol"] == "validation_best"
    assert payload["pts_cem_surrogate_retrain_identity_neutral"] is True
    assert "fixed_last is intentionally excluded from cache identity" in payload["note"]


def test_fixed_last_epoch_diagnostics_do_not_change_official_last_reward(monkeypatch) -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    config = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=PTSCEMSurrogateRetrainRuntimeConfig(
                        checkpoint_protocol="fixed_last",
                        validation_enabled=False,
                        reward_checkpoint="last",
                    ),
                    epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                        enabled=True,
                        epochs=(2,),
                        include_final_epoch=True,
                    ),
                ),
            ),
        ),
    )
    shared = SharedAttackArtifacts(
        stats=object(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        canonical_dataset=object(),
        export_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=0,
        shared_paths={},
    )
    backend = SRGNNBackend(config, n_node=10)
    trainer = SRGNNFullRetrainFixedLastInnerTrainer(
        train_config={"epochs": 4},
        max_epochs=4,
        log_epochs=False,
    )

    def fake_run(self, *args, **kwargs):
        kwargs["epoch_callback"]("epoch-2-model", {"epoch": 2, "train_loss": 0.2})
        return InnerTrainResult(
            model="last-model",
            history={
                "checkpoint_protocol": "fixed_last",
                "selected_checkpoint_epoch": 4,
                "official_reward_checkpoint_epoch": 4,
            },
        )

    trainer.run = types.MethodType(fake_run, trainer)
    score_calls: list[object] = []

    def fake_score_candidate_target_reward(**kwargs):
        model = kwargs["model"]
        score_calls.append(model)
        if model == "epoch-2-model":
            return 0.5, {"targeted_mrr@20": 0.5}, 0.01
        assert model == "last-model"
        return 1.5, {"targeted_mrr@20": 1.5}, 0.01

    monkeypatch.setattr(
        pts_runner,
        "_score_candidate_target_reward",
        fake_score_candidate_target_reward,
    )

    result = pts_runner._evaluate_candidate_retrain_validation_reward(
        config=config,
        evaluator_context={
            "shared": shared,
            "backend": backend,
            "inner_trainer": trainer,
            "validation_sessions": [[1, 2]],
            "validation_eval_data": None,
        },
        candidate_sessions=[[9, 10]],
        target_item=5334,
        iteration=0,
        population_size=1,
        candidate_id=0,
        candidate_seed=123,
    )

    assert score_calls == ["epoch-2-model", "last-model"]
    assert result.reward == pytest.approx(1.5)
    assert result.epoch_reward_diagnostics["official_reward_source"] == (
        "final_partial_retrain_protocol"
    )
    assert result.epoch_reward_diagnostics["rewards_by_epoch"]["4"][
        "official_reward_checkpoint_epoch"
    ] == 4
