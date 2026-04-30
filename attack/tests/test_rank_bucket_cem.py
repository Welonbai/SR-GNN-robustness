from __future__ import annotations

from dataclasses import replace
import json
import pickle
import random
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    RankBucketCEMConfig,
    SurrogateEvalPoisonBalanceConfig,
    load_config,
)
from attack.common.paths import (
    POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    attack_key,
    run_group_key,
    shared_attack_artifact_key,
    victim_prediction_key,
)
from attack.position_opt import position_opt_identity_payload
from attack.position_opt.cem import rank_bucket_cem_identity_payload
from attack.position_opt.cem.artifacts import RankBucketCEMArtifactPaths
from attack.position_opt.cem.availability import (
    build_availability_summary,
    build_rank_candidate_states,
)
from attack.position_opt.cem.optimizer import (
    CEMCandidate,
    CEMCandidateResult,
    initialize_cem_state,
    update_cem_state,
)
from attack.position_opt.cem.rank_policy import (
    RankBucketPolicy,
    build_rank_position_summary,
    sample_positions_from_rank_policy,
)
from attack.position_opt.cem.trainer import (
    NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD,
    RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
    RankBucketCEMTrainer,
    _resolve_reward_value,
    _with_global_normalized_lowk_rewards,
    _with_iteration_normalized_lowk_rewards,
)
from attack.surrogate.srgnn_backend import _TARGET_SCORE_REQUIRED_KEYS
from attack.position_opt.types import InnerTrainResult, SurrogateScoreResult
from attack.surrogate.srgnn_backend import _fixed_ratio_batch_sizes


CONFIG_PATH = (
    REPO_ROOT / "attack" / "configs" / "diginetica_attack_rank_bucket_cem.yaml"
)


def _config():
    return load_config(CONFIG_PATH)


def _session(length: int) -> list[int]:
    return list(range(1, length + 1))


def _identity_context(config) -> dict[str, object]:
    return {
        "position_opt": {
            "config": position_opt_identity_payload(config.attack.position_opt),
            "rank_bucket_cem": rank_bucket_cem_identity_payload(
                config.attack.rank_bucket_cem
            ),
            "seeds": {
                "position_opt_seed": int(config.seeds.position_opt_seed),
                "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
            },
            "clean_surrogate": {
                "type": "file_sha1",
                "sha1": "dummy-clean-checkpoint",
            },
        }
    }


class _DummyBackend:
    def __init__(
        self,
        *,
        trained_target_results: list[SurrogateScoreResult] | None = None,
    ) -> None:
        self.loaded_paths: list[str] = []
        self.fixed_ratio_calls: list[dict[str, object]] = []
        self.clean_target_result = SurrogateScoreResult.from_values(
            [0.2],
            metrics={
                "custom_metric@7": 0.1,
                "another_metric@11": 0.05,
                "targeted_mrr@10": 0.01,
                "targeted_mrr@20": 0.02,
                "targeted_recall@10": 0.03,
                "targeted_recall@20": 0.04,
            },
        )
        self.trained_target_result = SurrogateScoreResult.from_values(
            [0.8],
            metrics={
                "custom_metric@7": 0.6,
                "another_metric@11": 0.4,
                "targeted_mrr@10": 0.11,
                "targeted_mrr@20": 0.12,
                "targeted_recall@10": 0.13,
                "targeted_recall@20": 0.14,
            },
        )
        self.trained_target_results = (
            [self.trained_target_result]
            if trained_target_results is None
            else list(trained_target_results)
        )

    def load_clean_checkpoint(self, path) -> None:
        self.loaded_paths.append(str(path))

    def clone_clean_model(self) -> object:
        return {"kind": "clean"}

    def fine_tune(self, model, poisoned_train_data, *, fine_tune_config=None, eval_data=None):
        raise AssertionError("Dummy backend fine_tune() should not be called directly.")

    def fine_tune_fixed_ratio(
        self,
        model,
        *,
        clean_sessions,
        clean_labels,
        poison_sessions,
        poison_labels,
        poison_source_session_ids,
        fine_tune_config=None,
        poison_ratio_in_batch: float,
        seed=None,
    ):
        del clean_labels, poison_labels, seed
        if isinstance(model, dict):
            model["kind"] = "trained"
        clean_batch_size, poison_batch_size = _fixed_ratio_batch_sizes(
            100,
            poison_ratio_in_batch,
        )
        steps = 0 if fine_tune_config is None else int(fine_tune_config.steps)
        clean_seen = int(clean_batch_size * steps)
        poison_seen = int(poison_batch_size * steps)
        unique_prefixes = min(len(poison_sessions), poison_seen)
        unique_sources = min(len(set(poison_source_session_ids)), unique_prefixes)
        payload = {
            "clean_pool_size": len(clean_sessions),
            "poison_pool_size": len(poison_sessions),
            "poison_source_session_ids": list(poison_source_session_ids),
            "steps": steps,
        }
        self.fixed_ratio_calls.append(payload)
        return {
            "steps": steps,
            "epochs": 1 if steps else 0,
            "avg_loss": 0.0,
            "step_loss": [0.0] * steps,
            "train_loss": [0.0] if steps else [],
            "poison_balance": {
                "poison_balance_enabled": True,
                "poison_balance_mode": "fixed_ratio",
                "requested_poison_ratio_in_batch": float(poison_ratio_in_batch),
                "configured_clean_batch_size": int(clean_batch_size),
                "configured_poison_batch_size": int(poison_batch_size),
                "effective_poison_ratio_seen": (
                    None
                    if clean_seen + poison_seen == 0
                    else poison_seen / float(clean_seen + poison_seen)
                ),
                "configured_fine_tune_steps": int(steps),
                "actual_optimizer_steps": int(steps),
                "batch_size": 100,
                "clean_pool_size": len(clean_sessions),
                "poison_pool_size": len(poison_sessions),
                "clean_examples_seen": int(clean_seen),
                "poison_examples_seen": int(poison_seen),
                "unique_poison_prefix_examples_seen": int(unique_prefixes),
                "unique_poison_source_sessions_seen": int(unique_sources),
                "sampling_strategy": "shuffled_cycling",
                "sampling_wrapped": bool(
                    clean_seen > len(clean_sessions) or poison_seen > len(poison_sessions)
                ),
            },
        }

    def score_target(self, model, eval_sessions, target_item: int) -> SurrogateScoreResult:
        if isinstance(model, dict) and model.get("kind") == "clean":
            return self.clean_target_result
        score_index = 0
        if isinstance(model, dict):
            score_index = int(model.get("score_index", 0))
        return self.trained_target_results[score_index % len(self.trained_target_results)]

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        return SurrogateScoreResult.from_values([0.55])


class _DummyInnerTrainer:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []
        self.run_count = 0

    def run(
        self,
        surrogate_backend,
        clean_checkpoint_path,
        poisoned_train_data,
        *,
        config=None,
        eval_data=None,
        seed=None,
    ) -> InnerTrainResult:
        del surrogate_backend, clean_checkpoint_path, eval_data
        self.seeds.append(seed)
        score_index = self.run_count
        self.run_count += 1
        return InnerTrainResult(
            model={
                "kind": "trained",
                "seed": seed,
                "score_index": score_index,
                "poisoned_fake_count": poisoned_train_data.fake_count,
            },
            history={
                "steps": 0 if config is None else int(config.steps),
                "epochs": 0 if config is None else int(config.epochs),
                "avg_loss": 0.0,
            },
        )


def _shared_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        clean_sessions=[_session(4), _session(5)],
        clean_labels=[10, 11],
        validation_sessions=[_session(4), _session(5), _session(6), _session(7)],
        validation_labels=[20, 21, 22, 23],
    )


def _artifact_paths(tmp_path: Path) -> RankBucketCEMArtifactPaths:
    base_dir = tmp_path / "rank_bucket_cem"
    clean_checkpoint = tmp_path / "clean_surrogate.pt"
    clean_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    clean_checkpoint.write_text("checkpoint", encoding="utf-8")
    return RankBucketCEMArtifactPaths(
        base_dir=base_dir,
        clean_surrogate_checkpoint=clean_checkpoint,
        optimized_poisoned_sessions=base_dir / "optimized_poisoned_sessions.pkl",
        availability_summary=base_dir / "availability_summary.json",
        cem_trace=base_dir / "cem_trace.jsonl",
        cem_state_history=base_dir / "cem_state_history.json",
        cem_best_policy=base_dir / "cem_best_policy.json",
        final_selected_positions=base_dir / "final_selected_positions.jsonl",
        final_position_summary=base_dir / "final_position_summary.json",
        run_metadata=base_dir / "run_metadata.json",
    )


def _repo_temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / ".pytest_rank_bucket_cem" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_dummy_trainer(
    tmp_path: Path,
    *,
    backend: _DummyBackend | None = None,
    reward_mode: str = "poisoned_target_utility",
    reward_metric: str | None = None,
    surrogate_eval_poison_balance: SurrogateEvalPoisonBalanceConfig | None = None,
    save_candidate_selected_positions: bool = False,
    save_final_selected_positions: bool = False,
    save_optimized_poisoned_sessions: bool = True,
    save_replay_metadata: bool = True,
    fake_sessions: list[list[int]] | None = None,
    iterations: int = 1,
    population_size: int = 3,
    population_per_iteration: tuple[int, ...] | None = None,
) -> tuple[dict[str, object], RankBucketCEMArtifactPaths, _DummyInnerTrainer]:
    config = _config()
    position_opt = replace(
        config.attack.position_opt,
        reward_mode=reward_mode,
        fine_tune_steps=1,
        validation_subset_size=2,
        enable_gt_penalty=False,
    )
    rank_bucket_cem = replace(
        config.attack.rank_bucket_cem,
        iterations=int(iterations),
        population_size=int(population_size),
        population_per_iteration=population_per_iteration,
        reward_metric=reward_metric,
        surrogate_eval_poison_balance=(
            config.attack.rank_bucket_cem.surrogate_eval_poison_balance
            if surrogate_eval_poison_balance is None
            else surrogate_eval_poison_balance
        ),
        save_candidate_selected_positions=save_candidate_selected_positions,
        save_final_selected_positions=save_final_selected_positions,
        save_optimized_poisoned_sessions=save_optimized_poisoned_sessions,
        save_replay_metadata=save_replay_metadata,
    )
    config = replace(
        config,
        attack=replace(
            config.attack,
            position_opt=position_opt,
            rank_bucket_cem=rank_bucket_cem,
        ),
    )

    backend = _DummyBackend() if backend is None else backend
    inner_trainer = _DummyInnerTrainer()
    trainer = RankBucketCEMTrainer(
        backend,
        inner_trainer,
        clean_surrogate_checkpoint_path=_artifact_paths(tmp_path).clean_surrogate_checkpoint,
        position_opt_config=position_opt,
        rank_bucket_cem_config=rank_bucket_cem,
    )
    artifact_paths = _artifact_paths(tmp_path)
    result = trainer.train(
        [_session(2), _session(3), _session(5)] if fake_sessions is None else fake_sessions,
        5334,
        _shared_artifacts(),
        config,
        artifact_paths=artifact_paths,
    )
    trainer.save_artifacts(artifact_paths)
    return result, artifact_paths, inner_trainer


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lowk_candidate_result(
    *,
    iteration: int,
    candidate_id: int,
    mrr10: float,
    mrr20: float,
    recall10: float,
    recall20: float,
    reward: float = 0.0,
) -> CEMCandidateResult:
    return CEMCandidateResult(
        candidate=CEMCandidate(
            iteration=iteration,
            candidate_id=candidate_id,
            logits_g2=(float(candidate_id), 0.0),
            logits_g3=(float(candidate_id), 0.0, 0.0),
            pi_g2=(0.5, 0.5),
            pi_g3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
        reward=float(reward),
        metrics={
            "target_metrics": {
                "targeted_mrr@10": float(mrr10),
                "targeted_mrr@20": float(mrr20),
                "targeted_recall@10": float(recall10),
                "targeted_recall@20": float(recall20),
            },
            "reward_components": {
                "targeted_mrr@10": float(mrr10),
                "targeted_mrr@20": float(mrr20),
                "targeted_recall@10": float(recall10),
                "targeted_recall@20": float(recall20),
            },
        },
    )


def test_rank_bucket_cem_config_loads_with_clean_defaults() -> None:
    config = _config()

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.nonzero_action_when_possible is True
    assert config.attack.position_opt.enable_gt_penalty is False
    assert config.attack.rank_bucket_cem is not None
    assert config.attack.rank_bucket_cem.iterations == 3
    assert config.attack.rank_bucket_cem.population_size == 8
    assert config.attack.rank_bucket_cem.population_per_iteration is None
    assert config.attack.rank_bucket_cem.effective_population_schedule == (8, 8, 8)
    assert config.attack.rank_bucket_cem.candidate_count == 24
    assert config.attack.rank_bucket_cem.reward_metric is None
    assert config.attack.rank_bucket_cem.save_candidate_selected_positions is False
    assert config.attack.rank_bucket_cem.save_final_selected_positions is False
    assert config.attack.rank_bucket_cem.save_optimized_poisoned_sessions is True
    assert config.attack.rank_bucket_cem.save_replay_metadata is True
    poison_balance = config.attack.rank_bucket_cem.surrogate_eval_poison_balance
    assert poison_balance.enabled is False
    assert poison_balance.mode == "fixed_ratio"
    assert poison_balance.poison_ratio_in_batch == pytest.approx(0.20)
    assert poison_balance.loss_weighting == "none"
    assert config.victims.enabled == ("srgnn", "miasrec", "tron")


def test_rank_bucket_cem_population_schedule_validation() -> None:
    config = RankBucketCEMConfig(
        iterations=3,
        population_size=8,
        population_per_iteration=(16, 8, 8),
    )

    assert config.effective_population_schedule == (16, 8, 8)
    assert config.candidate_count == 32

    with pytest.raises(ValueError, match="must not be empty"):
        RankBucketCEMConfig(iterations=3, population_per_iteration=[])
    with pytest.raises(ValueError, match="must not be empty"):
        RankBucketCEMConfig(iterations=3, population_per_iteration=())
    with pytest.raises(ValueError, match="length must equal"):
        RankBucketCEMConfig(iterations=3, population_per_iteration=(16, 8))
    with pytest.raises(ValueError, match="entries must be positive"):
        RankBucketCEMConfig(iterations=3, population_per_iteration=(16, 0, 8))
    with pytest.raises(ValueError, match="entries must be positive"):
        RankBucketCEMConfig(iterations=3, population_per_iteration=(16, -1, 8))


def test_rank_bucket_cem_poison_balance_config_validation() -> None:
    with pytest.raises(ValueError, match="poison_ratio_in_batch must be > 0 and < 1"):
        SurrogateEvalPoisonBalanceConfig(poison_ratio_in_batch=0.0)
    with pytest.raises(ValueError, match="poison_ratio_in_batch must be > 0 and < 1"):
        SurrogateEvalPoisonBalanceConfig(poison_ratio_in_batch=1.0)
    with pytest.raises(ValueError, match="supports only 'fixed_ratio'"):
        SurrogateEvalPoisonBalanceConfig(mode="poison_only")
    with pytest.raises(ValueError, match="supports only 'none'"):
        SurrogateEvalPoisonBalanceConfig(loss_weighting="importance")


def test_fixed_ratio_batch_sizes_are_deterministic_and_clamped() -> None:
    assert _fixed_ratio_batch_sizes(100, 0.20) == (80, 20)
    assert _fixed_ratio_batch_sizes(7, 0.20) == (6, 1)
    assert _fixed_ratio_batch_sizes(7, 0.50) == (3, 4)
    assert _fixed_ratio_batch_sizes(100, 0.001) == (99, 1)
    assert _fixed_ratio_batch_sizes(100, 0.999) == (1, 99)
    with pytest.raises(ValueError, match="batch_size >= 2"):
        _fixed_ratio_batch_sizes(1, 0.20)


def test_availability_groups_and_summary_match_expected_candidate_shapes() -> None:
    states = build_rank_candidate_states(
        [_session(2), _session(3), _session(5)],
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
    )

    assert [state.candidate_count for state in states] == [1, 2, 4]
    assert [state.availability_group for state in states] == ["G1", "G2", "G3"]
    assert states[2].tail_positions == (3, 4)

    summary = build_availability_summary(states)
    assert summary["G1_count"] == 1
    assert summary["G2_count"] == 1
    assert summary["G3_count"] == 1
    assert summary["pos0_removed_count"] == 3


def test_rank_policy_obeys_rank2_only_and_tail_only_policies() -> None:
    states = build_rank_candidate_states(
        [_session(2), _session(3), _session(5)],
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
    )

    selected_positions, records = sample_positions_from_rank_policy(
        states,
        RankBucketPolicy(pi_g2=(0.0, 1.0), pi_g3=(0.0, 1.0, 0.0)),
        target_item=5334,
        rng=random.Random(7),
    )
    assert selected_positions == [1, 2, 2]
    assert [record.selected_rank for record in records] == ["rank1", "rank2", "rank2"]

    tail_positions, tail_records = sample_positions_from_rank_policy(
        [states[2]],
        RankBucketPolicy(pi_g2=(0.5, 0.5), pi_g3=(0.0, 0.0, 1.0)),
        target_item=5334,
        rng=random.Random(13),
    )
    assert tail_positions[0] in {3, 4}
    assert tail_records[0].selected_rank == "tail"


def test_rank_policy_sampling_is_reproducible_for_same_seed() -> None:
    states = build_rank_candidate_states(
        [_session(length) for length in range(5, 15)],
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
    )
    policy = RankBucketPolicy(pi_g2=(0.25, 0.75), pi_g3=(0.2, 0.3, 0.5))

    def collect(seed: int) -> list[int]:
        positions, _ = sample_positions_from_rank_policy(
            states,
            policy,
            target_item=5334,
            rng=random.Random(seed),
        )
        return positions

    assert collect(20260405) == collect(20260405)
    assert collect(1) != collect(2)


def test_cem_optimizer_updates_from_elites_with_smoothing_and_min_std() -> None:
    state = initialize_cem_state(1.0)
    results = [
        CEMCandidateResult(
            candidate=CEMCandidate(
                iteration=0,
                candidate_id=0,
                logits_g2=(4.0, 0.0),
                logits_g3=(3.0, 1.0, 0.0),
                pi_g2=(0.5, 0.5),
                pi_g3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            ),
            reward=10.0,
            metrics={},
        ),
        CEMCandidateResult(
            candidate=CEMCandidate(
                iteration=0,
                candidate_id=1,
                logits_g2=(2.0, 0.0),
                logits_g3=(1.0, 1.0, 0.0),
                pi_g2=(0.5, 0.5),
                pi_g3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            ),
            reward=9.0,
            metrics={},
        ),
        CEMCandidateResult(
            candidate=CEMCandidate(
                iteration=0,
                candidate_id=2,
                logits_g2=(-1.0, 0.0),
                logits_g3=(0.0, 0.0, 0.0),
                pi_g2=(0.5, 0.5),
                pi_g3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            ),
            reward=1.0,
            metrics={},
        ),
        CEMCandidateResult(
            candidate=CEMCandidate(
                iteration=0,
                candidate_id=3,
                logits_g2=(-3.0, 0.0),
                logits_g3=(-2.0, 0.0, 0.0),
                pi_g2=(0.5, 0.5),
                pi_g3=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            ),
            reward=0.5,
            metrics={},
        ),
    ]

    updated = update_cem_state(
        state,
        results,
        elite_ratio=0.5,
        smoothing=0.25,
        min_std=0.8,
    )

    assert updated.mean_g2 == pytest.approx([0.75, 0.0])
    assert updated.mean_g3 == pytest.approx([0.5, 0.25, 0.0])
    assert updated.std_g2 == pytest.approx([1.0, 0.8])
    assert updated.std_g3 == pytest.approx([1.0, 0.8, 0.8])


def test_reward_metric_null_uses_target_result_mean() -> None:
    result = SurrogateScoreResult.from_values(
        [0.9, 0.7],
        metrics={"custom_metric@7": 0.6},
    )

    assert _resolve_reward_value(target_result=result, reward_metric=None) == pytest.approx(0.8)


def test_arbitrary_available_reward_metric_is_supported_without_lowk_whitelist() -> None:
    result = SurrogateScoreResult.from_values(
        [0.8],
        metrics={"custom_metric@7": 0.6, "another_metric@11": 0.4},
    )

    assert _resolve_reward_value(
        target_result=result,
        reward_metric="custom_metric@7",
    ) == pytest.approx(0.6)


def test_missing_reward_metric_raises_clear_error_with_available_keys() -> None:
    result = SurrogateScoreResult.from_values(
        [0.8],
        metrics={"custom_metric@7": 0.6, "another_metric@11": 0.4},
    )

    with pytest.raises(ValueError, match="Available metric keys"):
        _resolve_reward_value(
            target_result=result,
            reward_metric="missing_metric@30",
        )


def test_normalized_lowk_reward_metric_uses_raw_family_value_for_baseline() -> None:
    result = SurrogateScoreResult.from_values(
        [0.8],
        metrics={
            "targeted_mrr@10": 0.10,
            "targeted_mrr@20": 0.20,
            "targeted_recall@10": 0.30,
            "targeted_recall@20": 0.50,
        },
    )

    assert _resolve_reward_value(
        target_result=result,
        reward_metric=NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD,
    ) == pytest.approx(0.275)


def test_normalized_lowk_reward_missing_metrics_reports_required_and_available_keys() -> None:
    result = SurrogateScoreResult.from_values(
        [0.8],
        metrics={
            "targeted_mrr@10": 0.10,
            "targeted_recall@10": 0.30,
        },
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_reward_value(
            target_result=result,
            reward_metric=NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD,
        )

    message = str(exc_info.value)
    assert "required_keys" in message
    assert "available_keys" in message
    assert "targeted_mrr@20" in message


def test_iteration_normalized_lowk_reward_computes_per_iteration_values() -> None:
    results = [
        _lowk_candidate_result(
            iteration=0,
            candidate_id=0,
            mrr10=1.0,
            mrr20=2.0,
            recall10=4.0,
            recall20=8.0,
        ),
        _lowk_candidate_result(
            iteration=0,
            candidate_id=1,
            mrr10=0.5,
            mrr20=1.0,
            recall10=2.0,
            recall20=4.0,
        ),
    ]

    normalized = _with_iteration_normalized_lowk_rewards(results)

    assert normalized[0].reward == pytest.approx(1.0)
    assert normalized[0].metrics["iteration_normalized_lowk_reward"] == pytest.approx(1.0)
    assert normalized[1].reward == pytest.approx(0.5)
    assert normalized[1].metrics["iteration_normalized_lowk_reward"] == pytest.approx(0.5)


def test_global_normalized_lowk_reward_recomputes_across_all_candidates() -> None:
    results = [
        _lowk_candidate_result(
            iteration=0,
            candidate_id=0,
            mrr10=1.0,
            mrr20=1.0,
            recall10=1.0,
            recall20=1.0,
            reward=1.0,
        ),
        _lowk_candidate_result(
            iteration=1,
            candidate_id=0,
            mrr10=2.0,
            mrr20=4.0,
            recall10=2.0,
            recall20=4.0,
            reward=0.25,
        ),
    ]

    normalized = _with_global_normalized_lowk_rewards(results)

    assert normalized[0].reward == pytest.approx(1.0)
    assert normalized[0].metrics["global_normalized_lowk_reward"] == pytest.approx(0.375)
    assert normalized[1].reward == pytest.approx(0.25)
    assert normalized[1].metrics["global_normalized_lowk_reward"] == pytest.approx(1.0)


def test_normalized_lowk_reward_zero_metric_max_stays_zero() -> None:
    results = [
        _lowk_candidate_result(
            iteration=0,
            candidate_id=0,
            mrr10=0.0,
            mrr20=0.0,
            recall10=0.0,
            recall20=0.0,
        ),
        _lowk_candidate_result(
            iteration=0,
            candidate_id=1,
            mrr10=0.0,
            mrr20=0.0,
            recall10=0.0,
            recall20=0.0,
        ),
    ]

    iteration_normalized = _with_iteration_normalized_lowk_rewards(results)
    global_normalized = _with_global_normalized_lowk_rewards(results)

    assert [result.reward for result in iteration_normalized] == pytest.approx([0.0, 0.0])
    assert [
        result.metrics["global_normalized_lowk_reward"]
        for result in global_normalized
    ] == pytest.approx([0.0, 0.0])


def test_srg_nn_score_target_required_metrics_include_mrr20() -> None:
    assert "targeted_mrr@20" in _TARGET_SCORE_REQUIRED_KEYS


def test_nonzero_candidate_setting_never_selects_position0() -> None:
    states = build_rank_candidate_states(
        [_session(3), _session(5), _session(8)],
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
    )
    positions, records = sample_positions_from_rank_policy(
        states,
        RankBucketPolicy(pi_g2=(0.5, 0.5), pi_g3=(0.2, 0.3, 0.5)),
        target_item=5334,
        rng=random.Random(17),
    )

    assert all(position > 0 for position in positions)
    assert all(record.selected_position > 0 for record in records)
    assert build_rank_position_summary(records)["pos0_pct"] == pytest.approx(0.0)


def test_cem_trace_omits_candidate_selected_positions_by_default_and_keeps_seed_fixed() -> None:
    tmp_path = _repo_temp_dir()
    try:
        result, artifact_paths, inner_trainer = _run_dummy_trainer(tmp_path)
        trace_rows = _read_jsonl(artifact_paths.cem_trace)
        best_policy = json.loads(artifact_paths.cem_best_policy.read_text(encoding="utf-8"))
        position_summary = json.loads(
            artifact_paths.final_position_summary.read_text(encoding="utf-8")
        )

        assert result["best_reward"] == pytest.approx(0.8)
        assert trace_rows
        assert all("selected_positions" not in row for row in trace_rows)
        assert len({row["surrogate_train_seed"] for row in trace_rows}) == 1
        assert len(set(seed for seed in inner_trainer.seeds if seed is not None)) == 1
        assert all(row["poison_balance_enabled"] is False for row in trace_rows)
        assert all(row["clean_examples_seen"] is None for row in trace_rows)
        assert all(row["poison_examples_seen"] is None for row in trace_rows)
        assert all(row["actual_optimizer_steps"] == 1 for row in trace_rows)
        assert all(row["fine_tune_seconds"] >= 0.0 for row in trace_rows)
        assert all(row["score_target_seconds"] >= 0.0 for row in trace_rows)
        assert all(row["candidate_total_seconds"] >= 0.0 for row in trace_rows)
        assert not artifact_paths.final_selected_positions.exists()
        assert artifact_paths.optimized_poisoned_sessions.exists()
        assert artifact_paths.final_position_summary.exists()
        assert position_summary["total_fake_sessions"] == 3
        assert best_policy["method_name"] == "rank_bucket_cem"
        assert best_policy["method_version"] == RANK_BUCKET_CEM_IMPLEMENTATION_TAG
        assert best_policy["replay_metadata"]["final_selection_seed"] == result["best_selection_seed"]
        assert best_policy["replay_metadata"]["replacement_topk_ratio"] == pytest.approx(1.0)
        assert best_policy["replay_metadata"]["nonzero_action_when_possible"] is True
        assert best_policy["replay_metadata"]["position_opt_seed"] == 20260405
        assert best_policy["replay_metadata"]["surrogate_train_seed"] == trace_rows[0][
            "surrogate_train_seed"
        ]
        assert best_policy["replay_metadata"]["validation_subset_strategy"] == (
            "deterministic_random_subset"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_population_per_iteration_controls_trace_rows_elites_and_global_ids() -> None:
    tmp_path = _repo_temp_dir()
    try:
        result, artifact_paths, _ = _run_dummy_trainer(
            tmp_path,
            iterations=3,
            population_size=8,
            population_per_iteration=(16, 8, 8),
        )
        trace_rows = _read_jsonl(artifact_paths.cem_trace)
        best_policy = json.loads(artifact_paths.cem_best_policy.read_text(encoding="utf-8"))

        assert result["effective_population_schedule"] == [16, 8, 8]
        assert result["candidate_count"] == 32
        assert len(trace_rows) == 32
        assert [sum(1 for row in trace_rows if row["iteration"] == i) for i in range(3)] == [
            16,
            8,
            8,
        ]
        assert [
            sum(
                1
                for row in trace_rows
                if row["iteration"] == i and row["selected_as_iteration_elite"]
            )
            for i in range(3)
        ] == [4, 2, 2]
        assert [row["global_candidate_id"] for row in trace_rows] == list(range(32))
        assert all(row["candidate_id"] == row["candidate_id_in_iteration"] for row in trace_rows)
        assert result["final_selected_global_candidate_id"] == 0
        assert best_policy["final_selected_global_candidate_id"] == 0
        assert best_policy["replay_metadata"]["final_selected_global_candidate_id"] == 0
        assert best_policy["cem_hyperparameters"]["effective_population_schedule"] == [
            16,
            8,
            8,
        ]
        assert best_policy["cem_hyperparameters"]["candidate_count"] == 32
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_fixed_ratio_poison_balance_records_trace_metadata_and_skips_inner_trainer() -> None:
    tmp_path = _repo_temp_dir()
    try:
        result, artifact_paths, inner_trainer = _run_dummy_trainer(
            tmp_path,
            surrogate_eval_poison_balance=SurrogateEvalPoisonBalanceConfig(
                enabled=True,
                mode="fixed_ratio",
                poison_ratio_in_batch=0.20,
                loss_weighting="none",
            ),
        )
        trace_rows = _read_jsonl(artifact_paths.cem_trace)
        best_policy = json.loads(artifact_paths.cem_best_policy.read_text(encoding="utf-8"))

        assert result["best_reward"] == pytest.approx(0.8)
        assert inner_trainer.seeds == []
        assert trace_rows
        assert all(row["poison_balance_enabled"] is True for row in trace_rows)
        assert all(row["poison_balance_mode"] == "fixed_ratio" for row in trace_rows)
        assert all(row["requested_poison_ratio_in_batch"] == pytest.approx(0.20) for row in trace_rows)
        assert all(row["configured_clean_batch_size"] == 80 for row in trace_rows)
        assert all(row["configured_poison_batch_size"] == 20 for row in trace_rows)
        assert all(row["actual_optimizer_steps"] == 1 for row in trace_rows)
        assert all(row["clean_examples_seen"] == 80 for row in trace_rows)
        assert all(row["poison_examples_seen"] == 20 for row in trace_rows)
        assert all(row["unique_poison_prefix_examples_seen"] == 7 for row in trace_rows)
        assert all(row["unique_poison_source_sessions_seen"] == 3 for row in trace_rows)
        assert all(row["sampling_strategy"] == "shuffled_cycling" for row in trace_rows)
        assert all(row["sampling_wrapped"] is True for row in trace_rows)
        assert all(row["fine_tune_seconds"] >= 0.0 for row in trace_rows)
        assert all(row["score_target_seconds"] >= 0.0 for row in trace_rows)
        assert all(row["candidate_total_seconds"] >= 0.0 for row in trace_rows)
        assert best_policy["replay_metadata"]["surrogate_eval_poison_balance"] == {
            "enabled": True,
            "mode": "fixed_ratio",
            "poison_ratio_in_batch": 0.20,
            "loss_weighting": "none",
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_fixed_ratio_poison_balance_rejects_empty_poison_prefix_pool() -> None:
    tmp_path = _repo_temp_dir()
    try:
        with pytest.raises(ValueError, match="non-empty poison pool"):
            _run_dummy_trainer(
                tmp_path,
                surrogate_eval_poison_balance=SurrogateEvalPoisonBalanceConfig(
                    enabled=True,
                    poison_ratio_in_batch=0.20,
                ),
                fake_sessions=[_session(1)],
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_normalized_lowk_cem_uses_iteration_reward_for_elites_and_global_reward_for_final() -> None:
    tmp_path = _repo_temp_dir()
    target_results = [
        SurrogateScoreResult.from_values(
            [0.10],
            metrics={
                "targeted_mrr@10": 1.0,
                "targeted_mrr@20": 1.0,
                "targeted_recall@10": 1.0,
                "targeted_recall@20": 1.0,
            },
        ),
        SurrogateScoreResult.from_values(
            [0.05],
            metrics={
                "targeted_mrr@10": 0.5,
                "targeted_mrr@20": 0.5,
                "targeted_recall@10": 0.5,
                "targeted_recall@20": 0.5,
            },
        ),
        SurrogateScoreResult.from_values(
            [0.20],
            metrics={
                "targeted_mrr@10": 2.0,
                "targeted_mrr@20": 2.0,
                "targeted_recall@10": 2.0,
                "targeted_recall@20": 2.0,
            },
        ),
        SurrogateScoreResult.from_values(
            [0.01],
            metrics={
                "targeted_mrr@10": 0.1,
                "targeted_mrr@20": 0.1,
                "targeted_recall@10": 0.1,
                "targeted_recall@20": 0.1,
            },
        ),
    ]
    try:
        result, artifact_paths, _ = _run_dummy_trainer(
            tmp_path,
            backend=_DummyBackend(trained_target_results=target_results),
            reward_metric=NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD,
            iterations=2,
            population_size=2,
        )
        trace_rows = _read_jsonl(artifact_paths.cem_trace)
        best_policy = json.loads(artifact_paths.cem_best_policy.read_text(encoding="utf-8"))

        assert result["final_selection_reward_name"] == "global_normalized_lowk_reward"
        assert result["final_selection_reward_value"] == pytest.approx(1.0)
        assert result["best_reward"] == pytest.approx(1.0)
        assert result["best_reward_name"] == "global_normalized_lowk_reward"
        assert result["best_iteration_reward"] == pytest.approx(1.0)
        assert result["best_iteration"] == 1
        assert result["best_candidate_id"] == 0
        assert result["final_selected_global_candidate_id"] == 2
        assert best_policy["final_selection_reward_name"] == "global_normalized_lowk_reward"
        assert best_policy["best_reward_name"] == "global_normalized_lowk_reward"
        assert best_policy["best_iteration_reward"] == pytest.approx(1.0)
        assert best_policy["final_selected_iteration"] == 1
        assert best_policy["final_selected_candidate_id"] == 0
        assert best_policy["final_selected_global_candidate_id"] == 2
        assert best_policy["replay_metadata"]["final_selected_global_candidate_id"] == 2
        assert len(trace_rows) == 4
        assert [row["reward"] for row in trace_rows] == pytest.approx(
            [1.0, 0.5, 1.0, 0.05]
        )
        assert [
            (row["iteration"], row["candidate_id"])
            for row in trace_rows
            if row["selected_as_iteration_elite"]
        ] == [(0, 0), (1, 0)]
        assert [
            (row["iteration"], row["candidate_id"], row["global_candidate_id"])
            for row in trace_rows
            if row.get("selected_as_global_best")
        ] == [(1, 0, 2)]
        assert all("absolute_raw_family_lowk_reward" in row for row in trace_rows)
        assert all("target_mean_reward" in row for row in trace_rows)
        assert all("reward_components" in row for row in trace_rows)
        assert all("targeted_mrr@10" in row for row in trace_rows)
        assert all("targeted_mrr@20" in row for row in trace_rows)
        assert all("targeted_recall@10" in row for row in trace_rows)
        assert all("targeted_recall@20" in row for row in trace_rows)
        assert all("global_normalized_lowk_reward" in row for row in trace_rows)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_cem_trace_can_include_candidate_selected_positions_when_enabled() -> None:
    tmp_path = _repo_temp_dir()
    try:
        _, artifact_paths, _ = _run_dummy_trainer(
            tmp_path,
            save_candidate_selected_positions=True,
        )
        trace_rows = _read_jsonl(artifact_paths.cem_trace)

        assert trace_rows
        assert all("selected_positions" in row for row in trace_rows)
        assert len(trace_rows[0]["selected_positions"]) == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_final_selected_positions_are_written_only_when_enabled() -> None:
    tmp_root = _repo_temp_dir()
    try:
        _, disabled_paths, _ = _run_dummy_trainer(tmp_root / "disabled")
        _, enabled_paths, _ = _run_dummy_trainer(
            tmp_root / "enabled",
            save_final_selected_positions=True,
        )

        assert not disabled_paths.final_selected_positions.exists()
        assert enabled_paths.final_selected_positions.exists()
        assert enabled_paths.final_position_summary.exists()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def test_replay_metadata_and_reward_metric_are_saved_for_metric_based_runs() -> None:
    tmp_path = _repo_temp_dir()
    try:
        result, artifact_paths, _ = _run_dummy_trainer(
            tmp_path,
            reward_metric="custom_metric@7",
        )
        best_policy = json.loads(artifact_paths.cem_best_policy.read_text(encoding="utf-8"))

        assert result["best_reward"] == pytest.approx(0.6)
        assert result["selected_reward_metric_name"] == "custom_metric@7"
        assert best_policy["selected_reward_metric_name"] == "custom_metric@7"
        assert best_policy["replay_metadata"]["cem_hyperparameters"]["reward_metric"] == (
            "custom_metric@7"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_rank_bucket_cem_identity_changes_without_touching_shared_fake_session_cache() -> None:
    config = _config()
    baseline_context = _identity_context(config)
    assert (
        "surrogate_eval_poison_balance"
        not in baseline_context["position_opt"]["rank_bucket_cem"]
    )
    assert baseline_context["position_opt"]["rank_bucket_cem"][
        "effective_population_schedule"
    ] == [8, 8, 8]
    assert baseline_context["position_opt"]["rank_bucket_cem"]["candidate_count"] == 24

    tweaked_cem = replace(config.attack.rank_bucket_cem, iterations=5)
    updated = replace(config, attack=replace(config.attack, rank_bucket_cem=tweaked_cem))
    updated_context = _identity_context(updated)

    assert attack_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != attack_key(
        updated,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=updated_context,
    )
    assert run_group_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != run_group_key(
        updated,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=updated_context,
    )
    assert victim_prediction_key(
        config,
        "srgnn",
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != victim_prediction_key(
        updated,
        "srgnn",
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=updated_context,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    ) == shared_attack_artifact_key(
        updated,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    )

    balanced_cem = replace(
        config.attack.rank_bucket_cem,
        surrogate_eval_poison_balance=SurrogateEvalPoisonBalanceConfig(
            enabled=True,
            poison_ratio_in_batch=0.20,
        ),
    )
    balanced = replace(config, attack=replace(config.attack, rank_bucket_cem=balanced_cem))
    balanced_context = _identity_context(balanced)
    assert (
        balanced_context["position_opt"]["rank_bucket_cem"][
            "surrogate_eval_poison_balance"
        ]["enabled"]
        is True
    )
    assert attack_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != attack_key(
        balanced,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=balanced_context,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    ) == shared_attack_artifact_key(
        balanced,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    )

    lowk_cem = replace(
        config.attack.rank_bucket_cem,
        reward_metric=NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD,
    )
    lowk = replace(config, attack=replace(config.attack, rank_bucket_cem=lowk_cem))
    lowk_context = _identity_context(lowk)
    assert lowk_context["position_opt"]["rank_bucket_cem"]["reward_metric"] == (
        NORMALIZED_LOWK_MRR_RECALL_10_20_REWARD
    )
    assert attack_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != attack_key(
        lowk,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=lowk_context,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    ) == shared_attack_artifact_key(
        lowk,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    )

    scheduled_cem = replace(
        config.attack.rank_bucket_cem,
        population_per_iteration=(16, 8, 8),
    )
    scheduled = replace(
        config,
        attack=replace(config.attack, rank_bucket_cem=scheduled_cem),
    )
    scheduled_context = _identity_context(scheduled)
    assert scheduled_context["position_opt"]["rank_bucket_cem"][
        "effective_population_schedule"
    ] == [16, 8, 8]
    assert scheduled_context["position_opt"]["rank_bucket_cem"]["candidate_count"] == 32
    assert attack_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=baseline_context,
    ) != attack_key(
        scheduled,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        attack_identity_context=scheduled_context,
    )
    assert shared_attack_artifact_key(
        config,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    ) == shared_attack_artifact_key(
        scheduled,
        run_type=POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    )


def test_optimized_poisoned_sessions_default_artifact_is_valid_pickle() -> None:
    tmp_path = _repo_temp_dir()
    try:
        _, artifact_paths, _ = _run_dummy_trainer(tmp_path)

        with artifact_paths.optimized_poisoned_sessions.open("rb") as handle:
            payload = pickle.load(handle)

        assert isinstance(payload, list)
        assert len(payload) == 3
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
