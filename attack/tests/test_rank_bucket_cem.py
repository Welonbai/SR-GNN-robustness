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
    RANK_BUCKET_CEM_IMPLEMENTATION_TAG,
    RankBucketCEMTrainer,
    _resolve_reward_value,
)
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
    def __init__(self) -> None:
        self.loaded_paths: list[str] = []
        self.fixed_ratio_calls: list[dict[str, object]] = []
        self.clean_target_result = SurrogateScoreResult.from_values(
            [0.2],
            metrics={
                "custom_metric@7": 0.1,
                "another_metric@11": 0.05,
            },
        )
        self.trained_target_result = SurrogateScoreResult.from_values(
            [0.8],
            metrics={
                "custom_metric@7": 0.6,
                "another_metric@11": 0.4,
            },
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
        return self.trained_target_result

    def score_gt(self, model, eval_sessions, ground_truth_items) -> SurrogateScoreResult:
        return SurrogateScoreResult.from_values([0.55])


class _DummyInnerTrainer:
    def __init__(self) -> None:
        self.seeds: list[int | None] = []

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
        return InnerTrainResult(
            model={
                "kind": "trained",
                "seed": seed,
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
    reward_mode: str = "poisoned_target_utility",
    reward_metric: str | None = None,
    surrogate_eval_poison_balance: SurrogateEvalPoisonBalanceConfig | None = None,
    save_candidate_selected_positions: bool = False,
    save_final_selected_positions: bool = False,
    save_optimized_poisoned_sessions: bool = True,
    save_replay_metadata: bool = True,
    fake_sessions: list[list[int]] | None = None,
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
        iterations=1,
        population_size=3,
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

    backend = _DummyBackend()
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


def test_rank_bucket_cem_config_loads_with_clean_defaults() -> None:
    config = _config()

    assert config.attack.position_opt is not None
    assert config.attack.position_opt.nonzero_action_when_possible is True
    assert config.attack.position_opt.enable_gt_penalty is False
    assert config.attack.rank_bucket_cem is not None
    assert config.attack.rank_bucket_cem.iterations == 3
    assert config.attack.rank_bucket_cem.population_size == 8
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
