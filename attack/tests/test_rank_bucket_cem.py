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

from attack.common.config import RankBucketCEMConfig, load_config
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
    save_candidate_selected_positions: bool = False,
    save_final_selected_positions: bool = False,
    save_optimized_poisoned_sessions: bool = True,
    save_replay_metadata: bool = True,
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
        [_session(2), _session(3), _session(5)],
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
    assert config.victims.enabled == ("srgnn",)


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
