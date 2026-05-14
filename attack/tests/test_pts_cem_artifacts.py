from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import uuid

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20
from attack.pts.artifacts import (
    build_epoch_reward_ranking_summary,
    write_pts_cem_artifacts,
)
from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSGroupedCEMTrainer,
)
from attack.pts.policy import CONSUME_ONE_ACTION_NAMES
from attack.pts.specs import get_default_pts_v1_specs


EXPECTED_SEED_ALIGNMENT = {
    "target_item": 99,
    "pts_cem_surrogate_seed_alignment_mode": "victim_effective_seed",
    "pts_cem_surrogate_seed_alignment_target_victim_name": "srgnn",
    "configured_surrogate_train_seed": 20260405,
    "configured_victim_train_seed": 20260405,
    "resolved_surrogate_effective_seed": 1386226870,
    "resolved_victim_effective_seed": 1386226870,
    "surrogate_victim_seed_aligned": True,
}


def _patch_generated_suffix(monkeypatch) -> None:
    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        return [200 + index for index in range(int(suffix_length))]

    monkeypatch.setattr(
        "attack.pts.suffix_constructor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )


def _evaluator_fn(**kwargs) -> PTSCEMEvaluationResult:
    policy = kwargs["policy"]
    reward = float(
        policy.group_probabilities["suffix_3plus"]["regenerate_residual_suffix"]
    )
    return PTSCEMEvaluationResult(
        reward=reward,
        reward_metrics={"reward": reward},
        metadata={
            "candidate_seed": int(kwargs["candidate_seed"]),
            **EXPECTED_SEED_ALIGNMENT,
            "pts_cem_surrogate_retrain_checkpoint_protocol": "fixed_last",
            "pts_cem_surrogate_retrain_validation_enabled": False,
            "pts_cem_surrogate_retrain_reward_checkpoint": "last",
            "pts_cem_surrogate_retrain_identity_neutral": True,
            "pts_cem_surrogate_retrain_identity_note": (
                "Surrogate retrain checkpoint protocol is intentionally excluded "
                "from PTS-CEM cache identity."
            ),
            "selected_checkpoint_epoch": 4,
            "selected_checkpoint_protocol": "fixed_last",
            "selected_checkpoint_source": "last_epoch",
            "selected_checkpoint_metric": None,
            "validation_best_metrics_recorded": False,
            "official_reward_checkpoint_epoch": 4,
        },
    )


def _ranking_evaluator_fn(**kwargs) -> PTSCEMEvaluationResult:
    candidate_id = int(kwargs["candidate_id"])
    official_rewards = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
    epoch2_rewards = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    epoch3_rewards = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}
    reward = official_rewards[candidate_id]

    def epoch_payload(value: float) -> dict[str, float]:
        return {
            "target_summary_value": float(value),
            PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20: float(value),
            "targeted_mrr@10": float(value),
            "targeted_mrr@20": float(value),
            "targeted_recall@10": float(value),
            "targeted_recall@20": float(value),
        }

    return PTSCEMEvaluationResult(
        reward=reward,
        reward_metrics={PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20: reward},
        metadata={
            "candidate_seed": int(kwargs["candidate_seed"]),
            **EXPECTED_SEED_ALIGNMENT,
        },
        epoch_reward_diagnostics={
            "enabled": True,
            "reward_name": PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20,
            "diagnostic_epochs": [2, 3],
            "include_final_epoch": True,
            "official_reward_source": "final_partial_retrain_protocol",
            "training_budget_epoch": 4,
            "selected_checkpoint_epoch": 4,
            "epoch_diagnostic_checkpoint_mode": "current_epoch",
            "rewards_by_epoch": {
                "2": epoch_payload(epoch2_rewards[candidate_id]),
                "3": epoch_payload(epoch3_rewards[candidate_id]),
                "4": epoch_payload(reward),
            },
        },
    )


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assert_ragged_policy_payload(policy_payload: dict[str, object]) -> None:
    group_probabilities = policy_payload["group_probabilities"]
    assert isinstance(group_probabilities, dict)
    for action in CONSUME_ONE_ACTION_NAMES:
        assert action not in group_probabilities["suffix_1"]
        assert action in group_probabilities["suffix_2"]
        assert action in group_probabilities["suffix_3plus"]


def _assert_seed_alignment_payload(payload: dict[str, object]) -> None:
    for key, expected in EXPECTED_SEED_ALIGNMENT.items():
        assert payload[key] == expected


def test_pts_cem_artifact_writer_creates_standalone_outputs(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=2,
            population_size=4,
            elite_ratio=0.5,
            base_seed=21,
            save_top_k_candidates=2,
        ),
        specs=get_default_pts_v1_specs(),
    )
    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7], [8, 9]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_evaluator_fn,
    )

    output_dir = REPO_ROOT / "outputs" / f"tmp_pts_cem_artifacts_{uuid.uuid4().hex}"
    try:
        paths = write_pts_cem_artifacts(result=result, output_dir=output_dir)

        expected_files = [
            "pts_cem_trace",
            "pts_policy_history",
            "pts_best_policy",
            "pts_final_policy",
            "pts_top_candidates",
            "pts_top_candidate_policies",
            "top_candidate_rank_1_policy",
            "top_candidate_rank_1_sessions",
            "top_candidate_rank_1_session_records",
            "top_candidate_rank_1_metadata",
        ]
        for key in expected_files:
            assert key in paths
            assert Path(paths[key]).exists()

        trace_rows = _read_jsonl(Path(paths["pts_cem_trace"]))
        assert len(trace_rows) == 8
        assert all("final_sessions" not in row for row in trace_rows)
        assert all("per_session_records" not in row for row in trace_rows)
        assert all(row["sample_origin"] == "global_policy" for row in trace_rows)
        assert all("sample_metadata" in row for row in trace_rows)
        assert all(row["sampled_policy_projection_enabled"] is True for row in trace_rows)
        assert all(row["sampled_policy_min_probability"] == 0.03 for row in trace_rows)
        assert all(row["sampled_policy_max_probability"] == 0.90 for row in trace_rows)
        assert all("parent_candidate_key" in row for row in trace_rows)
        _assert_seed_alignment_payload(trace_rows[0])
        assert trace_rows[0]["pts_cem_surrogate_retrain_checkpoint_protocol"] == (
            "fixed_last"
        )
        assert trace_rows[0]["pts_cem_surrogate_retrain_identity_neutral"] is True
        assert trace_rows[0]["selected_checkpoint_source"] == "last_epoch"

        best_policy = _read_json(Path(paths["pts_best_policy"]))
        final_policy = _read_json(Path(paths["pts_final_policy"]))
        top_candidates = _read_json(Path(paths["pts_top_candidates"]))
        top_policies = _read_json(Path(paths["pts_top_candidate_policies"]))
        rank1_policy = _read_json(Path(paths["top_candidate_rank_1_policy"]))
        rank1_metadata = _read_json(Path(paths["top_candidate_rank_1_metadata"]))
        rank1_sessions = _read_json(Path(paths["top_candidate_rank_1_sessions"]))
        rank1_records = _read_jsonl(Path(paths["top_candidate_rank_1_session_records"]))

        _assert_ragged_policy_payload(best_policy["policy"])
        _assert_ragged_policy_payload(final_policy)
        _assert_ragged_policy_payload(trace_rows[0]["policy"])
        _assert_ragged_policy_payload(rank1_policy)
        assert best_policy["selected_as_global_best"] is True
        best_key = best_policy["candidate_key"]
        assert sum(1 for row in trace_rows if row["selected_as_global_best"]) == 1
        assert next(row for row in trace_rows if row["selected_as_global_best"])[
            "candidate_key"
        ] == best_key
        assert top_candidates["candidates"][0]["candidate_key"] == best_key
        assert top_candidates["candidates"][0]["selected_as_global_best"] is True
        _assert_seed_alignment_payload(top_candidates["candidates"][0])
        assert "sample_origin" in top_candidates["candidates"][0]
        assert "sample_metadata" in top_candidates["candidates"][0]
        assert "policy" in top_candidates["candidates"][0]
        assert "parent_candidate_key" in top_candidates["candidates"][0]
        assert top_policies["candidates"][0]["candidate_key"] == best_key
        _assert_seed_alignment_payload(top_policies["candidates"][0])
        assert best_policy["pts_cem_surrogate_retrain_checkpoint_protocol"] == (
            "fixed_last"
        )
        assert top_candidates["candidates"][0][
            "pts_cem_surrogate_retrain_checkpoint_protocol"
        ] == "fixed_last"
        assert rank1_metadata["pts_cem_surrogate_retrain_checkpoint_protocol"] == (
            "fixed_last"
        )
        assert "sample_metadata" in top_policies["candidates"][0]
        assert "policy" in top_policies["candidates"][0]
        assert rank1_metadata["candidate_key"] == best_key
        assert rank1_metadata["selected_as_global_best"] is True
        _assert_seed_alignment_payload(rank1_metadata)
        _assert_seed_alignment_payload(best_policy)
        assert "sample_origin" in rank1_metadata
        assert "sample_metadata" in rank1_metadata
        assert "policy" in rank1_metadata
        assert "parent_candidate_key" in rank1_metadata
        assert isinstance(rank1_sessions, list)
        assert len(rank1_records) == len(rank1_sessions)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_pts_epoch_reward_diagnostics_artifacts_and_ranking_summary(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=1,
            population_size=4,
            elite_ratio=0.5,
            base_seed=31,
            save_top_k_candidates=2,
        ),
        specs=get_default_pts_v1_specs(),
    )
    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7], [8, 9]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_ranking_evaluator_fn,
    )

    assert result.iteration_results[0].elite_candidate_keys == [
        "iter0_cand0",
        "iter0_cand1",
    ]
    summary = build_epoch_reward_ranking_summary(result)
    iter0_epoch2 = summary["by_iteration"]["0"]["epoch_2"]
    iter0_epoch3 = summary["by_iteration"]["0"]["epoch_3"]

    assert iter0_epoch2["top1_match"] is False
    assert iter0_epoch2["official_elite_candidate_keys"] == [
        "iter0_cand0",
        "iter0_cand1",
    ]
    assert iter0_epoch2["epoch_elite_candidate_keys"] == [
        "iter0_cand3",
        "iter0_cand2",
    ]
    assert iter0_epoch2["elite_overlap_count"] == 0
    assert iter0_epoch2["epoch_best_candidate_key"] == "iter0_cand3"
    assert iter0_epoch2["official_best_candidate_key"] == "iter0_cand0"
    assert iter0_epoch2["epoch_best_official_rank"] == 4
    assert iter0_epoch2["official_best_epoch_rank"] == 4
    assert iter0_epoch2["spearman_vs_official"] == pytest.approx(-1.0)
    assert iter0_epoch2["kendall_tau_vs_official"] == pytest.approx(-1.0)
    assert iter0_epoch3["top1_match"] is True
    assert iter0_epoch3["elite_overlap_count"] == 2

    output_dir = REPO_ROOT / "outputs" / f"tmp_pts_cem_epoch_diag_{uuid.uuid4().hex}"
    try:
        paths = write_pts_cem_artifacts(
            result=result,
            output_dir=output_dir,
            write_candidate_epoch_metrics=True,
            write_epoch_reward_ranking_summary=True,
        )
        assert Path(paths["pts_epoch_reward_ranking_summary_json"]).exists()
        assert Path(paths["pts_epoch_reward_ranking_summary_csv"]).exists()

        trace_rows = _read_jsonl(Path(paths["pts_cem_trace"]))
        assert "epoch_reward_diagnostics" in trace_rows[0]
        assert "2" in trace_rows[0]["epoch_reward_diagnostics"]["rewards_by_epoch"]
        best_policy = _read_json(Path(paths["pts_best_policy"]))
        rank1_metadata = _read_json(Path(paths["top_candidate_rank_1_metadata"]))
        assert "epoch_reward_diagnostics" in best_policy
        assert "epoch_reward_diagnostics" in rank1_metadata
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_pts_epoch_ranking_summary_writes_when_candidate_metrics_disabled(monkeypatch) -> None:
    _patch_generated_suffix(monkeypatch)
    trainer = PTSGroupedCEMTrainer(
        cem_config=PTSCEMConfig(
            iterations=1,
            population_size=4,
            elite_ratio=0.5,
            base_seed=37,
            save_top_k_candidates=2,
        ),
        specs=get_default_pts_v1_specs(),
    )
    result = trainer.train(
        template_sessions=[[1, 2, 3, 4], [5, 6, 7], [8, 9]],
        target_item=99,
        poison_runner=object(),
        evaluator_fn=_ranking_evaluator_fn,
    )
    output_dir = REPO_ROOT / "outputs" / f"tmp_pts_cem_epoch_diag_off_{uuid.uuid4().hex}"
    try:
        paths = write_pts_cem_artifacts(
            result=result,
            output_dir=output_dir,
            write_candidate_epoch_metrics=False,
            write_epoch_reward_ranking_summary=True,
        )

        assert Path(paths["pts_epoch_reward_ranking_summary_json"]).exists()
        assert Path(paths["pts_epoch_reward_ranking_summary_csv"]).exists()
        summary = _read_json(Path(paths["pts_epoch_reward_ranking_summary_json"]))
        assert summary["by_iteration"]["0"]["epoch_2"]["epoch_best_candidate_key"] == (
            "iter0_cand3"
        )

        trace_rows = _read_jsonl(Path(paths["pts_cem_trace"]))
        best_policy = _read_json(Path(paths["pts_best_policy"]))
        top_candidates = _read_json(Path(paths["pts_top_candidates"]))
        rank1_metadata = _read_json(Path(paths["top_candidate_rank_1_metadata"]))
        assert "epoch_reward_diagnostics" not in trace_rows[0]
        assert "epoch_reward_diagnostics" not in best_policy
        assert "epoch_reward_diagnostics" not in top_candidates["candidates"][0]
        assert "epoch_reward_diagnostics" not in rank1_metadata
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
