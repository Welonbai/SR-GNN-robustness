from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.artifacts import write_pts_cem_artifacts
from attack.pts.cem import (
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSGroupedCEMTrainer,
)
from attack.pts.policy import CONSUME_ONE_ACTION_NAMES
from attack.pts.specs import get_default_pts_v1_specs


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
        metadata={"candidate_seed": int(kwargs["candidate_seed"])},
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
        assert "sample_origin" in top_candidates["candidates"][0]
        assert "sample_metadata" in top_candidates["candidates"][0]
        assert "policy" in top_candidates["candidates"][0]
        assert "parent_candidate_key" in top_candidates["candidates"][0]
        assert top_policies["candidates"][0]["candidate_key"] == best_key
        assert "sample_metadata" in top_policies["candidates"][0]
        assert "policy" in top_policies["candidates"][0]
        assert rank1_metadata["candidate_key"] == best_key
        assert rank1_metadata["selected_as_global_best"] is True
        assert "sample_origin" in rank1_metadata
        assert "sample_metadata" in rank1_metadata
        assert "policy" in rank1_metadata
        assert "parent_candidate_key" in rank1_metadata
        assert isinstance(rank1_sessions, list)
        assert len(rank1_records) == len(rank1_sessions)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
