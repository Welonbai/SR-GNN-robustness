from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json
from attack.pts.artifacts import write_pts_cem_artifacts
from attack.pts.cem import PTSCEMCandidateResult, PTSCEMIterationResult, PTSCEMResult
from attack.pts.continuous_policy import ContinuousBetaPolicy


def test_continuous_policy_artifacts_write_rank1_sessions(tmp_path: Path) -> None:
    policy = ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0])
    candidate = PTSCEMCandidateResult(
        iteration=0,
        candidate_id=0,
        candidate_seed=123,
        policy=policy,
        reward=1.0,
        reward_metrics={"score": 1.0},
        evaluator_metadata={},
        construction_summary={"action_counts": {"continuous_stop": 1}},
        per_session_records=[],
        final_sessions=[[1, 99]],
        selected_as_global_best=True,
    )
    result = PTSCEMResult(
        best_candidate=candidate,
        final_policy=policy,
        policy_history=[policy.to_dict()],
        iteration_results=[
            PTSCEMIterationResult(
                iteration=0,
                population_size=1,
                elite_count=1,
                candidates=[candidate],
                elite_candidate_keys=[candidate.candidate_key],
                policy_before=policy.to_dict(),
                policy_after=policy.to_dict(),
            )
        ],
        top_candidates=[candidate],
    )

    paths = write_pts_cem_artifacts(result=result, output_dir=tmp_path)

    assert Path(paths["top_candidate_rank_1_sessions"]).exists()
    policy_payload = load_json(Path(paths["top_candidate_rank_1_policy"]))
    assert policy_payload["type"] == "continuous_beta_policy"
