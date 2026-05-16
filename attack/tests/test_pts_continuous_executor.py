from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.continuous_executor import (
    CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX,
    CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX,
    CONTINUOUS_ACTION_STOP,
    PTSContinuousSessionContext,
    apply_pts_continuous_beta_construction_batch,
    build_continuous_shared_session_contexts,
)
from attack.pts.continuous_policy import (
    ContinuousBetaPolicy,
    build_suffix_length_percentile_lookup,
)


def _context(residual_suffix: list[int]) -> PTSContinuousSessionContext:
    template = [1, 2] + list(residual_suffix)
    return PTSContinuousSessionContext(
        fake_session_index=0,
        template_session=template,
        anchor_position=2,
        prefix=[1, 2],
        residual_suffix=list(residual_suffix),
        suffix_length_percentile=0.5,
    )


def test_continuous_executor_stop_branch(monkeypatch) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 1.0)
    result = apply_pts_continuous_beta_construction_batch(
        session_contexts=[_context([3, 4])],
        target_item=99,
        policy=ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0]),
        base_seed=7,
        candidate_key="iter0_cand0",
    )

    assert result.final_sessions == [[1, 2, 99]]
    record = result.per_session_records[0]
    assert record["action"] == CONTINUOUS_ACTION_STOP
    assert record["consume_count"] == 2
    assert record["continuation_source"] == "stop"
    assert record["target_position_final"] == 2
    assert result.summary["action_counts"][CONTINUOUS_ACTION_STOP] == 1


def test_continuous_executor_preserve_branch(monkeypatch) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 0.25)
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 1.0,
    )
    result = apply_pts_continuous_beta_construction_batch(
        session_contexts=[_context([3, 4, 5, 6])],
        target_item=99,
        policy=ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0]),
        base_seed=7,
        candidate_key="iter0_cand1",
    )

    assert result.final_sessions == [[1, 2, 99, 4, 5, 6]]
    record = result.per_session_records[0]
    assert record["action"] == CONTINUOUS_ACTION_PARTIAL_KEEP_SUFFIX
    assert record["consume_count"] == 1
    assert record["generated_suffix_length"] == 0


def test_continuous_executor_generate_branch_uses_remaining_length(monkeypatch) -> None:
    monkeypatch.setattr("attack.pts.continuous_executor.sample_beta", lambda *a, **k: 0.25)
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 0.0,
    )

    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        return [700 + index for index in range(int(suffix_length))]

    monkeypatch.setattr(
        "attack.pts.continuous_executor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )
    result = apply_pts_continuous_beta_construction_batch(
        session_contexts=[_context([3, 4, 5, 6])],
        target_item=99,
        policy=ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0]),
        base_seed=7,
        candidate_key="iter0_cand2",
        poison_runner=object(),
        generation_topk=10,
    )

    assert result.final_sessions == [[1, 2, 99, 700, 701, 702]]
    record = result.per_session_records[0]
    assert record["action"] == CONTINUOUS_ACTION_PARTIAL_GENERATE_SUFFIX
    assert record["consume_count"] == 1
    assert record["generated_suffix_length"] == 3
    assert result.summary["continuous"]["generated_source_ratio"] == 1.0


def test_shared_contexts_are_deterministic_and_q_is_run_relative() -> None:
    sessions = [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12]]
    first = build_continuous_shared_session_contexts(
        template_sessions=sessions,
        target_item=99,
        base_seed=123,
    )
    second = build_continuous_shared_session_contexts(
        template_sessions=sessions,
        target_item=99,
        base_seed=123,
    )

    assert first == second
    lengths = [context.residual_suffix_length for context in first]
    lookup = build_suffix_length_percentile_lookup(lengths)
    assert [context.suffix_length_percentile for context in first] == [
        lookup[length] for length in lengths
    ]
