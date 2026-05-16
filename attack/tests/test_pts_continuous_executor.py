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
    compute_half_up_consume_count,
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


def test_half_up_consume_count_boundaries_and_clamp() -> None:
    assert compute_half_up_consume_count(0.125, 4) == 1
    assert compute_half_up_consume_count(0.625, 4) == 3
    assert compute_half_up_consume_count(-0.25, 4) == 0
    assert compute_half_up_consume_count(1.25, 4) == 4


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
    assert record["source_generate_probability"] is None
    assert record["target_position_final"] == 2
    assert result.summary["action_counts"][CONTINUOUS_ACTION_STOP] == 1
    assert result.summary["continuous"]["source_generate_probability_count"] == 0


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


def test_continuous_summary_excludes_stop_from_source_probability_mean(monkeypatch) -> None:
    rho_values = iter([1.0, 0.0])
    monkeypatch.setattr(
        "attack.pts.continuous_executor.sample_beta",
        lambda *args, **kwargs: next(rho_values),
    )
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 1.0,
    )
    result = apply_pts_continuous_beta_construction_batch(
        session_contexts=[
            _context([3, 4]),
            PTSContinuousSessionContext(
                fake_session_index=1,
                template_session=[10, 11, 12, 13],
                anchor_position=2,
                prefix=[10, 11],
                residual_suffix=[12, 13],
                suffix_length_percentile=0.5,
            ),
        ],
        target_item=99,
        policy=ContinuousBetaPolicy.from_vector([0, 0, 0, 0, 0, 0, 0]),
        base_seed=7,
        candidate_key="iter0_cand1",
    )

    assert result.per_session_records[0]["action"] == CONTINUOUS_ACTION_STOP
    assert result.per_session_records[0]["source_generate_probability"] is None
    assert result.per_session_records[1]["source_generate_probability"] == 0.5
    assert result.summary["action_counts"][CONTINUOUS_ACTION_STOP] == 1
    assert result.summary["continuous"]["source_generate_probability_count"] == 1
    assert result.summary["continuous"]["source_generate_probability_mean"] == 0.5
    assert (
        result.summary["continuous"]["source_generate_probability_mean_non_stop"]
        == 0.5
    )


def test_continuous_executor_applies_smoothing_and_keeps_stop_probability_none(
    monkeypatch,
) -> None:
    rho_values = iter([1.0, 0.0])
    monkeypatch.setattr(
        "attack.pts.continuous_executor.sample_beta",
        lambda *args, **kwargs: next(rho_values),
    )
    monkeypatch.setattr(
        "attack.pts.continuous_executor.deterministic_unit_interval",
        lambda **kwargs: 1.0,
    )
    result = apply_pts_continuous_beta_construction_batch(
        session_contexts=[
            _context([3, 4]),
            PTSContinuousSessionContext(
                fake_session_index=1,
                template_session=[10, 11, 12, 13],
                anchor_position=2,
                prefix=[10, 11],
                residual_suffix=[12, 13],
                suffix_length_percentile=0.5,
            ),
        ],
        target_item=99,
        policy=ContinuousBetaPolicy.from_vector(
            [0, 0, 0, 0, 100, 0, 0],
            parameter_bounds=(-200.0, 200.0),
            smoothing_epsilon=0.1,
        ),
        base_seed=7,
        candidate_key="iter0_cand1",
    )

    stop_record, non_stop_record = result.per_session_records
    assert stop_record["source_generate_probability"] is None
    assert stop_record["smoothing_epsilon"] == 0.1
    assert non_stop_record["source_generate_probability"] == 0.9
    assert 0.1 <= non_stop_record["source_generate_probability"] <= 0.9
    assert non_stop_record["consume_smoothing"] == "beta_uniform_mixture"


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
