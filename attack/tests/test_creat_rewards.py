from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.creat.reward_table import build_v2_reward_table
from attack.creat.rewards_v2 import (
    RAW_REWARD_COMPONENTS,
    compose_v2_reward,
    compute_dpp_scores,
    compute_v2_raw_reward_components,
    reward_component_statistics,
)


class _Adapter:
    embedding_dim = 2

    def encode_session(self, session):
        return torch.tensor([float(sum(session)), float(len(session))])

    def encode_sessions(self, sessions):
        return torch.stack([self.encode_session(session) for session in sessions])

    def target_embedding(self, target_item):
        return torch.tensor([1.0, 0.0])

    def target_score_for_prefix(self, prefix, target_item):
        return float(sum(prefix))

    def valid_position_mask(self, session, target_item, topk_ratio, nonzero_when_possible=True):
        return torch.tensor([False] + [item != target_item for item in session[1:]])


class _CountingAdapter(_Adapter):
    def __init__(self):
        self.encode_sessions_calls = 0
        self.target_scores_calls = 0

    def encode_sessions(self, sessions):
        self.encode_sessions_calls += 1
        return super().encode_sessions(sessions)

    def target_scores_for_prefixes(self, prefixes, target_item):
        self.target_scores_calls += 1
        return [self.target_score_for_prefix(prefix, target_item) for prefix in prefixes]


def test_v2_pattern_reward_uses_nonempty_prefix_suffix_mean_cosine_distance() -> None:
    adapter = _Adapter()
    components = compute_v2_raw_reward_components(
        adapter,
        original_session=[1, 2, 3],
        selected_position=1,
        target_item=9,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    target = adapter.target_embedding(9)
    reps = adapter.encode_sessions([[1], [3]])
    expected = torch.mean(
        1.0 - torch.nn.functional.cosine_similarity(target.view(1, -1).expand_as(reps), reps)
    )
    assert components.pattern_reward == pytest.approx(float(expected.item()))
    assert components.pattern_segment_count == 2


def test_v2_local_consistency_skips_without_full_kgram_but_candidate_remains() -> None:
    table = build_v2_reward_table(
        _Adapter(),
        template_sessions=[[1, 2]],
        target_item=9,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    row = table.get(0, 1)
    assert row.local_consistency_reward == 0.0
    assert row.local_affected_kgram_count == 0
    assert row.local_skipped_count == 1
    serialized = table.to_serializable()
    assert serialized["rows"][0]["position"] == 1
    assert all(component in serialized["rows"][0] for component in RAW_REWARD_COMPONENTS)


def test_batched_reward_table_matches_direct_component_computation() -> None:
    adapter = _Adapter()
    table = build_v2_reward_table(
        adapter,
        template_sessions=[[1, 2, 3]],
        target_item=9,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    direct = compute_v2_raw_reward_components(
        adapter,
        original_session=[1, 2, 3],
        selected_position=1,
        target_item=9,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    assert table.get(0, 1).to_dict() == pytest.approx(direct.to_dict())


def test_v2_dpp_and_phase_composition() -> None:
    components = compute_v2_raw_reward_components(
        _Adapter(),
        original_session=[1, 2, 3],
        selected_position=1,
        target_item=9,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    assert math.isfinite(components.dpp_raw_logdet)
    assert 0.0 <= components.dpp_bounded_determinant <= 1.0
    attack = compose_v2_reward(
        components,
        phase="attack",
        pattern_reward_weight=0.1,
        dpp_reward_weight=0.0,
        global_consistency_weight=0.1,
        local_consistency_weight=0.1,
    )
    consistency = compose_v2_reward(
        components,
        phase="consistency",
        pattern_reward_weight=0.1,
        dpp_reward_weight=0.0,
        global_consistency_weight=0.1,
        local_consistency_weight=0.1,
    )
    assert attack != consistency
    assert compose_v2_reward(
        replace(components, dpp_reward=999.0),
        phase="attack",
        pattern_reward_weight=0.1,
        dpp_reward_weight=0.0,
        global_consistency_weight=0.1,
        local_consistency_weight=0.1,
    ) == pytest.approx(attack)


def test_dpp_bounded_determinant_semantics() -> None:
    _raw, identical_bounded, invalid = compute_dpp_scores(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        eps=1.0e-6,
    )
    assert invalid == 0
    assert identical_bounded == pytest.approx(0.0, abs=1.0e-6)

    _raw, orthogonal_bounded, invalid = compute_dpp_scores(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        eps=1.0e-6,
    )
    assert invalid == 0
    assert orthogonal_bounded == pytest.approx(1.0, abs=1.0e-6)

    raw, single_bounded, invalid = compute_dpp_scores(
        torch.tensor([[1.0, 0.0]]),
        eps=1.0e-6,
    )
    assert (raw, single_bounded, invalid) == (0.0, 0.0, 0)


def test_reward_table_reports_candidate_and_selected_composed_stats() -> None:
    table = build_v2_reward_table(
        _Adapter(),
        template_sessions=[[1, 2, 3], [3, 4, 5]],
        target_item=9,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    kwargs = {
        "pattern_reward_weight": 0.1,
        "dpp_reward_weight": 0.0,
        "global_consistency_weight": 0.1,
        "local_consistency_weight": 0.1,
    }
    candidate = table.composed_reward_stats(**kwargs)
    selected = table.composed_reward_stats(selected_positions=[1, 2], **kwargs)
    assert set(candidate) == {"attack", "consistency"}
    assert candidate["attack"]["count"] == len(table.rows)
    assert selected["attack"]["count"] == 2
    assert selected["consistency"]["mean"] != selected["attack"]["mean"]


def test_reward_table_batches_cached_representations_and_prints_progress(capsys) -> None:
    adapter = _CountingAdapter()
    table = build_v2_reward_table(
        adapter,
        template_sessions=[[1, 2, 3], [3, 4, 5]],
        target_item=9,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    output = capsys.readouterr().out
    assert "reward-table stage=representations" in output
    assert "reward-table stage=prefix_scores" in output
    assert "reward-table stage=assemble" in output
    assert adapter.encode_sessions_calls == 1
    assert adapter.target_scores_calls == 1
    assert table.build_metadata["implementation"] == "batched_cached_v1"
    assert table.build_metadata["candidate_count"] == len(table.rows)


def test_reward_table_reuses_target_independent_representations_across_targets() -> None:
    adapter = _CountingAdapter()
    first = build_v2_reward_table(
        adapter,
        template_sessions=[[1, 2, 3], [3, 4, 5]],
        target_item=9,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    second = build_v2_reward_table(
        adapter,
        template_sessions=[[1, 2, 3], [3, 4, 5]],
        target_item=8,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        local_window_size=3,
        dpp_score_mode="bounded_determinant",
        dpp_eps=1.0e-6,
    )
    assert first.build_metadata["shared_representation_cache_hit_count"] == 0
    assert second.build_metadata["shared_representation_cache_hit_count"] > 0
    assert (
        second.build_metadata["shared_representation_cache_miss_count"]
        < second.build_metadata["unique_representation_count"]
    )


def test_reward_statistics_cover_all_raw_components_and_count_nonfinite() -> None:
    rows = [{name: 1.0 for name in RAW_REWARD_COMPONENTS}]
    rows.append({name: (float("nan") if name == "dpp_raw_logdet" else 3.0) for name in RAW_REWARD_COMPONENTS})
    stats = reward_component_statistics(rows)
    assert set(stats) == set(RAW_REWARD_COMPONENTS)
    assert stats["attack_reward"]["mean"] == pytest.approx(2.0)
    assert stats["attack_reward"]["std"] == pytest.approx(1.0)
    assert stats["dpp_raw_logdet"]["count"] == 1
    assert stats["dpp_raw_logdet"]["invalid_count"] == 1
