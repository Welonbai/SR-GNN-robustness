from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from attack.creat.rewards_v2 import (
    CreatV2RawRewardComponents,
    compose_v2_reward,
    compute_dpp_scores,
    reward_component_statistics,
    scalar_reward_statistics,
)


_ENCODE_CHUNK_SIZE = 2048
_SCORE_CHUNK_SIZE = 2048


@dataclass(frozen=True)
class _CandidateSpec:
    template_index: int
    position: int
    original: tuple[int, ...]
    polluted: tuple[int, ...]
    prefix: tuple[int, ...]
    segments: tuple[tuple[int, ...], ...]
    local_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    local_skipped_count: int


@dataclass(frozen=True)
class CreatV2RewardTable:
    rows: dict[tuple[int, int], CreatV2RawRewardComponents]
    candidate_reward_stats: dict[str, dict[str, float | int | None]]
    build_metadata: dict[str, object]

    def get(self, template_index: int, position: int) -> CreatV2RawRewardComponents:
        return self.rows[(int(template_index), int(position))]

    def selected_reward_stats(
        self,
        selected_positions: Sequence[int],
    ) -> dict[str, dict[str, float | int | None]]:
        rows = [
            self.get(index, int(position)).to_dict()
            for index, position in enumerate(selected_positions)
        ]
        return reward_component_statistics(rows)

    def composed_reward_stats(
        self,
        *,
        selected_positions: Sequence[int] | None = None,
        pattern_reward_weight: float,
        dpp_reward_weight: float,
        global_consistency_weight: float,
        local_consistency_weight: float,
    ) -> dict[str, dict[str, float | int | None]]:
        components = (
            list(self.rows.values())
            if selected_positions is None
            else [
                self.get(index, int(position))
                for index, position in enumerate(selected_positions)
            ]
        )
        return {
            phase: scalar_reward_statistics(
                [
                    compose_v2_reward(
                        row,
                        phase=phase,
                        pattern_reward_weight=float(pattern_reward_weight),
                        dpp_reward_weight=float(dpp_reward_weight),
                        global_consistency_weight=float(global_consistency_weight),
                        local_consistency_weight=float(local_consistency_weight),
                    )
                    for row in components
                ]
            )
            for phase in ("attack", "consistency")
        }

    def to_serializable(self) -> dict[str, object]:
        return {
            "schema_version": "creat_v2_raw_reward_table_v2_batched",
            "build_metadata": self.build_metadata,
            "candidate_reward_stats": self.candidate_reward_stats,
            "rows": [
                {
                    "template_index": int(template_index),
                    "position": int(position),
                    **components.to_dict(),
                }
                for (template_index, position), components in sorted(self.rows.items())
            ],
        }


def build_v2_reward_table(
    adapter,
    *,
    template_sessions: Sequence[Sequence[int]],
    target_item: int,
    replacement_topk_ratio: float,
    nonzero_when_possible: bool,
    local_window_size: int,
    dpp_score_mode: str,
    dpp_eps: float,
) -> CreatV2RewardTable:
    started = time.monotonic()
    target = int(target_item)
    print(
        "[CREAT-Additive-SBR v2] "
        f"target={target} reward-table started; templates={len(template_sessions)}",
        flush=True,
    )
    specs = _build_candidate_specs(
        adapter,
        template_sessions=template_sessions,
        target_item=target,
        replacement_topk_ratio=float(replacement_topk_ratio),
        nonzero_when_possible=bool(nonzero_when_possible),
        local_window_size=int(local_window_size),
    )
    representation_keys = _representation_keys(specs)
    shared_representation_keys = _shared_representation_keys(specs)
    prefix_keys = sorted({spec.prefix for spec in specs})
    print(
        "[CREAT-Additive-SBR v2] "
        f"target={target} reward-table candidates ready; candidates={len(specs)}, "
        f"unique_representations={len(representation_keys)}, unique_prefixes={len(prefix_keys)}",
        flush=True,
    )
    representation_cache, representation_cache_metadata = _encode_representation_cache(
        adapter,
        representation_keys,
        shared_keys=shared_representation_keys,
        target_item=target,
    )
    prefix_score_cache = _score_prefix_cache(adapter, prefix_keys, target_item=target)
    target_embedding = adapter.target_embedding(target)
    rows: dict[tuple[int, int], CreatV2RawRewardComponents] = {}
    total_specs = len(specs)
    report_every = max(1, total_specs // 20)
    for completed, spec in enumerate(specs, start=1):
        rows[(spec.template_index, spec.position)] = _components_from_caches(
            spec,
            representation_cache=representation_cache,
            prefix_score_cache=prefix_score_cache,
            target_embedding=target_embedding,
            dpp_score_mode=str(dpp_score_mode),
            dpp_eps=float(dpp_eps),
        )
        if completed == total_specs or completed % report_every == 0:
            _print_progress(target, "assemble", completed, total_specs, started)
    elapsed = time.monotonic() - started
    metadata = {
        "implementation": "batched_cached_v1",
        "elapsed_seconds": round(float(elapsed), 3),
        "template_count": int(len(template_sessions)),
        "candidate_count": int(len(specs)),
        "unique_representation_count": int(len(representation_keys)),
        **representation_cache_metadata,
        "unique_prefix_count": int(len(prefix_keys)),
        "encode_chunk_size": int(_ENCODE_CHUNK_SIZE),
        "score_chunk_size": int(_SCORE_CHUNK_SIZE),
    }
    print(
        "[CREAT-Additive-SBR v2] "
        f"target={target} reward-table completed; candidates={len(specs)}, "
        f"elapsed_seconds={metadata['elapsed_seconds']}",
        flush=True,
    )
    return CreatV2RewardTable(
        rows=rows,
        candidate_reward_stats=reward_component_statistics(
            [row.to_dict() for row in rows.values()]
        ),
        build_metadata=metadata,
    )


def _build_candidate_specs(
    adapter,
    *,
    template_sessions: Sequence[Sequence[int]],
    target_item: int,
    replacement_topk_ratio: float,
    nonzero_when_possible: bool,
    local_window_size: int,
) -> list[_CandidateSpec]:
    specs: list[_CandidateSpec] = []
    for template_index, raw_session in enumerate(template_sessions):
        session = tuple(int(item) for item in raw_session)
        valid_mask = adapter.valid_position_mask(
            session,
            int(target_item),
            float(replacement_topk_ratio),
            nonzero_when_possible=bool(nonzero_when_possible),
        )
        for position, is_valid in enumerate(valid_mask.tolist()):
            if not bool(is_valid):
                continue
            polluted = list(session)
            polluted[position] = int(target_item)
            local_pairs = _full_affected_kgram_pairs(
                session,
                tuple(polluted),
                selected_position=int(position),
                window_size=int(local_window_size),
            )
            specs.append(
                _CandidateSpec(
                    template_index=int(template_index),
                    position=int(position),
                    original=session,
                    polluted=tuple(polluted),
                    prefix=session[:position],
                    segments=tuple(
                        segment
                        for segment in (session[:position], session[position + 1 :])
                        if segment
                    ),
                    local_pairs=local_pairs,
                    local_skipped_count=int(not local_pairs),
                )
            )
    return specs


def _full_affected_kgram_pairs(
    original: tuple[int, ...],
    polluted: tuple[int, ...],
    *,
    selected_position: int,
    window_size: int,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if window_size > len(original):
        return ()
    first_start = max(0, int(selected_position) - int(window_size) + 1)
    last_start = min(int(selected_position), len(original) - int(window_size))
    return tuple(
        (
            original[start : start + window_size],
            polluted[start : start + window_size],
        )
        for start in range(first_start, last_start + 1)
    )


def _representation_keys(specs: Sequence[_CandidateSpec]) -> list[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for spec in specs:
        keys.add(spec.original)
        keys.add(spec.polluted)
        keys.update(spec.segments)
        for original_window, polluted_window in spec.local_pairs:
            keys.add(original_window)
            keys.add(polluted_window)
    return sorted(keys)


def _shared_representation_keys(specs: Sequence[_CandidateSpec]) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for spec in specs:
        keys.add(spec.original)
        keys.update(spec.segments)
        keys.update(original_window for original_window, _polluted_window in spec.local_pairs)
    return keys


def _encode_representation_cache(
    adapter,
    keys: Sequence[tuple[int, ...]],
    *,
    shared_keys: set[tuple[int, ...]],
    target_item: int,
) -> tuple[dict[tuple[int, ...], torch.Tensor], dict[str, int]]:
    persistent_cache = getattr(adapter, "_creat_shared_representation_cache", None)
    if not isinstance(persistent_cache, dict):
        persistent_cache = {}
        setattr(adapter, "_creat_shared_representation_cache", persistent_cache)
    cache: dict[tuple[int, ...], torch.Tensor] = {
        key: persistent_cache[key] for key in keys if key in persistent_cache
    }
    missing_keys = [key for key in keys if key not in cache]
    started = time.monotonic()
    if cache:
        print(
            "[CREAT-Additive-SBR v2] "
            f"target={int(target_item)} reward-table shared representation cache "
            f"hits={len(cache)}, misses={len(missing_keys)}",
            flush=True,
        )
    for start in range(0, len(missing_keys), _ENCODE_CHUNK_SIZE):
        chunk = missing_keys[start : start + _ENCODE_CHUNK_SIZE]
        encoded = adapter.encode_sessions(chunk)
        cache.update({key: encoded[index] for index, key in enumerate(chunk)})
        _print_progress(
            int(target_item),
            "representations",
            min(start + len(chunk), len(missing_keys)),
            len(missing_keys),
            started,
        )
    persistent_cache.update({key: cache[key] for key in shared_keys})
    return cache, {
        "shared_representation_cache_hit_count": int(len(keys) - len(missing_keys)),
        "shared_representation_cache_miss_count": int(len(missing_keys)),
        "shared_representation_cache_size": int(len(persistent_cache)),
    }


def _score_prefix_cache(
    adapter,
    prefixes: Sequence[tuple[int, ...]],
    *,
    target_item: int,
) -> dict[tuple[int, ...], float]:
    cache: dict[tuple[int, ...], float] = {(): 0.0} if () in prefixes else {}
    nonempty_prefixes = [prefix for prefix in prefixes if prefix]
    started = time.monotonic()
    for start in range(0, len(nonempty_prefixes), _SCORE_CHUNK_SIZE):
        chunk = nonempty_prefixes[start : start + _SCORE_CHUNK_SIZE]
        if hasattr(adapter, "target_scores_for_prefixes"):
            scores = adapter.target_scores_for_prefixes(chunk, int(target_item))
        else:
            scores = [
                adapter.target_score_for_prefix(prefix, int(target_item))
                for prefix in chunk
            ]
        cache.update({key: float(scores[index]) for index, key in enumerate(chunk)})
        _print_progress(
            int(target_item),
            "prefix_scores",
            min(start + len(chunk), len(nonempty_prefixes)),
            len(nonempty_prefixes),
            started,
        )
    return cache


def _components_from_caches(
    spec: _CandidateSpec,
    *,
    representation_cache: dict[tuple[int, ...], torch.Tensor],
    prefix_score_cache: dict[tuple[int, ...], float],
    target_embedding: torch.Tensor,
    dpp_score_mode: str,
    dpp_eps: float,
) -> CreatV2RawRewardComponents:
    segment_reps = (
        torch.stack([representation_cache[segment] for segment in spec.segments])
        if spec.segments
        else torch.empty((0, int(target_embedding.shape[0])), dtype=target_embedding.dtype)
    )
    if len(spec.segments):
        target = target_embedding.view(1, -1).expand_as(segment_reps)
        pattern_reward = float(
            (1.0 - F.cosine_similarity(target, segment_reps, dim=1)).mean().item()
        )
    else:
        pattern_reward = 0.0
    dpp_raw, dpp_bounded, dpp_invalid_count = compute_dpp_scores(
        segment_reps,
        eps=float(dpp_eps),
    )
    local_distances = [
        torch.linalg.vector_norm(
            representation_cache[original_window] - representation_cache[polluted_window]
        )
        for original_window, polluted_window in spec.local_pairs
    ]
    local_consistency = (
        -float(torch.stack(local_distances).mean().item())
        if local_distances
        else 0.0
    )
    return CreatV2RawRewardComponents(
        attack_reward=float(prefix_score_cache[spec.prefix]),
        pattern_reward=float(pattern_reward),
        dpp_raw_logdet=float(dpp_raw),
        dpp_bounded_determinant=float(dpp_bounded),
        dpp_reward=float(dpp_raw if dpp_score_mode == "raw_logdet" else dpp_bounded),
        global_consistency_reward=-float(
            torch.linalg.vector_norm(
                representation_cache[spec.original] - representation_cache[spec.polluted]
            ).item()
        ),
        local_consistency_reward=float(local_consistency),
        pattern_segment_count=int(len(spec.segments)),
        dpp_segment_count=int(len(spec.segments)),
        dpp_invalid_count=int(dpp_invalid_count),
        local_affected_kgram_count=int(len(spec.local_pairs)),
        local_skipped_count=int(spec.local_skipped_count),
    )


def _print_progress(
    target_item: int,
    stage: str,
    completed: int,
    total: int,
    started: float,
) -> None:
    ratio = 1.0 if total <= 0 else float(completed) / float(total)
    print(
        "[CREAT-Additive-SBR v2] "
        f"target={int(target_item)} reward-table stage={stage} "
        f"progress={int(completed)}/{int(total)} ({ratio:.1%}); "
        f"elapsed_seconds={round(float(time.monotonic() - started), 3)}",
        flush=True,
    )


__all__ = ["CreatV2RewardTable", "build_v2_reward_table"]
