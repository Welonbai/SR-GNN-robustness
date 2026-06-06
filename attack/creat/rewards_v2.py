from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean, pstdev
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


RAW_REWARD_COMPONENTS = (
    "attack_reward",
    "pattern_reward",
    "dpp_raw_logdet",
    "dpp_bounded_determinant",
    "dpp_reward",
    "global_consistency_reward",
    "local_consistency_reward",
)


@dataclass(frozen=True)
class CreatV2RawRewardComponents:
    attack_reward: float
    pattern_reward: float
    dpp_raw_logdet: float
    dpp_bounded_determinant: float
    dpp_reward: float
    global_consistency_reward: float
    local_consistency_reward: float
    pattern_segment_count: int
    dpp_segment_count: int
    dpp_invalid_count: int
    local_affected_kgram_count: int
    local_skipped_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_v2_raw_reward_components(
    adapter,
    *,
    original_session: Sequence[int],
    selected_position: int,
    target_item: int,
    local_window_size: int,
    dpp_score_mode: str,
    dpp_eps: float,
) -> CreatV2RawRewardComponents:
    session = [int(item) for item in original_session]
    position = int(selected_position)
    if position < 0 or position >= len(session):
        raise ValueError("selected_position is outside the session.")
    polluted = list(session)
    polluted[position] = int(target_item)
    segments = [
        segment
        for segment in (session[:position], session[position + 1 :])
        if segment
    ]
    with torch.no_grad():
        segment_representations = (
            adapter.encode_sessions(segments)
            if segments
            else torch.empty((0, int(adapter.embedding_dim)), dtype=torch.float32)
        )
        target_embedding = adapter.target_embedding(int(target_item))
        pattern_reward = _pattern_inversion_reward(
            target_embedding,
            segment_representations,
        )
        dpp_raw, dpp_bounded, dpp_invalid_count = compute_dpp_scores(
            segment_representations,
            eps=float(dpp_eps),
        )
        dpp_reward = (
            dpp_raw if str(dpp_score_mode) == "raw_logdet" else dpp_bounded
        )
        original_rep = adapter.encode_session(session)
        polluted_rep = adapter.encode_session(polluted)
        global_consistency = -float(
            torch.linalg.vector_norm(original_rep - polluted_rep).item()
        )
        local_consistency, affected_count, skipped_count = _local_consistency_reward(
            adapter,
            original_session=session,
            polluted_session=polluted,
            selected_position=position,
            window_size=int(local_window_size),
        )
    attack_reward = (
        0.0
        if position <= 0
        else float(adapter.target_score_for_prefix(session[:position], int(target_item)))
    )
    return CreatV2RawRewardComponents(
        attack_reward=float(attack_reward),
        pattern_reward=float(pattern_reward),
        dpp_raw_logdet=float(dpp_raw),
        dpp_bounded_determinant=float(dpp_bounded),
        dpp_reward=float(dpp_reward),
        global_consistency_reward=float(global_consistency),
        local_consistency_reward=float(local_consistency),
        pattern_segment_count=int(len(segments)),
        dpp_segment_count=int(len(segments)),
        dpp_invalid_count=int(dpp_invalid_count),
        local_affected_kgram_count=int(affected_count),
        local_skipped_count=int(skipped_count),
    )


def compose_v2_reward(
    components: CreatV2RawRewardComponents | Mapping[str, float | int],
    *,
    phase: str,
    pattern_reward_weight: float,
    dpp_reward_weight: float,
    global_consistency_weight: float,
    local_consistency_weight: float,
) -> float:
    values = components.to_dict() if isinstance(components, CreatV2RawRewardComponents) else components
    total = (
        float(values["attack_reward"])
        + float(pattern_reward_weight) * float(values["pattern_reward"])
        + float(dpp_reward_weight) * float(values["dpp_reward"])
    )
    if phase == "consistency":
        total += (
            float(global_consistency_weight)
            * float(values["global_consistency_reward"])
            + float(local_consistency_weight)
            * float(values["local_consistency_reward"])
        )
    elif phase != "attack":
        raise ValueError("phase must be 'attack' or 'consistency'.")
    return float(total)


def reward_component_statistics(
    rows: Sequence[Mapping[str, float | int]],
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for component in RAW_REWARD_COMPONENTS:
        finite: list[float] = []
        invalid_count = 0
        for row in rows:
            value = float(row[component])
            if math.isfinite(value):
                finite.append(value)
            else:
                invalid_count += 1
        result[component] = {
            "count": int(len(finite)),
            "mean": float(fmean(finite)) if finite else None,
            "std": float(pstdev(finite)) if finite else None,
            "min": float(min(finite)) if finite else None,
            "max": float(max(finite)) if finite else None,
            "invalid_count": int(invalid_count),
        }
    return result


def scalar_reward_statistics(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": int(len(finite)),
        "mean": float(fmean(finite)) if finite else None,
        "std": float(pstdev(finite)) if finite else None,
        "min": float(min(finite)) if finite else None,
        "max": float(max(finite)) if finite else None,
        "invalid_count": int(len(values) - len(finite)),
    }


def _pattern_inversion_reward(
    target_embedding: torch.Tensor,
    segment_representations: torch.Tensor,
) -> float:
    if int(segment_representations.shape[0]) == 0:
        return 0.0
    target = target_embedding.view(1, -1).expand_as(segment_representations)
    distances = 1.0 - F.cosine_similarity(target, segment_representations, dim=1)
    return float(distances.mean().item())


def compute_dpp_scores(
    segment_representations: torch.Tensor,
    *,
    eps: float,
) -> tuple[float, float, int]:
    if int(segment_representations.shape[0]) < 2:
        return 0.0, 0.0, 0
    normalized = F.normalize(segment_representations, p=2, dim=1)
    kernel = normalized @ normalized.transpose(0, 1)
    identity = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
    sign, raw_logdet = torch.linalg.slogdet(kernel + float(eps) * identity)
    bounded = torch.clamp(torch.linalg.det(kernel), min=0.0, max=1.0)
    if (
        float(sign.item()) <= 0.0
        or not bool(torch.isfinite(raw_logdet).item())
        or not bool(torch.isfinite(bounded).item())
    ):
        return 0.0, 0.0, 1
    return float(raw_logdet.item()), float(bounded.item()), 0


def _local_consistency_reward(
    adapter,
    *,
    original_session: Sequence[int],
    polluted_session: Sequence[int],
    selected_position: int,
    window_size: int,
) -> tuple[float, int, int]:
    length = len(original_session)
    if window_size > length:
        return 0.0, 0, 1
    first_start = max(0, int(selected_position) - int(window_size) + 1)
    last_start = min(int(selected_position), length - int(window_size))
    starts = list(range(first_start, last_start + 1))
    if not starts:
        return 0.0, 0, 1
    original_windows = [
        list(original_session[start : start + window_size]) for start in starts
    ]
    polluted_windows = [
        list(polluted_session[start : start + window_size]) for start in starts
    ]
    original_reps = adapter.encode_sessions(original_windows)
    polluted_reps = adapter.encode_sessions(polluted_windows)
    distances = torch.linalg.vector_norm(original_reps - polluted_reps, dim=1)
    return -float(distances.mean().item()), int(len(starts)), 0


__all__ = [
    "CreatV2RawRewardComponents",
    "RAW_REWARD_COMPONENTS",
    "compose_v2_reward",
    "compute_dpp_scores",
    "compute_v2_raw_reward_components",
    "reward_component_statistics",
    "scalar_reward_statistics",
]
