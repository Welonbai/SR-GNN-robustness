from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import pstdev
from typing import Any, Sequence


@dataclass(frozen=True)
class CEMCandidate:
    iteration: int
    candidate_id: int
    logits_g2: tuple[float, float]
    logits_g3: tuple[float, float, float]
    pi_g2: tuple[float, float]
    pi_g3: tuple[float, float, float]


@dataclass(frozen=True)
class CEMCandidateResult:
    candidate: CEMCandidate
    reward: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class CEMState:
    mean_g2: list[float]
    std_g2: list[float]
    mean_g3: list[float]
    std_g3: list[float]


def initialize_cem_state(initial_std: float) -> CEMState:
    if float(initial_std) <= 0.0:
        raise ValueError("initial_std must be positive.")
    return CEMState(
        mean_g2=[0.0, 0.0],
        std_g2=[float(initial_std), float(initial_std)],
        mean_g3=[0.0, 0.0, 0.0],
        std_g3=[float(initial_std), float(initial_std), float(initial_std)],
    )


def sample_cem_candidates(
    state: CEMState,
    iteration: int,
    population_size: int,
    rng: random.Random,
) -> list[CEMCandidate]:
    if int(population_size) <= 0:
        raise ValueError("population_size must be positive.")

    candidates: list[CEMCandidate] = []
    for candidate_id in range(int(population_size)):
        logits_g2 = tuple(
            float(rng.gauss(mean, std))
            for mean, std in zip(state.mean_g2, state.std_g2)
        )
        logits_g3 = tuple(
            float(rng.gauss(mean, std))
            for mean, std in zip(state.mean_g3, state.std_g3)
        )
        candidates.append(
            CEMCandidate(
                iteration=int(iteration),
                candidate_id=int(candidate_id),
                logits_g2=(float(logits_g2[0]), float(logits_g2[1])),
                logits_g3=(
                    float(logits_g3[0]),
                    float(logits_g3[1]),
                    float(logits_g3[2]),
                ),
                pi_g2=_softmax2(logits_g2),
                pi_g3=_softmax3(logits_g3),
            )
        )
    return candidates


def update_cem_state(
    state: CEMState,
    results: Sequence[CEMCandidateResult],
    elite_ratio: float,
    smoothing: float,
    min_std: float,
) -> CEMState:
    if not results:
        raise ValueError("results must not be empty.")
    if not 0.0 < float(elite_ratio) <= 1.0:
        raise ValueError("elite_ratio must be in (0, 1].")
    if not 0.0 <= float(smoothing) <= 1.0:
        raise ValueError("smoothing must be in [0, 1].")
    if float(min_std) < 0.0:
        raise ValueError("min_std must be non-negative.")

    elite_count = max(1, int(math.ceil(len(results) * float(elite_ratio))))
    ranked_results = sorted(
        results,
        key=lambda result: (-float(result.reward), int(result.candidate.candidate_id)),
    )
    elite_results = ranked_results[:elite_count]

    elite_mean_g2, elite_std_g2 = _distribution_stats(
        [result.candidate.logits_g2 for result in elite_results]
    )
    elite_mean_g3, elite_std_g3 = _distribution_stats(
        [result.candidate.logits_g3 for result in elite_results]
    )

    return CEMState(
        mean_g2=_smooth_vector(state.mean_g2, elite_mean_g2, smoothing=smoothing),
        std_g2=_smooth_std_vector(
            state.std_g2,
            elite_std_g2,
            smoothing=smoothing,
            min_std=min_std,
        ),
        mean_g3=_smooth_vector(state.mean_g3, elite_mean_g3, smoothing=smoothing),
        std_g3=_smooth_std_vector(
            state.std_g3,
            elite_std_g3,
            smoothing=smoothing,
            min_std=min_std,
        ),
    )


def _distribution_stats(
    values: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    if not values:
        raise ValueError("values must not be empty.")
    width = len(values[0])
    means: list[float] = []
    stds: list[float] = []
    for column_index in range(width):
        column = [float(row[column_index]) for row in values]
        means.append(float(sum(column) / len(column)))
        stds.append(float(pstdev(column)))
    return means, stds


def _smooth_vector(
    old: Sequence[float],
    elite: Sequence[float],
    *,
    smoothing: float,
) -> list[float]:
    return [
        float((1.0 - float(smoothing)) * float(old_value) + float(smoothing) * float(elite_value))
        for old_value, elite_value in zip(old, elite)
    ]


def _smooth_std_vector(
    old: Sequence[float],
    elite: Sequence[float],
    *,
    smoothing: float,
    min_std: float,
) -> list[float]:
    updated: list[float] = []
    for old_value, elite_value in zip(old, elite):
        value = float((1.0 - float(smoothing)) * float(old_value) + float(smoothing) * float(elite_value))
        updated.append(float(max(float(min_std), value)))
    return updated


def _softmax2(logits: Sequence[float]) -> tuple[float, float]:
    probabilities = _softmax(logits)
    return float(probabilities[0]), float(probabilities[1])


def _softmax3(logits: Sequence[float]) -> tuple[float, float, float]:
    probabilities = _softmax(logits)
    return float(probabilities[0]), float(probabilities[1]), float(probabilities[2])


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("logits must not be empty.")
    max_logit = max(float(value) for value in logits)
    exp_values = [math.exp(float(value) - max_logit) for value in logits]
    total = sum(exp_values)
    return tuple(float(value / total) for value in exp_values)


__all__ = [
    "CEMCandidate",
    "CEMCandidateResult",
    "CEMState",
    "initialize_cem_state",
    "sample_cem_candidates",
    "update_cem_state",
]
