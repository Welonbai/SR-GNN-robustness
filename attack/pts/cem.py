from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the main project env
    np = None

from attack.pts.executor import apply_pts_construction_batch
from attack.pts.grouping import SuffixLengthBucket, default_suffix_length_buckets
from attack.pts.policy import GroupActionPolicy
from attack.pts.specs import PTSConstructionSpec


@dataclass(frozen=True)
class PTSCEMSamplerConfig:
    type: str = "dirichlet"
    concentration_scale: float = 20.0

    def __post_init__(self) -> None:
        if float(self.concentration_scale) <= 0.0:
            raise ValueError("concentration_scale must be positive.")


@dataclass(frozen=True)
class PTSCEMUpdateConfig:
    smoothing: float = 0.3
    min_probability: float = 0.03
    max_probability: float = 0.90

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.smoothing) <= 1.0:
            raise ValueError("smoothing must be in [0, 1].")
        minimum = float(self.min_probability)
        maximum = float(self.max_probability)
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("min_probability and max_probability must satisfy 0 <= min < max <= 1.")


@dataclass(frozen=True)
class PTSCEMInitConfig:
    mode: str = "uniform"


@dataclass(frozen=True)
class PTSCEMConfig:
    iterations: int
    population_schedule: list[int] | None = None
    population_size: int | None = None
    elite_ratio: float = 0.25
    sampler: PTSCEMSamplerConfig = field(default_factory=PTSCEMSamplerConfig)
    update: PTSCEMUpdateConfig = field(default_factory=PTSCEMUpdateConfig)
    init: PTSCEMInitConfig = field(default_factory=PTSCEMInitConfig)
    base_seed: int = 0
    candidate_seed_stride: int = 1000
    save_top_k_candidates: int = 3

    def __post_init__(self) -> None:
        if int(self.iterations) <= 0:
            raise ValueError("iterations must be positive.")
        if self.population_schedule is None and self.population_size is None:
            raise ValueError("Either population_schedule or population_size must be provided.")
        if self.population_schedule is not None:
            if len(self.population_schedule) != int(self.iterations):
                raise ValueError("population_schedule length must equal iterations.")
            if any(int(value) <= 0 for value in self.population_schedule):
                raise ValueError("All population_schedule values must be positive.")
        if self.population_size is not None and int(self.population_size) <= 0:
            raise ValueError("population_size must be positive.")
        if not 0.0 < float(self.elite_ratio) <= 1.0:
            raise ValueError("elite_ratio must be in (0, 1].")
        if int(self.candidate_seed_stride) <= 0:
            raise ValueError("candidate_seed_stride must be positive.")
        if int(self.save_top_k_candidates) < 0:
            raise ValueError("save_top_k_candidates must be >= 0.")


@dataclass(frozen=True)
class PTSCEMEvaluationResult:
    reward: float
    reward_metrics: dict[str, float]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class PTSCEMCandidateResult:
    iteration: int
    candidate_id: int
    candidate_seed: int
    policy: GroupActionPolicy
    reward: float
    reward_metrics: dict[str, float]
    evaluator_metadata: dict[str, object]
    construction_summary: dict[str, object]
    per_session_records: list[dict[str, object]]
    final_sessions: list[list[int]]
    selected_as_elite: bool = False
    selected_as_global_best: bool = False

    @property
    def candidate_key(self) -> str:
        return candidate_key(self.iteration, self.candidate_id)


@dataclass(frozen=True)
class PTSCEMIterationResult:
    iteration: int
    population_size: int
    elite_count: int
    candidates: list[PTSCEMCandidateResult]
    elite_candidate_keys: list[str]
    policy_before: dict[str, object]
    policy_after: dict[str, object]


@dataclass(frozen=True)
class PTSCEMResult:
    best_candidate: PTSCEMCandidateResult
    final_policy: GroupActionPolicy
    policy_history: list[dict[str, object]]
    iteration_results: list[PTSCEMIterationResult]
    top_candidates: list[PTSCEMCandidateResult]


EvaluatorFn = Callable[..., PTSCEMEvaluationResult]


class PTSGroupedCEMTrainer:
    def __init__(
        self,
        *,
        cem_config: PTSCEMConfig,
        specs: Sequence[PTSConstructionSpec],
        suffix_length_buckets: Sequence[SuffixLengthBucket] | None = None,
        disable_consume_one_when_suffix_len_leq_1: bool = True,
        generation_topk: int = 100,
        generation_rng_tag: str = "pts_generated_suffix",
    ) -> None:
        self.cem_config = cem_config
        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("specs must not be empty.")
        self.suffix_length_buckets = (
            default_suffix_length_buckets()
            if suffix_length_buckets is None
            else tuple(suffix_length_buckets)
        )
        if not self.suffix_length_buckets:
            raise ValueError("suffix_length_buckets must not be empty.")
        self.disable_consume_one_when_suffix_len_leq_1 = bool(
            disable_consume_one_when_suffix_len_leq_1
        )
        self.generation_topk = int(generation_topk)
        self.generation_rng_tag = str(generation_rng_tag)
        if int(self.generation_topk) <= 0:
            raise ValueError("generation_topk must be positive.")
        if self.cem_config.sampler.type != "dirichlet":
            raise ValueError("Phase 2 PTS-CEM supports only sampler.type='dirichlet'.")
        if self.cem_config.init.mode != "uniform":
            raise ValueError("Phase 2 PTS-CEM supports only init.mode='uniform'.")
        _validate_probability_bounds(
            group_count=len(self._action_names()),
            min_probability=float(self.cem_config.update.min_probability),
            max_probability=float(self.cem_config.update.max_probability),
        )

    def train(
        self,
        *,
        template_sessions: Sequence[Sequence[int]],
        target_item: int,
        poison_runner,
        evaluator_fn: EvaluatorFn,
    ) -> PTSCEMResult:
        if _specs_include_generate_action(self.specs) and poison_runner is None:
            raise ValueError("poison_runner is required when specs include generated suffix actions.")

        current_policy = self._initial_policy()
        policy_history = [current_policy.to_dict()]
        iteration_results: list[PTSCEMIterationResult] = []
        all_candidates: list[PTSCEMCandidateResult] = []

        for iteration in range(int(self.cem_config.iterations)):
            population_size = self._population_size(iteration)
            policy_before = current_policy.to_dict()
            candidates: list[PTSCEMCandidateResult] = []

            for candidate_id in range(population_size):
                seed = self._candidate_seed(iteration, candidate_id)
                candidate_policy = _sample_candidate_policy(
                    current_policy,
                    seed=seed,
                    concentration_scale=float(
                        self.cem_config.sampler.concentration_scale
                    ),
                    disable_consume_one_when_suffix_len_leq_1=(
                        self.disable_consume_one_when_suffix_len_leq_1
                    ),
                )
                construction_result = apply_pts_construction_batch(
                    template_sessions=template_sessions,
                    target_item=int(target_item),
                    specs=self.specs,
                    group_policy=candidate_policy,
                    rng=random.Random(seed),
                    poison_runner=poison_runner,
                    generation_topk=self.generation_topk,
                    generation_rng_base_seed=seed,
                    generation_rng_tag=self.generation_rng_tag,
                    suffix_length_buckets=self.suffix_length_buckets,
                    disable_consume_one_when_suffix_len_leq_1=(
                        self.disable_consume_one_when_suffix_len_leq_1
                    ),
                )
                evaluation = evaluator_fn(
                    candidate_sessions=construction_result.final_sessions,
                    candidate_session_records=construction_result.per_session_records,
                    candidate_summary=construction_result.summary,
                    iteration=int(iteration),
                    candidate_id=int(candidate_id),
                    candidate_seed=int(seed),
                    policy=candidate_policy,
                )
                if not isinstance(evaluation, PTSCEMEvaluationResult):
                    raise ValueError("evaluator_fn must return PTSCEMEvaluationResult.")
                candidate = PTSCEMCandidateResult(
                    iteration=int(iteration),
                    candidate_id=int(candidate_id),
                    candidate_seed=int(seed),
                    policy=candidate_policy,
                    reward=float(evaluation.reward),
                    reward_metrics={
                        str(key): float(value)
                        for key, value in evaluation.reward_metrics.items()
                    },
                    evaluator_metadata=dict(evaluation.metadata),
                    construction_summary=dict(construction_result.summary),
                    per_session_records=[
                        dict(record)
                        for record in construction_result.per_session_records
                    ],
                    final_sessions=[
                        [int(item) for item in session]
                        for session in construction_result.final_sessions
                    ],
                )
                candidates.append(candidate)
                all_candidates.append(candidate)

            elite_count = _elite_count(population_size, float(self.cem_config.elite_ratio))
            ranked = _rank_candidates(candidates)
            elites = ranked[:elite_count]
            for candidate in elites:
                candidate.selected_as_elite = True

            current_policy = _updated_policy_from_elites(
                old_policy=current_policy,
                elites=elites,
                update_config=self.cem_config.update,
                disable_consume_one_when_suffix_len_leq_1=(
                    self.disable_consume_one_when_suffix_len_leq_1
                ),
            )
            policy_after = current_policy.to_dict()
            policy_history.append(policy_after)
            iteration_results.append(
                PTSCEMIterationResult(
                    iteration=int(iteration),
                    population_size=int(population_size),
                    elite_count=int(elite_count),
                    candidates=candidates,
                    elite_candidate_keys=[
                        str(candidate.candidate_key) for candidate in elites
                    ],
                    policy_before=policy_before,
                    policy_after=policy_after,
                )
            )

        if not all_candidates:
            raise RuntimeError("PTS-CEM produced no candidates.")

        best_candidate = _rank_candidates(all_candidates)[0]
        best_candidate.selected_as_global_best = True
        top_candidates = _rank_candidates(all_candidates)[
            : int(self.cem_config.save_top_k_candidates)
        ]

        return PTSCEMResult(
            best_candidate=best_candidate,
            final_policy=current_policy,
            policy_history=policy_history,
            iteration_results=iteration_results,
            top_candidates=top_candidates,
        )

    def _initial_policy(self) -> GroupActionPolicy:
        return GroupActionPolicy.uniform(
            group_names=[bucket.name for bucket in self.suffix_length_buckets],
            action_names=self._action_names(),
            disable_consume_one_when_suffix_len_leq_1=(
                self.disable_consume_one_when_suffix_len_leq_1
            ),
        )

    def _action_names(self) -> tuple[str, ...]:
        return tuple(str(spec.name) for spec in self.specs)

    def _population_size(self, iteration: int) -> int:
        if self.cem_config.population_schedule is not None:
            return int(self.cem_config.population_schedule[int(iteration)])
        if self.cem_config.population_size is None:
            raise ValueError("population_size is required when population_schedule is absent.")
        return int(self.cem_config.population_size)

    def _candidate_seed(self, iteration: int, candidate_id: int) -> int:
        return int(self.cem_config.base_seed) + int(iteration) * int(
            self.cem_config.candidate_seed_stride
        ) + int(candidate_id)


def candidate_key(iteration: int, candidate_id: int) -> str:
    return f"iter{int(iteration)}_cand{int(candidate_id)}"


def _sample_candidate_policy(
    current_policy: GroupActionPolicy,
    *,
    seed: int,
    concentration_scale: float,
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> GroupActionPolicy:
    sampled: dict[str, dict[str, float]] = {}
    for group, probabilities in current_policy.group_probabilities.items():
        action_names = list(probabilities.keys())
        alpha = [
            max(float(probabilities[action]), 1e-12) * float(concentration_scale)
            for action in action_names
        ]
        sampled_values = _sample_dirichlet(alpha, seed=_group_seed(seed, group))
        sampled[group] = {
            action: float(value)
            for action, value in zip(action_names, sampled_values)
        }
    return GroupActionPolicy(
        sampled,
        disable_consume_one_when_suffix_len_leq_1=(
            disable_consume_one_when_suffix_len_leq_1
        ),
    )


def _sample_dirichlet(alpha: Sequence[float], *, seed: int) -> list[float]:
    if any(float(value) <= 0.0 for value in alpha):
        raise ValueError("Dirichlet alpha values must be positive.")
    if np is not None:
        rng = np.random.default_rng(int(seed))
        return [float(value) for value in rng.dirichlet([float(value) for value in alpha])]

    rng = random.Random(int(seed))
    values = [float(rng.gammavariate(float(value), 1.0)) for value in alpha]
    total = float(sum(values))
    if total <= 0.0:
        probability = 1.0 / float(len(values))
        return [probability for _ in values]
    return [float(value) / total for value in values]


def _updated_policy_from_elites(
    *,
    old_policy: GroupActionPolicy,
    elites: Sequence[PTSCEMCandidateResult],
    update_config: PTSCEMUpdateConfig,
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> GroupActionPolicy:
    if not elites:
        raise ValueError("elites must not be empty.")

    updated: dict[str, dict[str, float]] = {}
    for group, old_probabilities in old_policy.group_probabilities.items():
        action_names = list(old_probabilities.keys())
        elite_mean = {
            action: float(
                sum(
                    elite.policy.group_probabilities[group][action]
                    for elite in elites
                )
            )
            / float(len(elites))
            for action in action_names
        }
        smoothed = {
            action: (
                (1.0 - float(update_config.smoothing)) * elite_mean[action]
                + float(update_config.smoothing) * float(old_probabilities[action])
            )
            for action in action_names
        }
        updated[group] = _bounded_probability_mapping(
            smoothed,
            min_probability=float(update_config.min_probability),
            max_probability=float(update_config.max_probability),
        )
    return GroupActionPolicy(
        updated,
        disable_consume_one_when_suffix_len_leq_1=(
            disable_consume_one_when_suffix_len_leq_1
        ),
    )


def _bounded_probability_mapping(
    probabilities: dict[str, float],
    *,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    _validate_probability_bounds(
        group_count=len(probabilities),
        min_probability=float(min_probability),
        max_probability=float(max_probability),
    )
    values = {
        action: min(max(float(value), float(min_probability)), float(max_probability))
        for action, value in probabilities.items()
    }
    for _ in range(100):
        total = float(sum(values.values()))
        diff = 1.0 - total
        if abs(diff) <= 1e-12:
            break
        if diff > 0.0:
            adjustable = {
                action: float(max_probability) - value
                for action, value in values.items()
                if value < float(max_probability) - 1e-12
            }
            capacity = float(sum(adjustable.values()))
            if capacity <= 0.0:
                raise ValueError("Cannot renormalize probabilities within max_probability bound.")
            for action, action_capacity in adjustable.items():
                values[action] += diff * (action_capacity / capacity)
        else:
            adjustable = {
                action: value - float(min_probability)
                for action, value in values.items()
                if value > float(min_probability) + 1e-12
            }
            capacity = float(sum(adjustable.values()))
            if capacity <= 0.0:
                raise ValueError("Cannot renormalize probabilities within min_probability bound.")
            for action, action_capacity in adjustable.items():
                values[action] += diff * (action_capacity / capacity)

    total = float(sum(values.values()))
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Bounded probabilities failed to sum to 1.0: {total:.12f}.")
    if any(value < float(min_probability) - 1e-8 for value in values.values()):
        raise ValueError("Bounded probabilities violated min_probability.")
    if any(value > float(max_probability) + 1e-8 for value in values.values()):
        raise ValueError("Bounded probabilities violated max_probability.")
    return {action: float(value) for action, value in values.items()}


def _validate_probability_bounds(
    *,
    group_count: int,
    min_probability: float,
    max_probability: float,
) -> None:
    if int(group_count) <= 0:
        raise ValueError("group_count must be positive.")
    if float(min_probability) * int(group_count) > 1.0 + 1e-12:
        raise ValueError("min_probability is infeasible for the number of actions.")
    if float(max_probability) * int(group_count) < 1.0 - 1e-12:
        raise ValueError("max_probability is infeasible for the number of actions.")


def _elite_count(population_size: int, elite_ratio: float) -> int:
    return max(1, int(int(population_size) * float(elite_ratio)))


def _rank_candidates(
    candidates: Sequence[PTSCEMCandidateResult],
) -> list[PTSCEMCandidateResult]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.reward),
            int(candidate.iteration),
            int(candidate.candidate_id),
        ),
    )


def _specs_include_generate_action(specs: Sequence[PTSConstructionSpec]) -> bool:
    return any(spec.suffix_constructor.continuation_source == "generate" for spec in specs)


def _group_seed(seed: int, group_name: str) -> int:
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(group_name))


__all__ = [
    "PTSCEMCandidateResult",
    "PTSCEMConfig",
    "PTSCEMEvaluationResult",
    "PTSCEMInitConfig",
    "PTSCEMIterationResult",
    "PTSCEMResult",
    "PTSCEMSamplerConfig",
    "PTSCEMUpdateConfig",
    "PTSGroupedCEMTrainer",
    "candidate_key",
]
