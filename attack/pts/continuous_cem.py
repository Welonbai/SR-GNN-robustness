from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the main project env
    np = None

from attack.pts.cem import (
    EvaluatorFn,
    PTSCEMCandidateResult,
    PTSCEMConfig,
    PTSCEMEvaluationResult,
    PTSCEMIterationResult,
    PTSCEMResult,
    candidate_key,
)
from attack.pts.continuous_executor import (
    apply_pts_continuous_beta_construction_batch,
    build_continuous_shared_session_contexts,
)
from attack.pts.continuous_policy import (
    CONTINUOUS_BETA_DEFAULT_BOUNDS,
    CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1,
    CONTINUOUS_BETA_NORMALIZED_SAMPLER,
    CONTINUOUS_BETA_PARAMETER_NAMES,
    CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    ContinuousBetaPolicy,
)


@dataclass(frozen=True)
class PTSContinuousBetaCEMConfig:
    parameter_bounds: tuple[float, float] = CONTINUOUS_BETA_DEFAULT_BOUNDS
    initial_std: float = 2.0
    min_std: float = 0.25
    deterministic_sampling: bool = True
    initialization_mode: str = CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1
    gaussian_fill: bool = True


@dataclass(frozen=True)
class _ContinuousCandidateSampleSpec:
    vector: list[float]
    sample_origin: str
    sample_metadata: dict[str, object] = field(default_factory=dict)
    parent_iteration: int | None = None
    parent_candidate_id: int | None = None
    parent_candidate_key: str | None = None
    parent_reward: float | None = None
    parent_rank_among_elites: int | None = None


class PTSContinuousBetaCEMTrainer:
    def __init__(
        self,
        *,
        cem_config: PTSCEMConfig,
        continuous_config: PTSContinuousBetaCEMConfig,
        generation_topk: int = 100,
        generation_rng_tag: str = "pts_generated_suffix",
        shared_prefix_rng_tag: str = CONTINUOUS_BETA_SHARED_PREFIX_TAG,
    ) -> None:
        self.cem_config = cem_config
        self.continuous_config = continuous_config
        self.generation_topk = int(generation_topk)
        self.generation_rng_tag = str(generation_rng_tag)
        self.shared_prefix_rng_tag = str(shared_prefix_rng_tag)
        if int(self.generation_topk) <= 0:
            raise ValueError("generation_topk must be positive.")
        if not bool(self.continuous_config.deterministic_sampling):
            raise ValueError("continuous_beta.deterministic_sampling=false is not supported.")
        if self.continuous_config.initialization_mode != (
            CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1
        ):
            raise ValueError(
                "continuous_beta.initialization.mode must be "
                "'behavior_covering_v1'."
            )
        if float(self.continuous_config.initial_std) <= 0.0:
            raise ValueError("continuous_beta.initial_std must be positive.")
        if float(self.continuous_config.min_std) <= 0.0:
            raise ValueError("continuous_beta.min_std must be positive.")

    def train(
        self,
        *,
        template_sessions: Sequence[Sequence[int]],
        target_item: int,
        poison_runner,
        evaluator_fn: EvaluatorFn,
    ) -> PTSCEMResult:
        session_contexts = build_continuous_shared_session_contexts(
            template_sessions=template_sessions,
            target_item=int(target_item),
            base_seed=int(self.cem_config.base_seed),
            prefix_rng_tag=self.shared_prefix_rng_tag,
        )
        current_mean = [0.0 for _ in CONTINUOUS_BETA_PARAMETER_NAMES]
        current_std = [
            float(self.continuous_config.initial_std)
            for _ in CONTINUOUS_BETA_PARAMETER_NAMES
        ]
        policy_history = [
            self._distribution_payload(
                mean=current_mean,
                std=current_std,
                label="initial_search_distribution",
            )
        ]
        iteration_results: list[PTSCEMIterationResult] = []
        all_candidates: list[PTSCEMCandidateResult] = []

        for iteration in range(int(self.cem_config.iterations)):
            population_size = self._population_size(iteration)
            policy_before = self._distribution_payload(
                mean=current_mean,
                std=current_std,
                label="search_distribution_before_iteration",
            )
            sample_plan = self._candidate_sample_plan(
                iteration=int(iteration),
                population_size=int(population_size),
                mean=current_mean,
                std=current_std,
            )
            candidates: list[PTSCEMCandidateResult] = []
            for candidate_id, sample_spec in enumerate(sample_plan):
                seed = self._candidate_seed(iteration, candidate_id)
                key = candidate_key(iteration, candidate_id)
                candidate_policy = ContinuousBetaPolicy.from_vector(
                    sample_spec.vector,
                    parameter_bounds=self.continuous_config.parameter_bounds,
                )
                construction_result = apply_pts_continuous_beta_construction_batch(
                    session_contexts=session_contexts,
                    target_item=int(target_item),
                    policy=candidate_policy,
                    base_seed=int(self.cem_config.base_seed),
                    candidate_key=key,
                    poison_runner=poison_runner,
                    generation_topk=self.generation_topk,
                    generation_rng_base_seed=seed,
                    generation_rng_tag=self.generation_rng_tag,
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
                    sample_origin=sample_spec.sample_origin,
                    sample_metadata={
                        **dict(sample_spec.sample_metadata),
                        **self._sample_metadata(
                            policy=candidate_policy,
                            vector=sample_spec.vector,
                            distribution_mean=current_mean,
                            distribution_std=current_std,
                        ),
                    },
                    parent_iteration=sample_spec.parent_iteration,
                    parent_candidate_id=sample_spec.parent_candidate_id,
                    parent_candidate_key=sample_spec.parent_candidate_key,
                    parent_reward=sample_spec.parent_reward,
                    parent_rank_among_elites=sample_spec.parent_rank_among_elites,
                    epoch_reward_diagnostics=(
                        None
                        if evaluation.epoch_reward_diagnostics is None
                        else dict(evaluation.epoch_reward_diagnostics)
                    ),
                )
                candidates.append(candidate)
                all_candidates.append(candidate)

            elite_count = _elite_count(population_size, float(self.cem_config.elite_ratio))
            ranked = _rank_candidates(candidates)
            elites = ranked[:elite_count]
            for candidate in elites:
                candidate.selected_as_elite = True
            current_mean, current_std = self._updated_distribution_from_elites(
                old_mean=current_mean,
                elites=elites,
            )
            policy_after = self._distribution_payload(
                mean=current_mean,
                std=current_std,
                label="search_distribution_after_iteration",
                elite_candidate_keys=[str(candidate.candidate_key) for candidate in elites],
            )
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
            raise RuntimeError("Continuous PTS-CEM produced no candidates.")

        best_candidate = _rank_candidates(all_candidates)[0]
        best_candidate.selected_as_global_best = True
        top_candidates = _rank_candidates(all_candidates)[
            : int(self.cem_config.save_top_k_candidates)
        ]
        policy_history.append(
            {
                **self._distribution_payload(
                    mean=current_mean,
                    std=current_std,
                    label="final_search_distribution_diagnostic",
                ),
                "final_policy_selection": "global_best_candidate",
                "final_policy_candidate_key": str(best_candidate.candidate_key),
                "final_policy": best_candidate.policy.to_dict(),
            }
        )

        return PTSCEMResult(
            best_candidate=best_candidate,
            final_policy=best_candidate.policy,
            policy_history=policy_history,
            iteration_results=iteration_results,
            top_candidates=top_candidates,
        )

    def _candidate_sample_plan(
        self,
        *,
        iteration: int,
        population_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> list[_ContinuousCandidateSampleSpec]:
        if int(iteration) == 0:
            return self._initial_sample_plan(population_size=int(population_size))
        return [
            _ContinuousCandidateSampleSpec(
                vector=self._sample_gaussian_vector(
                    mean=mean,
                    std=std,
                    seed=self._candidate_seed(iteration, candidate_id),
                ),
                sample_origin="continuous_beta_elite_gaussian",
                sample_metadata={
                    "init_mode": self.continuous_config.initialization_mode,
                    "sample_origin": "continuous_beta_elite_gaussian",
                    "fixed_policy": False,
                },
            )
            for candidate_id in range(int(population_size))
        ]

    def _initial_sample_plan(self, *, population_size: int) -> list[_ContinuousCandidateSampleSpec]:
        prototypes = _behavior_covering_prototypes()
        selected: list[_ContinuousCandidateSampleSpec] = []
        for name, vector in prototypes[: int(population_size)]:
            selected.append(
                _ContinuousCandidateSampleSpec(
                    vector=self._clip_vector(vector),
                    sample_origin="continuous_beta_behavior_covering",
                    sample_metadata={
                        "init_mode": self.continuous_config.initialization_mode,
                        "prototype_name": name,
                        "fixed_policy": True,
                    },
                )
            )
        remaining = int(population_size) - len(selected)
        if remaining <= 0:
            return selected
        if not bool(self.continuous_config.gaussian_fill):
            raise ValueError(
                "behavior_covering_v1 initial population is smaller than "
                "population_size and gaussian_fill=false."
            )
        for offset in range(remaining):
            candidate_id = len(selected)
            selected.append(
                _ContinuousCandidateSampleSpec(
                    vector=self._sample_gaussian_vector(
                        mean=[0.0 for _ in CONTINUOUS_BETA_PARAMETER_NAMES],
                        std=[
                            float(self.continuous_config.initial_std)
                            for _ in CONTINUOUS_BETA_PARAMETER_NAMES
                        ],
                        seed=self._candidate_seed(0, candidate_id),
                    ),
                    sample_origin="continuous_beta_initial_gaussian",
                    sample_metadata={
                        "init_mode": self.continuous_config.initialization_mode,
                        "sample_origin": "continuous_beta_initial_gaussian",
                        "fixed_policy": False,
                        "gaussian_fill_index": int(offset),
                    },
                )
            )
        return selected

    def _updated_distribution_from_elites(
        self,
        *,
        old_mean: Sequence[float],
        elites: Sequence[PTSCEMCandidateResult],
    ) -> tuple[list[float], list[float]]:
        if not elites:
            raise ValueError("elites must not be empty.")
        vectors = [list(elite.policy.to_vector()) for elite in elites]
        if np is not None:
            arr = np.asarray(vectors, dtype=np.float64)
            elite_mean = [float(value) for value in arr.mean(axis=0)]
            elite_std = [float(value) for value in arr.std(axis=0)]
        else:
            elite_mean = [
                sum(vector[index] for vector in vectors) / float(len(vectors))
                for index in range(len(CONTINUOUS_BETA_PARAMETER_NAMES))
            ]
            elite_std = [
                _std([vector[index] for vector in vectors], elite_mean[index])
                for index in range(len(CONTINUOUS_BETA_PARAMETER_NAMES))
            ]
        smoothing = float(self.cem_config.update.smoothing)
        new_mean = self._clip_vector(
            [
                (1.0 - smoothing) * float(elite_mean[index])
                + smoothing * float(old_mean[index])
                for index in range(len(CONTINUOUS_BETA_PARAMETER_NAMES))
            ]
        )
        new_std = [
            max(float(elite_std[index]), float(self.continuous_config.min_std))
            for index in range(len(CONTINUOUS_BETA_PARAMETER_NAMES))
        ]
        return new_mean, new_std

    def _sample_gaussian_vector(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        seed: int,
    ) -> list[float]:
        if np is not None:
            rng = np.random.default_rng(int(seed))
            values = rng.normal(
                [float(value) for value in mean],
                [float(value) for value in std],
            )
            return self._clip_vector([float(value) for value in values.tolist()])
        rng = random.Random(int(seed))
        return self._clip_vector(
            [
                rng.gauss(float(center), float(width))
                for center, width in zip(mean, std)
            ]
        )

    def _clip_vector(self, vector: Sequence[float]) -> list[float]:
        lower, upper = self.continuous_config.parameter_bounds
        return [float(min(max(float(value), float(lower)), float(upper))) for value in vector]

    def _distribution_payload(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        label: str,
        elite_candidate_keys: Sequence[str] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "continuous_beta_cem_search_distribution",
            "label": str(label),
            "method": "continuous_beta_cem_v1",
            "normalized_sampler": CONTINUOUS_BETA_NORMALIZED_SAMPLER,
            "parameter_names": list(CONTINUOUS_BETA_PARAMETER_NAMES),
            "mean_vector": [float(value) for value in mean],
            "std_vector": [float(value) for value in std],
            "mean_policy": ContinuousBetaPolicy.from_vector(
                mean,
                parameter_bounds=self.continuous_config.parameter_bounds,
            ).to_dict(),
            "parameter_bounds": {
                "min": float(self.continuous_config.parameter_bounds[0]),
                "max": float(self.continuous_config.parameter_bounds[1]),
            },
            "initial_std": float(self.continuous_config.initial_std),
            "min_std": float(self.continuous_config.min_std),
            "initialization_mode": self.continuous_config.initialization_mode,
            "gaussian_fill": bool(self.continuous_config.gaussian_fill),
        }
        if elite_candidate_keys is not None:
            payload["elite_candidate_keys"] = [str(key) for key in elite_candidate_keys]
        return payload

    def _sample_metadata(
        self,
        *,
        policy: ContinuousBetaPolicy,
        vector: Sequence[float],
        distribution_mean: Sequence[float],
        distribution_std: Sequence[float],
    ) -> dict[str, object]:
        return {
            "method": "continuous_beta_cem_v1",
            "parameter_names": list(CONTINUOUS_BETA_PARAMETER_NAMES),
            "parameter_vector": [float(value) for value in vector],
            "policy_vector": policy.to_vector(),
            "bounds": {
                "min": float(self.continuous_config.parameter_bounds[0]),
                "max": float(self.continuous_config.parameter_bounds[1]),
            },
            "initial_std": float(self.continuous_config.initial_std),
            "min_std": float(self.continuous_config.min_std),
            "normalized_sampler": CONTINUOUS_BETA_NORMALIZED_SAMPLER,
            "search_distribution_mean": [float(value) for value in distribution_mean],
            "search_distribution_std": [float(value) for value in distribution_std],
        }

    def _candidate_seed(self, iteration: int, candidate_id: int) -> int:
        return int(self.cem_config.base_seed) + int(iteration) * int(
            self.cem_config.candidate_seed_stride
        ) + int(candidate_id)

    def _population_size(self, iteration: int) -> int:
        if self.cem_config.population_schedule is not None:
            return int(self.cem_config.population_schedule[int(iteration)])
        if self.cem_config.population_size is None:
            raise ValueError("population_size is required when population_schedule is absent.")
        return int(self.cem_config.population_size)


def _behavior_covering_prototypes() -> list[tuple[str, list[float]]]:
    return [
        ("near_zero_consume_preserve", [-3.0, 0.0, 2.0, 0.0, -4.0, 0.0, 0.0]),
        ("near_zero_consume_generate", [-3.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0]),
        ("near_one_consume_stop", [2.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0]),
        ("middle_consume_preserve", [2.0, 0.0, 2.0, 0.0, -4.0, 0.0, 0.0]),
        ("middle_consume_generate", [2.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0]),
        ("u_shaped_mixed_source", [-2.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0]),
        ("q_sensitive_generation", [0.0, 0.0, 0.0, 0.0, -2.0, 4.0, 0.0]),
        ("rho_sensitive_generation", [0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 4.0]),
    ]


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


def _std(values: Sequence[float], mean: float) -> float:
    if not values:
        return 0.0
    return float(
        (
            sum((float(value) - float(mean)) ** 2.0 for value in values)
            / float(len(values))
        )
        ** 0.5
    )


__all__ = [
    "PTSContinuousBetaCEMConfig",
    "PTSContinuousBetaCEMTrainer",
]
