from __future__ import annotations

import math
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
from attack.pts.direct_action_executor import (
    DIRECT_ACTION_FORMAL_GENERATION_TAG,
    DIRECT_ACTION_FORMAL_PREFIX_TAG,
    DIRECT_ACTION_FORMAL_SAMPLE_TAG,
    DirectActionContextStats,
    apply_pts_direct_action_construction_batch,
    build_direct_action_formal_session_contexts,
)
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_MLP_H2_PARAMETER_NAMES,
    DIRECT_ACTION_POLICY_MLP_H2,
    DirectActionMLPPolicy,
    normalize_direct_action_length_feature_mode,
)


DIRECT_ACTION_MLP_CEM_METHOD = "direct_action_mlp_cem"


@dataclass(frozen=True)
class PTSDirectActionMLPCEMConfig:
    length_feature_mode: str = DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M
    elite_min_std: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "length_feature_mode",
            normalize_direct_action_length_feature_mode(self.length_feature_mode),
        )
        if float(self.elite_min_std) <= 0.0:
            raise ValueError("direct-action CEM elite_min_std must be positive.")
        object.__setattr__(self, "elite_min_std", float(self.elite_min_std))


@dataclass(frozen=True)
class _DirectActionCandidateSampleSpec:
    vector: list[float]
    distribution_mean: list[float]
    distribution_std: list[float]
    sample_origin: str
    sample_metadata: dict[str, object] = field(default_factory=dict)


class PTSDirectActionMLPCEMTrainer:
    def __init__(
        self,
        *,
        cem_config: PTSCEMConfig,
        direct_action_config: PTSDirectActionMLPCEMConfig,
        generation_topk: int = 100,
        generation_rng_tag: str = DIRECT_ACTION_FORMAL_GENERATION_TAG,
        action_sampling_tag: str = DIRECT_ACTION_FORMAL_SAMPLE_TAG,
        shared_prefix_rng_tag: str = DIRECT_ACTION_FORMAL_PREFIX_TAG,
    ) -> None:
        self.cem_config = cem_config
        self.direct_action_config = direct_action_config
        self.generation_topk = int(generation_topk)
        self.generation_rng_tag = str(generation_rng_tag)
        self.action_sampling_tag = str(action_sampling_tag)
        self.shared_prefix_rng_tag = str(shared_prefix_rng_tag)
        self.parameterization = DIRECT_ACTION_POLICY_MLP_H2
        self.parameter_names = tuple(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
        if int(self.generation_topk) <= 0:
            raise ValueError("generation_topk must be positive.")
        if self.cem_config.sampler.type != "gaussian":
            raise ValueError("direct_action_mlp_cem requires cem.sampler.type='gaussian'.")

    def train(
        self,
        *,
        template_sessions: Sequence[Sequence[int]],
        target_item: int,
        poison_runner,
        evaluator_fn: EvaluatorFn,
    ) -> PTSCEMResult:
        contexts, context_stats = build_direct_action_formal_session_contexts(
            template_sessions=template_sessions,
            base_seed=int(self.cem_config.base_seed),
            prefix_rng_tag=self.shared_prefix_rng_tag,
        )
        current_mean = [0.0 for _ in self.parameter_names]
        current_std = [1.0 for _ in self.parameter_names]
        context_stats_payload = context_stats.to_dict()
        policy_history = [
            self._distribution_payload(
                mean=current_mean,
                std=current_std,
                label="initial_search_distribution",
                context_stats=context_stats,
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
                context_stats=context_stats,
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
                candidate_policy = DirectActionMLPPolicy.from_vector(
                    sample_spec.vector,
                    length_feature_mode=self.direct_action_config.length_feature_mode,
                    context_stats=context_stats_payload,
                )
                construction_result = apply_pts_direct_action_construction_batch(
                    session_contexts=contexts,
                    context_stats=context_stats,
                    target_item=int(target_item),
                    policy=candidate_policy,
                    base_seed=int(self.cem_config.base_seed),
                    iteration=int(iteration),
                    candidate_key=key,
                    poison_runner=poison_runner,
                    generation_topk=self.generation_topk,
                    sample_rng_tag=self.action_sampling_tag,
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
                        str(metric_key): float(metric_value)
                        for metric_key, metric_value in evaluation.reward_metrics.items()
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
                            distribution_mean=sample_spec.distribution_mean,
                            distribution_std=sample_spec.distribution_std,
                            context_stats=context_stats,
                            construction_summary=construction_result.summary,
                        ),
                    },
                    epoch_reward_diagnostics=(
                        None
                        if evaluation.epoch_reward_diagnostics is None
                        else dict(evaluation.epoch_reward_diagnostics)
                    ),
                )
                candidates.append(candidate)
                all_candidates.append(candidate)

            elite_count = direct_action_elite_count(
                population_size,
                float(self.cem_config.elite_ratio),
            )
            ranked = _rank_candidates(candidates)
            elites = ranked[:elite_count]
            for candidate in elites:
                candidate.selected_as_elite = True
            update = self._updated_distribution_from_elites(elites=elites)
            current_mean = update.elite_mean
            current_std = update.resample_std
            policy_after = self._distribution_payload(
                mean=current_mean,
                std=current_std,
                label="search_distribution_after_iteration",
                context_stats=context_stats,
                elite_candidate_keys=[str(candidate.candidate_key) for candidate in elites],
                elite_rewards=[float(candidate.reward) for candidate in elites],
                elite_mean=update.elite_mean,
                elite_std=update.elite_std,
                resample_std=update.resample_std,
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
            raise RuntimeError("Direct-action PTS-CEM produced no candidates.")

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
                    context_stats=context_stats,
                ),
                "final_policy_selection": "global_best_candidate",
                "final_policy_candidate_key": str(best_candidate.candidate_key),
                "final_policy": best_candidate.policy.to_dict(),
                "final_best_action_summary": dict(
                    best_candidate.construction_summary.get("direct_action", {})
                ),
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
    ) -> list[_DirectActionCandidateSampleSpec]:
        sample_origin = (
            "direct_action_initial_standard_normal"
            if int(iteration) == 0
            else "direct_action_elite_centered_empirical_gaussian"
        )
        metadata = (
            {
                "cem_init": {
                    "mode": "standard_normal",
                    "parameter_space": "standardized_policy_parameter_space",
                }
            }
            if int(iteration) == 0
            else {}
        )
        return [
            _DirectActionCandidateSampleSpec(
                vector=self._sample_gaussian_vector(
                    mean=mean,
                    std=std,
                    seed=self._candidate_seed(iteration, candidate_id),
                ),
                distribution_mean=[float(value) for value in mean],
                distribution_std=[float(value) for value in std],
                sample_origin=sample_origin,
                sample_metadata={
                    "method": DIRECT_ACTION_MLP_CEM_METHOD,
                    "sample_origin": sample_origin,
                    "fixed_policy": False,
                    **metadata,
                },
            )
            for candidate_id in range(int(population_size))
        ]

    def _updated_distribution_from_elites(
        self,
        *,
        elites: Sequence[PTSCEMCandidateResult],
    ) -> "_DirectActionEliteUpdate":
        if not elites:
            raise ValueError("elites must not be empty.")
        vectors = [list(elite.policy.to_vector()) for elite in elites]
        if np is not None:
            arr = np.asarray(vectors, dtype=np.float64)
            elite_mean = [float(value) for value in arr.mean(axis=0)]
            elite_std = [float(value) for value in arr.std(axis=0, ddof=0)]
        else:
            elite_mean = [
                sum(vector[index] for vector in vectors) / float(len(vectors))
                for index in range(len(self.parameter_names))
            ]
            elite_std = [
                _std([vector[index] for vector in vectors], elite_mean[index])
                for index in range(len(self.parameter_names))
            ]
        resample_std = [
            max(
                float(elite_std[index]),
                float(self.direct_action_config.elite_min_std),
            )
            for index in range(len(self.parameter_names))
        ]
        return _DirectActionEliteUpdate(
            elite_mean=elite_mean,
            elite_std=elite_std,
            resample_std=resample_std,
        )

    def _sample_gaussian_vector(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        seed: int,
    ) -> list[float]:
        if len(mean) != len(self.parameter_names) or len(std) != len(self.parameter_names):
            raise ValueError("direct-action Gaussian vectors have unexpected length.")
        if np is not None:
            rng = np.random.default_rng(int(seed))
            values = rng.normal(
                [float(value) for value in mean],
                [float(value) for value in std],
            )
            return [float(value) for value in values.tolist()]
        rng = random.Random(int(seed))
        return [
            float(rng.gauss(float(center), float(width)))
            for center, width in zip(mean, std)
        ]

    def _distribution_payload(
        self,
        *,
        mean: Sequence[float],
        std: Sequence[float],
        label: str,
        context_stats: DirectActionContextStats,
        elite_candidate_keys: Sequence[str] | None = None,
        elite_rewards: Sequence[float] | None = None,
        elite_mean: Sequence[float] | None = None,
        elite_std: Sequence[float] | None = None,
        resample_std: Sequence[float] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": "direct_action_mlp_cem_search_distribution",
            "label": str(label),
            "method": DIRECT_ACTION_MLP_CEM_METHOD,
            "parameterization": self.parameterization,
            "parameter_names": list(self.parameter_names),
            "mean_vector": [float(value) for value in mean],
            "std_vector": [float(value) for value in std],
            "mean_policy": DirectActionMLPPolicy.from_vector(
                mean,
                length_feature_mode=self.direct_action_config.length_feature_mode,
                context_stats=context_stats.to_dict(),
            ).to_dict(),
            "length_feature": self.direct_action_config.length_feature_mode,
            "cem_init": {
                "mode": "standard_normal",
                "parameter_space": "standardized_policy_parameter_space",
            },
            "cem_update": {
                "mode": "elite_centered_empirical_gaussian",
                "anti_collapse_min_std": float(self.direct_action_config.elite_min_std),
                "std_ddof": 0,
            },
            "context_stats": context_stats.to_dict(),
        }
        if elite_candidate_keys is not None:
            payload["elite_candidate_keys"] = [str(key) for key in elite_candidate_keys]
        if elite_rewards is not None:
            payload["elite_rewards"] = [float(value) for value in elite_rewards]
        if elite_mean is not None:
            payload["elite_mean"] = [float(value) for value in elite_mean]
        if elite_std is not None:
            payload["elite_std"] = [float(value) for value in elite_std]
        if resample_std is not None:
            payload["resample_std"] = [float(value) for value in resample_std]
        return payload

    def _sample_metadata(
        self,
        *,
        policy: DirectActionMLPPolicy,
        vector: Sequence[float],
        distribution_mean: Sequence[float],
        distribution_std: Sequence[float],
        context_stats: DirectActionContextStats,
        construction_summary: dict[str, object],
    ) -> dict[str, object]:
        return {
            "method": DIRECT_ACTION_MLP_CEM_METHOD,
            "parameterization": self.parameterization,
            "parameter_names": list(self.parameter_names),
            "parameter_vector": [float(value) for value in vector],
            "theta": [float(value) for value in vector],
            "direct_action_policy_payload": policy.to_dict(),
            "length_feature": self.direct_action_config.length_feature_mode,
            "cem_init": {
                "mode": "standard_normal",
                "parameter_space": "standardized_policy_parameter_space",
            },
            "cem_update": {
                "mode": "elite_centered_empirical_gaussian",
                "anti_collapse_min_std": float(self.direct_action_config.elite_min_std),
                "std_ddof": 0,
            },
            "direct_action_context_stats": context_stats.to_dict(),
            "direct_action_action_summary": dict(
                construction_summary.get("direct_action", {})
            ),
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


@dataclass(frozen=True)
class _DirectActionEliteUpdate:
    elite_mean: list[float]
    elite_std: list[float]
    resample_std: list[float]


def direct_action_elite_count(population_size: int, elite_ratio: float) -> int:
    population = int(population_size)
    if population <= 0:
        raise ValueError("population_size must be positive.")
    if population == 1:
        return 1
    return min(population, max(2, int(math.ceil(population * float(elite_ratio)))))


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
    "DIRECT_ACTION_MLP_CEM_METHOD",
    "PTSDirectActionMLPCEMConfig",
    "PTSDirectActionMLPCEMTrainer",
    "direct_action_elite_count",
]
