from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is available in the main project env
    np = None

from attack.pts.executor import apply_pts_construction_batch
from attack.pts.grouping import SuffixLengthBucket, default_suffix_length_buckets
from attack.pts.policy import GroupActionPolicy, build_valid_actions_by_group
from attack.pts.space_filling import (
    PTSSpaceFillingConfig,
    PTSSpaceFillingSample,
    build_vertex_stratified_initial_population,
)
from attack.pts.specs import CONSUME_ONE_GENERATE_ACTION_NAME, PTSConstructionSpec


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
    mandatory_enabled: bool = True
    extreme_count: int = 7
    moderate_count: int = 3
    balanced_count: int = 1
    extreme_pool_size: int = 1024
    moderate_pool_size: int = 512
    extreme_alpha: float = 0.3
    moderate_alpha: float = 2.0
    distance: str = "l1"

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"uniform", "vertex_stratified_space_filling"}:
            raise ValueError(
                "init.mode must be 'uniform' or "
                "'vertex_stratified_space_filling'."
            )
        if int(self.extreme_count) < 0:
            raise ValueError("init.extreme_count must be >= 0.")
        if int(self.moderate_count) < 0:
            raise ValueError("init.moderate_count must be >= 0.")
        if int(self.balanced_count) not in {0, 1}:
            raise ValueError("init.balanced_count must be 0 or 1.")
        if int(self.extreme_pool_size) < int(self.extreme_count):
            raise ValueError("init.extreme_pool_size must be >= extreme_count.")
        if int(self.moderate_pool_size) < int(self.moderate_count):
            raise ValueError("init.moderate_pool_size must be >= moderate_count.")
        if float(self.extreme_alpha) <= 0.0:
            raise ValueError("init.extreme_alpha must be positive.")
        if float(self.moderate_alpha) <= 0.0:
            raise ValueError("init.moderate_alpha must be positive.")
        distance = str(self.distance).strip().lower()
        if distance != "l1":
            raise ValueError("init.distance currently supports only 'l1'.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mandatory_enabled", bool(self.mandatory_enabled))
        object.__setattr__(self, "extreme_count", int(self.extreme_count))
        object.__setattr__(self, "moderate_count", int(self.moderate_count))
        object.__setattr__(self, "balanced_count", int(self.balanced_count))
        object.__setattr__(self, "extreme_pool_size", int(self.extreme_pool_size))
        object.__setattr__(self, "moderate_pool_size", int(self.moderate_pool_size))
        object.__setattr__(self, "extreme_alpha", float(self.extreme_alpha))
        object.__setattr__(self, "moderate_alpha", float(self.moderate_alpha))
        object.__setattr__(self, "distance", distance)


@dataclass(frozen=True)
class PTSCEMResamplingConfig:
    mode: str = "standard"
    local_concentration_scale: float = 30.0

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"standard", "elite_centered"}:
            raise ValueError("resampling.mode must be 'standard' or 'elite_centered'.")
        if float(self.local_concentration_scale) <= 0.0:
            raise ValueError("resampling.local_concentration_scale must be positive.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "local_concentration_scale",
            float(self.local_concentration_scale),
        )


@dataclass(frozen=True)
class PTSCEMConfig:
    iterations: int
    population_schedule: list[int] | None = None
    population_size: int | None = None
    elite_ratio: float = 0.25
    sampler: PTSCEMSamplerConfig = field(default_factory=PTSCEMSamplerConfig)
    update: PTSCEMUpdateConfig = field(default_factory=PTSCEMUpdateConfig)
    init: PTSCEMInitConfig = field(default_factory=PTSCEMInitConfig)
    resampling: PTSCEMResamplingConfig = field(default_factory=PTSCEMResamplingConfig)
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
    epoch_reward_diagnostics: dict[str, object] | None = None


@dataclass
class PTSCEMCandidateResult:
    iteration: int
    candidate_id: int
    candidate_seed: int
    policy: Any
    reward: float
    reward_metrics: dict[str, float]
    evaluator_metadata: dict[str, object]
    construction_summary: dict[str, object]
    per_session_records: list[dict[str, object]]
    final_sessions: list[list[int]]
    selected_as_elite: bool = False
    selected_as_global_best: bool = False
    sample_origin: str = "global_policy"
    sample_metadata: dict[str, object] = field(default_factory=dict)
    parent_iteration: int | None = None
    parent_candidate_id: int | None = None
    parent_candidate_key: str | None = None
    parent_reward: float | None = None
    parent_rank_among_elites: int | None = None
    epoch_reward_diagnostics: dict[str, object] | None = None

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
    final_policy: Any
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
        self.valid_actions_by_group = build_valid_actions_by_group(
            group_buckets=self.suffix_length_buckets,
            enabled_actions=self._action_names(),
            disable_consume_one_when_suffix_len_leq_1=(
                self.disable_consume_one_when_suffix_len_leq_1
            ),
        )
        self.generation_topk = int(generation_topk)
        self.generation_rng_tag = str(generation_rng_tag)
        if int(self.generation_topk) <= 0:
            raise ValueError("generation_topk must be positive.")
        if self.cem_config.sampler.type != "dirichlet":
            raise ValueError("Phase 2 PTS-CEM supports only sampler.type='dirichlet'.")
        if (
            self.cem_config.init.mode == "vertex_stratified_space_filling"
            and bool(self.cem_config.init.mandatory_enabled)
            and CONSUME_ONE_GENERATE_ACTION_NAME not in set(self._action_names())
        ):
            raise ValueError(
                "vertex_stratified_space_filling with mandatory_enabled=true "
                "requires consume_one_generate_continuation in actions.enabled, "
                "because the c1_generate_where_valid mandatory vertex cannot be "
                "constructed."
            )
        for group_name, actions in self.valid_actions_by_group.items():
            _validate_probability_bounds(
                group_count=len(actions),
                min_probability=float(self.cem_config.update.min_probability),
                max_probability=float(self.cem_config.update.max_probability),
                label=f"group {group_name!r}",
            )
        self._initial_space_filling_samples: list[PTSSpaceFillingSample] = []
        if self.cem_config.init.mode == "vertex_stratified_space_filling":
            self._initial_space_filling_samples = self._build_initial_space_filling_samples()
            initial_population_size = len(self._initial_space_filling_samples)
            configured_population_size = self._population_size(0)
            if int(configured_population_size) != int(initial_population_size):
                raise ValueError(
                    "vertex_stratified_space_filling requires "
                    "population_schedule[0] or population_size to equal the "
                    f"computed initial population size {initial_population_size}; "
                    f"received {configured_population_size}."
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
        previous_elites: list[PTSCEMCandidateResult] = []

        for iteration in range(int(self.cem_config.iterations)):
            population_size = self._population_size(iteration)
            policy_before = current_policy.to_dict()
            candidates: list[PTSCEMCandidateResult] = []
            sample_plan = self._candidate_sample_plan(
                iteration=int(iteration),
                population_size=int(population_size),
                current_policy=current_policy,
                previous_elites=previous_elites,
            )

            for candidate_id, sample_spec in enumerate(sample_plan):
                seed = self._candidate_seed(iteration, candidate_id)
                if sample_spec.fixed_policy is not None:
                    candidate_policy = sample_spec.fixed_policy
                else:
                    candidate_policy = _sample_candidate_policy_from_center(
                        center_policy=sample_spec.center_policy,
                        candidate_seed=seed,
                        concentration_scale=sample_spec.concentration_scale,
                        min_probability=float(self.cem_config.update.min_probability),
                        max_probability=float(self.cem_config.update.max_probability),
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
                    sample_origin=sample_spec.sample_origin,
                    sample_metadata=dict(sample_spec.sample_metadata),
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
            previous_elites = list(elites)

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
            valid_actions_by_group=self.valid_actions_by_group,
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

    def _build_initial_space_filling_samples(self) -> list[PTSSpaceFillingSample]:
        init = self.cem_config.init
        return build_vertex_stratified_initial_population(
            config=PTSSpaceFillingConfig(
                seed=int(self.cem_config.base_seed),
                mandatory_enabled=bool(init.mandatory_enabled),
                extreme_count=int(init.extreme_count),
                moderate_count=int(init.moderate_count),
                balanced_count=int(init.balanced_count),
                extreme_pool_size=int(init.extreme_pool_size),
                moderate_pool_size=int(init.moderate_pool_size),
                extreme_alpha=float(init.extreme_alpha),
                moderate_alpha=float(init.moderate_alpha),
                min_probability=float(self.cem_config.update.min_probability),
                max_probability=float(self.cem_config.update.max_probability),
                distance=str(init.distance),
            ),
            valid_actions_by_group=self.valid_actions_by_group,
            enabled_actions=self._action_names(),
            disable_consume_one_when_suffix_len_leq_1=(
                self.disable_consume_one_when_suffix_len_leq_1
            ),
        )

    def _candidate_seed(self, iteration: int, candidate_id: int) -> int:
        return int(self.cem_config.base_seed) + int(iteration) * int(
            self.cem_config.candidate_seed_stride
        ) + int(candidate_id)

    def _sample_projection_metadata(self) -> dict[str, object]:
        return {
            "sampled_policy_projection_enabled": True,
            "sampled_policy_min_probability": float(
                self.cem_config.update.min_probability
            ),
            "sampled_policy_max_probability": float(
                self.cem_config.update.max_probability
            ),
        }

    def _candidate_sample_plan(
        self,
        *,
        iteration: int,
        population_size: int,
        current_policy: GroupActionPolicy,
        previous_elites: Sequence[PTSCEMCandidateResult],
    ) -> list["_CandidateSampleSpec"]:
        mode = self.cem_config.resampling.mode
        init_mode = self.cem_config.init.mode
        if init_mode == "vertex_stratified_space_filling" and int(iteration) == 0:
            if len(self._initial_space_filling_samples) != int(population_size):
                raise ValueError(
                    "vertex_stratified_space_filling initial population size "
                    "does not match iteration-0 population_size."
                )
            return [
                _CandidateSampleSpec(
                    center_policy=current_policy,
                    concentration_scale=float(
                        self.cem_config.sampler.concentration_scale
                    ),
                    sample_origin=sample.sample_origin,
                    fixed_policy=sample.policy,
                    sample_metadata={
                        **sample.sample_metadata(init_mode=init_mode),
                        "fixed_policy": True,
                        "concentration_scale": None,
                        **self._sample_projection_metadata(),
                    },
                )
                for sample in self._initial_space_filling_samples
            ]
        if mode == "standard":
            return [
                _CandidateSampleSpec(
                    center_policy=current_policy,
                    concentration_scale=float(
                        self.cem_config.sampler.concentration_scale
                    ),
                    sample_origin="global_policy",
                    sample_metadata={
                        "init_mode": init_mode,
                        "sample_origin": "global_policy",
                        "fixed_policy": False,
                        "concentration_scale": float(
                            self.cem_config.sampler.concentration_scale
                        ),
                        **self._sample_projection_metadata(),
                    },
                )
                for _ in range(int(population_size))
            ]
        if mode != "elite_centered":
            raise ValueError(f"Unsupported resampling.mode {mode!r}.")
        if int(iteration) == 0:
            return [
                _CandidateSampleSpec(
                    center_policy=current_policy,
                    concentration_scale=float(
                        self.cem_config.sampler.concentration_scale
                    ),
                    sample_origin="initial_global_policy",
                    sample_metadata={
                        "init_mode": init_mode,
                        "sample_origin": "initial_global_policy",
                        "fixed_policy": False,
                        "concentration_scale": float(
                            self.cem_config.sampler.concentration_scale
                        ),
                        **self._sample_projection_metadata(),
                    },
                )
                for _ in range(int(population_size))
            ]
        if not previous_elites:
            raise RuntimeError(
                "elite_centered resampling requires previous iteration elites."
            )
        allocation = _allocate_children_to_elites(
            population_size=int(population_size),
            elite_count=len(previous_elites),
        )
        sample_plan: list[_CandidateSampleSpec] = []
        for elite_rank_index, (elite, child_count) in enumerate(
            zip(previous_elites, allocation),
            start=1,
        ):
            for _ in range(int(child_count)):
                sample_plan.append(
                    _CandidateSampleSpec(
                        center_policy=elite.policy,
                        concentration_scale=float(
                            self.cem_config.resampling.local_concentration_scale
                        ),
                        sample_origin="elite_centered",
                        parent_iteration=int(elite.iteration),
                        parent_candidate_id=int(elite.candidate_id),
                        parent_candidate_key=str(elite.candidate_key),
                        parent_reward=float(elite.reward),
                        parent_rank_among_elites=int(elite_rank_index),
                        sample_metadata={
                            "init_mode": init_mode,
                            "sample_origin": "elite_centered",
                            "fixed_policy": False,
                            "concentration_scale": float(
                                self.cem_config.resampling.local_concentration_scale
                            ),
                            "local_concentration_scale": float(
                                self.cem_config.resampling.local_concentration_scale
                            ),
                            "parent_iteration": int(elite.iteration),
                            "parent_candidate_id": int(elite.candidate_id),
                            "parent_candidate_key": str(elite.candidate_key),
                            "parent_reward": float(elite.reward),
                            "parent_rank_among_elites": int(elite_rank_index),
                            **self._sample_projection_metadata(),
                        },
                    )
                )
        if len(sample_plan) != int(population_size):
            raise RuntimeError("Elite-centered sample allocation size mismatch.")
        return sample_plan


def candidate_key(iteration: int, candidate_id: int) -> str:
    return f"iter{int(iteration)}_cand{int(candidate_id)}"


@dataclass(frozen=True)
class _CandidateSampleSpec:
    center_policy: GroupActionPolicy
    concentration_scale: float
    sample_origin: str
    fixed_policy: GroupActionPolicy | None = None
    sample_metadata: dict[str, object] = field(default_factory=dict)
    parent_iteration: int | None = None
    parent_candidate_id: int | None = None
    parent_candidate_key: str | None = None
    parent_reward: float | None = None
    parent_rank_among_elites: int | None = None


def _sample_candidate_policy_from_center(
    *,
    center_policy: GroupActionPolicy,
    candidate_seed: int,
    concentration_scale: float,
    min_probability: float,
    max_probability: float,
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> GroupActionPolicy:
    return _sample_candidate_policy(
        center_policy,
        seed=int(candidate_seed),
        concentration_scale=float(concentration_scale),
        min_probability=float(min_probability),
        max_probability=float(max_probability),
        disable_consume_one_when_suffix_len_leq_1=(
            disable_consume_one_when_suffix_len_leq_1
        ),
    )


def _sample_candidate_policy(
    center_policy: GroupActionPolicy,
    *,
    seed: int,
    concentration_scale: float,
    min_probability: float,
    max_probability: float,
    disable_consume_one_when_suffix_len_leq_1: bool,
) -> GroupActionPolicy:
    sampled: dict[str, dict[str, float]] = {}
    for group, probabilities in center_policy.group_probabilities.items():
        action_names = list(probabilities.keys())
        alpha = [
            max(float(probabilities[action]), 1e-12) * float(concentration_scale)
            for action in action_names
        ]
        sampled_values = _sample_dirichlet(alpha, seed=_group_seed(seed, group))
        sampled[group] = _bounded_probability_mapping(
            {
                action: float(value)
                for action, value in zip(action_names, sampled_values)
            },
            min_probability=float(min_probability),
            max_probability=float(max_probability),
        )
    return GroupActionPolicy(
        sampled,
        valid_actions_by_group=center_policy.valid_actions_by_group,
        enabled_actions=center_policy.enabled_actions,
        disable_consume_one_when_suffix_len_leq_1=(
            disable_consume_one_when_suffix_len_leq_1
        ),
    )


def _allocate_children_to_elites(
    *,
    population_size: int,
    elite_count: int,
) -> list[int]:
    population = int(population_size)
    elites = int(elite_count)
    if population < 0:
        raise ValueError("population_size must be non-negative.")
    if elites <= 0:
        raise ValueError("elite_count must be positive.")
    base = population // elites
    remainder = population % elites
    return [
        int(base + (1 if elite_index < remainder else 0))
        for elite_index in range(elites)
    ]


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
        valid_actions_by_group=old_policy.valid_actions_by_group,
        enabled_actions=old_policy.enabled_actions,
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
        label="probability mapping",
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
    label: str = "group",
) -> None:
    if int(group_count) <= 0:
        raise ValueError(f"{label} group_count must be positive.")
    if float(min_probability) * int(group_count) > 1.0 + 1e-12:
        raise ValueError(
            f"min_probability is infeasible for the number of actions in {label}."
        )
    if float(max_probability) * int(group_count) < 1.0 - 1e-12:
        raise ValueError(
            f"max_probability is infeasible for the number of actions in {label}."
        )


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
    "PTSCEMResamplingConfig",
    "PTSCEMResult",
    "PTSCEMSamplerConfig",
    "PTSCEMUpdateConfig",
    "PTSGroupedCEMTrainer",
    "_allocate_children_to_elites",
    "candidate_key",
]
