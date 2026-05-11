from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from attack.pts.grouping import default_suffix_length_buckets
from attack.pts.policy import CONSUME_ONE_ACTION_NAME, build_valid_actions_by_group
from attack.pts.specs import get_default_pts_v1_specs


DEFAULT_OUTPUT_DIR = Path(
    "outputs/diagnostics/pts_space_filling_initial_population_stratified"
)
GROUP_ORDER = ("suffix_1", "suffix_2", "suffix_3plus")
FLOAT_TOLERANCE = 1.0e-8
C1_GENERATE_ACTION_NAME = "consume_one_generate_continuation"
CONCEPTUAL_ACTION_ORDER = (
    "c0_preserve",
    "c0_generate",
    "c1_preserve",
    "c1_generate",
    "stop",
)
ACTION_CONCEPTUAL_ALIASES = {
    "keep_residual_suffix": "c0_preserve",
    "regenerate_residual_suffix": "c0_generate",
    "consume_one_keep_rest": "c1_preserve",
    "consume_one_generate_continuation": "c1_generate",
    "consume_all_stop": "stop",
    "c0_preserve": "c0_preserve",
    "c0_generate": "c0_generate",
    "c1_preserve": "c1_preserve",
    "c1_generate": "c1_generate",
    "stop": "stop",
}
METADATA_FIELDS = [
    "c1_generate_available_in_production",
    "diagnostic_c1_generate_injected",
    "selected_candidate_count",
    "expected_candidate_count_for_planned_5_action_space",
]


@dataclass(frozen=True)
class SpaceFillingConfig:
    seed: int = 20260405
    mandatory_enabled: bool = True
    extreme_count: int = 7
    moderate_count: int = 3
    balanced_count: int = 1
    extreme_pool_size: int = 1024
    moderate_pool_size: int = 512
    extreme_alpha: float = 0.3
    moderate_alpha: float = 2.0
    min_probability: float = 0.03
    max_probability: float = 0.90
    distance: str = "l1"
    include_diagnostic_c1_generate: bool = False
    require_c1_generate: bool = False
    inspect_elite_children: bool = True
    elite_child_count: int = 8
    elite_child_seed: int | None = None
    local_concentration_scales: tuple[float, ...] = (10.0, 30.0, 60.0, 100.0)
    elite_example_mode: str = "both"
    output_dir: Path = DEFAULT_OUTPUT_DIR

    @property
    def selected_size(self) -> int:
        mandatory_count = 5 if self.mandatory_enabled else 0
        return int(
            mandatory_count
            + self.extreme_count
            + self.moderate_count
            + self.balanced_count
        )

    @property
    def expected_candidate_count_for_planned_5_action_space(self) -> int:
        return int(5 + self.extreme_count + self.moderate_count + self.balanced_count)

    @property
    def effective_elite_child_seed(self) -> int:
        if self.elite_child_seed is not None:
            return int(self.elite_child_seed)
        return int(self.seed) + 2_000_003


@dataclass(frozen=True)
class ActionSpaceInfo:
    production_actions: list[str]
    enabled_actions: list[str]
    valid_actions_by_group: dict[str, list[str]]
    conceptual_aliases_by_group: dict[str, list[str]]
    c1_generate_available_in_production: bool
    diagnostic_c1_generate_injected: bool
    c1_generate_available: bool
    c1_generate_missing: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PoolCandidate:
    pool_index: int | None
    source_sampler: str
    policy: dict[str, dict[str, float]]
    vertex_name: str | None = None


@dataclass(frozen=True)
class SelectedCandidate:
    candidate_id: int
    pool_index: int | None
    source_sampler: str
    vertex_name: str | None
    distance_to_uniform: float
    min_distance_to_previous_selected: float | None
    policy: dict[str, dict[str, float]]
    entropy_by_group: dict[str, float]
    max_probability_by_group: dict[str, float]
    dominant_action_by_group: dict[str, str]
    dominant_conceptual_action_by_group: dict[str, str]
    dominant_probability_by_group: dict[str, float]

    def to_dict(
        self,
        diagnostic_metadata: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "candidate_id": int(self.candidate_id),
            "pool_index": None if self.pool_index is None else int(self.pool_index),
            "source_sampler": str(self.source_sampler),
            "vertex_name": self.vertex_name,
            "distance_to_uniform": float(self.distance_to_uniform),
            "min_distance_to_previous_selected": (
                None
                if self.min_distance_to_previous_selected is None
                else float(self.min_distance_to_previous_selected)
            ),
            "policy": {
                group: dict(probabilities)
                for group, probabilities in self.policy.items()
            },
            "entropy_by_group": dict(self.entropy_by_group),
            "max_probability_by_group": dict(self.max_probability_by_group),
            "dominant_action_by_group": dict(self.dominant_action_by_group),
            "dominant_conceptual_action_by_group": dict(
                self.dominant_conceptual_action_by_group
            ),
            "dominant_probability_by_group": dict(
                self.dominant_probability_by_group
            ),
        }
        if diagnostic_metadata is not None:
            payload.update({field: diagnostic_metadata[field] for field in METADATA_FIELDS})
        return payload


def main() -> None:
    config = parse_args()
    selected, pool_summary, pairwise = run_diagnostic(config)
    write_outputs(
        selected_candidates=selected,
        pool_summary=pool_summary,
        pairwise_distances=pairwise,
        output_dir=config.output_dir,
    )
    print_summary(config, pool_summary)
    print_selected_candidates(selected)
    print_elite_child_console_summary(pool_summary.get("elite_child_summary_rows", []))
    print(f"\nWrote diagnostics to {config.output_dir}")


def parse_args() -> SpaceFillingConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect target-independent stratified space-filling initial "
            "PTS-CEM policy populations without changing trainer behavior."
        )
    )
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument(
        "--mandatory-enabled",
        dest="mandatory_enabled",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-mandatory-enabled",
        dest="mandatory_enabled",
        action="store_false",
    )
    parser.add_argument("--extreme-count", type=int, default=7)
    parser.add_argument("--moderate-count", type=int, default=3)
    parser.add_argument("--balanced-count", type=int, default=1)
    parser.add_argument("--extreme-pool-size", type=int, default=1024)
    parser.add_argument("--moderate-pool-size", type=int, default=512)
    parser.add_argument("--extreme-alpha", type=float, default=0.3)
    parser.add_argument("--moderate-alpha", type=float, default=2.0)
    parser.add_argument("--min-probability", type=float, default=0.03)
    parser.add_argument("--max-probability", type=float, default=0.90)
    parser.add_argument("--distance", choices=["l1"], default="l1")
    parser.add_argument("--include-diagnostic-c1-generate", action="store_true")
    parser.add_argument("--require-c1-generate", action="store_true")
    parser.add_argument(
        "--inspect-elite-children",
        dest="inspect_elite_children",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-inspect-elite-children",
        dest="inspect_elite_children",
        action="store_false",
    )
    parser.add_argument("--elite-child-count", type=int, default=8)
    parser.add_argument("--elite-child-seed", type=int, default=None)
    parser.add_argument("--local-concentration-scales", default="10,30,60,100")
    parser.add_argument(
        "--elite-example-mode",
        choices=["mandatory", "selected", "both"],
        default="both",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    return SpaceFillingConfig(
        seed=int(args.seed),
        mandatory_enabled=bool(args.mandatory_enabled),
        extreme_count=int(args.extreme_count),
        moderate_count=int(args.moderate_count),
        balanced_count=int(args.balanced_count),
        extreme_pool_size=int(args.extreme_pool_size),
        moderate_pool_size=int(args.moderate_pool_size),
        extreme_alpha=float(args.extreme_alpha),
        moderate_alpha=float(args.moderate_alpha),
        min_probability=float(args.min_probability),
        max_probability=float(args.max_probability),
        distance=str(args.distance),
        include_diagnostic_c1_generate=bool(args.include_diagnostic_c1_generate),
        require_c1_generate=bool(args.require_c1_generate),
        inspect_elite_children=bool(args.inspect_elite_children),
        elite_child_count=int(args.elite_child_count),
        elite_child_seed=(
            None if args.elite_child_seed is None else int(args.elite_child_seed)
        ),
        local_concentration_scales=parse_float_list(
            str(args.local_concentration_scales),
            label="local_concentration_scales",
        ),
        elite_example_mode=str(args.elite_example_mode),
        output_dir=Path(args.output_dir),
    )


def parse_float_list(raw: str, *, label: str) -> tuple[float, ...]:
    values: list[float] = []
    for piece in str(raw).split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError(f"{label} must contain at least one numeric value.")
    if any(value <= 0.0 for value in values):
        raise ValueError(f"{label} values must be positive.")
    return tuple(values)


def run_diagnostic(
    config: SpaceFillingConfig,
) -> tuple[list[SelectedCandidate], dict[str, object], list[list[float]]]:
    action_space = build_action_space_info(
        include_diagnostic_c1_generate=config.include_diagnostic_c1_generate,
        require_c1_generate=config.require_c1_generate,
    )
    valid_actions = action_space.valid_actions_by_group
    validate_config(config, valid_actions)
    uniform_policy = build_uniform_policy(valid_actions)
    mandatory_vertices = (
        build_mandatory_vertices(
            valid_actions_by_group=valid_actions,
            min_probability=config.min_probability,
            max_probability=config.max_probability,
            c1_generate_available=action_space.c1_generate_available,
        )
        if config.mandatory_enabled
        else []
    )
    extreme_pool = generate_policy_pool(
        pool_size=config.extreme_pool_size,
        source_sampler="extreme",
        alpha=config.extreme_alpha,
        seed=int(config.seed),
        valid_actions_by_group=valid_actions,
        min_probability=config.min_probability,
        max_probability=config.max_probability,
    )
    moderate_pool = generate_policy_pool(
        pool_size=config.moderate_pool_size,
        source_sampler="moderate",
        alpha=config.moderate_alpha,
        seed=int(config.seed) + 1_000_003,
        valid_actions_by_group=valid_actions,
        min_probability=config.min_probability,
        max_probability=config.max_probability,
    )

    selected_policies: list[PoolCandidate] = []
    selected_policies.extend(mandatory_vertices)
    selected_extreme = select_pool_candidates_greedy_maximin(
        pool=extreme_pool,
        count=config.extreme_count,
        uniform_policy=uniform_policy,
        reference_policies=[candidate.policy for candidate in mandatory_vertices],
    )
    selected_policies.extend(selected_extreme)
    selected_moderate = select_pool_candidates_greedy_maximin(
        pool=moderate_pool,
        count=config.moderate_count,
        uniform_policy=uniform_policy,
        reference_policies=[candidate.policy for candidate in selected_policies],
    )
    selected_policies.extend(selected_moderate)
    for _ in range(config.balanced_count):
        selected_policies.append(
            PoolCandidate(
                pool_index=None,
                source_sampler="balanced",
                vertex_name=None,
                policy=uniform_policy,
            )
        )

    selected = build_selected_candidates(selected_policies, uniform_policy)
    pairwise = build_pairwise_distance_matrix([candidate.policy for candidate in selected])
    metadata = build_diagnostic_metadata(
        config=config,
        action_space=action_space,
        selected_candidate_count=len(selected),
        mandatory_count=len(mandatory_vertices),
    )
    action_distribution_summary = build_action_distribution_summary(
        config=config,
        action_space=action_space,
        selected_candidates=selected,
        pairwise_distances=pairwise,
        diagnostic_metadata=metadata,
    )
    elite_child_examples: list[dict[str, object]] = []
    elite_child_summary_rows: list[dict[str, object]] = []
    if config.inspect_elite_children:
        elite_child_examples = sample_elite_children(
            config=config,
            selected_candidates=selected,
            valid_actions_by_group=valid_actions,
            uniform_policy=uniform_policy,
            c1_generate_available=action_space.c1_generate_available,
            diagnostic_metadata=metadata,
        )
        elite_child_summary_rows = build_elite_child_summary(elite_child_examples)
    summary = build_pool_summary(
        config=config,
        action_space=action_space,
        selected_candidates=selected,
        pairwise_distances=pairwise,
        diagnostic_metadata=metadata,
        action_distribution_summary=action_distribution_summary,
        mandatory_vertices=mandatory_vertices,
        elite_child_examples=elite_child_examples,
        elite_child_summary_rows=elite_child_summary_rows,
    )
    return selected, summary, pairwise


def default_valid_actions_by_group(
    *,
    include_diagnostic_c1_generate: bool = False,
    require_c1_generate: bool = False,
) -> dict[str, list[str]]:
    action_space = build_action_space_info(
        include_diagnostic_c1_generate=include_diagnostic_c1_generate,
        require_c1_generate=require_c1_generate,
    )
    return {group: list(actions) for group, actions in action_space.valid_actions_by_group.items()}


def build_action_space_info(
    *,
    include_diagnostic_c1_generate: bool,
    require_c1_generate: bool,
) -> ActionSpaceInfo:
    production_actions = [spec.name for spec in get_default_pts_v1_specs()]
    c1_generate_available_in_production = any(
        conceptual_action_name(action) == "c1_generate"
        for action in production_actions
    )
    enabled_actions = list(production_actions)
    diagnostic_c1_generate_injected = False
    if (
        bool(include_diagnostic_c1_generate)
        and not c1_generate_available_in_production
    ):
        enabled_actions.append(C1_GENERATE_ACTION_NAME)
        diagnostic_c1_generate_injected = True

    buckets = default_suffix_length_buckets()
    raw_valid_actions = build_valid_actions_by_group(
        group_buckets=buckets,
        enabled_actions=enabled_actions,
        disable_consume_one_when_suffix_len_leq_1=True,
    )
    valid_actions: dict[str, list[str]] = {}
    for group in GROUP_ORDER:
        filtered = [
            action
            for action in raw_valid_actions[group]
            if action_is_valid_for_group(action, group)
        ]
        valid_actions[group] = order_actions_by_concept(filtered)

    c1_generate_available = any(
        conceptual_action_name(action) == "c1_generate"
        for group in ("suffix_2", "suffix_3plus")
        for action in valid_actions[group]
    )
    c1_generate_missing = not c1_generate_available
    warnings: list[str] = []
    if c1_generate_missing:
        warnings.append(
            "WARNING: c1_generate / consume_one_generate_continuation is not "
            "available in the diagnostic action space; skipping the "
            "c1_generate_where_valid mandatory vertex."
        )
    if diagnostic_c1_generate_injected:
        warnings.append(
            "WARNING: injected diagnostic-only action "
            "consume_one_generate_continuation into suffix_2 and suffix_3plus; "
            "production PTS specs were not changed."
        )
    if bool(require_c1_generate) and c1_generate_missing:
        raise ValueError(
            "c1_generate is required but absent. Pass "
            "--include-diagnostic-c1-generate to inspect the planned diagnostic "
            "space without changing production specs."
        )
    conceptual_aliases = {
        group: [conceptual_action_name(action) for action in actions]
        for group, actions in valid_actions.items()
    }
    return ActionSpaceInfo(
        production_actions=production_actions,
        enabled_actions=enabled_actions,
        valid_actions_by_group=valid_actions,
        conceptual_aliases_by_group=conceptual_aliases,
        c1_generate_available_in_production=(
            c1_generate_available_in_production
        ),
        diagnostic_c1_generate_injected=diagnostic_c1_generate_injected,
        c1_generate_available=c1_generate_available,
        c1_generate_missing=c1_generate_missing,
        warnings=tuple(warnings),
    )


def conceptual_action_name(action_name: str) -> str:
    return ACTION_CONCEPTUAL_ALIASES.get(str(action_name), str(action_name))


def action_is_valid_for_group(action_name: str, group: str) -> bool:
    concept = conceptual_action_name(action_name)
    if str(group) == "suffix_1" and concept in {"c1_preserve", "c1_generate"}:
        return False
    return True


def order_actions_by_concept(actions: Sequence[str]) -> list[str]:
    concept_rank = {
        concept: index for index, concept in enumerate(CONCEPTUAL_ACTION_ORDER)
    }
    return sorted(
        [str(action) for action in actions],
        key=lambda action: (
            concept_rank.get(conceptual_action_name(action), len(concept_rank)),
            str(action),
        ),
    )


def build_diagnostic_metadata(
    *,
    config: SpaceFillingConfig,
    action_space: ActionSpaceInfo,
    selected_candidate_count: int,
    mandatory_count: int,
) -> dict[str, object]:
    return {
        "c1_generate_available_in_production": bool(
            action_space.c1_generate_available_in_production
        ),
        "diagnostic_c1_generate_injected": bool(
            action_space.diagnostic_c1_generate_injected
        ),
        "selected_candidate_count": int(selected_candidate_count),
        "expected_candidate_count_for_planned_5_action_space": int(
            config.expected_candidate_count_for_planned_5_action_space
        ),
        "mandatory_count": int(mandatory_count),
        "mandatory_enabled": bool(config.mandatory_enabled),
        "c1_generate_available": bool(action_space.c1_generate_available),
        "c1_generate_missing": bool(action_space.c1_generate_missing),
    }


def validate_config(
    config: SpaceFillingConfig,
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> None:
    if config.extreme_count < 0:
        raise ValueError("extreme_count must be >= 0.")
    if config.moderate_count < 0:
        raise ValueError("moderate_count must be >= 0.")
    if config.balanced_count not in {0, 1}:
        raise ValueError("balanced_count must be 0 or 1.")
    if config.selected_size <= 0:
        raise ValueError("Total selected count must be positive.")
    if config.extreme_pool_size < config.extreme_count:
        raise ValueError("extreme_pool_size must be >= extreme_count.")
    if config.moderate_pool_size < config.moderate_count:
        raise ValueError("moderate_pool_size must be >= moderate_count.")
    if config.extreme_alpha <= 0.0:
        raise ValueError("extreme_alpha must be positive.")
    if config.moderate_alpha <= 0.0:
        raise ValueError("moderate_alpha must be positive.")
    if not 0.0 <= config.min_probability < config.max_probability <= 1.0:
        raise ValueError("Require 0 <= min_probability < max_probability <= 1.")
    if config.distance != "l1":
        raise ValueError("Only distance='l1' is supported.")
    if config.elite_child_count < 0:
        raise ValueError("elite_child_count must be >= 0.")
    if config.elite_example_mode not in {"mandatory", "selected", "both"}:
        raise ValueError("elite_example_mode must be mandatory, selected, or both.")
    for group, actions in valid_actions_by_group.items():
        if not actions:
            raise ValueError(f"Valid action set for {group!r} must not be empty.")
        count = len(actions)
        if count * config.min_probability > 1.0 + FLOAT_TOLERANCE:
            raise ValueError(
                f"min_probability is infeasible for group {group!r} with "
                f"{count} valid actions."
            )
        if count * config.max_probability < 1.0 - FLOAT_TOLERANCE:
            raise ValueError(
                f"max_probability is infeasible for group {group!r} with "
                f"{count} valid actions."
            )
    suffix_1_concepts = {
        conceptual_action_name(action)
        for action in valid_actions_by_group.get("suffix_1", ())
    }
    if suffix_1_concepts & {"c1_preserve", "c1_generate"}:
        raise ValueError("suffix_1 must not contain c1 preserve/generate actions.")


def build_mandatory_vertices(
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    min_probability: float,
    max_probability: float,
    c1_generate_available: bool,
) -> list[PoolCandidate]:
    definitions: list[tuple[str, dict[str, str]]] = [
        (
            "c0_preserve",
            {group: "c0_preserve" for group in GROUP_ORDER},
        ),
        (
            "c0_generate",
            {group: "c0_generate" for group in GROUP_ORDER},
        ),
        (
            "c1_preserve_where_valid",
            {
                "suffix_1": "stop",
                "suffix_2": "c1_preserve",
                "suffix_3plus": "c1_preserve",
            },
        ),
    ]
    if c1_generate_available:
        definitions.append(
            (
                "c1_generate_where_valid",
                {
                    "suffix_1": "stop",
                    "suffix_2": "c1_generate",
                    "suffix_3plus": "c1_generate",
                },
            )
        )
    definitions.append(("stop", {group: "stop" for group in GROUP_ORDER}))

    vertices: list[PoolCandidate] = []
    for vertex_name, dominant_concepts in definitions:
        vertices.append(
            PoolCandidate(
                pool_index=None,
                source_sampler="mandatory_vertex",
                vertex_name=vertex_name,
                policy=build_near_vertex_policy(
                    valid_actions_by_group=valid_actions_by_group,
                    dominant_concept_by_group=dominant_concepts,
                    min_probability=min_probability,
                    max_probability=max_probability,
                ),
            )
        )
    return vertices


def build_near_vertex_policy(
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    dominant_concept_by_group: Mapping[str, str],
    min_probability: float,
    max_probability: float,
) -> dict[str, dict[str, float]]:
    policy: dict[str, dict[str, float]] = {}
    for group in GROUP_ORDER:
        actions = list(valid_actions_by_group[group])
        concept = str(dominant_concept_by_group[group])
        dominant_action = find_action_for_concept(actions, concept)
        if dominant_action is None:
            raise ValueError(
                f"Cannot build near-vertex policy for group {group!r}: "
                f"concept {concept!r} is absent from valid actions {actions}."
            )
        policy[group] = build_near_vertex_distribution(
            actions=actions,
            dominant_action=dominant_action,
            min_probability=min_probability,
            max_probability=max_probability,
        )
    return policy


def build_near_vertex_distribution(
    *,
    actions: Sequence[str],
    dominant_action: str,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    ordered_actions = [str(action) for action in actions]
    if str(dominant_action) not in ordered_actions:
        raise ValueError(f"Dominant action {dominant_action!r} is not valid.")
    count = len(ordered_actions)
    if count == 1:
        return {ordered_actions[0]: 1.0}
    high = min(float(max_probability), 1.0 - float(count - 1) * min_probability)
    low = (1.0 - high) / float(count - 1)
    probabilities = {
        action: (high if action == str(dominant_action) else low)
        for action in ordered_actions
    }
    _validate_probability_bounds(
        list(probabilities.values()),
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return {action: float(probability) for action, probability in probabilities.items()}


def find_action_for_concept(
    actions: Sequence[str],
    concept: str,
) -> str | None:
    for action in actions:
        if conceptual_action_name(action) == str(concept):
            return str(action)
    return None


def generate_policy_pool(
    *,
    pool_size: int,
    source_sampler: str,
    alpha: float,
    seed: int,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    min_probability: float,
    max_probability: float,
) -> list[PoolCandidate]:
    rng = np.random.default_rng(int(seed))
    pool: list[PoolCandidate] = []
    for pool_index in range(int(pool_size)):
        policy = {
            group: _sample_group_distribution(
                rng=rng,
                actions=actions,
                alpha=alpha,
                min_probability=min_probability,
                max_probability=max_probability,
            )
            for group, actions in valid_actions_by_group.items()
        }
        validate_policy(
            policy,
            valid_actions_by_group,
            min_probability=min_probability,
            max_probability=max_probability,
        )
        pool.append(
            PoolCandidate(
                pool_index=pool_index,
                source_sampler=source_sampler,
                policy=policy,
            )
        )
    return pool


def _sample_group_distribution(
    *,
    rng: np.random.Generator,
    actions: Sequence[str],
    alpha: float,
    min_probability: float,
    max_probability: float,
) -> dict[str, float]:
    values = rng.dirichlet([float(alpha)] * len(actions))
    bounded = project_to_bounded_simplex(
        values,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return {
        str(action): float(value)
        for action, value in zip(actions, bounded, strict=True)
    }


def project_to_bounded_simplex(
    values: Sequence[float],
    *,
    min_probability: float,
    max_probability: float,
) -> list[float]:
    raw = np.asarray([max(0.0, float(value)) for value in values], dtype=float)
    count = int(raw.size)
    if count <= 0:
        raise ValueError("Cannot project an empty probability vector.")
    if count * min_probability > 1.0 + FLOAT_TOLERANCE:
        raise ValueError("min_probability is infeasible for this vector.")
    if count * max_probability < 1.0 - FLOAT_TOLERANCE:
        raise ValueError("max_probability is infeasible for this vector.")

    if float(raw.sum()) <= 0.0:
        raw = np.full(count, 1.0 / count, dtype=float)
    else:
        raw = raw / float(raw.sum())

    lower_tau = float(np.min(raw - max_probability))
    upper_tau = float(np.max(raw - min_probability))
    result = np.clip(raw, min_probability, max_probability)
    for _ in range(100):
        tau = (lower_tau + upper_tau) / 2.0
        candidate = np.clip(raw - tau, min_probability, max_probability)
        total = float(candidate.sum())
        result = candidate
        if abs(total - 1.0) <= FLOAT_TOLERANCE:
            break
        if total > 1.0:
            lower_tau = tau
        else:
            upper_tau = tau

    total = float(result.sum())
    if abs(total - 1.0) > FLOAT_TOLERANCE:
        result = result / total
    _validate_probability_bounds(
        result,
        min_probability=min_probability,
        max_probability=max_probability,
    )
    return [float(value) for value in result]


def _validate_probability_bounds(
    probabilities: Sequence[float],
    *,
    min_probability: float,
    max_probability: float,
) -> None:
    total = float(sum(probabilities))
    if abs(total - 1.0) > FLOAT_TOLERANCE:
        raise ValueError(f"Probability vector sums to {total}, not 1.")
    for value in probabilities:
        if float(value) < min_probability - FLOAT_TOLERANCE:
            raise ValueError("Probability vector violates the lower bound.")
        if float(value) > max_probability + FLOAT_TOLERANCE:
            raise ValueError("Probability vector violates the upper bound.")


def build_uniform_policy(
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, float]]:
    return {
        str(group): {
            str(action): 1.0 / float(len(actions))
            for action in actions
        }
        for group, actions in valid_actions_by_group.items()
    }


def select_pool_candidates_greedy_maximin(
    *,
    pool: Sequence[PoolCandidate],
    count: int,
    uniform_policy: Mapping[str, Mapping[str, float]],
    reference_policies: Sequence[Mapping[str, Mapping[str, float]]],
) -> list[PoolCandidate]:
    if count < 0:
        raise ValueError("count must be >= 0.")
    if count > len(pool):
        raise ValueError("count must be <= pool length.")
    selected_indices: list[int] = []
    references = [dict(policy) for policy in reference_policies]
    distance_to_uniform = [
        policy_l1_distance(candidate.policy, uniform_policy)
        for candidate in pool
    ]

    while len(selected_indices) < count:
        selected_set = set(selected_indices)
        best_index: int | None = None
        best_distance = -1.0
        for index, candidate in enumerate(pool):
            if index in selected_set:
                continue
            if not references:
                candidate_distance = distance_to_uniform[index]
            else:
                candidate_distance = min(
                    policy_l1_distance(candidate.policy, reference)
                    for reference in references
                )
            if (
                candidate_distance > best_distance + FLOAT_TOLERANCE
                or (
                    abs(candidate_distance - best_distance) <= FLOAT_TOLERANCE
                    and (
                        best_index is None
                        or int(candidate.pool_index or 0)
                        < int(pool[best_index].pool_index or 0)
                    )
                )
            ):
                best_distance = candidate_distance
                best_index = index
        if best_index is None:
            raise ValueError("Could not select a next maximin candidate.")
        selected_indices.append(best_index)
        references.append(pool[best_index].policy)

    return [pool[index] for index in selected_indices]


def build_selected_candidates(
    selected_pool_candidates: Sequence[PoolCandidate],
    uniform_policy: Mapping[str, Mapping[str, float]],
) -> list[SelectedCandidate]:
    selected: list[SelectedCandidate] = []
    previous_policies: list[Mapping[str, Mapping[str, float]]] = []
    for candidate_id, pool_candidate in enumerate(selected_pool_candidates):
        if previous_policies:
            min_distance_to_previous = min(
                policy_l1_distance(pool_candidate.policy, previous_policy)
                for previous_policy in previous_policies
            )
        else:
            min_distance_to_previous = None
        dominant_actions = {
            group: max(
                probabilities.items(),
                key=lambda item: (item[1], item[0]),
            )[0]
            for group, probabilities in pool_candidate.policy.items()
        }
        selected.append(
            SelectedCandidate(
                candidate_id=candidate_id,
                pool_index=pool_candidate.pool_index,
                source_sampler=pool_candidate.source_sampler,
                vertex_name=pool_candidate.vertex_name,
                distance_to_uniform=policy_l1_distance(
                    pool_candidate.policy,
                    uniform_policy,
                ),
                min_distance_to_previous_selected=min_distance_to_previous,
                policy=pool_candidate.policy,
                entropy_by_group={
                    group: entropy(probabilities)
                    for group, probabilities in pool_candidate.policy.items()
                },
                max_probability_by_group={
                    group: max(probabilities.values())
                    for group, probabilities in pool_candidate.policy.items()
                },
                dominant_action_by_group=dominant_actions,
                dominant_conceptual_action_by_group={
                    group: conceptual_action_name(action)
                    for group, action in dominant_actions.items()
                },
                dominant_probability_by_group={
                    group: pool_candidate.policy[group][action]
                    for group, action in dominant_actions.items()
                },
            )
        )
        previous_policies.append(pool_candidate.policy)
    return selected


def policy_l1_distance(
    policy_a: Mapping[str, Mapping[str, float]],
    policy_b: Mapping[str, Mapping[str, float]],
) -> float:
    distance = 0.0
    for group in GROUP_ORDER:
        actions = sorted(set(policy_a[group]) | set(policy_b[group]))
        distance += sum(
            abs(
                float(policy_a[group].get(action, 0.0))
                - float(policy_b[group].get(action, 0.0))
            )
            for action in actions
        )
    return float(distance)


def entropy(probabilities: Mapping[str, float]) -> float:
    total = 0.0
    for probability in probabilities.values():
        value = float(probability)
        if value > 0.0:
            total -= value * math.log(value)
    return float(total)


def validate_policy(
    policy: Mapping[str, Mapping[str, float]],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    *,
    min_probability: float | None = None,
    max_probability: float | None = None,
) -> None:
    for group, actions in valid_actions_by_group.items():
        expected = list(actions)
        probabilities = policy.get(group)
        if probabilities is None:
            raise ValueError(f"Policy is missing group {group!r}.")
        if list(probabilities.keys()) != expected:
            raise ValueError(
                f"Policy actions for group {group!r} must match valid actions."
            )
        values = [float(value) for value in probabilities.values()]
        _validate_probability_bounds(
            values,
            min_probability=0.0 if min_probability is None else min_probability,
            max_probability=1.0 if max_probability is None else max_probability,
        )
    suffix_1_concepts = {
        conceptual_action_name(action) for action in policy.get("suffix_1", {})
    }
    if suffix_1_concepts & {"c1_preserve", "c1_generate"}:
        raise ValueError("suffix_1 policy must not contain c1 preserve/generate.")


def count_probability_bound_violations(
    selected_candidates: Sequence[SelectedCandidate],
    *,
    min_probability: float,
    max_probability: float,
) -> int:
    count = 0
    for candidate in selected_candidates:
        for probabilities in candidate.policy.values():
            for value in probabilities.values():
                if (
                    float(value) < min_probability - FLOAT_TOLERANCE
                    or float(value) > max_probability + FLOAT_TOLERANCE
                ):
                    count += 1
    return count


def build_pairwise_distance_matrix(
    policies: Sequence[Mapping[str, Mapping[str, float]]],
) -> list[list[float]]:
    return [
        [policy_l1_distance(policy_a, policy_b) for policy_b in policies]
        for policy_a in policies
    ]


def build_action_distribution_summary(
    *,
    config: SpaceFillingConfig,
    action_space: ActionSpaceInfo,
    selected_candidates: Sequence[SelectedCandidate],
    pairwise_distances: Sequence[Sequence[float]],
    diagnostic_metadata: Mapping[str, object],
) -> dict[str, object]:
    entropy_stats = {
        group: _summary_stats(
            [candidate.entropy_by_group[group] for candidate in selected_candidates]
        )
        for group in GROUP_ORDER
    }
    max_probability_stats = {
        group: _summary_stats(
            [
                candidate.max_probability_by_group[group]
                for candidate in selected_candidates
            ]
        )
        for group in GROUP_ORDER
    }
    distance_to_uniform = [
        candidate.distance_to_uniform for candidate in selected_candidates
    ]
    off_diagonal = [
        float(value)
        for row_index, row in enumerate(pairwise_distances)
        for column_index, value in enumerate(row)
        if row_index != column_index
    ]
    source_counts = Counter(
        candidate.source_sampler for candidate in selected_candidates
    )
    dominant_action_counts = {
        group: dict(
            Counter(
                candidate.dominant_action_by_group[group]
                for candidate in selected_candidates
            )
        )
        for group in GROUP_ORDER
    }
    dominant_conceptual_action_counts = {
        group: dict(
            Counter(
                candidate.dominant_conceptual_action_by_group[group]
                for candidate in selected_candidates
            )
        )
        for group in GROUP_ORDER
    }
    probability_stats_by_group_action: dict[str, dict[str, dict[str, float | None]]] = {}
    average_probability_by_group_action: dict[str, dict[str, float | None]] = {}
    for group in GROUP_ORDER:
        probability_stats_by_group_action[group] = {}
        average_probability_by_group_action[group] = {}
        for action in action_space.valid_actions_by_group[group]:
            values = [
                float(candidate.policy[group][action])
                for candidate in selected_candidates
            ]
            stats = _summary_stats(values)
            probability_stats_by_group_action[group][action] = stats
            average_probability_by_group_action[group][action] = stats["mean"]

    payload: dict[str, object] = {
        **{field: diagnostic_metadata[field] for field in METADATA_FIELDS},
        "count_by_source_sampler": dict(source_counts),
        "dominant_action_count_by_group": dominant_action_counts,
        "dominant_conceptual_action_count_by_group": (
            dominant_conceptual_action_counts
        ),
        "average_probability_per_action_by_group": (
            average_probability_by_group_action
        ),
        "probability_stats_per_action_by_group": (
            probability_stats_by_group_action
        ),
        "entropy_min_mean_max_by_group": entropy_stats,
        "max_probability_min_mean_max_by_group": max_probability_stats,
        "pairwise_l1_min_mean_max": _summary_stats(off_diagonal),
        "distance_to_uniform_min_mean_max": _summary_stats(distance_to_uniform),
        "probability_bound_violations": count_probability_bound_violations(
            selected_candidates,
            min_probability=config.min_probability,
            max_probability=config.max_probability,
        ),
    }
    return payload


def build_pool_summary(
    *,
    config: SpaceFillingConfig,
    action_space: ActionSpaceInfo,
    selected_candidates: Sequence[SelectedCandidate],
    pairwise_distances: Sequence[Sequence[float]],
    diagnostic_metadata: Mapping[str, object],
    action_distribution_summary: Mapping[str, object],
    mandatory_vertices: Sequence[PoolCandidate],
    elite_child_examples: Sequence[Mapping[str, object]],
    elite_child_summary_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        **dict(diagnostic_metadata),
        "seed": int(config.seed),
        "mandatory_enabled": bool(config.mandatory_enabled),
        "mandatory_count": int(len(mandatory_vertices)),
        "extreme_count": int(config.extreme_count),
        "moderate_count": int(config.moderate_count),
        "balanced_count": int(config.balanced_count),
        "selected_size": int(len(selected_candidates)),
        "extreme_pool_size": int(config.extreme_pool_size),
        "moderate_pool_size": int(config.moderate_pool_size),
        "extreme_alpha": float(config.extreme_alpha),
        "moderate_alpha": float(config.moderate_alpha),
        "min_probability": float(config.min_probability),
        "max_probability": float(config.max_probability),
        "distance": str(config.distance),
        "include_diagnostic_c1_generate": bool(
            config.include_diagnostic_c1_generate
        ),
        "require_c1_generate": bool(config.require_c1_generate),
        "inspect_elite_children": bool(config.inspect_elite_children),
        "elite_child_count": int(config.elite_child_count),
        "elite_child_seed": int(config.effective_elite_child_seed),
        "local_concentration_scales": [
            float(value) for value in config.local_concentration_scales
        ],
        "elite_example_mode": str(config.elite_example_mode),
        "production_actions": list(action_space.production_actions),
        "enabled_actions": list(action_space.enabled_actions),
        "valid_actions_by_group": {
            group: list(actions)
            for group, actions in action_space.valid_actions_by_group.items()
        },
        "conceptual_action_aliases_by_group": {
            group: list(actions)
            for group, actions in action_space.conceptual_aliases_by_group.items()
        },
        "warnings": list(action_space.warnings),
        "selected_count_by_source_sampler": dict(
            action_distribution_summary["count_by_source_sampler"]
        ),
        "dominant_action_counts_by_group": dict(
            action_distribution_summary["dominant_action_count_by_group"]
        ),
        "dominant_conceptual_action_counts_by_group": dict(
            action_distribution_summary[
                "dominant_conceptual_action_count_by_group"
            ]
        ),
        "selected_entropy_by_group": dict(
            action_distribution_summary["entropy_min_mean_max_by_group"]
        ),
        "selected_max_probability_by_group": dict(
            action_distribution_summary["max_probability_min_mean_max_by_group"]
        ),
        "distance_to_uniform": dict(
            action_distribution_summary["distance_to_uniform_min_mean_max"]
        ),
        "pairwise_selected_distance": dict(
            action_distribution_summary["pairwise_l1_min_mean_max"]
        ),
        "probability_bound_violations_count": int(
            action_distribution_summary["probability_bound_violations"]
        ),
        "action_distribution_summary": dict(action_distribution_summary),
        "elite_child_examples": list(elite_child_examples),
        "elite_child_summary_rows": list(elite_child_summary_rows),
    }
    return payload


def _summary_stats(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    normalized = [float(value) for value in values]
    return {
        "min": min(normalized),
        "mean": sum(normalized) / float(len(normalized)),
        "max": max(normalized),
    }


def sample_elite_children(
    *,
    config: SpaceFillingConfig,
    selected_candidates: Sequence[SelectedCandidate],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    uniform_policy: Mapping[str, Mapping[str, float]],
    c1_generate_available: bool,
    diagnostic_metadata: Mapping[str, object],
) -> list[dict[str, object]]:
    rng = np.random.default_rng(config.effective_elite_child_seed)
    parents = build_elite_parent_examples(
        config=config,
        selected_candidates=selected_candidates,
        valid_actions_by_group=valid_actions_by_group,
        c1_generate_available=c1_generate_available,
    )
    rows: list[dict[str, object]] = []
    for parent in parents:
        parent_policy = parent["policy"]
        if not isinstance(parent_policy, Mapping):
            raise TypeError("Parent policy must be a mapping.")
        parent_dominants = dominant_concepts_for_policy(parent_policy)
        for scale in config.local_concentration_scales:
            for child_id in range(config.elite_child_count):
                child_policy = sample_child_policy_from_parent(
                    rng=rng,
                    parent_policy=parent_policy,
                    valid_actions_by_group=valid_actions_by_group,
                    local_concentration_scale=float(scale),
                    min_probability=config.min_probability,
                    max_probability=config.max_probability,
                )
                validate_policy(
                    child_policy,
                    valid_actions_by_group,
                    min_probability=config.min_probability,
                    max_probability=config.max_probability,
                )
                child_dominants = dominant_concepts_for_policy(child_policy)
                row: dict[str, object] = {
                    **{field: diagnostic_metadata[field] for field in METADATA_FIELDS},
                    "parent_name": str(parent["parent_name"]),
                    "parent_source_sampler": str(parent["parent_source_sampler"]),
                    "parent_vertex_name": parent.get("parent_vertex_name"),
                    "scale": float(scale),
                    "child_id": int(child_id),
                    "child_parent_l1": policy_l1_distance(
                        child_policy,
                        parent_policy,
                    ),
                    "child_uniform_l1": policy_l1_distance(
                        child_policy,
                        uniform_policy,
                    ),
                    "child_policy": child_policy,
                    "entropy_by_group": {
                        group: entropy(child_policy[group])
                        for group in GROUP_ORDER
                    },
                    "max_probability_by_group": {
                        group: max(child_policy[group].values())
                        for group in GROUP_ORDER
                    },
                }
                for group in GROUP_ORDER:
                    row[f"{group}_parent_dom"] = parent_dominants[group]
                    row[f"{group}_child_dom"] = child_dominants[group]
                    row[f"{group}_same_dom"] = (
                        parent_dominants[group] == child_dominants[group]
                    )
                rows.append(row)
    return rows


def build_elite_parent_examples(
    *,
    config: SpaceFillingConfig,
    selected_candidates: Sequence[SelectedCandidate],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    c1_generate_available: bool,
) -> list[dict[str, object]]:
    parents: list[dict[str, object]] = []
    if config.elite_example_mode in {"mandatory", "both"}:
        for candidate in selected_candidates:
            if candidate.source_sampler != "mandatory_vertex":
                continue
            parents.append(
                {
                    "parent_name": f"{candidate.vertex_name}_vertex",
                    "parent_source_sampler": candidate.source_sampler,
                    "parent_vertex_name": candidate.vertex_name,
                    "policy": candidate.policy,
                }
            )
    if config.elite_example_mode in {"selected", "both"}:
        extreme_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.source_sampler == "extreme"
        ]
        if extreme_candidates:
            extreme_parent = max(
                extreme_candidates,
                key=lambda candidate: (
                    candidate.distance_to_uniform,
                    -candidate.candidate_id,
                ),
            )
            parents.append(
                {
                    "parent_name": (
                        "selected_extreme_largest_distance_to_uniform_"
                        f"candidate_{extreme_parent.candidate_id}"
                    ),
                    "parent_source_sampler": extreme_parent.source_sampler,
                    "parent_vertex_name": extreme_parent.vertex_name,
                    "policy": extreme_parent.policy,
                }
            )
        moderate_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.source_sampler == "moderate"
        ]
        if moderate_candidates:
            moderate_parent = min(
                moderate_candidates,
                key=lambda candidate: (
                    candidate.distance_to_uniform,
                    candidate.candidate_id,
                ),
            )
            parents.append(
                {
                    "parent_name": (
                        "selected_moderate_smallest_distance_to_uniform_"
                        f"candidate_{moderate_parent.candidate_id}"
                    ),
                    "parent_source_sampler": moderate_parent.source_sampler,
                    "parent_vertex_name": moderate_parent.vertex_name,
                    "policy": moderate_parent.policy,
                }
            )
        balanced_candidates = [
            candidate
            for candidate in selected_candidates
            if candidate.source_sampler == "balanced"
        ]
        if balanced_candidates:
            balanced_parent = balanced_candidates[-1]
            parents.append(
                {
                    "parent_name": f"balanced_candidate_{balanced_parent.candidate_id}",
                    "parent_source_sampler": balanced_parent.source_sampler,
                    "parent_vertex_name": balanced_parent.vertex_name,
                    "policy": balanced_parent.policy,
                }
            )
    parents.append(
        {
            "parent_name": "synthetic_cross_group_mixed",
            "parent_source_sampler": "synthetic",
            "parent_vertex_name": "synthetic_cross_group_mixed",
            "policy": build_synthetic_cross_group_mixed_parent(
                valid_actions_by_group=valid_actions_by_group,
                min_probability=config.min_probability,
                max_probability=config.max_probability,
                c1_generate_available=c1_generate_available,
            ),
        }
    )
    return parents


def build_synthetic_cross_group_mixed_parent(
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    min_probability: float,
    max_probability: float,
    c1_generate_available: bool,
) -> dict[str, dict[str, float]]:
    suffix_3plus_concept = "c1_generate" if c1_generate_available else "c1_preserve"
    return build_near_vertex_policy(
        valid_actions_by_group=valid_actions_by_group,
        dominant_concept_by_group={
            "suffix_1": "stop",
            "suffix_2": "c0_generate",
            "suffix_3plus": suffix_3plus_concept,
        },
        min_probability=min_probability,
        max_probability=max_probability,
    )


def sample_child_policy_from_parent(
    *,
    rng: np.random.Generator,
    parent_policy: Mapping[str, Mapping[str, float]],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    local_concentration_scale: float,
    min_probability: float,
    max_probability: float,
) -> dict[str, dict[str, float]]:
    child: dict[str, dict[str, float]] = {}
    for group in GROUP_ORDER:
        actions = list(valid_actions_by_group[group])
        parent_values = np.asarray(
            [float(parent_policy[group][action]) for action in actions],
            dtype=float,
        )
        alpha_vector = np.maximum(
            parent_values * float(local_concentration_scale),
            1.0e-12,
        )
        sampled = rng.dirichlet(alpha_vector)
        bounded = project_to_bounded_simplex(
            sampled,
            min_probability=min_probability,
            max_probability=max_probability,
        )
        child[group] = {
            action: float(value)
            for action, value in zip(actions, bounded, strict=True)
        }
    return child


def dominant_concepts_for_policy(
    policy: Mapping[str, Mapping[str, float]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in GROUP_ORDER:
        dominant_action = max(
            policy[group].items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        result[group] = conceptual_action_name(dominant_action)
    return result


def build_elite_child_summary(
    examples: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[Mapping[str, object]]] = {}
    for row in examples:
        key = (str(row["parent_name"]), float(row["scale"]))
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (parent_name, scale), rows in sorted(grouped.items(), key=lambda item: item[0]):
        first = rows[0]
        child_parent_l1 = [float(row["child_parent_l1"]) for row in rows]
        child_uniform_l1 = [float(row["child_uniform_l1"]) for row in rows]
        payload: dict[str, object] = {
            **{field: first[field] for field in METADATA_FIELDS},
            "parent_name": parent_name,
            "parent_source_sampler": str(first["parent_source_sampler"]),
            "scale": float(scale),
            "child_count": int(len(rows)),
            "min_child_parent_l1": min(child_parent_l1),
            "mean_child_parent_l1": sum(child_parent_l1) / float(len(rows)),
            "max_child_parent_l1": max(child_parent_l1),
            "min_child_uniform_l1": min(child_uniform_l1),
            "mean_child_uniform_l1": sum(child_uniform_l1) / float(len(rows)),
            "max_child_uniform_l1": max(child_uniform_l1),
        }
        for group in GROUP_ORDER:
            same_values = [bool(row[f"{group}_same_dom"]) for row in rows]
            payload[f"{group}_same_dom_ratio"] = (
                sum(1 for value in same_values if value) / float(len(same_values))
            )
            payload[f"{group}_average_entropy"] = sum(
                float(row["entropy_by_group"][group]) for row in rows
            ) / float(len(rows))
            payload[f"{group}_average_max_probability"] = sum(
                float(row["max_probability_by_group"][group]) for row in rows
            ) / float(len(rows))
        summary_rows.append(payload)
    return summary_rows


def write_outputs(
    *,
    selected_candidates: Sequence[SelectedCandidate],
    pool_summary: Mapping[str, object],
    pairwise_distances: Sequence[Sequence[float]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = extract_metadata(pool_summary)
    valid_actions_by_group = coerce_valid_actions(
        pool_summary["valid_actions_by_group"]
    )
    (output_dir / "selected_candidates.json").write_text(
        json.dumps(
            [
                candidate.to_dict(diagnostic_metadata=metadata)
                for candidate in selected_candidates
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "pool_summary.json").write_text(
        json.dumps(dict(pool_summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    action_distribution_summary = dict(pool_summary["action_distribution_summary"])
    (output_dir / "action_distribution_summary.json").write_text(
        json.dumps(action_distribution_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_selected_candidates_csv(
        selected_candidates,
        output_dir / "selected_candidates.csv",
        valid_actions_by_group=valid_actions_by_group,
        diagnostic_metadata=metadata,
    )
    write_pairwise_distances_csv(
        pairwise_distances,
        output_dir / "pairwise_distances.csv",
        diagnostic_metadata=metadata,
    )
    write_action_distribution_summary_csv(
        action_distribution_summary,
        output_dir / "action_distribution_summary.csv",
        diagnostic_metadata=metadata,
    )
    write_elite_child_outputs(
        elite_child_examples=list(pool_summary.get("elite_child_examples", [])),
        elite_child_summary_rows=list(pool_summary.get("elite_child_summary_rows", [])),
        valid_actions_by_group=valid_actions_by_group,
        diagnostic_metadata=metadata,
        output_dir=output_dir,
    )
    write_markdown_report(
        selected_candidates=selected_candidates,
        pool_summary=pool_summary,
        pairwise_distances=pairwise_distances,
        output_dir=output_dir,
    )


def extract_metadata(source: Mapping[str, object]) -> dict[str, object]:
    return {field: source[field] for field in METADATA_FIELDS}


def coerce_valid_actions(payload: object) -> dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        raise TypeError("valid_actions_by_group must be a mapping.")
    return {
        str(group): [str(action) for action in actions]
        for group, actions in payload.items()
    }


def write_selected_candidates_csv(
    selected_candidates: Sequence[SelectedCandidate],
    path: Path,
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    diagnostic_metadata: Mapping[str, object],
) -> None:
    raw_probability_fields = [
        f"{group}_{action}"
        for group in GROUP_ORDER
        for action in valid_actions_by_group[group]
    ]
    conceptual_actions = concepts_present(valid_actions_by_group)
    conceptual_probability_fields = [
        f"{group}_{concept}"
        for group in GROUP_ORDER
        for concept in conceptual_actions
    ]
    fields = (
        METADATA_FIELDS
        + [
            "candidate_id",
            "source_sampler",
            "vertex_name",
            "pool_index",
            "distance_to_uniform",
            "min_distance_to_previous_selected",
        ]
        + raw_probability_fields
        + conceptual_probability_fields
        + [
            f"{group}_dominant_action" for group in GROUP_ORDER
        ]
        + [
            f"{group}_dominant_conceptual_action" for group in GROUP_ORDER
        ]
        + [
            f"{group}_dominant_prob" for group in GROUP_ORDER
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for candidate in selected_candidates:
            row: dict[str, object] = {
                **{field: diagnostic_metadata[field] for field in METADATA_FIELDS},
                "candidate_id": candidate.candidate_id,
                "source_sampler": candidate.source_sampler,
                "vertex_name": "" if candidate.vertex_name is None else candidate.vertex_name,
                "pool_index": "" if candidate.pool_index is None else candidate.pool_index,
                "distance_to_uniform": candidate.distance_to_uniform,
                "min_distance_to_previous_selected": (
                    ""
                    if candidate.min_distance_to_previous_selected is None
                    else candidate.min_distance_to_previous_selected
                ),
            }
            for group in GROUP_ORDER:
                for action in valid_actions_by_group[group]:
                    row[f"{group}_{action}"] = candidate.policy[group].get(action, "")
                for concept in conceptual_actions:
                    value = probability_for_concept(
                        candidate.policy[group],
                        concept,
                    )
                    row[f"{group}_{concept}"] = "" if value is None else value
                row[f"{group}_dominant_action"] = candidate.dominant_action_by_group[group]
                row[f"{group}_dominant_conceptual_action"] = (
                    candidate.dominant_conceptual_action_by_group[group]
                )
                row[f"{group}_dominant_prob"] = (
                    candidate.dominant_probability_by_group[group]
                )
            writer.writerow(row)


def concepts_present(
    valid_actions_by_group: Mapping[str, Sequence[str]],
) -> list[str]:
    present = {
        conceptual_action_name(action)
        for actions in valid_actions_by_group.values()
        for action in actions
    }
    return [concept for concept in CONCEPTUAL_ACTION_ORDER if concept in present]


def probability_for_concept(
    probabilities: Mapping[str, float],
    concept: str,
) -> float | None:
    values = [
        float(probability)
        for action, probability in probabilities.items()
        if conceptual_action_name(action) == str(concept)
    ]
    if not values:
        return None
    return float(sum(values))


def write_pairwise_distances_csv(
    pairwise_distances: Sequence[Sequence[float]],
    path: Path,
    *,
    diagnostic_metadata: Mapping[str, object],
) -> None:
    fields = METADATA_FIELDS + ["candidate_id"] + [
        f"candidate_{index}" for index in range(len(pairwise_distances))
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row_index, row in enumerate(pairwise_distances):
            payload: dict[str, object] = {
                **{field: diagnostic_metadata[field] for field in METADATA_FIELDS},
                "candidate_id": row_index,
            }
            payload.update(
                {
                    f"candidate_{column_index}": float(value)
                    for column_index, value in enumerate(row)
                }
            )
            writer.writerow(payload)


def write_action_distribution_summary_csv(
    summary: Mapping[str, object],
    path: Path,
    *,
    diagnostic_metadata: Mapping[str, object],
) -> None:
    fields = METADATA_FIELDS + [
        "section",
        "group",
        "action",
        "conceptual_action",
        "source_sampler",
        "metric",
        "count",
        "min",
        "mean",
        "max",
    ]
    rows: list[dict[str, object]] = []
    for source_sampler, count in dict(summary["count_by_source_sampler"]).items():
        rows.append(
            {
                "section": "count_by_source_sampler",
                "source_sampler": source_sampler,
                "count": count,
            }
        )
    for group, counts in dict(summary["dominant_action_count_by_group"]).items():
        for action, count in dict(counts).items():
            rows.append(
                {
                    "section": "dominant_action_count_by_group",
                    "group": group,
                    "action": action,
                    "conceptual_action": conceptual_action_name(action),
                    "count": count,
                }
            )
    for group, counts in dict(
        summary["dominant_conceptual_action_count_by_group"]
    ).items():
        for concept, count in dict(counts).items():
            rows.append(
                {
                    "section": "dominant_conceptual_action_count_by_group",
                    "group": group,
                    "conceptual_action": concept,
                    "count": count,
                }
            )
    for group, action_payload in dict(
        summary["probability_stats_per_action_by_group"]
    ).items():
        for action, stats in dict(action_payload).items():
            rows.append(
                {
                    "section": "probability_stats_per_action_by_group",
                    "group": group,
                    "action": action,
                    "conceptual_action": conceptual_action_name(action),
                    "metric": "probability",
                    "min": stats["min"],
                    "mean": stats["mean"],
                    "max": stats["max"],
                }
            )
    for group, stats in dict(summary["entropy_min_mean_max_by_group"]).items():
        rows.append(
            {
                "section": "entropy_min_mean_max_by_group",
                "group": group,
                "metric": "entropy",
                "min": stats["min"],
                "mean": stats["mean"],
                "max": stats["max"],
            }
        )
    for group, stats in dict(summary["max_probability_min_mean_max_by_group"]).items():
        rows.append(
            {
                "section": "max_probability_min_mean_max_by_group",
                "group": group,
                "metric": "max_probability",
                "min": stats["min"],
                "mean": stats["mean"],
                "max": stats["max"],
            }
        )
    for section, metric in (
        ("pairwise_l1_min_mean_max", "pairwise_l1"),
        ("distance_to_uniform_min_mean_max", "distance_to_uniform"),
    ):
        stats = dict(summary[section])
        rows.append(
            {
                "section": section,
                "metric": metric,
                "min": stats["min"],
                "mean": stats["mean"],
                "max": stats["max"],
            }
        )
    rows.append(
        {
            "section": "probability_bound_violations",
            "metric": "violations",
            "count": int(summary["probability_bound_violations"]),
        }
    )
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: "" for field in fields}
            payload.update({field: diagnostic_metadata[field] for field in METADATA_FIELDS})
            payload.update(row)
            writer.writerow(payload)


def write_elite_child_outputs(
    *,
    elite_child_examples: Sequence[Mapping[str, object]],
    elite_child_summary_rows: Sequence[Mapping[str, object]],
    valid_actions_by_group: Mapping[str, Sequence[str]],
    diagnostic_metadata: Mapping[str, object],
    output_dir: Path,
) -> None:
    (output_dir / "elite_child_examples.json").write_text(
        json.dumps(
            {
                "metadata": dict(diagnostic_metadata),
                "examples": list(elite_child_examples),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_elite_child_examples_csv(
        elite_child_examples,
        output_dir / "elite_child_examples.csv",
        valid_actions_by_group=valid_actions_by_group,
        diagnostic_metadata=diagnostic_metadata,
    )
    write_elite_child_summary_csv(
        elite_child_summary_rows,
        output_dir / "elite_child_summary.csv",
        diagnostic_metadata=diagnostic_metadata,
    )


def write_elite_child_examples_csv(
    examples: Sequence[Mapping[str, object]],
    path: Path,
    *,
    valid_actions_by_group: Mapping[str, Sequence[str]],
    diagnostic_metadata: Mapping[str, object],
) -> None:
    conceptual_actions = concepts_present(valid_actions_by_group)
    raw_probability_fields = [
        f"{group}_{action}"
        for group in GROUP_ORDER
        for action in valid_actions_by_group[group]
    ]
    conceptual_probability_fields = [
        f"{group}_{concept}"
        for group in GROUP_ORDER
        for concept in conceptual_actions
    ]
    fields = (
        METADATA_FIELDS
        + [
            "parent_name",
            "parent_source_sampler",
            "parent_vertex_name",
            "scale",
            "child_id",
            "child_parent_l1",
            "child_uniform_l1",
        ]
        + [
            item
            for group in GROUP_ORDER
            for item in (
                f"{group}_parent_dom",
                f"{group}_child_dom",
                f"{group}_same_dom",
            )
        ]
        + raw_probability_fields
        + conceptual_probability_fields
    )
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            child_policy = example["child_policy"]
            if not isinstance(child_policy, Mapping):
                raise TypeError("child_policy must be a mapping.")
            row: dict[str, object] = {
                **{field: diagnostic_metadata[field] for field in METADATA_FIELDS},
                "parent_name": example["parent_name"],
                "parent_source_sampler": example["parent_source_sampler"],
                "parent_vertex_name": example.get("parent_vertex_name", ""),
                "scale": example["scale"],
                "child_id": example["child_id"],
                "child_parent_l1": example["child_parent_l1"],
                "child_uniform_l1": example["child_uniform_l1"],
            }
            for group in GROUP_ORDER:
                row[f"{group}_parent_dom"] = example[f"{group}_parent_dom"]
                row[f"{group}_child_dom"] = example[f"{group}_child_dom"]
                row[f"{group}_same_dom"] = example[f"{group}_same_dom"]
                group_probabilities = child_policy[group]
                for action in valid_actions_by_group[group]:
                    row[f"{group}_{action}"] = group_probabilities.get(action, "")
                for concept in conceptual_actions:
                    value = probability_for_concept(group_probabilities, concept)
                    row[f"{group}_{concept}"] = "" if value is None else value
            writer.writerow(row)


def write_elite_child_summary_csv(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    *,
    diagnostic_metadata: Mapping[str, object],
) -> None:
    fields = METADATA_FIELDS + [
        "parent_name",
        "parent_source_sampler",
        "scale",
        "child_count",
        "min_child_parent_l1",
        "mean_child_parent_l1",
        "max_child_parent_l1",
        "min_child_uniform_l1",
        "mean_child_uniform_l1",
        "max_child_uniform_l1",
    ]
    for group in GROUP_ORDER:
        fields.extend(
            [
                f"{group}_same_dom_ratio",
                f"{group}_average_entropy",
                f"{group}_average_max_probability",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {field: "" for field in fields}
            payload.update({field: diagnostic_metadata[field] for field in METADATA_FIELDS})
            payload.update({field: row.get(field, "") for field in fields})
            writer.writerow(payload)


def write_markdown_report(
    *,
    selected_candidates: Sequence[SelectedCandidate],
    pool_summary: Mapping[str, object],
    pairwise_distances: Sequence[Sequence[float]],
    output_dir: Path,
) -> None:
    del pairwise_distances
    metadata = extract_metadata(pool_summary)
    valid_actions_by_group = coerce_valid_actions(
        pool_summary["valid_actions_by_group"]
    )
    action_summary = dict(pool_summary["action_distribution_summary"])
    lines: list[str] = [
        "# PTS Space-Filling Initial Population Diagnostic",
        "",
        "This report is a policy-space diagnostic only. It does not make claims "
        "about attack performance.",
        "",
        "## Configuration",
    ]
    config_rows = [
        ("seed", pool_summary["seed"]),
        ("mandatory_enabled", pool_summary["mandatory_enabled"]),
        ("mandatory_count", pool_summary["mandatory_count"]),
        ("extreme_count", pool_summary["extreme_count"]),
        ("moderate_count", pool_summary["moderate_count"]),
        ("balanced_count", pool_summary["balanced_count"]),
        ("extreme_alpha", pool_summary["extreme_alpha"]),
        ("moderate_alpha", pool_summary["moderate_alpha"]),
        ("min_probability", pool_summary["min_probability"]),
        ("max_probability", pool_summary["max_probability"]),
        ("elite_child_count", pool_summary["elite_child_count"]),
        ("elite_child_seed", pool_summary["elite_child_seed"]),
        (
            "local_concentration_scales",
            ",".join(str(value) for value in pool_summary["local_concentration_scales"]),
        ),
    ]
    config_rows.extend((field, metadata[field]) for field in METADATA_FIELDS)
    config_rows.extend(
        [
            ("c1_generate_missing", pool_summary["c1_generate_missing"]),
            ("c1_generate_available", pool_summary["c1_generate_available"]),
        ]
    )
    lines.extend(markdown_table(["field", "value"], config_rows))
    warnings = list(pool_summary.get("warnings", []))
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(["", "## Valid Actions", ""])
    lines.extend(
        markdown_table(
            ["group", "raw_actions", "conceptual_aliases"],
            [
                (
                    group,
                    ", ".join(valid_actions_by_group[group]),
                    ", ".join(
                        conceptual_action_name(action)
                        for action in valid_actions_by_group[group]
                    ),
                )
                for group in GROUP_ORDER
            ],
        )
    )

    mandatory_rows: list[tuple[object, ...]] = []
    for candidate in selected_candidates:
        if candidate.source_sampler != "mandatory_vertex":
            continue
        for group in GROUP_ORDER:
            for action, probability in candidate.policy[group].items():
                mandatory_rows.append(
                    (
                        candidate.vertex_name,
                        group,
                        action,
                        conceptual_action_name(action),
                        format_float(probability),
                    )
                )
    lines.extend(["", "## Mandatory Vertex Distributions", ""])
    lines.extend(
        markdown_table(
            ["vertex_name", "group", "raw_action", "conceptual_action", "probability"],
            mandatory_rows,
        )
    )

    lines.extend(["", "## Selected Initial Population", ""])
    lines.extend(
        markdown_table(
            [
                "candidate_id",
                "source_sampler",
                "vertex_name",
                "suffix_1 dominant/prob",
                "suffix_2 dominant/prob",
                "suffix_3plus dominant/prob",
                "distance_to_uniform",
                "min_distance_to_previous_selected",
            ],
            [
                (
                    candidate.candidate_id,
                    candidate.source_sampler,
                    "" if candidate.vertex_name is None else candidate.vertex_name,
                    format_dominant_for_report(candidate, "suffix_1"),
                    format_dominant_for_report(candidate, "suffix_2"),
                    format_dominant_for_report(candidate, "suffix_3plus"),
                    format_float(candidate.distance_to_uniform),
                    format_optional_float(
                        candidate.min_distance_to_previous_selected
                    ),
                )
                for candidate in selected_candidates
            ],
        )
    )

    lines.extend(["", "## Action Distribution Summary", ""])
    source_counts = dict(action_summary["count_by_source_sampler"])
    lines.extend(
        markdown_table(
            ["source_sampler", "count"],
            sorted(source_counts.items()),
        )
    )
    dominant_rows: list[tuple[object, ...]] = []
    for group, counts in dict(
        action_summary["dominant_conceptual_action_count_by_group"]
    ).items():
        for concept, count in dict(counts).items():
            dominant_rows.append((group, concept, count))
    lines.extend(["", "Dominant conceptual action counts:", ""])
    lines.extend(
        markdown_table(["group", "conceptual_action", "count"], dominant_rows)
    )
    probability_rows: list[tuple[object, ...]] = []
    for group, action_payload in dict(
        action_summary["probability_stats_per_action_by_group"]
    ).items():
        for action, stats in dict(action_payload).items():
            probability_rows.append(
                (
                    group,
                    action,
                    conceptual_action_name(action),
                    format_optional_float(stats["min"]),
                    format_optional_float(stats["mean"]),
                    format_optional_float(stats["max"]),
                )
            )
    lines.extend(["", "Probability stats by raw action:", ""])
    lines.extend(
        markdown_table(
            ["group", "raw_action", "conceptual_action", "min", "mean", "max"],
            probability_rows,
        )
    )
    lines.extend(["", "## Pairwise Distance Summary", ""])
    lines.extend(
        markdown_table(
            ["metric", "min", "mean", "max"],
            [
                (
                    "pairwise_l1",
                    format_optional_float(
                        action_summary["pairwise_l1_min_mean_max"]["min"]
                    ),
                    format_optional_float(
                        action_summary["pairwise_l1_min_mean_max"]["mean"]
                    ),
                    format_optional_float(
                        action_summary["pairwise_l1_min_mean_max"]["max"]
                    ),
                ),
                (
                    "distance_to_uniform",
                    format_optional_float(
                        action_summary["distance_to_uniform_min_mean_max"]["min"]
                    ),
                    format_optional_float(
                        action_summary["distance_to_uniform_min_mean_max"]["mean"]
                    ),
                    format_optional_float(
                        action_summary["distance_to_uniform_min_mean_max"]["max"]
                    ),
                ),
                (
                    "probability_bound_violations",
                    action_summary["probability_bound_violations"],
                    "",
                    "",
                ),
            ],
        )
    )

    elite_summary_rows = list(pool_summary.get("elite_child_summary_rows", []))
    lines.extend(["", "## Elite Child Summary", ""])
    if elite_summary_rows:
        lines.extend(
            markdown_table(
                [
                    "parent_name",
                    "scale",
                    "mean_child_parent_l1",
                    "mean_child_uniform_l1",
                    "suffix_1_same_dom",
                    "suffix_2_same_dom",
                    "suffix_3plus_same_dom",
                    "suffix_1_avg_max_prob",
                    "suffix_2_avg_max_prob",
                    "suffix_3plus_avg_max_prob",
                ],
                [
                    (
                        row["parent_name"],
                        format_float(row["scale"]),
                        format_float(row["mean_child_parent_l1"]),
                        format_float(row["mean_child_uniform_l1"]),
                        format_float(row["suffix_1_same_dom_ratio"]),
                        format_float(row["suffix_2_same_dom_ratio"]),
                        format_float(row["suffix_3plus_same_dom_ratio"]),
                        format_float(row["suffix_1_average_max_probability"]),
                        format_float(row["suffix_2_average_max_probability"]),
                        format_float(row["suffix_3plus_average_max_probability"]),
                    )
                    for row in elite_summary_rows
                ],
            )
        )
    else:
        lines.append("Elite child inspection was disabled or no rows were generated.")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Low local concentration scales may let children drift far from the parent policy.",
            "- High local concentration scales make children closer to parent copies.",
            "- Scale around 30 should be evaluated using child-parent L1 and same-dominant ratios.",
            "- Child policies are projected to the bounded simplex, so high concentration scales may be affected by max_probability clipping.",
            "- This is only a policy-space diagnostic and does not evaluate attack performance.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> list[str]:
    normalized_headers = [escape_markdown_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(normalized_headers) + " |",
        "| " + " | ".join("---" for _ in normalized_headers) + " |",
    ]
    for row in rows:
        values = [escape_markdown_cell(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def escape_markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def format_float(value: object) -> str:
    return f"{float(value):.6f}"


def format_optional_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return format_float(value)


def format_dominant_for_report(candidate: SelectedCandidate, group: str) -> str:
    concept = candidate.dominant_conceptual_action_by_group[group]
    probability = candidate.dominant_probability_by_group[group]
    return f"{concept}/{probability:.6f}"


def print_summary(
    config: SpaceFillingConfig,
    pool_summary: Mapping[str, object],
) -> None:
    print("PTS space-filling initial population diagnostic")
    for warning in pool_summary.get("warnings", []):
        print(warning)
    print(
        "parameters: "
        f"seed={config.seed}, mandatory_enabled={config.mandatory_enabled}, "
        f"mandatory_count={pool_summary['mandatory_count']}, "
        f"extreme={config.extreme_count}/{config.extreme_pool_size} "
        f"alpha={config.extreme_alpha}, "
        f"moderate={config.moderate_count}/{config.moderate_pool_size} "
        f"alpha={config.moderate_alpha}, balanced={config.balanced_count}, "
        f"bounds=[{config.min_probability}, {config.max_probability}]"
    )
    print(
        "c1_generate_available_in_production="
        f"{pool_summary['c1_generate_available_in_production']}, "
        "diagnostic_c1_generate_injected="
        f"{pool_summary['diagnostic_c1_generate_injected']}"
    )
    print(
        "selected_candidate_count="
        f"{pool_summary['selected_candidate_count']} "
        "expected_candidate_count_for_planned_5_action_space="
        f"{pool_summary['expected_candidate_count_for_planned_5_action_space']}"
    )
    print(
        "selected_count_by_source_sampler: "
        f"{pool_summary['selected_count_by_source_sampler']}"
    )
    print(
        "pairwise_selected_distance: "
        f"{pool_summary['pairwise_selected_distance']}"
    )
    print(
        "dominant_conceptual_action_counts_by_group: "
        f"{pool_summary['dominant_conceptual_action_counts_by_group']}"
    )
    print(
        "probability_bound_violations_count: "
        f"{pool_summary['probability_bound_violations_count']}"
    )
    print()


def print_selected_candidates(selected_candidates: Sequence[SelectedCandidate]) -> None:
    header = (
        "candidate_id  source_sampler     vertex_name                    "
        "suffix_1_dom     suffix_2_dom     suffix_3plus_dom  "
        "distance_to_uniform  min_distance_to_previous_selected"
    )
    print(header)
    print("-" * len(header))
    for candidate in selected_candidates:
        min_previous = (
            "-"
            if candidate.min_distance_to_previous_selected is None
            else f"{candidate.min_distance_to_previous_selected:.6f}"
        )
        vertex_name = "-" if candidate.vertex_name is None else candidate.vertex_name
        print(
            f"{candidate.candidate_id:>12}  "
            f"{candidate.source_sampler:<17} "
            f"{vertex_name:<30} "
            f"{_format_dominant(candidate, 'suffix_1'):<16} "
            f"{_format_dominant(candidate, 'suffix_2'):<16} "
            f"{_format_dominant(candidate, 'suffix_3plus'):<17} "
            f"{candidate.distance_to_uniform:>19.6f}  "
            f"{min_previous:>33}"
        )
    print("\nDetailed selected policy distributions:")
    for candidate in selected_candidates:
        print(
            f"\ncandidate {candidate.candidate_id} "
            f"(source={candidate.source_sampler}, "
            f"vertex={candidate.vertex_name}, pool_index={candidate.pool_index}):"
        )
        for group in GROUP_ORDER:
            print(f"  {group}:")
            for action, probability in candidate.policy[group].items():
                concept = conceptual_action_name(action)
                print(f"    {action} ({concept}): {probability:.6f}")


def _format_dominant(candidate: SelectedCandidate, group: str) -> str:
    concept = candidate.dominant_conceptual_action_by_group[group]
    probability = candidate.dominant_probability_by_group[group]
    return f"{concept}={probability:.3f}"


def print_elite_child_console_summary(rows: object) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return
    if not rows:
        return
    print("\nElite-centered child diagnostic summary:")
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        print(f"parent={row['parent_name']} scale={float(row['scale']):g}:")
        print(
            "  child_parent_l1 "
            f"mean={float(row['mean_child_parent_l1']):.6f} "
            f"min={float(row['min_child_parent_l1']):.6f} "
            f"max={float(row['max_child_parent_l1']):.6f}"
        )
        print(
            "  same_dom "
            f"suffix_1={float(row['suffix_1_same_dom_ratio']):.3f}, "
            f"suffix_2={float(row['suffix_2_same_dom_ratio']):.3f}, "
            f"suffix_3plus={float(row['suffix_3plus_same_dom_ratio']):.3f}"
        )
        print(
            "  avg max prob "
            f"suffix_1={float(row['suffix_1_average_max_probability']):.3f}, "
            f"suffix_2={float(row['suffix_2_average_max_probability']):.3f}, "
            f"suffix_3plus={float(row['suffix_3plus_average_max_probability']):.3f}"
        )


if __name__ == "__main__":
    main()
