from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.generated_continuation_suffix import PURE_GENERATED_MODE_RNG_TAG
from attack.insertion.internal_random_replacement_generated_continuation import (
    InternalRandomReplacementGeneratedContinuationPolicy,
    InternalRandomReplacementGeneratedContinuationResult,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.position_stats import build_position_stats_payload


DEFAULT_INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_internal_random_replacement_generated_continuation_"
    "nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)


def run_internal_random_replacement_generated_continuation_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Internal-Random-Replacement-Generated-Continuation-NZ expects "
            "data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    if shared.poison_runner is None:
        raise RuntimeError(
            "Poison runner is required for generated-continuation suffix generation."
        )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = InternalRandomReplacementGeneratedContinuationPolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            poison_runner=shared.poison_runner,
            generation_topk=config.attack.fake_session_generation_topk,
            replacement_rng=random.Random(config.seeds.fake_session_seed),
            generation_rng_base_seed=config.seeds.fake_session_seed,
        )
        results = [
            policy.apply_with_metadata(session, int(target_item), index)
            for index, session in enumerate(shared.template_sessions)
        ]
        fake_sessions = [result.session for result in results]
        replacement_positions = [
            int(result.replacement_position) for result in results
        ]

        max_item = max(shared.stats.item_counts)
        _validate_internal_replacement_generated_continuation_sessions(
            template_sessions=shared.template_sessions,
            final_sessions=fake_sessions,
            results=results,
            target_item=int(target_item),
            max_item_id=max_item,
        )

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        position_stats_payload = build_position_stats_payload(
            sessions=shared.template_sessions,
            positions=replacement_positions,
            run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Internal-Random-Replacement-Generated-Continuation-NZ samples "
                "replacement positions with Internal-Random-Replacement-NZ, "
                "discards the original suffix after target, and regenerates a "
                "suffix of the same length."
            ),
        )
        position_stats_path = (
            target_root / "internal_replacement_generated_continuation_slot_stats.json"
        )
        save_json(position_stats_payload, position_stats_path)

        metadata = build_internal_random_replacement_generated_continuation_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            position_stats_payload=position_stats_payload,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
            generation_topk=config.attack.fake_session_generation_topk,
            generation_rng_base_seed=config.seeds.fake_session_seed,
        )
        metadata_path = (
            target_root
            / "internal_random_replacement_generated_continuation_metadata.json"
        )
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=fake_sessions,
            metadata={
                "internal_replacement_generated_continuation_slot_stats_path": str(
                    position_stats_path
                ),
                "internal_random_replacement_generated_continuation_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_internal_random_replacement_generated_continuation_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomReplacementGeneratedContinuationResult],
    clean_train_sessions: Sequence[Sequence[int]],
    position_stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    generation_topk: int,
    generation_rng_base_seed: int,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(results):
        raise ValueError("results must align 1:1 with template_sessions.")

    target = int(target_item)
    total = int(len(results))
    final_sessions = [result.session for result in results]
    replaced_sessions = [result.replaced_session_before_generation for result in results]
    original_suffixes = [result.original_suffix for result in results]
    generated_suffixes = [result.generated_suffix for result in results]
    replacement_position_counts = Counter(
        int(result.replacement_position) for result in results
    )
    target_position_counts = Counter(
        int(result.generated_result.final_target_position) for result in results
    )
    position_group_counts = _position_group_counts(results)
    suffix_lengths = [int(result.suffix_length) for result in results]
    suffix_length_zero_count = int(sum(1 for length in suffix_lengths if length == 0))
    generated_applied_count = int(sum(1 for length in suffix_lengths if length > 0))
    generated_contains_target_counts = [
        int(result.generated_result.generated_suffix_contains_target_count)
        for result in results
    ]
    generated_suffix_contains_target_count = int(
        sum(1 for count in generated_contains_target_counts if count > 0)
    )
    generated_suffix_equals_original_count = int(
        sum(
            1
            for result in results
            if [int(item) for item in result.generated_suffix]
            == [int(item) for item in result.original_suffix]
        )
    )
    final_target_counts = [
        int(result.target_occurrence_count_final) for result in results
    ]
    after_replacement_counts = [
        int(result.target_occurrence_count_after_replacement) for result in results
    ]
    generated_first_items = [
        int(result.generated_suffix[0]) for result in results if result.generated_suffix
    ]
    generated_item_counts: Counter[int] = Counter(
        int(item) for suffix in generated_suffixes for item in suffix
    )
    generated_item_total = int(sum(generated_item_counts.values()))
    target_tail_count = int(
        sum(1 for session in final_sessions if session and int(session[-1]) == target)
    )
    adjacent_target_count = int(
        sum(1 for session in final_sessions if _has_adjacent_pair(session, target, target))
    )

    return {
        "run_type": INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "internal_random_replacement_generated_continuation",
        "exposure_operation": "internal_random_replacement_nonzero_when_possible",
        "replacement_policy": "internal_random_nonzero_when_possible",
        "suffix_strategy": "target_conditioned_generated_continuation",
        "target_item": target,
        "fake_session_count": total,
        "template_fake_sessions_path": str(template_fake_sessions_path),
        "poison_model_checkpoint_path": (
            None
            if poison_model_checkpoint_path is None
            else str(poison_model_checkpoint_path)
        ),
        "poison_model_key": poison_model_key(config),
        "poison_model_identity": poison_model_key_payload(config),
        "shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "generation_topk": int(generation_topk),
        "generated_suffix_source": "poison_model_score_session_autoregressive",
        "suffix_length_strategy": "preserve_replacement_suffix_length",
        "original_suffix_preserved": False,
        "all_sessions_regenerate_suffix_when_suffix_exists": True,
        "allow_target_in_generated_suffix": True,
        "allow_repeated_items_in_generated_suffix": True,
        "generation_rng_strategy": "per_target_session_deterministic",
        "generation_rng_base_seed": int(generation_rng_base_seed),
        "pure_generated_mode_rng_tag": PURE_GENERATED_MODE_RNG_TAG,
        "original_length_distribution": _length_distribution(template_sessions),
        "replaced_length_distribution": _length_distribution(replaced_sessions),
        "final_length_distribution": _length_distribution(final_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "length_shift_from_template_summary": _length_shift_summary_from_pairs(
            template_sessions,
            final_sessions,
        ),
        "length_shift_from_replaced_summary": _length_shift_summary_from_pairs(
            replaced_sessions,
            final_sessions,
        ),
        "replacement_position_counts": _stringify_counts(replacement_position_counts),
        "replacement_position_ratios": _stringify_ratios(
            replacement_position_counts,
            total=total,
        ),
        "replacement_position_group_counts": _stringify_named_counts(
            position_group_counts
        ),
        "replacement_position_group_ratios": _stringify_named_ratios(
            position_group_counts,
            total=total,
        ),
        "target_position_counts": _stringify_counts(target_position_counts),
        "target_position_ratios": _stringify_ratios(
            target_position_counts,
            total=total,
        ),
        "position_stats": position_stats_payload,
        "target_tail_count": target_tail_count,
        "target_tail_ratio": float(target_tail_count) / float(total) if total else 0.0,
        "suffix_length_zero_count": suffix_length_zero_count,
        "suffix_length_zero_ratio": (
            float(suffix_length_zero_count) / float(total) if total else 0.0
        ),
        "generated_suffix_applied_count": generated_applied_count,
        "generated_suffix_applied_ratio": (
            float(generated_applied_count) / float(total) if total else 0.0
        ),
        "all_final_sessions_contain_target": bool(
            all(target in {int(item) for item in session} for session in final_sessions)
        ),
        "original_suffix_length_distribution": _length_distribution(original_suffixes),
        "generated_suffix_length_distribution": _length_distribution(generated_suffixes),
        "generated_suffix_length_summary": _numeric_summary(suffix_lengths),
        "generated_suffix_replaced_count": generated_applied_count,
        "generated_suffix_replaced_ratio": (
            float(generated_applied_count) / float(total) if total else 0.0
        ),
        "generated_suffix_equals_original_suffix_count": generated_suffix_equals_original_count,
        "generated_suffix_equals_original_suffix_ratio": (
            float(generated_suffix_equals_original_count) / float(total)
            if total
            else 0.0
        ),
        "generated_suffix_contains_target_count": generated_suffix_contains_target_count,
        "generated_suffix_contains_target_ratio": (
            float(generated_suffix_contains_target_count) / float(total)
            if total
            else 0.0
        ),
        "generated_suffix_target_occurrence_count": _numeric_summary(
            generated_contains_target_counts
        ),
        "final_sessions_with_adjacent_target_target_count": adjacent_target_count,
        "final_sessions_with_adjacent_target_target_ratio": (
            float(adjacent_target_count) / float(total) if total else 0.0
        ),
        "generated_suffix_unique_item_count_summary": _numeric_summary(
            [
                int(result.generated_result.generated_suffix_unique_item_count)
                for result in results
            ]
        ),
        "generated_suffix_first_item_counts": _stringify_counts(
            Counter(generated_first_items)
        ),
        "generated_suffix_first_item_ratios": _stringify_ratios(
            Counter(generated_first_items),
            total=len(generated_first_items),
        ),
        "generated_suffix_item_counts_top20": _top_counts(generated_item_counts, 20),
        "generated_suffix_item_ratios_top20": _top_ratios(
            generated_item_counts,
            20,
            total=generated_item_total,
        ),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in results if result.pre_existing_target_count > 0)
        ),
        "injected_sessions_containing_target_count": int(
            sum(1 for count in final_target_counts if count > 0)
        ),
        "target_occurrence_count_after_replacement": _numeric_summary(
            after_replacement_counts
        ),
        "target_occurrence_count_final": _numeric_summary(final_target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in final_target_counts if count > 1)
        ),
        "method_is_diagnostic": True,
        "tests_generated_continuation_as_suffix_strategy": True,
        "length_preserving_relative_to_replaced_session": True,
        "removes_original_suffix": True,
        "previews": _replacement_generated_continuation_previews(
            template_sessions=template_sessions,
            results=results,
            limit=preview_limit,
        ),
    }


def _validate_internal_replacement_generated_continuation_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomReplacementGeneratedContinuationResult],
    target_item: int,
    max_item_id: int | None = None,
) -> None:
    if len(template_sessions) != len(final_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(results):
        raise RuntimeError("Result metadata count does not equal template count.")
    target = int(target_item)
    for original, final, result in zip(template_sessions, final_sessions, results):
        original_list = [int(item) for item in original]
        final_list = [int(item) for item in final]
        replaced_list = [int(item) for item in result.replaced_session_before_generation]
        generated_suffix = [int(item) for item in result.generated_suffix]
        position = int(result.replacement_position)
        if len(original_list) < 2:
            raise RuntimeError(
                "Internal-Replacement-Generated-Continuation requires template length >= 2."
            )
        if len(replaced_list) != len(original_list):
            raise RuntimeError("Replacement changed template length.")
        if int(result.original_length) != len(original_list):
            raise RuntimeError("Original length metadata is invalid.")
        if int(result.replaced_length) != len(replaced_list):
            raise RuntimeError("Replaced length metadata is invalid.")
        if int(result.final_length) != len(final_list):
            raise RuntimeError("Final length metadata is invalid.")
        if len(final_list) != len(replaced_list):
            raise RuntimeError("Generated continuation changed replaced length.")
        if len(final_list) != len(original_list):
            raise RuntimeError("Replacement generated continuation changed original length.")
        if position != int(result.generated_result.final_target_position):
            raise RuntimeError("Replacement position and target position differ.")
        if position < 1 or position >= len(original_list):
            raise RuntimeError("Replacement position is outside valid nonzero bounds.")
        expected_replaced = list(original_list)
        expected_replaced[position] = target
        if replaced_list != expected_replaced:
            raise RuntimeError("Replaced session before generation is invalid.")
        if [int(item) for item in result.original_suffix] != replaced_list[position + 1 :]:
            raise RuntimeError("Original suffix metadata is invalid.")
        if len(generated_suffix) != len(result.original_suffix):
            raise RuntimeError("Generated suffix length does not match original suffix.")
        expected_final = replaced_list[: position + 1] + generated_suffix
        if final_list != expected_final:
            raise RuntimeError("Final session does not replace suffix after target.")
        if final_list[position] != target:
            raise RuntimeError("Generated continuation moved the replacement target.")
        if target not in set(final_list):
            raise RuntimeError("Final session is missing target item.")
        if result.suffix_length == 0:
            if generated_suffix:
                raise RuntimeError("Zero-length suffix generated unexpected items.")
            if final_list != replaced_list[: position + 1]:
                raise RuntimeError("Zero-length suffix final session is invalid.")
        if position not in {
            int(item) for item in result.replacement_result.restricted_candidate_positions
        }:
            raise RuntimeError(
                "Replacement position was not sampled from restricted candidate positions."
            )
        for item in generated_suffix:
            if int(item) < 1:
                raise RuntimeError("Generated suffix contains non-positive item id.")
            if max_item_id is not None and int(item) > int(max_item_id):
                raise RuntimeError("Generated suffix contains item id above max item id.")
        if max_item_id is not None and any(int(item) > int(max_item_id) for item in final_list):
            raise RuntimeError("Final fake session contains item id above max item id.")


def _existing_path_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    return str(path_obj) if path_obj.exists() else None


def _length_distribution(sessions: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: Counter[int] = Counter(int(len(session)) for session in sessions)
    return {f"len{length}": int(count) for length, count in sorted(counts.items())}


def _numeric_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [int(value) for value in values]
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
    }


def _length_shift_summary_from_pairs(
    before_sessions: Sequence[Sequence[int]],
    after_sessions: Sequence[Sequence[int]],
) -> dict[str, float]:
    if len(before_sessions) != len(after_sessions):
        raise ValueError("before_sessions and after_sessions must have the same length.")
    return _numeric_summary(
        [
            int(len(after)) - int(len(before))
            for before, after in zip(before_sessions, after_sessions)
        ]
    )


def _position_group_counts(
    results: Sequence[InternalRandomReplacementGeneratedContinuationResult],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        counts[_position_group(result)] += 1
    for group in _POSITION_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _position_group(result: InternalRandomReplacementGeneratedContinuationResult) -> str:
    position = int(result.replacement_position)
    original_length = int(result.original_length)
    if position == original_length - 1:
        return "tail_position"
    if position == 1:
        return "pos1"
    if position == 2:
        return "pos2"
    if position == 3:
        return "pos3"
    if 4 <= position <= 5:
        return "pos4_5"
    return "pos6_plus"


_POSITION_GROUP_ORDER = (
    "pos1",
    "pos2",
    "pos3",
    "pos4_5",
    "pos6_plus",
    "tail_position",
)


def _stringify_counts(counter: Counter[int]) -> dict[str, int]:
    return {str(position): int(count) for position, count in sorted(counter.items())}


def _stringify_ratios(counter: Counter[int], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {str(position): 0.0 for position, _ in sorted(counter.items())}
    return {
        str(position): float(count) / float(total)
        for position, count in sorted(counter.items())
    }


def _stringify_named_counts(counter: Counter[str]) -> dict[str, int]:
    return {group: int(counter.get(group, 0)) for group in _POSITION_GROUP_ORDER}


def _stringify_named_ratios(counter: Counter[str], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {group: 0.0 for group in _POSITION_GROUP_ORDER}
    return {
        group: float(counter.get(group, 0)) / float(total)
        for group in _POSITION_GROUP_ORDER
    }


def _top_counts(counter: Counter[int], limit: int) -> dict[str, int]:
    return {
        str(item): int(count)
        for item, count in counter.most_common(int(limit))
    }


def _top_ratios(counter: Counter[int], limit: int, *, total: int) -> dict[str, float]:
    if total <= 0:
        return {}
    return {
        str(item): float(count) / float(total)
        for item, count in counter.most_common(int(limit))
    }


def _has_adjacent_pair(session: Sequence[int], left: int, right: int) -> bool:
    normalized = [int(item) for item in session]
    return any(
        current == int(left) and following == int(right)
        for current, following in zip(normalized, normalized[1:])
    )


def _replacement_generated_continuation_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomReplacementGeneratedContinuationResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, result in zip(template_sessions[:limit], results[:limit]):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "replaced_session_before_generation": [
                    int(item) for item in result.replaced_session_before_generation
                ],
                "final_session": [int(item) for item in result.session],
                "replacement_position": int(result.replacement_position),
                "target_position": int(result.generated_result.final_target_position),
                "left_item": (
                    None if result.left_item is None else int(result.left_item)
                ),
                "original_right_item": (
                    None
                    if result.original_right_item is None
                    else int(result.original_right_item)
                ),
                "original_suffix": [int(item) for item in result.original_suffix],
                "generated_suffix": [int(item) for item in result.generated_suffix],
                "original_length": int(result.original_length),
                "replaced_length": int(result.replaced_length),
                "final_length": int(result.final_length),
                "suffix_length": int(result.suffix_length),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_internal_random_replacement_generated_continuation_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
