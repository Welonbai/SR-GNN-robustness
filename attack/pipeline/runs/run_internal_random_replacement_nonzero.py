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
    INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    attack_key,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.internal_random_replacement_nonzero_when_possible import (
    InternalRandomReplacementNonzeroWhenPossiblePolicy,
    InternalRandomReplacementResult,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.position_stats import save_position_stats


DEFAULT_INTERNAL_RANDOM_REPLACEMENT_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_internal_random_replacement_nonzero_when_possible_"
    "ratio1_srgnn_partial4.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_internal_random_replacement_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Internal-Random-Replacement-NZ expects data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = InternalRandomReplacementNonzeroWhenPossiblePolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            rng=random.Random(config.seeds.fake_session_seed),
        )
        results = [
            policy.apply_with_metadata(session, int(target_item))
            for session in shared.template_sessions
        ]
        fake_sessions = [result.session for result in results]
        replacement_positions = [
            int(result.replacement_position) for result in results
        ]

        _validate_internal_replaced_sessions(
            template_sessions=shared.template_sessions,
            replaced_sessions=fake_sessions,
            results=results,
            target_item=int(target_item),
        )

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=replacement_positions,
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Internal-Random-Replacement-NZ samples zero-based replacement "
                "positions from internal nonzero positions [1, len(session)-2]; "
                "length-2 sessions fall back to tail position 1."
            ),
        )

        metadata = build_internal_random_replacement_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            replaced_sessions=fake_sessions,
            replacement_results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
        )
        metadata_path = target_root / "internal_random_replacement_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "position_stats_path": str(position_stats_path),
                "internal_random_replacement_metadata_path": str(metadata_path),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_internal_random_replacement_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    replacement_results: Sequence[InternalRandomReplacementResult],
    clean_train_sessions: Sequence[Sequence[int]],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(replaced_sessions):
        raise ValueError("template_sessions and replaced_sessions must have the same length.")
    if len(template_sessions) != len(replacement_results):
        raise ValueError("replacement_results must align 1:1 with template_sessions.")

    replacement_positions = [
        int(result.replacement_position) for result in replacement_results
    ]
    position_counts: Counter[int] = Counter(replacement_positions)
    position_group_counts = _position_group_counts(
        replacement_results,
    )
    total = int(len(replacement_results))
    target_counts = [
        int(result.target_occurrence_count_after_replacement)
        for result in replacement_results
    ]
    length_deltas = [
        int(result.replaced_length) - int(result.original_length)
        for result in replacement_results
    ]
    internal_replacement_count = int(
        sum(1 for result in replacement_results if result.used_internal_position)
    )
    tail_fallback_count = int(
        sum(1 for result in replacement_results if result.used_tail_fallback)
    )
    left_items = [
        int(result.left_item)
        for result in replacement_results
        if result.left_item is not None
    ]
    right_items = [
        int(result.right_item)
        for result in replacement_results
        if result.right_item is not None
    ]
    left_right_pairs = [
        (int(result.left_item), int(result.right_item))
        for result in replacement_results
        if result.left_item is not None and result.right_item is not None
    ]
    shared_key = shared_attack_artifact_key(
        config,
        run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    return {
        "run_type": INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "replacement",
        "replacement_policy": "internal_random_nonzero_when_possible",
        "target_item": int(target_item),
        "fake_session_count": int(len(replaced_sessions)),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "internal_replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "topk_sampling_note": (
            "top-k ratio is applied to the ordered valid replacement positions; "
            "replacement_position is sampled uniformly from "
            "restricted_candidate_positions."
        ),
        "template_fake_sessions_path": str(template_fake_sessions_path),
        "poison_model_checkpoint_path": (
            None
            if poison_model_checkpoint_path is None
            else str(poison_model_checkpoint_path)
        ),
        "poison_model_key": poison_model_key(config),
        "poison_model_identity": poison_model_key_payload(config),
        "shared_fake_sessions_key": shared_key,
        "random_nz_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_NZ_RUN_TYPE,
        ),
        "internal_random_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "tail_replacement_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "same_shared_fake_sessions_as_internal_random_insertion_expected": True,
        "same_shared_fake_sessions_as_tail_replacement_expected": True,
        "attack_key": attack_key(
            config,
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "random_nz_attack_key": attack_key(config, run_type=RANDOM_NZ_RUN_TYPE),
        "internal_random_insertion_attack_key": attack_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "tail_replacement_attack_key": attack_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "original_length_distribution": _length_distribution(template_sessions),
        "replaced_length_distribution": _length_distribution(replaced_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "length_shift_summary": _numeric_summary(length_deltas),
        "replacement_position_counts": _stringify_counts(position_counts),
        "replacement_position_ratios": _stringify_ratios(position_counts, total=total),
        "replacement_position_group_counts": _stringify_named_counts(
            position_group_counts
        ),
        "replacement_position_group_ratios": _stringify_named_ratios(
            position_group_counts,
            total=total,
        ),
        "tail_position_count": int(position_group_counts["tail_position"]),
        "tail_position_ratio": (
            float(position_group_counts["tail_position"]) / float(total)
            if total
            else 0.0
        ),
        "tail_position_is_fallback_group": True,
        "pos0_excluded": True,
        "tail_position_excluded_when_internal_available": True,
        "tail_fallback_allowed_for_len2": True,
        "internal_replacement_count": internal_replacement_count,
        "tail_fallback_count": tail_fallback_count,
        "tail_fallback_ratio": (
            float(tail_fallback_count) / float(total) if total else 0.0
        ),
        "internal_replacement_ratio": (
            float(internal_replacement_count) / float(total) if total else 0.0
        ),
        "every_internal_replaced_target_has_left_neighbor": bool(
            all(
                result.left_item is not None
                for result in replacement_results
                if result.used_internal_position
            )
        ),
        "every_internal_replaced_target_has_right_neighbor": bool(
            all(
                result.right_item is not None
                for result in replacement_results
                if result.used_internal_position
            )
        ),
        "every_replaced_target_has_left_neighbor": bool(
            all(result.left_item is not None for result in replacement_results)
        ),
        "every_replaced_target_has_right_neighbor": bool(tail_fallback_count == 0),
        "left_item_count_summary": _counter_summary(left_items),
        "right_item_count_summary": _counter_summary(right_items),
        "unique_left_item_count": int(len(set(left_items))),
        "unique_right_item_count": int(len(set(right_items))),
        "unique_left_right_pair_count": int(len(set(left_right_pairs))),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in replacement_results if result.pre_existing_target_count > 0)
        ),
        "noop_replacement_count": int(
            sum(1 for result in replacement_results if result.was_noop)
        ),
        "injected_sessions_containing_target_count": int(
            sum(1 for count in target_counts if count > 0)
        ),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in target_counts)
        ),
        "target_occurrence_count_after_replacement": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "previews": _internal_replacement_previews(
            template_sessions=template_sessions,
            replaced_sessions=replaced_sessions,
            replacement_results=replacement_results,
            limit=preview_limit,
        ),
    }


def _validate_internal_replaced_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomReplacementResult],
    target_item: int,
) -> None:
    if len(template_sessions) != len(replaced_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(results):
        raise RuntimeError("Result metadata count does not equal template count.")
    for original, replaced, result in zip(template_sessions, replaced_sessions, results):
        original_list = [int(item) for item in original]
        replaced_list = [int(item) for item in replaced]
        position = int(result.replacement_position)
        target = int(target_item)
        if len(original_list) < 2:
            raise RuntimeError(
                "Internal-Random-Replacement-NZ requires template session length >= 2."
            )
        if len(replaced_list) != len(original_list):
            raise RuntimeError(
                "Internal-Random-Replacement-NZ injected session length changed."
            )
        if int(result.original_length) != len(original_list):
            raise RuntimeError("Original length metadata is invalid.")
        if int(result.replaced_length) != len(replaced_list):
            raise RuntimeError("Replaced length metadata is invalid.")
        if target not in set(replaced_list):
            raise RuntimeError(
                "Internal-Random-Replacement-NZ injected session is missing target item."
            )
        if position == 0:
            raise RuntimeError("Internal-Random-Replacement-NZ selected position 0.")
        if position < 0 or position >= len(original_list):
            raise RuntimeError("Replacement position is outside the template session.")
        if position not in {int(item) for item in result.restricted_candidate_positions}:
            raise RuntimeError(
                "Replacement position was not sampled from restricted candidate positions."
            )

        expected_candidates = (
            list(range(1, len(original_list) - 1))
            if len(original_list) >= 3
            else [1]
        )
        if [int(item) for item in result.candidate_positions] != expected_candidates:
            raise RuntimeError("Candidate position metadata is invalid.")
        restricted = [int(item) for item in result.restricted_candidate_positions]
        if not restricted:
            raise RuntimeError("Restricted candidate positions must not be empty.")
        if not set(restricted).issubset(set(expected_candidates)):
            raise RuntimeError("Restricted candidate positions are not valid candidates.")
        if restricted != expected_candidates[: len(restricted)]:
            raise RuntimeError(
                "Restricted candidate positions must be the prefix of candidate positions."
            )

        if len(original_list) >= 3:
            if position < 1 or position > len(original_list) - 2:
                raise RuntimeError(
                    "Internal-Random-Replacement-NZ selected a non-internal position."
                )
            if not result.used_internal_position or result.used_tail_fallback:
                raise RuntimeError("Internal replacement flags are invalid.")
            if result.right_item is None:
                raise RuntimeError("Internal replacement right neighbor is missing.")
            if int(result.right_item) != original_list[position + 1]:
                raise RuntimeError("Internal replacement right neighbor metadata is invalid.")
            if int(result.internal_candidate_count) != len(expected_candidates):
                raise RuntimeError("Internal candidate count metadata is invalid.")
        else:
            if position != 1:
                raise RuntimeError("Length-2 fallback must select position 1.")
            if result.used_internal_position or not result.used_tail_fallback:
                raise RuntimeError("Length-2 tail fallback flags are invalid.")
            if result.right_item is not None:
                raise RuntimeError("Length-2 tail fallback right neighbor must be null.")
            if int(result.internal_candidate_count) != 0:
                raise RuntimeError("Length-2 fallback has no internal candidates.")

        if result.left_item is None:
            raise RuntimeError("Replacement left neighbor metadata is missing.")
        if int(result.left_item) != original_list[position - 1]:
            raise RuntimeError("Replacement left neighbor metadata is invalid.")
        if int(result.original_item) != original_list[position]:
            raise RuntimeError("Original item metadata is invalid.")
        if bool(result.was_noop) != bool(original_list[position] == target):
            raise RuntimeError("Noop replacement metadata is invalid.")

        changed_positions = [
            index
            for index, (before, after) in enumerate(zip(original_list, replaced_list))
            if before != after
        ]
        if result.was_noop:
            if changed_positions:
                raise RuntimeError("Noop replacement changed the session.")
        else:
            if len(changed_positions) > 1:
                raise RuntimeError("Replacement changed more than one position.")
            if changed_positions != [position]:
                raise RuntimeError("Replacement did not change the selected position.")

        expected = list(original_list)
        expected[position] = target
        if replaced_list != expected:
            raise RuntimeError(
                "Internal-Random-Replacement-NZ did not preserve non-replaced items."
            )


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


def _counter_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {
            "unique": 0.0,
            "max_count": 0.0,
            "mean_count_per_unique": 0.0,
        }
    counts: Counter[int] = Counter(int(value) for value in values)
    return {
        "unique": float(len(counts)),
        "max_count": float(max(counts.values())),
        "mean_count_per_unique": float(sum(counts.values()) / len(counts)),
    }


def _position_group_counts(
    replacement_results: Sequence[InternalRandomReplacementResult],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in replacement_results:
        counts[_position_group(result)] += 1
    for group in _POSITION_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _position_group(result: InternalRandomReplacementResult) -> str:
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


def _internal_replacement_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    replacement_results: Sequence[InternalRandomReplacementResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, replaced, result in zip(
        template_sessions[:limit],
        replaced_sessions[:limit],
        replacement_results[:limit],
    ):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "replaced_session": [int(item) for item in replaced],
                "replacement_position": int(result.replacement_position),
                "original_item": int(result.original_item),
                "left_item": (
                    None if result.left_item is None else int(result.left_item)
                ),
                "right_item": (
                    None if result.right_item is None else int(result.right_item)
                ),
                "original_length": int(result.original_length),
                "replaced_length": int(result.replaced_length),
                "was_noop": bool(result.was_noop),
                "used_internal_position": bool(result.used_internal_position),
                "used_tail_fallback": bool(result.used_tail_fallback),
                "candidate_positions": [
                    int(position) for position in result.candidate_positions
                ],
                "restricted_candidate_positions": [
                    int(position)
                    for position in result.restricted_candidate_positions
                ],
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_INTERNAL_RANDOM_REPLACEMENT_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_internal_random_replacement_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
