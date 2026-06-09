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
    INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.internal_random_insertion_suffix_variants import (
    InternalRandomInsertionTruncateSuffixPolicy,
    InternalRandomInsertionTruncateSuffixResult,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload


DEFAULT_INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_internal_random_insertion_truncate_suffix_"
    "nonzero_when_possible_ratio1_srgnn_targets5418_4092_9496_partial4.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_internal_random_insertion_truncate_suffix_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Internal-Random-Insertion-Truncate-Suffix-NZ expects "
            "data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = InternalRandomInsertionTruncateSuffixPolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            rng=random.Random(config.seeds.fake_session_seed),
        )
        results = [
            policy.apply_with_metadata(session, int(target_item))
            for session in shared.template_sessions
        ]
        fake_sessions = [result.session for result in results]
        insertion_slots = [int(result.insertion_slot) for result in results]

        _validate_internal_insertion_truncate_suffix_sessions(
            template_sessions=shared.template_sessions,
            final_sessions=fake_sessions,
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
            run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Internal-Random-Insertion-Truncate-Suffix-NZ samples zero-based "
                "Python insertion slots from [1, len(session)-1], then truncates "
                "after the inserted target; slot0 and tail_slot are excluded."
            ),
        )
        slot_stats_path = target_root / "internal_insertion_truncate_suffix_slot_stats.json"
        save_json(slot_stats_payload, slot_stats_path)

        metadata = build_internal_random_insertion_truncate_suffix_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            insertion_results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            slot_stats_payload=slot_stats_payload,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
        )
        metadata_path = target_root / "internal_random_insertion_truncate_suffix_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=fake_sessions,
            metadata={
                "internal_insertion_truncate_suffix_slot_stats_path": str(
                    slot_stats_path
                ),
                "internal_random_insertion_truncate_suffix_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=(
                        INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE
                    ),
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_internal_random_insertion_truncate_suffix_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[InternalRandomInsertionTruncateSuffixResult],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(insertion_results):
        raise ValueError("insertion_results must align 1:1 with template_sessions.")

    final_sessions = [result.session for result in insertion_results]
    inserted_sessions = [
        result.inserted_session_before_truncation for result in insertion_results
    ]
    overall_stats = slot_stats_payload.get("overall", {})
    if not isinstance(overall_stats, dict):
        raise ValueError("slot_stats_payload must contain an overall object.")

    target = int(target_item)
    final_target_counts = [
        int(result.target_occurrence_count_final) for result in insertion_results
    ]
    after_insertion_target_counts = [
        int(result.target_occurrence_count_after_insertion)
        for result in insertion_results
    ]
    truncated_lengths = [
        int(len(result.truncated_suffix)) for result in insertion_results
    ]
    target_tail_count = int(
        sum(1 for session in final_sessions if session and int(session[-1]) == target)
    )
    total = int(len(insertion_results))
    insertion_slot_counts = Counter(
        int(result.insertion_slot) for result in insertion_results
    )

    shared_key = shared_attack_artifact_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    return {
        "run_type": INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "internal_insertion_truncate_suffix",
        "insertion_policy": "internal_random_nonzero_when_possible",
        "suffix_strategy": "truncate_after_target",
        "method_is_diagnostic": True,
        "suffix_truncation_changes_length_distribution": True,
        "not_length_matched_to_internal_insertion": True,
        "not_length_matched_to_random_nz": True,
        "truncate_may_remove_pre_existing_target_in_suffix": True,
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
        "shared_fake_sessions_key": shared_key,
        "random_nz_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_NZ_RUN_TYPE,
        ),
        "internal_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "internal_replacement_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "same_shared_fake_sessions_as_internal_insertion_expected": True,
        "same_shared_fake_sessions_as_internal_replacement_expected": True,
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "internal_insertion_slot_topk_ratio": float(config.attack.replacement_topk_ratio),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "original_length_distribution": _length_distribution(template_sessions),
        "inserted_length_distribution": _length_distribution(inserted_sessions),
        "final_length_distribution": _length_distribution(final_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "length_shift_from_template_summary": _length_shift_summary_from_pairs(
            template_sessions,
            final_sessions,
        ),
        "length_shift_from_inserted_summary": _length_shift_summary_from_pairs(
            inserted_sessions,
            final_sessions,
        ),
        "insertion_slot_counts": dict(overall_stats.get("slot_counts", {})),
        "insertion_slot_ratios": dict(overall_stats.get("slot_ratios", {})),
        "insertion_slot_group_counts": dict(
            overall_stats.get("slot_group_counts", {})
        ),
        "insertion_slot_group_ratios": dict(
            overall_stats.get("slot_group_ratios", {})
        ),
        "insertion_slot_in_template": {
            "counts": _distribution_from_counter(insertion_slot_counts),
            "ratios": _ratio_dict(insertion_slot_counts),
        },
        "target_position_in_inserted_session": {
            "counts": _distribution_from_counter(insertion_slot_counts),
            "ratios": _ratio_dict(insertion_slot_counts),
        },
        "target_position_in_final_session": {
            "counts": _distribution_from_counter(insertion_slot_counts),
            "ratios": _ratio_dict(insertion_slot_counts),
        },
        "tail_slot_count": int(overall_stats.get("tail_slot_count", 0)),
        "tail_slot_ratio": float(overall_stats.get("tail_slot_ratio", 0.0)),
        "tail_slot_excluded": True,
        "slot0_excluded": True,
        "every_inserted_target_had_left_neighbor_before_truncation": bool(
            all(result.insertion_slot > 0 for result in insertion_results)
        ),
        "every_inserted_target_had_right_neighbor_before_truncation": bool(
            all(result.insertion_slot < result.original_length for result in insertion_results)
        ),
        "every_final_target_has_left_neighbor": bool(
            all(result.session and result.session[-1] == target and len(result.session) >= 2 for result in insertion_results)
        ),
        "every_final_target_is_tail": bool(target_tail_count == total),
        "target_tail_count": target_tail_count,
        "target_tail_ratio": float(target_tail_count) / float(total) if total else 0.0,
        "truncated_suffix_total_item_count": int(sum(truncated_lengths)),
        "truncated_suffix_length_distribution": _distribution_from_counter(
            Counter(truncated_lengths)
        ),
        "truncated_suffix_length_summary": _numeric_summary(truncated_lengths),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in insertion_results if result.pre_existing_target_count > 0)
        ),
        "injected_sessions_containing_target_count": int(
            sum(1 for count in final_target_counts if count > 0)
        ),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in final_target_counts)
        ),
        "target_occurrence_count_after_insertion": _numeric_summary(
            after_insertion_target_counts
        ),
        "target_occurrence_count_final": _numeric_summary(final_target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in final_target_counts if count > 1)
        ),
        "previews": _internal_insertion_truncate_suffix_previews(
            template_sessions=template_sessions,
            insertion_results=insertion_results,
            limit=preview_limit,
        ),
    }


def _validate_internal_insertion_truncate_suffix_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomInsertionTruncateSuffixResult],
    target_item: int,
) -> None:
    if len(template_sessions) != len(final_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(results):
        raise RuntimeError("Result metadata count does not equal template count.")
    target = int(target_item)
    for original, final, result in zip(template_sessions, final_sessions, results):
        original_list = [int(item) for item in original]
        final_list = [int(item) for item in final]
        inserted_list = [int(item) for item in result.inserted_session_before_truncation]
        slot = int(result.insertion_slot)
        if len(original_list) < 2:
            raise RuntimeError(
                "Internal-Random-Insertion-Truncate-Suffix-NZ requires template "
                "session length >= 2."
            )
        if slot < 1 or slot > len(original_list) - 1:
            raise RuntimeError(
                "Internal-Random-Insertion-Truncate-Suffix-NZ selected slot0 or tail slot."
            )
        expected_inserted = original_list[:slot] + [target] + original_list[slot:]
        if inserted_list != expected_inserted:
            raise RuntimeError("Inserted session before truncation is invalid.")
        if not final_list or final_list[-1] != target:
            raise RuntimeError("Truncated final session does not end in target.")
        expected_final = inserted_list[: slot + 1]
        if final_list != expected_final:
            raise RuntimeError("Final session is not the inserted prefix through target.")
        if len(final_list) < 2:
            raise RuntimeError("Truncated final session length is less than 2.")
        if target not in set(final_list):
            raise RuntimeError("Truncated final session is missing target item.")
        if int(result.left_item) != original_list[slot - 1]:
            raise RuntimeError("Truncate left neighbor metadata is invalid.")
        if int(result.original_right_item_before_truncation) != original_list[slot]:
            raise RuntimeError("Truncate right neighbor metadata is invalid.")


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


def _distribution_from_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(item): int(count) for item, count in sorted(counter.items())}


def _ratio_dict(counter: Counter[int]) -> dict[str, float]:
    total = int(sum(counter.values()))
    if total <= 0:
        return {str(item): 0.0 for item in sorted(counter)}
    return {
        str(item): float(count) / float(total)
        for item, count in sorted(counter.items())
    }


def _internal_insertion_truncate_suffix_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[InternalRandomInsertionTruncateSuffixResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, result in zip(template_sessions[:limit], insertion_results[:limit]):
        slot = int(result.insertion_slot)
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "inserted_session_before_truncation": [
                    int(item) for item in result.inserted_session_before_truncation
                ],
                "final_session": [int(item) for item in result.session],
                "insertion_slot": slot,
                "insertion_slot_in_template": slot,
                "target_position_in_inserted_session": slot,
                "target_position_in_final_session": slot,
                "left_item": int(result.left_item),
                "original_right_item_before_truncation": int(
                    result.original_right_item_before_truncation
                ),
                "truncated_suffix": [int(item) for item in result.truncated_suffix],
                "original_length": int(result.original_length),
                "inserted_length": int(result.inserted_length),
                "final_length": int(result.final_length),
                "length_shift_from_template": int(result.final_length)
                - int(result.original_length),
                "length_shift_from_inserted": int(result.final_length)
                - int(result.inserted_length),
                "pre_existing_target_count": int(result.pre_existing_target_count),
                "target_occurrence_count_after_insertion": int(
                    result.target_occurrence_count_after_insertion
                ),
                "target_occurrence_count_final": int(
                    result.target_occurrence_count_final
                ),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_internal_random_insertion_truncate_suffix_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
