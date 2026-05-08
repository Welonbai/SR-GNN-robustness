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
    INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.internal_random_insertion_suffix_variants import (
    InternalRandomInsertionSuccessorRepairPolicy,
    InternalRandomInsertionSuccessorRepairResult,
    build_target_successor_counts,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload


DEFAULT_INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_internal_random_insertion_successor_repair_"
    "nonzero_when_possible_ratio1_srgnn_targets5418_4092_9496_partial4.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"
SUCCESSOR_TOP_K = 5


def run_internal_random_insertion_successor_repair_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Internal-Random-Insertion-Successor-Repair-NZ expects "
            "data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        successor_counts = build_target_successor_counts(
            shared.canonical_dataset.train_sub,
            int(target_item),
            exclude_target=False,
        )
        successor_rng_seed = int(config.seeds.position_opt_seed) + int(target_item)
        policy = InternalRandomInsertionSuccessorRepairPolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            successor_counts=successor_counts,
            successor_top_k=SUCCESSOR_TOP_K,
            insertion_rng=random.Random(config.seeds.fake_session_seed),
            successor_rng=random.Random(successor_rng_seed),
            exclude_target_from_successor_pool=False,
        )
        results = [
            policy.apply_with_metadata(session, int(target_item))
            for session in shared.template_sessions
        ]
        fake_sessions = [result.session for result in results]
        insertion_slots = [int(result.insertion_slot) for result in results]

        _validate_internal_insertion_successor_repair_sessions(
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
            run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Internal-Random-Insertion-Successor-Repair-NZ samples zero-based "
                "Python insertion slots from [1, len(session)-1], then replaces "
                "the first right item with an empirical target successor when "
                "available; slot0 and tail_slot are excluded."
            ),
        )
        slot_stats_path = target_root / "internal_insertion_successor_repair_slot_stats.json"
        save_json(slot_stats_payload, slot_stats_path)

        metadata = build_internal_random_insertion_successor_repair_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            insertion_results=results,
            successor_counts=successor_counts,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            slot_stats_payload=slot_stats_payload,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
            successor_rng_seed=successor_rng_seed,
        )
        metadata_path = target_root / "internal_random_insertion_successor_repair_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "internal_insertion_successor_repair_slot_stats_path": str(
                    slot_stats_path
                ),
                "internal_random_insertion_successor_repair_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=(
                        INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE
                    ),
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_internal_random_insertion_successor_repair_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[InternalRandomInsertionSuccessorRepairResult],
    successor_counts: Counter[int],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    successor_rng_seed: int,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(insertion_results):
        raise ValueError("insertion_results must align 1:1 with template_sessions.")

    final_sessions = [result.session for result in insertion_results]
    inserted_sessions = [result.inserted_session_before_repair for result in insertion_results]
    overall_stats = slot_stats_payload.get("overall", {})
    if not isinstance(overall_stats, dict):
        raise ValueError("slot_stats_payload must contain an overall object.")

    target = int(target_item)
    total = int(len(insertion_results))
    final_target_counts = [
        int(result.target_occurrence_count_final) for result in insertion_results
    ]
    after_insertion_target_counts = [
        int(result.target_occurrence_count_after_insertion)
        for result in insertion_results
    ]
    insertion_slot_counts = Counter(
        int(result.insertion_slot) for result in insertion_results
    )
    repair_applied_count = int(
        sum(1 for result in insertion_results if result.repair_applied)
    )
    repair_changed_item_count = int(
        sum(1 for result in insertion_results if result.repair_changed_item)
    )
    empty_pool_count = int(
        sum(1 for result in insertion_results if result.successor_pool_empty)
    )
    original_right_items = [
        int(result.original_right_item) for result in insertion_results
    ]
    repaired_right_items = [
        int(result.repaired_right_item) for result in insertion_results
    ]
    sampled_successor_counts = _sampled_successor_counts(insertion_results)
    successor_payload = _successor_topk_payload(successor_counts, SUCCESSOR_TOP_K)
    successor_pool_size = int(len(successor_payload["top_successor_items"]))
    successor_pool_empty = successor_pool_size == 0
    self_successor_count = int(successor_counts.get(target, 0))
    successor_total_count = int(successor_payload["successor_total_count"])
    self_successor_rank = _successor_rank(successor_counts, target)
    sampled_self_successor_count = int(sampled_successor_counts.get(target, 0))
    final_adjacent_target_count = int(
        sum(1 for session in final_sessions if _has_adjacent_pair(session, target, target))
    )
    repair_created_adjacent_target_count = int(
        sum(
            1
            for result in insertion_results
            if result.repair_applied
            and result.repair_changed_item
            and int(result.repaired_right_item) == target
        )
    )

    shared_key = shared_attack_artifact_key(
        config,
        run_type=INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    return {
        "run_type": INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "internal_insertion_successor_repair",
        "insertion_policy": "internal_random_nonzero_when_possible",
        "suffix_strategy": "one_step_target_successor_repair",
        "method_is_diagnostic": True,
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
        "successor_top_k": SUCCESSOR_TOP_K,
        "successor_top_k_configurable": False,
        "successor_top_k_source": "runner_default",
        "successor_sampling": "empirical_frequency",
        "successor_pool_source": "train_sub_immediate_successors",
        "successor_repair_definition": "empirical_immediate_successor_allow_self",
        "successor_counts_item_id_space": "canonical_remapped_item_ids",
        "exclude_target_from_successor_pool": False,
        "insertion_rng_seed": int(config.seeds.fake_session_seed),
        "successor_rng_seed": int(successor_rng_seed),
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
        "every_inserted_target_has_left_neighbor": bool(
            all(result.insertion_slot > 0 for result in insertion_results)
        ),
        "every_inserted_target_has_right_neighbor": bool(
            all(result.insertion_slot < result.original_length for result in insertion_results)
        ),
        "every_final_target_has_left_neighbor": bool(
            all(result.insertion_slot > 0 for result in insertion_results)
        ),
        "every_final_target_has_right_neighbor": bool(
            all(result.insertion_slot < result.final_length - 1 for result in insertion_results)
        ),
        "repair_applied_count": repair_applied_count,
        "repair_applied_ratio": (
            float(repair_applied_count) / float(total) if total else 0.0
        ),
        "repair_changed_item_count": repair_changed_item_count,
        "repair_changed_item_ratio": (
            float(repair_changed_item_count) / float(total) if total else 0.0
        ),
        "successor_repair_available": not successor_pool_empty,
        "repair_skipped_due_to_empty_pool_count": empty_pool_count,
        "successor_pool_empty": successor_pool_empty,
        "successor_pool_size": successor_pool_size,
        "successor_total_count": successor_total_count,
        "successor_pool_total_count": int(successor_payload["successor_pool_total_count"]),
        "top_successor_items": successor_payload["top_successor_items"],
        "top_successor_counts": successor_payload["top_successor_counts"],
        "top1_successor_share": float(successor_payload["top1_successor_share"]),
        "top5_successor_share": float(successor_payload["top5_successor_share"]),
        "self_successor_count": self_successor_count,
        "self_successor_share": (
            float(self_successor_count) / float(successor_total_count)
            if successor_total_count
            else 0.0
        ),
        "self_successor_in_topk": bool(
            target in {int(item) for item in successor_payload["top_successor_items"]}
        ),
        "self_successor_rank_in_successor_counts": self_successor_rank,
        "sampled_self_successor_count": sampled_self_successor_count,
        "sampled_self_successor_ratio": (
            float(sampled_self_successor_count) / float(repair_applied_count)
            if repair_applied_count
            else 0.0
        ),
        "final_sessions_with_adjacent_target_target_count": final_adjacent_target_count,
        "final_sessions_with_adjacent_target_target_ratio": (
            float(final_adjacent_target_count) / float(total) if total else 0.0
        ),
        "repair_created_adjacent_target_target_count": repair_created_adjacent_target_count,
        "repair_created_adjacent_target_target_ratio": (
            float(repair_created_adjacent_target_count) / float(total)
            if total
            else 0.0
        ),
        "sampled_successor_counts": _distribution_from_counter(
            sampled_successor_counts
        ),
        "sampled_successor_ratios": _ratio_dict(sampled_successor_counts),
        "original_right_item_count_summary": _counter_summary(original_right_items),
        "repaired_right_item_count_summary": _counter_summary(repaired_right_items),
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
        "previews": _internal_insertion_successor_repair_previews(
            template_sessions=template_sessions,
            insertion_results=insertion_results,
            limit=preview_limit,
        ),
    }


def _validate_internal_insertion_successor_repair_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[InternalRandomInsertionSuccessorRepairResult],
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
        inserted_list = [int(item) for item in result.inserted_session_before_repair]
        slot = int(result.insertion_slot)
        if len(original_list) < 2:
            raise RuntimeError(
                "Internal-Random-Insertion-Successor-Repair-NZ requires template "
                "session length >= 2."
            )
        if slot < 1 or slot > len(original_list) - 1:
            raise RuntimeError(
                "Internal-Random-Insertion-Successor-Repair-NZ selected slot0 or tail slot."
            )
        expected_inserted = original_list[:slot] + [target] + original_list[slot:]
        if inserted_list != expected_inserted:
            raise RuntimeError("Inserted session before repair is invalid.")
        if len(final_list) != len(inserted_list):
            raise RuntimeError("Successor repair changed session length.")
        if final_list[slot] != target:
            raise RuntimeError("Successor repair moved the inserted target.")
        if slot <= 0 or slot >= len(final_list) - 1:
            raise RuntimeError("Successor repaired target lacks left or right neighbor.")
        changed_positions = [
            index
            for index, (before, after) in enumerate(zip(inserted_list, final_list))
            if before != after
        ]
        if any(index != slot + 1 for index in changed_positions):
            raise RuntimeError(
                "Successor repair changed a position other than the first right item."
            )
        if final_list[: slot + 1] != inserted_list[: slot + 1]:
            raise RuntimeError("Successor repair changed prefix through target.")
        if final_list[slot + 2 :] != inserted_list[slot + 2 :]:
            raise RuntimeError("Successor repair changed suffix after first right item.")
        if result.successor_pool_empty:
            if result.repair_applied:
                raise RuntimeError("Empty successor pool cannot apply repair.")
            if final_list != inserted_list:
                raise RuntimeError("Empty successor pool changed the inserted session.")
        if result.repair_applied and not result.successor_pool_empty:
            if int(result.repaired_right_item) not in {
                int(item) for item in result.successor_pool
            }:
                raise RuntimeError("Repaired right item is not in successor pool.")
        if target not in set(final_list):
            raise RuntimeError("Successor repaired final session is missing target item.")


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


def _successor_topk_payload(
    successor_counts: Counter[int],
    top_k: int,
) -> dict[str, object]:
    ordered = sorted(
        (
            (int(item), int(count))
            for item, count in successor_counts.items()
            if int(count) > 0
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    top = ordered[: int(top_k)]
    total_count = int(sum(count for _, count in ordered))
    top_total = int(sum(count for _, count in top))
    top1_count = int(top[0][1]) if top else 0
    return {
        "top_successor_items": [int(item) for item, _ in top],
        "top_successor_counts": [int(count) for _, count in top],
        "successor_total_count": total_count,
        "successor_pool_total_count": top_total,
        "top1_successor_share": (
            float(top1_count) / float(total_count) if total_count else 0.0
        ),
        "top5_successor_share": (
            float(top_total) / float(total_count) if total_count else 0.0
        ),
    }


def _successor_rank(successor_counts: Counter[int], target_item: int) -> int | None:
    ordered = sorted(
        (
            (int(item), int(count))
            for item, count in successor_counts.items()
            if int(count) > 0
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )
    target = int(target_item)
    for rank, (item, _) in enumerate(ordered, start=1):
        if item == target:
            return int(rank)
    return None


def _has_adjacent_pair(
    session: Sequence[int],
    left_item: int,
    right_item: int,
) -> bool:
    left = int(left_item)
    right = int(right_item)
    normalized = [int(item) for item in session]
    return any(
        first == left and second == right
        for first, second in zip(normalized, normalized[1:])
    )


def _sampled_successor_counts(
    results: Sequence[InternalRandomInsertionSuccessorRepairResult],
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for result in results:
        if result.sampled_successor is not None:
            counts[int(result.sampled_successor)] += 1
    return counts


def _internal_insertion_successor_repair_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[InternalRandomInsertionSuccessorRepairResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, result in zip(template_sessions[:limit], insertion_results[:limit]):
        slot = int(result.insertion_slot)
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "inserted_session_before_repair": [
                    int(item) for item in result.inserted_session_before_repair
                ],
                "final_session": [int(item) for item in result.session],
                "insertion_slot": slot,
                "insertion_slot_in_template": slot,
                "target_position_in_inserted_session": slot,
                "target_position_in_final_session": slot,
                "left_item": int(result.left_item),
                "original_right_item": int(result.original_right_item),
                "repaired_right_item": int(result.repaired_right_item),
                "repair_applied": bool(result.repair_applied),
                "repair_changed_item": bool(result.repair_changed_item),
                "successor_pool": [int(item) for item in result.successor_pool],
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
        default=DEFAULT_INTERNAL_RANDOM_INSERTION_SUCCESSOR_REPAIR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_internal_random_insertion_successor_repair_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
