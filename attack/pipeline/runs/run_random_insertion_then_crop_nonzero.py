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
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.random_insertion_then_crop_nonzero_when_possible import (
    DEFAULT_CROP_POLICY,
    RandomInsertionThenCropNonzeroWhenPossiblePolicy,
    RandomInsertionThenCropResult,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload


DEFAULT_INSERTION_THEN_CROP_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_random_insertion_then_crop_nonzero_when_possible_"
    "ratio1_srgnn_target11103_partial5.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_random_insertion_then_crop_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Random-Insertion-Then-Crop-NZ expects data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = RandomInsertionThenCropNonzeroWhenPossiblePolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            rng=random.Random(config.seeds.fake_session_seed),
            crop_policy=DEFAULT_CROP_POLICY,
        )
        results = [
            policy.apply_with_metadata(session, int(target_item))
            for session in shared.template_sessions
        ]
        fake_sessions = [result.session for result in results]
        insertion_slots = [int(result.insertion_slot) for result in results]

        _validate_insertion_then_crop_sessions(
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
            run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Random-Insertion-Then-Crop-NZ samples a nonzero insertion slot, "
                "then deterministically crops the last non-target item."
            ),
        )
        stats_payload = build_insertion_then_crop_stats_payload(
            run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            results=results,
            slot_stats_payload=slot_stats_payload,
        )
        stats_path = target_root / "insertion_then_crop_stats.json"
        save_json(stats_payload, stats_path)

        metadata = build_random_insertion_then_crop_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            final_sessions=fake_sessions,
            results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            stats_payload=stats_payload,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
        )
        metadata_path = target_root / "random_insertion_then_crop_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "insertion_then_crop_stats_path": str(stats_path),
                "random_insertion_then_crop_metadata_path": str(metadata_path),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_insertion_then_crop_stats_payload(
    *,
    run_type: str,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    results: Sequence[RandomInsertionThenCropResult],
    slot_stats_payload: dict[str, object],
) -> dict[str, object]:
    if len(template_sessions) != len(results):
        raise ValueError("results must align 1:1 with template_sessions.")
    crop_positions = [int(result.crop_position) for result in results]
    crop_position_counts: Counter[int] = Counter(crop_positions)
    crop_group_counts = _crop_position_group_counts(results)
    target_positions = [
        int(result.target_position_after_crop)
        for result in results
        if result.target_position_after_crop is not None
    ]
    target_position_counts: Counter[int] = Counter(target_positions)
    target_position_group_counts = _target_position_group_counts(results)
    total = int(len(results))
    overall_slot_stats = slot_stats_payload.get("overall", {})
    if not isinstance(overall_slot_stats, dict):
        raise ValueError("slot_stats_payload must contain an overall object.")

    return {
        "run_type": run_type,
        "target_item": int(target_item),
        "total_sessions": total,
        "insertion_slot_stats": overall_slot_stats,
        "tail_slot_is_overlapping_group": True,
        "crop_position_counts": _stringify_counts(crop_position_counts),
        "crop_position_ratios": _stringify_ratios(crop_position_counts, total=total),
        "crop_position_group_counts": _stringify_named_counts(
            crop_group_counts,
            order=_CROP_POSITION_GROUP_ORDER,
        ),
        "crop_position_group_ratios": _stringify_named_ratios(
            crop_group_counts,
            order=_CROP_POSITION_GROUP_ORDER,
            total=total,
        ),
        "cropped_item_target_count": int(
            sum(1 for result in results if int(result.cropped_item) == int(target_item))
        ),
        "crop_after_inserted_target_count": int(
            sum(1 for result in results if result.crop_position > result.insertion_slot)
        ),
        "crop_before_inserted_target_count": int(
            sum(1 for result in results if result.crop_position < result.insertion_slot)
        ),
        "crop_at_inserted_target_count": int(
            sum(1 for result in results if result.crop_position == result.insertion_slot)
        ),
        "cropped_tail_count": int(
            sum(1 for result in results if result.crop_position == result.inserted_length - 1)
        ),
        "cropped_tail_ratio": (
            float(
                sum(
                    1
                    for result in results
                    if result.crop_position == result.inserted_length - 1
                )
            )
            / float(total)
            if total
            else 0.0
        ),
        "target_position_after_crop_counts": _stringify_counts(target_position_counts),
        "target_position_after_crop_group_counts": _stringify_named_counts(
            target_position_group_counts,
            order=_TARGET_POSITION_GROUP_ORDER,
        ),
        "target_position_after_crop_group_ratios": _stringify_named_ratios(
            target_position_group_counts,
            order=_TARGET_POSITION_GROUP_ORDER,
            total=total,
        ),
        "tail_target_position_after_crop_ratio": (
            float(target_position_group_counts["tail_position_after_crop"]) / float(total)
            if total
            else 0.0
        ),
    }


def build_random_insertion_then_crop_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[RandomInsertionThenCropResult],
    clean_train_sessions: Sequence[Sequence[int]],
    stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(final_sessions):
        raise ValueError("template_sessions and final_sessions must have the same length.")
    if len(template_sessions) != len(results):
        raise ValueError("results must align 1:1 with template_sessions.")

    inserted_sessions = [result.inserted_session for result in results]
    target_counts = [int(result.target_occurrence_count_after_crop) for result in results]
    insertion_length_deltas = [
        int(result.inserted_length) - int(result.original_length) for result in results
    ]
    final_length_deltas = [
        int(result.final_length) - int(result.original_length) for result in results
    ]
    insertion_slot_stats = stats_payload.get("insertion_slot_stats", {})
    if not isinstance(insertion_slot_stats, dict):
        raise ValueError("stats_payload must contain insertion_slot_stats.")
    shared_key = shared_attack_artifact_key(
        config,
        run_type=RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    return {
        "run_type": RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "insertion_then_crop",
        "insertion_policy": "random_nonzero_when_possible",
        "crop_policy": DEFAULT_CROP_POLICY,
        "target_item": int(target_item),
        "fake_session_count": int(len(final_sessions)),
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
        "random_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "tail_replacement_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "tail_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "same_shared_fake_sessions_as_random_insertion_expected": True,
        "original_length_distribution": _length_distribution(template_sessions),
        "inserted_length_distribution": _length_distribution(inserted_sessions),
        "final_length_distribution": _length_distribution(final_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "insertion_length_shift_summary": _numeric_summary(insertion_length_deltas),
        "final_length_shift_summary": _numeric_summary(final_length_deltas),
        "insertion_slot_counts": dict(insertion_slot_stats.get("slot_counts", {})),
        "insertion_slot_ratios": dict(insertion_slot_stats.get("slot_ratios", {})),
        "insertion_slot_group_counts": dict(
            insertion_slot_stats.get("slot_group_counts", {})
        ),
        "insertion_slot_group_ratios": dict(
            insertion_slot_stats.get("slot_group_ratios", {})
        ),
        "tail_slot_count": int(insertion_slot_stats.get("tail_slot_count", 0)),
        "tail_slot_ratio": float(insertion_slot_stats.get("tail_slot_ratio", 0.0)),
        "tail_slot_is_overlapping_group": True,
        "crop_position_counts": dict(stats_payload["crop_position_counts"]),
        "crop_position_ratios": dict(stats_payload["crop_position_ratios"]),
        "crop_position_group_counts": dict(stats_payload["crop_position_group_counts"]),
        "crop_position_group_ratios": dict(stats_payload["crop_position_group_ratios"]),
        "cropped_item_target_count": int(stats_payload["cropped_item_target_count"]),
        "crop_after_inserted_target_count": int(
            stats_payload["crop_after_inserted_target_count"]
        ),
        "crop_before_inserted_target_count": int(
            stats_payload["crop_before_inserted_target_count"]
        ),
        "crop_at_inserted_target_count": int(
            stats_payload["crop_at_inserted_target_count"]
        ),
        "cropped_tail_count": int(stats_payload["cropped_tail_count"]),
        "cropped_tail_ratio": float(stats_payload["cropped_tail_ratio"]),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in results if result.pre_existing_target_count > 0)
        ),
        "injected_sessions_containing_target_count": int(
            sum(1 for count in target_counts if count > 0)
        ),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in target_counts)
        ),
        "target_occurrence_count_after_crop": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "target_position_after_crop_counts": dict(
            stats_payload["target_position_after_crop_counts"]
        ),
        "target_position_after_crop_group_counts": dict(
            stats_payload["target_position_after_crop_group_counts"]
        ),
        "target_position_after_crop_group_ratios": dict(
            stats_payload["target_position_after_crop_group_ratios"]
        ),
        "tail_target_position_after_crop_ratio": float(
            stats_payload["tail_target_position_after_crop_ratio"]
        ),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "insertion_slot_topk_ratio": float(config.attack.replacement_topk_ratio),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "replacement_topk_ratio_used_for_replacement": False,
        "topk_sampling_note": (
            "topk ratio is interpreted as insertion slot top-k ratio before "
            "deterministic crop."
        ),
        "previews": _insertion_then_crop_previews(
            template_sessions=template_sessions,
            results=results,
            limit=preview_limit,
        ),
    }


def _validate_insertion_then_crop_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[RandomInsertionThenCropResult],
    target_item: int,
) -> None:
    if len(template_sessions) != len(final_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(results):
        raise RuntimeError("Result metadata count does not equal template count.")
    for original, final, result in zip(template_sessions, final_sessions, results):
        original_list = [int(item) for item in original]
        final_list = [int(item) for item in final]
        if int(target_item) not in set(final_list):
            raise RuntimeError("Final insertion-then-crop session is missing target item.")
        if int(result.cropped_item) == int(target_item):
            raise RuntimeError("Insertion-then-crop cropped target item.")
        if int(result.inserted_length) != len(original_list) + 1:
            raise RuntimeError("Inserted session length delta is not +1.")
        if len(final_list) != len(original_list):
            raise RuntimeError("Final insertion-then-crop session length changed.")
        if final_list != [int(item) for item in result.session]:
            raise RuntimeError("Final session does not match result metadata.")
        if result.crop_position == result.insertion_slot:
            raise RuntimeError("Insertion-then-crop removed the inserted target.")
        if not _remaining_original_order_is_preserved(
            original=original_list,
            final_session=final_list,
            target_position_after_crop=result.target_position_after_crop,
        ):
            raise RuntimeError("Remaining original items did not preserve relative order.")


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


def _remaining_original_order_is_preserved(
    *,
    original: Sequence[int],
    final_session: Sequence[int],
    target_position_after_crop: int | None,
) -> bool:
    if target_position_after_crop is None:
        return False
    final_without_inserted_target = [
        int(item)
        for index, item in enumerate(final_session)
        if index != int(target_position_after_crop)
    ]
    original_index = 0
    for item in final_without_inserted_target:
        while original_index < len(original) and int(original[original_index]) != int(item):
            original_index += 1
        if original_index >= len(original):
            return False
        original_index += 1
    return True


def _crop_position_group_counts(
    results: Sequence[RandomInsertionThenCropResult],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        for group in _crop_position_groups(
            int(result.crop_position),
            int(result.inserted_length),
        ):
            counts[group] += 1
    for group in _CROP_POSITION_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _crop_position_groups(position: int, inserted_length: int) -> tuple[str, ...]:
    groups: list[str] = []
    if position == 0:
        groups.append("pos0")
    elif position == 1:
        groups.append("pos1")
    elif position == 2:
        groups.append("pos2")
    elif position == 3:
        groups.append("pos3")
    elif 4 <= position <= 5:
        groups.append("pos4_5")
    elif position >= 6:
        groups.append("pos6_plus")
    if position == inserted_length - 1:
        groups.append("tail_crop_position")
    return tuple(groups)


_CROP_POSITION_GROUP_ORDER = (
    "pos0",
    "pos1",
    "pos2",
    "pos3",
    "pos4_5",
    "pos6_plus",
    "tail_crop_position",
)


def _target_position_group_counts(
    results: Sequence[RandomInsertionThenCropResult],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in results:
        if result.target_position_after_crop is None:
            continue
        for group in _target_position_groups(
            int(result.target_position_after_crop),
            int(result.final_length),
        ):
            counts[group] += 1
    for group in _TARGET_POSITION_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _target_position_groups(position: int, final_length: int) -> tuple[str, ...]:
    groups: list[str] = []
    if position == 0:
        groups.append("pos0")
    elif position == 1:
        groups.append("pos1")
    elif position == 2:
        groups.append("pos2")
    elif position == 3:
        groups.append("pos3")
    elif 4 <= position <= 5:
        groups.append("pos4_5")
    elif position >= 6:
        groups.append("pos6_plus")
    if position == final_length - 1:
        groups.append("tail_position_after_crop")
    return tuple(groups)


_TARGET_POSITION_GROUP_ORDER = (
    "pos0",
    "pos1",
    "pos2",
    "pos3",
    "pos4_5",
    "pos6_plus",
    "tail_position_after_crop",
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


def _stringify_named_counts(counter: Counter[str], *, order: Sequence[str]) -> dict[str, int]:
    return {group: int(counter.get(group, 0)) for group in order}


def _stringify_named_ratios(
    counter: Counter[str],
    *,
    order: Sequence[str],
    total: int,
) -> dict[str, float]:
    if total <= 0:
        return {group: 0.0 for group in order}
    return {group: float(counter.get(group, 0)) / float(total) for group in order}


def _insertion_then_crop_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    results: Sequence[RandomInsertionThenCropResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, result in zip(template_sessions[:limit], results[:limit]):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "inserted_session": [int(item) for item in result.inserted_session],
                "final_cropped_session": [int(item) for item in result.session],
                "insertion_slot": int(result.insertion_slot),
                "crop_position": int(result.crop_position),
                "cropped_item": int(result.cropped_item),
                "target_position_after_crop": (
                    None
                    if result.target_position_after_crop is None
                    else int(result.target_position_after_crop)
                ),
                "pre_existing_target_count": int(result.pre_existing_target_count),
                "target_occurrence_count_after_crop": int(
                    result.target_occurrence_count_after_crop
                ),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_INSERTION_THEN_CROP_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_random_insertion_then_crop_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
