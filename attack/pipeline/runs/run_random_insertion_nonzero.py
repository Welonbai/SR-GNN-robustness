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
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.random_insertion_nonzero_when_possible import (
    RandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload


DEFAULT_RANDOM_INSERTION_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_random_insertion_nonzero_when_possible_ratio1_"
    "srgnn_target11103_partial4.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_random_insertion_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Random-Insertion-NZ expects data.poison_train_only to be true.")
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        rng = random.Random(config.seeds.fake_session_seed)
        policy = RandomInsertionNonzeroWhenPossiblePolicy(
            config.attack.replacement_topk_ratio,
            rng=rng,
        )
        fake_sessions: list[list[int]] = []
        insertion_slots: list[int] = []
        for session in shared.template_sessions:
            result = policy.apply_with_metadata(session, int(target_item))
            fake_sessions.append(result.session)
            insertion_slots.append(int(result.insertion_slot))

        _validate_inserted_sessions(
            template_sessions=shared.template_sessions,
            inserted_sessions=fake_sessions,
            insertion_slots=insertion_slots,
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
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Random-Insertion-NZ uses zero-based Python insertion slots; "
                "tail_slot is an overlapping summary group."
            ),
        )
        insertion_slot_stats_path = target_root / "insertion_slot_stats.json"
        save_json(slot_stats_payload, insertion_slot_stats_path)

        metadata = build_random_insertion_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            inserted_sessions=fake_sessions,
            insertion_slots=insertion_slots,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            slot_stats_payload=slot_stats_payload,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=shared.shared_paths["poison_model"],
        )
        metadata_path = target_root / "random_insertion_slot_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "insertion_slot_stats_path": str(insertion_slot_stats_path),
                "random_insertion_slot_metadata_path": str(metadata_path),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_random_insertion_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    inserted_sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(inserted_sessions):
        raise ValueError("template_sessions and inserted_sessions must have the same length.")
    if len(template_sessions) != len(insertion_slots):
        raise ValueError("template_sessions and insertion_slots must have the same length.")

    target_counts = [
        sum(1 for item in session if int(item) == int(target_item))
        for session in inserted_sessions
    ]
    length_deltas = [
        int(len(inserted)) - int(len(original))
        for original, inserted in zip(template_sessions, inserted_sessions)
    ]
    overall_stats = slot_stats_payload.get("overall", {})
    if not isinstance(overall_stats, dict):
        raise ValueError("slot_stats_payload must contain an overall object.")

    shared_key = shared_attack_artifact_key(
        config,
        run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    random_nz_shared_key = shared_attack_artifact_key(
        config,
        run_type=RANDOM_NZ_RUN_TYPE,
    )
    return {
        "run_type": RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "insertion",
        "insertion_policy": "random_nonzero_when_possible",
        "target_item": int(target_item),
        "fake_session_count": int(len(inserted_sessions)),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "insertion_slot_topk_ratio": float(config.attack.replacement_topk_ratio),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "shared_fake_sessions_key": shared_key,
        "random_nz_shared_fake_sessions_key": random_nz_shared_key,
        "template_fake_sessions_path": str(template_fake_sessions_path),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "poison_model_checkpoint_path": str(poison_model_checkpoint_path),
        "poison_model_key": poison_model_key(config),
        "poison_model_identity": poison_model_key_payload(config),
        "original_length_distribution": _length_distribution(template_sessions),
        "inserted_length_distribution": _length_distribution(inserted_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "insertion_slot_counts": dict(overall_stats.get("slot_counts", {})),
        "insertion_slot_ratios": dict(overall_stats.get("slot_ratios", {})),
        "insertion_slot_group_counts": dict(overall_stats.get("slot_group_counts", {})),
        "insertion_slot_group_ratios": dict(overall_stats.get("slot_group_ratios", {})),
        "tail_slot_count": int(overall_stats.get("tail_slot_count", 0)),
        "tail_slot_ratio": float(overall_stats.get("tail_slot_ratio", 0.0)),
        "tail_slot_is_overlapping_group": True,
        "pre_existing_target_in_template_sessions_count": int(
            sum(
                1
                for session in template_sessions
                if int(target_item) in {int(item) for item in session}
            )
        ),
        "injected_sessions_containing_target_count": int(
            sum(1 for count in target_counts if count > 0)
        ),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in target_counts)
        ),
        "target_occurrence_count_after_insertion": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "length_shift_summary": _numeric_summary(length_deltas),
        "previews": _insertion_previews(
            template_sessions=template_sessions,
            inserted_sessions=inserted_sessions,
            insertion_slots=insertion_slots,
            limit=preview_limit,
        ),
    }


def _validate_inserted_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    inserted_sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
    target_item: int,
) -> None:
    if len(template_sessions) != len(inserted_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(insertion_slots):
        raise RuntimeError("Insertion slot count does not equal template count.")
    for original, inserted, slot in zip(
        template_sessions,
        inserted_sessions,
        insertion_slots,
    ):
        original_list = [int(item) for item in original]
        inserted_list = [int(item) for item in inserted]
        insertion_slot = int(slot)
        if int(target_item) not in set(inserted_list):
            raise RuntimeError("Random-Insertion-NZ injected session is missing target item.")
        if len(inserted_list) != len(original_list) + 1:
            raise RuntimeError("Random-Insertion-NZ injected session length delta is not +1.")
        expected = (
            original_list[:insertion_slot]
            + [int(target_item)]
            + original_list[insertion_slot:]
        )
        if inserted_list != expected:
            raise RuntimeError("Random-Insertion-NZ did not preserve original item order.")


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


def _insertion_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    inserted_sessions: Sequence[Sequence[int]],
    insertion_slots: Sequence[int],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, inserted, slot in zip(
        template_sessions[:limit],
        inserted_sessions[:limit],
        insertion_slots[:limit],
    ):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "inserted_session": [int(item) for item in inserted],
                "insertion_slot": int(slot),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_RANDOM_INSERTION_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_random_insertion_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
