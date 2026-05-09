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
    INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.internal_random_insertion_generated_suffix import (
    InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy,
    InternalRandomInsertionSuccessorSeededGeneratedContinuationResult,
    successor_rank,
    successor_smoothed_payload,
)
from attack.insertion.internal_random_insertion_suffix_variants import (
    build_target_successor_counts,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_internal_random_insertion_generated_continuation import (
    _distribution_from_counter,
    _existing_path_or_none,
    _has_adjacent_pair,
    _ratio_dict,
    _validate_internal_insertion_generated_continuation_sessions,
    build_internal_random_insertion_generated_continuation_metadata,
)


DEFAULT_INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_internal_random_insertion_successor_seeded_"
    "generated_continuation_nonzero_when_possible_ratio1_srgnn_partial4.yaml"
)
SUCCESSOR_SEED_RATIO = 0.25
SUCCESSOR_POOL_TOP_K = 10
SUCCESSOR_SMOOTHING_ALPHA = 0.5


def run_internal_random_insertion_successor_seeded_generated_continuation_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError(
            "Internal-Random-Insertion-Successor-Seeded-Generated-Continuation-NZ "
            "expects data.poison_train_only to be true."
        )
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=(
            INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
        ),
        require_poison_runner=True,
        config_path=config_path,
    )
    if shared.poison_runner is None:
        raise RuntimeError(
            "Poison runner is required for successor-seeded generated-continuation "
            "suffix generation."
        )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        successor_counts = build_target_successor_counts(
            shared.canonical_dataset.train_sub,
            int(target_item),
            exclude_target=False,
        )
        policy = InternalRandomInsertionSuccessorSeededGeneratedContinuationPolicy(
            topk_ratio=config.attack.replacement_topk_ratio,
            poison_runner=shared.poison_runner,
            generation_topk=config.attack.fake_session_generation_topk,
            successor_counts=successor_counts,
            successor_pool_top_k=SUCCESSOR_POOL_TOP_K,
            successor_seed_ratio=SUCCESSOR_SEED_RATIO,
            successor_smoothing_alpha=SUCCESSOR_SMOOTHING_ALPHA,
            insertion_rng=random.Random(config.seeds.fake_session_seed),
            successor_seed_rng=random.Random(
                int(config.seeds.position_opt_seed) + int(target_item)
            ),
            successor_item_rng=random.Random(
                int(config.seeds.position_opt_seed) + int(target_item) + 1000003
            ),
            generation_rng_base_seed=config.seeds.fake_session_seed,
        )
        results = [
            policy.apply_with_metadata(session, int(target_item), index)
            for index, session in enumerate(shared.template_sessions)
        ]
        fake_sessions = [result.session for result in results]
        insertion_slots = [int(result.insertion_slot) for result in results]

        max_item = max(shared.stats.item_counts)
        _validate_internal_insertion_successor_seeded_generated_continuation_sessions(
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
            run_type=(
                INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
            ),
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=(
                INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
            ),
            target_item=int(target_item),
            note=(
                "Internal-Random-Insertion-Successor-Seeded-Generated-Continuation-NZ "
                "samples internal non-tail insertion slots, discards the original "
                "suffix, optionally seeds the first regenerated suffix item from "
                "empirical target successors, and regenerates all suffixes."
            ),
        )
        slot_stats_path = (
            target_root
            / "internal_insertion_successor_seeded_generated_continuation_slot_stats.json"
        )
        save_json(slot_stats_payload, slot_stats_path)

        metadata = build_internal_random_insertion_successor_seeded_generated_continuation_metadata(
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
            generation_topk=config.attack.fake_session_generation_topk,
            generation_rng_base_seed=config.seeds.fake_session_seed,
        )
        metadata_path = (
            target_root
            / "internal_random_insertion_successor_seeded_generated_continuation_metadata.json"
        )
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "internal_insertion_successor_seeded_generated_continuation_slot_stats_path": str(
                    slot_stats_path
                ),
                "internal_random_insertion_successor_seeded_generated_continuation_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=(
                        INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
                    ),
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=(
            INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
        ),
        build_poisoned=build_poisoned,
    )


def build_internal_random_insertion_successor_seeded_generated_continuation_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[
        InternalRandomInsertionSuccessorSeededGeneratedContinuationResult
    ],
    successor_counts: Counter[int],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    generation_topk: int,
    generation_rng_base_seed: int,
    preview_limit: int = 20,
) -> dict[str, object]:
    metadata = build_internal_random_insertion_generated_continuation_metadata(
        config=config,
        run_type=(
            INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE
        ),
        operation="internal_insertion_successor_seeded_generated_continuation",
        suffix_strategy="successor_seeded_target_conditioned_generated_continuation",
        target_item=target_item,
        template_sessions=template_sessions,
        insertion_results=insertion_results,
        clean_train_sessions=clean_train_sessions,
        slot_stats_payload=slot_stats_payload,
        template_fake_sessions_path=template_fake_sessions_path,
        poison_model_checkpoint_path=poison_model_checkpoint_path,
        generation_topk=generation_topk,
        generation_rng_base_seed=generation_rng_base_seed,
        preview_limit=preview_limit,
    )
    total = int(len(insertion_results))
    target = int(target_item)
    successor_payload = successor_smoothed_payload(
        successor_counts,
        SUCCESSOR_POOL_TOP_K,
        SUCCESSOR_SMOOTHING_ALPHA,
    )
    successor_pool = [int(item) for item in successor_payload["top_successor_items"]]
    successor_pool_empty = not successor_pool
    eligible_count = int(
        sum(
            1
            for result in insertion_results
            if int(result.suffix_length) > 0 and not successor_pool_empty
        )
    )
    attempted_count = int(
        sum(1 for result in insertion_results if result.successor_seed_attempted)
    )
    applied_count = int(
        sum(1 for result in insertion_results if result.successor_seed_applied)
    )
    sampled_seed_counts: Counter[int] = Counter(
        int(result.successor_seed_item)
        for result in insertion_results
        if result.successor_seed_item is not None
    )
    self_seed_count = int(sampled_seed_counts.get(target, 0))
    adjacent_count = int(
        sum(1 for result in insertion_results if _has_adjacent_pair(result.session, target, target))
    )
    seed_created_adjacent_count = int(
        sum(
            1
            for result in insertion_results
            if result.successor_seed_applied
            and result.successor_seed_item == target
        )
    )
    seeded_mode_count = int(
        sum(
            1
            for result in insertion_results
            if result.repair_generation_mode == "successor_seeded"
        )
    )
    pure_mode_count = int(
        sum(
            1
            for result in insertion_results
            if result.repair_generation_mode == "pure_generated"
        )
    )
    successor_total_count = int(successor_payload["successor_total_count"])
    self_successor_count = int(successor_counts.get(target, 0))

    metadata.update(
        {
            "successor_seed_ratio": SUCCESSOR_SEED_RATIO,
            "successor_pool_top_k": SUCCESSOR_POOL_TOP_K,
            "successor_sampling": "power_smoothed_empirical_frequency",
            "successor_smoothing_alpha": SUCCESSOR_SMOOTHING_ALPHA,
            "successor_pool_source": "train_sub_immediate_successors",
            "allow_self_successor": True,
            "non_seeded_sessions_use_generated_continuation": True,
            "original_suffix_preserved_for_non_seeded_sessions": False,
            "successor_total_count": successor_total_count,
            "successor_pool_total_count": int(
                successor_payload["successor_pool_total_count"]
            ),
            "successor_pool_size": int(successor_payload["successor_pool_size"]),
            "top_successor_items": successor_payload["top_successor_items"],
            "top_successor_counts": successor_payload["top_successor_counts"],
            "top_successor_smoothed_weights": successor_payload[
                "top_successor_smoothed_weights"
            ],
            "top_successor_smoothed_probabilities": successor_payload[
                "top_successor_smoothed_probabilities"
            ],
            "top1_successor_share": float(successor_payload["top1_successor_share"]),
            "top10_successor_share": float(successor_payload["top10_successor_share"]),
            "self_successor_count": self_successor_count,
            "self_successor_share": (
                float(self_successor_count) / float(successor_total_count)
                if successor_total_count
                else 0.0
            ),
            "self_successor_in_topk": bool(target in set(successor_pool)),
            "self_successor_rank_in_successor_counts": successor_rank(
                successor_counts,
                target,
            ),
            "successor_seed_eligible_count": eligible_count,
            "successor_seed_eligible_ratio": (
                float(eligible_count) / float(total) if total else 0.0
            ),
            "successor_seed_attempted_count": attempted_count,
            "successor_seed_attempted_ratio": (
                float(attempted_count) / float(total) if total else 0.0
            ),
            "successor_seed_attempted_definition": (
                "eligible_sessions_selected_by_bernoulli_draw"
            ),
            "successor_seed_applied_count": applied_count,
            "successor_seed_applied_ratio": (
                float(applied_count) / float(total) if total else 0.0
            ),
            "successor_pool_empty": successor_pool_empty,
            "sampled_successor_seed_counts": _distribution_from_counter(
                sampled_seed_counts
            ),
            "sampled_successor_seed_ratios": _ratio_dict(sampled_seed_counts),
            "sampled_self_successor_seed_count": self_seed_count,
            "sampled_self_successor_seed_ratio": (
                float(self_seed_count) / float(applied_count)
                if applied_count
                else 0.0
            ),
            "final_sessions_with_adjacent_target_target_count": adjacent_count,
            "final_sessions_with_adjacent_target_target_ratio": (
                float(adjacent_count) / float(total) if total else 0.0
            ),
            "seed_created_adjacent_target_target_count": seed_created_adjacent_count,
            "seed_created_adjacent_target_target_ratio": (
                float(seed_created_adjacent_count) / float(total) if total else 0.0
            ),
            "pure_generated_mode_count": pure_mode_count,
            "pure_generated_mode_ratio": (
                float(pure_mode_count) / float(total) if total else 0.0
            ),
            "successor_seeded_mode_count": seeded_mode_count,
            "successor_seeded_mode_ratio": (
                float(seeded_mode_count) / float(total) if total else 0.0
            ),
            "previews": _successor_seeded_previews(
                template_sessions=template_sessions,
                insertion_results=insertion_results,
                limit=preview_limit,
            ),
        }
    )
    return metadata


def _validate_internal_insertion_successor_seeded_generated_continuation_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    final_sessions: Sequence[Sequence[int]],
    results: Sequence[
        InternalRandomInsertionSuccessorSeededGeneratedContinuationResult
    ],
    target_item: int,
    max_item_id: int | None = None,
) -> None:
    _validate_internal_insertion_generated_continuation_sessions(
        template_sessions=template_sessions,
        final_sessions=final_sessions,
        results=results,
        target_item=target_item,
        max_item_id=max_item_id,
    )
    for result in results:
        if result.successor_pool_empty and result.successor_seed_applied:
            raise RuntimeError("Empty successor pool cannot apply successor seed.")
        if result.successor_seed_applied:
            if not result.generated_suffix:
                raise RuntimeError("Seeded result must have a generated suffix.")
            if result.successor_seed_item is None:
                raise RuntimeError("Seeded result is missing successor_seed_item.")
            if int(result.generated_suffix[0]) != int(result.successor_seed_item):
                raise RuntimeError("Seeded suffix does not start with successor seed.")
            if int(result.successor_seed_item) not in {
                int(item) for item in result.successor_pool
            }:
                raise RuntimeError("Successor seed item is not in successor pool.")
            if result.repair_generation_mode != "successor_seeded":
                raise RuntimeError("Seeded result has invalid generation mode.")
        else:
            if result.repair_generation_mode != "pure_generated":
                raise RuntimeError("Non-seeded result has invalid generation mode.")


def _successor_seeded_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    insertion_results: Sequence[
        InternalRandomInsertionSuccessorSeededGeneratedContinuationResult
    ],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, result in zip(template_sessions[:limit], insertion_results[:limit]):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "inserted_session_before_generation": [
                    int(item) for item in result.inserted_session_before_generation
                ],
                "final_session": [int(item) for item in result.session],
                "insertion_slot": int(result.insertion_slot),
                "left_item": int(result.left_item),
                "original_right_item": int(result.original_right_item),
                "original_suffix": [int(item) for item in result.original_suffix],
                "generated_suffix": [int(item) for item in result.generated_suffix],
                "successor_seed_attempted": bool(result.successor_seed_attempted),
                "successor_seed_applied": bool(result.successor_seed_applied),
                "successor_seed_item": (
                    None
                    if result.successor_seed_item is None
                    else int(result.successor_seed_item)
                ),
                "successor_pool": [int(item) for item in result.successor_pool],
                "self_successor_seed": bool(result.self_successor_seed),
                "repair_generation_mode": str(result.repair_generation_mode),
                "generated_suffix_after_seed": [
                    int(item) for item in result.generated_suffix_after_seed
                ],
                "original_length": int(result.original_length),
                "inserted_length": int(result.inserted_length),
                "final_length": int(result.final_length),
                "target_position": int(result.target_position),
                "suffix_length": int(result.suffix_length),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            DEFAULT_INTERNAL_RANDOM_INSERTION_SUCCESSOR_SEEDED_GENERATED_CONTINUATION_CONFIG_PATH
        ),
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_internal_random_insertion_successor_seeded_generated_continuation_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
