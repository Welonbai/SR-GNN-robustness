from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.vulnerable_anchor_internal_construction import (
    VulnerableAnchorInternalConstructionPolicy as AnchorInternalConstructionPolicy,
    VulnerableAnchorInternalConstructionResult as AnchorInternalConstructionResult,
    stable_anchor_pool_hash,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload
from attack.pipeline.runs.run_vulnerable_anchor_internal_construction import (
    RANDOM_NZ_RUN_TYPE,
    _construction_previews,
    _counter_ratios,
    _counter_summary,
    _entropy,
    _existing_path_or_none,
    _frequency_delta_summary,
    _item_popularity_ranks,
    _length_distribution,
    _numeric_summary,
    _rank_summary,
    _stringify_counts,
    _train_frequency_delta_summary,
    _train_frequency_summary,
    _validate_constructed_sessions,
)


DEFAULT_POPULAR_ANCHOR_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_popular_anchor_internal_construction_"
    "top20_ratio1_srgnn_target39588_partial5.yaml"
)


@dataclass(frozen=True)
class LoadedPopularAnchorPool:
    target_item: int
    anchor_pool: list[int]
    anchor_rows: list[dict[str, int]]

    @property
    def selected_anchor_pool_hash(self) -> str:
        return stable_anchor_pool_hash(self.anchor_pool)


def run_popular_anchor_internal_construction(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_config_for_runner(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    loaded_anchors_by_target = {
        int(target_item): build_popular_anchor_pool(
            train_sub_sessions=shared.canonical_dataset.train_sub,
            target_item=int(target_item),
            anchor_top_m=int(config.anchor_construction.anchor_top_m),
        )
        for target_item in config.targets.explicit_list
    }
    attack_identity_context = build_popular_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target=loaded_anchors_by_target,
    )
    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        loaded_anchors = loaded_anchors_by_target[int(target_item)]
        rng = random.Random(
            derive_seed(
                int(config.seeds.fake_session_seed),
                "popular_anchor_internal_construction",
                int(target_item),
            )
        )
        policy = AnchorInternalConstructionPolicy(
            anchor_pool=loaded_anchors.anchor_pool,
            topk_ratio=float(config.attack.replacement_topk_ratio),
            rng=rng,
            anchor_assignment_strategy=(
                config.anchor_construction.anchor_assignment_strategy
            ),
        )
        results = [
            policy.apply_with_metadata(session, int(target_item), session_index)
            for session_index, session in enumerate(shared.template_sessions)
        ]
        fake_sessions = [result.session for result in results]
        insertion_slots = [int(result.target_insertion_slot) for result in results]

        _validate_constructed_sessions(
            template_sessions=shared.template_sessions,
            constructed_sessions=fake_sessions,
            results=results,
            target_item=int(target_item),
            anchor_pool=loaded_anchors.anchor_pool,
        )

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Constructed fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Popular-Anchor Internal Construction samples zero-based target "
                "insertion slots from [1, len(session)-1], replaces slot-1 with "
                "a selected popular train_sub anchor, and excludes slot0 and tail target insertion."
            ),
        )
        metadata = build_popular_anchor_construction_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            constructed_sessions=fake_sessions,
            construction_results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            slot_stats_payload=slot_stats_payload,
            loaded_anchors=loaded_anchors,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
            item_counts=shared.stats.item_counts,
        )
        stats_path = target_root / "popular_anchor_internal_construction_stats.json"
        metadata_path = target_root / "popular_anchor_internal_construction_metadata.json"
        save_json(metadata, stats_path)
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "popular_anchor_internal_construction_stats_path": str(stats_path),
                "popular_anchor_internal_construction_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "selected_anchor_pool_hash": loaded_anchors.selected_anchor_pool_hash,
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )


def build_popular_anchor_pool(
    *,
    train_sub_sessions: Sequence[Sequence[int]],
    target_item: int,
    anchor_top_m: int,
) -> LoadedPopularAnchorPool:
    if anchor_top_m < 1:
        raise ValueError("anchor_top_m must be >= 1.")
    counts: Counter[int] = Counter()
    for session in train_sub_sessions:
        for item in session:
            item_id = int(item)
            if item_id == 0 or item_id == int(target_item):
                continue
            counts[item_id] += 1
    if not counts:
        raise ValueError(
            f"No popular train_sub anchors are available for target {int(target_item)}."
        )
    ordered = sorted(counts.items(), key=lambda pair: (-int(pair[1]), int(pair[0])))
    selected = ordered[: int(anchor_top_m)]
    return LoadedPopularAnchorPool(
        target_item=int(target_item),
        anchor_pool=[int(item) for item, _ in selected],
        anchor_rows=[
            {
                "anchor_item": int(item),
                "train_sub_frequency": int(count),
                "train_sub_popularity_rank": int(rank),
            }
            for rank, (item, count) in enumerate(selected, start=1)
        ],
    )


def build_popular_anchor_attack_identity_context(
    config: Config,
    *,
    loaded_anchors_by_target: Mapping[int, LoadedPopularAnchorPool],
) -> dict[str, object]:
    return {
        "popular_anchor_internal_construction": {
            "anchor_source": config.anchor_construction.anchor_source,
            "anchor_top_m": int(config.anchor_construction.anchor_top_m),
            "anchor_assignment_strategy": (
                config.anchor_construction.anchor_assignment_strategy
            ),
            "targets": {
                str(int(target_item)): {
                    "selected_anchor_pool": [
                        int(item) for item in loaded.anchor_pool
                    ],
                    "selected_anchor_pool_hash": loaded.selected_anchor_pool_hash,
                }
                for target_item, loaded in sorted(loaded_anchors_by_target.items())
            },
        }
    }


def build_popular_anchor_construction_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    constructed_sessions: Sequence[Sequence[int]],
    construction_results: Sequence[AnchorInternalConstructionResult],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    loaded_anchors: LoadedPopularAnchorPool,
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    item_counts: Mapping[int, int] | None = None,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(constructed_sessions):
        raise ValueError("template_sessions and constructed_sessions must align.")
    if len(template_sessions) != len(construction_results):
        raise ValueError("construction_results must align with template_sessions.")
    overall_stats = slot_stats_payload.get("overall", {})
    if not isinstance(overall_stats, dict):
        raise ValueError("slot_stats_payload must contain an overall object.")

    anchor_items = [int(result.anchor_item) for result in construction_results]
    replaced_items = [
        int(result.original_replaced_item) for result in construction_results
    ]
    right_items = [int(result.right_item) for result in construction_results]
    anchor_right_pairs = [
        (int(result.anchor_item), int(result.right_item))
        for result in construction_results
    ]
    target_counts = [
        int(result.target_occurrence_count_after_construction)
        for result in construction_results
    ]
    length_deltas = [
        int(result.final_length) - int(result.original_length)
        for result in construction_results
    ]
    anchor_usage_counts = Counter(anchor_items)
    anchor_usage_ratios = _counter_ratios(anchor_usage_counts)
    popularity_ranks = _item_popularity_ranks(item_counts)
    shared_key = shared_attack_artifact_key(
        config,
        run_type=POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    )
    return {
        "run_type": POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        "operation": "anchor_construction_internal_insertion",
        "diagnostic_purpose": "active popular-anchor construction",
        "baseline_to_compare": [
            RANDOM_NZ_RUN_TYPE,
            INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        ],
        "anchor_source": config.anchor_construction.anchor_source,
        "target_item": int(target_item),
        "fake_session_count": int(len(constructed_sessions)),
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
        "vulnerable_anchor_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        ),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "original_length_distribution": _length_distribution(template_sessions),
        "final_length_distribution": _length_distribution(constructed_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "length_shift_summary": _numeric_summary(length_deltas),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in target_counts)
        ),
        "every_inserted_target_has_left_neighbor": bool(
            all(result.target_insertion_slot > 0 for result in construction_results)
        ),
        "every_inserted_target_has_right_neighbor": bool(
            all(
                result.target_insertion_slot < result.original_length
                for result in construction_results
            )
        ),
        "tail_slot_excluded": True,
        "slot0_target_insertion_excluded": True,
        "anchor_top_m": int(config.anchor_construction.anchor_top_m),
        "anchor_assignment_strategy": (
            config.anchor_construction.anchor_assignment_strategy
        ),
        "selected_anchor_pool": [int(item) for item in loaded_anchors.anchor_pool],
        "selected_anchor_pool_size": int(len(loaded_anchors.anchor_pool)),
        "selected_anchor_pool_hash": loaded_anchors.selected_anchor_pool_hash,
        "selected_anchor_table": loaded_anchors.anchor_rows,
        "anchor_usage_counts": _stringify_counts(anchor_usage_counts),
        "anchor_usage_ratios": {
            str(item): float(ratio)
            for item, ratio in sorted(anchor_usage_ratios.items())
        },
        "anchor_usage_entropy": _entropy(anchor_usage_counts),
        "max_anchor_usage_ratio": (
            max(anchor_usage_ratios.values()) if anchor_usage_ratios else 0.0
        ),
        "min_anchor_usage_ratio": (
            min(anchor_usage_ratios.values()) if anchor_usage_ratios else 0.0
        ),
        "unique_anchor_used_count": int(len(anchor_usage_counts)),
        "anchor_replacement_noop_count": int(
            sum(1 for result in construction_results if result.was_anchor_replacement_noop)
        ),
        "anchor_already_in_original_session_count": int(
            sum(
                1
                for result in construction_results
                if result.anchor_already_in_original_session
            )
        ),
        "anchor_train_frequency_summary": _train_frequency_summary(
            anchor_items,
            item_counts,
        ),
        "anchor_popularity_rank_summary": _rank_summary(anchor_items, popularity_ranks),
        "replaced_item_train_frequency_summary": _train_frequency_summary(
            replaced_items,
            item_counts,
        ),
        "replaced_item_popularity_rank_summary": _rank_summary(
            replaced_items,
            popularity_ranks,
        ),
        "anchor_minus_replaced_train_frequency_summary": (
            _train_frequency_delta_summary(anchor_items, replaced_items, item_counts)
        ),
        "anchor_minus_replaced_popularity_rank_summary": (
            _popularity_rank_delta_summary(anchor_items, replaced_items, popularity_ranks)
        ),
        "target_insertion_slot_counts": dict(overall_stats.get("slot_counts", {})),
        "target_insertion_slot_group_counts": dict(
            overall_stats.get("slot_group_counts", {})
        ),
        "target_insertion_slot_ratios": dict(overall_stats.get("slot_ratios", {})),
        "tail_slot_count": int(overall_stats.get("tail_slot_count", 0)),
        "tail_slot_ratio": float(overall_stats.get("tail_slot_ratio", 0.0)),
        "anchor_replace_position_counts": _stringify_counts(
            Counter(int(result.anchor_replace_position) for result in construction_results)
        ),
        "anchor_replace_position_group_counts": _anchor_replace_position_group_counts(
            construction_results
        ),
        "anchor_replace_pos0_count": int(
            sum(1 for result in construction_results if result.anchor_replace_position == 0)
        ),
        "anchor_replace_pos0_ratio": (
            float(
                sum(
                    1
                    for result in construction_results
                    if result.anchor_replace_position == 0
                )
            )
            / float(len(construction_results))
            if construction_results
            else 0.0
        ),
        "right_item_count_summary": _counter_summary(right_items),
        "unique_right_item_count": int(len(set(right_items))),
        "unique_anchor_right_pair_count": int(len(set(anchor_right_pairs))),
        "anchor_right_pair_usage_entropy": _entropy(Counter(anchor_right_pairs)),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in construction_results if result.pre_existing_target_count > 0)
        ),
        "target_occurrence_count_after_construction": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "internal_insertion_slot_topk_ratio": float(
            config.attack.replacement_topk_ratio
        ),
        "topk_ratio_field_name": "attack.replacement_topk_ratio",
        "replacement_topk_ratio_used_for_replacement": False,
        "previews": _construction_previews(
            template_sessions=template_sessions,
            constructed_sessions=constructed_sessions,
            construction_results=construction_results,
            limit=preview_limit,
        ),
    }


def _anchor_replace_position_group_counts(
    results: Sequence[AnchorInternalConstructionResult],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        position = int(result.anchor_replace_position)
        if position == 0:
            counts["pos0"] += 1
        elif position == 1:
            counts["pos1"] += 1
        elif position == 2:
            counts["pos2"] += 1
        elif position == 3:
            counts["pos3"] += 1
        elif 4 <= position <= 5:
            counts["pos4_5"] += 1
        else:
            counts["pos6_plus"] += 1
    return {
        key: int(counts.get(key, 0))
        for key in ("pos0", "pos1", "pos2", "pos3", "pos4_5", "pos6_plus")
    }


def _popularity_rank_delta_summary(
    anchor_items: Sequence[int],
    replaced_items: Sequence[int],
    popularity_ranks: Mapping[int, int] | None,
) -> dict[str, float] | None:
    if popularity_ranks is None:
        return None
    deltas = [
        int(popularity_ranks[int(anchor)]) - int(popularity_ranks[int(replaced)])
        for anchor, replaced in zip(anchor_items, replaced_items)
        if int(anchor) in popularity_ranks and int(replaced) in popularity_ranks
    ]
    if not deltas:
        return None
    return _numeric_summary(deltas)


def _validate_config_for_runner(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError(
            "Popular-Anchor Internal Construction expects data.poison_train_only to be true."
        )
    if config.targets.mode != "explicit_list" or not config.targets.explicit_list:
        raise ValueError(
            "Popular-Anchor Internal Construction requires targets.mode == "
            "explicit_list and a non-empty targets.explicit_list."
        )
    if not config.anchor_construction.enabled:
        raise ValueError(
            "Popular-Anchor Internal Construction requires "
            "anchor_construction.enabled == true."
        )
    if config.anchor_construction.anchor_source != "popular_train_items":
        raise ValueError(
            "Popular-Anchor Internal Construction requires "
            "anchor_construction.anchor_source == 'popular_train_items'."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_POPULAR_ANCHOR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_popular_anchor_internal_construction(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
