from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_artifact_paths,
    shared_attack_artifact_key,
    target_dir,
)
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.vulnerable_anchor_internal_construction import (
    LoadedVulnerableAnchorPool,
    VulnerableAnchorInternalConstructionPolicy,
    VulnerableAnchorInternalConstructionResult,
    load_vulnerable_anchor_pool,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.slot_stats import build_slot_stats_payload


DEFAULT_VULNERABLE_ANCHOR_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_vulnerable_anchor_internal_construction_"
    "top20_ratio1_srgnn_target39588_partial5.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_vulnerable_anchor_internal_construction(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_config_for_runner(config)
    loaded_anchors_by_target = _load_anchor_pools_for_config(config)
    attack_identity_context = build_vulnerable_anchor_attack_identity_context(
        config,
        loaded_anchors_by_target=loaded_anchors_by_target,
    )
    _require_existing_shared_fake_sessions(config)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        loaded_anchors = loaded_anchors_by_target[int(target_item)]
        rng = random.Random(
            derive_seed(
                int(config.seeds.fake_session_seed),
                "vulnerable_anchor_internal_construction",
                int(target_item),
            )
        )
        policy = VulnerableAnchorInternalConstructionPolicy(
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
            run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        slot_stats_payload = build_slot_stats_payload(
            sessions=shared.template_sessions,
            insertion_slots=insertion_slots,
            run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Vulnerable-Anchor Internal Construction samples zero-based target "
                "insertion slots from [1, len(session)-1], replaces slot-1 with "
                "the selected vulnerable anchor, and excludes slot0 and tail target insertion."
            ),
        )
        stats_payload = build_vulnerable_anchor_construction_metadata(
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
        stats_path = target_root / "vulnerable_anchor_internal_construction_stats.json"
        metadata_path = target_root / "vulnerable_anchor_internal_construction_metadata.json"
        save_json(stats_payload, stats_path)
        save_json(stats_payload, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "vulnerable_anchor_internal_construction_stats_path": str(stats_path),
                "vulnerable_anchor_internal_construction_metadata_path": str(
                    metadata_path
                ),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "survey_file_path": loaded_anchors.survey_file_path,
                "survey_file_hash": loaded_anchors.survey_file_hash,
                "selected_anchor_pool_hash": loaded_anchors.selected_anchor_pool_hash,
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )


def build_vulnerable_anchor_attack_identity_context(
    config: Config,
    *,
    loaded_anchors_by_target: Mapping[int, LoadedVulnerableAnchorPool],
) -> dict[str, object]:
    return {
        "vulnerable_anchor_internal_construction": {
            "anchor_source": config.anchor_construction.anchor_source,
            "anchor_top_m": int(config.anchor_construction.anchor_top_m),
            "anchor_assignment_strategy": (
                config.anchor_construction.anchor_assignment_strategy
            ),
            "survey_output_dir": config.anchor_construction.survey_output_dir,
            "targets": {
                str(int(target_item)): {
                    "selected_anchor_pool": [
                        int(item) for item in loaded.anchor_pool
                    ],
                    "selected_anchor_pool_hash": loaded.selected_anchor_pool_hash,
                    "survey_file_path": loaded.survey_file_path,
                    "survey_file_hash": loaded.survey_file_hash,
                    "survey_source_format": loaded.source_format,
                }
                for target_item, loaded in sorted(loaded_anchors_by_target.items())
            },
        }
    }


def build_vulnerable_anchor_construction_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    constructed_sessions: Sequence[Sequence[int]],
    construction_results: Sequence[VulnerableAnchorInternalConstructionResult],
    clean_train_sessions: Sequence[Sequence[int]],
    slot_stats_payload: dict[str, object],
    loaded_anchors: LoadedVulnerableAnchorPool,
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
    anchor_usage_count_summary = _counter_summary(anchor_items)
    replaced_item_usage_count_summary = _counter_summary(replaced_items)

    shared_key = shared_attack_artifact_key(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    )
    return {
        "run_type": VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        "operation": "anchor_construction_internal_insertion",
        "diagnostic_purpose": "active vulnerable-anchor construction",
        "baseline_to_compare": [
            RANDOM_NZ_RUN_TYPE,
            INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ],
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
        "random_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
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
        "anchor_source": config.anchor_construction.anchor_source,
        "anchor_top_m": int(config.anchor_construction.anchor_top_m),
        "anchor_assignment_strategy": (
            config.anchor_construction.anchor_assignment_strategy
        ),
        "anchor_pool": [int(item) for item in loaded_anchors.anchor_pool],
        "anchor_pool_size": int(len(loaded_anchors.anchor_pool)),
        "selected_anchor_pool_hash": loaded_anchors.selected_anchor_pool_hash,
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
        "original_replaced_item_count_summary": _counter_summary(replaced_items),
        "original_replaced_item_unique_count": int(len(set(replaced_items))),
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
        "anchor_usage_count_summary": anchor_usage_count_summary,
        "replaced_item_usage_count_summary": replaced_item_usage_count_summary,
        "anchor_item_frequency_summary": anchor_usage_count_summary,
        "replaced_item_frequency_summary": replaced_item_usage_count_summary,
        "anchor_minus_replaced_frequency_summary": _frequency_delta_summary(
            anchor_items,
            replaced_items,
        ),
        "anchor_train_frequency_summary": _train_frequency_summary(
            anchor_items,
            item_counts,
        ),
        "replaced_item_train_frequency_summary": _train_frequency_summary(
            replaced_items,
            item_counts,
        ),
        "anchor_minus_replaced_train_frequency_summary": (
            _train_frequency_delta_summary(anchor_items, replaced_items, item_counts)
        ),
        "anchor_popularity_rank_summary": _rank_summary(anchor_items, popularity_ranks),
        "replaced_item_popularity_rank_summary": _rank_summary(
            replaced_items,
            popularity_ranks,
        ),
        "pre_existing_target_in_template_sessions_count": int(
            sum(1 for result in construction_results if result.pre_existing_target_count > 0)
        ),
        "target_occurrence_count_after_construction": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "survey_file_path": loaded_anchors.survey_file_path,
        "survey_file_hash": loaded_anchors.survey_file_hash,
        "survey_source_format": loaded_anchors.source_format,
        "rank_min": loaded_anchors.rank_min,
        "rank_max": loaded_anchors.rank_max,
        "top_anchors_table": loaded_anchors.top_anchor_rows,
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


def _validate_config_for_runner(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError(
            "Vulnerable-Anchor Internal Construction expects "
            "data.poison_train_only to be true."
        )
    if config.targets.mode != "explicit_list" or not config.targets.explicit_list:
        raise ValueError(
            "Vulnerable-Anchor Internal Construction requires targets.mode == "
            "explicit_list and a non-empty targets.explicit_list."
        )
    if not config.anchor_construction.enabled:
        raise ValueError(
            "Vulnerable-Anchor Internal Construction requires "
            "anchor_construction.enabled == true."
        )


def _load_anchor_pools_for_config(
    config: Config,
) -> dict[int, LoadedVulnerableAnchorPool]:
    return {
        int(target_item): load_vulnerable_anchor_pool(
            survey_output_dir=config.anchor_construction.survey_output_dir,
            target_item=int(target_item),
            anchor_top_m=int(config.anchor_construction.anchor_top_m),
            require_survey_file=bool(config.anchor_construction.require_survey_file),
        )
        for target_item in config.targets.explicit_list
    }


def _require_existing_shared_fake_sessions(config: Config) -> None:
    paths = shared_artifact_paths(
        config,
        run_type=VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    )
    fake_sessions_path = paths["fake_sessions"]
    if not fake_sessions_path.exists():
        raise FileNotFoundError(
            "Vulnerable-Anchor Internal Construction requires existing shared "
            "fake-session templates and will not regenerate them. Missing: "
            f"{fake_sessions_path}"
        )


def _validate_constructed_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    constructed_sessions: Sequence[Sequence[int]],
    results: Sequence[VulnerableAnchorInternalConstructionResult],
    target_item: int,
    anchor_pool: Sequence[int],
) -> None:
    if len(template_sessions) != len(constructed_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(results):
        raise RuntimeError("Result metadata count does not equal template count.")
    anchor_set = {int(item) for item in anchor_pool}
    for original, constructed, result in zip(
        template_sessions,
        constructed_sessions,
        results,
    ):
        original_list = [int(item) for item in original]
        constructed_list = [int(item) for item in constructed]
        slot = int(result.target_insertion_slot)
        if len(original_list) < 2:
            raise RuntimeError(
                "Vulnerable-Anchor Internal Construction requires template length >= 2."
            )
        if slot < 1 or slot > len(original_list) - 1:
            raise RuntimeError(
                "Vulnerable-Anchor Internal Construction selected slot0 or tail slot."
            )
        if int(target_item) not in set(constructed_list):
            raise RuntimeError("Constructed session is missing target item.")
        if len(constructed_list) != len(original_list) + 1:
            raise RuntimeError("Constructed session length delta is not +1.")
        if int(result.anchor_item) not in anchor_set:
            raise RuntimeError("Constructed session used an anchor outside the pool.")
        expected = (
            original_list[: slot - 1]
            + [int(result.anchor_item), int(target_item)]
            + original_list[slot:]
        )
        if constructed_list != expected:
            raise RuntimeError(
                "Constructed session did not preserve order except replacement and insertion."
            )
        if constructed_list[slot - 1] != int(result.anchor_item):
            raise RuntimeError("Target left neighbor is not the selected anchor.")
        if constructed_list[slot + 1] != int(result.right_item):
            raise RuntimeError("Target right neighbor metadata is invalid.")


def _existing_path_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    return str(path_obj) if path_obj.exists() else None


def _length_distribution(sessions: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: Counter[int] = Counter(int(len(session)) for session in sessions)
    return {f"len{length}": int(count) for length, count in sorted(counts.items())}


def _numeric_summary(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [float(value) for value in values]
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


def _counter_ratios(counter: Counter[Any]) -> dict[Any, float]:
    total = int(sum(counter.values()))
    if total <= 0:
        return {}
    return {key: float(count) / float(total) for key, count in counter.items()}


def _stringify_counts(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _entropy(counter: Counter[Any]) -> float:
    total = float(sum(counter.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        probability = float(count) / total
        if probability > 0.0:
            entropy -= probability * math.log(probability, 2)
    return float(entropy)


def _frequency_delta_summary(
    anchor_items: Sequence[int],
    replaced_items: Sequence[int],
    *,
    limit: int = 10,
) -> dict[str, object]:
    anchor_counts = Counter(int(item) for item in anchor_items)
    replaced_counts = Counter(int(item) for item in replaced_items)
    deltas = {
        int(item): int(anchor_counts.get(item, 0) - replaced_counts.get(item, 0))
        for item in set(anchor_counts) | set(replaced_counts)
    }
    positives = sorted(
        ((item, delta) for item, delta in deltas.items() if delta > 0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    negatives = sorted(
        ((item, delta) for item, delta in deltas.items() if delta < 0),
        key=lambda pair: (pair[1], pair[0]),
    )
    return {
        "nonzero_item_count": int(sum(1 for delta in deltas.values() if delta != 0)),
        "total_positive_delta": int(sum(delta for delta in deltas.values() if delta > 0)),
        "total_negative_delta_abs": int(
            -sum(delta for delta in deltas.values() if delta < 0)
        ),
        "max_positive_delta": int(positives[0][1]) if positives else 0,
        "max_negative_delta": int(negatives[0][1]) if negatives else 0,
        "top_positive": [
            {"item": int(item), "delta": int(delta)}
            for item, delta in positives[:limit]
        ],
        "top_negative": [
            {"item": int(item), "delta": int(delta)}
            for item, delta in negatives[:limit]
        ],
    }


def _train_frequency_summary(
    items: Sequence[int],
    item_counts: Mapping[int, int] | None,
) -> dict[str, float] | None:
    if not item_counts:
        return None
    frequencies = [
        int(item_counts.get(int(item), 0))
        for item in items
    ]
    if not frequencies:
        return None
    summary = _numeric_summary(frequencies)
    summary["zero_frequency_count"] = float(
        sum(1 for frequency in frequencies if int(frequency) == 0)
    )
    return summary


def _train_frequency_delta_summary(
    anchor_items: Sequence[int],
    replaced_items: Sequence[int],
    item_counts: Mapping[int, int] | None,
) -> dict[str, float] | None:
    if not item_counts:
        return None
    deltas = [
        int(item_counts.get(int(anchor), 0)) - int(item_counts.get(int(replaced), 0))
        for anchor, replaced in zip(anchor_items, replaced_items)
    ]
    if not deltas:
        return None
    return _numeric_summary(deltas)


def _item_popularity_ranks(item_counts: Mapping[int, int] | None) -> dict[int, int] | None:
    if not item_counts:
        return None
    ordered = sorted(
        ((int(item), int(count)) for item, count in item_counts.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return {item: index for index, (item, _) in enumerate(ordered, start=1)}


def _rank_summary(
    items: Sequence[int],
    popularity_ranks: Mapping[int, int] | None,
) -> dict[str, float] | None:
    if popularity_ranks is None:
        return None
    ranks = [
        int(popularity_ranks[int(item)])
        for item in items
        if int(item) in popularity_ranks
    ]
    if not ranks:
        return None
    return _numeric_summary(ranks)


def _construction_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    constructed_sessions: Sequence[Sequence[int]],
    construction_results: Sequence[VulnerableAnchorInternalConstructionResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, constructed, result in zip(
        template_sessions[:limit],
        constructed_sessions[:limit],
        construction_results[:limit],
    ):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "constructed_session": [int(item) for item in constructed],
                "anchor_item": int(result.anchor_item),
                "target_item": int(result.target_item),
                "target_insertion_slot": int(result.target_insertion_slot),
                "anchor_replace_position": int(result.anchor_replace_position),
                "original_replaced_item": int(result.original_replaced_item),
                "right_item": int(result.right_item),
                "was_anchor_replacement_noop": bool(result.was_anchor_replacement_noop),
                "anchor_already_in_original_session": bool(
                    result.anchor_already_in_original_session
                ),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_VULNERABLE_ANCHOR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_vulnerable_anchor_internal_construction(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
