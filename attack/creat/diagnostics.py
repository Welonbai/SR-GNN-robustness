from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def position_collapse_summary(positions: Sequence[int]) -> dict[str, object]:
    counts = Counter(int(position) for position in positions)
    total = int(sum(counts.values()))
    if total <= 0:
        return {
            "position_entropy": 0.0,
            "position_top1_index": None,
            "position_top1_count": 0,
            "position_top1_ratio": 0.0,
            "is_position_collapsed": False,
        }
    top1_index, top1_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    entropy = -sum(
        (float(count) / float(total)) * math.log(float(count) / float(total))
        for count in counts.values()
    )
    top1_ratio = float(top1_count) / float(total)
    return {
        "position_entropy": float(entropy),
        "position_top1_index": int(top1_index),
        "position_top1_count": int(top1_count),
        "position_top1_ratio": float(top1_ratio),
        "is_position_collapsed": bool(top1_ratio >= 0.9),
    }


def creat_fidelity_metadata(*, variant: str, dpp_reward_weight: float) -> dict[str, object]:
    if str(variant) == "v1":
        components = {
            "two_stage_training": "not_implemented",
            "pattern_inversion": "not_implemented",
            "pattern_diversity_dpp": "not_implemented",
            "global_distribution_consistency": "approximated_representation_l2",
            "local_distribution_consistency": "approximated_neighbor_embedding_compatibility",
            "unbalanced_co_optimal_transport": "not_implemented",
            "dynamic_barrier": "not_implemented",
            "constrained_group_relative_replay": "not_implemented",
        }
    else:
        components = {
            "two_stage_training": "implemented",
            "pattern_inversion": "approximated_prefix_suffix_srgnn",
            "pattern_diversity_dpp": (
                "implemented_enabled"
                if float(dpp_reward_weight) > 0.0
                else "implemented_disabled"
            ),
            "global_distribution_consistency": "approximated_representation_l2",
            "local_distribution_consistency": "approximated_full_affected_kgram_l2",
            "unbalanced_co_optimal_transport": "not_implemented",
            "dynamic_barrier": "not_implemented",
            "constrained_group_relative_replay": "not_implemented",
        }
    return {
        "variant": str(variant),
        "original_creat_components": components,
        "sbr_migration": {
            "profile_overwrite": "replaced_by_additive_copied_session_injection",
            "backbone": "frozen_srgnn_surrogate",
            "pollution_operation": "single_target_replacement",
            "clean_training_sessions_overwritten": False,
        },
    }


__all__ = ["creat_fidelity_metadata", "position_collapse_summary"]
