from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.carrier_selection import (
    HybridTargetSessionCompatibilityScorer,
    build_targetized_selected_sessions,
    select_carriers,
)
from attack.common.artifact_io import save_json
from attack.common.config import CarrierSelectionConfig, Config, load_config
from attack.common.paths import (
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    target_dir,
)
from attack.common.seed import derive_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import (
    fake_session_count_from_ratio,
    prepare_shared_attack_artifacts,
)
from attack.pipeline.core.position_stats import save_position_stats


DEFAULT_TACS_NZ_CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_tacs_nz_pool3_final1_srgnn_target11103.yaml"
)


def run_target_aware_carrier_selection_nz(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    carrier_selection = _validate_tacs_nz_config(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )

    candidate_pool_count = fake_session_count_from_ratio(
        carrier_selection.candidate_pool_size,
        len(shared.clean_sessions),
    )
    final_count = fake_session_count_from_ratio(
        carrier_selection.final_attack_size,
        len(shared.clean_sessions),
    )
    if len(shared.template_sessions) != candidate_pool_count:
        raise ValueError(
            "TACS-NZ candidate fake-session pool has an unexpected size: "
            f"expected {candidate_pool_count}, got {len(shared.template_sessions)}."
        )

    context = replace(RunContext.from_shared(shared), fake_session_count=final_count)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        scorer = HybridTargetSessionCompatibilityScorer(
            train_sub_sessions=shared.canonical_dataset.train_sub,
            config=carrier_selection,
            poison_runner=shared.poison_runner,
        )
        score_records, scorer_metadata = scorer.score(
            candidate_sessions=shared.template_sessions,
            target_item=int(target_item),
        )
        selection_result = select_carriers(
            candidate_sessions=shared.template_sessions,
            score_records=score_records,
            final_count=final_count,
            target_item=int(target_item),
            use_length_control=bool(carrier_selection.use_length_control),
            length_buckets=carrier_selection.length_buckets,
        )
        position_seed = derive_seed(
            int(config.seeds.fake_session_seed),
            TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
            int(target_item),
            "random_nonzero_ratio1",
        )
        rng = random.Random(position_seed)
        policy = RandomNonzeroWhenPossiblePolicy(
            config.attack.replacement_topk_ratio,
            rng=rng,
        )
        targetized = build_targetized_selected_sessions(
            candidate_sessions=shared.template_sessions,
            selected_indices=selection_result.selected_indices,
            target_item=int(target_item),
            policy=policy,
        )
        if len(targetized.fake_sessions) != final_count:
            raise RuntimeError("TACS-NZ selected fake-session count does not equal final_count.")

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in targetized.fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            targetized.fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)
        selected_candidate_sessions = [
            shared.template_sessions[index]
            for index in targetized.selected_candidate_indices
        ]
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=selected_candidate_sessions,
            positions=targetized.selected_positions,
            run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "TACS-NZ v1 keeps placement as Random-NZ ratio1 to isolate "
                "target-aware carrier selection."
            ),
        )
        targetized_previews = _targetized_previews(
            original_sessions=selected_candidate_sessions,
            targetized_sessions=targetized.fake_sessions,
            selected_indices=targetized.selected_candidate_indices,
            selected_positions=targetized.selected_positions,
            limit=20,
        )
        carrier_selection_metadata = {
            **selection_result.metadata,
            **scorer_metadata,
            "run_type": TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
            "diagnostic": "TACS-NZ v1 target-aware carrier selection with Random-NZ ratio1 placement.",
            "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
            "poison_model_key": poison_model_key(config),
            "poison_model_identity": poison_model_key_payload(config),
            "candidate_pool_size": float(carrier_selection.candidate_pool_size),
            "candidate_pool_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
            "scorer_config": _carrier_selection_config_payload(carrier_selection),
            "targetized_selected_session_previews": targetized_previews,
        }
        carrier_selection_metadata_path = target_root / "carrier_selection_metadata.json"
        save_json(carrier_selection_metadata, carrier_selection_metadata_path)

        position_metadata = {
            "run_type": TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
            "placement_policy": "RandomNonzeroWhenPossiblePolicy",
            "diagnostic_note": (
                "TACS-NZ v1 intentionally keeps placement as Random-NZ ratio1; "
                "candidate_pool_size controls scoring pool size and attack.size/final_attack_size "
                "controls actual injected fake sessions."
            ),
            "target_item": int(target_item),
            "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
            "poison_model_key": poison_model_key(config),
            "candidate_pool_count": int(candidate_pool_count),
            "candidate_pool_size": float(carrier_selection.candidate_pool_size),
            "final_count": int(final_count),
            "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
            "position_seed": int(position_seed),
            "seed_source": {
                "base_fake_session_seed": int(config.seeds.fake_session_seed),
                "derived_components": [
                    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
                    int(target_item),
                    "random_nonzero_ratio1",
                ],
            },
            "selected_candidate_indices": targetized.selected_candidate_indices,
            "selected_positions": targetized.selected_positions,
            "targetized_selected_session_previews": targetized_previews,
        }
        tacs_nz_position_metadata_path = target_root / "tacs_nz_position_metadata.json"
        save_json(position_metadata, tacs_nz_position_metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "carrier_selection_metadata_path": str(carrier_selection_metadata_path),
                "position_stats_path": str(position_stats_path),
                "tacs_nz_position_metadata_path": str(tacs_nz_position_metadata_path),
                "candidate_pool_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
                "poison_model_key": poison_model_key(config),
                "candidate_pool_count": int(candidate_pool_count),
                "final_injected_fake_session_count": int(final_count),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def _validate_tacs_nz_config(config: Config) -> CarrierSelectionConfig:
    if not config.data.poison_train_only:
        raise ValueError("TACS-NZ expects data.poison_train_only to be true.")
    carrier_selection = config.attack.carrier_selection
    if carrier_selection is None:
        raise ValueError("TACS-NZ requires attack.carrier_selection to be configured.")
    if not carrier_selection.enabled:
        raise ValueError("TACS-NZ requires attack.carrier_selection.enabled == true.")
    if abs(float(carrier_selection.final_attack_size) - float(config.attack.size)) > 1e-12:
        raise ValueError(
            "TACS-NZ v1 requires attack.carrier_selection.final_attack_size "
            "to equal attack.size."
        )
    if float(config.attack.replacement_topk_ratio) != 1.0:
        raise ValueError(
            "TACS-NZ v1 requires attack.replacement_topk_ratio == 1.0 because "
            "this diagnostic isolates carrier selection under Random-NZ ratio1 placement."
        )
    return carrier_selection


def _carrier_selection_config_payload(config: CarrierSelectionConfig) -> dict[str, object]:
    return {
        "enabled": bool(config.enabled),
        "candidate_pool_size": float(config.candidate_pool_size),
        "final_attack_size": float(config.final_attack_size),
        "scorer": config.scorer,
        "embedding_weight": float(config.embedding_weight),
        "cooccurrence_weight": float(config.cooccurrence_weight),
        "transition_weight": float(config.transition_weight),
        "use_length_control": bool(config.use_length_control),
        "length_buckets": config.length_buckets,
        "normalize": config.normalize,
    }


def _targetized_previews(
    *,
    original_sessions: list[list[int]],
    targetized_sessions: list[list[int]],
    selected_indices: list[int],
    selected_positions: list[int],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for index, original, targetized, position in zip(
        selected_indices[:limit],
        original_sessions[:limit],
        targetized_sessions[:limit],
        selected_positions[:limit],
    ):
        previews.append(
            {
                "candidate_index": int(index),
                "original_session": [int(item) for item in original],
                "targetized_session": [int(item) for item in targetized],
                "selected_position": int(position),
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_TACS_NZ_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_target_aware_carrier_selection_nz(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
