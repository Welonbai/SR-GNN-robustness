from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.carrier_selection import (
    CoverageAwareLocalPositionScorer,
    build_targetized_selected_sessions_with_fixed_positions,
    build_vulnerable_validation_prefix_bank,
    select_local_position_carriers,
)
from attack.carrier_selection.local_position_scorer import INDEX_BASE
from attack.common.artifact_io import save_json
from attack.common.config import (
    COVERAGE_AWARE_LOCAL_POSITION_SCORER,
    CarrierSelectionConfig,
    Config,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE,
    load_config,
)
from attack.common.paths import (
    TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
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


DEFAULT_COVERAGE_LOCAL_POSITION_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_coverage_local_position_pool3_final1_srgnn_target11103_partial4.yaml"
)


def run_target_aware_coverage_local_position(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    carrier_selection = _validate_coverage_local_position_config(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    if shared.poison_runner is None:
        raise RuntimeError("Coverage-LocalPosition requires an initialized poison runner.")

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
            "Coverage-LocalPosition candidate fake-session pool has an unexpected size: "
            f"expected {candidate_pool_count}, got {len(shared.template_sessions)}."
        )

    context = replace(RunContext.from_shared(shared), fake_session_count=final_count)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        target = int(target_item)
        prefix_bank = build_vulnerable_validation_prefix_bank(
            canonical_dataset=shared.canonical_dataset,
            target_item=target,
            config=carrier_selection,
            poison_runner=shared.poison_runner,
        )
        scorer = CoverageAwareLocalPositionScorer(
            config=carrier_selection,
            prefix_bank=prefix_bank,
            poison_runner=shared.poison_runner,
        )
        session_records, scorer_metadata = scorer.score(
            candidate_sessions=shared.template_sessions,
            target_item=target,
        )
        debug_position_records = scorer_metadata.pop("_debug_all_position_records", None)
        selection_result = select_local_position_carriers(
            candidate_sessions=shared.template_sessions,
            session_records=session_records,
            final_count=final_count,
            target_item=target,
            use_length_control=bool(carrier_selection.use_length_control),
            length_buckets=carrier_selection.length_buckets,
        )
        records_by_index = {int(record.index): record for record in session_records}
        selected_records = [
            records_by_index[int(index)] for index in selection_result.selected_indices
        ]
        targetized = build_targetized_selected_sessions_with_fixed_positions(
            candidate_sessions=shared.template_sessions,
            selected_records=selected_records,
            target_item=target,
        )
        if len(targetized.fake_sessions) != final_count:
            raise RuntimeError(
                "Coverage-LocalPosition selected fake-session count does not equal final_count."
            )
        if any(target not in {int(item) for item in session} for session in targetized.fake_sessions):
            raise RuntimeError(
                "Coverage-LocalPosition injected fake session is missing target item."
            )

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
            target,
            run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)
        prefix_bank_path = target_root / "coverage_prefix_bank.json"
        prefix_bank_metadata = {
            **prefix_bank.metadata,
            "prefix_bank_path": str(prefix_bank_path),
        }
        save_json(
            {
                "metadata": prefix_bank_metadata,
                "records": [record.to_metadata() for record in prefix_bank.records],
            },
            prefix_bank_path,
        )

        selected_candidate_sessions = [
            shared.template_sessions[index]
            for index in targetized.selected_candidate_indices
        ]
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=selected_candidate_sessions,
            positions=targetized.selected_positions,
            run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
            target_item=target,
            note=(
                "Coverage-LocalPosition uses deterministic best coverage-aware "
                "nonzero replacement positions; position indices are zero-based."
            ),
        )

        previews = _selected_previews(
            selected_records=selected_records,
            targetized_sessions=targetized.fake_sessions,
            selected_candidate_indices=targetized.selected_candidate_indices,
            limit=20,
        )
        selected_best_positions = [
            int(record.best_position)
            for record in selected_records
            if record.best_position is not None
        ]
        debug_position_records_path: Path | None = None
        if debug_position_records is not None:
            debug_position_records_path = target_root / "coverage_position_records_debug.json"
            save_json(
                {
                    "run_type": TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
                    "target_item": target,
                    "position_records": debug_position_records,
                },
                debug_position_records_path,
            )

        local_metadata: dict[str, object] = {
            **selection_result.metadata,
            **scorer_metadata,
            "run_type": TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
            "diagnostic": (
                "Coverage-LocalPosition scores deterministic local replacement "
                "positions by coverage of vulnerable validation prefixes."
            ),
            "index_base": INDEX_BASE,
            "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
            "poison_model_key": poison_model_key(config),
            "poison_model_identity": poison_model_key_payload(config),
            "candidate_pool_size": float(carrier_selection.candidate_pool_size),
            "candidate_pool_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
            "candidate_pool_count": int(candidate_pool_count),
            "pool_count": int(candidate_pool_count),
            "final_count": int(final_count),
            "final_injected_count": int(len(targetized.fake_sessions)),
            "fake_session_count": int(final_count),
            "prefix_bank_metadata": prefix_bank_metadata,
            "scorer_config": _carrier_selection_config_payload(carrier_selection),
            "selected_candidate_indices": list(targetized.selected_candidate_indices),
            "selected_best_positions": selected_best_positions,
            "selected_best_position_labels": [
                _position_label(pos) for pos in selected_best_positions
            ],
            "valid_position_count_summary": _valid_position_count_summary(
                selection_metadata=selection_result.metadata,
                scorer_metadata=scorer_metadata,
            ),
            "position_distribution": _position_distribution(selected_best_positions),
            "top_20_selected_previews": previews,
            "targetized_selected_session_previews": previews,
        }
        if debug_position_records_path is not None:
            local_metadata["debug_position_records_path"] = str(debug_position_records_path)

        coverage_metadata_path = target_root / "coverage_local_position_metadata.json"
        save_json(local_metadata, coverage_metadata_path)

        placement_metadata = {
            "run_type": TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
            "placement_policy": "best_local_position",
            "operation": "replacement",
            "candidate_positions": "nonzero",
            "index_base": INDEX_BASE,
            "diagnostic_note": (
                "This diagnostic does not sample Random-NZ positions; it injects "
                "only selected candidates at their stored best coverage-aware "
                "replacement position."
            ),
            "target_item": target,
            "candidate_pool_count": int(candidate_pool_count),
            "candidate_pool_size": float(carrier_selection.candidate_pool_size),
            "final_count": int(final_count),
            "final_injected_count": int(len(targetized.fake_sessions)),
            "fake_session_count": int(final_count),
            "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
            "poison_model_key": poison_model_key(config),
            "candidate_pool_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
            "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
            "prefix_bank_metadata": prefix_bank_metadata,
            "selected_candidate_indices": list(targetized.selected_candidate_indices),
            "selected_positions": list(targetized.selected_positions),
            "selected_position_labels": [
                _position_label(pos) for pos in targetized.selected_positions
            ],
            "position_distribution": _position_distribution(targetized.selected_positions),
            "targetized_selected_session_previews": previews,
        }
        tacs_metadata_path = target_root / "tacs_coverage_local_position_metadata.json"
        save_json(placement_metadata, tacs_metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "coverage_local_position_metadata_path": str(coverage_metadata_path),
                "tacs_coverage_local_position_metadata_path": str(tacs_metadata_path),
                "coverage_prefix_bank_path": str(prefix_bank_path),
                "position_stats_path": str(position_stats_path),
                "candidate_pool_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": str(shared.shared_paths["poison_model"]),
                "poison_model_key": poison_model_key(config),
                "candidate_pool_count": int(candidate_pool_count),
                "final_injected_fake_session_count": int(final_count),
                "run_type": TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def _validate_coverage_local_position_config(config: Config) -> CarrierSelectionConfig:
    if not config.data.poison_train_only:
        raise ValueError("Coverage-LocalPosition expects data.poison_train_only to be true.")
    carrier_selection = config.attack.carrier_selection
    if carrier_selection is None:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection to be configured."
        )
    if not carrier_selection.enabled:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.enabled == true."
        )
    if abs(float(carrier_selection.final_attack_size) - float(config.attack.size)) > 1e-12:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.final_attack_size "
            "to equal attack.size."
        )
    if float(config.attack.replacement_topk_ratio) != 1.0:
        raise ValueError(
            "Coverage-LocalPosition requires attack.replacement_topk_ratio == 1.0."
        )
    if carrier_selection.scorer != COVERAGE_AWARE_LOCAL_POSITION_SCORER:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.scorer == "
            f"{COVERAGE_AWARE_LOCAL_POSITION_SCORER!r}."
        )
    if carrier_selection.placement_mode != TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.placement_mode == "
            f"{TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE!r}."
        )
    if carrier_selection.operation != TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.operation == "
            f"{TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION!r}."
        )
    if carrier_selection.candidate_positions != TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS:
        raise ValueError(
            "Coverage-LocalPosition requires attack.carrier_selection.candidate_positions == "
            f"{TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS!r}."
        )
    return carrier_selection


def _carrier_selection_config_payload(config: CarrierSelectionConfig) -> dict[str, object]:
    return {
        "enabled": bool(config.enabled),
        "candidate_pool_size": float(config.candidate_pool_size),
        "final_attack_size": float(config.final_attack_size),
        "scorer": config.scorer,
        "placement_mode": config.placement_mode,
        "operation": config.operation,
        "candidate_positions": config.candidate_positions,
        "coverage_prefix_source": config.coverage_prefix_source,
        "vulnerable_rank_min": int(config.vulnerable_rank_min),
        "vulnerable_rank_max": int(config.vulnerable_rank_max),
        "max_vulnerable_prefixes": int(config.max_vulnerable_prefixes),
        "prefix_representation": config.prefix_representation,
        "candidate_representation": config.candidate_representation,
        "top_m_coverage": int(config.top_m_coverage),
        "rank_weighting": config.rank_weighting,
        "coverage_similarity": config.coverage_similarity,
        "use_length_control": bool(config.use_length_control),
        "length_buckets": config.length_buckets,
        "normalize": config.normalize,
        "debug_save_all_position_records": bool(config.debug_save_all_position_records),
    }


def _selected_previews(
    *,
    selected_records: Sequence[Any],
    targetized_sessions: Sequence[Sequence[int]],
    selected_candidate_indices: Sequence[int],
    limit: int,
) -> list[dict[str, object]]:
    targetized_by_index = {
        int(index): [int(item) for item in session]
        for index, session in zip(selected_candidate_indices, targetized_sessions)
    }
    ranked_records = sorted(
        selected_records,
        key=lambda record: (-float(record.best_position_score), int(record.index)),
    )
    previews: list[dict[str, object]] = []
    for record in ranked_records[: int(limit)]:
        if record.best_position_record is None:
            continue
        previews.append(
            record.best_position_record.to_preview(
                targetized_session=targetized_by_index.get(int(record.index))
            )
        )
    return previews


def _position_label(position: int) -> str:
    return f"pos{int(position)}"


def _position_distribution(positions: Sequence[int]) -> dict[str, int]:
    buckets = {
        "pos1": 0,
        "pos2": 0,
        "pos3": 0,
        "pos4_5": 0,
        "pos6_plus": 0,
    }
    for position in positions:
        pos = int(position)
        if pos <= 1:
            buckets["pos1"] += 1
        elif pos == 2:
            buckets["pos2"] += 1
        elif pos == 3:
            buckets["pos3"] += 1
        elif pos in (4, 5):
            buckets["pos4_5"] += 1
        else:
            buckets["pos6_plus"] += 1
    return buckets


def _valid_position_count_summary(
    *,
    selection_metadata: dict[str, object],
    scorer_metadata: dict[str, object],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    scorer_summary = scorer_metadata.get("valid_position_count_summary")
    selection_summary = selection_metadata.get("valid_position_count_summary")
    if isinstance(scorer_summary, dict):
        summary.update(scorer_summary)
    if isinstance(selection_summary, dict):
        summary.update(selection_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run coverage-aware local-position target-aware carrier attack."
    )
    parser.add_argument("--config", default=DEFAULT_COVERAGE_LOCAL_POSITION_CONFIG_PATH)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    summary = run_target_aware_coverage_local_position(
        config,
        config_path=config_path,
    )
    print(summary)


if __name__ == "__main__":
    main()
