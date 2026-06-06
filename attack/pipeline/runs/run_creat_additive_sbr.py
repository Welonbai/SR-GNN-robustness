from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import (
    Config,
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    load_config,
)
from attack.common.artifact_io import load_json
from attack.common.paths import CREAT_ADDITIVE_SBR_RUN_TYPE, target_dir
from attack.creat import METHOD_LABEL
from attack.creat.candidates import (
    filter_effective_templates,
    filter_templates_with_valid_candidates,
    position_distribution,
    sessions_sha1,
    target_exposure_counts,
)
from attack.creat.poison_builder import build_creat_poisoned_sessions
from attack.creat.srgnn_adapter import SRGNNRepresentationAdapter
from attack.creat.trainer import CreatAdditiveSBRTrainer
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts


DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
)


def run_creat_additive_sbr(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_creat_run_config(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    if shared.poison_runner is None:
        raise RuntimeError("CREAT-Additive-SBR requires an SR-GNN poison runner.")

    base_template_sessions = [list(session) for session in shared.template_sessions]
    base_template_hash = sessions_sha1(base_template_sessions)
    shared_template_sessions_sha1 = _load_shared_template_sessions_sha1(
        shared.shared_paths,
        expected_hash=base_template_hash,
    )
    effective_templates, template_counts = filter_effective_templates(base_template_sessions)
    effective_template_hash = sessions_sha1(effective_templates)
    if not effective_templates:
        raise ValueError("CREAT-Additive-SBR has no effective templates after len < 2 filtering.")

    context = RunContext.from_shared(shared)
    creat_config = config.attack.creat_additive_sbr
    if creat_config is None:
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.")

    adapter = SRGNNRepresentationAdapter(shared.poison_runner)
    max_item_id = int(adapter.max_item_id)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        creat_attack_started_at = _timestamp_utc()
        creat_attack_started_monotonic = time.monotonic()
        print(
            "[CREAT-Additive-SBR] "
            f"target={int(target_item)} attack construction started at "
            f"{creat_attack_started_at}",
            flush=True,
        )
        target_root = target_dir(
            config,
            target_item,
            run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        target_templates, target_filter_counts = filter_templates_with_valid_candidates(
            effective_templates,
            target_item=int(target_item),
            topk_ratio=float(config.attack.replacement_topk_ratio),
            nonzero_when_possible=bool(creat_config.nonzero_when_possible),
        )
        if not target_templates:
            raise ValueError(
                "CREAT-Additive-SBR has no templates with a valid replacement "
                f"candidate for target_item={int(target_item)}."
            )
        target_effective_template_hash = sessions_sha1(target_templates)
        pre_existing_exposure = target_exposure_counts(
            target_templates,
            target_item=int(target_item),
        )

        trainer = CreatAdditiveSBRTrainer(
            adapter=adapter,
            config=creat_config,
            replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
            seed=int(config.seeds.position_opt_seed),
        )
        train_result = trainer.train(
            target_item=int(target_item),
            template_sessions=target_templates,
        )
        poison_build_started_at = _timestamp_utc()
        poison_build_started_monotonic = time.monotonic()
        build_result = build_creat_poisoned_sessions(
            adapter=adapter,
            masker=train_result.masker,
            target_item=int(target_item),
            template_sessions=target_templates,
            replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
            nonzero_when_possible=bool(creat_config.nonzero_when_possible),
            max_item_id=int(max_item_id),
        )
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            build_result.poisoned_sessions,
        )
        poison_build_completed_at = _timestamp_utc()
        poison_build_elapsed_seconds = time.monotonic() - poison_build_started_monotonic
        post_poison_exposure = target_exposure_counts(
            build_result.poisoned_sessions,
            target_item=int(target_item),
        )
        original_template_count = int(template_counts["original_template_count"])
        effective_poisoned_count = int(
            build_result.metadata["effective_poisoned_copied_session_count"]
        )

        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=target_templates,
            positions=build_result.selected_positions,
            run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
            target_item=int(target_item),
        )
        train_history_path = target_root / "creat_additive_sbr_train_history.json"
        _save_json(train_result.history, train_history_path)
        selected_positions_path = target_root / "creat_additive_sbr_selected_positions.json"
        _save_json(
            [int(position) for position in build_result.selected_positions],
            selected_positions_path,
        )
        creat_attack_completed_at = _timestamp_utc()
        creat_attack_elapsed_seconds = time.monotonic() - creat_attack_started_monotonic
        print(
            "[CREAT-Additive-SBR] "
            f"target={int(target_item)} attack construction completed at "
            f"{creat_attack_completed_at}; "
            f"elapsed_seconds={round(float(creat_attack_elapsed_seconds), 3)}",
            flush=True,
        )
        metadata = {
            "run_type": CREAT_ADDITIVE_SBR_RUN_TYPE,
            "method_label": METHOD_LABEL,
            "target_item": int(target_item),
            "creat_attack_started_at": creat_attack_started_at,
            "creat_attack_completed_at": creat_attack_completed_at,
            "creat_attack_elapsed_seconds": round(
                float(creat_attack_elapsed_seconds),
                3,
            ),
            "creat_masker_train_started_at": train_result.history.get("started_at"),
            "creat_masker_train_completed_at": train_result.history.get("completed_at"),
            "creat_masker_train_elapsed_seconds": train_result.history.get(
                "elapsed_seconds"
            ),
            "poison_build_started_at": poison_build_started_at,
            "poison_build_completed_at": poison_build_completed_at,
            "poison_build_elapsed_seconds": round(
                float(poison_build_elapsed_seconds),
                3,
            ),
            **template_counts,
            **target_filter_counts,
            "effective_poisoned_copied_session_count": effective_poisoned_count,
            "effective_budget_ratio": (
                0.0
                if original_template_count <= 0
                else float(effective_poisoned_count) / float(original_template_count)
            ),
            "expanded_poisoned_prefix_label_pair_count": int(
                build_result.metadata["expanded_poisoned_prefix_label_pair_count"]
            ),
            "target_label_poisoned_pair_count": int(
                build_result.metadata["target_label_poisoned_pair_count"]
            ),
            "selected_replacement_target_pair_count": int(
                build_result.metadata["selected_replacement_target_pair_count"]
            ),
            "expanded_target_label_pair_count": int(
                build_result.metadata["expanded_target_label_pair_count"]
            ),
            "pre_existing_target_session_count": int(
                pre_existing_exposure["target_session_count"]
            ),
            "pre_existing_target_item_count": int(
                pre_existing_exposure["target_item_count"]
            ),
            "pre_existing_target_label_pair_count": int(
                pre_existing_exposure["target_label_pair_count"]
            ),
            "post_poison_target_label_pair_count": int(
                post_poison_exposure["target_label_pair_count"]
            ),
            "new_target_label_pair_count": int(
                post_poison_exposure["target_label_pair_count"]
                - pre_existing_exposure["target_label_pair_count"]
            ),
            "selected_position_distribution": position_distribution(
                build_result.selected_positions
            ),
            "base_template_hash": base_template_hash,
            "shared_template_sessions_sha1": shared_template_sessions_sha1,
            "effective_template_hash": effective_template_hash,
            "target_effective_template_hash": target_effective_template_hash,
            "template_source": {
                "type": config.attack.fake_session_source.type,
                "config": config.to_primitive()["attack"]["fake_session_source"],
                "path": str(shared.shared_paths["fake_sessions"]),
                "sampling_seed_source": "seeds.fake_session_seed",
            },
            "attack_reward_mode": creat_config.attack_reward_mode,
            "max_attack_num": int(creat_config.max_attack_num),
            "operation": "replacement",
            "source_semantics": (
                "Original CREAT is profile pollution; CREAT-Additive-SBR copies "
                "clean train sessions, replaces one item in each copy, and appends "
                "the polluted copies without overwriting clean train data."
            ),
            "position_stats_path": str(position_stats_path),
            "creat_additive_sbr_train_history_path": str(train_history_path),
            "creat_additive_sbr_selected_positions_path": str(selected_positions_path),
        }
        metadata_path = target_root / "creat_additive_sbr_metadata.json"
        _save_json(metadata, metadata_path)
        metadata["creat_additive_sbr_metadata_path"] = str(metadata_path)
        return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def _validate_creat_run_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("CREAT-Additive-SBR requires data.poison_train_only == true.")
    if (
        config.attack.fake_session_source.type
        != FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    ):
        raise ValueError(
            "CREAT-Additive-SBR requires "
            "attack.fake_session_source.type == 'train_template_clean_exact_length_matched'."
        )
    if config.attack.creat_additive_sbr is None:
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.")
    if not bool(config.attack.creat_additive_sbr.enabled):
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.enabled == true.")


def _load_shared_template_sessions_sha1(
    shared_paths: dict[str, Path],
    *,
    expected_hash: str,
) -> str | None:
    summary_path = Path(shared_paths["attack_shared_dir"]) / "fake_session_source_summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Malformed fake session source summary: {summary_path}")
    value = summary.get("template_sessions_sha1")
    if value is None:
        return None
    shared_hash = str(value)
    if shared_hash != str(expected_hash):
        raise ValueError(
            "Shared template_sessions_sha1 does not match loaded base templates: "
            f"{shared_hash} != {expected_hash}"
        )
    return shared_hash


def _save_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_creat_additive_sbr(config, config_path=args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH",
    "run_creat_additive_sbr",
]
