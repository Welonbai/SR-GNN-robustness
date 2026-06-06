from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import time
from datetime import datetime, timezone

from attack.common.artifact_io import load_json, save_fake_sessions, save_json
from attack.common.config import Config
from attack.common.paths import target_dir
from attack.creat import METHOD_LABEL
from attack.creat.candidates import (
    filter_effective_templates,
    filter_templates_with_valid_candidates,
    position_distribution,
    sessions_sha1,
    target_exposure_counts,
)
from attack.creat.diagnostics import creat_fidelity_metadata, position_collapse_summary
from attack.creat.poison_builder import build_creat_poisoned_sessions
from attack.creat.srgnn_adapter import SRGNNRepresentationAdapter
from attack.creat.trainer import CreatAdditiveSBRTrainer
from attack.pipeline.core.position_stats import save_position_stats


@dataclass(frozen=True)
class CreatPreparedArtifacts:
    adapter: object
    base_template_sessions: list[list[int]]
    effective_templates: list[list[int]]
    template_counts: dict[str, int]
    base_template_hash: str
    effective_template_hash: str
    shared_template_sessions_sha1: str | None


@dataclass(frozen=True)
class CreatTargetGenerationResult:
    poisoned_sessions: list[list[int]]
    metadata: dict[str, object]


def prepare_creat_artifacts(shared, *, adapter_class=SRGNNRepresentationAdapter) -> CreatPreparedArtifacts:
    if shared.poison_runner is None:
        raise RuntimeError("CREAT-Additive-SBR requires an SR-GNN poison runner.")
    base = [list(session) for session in shared.template_sessions]
    base_hash = sessions_sha1(base)
    shared_hash = _load_shared_template_sessions_sha1(
        shared.shared_paths,
        expected_hash=base_hash,
    )
    effective, counts = filter_effective_templates(base)
    if not effective:
        raise ValueError("CREAT-Additive-SBR has no effective templates after len < 2 filtering.")
    return CreatPreparedArtifacts(
        adapter=adapter_class(shared.poison_runner),
        base_template_sessions=base,
        effective_templates=effective,
        template_counts=counts,
        base_template_hash=base_hash,
        effective_template_hash=sessions_sha1(effective),
        shared_template_sessions_sha1=shared_hash,
    )


def generate_creat_target(
    *,
    config: Config,
    shared,
    prepared: CreatPreparedArtifacts,
    target_item: int,
    run_type: str,
    save_poisoned_sessions: bool = False,
    trainer_class=CreatAdditiveSBRTrainer,
) -> CreatTargetGenerationResult:
    creat_config = config.attack.creat_additive_sbr
    if creat_config is None:
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.")
    started_at = _timestamp_utc()
    started_monotonic = time.monotonic()
    target_root = target_dir(config, int(target_item), run_type=run_type)
    target_root.mkdir(parents=True, exist_ok=True)
    target_templates, target_filter_counts = filter_templates_with_valid_candidates(
        prepared.effective_templates,
        target_item=int(target_item),
        topk_ratio=float(config.attack.replacement_topk_ratio),
        nonzero_when_possible=bool(creat_config.nonzero_when_possible),
    )
    if not target_templates:
        raise ValueError(
            "CREAT-Additive-SBR has no templates with a valid replacement candidate "
            f"for target_item={int(target_item)}."
        )
    pre_existing_exposure = target_exposure_counts(target_templates, target_item=int(target_item))
    trainer = trainer_class(
        adapter=prepared.adapter,
        config=creat_config,
        replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
        seed=int(config.seeds.position_opt_seed),
    )
    train_result = trainer.train(
        target_item=int(target_item),
        template_sessions=target_templates,
    )
    build_started_at = _timestamp_utc()
    build_started_monotonic = time.monotonic()
    build_result = build_creat_poisoned_sessions(
        adapter=prepared.adapter,
        masker=train_result.masker,
        target_item=int(target_item),
        template_sessions=target_templates,
        replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
        nonzero_when_possible=bool(creat_config.nonzero_when_possible),
        max_item_id=int(prepared.adapter.max_item_id),
    )
    build_completed_at = _timestamp_utc()
    build_elapsed = time.monotonic() - build_started_monotonic
    post_exposure = target_exposure_counts(
        build_result.poisoned_sessions,
        target_item=int(target_item),
    )
    reward_table = getattr(train_result, "reward_table", None)
    selected_reward_stats = (
        reward_table.selected_reward_stats(build_result.selected_positions)
        if reward_table is not None
        else None
    )
    candidate_reward_stats = (
        reward_table.candidate_reward_stats
        if reward_table is not None
        else None
    )
    composed_stat_kwargs = {
        "pattern_reward_weight": float(creat_config.pattern_reward_weight),
        "dpp_reward_weight": float(creat_config.dpp_reward_weight),
        "global_consistency_weight": float(creat_config.global_consistency_weight),
        "local_consistency_weight": float(creat_config.local_consistency_weight),
    }
    candidate_composed_reward_stats = (
        reward_table.composed_reward_stats(**composed_stat_kwargs)
        if reward_table is not None
        else None
    )
    selected_composed_reward_stats = (
        reward_table.composed_reward_stats(
            selected_positions=build_result.selected_positions,
            **composed_stat_kwargs,
        )
        if reward_table is not None
        else None
    )
    reward_table_path = None
    if reward_table is not None:
        reward_table_path = target_root / "creat_additive_sbr_v2_raw_reward_table.pkl"
        with reward_table_path.open("wb") as handle:
            pickle.dump(reward_table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    history_path = target_root / "creat_additive_sbr_train_history.json"
    save_json(train_result.history, history_path)
    selected_positions_path = target_root / "creat_additive_sbr_selected_positions.json"
    save_json([int(position) for position in build_result.selected_positions], selected_positions_path)
    position_stats_path = save_position_stats(
        target_root / "position_stats.json",
        sessions=target_templates,
        positions=build_result.selected_positions,
        run_type=run_type,
        target_item=int(target_item),
    )
    poisoned_sessions_path = None
    if save_poisoned_sessions:
        poisoned_sessions_path = target_root / "creat_additive_sbr_poisoned_sessions.pkl"
        save_fake_sessions(build_result.poisoned_sessions, poisoned_sessions_path)
        save_json(
            {
                "target_item": int(target_item),
                "session_count": int(len(build_result.poisoned_sessions)),
                "sessions_sha1": sessions_sha1(build_result.poisoned_sessions),
            },
            target_root / "creat_additive_sbr_poisoned_sessions_summary.json",
        )
    completed_at = _timestamp_utc()
    elapsed = time.monotonic() - started_monotonic
    original_count = int(prepared.template_counts["original_template_count"])
    poisoned_count = int(build_result.metadata["effective_poisoned_copied_session_count"])
    metadata: dict[str, object] = {
        "run_type": run_type,
        "method_label": METHOD_LABEL,
        "variant": str(creat_config.variant),
        "target_item": int(target_item),
        "creat_attack_started_at": started_at,
        "creat_attack_completed_at": completed_at,
        "creat_attack_elapsed_seconds": round(float(elapsed), 3),
        "creat_masker_train_started_at": train_result.history.get("started_at"),
        "creat_masker_train_completed_at": train_result.history.get("completed_at"),
        "creat_masker_train_elapsed_seconds": train_result.history.get("elapsed_seconds"),
        "poison_build_started_at": build_started_at,
        "poison_build_completed_at": build_completed_at,
        "poison_build_elapsed_seconds": round(float(build_elapsed), 3),
        **prepared.template_counts,
        **target_filter_counts,
        "effective_poisoned_copied_session_count": poisoned_count,
        "effective_budget_ratio": (
            0.0 if original_count <= 0 else float(poisoned_count) / float(original_count)
        ),
        **build_result.metadata,
        "pre_existing_target_session_count": int(pre_existing_exposure["target_session_count"]),
        "pre_existing_target_item_count": int(pre_existing_exposure["target_item_count"]),
        "pre_existing_target_label_pair_count": int(pre_existing_exposure["target_label_pair_count"]),
        "post_poison_target_label_pair_count": int(post_exposure["target_label_pair_count"]),
        "new_target_label_pair_count": int(
            post_exposure["target_label_pair_count"]
            - pre_existing_exposure["target_label_pair_count"]
        ),
        "selected_position_distribution": position_distribution(build_result.selected_positions),
        **position_collapse_summary(build_result.selected_positions),
        "candidate_reward_stats": candidate_reward_stats,
        "selected_reward_stats": selected_reward_stats,
        "candidate_composed_reward_stats": candidate_composed_reward_stats,
        "selected_composed_reward_stats": selected_composed_reward_stats,
        "reward_table_build_metadata": (
            reward_table.build_metadata if reward_table is not None else None
        ),
        "base_template_hash": prepared.base_template_hash,
        "shared_template_sessions_sha1": prepared.shared_template_sessions_sha1,
        "effective_template_hash": prepared.effective_template_hash,
        "target_effective_template_hash": sessions_sha1(target_templates),
        "template_source": {
            "type": config.attack.fake_session_source.type,
            "config": config.to_primitive()["attack"]["fake_session_source"],
            "path": str(shared.shared_paths["fake_sessions"]),
            "sampling_seed_source": "seeds.fake_session_seed",
        },
        "attack_reward_mode": creat_config.attack_reward_mode,
        "max_attack_num": int(creat_config.max_attack_num),
        "operation": "replacement",
        "dpp_style_diversity_implemented": bool(str(creat_config.variant) == "v2"),
        "dpp_style_diversity_enabled": bool(
            str(creat_config.variant) == "v2" and float(creat_config.dpp_reward_weight) > 0.0
        ),
        "creat_fidelity": creat_fidelity_metadata(
            variant=str(creat_config.variant),
            dpp_reward_weight=float(creat_config.dpp_reward_weight),
        ),
        "source_semantics": (
            "Original CREAT is profile pollution; CREAT-Additive-SBR copies clean train "
            "sessions, replaces one item in each copy, and appends the polluted copies "
            "without overwriting clean train data."
        ),
        "position_stats_path": str(position_stats_path),
        "creat_additive_sbr_train_history_path": str(history_path),
        "creat_additive_sbr_selected_positions_path": str(selected_positions_path),
        "creat_additive_sbr_poisoned_sessions_path": (
            str(poisoned_sessions_path) if poisoned_sessions_path is not None else None
        ),
        "creat_additive_sbr_raw_reward_table_path": (
            str(reward_table_path) if reward_table_path is not None else None
        ),
    }
    metadata_path = target_root / "creat_additive_sbr_metadata.json"
    save_json(metadata, metadata_path)
    metadata["creat_additive_sbr_metadata_path"] = str(metadata_path)
    return CreatTargetGenerationResult(
        poisoned_sessions=build_result.poisoned_sessions,
        metadata=metadata,
    )


def _load_shared_template_sessions_sha1(
    shared_paths: dict[str, Path],
    *,
    expected_hash: str,
) -> str | None:
    summary_path = Path(shared_paths["attack_shared_dir"]) / "fake_session_source_summary.json"
    summary = load_json(summary_path)
    if summary is None:
        return None
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


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CreatPreparedArtifacts",
    "CreatTargetGenerationResult",
    "generate_creat_target",
    "prepare_creat_artifacts",
]
