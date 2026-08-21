from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import (
    Config,
    FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED,
    PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_UNIFORM,
    PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX,
    PTS_PREFIX_RANGE_INTERNAL,
    PTS_PREFIX_SAMPLER_UNIFORM,
    load_config,
)
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pts.direct_action_executor import (
    DIRECT_ACTION_FORMAL_GENERATION_TAG,
    DIRECT_ACTION_FORMAL_PREFIX_TAG,
    DIRECT_ACTION_FORMAL_SAMPLE_TAG,
    apply_pts_direct_action_construction_batch,
    build_direct_action_formal_session_contexts,
)
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_MLP_H2_PARAMETER_NAMES,
    DirectActionMLPPolicy,
)


DEFAULT_CONFIG_PATH = (
    "attack/configs/"
    "ssh_yoochoose1_64_valbest_attack_ptsuniform_direct_srgnn_generated_"
    "budget0p01_popular_all_victims.yaml"
)
_LOG_PREFIX = "[pts-direct-action-uniform]"
_ARTIFACT_DIR_NAME = "pts_direct_action_uniform"
_POLICY_MODE = "zero_logits_uniform_atomic_actions"
_CANDIDATE_KEY = "fixed_zero_policy"


def run_pts_direct_action_uniform(
    config: Config,
    *,
    config_path: str | Path | None = None,
    max_targets_per_execution: int | None = None,
) -> dict[str, object]:
    """Run the no-search direct-action ablation and evaluate requested victims."""

    _validate_uniform_run_config(config)
    pts_config = _require_pts_config(config)
    base_seed = int(config.seeds.position_opt_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    if shared.poison_runner is None:
        raise RuntimeError("Direct-action uniform construction requires a poison runner.")
    context = RunContext.from_shared(shared)

    print(
        f"{_LOG_PREFIX} loaded {len(shared.template_sessions)} generated base sessions "
        f"from {shared.shared_paths['fake_sessions']}"
    )
    print(
        f"{_LOG_PREFIX} method={pts_config.method} policy={_POLICY_MODE} "
        "surrogate_evaluations=0 cem_enabled=false"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        construction_started = time.perf_counter()
        session_contexts, context_stats = build_direct_action_formal_session_contexts(
            template_sessions=shared.template_sessions,
            base_seed=base_seed,
            prefix_rng_tag=DIRECT_ACTION_FORMAL_PREFIX_TAG,
        )
        zero_vector = [0.0] * len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
        policy = DirectActionMLPPolicy.from_vector(
            zero_vector,
            length_feature_mode=pts_config.direct_action_policy.length_feature,
            context_stats=context_stats.to_dict(),
        )
        result = apply_pts_direct_action_construction_batch(
            session_contexts=session_contexts,
            context_stats=context_stats,
            target_item=int(target_item),
            policy=policy,
            base_seed=base_seed,
            iteration=0,
            candidate_key=_CANDIDATE_KEY,
            poison_runner=shared.poison_runner,
            generation_topk=int(pts_config.generation.topk),
            sample_rng_tag=DIRECT_ACTION_FORMAL_SAMPLE_TAG,
            generation_rng_tag=DIRECT_ACTION_FORMAL_GENERATION_TAG,
        )
        _validate_constructed_sessions(
            sessions=result.final_sessions,
            target_item=int(target_item),
            expected_count=len(shared.template_sessions),
            max_item=max(shared.stats.item_counts),
        )

        artifact_dir = target_dir(
            config,
            int(target_item),
            run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
        ) / _ARTIFACT_DIR_NAME
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sessions_path = artifact_dir / "sessions.json"
        summary_path = artifact_dir / "summary.json"
        policy_path = artifact_dir / "policy.json"
        records_path = artifact_dir / "session_records.jsonl"
        complete_path = artifact_dir / "construction_complete.json"

        save_json(result.final_sessions, sessions_path)
        save_json(result.summary, summary_path)
        save_json(
            {
                **policy.to_dict(),
                "mode": _POLICY_MODE,
                "cem_enabled": False,
                "surrogate_evaluation_count": 0,
                "candidate_key": _CANDIDATE_KEY,
                "base_seed": base_seed,
            },
            policy_path,
        )
        if bool(pts_config.artifacts.save_per_session_records):
            _write_jsonl(result.per_session_records, records_path)

        construction_seconds = float(time.perf_counter() - construction_started)
        sessions_sha1 = _sessions_sha1(result.final_sessions)
        complete_payload = {
            "schema_version": "pts_direct_action_uniform_v1",
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_type": PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
            "pts_construction_method": pts_config.method,
            "target_item": int(target_item),
            "policy_mode": _POLICY_MODE,
            "parameter_count": len(zero_vector),
            "parameter_vector": zero_vector,
            "cem_enabled": False,
            "surrogate_evaluation_count": 0,
            "base_seed": base_seed,
            "candidate_key": _CANDIDATE_KEY,
            "poisoned_session_count": len(result.final_sessions),
            "selected_sessions_sha1": sessions_sha1,
            "attack_construction_seconds": construction_seconds,
            "attack_construction_minutes": construction_seconds / 60.0,
            "context_stats": context_stats.to_dict(),
            "summary": result.summary,
            "artifacts": {
                "sessions": str(sessions_path),
                "summary": str(summary_path),
                "policy": str(policy_path),
                "session_records": (
                    str(records_path)
                    if bool(pts_config.artifacts.save_per_session_records)
                    else None
                ),
            },
        }
        save_json(complete_payload, complete_path)

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            result.final_sessions,
        )
        metadata = {
            "pts_construction_method": pts_config.method,
            "pts_uniform_policy_mode": _POLICY_MODE,
            "pts_uniform_cem_enabled": False,
            "pts_uniform_surrogate_evaluation_count": 0,
            "pts_uniform_parameter_vector": zero_vector,
            "pts_uniform_base_seed": base_seed,
            "pts_uniform_candidate_key": _CANDIDATE_KEY,
            "pts_uniform_sessions_sha1": sessions_sha1,
            "pts_uniform_attack_construction_seconds": construction_seconds,
            "pts_uniform_attack_construction_minutes": construction_seconds / 60.0,
            "pts_uniform_sessions_path": str(sessions_path),
            "pts_uniform_summary_path": str(summary_path),
            "pts_uniform_policy_path": str(policy_path),
            "pts_uniform_complete_path": str(complete_path),
            "pts_uniform_action_summary": result.summary,
        }
        if bool(pts_config.artifacts.save_per_session_records):
            metadata["pts_uniform_session_records_path"] = str(records_path)

        print(
            f"{_LOG_PREFIX} target={int(target_item)} sessions={len(result.final_sessions)} "
            f"construction_minutes={construction_seconds / 60.0:.3f}"
        )
        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=result.final_sessions,
            metadata=metadata,
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
        build_poisoned=build_poisoned,
        max_targets_per_execution=max_targets_per_execution,
    )


def _validate_uniform_run_config(config: Config) -> None:
    if not bool(config.data.poison_train_only):
        raise ValueError("Direct-action uniform requires data.poison_train_only=true.")
    pts_config = _require_pts_config(config)
    if not bool(pts_config.enabled):
        raise ValueError("Direct-action uniform requires attack.pts_construction.enabled=true.")
    if pts_config.method != PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_UNIFORM:
        raise ValueError(
            "Direct-action uniform runner requires "
            "attack.pts_construction.method='direct_action_mlp_uniform'."
        )
    if config.attack.fake_session_source.type != FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED:
        raise ValueError(
            "TC-SACP-G-Uniform requires attack.fake_session_source.type="
            "'poison_model_generated'."
        )
    if (
        pts_config.prefix_selector.range != PTS_PREFIX_RANGE_INTERNAL
        or pts_config.prefix_selector.sampler != PTS_PREFIX_SAMPLER_UNIFORM
    ):
        raise ValueError(
            "Direct-action uniform requires the same internal/uniform prefix selector "
            "as TC-SACP-G."
        )
    if (
        pts_config.generation.length_policy
        != PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX
    ):
        raise ValueError(
            "Direct-action uniform requires generation.length_policy="
            "'same_as_residual_suffix'."
        )


def _require_pts_config(config: Config):
    pts_config = config.attack.pts_construction
    if pts_config is None:
        raise ValueError("Direct-action uniform requires attack.pts_construction.")
    return pts_config


def _validate_constructed_sessions(
    *,
    sessions: Sequence[Sequence[int]],
    target_item: int,
    expected_count: int,
    max_item: int,
) -> None:
    if len(sessions) != int(expected_count):
        raise ValueError(
            "Uniform construction session count mismatch: "
            f"expected {int(expected_count)}, got {len(sessions)}."
        )
    target = int(target_item)
    for index, session in enumerate(sessions):
        values = [int(item) for item in session]
        if not values:
            raise ValueError(f"Uniform construction produced empty session at index {index}.")
        if values[0] == target:
            raise ValueError(
                f"Uniform construction placed target first at session index {index}."
            )
        if target not in values:
            raise ValueError(
                f"Uniform construction omitted target at session index {index}."
            )
        if min(values) < 1 or max(values) > int(max_item):
            raise ValueError(
                f"Uniform construction produced invalid item ID at session index {index}."
            )


def _sessions_sha1(sessions: Sequence[Sequence[int]]) -> str:
    payload = [[int(item) for item in session] for session in sessions]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _write_jsonl(records: Sequence[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-zero uniform direct-action TC-SACP-G ablation."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to a YAML config.")
    parser.add_argument(
        "--max-targets-per-execution",
        type=int,
        default=None,
        help="Optional target limit for smoke tests or resumable server batches.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    run_pts_direct_action_uniform(
        config,
        config_path=config_path,
        max_targets_per_execution=args.max_targets_per_execution,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "main",
    "run_pts_direct_action_uniform",
]
