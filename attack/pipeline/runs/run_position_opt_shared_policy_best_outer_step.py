from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    build_position_opt_attack_identity_context,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import (
    prepare_shared_attack_artifacts,
    requested_target_prefix,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.position_opt import (
    POSITION_OPT_SHARED_POLICY_RUN_TYPE,
    build_position_opt_artifact_paths,
    ensure_position_opt_artifact_dirs,
    position_opt_identity_payload,
    resolve_clean_surrogate_checkpoint_path,
    resolve_position_opt_config,
)
from attack.position_opt.poison_builder import replace_item_at_position


DEFAULT_CONFIG_PATH = (
    "attack/configs/diginetica_attack_position_opt_shared_policy_nz_ft3000_srgnn_sample3.yaml"
)
DEFAULT_SOURCE_RUN_ROOT = (
    "outputs/runs/diginetica/attack_position_opt_shared_policy_nz_ft3000_srgnn_sample3/"
    "run_group_6a623ff2f6"
)
_LOG_PREFIX = "[position-opt-shared-best-step]"


def run_position_opt_shared_policy_best_outer_step(
    config: Config,
    *,
    source_run_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Shared-policy best-step replay expects data.poison_train_only=true.")
    if config.attack.position_opt is None:
        raise ValueError("Shared-policy best-step replay requires attack.position_opt config.")

    source_root = Path(source_run_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source shared-policy run root not found: {source_root}")

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)
    target_registry = requested_target_prefix(
        config,
        target_registry=_ensure_target_registry(shared, config),
    )

    resolved_position_opt_config = resolve_position_opt_config(config.attack.position_opt)
    clean_checkpoint = resolve_clean_surrogate_checkpoint_path(
        config,
        run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        override=resolved_position_opt_config.clean_surrogate_checkpoint,
    ).resolve()
    if not clean_checkpoint.exists():
        raise FileNotFoundError(f"Clean surrogate checkpoint not found: {clean_checkpoint}")
    resolved_position_opt_config = replace(
        resolved_position_opt_config,
        clean_surrogate_checkpoint=str(clean_checkpoint),
    )

    source_history_hashes = {
        str(target_item): _sha1_file(_source_training_history_path(source_root, target_item))
        for target_item in target_registry
    }
    identity_position_opt_payload = position_opt_identity_payload(resolved_position_opt_config)
    identity_position_opt_payload["replay"] = {
        "mode": "best_sampled_outer_step",
        "selection_metric": "reward",
        "source_experiment": source_root.parent.name,
        "source_run_group": source_root.name,
        "source_training_history_sha1": source_history_hashes,
    }
    attack_identity_context = build_position_opt_attack_identity_context(
        position_opt_config=identity_position_opt_payload,
        clean_surrogate_checkpoint=clean_checkpoint,
        runtime_seeds={
            "position_opt_seed": int(config.seeds.position_opt_seed),
            "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        },
    )

    print(
        f"{_LOG_PREFIX} source_run_root={source_root} "
        f"targets={','.join(str(item) for item in target_registry)} "
        f"fake_sessions={len(shared.template_sessions)}"
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        artifact_paths = ensure_position_opt_artifact_dirs(
            build_position_opt_artifact_paths(
                config,
                run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                target_item=target_item,
                clean_checkpoint_override=clean_checkpoint,
                attack_identity_context=attack_identity_context,
            )
        )
        history_path = _source_training_history_path(source_root, target_item)
        history_payload = _load_json(history_path)
        best_row = _best_training_history_row(history_payload, target_item=target_item)
        selected_positions = _coerce_selected_positions(
            best_row,
            target_item=target_item,
            expected_count=len(shared.template_sessions),
        )
        optimized_poisoned_sessions = [
            replace_item_at_position(session, position, target_item)
            for session, position in zip(shared.template_sessions, selected_positions)
        ]
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            optimized_poisoned_sessions,
        )

        with artifact_paths.optimized_poisoned_sessions.open("wb") as handle:
            pickle.dump(optimized_poisoned_sessions, handle)
        save_json(
            [
                {"position": int(position), "source": "best_outer_step"}
                for position in selected_positions
            ],
            artifact_paths.selected_positions,
        )
        target_root = target_dir(
            config,
            int(target_item),
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=selected_positions,
            run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Replay of best sampled outer_step from an existing shared-policy "
                "training_history.json; shared-policy training was not rerun."
            ),
        )
        replay_metadata = _build_replay_metadata(
            config=config,
            target_item=target_item,
            source_root=source_root,
            history_path=history_path,
            best_row=best_row,
            selected_positions=selected_positions,
            optimized_poisoned_sessions_path=artifact_paths.optimized_poisoned_sessions,
            selected_positions_path=artifact_paths.selected_positions,
            position_stats_path=position_stats_path,
        )
        replay_metadata_path = artifact_paths.base_dir / "replay_metadata.json"
        save_json(replay_metadata, replay_metadata_path)
        save_json(
            {
                **replay_metadata,
                "clean_surrogate_checkpoint": str(clean_checkpoint),
                "position_opt_config": asdict(resolved_position_opt_config),
                "attack_identity_context": attack_identity_context,
            },
            artifact_paths.run_metadata,
        )

        print(
            f"{_LOG_PREFIX} target={int(target_item)} "
            f"best_outer_step={int(best_row['outer_step'])} "
            f"best_reward={float(best_row['reward']):.6g} "
            f"poison_sessions={len(optimized_poisoned_sessions)}"
        )
        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "position_opt_method": POSITION_OPT_SHARED_POLICY_RUN_TYPE,
                "position_opt_replay_mode": "best_sampled_outer_step",
                "position_opt_source_run_root": str(source_root),
                "position_opt_source_training_history_path": str(history_path),
                "position_opt_best_outer_step": int(best_row["outer_step"]),
                "position_opt_best_reward": float(best_row["reward"]),
                "position_opt_best_target_result_mean": _optional_float(
                    best_row.get("target_result_mean")
                ),
                "position_opt_best_targeted_mrr_at_10": _optional_float(
                    best_row.get("targeted_mrr@10")
                ),
                "position_opt_best_targeted_recall_at_10": _optional_float(
                    best_row.get("targeted_recall@10")
                ),
                "position_opt_best_targeted_recall_at_20": _optional_float(
                    best_row.get("targeted_recall@20")
                ),
                "position_opt_best_targeted_recall_at_30": _optional_float(
                    best_row.get("targeted_recall@30")
                ),
                "position_opt_optimized_poisoned_sessions_path": str(
                    artifact_paths.optimized_poisoned_sessions
                ),
                "position_opt_selected_positions_path": str(
                    artifact_paths.selected_positions
                ),
                "position_opt_replay_metadata_path": str(replay_metadata_path),
                "position_stats_path": str(position_stats_path),
            },
        )

    summary = run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=attack_identity_context,
    )
    print(f"{_LOG_PREFIX} Final victim evaluation completed.")
    return summary


def _ensure_target_registry(shared: Any, config: Config) -> Mapping[str, Any]:
    from attack.pipeline.core.pipeline_utils import ensure_target_registry_prefix

    return ensure_target_registry_prefix(
        shared.stats,
        config,
        shared_paths=shared.shared_paths,
    )


def _source_training_history_path(source_root: Path, target_item: int) -> Path:
    path = source_root / "targets" / str(int(target_item)) / "position_opt" / "training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"Source training_history.json not found: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _best_training_history_row(
    history_payload: Mapping[str, Any],
    *,
    target_item: int,
) -> Mapping[str, Any]:
    rows = history_payload.get("training_history")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"training_history is empty for target {int(target_item)}.")
    best_row = max(rows, key=lambda row: float(row.get("reward", float("-inf"))))
    if "outer_step" not in best_row or "reward" not in best_row:
        raise ValueError(f"Best row is missing outer_step/reward for target {int(target_item)}.")
    return best_row


def _coerce_selected_positions(
    row: Mapping[str, Any],
    *,
    target_item: int,
    expected_count: int,
) -> list[int]:
    raw_positions = row.get("selected_positions")
    if not isinstance(raw_positions, list):
        raise ValueError(
            f"Best row for target {int(target_item)} does not contain selected_positions."
        )
    if len(raw_positions) != int(expected_count):
        raise ValueError(
            f"Best row selected_positions length mismatch for target {int(target_item)}: "
            f"{len(raw_positions)} != {int(expected_count)}."
        )
    return [int(position) for position in raw_positions]


def _build_replay_metadata(
    *,
    config: Config,
    target_item: int,
    source_root: Path,
    history_path: Path,
    best_row: Mapping[str, Any],
    selected_positions: list[int],
    optimized_poisoned_sessions_path: Path,
    selected_positions_path: Path,
    position_stats_path: Path,
) -> dict[str, Any]:
    return {
        "target_item": int(target_item),
        "replay_mode": "best_sampled_outer_step",
        "selection_metric": "reward",
        "source_run_root": str(source_root),
        "source_training_history_path": str(history_path),
        "source_training_history_sha1": _sha1_file(history_path),
        "best_outer_step": int(best_row["outer_step"]),
        "best_reward": float(best_row["reward"]),
        "best_reward_metric_snapshot": {
            "target_result_mean": _optional_float(best_row.get("target_result_mean")),
            "targeted_mrr@10": _optional_float(best_row.get("targeted_mrr@10")),
            "targeted_mrr@30": _optional_float(best_row.get("targeted_mrr@30")),
            "targeted_recall@10": _optional_float(best_row.get("targeted_recall@10")),
            "targeted_recall@20": _optional_float(best_row.get("targeted_recall@20")),
            "targeted_recall@30": _optional_float(best_row.get("targeted_recall@30")),
        },
        "final_policy_selection_from_source": str(
            _optional_source_field(history_path, "final_policy_selection")
        ),
        "source_exported_policy_source": str(
            _optional_source_field(history_path, "exported_policy_source")
        ),
        "selected_position_count": int(len(selected_positions)),
        "optimized_poisoned_sessions_path": str(optimized_poisoned_sessions_path),
        "selected_positions_path": str(selected_positions_path),
        "position_stats_path": str(position_stats_path),
        "victims": list(config.victims.enabled),
        "note": (
            "This run replays the sampled selected_positions from the best reward "
            "outer step. It does not rerun shared-policy optimization."
        ),
    }


def _optional_source_field(history_path: Path, key: str) -> object | None:
    payload = _load_json(history_path)
    return payload.get(key)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML config.")
    parser.add_argument(
        "--source-run-root",
        default=DEFAULT_SOURCE_RUN_ROOT,
        help="Existing shared-policy run_group root containing per-target training_history.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_position_opt_shared_policy_best_outer_step(
        config,
        source_run_root=args.source_run_root,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
