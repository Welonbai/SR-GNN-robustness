from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import run_artifact_paths
from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.dpsbr_baseline import DPSBRBaselinePolicy
from attack.pipeline.core.evaluator import save_metrics
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts, resolve_target_item
from attack.pipeline.core.victim_execution import execute_single_victim


def _prepare_run_artifacts(
    config: Config, config_path: str | Path | None, *, target_item: int
) -> dict[str, Path]:
    artifacts = run_artifact_paths(config, target_id=target_item)
    run_dir = artifacts["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def run_dp_sbr_baseline(
    config: Config,
    config_path: str | Path | None = None,
    poison_epochs: int = 1,
    attack_epochs: int = 1,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Batch 6 expects data.poison_train_only to be true.")
    if len(config.victims.enabled) != 1:
        raise ValueError("Batch 7 supports exactly one victim model.")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        poison_epochs=poison_epochs,
        require_poison_runner=False,
        config_path=config_path,
    )
    target_item = resolve_target_item(shared.stats, config, shared_paths=shared.shared_paths)
    artifacts = _prepare_run_artifacts(config, config_path, target_item=target_item)

    rng = random.Random(config.seeds.fake_session_seed)
    policy = DPSBRBaselinePolicy(config.attack.replacement_topk_ratio, rng=rng)
    fake_sessions = []
    position_counts: Counter[int] = Counter()
    for session in shared.template_sessions:
        result = policy.apply_with_metadata(session, target_item)
        fake_sessions.append(result.session)
        position_counts[int(result.position)] += 1

    max_item = max(shared.stats.item_counts)
    if any(max(session) > max_item for session in fake_sessions):
        raise ValueError("Generated fake sessions contain invalid item IDs.")

    poisoned = build_poisoned_dataset(shared.clean_sessions, shared.clean_labels, fake_sessions)

    victim_name = config.victims.enabled[0]

    total_positions = sum(position_counts.values())
    ratios = {
        str(pos): (count / total_positions if total_positions else 0.0)
        for pos, count in position_counts.items()
    }
    positions_payload = {
        "total": int(total_positions),
        "counts": {str(pos): int(count) for pos, count in position_counts.items()},
        "ratios": ratios,
    }
    positions_path = artifacts["dpsbr_position_metadata"]
    with positions_path.open("w", encoding="utf-8") as handle:
        json.dump(positions_payload, handle, indent=2, sort_keys=True)

    victim_result = execute_single_victim(
        config,
        victim_name=victim_name,
        canonical_dataset=shared.canonical_dataset,
        poisoned_sessions=poisoned.sessions,
        poisoned_labels=poisoned.labels,
        run_dir=artifacts["run_dir"],
        poisoned_train_path=artifacts["poisoned_train"],
        target_item=target_item,
        attack_epochs=attack_epochs,
        eval_topk=config.evaluation.topk,
        srg_nn_export_paths=shared.export_paths,
    )

    payload = {
        "run_type": "dpsbr_baseline",
        "target_item": int(target_item),
        "fake_session_count": int(shared.fake_session_count),
        "clean_session_count": int(len(shared.clean_sessions)),
        "poison_epochs": int(poison_epochs),
        "attack_epochs": int(attack_epochs),
        "dpsbr_position_metadata_path": str(positions_path),
    }
    if victim_result.metrics is not None:
        payload["metrics"] = victim_result.metrics
    if victim_result.poisoned_train_path is not None:
        payload["poisoned_train_path"] = str(victim_result.poisoned_train_path)
    payload.update(victim_result.extra)
    save_metrics(payload, artifacts["metrics"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica_attack_dpsbr.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--poison-epochs", type=int, default=1, help="Poison model epochs.")
    parser.add_argument("--attack-epochs", type=int, default=1, help="Attack model epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dp_sbr_baseline(
        config,
        config_path=args.config,
        poison_epochs=args.poison_epochs,
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main()
