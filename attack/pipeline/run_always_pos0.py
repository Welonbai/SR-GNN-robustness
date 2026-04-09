from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import run_artifact_paths
from attack.common.seed import set_seed
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.always_position0 import AlwaysPositionZeroPolicy
from attack.models.victim import srgnn_runner as _srgnn_runner
from attack.models.victim.registry import get_victim_runner
from attack.pipeline.evaluator import (
    evaluate_runner,
    evaluate_targeted_precision_at_k,
    save_metrics,
)
from attack.pipeline.pipeline_utils import (
    build_default_opt,
    prepare_shared_attack_artifacts,
    resolve_target_item,
)


def _prepare_run_artifacts(
    config: Config, config_path: str | Path | None, *, target_item: int
) -> dict[str, Path]:
    artifacts = run_artifact_paths(config, target_id=target_item)
    run_dir = artifacts["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def run_always_pos0(
    config: Config,
    config_path: str | Path | None = None,
    poison_epochs: int = 1,
    attack_epochs: int = 1,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Batch 6 expects data.poison_train_only to be true.")
    if tuple(config.victims.enabled) != ("srgnn",):
        raise ValueError("Batch 6 supports only victims.enabled == ['srgnn'].")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        poison_epochs=poison_epochs,
        require_poison_runner=False,
        config_path=config_path,
    )
    target_item = resolve_target_item(shared.stats, config, shared_paths=shared.shared_paths)
    artifacts = _prepare_run_artifacts(config, config_path, target_item=target_item)

    policy = AlwaysPositionZeroPolicy()
    fake_sessions = [policy.apply(session, target_item) for session in shared.template_sessions]

    max_item = max(shared.stats.item_counts)
    if any(max(session) > max_item for session in fake_sessions):
        raise ValueError("Generated fake sessions contain invalid item IDs.")

    poisoned = build_poisoned_dataset(shared.clean_sessions, shared.clean_labels, fake_sessions)

    exporter = SRGNNExporter()
    poisoned_train_path = exporter.export_train_pairs(
        poisoned.sessions, poisoned.labels, artifacts["poisoned_train"]
    )

    victim_name = config.victims.enabled[0]
    victim_cls = get_victim_runner(victim_name)
    attacked_runner = victim_cls(config)
    attacked_runner.build_model(build_default_opt(attack_epochs))
    attacked_train_data, attacked_valid_data = attacked_runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=shared.export_paths["valid"],
    )
    if attack_epochs > 0:
        attacked_runner.train(
            attacked_train_data,
            attacked_valid_data,
            attack_epochs,
            target_item=target_item,
            topk=config.evaluation.topk,
        )

    _, attacked_test_data = attacked_runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=shared.export_paths["test"],
        shuffle_train=False,
    )
    attack_metrics = evaluate_runner(
        attacked_runner, attacked_test_data, topk=config.evaluation.topk
    )
    targeted = evaluate_targeted_precision_at_k(
        attacked_runner,
        attacked_test_data,
        target_item=target_item,
        topk=config.evaluation.topk,
    )
    attack_metrics["targeted_precision_at_k"] = float(targeted)
    payload = {
        "run_type": "always_pos0",
        "metrics": attack_metrics,
        "target_item": int(target_item),
        "fake_session_count": int(shared.fake_session_count),
        "clean_session_count": int(len(shared.clean_sessions)),
        "poison_epochs": int(poison_epochs),
        "attack_epochs": int(attack_epochs),
        "poisoned_train_path": str(poisoned_train_path),
    }
    save_metrics(payload, artifacts["metrics"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica_attack_pos0.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--poison-epochs", type=int, default=1, help="Poison model epochs.")
    parser.add_argument("--attack-epochs", type=int, default=1, help="Attack model epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_always_pos0(
        config,
        config_path=args.config,
        poison_epochs=args.poison_epochs,
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main()
