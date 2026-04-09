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
from attack.data.dataset_serializer import save_srg_nn_train
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.random_nonzero_when_possible import RandomNonzeroWhenPossiblePolicy
from attack.models.srgnn_runner import SRGNNRunner
from attack.pipeline.evaluator import (
    evaluate_runner,
    evaluate_targeted_precision_at_k,
    save_metrics,
)
from attack.pipeline.pipeline_utils import build_default_opt, prepare_shared_attack_artifacts


def _prepare_run_artifacts(config: Config, config_path: str | Path | None) -> dict[str, Path]:
    artifacts = run_artifact_paths(config, config.experiment.name)
    run_dir = artifacts["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def run_random_nonzero(
    config: Config,
    config_path: str | Path | None = None,
    poison_epochs: int = 1,
    attack_epochs: int = 1,
) -> dict[str, object]:
    set_seed(config.experiment.seed)
    artifacts = _prepare_run_artifacts(config, config_path)
    shared = prepare_shared_attack_artifacts(
        config,
        poison_epochs=poison_epochs,
        require_poison_runner=False,
        config_path=config_path,
    )

    target_item = shared.target_item
    rng = random.Random(config.experiment.seed)
    policy = RandomNonzeroWhenPossiblePolicy(
        config.attack.replacement_topk_ratio, rng=rng
    )
    fake_sessions = []
    position_counts: Counter[int] = Counter()
    for session in shared.template_sessions:
        updated = policy.apply(session, target_item)
        fake_sessions.append(updated)
        replace_index = updated.index(target_item)
        position_counts[int(replace_index)] += 1

    max_item = max(shared.stats.item_counts)
    if any(max(session) > max_item for session in fake_sessions):
        raise ValueError("Generated fake sessions contain invalid item IDs.")

    poisoned = build_poisoned_dataset(shared.clean_sessions, shared.clean_labels, fake_sessions)

    poisoned_train_path = artifacts["poisoned_train"]
    save_srg_nn_train(poisoned_train_path, poisoned.sessions, poisoned.labels)

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
    positions_path = artifacts["random_nonzero_position_metadata"]
    with positions_path.open("w", encoding="utf-8") as handle:
        json.dump(positions_payload, handle, indent=2, sort_keys=True)

    attacked_runner = SRGNNRunner(config)
    attacked_runner.build_model(build_default_opt(attack_epochs))
    attacked_train_data, attacked_test_data = attacked_runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=config.dataset.test,
    )
    if attack_epochs > 0:
        attacked_runner.train(
            attacked_train_data,
            attacked_test_data,
            attack_epochs,
            target_item=target_item,
            topk=config.evaluation.topk,
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
        "run_type": "random_nonzero_when_possible",
        "metrics": attack_metrics,
        "target_item": int(target_item),
        "fake_session_count": int(shared.fake_session_count),
        "clean_session_count": int(len(shared.clean_sessions)),
        "poison_epochs": int(poison_epochs),
        "attack_epochs": int(attack_epochs),
        "poisoned_train_path": str(poisoned_train_path),
        "random_nonzero_position_metadata_path": str(positions_path),
    }
    save_metrics(payload, artifacts["metrics"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica_attack_random_nonzero_when_possible.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--poison-epochs", type=int, default=1, help="Poison model epochs.")
    parser.add_argument("--attack-epochs", type=int, default=1, help="Attack model epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_random_nonzero(
        config,
        config_path=args.config,
        poison_epochs=args.poison_epochs,
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main()
