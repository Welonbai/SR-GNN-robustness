from __future__ import annotations

import argparse
import math
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace

from attack.common.config import Config, load_config
from attack.common.paths import artifact_paths
from attack.common.seed import set_seed
from attack.data.dataset_serializer import load_srg_nn_train, save_srg_nn_train
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.data.session_stats import compute_session_stats
from attack.data.target_selector import resolve_target_item
from attack.generation.fake_session_generator import FakeSessionGenerator
from attack.generation.fake_session_parameter_sampler import FakeSessionParameterSampler
from attack.insertion.random_topk_replace import RandomTopKReplacePolicy
from attack.models.srgnn_runner import SRGNNRunner
from attack.pipeline.evaluator import (
    evaluate_runner,
    evaluate_targeted_precision_at_k,
    save_metrics,
)


def _build_default_opt(epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=100,
        hiddenSize=100,
        epoch=epochs,
        lr=0.001,
        lr_dc=0.1,
        lr_dc_step=3,
        l2=1e-5,
        step=1,
        patience=10,
        nonhybrid=False,
    )


def _prepare_artifacts(config: Config, config_path: str | Path | None) -> dict[str, Path]:
    artifacts = artifact_paths(config)
    run_dir = artifacts["config_snapshot"].parent
    run_dir.mkdir(parents=True, exist_ok=True)
    for key in ("logs", "checkpoints", "fake_sessions", "poisoned_dataset"):
        artifacts[key].mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def _fake_session_count(ratio: float, clean_count: int) -> int:
    if ratio <= 0:
        return 0
    count = int(round(clean_count * ratio))
    return max(1, count)


def run_dp_sbr_baseline(
    config: Config,
    config_path: str | Path | None = None,
    poison_epochs: int = 1,
    attack_epochs: int = 1,
) -> dict[str, object]:
    set_seed(config.experiment.seed)
    artifacts = _prepare_artifacts(config, config_path)

    clean_sessions, clean_labels = load_srg_nn_train(config.dataset.train)
    stats = compute_session_stats(clean_sessions)

    target_item = resolve_target_item(
        stats,
        mode=config.attack.target_selection_mode,
        explicit_item=config.attack.target_item,
        seed=config.experiment.seed,
    )

    poison_runner = SRGNNRunner(config)
    poison_runner.build_model(_build_default_opt(poison_epochs))
    poison_train_data, poison_test_data = poison_runner.load_dataset()
    if poison_epochs > 0:
        poison_runner.train(poison_train_data, poison_test_data, poison_epochs)

    sampler = FakeSessionParameterSampler(stats)
    generator = FakeSessionGenerator(
        poison_runner,
        sampler,
        topk=config.attack.fake_session_generation_topk,
    )
    fake_count = _fake_session_count(config.attack.size, len(clean_sessions))
    template_sessions = [s.items for s in generator.generate_many(fake_count)]

    policy = RandomTopKReplacePolicy(config.attack.replacement_topk_ratio)
    fake_sessions = [policy.apply(session, target_item) for session in template_sessions]

    max_item = max(stats.item_counts)
    if any(max(session) > max_item for session in fake_sessions):
        raise ValueError("Generated fake sessions contain invalid item IDs.")

    poisoned = build_poisoned_dataset(clean_sessions, clean_labels, fake_sessions)

    fake_sessions_path = artifacts["fake_sessions"] / "fake_sessions.pkl"
    with fake_sessions_path.open("wb") as handle:
        pickle.dump(fake_sessions, handle)

    poisoned_train_path = artifacts["poisoned_dataset"] / "poisoned_train.txt"
    save_srg_nn_train(poisoned_train_path, poisoned.sessions, poisoned.labels)

    attacked_runner = SRGNNRunner(config)
    attacked_runner.build_model(_build_default_opt(attack_epochs))
    attacked_train_data, attacked_test_data = attacked_runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=config.dataset.test,
    )
    if attack_epochs > 0:
        attacked_runner.train(attacked_train_data, attacked_test_data, attack_epochs)

    attack_metrics = evaluate_runner(
        attacked_runner, attacked_test_data, topk=config.evaluation.topk
    )
    targeted = evaluate_targeted_precision_at_k(
        attacked_runner,
        attacked_test_data,
        target_item=target_item,
        topk=config.evaluation.topk,
    )
    metrics = {
        "attack": attack_metrics,
        "targeted_precision_at_k": float(targeted),
        "target_item": int(target_item),
        "fake_session_count": int(fake_count),
        "clean_session_count": int(len(clean_sessions)),
        "poison_epochs": int(poison_epochs),
        "attack_epochs": int(attack_epochs),
        "poisoned_train_path": str(poisoned_train_path),
    }
    save_metrics(metrics, artifacts["metrics"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica.yaml",
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
