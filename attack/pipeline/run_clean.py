from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from types import SimpleNamespace

from attack.common.config import Config, load_config
from attack.common.paths import artifact_paths
from attack.common.seed import set_seed
from attack.data.session_stats import compute_session_stats, load_train_sessions
from attack.data.target_selector import resolve_target_item
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


def run_clean(config: Config, config_path: str | Path | None = None, epochs: int = 1) -> dict[str, float]:
    set_seed(config.experiment.seed)
    artifacts = _prepare_artifacts(config, config_path)

    runner = SRGNNRunner(config)
    runner.build_model(_build_default_opt(epochs))
    train_data, test_data = runner.load_dataset()
    if epochs > 0:
        runner.train(train_data, test_data, epochs)

    clean_sessions = load_train_sessions(config.dataset.train)
    stats = compute_session_stats(clean_sessions)
    target_item = resolve_target_item(
        stats,
        mode=config.attack.target_selection_mode,
        explicit_item=config.attack.target_item,
        seed=config.experiment.seed,
    )

    metrics = evaluate_runner(runner, test_data, topk=config.evaluation.topk)
    targeted = evaluate_targeted_precision_at_k(
        runner, test_data, target_item=target_item, topk=config.evaluation.topk
    )
    metrics["target_item"] = int(target_item)
    metrics["targeted_precision_at_k"] = float(targeted)
    save_metrics({"clean": metrics}, artifacts["metrics"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_clean(config, config_path=args.config, epochs=args.epochs)


if __name__ == "__main__":
    main()
