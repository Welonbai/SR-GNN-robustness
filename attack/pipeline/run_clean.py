from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import run_artifact_paths
from attack.common.seed import set_seed
from attack.data.session_stats import compute_session_stats, load_train_sessions
from attack.data.target_selector import sample_one_from_popular, sample_one_from_unpopular
from attack.models.srgnn_runner import SRGNNRunner
from attack.pipeline.evaluator import (
    evaluate_runner,
    evaluate_targeted_precision_at_k,
    save_metrics,
)
from attack.pipeline.pipeline_utils import build_default_opt


def _prepare_run_artifacts(
    config: Config, method_name: str, config_path: str | Path | None
) -> dict[str, Path]:
    artifacts = run_artifact_paths(config, method_name)
    run_dir = artifacts["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def run_clean(config: Config, config_path: str | Path | None = None, epochs: int = 1) -> dict[str, float]:
    set_seed(config.experiment.seed)
    artifacts = _prepare_run_artifacts(config, "clean", config_path)

    runner = SRGNNRunner(config)
    runner.build_model(build_default_opt(epochs))
    train_data, test_data = runner.load_dataset()

    clean_sessions = load_train_sessions(config.dataset.train)
    stats = compute_session_stats(clean_sessions)
    if config.attack.target_selection_mode == "sample_one_from_popular":
        target_item = sample_one_from_popular(stats, seed=config.experiment.seed)
    elif config.attack.target_selection_mode == "sample_one_from_unpopular":
        target_item = sample_one_from_unpopular(stats, seed=config.experiment.seed)
    else:
        raise ValueError("Unsupported target_selection_mode.")
    if epochs > 0:
        runner.train(
            train_data,
            test_data,
            epochs,
            target_item=target_item,
            topk=config.evaluation.topk,
        )

    metrics = evaluate_runner(runner, test_data, topk=config.evaluation.topk)
    targeted = evaluate_targeted_precision_at_k(
        runner, test_data, target_item=target_item, topk=config.evaluation.topk
    )
    metrics["targeted_precision_at_k"] = float(targeted)
    payload = {
        "run_type": "clean",
        "metrics": metrics,
        "target_item": int(target_item),
    }
    save_metrics(payload, artifacts["metrics"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica_clean.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_clean(config, config_path=args.config, epochs=args.epochs)


if __name__ == "__main__":
    main()
