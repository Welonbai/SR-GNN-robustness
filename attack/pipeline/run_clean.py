from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import run_artifact_paths, shared_artifact_paths
from attack.common.seed import set_seed
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.models.victim import srgnn_runner as _srgnn_runner
from attack.models.victim.registry import get_victim_runner
from attack.pipeline.evaluator import (
    evaluate_runner,
    evaluate_targeted_precision_at_k,
    save_metrics,
)
from attack.pipeline.pipeline_utils import build_default_opt, resolve_target_item


def _prepare_run_artifacts(
    config: Config, config_path: str | Path | None, *, target_item: int
) -> dict[str, Path]:
    artifacts = run_artifact_paths(config, target_id=target_item)
    run_dir = artifacts["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path:
        shutil.copyfile(config_path, artifacts["config_snapshot"])
    return artifacts


def run_clean(
    config: Config,
    config_path: str | Path | None = None,
    epochs: int = 1,
) -> dict[str, float]:
    if tuple(config.victims.enabled) != ("srgnn",):
        raise ValueError("Batch 6 supports only victims.enabled == ['srgnn'].")
    set_seed(config.seeds.fake_session_seed)

    canonical_dataset = ensure_canonical_dataset(config)
    stats = compute_session_stats(canonical_dataset.train_sub)
    shared_paths = shared_artifact_paths(config)
    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    target_item = resolve_target_item(stats, config, shared_paths=shared_paths)

    artifacts = _prepare_run_artifacts(config, config_path, target_item=target_item)

    export_dir = artifacts["run_dir"] / "export"
    export_result = SRGNNExporter().export(canonical_dataset, export_dir)

    victim_name = config.victims.enabled[0]
    victim_cls = get_victim_runner(victim_name)
    runner = victim_cls(config)
    runner.build_model(build_default_opt(epochs))
    train_data, valid_data = runner.load_dataset(
        train_path=export_result.files["train"],
        test_path=export_result.files["valid"],
    )
    if epochs > 0:
        runner.train(
            train_data,
            valid_data,
            epochs,
            target_item=target_item,
            topk=config.evaluation.topk,
        )

    _, test_data = runner.load_dataset(
        train_path=export_result.files["train"],
        test_path=export_result.files["test"],
        shuffle_train=False,
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
