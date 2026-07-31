from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from attack.common.artifact_io import save_json
from attack.common.config import load_config
from attack.common.seed import set_seed
from attack.common.srgnn_training_protocol import srgnn_validation_best_enabled
from attack.models.srgnn_validation_training import (
    srgnn_validation_train_history_extra,
    train_srgnn_validation_best,
)
from attack.models.victim.srgnn_runner import SRGNNVictimRunner
from attack.pipeline.core.evaluator import save_predictions
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from attack.pipeline.core.train_history import save_train_history


def run_srgnn_victim_subprocess(
    *,
    config_path: str | Path,
    poisoned_train_path: str | Path,
    valid_path: str | Path,
    test_path: str | Path,
    run_dir: str | Path,
    predictions_path: str | Path,
    target_item: int,
    topk: int,
    seed: int,
    clean_run: bool,
) -> dict[str, Any]:
    config_path = Path(config_path)
    poisoned_train_path = Path(poisoned_train_path)
    valid_path = Path(valid_path)
    test_path = Path(test_path)
    run_dir = Path(run_dir)
    predictions_path = Path(predictions_path)
    for path in (config_path, poisoned_train_path, valid_path, test_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = load_config(config_path)
    train_config = dict(config.victims.params["srgnn"]["train"])
    epochs = int(train_config["epochs"])
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))

    runner = SRGNNVictimRunner(config)
    runner.build_model(build_srgnn_opt_from_train_config(train_config))
    train_data, valid_data = runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=valid_path,
    )
    if srgnn_validation_best_enabled(train_config):
        result = train_srgnn_validation_best(
            runner,
            train_data,
            valid_data,
            train_config=train_config,
            max_epochs=epochs,
            patience=int(train_config["patience"]),
            best_checkpoint_path=run_dir / "best_validation.pt",
            log_prefix="[victim:srgnn-validation-best]",
        )
        save_train_history(
            run_dir / "train_history.json",
            role="victim",
            model="srgnn",
            epochs=len(result.rows),
            train_loss=[float(row["train_loss"]) for row in result.rows],
            valid_loss=[None] * len(result.rows),
            notes=(
                "SRGNN victim training selected the checkpoint with highest "
                "validation ground-truth MRR@20. Test metrics were not used."
            ),
            extra=srgnn_validation_train_history_extra(result),
        )
        completed_epochs = len(result.rows)
    elif epochs > 0:
        runner.train(
            train_data,
            valid_data,
            epochs,
            target_item=None if clean_run else int(target_item),
            topk=int(topk),
        )
        if runner.train_loss_history:
            save_train_history(
                run_dir / "train_history.json",
                role="victim",
                model="srgnn",
                epochs=len(runner.train_loss_history),
                train_loss=runner.train_loss_history,
                valid_loss=[None] * len(runner.train_loss_history),
                notes="valid_loss not available for SRGNN victim training.",
            )
        completed_epochs = len(runner.train_loss_history)
    else:
        completed_epochs = 0

    _, test_data = runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=test_path,
        shuffle_train=False,
    )
    rankings = runner.predict_topk(test_data, topk=int(topk))
    save_predictions(
        predictions_path,
        topk=int(topk),
        rankings=rankings,
        victim="srgnn",
        target_item=int(target_item),
    )
    summary = {
        "victim": "srgnn",
        "target_item": int(target_item),
        "seed": int(seed),
        "epochs_requested": int(epochs),
        "epochs_completed": int(completed_epochs),
        "topk": int(topk),
        "prediction_count": int(len(rankings)),
        "predictions_path": str(predictions_path),
        "execution_mode": "isolated_subprocess",
    }
    save_json(summary, run_dir / "srgnn_subprocess_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated SR-GNN victim.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--poisoned-train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--target-item", required=True, type=int)
    parser.add_argument("--topk", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--clean-run", action="store_true")
    args = parser.parse_args()
    run_srgnn_victim_subprocess(
        config_path=args.config,
        poisoned_train_path=args.poisoned_train,
        valid_path=args.valid,
        test_path=args.test,
        run_dir=args.run_dir,
        predictions_path=args.predictions,
        target_item=int(args.target_item),
        topk=int(args.topk),
        seed=int(args.seed),
        clean_run=bool(args.clean_run),
    )


if __name__ == "__main__":
    main()
