from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.artifact_io import load_json, save_json
from attack.common.config import Config, load_config
from attack.common.seed import set_seed
from attack.models.victim.registry import get_victim_runner
from attack.pipeline.core.evaluator import evaluate_prediction_metrics, save_predictions
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.pipeline.core.pipeline_utils import (
    build_srgnn_opt_from_train_config,
    prepare_shared_attack_artifacts,
)


LOWK_KEYS = (
    "targeted_mrr@10",
    "targeted_mrr@20",
    "targeted_recall@10",
    "targeted_recall@20",
)


def evaluate_srgnn_checkpoint(
    config: Config,
    *,
    target_item: int,
    checkpoint_path: str | Path,
    poisoned_train_path: str | Path,
    output_dir: str | Path | None = None,
    selected_checkpoint_epoch: int | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    poisoned_train_path = Path(poisoned_train_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not poisoned_train_path.exists():
        raise FileNotFoundError(f"Poisoned train file not found: {poisoned_train_path}")

    if output_dir is None:
        output_dir = checkpoint_path.parent / "checkpoint_eval"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type="srgnn_checkpoint_eval",
        require_poison_runner=False,
        config_path=config_path,
    )
    train_config = _require_srgnn_train_config(config)
    max_topk = max(int(k) for k in config.evaluation.topk)

    set_seed(int(config.seeds.victim_train_seed))
    runner = get_victim_runner("srgnn")(config)
    runner.build_model(build_srgnn_opt_from_train_config(train_config))
    runner.load_model(checkpoint_path)
    _, test_data = runner.load_dataset(
        train_path=poisoned_train_path,
        test_path=shared.export_paths["test"],
        shuffle_train=False,
    )
    rankings = runner.predict_topk(test_data, topk=max_topk)
    predictions_path = output_dir / "predictions.json"
    save_predictions(
        predictions_path,
        topk=max_topk,
        rankings=rankings,
        victim="srgnn",
        target_item=int(target_item),
    )
    ground_truth_labels = resolve_ground_truth_labels(
        config,
        victim_name="srgnn",
        canonical_dataset=shared.canonical_dataset,
        predictions=rankings,
    )
    metrics, available = evaluate_prediction_metrics(
        rankings,
        target_item=int(target_item),
        ground_truth_labels=ground_truth_labels,
        targeted_metrics=config.evaluation.targeted_metrics,
        ground_truth_metrics=config.evaluation.ground_truth_metrics,
        topk=config.evaluation.topk,
    )
    payload = {
        "mode": "srgnn_checkpoint_eval",
        "target_item": int(target_item),
        "checkpoint_path": str(checkpoint_path),
        "poisoned_train_path": str(poisoned_train_path),
        "selected_checkpoint_epoch": (
            None if selected_checkpoint_epoch is None else int(selected_checkpoint_epoch)
        ),
        "metrics_available": bool(available),
        "metrics": metrics,
        "raw_lowk": _raw_lowk(metrics),
        "predictions_path": str(predictions_path),
        "source_train_history_path": _source_train_history_path(checkpoint_path),
        "source_train_history": _source_train_history_payload(checkpoint_path),
        "warning": (
            "Checkpoint evaluation only; no training was run and no validation-best "
            "selection was performed by this tool."
        ),
    }
    save_json(payload, output_dir / "metrics.json")
    return payload


def _require_srgnn_train_config(config: Config) -> Mapping[str, Any]:
    try:
        train_config = config.victims.params["srgnn"]["train"]
    except KeyError as exc:
        raise ValueError("Config must contain victims.params.srgnn.train.") from exc
    if not isinstance(train_config, Mapping):
        raise ValueError("victims.params.srgnn.train must be a mapping.")
    return train_config


def _raw_lowk(metrics: Mapping[str, Any]) -> float | None:
    if any(key not in metrics or metrics[key] is None for key in LOWK_KEYS):
        return None
    return float(sum(float(metrics[key]) for key in LOWK_KEYS) / 4.0)


def _source_train_history_path(checkpoint_path: Path) -> str | None:
    path = checkpoint_path.parent / "train_history.json"
    return str(path) if path.exists() else None


def _source_train_history_payload(checkpoint_path: Path) -> dict[str, Any] | None:
    path = checkpoint_path.parent / "train_history.json"
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing SR-GNN victim checkpoint on the test split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-item", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--poisoned-train", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--selected-checkpoint-epoch", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    payload = evaluate_srgnn_checkpoint(
        load_config(config_path),
        target_item=int(args.target_item),
        checkpoint_path=args.checkpoint,
        poisoned_train_path=args.poisoned_train,
        output_dir=args.output_dir,
        selected_checkpoint_epoch=args.selected_checkpoint_epoch,
        config_path=config_path,
    )
    print(
        "Wrote checkpoint evaluation metrics to "
        f"{Path(args.output_dir) / 'metrics.json' if args.output_dir else Path(args.checkpoint).parent / 'checkpoint_eval' / 'metrics.json'}"
    )
    print(f"raw_lowk={payload['raw_lowk']}")


__all__ = ["_raw_lowk", "evaluate_srgnn_checkpoint"]


if __name__ == "__main__":
    main()
