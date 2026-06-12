from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from attack.common.config import load_config


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "attack"
    / "configs"
    / "diginetica_attack_dpsbr.yaml"
)


def freqrec_train(**overrides):
    values = {
        "model_type": "freqrec",
        "epochs": 2,
        "batch_size": 4,
        "lr": 0.001,
        "max_seq_length": 5,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "hidden_act": "gelu",
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "alpha": 0.5,
        "gama": 0.5,
        "alpha_loss": 0.1,
        "fft_loss_type": "l1",
        "chux": "p",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "patience": 10,
        "fre": 1.0,
        "fourier_loss": True,
        "checkpoint_protocol": "fixed_epoch",
        "validation_metric": "ndcg@20",
        "metric_cutoffs": [20],
    }
    values.update(overrides)
    return values


def freqrec_config(
    tmp_path: Path,
    *,
    train_overrides=None,
    runtime_overrides=None,
):
    config = load_config(CONFIG_PATH)
    params = dict(config.victims.params)
    params["freqrec"] = {"train": freqrec_train(**(train_overrides or {}))}
    runtime = dict(config.victims.runtime or {})
    freqrec_runtime = {
        "python_executable": "python",
        "repo_root": str(tmp_path / "freqrec"),
        "working_dir": str(tmp_path / "freqrec"),
        "device": {"use_gpu": False, "gpu_id": "0"},
        "dataloader": {"num_workers": 0},
        "diagnostics": {
            "epoch_metrics": False,
            "per_epoch_predictions": False,
            "save_checkpoint": False,
        },
    }
    if runtime_overrides:
        for key, value in runtime_overrides.items():
            if isinstance(value, dict) and isinstance(freqrec_runtime.get(key), dict):
                freqrec_runtime[key] = {**freqrec_runtime[key], **value}
            else:
                freqrec_runtime[key] = value
    runtime["freqrec"] = freqrec_runtime
    return replace(
        config,
        victims=replace(
            config.victims,
            enabled=("freqrec",),
            params=params,
            runtime=runtime,
        ),
    )


def prediction_payload(
    *,
    item_count=5,
    example_count=3,
    requested_topk=8,
    evaluation_topk=5,
    batch_size=4,
    epochs=2,
    protocol="fixed_epoch",
    selected_epoch=None,
    best_metric=None,
    split="test",
    current_epoch=None,
    epochs_completed=None,
    num_workers=0,
    seed=7,
):
    selected_epoch = selected_epoch or epochs
    current_epoch = current_epoch or selected_epoch
    epochs_completed = epochs if epochs_completed is None else epochs_completed
    topk = min(requested_topk, item_count)
    rankings = [
        {"example_id": index, "items": list(range(1, item_count + 1))[:topk]}
        for index in range(example_count)
    ]
    return {
        "schema_version": 1,
        "model": "freqrec",
        "mode": "canonical_sbr",
        "split": split,
        "checkpoint_protocol": protocol,
        "current_epoch": current_epoch,
        "selected_epoch": selected_epoch,
        "epochs_requested": epochs,
        "epochs_completed": epochs_completed,
        "best_epoch": selected_epoch if protocol == "validation_best" else None,
        "best_metric": best_metric if protocol == "validation_best" else None,
        "validation_metric": "ndcg@20",
        "requested_topk": requested_topk,
        "topk": topk,
        "evaluation_topk": evaluation_topk,
        "item_count": item_count,
        "example_count": example_count,
        "batch_size": batch_size,
        "batch_count": (example_count + batch_size - 1) // batch_size,
        "final_batch_size": example_count - batch_size * (
            (example_count + batch_size - 1) // batch_size - 1
        ),
        "num_workers": num_workers,
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "seed": seed,
        "rankings": rankings,
    }
