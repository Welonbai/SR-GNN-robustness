from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from attack.common.config import load_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "diginetica_attack_dpsbr.yaml"


def wearec_train(**overrides):
    values = {
        "epochs": 2,
        "batch_size": 4,
        "lr": 0.001,
        "max_seq_length": 6,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "num_heads": 2,
        "alpha": 0.3,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "checkpoint_protocol": "fixed_epoch",
        "metric_cutoffs": [1, 3, 5],
    }
    values.update(overrides)
    return values


def wearec_config(
    tmp_path: Path,
    *,
    python_executable: str | None = None,
    train_overrides=None,
    runtime_overrides=None,
):
    config = load_config(CONFIG_PATH)
    params = dict(config.victims.params)
    params["wearec"] = {"train": wearec_train(**(train_overrides or {}))}
    runtime = dict(config.victims.runtime or {})
    value = {
        "python_executable": python_executable or str(tmp_path / "python.exe"),
        "repo_root": str(tmp_path / "wearec"),
        "working_dir": str(tmp_path / "wearec"),
        "device": {"use_gpu": False, "gpu_id": "0"},
        "dataloader": {"num_workers": 0},
        "diagnostics": {"per_epoch_predictions": False},
    }
    for key, override in (runtime_overrides or {}).items():
        if isinstance(override, dict) and isinstance(value.get(key), dict):
            value[key] = {**value[key], **override}
        else:
            value[key] = override
    runtime["wearec"] = value
    return replace(
        config,
        victims=replace(
            config.victims,
            enabled=("wearec",),
            params=params,
            runtime=runtime,
        ),
    )


def raw_prediction_payload(*, epochs=2, batch_size=4, seed=7):
    rankings = []
    for index, label in enumerate((1, 2, 3)):
        rankings.append(
            {
                "example_id": index,
                "label": label,
                "items": [label] + [item for item in range(1, 6) if item != label],
            }
        )
    return {
        "schema_version": 1,
        "model": "wearec",
        "mode": "canonical_sbr",
        "split": "test",
        "dataset_name": "toy",
        "training_mode": "clean",
        "checkpoint_protocol": "fixed_epoch",
        "epochs_requested": epochs,
        "epochs_completed": epochs,
        "current_epoch": epochs,
        "final_epoch": epochs,
        "selected_epoch": epochs,
        "item_count": 5,
        "max_seq_length": 6,
        "metric_cutoffs": [1, 3, 5],
        "requested_topk": 5,
        "topk": 5,
        "evaluation_topk": 5,
        "example_count": 3,
        "batch_size": batch_size,
        "batch_count": 1,
        "final_batch_size": 3,
        "num_workers": 0,
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "seed": seed,
        "model_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "num_heads": 2,
            "alpha": 0.3,
        },
        "test_metrics": {
            "hr@1": 1.0, "mrr@1": 1.0, "ndcg@1": 1.0,
            "hr@3": 1.0, "mrr@3": 1.0, "ndcg@3": 1.0,
            "hr@5": 1.0, "mrr@5": 1.0, "ndcg@5": 1.0,
        },
        "rankings": rankings,
    }
