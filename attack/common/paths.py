from __future__ import annotations

from pathlib import Path

from .config import Config


def output_root(config: Config) -> Path:
    return Path(config.output.root)


def dataset_name(config: Config) -> str:
    return Path(config.dataset.train).parent.name or "dataset"


def shared_dir(config: Config) -> Path:
    return output_root(config) / "shared" / dataset_name(config) / f"seed_{config.experiment.seed}"


def runs_dir(config: Config) -> Path:
    return output_root(config) / "runs" / dataset_name(config) / f"seed_{config.experiment.seed}"


def run_dir(config: Config, method_name: str) -> Path:
    return runs_dir(config) / method_name


def shared_artifact_paths(config: Config) -> dict[str, Path]:
    base = shared_dir(config)
    return {
        "shared_dir": base,
        "poison_model": base / "poison_model.pt",
        "fake_sessions": base / "fake_sessions.pkl",
        "target_info": base / "target_info.json",
    }


def run_artifact_paths(config: Config, method_name: str) -> dict[str, Path]:
    base = run_dir(config, method_name)
    return {
        "run_dir": base,
        "config_snapshot": base / "config.yaml",
        "metrics": base / "metrics.json",
        "poisoned_train": base / "poisoned_train.txt",
        "best_position_metadata": base / "best_position_metadata.pkl",
    }


__all__ = [
    "output_root",
    "dataset_name",
    "shared_dir",
    "runs_dir",
    "run_dir",
    "shared_artifact_paths",
    "run_artifact_paths",
]
