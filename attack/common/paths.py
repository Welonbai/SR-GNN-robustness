from __future__ import annotations

from pathlib import Path

from .config import Config


def output_root(config: Config) -> Path:
    return Path(config.output.root)


def run_dir(config: Config) -> Path:
    root = Path(config.output.root)
    run_path = Path(config.output.run_dir)
    if run_path.is_absolute() or root == Path("."):
        return run_path
    try:
        run_path.relative_to(root)
        return run_path
    except ValueError:
        return root / run_path


def artifact_paths(config: Config) -> dict[str, Path]:
    base = run_dir(config)
    return {
        "config_snapshot": base / "config.yaml",
        "logs": base / "logs",
        "checkpoints": base / "checkpoints",
        "fake_sessions": base / "fake_sessions",
        "poisoned_dataset": base / "poisoned_dataset",
        "metrics": base / "metrics.json",
    }


__all__ = ["output_root", "run_dir", "artifact_paths"]
