from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class DatasetPaths:
    train: str
    test: str
    all_train_seq: str


@dataclass(frozen=True)
class OutputPaths:
    root: str
    run_dir: str


@dataclass(frozen=True)
class AttackConfig:
    size: float
    target_selection_mode: str
    fake_session_generation_topk: int
    replacement_topk_ratio: float


@dataclass(frozen=True)
class EvaluationConfig:
    topk: int


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    dataset: DatasetPaths
    attack: AttackConfig
    evaluation: EvaluationConfig
    output: OutputPaths


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required field: {context}.{key}")
    return mapping[key]


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected {context} to be a string, got {type(value).__name__}")
    return value


def _as_int(value: Any, context: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"Expected {context} to be an int, got {type(value).__name__}")
    return value


def _as_float(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected {context} to be a number, got {type(value).__name__}")
    return float(value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load YAML configs. Install with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data = _load_yaml(path)

    experiment = _require(data, "experiment", "root")
    paths = _require(data, "paths", "root")
    attack = _require(data, "attack", "root")
    evaluation = _require(data, "evaluation", "root")

    dataset = _require(paths, "dataset", "paths")
    output = _require(paths, "output", "paths")

    experiment_cfg = ExperimentConfig(
        name=_as_str(_require(experiment, "name", "experiment"), "experiment.name"),
        seed=_as_int(_require(experiment, "seed", "experiment"), "experiment.seed"),
    )

    dataset_cfg = DatasetPaths(
        train=_as_str(_require(dataset, "train", "paths.dataset"), "paths.dataset.train"),
        test=_as_str(_require(dataset, "test", "paths.dataset"), "paths.dataset.test"),
        all_train_seq=_as_str(
            _require(dataset, "all_train_seq", "paths.dataset"),
            "paths.dataset.all_train_seq",
        ),
    )

    output_cfg = OutputPaths(
        root=_as_str(_require(output, "root", "paths.output"), "paths.output.root"),
        run_dir=_as_str(_require(output, "run_dir", "paths.output"), "paths.output.run_dir"),
    )

    attack_cfg = AttackConfig(
        size=_as_float(_require(attack, "size", "attack"), "attack.size"),
        target_selection_mode=_as_str(
            _require(attack, "target_selection_mode", "attack"),
            "attack.target_selection_mode",
        ),
        fake_session_generation_topk=_as_int(
            _require(attack, "fake_session_generation_topk", "attack"),
            "attack.fake_session_generation_topk",
        ),
        replacement_topk_ratio=_as_float(
            _require(attack, "replacement_topk_ratio", "attack"),
            "attack.replacement_topk_ratio",
        ),
    )

    evaluation_cfg = EvaluationConfig(
        topk=_as_int(_require(evaluation, "topk", "evaluation"), "evaluation.topk")
    )

    return Config(
        experiment=experiment_cfg,
        dataset=dataset_cfg,
        attack=attack_cfg,
        evaluation=evaluation_cfg,
        output=output_cfg,
    )


__all__ = ["Config", "load_config"]
