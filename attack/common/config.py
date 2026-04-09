from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentConfig:
    name: str


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    split_protocol: str
    poison_train_only: bool


@dataclass(frozen=True)
class SeedsConfig:
    fake_session_seed: int
    target_selection_seed: int


@dataclass(frozen=True)
class PoisonModelConfig:
    name: str


@dataclass(frozen=True)
class AttackConfig:
    size: float
    fake_session_generation_topk: int
    replacement_topk_ratio: float
    poison_model: PoisonModelConfig


@dataclass(frozen=True)
class TargetsConfig:
    mode: str
    explicit_list: tuple[int, ...]
    bucket: str
    count: int
    reuse_saved_targets: bool


@dataclass(frozen=True)
class VictimsConfig:
    enabled: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    topk: int
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactsConfig:
    root: str
    shared_dir: str
    runs_dir: str


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    seeds: SeedsConfig
    attack: AttackConfig
    targets: TargetsConfig
    victims: VictimsConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required field: {context}.{key}")
    return mapping[key]


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected {context} to be a string, got {type(value).__name__}")
    return value


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {context} to be an int, got {type(value).__name__}")
    return value


def _as_float(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected {context} to be a number, got {type(value).__name__}")
    return float(value)


def _as_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Expected {context} to be a bool, got {type(value).__name__}")
    return value


def _as_str_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {context} to be a list of strings.")
    result = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"Expected {context} to contain only strings.")
        result.append(item)
    return tuple(result)


def _as_int_list(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {context} to be a list of ints.")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"Expected {context} to contain only ints.")
        result.append(int(item))
    return tuple(result)


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
    data_cfg = _require(data, "data", "root")
    seeds = _require(data, "seeds", "root")
    attack = _require(data, "attack", "root")
    targets = _require(data, "targets", "root")
    victims = _require(data, "victims", "root")
    evaluation = _require(data, "evaluation", "root")
    artifacts = _require(data, "artifacts", "root")

    if "seed" in experiment:
        raise ValueError("experiment.seed is not supported. Use seeds.fake_session_seed and seeds.target_selection_seed.")

    experiment_cfg = ExperimentConfig(
        name=_as_str(_require(experiment, "name", "experiment"), "experiment.name"),
    )

    data_cfg = DataConfig(
        dataset_name=_as_str(_require(data_cfg, "dataset_name", "data"), "data.dataset_name"),
        split_protocol=_as_str(_require(data_cfg, "split_protocol", "data"), "data.split_protocol"),
        poison_train_only=_as_bool(_require(data_cfg, "poison_train_only", "data"), "data.poison_train_only"),
    )

    if data_cfg.split_protocol != "unified":
        raise ValueError("data.split_protocol must be 'unified'.")

    seeds_cfg = SeedsConfig(
        fake_session_seed=_as_int(_require(seeds, "fake_session_seed", "seeds"), "seeds.fake_session_seed"),
        target_selection_seed=_as_int(
            _require(seeds, "target_selection_seed", "seeds"),
            "seeds.target_selection_seed",
        ),
    )

    poison_model = _require(attack, "poison_model", "attack")
    poison_model_cfg = PoisonModelConfig(
        name=_as_str(_require(poison_model, "name", "attack.poison_model"), "attack.poison_model.name")
    )
    if poison_model_cfg.name.lower() != "srgnn":
        raise ValueError("attack.poison_model.name must be 'srgnn' for Batch 1.")

    attack_cfg = AttackConfig(
        size=_as_float(_require(attack, "size", "attack"), "attack.size"),
        fake_session_generation_topk=_as_int(
            _require(attack, "fake_session_generation_topk", "attack"),
            "attack.fake_session_generation_topk",
        ),
        replacement_topk_ratio=_as_float(
            _require(attack, "replacement_topk_ratio", "attack"),
            "attack.replacement_topk_ratio",
        ),
        poison_model=poison_model_cfg,
    )

    targets_mode = _as_str(_require(targets, "mode", "targets"), "targets.mode")
    explicit_list = _as_int_list(targets.get("explicit_list", []), "targets.explicit_list")
    bucket = _as_str(targets.get("bucket", "popular"), "targets.bucket")
    count = _as_int(targets.get("count", 1), "targets.count")
    reuse_saved_targets = _as_bool(targets.get("reuse_saved_targets", True), "targets.reuse_saved_targets")

    if targets_mode not in {"explicit_list", "sampled"}:
        raise ValueError("targets.mode must be 'explicit_list' or 'sampled'.")
    if targets_mode == "explicit_list" and not explicit_list:
        raise ValueError("targets.explicit_list must be non-empty when mode is explicit_list.")
    if targets_mode == "sampled":
        if bucket not in {"popular", "unpopular", "all"}:
            raise ValueError("targets.bucket must be one of: popular, unpopular, all.")
        if count <= 0:
            raise ValueError("targets.count must be positive when mode is sampled.")

    targets_cfg = TargetsConfig(
        mode=targets_mode,
        explicit_list=explicit_list,
        bucket=bucket,
        count=count,
        reuse_saved_targets=reuse_saved_targets,
    )

    victims_cfg = VictimsConfig(
        enabled=_as_str_list(_require(victims, "enabled", "victims"), "victims.enabled")
    )
    if not victims_cfg.enabled:
        raise ValueError("victims.enabled must include at least one victim model.")
    allowed_victims = {"srgnn"}
    if not set(victims_cfg.enabled).issubset(allowed_victims):
        raise ValueError(
            f"victims.enabled must be subset of {sorted(allowed_victims)}, "
            f"got {victims_cfg.enabled}"
        )

    evaluation_cfg = EvaluationConfig(
        topk=_as_int(_require(evaluation, "topk", "evaluation"), "evaluation.topk"),
        metrics=_as_str_list(_require(evaluation, "metrics", "evaluation"), "evaluation.metrics"),
    )
    if not evaluation_cfg.metrics:
        raise ValueError("evaluation.metrics must include at least one metric.")
    allowed_metrics = {"targeted_precision", "targeted_mrr"}
    if not set(evaluation_cfg.metrics).issubset(allowed_metrics):
        raise ValueError("evaluation.metrics must be a subset of: targeted_precision, targeted_mrr.")

    artifacts_cfg = ArtifactsConfig(
        root=_as_str(_require(artifacts, "root", "artifacts"), "artifacts.root"),
        shared_dir=_as_str(artifacts.get("shared_dir", "shared"), "artifacts.shared_dir"),
        runs_dir=_as_str(artifacts.get("runs_dir", "runs"), "artifacts.runs_dir"),
    )

    return Config(
        experiment=experiment_cfg,
        data=data_cfg,
        seeds=seeds_cfg,
        attack=attack_cfg,
        targets=targets_cfg,
        victims=victims_cfg,
        evaluation=evaluation_cfg,
        artifacts=artifacts_cfg,
    )


__all__ = ["Config", "load_config"]
