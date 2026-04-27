from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.position_opt_policy_feature_sets import (
    ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS,
)


_ALLOWED_VICTIMS = {"srgnn", "miasrec", "tron"}
_ALLOWED_TARGET_BUCKETS = {"popular", "unpopular", "all"}
_ALLOWED_EVAL_METRICS = {"precision", "recall", "mrr", "ndcg"}
_ALLOWED_POSITION_OPT_REWARD_MODES = {
    "poisoned_target_utility",
    "delta_target_utility",
    "delta_lowk_rank_utility",
}
_ALLOWED_POSITION_OPT_FINAL_POLICY_SELECTIONS = {
    "last",
    "best_deterministic",
}
_REQUIRED_SRGNN_TRAIN_KEYS = (
    "epochs",
    "batch_size",
    "hidden_size",
    "lr",
    "lr_dc",
    "lr_dc_step",
    "l2",
    "step",
    "patience",
    "nonhybrid",
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str


@dataclass(frozen=True)
class CanonicalSplitConfig:
    min_item_count: int
    min_session_len: int
    valid_ratio: float
    test_days: int


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    split_protocol: str
    poison_train_only: bool
    canonical_split: CanonicalSplitConfig


@dataclass(frozen=True)
class SeedsConfig:
    fake_session_seed: int
    target_selection_seed: int
    position_opt_seed: int
    surrogate_train_seed: int
    victim_train_seed: int


@dataclass(frozen=True)
class PoisonModelConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PositionOptConfig:
    clean_surrogate_checkpoint: str | None = None
    outer_steps: int = 30
    policy_lr: float = 0.05
    policy_embedding_dim: int = 16
    policy_hidden_dim: int = 32
    policy_feature_set: str = "local_context"
    nonzero_action_when_possible: bool = False
    fine_tune_steps: int = 20
    validation_subset_size: int | None = None
    reward_baseline_momentum: float = 0.9
    reward_mode: str = "poisoned_target_utility"
    entropy_coef: float = 0.0
    enable_gt_penalty: bool = False
    gt_penalty_weight: float = 0.0
    gt_tolerance: float = 0.0
    final_selection: str = "argmax"
    deterministic_eval_every: int = 0
    deterministic_eval_include_final: bool = True
    final_policy_selection: str = "last"

    def __post_init__(self) -> None:
        checkpoint = self.clean_surrogate_checkpoint
        if checkpoint is not None:
            checkpoint = _as_str(
                checkpoint,
                "attack.position_opt.clean_surrogate_checkpoint",
            ).strip()
            if not checkpoint:
                raise ValueError(
                    "attack.position_opt.clean_surrogate_checkpoint must be a non-empty "
                    "string when provided."
                )
            object.__setattr__(self, "clean_surrogate_checkpoint", checkpoint)

        outer_steps = _as_int(self.outer_steps, "attack.position_opt.outer_steps")
        if outer_steps < 0:
            raise ValueError("attack.position_opt.outer_steps must be non-negative.")
        object.__setattr__(self, "outer_steps", outer_steps)

        policy_lr = _as_float(self.policy_lr, "attack.position_opt.policy_lr")
        if policy_lr <= 0.0:
            raise ValueError("attack.position_opt.policy_lr must be positive.")
        object.__setattr__(self, "policy_lr", policy_lr)

        policy_embedding_dim = _as_int(
            self.policy_embedding_dim,
            "attack.position_opt.policy_embedding_dim",
        )
        if policy_embedding_dim <= 0:
            raise ValueError("attack.position_opt.policy_embedding_dim must be positive.")
        object.__setattr__(self, "policy_embedding_dim", policy_embedding_dim)

        policy_hidden_dim = _as_int(
            self.policy_hidden_dim,
            "attack.position_opt.policy_hidden_dim",
        )
        if policy_hidden_dim <= 0:
            raise ValueError("attack.position_opt.policy_hidden_dim must be positive.")
        object.__setattr__(self, "policy_hidden_dim", policy_hidden_dim)

        policy_feature_set = _as_str(
            self.policy_feature_set,
            "attack.position_opt.policy_feature_set",
        ).strip().lower()
        if policy_feature_set not in ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS:
            allowed_feature_sets = ", ".join(sorted(ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS))
            raise ValueError(
                "attack.position_opt.policy_feature_set must be one of: "
                f"{allowed_feature_sets}."
            )
        object.__setattr__(self, "policy_feature_set", policy_feature_set)

        nonzero_action_when_possible = _as_bool(
            self.nonzero_action_when_possible,
            "attack.position_opt.nonzero_action_when_possible",
        )
        object.__setattr__(
            self,
            "nonzero_action_when_possible",
            nonzero_action_when_possible,
        )

        fine_tune_steps = _as_int(
            self.fine_tune_steps,
            "attack.position_opt.fine_tune_steps",
        )
        if fine_tune_steps < 0:
            raise ValueError("attack.position_opt.fine_tune_steps must be non-negative.")
        object.__setattr__(self, "fine_tune_steps", fine_tune_steps)

        subset_size = self.validation_subset_size
        if subset_size is not None:
            subset_size = _as_int(
                subset_size,
                "attack.position_opt.validation_subset_size",
            )
            if subset_size <= 0:
                raise ValueError(
                    "attack.position_opt.validation_subset_size must be positive when provided."
                )
        object.__setattr__(self, "validation_subset_size", subset_size)

        momentum = _as_float(
            self.reward_baseline_momentum,
            "attack.position_opt.reward_baseline_momentum",
        )
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(
                "attack.position_opt.reward_baseline_momentum must be in [0, 1]."
            )
        object.__setattr__(self, "reward_baseline_momentum", momentum)

        reward_mode = _as_str(
            self.reward_mode,
            "attack.position_opt.reward_mode",
        ).strip().lower()
        if reward_mode not in _ALLOWED_POSITION_OPT_REWARD_MODES:
            allowed_modes = ", ".join(sorted(_ALLOWED_POSITION_OPT_REWARD_MODES))
            raise ValueError(
                "attack.position_opt.reward_mode must be one of: "
                f"{allowed_modes}."
            )
        object.__setattr__(self, "reward_mode", reward_mode)

        entropy_coef = _as_float(
            self.entropy_coef,
            "attack.position_opt.entropy_coef",
        )
        if entropy_coef < 0.0:
            raise ValueError("attack.position_opt.entropy_coef must be non-negative.")
        object.__setattr__(self, "entropy_coef", entropy_coef)

        enable_gt_penalty = _as_bool(
            self.enable_gt_penalty,
            "attack.position_opt.enable_gt_penalty",
        )
        object.__setattr__(self, "enable_gt_penalty", enable_gt_penalty)

        gt_penalty_weight = _as_float(
            self.gt_penalty_weight,
            "attack.position_opt.gt_penalty_weight",
        )
        if gt_penalty_weight < 0.0:
            raise ValueError(
                "attack.position_opt.gt_penalty_weight must be non-negative."
            )
        object.__setattr__(self, "gt_penalty_weight", gt_penalty_weight)

        gt_tolerance = _as_float(
            self.gt_tolerance,
            "attack.position_opt.gt_tolerance",
        )
        if gt_tolerance < 0.0:
            raise ValueError("attack.position_opt.gt_tolerance must be non-negative.")
        object.__setattr__(self, "gt_tolerance", gt_tolerance)

        final_selection = _as_str(
            self.final_selection,
            "attack.position_opt.final_selection",
        ).strip().lower()
        if final_selection != "argmax":
            raise ValueError(
                "attack.position_opt.final_selection must be 'argmax' for the current MVP."
            )
        object.__setattr__(self, "final_selection", final_selection)

        deterministic_eval_every = _as_int(
            self.deterministic_eval_every,
            "attack.position_opt.deterministic_eval_every",
        )
        if deterministic_eval_every < 0:
            raise ValueError(
                "attack.position_opt.deterministic_eval_every must be non-negative."
            )
        object.__setattr__(self, "deterministic_eval_every", deterministic_eval_every)

        deterministic_eval_include_final = _as_bool(
            self.deterministic_eval_include_final,
            "attack.position_opt.deterministic_eval_include_final",
        )
        object.__setattr__(
            self,
            "deterministic_eval_include_final",
            deterministic_eval_include_final,
        )

        final_policy_selection = _as_str(
            self.final_policy_selection,
            "attack.position_opt.final_policy_selection",
        ).strip().lower()
        if final_policy_selection not in _ALLOWED_POSITION_OPT_FINAL_POLICY_SELECTIONS:
            allowed_final_policy_selections = ", ".join(
                sorted(_ALLOWED_POSITION_OPT_FINAL_POLICY_SELECTIONS)
            )
            raise ValueError(
                "attack.position_opt.final_policy_selection must be one of: "
                f"{allowed_final_policy_selections}."
            )
        object.__setattr__(self, "final_policy_selection", final_policy_selection)


@dataclass(frozen=True)
class RankBucketCEMConfig:
    iterations: int = 3
    population_size: int = 8
    elite_ratio: float = 0.25
    initial_std: float = 1.0
    min_std: float = 0.2
    smoothing: float = 0.3
    reward_metric: str | None = None
    save_candidate_selected_positions: bool = False
    save_final_selected_positions: bool = False
    save_optimized_poisoned_sessions: bool = True
    save_replay_metadata: bool = True

    def __post_init__(self) -> None:
        iterations = _as_int(self.iterations, "attack.rank_bucket_cem.iterations")
        if iterations <= 0:
            raise ValueError("attack.rank_bucket_cem.iterations must be positive.")
        object.__setattr__(self, "iterations", iterations)

        population_size = _as_int(
            self.population_size,
            "attack.rank_bucket_cem.population_size",
        )
        if population_size <= 0:
            raise ValueError("attack.rank_bucket_cem.population_size must be positive.")
        object.__setattr__(self, "population_size", population_size)

        elite_ratio = _as_float(self.elite_ratio, "attack.rank_bucket_cem.elite_ratio")
        if not 0.0 < elite_ratio <= 1.0:
            raise ValueError("attack.rank_bucket_cem.elite_ratio must be in (0, 1].")
        object.__setattr__(self, "elite_ratio", elite_ratio)

        initial_std = _as_float(self.initial_std, "attack.rank_bucket_cem.initial_std")
        if initial_std <= 0.0:
            raise ValueError("attack.rank_bucket_cem.initial_std must be positive.")
        object.__setattr__(self, "initial_std", initial_std)

        min_std = _as_float(self.min_std, "attack.rank_bucket_cem.min_std")
        if min_std < 0.0:
            raise ValueError("attack.rank_bucket_cem.min_std must be non-negative.")
        object.__setattr__(self, "min_std", min_std)

        smoothing = _as_float(self.smoothing, "attack.rank_bucket_cem.smoothing")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("attack.rank_bucket_cem.smoothing must be in [0, 1].")
        object.__setattr__(self, "smoothing", smoothing)

        reward_metric = self.reward_metric
        if reward_metric is not None:
            reward_metric = _as_str(
                reward_metric,
                "attack.rank_bucket_cem.reward_metric",
            ).strip()
            if not reward_metric:
                raise ValueError(
                    "attack.rank_bucket_cem.reward_metric must be a non-empty string "
                    "when provided."
                )
        object.__setattr__(self, "reward_metric", reward_metric)

        object.__setattr__(
            self,
            "save_candidate_selected_positions",
            _as_bool(
                self.save_candidate_selected_positions,
                "attack.rank_bucket_cem.save_candidate_selected_positions",
            ),
        )
        object.__setattr__(
            self,
            "save_final_selected_positions",
            _as_bool(
                self.save_final_selected_positions,
                "attack.rank_bucket_cem.save_final_selected_positions",
            ),
        )
        object.__setattr__(
            self,
            "save_optimized_poisoned_sessions",
            _as_bool(
                self.save_optimized_poisoned_sessions,
                "attack.rank_bucket_cem.save_optimized_poisoned_sessions",
            ),
        )
        object.__setattr__(
            self,
            "save_replay_metadata",
            _as_bool(
                self.save_replay_metadata,
                "attack.rank_bucket_cem.save_replay_metadata",
            ),
        )


@dataclass(frozen=True)
class AttackConfig:
    size: float
    fake_session_generation_topk: int
    replacement_topk_ratio: float
    poison_model: PoisonModelConfig
    position_opt: PositionOptConfig | None = None
    rank_bucket_cem: RankBucketCEMConfig | None = None


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
    params: dict[str, dict[str, Any]]
    runtime: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    topk: tuple[int, ...]
    targeted_metrics: tuple[str, ...]
    ground_truth_metrics: tuple[str, ...]


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

    def to_primitive(self) -> dict[str, Any]:
        return _primitive_from_obj(self)

    def result_config_dict(self) -> dict[str, Any]:
        payload = self.to_primitive()
        victims = payload["victims"]
        return {
            "data": payload["data"],
            "seeds": payload["seeds"],
            "targets": payload["targets"],
            "attack": payload["attack"],
            "victims": {
                "enabled": victims["enabled"],
                "params": victims["params"],
            },
            "evaluation": payload["evaluation"],
        }

    def runtime_config_dict(self) -> dict[str, Any]:
        payload = self.to_primitive()
        return {
            "victims": {
                "runtime": payload["victims"]["runtime"],
            }
        }


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required field: {context}.{key}")
    return mapping[key]


def _as_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected {context} to be a mapping.")
    return value


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Expected {context} to be a string, got {type(value).__name__}")
    return value


def _as_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {context} to be an int, got {type(value).__name__}")
    return value


def _as_float(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected {context} to be a number, got {type(value).__name__}")
    return float(value)


def _as_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"Expected {context} to be a bool, got {type(value).__name__}")
    return value


def _as_gpu_id(value: Any, context: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"Expected {context} to be a string or int, got bool")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise TypeError(f"Expected {context} to be a non-empty string or int.")


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


def _unique_preserve_order(items: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    result: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _normalize_primitive(value: Any, context: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_normalize_primitive(item, f"{context}[]") for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Expected {context} keys to be strings.")
            normalized[key] = _normalize_primitive(item, f"{context}.{key}")
        return normalized
    raise TypeError(
        f"Unsupported value type for {context}: {type(value).__name__}"
    )


def _primitive_from_obj(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _primitive_from_obj(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_primitive_from_obj(item) for item in value]
    if isinstance(value, list):
        return [_primitive_from_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: _primitive_from_obj(item) for key, item in value.items()}
    return value


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


def parse_config(path: str | Path) -> dict[str, Any]:
    return parse_config_mapping(_load_yaml(Path(path)))


def parse_config_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    root = _as_mapping(data, "root")
    sections = {
        "experiment": _as_mapping(_require(root, "experiment", "root"), "experiment"),
        "data": _as_mapping(_require(root, "data", "root"), "data"),
        "seeds": _as_mapping(_require(root, "seeds", "root"), "seeds"),
        "attack": _as_mapping(_require(root, "attack", "root"), "attack"),
        "targets": _as_mapping(_require(root, "targets", "root"), "targets"),
        "victims": _as_mapping(_require(root, "victims", "root"), "victims"),
        "evaluation": _as_mapping(_require(root, "evaluation", "root"), "evaluation"),
        "artifacts": _as_mapping(root.get("artifacts", {}), "artifacts"),
    }
    if "seed" in sections["experiment"]:
        raise ValueError(
            "experiment.seed is not supported. Use seeds.fake_session_seed and "
            "seeds.target_selection_seed."
        )
    return {key: dict(value) for key, value in sections.items()}


def validate_config_mapping(data: Mapping[str, Any]) -> None:
    _normalize_config_mapping(data)


def normalize_config_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_config_mapping(data)


def _normalize_config_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    parsed = parse_config_mapping(data)

    experiment = parsed["experiment"]
    data_cfg = parsed["data"]
    seeds = parsed["seeds"]
    attack = parsed["attack"]
    targets = parsed["targets"]
    victims = parsed["victims"]
    evaluation = parsed["evaluation"]
    artifacts = parsed["artifacts"]

    normalized_canonical_split = _normalize_canonical_split(
        _require(data_cfg, "canonical_split", "data"),
        "data.canonical_split",
    )

    normalized_data = {
        "dataset_name": _as_str(
            _require(data_cfg, "dataset_name", "data"),
            "data.dataset_name",
        ),
        "split_protocol": _as_str(
            _require(data_cfg, "split_protocol", "data"),
            "data.split_protocol",
        ),
        "poison_train_only": _as_bool(
            _require(data_cfg, "poison_train_only", "data"),
            "data.poison_train_only",
        ),
        "canonical_split": normalized_canonical_split,
    }
    if normalized_data["split_protocol"] != "unified":
        raise ValueError("data.split_protocol must be 'unified'.")

    normalized_seeds = {
        "fake_session_seed": _as_int(
            _require(seeds, "fake_session_seed", "seeds"),
            "seeds.fake_session_seed",
        ),
        "target_selection_seed": _as_int(
            _require(seeds, "target_selection_seed", "seeds"),
            "seeds.target_selection_seed",
        ),
        "position_opt_seed": _as_int(
            seeds.get("position_opt_seed", _require(seeds, "fake_session_seed", "seeds")),
            "seeds.position_opt_seed",
        ),
        "surrogate_train_seed": _as_int(
            seeds.get("surrogate_train_seed", _require(seeds, "fake_session_seed", "seeds")),
            "seeds.surrogate_train_seed",
        ),
        "victim_train_seed": _as_int(
            seeds.get("victim_train_seed", _require(seeds, "fake_session_seed", "seeds")),
            "seeds.victim_train_seed",
        ),
    }

    normalized_attack = _normalize_attack_config(attack)
    normalized_targets = _normalize_targets_config(targets)
    normalized_victims = _normalize_victims_config(victims)
    normalized_evaluation = _normalize_evaluation_config(evaluation)

    normalized_artifacts = {
        "root": _as_str(artifacts.get("root", "outputs"), "artifacts.root"),
        "shared_dir": _as_str(
            artifacts.get("shared_dir", "shared"),
            "artifacts.shared_dir",
        ),
        "runs_dir": _as_str(
            artifacts.get("runs_dir", "runs"),
            "artifacts.runs_dir",
        ),
    }

    return {
        "experiment": {
            "name": _as_str(
                _require(experiment, "name", "experiment"),
                "experiment.name",
            ),
        },
        "data": normalized_data,
        "seeds": normalized_seeds,
        "attack": normalized_attack,
        "targets": normalized_targets,
        "victims": normalized_victims,
        "evaluation": normalized_evaluation,
        "artifacts": normalized_artifacts,
    }


def _normalize_canonical_split(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    normalized = {
        "min_item_count": _as_int(
            _require(mapping, "min_item_count", context),
            f"{context}.min_item_count",
        ),
        "min_session_len": _as_int(
            _require(mapping, "min_session_len", context),
            f"{context}.min_session_len",
        ),
        "valid_ratio": _as_float(
            _require(mapping, "valid_ratio", context),
            f"{context}.valid_ratio",
        ),
        "test_days": _as_int(
            _require(mapping, "test_days", context),
            f"{context}.test_days",
        ),
    }
    if normalized["min_item_count"] <= 0:
        raise ValueError(f"{context}.min_item_count must be positive.")
    if normalized["min_session_len"] < 2:
        raise ValueError(f"{context}.min_session_len must be at least 2.")
    if not 0.0 < normalized["valid_ratio"] < 1.0:
        raise ValueError(f"{context}.valid_ratio must be in (0, 1).")
    if normalized["test_days"] <= 0:
        raise ValueError(f"{context}.test_days must be positive.")
    return normalized


def _normalize_attack_config(attack: Mapping[str, Any]) -> dict[str, Any]:
    poison_model = _as_mapping(
        _require(attack, "poison_model", "attack"),
        "attack.poison_model",
    )
    poison_model_name = _as_str(
        _require(poison_model, "name", "attack.poison_model"),
        "attack.poison_model.name",
    ).lower()
    if poison_model_name != "srgnn":
        raise ValueError("attack.poison_model.name must be 'srgnn' for Batch 1.")

    normalized = {
        "size": _as_float(_require(attack, "size", "attack"), "attack.size"),
        "fake_session_generation_topk": _as_int(
            _require(attack, "fake_session_generation_topk", "attack"),
            "attack.fake_session_generation_topk",
        ),
        "replacement_topk_ratio": _as_float(
            _require(attack, "replacement_topk_ratio", "attack"),
            "attack.replacement_topk_ratio",
        ),
        "poison_model": {
            "name": poison_model_name,
            "params": _normalize_poison_model_params(
                _require(poison_model, "params", "attack.poison_model"),
                "attack.poison_model.params",
                model_name=poison_model_name,
            ),
        },
        "position_opt": (
            _normalize_position_opt_config(
                attack["position_opt"],
                "attack.position_opt",
            )
            if "position_opt" in attack and attack["position_opt"] is not None
            else None
        ),
        "rank_bucket_cem": (
            _normalize_rank_bucket_cem_config(
                attack["rank_bucket_cem"],
                "attack.rank_bucket_cem",
            )
            if "rank_bucket_cem" in attack and attack["rank_bucket_cem"] is not None
            else None
        ),
    }

    if not 0.0 < normalized["size"] <= 1.0:
        raise ValueError("attack.size must be in (0, 1].")
    if normalized["fake_session_generation_topk"] <= 0:
        raise ValueError("attack.fake_session_generation_topk must be positive.")
    if not 0.0 < normalized["replacement_topk_ratio"] <= 1.0:
        raise ValueError("attack.replacement_topk_ratio must be in (0, 1].")
    return normalized


def _normalize_position_opt_config(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {field.name for field in fields(PositionOptConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown position-opt config keys: " + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "clean_surrogate_checkpoint" in mapping:
        raw_checkpoint = mapping["clean_surrogate_checkpoint"]
        payload["clean_surrogate_checkpoint"] = (
            None
            if raw_checkpoint is None
            else _as_str(
                raw_checkpoint,
                f"{context}.clean_surrogate_checkpoint",
            )
        )
    if "outer_steps" in mapping:
        payload["outer_steps"] = _as_int(mapping["outer_steps"], f"{context}.outer_steps")
    if "policy_lr" in mapping:
        payload["policy_lr"] = _as_float(mapping["policy_lr"], f"{context}.policy_lr")
    if "policy_embedding_dim" in mapping:
        payload["policy_embedding_dim"] = _as_int(
            mapping["policy_embedding_dim"],
            f"{context}.policy_embedding_dim",
        )
    if "policy_hidden_dim" in mapping:
        payload["policy_hidden_dim"] = _as_int(
            mapping["policy_hidden_dim"],
            f"{context}.policy_hidden_dim",
        )
    if "policy_feature_set" in mapping:
        payload["policy_feature_set"] = _as_str(
            mapping["policy_feature_set"],
            f"{context}.policy_feature_set",
        )
    if "nonzero_action_when_possible" in mapping:
        payload["nonzero_action_when_possible"] = _as_bool(
            mapping["nonzero_action_when_possible"],
            f"{context}.nonzero_action_when_possible",
        )
    if "fine_tune_steps" in mapping:
        payload["fine_tune_steps"] = _as_int(
            mapping["fine_tune_steps"],
            f"{context}.fine_tune_steps",
        )
    if "validation_subset_size" in mapping:
        subset_size = mapping["validation_subset_size"]
        payload["validation_subset_size"] = (
            None
            if subset_size is None
            else _as_int(subset_size, f"{context}.validation_subset_size")
        )
    if "reward_baseline_momentum" in mapping:
        payload["reward_baseline_momentum"] = _as_float(
            mapping["reward_baseline_momentum"],
            f"{context}.reward_baseline_momentum",
        )
    if "reward_mode" in mapping:
        payload["reward_mode"] = _as_str(
            mapping["reward_mode"],
            f"{context}.reward_mode",
        )
    if "entropy_coef" in mapping:
        payload["entropy_coef"] = _as_float(
            mapping["entropy_coef"],
            f"{context}.entropy_coef",
        )
    if "enable_gt_penalty" in mapping:
        payload["enable_gt_penalty"] = _as_bool(
            mapping["enable_gt_penalty"],
            f"{context}.enable_gt_penalty",
        )
    if "gt_penalty_weight" in mapping:
        payload["gt_penalty_weight"] = _as_float(
            mapping["gt_penalty_weight"],
            f"{context}.gt_penalty_weight",
        )
    if "gt_tolerance" in mapping:
        payload["gt_tolerance"] = _as_float(
            mapping["gt_tolerance"],
            f"{context}.gt_tolerance",
        )
    if "final_selection" in mapping:
        payload["final_selection"] = _as_str(
            mapping["final_selection"],
            f"{context}.final_selection",
        )
    if "deterministic_eval_every" in mapping:
        payload["deterministic_eval_every"] = _as_int(
            mapping["deterministic_eval_every"],
            f"{context}.deterministic_eval_every",
        )
    if "deterministic_eval_include_final" in mapping:
        payload["deterministic_eval_include_final"] = _as_bool(
            mapping["deterministic_eval_include_final"],
            f"{context}.deterministic_eval_include_final",
        )
    if "final_policy_selection" in mapping:
        payload["final_policy_selection"] = _as_str(
            mapping["final_policy_selection"],
            f"{context}.final_policy_selection",
        )

    return _primitive_from_obj(PositionOptConfig(**payload))


def _normalize_rank_bucket_cem_config(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {field.name for field in fields(RankBucketCEMConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown rank_bucket_cem config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "iterations" in mapping:
        payload["iterations"] = _as_int(mapping["iterations"], f"{context}.iterations")
    if "population_size" in mapping:
        payload["population_size"] = _as_int(
            mapping["population_size"],
            f"{context}.population_size",
        )
    if "elite_ratio" in mapping:
        payload["elite_ratio"] = _as_float(
            mapping["elite_ratio"],
            f"{context}.elite_ratio",
        )
    if "initial_std" in mapping:
        payload["initial_std"] = _as_float(
            mapping["initial_std"],
            f"{context}.initial_std",
        )
    if "min_std" in mapping:
        payload["min_std"] = _as_float(
            mapping["min_std"],
            f"{context}.min_std",
        )
    if "smoothing" in mapping:
        payload["smoothing"] = _as_float(
            mapping["smoothing"],
            f"{context}.smoothing",
        )
    if "reward_metric" in mapping:
        raw_reward_metric = mapping["reward_metric"]
        payload["reward_metric"] = (
            None
            if raw_reward_metric is None
            else _as_str(raw_reward_metric, f"{context}.reward_metric")
        )
    if "save_candidate_selected_positions" in mapping:
        payload["save_candidate_selected_positions"] = _as_bool(
            mapping["save_candidate_selected_positions"],
            f"{context}.save_candidate_selected_positions",
        )
    if "save_final_selected_positions" in mapping:
        payload["save_final_selected_positions"] = _as_bool(
            mapping["save_final_selected_positions"],
            f"{context}.save_final_selected_positions",
        )
    if "save_optimized_poisoned_sessions" in mapping:
        payload["save_optimized_poisoned_sessions"] = _as_bool(
            mapping["save_optimized_poisoned_sessions"],
            f"{context}.save_optimized_poisoned_sessions",
        )
    if "save_replay_metadata" in mapping:
        payload["save_replay_metadata"] = _as_bool(
            mapping["save_replay_metadata"],
            f"{context}.save_replay_metadata",
        )

    return _primitive_from_obj(RankBucketCEMConfig(**payload))


def _normalize_poison_model_params(
    value: Any,
    context: str,
    *,
    model_name: str,
) -> dict[str, Any]:
    normalized = _normalize_primitive(value, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    if model_name == "srgnn":
        train = _as_mapping(_require(normalized, "train", context), f"{context}.train")
        return {
            **normalized,
            "train": _normalize_srgnn_train(train, f"{context}.train"),
        }
    raise ValueError(f"Unsupported poison model: {model_name}")


def _normalize_targets_config(targets: Mapping[str, Any]) -> dict[str, Any]:
    mode = _as_str(_require(targets, "mode", "targets"), "targets.mode")
    explicit_list = list(
        _unique_preserve_order(
            _as_int_list(targets.get("explicit_list", []), "targets.explicit_list")
        )
    )
    bucket = _as_str(targets.get("bucket", "popular"), "targets.bucket")
    count = _as_int(targets.get("count", 1), "targets.count")
    reuse_saved_targets = _as_bool(
        targets.get("reuse_saved_targets", True),
        "targets.reuse_saved_targets",
    )

    if mode not in {"explicit_list", "sampled"}:
        raise ValueError("targets.mode must be 'explicit_list' or 'sampled'.")
    if mode == "explicit_list" and not explicit_list:
        raise ValueError(
            "targets.explicit_list must be non-empty when mode is explicit_list."
        )
    if mode == "sampled":
        if bucket not in _ALLOWED_TARGET_BUCKETS:
            raise ValueError(
                "targets.bucket must be one of: popular, unpopular, all."
            )
        if count <= 0:
            raise ValueError("targets.count must be positive when mode is sampled.")

    return {
        "mode": mode,
        "explicit_list": explicit_list,
        "bucket": bucket,
        "count": count,
        "reuse_saved_targets": reuse_saved_targets,
    }


def _normalize_victims_config(victims: Mapping[str, Any]) -> dict[str, Any]:
    enabled = list(
        _as_str_list(_require(victims, "enabled", "victims"), "victims.enabled")
    )
    if not enabled:
        raise ValueError("victims.enabled must include at least one victim model.")
    if len(set(enabled)) != len(enabled):
        raise ValueError("victims.enabled must not contain duplicates.")
    if not set(enabled).issubset(_ALLOWED_VICTIMS):
        raise ValueError(
            f"victims.enabled must be subset of {sorted(_ALLOWED_VICTIMS)}, got {enabled}"
        )

    params = _normalize_victim_params(
        _require(victims, "params", "victims"),
        "victims.params",
    )
    for victim_name in enabled:
        if victim_name not in params:
            raise ValueError(
                f"Missing required configuration: victims.params.{victim_name}"
            )

    runtime_value = victims.get("runtime")
    runtime = _normalize_victim_runtime(runtime_value) if runtime_value is not None else None

    if "miasrec" in enabled:
        if runtime is None or "miasrec" not in runtime:
            raise ValueError("Missing required runtime configuration: victims.runtime.miasrec")
        _validate_miasrec_runtime(runtime["miasrec"], "victims.runtime.miasrec")
    if "tron" in enabled:
        if runtime is None or "tron" not in runtime:
            raise ValueError("Missing required runtime configuration: victims.runtime.tron")
        _validate_tron_runtime(runtime["tron"], "victims.runtime.tron")

    return {
        "enabled": enabled,
        "params": params,
        "runtime": runtime,
    }


def _normalize_victim_params(value: Any, context: str) -> dict[str, dict[str, Any]]:
    mapping = _as_mapping(value, context)
    normalized: dict[str, dict[str, Any]] = {}
    for victim_name, victim_params in mapping.items():
        if not isinstance(victim_name, str):
            raise TypeError(f"Expected {context} keys to be victim names.")
        if victim_name not in _ALLOWED_VICTIMS:
            raise ValueError(
                f"{context} keys must be subset of {sorted(_ALLOWED_VICTIMS)}, got {victim_name!r}"
            )
        victim_mapping = _as_mapping(victim_params, f"{context}.{victim_name}")
        primitive = _normalize_primitive(victim_mapping, f"{context}.{victim_name}")
        if not isinstance(primitive, dict):
            raise TypeError(f"Expected {context}.{victim_name} to be a mapping.")
        train = _as_mapping(
            _require(primitive, "train", f"{context}.{victim_name}"),
            f"{context}.{victim_name}.train",
        )
        if victim_name == "srgnn":
            primitive["train"] = _normalize_srgnn_train(train, f"{context}.{victim_name}.train")
        elif victim_name == "miasrec":
            primitive["train"] = _normalize_miasrec_train(train, f"{context}.{victim_name}.train")
        elif victim_name == "tron":
            primitive["train"] = _normalize_tron_train(train, f"{context}.{victim_name}.train")
        normalized[victim_name] = primitive
    return normalized


def _normalize_victim_runtime(value: Any) -> dict[str, dict[str, Any]]:
    mapping = _as_mapping(value, "victims.runtime")
    normalized: dict[str, dict[str, Any]] = {}
    for victim_name, victim_runtime in mapping.items():
        if not isinstance(victim_name, str):
            raise TypeError("Expected victims.runtime keys to be victim names.")
        if victim_name not in _ALLOWED_VICTIMS:
            raise ValueError(
                f"victims.runtime keys must be subset of {sorted(_ALLOWED_VICTIMS)}, got {victim_name!r}"
            )
        primitive = _normalize_primitive(
            _as_mapping(victim_runtime, f"victims.runtime.{victim_name}"),
            f"victims.runtime.{victim_name}",
        )
        if not isinstance(primitive, dict):
            raise TypeError(f"Expected victims.runtime.{victim_name} to be a mapping.")
        normalized[victim_name] = primitive
    return normalized


def _validate_miasrec_runtime(runtime: dict[str, Any], context: str) -> None:
    _as_str(_require(runtime, "python_executable", context), f"{context}.python_executable")
    _as_str(_require(runtime, "repo_root", context), f"{context}.repo_root")
    _as_str(_require(runtime, "working_dir", context), f"{context}.working_dir")
    device = _as_mapping(_require(runtime, "device", context), f"{context}.device")
    _as_bool(_require(device, "use_gpu", f"{context}.device"), f"{context}.device.use_gpu")
    _as_gpu_id(_require(device, "gpu_id", f"{context}.device"), f"{context}.device.gpu_id")
    logging = _as_mapping(_require(runtime, "logging", context), f"{context}.logging")
    _as_bool(
        _require(logging, "show_progress", f"{context}.logging"),
        f"{context}.logging.show_progress",
    )


def _validate_tron_runtime(runtime: dict[str, Any], context: str) -> None:
    _as_str(_require(runtime, "python_executable", context), f"{context}.python_executable")
    _as_str(_require(runtime, "repo_root", context), f"{context}.repo_root")
    _as_str(_require(runtime, "working_dir", context), f"{context}.working_dir")
    device = _as_mapping(_require(runtime, "device", context), f"{context}.device")
    _as_bool(_require(device, "use_gpu", f"{context}.device"), f"{context}.device.use_gpu")
    _as_gpu_id(_require(device, "gpu_id", f"{context}.device"), f"{context}.device.gpu_id")
    dataloader = _as_mapping(
        _require(runtime, "dataloader", context),
        f"{context}.dataloader",
    )
    num_workers = _as_int(
        _require(dataloader, "num_workers", f"{context}.dataloader"),
        f"{context}.dataloader.num_workers",
    )
    if num_workers < 0:
        raise ValueError(f"{context}.dataloader.num_workers must be non-negative.")


def _normalize_srgnn_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    for key in _REQUIRED_SRGNN_TRAIN_KEYS:
        _require(normalized, key, context)
    normalized["epochs"] = _as_int(normalized["epochs"], f"{context}.epochs")
    normalized["batch_size"] = _as_int(normalized["batch_size"], f"{context}.batch_size")
    normalized["hidden_size"] = _as_int(normalized["hidden_size"], f"{context}.hidden_size")
    normalized["lr"] = _as_float(normalized["lr"], f"{context}.lr")
    normalized["lr_dc"] = _as_float(normalized["lr_dc"], f"{context}.lr_dc")
    normalized["lr_dc_step"] = _as_int(normalized["lr_dc_step"], f"{context}.lr_dc_step")
    normalized["l2"] = _as_float(normalized["l2"], f"{context}.l2")
    normalized["step"] = _as_int(normalized["step"], f"{context}.step")
    normalized["patience"] = _as_int(normalized["patience"], f"{context}.patience")
    normalized["nonhybrid"] = _as_bool(normalized["nonhybrid"], f"{context}.nonhybrid")

    if normalized["epochs"] <= 0:
        raise ValueError(f"{context}.epochs must be positive.")
    if normalized["batch_size"] <= 0:
        raise ValueError(f"{context}.batch_size must be positive.")
    if normalized["hidden_size"] <= 0:
        raise ValueError(f"{context}.hidden_size must be positive.")
    if normalized["lr"] <= 0:
        raise ValueError(f"{context}.lr must be positive.")
    if normalized["lr_dc"] <= 0:
        raise ValueError(f"{context}.lr_dc must be positive.")
    if normalized["lr_dc_step"] <= 0:
        raise ValueError(f"{context}.lr_dc_step must be positive.")
    if normalized["l2"] < 0:
        raise ValueError(f"{context}.l2 must be non-negative.")
    if normalized["step"] <= 0:
        raise ValueError(f"{context}.step must be positive.")
    if normalized["patience"] <= 0:
        raise ValueError(f"{context}.patience must be positive.")
    return normalized


def _normalize_miasrec_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    required = ("epochs", "train_batch_size", "eval_batch_size")
    for key in required:
        _require(normalized, key, context)
    normalized["epochs"] = _as_int(normalized["epochs"], f"{context}.epochs")
    normalized["train_batch_size"] = _as_int(
        normalized["train_batch_size"],
        f"{context}.train_batch_size",
    )
    normalized["eval_batch_size"] = _as_int(
        normalized["eval_batch_size"],
        f"{context}.eval_batch_size",
    )
    if normalized["epochs"] <= 0:
        raise ValueError(f"{context}.epochs must be positive.")
    if normalized["train_batch_size"] <= 0:
        raise ValueError(f"{context}.train_batch_size must be positive.")
    if normalized["eval_batch_size"] <= 0:
        raise ValueError(f"{context}.eval_batch_size must be positive.")
    return normalized


def _normalize_tron_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    _require(normalized, "max_epochs", context)
    normalized["max_epochs"] = _as_int(normalized["max_epochs"], f"{context}.max_epochs")
    if normalized["max_epochs"] <= 0:
        raise ValueError(f"{context}.max_epochs must be positive.")
    return normalized


def _normalize_evaluation_config(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    raw_topk = _as_int_list(_require(evaluation, "topk", "evaluation"), "evaluation.topk")
    if not raw_topk:
        raise ValueError("evaluation.topk must include at least one K value.")
    normalized_topk = _unique_preserve_order(raw_topk)
    if any(k <= 0 for k in normalized_topk):
        raise ValueError("evaluation.topk values must be positive integers.")

    raw_targeted_metrics = _as_str_list(
        evaluation.get("targeted_metrics", []),
        "evaluation.targeted_metrics",
    )
    raw_ground_truth_metrics = _as_str_list(
        evaluation.get("ground_truth_metrics", []),
        "evaluation.ground_truth_metrics",
    )
    if not raw_targeted_metrics and not raw_ground_truth_metrics:
        raise ValueError(
            "evaluation.targeted_metrics and evaluation.ground_truth_metrics "
            "cannot both be empty."
        )
    normalized_targeted_metrics = _unique_preserve_order(raw_targeted_metrics)
    normalized_ground_truth_metrics = _unique_preserve_order(raw_ground_truth_metrics)
    if not set(normalized_targeted_metrics).issubset(_ALLOWED_EVAL_METRICS):
        raise ValueError(
            "evaluation.targeted_metrics must be a subset of: "
            "precision, recall, mrr, ndcg."
        )
    if not set(normalized_ground_truth_metrics).issubset(_ALLOWED_EVAL_METRICS):
        raise ValueError(
            "evaluation.ground_truth_metrics must be a subset of: "
            "precision, recall, mrr, ndcg."
        )

    return {
        "topk": list(normalized_topk),
        "targeted_metrics": list(normalized_targeted_metrics),
        "ground_truth_metrics": list(normalized_ground_truth_metrics),
    }


def _build_config(normalized: Mapping[str, Any]) -> Config:
    experiment = _as_mapping(_require(normalized, "experiment", "root"), "experiment")
    data_cfg = _as_mapping(_require(normalized, "data", "root"), "data")
    seeds = _as_mapping(_require(normalized, "seeds", "root"), "seeds")
    attack = _as_mapping(_require(normalized, "attack", "root"), "attack")
    targets = _as_mapping(_require(normalized, "targets", "root"), "targets")
    victims = _as_mapping(_require(normalized, "victims", "root"), "victims")
    evaluation = _as_mapping(_require(normalized, "evaluation", "root"), "evaluation")
    artifacts = _as_mapping(_require(normalized, "artifacts", "root"), "artifacts")

    canonical_split = _as_mapping(
        _require(data_cfg, "canonical_split", "data"),
        "data.canonical_split",
    )

    return Config(
        experiment=ExperimentConfig(
            name=_as_str(_require(experiment, "name", "experiment"), "experiment.name"),
        ),
        data=DataConfig(
            dataset_name=_as_str(
                _require(data_cfg, "dataset_name", "data"),
                "data.dataset_name",
            ),
            split_protocol=_as_str(
                _require(data_cfg, "split_protocol", "data"),
                "data.split_protocol",
            ),
            poison_train_only=_as_bool(
                _require(data_cfg, "poison_train_only", "data"),
                "data.poison_train_only",
            ),
            canonical_split=CanonicalSplitConfig(
                min_item_count=_as_int(
                    _require(canonical_split, "min_item_count", "data.canonical_split"),
                    "data.canonical_split.min_item_count",
                ),
                min_session_len=_as_int(
                    _require(canonical_split, "min_session_len", "data.canonical_split"),
                    "data.canonical_split.min_session_len",
                ),
                valid_ratio=_as_float(
                    _require(canonical_split, "valid_ratio", "data.canonical_split"),
                    "data.canonical_split.valid_ratio",
                ),
                test_days=_as_int(
                    _require(canonical_split, "test_days", "data.canonical_split"),
                    "data.canonical_split.test_days",
                ),
            ),
        ),
        seeds=SeedsConfig(
            fake_session_seed=_as_int(
                _require(seeds, "fake_session_seed", "seeds"),
                "seeds.fake_session_seed",
            ),
            target_selection_seed=_as_int(
                _require(seeds, "target_selection_seed", "seeds"),
                "seeds.target_selection_seed",
            ),
            position_opt_seed=_as_int(
                _require(seeds, "position_opt_seed", "seeds"),
                "seeds.position_opt_seed",
            ),
            surrogate_train_seed=_as_int(
                _require(seeds, "surrogate_train_seed", "seeds"),
                "seeds.surrogate_train_seed",
            ),
            victim_train_seed=_as_int(
                _require(seeds, "victim_train_seed", "seeds"),
                "seeds.victim_train_seed",
            ),
        ),
        attack=AttackConfig(
            size=_as_float(_require(attack, "size", "attack"), "attack.size"),
            fake_session_generation_topk=_as_int(
                _require(attack, "fake_session_generation_topk", "attack"),
                "attack.fake_session_generation_topk",
            ),
            replacement_topk_ratio=_as_float(
                _require(attack, "replacement_topk_ratio", "attack"),
                "attack.replacement_topk_ratio",
            ),
            poison_model=PoisonModelConfig(
                name=_as_str(
                    _require(
                        _as_mapping(
                            _require(attack, "poison_model", "attack"),
                            "attack.poison_model",
                        ),
                        "name",
                        "attack.poison_model",
                    ),
                    "attack.poison_model.name",
                ),
                params=_normalize_primitive(
                    _require(
                        _as_mapping(
                            _require(attack, "poison_model", "attack"),
                            "attack.poison_model",
                        ),
                        "params",
                        "attack.poison_model",
                    ),
                    "attack.poison_model.params",
                ),
            ),
            position_opt=(
                PositionOptConfig(
                    **dict(
                        _as_mapping(
                            attack["position_opt"],
                            "attack.position_opt",
                        )
                    )
                )
                if attack.get("position_opt") is not None
                else None
            ),
            rank_bucket_cem=(
                RankBucketCEMConfig(
                    **dict(
                        _as_mapping(
                            attack["rank_bucket_cem"],
                            "attack.rank_bucket_cem",
                        )
                    )
                )
                if attack.get("rank_bucket_cem") is not None
                else None
            ),
        ),
        targets=TargetsConfig(
            mode=_as_str(_require(targets, "mode", "targets"), "targets.mode"),
            explicit_list=tuple(
                _as_int_list(targets.get("explicit_list", []), "targets.explicit_list")
            ),
            bucket=_as_str(targets.get("bucket", "popular"), "targets.bucket"),
            count=_as_int(targets.get("count", 1), "targets.count"),
            reuse_saved_targets=_as_bool(
                targets.get("reuse_saved_targets", True),
                "targets.reuse_saved_targets",
            ),
        ),
        victims=VictimsConfig(
            enabled=tuple(
                _as_str_list(_require(victims, "enabled", "victims"), "victims.enabled")
            ),
            params=_normalize_primitive(
                _require(victims, "params", "victims"),
                "victims.params",
            ),
            runtime=(
                _normalize_primitive(victims.get("runtime"), "victims.runtime")
                if victims.get("runtime") is not None
                else None
            ),
        ),
        evaluation=EvaluationConfig(
            topk=tuple(_as_int_list(_require(evaluation, "topk", "evaluation"), "evaluation.topk")),
            targeted_metrics=tuple(
                _as_str_list(
                    evaluation.get("targeted_metrics", []),
                    "evaluation.targeted_metrics",
                )
            ),
            ground_truth_metrics=tuple(
                _as_str_list(
                    evaluation.get("ground_truth_metrics", []),
                    "evaluation.ground_truth_metrics",
                )
            ),
        ),
        artifacts=ArtifactsConfig(
            root=_as_str(artifacts.get("root", "outputs"), "artifacts.root"),
            shared_dir=_as_str(artifacts.get("shared_dir", "shared"), "artifacts.shared_dir"),
            runs_dir=_as_str(artifacts.get("runs_dir", "runs"), "artifacts.runs_dir"),
        ),
    )


def load_config(path: str | Path) -> Config:
    parsed = parse_config(path)
    normalized = normalize_config_mapping(parsed)
    return _build_config(normalized)


__all__ = [
    "CanonicalSplitConfig",
    "Config",
    "PositionOptConfig",
    "RankBucketCEMConfig",
    "load_config",
    "normalize_config_mapping",
    "parse_config",
    "parse_config_mapping",
    "validate_config_mapping",
]
