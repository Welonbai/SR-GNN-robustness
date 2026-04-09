from __future__ import annotations

from pathlib import Path
import hashlib
import json

from .config import Config


def output_root(config: Config) -> Path:
    return Path(config.artifacts.root)


def dataset_name(config: Config) -> str:
    return config.data.dataset_name or "dataset"


def dataset_root(config: Config) -> Path:
    return Path("datasets") / dataset_name(config)


def dataset_paths(config: Config) -> dict[str, Path]:
    base = dataset_root(config)
    return {
        "train": base / "train.txt",
        "test": base / "test.txt",
        "all_train_seq": base / "all_train_seq.txt",
    }


def _format_float_token(value: float) -> str:
    token = f"{value:g}"
    return token.replace(".", "p")


def _hash_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _target_config_token(config: Config) -> str:
    if config.targets.mode == "explicit_list":
        joined = ",".join(str(item) for item in config.targets.explicit_list)
        digest = _hash_token(joined or "empty")
        return f"explicit_{len(config.targets.explicit_list)}_{digest}"
    return f"sampled_{config.targets.bucket}_{config.targets.count}"


def _fake_session_key(config: Config) -> str:
    size_token = _format_float_token(config.attack.size)
    return "_".join(
        [
            f"split_{config.data.split_protocol}",
            f"poison_{config.attack.poison_model.name}",
            f"size_{size_token}",
            f"topk_{config.attack.fake_session_generation_topk}",
            f"fs_{config.seeds.fake_session_seed}",
        ]
    )


def _target_selection_key(config: Config) -> str:
    return "_".join(
        [
            f"split_{config.data.split_protocol}",
            f"ts_{config.seeds.target_selection_seed}",
            f"targets_{_target_config_token(config)}",
        ]
    )


def shared_root(config: Config) -> Path:
    return output_root(config) / config.artifacts.shared_dir / dataset_name(config)


def shared_attack_dir(config: Config) -> Path:
    return shared_root(config) / "attack" / _fake_session_key(config)


def target_selection_dir(config: Config) -> Path:
    return shared_root(config) / "targets" / _target_selection_key(config)


def canonical_split_dir(config: Config, *, split_key: str) -> Path:
    return shared_root(config) / "canonical" / split_key


def canonical_split_paths(config: Config, *, split_key: str) -> dict[str, Path]:
    base = canonical_split_dir(config, split_key=split_key)
    return {
        "canonical_dir": base,
        "metadata": base / "metadata.json",
        "item_map": base / "item_map.pkl",
        "train_sub": base / "train_sub.pkl",
        "valid": base / "valid.pkl",
        "test": base / "test.pkl",
    }


def _run_config_key(config: Config) -> str:
    victim_token = _hash_token(",".join(sorted(config.victims.enabled)))
    metrics_token = _hash_token(",".join(sorted(config.evaluation.metrics)))
    tokens = [
        f"split_{config.data.split_protocol}",
        f"poison_train_only_{int(config.data.poison_train_only)}",
        f"poison_{config.attack.poison_model.name}",
        f"size_{_format_float_token(config.attack.size)}",
        f"topk_{config.attack.fake_session_generation_topk}",
        f"replace_{_format_float_token(config.attack.replacement_topk_ratio)}",
        f"fs_{config.seeds.fake_session_seed}",
        f"ts_{config.seeds.target_selection_seed}",
        f"targets_{_target_config_token(config)}",
        f"victims_{victim_token}",
        f"eval_{config.evaluation.topk}_{metrics_token}",
    ]
    payload = json.dumps(tokens)
    return f"run_{_hash_token(payload)}"


def runs_root(config: Config) -> Path:
    return output_root(config) / config.artifacts.runs_dir / dataset_name(config) / config.experiment.name


def run_config_dir(config: Config) -> Path:
    return runs_root(config) / _run_config_key(config)


def target_dir(config: Config, target_id: str | int) -> Path:
    return run_config_dir(config) / "targets" / str(target_id)


def _primary_victim(config: Config) -> str:
    return config.victims.enabled[0] if config.victims.enabled else "srgnn"


def victim_dir(config: Config, target_id: str | int, victim_name: str | None = None) -> Path:
    victim = victim_name or _primary_victim(config)
    return target_dir(config, target_id) / "victims" / victim


def shared_artifact_paths(config: Config) -> dict[str, Path]:
    attack_dir = shared_attack_dir(config)
    target_dir = target_selection_dir(config)
    return {
        "attack_shared_dir": attack_dir,
        "attack_config_snapshot": attack_dir / "config.yaml",
        "poison_model": attack_dir / "poison_model.pt",
        "fake_sessions": attack_dir / "fake_sessions.pkl",
        "target_shared_dir": target_dir,
        "target_config_snapshot": target_dir / "config.yaml",
        "target_info": target_dir / "target_info.json",
    }


def run_artifact_paths(
    config: Config,
    *,
    target_id: str | int,
    victim_name: str | None = None,
) -> dict[str, Path]:
    base = victim_dir(config, target_id, victim_name=victim_name)
    return {
        "run_dir": base,
        "config_snapshot": base / "config.yaml",
        "metrics": base / "metrics.json",
        "poisoned_train": base / "poisoned_train.txt",
        "best_position_metadata": base / "best_position_metadata.pkl",
        "dpsbr_position_metadata": base / "dpsbr_position_metadata.json",
        "random_nonzero_position_metadata": base / "random_nonzero_position_metadata.json",
    }


__all__ = [
    "output_root",
    "dataset_name",
    "dataset_root",
    "dataset_paths",
    "shared_root",
    "shared_attack_dir",
    "target_selection_dir",
    "canonical_split_dir",
    "canonical_split_paths",
    "runs_root",
    "run_config_dir",
    "target_dir",
    "victim_dir",
    "shared_artifact_paths",
    "run_artifact_paths",
]
