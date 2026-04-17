from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

from .config import Config


POSITION_OPT_RUN_TYPE = "position_opt_mvp"


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


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _hash_token(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _normalize_identity_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalize_identity_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_identity_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_identity_value(item) for item in value]
    return value


def checkpoint_identity_payload(checkpoint_path: str | Path) -> dict[str, Any]:
    """Return a content-based identity for an explicit external checkpoint.

    This helper is for downstream attack-result identity only. It must not be
    used to key split caches, target caches, or shared fake-session generation
    caches.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found for identity hashing: {path}")

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "type": "file_sha1",
        "sha1": digest.hexdigest(),
    }


def build_position_opt_attack_identity_context(
    *,
    position_opt_config: Mapping[str, Any],
    clean_surrogate_checkpoint: str | Path,
) -> dict[str, Any]:
    """Build the position-opt-specific runtime identity payload.

    The final position-opt poisoned result can change when either the outer-loop
    settings or the clean surrogate checkpoint changes, so both belong in the
    final attack identity layer.
    """
    return {
        "position_opt": {
            "config": _normalize_identity_value(position_opt_config),
            "clean_surrogate": checkpoint_identity_payload(clean_surrogate_checkpoint),
        }
    }


def split_key_payload(config: Config) -> dict[str, Any]:
    # Canonical dataset cache is split-only. It must not depend on targets,
    # attack settings, victims, evaluation, or position-opt runtime overrides.
    split_cfg = config.data.canonical_split
    return {
        "dataset_name": config.data.dataset_name,
        "split_protocol": config.data.split_protocol,
        "poison_train_only": bool(config.data.poison_train_only),
        "canonical_split": {
            "min_item_count": int(split_cfg.min_item_count),
            "min_session_len": int(split_cfg.min_session_len),
            "valid_ratio": float(split_cfg.valid_ratio),
            "test_days": int(split_cfg.test_days),
        },
    }


def split_key(config: Config) -> str:
    split_cfg = config.data.canonical_split
    ratio_token = f"{float(split_cfg.valid_ratio):.4f}".rstrip("0").rstrip(".")
    ratio_token = ratio_token.replace(".", "p")
    return (
        f"split_{config.data.dataset_name.lower()}"
        f"_{config.data.split_protocol}"
        f"_trainonly{int(bool(config.data.poison_train_only))}"
        f"_minitems{int(split_cfg.min_item_count)}"
        f"_minsess{int(split_cfg.min_session_len)}"
        f"_testdays{int(split_cfg.test_days)}"
        f"_valid{ratio_token}"
    )


def target_selection_key_payload(config: Config) -> dict[str, Any]:
    # Target selection cache is target-choice-only. It depends on the split and
    # the target sampling/selection settings, but not on downstream attacks.
    return {
        "split_key": split_key(config),
        "targets": {
            "mode": config.targets.mode,
            "explicit_list": [int(item) for item in config.targets.explicit_list],
            "bucket": config.targets.bucket,
            "count": int(config.targets.count),
            "reuse_saved_targets": bool(config.targets.reuse_saved_targets),
        },
        "target_selection_seed": int(config.seeds.target_selection_seed),
    }


def target_selection_key(config: Config) -> str:
    return f"targets_{_hash_token(_stable_json(target_selection_key_payload(config)))}"


def attack_key_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Final attack identity is downstream-facing. It intentionally includes
    # replacement-policy semantics and, for position-opt runs, the runtime
    # settings that can change the optimized poisoned result.
    if run_type == "clean":
        return {
            "run_type": "clean",
            "split_key": split_key(config),
        }
    payload = {
        "split_key": split_key(config),
        "target_selection_key": target_selection_key(config),
        "fake_session_seed": int(config.seeds.fake_session_seed),
        "attack": {
            "size": float(config.attack.size),
            "fake_session_generation_topk": int(config.attack.fake_session_generation_topk),
            "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
            "poison_model": {
                "name": config.attack.poison_model.name,
                "params": config.attack.poison_model.params,
            },
        },
    }
    if run_type == POSITION_OPT_RUN_TYPE:
        if attack_identity_context is None:
            raise ValueError(
                "position_opt_mvp final attack identity requires explicit "
                "attack_identity_context with position-opt settings and clean "
                "surrogate identity."
            )
        payload["attack_runtime_identity"] = _normalize_identity_value(attack_identity_context)
    return payload


def attack_key(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> str:
    payload = attack_key_payload(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return f"attack_{_hash_token(_stable_json(payload))}"


def shared_attack_artifact_key_payload(config: Config, *, run_type: str) -> dict[str, Any]:
    if run_type == "clean":
        return {
            "run_type": "clean",
            "split_key": split_key(config),
        }
    # Shared generation cache is generation-only: fake-session templates and the
    # poison model used to generate them. It must not depend on target choice,
    # replacement policy, victim settings, or position-opt runtime overrides.
    return {
        "split_key": split_key(config),
        "fake_session_seed": int(config.seeds.fake_session_seed),
        "attack_generation": {
            "size": float(config.attack.size),
            "fake_session_generation_topk": int(config.attack.fake_session_generation_topk),
            "poison_model": {
                "name": config.attack.poison_model.name,
                "params": config.attack.poison_model.params,
            },
        },
    }


def shared_attack_artifact_key(config: Config, *, run_type: str) -> str:
    payload = shared_attack_artifact_key_payload(config, run_type=run_type)
    return f"attack_shared_{_hash_token(_stable_json(payload))}"


def victim_prediction_key_payload(
    config: Config,
    victim_name: str,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if run_type == "clean":
        base_context: dict[str, Any] = {
            "run_type": "clean",
            "split_key": split_key(config),
        }
    else:
        base_context = {
            "run_type": run_type,
            "attack_key": attack_key(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        }
    # Victim prediction identity covers settings that can change victim training
    # results. Runtime-only fields stay excluded because they live under
    # victims.runtime rather than victims.params.
    victim_params = config.victims.params[victim_name]
    return {
        **base_context,
        "victim_name": victim_name,
        "victim_params": victim_params,
    }


def victim_prediction_key(
    config: Config,
    victim_name: str,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> str:
    payload = victim_prediction_key_payload(
        config,
        victim_name,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return f"victim_{victim_name}_{_hash_token(_stable_json(payload))}"


def evaluation_key_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # Final reporting identity composes the upstream final attack identity, the
    # victim-training-result identities, and the evaluation metric settings.
    return {
        "run_type": run_type,
        "target_selection_key": target_selection_key(config),
        "victim_prediction_keys": {
            victim_name: victim_prediction_key(
                config,
                victim_name,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            )
            for victim_name in config.victims.enabled
        },
        "evaluation": {
            "topk": [int(k) for k in config.evaluation.topk],
            "targeted_metrics": list(config.evaluation.targeted_metrics),
            "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
        },
    }


def evaluation_key(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> str:
    payload = evaluation_key_payload(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return f"eval_{_hash_token(_stable_json(payload))}"


def shared_root(config: Config) -> Path:
    return output_root(config) / config.artifacts.shared_dir / dataset_name(config)


def canonical_split_dir(config: Config, *, split_key_value: str | None = None) -> Path:
    key = split_key_value or split_key(config)
    return shared_root(config) / "canonical" / key


def canonical_split_paths(
    config: Config,
    *,
    split_key: str | None = None,
) -> dict[str, Path]:
    base = canonical_split_dir(config, split_key_value=split_key)
    return {
        "canonical_dir": base,
        "metadata": base / "metadata.json",
        "item_map": base / "item_map.pkl",
        "train_sub": base / "train_sub.pkl",
        "valid": base / "valid.pkl",
        "test": base / "test.pkl",
    }


def target_selection_dir(config: Config) -> Path:
    return shared_root(config) / "targets" / target_selection_key(config)


def shared_attack_dir(config: Config, *, run_type: str) -> Path:
    # Shared fake-session / poison-model artifacts should only depend on the
    # inputs that actually affect their generation. Final attack/evaluation keys
    # still use attack_key(...), which may include downstream replacement-policy
    # settings such as replacement_topk_ratio.
    return shared_root(config) / "attack" / shared_attack_artifact_key(config, run_type=run_type)


def shared_victim_dir(
    config: Config,
    *,
    run_type: str,
    target_id: str | int,
    victim_name: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    return (
        shared_root(config)
        / "victim_predictions"
        / victim_name
        / victim_prediction_key(
            config,
            victim_name,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        )
        / "targets"
        / str(target_id)
    )


def runs_root(config: Config) -> Path:
    return (
        output_root(config)
        / config.artifacts.runs_dir
        / dataset_name(config)
        / config.experiment.name
    )


def run_config_dir(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    return runs_root(config) / evaluation_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )


def target_dir(
    config: Config,
    target_id: str | int,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    return (
        run_config_dir(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        )
        / "targets"
        / str(target_id)
    )


def _primary_victim(config: Config) -> str:
    return config.victims.enabled[0] if config.victims.enabled else "srgnn"


def victim_dir(
    config: Config,
    target_id: str | int,
    *,
    run_type: str,
    victim_name: str | None = None,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    victim = victim_name or _primary_victim(config)
    return (
        target_dir(
            config,
            target_id,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        )
        / "victims"
        / victim
    )


def shared_artifact_paths(config: Config, *, run_type: str) -> dict[str, Path]:
    attack_dir = shared_attack_dir(config, run_type=run_type)
    target_dir_path = target_selection_dir(config)
    return {
        "attack_shared_dir": attack_dir,
        "attack_config_snapshot": attack_dir / "config.yaml",
        "poison_model": attack_dir / "poison_model.pt",
        "fake_sessions": attack_dir / "fake_sessions.pkl",
        "poison_train_history": attack_dir / "poison_train_history.json",
        "target_shared_dir": target_dir_path,
        "target_config_snapshot": target_dir_path / "config.yaml",
        "selected_targets": target_dir_path / "selected_targets.json",
        "target_selection_meta": target_dir_path / "target_selection_meta.json",
        "target_info": target_dir_path / "target_info.json",
    }


def run_metadata_paths(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    run_root = run_config_dir(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return {
        "run_root": run_root,
        "resolved_config": run_root / "resolved_config.json",
        "key_payloads": run_root / "key_payloads.json",
        "artifact_manifest": run_root / "artifact_manifest.json",
        "progress": run_root / "progress.json",
        "summary": run_root / f"summary_{run_type}.json",
    }


def run_artifact_paths(
    config: Config,
    *,
    run_type: str,
    target_id: str | int,
    victim_name: str | None = None,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    victim = victim_name or _primary_victim(config)
    local_base = victim_dir(
        config,
        target_id,
        run_type=run_type,
        victim_name=victim,
        attack_identity_context=attack_identity_context,
    )
    shared_base = shared_victim_dir(
        config,
        run_type=run_type,
        target_id=target_id,
        victim_name=victim,
        attack_identity_context=attack_identity_context,
    )
    return {
        "run_dir": local_base,
        "config_snapshot": local_base / "config.yaml",
        "resolved_config": local_base / "resolved_config.json",
        "metrics": local_base / "metrics.json",
        "predictions": local_base / "predictions.json",
        "train_history": local_base / "train_history.json",
        "poisoned_train": local_base / "poisoned_train.txt",
        "prefix_nonzero_when_possible_metadata": (
            local_base / "prefix_nonzero_when_possible_metadata.pkl"
        ),
        "dpsbr_position_metadata": local_base / "dpsbr_position_metadata.json",
        "random_nonzero_position_metadata": local_base / "random_nonzero_position_metadata.json",
        "shared_dir": shared_base,
        "shared_predictions": shared_base / "predictions.json",
        "shared_train_history": shared_base / "train_history.json",
        "shared_execution_result": shared_base / "execution_result.json",
        "shared_poisoned_train": shared_base / "poisoned_train.txt",
    }


__all__ = [
    "attack_key",
    "attack_key_payload",
    "build_position_opt_attack_identity_context",
    "canonical_split_dir",
    "canonical_split_paths",
    "checkpoint_identity_payload",
    "dataset_name",
    "dataset_paths",
    "dataset_root",
    "evaluation_key",
    "evaluation_key_payload",
    "output_root",
    "run_artifact_paths",
    "run_config_dir",
    "run_metadata_paths",
    "runs_root",
    "shared_artifact_paths",
    "shared_attack_dir",
    "shared_root",
    "shared_victim_dir",
    "split_key",
    "split_key_payload",
    "target_dir",
    "target_selection_dir",
    "target_selection_key",
    "target_selection_key_payload",
    "victim_dir",
    "victim_prediction_key",
    "victim_prediction_key_payload",
]
