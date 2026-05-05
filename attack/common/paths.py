from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

from .config import Config, COVERAGE_AWARE_LOCAL_POSITION_SCORER
from .srgnn_training_protocol import (
    SRGNN_VALIDATION_BEST_PROTOCOL,
    srgnn_checkpoint_protocol,
    srgnn_validation_protocol_identity,
)


POSITION_OPT_RUN_TYPE = "position_opt_mvp"
POSITION_OPT_SHARED_POLICY_RUN_TYPE = "position_opt_shared_policy"
POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE = "rank_bucket_cem"
POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE = "rank_bucket_cem_candidate_replay"
TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE = "target_aware_carrier_selection_nz"
TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE = "target_aware_carrier_local_position"
TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE = "target_aware_coverage_local_position"
RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "random_insertion_nonzero_when_possible"
)
_TARGET_AWARE_CANDIDATE_POOL_RUN_TYPES = {
    TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
    TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
}
_POSITION_OPT_RUNTIME_RUN_TYPES = {
    POSITION_OPT_RUN_TYPE,
    POSITION_OPT_SHARED_POLICY_RUN_TYPE,
    POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
}
TARGET_COHORT_SELECTION_POLICY_VERSION = "appendable_target_cohort_v1"


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
    runtime_seeds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the position-opt-specific runtime identity payload.

    The final position-opt poisoned result can change when either the outer-loop
    settings or the clean surrogate checkpoint changes, so both belong in the
    final attack identity layer.
    """
    return {
        "position_opt": {
            "config": _normalize_identity_value(position_opt_config),
            "seeds": (
                None
                if runtime_seeds is None
                else _normalize_identity_value(runtime_seeds)
            ),
            "clean_surrogate": checkpoint_identity_payload(clean_surrogate_checkpoint),
        }
    }


def carrier_selection_identity_payload(config: Config) -> dict[str, Any]:
    carrier_selection = config.attack.carrier_selection
    if carrier_selection is None:
        raise ValueError("attack.carrier_selection is required for TACS-NZ identity.")
    payload = {
        "enabled": bool(carrier_selection.enabled),
        "candidate_pool_size": float(carrier_selection.candidate_pool_size),
        "final_attack_size": float(carrier_selection.final_attack_size),
        "scorer": carrier_selection.scorer,
        "embedding_weight": float(carrier_selection.embedding_weight),
        "cooccurrence_weight": float(carrier_selection.cooccurrence_weight),
        "transition_weight": float(carrier_selection.transition_weight),
        "use_length_control": bool(carrier_selection.use_length_control),
        "length_buckets": carrier_selection.length_buckets,
        "normalize": carrier_selection.normalize,
        "placement_mode": carrier_selection.placement_mode,
        "operation": carrier_selection.operation,
        "candidate_positions": carrier_selection.candidate_positions,
        "local_embedding_weight": float(carrier_selection.local_embedding_weight),
        "local_transition_weight": float(carrier_selection.local_transition_weight),
        "session_compatibility_weight": float(
            carrier_selection.session_compatibility_weight
        ),
        "left_to_target_weight": float(carrier_selection.left_to_target_weight),
        "target_to_right_weight": float(carrier_selection.target_to_right_weight),
    }
    if carrier_selection.scorer == COVERAGE_AWARE_LOCAL_POSITION_SCORER:
        payload.update(
            {
                "coverage_prefix_source": carrier_selection.coverage_prefix_source,
                "vulnerable_rank_min": int(carrier_selection.vulnerable_rank_min),
                "vulnerable_rank_max": int(carrier_selection.vulnerable_rank_max),
                "max_vulnerable_prefixes": int(
                    carrier_selection.max_vulnerable_prefixes
                ),
                "prefix_representation": carrier_selection.prefix_representation,
                "candidate_representation": carrier_selection.candidate_representation,
                "top_m_coverage": int(carrier_selection.top_m_coverage),
                "rank_weighting": carrier_selection.rank_weighting,
                "coverage_similarity": carrier_selection.coverage_similarity,
            }
        )
    return payload


def carrier_selection_shared_generation_payload(config: Config) -> dict[str, Any]:
    carrier_selection = config.attack.carrier_selection
    if carrier_selection is None:
        raise ValueError("attack.carrier_selection is required for TACS-NZ generation identity.")
    return {
        "family": "target_aware_candidate_pool",
        "candidate_pool_size": float(carrier_selection.candidate_pool_size),
    }


def poison_model_key_payload(config: Config) -> dict[str, Any]:
    return {
        "split_key": split_key(config),
        "fake_session_seed": int(config.seeds.fake_session_seed),
        "poison_model": _poison_model_identity_payload(config),
    }


def poison_model_key(config: Config) -> str:
    return f"poison_model_{_hash_token(_stable_json(poison_model_key_payload(config)))}"


def poison_model_dir(config: Config) -> Path:
    return shared_root(config) / "poison_models" / poison_model_key(config)


def _poison_model_identity_payload(config: Config) -> dict[str, Any]:
    poison_model_payload: dict[str, Any] = {
        "name": config.attack.poison_model.name,
        "params": config.attack.poison_model.params,
    }
    if config.attack.poison_model.name == "srgnn":
        train_config = config.attack.poison_model.params.get("train", {})
        if (
            isinstance(train_config, Mapping)
            and srgnn_checkpoint_protocol(train_config) == SRGNN_VALIDATION_BEST_PROTOCOL
        ):
            poison_model_payload.update(
                srgnn_validation_protocol_identity(train_config, prefix="poison_model")
            )
    return poison_model_payload


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
    """Legacy batch-era target-selection identity.

    Keep this payload for compatibility and future migration tooling, but do
    not treat it as the authoritative target-cohort identity.
    """
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


def target_cohort_key_payload(config: Config) -> dict[str, Any]:
    mode = config.targets.mode
    bucket: str | None = None
    explicit_list: list[int] = []
    target_selection_seed: int | None = None
    if mode == "sampled":
        bucket = config.targets.bucket
        target_selection_seed = int(config.seeds.target_selection_seed)
    elif mode == "explicit_list":
        explicit_list = [int(item) for item in config.targets.explicit_list]
    return {
        "split_key": split_key(config),
        "selection_policy_version": TARGET_COHORT_SELECTION_POLICY_VERSION,
        "mode": mode,
        "bucket": bucket,
        "explicit_list": explicit_list,
        "target_selection_seed": target_selection_seed,
    }


def target_cohort_key(config: Config) -> str:
    payload = target_cohort_key_payload(config)
    return f"target_cohort_{_hash_token(_stable_json(payload))}"


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
        "run_type": run_type,
        "split_key": split_key(config),
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
    if run_type in _TARGET_AWARE_CANDIDATE_POOL_RUN_TYPES:
        payload["attack"]["carrier_selection"] = carrier_selection_identity_payload(config)
    if run_type in _POSITION_OPT_RUNTIME_RUN_TYPES:
        if attack_identity_context is None:
            raise ValueError(
                f"{run_type} final attack identity requires explicit "
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
    poison_model_payload = _poison_model_identity_payload(config)

    generation_size = float(config.attack.size)
    carrier_generation_payload: dict[str, Any] | None = None
    if run_type in _TARGET_AWARE_CANDIDATE_POOL_RUN_TYPES:
        carrier_generation_payload = carrier_selection_shared_generation_payload(config)
        generation_size = float(carrier_generation_payload["candidate_pool_size"])

    attack_generation: dict[str, Any] = {
        "split_key": split_key(config),
        "fake_session_seed": int(config.seeds.fake_session_seed),
        "attack_generation": {
            "size": generation_size,
            "fake_session_generation_topk": int(config.attack.fake_session_generation_topk),
            "poison_model": poison_model_payload,
        },
    }
    if carrier_generation_payload is not None:
        attack_generation["attack_generation"][
            "carrier_selection_candidate_pool"
        ] = carrier_generation_payload
    return attack_generation


def shared_attack_artifact_key(config: Config, *, run_type: str) -> str:
    payload = shared_attack_artifact_key_payload(config, run_type=run_type)
    return f"attack_shared_{_hash_token(_stable_json(payload))}"


_VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS = frozenset(
    {
        "batch_size",
        "train_batch_size",
        "eval_batch_size",
    }
)


def _victim_identity_params(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _victim_identity_params(inner_value)
            for key, inner_value in value.items()
            if key not in _VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS
        }
    if isinstance(value, list):
        return [_victim_identity_params(item) for item in value]
    if isinstance(value, tuple):
        return [_victim_identity_params(item) for item in value]
    return value


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
    # Victim prediction identity excludes runtime-only fields and batch-size
    # tuning knobs so append/retry can keep reusing victim state across
    # resource-only batch-size adjustments.
    victim_params = _victim_identity_params(config.victims.params[victim_name])
    payload = {
        **base_context,
        "victim_name": victim_name,
        "victim_train_seed": int(config.seeds.victim_train_seed),
        "victim_params": victim_params,
    }
    if victim_name == "srgnn":
        train_config = config.victims.params[victim_name].get("train", {})
        if (
            isinstance(train_config, Mapping)
            and srgnn_checkpoint_protocol(train_config) == SRGNN_VALIDATION_BEST_PROTOCOL
        ):
            payload.update(
                srgnn_validation_protocol_identity(train_config, prefix="victim_srgnn")
            )
    return payload


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


def run_group_key_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "run_type": run_type,
        "split_key": split_key(config),
        "target_cohort_key": target_cohort_key(config),
        "shared_attack_artifact_key": shared_attack_artifact_key(config, run_type=run_type),
        "final_attack_key": attack_key(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ),
        "evaluation_schema": {
            "topk": [int(k) for k in config.evaluation.topk],
            "targeted_metrics": list(config.evaluation.targeted_metrics),
            "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
        },
    }


def run_group_key(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> str:
    payload = run_group_key_payload(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return f"run_group_{_hash_token(_stable_json(payload))}"


def evaluation_key_payload(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy batch-era evaluation identity.

    Keep this payload for compatibility and future migration tooling, but do
    not use it as the primary run-group identity.
    """
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
    # Deprecated authoritative identity path. Keep for compatibility with the
    # batch-era selected-target artifacts until target_registry.json lands.
    return shared_root(config) / "targets" / target_selection_key(config)


def target_cohort_dir(config: Config) -> Path:
    return shared_root(config) / "target_cohorts" / target_cohort_key(config)


def target_registry_path(config: Config) -> Path:
    return target_cohort_dir(config) / "target_registry.json"


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
    base = (
        shared_root(config)
        / "victim_predictions"
        / victim_name
        / victim_prediction_key(
            config,
            victim_name,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        )
    )
    if run_type == "clean":
        # Clean victim execution is target-agnostic: one trained model and one
        # prediction export are reused across all target evaluations.
        return base / "shared"
    return base / "targets" / str(target_id)


def runs_root(config: Config) -> Path:
    return (
        output_root(config)
        / config.artifacts.runs_dir
        / dataset_name(config)
        / config.experiment.name
    )


def run_group_dir(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    return runs_root(config) / run_group_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )


def run_config_dir(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> Path:
    # Deprecated compatibility alias. The primary execution identity is now the
    # run-group key, not the batch-era evaluation key.
    return run_group_dir(
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
    poison_dir_path = poison_model_dir(config)
    legacy_target_dir_path = target_selection_dir(config)
    cohort_dir_path = target_cohort_dir(config)
    return {
        "attack_shared_dir": attack_dir,
        "attack_config_snapshot": attack_dir / "config.yaml",
        "poison_model_dir": poison_dir_path,
        "poison_model": poison_dir_path / "poison_model.pt",
        "poison_model_identity": poison_dir_path / "identity.json",
        "fake_sessions": attack_dir / "fake_sessions.pkl",
        "poison_train_history": poison_dir_path / "poison_train_history.json",
        "legacy_attack_poison_model": attack_dir / "poison_model.pt",
        "legacy_attack_poison_train_history": attack_dir / "poison_train_history.json",
        "target_cohort_dir": cohort_dir_path,
        "target_registry": cohort_dir_path / "target_registry.json",
        "target_shared_dir": legacy_target_dir_path,
        "target_config_snapshot": legacy_target_dir_path / "config.yaml",
        "selected_targets": legacy_target_dir_path / "selected_targets.json",
        "target_selection_meta": legacy_target_dir_path / "target_selection_meta.json",
        "target_info": legacy_target_dir_path / "target_info.json",
    }


def run_metadata_paths(
    config: Config,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    run_root = run_group_dir(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    return {
        "run_root": run_root,
        "resolved_config": run_root / "resolved_config.json",
        "key_payloads": run_root / "key_payloads.json",
        "artifact_manifest": run_root / "artifact_manifest.json",
        "run_coverage": run_root / "run_coverage.json",
        "execution_log": run_root / "execution_log.json",
        "summary_current": run_root / "summary_current.json",
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
        "random_insertion_slot_metadata": local_base / "random_insertion_slot_metadata.json",
        "shared_dir": shared_base,
        "shared_predictions": shared_base / "predictions.json",
        "shared_train_history": shared_base / "train_history.json",
        "shared_execution_result": shared_base / "execution_result.json",
        "shared_poisoned_train": shared_base / "poisoned_train.txt",
    }


__all__ = [
    "POSITION_OPT_RUN_TYPE",
    "POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE",
    "POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE",
    "POSITION_OPT_SHARED_POLICY_RUN_TYPE",
    "RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE",
    "TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE",
    "TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE",
    "attack_key",
    "attack_key_payload",
    "carrier_selection_identity_payload",
    "carrier_selection_shared_generation_payload",
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
    "poison_model_dir",
    "poison_model_key",
    "poison_model_key_payload",
    "run_group_dir",
    "run_group_key",
    "run_group_key_payload",
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
    "target_cohort_dir",
    "target_cohort_key",
    "target_cohort_key_payload",
    "target_registry_path",
    "target_selection_dir",
    "target_selection_key",
    "target_selection_key_payload",
    "victim_dir",
    "victim_prediction_key",
    "victim_prediction_key_payload",
]
