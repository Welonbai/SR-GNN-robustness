from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import hashlib
import json

from .config import (
    Config,
    COVERAGE_AWARE_LOCAL_POSITION_SCORER,
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    TRAIN_TEMPLATE_LENGTH_MATCHING_EXACT_LARGEST_REMAINDER,
)
from .srgnn_training_protocol import (
    SRGNN_VALIDATION_BEST_PROTOCOL,
    srgnn_checkpoint_protocol,
    srgnn_validation_protocol_identity,
)


POSITION_OPT_RUN_TYPE = "position_opt_mvp"
POSITION_OPT_SHARED_POLICY_RUN_TYPE = "position_opt_shared_policy"
POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE = "rank_bucket_cem"
POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE = "rank_bucket_cem_candidate_replay"
PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE = "pts_construction_grouped_cem"
PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE = "pts_construction_direct_action_mlp_cem"
PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE = "pts_construction_candidate_replay"
CREAT_ADDITIVE_SBR_RUN_TYPE = "creat_additive_sbr"
CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE = "creat_additive_sbr_generate_only"
POISONING_SSL_SBR_RUN_TYPE = "poisoning_ssl_sbr"
PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE = "prefix_nonzero_when_possible"
TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE = "target_aware_carrier_selection_nz"
TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE = "target_aware_carrier_local_position"
TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE = "target_aware_coverage_local_position"
RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "random_insertion_nonzero_when_possible"
)
TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "tail_replacement_nonzero_when_possible"
)
TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "tail_insertion_nonzero_when_possible"
)
RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "random_insertion_then_crop_nonzero_when_possible"
)
INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "internal_random_insertion_nonzero_when_possible"
)
INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "internal_random_insertion_truncate_suffix_nonzero_when_possible"
)
INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "internal_random_insertion_generated_continuation_nonzero_when_possible"
)
INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "internal_random_replacement_nonzero_when_possible"
)
INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE = (
    "internal_random_replacement_generated_continuation_nonzero_when_possible"
)
VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE = (
    "vulnerable_anchor_internal_construction"
)
POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE = (
    "popular_anchor_internal_construction"
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
_ANCHOR_CONSTRUCTION_RUNTIME_RUN_TYPES = {
    POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
    VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
}
TARGET_COHORT_SELECTION_POLICY_VERSION = "appendable_target_cohort_v1"
TRON_VICTIM_DATA_SEMANTICS = "tron_raw_session_export_v1"
MDHG_VICTIM_DATA_SEMANTICS = (
    "mdhg_expanded_pairs_plus_raw_sessions_v3_zero_degree_safe_unique_last_eval"
)
FREQREC_VICTIM_DATA_SEMANTICS = "freqrec_canonical_explicit_prefix_label_v1"
WEAREC_VICTIM_DATA_SEMANTICS = "wearec_canonical_explicit_prefix_label_v1"

_LEGACY_POISONED_VICTIM_RUN_TYPES = frozenset(
    {"always_pos0", "dpsbr_baseline", "random_nonzero_when_possible"}
)
_WEAREC_POISONED_VICTIM_RUN_TYPES = frozenset(
    {
        CREAT_ADDITIVE_SBR_RUN_TYPE,
        POISONING_SSL_SBR_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        POSITION_OPT_SHARED_POLICY_RUN_TYPE,
        POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
        POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE,
        PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
        PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
        TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE,
        *_LEGACY_POISONED_VICTIM_RUN_TYPES,
    }
)


def classify_victim_training_run_type(run_type: str) -> str:
    if run_type == "clean":
        return "clean"
    from attack.position_opt.bucket_selector import BUCKET_METHODS

    if run_type in _WEAREC_POISONED_VICTIM_RUN_TYPES or run_type in BUCKET_METHODS:
        return "poisoned"
    return "unsupported"


def shared_attack_identity_requires_poison_runner(run_type: str) -> bool:
    return run_type in {
        CREAT_ADDITIVE_SBR_RUN_TYPE,
        CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE,
        TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE,
        TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE,
        TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE,
        INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    }


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
    if run_type == POISONING_SSL_SBR_RUN_TYPE:
        if config.attack.poisoning_ssl_sbr is None:
            raise ValueError(
                "poisoning_ssl_sbr final attack identity requires "
                "attack.poisoning_ssl_sbr."
            )
        return {
            "run_type": run_type,
            "split_key": split_key(config),
            "fake_session_seed": int(config.seeds.fake_session_seed),
            "attack": {
                "size": float(config.attack.size),
                "poisoning_ssl_sbr": _poisoning_ssl_sbr_identity_payload(config),
            },
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
    if (
        config.attack.fake_session_source.type
        == FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    ):
        payload["attack"][
            "fake_session_source"
        ] = _train_template_fake_session_source_identity_payload(config)
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
    if run_type in _ANCHOR_CONSTRUCTION_RUNTIME_RUN_TYPES:
        payload["attack"]["anchor_construction"] = _normalize_identity_value(
            config.anchor_construction.__dict__
        )
        if attack_identity_context is None:
            raise ValueError(
                f"{run_type} final attack identity requires explicit "
                "attack_identity_context with selected anchor pools and survey hashes."
            )
        payload["attack_runtime_identity"] = _normalize_identity_value(
            attack_identity_context
        )
    if run_type in {
        PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    }:
        if config.attack.pts_construction is None:
            raise ValueError(
                f"{run_type} final attack identity requires "
                "attack.pts_construction."
            )
        payload["attack"]["pts_construction"] = _pts_construction_identity_payload(
            config
        )
        payload["pts_runtime_seeds"] = {
            "position_opt_seed": int(config.seeds.position_opt_seed),
            "surrogate_train_seed": int(config.seeds.surrogate_train_seed),
        }
        if attack_identity_context is not None:
            payload["attack_runtime_identity"] = _normalize_identity_value(
                attack_identity_context
            )
    if run_type == PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE:
        if attack_identity_context is None:
            raise ValueError(
                "pts_construction_candidate_replay final attack identity requires "
                "explicit attack_identity_context with source candidate identity."
            )
        payload["attack_runtime_identity"] = _normalize_identity_value(
            attack_identity_context
        )
    if run_type in {CREAT_ADDITIVE_SBR_RUN_TYPE, CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE}:
        if config.attack.creat_additive_sbr is None:
            raise ValueError(
                "creat_additive_sbr final attack identity requires "
                "attack.creat_additive_sbr."
            )
        payload["attack"]["creat_additive_sbr"] = _normalize_identity_value(
            config.to_primitive()["attack"]["creat_additive_sbr"]
        )
    return payload


def _pts_construction_identity_payload(config: Config) -> Any:
    """Return PTS construction semantics, excluding artifact-only persistence knobs."""
    payload = _normalize_identity_value(
        config.to_primitive()["attack"]["pts_construction"]
    )
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("artifacts", None)
        method = payload.get("method")
        if method == "grouped_cem_v1":
            payload.pop("continuous_policy", None)
            payload.pop("direct_action_policy", None)
        elif method == "continuous_mlp_cem":
            payload.pop("grouping", None)
            payload.pop("actions", None)
            payload.pop("direct_action_policy", None)
            continuous = payload.get("continuous_policy")
            if isinstance(continuous, dict):
                payload["continuous_policy"] = dict(continuous)
        elif method == "direct_action_mlp_cem":
            payload.pop("grouping", None)
            payload.pop("actions", None)
            payload.pop("continuous_policy", None)
            direct_action = payload.get("direct_action_policy")
            if isinstance(direct_action, dict):
                payload["direct_action_policy"] = {
                    "parameterization": direct_action.get("parameterization"),
                    "length_feature": direct_action.get("length_feature"),
                    "cem_init": {
                        "mode": "standard_normal",
                        "parameter_space": "standardized_policy_parameter_space",
                    },
                }
        cem = payload.get("cem")
        if isinstance(cem, dict):
            cem = dict(cem)
            cem.pop("save_top_k_candidates", None)
            cem.pop("epoch_reward_diagnostics", None)
            cem.pop("surrogate_retrain", None)
            if method == "continuous_mlp_cem":
                sampler = cem.get("sampler")
                if isinstance(sampler, dict):
                    cem["sampler"] = {"type": "gaussian_parameter_space_v1"}
                update = cem.get("update")
                if isinstance(update, dict):
                    cem["update"] = {"smoothing": update.get("smoothing")}
                cem.pop("init", None)
                cem.pop("resampling", None)
            elif method == "direct_action_mlp_cem":
                sampler = cem.get("sampler")
                if isinstance(sampler, dict):
                    cem["sampler"] = {"type": "gaussian"}
                update = cem.get("update")
                if isinstance(update, dict):
                    cem["update"] = {
                        "mode": update.get("mode"),
                        "elite_min_std": update.get("elite_min_std"),
                    }
                cem["init"] = None
                cem.pop("resampling", None)
            payload = dict(payload)
            payload["cem"] = cem
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


def shared_attack_artifact_key_payload(
    config: Config,
    *,
    run_type: str,
    require_poison_runner: bool = False,
) -> dict[str, Any]:
    require_poison_runner = bool(
        require_poison_runner or shared_attack_identity_requires_poison_runner(run_type)
    )
    if run_type == "clean":
        return {
            "run_type": "clean",
            "split_key": split_key(config),
        }
    if run_type == POISONING_SSL_SBR_RUN_TYPE:
        if config.attack.poisoning_ssl_sbr is None:
            raise ValueError(
                "poisoning_ssl_sbr shared artifact identity requires "
                "attack.poisoning_ssl_sbr."
            )
        return {
            "run_type": run_type,
            "split_key": split_key(config),
            "fake_session_seed": int(config.seeds.fake_session_seed),
            "attack_generation": {
                "size": float(config.attack.size),
                "poisoning_ssl_sbr": _poisoning_ssl_sbr_identity_payload(config),
            },
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

    fake_session_source = config.attack.fake_session_source
    if fake_session_source.type == FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED:
        fake_session_source_payload = _train_template_fake_session_source_identity_payload(config)
        attack_generation = {
            "split_key": split_key(config),
            "fake_session_seed": int(config.seeds.fake_session_seed),
            "attack_generation": {
                "size": generation_size,
                "fake_session_source": fake_session_source_payload,
                "shared_identity_includes_poison_model": bool(require_poison_runner),
            },
        }
        if bool(require_poison_runner):
            attack_generation["attack_generation"]["poison_model"] = poison_model_payload
            attack_generation["attack_generation"]["fake_session_generation_topk"] = int(
                config.attack.fake_session_generation_topk
            )
        if carrier_generation_payload is not None:
            attack_generation["attack_generation"][
                "carrier_selection_candidate_pool"
            ] = carrier_generation_payload
        return attack_generation

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


def _train_template_fake_session_source_identity_payload(config: Config) -> dict[str, Any]:
    train_template = config.attack.fake_session_source.train_template
    return {
        "type": FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
        "reference_split": train_template.reference_split,
        "length_matching_mode": TRAIN_TEMPLATE_LENGTH_MATCHING_EXACT_LARGEST_REMAINDER,
        "target_filtering": train_template.target_filtering,
        "replacement": bool(train_template.replacement),
        "fallback": {
            "nearest_length_redistribution": bool(
                train_template.fallback.nearest_length_redistribution
            ),
            "replacement_if_needed": bool(
                train_template.fallback.replacement_if_needed
            ),
        },
        "denominator_source": "build_clean_pairs(canonical_dataset)[0]",
        "denominator_representation": "expanded_prefix_label_pairs",
    }


def _poisoning_ssl_sbr_identity_payload(config: Config) -> dict[str, Any]:
    poisoning = config.attack.poisoning_ssl_sbr
    if poisoning is None:
        raise ValueError("attack.poisoning_ssl_sbr is required for identity payload.")
    return {
        "max_seq_len_policy": poisoning.max_seq_len_policy,
        "original_max_seq_len_cap": int(poisoning.original_max_seq_len_cap),
        "max_seq_len_override": (
            None
            if poisoning.max_seq_len_override is None
            else int(poisoning.max_seq_len_override)
        ),
        "enforce_single_target": bool(poisoning.enforce_single_target),
        "enforce_nonzero_target_position": bool(
            poisoning.enforce_nonzero_target_position
        ),
        "filter_no_target": bool(poisoning.filter_no_target),
        "filter_short_sessions": bool(poisoning.filter_short_sessions),
        "candidate_multiplier": int(poisoning.candidate_multiplier),
        "max_generation_rounds": int(poisoning.max_generation_rounds),
        "generation_seed_offset": int(poisoning.generation_seed_offset),
        "generation_backend": poisoning.generation_backend,
        "device": poisoning.device,
        "gpu_id": poisoning.gpu_id,
        "classifier_epochs": poisoning.classifier_epochs,
        "mle_epochs": poisoning.mle_epochs,
        "adversarial_epochs": poisoning.adversarial_epochs,
        "discriminator_pretrain_steps": poisoning.discriminator_pretrain_steps,
        "discriminator_pretrain_epochs": poisoning.discriminator_pretrain_epochs,
        "discriminator_adversarial_steps": poisoning.discriminator_adversarial_steps,
        "discriminator_adversarial_epochs": poisoning.discriminator_adversarial_epochs,
        "batch_size": poisoning.batch_size,
        "learning_rate": poisoning.learning_rate,
        "classifier_learning_rate": poisoning.classifier_learning_rate,
        "embedding_dim": poisoning.embedding_dim,
        "hidden_dim": poisoning.hidden_dim,
        "discriminator_embedding_dim": poisoning.discriminator_embedding_dim,
        "discriminator_hidden_dim": poisoning.discriminator_hidden_dim,
        "classifier_embedding_dim": poisoning.classifier_embedding_dim,
        "pos_neg_samples": poisoning.pos_neg_samples,
        "max_train_sequences": poisoning.max_train_sequences,
        "reward_target_weight": poisoning.reward_target_weight,
        "reward_classifier_weight": poisoning.reward_classifier_weight,
        "reward_discriminator_weight": poisoning.reward_discriminator_weight,
    }


def shared_attack_artifact_key(
    config: Config,
    *,
    run_type: str,
    require_poison_runner: bool = False,
) -> str:
    payload = shared_attack_artifact_key_payload(
        config,
        run_type=run_type,
        require_poison_runner=require_poison_runner,
    )
    return f"attack_shared_{_hash_token(_stable_json(payload))}"


_VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS = frozenset(
    {
        "batch_size",
        "train_batch_size",
        "eval_batch_size",
        "validation_enabled",
    }
)
_EXTERNAL_VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS = frozenset(
    {
        "checkpoint_protocol",
        "export_model",
    }
)


def _victim_identity_params(
    value: Any,
    *,
    exclude_external_protocol: bool = False,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _victim_identity_params(
                inner_value,
                exclude_external_protocol=exclude_external_protocol,
            )
            for key, inner_value in value.items()
            if key not in _VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS
            and not (
                exclude_external_protocol
                and key in _EXTERNAL_VICTIM_IDENTITY_EXCLUDED_PARAM_KEYS
            )
            and not (
                exclude_external_protocol
                and key == "epochs"
                and "max_epochs" in value
            )
        }
    if isinstance(value, list):
        return [
            _victim_identity_params(
                item,
                exclude_external_protocol=exclude_external_protocol,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            _victim_identity_params(
                item,
                exclude_external_protocol=exclude_external_protocol,
            )
            for item in value
        ]
    return value


def _freqrec_identity_params(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return _normalize_identity_value(value)
    normalized = _normalize_identity_value(value)
    if not isinstance(normalized, dict):
        return normalized
    train = normalized.get("train")
    if not isinstance(train, dict):
        return normalized
    protocol = str(train.get("checkpoint_protocol", ""))
    projected_train = dict(train)
    projected_train.pop("patience", None)
    if protocol == "fixed_epoch":
        projected_train.pop("validation_metric", None)
    else:
        metric = str(projected_train.get("validation_metric", ""))
        cutoff = int(metric.rsplit("@", 1)[1]) if "@" in metric else None
        projected_train["monitor_cutoff"] = cutoff
    normalized["train"] = projected_train
    return normalized


def victim_prediction_key_payload(
    config: Config,
    victim_name: str,
    *,
    run_type: str,
    attack_identity_context: Mapping[str, Any] | None = None,
    victim_attack_identity_context: Mapping[str, Any] | None = None,
    victim_effective_train_seed: int | None = None,
) -> dict[str, Any]:
    if victim_name == "wearec":
        base_context = {
            "run_type": run_type,
            "wearec_scientific_identity": (
                {"state": "pre_export"}
                if victim_attack_identity_context is None
                else _normalize_identity_value(victim_attack_identity_context)
            ),
        }
    elif run_type == "clean":
        base_context: dict[str, Any] = {
            "run_type": "clean",
            "split_key": split_key(config),
        }
    elif victim_attack_identity_context is not None:
        base_context = {
            "run_type": run_type,
            "victim_attack_identity": _normalize_identity_value(
                victim_attack_identity_context
            ),
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
    if victim_name == "freqrec":
        victim_params = _freqrec_identity_params(config.victims.params[victim_name])
    else:
        victim_params = _victim_identity_params(
            config.victims.params[victim_name],
            exclude_external_protocol=victim_name in {"miasrec", "tron", "mdhg"},
        )
    payload = {
        **base_context,
        "victim_name": victim_name,
        "victim_train_seed": int(config.seeds.victim_train_seed),
        "victim_params": victim_params,
    }
    if victim_effective_train_seed is not None:
        payload["victim_effective_train_seed"] = int(victim_effective_train_seed)
    if victim_name == "tron":
        payload["victim_data_semantics"] = TRON_VICTIM_DATA_SEMANTICS
    if victim_name == "mdhg":
        payload["victim_data_semantics"] = MDHG_VICTIM_DATA_SEMANTICS
    if victim_name == "freqrec":
        payload["victim_data_semantics"] = FREQREC_VICTIM_DATA_SEMANTICS
    if victim_name == "wearec":
        payload["victim_data_semantics"] = WEAREC_VICTIM_DATA_SEMANTICS
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
    victim_attack_identity_context: Mapping[str, Any] | None = None,
    victim_effective_train_seed: int | None = None,
) -> str:
    payload = victim_prediction_key_payload(
        config,
        victim_name,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
        victim_attack_identity_context=victim_attack_identity_context,
        victim_effective_train_seed=victim_effective_train_seed,
    )
    return f"victim_{victim_name}_{_hash_token(_stable_json(payload))}"


def freqrec_diagnostic_key_payload(
    config: Config,
    *,
    effective_epochs: int,
) -> dict[str, Any]:
    train = config.victims.params["freqrec"]["train"]
    return {
        "victim_prediction": victim_prediction_key_payload(
            config,
            "freqrec",
            run_type="clean",
        ),
        "diagnostic": {
            "effective_epochs": int(effective_epochs),
            "validation_metric": str(train["validation_metric"]),
            "metric_cutoffs": [int(value) for value in train["metric_cutoffs"]],
        },
    }


def freqrec_diagnostic_key(config: Config, *, effective_epochs: int) -> str:
    payload = freqrec_diagnostic_key_payload(
        config,
        effective_epochs=effective_epochs,
    )
    return f"freqrec_diagnostic_{_hash_token(_stable_json(payload))}"


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
        "shared_attack_artifact_key": shared_attack_artifact_key(
            config,
            run_type=run_type,
            require_poison_runner=shared_attack_identity_requires_poison_runner(run_type),
        ),
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


def shared_attack_dir(
    config: Config,
    *,
    run_type: str,
    require_poison_runner: bool = False,
) -> Path:
    # Shared fake-session / poison-model artifacts should only depend on the
    # inputs that actually affect their generation. Final attack/evaluation keys
    # still use attack_key(...), which may include downstream replacement-policy
    # settings such as replacement_topk_ratio.
    return (
        shared_root(config)
        / "attack"
        / shared_attack_artifact_key(
            config,
            run_type=run_type,
            require_poison_runner=require_poison_runner,
        )
    )


def shared_victim_dir(
    config: Config,
    *,
    run_type: str,
    target_id: str | int,
    victim_name: str,
    attack_identity_context: Mapping[str, Any] | None = None,
    victim_attack_identity_context: Mapping[str, Any] | None = None,
    victim_effective_train_seed: int | None = None,
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
            victim_attack_identity_context=victim_attack_identity_context,
            victim_effective_train_seed=victim_effective_train_seed,
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


def shared_artifact_paths(
    config: Config,
    *,
    run_type: str,
    require_poison_runner: bool = False,
) -> dict[str, Path]:
    attack_dir = shared_attack_dir(
        config,
        run_type=run_type,
        require_poison_runner=require_poison_runner,
    )
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
    victim_attack_identity_context: Mapping[str, Any] | None = None,
    victim_effective_train_seed: int | None = None,
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
        victim_attack_identity_context=victim_attack_identity_context,
        victim_effective_train_seed=victim_effective_train_seed,
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
        "tail_replacement_position_metadata": (
            local_base / "tail_replacement_position_metadata.json"
        ),
        "tail_insertion_slot_metadata": (
            local_base / "tail_insertion_slot_metadata.json"
        ),
        "random_insertion_then_crop_metadata": (
            local_base / "random_insertion_then_crop_metadata.json"
        ),
        "internal_random_insertion_metadata": (
            local_base / "internal_random_insertion_metadata.json"
        ),
        "internal_random_replacement_metadata": (
            local_base / "internal_random_replacement_metadata.json"
        ),
        "vulnerable_anchor_internal_construction_metadata": (
            local_base / "vulnerable_anchor_internal_construction_metadata.json"
        ),
        "popular_anchor_internal_construction_metadata": (
            local_base / "popular_anchor_internal_construction_metadata.json"
        ),
        "shared_dir": shared_base,
        "shared_predictions": shared_base / "predictions.json",
        "shared_train_history": shared_base / "train_history.json",
        "shared_execution_result": shared_base / "execution_result.json",
        "shared_poisoned_train": shared_base / "poisoned_train.txt",
        "wearec_raw_predictions": local_base / "wearec_topk_raw.json",
        "wearec_checkpoint": local_base / "wearec_checkpoint.pt",
        "wearec_log": local_base / "wearec_stdout.log",
        "shared_wearec_raw_predictions": shared_base / "wearec_topk_raw.json",
        "shared_wearec_checkpoint": shared_base / "wearec_checkpoint.pt",
        "shared_wearec_log": shared_base / "wearec_stdout.log",
        "shared_artifact_manifest": shared_base / "artifact_manifest.json",
    }


__all__ = [
    "POSITION_OPT_RUN_TYPE",
    "POSITION_OPT_RANK_BUCKET_CEM_CANDIDATE_REPLAY_RUN_TYPE",
    "POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE",
    "POSITION_OPT_SHARED_POLICY_RUN_TYPE",
    "POISONING_SSL_SBR_RUN_TYPE",
    "PREFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE",
    "CREAT_ADDITIVE_SBR_RUN_TYPE",
    "CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE",
    "PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE",
    "PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE",
    "INTERNAL_RANDOM_INSERTION_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "INTERNAL_RANDOM_INSERTION_TRUNCATE_SUFFIX_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "INTERNAL_RANDOM_REPLACEMENT_GENERATED_CONTINUATION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "INTERNAL_RANDOM_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "POPULAR_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE",
    "VULNERABLE_ANCHOR_INTERNAL_CONSTRUCTION_RUN_TYPE",
    "RANDOM_INSERTION_THEN_CROP_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "TAIL_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_RUN_TYPE",
    "TARGET_AWARE_CARRIER_SELECTION_NZ_RUN_TYPE",
    "TARGET_AWARE_COVERAGE_LOCAL_POSITION_RUN_TYPE",
    "TRON_VICTIM_DATA_SEMANTICS",
    "MDHG_VICTIM_DATA_SEMANTICS",
    "FREQREC_VICTIM_DATA_SEMANTICS",
    "WEAREC_VICTIM_DATA_SEMANTICS",
    "classify_victim_training_run_type",
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
    "freqrec_diagnostic_key",
    "freqrec_diagnostic_key_payload",
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
    "shared_attack_identity_requires_poison_runner",
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
