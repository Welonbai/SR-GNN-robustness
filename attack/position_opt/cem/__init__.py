from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from attack.common.config import RankBucketCEMConfig
from attack.common.paths import (
    POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE,
    checkpoint_identity_payload,
)

from .artifacts import (
    RankBucketCEMArtifactPaths,
    build_rank_bucket_cem_artifact_paths,
    ensure_rank_bucket_cem_artifact_dirs,
)
from .availability import (
    RankCandidateSessionState,
    build_availability_summary,
    build_rank_candidate_states,
)
from .optimizer import (
    CEMCandidate,
    CEMCandidateResult,
    CEMState,
    initialize_cem_state,
    sample_cem_candidates,
    update_cem_state,
)
from .rank_policy import (
    RankBucketPolicy,
    RankBucketSelectionRecord,
    build_rank_position_summary,
    sample_positions_from_rank_policy,
)

if TYPE_CHECKING:
    from .trainer import RankBucketCEMTrainer


RANK_BUCKET_CEM_DEFAULTS = RankBucketCEMConfig()


def resolve_rank_bucket_cem_config(
    config: RankBucketCEMConfig | Mapping[str, Any] | None = None,
    overrides: RankBucketCEMConfig | Mapping[str, Any] | None = None,
) -> RankBucketCEMConfig:
    merged = asdict(RANK_BUCKET_CEM_DEFAULTS)
    merged.update(_coerce_rank_bucket_cem_layer(config))
    merged.update(_coerce_rank_bucket_cem_layer(overrides))
    return RankBucketCEMConfig(**merged)


def rank_bucket_cem_identity_payload(config: RankBucketCEMConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in (
        "save_candidate_selected_positions",
        "save_final_selected_positions",
        "save_optimized_poisoned_sessions",
        "save_replay_metadata",
    ):
        payload.pop(key, None)
    return payload


def build_rank_bucket_cem_attack_identity_context(
    *,
    position_opt_config: Mapping[str, Any],
    rank_bucket_cem_config: Mapping[str, Any],
    clean_surrogate_checkpoint: str | Path,
    runtime_seeds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "position_opt": {
            "config": _normalize_identity_value(position_opt_config),
            "rank_bucket_cem": _normalize_identity_value(rank_bucket_cem_config),
            "seeds": (
                None
                if runtime_seeds is None
                else _normalize_identity_value(runtime_seeds)
            ),
            "clean_surrogate": checkpoint_identity_payload(clean_surrogate_checkpoint),
        }
    }


def __getattr__(name: str):
    if name == "RankBucketCEMTrainer":
        from .trainer import RankBucketCEMTrainer

        return RankBucketCEMTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _coerce_rank_bucket_cem_layer(
    value: RankBucketCEMConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, RankBucketCEMConfig):
        return asdict(value)
    if isinstance(value, Mapping):
        payload = dict(value)
        allowed_fields = {
            "iterations",
            "population_size",
            "elite_ratio",
            "initial_std",
            "min_std",
            "smoothing",
            "reward_metric",
            "save_candidate_selected_positions",
            "save_final_selected_positions",
            "save_optimized_poisoned_sessions",
            "save_replay_metadata",
        }
        unknown = set(payload) - allowed_fields
        if unknown:
            raise ValueError(
                "Unknown rank_bucket_cem config keys: "
                + ", ".join(sorted(map(str, unknown)))
            )
        return payload
    raise TypeError(
        "rank_bucket_cem config must be a RankBucketCEMConfig instance, a mapping, or None."
    )


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


__all__ = [
    "CEMCandidate",
    "CEMCandidateResult",
    "CEMState",
    "POSITION_OPT_RANK_BUCKET_CEM_RUN_TYPE",
    "RANK_BUCKET_CEM_DEFAULTS",
    "RankBucketCEMArtifactPaths",
    "RankBucketCEMTrainer",
    "RankBucketPolicy",
    "RankBucketSelectionRecord",
    "RankCandidateSessionState",
    "build_availability_summary",
    "build_rank_bucket_cem_artifact_paths",
    "build_rank_bucket_cem_attack_identity_context",
    "build_rank_candidate_states",
    "build_rank_position_summary",
    "ensure_rank_bucket_cem_artifact_dirs",
    "initialize_cem_state",
    "rank_bucket_cem_identity_payload",
    "resolve_rank_bucket_cem_config",
    "sample_cem_candidates",
    "sample_positions_from_rank_policy",
    "update_cem_state",
]
