from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from attack.common.config import Config
from attack.common.paths import shared_attack_dir, target_dir

from .types import PositionOptArtifactPaths


POSITION_OPT_RUN_TYPE = "position_opt_mvp"


def resolve_clean_surrogate_checkpoint_path(
    config: Config,
    *,
    run_type: str = POSITION_OPT_RUN_TYPE,
    override: str | Path | None = None,
) -> Path:
    del run_type
    if override is not None:
        override_text = str(override).strip()
        if not override_text:
            raise ValueError(
                "Clean surrogate checkpoint override must be a non-empty path."
            )
        return Path(override_text)

    position_opt_config = config.attack.position_opt
    if (
        position_opt_config is not None
        and position_opt_config.clean_surrogate_checkpoint is not None
    ):
        return Path(position_opt_config.clean_surrogate_checkpoint)

    raise ValueError(
        "Position optimization requires a clean surrogate checkpoint from either "
        "attack.position_opt.clean_surrogate_checkpoint or "
        "--clean-surrogate-checkpoint."
    )


def build_position_opt_artifact_paths(
    config: Config,
    *,
    run_type: str = POSITION_OPT_RUN_TYPE,
    target_item: int | None = None,
    clean_checkpoint_override: str | Path | None = None,
    attack_identity_context: Mapping[str, Any] | None = None,
) -> PositionOptArtifactPaths:
    if target_item is None:
        base_dir = shared_attack_dir(config, run_type=run_type) / "position_opt"
    else:
        base_dir = (
            target_dir(
                config,
                target_item,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            )
            / "position_opt"
        )

    return PositionOptArtifactPaths(
        base_dir=base_dir,
        clean_surrogate_checkpoint=resolve_clean_surrogate_checkpoint_path(
            config,
            run_type=run_type,
            override=clean_checkpoint_override,
        ),
        optimized_poisoned_sessions=base_dir / "optimized_poisoned_sessions.pkl",
        selected_positions=base_dir / "selected_positions.json",
        training_history=base_dir / "training_history.json",
        learned_logits=base_dir / "learned_logits.pt",
        run_metadata=base_dir / "run_metadata.json",
    )


def ensure_position_opt_artifact_dirs(paths: PositionOptArtifactPaths) -> PositionOptArtifactPaths:
    paths.base_dir.mkdir(parents=True, exist_ok=True)
    paths.clean_surrogate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.optimized_poisoned_sessions.parent.mkdir(parents=True, exist_ok=True)
    if paths.selected_positions is not None:
        paths.selected_positions.parent.mkdir(parents=True, exist_ok=True)
    if paths.training_history is not None:
        paths.training_history.parent.mkdir(parents=True, exist_ok=True)
    if paths.learned_logits is not None:
        paths.learned_logits.parent.mkdir(parents=True, exist_ok=True)
    if paths.run_metadata is not None:
        paths.run_metadata.parent.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = [
    "POSITION_OPT_RUN_TYPE",
    "build_position_opt_artifact_paths",
    "ensure_position_opt_artifact_dirs",
    "resolve_clean_surrogate_checkpoint_path",
]
