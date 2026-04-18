from typing import TYPE_CHECKING

from .artifacts import (
    POSITION_OPT_RUN_TYPE,
    build_position_opt_artifact_paths,
    ensure_position_opt_artifact_dirs,
    resolve_clean_surrogate_checkpoint_path,
)
from .candidate_builder import build_candidate_positions
from .objective import (
    PositionOptObjectiveResult,
    compute_asymmetric_gt_penalty,
    compute_position_opt_objective,
)
from .policy import PerSessionLogitPolicy
from .poison_builder import replace_item_at_position
from .selector import sample_position_reinforce, select_position_eval, select_position_train
from .types import (
    CandidateMetadata,
    InnerTrainResult,
    POSITION_OPT_DEFAULTS,
    PositionOptConfig,
    PositionOptArtifactPaths,
    SelectedPositionResult,
    SurrogateScoreResult,
    TruncatedFineTuneConfig,
    position_opt_identity_payload,
    resolve_position_opt_config,
)

if TYPE_CHECKING:
    from .trainer import PositionOptMVPTrainer


def __getattr__(name: str):
    if name == "PositionOptMVPTrainer":
        from .trainer import PositionOptMVPTrainer

        return PositionOptMVPTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "POSITION_OPT_DEFAULTS",
    "POSITION_OPT_RUN_TYPE",
    "CandidateMetadata",
    "InnerTrainResult",
    "PerSessionLogitPolicy",
    "PositionOptConfig",
    "PositionOptMVPTrainer",
    "PositionOptObjectiveResult",
    "PositionOptArtifactPaths",
    "SelectedPositionResult",
    "SurrogateScoreResult",
    "TruncatedFineTuneConfig",
    "build_candidate_positions",
    "build_position_opt_artifact_paths",
    "compute_asymmetric_gt_penalty",
    "compute_position_opt_objective",
    "ensure_position_opt_artifact_dirs",
    "position_opt_identity_payload",
    "replace_item_at_position",
    "resolve_position_opt_config",
    "resolve_clean_surrogate_checkpoint_path",
    "sample_position_reinforce",
    "select_position_eval",
    "select_position_train",
]
