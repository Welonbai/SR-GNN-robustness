from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.position_opt_policy_feature_sets import (
    ALLOWED_POSITION_OPT_POLICY_FEATURE_SETS,
)
from attack.common.srgnn_training_protocol import (
    ALLOWED_SRGNN_BEST_METRICS,
    ALLOWED_SRGNN_CHECKPOINT_PROTOCOLS,
    ALLOWED_SRGNN_PATIENCE_METRICS,
    SRGNN_FIXED_LAST_PROTOCOL,
    SRGNN_VALIDATION_BEST_METRIC,
    SRGNN_VALIDATION_BEST_PROTOCOL,
    SRGNN_VALIDATION_PATIENCE_METRIC,
)


_ALLOWED_VICTIMS = {"srgnn", "miasrec", "tron", "mdhg", "freqrec", "wearec"}
_ALLOWED_TARGET_BUCKETS = {"popular", "unpopular", "all"}
_ALLOWED_EVAL_METRICS = {"precision", "recall", "mrr", "ndcg"}
VICTIM_VALIDATION_BEST_PROTOCOL = "validation_best"
VICTIM_FIXED_EPOCH_PROTOCOL = "fixed_epoch"
VICTIM_EXPORT_BEST = "best"
VICTIM_EXPORT_LAST = "last"
_ALLOWED_EXTERNAL_VICTIM_CHECKPOINT_PROTOCOLS = {
    VICTIM_VALIDATION_BEST_PROTOCOL,
    VICTIM_FIXED_EPOCH_PROTOCOL,
}
_ALLOWED_EXTERNAL_VICTIM_EXPORT_MODELS = {
    VICTIM_EXPORT_BEST,
    VICTIM_EXPORT_LAST,
}
_ALLOWED_POSITION_OPT_REWARD_MODES = {
    "poisoned_target_utility",
    "delta_target_utility",
    "delta_lowk_rank_utility",
}
CREAT_ADDITIVE_SBR_ATTACK_REWARD_SCORE = "score"
CREAT_ADDITIVE_SBR_SEED_SOURCE_POSITION_OPT_SEED = "position_opt_seed"
CREAT_ADDITIVE_SBR_VARIANT_V1 = "v1"
CREAT_ADDITIVE_SBR_VARIANT_V2 = "v2"
CREAT_ADDITIVE_SBR_DPP_BOUNDED_DETERMINANT = "bounded_determinant"
CREAT_ADDITIVE_SBR_DPP_RAW_LOGDET = "raw_logdet"
CREAT_ADDITIVE_SBR_CONSISTENCY_LOCAL_GLOBAL = "local_global"
CREAT_ADDITIVE_SBR_FINAL_POLICY_LAST = "last"
_ALLOWED_POSITION_OPT_FINAL_POLICY_SELECTIONS = {
    "last",
    "best_deterministic",
}
TARGET_AWARE_CARRIER_SELECTION_SCORER = "hybrid_target_session_compatibility"
TARGET_AWARE_CARRIER_LOCAL_POSITION_SCORER = "hybrid_local_position_compatibility"
COVERAGE_AWARE_LOCAL_POSITION_SCORER = "coverage_aware_local_position"
TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE = "best_local_position"
TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION = "replacement"
TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS = "nonzero"
TARGET_AWARE_CARRIER_SELECTION_NORMALIZE_MINMAX = "minmax"
TARGET_AWARE_CARRIER_SELECTION_LENGTH_BUCKETS_EXACT_UNTIL_4_PLUS = "exact_until_4_plus"
COVERAGE_PREFIX_SOURCE_VALIDATION = "validation"
COVERAGE_PREFIX_REPRESENTATION_MEAN_ITEM_EMBEDDING = "mean_item_embedding"
COVERAGE_CANDIDATE_REPRESENTATION_TARGETIZED_PREFIX_MEAN_EMBEDDING = (
    "targetized_prefix_mean_embedding"
)
FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED = "poison_model_generated"
FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED = (
    "train_template_clean_exact_length_matched"
)
TRAIN_TEMPLATE_REFERENCE_SPLIT_TRAIN_SUB = "train_sub"
TRAIN_TEMPLATE_TARGET_FILTERING_NONE = "none"
TRAIN_TEMPLATE_LENGTH_MATCHING_EXACT_LARGEST_REMAINDER = "exact_largest_remainder"
COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK = "inverse_log_rank"
COVERAGE_RANK_WEIGHTING_NONE = "none"
COVERAGE_SIMILARITY_COSINE = "cosine"
_ALLOWED_CARRIER_SELECTION_SCORERS = {
    TARGET_AWARE_CARRIER_SELECTION_SCORER,
    TARGET_AWARE_CARRIER_LOCAL_POSITION_SCORER,
    COVERAGE_AWARE_LOCAL_POSITION_SCORER,
}
_ALLOWED_CARRIER_SELECTION_PLACEMENT_MODES = {
    TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE,
}
_ALLOWED_CARRIER_SELECTION_OPERATIONS = {
    TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION,
}
_ALLOWED_CARRIER_SELECTION_CANDIDATE_POSITIONS = {
    TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS,
}
_ALLOWED_CARRIER_SELECTION_NORMALIZE = {
    TARGET_AWARE_CARRIER_SELECTION_NORMALIZE_MINMAX,
}
_ALLOWED_CARRIER_SELECTION_LENGTH_BUCKETS = {
    TARGET_AWARE_CARRIER_SELECTION_LENGTH_BUCKETS_EXACT_UNTIL_4_PLUS,
}
_ALLOWED_COVERAGE_PREFIX_SOURCES = {COVERAGE_PREFIX_SOURCE_VALIDATION}
_ALLOWED_COVERAGE_PREFIX_REPRESENTATIONS = {
    COVERAGE_PREFIX_REPRESENTATION_MEAN_ITEM_EMBEDDING,
}
_ALLOWED_COVERAGE_CANDIDATE_REPRESENTATIONS = {
    COVERAGE_CANDIDATE_REPRESENTATION_TARGETIZED_PREFIX_MEAN_EMBEDDING,
}
_ALLOWED_COVERAGE_RANK_WEIGHTINGS = {
    COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK,
    COVERAGE_RANK_WEIGHTING_NONE,
}
_ALLOWED_COVERAGE_SIMILARITIES = {COVERAGE_SIMILARITY_COSINE}
RANK_BUCKET_CEM_WARM_START_SURROGATE_EVALUATOR = "warm_start_fine_tune"
RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR = "full_retrain_validation_best"
_ALLOWED_RANK_BUCKET_CEM_SURROGATE_EVALUATORS = {
    RANK_BUCKET_CEM_WARM_START_SURROGATE_EVALUATOR,
    RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR,
}
RANK_BUCKET_CEM_ZERO_MEAN_INIT_MODE = "zero_mean"
RANK_BUCKET_CEM_TAIL_BOOSTED_INIT_MODE = "tail_boosted"
_ALLOWED_RANK_BUCKET_CEM_INIT_MODES = {
    RANK_BUCKET_CEM_ZERO_MEAN_INIT_MODE,
    RANK_BUCKET_CEM_TAIL_BOOSTED_INIT_MODE,
}
ANCHOR_CONSTRUCTION_SOURCE_VULNERABLE_VALIDATION_LAST_ITEM = (
    "vulnerable_validation_last_item"
)
ANCHOR_CONSTRUCTION_SOURCE_POPULAR_TRAIN_ITEMS = "popular_train_items"
ANCHOR_CONSTRUCTION_STRATEGY_ROUND_ROBIN = "round_robin"
_ALLOWED_ANCHOR_CONSTRUCTION_SOURCES = {
    ANCHOR_CONSTRUCTION_SOURCE_POPULAR_TRAIN_ITEMS,
    ANCHOR_CONSTRUCTION_SOURCE_VULNERABLE_VALIDATION_LAST_ITEM,
}
_ALLOWED_ANCHOR_ASSIGNMENT_STRATEGIES = {
    ANCHOR_CONSTRUCTION_STRATEGY_ROUND_ROBIN,
}
PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1 = "grouped_cem_v1"
PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM = "continuous_mlp_cem"
PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM = "direct_action_mlp_cem"
PTS_PREFIX_RANGE_INTERNAL = "internal"
PTS_PREFIX_SAMPLER_UNIFORM = "uniform"
PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH = "residual_suffix_length"
PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX = "same_as_residual_suffix"
PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20 = "raw_lowk_mrr_recall_10_20"
PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE = "global_best_candidate"
PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED = "position_opt_seed"
PTS_CEM_SAMPLER_DIRICHLET = "dirichlet"
PTS_CEM_SAMPLER_GAUSSIAN = "gaussian"
PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN = "elite_centered_gaussian"
PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN = (
    "elite_centered_empirical_gaussian"
)
PTS_DIRECT_ACTION_POLICY_PARAMETERIZATION_MLP_H2 = "direct_action_mlp_h2"
PTS_DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE = "z_score"
PTS_CEM_INIT_UNIFORM = "uniform"
PTS_CEM_INIT_VERTEX_STRATIFIED_SPACE_FILLING = "vertex_stratified_space_filling"
PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING = (
    "two_pool_behavior_curve_space_filling"
)
PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST = "validation_best"
PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST = "fixed_last"
PTS_CEM_SURROGATE_REWARD_BEST = "best"
PTS_CEM_SURROGATE_REWARD_LAST = "last"
_ALLOWED_PTS_CEM_SURROGATE_RETRAIN_PROTOCOLS = {
    PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST,
    PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST,
}
_ALLOWED_PTS_CEM_SURROGATE_REWARD_CHECKPOINTS = {
    PTS_CEM_SURROGATE_REWARD_BEST,
    PTS_CEM_SURROGATE_REWARD_LAST,
}
_ALLOWED_PTS_V1_ACTIONS = {
    "keep_residual_suffix",
    "regenerate_residual_suffix",
    "consume_one_keep_rest",
    "consume_one_generate_continuation",
    "consume_all_stop",
}
PTS_CONTINUOUS_BETA_INPUT_SUFFIX_LENGTH_PERCENTILE = "suffix_length_percentile"
PTS_CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA = "linear_log_beta"
PTS_CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2 = "tiny_mlp_log_beta_h2"
PTS_CONTINUOUS_POLICY_PARAMETERIZATION_SUFFIX_LENGTH_MLP = "suffix_length_mlp"
PTS_CONTINUOUS_POLICY_CONSUME_DISTRIBUTION_BETA = "beta"
PTS_CONTINUOUS_BETA_SOURCE_POLICY_Q_AND_RHO_LOGISTIC = "q_and_rho_logistic"
PTS_CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1 = "behavior_covering_v1"
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
class AnchorConstructionConfig:
    enabled: bool = False
    anchor_source: str = ANCHOR_CONSTRUCTION_SOURCE_VULNERABLE_VALIDATION_LAST_ITEM
    anchor_top_m: int = 20
    anchor_assignment_strategy: str = ANCHOR_CONSTRUCTION_STRATEGY_ROUND_ROBIN
    survey_output_dir: str = "outputs/analysis/target_anchor_survey"
    require_survey_file: bool = True

    def __post_init__(self) -> None:
        enabled = _as_bool(self.enabled, "anchor_construction.enabled")
        object.__setattr__(self, "enabled", enabled)

        anchor_source = _as_str(
            self.anchor_source,
            "anchor_construction.anchor_source",
        ).strip().lower()
        if anchor_source not in _ALLOWED_ANCHOR_CONSTRUCTION_SOURCES:
            allowed = ", ".join(sorted(_ALLOWED_ANCHOR_CONSTRUCTION_SOURCES))
            raise ValueError(
                "anchor_construction.anchor_source must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "anchor_source", anchor_source)

        anchor_top_m = _as_int(
            self.anchor_top_m,
            "anchor_construction.anchor_top_m",
        )
        if anchor_top_m < 1:
            raise ValueError("anchor_construction.anchor_top_m must be >= 1.")
        object.__setattr__(self, "anchor_top_m", anchor_top_m)

        strategy = _as_str(
            self.anchor_assignment_strategy,
            "anchor_construction.anchor_assignment_strategy",
        ).strip().lower()
        if strategy not in _ALLOWED_ANCHOR_ASSIGNMENT_STRATEGIES:
            allowed = ", ".join(sorted(_ALLOWED_ANCHOR_ASSIGNMENT_STRATEGIES))
            raise ValueError(
                "anchor_construction.anchor_assignment_strategy must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "anchor_assignment_strategy", strategy)

        survey_output_dir = _as_str(
            self.survey_output_dir,
            "anchor_construction.survey_output_dir",
        ).strip()
        if not survey_output_dir:
            raise ValueError(
                "anchor_construction.survey_output_dir must be a non-empty string."
            )
        object.__setattr__(self, "survey_output_dir", survey_output_dir)

        require_survey_file = _as_bool(
            self.require_survey_file,
            "anchor_construction.require_survey_file",
        )
        object.__setattr__(self, "require_survey_file", require_survey_file)


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
class SurrogateEvalPoisonBalanceConfig:
    enabled: bool = False
    mode: str = "fixed_ratio"
    poison_ratio_in_batch: float = 0.20
    loss_weighting: str = "none"

    def __post_init__(self) -> None:
        enabled = _as_bool(
            self.enabled,
            "attack.rank_bucket_cem.surrogate_eval_poison_balance.enabled",
        )
        object.__setattr__(self, "enabled", enabled)

        mode = _as_str(
            self.mode,
            "attack.rank_bucket_cem.surrogate_eval_poison_balance.mode",
        ).strip().lower()
        if mode != "fixed_ratio":
            raise ValueError(
                "attack.rank_bucket_cem.surrogate_eval_poison_balance.mode "
                "currently supports only 'fixed_ratio'."
            )
        object.__setattr__(self, "mode", mode)

        poison_ratio = _as_float(
            self.poison_ratio_in_batch,
            "attack.rank_bucket_cem.surrogate_eval_poison_balance.poison_ratio_in_batch",
        )
        if not 0.0 < poison_ratio < 1.0:
            raise ValueError(
                "attack.rank_bucket_cem.surrogate_eval_poison_balance."
                "poison_ratio_in_batch must be > 0 and < 1."
            )
        object.__setattr__(self, "poison_ratio_in_batch", poison_ratio)

        loss_weighting = _as_str(
            self.loss_weighting,
            "attack.rank_bucket_cem.surrogate_eval_poison_balance.loss_weighting",
        ).strip().lower()
        if loss_weighting != "none":
            raise ValueError(
                "attack.rank_bucket_cem.surrogate_eval_poison_balance.loss_weighting "
                "currently supports only 'none'."
            )
        object.__setattr__(self, "loss_weighting", loss_weighting)


@dataclass(frozen=True)
class RankBucketCEMSurrogateEvaluatorConfig:
    mode: str = RANK_BUCKET_CEM_WARM_START_SURROGATE_EVALUATOR
    max_epochs: int | None = None
    patience: int | None = None

    def __post_init__(self) -> None:
        mode = _as_str(
            self.mode,
            "attack.rank_bucket_cem.surrogate_evaluator.mode",
        ).strip().lower()
        if mode not in _ALLOWED_RANK_BUCKET_CEM_SURROGATE_EVALUATORS:
            allowed = ", ".join(sorted(_ALLOWED_RANK_BUCKET_CEM_SURROGATE_EVALUATORS))
            raise ValueError(
                "attack.rank_bucket_cem.surrogate_evaluator.mode must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "mode", mode)

        if self.max_epochs is not None:
            max_epochs = _as_int(
                self.max_epochs,
                "attack.rank_bucket_cem.surrogate_evaluator.max_epochs",
            )
            if max_epochs <= 0:
                raise ValueError(
                    "attack.rank_bucket_cem.surrogate_evaluator.max_epochs "
                    "must be positive when provided."
                )
            object.__setattr__(self, "max_epochs", max_epochs)

        if self.patience is not None:
            patience = _as_int(
                self.patience,
                "attack.rank_bucket_cem.surrogate_evaluator.patience",
            )
            if patience <= 0:
                raise ValueError(
                    "attack.rank_bucket_cem.surrogate_evaluator.patience "
                    "must be positive when provided."
                )
            object.__setattr__(self, "patience", patience)


@dataclass(frozen=True)
class RankBucketCEMConfig:
    iterations: int = 3
    population_size: int = 8
    population_per_iteration: tuple[int, ...] | None = None
    elite_ratio: float = 0.25
    initial_std: float = 1.0
    cem_init_mode: str = RANK_BUCKET_CEM_ZERO_MEAN_INIT_MODE
    g2_initial_pi: tuple[float, float] | None = None
    g3_initial_pi: tuple[float, float, float] | None = None
    min_std: float = 0.2
    smoothing: float = 0.3
    reward_metric: str | None = None
    save_candidate_selected_positions: bool = False
    save_final_selected_positions: bool = False
    save_optimized_poisoned_sessions: bool = True
    save_replay_metadata: bool = True
    surrogate_eval_poison_balance: SurrogateEvalPoisonBalanceConfig = field(
        default_factory=SurrogateEvalPoisonBalanceConfig
    )
    surrogate_evaluator: RankBucketCEMSurrogateEvaluatorConfig = field(
        default_factory=RankBucketCEMSurrogateEvaluatorConfig
    )

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

        population_per_iteration = self.population_per_iteration
        if population_per_iteration is not None:
            schedule = _as_int_list(
                population_per_iteration,
                "attack.rank_bucket_cem.population_per_iteration",
            )
            if not schedule:
                raise ValueError(
                    "attack.rank_bucket_cem.population_per_iteration must not be empty."
                )
            if len(schedule) != iterations:
                raise ValueError(
                    "attack.rank_bucket_cem.population_per_iteration length must "
                    "equal attack.rank_bucket_cem.iterations."
                )
            if any(int(value) <= 0 for value in schedule):
                raise ValueError(
                    "attack.rank_bucket_cem.population_per_iteration entries must "
                    "be positive."
                )
            population_per_iteration = tuple(int(value) for value in schedule)
        object.__setattr__(self, "population_per_iteration", population_per_iteration)

        elite_ratio = _as_float(self.elite_ratio, "attack.rank_bucket_cem.elite_ratio")
        if not 0.0 < elite_ratio <= 1.0:
            raise ValueError("attack.rank_bucket_cem.elite_ratio must be in (0, 1].")
        object.__setattr__(self, "elite_ratio", elite_ratio)

        initial_std = _as_float(self.initial_std, "attack.rank_bucket_cem.initial_std")
        if initial_std <= 0.0:
            raise ValueError("attack.rank_bucket_cem.initial_std must be positive.")
        object.__setattr__(self, "initial_std", initial_std)

        cem_init_mode = _as_str(
            self.cem_init_mode,
            "attack.rank_bucket_cem.cem_init_mode",
        ).strip().lower()
        if cem_init_mode not in _ALLOWED_RANK_BUCKET_CEM_INIT_MODES:
            raise ValueError(
                "attack.rank_bucket_cem.cem_init_mode must be one of "
                f"{sorted(_ALLOWED_RANK_BUCKET_CEM_INIT_MODES)}."
            )
        object.__setattr__(self, "cem_init_mode", cem_init_mode)

        g2_initial_pi = self.g2_initial_pi
        if g2_initial_pi is not None:
            g2_initial_pi = _coerce_probability_tuple(
                g2_initial_pi,
                length=2,
                context="attack.rank_bucket_cem.g2_initial_pi",
            )
        object.__setattr__(self, "g2_initial_pi", g2_initial_pi)

        g3_initial_pi = self.g3_initial_pi
        if g3_initial_pi is not None:
            g3_initial_pi = _coerce_probability_tuple(
                g3_initial_pi,
                length=3,
                context="attack.rank_bucket_cem.g3_initial_pi",
            )
        object.__setattr__(self, "g3_initial_pi", g3_initial_pi)

        if cem_init_mode == RANK_BUCKET_CEM_ZERO_MEAN_INIT_MODE:
            if g2_initial_pi is not None or g3_initial_pi is not None:
                raise ValueError(
                    "attack.rank_bucket_cem.g2_initial_pi/g3_initial_pi require "
                    "cem_init_mode='tail_boosted'."
                )
        elif g2_initial_pi is None or g3_initial_pi is None:
            raise ValueError(
                "attack.rank_bucket_cem.cem_init_mode='tail_boosted' requires both "
                "g2_initial_pi and g3_initial_pi."
            )

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
        poison_balance = self.surrogate_eval_poison_balance
        if isinstance(poison_balance, SurrogateEvalPoisonBalanceConfig):
            resolved_poison_balance = poison_balance
        elif isinstance(poison_balance, Mapping):
            resolved_poison_balance = SurrogateEvalPoisonBalanceConfig(**dict(poison_balance))
        else:
            raise TypeError(
                "attack.rank_bucket_cem.surrogate_eval_poison_balance must be a mapping "
                "or SurrogateEvalPoisonBalanceConfig."
            )
        object.__setattr__(
            self,
            "surrogate_eval_poison_balance",
            resolved_poison_balance,
        )

        surrogate_evaluator = self.surrogate_evaluator
        if isinstance(surrogate_evaluator, RankBucketCEMSurrogateEvaluatorConfig):
            resolved_surrogate_evaluator = surrogate_evaluator
        elif isinstance(surrogate_evaluator, Mapping):
            resolved_surrogate_evaluator = RankBucketCEMSurrogateEvaluatorConfig(
                **dict(surrogate_evaluator)
            )
        else:
            raise TypeError(
                "attack.rank_bucket_cem.surrogate_evaluator must be a mapping "
                "or RankBucketCEMSurrogateEvaluatorConfig."
            )
        object.__setattr__(
            self,
            "surrogate_evaluator",
            resolved_surrogate_evaluator,
        )

    @property
    def effective_population_schedule(self) -> tuple[int, ...]:
        if self.population_per_iteration is None:
            return tuple([int(self.population_size)] * int(self.iterations))
        return tuple(int(value) for value in self.population_per_iteration)

    @property
    def candidate_count(self) -> int:
        return int(sum(self.effective_population_schedule))


@dataclass(frozen=True)
class PTSPrefixSelectorConfig:
    range: str = PTS_PREFIX_RANGE_INTERNAL
    sampler: str = PTS_PREFIX_SAMPLER_UNIFORM

    def __post_init__(self) -> None:
        range_name = _as_str(self.range, "attack.pts_construction.prefix_selector.range").strip().lower()
        sampler = _as_str(self.sampler, "attack.pts_construction.prefix_selector.sampler").strip().lower()
        if range_name != PTS_PREFIX_RANGE_INTERNAL or sampler != PTS_PREFIX_SAMPLER_UNIFORM:
            raise ValueError(
                "attack.pts_construction.prefix_selector supports only "
                "range='internal' and sampler='uniform' in Phase 3."
            )
        object.__setattr__(self, "range", range_name)
        object.__setattr__(self, "sampler", sampler)


@dataclass(frozen=True)
class PTSSuffixLengthBucketConfig:
    name: str
    min: int
    max: int | None = None

    def __post_init__(self) -> None:
        name = _as_str(self.name, "attack.pts_construction.grouping.buckets[].name").strip()
        if not name:
            raise ValueError("attack.pts_construction.grouping.buckets[].name must be non-empty.")
        min_len = _as_int(self.min, "attack.pts_construction.grouping.buckets[].min")
        if min_len < 1:
            raise ValueError("attack.pts_construction.grouping.buckets[].min must be >= 1.")
        max_len = self.max
        if max_len is not None:
            max_len = _as_int(max_len, "attack.pts_construction.grouping.buckets[].max")
            if max_len < min_len:
                raise ValueError(
                    "attack.pts_construction.grouping.buckets[].max must be >= min."
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "min", min_len)
        object.__setattr__(self, "max", max_len)


def _default_pts_suffix_buckets() -> tuple[PTSSuffixLengthBucketConfig, ...]:
    return (
        PTSSuffixLengthBucketConfig(name="suffix_1", min=1, max=1),
        PTSSuffixLengthBucketConfig(name="suffix_2", min=2, max=2),
        PTSSuffixLengthBucketConfig(name="suffix_3plus", min=3, max=None),
    )


@dataclass(frozen=True)
class PTSGroupingConfig:
    mode: str = PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH
    buckets: tuple[PTSSuffixLengthBucketConfig, ...] = field(
        default_factory=_default_pts_suffix_buckets
    )

    def __post_init__(self) -> None:
        mode = _as_str(self.mode, "attack.pts_construction.grouping.mode").strip().lower()
        if mode != PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH:
            raise ValueError(
                "attack.pts_construction.grouping.mode must be "
                "'residual_suffix_length' in Phase 3."
            )
        buckets = _coerce_pts_bucket_configs(
            self.buckets,
            "attack.pts_construction.grouping.buckets",
        )
        if not buckets:
            raise ValueError("attack.pts_construction.grouping.buckets must not be empty.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "buckets", buckets)


@dataclass(frozen=True)
class PTSActionsDynamicMasksConfig:
    disable_consume_one_when_suffix_len_leq_1: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "disable_consume_one_when_suffix_len_leq_1",
            _as_bool(
                self.disable_consume_one_when_suffix_len_leq_1,
                "attack.pts_construction.actions.dynamic_masks."
                "disable_consume_one_when_suffix_len_leq_1",
            ),
        )


@dataclass(frozen=True)
class PTSActionsConfig:
    enabled: tuple[str, ...] = (
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_one_generate_continuation",
        "consume_all_stop",
    )
    dynamic_masks: PTSActionsDynamicMasksConfig = field(
        default_factory=PTSActionsDynamicMasksConfig
    )

    def __post_init__(self) -> None:
        enabled = tuple(
            str(action).strip()
            for action in _as_str_list(
                self.enabled,
                "attack.pts_construction.actions.enabled",
            )
        )
        if not enabled:
            raise ValueError("attack.pts_construction.actions.enabled must not be empty.")
        unknown = set(enabled) - _ALLOWED_PTS_V1_ACTIONS
        if unknown:
            raise ValueError(
                "attack.pts_construction.actions.enabled contains unsupported "
                "Phase 3 actions: "
                + ", ".join(sorted(unknown))
            )
        if len(set(enabled)) != len(enabled):
            raise ValueError("attack.pts_construction.actions.enabled must not contain duplicates.")

        dynamic_masks = self.dynamic_masks
        if isinstance(dynamic_masks, PTSActionsDynamicMasksConfig):
            resolved_dynamic_masks = dynamic_masks
        elif isinstance(dynamic_masks, Mapping):
            resolved_dynamic_masks = PTSActionsDynamicMasksConfig(**dict(dynamic_masks))
        else:
            raise TypeError(
                "attack.pts_construction.actions.dynamic_masks must be a mapping "
                "or PTSActionsDynamicMasksConfig."
            )
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "dynamic_masks", resolved_dynamic_masks)


@dataclass(frozen=True)
class PTSGenerationConfig:
    topk: int = 100
    length_policy: str = PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX

    def __post_init__(self) -> None:
        topk = _as_int(self.topk, "attack.pts_construction.generation.topk")
        if topk <= 0:
            raise ValueError("attack.pts_construction.generation.topk must be positive.")
        length_policy = _as_str(
            self.length_policy,
            "attack.pts_construction.generation.length_policy",
        ).strip().lower()
        if length_policy != PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX:
            raise ValueError(
                "attack.pts_construction.generation.length_policy must be "
                "'same_as_residual_suffix' in Phase 3."
            )
        object.__setattr__(self, "topk", topk)
        object.__setattr__(self, "length_policy", length_policy)


@dataclass(frozen=True)
class PTSCEMSamplerRuntimeConfig:
    type: str = PTS_CEM_SAMPLER_DIRICHLET
    concentration_scale: float = 20.0

    def __post_init__(self) -> None:
        sampler_type = _as_str(self.type, "attack.pts_construction.cem.sampler.type").strip().lower()
        if sampler_type not in {PTS_CEM_SAMPLER_DIRICHLET, PTS_CEM_SAMPLER_GAUSSIAN}:
            raise ValueError(
                "attack.pts_construction.cem.sampler.type must be "
                "'dirichlet' or 'gaussian'."
            )
        concentration_scale = _as_float(
            self.concentration_scale,
            "attack.pts_construction.cem.sampler.concentration_scale",
        )
        if concentration_scale <= 0.0:
            raise ValueError(
                "attack.pts_construction.cem.sampler.concentration_scale must be positive."
            )
        object.__setattr__(self, "type", sampler_type)
        object.__setattr__(self, "concentration_scale", concentration_scale)


@dataclass(frozen=True)
class PTSCEMUpdateRuntimeConfig:
    mode: str = "standard"
    smoothing: float = 0.3
    min_probability: float = 0.03
    max_probability: float = 0.90
    min_std: float = 0.25
    elite_min_std: float = 0.25
    elite_std_scale: float = 1.0

    def __post_init__(self) -> None:
        mode = _as_str(self.mode, "attack.pts_construction.cem.update.mode").strip().lower()
        if mode not in {
            "standard",
            PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN,
            PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN,
        }:
            raise ValueError(
                "attack.pts_construction.cem.update.mode must be 'standard', "
                "'elite_centered_gaussian', or "
                "'elite_centered_empirical_gaussian'."
            )
        smoothing = _as_float(self.smoothing, "attack.pts_construction.cem.update.smoothing")
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError("attack.pts_construction.cem.update.smoothing must be in [0, 1].")
        min_probability = _as_float(
            self.min_probability,
            "attack.pts_construction.cem.update.min_probability",
        )
        max_probability = _as_float(
            self.max_probability,
            "attack.pts_construction.cem.update.max_probability",
        )
        if not 0.0 <= min_probability < max_probability <= 1.0:
            raise ValueError(
                "attack.pts_construction.cem.update min/max probabilities must satisfy "
                "0 <= min < max <= 1."
            )
        min_std = _as_float(self.min_std, "attack.pts_construction.cem.update.min_std")
        if min_std <= 0.0:
            raise ValueError("attack.pts_construction.cem.update.min_std must be positive.")
        elite_min_std = _as_float(
            self.elite_min_std,
            "attack.pts_construction.cem.update.elite_min_std",
        )
        if elite_min_std <= 0.0:
            raise ValueError(
                "attack.pts_construction.cem.update.elite_min_std must be positive."
            )
        elite_std_scale = _as_float(
            self.elite_std_scale,
            "attack.pts_construction.cem.update.elite_std_scale",
        )
        if elite_std_scale < 0.0:
            raise ValueError(
                "attack.pts_construction.cem.update.elite_std_scale must be non-negative."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "smoothing", smoothing)
        object.__setattr__(self, "min_probability", min_probability)
        object.__setattr__(self, "max_probability", max_probability)
        object.__setattr__(self, "min_std", min_std)
        object.__setattr__(self, "elite_min_std", elite_min_std)
        object.__setattr__(self, "elite_std_scale", elite_std_scale)


@dataclass(frozen=True)
class PTSCEMInitRuntimeConfig:
    mode: str = PTS_CEM_INIT_VERTEX_STRATIFIED_SPACE_FILLING
    mandatory_enabled: bool = True
    extreme_count: int = 7
    moderate_count: int = 3
    balanced_count: int = 1
    extreme_pool_size: int = 1024
    moderate_pool_size: int = 512
    extreme_alpha: float = 0.3
    moderate_alpha: float = 2.0
    distance: str = "l1"
    soft_extreme_pool_size: int = 512
    moderate_pool_size: int = 512
    soft_extreme_select_size: int = 5
    moderate_select_size: int = 11
    soft_extreme_initial_std: float = 1.25
    moderate_initial_std: float = 0.8
    q_grid_size: int = 19
    behavior_distance: str = "l1"
    init_materialize_generated_suffix: bool = False

    def __post_init__(self) -> None:
        mode = _as_str(self.mode, "attack.pts_construction.cem.init.mode").strip().lower()
        if mode not in {
            PTS_CEM_INIT_UNIFORM,
            PTS_CEM_INIT_VERTEX_STRATIFIED_SPACE_FILLING,
            PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING,
        }:
            raise ValueError(
                "attack.pts_construction.cem.init.mode must be 'uniform', "
                "'vertex_stratified_space_filling', or "
                "'two_pool_behavior_curve_space_filling'."
            )
        mandatory_enabled = _as_bool(
            self.mandatory_enabled,
            "attack.pts_construction.cem.init.mandatory_enabled",
        )
        extreme_count = _as_int(
            self.extreme_count,
            "attack.pts_construction.cem.init.extreme_count",
        )
        moderate_count = _as_int(
            self.moderate_count,
            "attack.pts_construction.cem.init.moderate_count",
        )
        balanced_count = _as_int(
            self.balanced_count,
            "attack.pts_construction.cem.init.balanced_count",
        )
        extreme_pool_size = _as_int(
            self.extreme_pool_size,
            "attack.pts_construction.cem.init.extreme_pool_size",
        )
        moderate_pool_size = _as_int(
            self.moderate_pool_size,
            "attack.pts_construction.cem.init.moderate_pool_size",
        )
        extreme_alpha = _as_float(
            self.extreme_alpha,
            "attack.pts_construction.cem.init.extreme_alpha",
        )
        moderate_alpha = _as_float(
            self.moderate_alpha,
            "attack.pts_construction.cem.init.moderate_alpha",
        )
        distance = _as_str(
            self.distance,
            "attack.pts_construction.cem.init.distance",
        ).strip().lower()
        soft_extreme_pool_size = _as_int(
            self.soft_extreme_pool_size,
            "attack.pts_construction.cem.init.soft_extreme_pool_size",
        )
        moderate_pool_size_formal = _as_int(
            self.moderate_pool_size,
            "attack.pts_construction.cem.init.moderate_pool_size",
        )
        soft_extreme_select_size = _as_int(
            self.soft_extreme_select_size,
            "attack.pts_construction.cem.init.soft_extreme_select_size",
        )
        moderate_select_size = _as_int(
            self.moderate_select_size,
            "attack.pts_construction.cem.init.moderate_select_size",
        )
        soft_extreme_initial_std = _as_float(
            self.soft_extreme_initial_std,
            "attack.pts_construction.cem.init.soft_extreme_initial_std",
        )
        moderate_initial_std = _as_float(
            self.moderate_initial_std,
            "attack.pts_construction.cem.init.moderate_initial_std",
        )
        q_grid_size = _as_int(
            self.q_grid_size,
            "attack.pts_construction.cem.init.q_grid_size",
        )
        behavior_distance = _as_str(
            self.behavior_distance,
            "attack.pts_construction.cem.init.behavior_distance",
        ).strip().lower()
        init_materialize_generated_suffix = _as_bool(
            self.init_materialize_generated_suffix,
            "attack.pts_construction.cem.init.init_materialize_generated_suffix",
        )
        if extreme_count < 0:
            raise ValueError("attack.pts_construction.cem.init.extreme_count must be >= 0.")
        if moderate_count < 0:
            raise ValueError("attack.pts_construction.cem.init.moderate_count must be >= 0.")
        if balanced_count not in {0, 1}:
            raise ValueError("attack.pts_construction.cem.init.balanced_count must be 0 or 1.")
        if extreme_pool_size < extreme_count:
            raise ValueError(
                "attack.pts_construction.cem.init.extreme_pool_size must be "
                ">= extreme_count."
            )
        if moderate_pool_size < moderate_count:
            raise ValueError(
                "attack.pts_construction.cem.init.moderate_pool_size must be "
                ">= moderate_count."
            )
        if extreme_alpha <= 0.0:
            raise ValueError("attack.pts_construction.cem.init.extreme_alpha must be positive.")
        if moderate_alpha <= 0.0:
            raise ValueError("attack.pts_construction.cem.init.moderate_alpha must be positive.")
        if distance != "l1":
            raise ValueError("attack.pts_construction.cem.init.distance must be 'l1'.")
        if soft_extreme_pool_size <= 0 or moderate_pool_size_formal <= 0:
            raise ValueError("continuous MLP init pool sizes must be positive.")
        if soft_extreme_select_size < 0 or moderate_select_size < 0:
            raise ValueError("continuous MLP init select sizes must be non-negative.")
        if soft_extreme_initial_std <= 0.0 or moderate_initial_std <= 0.0:
            raise ValueError("continuous MLP init std values must be positive.")
        if q_grid_size <= 1:
            raise ValueError("attack.pts_construction.cem.init.q_grid_size must be > 1.")
        if behavior_distance != "l1":
            raise ValueError(
                "attack.pts_construction.cem.init.behavior_distance must be 'l1'."
            )
        if bool(init_materialize_generated_suffix):
            raise ValueError(
                "attack.pts_construction.cem.init.init_materialize_generated_suffix "
                "must be false for target-independent initialization."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "mandatory_enabled", mandatory_enabled)
        object.__setattr__(self, "extreme_count", extreme_count)
        object.__setattr__(self, "moderate_count", moderate_count)
        object.__setattr__(self, "balanced_count", balanced_count)
        object.__setattr__(self, "extreme_pool_size", extreme_pool_size)
        object.__setattr__(self, "moderate_pool_size", moderate_pool_size)
        object.__setattr__(self, "extreme_alpha", extreme_alpha)
        object.__setattr__(self, "moderate_alpha", moderate_alpha)
        object.__setattr__(self, "distance", distance)
        object.__setattr__(self, "soft_extreme_pool_size", soft_extreme_pool_size)
        object.__setattr__(self, "moderate_pool_size", moderate_pool_size_formal)
        object.__setattr__(self, "soft_extreme_select_size", soft_extreme_select_size)
        object.__setattr__(self, "moderate_select_size", moderate_select_size)
        object.__setattr__(self, "soft_extreme_initial_std", soft_extreme_initial_std)
        object.__setattr__(self, "moderate_initial_std", moderate_initial_std)
        object.__setattr__(self, "q_grid_size", q_grid_size)
        object.__setattr__(self, "behavior_distance", behavior_distance)
        object.__setattr__(
            self,
            "init_materialize_generated_suffix",
            init_materialize_generated_suffix,
        )


@dataclass(frozen=True)
class PTSCEMResamplingRuntimeConfig:
    mode: str = "elite_centered"
    local_concentration_scale: float = 30.0

    def __post_init__(self) -> None:
        mode = _as_str(
            self.mode,
            "attack.pts_construction.cem.resampling.mode",
        ).strip().lower()
        if mode not in {"standard", "elite_centered"}:
            raise ValueError(
                "attack.pts_construction.cem.resampling.mode must be "
                "'standard' or 'elite_centered'."
            )
        local_concentration_scale = _as_float(
            self.local_concentration_scale,
            "attack.pts_construction.cem.resampling.local_concentration_scale",
        )
        if local_concentration_scale <= 0.0:
            raise ValueError(
                "attack.pts_construction.cem.resampling.local_concentration_scale "
                "must be positive."
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "local_concentration_scale",
            local_concentration_scale,
        )


@dataclass(frozen=True)
class PTSCEMEpochRewardDiagnosticsRuntimeConfig:
    enabled: bool = False
    epochs: tuple[int, ...] = ()
    include_final_epoch: bool = True
    write_candidate_epoch_metrics: bool = True
    write_ranking_summary: bool = True

    def __post_init__(self) -> None:
        enabled = _as_bool(
            self.enabled,
            "attack.pts_construction.cem.epoch_reward_diagnostics.enabled",
        )
        epochs = tuple(
            _as_int_list(
                self.epochs,
                "attack.pts_construction.cem.epoch_reward_diagnostics.epochs",
            )
        )
        if enabled and not epochs:
            raise ValueError(
                "attack.pts_construction.cem.epoch_reward_diagnostics.epochs "
                "must not be empty when enabled=true."
            )
        if any(epoch <= 0 for epoch in epochs):
            raise ValueError(
                "attack.pts_construction.cem.epoch_reward_diagnostics.epochs "
                "must contain only positive integers."
            )
        if len(set(epochs)) != len(epochs):
            raise ValueError(
                "attack.pts_construction.cem.epoch_reward_diagnostics.epochs "
                "must not contain duplicates."
            )
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(
            self,
            "include_final_epoch",
            _as_bool(
                self.include_final_epoch,
                "attack.pts_construction.cem.epoch_reward_diagnostics."
                "include_final_epoch",
            ),
        )
        object.__setattr__(
            self,
            "write_candidate_epoch_metrics",
            _as_bool(
                self.write_candidate_epoch_metrics,
                "attack.pts_construction.cem.epoch_reward_diagnostics."
                "write_candidate_epoch_metrics",
            ),
        )
        object.__setattr__(
            self,
            "write_ranking_summary",
            _as_bool(
                self.write_ranking_summary,
                "attack.pts_construction.cem.epoch_reward_diagnostics."
                "write_ranking_summary",
            ),
        )


@dataclass(frozen=True)
class PTSCEMSurrogateRetrainRuntimeConfig:
    checkpoint_protocol: str = PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST
    validation_enabled: bool = True
    reward_checkpoint: str = PTS_CEM_SURROGATE_REWARD_BEST

    def __post_init__(self) -> None:
        context = "attack.pts_construction.cem.surrogate_retrain"
        checkpoint_protocol = _as_str(
            self.checkpoint_protocol,
            f"{context}.checkpoint_protocol",
        ).strip().lower()
        if checkpoint_protocol not in _ALLOWED_PTS_CEM_SURROGATE_RETRAIN_PROTOCOLS:
            allowed = ", ".join(sorted(_ALLOWED_PTS_CEM_SURROGATE_RETRAIN_PROTOCOLS))
            raise ValueError(
                f"{context}.checkpoint_protocol must be one of: {allowed}."
            )
        validation_enabled = _as_bool(
            self.validation_enabled,
            f"{context}.validation_enabled",
        )
        reward_checkpoint = _as_str(
            self.reward_checkpoint,
            f"{context}.reward_checkpoint",
        ).strip().lower()
        if reward_checkpoint not in _ALLOWED_PTS_CEM_SURROGATE_REWARD_CHECKPOINTS:
            allowed = ", ".join(sorted(_ALLOWED_PTS_CEM_SURROGATE_REWARD_CHECKPOINTS))
            raise ValueError(
                f"{context}.reward_checkpoint must be one of: {allowed}."
            )
        if checkpoint_protocol == PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST:
            if not validation_enabled or reward_checkpoint != PTS_CEM_SURROGATE_REWARD_BEST:
                raise ValueError(
                    f"{context}.checkpoint_protocol=validation_best requires "
                    "validation_enabled=true and reward_checkpoint=best."
                )
        if checkpoint_protocol == PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST:
            if validation_enabled or reward_checkpoint != PTS_CEM_SURROGATE_REWARD_LAST:
                raise ValueError(
                    f"{context}.checkpoint_protocol=fixed_last requires "
                    "validation_enabled=false and reward_checkpoint=last."
                )
        object.__setattr__(self, "checkpoint_protocol", checkpoint_protocol)
        object.__setattr__(self, "validation_enabled", validation_enabled)
        object.__setattr__(self, "reward_checkpoint", reward_checkpoint)


@dataclass(frozen=True)
class PTSCEMRuntimeConfig:
    iterations: int = 3
    population_schedule: tuple[int, ...] | None = (16, 8, 8)
    population_size: int | None = None
    elite_ratio: float = 0.25
    sampler: PTSCEMSamplerRuntimeConfig = field(default_factory=PTSCEMSamplerRuntimeConfig)
    update: PTSCEMUpdateRuntimeConfig = field(default_factory=PTSCEMUpdateRuntimeConfig)
    init: PTSCEMInitRuntimeConfig = field(default_factory=PTSCEMInitRuntimeConfig)
    resampling: PTSCEMResamplingRuntimeConfig = field(
        default_factory=PTSCEMResamplingRuntimeConfig
    )
    seed_source: str = PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED
    candidate_seed_stride: int = 1000
    save_top_k_candidates: int = 3
    epoch_reward_diagnostics: PTSCEMEpochRewardDiagnosticsRuntimeConfig = field(
        default_factory=PTSCEMEpochRewardDiagnosticsRuntimeConfig
    )
    surrogate_retrain: PTSCEMSurrogateRetrainRuntimeConfig = field(
        default_factory=PTSCEMSurrogateRetrainRuntimeConfig
    )

    def __post_init__(self) -> None:
        iterations = _as_int(self.iterations, "attack.pts_construction.cem.iterations")
        if iterations <= 0:
            raise ValueError("attack.pts_construction.cem.iterations must be positive.")
        population_schedule = self.population_schedule
        if population_schedule is not None:
            population_schedule = tuple(
                _as_int_list(
                    population_schedule,
                    "attack.pts_construction.cem.population_schedule",
                )
            )
            if len(population_schedule) != iterations:
                raise ValueError(
                    "attack.pts_construction.cem.population_schedule length must "
                    "equal iterations."
                )
            if any(value <= 0 for value in population_schedule):
                raise ValueError(
                    "attack.pts_construction.cem.population_schedule values must be positive."
                )
        population_size = self.population_size
        if population_size is not None:
            population_size = _as_int(
                population_size,
                "attack.pts_construction.cem.population_size",
            )
            if population_size <= 0:
                raise ValueError("attack.pts_construction.cem.population_size must be positive.")
        if population_schedule is None and population_size is None:
            raise ValueError(
                "attack.pts_construction.cem requires population_schedule or population_size."
            )
        elite_ratio = _as_float(self.elite_ratio, "attack.pts_construction.cem.elite_ratio")
        if not 0.0 < elite_ratio <= 1.0:
            raise ValueError("attack.pts_construction.cem.elite_ratio must be in (0, 1].")
        sampler = _coerce_pts_dataclass(
            self.sampler,
            PTSCEMSamplerRuntimeConfig,
            "attack.pts_construction.cem.sampler",
        )
        update = _coerce_pts_dataclass(
            self.update,
            PTSCEMUpdateRuntimeConfig,
            "attack.pts_construction.cem.update",
        )
        init = _coerce_pts_dataclass(
            self.init,
            PTSCEMInitRuntimeConfig,
            "attack.pts_construction.cem.init",
        )
        resampling = _coerce_pts_dataclass(
            self.resampling,
            PTSCEMResamplingRuntimeConfig,
            "attack.pts_construction.cem.resampling",
        )
        epoch_reward_diagnostics = _coerce_pts_dataclass(
            self.epoch_reward_diagnostics,
            PTSCEMEpochRewardDiagnosticsRuntimeConfig,
            "attack.pts_construction.cem.epoch_reward_diagnostics",
        )
        surrogate_retrain = _coerce_pts_dataclass(
            self.surrogate_retrain,
            PTSCEMSurrogateRetrainRuntimeConfig,
            "attack.pts_construction.cem.surrogate_retrain",
        )
        seed_source = _as_str(
            self.seed_source,
            "attack.pts_construction.cem.seed_source",
        ).strip().lower()
        if seed_source != PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED:
            raise ValueError(
                "attack.pts_construction.cem.seed_source currently supports only "
                "'position_opt_seed'."
            )
        candidate_seed_stride = _as_int(
            self.candidate_seed_stride,
            "attack.pts_construction.cem.candidate_seed_stride",
        )
        if candidate_seed_stride <= 0:
            raise ValueError(
                "attack.pts_construction.cem.candidate_seed_stride must be positive."
            )
        save_top_k_candidates = _as_int(
            self.save_top_k_candidates,
            "attack.pts_construction.cem.save_top_k_candidates",
        )
        if save_top_k_candidates < 0:
            raise ValueError(
                "attack.pts_construction.cem.save_top_k_candidates must be >= 0."
            )
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "population_schedule", population_schedule)
        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "elite_ratio", elite_ratio)
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "update", update)
        object.__setattr__(self, "init", init)
        object.__setattr__(self, "resampling", resampling)
        object.__setattr__(
            self,
            "epoch_reward_diagnostics",
            epoch_reward_diagnostics,
        )
        object.__setattr__(self, "surrogate_retrain", surrogate_retrain)
        object.__setattr__(self, "seed_source", seed_source)
        object.__setattr__(self, "candidate_seed_stride", candidate_seed_stride)
        object.__setattr__(self, "save_top_k_candidates", save_top_k_candidates)


@dataclass(frozen=True)
class PTSContinuousParameterBoundsConfig:
    min: float = -5.0
    max: float = 5.0

    def __post_init__(self) -> None:
        minimum = _as_float(
            self.min,
            "attack.pts_construction.continuous_policy.parameter_bounds.min",
        )
        maximum = _as_float(
            self.max,
            "attack.pts_construction.continuous_policy.parameter_bounds.max",
        )
        if not minimum < maximum:
            raise ValueError(
                "attack.pts_construction.continuous_policy.parameter_bounds "
                "must satisfy min < max."
            )
        object.__setattr__(self, "min", minimum)
        object.__setattr__(self, "max", maximum)


@dataclass(frozen=True)
class PTSContinuousPolicyConfig:
    parameterization: str = PTS_CONTINUOUS_POLICY_PARAMETERIZATION_SUFFIX_LENGTH_MLP
    hidden_size: int = 2
    consume_distribution: str = PTS_CONTINUOUS_POLICY_CONSUME_DISTRIBUTION_BETA
    smoothing_epsilon: float = 0.0
    source_policy: str = PTS_CONTINUOUS_BETA_SOURCE_POLICY_Q_AND_RHO_LOGISTIC
    parameter_bounds: PTSContinuousParameterBoundsConfig = field(
        default_factory=PTSContinuousParameterBoundsConfig
    )
    deterministic_sampling: bool = True

    def __post_init__(self) -> None:
        parameterization = _as_str(
            self.parameterization,
            "attack.pts_construction.continuous_policy.parameterization",
        ).strip().lower()
        if parameterization != PTS_CONTINUOUS_POLICY_PARAMETERIZATION_SUFFIX_LENGTH_MLP:
            raise ValueError(
                "attack.pts_construction.continuous_policy.parameterization "
                "must be 'suffix_length_mlp'."
            )
        hidden_size = _as_int(
            self.hidden_size,
            "attack.pts_construction.continuous_policy.hidden_size",
        )
        if hidden_size != 2:
            raise ValueError(
                "attack.pts_construction.continuous_policy.hidden_size must be 2."
            )
        consume_distribution = _as_str(
            self.consume_distribution,
            "attack.pts_construction.continuous_policy.consume_distribution",
        ).strip().lower()
        if consume_distribution != PTS_CONTINUOUS_POLICY_CONSUME_DISTRIBUTION_BETA:
            raise ValueError(
                "attack.pts_construction.continuous_policy.consume_distribution "
                "must be 'beta'."
            )
        source_policy = _as_str(
            self.source_policy,
            "attack.pts_construction.continuous_policy.source_policy",
        ).strip().lower()
        if source_policy != PTS_CONTINUOUS_BETA_SOURCE_POLICY_Q_AND_RHO_LOGISTIC:
            raise ValueError(
                "attack.pts_construction.continuous_policy.source_policy "
                "must be 'q_and_rho_logistic'."
            )
        smoothing_epsilon = _as_float(
            self.smoothing_epsilon,
            "attack.pts_construction.continuous_policy.smoothing_epsilon",
        )
        if not 0.0 <= smoothing_epsilon < 0.5:
            raise ValueError(
                "attack.pts_construction.continuous_policy.smoothing_epsilon "
                "must satisfy 0.0 <= epsilon < 0.5."
            )
        parameter_bounds = _coerce_pts_dataclass(
            self.parameter_bounds,
            PTSContinuousParameterBoundsConfig,
            "attack.pts_construction.continuous_policy.parameter_bounds",
        )
        deterministic_sampling = _as_bool(
            self.deterministic_sampling,
            "attack.pts_construction.continuous_policy.deterministic_sampling",
        )
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "consume_distribution", consume_distribution)
        object.__setattr__(self, "smoothing_epsilon", smoothing_epsilon)
        object.__setattr__(self, "source_policy", source_policy)
        object.__setattr__(self, "parameter_bounds", parameter_bounds)
        object.__setattr__(self, "deterministic_sampling", deterministic_sampling)

    @property
    def internal_parameterization(self) -> str:
        return PTS_CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2


@dataclass(frozen=True)
class PTSDirectActionPolicyConfig:
    parameterization: str = PTS_DIRECT_ACTION_POLICY_PARAMETERIZATION_MLP_H2
    length_feature: str = PTS_DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE

    def __post_init__(self) -> None:
        parameterization = _as_str(
            self.parameterization,
            "attack.pts_construction.direct_action_policy.parameterization",
        ).strip().lower()
        if parameterization != PTS_DIRECT_ACTION_POLICY_PARAMETERIZATION_MLP_H2:
            raise ValueError(
                "attack.pts_construction.direct_action_policy.parameterization "
                "must be 'direct_action_mlp_h2'."
            )
        length_feature = _as_str(
            self.length_feature,
            "attack.pts_construction.direct_action_policy.length_feature",
        ).strip().lower()
        if length_feature not in {"z_score", "z_score_m"}:
            raise ValueError(
                "attack.pts_construction.direct_action_policy.length_feature "
                "must be 'z_score'."
            )
        object.__setattr__(self, "parameterization", parameterization)
        object.__setattr__(self, "length_feature", "z_score")


@dataclass(frozen=True)
class PTSRewardConfig:
    target_summary: str = PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20
    enable_gt_penalty: bool = False
    gt_penalty_weight: float = 0.0
    enable_length_penalty: bool = False
    length_penalty_weight: float = 0.0

    def __post_init__(self) -> None:
        target_summary = _as_str(
            self.target_summary,
            "attack.pts_construction.reward.target_summary",
        ).strip().lower()
        if target_summary != PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20:
            raise ValueError(
                "attack.pts_construction.reward.target_summary currently supports "
                "only 'raw_lowk_mrr_recall_10_20'."
            )
        enable_gt_penalty = _as_bool(
            self.enable_gt_penalty,
            "attack.pts_construction.reward.enable_gt_penalty",
        )
        gt_penalty_weight = _as_float(
            self.gt_penalty_weight,
            "attack.pts_construction.reward.gt_penalty_weight",
        )
        enable_length_penalty = _as_bool(
            self.enable_length_penalty,
            "attack.pts_construction.reward.enable_length_penalty",
        )
        length_penalty_weight = _as_float(
            self.length_penalty_weight,
            "attack.pts_construction.reward.length_penalty_weight",
        )
        if enable_gt_penalty:
            raise NotImplementedError(
                "attack.pts_construction.reward.enable_gt_penalty is a Phase 3 "
                "placeholder and is not implemented."
            )
        if enable_length_penalty:
            raise NotImplementedError(
                "attack.pts_construction.reward.enable_length_penalty is a Phase 3 "
                "placeholder and is not implemented."
            )
        if gt_penalty_weight < 0.0 or length_penalty_weight < 0.0:
            raise ValueError("PTS reward penalty weights must be non-negative.")
        object.__setattr__(self, "target_summary", target_summary)
        object.__setattr__(self, "enable_gt_penalty", enable_gt_penalty)
        object.__setattr__(self, "gt_penalty_weight", gt_penalty_weight)
        object.__setattr__(self, "enable_length_penalty", enable_length_penalty)
        object.__setattr__(self, "length_penalty_weight", length_penalty_weight)


@dataclass(frozen=True)
class PTSArtifactsConfig:
    save_cem_trace: bool = True
    save_best_policy: bool = True
    save_final_policy: bool = True
    save_per_session_records: bool = True
    save_candidate_sessions: bool = False
    save_best_sessions: bool = True
    save_top_candidate_sessions: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "save_cem_trace",
            "save_best_policy",
            "save_final_policy",
            "save_per_session_records",
            "save_candidate_sessions",
            "save_best_sessions",
            "save_top_candidate_sessions",
        ):
            object.__setattr__(
                self,
                field_name,
                _as_bool(
                    getattr(self, field_name),
                    f"attack.pts_construction.artifacts.{field_name}",
                ),
            )
        if bool(self.save_candidate_sessions):
            raise ValueError(
                "attack.pts_construction.artifacts.save_candidate_sessions is not "
                "supported in Phase 3 and must be false."
            )


@dataclass(frozen=True)
class PTSFinalSelectionConfig:
    mode: str = PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE

    def __post_init__(self) -> None:
        mode = _as_str(
            self.mode,
            "attack.pts_construction.final_selection.mode",
        ).strip().lower()
        if mode != PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE:
            raise ValueError(
                "attack.pts_construction.final_selection.mode currently supports "
                "only 'global_best_candidate'."
            )
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class PTSConstructionConfig:
    enabled: bool = False
    method: str = PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1
    prefix_selector: PTSPrefixSelectorConfig = field(default_factory=PTSPrefixSelectorConfig)
    grouping: PTSGroupingConfig = field(default_factory=PTSGroupingConfig)
    actions: PTSActionsConfig = field(default_factory=PTSActionsConfig)
    generation: PTSGenerationConfig = field(default_factory=PTSGenerationConfig)
    continuous_policy: PTSContinuousPolicyConfig = field(
        default_factory=PTSContinuousPolicyConfig
    )
    direct_action_policy: PTSDirectActionPolicyConfig = field(
        default_factory=PTSDirectActionPolicyConfig
    )
    cem: PTSCEMRuntimeConfig = field(default_factory=PTSCEMRuntimeConfig)
    reward: PTSRewardConfig = field(default_factory=PTSRewardConfig)
    artifacts: PTSArtifactsConfig = field(default_factory=PTSArtifactsConfig)
    final_selection: PTSFinalSelectionConfig = field(default_factory=PTSFinalSelectionConfig)

    def __post_init__(self) -> None:
        enabled = _as_bool(self.enabled, "attack.pts_construction.enabled")
        method = _as_str(self.method, "attack.pts_construction.method").strip().lower()
        if method not in {
            PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1,
            PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM,
            PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM,
        }:
            raise ValueError(
                "attack.pts_construction.method must be 'grouped_cem_v1', "
                "'continuous_mlp_cem', or 'direct_action_mlp_cem'."
            )
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self,
            "prefix_selector",
            _coerce_pts_dataclass(
                self.prefix_selector,
                PTSPrefixSelectorConfig,
                "attack.pts_construction.prefix_selector",
            ),
        )
        object.__setattr__(
            self,
            "grouping",
            _coerce_pts_dataclass(
                self.grouping,
                PTSGroupingConfig,
                "attack.pts_construction.grouping",
            ),
        )
        object.__setattr__(
            self,
            "actions",
            _coerce_pts_dataclass(
                self.actions,
                PTSActionsConfig,
                "attack.pts_construction.actions",
            ),
        )
        object.__setattr__(
            self,
            "generation",
            _coerce_pts_dataclass(
                self.generation,
                PTSGenerationConfig,
                "attack.pts_construction.generation",
            ),
        )
        object.__setattr__(
            self,
            "continuous_policy",
            _coerce_pts_dataclass(
                self.continuous_policy,
                PTSContinuousPolicyConfig,
                "attack.pts_construction.continuous_policy",
            ),
        )
        object.__setattr__(
            self,
            "direct_action_policy",
            _coerce_pts_dataclass(
                self.direct_action_policy,
                PTSDirectActionPolicyConfig,
                "attack.pts_construction.direct_action_policy",
            ),
        )
        object.__setattr__(
            self,
            "cem",
            _coerce_pts_dataclass(
                self.cem,
                PTSCEMRuntimeConfig,
                "attack.pts_construction.cem",
            ),
        )
        object.__setattr__(
            self,
            "reward",
            _coerce_pts_dataclass(
                self.reward,
                PTSRewardConfig,
                "attack.pts_construction.reward",
            ),
        )
        object.__setattr__(
            self,
            "artifacts",
            _coerce_pts_dataclass(
                self.artifacts,
                PTSArtifactsConfig,
                "attack.pts_construction.artifacts",
            ),
        )
        object.__setattr__(
            self,
            "final_selection",
            _coerce_pts_dataclass(
                self.final_selection,
                PTSFinalSelectionConfig,
                "attack.pts_construction.final_selection",
            ),
        )
        if (
            method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1
            and self.cem.sampler.type != PTS_CEM_SAMPLER_DIRICHLET
        ):
            raise ValueError(
                "attack.pts_construction.cem.sampler.type must be 'dirichlet' "
                "for method='grouped_cem_v1'."
            )
        if method == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
            if self.cem.sampler.type != PTS_CEM_SAMPLER_GAUSSIAN:
                raise ValueError(
                    "Direct-action MLP-CEM requires cem.sampler.type='gaussian'."
                )
            if (
                self.cem.update.mode
                != PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN
            ):
                raise ValueError(
                    "Direct-action MLP-CEM requires "
                    "cem.update.mode='elite_centered_empirical_gaussian'."
                )
        if (
            method == PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1
            and self.cem.init.mode == PTS_CEM_INIT_VERTEX_STRATIFIED_SPACE_FILLING
            and bool(self.cem.init.mandatory_enabled)
            and "consume_one_generate_continuation" not in set(self.actions.enabled)
        ):
            raise ValueError(
                "vertex_stratified_space_filling with mandatory_enabled=true "
                "requires consume_one_generate_continuation in actions.enabled, "
                "because the c1_generate_where_valid mandatory vertex cannot be "
                "constructed."
            )


@dataclass(frozen=True)
class CreatAdditiveSBRConfig:
    enabled: bool = False
    variant: str = CREAT_ADDITIVE_SBR_VARIANT_V1
    epochs: int = 10
    attack_epochs: int = 10
    consistency_epochs: int = 10
    batch_size: int = 128
    lr: float = 1.0e-3
    hidden_dim: int = 64
    position_embedding_dim: int = 16
    max_attack_num: int = 1
    nonzero_when_possible: bool = True
    stealth_weight: float = 0.1
    local_weight: float = 0.0
    entropy_weight: float = 0.0
    pattern_reward_weight: float = 0.1
    dpp_reward_weight: float = 0.0
    dpp_score_mode: str = CREAT_ADDITIVE_SBR_DPP_BOUNDED_DETERMINANT
    dpp_eps: float = 1.0e-6
    global_consistency_weight: float = 0.1
    local_consistency_weight: float = 0.1
    local_window_size: int = 3
    consistency_mode: str = CREAT_ADDITIVE_SBR_CONSISTENCY_LOCAL_GLOBAL
    final_policy_selection: str = CREAT_ADDITIVE_SBR_FINAL_POLICY_LAST
    attack_reward_mode: str = CREAT_ADDITIVE_SBR_ATTACK_REWARD_SCORE
    seed_source: str = CREAT_ADDITIVE_SBR_SEED_SOURCE_POSITION_OPT_SEED

    def __post_init__(self) -> None:
        enabled = _as_bool(self.enabled, "attack.creat_additive_sbr.enabled")
        variant = _as_str(self.variant, "attack.creat_additive_sbr.variant").strip().lower()
        epochs = _as_int(self.epochs, "attack.creat_additive_sbr.epochs")
        attack_epochs = _as_int(
            self.attack_epochs, "attack.creat_additive_sbr.attack_epochs"
        )
        consistency_epochs = _as_int(
            self.consistency_epochs, "attack.creat_additive_sbr.consistency_epochs"
        )
        batch_size = _as_int(self.batch_size, "attack.creat_additive_sbr.batch_size")
        lr = _as_float(self.lr, "attack.creat_additive_sbr.lr")
        hidden_dim = _as_int(self.hidden_dim, "attack.creat_additive_sbr.hidden_dim")
        position_embedding_dim = _as_int(
            self.position_embedding_dim,
            "attack.creat_additive_sbr.position_embedding_dim",
        )
        max_attack_num = _as_int(
            self.max_attack_num,
            "attack.creat_additive_sbr.max_attack_num",
        )
        nonzero_when_possible = _as_bool(
            self.nonzero_when_possible,
            "attack.creat_additive_sbr.nonzero_when_possible",
        )
        stealth_weight = _as_float(
            self.stealth_weight,
            "attack.creat_additive_sbr.stealth_weight",
        )
        local_weight = _as_float(
            self.local_weight,
            "attack.creat_additive_sbr.local_weight",
        )
        entropy_weight = _as_float(
            self.entropy_weight,
            "attack.creat_additive_sbr.entropy_weight",
        )
        pattern_reward_weight = _as_float(
            self.pattern_reward_weight,
            "attack.creat_additive_sbr.pattern_reward_weight",
        )
        dpp_reward_weight = _as_float(
            self.dpp_reward_weight,
            "attack.creat_additive_sbr.dpp_reward_weight",
        )
        dpp_score_mode = _as_str(
            self.dpp_score_mode,
            "attack.creat_additive_sbr.dpp_score_mode",
        ).strip().lower()
        dpp_eps = _as_float(self.dpp_eps, "attack.creat_additive_sbr.dpp_eps")
        global_consistency_weight = _as_float(
            self.global_consistency_weight,
            "attack.creat_additive_sbr.global_consistency_weight",
        )
        local_consistency_weight = _as_float(
            self.local_consistency_weight,
            "attack.creat_additive_sbr.local_consistency_weight",
        )
        local_window_size = _as_int(
            self.local_window_size,
            "attack.creat_additive_sbr.local_window_size",
        )
        consistency_mode = _as_str(
            self.consistency_mode,
            "attack.creat_additive_sbr.consistency_mode",
        ).strip().lower()
        final_policy_selection = _as_str(
            self.final_policy_selection,
            "attack.creat_additive_sbr.final_policy_selection",
        ).strip().lower()
        attack_reward_mode = _as_str(
            self.attack_reward_mode,
            "attack.creat_additive_sbr.attack_reward_mode",
        ).strip().lower()
        seed_source = _as_str(
            self.seed_source,
            "attack.creat_additive_sbr.seed_source",
        ).strip().lower()

        if variant not in {CREAT_ADDITIVE_SBR_VARIANT_V1, CREAT_ADDITIVE_SBR_VARIANT_V2}:
            raise ValueError("attack.creat_additive_sbr.variant must be 'v1' or 'v2'.")
        if epochs < 0:
            raise ValueError("attack.creat_additive_sbr.epochs must be non-negative.")
        if enabled and variant == CREAT_ADDITIVE_SBR_VARIANT_V1 and epochs <= 0:
            raise ValueError(
                "attack.creat_additive_sbr.epochs must be positive when "
                "v1 is enabled."
            )
        if attack_epochs < 0 or consistency_epochs < 0:
            raise ValueError("attack.creat_additive_sbr v2 epochs must be non-negative.")
        if enabled and variant == CREAT_ADDITIVE_SBR_VARIANT_V2:
            if attack_epochs <= 0 or consistency_epochs <= 0:
                raise ValueError(
                    "attack.creat_additive_sbr attack_epochs and consistency_epochs "
                    "must be positive when v2 is enabled."
                )
        if batch_size <= 0:
            raise ValueError("attack.creat_additive_sbr.batch_size must be positive.")
        if lr <= 0.0:
            raise ValueError("attack.creat_additive_sbr.lr must be positive.")
        if hidden_dim <= 0:
            raise ValueError("attack.creat_additive_sbr.hidden_dim must be positive.")
        if position_embedding_dim <= 0:
            raise ValueError(
                "attack.creat_additive_sbr.position_embedding_dim must be positive."
            )
        if max_attack_num != 1:
            raise ValueError("attack.creat_additive_sbr.max_attack_num must be 1.")
        reward_weights = (
            stealth_weight,
            local_weight,
            entropy_weight,
            pattern_reward_weight,
            dpp_reward_weight,
            global_consistency_weight,
            local_consistency_weight,
        )
        if any(weight < 0.0 for weight in reward_weights):
            raise ValueError("attack.creat_additive_sbr reward weights must be non-negative.")
        if dpp_score_mode not in {
            CREAT_ADDITIVE_SBR_DPP_BOUNDED_DETERMINANT,
            CREAT_ADDITIVE_SBR_DPP_RAW_LOGDET,
        }:
            raise ValueError(
                "attack.creat_additive_sbr.dpp_score_mode must be "
                "'bounded_determinant' or 'raw_logdet'."
            )
        if dpp_eps <= 0.0:
            raise ValueError("attack.creat_additive_sbr.dpp_eps must be positive.")
        if local_window_size <= 0 or local_window_size % 2 == 0:
            raise ValueError(
                "attack.creat_additive_sbr.local_window_size must be a positive odd integer."
            )
        if consistency_mode != CREAT_ADDITIVE_SBR_CONSISTENCY_LOCAL_GLOBAL:
            raise ValueError(
                "attack.creat_additive_sbr.consistency_mode currently supports "
                "only 'local_global'."
            )
        if final_policy_selection != CREAT_ADDITIVE_SBR_FINAL_POLICY_LAST:
            raise ValueError(
                "attack.creat_additive_sbr.final_policy_selection currently supports "
                "only 'last'."
            )
        if attack_reward_mode != CREAT_ADDITIVE_SBR_ATTACK_REWARD_SCORE:
            raise ValueError(
                "attack.creat_additive_sbr.attack_reward_mode currently supports only 'score'."
            )
        if seed_source != CREAT_ADDITIVE_SBR_SEED_SOURCE_POSITION_OPT_SEED:
            raise ValueError(
                "attack.creat_additive_sbr.seed_source currently supports only "
                "'position_opt_seed'."
            )

        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "attack_epochs", attack_epochs)
        object.__setattr__(self, "consistency_epochs", consistency_epochs)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "hidden_dim", hidden_dim)
        object.__setattr__(self, "position_embedding_dim", position_embedding_dim)
        object.__setattr__(self, "max_attack_num", max_attack_num)
        object.__setattr__(self, "nonzero_when_possible", nonzero_when_possible)
        object.__setattr__(self, "stealth_weight", stealth_weight)
        object.__setattr__(self, "local_weight", local_weight)
        object.__setattr__(self, "entropy_weight", entropy_weight)
        object.__setattr__(self, "pattern_reward_weight", pattern_reward_weight)
        object.__setattr__(self, "dpp_reward_weight", dpp_reward_weight)
        object.__setattr__(self, "dpp_score_mode", dpp_score_mode)
        object.__setattr__(self, "dpp_eps", dpp_eps)
        object.__setattr__(self, "global_consistency_weight", global_consistency_weight)
        object.__setattr__(self, "local_consistency_weight", local_consistency_weight)
        object.__setattr__(self, "local_window_size", local_window_size)
        object.__setattr__(self, "consistency_mode", consistency_mode)
        object.__setattr__(self, "final_policy_selection", final_policy_selection)
        object.__setattr__(self, "attack_reward_mode", attack_reward_mode)
        object.__setattr__(self, "seed_source", seed_source)


def _coerce_pts_dataclass(value: Any, cls: type[Any], context: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls(**dict(value))
    raise TypeError(f"{context} must be a mapping or {cls.__name__}.")


def _coerce_pts_bucket_configs(
    value: Any,
    context: str,
) -> tuple[PTSSuffixLengthBucketConfig, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{context} must be a list of suffix-length buckets.")
    buckets: list[PTSSuffixLengthBucketConfig] = []
    for index, item in enumerate(value):
        item_context = f"{context}[{index}]"
        if isinstance(item, PTSSuffixLengthBucketConfig):
            buckets.append(item)
        elif isinstance(item, Mapping):
            buckets.append(PTSSuffixLengthBucketConfig(**dict(item)))
        else:
            raise TypeError(f"{item_context} must be a mapping or PTSSuffixLengthBucketConfig.")
    return tuple(buckets)


@dataclass(frozen=True)
class CarrierSelectionConfig:
    enabled: bool = False
    candidate_pool_size: float = 0.03
    final_attack_size: float = 0.01
    scorer: str = TARGET_AWARE_CARRIER_SELECTION_SCORER
    embedding_weight: float = 0.4
    cooccurrence_weight: float = 0.3
    transition_weight: float = 0.3
    use_length_control: bool = True
    length_buckets: str = TARGET_AWARE_CARRIER_SELECTION_LENGTH_BUCKETS_EXACT_UNTIL_4_PLUS
    normalize: str = TARGET_AWARE_CARRIER_SELECTION_NORMALIZE_MINMAX
    placement_mode: str | None = None
    operation: str | None = None
    candidate_positions: str | None = None
    local_embedding_weight: float = 0.5
    local_transition_weight: float = 0.5
    session_compatibility_weight: float = 0.0
    left_to_target_weight: float = 0.5
    target_to_right_weight: float = 0.5
    debug_save_all_session_records: bool = False
    coverage_prefix_source: str = COVERAGE_PREFIX_SOURCE_VALIDATION
    vulnerable_rank_min: int = 20
    vulnerable_rank_max: int = 200
    max_vulnerable_prefixes: int = 5000
    prefix_representation: str = COVERAGE_PREFIX_REPRESENTATION_MEAN_ITEM_EMBEDDING
    candidate_representation: str = (
        COVERAGE_CANDIDATE_REPRESENTATION_TARGETIZED_PREFIX_MEAN_EMBEDDING
    )
    top_m_coverage: int = 20
    rank_weighting: str = COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK
    coverage_similarity: str = COVERAGE_SIMILARITY_COSINE
    debug_save_all_position_records: bool = False

    def __post_init__(self) -> None:
        enabled = _as_bool(self.enabled, "attack.carrier_selection.enabled")
        object.__setattr__(self, "enabled", enabled)

        candidate_pool_size = _as_float(
            self.candidate_pool_size,
            "attack.carrier_selection.candidate_pool_size",
        )
        if not 0.0 < candidate_pool_size <= 1.0:
            raise ValueError(
                "attack.carrier_selection.candidate_pool_size must be in (0, 1]."
            )
        object.__setattr__(self, "candidate_pool_size", candidate_pool_size)

        final_attack_size = _as_float(
            self.final_attack_size,
            "attack.carrier_selection.final_attack_size",
        )
        if not 0.0 < final_attack_size <= candidate_pool_size:
            raise ValueError(
                "attack.carrier_selection.final_attack_size must be in "
                "(0, candidate_pool_size]."
            )
        object.__setattr__(self, "final_attack_size", final_attack_size)

        scorer = _as_str(self.scorer, "attack.carrier_selection.scorer").strip().lower()
        if scorer not in _ALLOWED_CARRIER_SELECTION_SCORERS:
            allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_SCORERS))
            raise ValueError(f"attack.carrier_selection.scorer must be one of: {allowed}.")
        object.__setattr__(self, "scorer", scorer)

        embedding_weight = _as_float(
            self.embedding_weight,
            "attack.carrier_selection.embedding_weight",
        )
        cooccurrence_weight = _as_float(
            self.cooccurrence_weight,
            "attack.carrier_selection.cooccurrence_weight",
        )
        transition_weight = _as_float(
            self.transition_weight,
            "attack.carrier_selection.transition_weight",
        )
        weights = (embedding_weight, cooccurrence_weight, transition_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("attack.carrier_selection weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError("attack.carrier_selection weights must sum to > 0.")
        object.__setattr__(self, "embedding_weight", embedding_weight)
        object.__setattr__(self, "cooccurrence_weight", cooccurrence_weight)
        object.__setattr__(self, "transition_weight", transition_weight)

        use_length_control = _as_bool(
            self.use_length_control,
            "attack.carrier_selection.use_length_control",
        )
        object.__setattr__(self, "use_length_control", use_length_control)

        length_buckets = _as_str(
            self.length_buckets,
            "attack.carrier_selection.length_buckets",
        ).strip().lower()
        if length_buckets not in _ALLOWED_CARRIER_SELECTION_LENGTH_BUCKETS:
            allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_LENGTH_BUCKETS))
            raise ValueError(
                "attack.carrier_selection.length_buckets must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "length_buckets", length_buckets)

        normalize = _as_str(
            self.normalize,
            "attack.carrier_selection.normalize",
        ).strip().lower()
        if normalize not in _ALLOWED_CARRIER_SELECTION_NORMALIZE:
            allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_NORMALIZE))
            raise ValueError(
                "attack.carrier_selection.normalize must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "normalize", normalize)

        placement_mode = self.placement_mode
        if placement_mode is not None:
            placement_mode = _as_str(
                placement_mode,
                "attack.carrier_selection.placement_mode",
            ).strip().lower()
            if placement_mode not in _ALLOWED_CARRIER_SELECTION_PLACEMENT_MODES:
                allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_PLACEMENT_MODES))
                raise ValueError(
                    "attack.carrier_selection.placement_mode must be one of: "
                    f"{allowed}."
                )
        object.__setattr__(self, "placement_mode", placement_mode)

        operation = self.operation
        if operation is not None:
            operation = _as_str(
                operation,
                "attack.carrier_selection.operation",
            ).strip().lower()
            if operation not in _ALLOWED_CARRIER_SELECTION_OPERATIONS:
                allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_OPERATIONS))
                raise ValueError(
                    "attack.carrier_selection.operation must be one of: "
                    f"{allowed}."
                )
        object.__setattr__(self, "operation", operation)

        candidate_positions = self.candidate_positions
        if candidate_positions is not None:
            candidate_positions = _as_str(
                candidate_positions,
                "attack.carrier_selection.candidate_positions",
            ).strip().lower()
            if candidate_positions not in _ALLOWED_CARRIER_SELECTION_CANDIDATE_POSITIONS:
                allowed = ", ".join(sorted(_ALLOWED_CARRIER_SELECTION_CANDIDATE_POSITIONS))
                raise ValueError(
                    "attack.carrier_selection.candidate_positions must be one of: "
                    f"{allowed}."
                )
        object.__setattr__(self, "candidate_positions", candidate_positions)

        local_embedding_weight = _as_float(
            self.local_embedding_weight,
            "attack.carrier_selection.local_embedding_weight",
        )
        local_transition_weight = _as_float(
            self.local_transition_weight,
            "attack.carrier_selection.local_transition_weight",
        )
        session_compatibility_weight = _as_float(
            self.session_compatibility_weight,
            "attack.carrier_selection.session_compatibility_weight",
        )
        local_weights = (
            local_embedding_weight,
            local_transition_weight,
            session_compatibility_weight,
        )
        if any(weight < 0.0 for weight in local_weights):
            raise ValueError("attack.carrier_selection local weights must be non-negative.")
        if scorer == TARGET_AWARE_CARRIER_LOCAL_POSITION_SCORER and sum(local_weights) <= 0.0:
            raise ValueError("attack.carrier_selection local weights must sum to > 0.")
        object.__setattr__(self, "local_embedding_weight", local_embedding_weight)
        object.__setattr__(self, "local_transition_weight", local_transition_weight)
        object.__setattr__(self, "session_compatibility_weight", session_compatibility_weight)

        left_to_target_weight = _as_float(
            self.left_to_target_weight,
            "attack.carrier_selection.left_to_target_weight",
        )
        target_to_right_weight = _as_float(
            self.target_to_right_weight,
            "attack.carrier_selection.target_to_right_weight",
        )
        direction_weights = (left_to_target_weight, target_to_right_weight)
        if any(weight < 0.0 for weight in direction_weights):
            raise ValueError(
                "attack.carrier_selection directional transition weights must be non-negative."
            )
        if sum(direction_weights) <= 0.0:
            raise ValueError(
                "attack.carrier_selection directional transition weights must sum to > 0."
            )
        object.__setattr__(self, "left_to_target_weight", left_to_target_weight)
        object.__setattr__(self, "target_to_right_weight", target_to_right_weight)

        debug_save_all_session_records = _as_bool(
            self.debug_save_all_session_records,
            "attack.carrier_selection.debug_save_all_session_records",
        )
        object.__setattr__(
            self,
            "debug_save_all_session_records",
            debug_save_all_session_records,
        )

        coverage_prefix_source = _as_str(
            self.coverage_prefix_source,
            "attack.carrier_selection.coverage_prefix_source",
        ).strip().lower()
        if coverage_prefix_source not in _ALLOWED_COVERAGE_PREFIX_SOURCES:
            allowed = ", ".join(sorted(_ALLOWED_COVERAGE_PREFIX_SOURCES))
            raise ValueError(
                "attack.carrier_selection.coverage_prefix_source must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "coverage_prefix_source", coverage_prefix_source)

        vulnerable_rank_min = _as_int(
            self.vulnerable_rank_min,
            "attack.carrier_selection.vulnerable_rank_min",
        )
        if vulnerable_rank_min < 1:
            raise ValueError(
                "attack.carrier_selection.vulnerable_rank_min must be >= 1."
            )
        object.__setattr__(self, "vulnerable_rank_min", vulnerable_rank_min)

        vulnerable_rank_max = _as_int(
            self.vulnerable_rank_max,
            "attack.carrier_selection.vulnerable_rank_max",
        )
        if vulnerable_rank_max <= vulnerable_rank_min:
            raise ValueError(
                "attack.carrier_selection.vulnerable_rank_max must be > "
                "vulnerable_rank_min."
            )
        object.__setattr__(self, "vulnerable_rank_max", vulnerable_rank_max)

        max_vulnerable_prefixes = _as_int(
            self.max_vulnerable_prefixes,
            "attack.carrier_selection.max_vulnerable_prefixes",
        )
        if max_vulnerable_prefixes < 1:
            raise ValueError(
                "attack.carrier_selection.max_vulnerable_prefixes must be >= 1."
            )
        object.__setattr__(self, "max_vulnerable_prefixes", max_vulnerable_prefixes)

        prefix_representation = _as_str(
            self.prefix_representation,
            "attack.carrier_selection.prefix_representation",
        ).strip().lower()
        if prefix_representation not in _ALLOWED_COVERAGE_PREFIX_REPRESENTATIONS:
            allowed = ", ".join(sorted(_ALLOWED_COVERAGE_PREFIX_REPRESENTATIONS))
            raise ValueError(
                "attack.carrier_selection.prefix_representation must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "prefix_representation", prefix_representation)

        candidate_representation = _as_str(
            self.candidate_representation,
            "attack.carrier_selection.candidate_representation",
        ).strip().lower()
        if candidate_representation not in _ALLOWED_COVERAGE_CANDIDATE_REPRESENTATIONS:
            allowed = ", ".join(sorted(_ALLOWED_COVERAGE_CANDIDATE_REPRESENTATIONS))
            raise ValueError(
                "attack.carrier_selection.candidate_representation must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "candidate_representation", candidate_representation)

        top_m_coverage = _as_int(
            self.top_m_coverage,
            "attack.carrier_selection.top_m_coverage",
        )
        if top_m_coverage < 1:
            raise ValueError("attack.carrier_selection.top_m_coverage must be >= 1.")
        object.__setattr__(self, "top_m_coverage", top_m_coverage)

        rank_weighting = _as_str(
            self.rank_weighting,
            "attack.carrier_selection.rank_weighting",
        ).strip().lower()
        if rank_weighting not in _ALLOWED_COVERAGE_RANK_WEIGHTINGS:
            allowed = ", ".join(sorted(_ALLOWED_COVERAGE_RANK_WEIGHTINGS))
            raise ValueError(
                "attack.carrier_selection.rank_weighting must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "rank_weighting", rank_weighting)

        coverage_similarity = _as_str(
            self.coverage_similarity,
            "attack.carrier_selection.coverage_similarity",
        ).strip().lower()
        if coverage_similarity not in _ALLOWED_COVERAGE_SIMILARITIES:
            allowed = ", ".join(sorted(_ALLOWED_COVERAGE_SIMILARITIES))
            raise ValueError(
                "attack.carrier_selection.coverage_similarity must be one of: "
                f"{allowed}."
            )
        object.__setattr__(self, "coverage_similarity", coverage_similarity)

        debug_save_all_position_records = _as_bool(
            self.debug_save_all_position_records,
            "attack.carrier_selection.debug_save_all_position_records",
        )
        object.__setattr__(
            self,
            "debug_save_all_position_records",
            debug_save_all_position_records,
        )


@dataclass(frozen=True)
class TrainTemplateFallbackConfig:
    nearest_length_redistribution: bool = True
    replacement_if_needed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nearest_length_redistribution",
            _as_bool(
                self.nearest_length_redistribution,
                "attack.fake_session_source.train_template.fallback.nearest_length_redistribution",
            ),
        )
        object.__setattr__(
            self,
            "replacement_if_needed",
            _as_bool(
                self.replacement_if_needed,
                "attack.fake_session_source.train_template.fallback.replacement_if_needed",
            ),
        )


@dataclass(frozen=True)
class TrainTemplateSourceConfig:
    reference_split: str = TRAIN_TEMPLATE_REFERENCE_SPLIT_TRAIN_SUB
    target_filtering: str = TRAIN_TEMPLATE_TARGET_FILTERING_NONE
    replacement: bool = False
    fallback: TrainTemplateFallbackConfig | Mapping[str, Any] | None = field(
        default_factory=TrainTemplateFallbackConfig
    )
    record_distribution_diagnostics: bool = True

    def __post_init__(self) -> None:
        reference_split = _as_str(
            self.reference_split,
            "attack.fake_session_source.train_template.reference_split",
        ).strip()
        if reference_split != TRAIN_TEMPLATE_REFERENCE_SPLIT_TRAIN_SUB:
            raise ValueError(
                "attack.fake_session_source.train_template.reference_split currently "
                f"supports only {TRAIN_TEMPLATE_REFERENCE_SPLIT_TRAIN_SUB!r}."
            )
        object.__setattr__(self, "reference_split", reference_split)

        target_filtering = _as_str(
            self.target_filtering,
            "attack.fake_session_source.train_template.target_filtering",
        ).strip().lower()
        if target_filtering != TRAIN_TEMPLATE_TARGET_FILTERING_NONE:
            raise ValueError(
                "attack.fake_session_source.train_template.target_filtering currently "
                f"supports only {TRAIN_TEMPLATE_TARGET_FILTERING_NONE!r}."
            )
        object.__setattr__(self, "target_filtering", target_filtering)

        object.__setattr__(
            self,
            "replacement",
            _as_bool(
                self.replacement,
                "attack.fake_session_source.train_template.replacement",
            ),
        )
        if bool(self.replacement):
            raise ValueError(
                "train_template.replacement=true is not supported; current "
                "train-template source uses without-replacement sampling with "
                "fallback replacement only when needed."
            )
        fallback = self.fallback
        if fallback is None:
            fallback_config = TrainTemplateFallbackConfig()
        elif isinstance(fallback, TrainTemplateFallbackConfig):
            fallback_config = fallback
        else:
            fallback_config = TrainTemplateFallbackConfig(
                **dict(
                    _as_mapping(
                        fallback,
                        "attack.fake_session_source.train_template.fallback",
                    )
                )
            )
        object.__setattr__(self, "fallback", fallback_config)
        object.__setattr__(
            self,
            "record_distribution_diagnostics",
            _as_bool(
                self.record_distribution_diagnostics,
                "attack.fake_session_source.train_template.record_distribution_diagnostics",
            ),
        )


@dataclass(frozen=True)
class FakeSessionSourceConfig:
    type: str = FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED
    train_template: TrainTemplateSourceConfig | Mapping[str, Any] | None = field(
        default_factory=TrainTemplateSourceConfig
    )

    def __post_init__(self) -> None:
        source_type = _as_str(self.type, "attack.fake_session_source.type").strip().lower()
        allowed = {
            FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED,
            FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
        }
        if source_type not in allowed:
            raise ValueError(
                "attack.fake_session_source.type must be one of: "
                + ", ".join(sorted(allowed))
            )
        object.__setattr__(self, "type", source_type)

        train_template = self.train_template
        if train_template is None:
            train_template_config = TrainTemplateSourceConfig()
        elif isinstance(train_template, TrainTemplateSourceConfig):
            train_template_config = train_template
        else:
            train_template_config = TrainTemplateSourceConfig(
                **dict(
                    _as_mapping(
                        train_template,
                        "attack.fake_session_source.train_template",
                    )
                )
            )
        object.__setattr__(self, "train_template", train_template_config)


@dataclass(frozen=True)
class AttackConfig:
    size: float
    fake_session_generation_topk: int
    replacement_topk_ratio: float
    poison_model: PoisonModelConfig
    fake_session_source: FakeSessionSourceConfig = field(
        default_factory=FakeSessionSourceConfig
    )
    position_opt: PositionOptConfig | None = None
    rank_bucket_cem: RankBucketCEMConfig | None = None
    carrier_selection: CarrierSelectionConfig | None = None
    pts_construction: PTSConstructionConfig | None = None
    creat_additive_sbr: CreatAdditiveSBRConfig | None = None


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
    cleanup_victim_intermediates: bool = False


@dataclass(frozen=True)
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    seeds: SeedsConfig
    attack: AttackConfig
    anchor_construction: AnchorConstructionConfig
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
            "anchor_construction": payload["anchor_construction"],
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


def _coerce_probability_tuple(
    value: Any,
    *,
    length: int,
    context: str,
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        keys = ("rank1", "rank2") if int(length) == 2 else ("rank1", "rank2", "tail")
        missing = [key for key in keys if key not in value]
        extra = sorted(str(key) for key in set(value) - set(keys))
        if missing or extra:
            raise ValueError(
                f"Expected {context} keys {list(keys)}; "
                f"missing={missing}, extra={extra}."
            )
        value = [value[key] for key in keys]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected {context} to be a list of probabilities.")
    if len(value) != int(length):
        raise ValueError(f"Expected {context} to contain exactly {int(length)} values.")
    result = tuple(_as_float(item, f"{context}[]") for item in value)
    if any(float(item) <= 0.0 for item in result):
        raise ValueError(f"Expected {context} probabilities to be positive.")
    total = float(sum(result))
    if abs(total - 1.0) > 1.0e-6:
        raise ValueError(f"Expected {context} probabilities to sum to 1.0, got {total}.")
    return tuple(float(item) for item in result)


def _coerce_rank_bucket_pi_mapping_or_sequence(
    value: Any,
    *,
    keys: Sequence[str],
    context: str,
) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        missing = [key for key in keys if key not in value]
        extra = sorted(str(key) for key in set(value) - set(keys))
        if missing or extra:
            raise ValueError(
                f"Expected {context} keys {list(keys)}; "
                f"missing={missing}, extra={extra}."
            )
        value = [value[key] for key in keys]
    return _coerce_probability_tuple(
        value,
        length=len(keys),
        context=context,
    )


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
        "anchor_construction": _as_mapping(
            root.get("anchor_construction", {}),
            "anchor_construction",
        ),
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
    anchor_construction = parsed["anchor_construction"]
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
    normalized_anchor_construction = _normalize_anchor_construction_config(
        anchor_construction
    )
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
        "cleanup_victim_intermediates": _as_bool(
            artifacts.get("cleanup_victim_intermediates", False),
            "artifacts.cleanup_victim_intermediates",
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
        "anchor_construction": normalized_anchor_construction,
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
        "fake_session_source": (
            _normalize_primitive(
                attack["fake_session_source"],
                "attack.fake_session_source",
            )
            if "fake_session_source" in attack and attack["fake_session_source"] is not None
            else None
        ),
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
        "carrier_selection": (
            _normalize_carrier_selection_config(
                attack["carrier_selection"],
                "attack.carrier_selection",
            )
            if "carrier_selection" in attack and attack["carrier_selection"] is not None
            else None
        ),
        "pts_construction": (
            _normalize_pts_construction_config(
                attack["pts_construction"],
                "attack.pts_construction",
            )
            if "pts_construction" in attack and attack["pts_construction"] is not None
            else None
        ),
        "creat_additive_sbr": (
            _normalize_creat_additive_sbr_config(
                attack["creat_additive_sbr"],
                "attack.creat_additive_sbr",
            )
            if "creat_additive_sbr" in attack and attack["creat_additive_sbr"] is not None
            else None
        ),
    }

    if not 0.0 < normalized["size"] <= 1.0:
        raise ValueError("attack.size must be in (0, 1].")
    if normalized["fake_session_generation_topk"] <= 0:
        raise ValueError("attack.fake_session_generation_topk must be positive.")
    if not 0.0 < normalized["replacement_topk_ratio"] <= 1.0:
        raise ValueError("attack.replacement_topk_ratio must be in (0, 1].")
    carrier_selection = normalized["carrier_selection"]
    if carrier_selection is not None:
        final_attack_size = float(carrier_selection["final_attack_size"])
        if abs(final_attack_size - float(normalized["size"])) > 1e-12:
            raise ValueError(
                "attack.carrier_selection.final_attack_size must equal attack.size "
                "for TACS-NZ v1."
            )
    return normalized


def _normalize_creat_additive_sbr_config(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {field.name for field in fields(CreatAdditiveSBRConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown CREAT-Additive-SBR config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )
    return _primitive_from_obj(CreatAdditiveSBRConfig(**dict(mapping)))


def _normalize_pts_construction_config(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {field.name for field in fields(PTSConstructionConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown PTS construction config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )
    if str(mapping.get("method", "")).strip().lower() == PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM:
        direct_action_policy = mapping.get("direct_action_policy", {})
        if isinstance(direct_action_policy, Mapping) and "initial_std" in direct_action_policy:
            raise ValueError(
                "direct_action_mlp_cem no longer supports "
                "direct_action_policy.initial_std; initialization is standard_normal."
            )
        cem = mapping.get("cem", {})
        if isinstance(cem, Mapping):
            update = cem.get("update", {})
            if isinstance(update, Mapping) and "elite_std_scale" in update:
                raise ValueError(
                    "direct_action_mlp_cem no longer supports "
                    "cem.update.elite_std_scale; update uses empirical elite std."
                )
    return _primitive_from_obj(PTSConstructionConfig(**dict(mapping)))


def _normalize_anchor_construction_config(value: Mapping[str, Any]) -> dict[str, Any]:
    mapping = _as_mapping(value, "anchor_construction")
    allowed_fields = {field.name for field in fields(AnchorConstructionConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown anchor_construction config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "enabled" in mapping:
        payload["enabled"] = _as_bool(
            mapping["enabled"],
            "anchor_construction.enabled",
        )
    if "anchor_source" in mapping:
        payload["anchor_source"] = _as_str(
            mapping["anchor_source"],
            "anchor_construction.anchor_source",
        )
    if "anchor_top_m" in mapping:
        payload["anchor_top_m"] = _as_int(
            mapping["anchor_top_m"],
            "anchor_construction.anchor_top_m",
        )
    if "anchor_assignment_strategy" in mapping:
        payload["anchor_assignment_strategy"] = _as_str(
            mapping["anchor_assignment_strategy"],
            "anchor_construction.anchor_assignment_strategy",
        )
    if "survey_output_dir" in mapping:
        payload["survey_output_dir"] = _as_str(
            mapping["survey_output_dir"],
            "anchor_construction.survey_output_dir",
        )
    if "require_survey_file" in mapping:
        payload["require_survey_file"] = _as_bool(
            mapping["require_survey_file"],
            "anchor_construction.require_survey_file",
        )
    return _primitive_from_obj(AnchorConstructionConfig(**payload))


def _normalize_carrier_selection_config(value: Any, context: str) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {field.name for field in fields(CarrierSelectionConfig)}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown carrier-selection config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "enabled" in mapping:
        payload["enabled"] = _as_bool(mapping["enabled"], f"{context}.enabled")
    if "candidate_pool_size" in mapping:
        payload["candidate_pool_size"] = _as_float(
            mapping["candidate_pool_size"],
            f"{context}.candidate_pool_size",
        )
    if "final_attack_size" in mapping:
        payload["final_attack_size"] = _as_float(
            mapping["final_attack_size"],
            f"{context}.final_attack_size",
        )
    if "scorer" in mapping:
        payload["scorer"] = _as_str(mapping["scorer"], f"{context}.scorer")
    if "embedding_weight" in mapping:
        payload["embedding_weight"] = _as_float(
            mapping["embedding_weight"],
            f"{context}.embedding_weight",
        )
    if "cooccurrence_weight" in mapping:
        payload["cooccurrence_weight"] = _as_float(
            mapping["cooccurrence_weight"],
            f"{context}.cooccurrence_weight",
        )
    if "transition_weight" in mapping:
        payload["transition_weight"] = _as_float(
            mapping["transition_weight"],
            f"{context}.transition_weight",
        )
    if "use_length_control" in mapping:
        payload["use_length_control"] = _as_bool(
            mapping["use_length_control"],
            f"{context}.use_length_control",
        )
    if "length_buckets" in mapping:
        payload["length_buckets"] = _as_str(
            mapping["length_buckets"],
            f"{context}.length_buckets",
        )
    if "normalize" in mapping:
        payload["normalize"] = _as_str(mapping["normalize"], f"{context}.normalize")
    if "placement_mode" in mapping:
        placement_mode = mapping["placement_mode"]
        payload["placement_mode"] = (
            None
            if placement_mode is None
            else _as_str(placement_mode, f"{context}.placement_mode")
        )
    if "operation" in mapping:
        operation = mapping["operation"]
        payload["operation"] = (
            None
            if operation is None
            else _as_str(operation, f"{context}.operation")
        )
    if "candidate_positions" in mapping:
        candidate_positions = mapping["candidate_positions"]
        payload["candidate_positions"] = (
            None
            if candidate_positions is None
            else _as_str(candidate_positions, f"{context}.candidate_positions")
        )
    if "local_embedding_weight" in mapping:
        payload["local_embedding_weight"] = _as_float(
            mapping["local_embedding_weight"],
            f"{context}.local_embedding_weight",
        )
    if "local_transition_weight" in mapping:
        payload["local_transition_weight"] = _as_float(
            mapping["local_transition_weight"],
            f"{context}.local_transition_weight",
        )
    if "session_compatibility_weight" in mapping:
        payload["session_compatibility_weight"] = _as_float(
            mapping["session_compatibility_weight"],
            f"{context}.session_compatibility_weight",
        )
    if "left_to_target_weight" in mapping:
        payload["left_to_target_weight"] = _as_float(
            mapping["left_to_target_weight"],
            f"{context}.left_to_target_weight",
        )
    if "target_to_right_weight" in mapping:
        payload["target_to_right_weight"] = _as_float(
            mapping["target_to_right_weight"],
            f"{context}.target_to_right_weight",
        )
    if "debug_save_all_session_records" in mapping:
        payload["debug_save_all_session_records"] = _as_bool(
            mapping["debug_save_all_session_records"],
            f"{context}.debug_save_all_session_records",
        )
    if "coverage_prefix_source" in mapping:
        payload["coverage_prefix_source"] = _as_str(
            mapping["coverage_prefix_source"],
            f"{context}.coverage_prefix_source",
        )
    if "vulnerable_rank_min" in mapping:
        payload["vulnerable_rank_min"] = _as_int(
            mapping["vulnerable_rank_min"],
            f"{context}.vulnerable_rank_min",
        )
    if "vulnerable_rank_max" in mapping:
        payload["vulnerable_rank_max"] = _as_int(
            mapping["vulnerable_rank_max"],
            f"{context}.vulnerable_rank_max",
        )
    if "max_vulnerable_prefixes" in mapping:
        payload["max_vulnerable_prefixes"] = _as_int(
            mapping["max_vulnerable_prefixes"],
            f"{context}.max_vulnerable_prefixes",
        )
    if "prefix_representation" in mapping:
        payload["prefix_representation"] = _as_str(
            mapping["prefix_representation"],
            f"{context}.prefix_representation",
        )
    if "candidate_representation" in mapping:
        payload["candidate_representation"] = _as_str(
            mapping["candidate_representation"],
            f"{context}.candidate_representation",
        )
    if "top_m_coverage" in mapping:
        payload["top_m_coverage"] = _as_int(
            mapping["top_m_coverage"],
            f"{context}.top_m_coverage",
        )
    if "rank_weighting" in mapping:
        payload["rank_weighting"] = _as_str(
            mapping["rank_weighting"],
            f"{context}.rank_weighting",
        )
    if "coverage_similarity" in mapping:
        payload["coverage_similarity"] = _as_str(
            mapping["coverage_similarity"],
            f"{context}.coverage_similarity",
        )
    if "debug_save_all_position_records" in mapping:
        payload["debug_save_all_position_records"] = _as_bool(
            mapping["debug_save_all_position_records"],
            f"{context}.debug_save_all_position_records",
        )

    config = CarrierSelectionConfig(**payload)
    return _primitive_from_obj(config)


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
    if "population_per_iteration" in mapping:
        raw_schedule = mapping["population_per_iteration"]
        payload["population_per_iteration"] = (
            None
            if raw_schedule is None
            else _as_int_list(raw_schedule, f"{context}.population_per_iteration")
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
    if "cem_init_mode" in mapping:
        payload["cem_init_mode"] = _as_str(
            mapping["cem_init_mode"],
            f"{context}.cem_init_mode",
        )
    if "g2_initial_pi" in mapping:
        raw_g2_initial_pi = mapping["g2_initial_pi"]
        payload["g2_initial_pi"] = (
            None
            if raw_g2_initial_pi is None
            else _coerce_rank_bucket_pi_mapping_or_sequence(
                raw_g2_initial_pi,
                keys=("rank1", "rank2"),
                context=f"{context}.g2_initial_pi",
            )
        )
    if "g3_initial_pi" in mapping:
        raw_g3_initial_pi = mapping["g3_initial_pi"]
        payload["g3_initial_pi"] = (
            None
            if raw_g3_initial_pi is None
            else _coerce_rank_bucket_pi_mapping_or_sequence(
                raw_g3_initial_pi,
                keys=("rank1", "rank2", "tail"),
                context=f"{context}.g3_initial_pi",
            )
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
    if "surrogate_eval_poison_balance" in mapping:
        payload["surrogate_eval_poison_balance"] = (
            _normalize_surrogate_eval_poison_balance_config(
                mapping["surrogate_eval_poison_balance"],
                f"{context}.surrogate_eval_poison_balance",
            )
        )
    if "surrogate_evaluator" in mapping:
        payload["surrogate_evaluator"] = _normalize_rank_bucket_cem_surrogate_evaluator_config(
            mapping["surrogate_evaluator"],
            f"{context}.surrogate_evaluator",
        )

    return _primitive_from_obj(RankBucketCEMConfig(**payload))


def _normalize_rank_bucket_cem_surrogate_evaluator_config(
    value: Any,
    context: str,
) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {"mode", "max_epochs", "patience"}
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown rank_bucket_cem surrogate_evaluator config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "mode" in mapping:
        payload["mode"] = _as_str(mapping["mode"], f"{context}.mode")
    if "max_epochs" in mapping:
        raw_max_epochs = mapping["max_epochs"]
        payload["max_epochs"] = (
            None
            if raw_max_epochs is None
            else _as_int(raw_max_epochs, f"{context}.max_epochs")
        )
    if "patience" in mapping:
        raw_patience = mapping["patience"]
        payload["patience"] = (
            None
            if raw_patience is None
            else _as_int(raw_patience, f"{context}.patience")
        )
    return _primitive_from_obj(RankBucketCEMSurrogateEvaluatorConfig(**payload))


def _normalize_surrogate_eval_poison_balance_config(
    value: Any,
    context: str,
) -> dict[str, Any]:
    mapping = _as_mapping(value, context)
    allowed_fields = {
        "enabled",
        "mode",
        "poison_ratio_in_batch",
        "loss_weighting",
    }
    unknown = set(mapping) - allowed_fields
    if unknown:
        raise ValueError(
            "Unknown surrogate_eval_poison_balance config keys: "
            + ", ".join(sorted(map(str, unknown)))
        )

    payload: dict[str, Any] = {}
    if "enabled" in mapping:
        payload["enabled"] = _as_bool(mapping["enabled"], f"{context}.enabled")
    if "mode" in mapping:
        payload["mode"] = _as_str(mapping["mode"], f"{context}.mode")
    if "poison_ratio_in_batch" in mapping:
        payload["poison_ratio_in_batch"] = _as_float(
            mapping["poison_ratio_in_batch"],
            f"{context}.poison_ratio_in_batch",
        )
    if "loss_weighting" in mapping:
        payload["loss_weighting"] = _as_str(
            mapping["loss_weighting"],
            f"{context}.loss_weighting",
        )
    return _primitive_from_obj(SurrogateEvalPoisonBalanceConfig(**payload))


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
    if "mdhg" in enabled:
        if runtime is None or "mdhg" not in runtime:
            raise ValueError("Missing required runtime configuration: victims.runtime.mdhg")
        _validate_mdhg_runtime(runtime["mdhg"], "victims.runtime.mdhg")
    if "freqrec" in enabled:
        if runtime is None or "freqrec" not in runtime:
            raise ValueError("Missing required runtime configuration: victims.runtime.freqrec")
        _validate_freqrec_runtime(runtime["freqrec"], "victims.runtime.freqrec")
    if "wearec" in enabled:
        if runtime is None or "wearec" not in runtime:
            raise ValueError("Missing required runtime configuration: victims.runtime.wearec")
        _validate_wearec_runtime(runtime["wearec"], "victims.runtime.wearec")

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
        elif victim_name == "mdhg":
            primitive["train"] = _normalize_mdhg_train(train, f"{context}.{victim_name}.train")
        elif victim_name == "freqrec":
            primitive["train"] = _normalize_freqrec_train(
                train, f"{context}.{victim_name}.train"
            )
        elif victim_name == "wearec":
            primitive["train"] = _normalize_wearec_train(
                train, f"{context}.{victim_name}.train"
            )
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


def _validate_mdhg_runtime(runtime: dict[str, Any], context: str) -> None:
    _as_str(_require(runtime, "python_executable", context), f"{context}.python_executable")
    _as_str(_require(runtime, "repo_root", context), f"{context}.repo_root")
    _as_str(_require(runtime, "working_dir", context), f"{context}.working_dir")
    device = _as_mapping(_require(runtime, "device", context), f"{context}.device")
    use_gpu = _as_bool(
        _require(device, "use_gpu", f"{context}.device"),
        f"{context}.device.use_gpu",
    )
    if not use_gpu:
        raise ValueError(f"{context}.device.use_gpu must be true; MDHG Phase 1A is GPU-only.")
    _as_gpu_id(_require(device, "gpu_id", f"{context}.device"), f"{context}.device.gpu_id")
    diagnostics_value = runtime.get("diagnostics")
    if diagnostics_value is not None:
        diagnostics = _as_mapping(diagnostics_value, f"{context}.diagnostics")
        for key in ("epoch_metrics", "per_epoch_predictions"):
            if key in diagnostics:
                _as_bool(diagnostics[key], f"{context}.diagnostics.{key}")


def _validate_freqrec_runtime(runtime: dict[str, Any], context: str) -> None:
    _as_str(_require(runtime, "python_executable", context), f"{context}.python_executable")
    _as_str(_require(runtime, "repo_root", context), f"{context}.repo_root")
    _as_str(_require(runtime, "working_dir", context), f"{context}.working_dir")
    device = _as_mapping(_require(runtime, "device", context), f"{context}.device")
    use_gpu = _as_bool(
        _require(device, "use_gpu", f"{context}.device"),
        f"{context}.device.use_gpu",
    )
    gpu_id = _as_gpu_id(
        _require(device, "gpu_id", f"{context}.device"),
        f"{context}.device.gpu_id",
    )
    if use_gpu and ("," in gpu_id or not gpu_id.isdigit()):
        raise ValueError(
            f"{context}.device.gpu_id must identify exactly one non-negative physical GPU."
        )
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
    diagnostics_value = runtime.get("diagnostics")
    if diagnostics_value is not None:
        diagnostics = _as_mapping(diagnostics_value, f"{context}.diagnostics")
        unknown = set(diagnostics) - {
            "epoch_metrics",
            "per_epoch_predictions",
            "save_checkpoint",
        }
        if unknown:
            raise ValueError(
                f"Unknown {context}.diagnostics keys: "
                + ", ".join(sorted(map(str, unknown)))
            )
        for key in diagnostics:
            _as_bool(diagnostics[key], f"{context}.diagnostics.{key}")


def _validate_wearec_runtime(runtime: dict[str, Any], context: str) -> None:
    for field in ("python_executable", "repo_root", "working_dir"):
        value = _as_str(_require(runtime, field, context), f"{context}.{field}")
        if not value.strip():
            raise ValueError(f"{context}.{field} must be a non-empty string.")
    device = _as_mapping(_require(runtime, "device", context), f"{context}.device")
    _as_bool(_require(device, "use_gpu", f"{context}.device"), f"{context}.device.use_gpu")
    gpu_id = _as_gpu_id(
        _require(device, "gpu_id", f"{context}.device"),
        f"{context}.device.gpu_id",
    )
    if not gpu_id.isdigit():
        raise ValueError(
            f"{context}.device.gpu_id must be one non-negative integer."
        )
    dataloader = _as_mapping(_require(runtime, "dataloader", context), f"{context}.dataloader")
    workers = _as_int(
        _require(dataloader, "num_workers", f"{context}.dataloader"),
        f"{context}.dataloader.num_workers",
    )
    if workers != 0:
        raise ValueError(f"{context}.dataloader.num_workers must be 0 in Phase 2.")
    diagnostics = runtime.get("diagnostics")
    if diagnostics is not None:
        mapping = _as_mapping(diagnostics, f"{context}.diagnostics")
        unknown = set(mapping) - {"per_epoch_predictions"}
        if unknown:
            raise ValueError(
                f"Unknown {context}.diagnostics keys: " + ", ".join(sorted(unknown))
            )
        if "per_epoch_predictions" in mapping:
            _as_bool(mapping["per_epoch_predictions"], f"{context}.diagnostics.per_epoch_predictions")


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
    protocol = SRGNN_FIXED_LAST_PROTOCOL
    if "checkpoint_protocol" in normalized:
        protocol = _as_str(
            normalized["checkpoint_protocol"],
            f"{context}.checkpoint_protocol",
        ).strip().lower()
        if protocol not in ALLOWED_SRGNN_CHECKPOINT_PROTOCOLS:
            allowed = ", ".join(sorted(ALLOWED_SRGNN_CHECKPOINT_PROTOCOLS))
            raise ValueError(f"{context}.checkpoint_protocol must be one of: {allowed}.")
        normalized["checkpoint_protocol"] = protocol

    if protocol == SRGNN_VALIDATION_BEST_PROTOCOL and "best_metric" not in normalized:
        normalized["best_metric"] = SRGNN_VALIDATION_BEST_METRIC
    if "best_metric" in normalized:
        best_metric = _as_str(normalized["best_metric"], f"{context}.best_metric").strip().lower()
        if best_metric not in ALLOWED_SRGNN_BEST_METRICS:
            allowed = ", ".join(sorted(ALLOWED_SRGNN_BEST_METRICS))
            raise ValueError(f"{context}.best_metric must be one of: {allowed}.")
        normalized["best_metric"] = best_metric

    if protocol == SRGNN_VALIDATION_BEST_PROTOCOL and "patience_metric" not in normalized:
        normalized["patience_metric"] = SRGNN_VALIDATION_PATIENCE_METRIC
    if "patience_metric" in normalized:
        patience_metric = _as_str(
            normalized["patience_metric"],
            f"{context}.patience_metric",
        ).strip().lower()
        if patience_metric not in ALLOWED_SRGNN_PATIENCE_METRICS:
            allowed = ", ".join(sorted(ALLOWED_SRGNN_PATIENCE_METRICS))
            raise ValueError(f"{context}.patience_metric must be one of: {allowed}.")
        normalized["patience_metric"] = patience_metric

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
    _normalize_external_victim_train_protocol(
        normalized,
        context,
        default_export_model=VICTIM_EXPORT_BEST,
    )
    return normalized


def _normalize_tron_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    has_epochs = "epochs" in normalized
    has_max_epochs = "max_epochs" in normalized
    if not has_epochs and not has_max_epochs:
        raise ValueError(f"Missing required configuration: {context}.epochs")
    epochs = (
        _as_int(normalized["epochs"], f"{context}.epochs")
        if has_epochs
        else _as_int(normalized["max_epochs"], f"{context}.max_epochs")
    )
    if has_max_epochs:
        max_epochs = _as_int(normalized["max_epochs"], f"{context}.max_epochs")
        if max_epochs != epochs:
            raise ValueError(f"{context}.epochs and {context}.max_epochs must match.")
    if epochs <= 0:
        raise ValueError(f"{context}.epochs must be positive.")
    normalized["epochs"] = int(epochs)
    normalized["max_epochs"] = int(epochs)
    _normalize_external_victim_train_protocol(
        normalized,
        context,
        default_export_model=VICTIM_EXPORT_LAST,
    )
    return normalized


def _normalize_mdhg_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    for key in ("epochs", "batch_size", "lr"):
        _require(normalized, key, context)
    normalized["epochs"] = _as_int(normalized["epochs"], f"{context}.epochs")
    normalized["batch_size"] = _as_int(normalized["batch_size"], f"{context}.batch_size")
    normalized["lr"] = _as_float(normalized["lr"], f"{context}.lr")
    if normalized["epochs"] <= 0:
        raise ValueError(f"{context}.epochs must be positive.")
    if normalized["batch_size"] <= 0:
        raise ValueError(f"{context}.batch_size must be positive.")
    if normalized["lr"] <= 0:
        raise ValueError(f"{context}.lr must be positive.")
    normalized.setdefault("checkpoint_protocol", VICTIM_FIXED_EPOCH_PROTOCOL)
    normalized.setdefault("validation_enabled", False)
    normalized.setdefault("export_model", VICTIM_EXPORT_LAST)
    _normalize_external_victim_train_protocol(
        normalized,
        context,
        default_export_model=VICTIM_EXPORT_LAST,
    )
    if normalized["checkpoint_protocol"] != VICTIM_FIXED_EPOCH_PROTOCOL:
        raise ValueError(f"{context}.checkpoint_protocol must be fixed_epoch for MDHG.")
    if normalized["validation_enabled"]:
        raise ValueError(f"{context}.validation_enabled must be false for MDHG.")
    if normalized["export_model"] != VICTIM_EXPORT_LAST:
        raise ValueError(f"{context}.export_model must be last for MDHG.")
    return normalized


def _normalize_freqrec_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    required = (
        "model_type",
        "epochs",
        "batch_size",
        "lr",
        "max_seq_length",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "hidden_act",
        "attention_probs_dropout_prob",
        "hidden_dropout_prob",
        "initializer_range",
        "alpha",
        "gama",
        "alpha_loss",
        "fft_loss_type",
        "chux",
        "adam_beta1",
        "adam_beta2",
        "weight_decay",
        "patience",
        "fre",
        "fourier_loss",
        "checkpoint_protocol",
        "validation_metric",
        "metric_cutoffs",
    )
    for key in required:
        _require(normalized, key, context)
    int_fields = (
        "epochs",
        "batch_size",
        "max_seq_length",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "patience",
    )
    float_fields = (
        "lr",
        "attention_probs_dropout_prob",
        "hidden_dropout_prob",
        "initializer_range",
        "alpha",
        "gama",
        "alpha_loss",
        "adam_beta1",
        "adam_beta2",
        "weight_decay",
        "fre",
    )
    for key in int_fields:
        normalized[key] = _as_int(normalized[key], f"{context}.{key}")
    for key in float_fields:
        normalized[key] = _as_float(normalized[key], f"{context}.{key}")
    for key in ("model_type", "hidden_act", "fft_loss_type", "chux"):
        normalized[key] = _as_str(normalized[key], f"{context}.{key}")
    normalized["fourier_loss"] = _as_bool(
        normalized["fourier_loss"], f"{context}.fourier_loss"
    )
    normalized["checkpoint_protocol"] = _as_str(
        normalized["checkpoint_protocol"], f"{context}.checkpoint_protocol"
    ).strip().lower()
    normalized["validation_metric"] = _as_str(
        normalized["validation_metric"], f"{context}.validation_metric"
    ).strip().lower()
    normalized["metric_cutoffs"] = list(
        _unique_preserve_order(
            _as_int_list(normalized["metric_cutoffs"], f"{context}.metric_cutoffs")
        )
    )
    if normalized["model_type"].strip().lower() != "freqrec":
        raise ValueError(f"{context}.model_type must be 'freqrec'.")
    normalized["model_type"] = "freqrec"
    if normalized["checkpoint_protocol"] not in {
        VICTIM_FIXED_EPOCH_PROTOCOL,
        VICTIM_VALIDATION_BEST_PROTOCOL,
    }:
        raise ValueError(
            f"{context}.checkpoint_protocol must be fixed_epoch or validation_best."
        )
    if normalized["validation_metric"] not in {"hr@20", "mrr@20", "ndcg@20"}:
        raise ValueError(f"{context}.validation_metric must be hr@20, mrr@20, or ndcg@20.")
    if not normalized["metric_cutoffs"] or any(
        cutoff <= 0 for cutoff in normalized["metric_cutoffs"]
    ):
        raise ValueError(f"{context}.metric_cutoffs must contain positive integers.")
    normalized["metric_cutoffs"] = sorted(set(normalized["metric_cutoffs"]) | {20})
    if any(normalized[key] <= 0 for key in int_fields):
        raise ValueError(f"{context} integer training fields must be positive.")
    if normalized["lr"] <= 0:
        raise ValueError(f"{context}.lr must be positive.")
    for key in ("attention_probs_dropout_prob", "hidden_dropout_prob"):
        if not 0.0 <= normalized[key] < 1.0:
            raise ValueError(f"{context}.{key} must be in [0, 1).")
    if normalized["initializer_range"] <= 0:
        raise ValueError(f"{context}.initializer_range must be positive.")
    if not 0.0 < normalized["alpha"] < 1.0:
        raise ValueError(f"{context}.alpha must be in (0, 1).")
    if not 0.0 < normalized["gama"] < 1.0:
        raise ValueError(f"{context}.gama must be in (0, 1).")
    if not 0.0 < normalized["alpha_loss"] < 1.0:
        raise ValueError(f"{context}.alpha_loss must be in (0, 1).")
    if normalized["fft_loss_type"] not in {
        "l1",
        "l2",
        "SmoothL1Loss",
        "mix_loss",
    }:
        raise ValueError(
            f"{context}.fft_loss_type must be l1, l2, SmoothL1Loss, or mix_loss."
        )
    if normalized["chux"] not in {"p", "c"}:
        raise ValueError(f"{context}.chux must be 'p' or 'c'.")
    if normalized["hidden_act"] not in {
        "gelu",
        "relu",
        "swish",
        "tanh",
        "sigmoid",
    }:
        raise ValueError(
            f"{context}.hidden_act must be gelu, relu, swish, tanh, or sigmoid."
        )
    if not 0.0 < normalized["adam_beta1"] < 1.0:
        raise ValueError(f"{context}.adam_beta1 must be in (0, 1).")
    if not 0.0 < normalized["adam_beta2"] < 1.0:
        raise ValueError(f"{context}.adam_beta2 must be in (0, 1).")
    if normalized["weight_decay"] < 0:
        raise ValueError(f"{context}.weight_decay must be non-negative.")
    if normalized["fre"] != 1.0:
        raise ValueError(f"{context}.fre must be exactly 1.0.")
    if normalized["fourier_loss"] is not True:
        raise ValueError(f"{context}.fourier_loss must be true.")
    if normalized["hidden_size"] % normalized["num_attention_heads"] != 0:
        raise ValueError(
            f"{context}.hidden_size must be divisible by num_attention_heads."
        )
    return normalized


def _normalize_wearec_train(train: Mapping[str, Any], context: str) -> dict[str, Any]:
    normalized = _normalize_primitive(train, context)
    if not isinstance(normalized, dict):
        raise TypeError(f"Expected {context} to be a mapping.")
    required = (
        "epochs", "batch_size", "lr", "max_seq_length", "hidden_size",
        "num_hidden_layers", "hidden_act", "hidden_dropout_prob",
        "initializer_range", "num_heads", "alpha", "adam_beta1",
        "adam_beta2", "weight_decay", "checkpoint_protocol", "metric_cutoffs",
    )
    for key in required:
        _require(normalized, key, context)
    for key in (
        "epochs", "batch_size", "max_seq_length", "hidden_size",
        "num_hidden_layers", "num_heads",
    ):
        normalized[key] = _as_int(normalized[key], f"{context}.{key}")
        if normalized[key] <= 0:
            raise ValueError(f"{context}.{key} must be positive.")
    if normalized["max_seq_length"] % 2:
        raise ValueError(f"{context}.max_seq_length must be even.")
    if normalized["hidden_size"] % normalized["num_heads"]:
        raise ValueError(
            f"{context}.hidden_size must be divisible by num_heads."
        )
    for key in (
        "lr", "hidden_dropout_prob", "initializer_range", "alpha",
        "adam_beta1", "adam_beta2", "weight_decay",
    ):
        normalized[key] = _as_float(normalized[key], f"{context}.{key}")
    if normalized["lr"] <= 0 or normalized["initializer_range"] <= 0:
        raise ValueError(f"{context} lr and initializer_range must be positive.")
    if not 0 <= normalized["hidden_dropout_prob"] < 1:
        raise ValueError(f"{context}.hidden_dropout_prob must be in [0, 1).")
    if not 0 < normalized["alpha"] < 1:
        raise ValueError(f"{context}.alpha must be in (0, 1).")
    for key in ("adam_beta1", "adam_beta2"):
        if not 0 < normalized[key] < 1:
            raise ValueError(f"{context}.{key} must be in (0, 1).")
    if normalized["weight_decay"] < 0:
        raise ValueError(f"{context}.weight_decay must be non-negative.")
    normalized["hidden_act"] = _as_str(normalized["hidden_act"], f"{context}.hidden_act").strip().lower()
    normalized["checkpoint_protocol"] = _as_str(
        normalized["checkpoint_protocol"], f"{context}.checkpoint_protocol"
    ).strip().lower()
    if normalized["checkpoint_protocol"] != VICTIM_FIXED_EPOCH_PROTOCOL:
        raise ValueError(f"{context}.checkpoint_protocol must be fixed_epoch.")
    cutoffs = list(_as_int_list(normalized["metric_cutoffs"], f"{context}.metric_cutoffs"))
    if not cutoffs or any(value <= 0 for value in cutoffs):
        raise ValueError(f"{context}.metric_cutoffs must contain positive integers.")
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError(f"{context}.metric_cutoffs must not contain duplicates.")
    normalized["metric_cutoffs"] = sorted(cutoffs)
    return normalized


def _normalize_external_victim_train_protocol(
    normalized: dict[str, Any],
    context: str,
    *,
    default_export_model: str,
) -> None:
    protocol = _as_str(
        normalized.get("checkpoint_protocol", VICTIM_VALIDATION_BEST_PROTOCOL),
        f"{context}.checkpoint_protocol",
    ).strip().lower()
    if protocol not in _ALLOWED_EXTERNAL_VICTIM_CHECKPOINT_PROTOCOLS:
        allowed = ", ".join(sorted(_ALLOWED_EXTERNAL_VICTIM_CHECKPOINT_PROTOCOLS))
        raise ValueError(f"{context}.checkpoint_protocol must be one of: {allowed}.")
    normalized["checkpoint_protocol"] = protocol

    validation_enabled = _as_bool(
        normalized.get("validation_enabled", True),
        f"{context}.validation_enabled",
    )
    normalized["validation_enabled"] = bool(validation_enabled)

    export_model = _as_str(
        normalized.get("export_model", default_export_model),
        f"{context}.export_model",
    ).strip().lower()
    if export_model not in _ALLOWED_EXTERNAL_VICTIM_EXPORT_MODELS:
        allowed = ", ".join(sorted(_ALLOWED_EXTERNAL_VICTIM_EXPORT_MODELS))
        raise ValueError(f"{context}.export_model must be one of: {allowed}.")
    normalized["export_model"] = export_model

    if not validation_enabled and export_model == VICTIM_EXPORT_BEST:
        raise ValueError(
            f"{context}.validation_enabled=false is incompatible with export_model=best."
        )
    if protocol == VICTIM_FIXED_EPOCH_PROTOCOL and export_model != VICTIM_EXPORT_LAST:
        raise ValueError(
            f"{context}.checkpoint_protocol=fixed_epoch requires export_model=last."
        )
    if protocol == VICTIM_VALIDATION_BEST_PROTOCOL and not validation_enabled:
        raise ValueError(
            f"{context}.checkpoint_protocol=validation_best requires validation_enabled=true."
        )


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
    anchor_construction = _as_mapping(
        _require(normalized, "anchor_construction", "root"),
        "anchor_construction",
    )
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
            fake_session_source=(
                FakeSessionSourceConfig(
                    **dict(
                        _as_mapping(
                            attack["fake_session_source"],
                            "attack.fake_session_source",
                        )
                    )
                )
                if attack.get("fake_session_source") is not None
                else FakeSessionSourceConfig()
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
            carrier_selection=(
                CarrierSelectionConfig(
                    **dict(
                        _as_mapping(
                            attack["carrier_selection"],
                            "attack.carrier_selection",
                        )
                    )
                )
                if attack.get("carrier_selection") is not None
                else None
            ),
            pts_construction=(
                PTSConstructionConfig(
                    **dict(
                        _as_mapping(
                            attack["pts_construction"],
                            "attack.pts_construction",
                        )
                    )
                )
                if attack.get("pts_construction") is not None
                else None
            ),
            creat_additive_sbr=(
                CreatAdditiveSBRConfig(
                    **dict(
                        _as_mapping(
                            attack["creat_additive_sbr"],
                            "attack.creat_additive_sbr",
                        )
                    )
                )
                if attack.get("creat_additive_sbr") is not None
                else None
            ),
        ),
        anchor_construction=AnchorConstructionConfig(
            **dict(anchor_construction)
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
            cleanup_victim_intermediates=_as_bool(
                artifacts.get("cleanup_victim_intermediates", False),
                "artifacts.cleanup_victim_intermediates",
            ),
        ),
    )


def load_config(path: str | Path) -> Config:
    parsed = parse_config(path)
    normalized = normalize_config_mapping(parsed)
    return _build_config(normalized)


__all__ = [
    "CanonicalSplitConfig",
    "AnchorConstructionConfig",
    "ANCHOR_CONSTRUCTION_SOURCE_POPULAR_TRAIN_ITEMS",
    "ANCHOR_CONSTRUCTION_SOURCE_VULNERABLE_VALIDATION_LAST_ITEM",
    "ANCHOR_CONSTRUCTION_STRATEGY_ROUND_ROBIN",
    "CarrierSelectionConfig",
    "Config",
    "CreatAdditiveSBRConfig",
    "CREAT_ADDITIVE_SBR_ATTACK_REWARD_SCORE",
    "CREAT_ADDITIVE_SBR_CONSISTENCY_LOCAL_GLOBAL",
    "CREAT_ADDITIVE_SBR_DPP_BOUNDED_DETERMINANT",
    "CREAT_ADDITIVE_SBR_DPP_RAW_LOGDET",
    "CREAT_ADDITIVE_SBR_FINAL_POLICY_LAST",
    "CREAT_ADDITIVE_SBR_SEED_SOURCE_POSITION_OPT_SEED",
    "CREAT_ADDITIVE_SBR_VARIANT_V1",
    "CREAT_ADDITIVE_SBR_VARIANT_V2",
    "FakeSessionSourceConfig",
    "FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED",
    "FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED",
    "PositionOptConfig",
    "PTSArtifactsConfig",
    "PTSActionsConfig",
    "PTSActionsDynamicMasksConfig",
    "PTSCEMEpochRewardDiagnosticsRuntimeConfig",
    "PTSCEMInitRuntimeConfig",
    "PTSCEMRuntimeConfig",
    "PTSCEMResamplingRuntimeConfig",
    "PTSCEMSamplerRuntimeConfig",
    "PTSCEMSurrogateRetrainRuntimeConfig",
    "PTSCEMUpdateRuntimeConfig",
    "PTSContinuousParameterBoundsConfig",
    "PTSContinuousPolicyConfig",
    "PTSConstructionConfig",
    "PTSDirectActionPolicyConfig",
    "PTS_CONSTRUCTION_METHOD_CONTINUOUS_MLP_CEM",
    "PTS_CONSTRUCTION_METHOD_DIRECT_ACTION_MLP_CEM",
    "PTS_CONSTRUCTION_METHOD_GROUPED_CEM_V1",
    "PTS_CONTINUOUS_BETA_INITIALIZATION_BEHAVIOR_COVERING_V1",
    "PTS_CONTINUOUS_BETA_INPUT_SUFFIX_LENGTH_PERCENTILE",
    "PTS_CONTINUOUS_BETA_PARAMETERIZATION_LINEAR_LOG_BETA",
    "PTS_CONTINUOUS_BETA_PARAMETERIZATION_TINY_MLP_LOG_BETA_H2",
    "PTS_CONTINUOUS_BETA_SOURCE_POLICY_Q_AND_RHO_LOGISTIC",
    "PTS_CEM_INIT_UNIFORM",
    "PTS_CEM_INIT_TWO_POOL_BEHAVIOR_CURVE_SPACE_FILLING",
    "PTS_CEM_INIT_VERTEX_STRATIFIED_SPACE_FILLING",
    "PTS_CEM_SAMPLER_DIRICHLET",
    "PTS_CEM_SAMPLER_GAUSSIAN",
    "PTS_CEM_SEED_SOURCE_POSITION_OPT_SEED",
    "PTS_CEM_UPDATE_MODE_ELITE_CENTERED_GAUSSIAN",
    "PTS_CEM_UPDATE_MODE_ELITE_CENTERED_EMPIRICAL_GAUSSIAN",
    "PTS_CEM_SURROGATE_RETRAIN_FIXED_LAST",
    "PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST",
    "PTS_CEM_SURROGATE_REWARD_BEST",
    "PTS_CEM_SURROGATE_REWARD_LAST",
    "PTS_FINAL_SELECTION_GLOBAL_BEST_CANDIDATE",
    "PTS_GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX",
    "PTS_GROUPING_RESIDUAL_SUFFIX_LENGTH",
    "PTS_PREFIX_RANGE_INTERNAL",
    "PTS_PREFIX_SAMPLER_UNIFORM",
    "PTS_DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE",
    "PTS_DIRECT_ACTION_POLICY_PARAMETERIZATION_MLP_H2",
    "PTS_REWARD_RAW_LOWK_MRR_RECALL_10_20",
    "PTSFinalSelectionConfig",
    "PTSGenerationConfig",
    "PTSGroupingConfig",
    "PTSPrefixSelectorConfig",
    "PTSRewardConfig",
    "PTSSuffixLengthBucketConfig",
    "RankBucketCEMConfig",
    "RankBucketCEMSurrogateEvaluatorConfig",
    "COVERAGE_AWARE_LOCAL_POSITION_SCORER",
    "COVERAGE_CANDIDATE_REPRESENTATION_TARGETIZED_PREFIX_MEAN_EMBEDDING",
    "COVERAGE_PREFIX_REPRESENTATION_MEAN_ITEM_EMBEDDING",
    "COVERAGE_PREFIX_SOURCE_VALIDATION",
    "COVERAGE_RANK_WEIGHTING_INVERSE_LOG_RANK",
    "COVERAGE_RANK_WEIGHTING_NONE",
    "COVERAGE_SIMILARITY_COSINE",
    "RANK_BUCKET_CEM_FULL_RETRAIN_SURROGATE_EVALUATOR",
    "RANK_BUCKET_CEM_TAIL_BOOSTED_INIT_MODE",
    "RANK_BUCKET_CEM_WARM_START_SURROGATE_EVALUATOR",
    "RANK_BUCKET_CEM_ZERO_MEAN_INIT_MODE",
    "SurrogateEvalPoisonBalanceConfig",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_CANDIDATE_POSITIONS",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_OPERATION",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_PLACEMENT_MODE",
    "TARGET_AWARE_CARRIER_LOCAL_POSITION_SCORER",
    "TARGET_AWARE_CARRIER_SELECTION_LENGTH_BUCKETS_EXACT_UNTIL_4_PLUS",
    "TARGET_AWARE_CARRIER_SELECTION_NORMALIZE_MINMAX",
    "TARGET_AWARE_CARRIER_SELECTION_SCORER",
    "TrainTemplateFallbackConfig",
    "TrainTemplateSourceConfig",
    "TRAIN_TEMPLATE_LENGTH_MATCHING_EXACT_LARGEST_REMAINDER",
    "TRAIN_TEMPLATE_REFERENCE_SPLIT_TRAIN_SUB",
    "TRAIN_TEMPLATE_TARGET_FILTERING_NONE",
    "load_config",
    "normalize_config_mapping",
    "parse_config",
    "parse_config_mapping",
    "validate_config_mapping",
]
