from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import load_fake_sessions
from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
)
from attack.pts.direct_action_diagnostic import (
    DirectActionInitDiagnosticResult,
    ELITE_SELECT_MODES,
    ELITE_SELECT_MODE_DIVERSE,
    run_direct_action_init_diagnostic,
)
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
    DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M,
    DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
    DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
    DIRECT_ACTION_POLICY_LINEAR_LENGTH,
    DIRECT_ACTION_POLICY_MLP_H2,
)


def parse_initial_stds(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("initial std list must not be empty.")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "initial std list must contain numeric values."
        ) from exc
    if any(item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("initial std values must be non-negative.")
    return values


def run_from_config_path(
    *,
    config_path: str | Path,
    policy_variant: str,
    length_feature_mode: str,
    initial_stds: Sequence[float],
    num_candidates: int,
    sample_sessions: int,
    output_dir: str | Path | None,
    seed: int | None,
    prefix_seed_scope: str,
    include_elite_centered_diagnostic: bool = False,
    elite_select_mode: str = ELITE_SELECT_MODE_DIVERSE,
    elite_count: int = 4,
    elite_resample_count: int = 8,
    elite_min_std: float = 0.25,
    elite_std_scale: float = 1.0,
    elite_centered_seed: int | None = None,
) -> DirectActionInitDiagnosticResult:
    config = load_config(config_path)
    shared_paths = shared_artifact_paths(
        config,
        run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    )
    fake_sessions_path = shared_paths["fake_sessions"]
    fake_sessions = load_fake_sessions(fake_sessions_path)
    if fake_sessions is None:
        raise FileNotFoundError(
            "Direct-action init diagnostic requires existing shared fake sessions "
            f"and will not generate them: {fake_sessions_path}"
        )
    return run_direct_action_init_diagnostic(
        config=config,
        config_path=config_path,
        fake_sessions=fake_sessions,
        fake_sessions_path=fake_sessions_path,
        policy_variant=policy_variant,
        length_feature_mode=length_feature_mode,
        initial_stds=initial_stds,
        num_candidates=int(num_candidates),
        sample_sessions=int(sample_sessions),
        output_dir=output_dir,
        seed=seed,
        prefix_seed_scope=prefix_seed_scope,
        include_elite_centered_diagnostic=bool(include_elite_centered_diagnostic),
        elite_select_mode=str(elite_select_mode),
        elite_count=int(elite_count),
        elite_resample_count=int(elite_resample_count),
        elite_min_std=float(elite_min_std),
        elite_std_scale=float(elite_std_scale),
        elite_centered_seed=elite_centered_seed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect direct-action categorical policy initialization.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--policy",
        default=DIRECT_ACTION_POLICY_MLP_H2,
        choices=[
            DIRECT_ACTION_POLICY_LINEAR_LENGTH,
            DIRECT_ACTION_POLICY_MLP_H2,
        ],
    )
    parser.add_argument(
        "--length-feature",
        default=DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
        choices=[
            DIRECT_ACTION_LENGTH_FEATURE_LOG1P,
            DIRECT_ACTION_LENGTH_FEATURE_M_OVER_MAX_M,
            DIRECT_ACTION_LENGTH_FEATURE_RAW_M,
            DIRECT_ACTION_LENGTH_FEATURE_Z_SCORE_M,
        ],
    )
    parser.add_argument("--initial-stds", type=parse_initial_stds, default=[0.5, 1.0, 1.5])
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--sample-sessions", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include-elite-centered-diagnostic", action="store_true")
    parser.add_argument(
        "--elite-select-mode",
        default=ELITE_SELECT_MODE_DIVERSE,
        choices=list(ELITE_SELECT_MODES),
    )
    parser.add_argument("--elite-count", type=int, default=4)
    parser.add_argument("--elite-resample-count", type=int, default=8)
    parser.add_argument("--elite-min-std", type=float, default=0.25)
    parser.add_argument("--elite-std-scale", type=float, default=1.0)
    parser.add_argument("--elite-centered-seed", type=int, default=None)
    parser.add_argument(
        "--prefix-seed-scope",
        default="target_independent",
        choices=["target_independent"],
    )
    args = parser.parse_args(argv)

    result = run_from_config_path(
        config_path=args.config,
        policy_variant=str(args.policy),
        length_feature_mode=str(args.length_feature),
        initial_stds=args.initial_stds,
        num_candidates=int(args.num_candidates),
        sample_sessions=int(args.sample_sessions),
        output_dir=args.output_dir,
        seed=args.seed,
        prefix_seed_scope=str(args.prefix_seed_scope),
        include_elite_centered_diagnostic=bool(args.include_elite_centered_diagnostic),
        elite_select_mode=str(args.elite_select_mode),
        elite_count=int(args.elite_count),
        elite_resample_count=int(args.elite_resample_count),
        elite_min_std=float(args.elite_min_std),
        elite_std_scale=float(args.elite_std_scale),
        elite_centered_seed=args.elite_centered_seed,
    )
    print(f"[direct-action-init-diagnostic] output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "main",
    "parse_initial_stds",
    "run_from_config_path",
]
