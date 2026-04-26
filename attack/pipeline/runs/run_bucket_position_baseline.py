from __future__ import annotations

import argparse
import random
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import load_target_registry, save_json
from attack.common.config import Config, load_config
from attack.common.paths import target_dir
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import (
    prepare_shared_attack_artifacts,
    requested_target_prefix,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.position_opt.bucket_diagnostics import (
    build_bucket_diagnostics,
    build_bucket_position_summary,
    write_selected_positions_jsonl,
)
from attack.position_opt.bucket_selector import (
    BUCKET_METHODS,
    select_bucket_session_position,
    validate_bucket_method,
)
from attack.position_opt.poison_builder import replace_item_at_position


DEFAULT_BUCKET_CONFIG_PATH = (
    "attack/configs/diginetica_attack_bucket_position_baselines_ratio1.yaml"
)
# This strict cohort guard is intentionally pilot-specific for the current
# 3-target popular bucket sweep. For later larger popular/unpopular sweeps, this
# should become config-driven instead of hardcoded here.
EXPECTED_BUCKET_TARGET_COHORT_KEY = "target_cohort_8be070ab82"
EXPECTED_BUCKET_TARGET_PREFIX = (11103, 39588, 5334)


def run_bucket_position_baseline(
    config: Config,
    *,
    bucket_method: str,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    validated_method = validate_bucket_method(bucket_method)
    _validate_bucket_run_config(config)

    shared = prepare_shared_attack_artifacts(
        config,
        run_type=validated_method,
        require_poison_runner=False,
        config_path=config_path,
    )
    target_registry, resolved_target_prefix, cohort_validation = _validate_bucket_target_cohort(
        config,
        target_registry=load_target_registry(shared.shared_paths["target_registry"]),
    )
    print(
        "[bucket-baseline] "
        f"method={validated_method} "
        f"shared_fake_sessions={shared.shared_paths['fake_sessions']} "
        f"target_cohort_key={target_registry['target_cohort_key']} "
        f"resolved_target_prefix={resolved_target_prefix}"
    )

    context = RunContext.from_shared(shared)
    nonzero_action_when_possible = bool(config.attack.position_opt.nonzero_action_when_possible)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        rng = random.Random(config.seeds.position_opt_seed)
        records = [
            select_bucket_session_position(
                method_name=validated_method,
                fake_session_index=session_index,
                session=session,
                target_item=int(target_item),
                replacement_topk_ratio=config.attack.replacement_topk_ratio,
                nonzero_action_when_possible=nonzero_action_when_possible,
                rng=rng,
            )
            for session_index, session in enumerate(shared.template_sessions)
        ]
        fake_sessions = [
            replace_item_at_position(
                session,
                record.selected_position,
                int(target_item),
            )
            for session, record in zip(shared.template_sessions, records)
        ]

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=validated_method,
        )
        target_root.mkdir(parents=True, exist_ok=True)
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=[int(record.selected_position) for record in records],
            run_type=validated_method,
            target_item=int(target_item),
        )
        selected_positions_path = write_selected_positions_jsonl(
            target_root / "selected_positions.jsonl",
            records,
        )
        position_summary = build_bucket_position_summary(
            records,
            method_name=validated_method,
            target_item=int(target_item),
            seed=int(config.seeds.position_opt_seed),
            seed_source="position_opt_seed",
            replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
            nonzero_action_when_possible=nonzero_action_when_possible,
        )
        position_summary_path = target_root / "position_summary.json"
        save_json(position_summary, position_summary_path)
        bucket_diagnostics = build_bucket_diagnostics(
            records,
            method_name=validated_method,
            target_item=int(target_item),
            seed=int(config.seeds.position_opt_seed),
            seed_source="position_opt_seed",
            replacement_topk_ratio=float(config.attack.replacement_topk_ratio),
            nonzero_action_when_possible=nonzero_action_when_possible,
            shared_fake_sessions_path=str(shared.shared_paths["fake_sessions"]),
            target_cohort_key=str(target_registry["target_cohort_key"]),
            resolved_target_prefix=resolved_target_prefix,
            cohort_validation=cohort_validation,
        )
        bucket_diagnostics_path = target_root / "bucket_diagnostics.json"
        save_json(bucket_diagnostics, bucket_diagnostics_path)

        metadata = {
            "bucket_method": validated_method,
            "position_selection_seed": int(config.seeds.position_opt_seed),
            "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
            "nonzero_action_when_possible": nonzero_action_when_possible,
            "position_stats_path": str(position_stats_path),
            "selected_positions_path": str(selected_positions_path),
            "position_summary_path": str(position_summary_path),
            "bucket_diagnostics_path": str(bucket_diagnostics_path),
            "shared_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
            "bucket_target_cohort_key": str(target_registry["target_cohort_key"]),
            "bucket_resolved_target_prefix": [int(item) for item in resolved_target_prefix],
            "bucket_cohort_validation": cohort_validation,
        }
        return TargetPoisonOutput(poisoned=poisoned, metadata=metadata)

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=validated_method,
        build_poisoned=build_poisoned,
    )


def _validate_bucket_run_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("Bucket baseline run expects data.poison_train_only to be true.")
    if float(config.attack.replacement_topk_ratio) != 1.0:
        raise ValueError("Bucket baseline run requires attack.replacement_topk_ratio == 1.0.")
    if int(config.data.canonical_split.min_session_len) < 2:
        raise ValueError(
            "Bucket baseline run requires data.canonical_split.min_session_len >= 2."
        )
    if config.attack.position_opt is None:
        raise ValueError(
            "Bucket baseline run requires attack.position_opt.nonzero_action_when_possible."
        )
    if not bool(config.attack.position_opt.nonzero_action_when_possible):
        raise ValueError(
            "Bucket baseline run requires attack.position_opt.nonzero_action_when_possible == true."
        )


def _validate_bucket_target_cohort(
    config: Config,
    *,
    target_registry: dict[str, object] | None,
) -> tuple[dict[str, object], list[int], dict[str, object]]:
    if target_registry is None:
        raise RuntimeError("Bucket baseline run requires a resolved target_registry.json artifact.")
    resolved_prefix = requested_target_prefix(
        config,
        target_registry=target_registry,
    )
    actual_key = str(target_registry.get("target_cohort_key"))
    validation = {
        "expected_target_cohort_key": EXPECTED_BUCKET_TARGET_COHORT_KEY,
        "actual_target_cohort_key": actual_key,
        "expected_target_prefix": [int(item) for item in EXPECTED_BUCKET_TARGET_PREFIX],
        "actual_target_prefix": [int(item) for item in resolved_prefix],
        "passed": False,
    }
    if actual_key != EXPECTED_BUCKET_TARGET_COHORT_KEY:
        raise RuntimeError(
            "Bucket baseline run resolved an unexpected target cohort key: "
            f"{actual_key}. Expected {EXPECTED_BUCKET_TARGET_COHORT_KEY}."
        )
    if list(resolved_prefix) != list(EXPECTED_BUCKET_TARGET_PREFIX):
        raise RuntimeError(
            "Bucket baseline run resolved an unexpected target prefix: "
            f"{resolved_prefix}. Expected {list(EXPECTED_BUCKET_TARGET_PREFIX)}."
        )
    validation["passed"] = True
    return target_registry, [int(item) for item in resolved_prefix], validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_BUCKET_CONFIG_PATH,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--bucket-method",
        required=True,
        choices=sorted(BUCKET_METHODS),
        help="Bucket baseline method key.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_bucket_position_baseline(
        config,
        bucket_method=args.bucket_method,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
