from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import Config, load_config
from attack.common.paths import POISONING_SSL_SBR_RUN_TYPE
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_lightweight_attack_artifacts
from attack.poisoning_ssl.generator import CandidateGenerator
from attack.poisoning_ssl.pipeline import generate_poisoning_ssl_sbr_target


DEFAULT_POISONING_SSL_SBR_CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1.yaml"
)


def run_poisoning_ssl_sbr(
    config: Config,
    config_path: str | Path | None = None,
    *,
    candidate_generator: CandidateGenerator | None = None,
) -> dict[str, object]:
    _validate_poisoning_ssl_sbr_run_config(config)
    shared = prepare_lightweight_attack_artifacts(
        config,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        generated = generate_poisoning_ssl_sbr_target(
            config=config,
            shared=shared,
            target_item=int(target_item),
            run_type=POISONING_SSL_SBR_RUN_TYPE,
            n_fake_requested=int(context.fake_session_count),
            candidate_generator=candidate_generator,
        )
        if len(generated.raw_fake_sessions) != int(context.fake_session_count):
            raise RuntimeError(
                "SeqPoison-SBR final injected fake-session count does not match "
                f"the shared budget: {len(generated.raw_fake_sessions)} != "
                f"{int(context.fake_session_count)}."
            )
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            generated.raw_fake_sessions,
        )
        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=generated.raw_fake_sessions,
            metadata=generated.metadata,
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def _validate_poisoning_ssl_sbr_run_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("SeqPoison-SBR requires data.poison_train_only == true.")
    poisoning_config = config.attack.poisoning_ssl_sbr
    if poisoning_config is None:
        raise ValueError("SeqPoison-SBR requires attack.poisoning_ssl_sbr.")
    if not bool(poisoning_config.enabled):
        raise ValueError("SeqPoison-SBR requires attack.poisoning_ssl_sbr.enabled == true.")
    if bool(poisoning_config.enforce_nonzero_target_position):
        raise NotImplementedError(
            "SeqPoison-SBR Phase 2 does not implement "
            "enforce_nonzero_target_position=true."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_POISONING_SSL_SBR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_poisoning_ssl_sbr(config, config_path=args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_POISONING_SSL_SBR_CONFIG_PATH",
    "run_poisoning_ssl_sbr",
]
