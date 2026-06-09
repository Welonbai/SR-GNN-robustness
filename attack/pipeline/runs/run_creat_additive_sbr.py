from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import (
    Config,
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    load_config,
)
from attack.common.paths import CREAT_ADDITIVE_SBR_RUN_TYPE
from attack.creat.pipeline import generate_creat_target, prepare_creat_artifacts
from attack.creat.srgnn_adapter import SRGNNRepresentationAdapter
from attack.creat.trainer import CreatAdditiveSBRTrainer
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts


DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
)


def run_creat_additive_sbr(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_creat_run_config(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    prepared = prepare_creat_artifacts(shared, adapter_class=SRGNNRepresentationAdapter)
    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        generated = generate_creat_target(
            config=config,
            shared=shared,
            prepared=prepared,
            target_item=int(target_item),
            run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
            trainer_class=CreatAdditiveSBRTrainer,
        )
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            generated.poisoned_sessions,
        )
        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=generated.poisoned_sessions,
            metadata=generated.metadata,
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=CREAT_ADDITIVE_SBR_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def _validate_creat_run_config(config: Config) -> None:
    if not config.data.poison_train_only:
        raise ValueError("CREAT-Additive-SBR requires data.poison_train_only == true.")
    if (
        config.attack.fake_session_source.type
        != FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    ):
        raise ValueError(
            "CREAT-Additive-SBR requires "
            "attack.fake_session_source.type == 'train_template_clean_exact_length_matched'."
        )
    if config.attack.creat_additive_sbr is None:
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.")
    if not bool(config.attack.creat_additive_sbr.enabled):
        raise ValueError("CREAT-Additive-SBR requires attack.creat_additive_sbr.enabled == true.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_creat_additive_sbr(config, config_path=args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CREAT_ADDITIVE_SBR_CONFIG_PATH",
    "run_creat_additive_sbr",
]
