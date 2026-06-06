from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE, run_metadata_paths
from attack.creat.pipeline import generate_creat_target, prepare_creat_artifacts
from attack.pipeline.core.pipeline_utils import (
    ensure_target_registry_prefix,
    prepare_shared_attack_artifacts,
    requested_target_prefix,
)
from attack.pipeline.runs.run_creat_additive_sbr import _validate_creat_run_config


DEFAULT_CREAT_V2_GENERATE_ONLY_CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_create_copy_source_popular_sample.yaml"
)


def run_creat_additive_sbr_generate_only(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _validate_creat_run_config(config)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE,
        require_poison_runner=True,
        config_path=config_path,
    )
    prepared = prepare_creat_artifacts(shared)
    registry = ensure_target_registry_prefix(
        shared.stats,
        config,
        shared_paths=shared.shared_paths,
    )
    targets = requested_target_prefix(config, target_registry=registry)
    results: dict[str, object] = {}
    for target_item in targets:
        generated = generate_creat_target(
            config=config,
            shared=shared,
            prepared=prepared,
            target_item=int(target_item),
            run_type=CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE,
            save_poisoned_sessions=True,
        )
        results[str(int(target_item))] = generated.metadata
    summary = {
        "run_type": CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE,
        "target_items": [int(item) for item in targets],
        "victim_execution_skipped": True,
        "targets": results,
    }
    paths = run_metadata_paths(config, run_type=CREAT_ADDITIVE_SBR_GENERATE_ONLY_RUN_TYPE)
    paths["run_root"].mkdir(parents=True, exist_ok=True)
    save_json(summary, paths["run_root"] / "generate_only_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CREAT_V2_GENERATE_ONLY_CONFIG_PATH)
    args = parser.parse_args()
    run_creat_additive_sbr_generate_only(load_config(args.config), config_path=args.config)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CREAT_V2_GENERATE_ONLY_CONFIG_PATH",
    "run_creat_additive_sbr_generate_only",
]
