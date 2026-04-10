from __future__ import annotations

import argparse
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import run_config_dir, shared_artifact_paths
from attack.common.seed import set_seed
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import build_clean_pairs


def run_clean(
    config: Config,
    config_path: str | Path | None = None,
    epochs: int = 1,
) -> dict[str, float]:
    set_seed(config.seeds.fake_session_seed)

    canonical_dataset = ensure_canonical_dataset(config)
    stats = compute_session_stats(canonical_dataset.train_sub)
    shared_paths = shared_artifact_paths(config)
    clean_sessions, clean_labels = build_clean_pairs(canonical_dataset)

    export_paths: dict[str, Path] | None = None
    if "srgnn" in config.victims.enabled:
        export_root = run_config_dir(config) / "export" / "srgnn_clean"
        export_result = SRGNNExporter().export(canonical_dataset, export_root)
        export_paths = export_result.files

    context = RunContext(
        canonical_dataset=canonical_dataset,
        stats=stats,
        clean_sessions=clean_sessions,
        clean_labels=clean_labels,
        export_paths=export_paths,
        shared_paths=shared_paths,
        fake_session_count=0,
    )

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        poisoned = build_poisoned_dataset(clean_sessions, clean_labels, [])
        return TargetPoisonOutput(poisoned=poisoned, metadata={})

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type="clean",
        poison_epochs=0,
        attack_epochs=epochs,
        build_poisoned=build_poisoned,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/dp_sbr_diginetica_clean.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_clean(config, config_path=args.config, epochs=args.epochs)


if __name__ == "__main__":
    main()
