from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import Config, load_config
from attack.common.paths import run_config_dir, shared_artifact_paths, target_dir
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.pipeline.core.pipeline_utils import build_clean_pairs


def run_clean(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, float]:
    canonical_dataset = ensure_canonical_dataset(config)
    stats = compute_session_stats(canonical_dataset.train_sub)
    shared_paths = shared_artifact_paths(config, run_type="clean")
    clean_sessions, clean_labels = build_clean_pairs(canonical_dataset)
    clean_poisoned = build_poisoned_dataset(clean_sessions, clean_labels, [])

    export_paths: dict[str, Path] | None = None
    if "srgnn" in config.victims.enabled:
        export_root = run_config_dir(config, run_type="clean") / "export" / "srgnn_clean"
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
        target_root = target_dir(config, target_item, run_type="clean")
        target_root.mkdir(parents=True, exist_ok=True)
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=[],
            positions=[],
            run_type="clean",
            target_item=int(target_item),
            note="Clean run injects no fake sessions, so no replacement positions are used.",
        )
        return TargetPoisonOutput(
            poisoned=clean_poisoned,
            metadata={"position_stats_path": str(position_stats_path)},
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type="clean",
        build_poisoned=build_poisoned,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/diginetica_clean.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_clean(config, config_path=args.config)


if __name__ == "__main__":
    main()
