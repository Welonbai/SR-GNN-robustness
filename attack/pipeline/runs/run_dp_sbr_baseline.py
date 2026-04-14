from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import Config, load_config
from attack.common.paths import target_dir
from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.dpsbr_baseline import DPSBRBaselinePolicy
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts


def run_dp_sbr_baseline(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Batch 6 expects data.poison_train_only to be true.")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type="dpsbr_baseline",
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        rng = random.Random(config.seeds.fake_session_seed)
        policy = DPSBRBaselinePolicy(config.attack.replacement_topk_ratio, rng=rng)
        fake_sessions = []
        position_counts: Counter[int] = Counter()
        for session in shared.template_sessions:
            result = policy.apply_with_metadata(session, target_item)
            fake_sessions.append(result.session)
            position_counts[int(result.position)] += 1

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions, shared.clean_labels, fake_sessions
        )

        total_positions = sum(position_counts.values())
        ratios = {
            str(pos): (count / total_positions if total_positions else 0.0)
            for pos, count in position_counts.items()
        }
        positions_payload = {
            "total": int(total_positions),
            "counts": {str(pos): int(count) for pos, count in position_counts.items()},
            "ratios": ratios,
        }
        target_root = target_dir(config, target_item, run_type="dpsbr_baseline")
        target_root.mkdir(parents=True, exist_ok=True)
        positions_path = target_root / "dpsbr_position_metadata.json"
        with positions_path.open("w", encoding="utf-8") as handle:
            json.dump(positions_payload, handle, indent=2, sort_keys=True)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "dpsbr_position_metadata_path": str(positions_path),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type="dpsbr_baseline",
        build_poisoned=build_poisoned,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/diginetica_attack_dpsbr.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dp_sbr_baseline(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
