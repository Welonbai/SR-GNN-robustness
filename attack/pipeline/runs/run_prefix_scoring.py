from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from attack.common.config import Config, load_config
from attack.common.paths import target_dir
from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.prefix_scoring import BestPositionPrefixPolicy
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts


def run_prefix_scoring(
    config: Config,
    config_path: str | Path | None = None,
    poison_epochs: int = 1,
    attack_epochs: int = 1,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Batch 6 expects data.poison_train_only to be true.")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        poison_epochs=poison_epochs,
        require_poison_runner=True,
        config_path=config_path,
    )

    if shared.poison_runner is None:
        raise RuntimeError("Poison runner is required for position-based scoring.")

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = BestPositionPrefixPolicy(
            shared.poison_runner, config.attack.replacement_topk_ratio
        )
        fake_sessions = []
        position_meta = []
        for session in shared.template_sessions:
            result = policy.apply_with_metadata(session, target_item)
            fake_sessions.append(result.session)
            position_meta.append(
                {"position": result.position, "target_score": result.target_score}
            )

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions, shared.clean_labels, fake_sessions
        )

        target_root = target_dir(config, target_item)
        target_root.mkdir(parents=True, exist_ok=True)
        positions_path = target_root / "best_position_metadata.pkl"
        with positions_path.open("wb") as handle:
            pickle.dump(position_meta, handle)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "best_position_metadata_path": str(positions_path),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type="position_prefix",
        poison_epochs=poison_epochs,
        attack_epochs=attack_epochs,
        build_poisoned=build_poisoned,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/diginetica_attack_dpsbr.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument("--poison-epochs", type=int, default=1, help="Poison model epochs.")
    parser.add_argument("--attack-epochs", type=int, default=1, help="Attack model epochs.")
    args = parser.parse_args()

    config = load_config(args.config)
    run_prefix_scoring(
        config,
        config_path=args.config,
        poison_epochs=args.poison_epochs,
        attack_epochs=args.attack_epochs,
    )


if __name__ == "__main__":
    main()
