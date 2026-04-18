from __future__ import annotations

import argparse
import pickle
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import Config, load_config
from attack.common.paths import target_dir
from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.prefix_nonzero_when_possible import (
    PrefixNonzeroWhenPossiblePolicy,
)
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.position_stats import save_position_stats
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts


def run_prefix_nonzero_when_possible(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Batch 6 expects data.poison_train_only to be true.")
    set_seed(config.seeds.fake_session_seed)
    shared = prepare_shared_attack_artifacts(
        config,
        run_type="prefix_nonzero_when_possible",
        require_poison_runner=True,
        config_path=config_path,
    )

    if shared.poison_runner is None:
        raise RuntimeError("Poison runner is required for prefix scoring.")

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = PrefixNonzeroWhenPossiblePolicy(
            shared.poison_runner, config.attack.replacement_topk_ratio
        )
        fake_sessions = []
        selected_positions: list[int] = []
        position_meta = []
        for session in shared.template_sessions:
            result = policy.apply_with_metadata(session, target_item)
            fake_sessions.append(result.session)
            selected_positions.append(int(result.position))
            position_meta.append(
                {"position": result.position, "target_score": result.target_score}
            )

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions, shared.clean_labels, fake_sessions
        )

        target_root = target_dir(
            config,
            target_item,
            run_type="prefix_nonzero_when_possible",
        )
        target_root.mkdir(parents=True, exist_ok=True)
        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=selected_positions,
            run_type="prefix_nonzero_when_possible",
            target_item=int(target_item),
        )
        positions_path = target_root / "prefix_nonzero_when_possible_metadata.pkl"
        with positions_path.open("wb") as handle:
            pickle.dump(position_meta, handle)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "position_stats_path": str(position_stats_path),
                "prefix_nonzero_when_possible_metadata_path": str(positions_path),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type="prefix_nonzero_when_possible",
        build_poisoned=build_poisoned,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="attack/configs/diginetica_attack_prefix_nonzero_when_possible.yaml",
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_prefix_nonzero_when_possible(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
