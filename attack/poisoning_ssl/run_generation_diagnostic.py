from __future__ import annotations

import argparse
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import load_config
from attack.common.paths import POISONING_SSL_SBR_RUN_TYPE
from attack.pipeline.core.pipeline_utils import (
    prepare_lightweight_attack_artifacts,
    resolve_target_items,
)
from attack.poisoning_ssl.pipeline import generate_poisoning_ssl_sbr_target


def run_generation_diagnostic(
    *,
    config_path: str | Path,
    target_item: int | None = None,
) -> dict[str, object]:
    config = load_config(config_path)
    shared = prepare_lightweight_attack_artifacts(
        config,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        config_path=config_path,
    )
    targets = resolve_target_items(
        shared.stats,
        config,
        shared_paths=shared.shared_paths,
    )
    target = int(target_item) if target_item is not None else int(targets[0])
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=target,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=int(shared.fake_session_count),
        config_path=config_path,
    )
    return {
        "target_item": target,
        "n_final_injected": int(len(result.raw_fake_sessions)),
        "metadata": result.metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SeqPoison-SBR generation only; victim training is not invoked."
    )
    parser.add_argument("--config", required=True, help="Path to poisoning_ssl_sbr YAML.")
    parser.add_argument("--target", type=int, default=None, help="Optional target item.")
    args = parser.parse_args()
    summary = run_generation_diagnostic(
        config_path=args.config,
        target_item=args.target,
    )
    metadata = summary["metadata"]
    print(
        "SeqPoison-SBR generation diagnostic completed: "
        f"target={summary['target_item']} "
        f"n_final_injected={summary['n_final_injected']} "
        f"n_generated_candidates={metadata.get('n_generated_candidates')} "
        f"acceptance_rate={metadata.get('acceptance_rate')}"
    )


if __name__ == "__main__":
    main()


__all__ = ["run_generation_diagnostic"]
