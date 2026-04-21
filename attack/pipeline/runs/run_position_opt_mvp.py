from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.config import Config
from attack.position_opt import PositionOptConfig

from attack.pipeline.runs.run_position_opt_shared_policy import (
    DEFAULT_SHARED_POLICY_CONFIG_PATH,
    main as _shared_policy_main,
    run_position_opt_shared_policy,
)


def run_position_opt_mvp(
    config: Config,
    *,
    clean_surrogate_checkpoint_path: str | Path | None = None,
    config_path: str | Path | None = None,
    position_opt_config: PositionOptConfig | Mapping[str, Any] | None = None,
) -> dict[str, object]:
    return run_position_opt_shared_policy(
        config,
        clean_surrogate_checkpoint_path=clean_surrogate_checkpoint_path,
        config_path=config_path,
        position_opt_config=position_opt_config,
    )


def main() -> None:
    _shared_policy_main()


__all__ = [
    "DEFAULT_SHARED_POLICY_CONFIG_PATH",
    "run_position_opt_mvp",
]


if __name__ == "__main__":
    main()
