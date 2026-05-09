from __future__ import annotations

import random
from typing import Sequence

from attack.pts.specs import PrefixSelectorSpec


def select_internal_uniform_anchor(
    session_length: int,
    *,
    rng: random.Random,
) -> int:
    length = int(session_length)
    if length < 2:
        raise ValueError(
            "Internal uniform anchor selection requires session length >= 2; "
            "valid anchors are in [1, L - 1]."
        )
    return int(rng.randint(1, length - 1))


def select_anchor_position(
    session: Sequence[int],
    *,
    spec: PrefixSelectorSpec,
    rng: random.Random,
) -> int:
    _validate_phase1_prefix_selector(spec)
    return select_internal_uniform_anchor(len(session), rng=rng)


def _validate_phase1_prefix_selector(spec: PrefixSelectorSpec) -> None:
    if spec.range_name != "internal" or spec.sampler_name != "uniform":
        raise ValueError(
            "Phase 1 PTS supports only internal/uniform prefix selection; "
            f"received range_name={spec.range_name!r}, "
            f"sampler_name={spec.sampler_name!r}."
        )


__all__ = [
    "select_anchor_position",
    "select_internal_uniform_anchor",
]
