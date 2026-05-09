from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrefixSelectorSpec:
    range_name: str
    sampler_name: str


@dataclass(frozen=True)
class SuffixConstructionSpec:
    consume_policy: str
    continuation_source: str
    generation_length_policy: str | None = None


@dataclass(frozen=True)
class PTSConstructionSpec:
    name: str
    prefix_selector: PrefixSelectorSpec
    suffix_constructor: SuffixConstructionSpec


_INTERNAL_UNIFORM_PREFIX_SELECTOR = PrefixSelectorSpec(
    range_name="internal",
    sampler_name="uniform",
)

DEFAULT_PTS_V1_SPECS: tuple[PTSConstructionSpec, ...] = (
    PTSConstructionSpec(
        name="keep_residual_suffix",
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="zero",
            continuation_source="keep",
            generation_length_policy=None,
        ),
    ),
    PTSConstructionSpec(
        name="regenerate_residual_suffix",
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="zero",
            continuation_source="generate",
            generation_length_policy="same_as_residual_suffix",
        ),
    ),
    PTSConstructionSpec(
        name="consume_one_keep_rest",
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="one",
            continuation_source="keep",
            generation_length_policy=None,
        ),
    ),
    PTSConstructionSpec(
        name="consume_all_stop",
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="all",
            continuation_source="keep",
            generation_length_policy=None,
        ),
    ),
)


def get_default_pts_v1_specs() -> tuple[PTSConstructionSpec, ...]:
    return tuple(DEFAULT_PTS_V1_SPECS)


def lookup_spec_by_name(
    specs: Sequence[PTSConstructionSpec],
    name: str,
) -> PTSConstructionSpec:
    matches = [spec for spec in specs if spec.name == str(name)]
    if not matches:
        raise ValueError(f"Unknown PTS construction spec {name!r}.")
    if len(matches) > 1:
        raise ValueError(f"PTS construction spec {name!r} is defined multiple times.")
    return matches[0]


__all__ = [
    "DEFAULT_PTS_V1_SPECS",
    "PTSConstructionSpec",
    "PrefixSelectorSpec",
    "SuffixConstructionSpec",
    "get_default_pts_v1_specs",
    "lookup_spec_by_name",
]
