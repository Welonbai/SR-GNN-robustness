from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


KEEP_RESIDUAL_SUFFIX_ACTION_NAME = "keep_residual_suffix"
REGENERATE_RESIDUAL_SUFFIX_ACTION_NAME = "regenerate_residual_suffix"
CONSUME_ONE_KEEP_REST_ACTION_NAME = "consume_one_keep_rest"
CONSUME_ONE_GENERATE_ACTION_NAME = "consume_one_generate_continuation"
CONSUME_ALL_STOP_ACTION_NAME = "consume_all_stop"
GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX = "same_as_residual_suffix"
GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE = "residual_suffix_minus_one"


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
        name=KEEP_RESIDUAL_SUFFIX_ACTION_NAME,
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="zero",
            continuation_source="keep",
            generation_length_policy=None,
        ),
    ),
    PTSConstructionSpec(
        name=REGENERATE_RESIDUAL_SUFFIX_ACTION_NAME,
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="zero",
            continuation_source="generate",
            generation_length_policy=GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX,
        ),
    ),
    PTSConstructionSpec(
        name=CONSUME_ONE_KEEP_REST_ACTION_NAME,
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="one",
            continuation_source="keep",
            generation_length_policy=None,
        ),
    ),
    PTSConstructionSpec(
        name=CONSUME_ONE_GENERATE_ACTION_NAME,
        prefix_selector=_INTERNAL_UNIFORM_PREFIX_SELECTOR,
        suffix_constructor=SuffixConstructionSpec(
            consume_policy="one",
            continuation_source="generate",
            generation_length_policy=(
                GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE
            ),
        ),
    ),
    PTSConstructionSpec(
        name=CONSUME_ALL_STOP_ACTION_NAME,
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
    "CONSUME_ALL_STOP_ACTION_NAME",
    "CONSUME_ONE_GENERATE_ACTION_NAME",
    "CONSUME_ONE_KEEP_REST_ACTION_NAME",
    "DEFAULT_PTS_V1_SPECS",
    "GENERATION_LENGTH_POLICY_RESIDUAL_SUFFIX_MINUS_ONE",
    "GENERATION_LENGTH_POLICY_SAME_AS_RESIDUAL_SUFFIX",
    "KEEP_RESIDUAL_SUFFIX_ACTION_NAME",
    "PTSConstructionSpec",
    "PrefixSelectorSpec",
    "REGENERATE_RESIDUAL_SUFFIX_ACTION_NAME",
    "SuffixConstructionSpec",
    "get_default_pts_v1_specs",
    "lookup_spec_by_name",
]
