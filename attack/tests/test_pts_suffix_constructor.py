from __future__ import annotations

from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.specs import get_default_pts_v1_specs, lookup_spec_by_name
from attack.pts.suffix_constructor import apply_suffix_construction


def _spec(name: str):
    return lookup_spec_by_name(get_default_pts_v1_specs(), name)


def test_keep_residual_suffix_construction() -> None:
    result = apply_suffix_construction(
        prefix=[1, 2],
        target_item=99,
        residual_suffix=[3, 4],
        spec=_spec("keep_residual_suffix"),
    )

    assert result.final_session == [1, 2, 99, 3, 4]
    assert result.generated_suffix == []


def test_consume_one_keep_rest_construction() -> None:
    result = apply_suffix_construction(
        prefix=[1, 2],
        target_item=99,
        residual_suffix=[3, 4],
        spec=_spec("consume_one_keep_rest"),
    )

    assert result.final_session == [1, 2, 99, 4]


def test_consume_all_stop_construction() -> None:
    result = apply_suffix_construction(
        prefix=[1, 2],
        target_item=99,
        residual_suffix=[3, 4],
        spec=_spec("consume_all_stop"),
    )

    assert result.final_session == [1, 2, 99]


def test_regenerate_residual_suffix_uses_monkeypatched_generator(monkeypatch) -> None:
    calls = []

    def fake_generate_poison_model_suffix(*, runner, prefix, suffix_length, topk, rng):
        calls.append(
            {
                "runner": runner,
                "prefix": list(prefix),
                "suffix_length": suffix_length,
                "topk": topk,
                "rng": rng,
            }
        )
        return [7, 8]

    monkeypatch.setattr(
        "attack.pts.suffix_constructor.generate_poison_model_suffix",
        fake_generate_poison_model_suffix,
    )
    poison_runner = object()

    result = apply_suffix_construction(
        prefix=[1, 2],
        target_item=99,
        residual_suffix=[3, 4],
        spec=_spec("regenerate_residual_suffix"),
        poison_runner=poison_runner,
        generation_topk=5,
        generation_rng_base_seed=123,
    )

    assert result.final_session == [1, 2, 99, 7, 8]
    assert result.generated_suffix == [7, 8]
    assert len(result.generated_suffix) == 2
    assert calls[0]["runner"] is poison_runner
    assert calls[0]["prefix"] == [1, 2, 99]
    assert calls[0]["suffix_length"] == 2
    assert calls[0]["topk"] == 5


def test_regenerate_residual_suffix_requires_poison_runner() -> None:
    with pytest.raises(ValueError, match="poison_runner is required"):
        apply_suffix_construction(
            prefix=[1, 2],
            target_item=99,
            residual_suffix=[3, 4],
            spec=_spec("regenerate_residual_suffix"),
            poison_runner=None,
        )
