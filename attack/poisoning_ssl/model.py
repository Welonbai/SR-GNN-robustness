from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeqPoisonModelSpec:
    component: str
    upstream_source: str
    status: str = "phase1_interface_only"


GENERATOR_MODEL_SPEC = SeqPoisonModelSpec(
    component="generator",
    upstream_source="Seq-poison/generator.py",
)
DISCRIMINATOR_MODEL_SPEC = SeqPoisonModelSpec(
    component="discriminator",
    upstream_source="Seq-poison/discriminator.py",
)
CLASSIFIER_MODEL_SPEC = SeqPoisonModelSpec(
    component="classifier",
    upstream_source="Seq-poison/classify.py",
)


__all__ = [
    "CLASSIFIER_MODEL_SPEC",
    "DISCRIMINATOR_MODEL_SPEC",
    "GENERATOR_MODEL_SPEC",
    "SeqPoisonModelSpec",
]
