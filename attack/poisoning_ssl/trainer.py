from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeqPoisonTrainerResult:
    output_dir: Path
    metadata: dict[str, object]


class SeqPoisonTrainer:
    def train(self, *, output_dir: str | Path) -> SeqPoisonTrainerResult:
        raise NotImplementedError(
            "SeqPoison-SBR Phase 1 provides the training interface only. "
            "Real Poisoning-SSL classifier/generator/discriminator training is "
            "not implemented; tests must inject generated candidates explicitly."
        )


__all__ = ["SeqPoisonTrainer", "SeqPoisonTrainerResult"]
