from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    target_item: int
    n_candidates: int
    max_seq_len: int
    seed: int
    output_dir: Path
    round_index: int


class CandidateGenerator(Protocol):
    """Generate upstream-style fake user sequences.

    The returned sequence contract is [user_id, item1, item2, ...]. The first
    token is a synthetic user id and is removed by SeqPoison-SBR postprocessing.
    Implementations must not return item-only SBR sessions.
    """

    def generate(self, request: GenerationRequest) -> list[list[int]]:
        ...


class UnimplementedSeqPoisonCandidateGenerator:
    def generate(self, request: GenerationRequest) -> list[list[int]]:
        raise NotImplementedError(
            "SeqPoison-SBR Phase 1 has no real candidate generator implementation. "
            "Mock candidates are allowed only through explicit dependency injection "
            "in tests; experiment YAML must not silently use mock fake sessions."
        )


class StaticCandidateGenerator:
    """Test helper for explicit dependency injection only."""

    def __init__(self, rounds: Sequence[Sequence[Sequence[int]]]) -> None:
        self._rounds = [
            [[int(item) for item in session] for session in round_sessions]
            for round_sessions in rounds
        ]

    def generate(self, request: GenerationRequest) -> list[list[int]]:
        index = int(request.round_index)
        if index >= len(self._rounds):
            return []
        return [list(session) for session in self._rounds[index]]


__all__ = [
    "CandidateGenerator",
    "GenerationRequest",
    "StaticCandidateGenerator",
    "UnimplementedSeqPoisonCandidateGenerator",
]
