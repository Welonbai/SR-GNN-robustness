from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PerSessionLogitPolicy(nn.Module):
    """MVP position policy with one learnable logit vector per fake session."""

    def __init__(self, candidate_sizes: Sequence[int]) -> None:
        super().__init__()
        normalized_sizes = [int(size) for size in candidate_sizes]
        if not normalized_sizes:
            raise ValueError("candidate_sizes must contain at least one session.")
        if any(size <= 0 for size in normalized_sizes):
            raise ValueError("Each session must have at least one candidate position.")

        self._candidate_sizes = tuple(normalized_sizes)
        self._logits = nn.ParameterList(
            [nn.Parameter(torch.zeros(size, dtype=torch.float32)) for size in normalized_sizes]
        )

    @property
    def num_sessions(self) -> int:
        return len(self._logits)

    @property
    def candidate_sizes(self) -> tuple[int, ...]:
        return self._candidate_sizes

    def get_logits(self, session_idx: int) -> torch.Tensor:
        if session_idx < 0 or session_idx >= len(self._logits):
            raise IndexError("session_idx is outside the policy range.")
        return self._logits[int(session_idx)]

    def export_logits(self) -> list[torch.Tensor]:
        return [param.detach().cpu().clone() for param in self._logits]


__all__ = ["PerSessionLogitPolicy"]
