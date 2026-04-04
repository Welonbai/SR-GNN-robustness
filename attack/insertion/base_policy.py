from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class InsertionPolicy(ABC):
    @abstractmethod
    def apply(self, session: Sequence[int], target_item: int) -> list[int]:
        raise NotImplementedError


__all__ = ["InsertionPolicy"]
