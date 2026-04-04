from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent in minimal envs
    torch = None


def min_max_smooth(scores):
    """
    Apply min-max scaling to a score vector. Returns zeros if the vector is degenerate.
    """
    if torch is not None and isinstance(scores, torch.Tensor):
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        if torch.equal(min_val, max_val):
            return torch.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_val = float(arr.min())
    max_val = float(arr.max())
    if min_val == max_val:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


__all__ = ["min_max_smooth"]
