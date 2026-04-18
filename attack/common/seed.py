from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derive_seed(base_seed: int, *components: object) -> int:
    normalized_base = int(base_seed)
    if not components:
        return normalized_base
    payload = json.dumps(
        [normalized_base, *components],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(derived % (2**31 - 1))


__all__ = ["derive_seed", "set_seed"]
