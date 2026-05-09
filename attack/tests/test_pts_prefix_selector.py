from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.prefix_selector import select_internal_uniform_anchor


def test_internal_uniform_anchor_uses_internal_positions_only() -> None:
    anchors = {
        select_internal_uniform_anchor(4, rng=random.Random(seed))
        for seed in range(100)
    }

    assert anchors <= {1, 2, 3}
    assert 0 not in anchors
    assert 4 not in anchors
    assert anchors == {1, 2, 3}


def test_internal_uniform_anchor_rejects_short_sessions() -> None:
    with pytest.raises(ValueError, match="session length >= 2"):
        select_internal_uniform_anchor(1, rng=random.Random(1))
