from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.creat.masker import CreatMasker


def test_creat_masker_logits_shape_equals_session_length() -> None:
    masker = CreatMasker(
        session_dim=4,
        item_dim=4,
        hidden_dim=8,
        position_embedding_dim=3,
        max_session_length=5,
    )
    logits = masker(
        torch.ones(4),
        torch.ones(3, 4),
        torch.tensor([True, True, True]),
    )
    assert logits.shape == (3,)


def test_creat_masker_masks_invalid_positions() -> None:
    masker = CreatMasker(
        session_dim=4,
        item_dim=4,
        hidden_dim=8,
        position_embedding_dim=3,
        max_session_length=5,
    )
    logits = masker(
        torch.ones(4),
        torch.ones(4, 4),
        torch.tensor([False, True, False, True]),
    )
    assert logits[0].item() < -1.0e30
    assert logits[2].item() < -1.0e30
    assert torch.isfinite(logits[1])
    assert torch.isfinite(logits[3])
