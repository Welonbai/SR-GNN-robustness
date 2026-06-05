from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.creat.candidates import valid_position_mask
from attack.creat.poison_builder import build_creat_poisoned_sessions


class _FakeAdapter:
    def encode_session(self, session):
        return torch.ones(2)

    def item_embeddings(self, session):
        return torch.ones(len(session), 2)

    def valid_position_mask(
        self,
        session,
        target_item,
        topk_ratio,
        nonzero_when_possible=True,
    ):
        return torch.tensor(
            [
                is_valid and int(item) != int(target_item)
                for is_valid, item in zip(
                    valid_position_mask(
                        len(session),
                        topk_ratio,
                        nonzero_when_possible=nonzero_when_possible,
                    ),
                    session,
                )
            ]
        )


class _LastValidMasker:
    def eval(self):
        return self

    def __call__(self, session_rep, item_embeddings, valid_mask):
        logits = torch.arange(len(valid_mask), dtype=torch.float32)
        return logits.masked_fill(~valid_mask, -1.0e38)


def test_creat_poison_builder_replaces_target_without_mutating_templates() -> None:
    templates = [[1, 2], [3, 4, 5]]
    original = [list(session) for session in templates]
    result = build_creat_poisoned_sessions(
        adapter=_FakeAdapter(),
        masker=_LastValidMasker(),
        target_item=9,
        template_sessions=templates,
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        max_item_id=10,
    )
    assert templates == original
    assert len(result.poisoned_sessions) == len(templates)
    assert result.selected_positions == [1, 2]
    assert result.poisoned_sessions == [[1, 9], [3, 4, 9]]


def test_creat_poison_builder_metadata_counts_pairs() -> None:
    result = build_creat_poisoned_sessions(
        adapter=_FakeAdapter(),
        masker=_LastValidMasker(),
        target_item=9,
        template_sessions=[[1, 2], [3, 4, 5]],
        replacement_topk_ratio=1.0,
        nonzero_when_possible=True,
        max_item_id=10,
    )
    assert result.metadata["effective_poisoned_copied_session_count"] == 2
    assert result.metadata["expanded_poisoned_prefix_label_pair_count"] == 3
    assert result.metadata["target_label_poisoned_pair_count"] == 2
    assert result.metadata["selected_replacement_target_pair_count"] == 2
    assert result.metadata["expanded_target_label_pair_count"] == 2
    assert result.metadata["selected_position_distribution"]["counts"] == {
        "1": 1,
        "2": 1,
    }
