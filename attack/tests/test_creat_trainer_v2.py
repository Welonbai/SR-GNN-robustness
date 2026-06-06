from __future__ import annotations

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import CreatAdditiveSBRConfig
from attack.creat.trainer import CreatAdditiveSBRTrainer


class _Adapter:
    embedding_dim = 2

    def encode_session(self, session):
        return torch.tensor([float(sum(session)), float(len(session))])

    def encode_sessions(self, sessions):
        return torch.stack([self.encode_session(session) for session in sessions])

    def item_embeddings(self, session):
        return torch.tensor([[float(item), 1.0] for item in session])

    def target_embedding(self, target_item):
        return torch.tensor([float(target_item), 1.0])

    def target_score_for_prefix(self, prefix, target_item):
        return float(sum(prefix))

    def valid_position_mask(self, session, target_item, topk_ratio, nonzero_when_possible=True):
        return torch.tensor([False] + [item != target_item for item in session[1:]])


def test_v2_trainer_runs_both_phases_and_records_raw_stats() -> None:
    config = CreatAdditiveSBRConfig(
        enabled=True,
        variant="v2",
        attack_epochs=1,
        consistency_epochs=1,
        batch_size=2,
    )
    result = CreatAdditiveSBRTrainer(
        adapter=_Adapter(),
        config=config,
        replacement_topk_ratio=1.0,
        seed=7,
    ).train(target_item=9, template_sessions=[[1, 2, 3], [3, 4, 5]])
    assert [row["phase"] for row in result.history["epochs"]] == ["attack", "consistency"]
    assert result.history["final_policy_phase"] == "consistency"
    assert result.reward_table is not None
    assert "dpp_raw_logdet" in result.history["candidate_reward_stats"]
    assert all("position_entropy" in row for row in result.history["epochs"])
