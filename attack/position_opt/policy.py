from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from attack.position_opt.feature_builder import (
    CandidateFeatureTensors,
    SessionCandidateFeatures,
)

_BASE_SCALAR_FEATURE_NAMES = (
    "position_index",
    "normalized_position",
    "session_length",
)
_PREFIX_SCORE_SCALAR_FEATURE_NAMES = (
    *_BASE_SCALAR_FEATURE_NAMES,
    "prefix_score",
    "has_prefix",
)


class SharedContextualPositionPolicy(nn.Module):
    """Shared contextual scorer for candidate replacement positions."""

    def __init__(
        self,
        *,
        num_item_embeddings: int,
        embedding_dim: int,
        hidden_dim: int,
        policy_feature_set: str = "local_context",
    ) -> None:
        super().__init__()
        normalized_vocab_size = int(num_item_embeddings)
        normalized_embedding_dim = int(embedding_dim)
        normalized_hidden_dim = int(hidden_dim)
        normalized_policy_feature_set = str(policy_feature_set).strip().lower()
        if normalized_vocab_size <= 0:
            raise ValueError("num_item_embeddings must be positive.")
        if normalized_embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if normalized_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if normalized_policy_feature_set not in {
            "local_context",
            "local_context_prefix_score_prob",
        }:
            raise ValueError(
                "policy_feature_set must be 'local_context' or "
                "'local_context_prefix_score_prob'."
            )

        self.num_item_embeddings = normalized_vocab_size
        self.embedding_dim = normalized_embedding_dim
        self.hidden_dim = normalized_hidden_dim
        self.policy_feature_set = normalized_policy_feature_set
        self.scalar_feature_names = (
            _BASE_SCALAR_FEATURE_NAMES
            if self.policy_feature_set == "local_context"
            else _PREFIX_SCORE_SCALAR_FEATURE_NAMES
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.num_item_embeddings,
            embedding_dim=self.embedding_dim,
        )
        self.scorer = nn.Sequential(
            nn.Linear((self.embedding_dim * 4) + len(self.scalar_feature_names), self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def score_candidates(self, features: CandidateFeatureTensors) -> torch.Tensor:
        device = self.item_embedding.weight.device
        batch = features.to(device)
        _validate_item_id_range(
            batch,
            num_item_embeddings=self.num_item_embeddings,
        )

        target_embedding = self.item_embedding(batch.target_item_ids)
        original_embedding = self.item_embedding(batch.original_item_ids)
        left_embedding = self.item_embedding(batch.left_item_ids)
        right_embedding = self.item_embedding(batch.right_item_ids)
        scalar_feature_columns = [
            batch.position_indices.to(dtype=torch.float32),
            batch.normalized_positions.to(dtype=torch.float32),
            batch.session_lengths.to(dtype=torch.float32),
        ]
        if self.policy_feature_set == "local_context_prefix_score_prob":
            scalar_feature_columns.extend(
                (
                    batch.prefix_scores.to(dtype=torch.float32),
                    batch.has_prefixes.to(dtype=torch.float32),
                )
            )
        scalar_features = torch.stack(tuple(scalar_feature_columns), dim=1)
        model_input = torch.cat(
            (
                target_embedding,
                original_embedding,
                left_embedding,
                right_embedding,
                scalar_features,
            ),
            dim=1,
        )
        logits = self.scorer(model_input).squeeze(-1)
        if logits.ndim != 1:
            raise RuntimeError("Policy scorer must return a 1D logit per candidate.")
        return logits

    def export_logits(
        self,
        features: Sequence[SessionCandidateFeatures],
    ) -> list[torch.Tensor]:
        with torch.no_grad():
            return [
                self.score_candidates(session_features.tensors).detach().cpu().clone()
                for session_features in features
            ]


def _validate_item_id_range(
    features: CandidateFeatureTensors,
    *,
    num_item_embeddings: int,
) -> None:
    for tensor in (
        features.target_item_ids,
        features.original_item_ids,
        features.left_item_ids,
        features.right_item_ids,
    ):
        if int(torch.min(tensor).item()) < 0:
            raise ValueError("Candidate item ids must be non-negative.")
        if int(torch.max(tensor).item()) >= int(num_item_embeddings):
            raise ValueError("Candidate item ids exceed the embedding table range.")


__all__ = ["SharedContextualPositionPolicy"]
