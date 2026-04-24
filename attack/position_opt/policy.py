from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from attack.common.position_opt_policy_feature_sets import (
    PositionOptPolicyFeatureSetSpec,
    resolve_position_opt_policy_feature_set,
)
from attack.position_opt.feature_builder import (
    CandidateFeatureTensors,
    SessionCandidateFeatures,
)


_ITEM_FEATURE_TENSOR_NAMES = {
    "target_item": "target_item_ids",
    "original_item": "original_item_ids",
    "left_item": "left_item_ids",
    "right_item": "right_item_ids",
}
_SCALAR_FEATURE_TENSOR_NAMES = {
    "position_index": "position_indices",
    "normalized_position": "normalized_positions",
    "session_length": "session_lengths",
    "prefix_score": "prefix_scores",
    "has_prefix": "has_prefixes",
}


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
        if normalized_vocab_size <= 0:
            raise ValueError("num_item_embeddings must be positive.")
        if normalized_embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if normalized_hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        feature_set_spec = resolve_position_opt_policy_feature_set(policy_feature_set)
        policy_input_dim = feature_set_spec.input_dim(
            embedding_dim=normalized_embedding_dim
        )
        if policy_input_dim <= 0:
            raise ValueError("policy_input_dim must be positive.")

        self.num_item_embeddings = normalized_vocab_size
        self.embedding_dim = normalized_embedding_dim
        self.hidden_dim = normalized_hidden_dim
        self.policy_feature_set = feature_set_spec.name
        self.feature_set_spec = feature_set_spec
        self.active_item_features = tuple(feature_set_spec.item_features)
        self.active_scalar_features = tuple(feature_set_spec.scalar_features)
        # Preserve the historical name used in artifacts/tests.
        self.scalar_feature_names = self.active_scalar_features
        self.item_feature_names = self.active_item_features
        self.policy_input_dim = int(policy_input_dim)
        self.item_embedding = (
            nn.Embedding(
                num_embeddings=self.num_item_embeddings,
                embedding_dim=self.embedding_dim,
            )
            if self.active_item_features
            else None
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.policy_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def score_candidates(self, features: CandidateFeatureTensors) -> torch.Tensor:
        device = self._model_device()
        batch = features.to(device)
        model_input_columns: list[torch.Tensor] = []
        if self.active_item_features:
            if self.item_embedding is None:
                raise RuntimeError("item_embedding must be initialized when item features are active.")
            item_id_tensors = tuple(
                getattr(batch, _ITEM_FEATURE_TENSOR_NAMES[feature_name])
                for feature_name in self.active_item_features
            )
            _validate_item_id_range(
                item_id_tensors,
                num_item_embeddings=self.num_item_embeddings,
            )
            model_input_columns.extend(
                self.item_embedding(item_tensor)
                for item_tensor in item_id_tensors
            )

        scalar_feature_columns = [
            getattr(batch, _SCALAR_FEATURE_TENSOR_NAMES[feature_name]).to(
                dtype=torch.float32
            )
            for feature_name in self.active_scalar_features
        ]
        if scalar_feature_columns:
            model_input_columns.append(
                torch.stack(tuple(scalar_feature_columns), dim=1)
            )
        if not model_input_columns:
            raise RuntimeError("Policy feature set produced no model inputs.")

        model_input = torch.cat(tuple(model_input_columns), dim=1)
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

    def input_metadata(self) -> dict[str, object]:
        return {
            "policy_feature_set": str(self.policy_feature_set),
            "active_item_features": list(self.active_item_features),
            "active_scalar_features": list(self.active_scalar_features),
            "policy_input_dim": int(self.policy_input_dim),
            "policy_embedding_dim": int(self.embedding_dim),
            "policy_hidden_dim": int(self.hidden_dim),
        }

    def _model_device(self) -> torch.device:
        if self.item_embedding is not None:
            return self.item_embedding.weight.device
        return next(self.scorer.parameters()).device


def resolve_policy_feature_set_spec(
    policy_feature_set: str,
) -> PositionOptPolicyFeatureSetSpec:
    return resolve_position_opt_policy_feature_set(policy_feature_set)


def _validate_item_id_range(
    item_tensors: Sequence[torch.Tensor],
    *,
    num_item_embeddings: int,
) -> None:
    for tensor in item_tensors:
        if int(torch.min(tensor).item()) < 0:
            raise ValueError("Candidate item ids must be non-negative.")
        if int(torch.max(tensor).item()) >= int(num_item_embeddings):
            raise ValueError("Candidate item ids exceed the embedding table range.")


__all__ = ["SharedContextualPositionPolicy", "resolve_policy_feature_set_spec"]
