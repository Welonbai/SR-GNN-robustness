from __future__ import annotations

import torch
from torch import nn


class CreatMasker(nn.Module):
    def __init__(
        self,
        *,
        session_dim: int,
        item_dim: int,
        hidden_dim: int,
        position_embedding_dim: int,
        max_session_length: int,
    ) -> None:
        super().__init__()
        if max_session_length <= 0:
            raise ValueError("max_session_length must be positive.")
        self.position_embedding = nn.Embedding(
            int(max_session_length),
            int(position_embedding_dim),
        )
        input_dim = int(session_dim) + int(item_dim) + int(position_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        session_representation: torch.Tensor,
        item_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if session_representation.dim() != 1:
            raise ValueError("session_representation must be a 1-D tensor.")
        if item_embeddings.dim() != 2:
            raise ValueError("item_embeddings must be a 2-D tensor.")
        length = int(item_embeddings.shape[0])
        if valid_mask.shape[0] != length:
            raise ValueError("valid_mask must align with item_embeddings length.")
        if length > int(self.position_embedding.num_embeddings):
            raise ValueError("Session length exceeds max_session_length.")

        positions = torch.arange(length, device=item_embeddings.device, dtype=torch.long)
        pos_embeddings = self.position_embedding(positions)
        repeated_session = session_representation.to(item_embeddings.device).view(1, -1).expand(length, -1)
        features = torch.cat([repeated_session, item_embeddings, pos_embeddings], dim=1)
        logits = self.mlp(features).squeeze(-1)
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)


__all__ = ["CreatMasker"]
