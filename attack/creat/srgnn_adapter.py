from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from attack.creat.candidates import valid_position_mask_for_session
from pytorch_code.model import trans_to_cpu, trans_to_cuda, validate_session_mask_array
from pytorch_code.utils import Data


class SRGNNRepresentationAdapter:
    def __init__(self, runner) -> None:
        if runner.model is None:
            raise RuntimeError("SR-GNN runner must have an initialized model.")
        self.runner = runner
        self.model = runner.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.hidden_size)

    @property
    def max_item_id(self) -> int:
        return int(self.model.embedding.num_embeddings - 1)

    def encode_session(self, session: Sequence[int]) -> torch.Tensor:
        return self.encode_sessions([session]).squeeze(0)

    def encode_sessions(self, sessions: Sequence[Sequence[int]]) -> torch.Tensor:
        normalized = _normalize_sessions(sessions)
        data = Data((normalized, [1] * len(normalized)), shuffle=False)
        representations: list[torch.Tensor] = []
        with torch.no_grad():
            for batch_indices in data.generate_batch(self.model.batch_size):
                seq_hidden, mask = self._seq_hidden_and_mask(data, batch_indices)
                reps = self.model.compute_session_representation(seq_hidden, mask)
                representations.append(trans_to_cpu(reps.detach()))
        return torch.cat(representations, dim=0)

    def item_embeddings(self, session: Sequence[int]) -> torch.Tensor:
        normalized = _normalize_session(session)
        item_ids = torch.as_tensor(normalized, dtype=torch.long, device=self.model.embedding.weight.device)
        if torch.any(item_ids <= 0) or torch.any(item_ids >= self.model.embedding.num_embeddings):
            raise ValueError("Session contains item ids outside the SR-GNN embedding range.")
        with torch.no_grad():
            embeddings = self.model.embedding(item_ids)
        return trans_to_cpu(embeddings.detach())

    def target_embedding(self, target_item: int) -> torch.Tensor:
        item = int(target_item)
        if item <= 0 or item >= self.model.embedding.num_embeddings:
            raise ValueError("target_item is outside the SR-GNN embedding range.")
        item_id = torch.as_tensor([item], dtype=torch.long, device=self.model.embedding.weight.device)
        with torch.no_grad():
            embedding = self.model.embedding(item_id).squeeze(0)
        return trans_to_cpu(embedding.detach())

    def valid_position_mask(
        self,
        session: Sequence[int],
        target_item: int,
        topk_ratio: float,
        nonzero_when_possible: bool = True,
    ) -> torch.Tensor:
        if int(target_item) <= 0:
            raise ValueError("target_item must be positive.")
        mask = valid_position_mask_for_session(
            session,
            int(target_item),
            topk_ratio,
            nonzero_when_possible=nonzero_when_possible,
        )
        return torch.as_tensor(mask, dtype=torch.bool)

    def target_score_for_prefix(self, prefix: Sequence[int], target_item: int) -> float:
        if not prefix:
            return 0.0
        return float(self.target_scores_for_prefixes([prefix], target_item)[0])

    def target_scores_for_prefixes(
        self,
        prefixes: Sequence[Sequence[int]],
        target_item: int,
    ) -> list[float]:
        normalized = _normalize_sessions(prefixes)
        target_index = int(target_item) - 1
        if target_index < 0 or target_index >= self.model.embedding.num_embeddings - 1:
            raise ValueError("target_item is outside the score vector range.")
        data = Data((normalized, [1] * len(normalized)), shuffle=False)
        target_scores: list[float] = []
        with torch.no_grad():
            for batch_indices in data.generate_batch(self.model.batch_size):
                seq_hidden, mask = self._seq_hidden_and_mask(data, batch_indices)
                scores = self.model.compute_scores(seq_hidden, mask)
                values = trans_to_cpu(scores[:, target_index].detach()).tolist()
                target_scores.extend(float(value) for value in values)
        return target_scores

    def _seq_hidden_and_mask(self, data: Data, batch_indices) -> tuple[torch.Tensor, torch.Tensor]:
        alias_inputs, A, items, mask, _targets = data.get_slice(batch_indices)
        alias_inputs = trans_to_cuda(torch.from_numpy(np.asarray(alias_inputs, dtype=np.int64)))
        items = trans_to_cuda(torch.from_numpy(np.asarray(items, dtype=np.int64)))
        A = trans_to_cuda(torch.from_numpy(np.asarray(A, dtype=np.float32)))
        mask_tensor = trans_to_cuda(torch.from_numpy(validate_session_mask_array(mask)))
        hidden = self.model(items, A)
        seq_hidden = torch.stack(
            [
                hidden[row_index][alias_inputs[row_index]]
                for row_index in range(len(alias_inputs))
            ]
        )
        return seq_hidden, mask_tensor


def _normalize_session(session: Sequence[int]) -> list[int]:
    normalized = [int(item) for item in session]
    if not normalized:
        raise ValueError("Session must contain at least one item.")
    return normalized


def _normalize_sessions(sessions: Sequence[Sequence[int]]) -> list[list[int]]:
    normalized = [_normalize_session(session) for session in sessions]
    if not normalized:
        raise ValueError("At least one session is required.")
    return normalized


__all__ = ["SRGNNRepresentationAdapter"]
