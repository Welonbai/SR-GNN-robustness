from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class PolicySpecialItemIds:
    left_boundary: int
    right_boundary: int

    @property
    def num_item_embeddings(self) -> int:
        return int(self.right_boundary) + 1

    def to_payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateFeatureMetadata:
    candidate_index: int
    position: int
    normalized_position: float
    session_length: int
    target_item: int
    original_item: int
    left_item: int
    right_item: int
    left_is_boundary: bool
    right_is_boundary: bool


@dataclass(frozen=True)
class CandidateFeatureTensors:
    target_item_ids: torch.Tensor
    original_item_ids: torch.Tensor
    left_item_ids: torch.Tensor
    right_item_ids: torch.Tensor
    position_indices: torch.Tensor
    normalized_positions: torch.Tensor
    session_lengths: torch.Tensor

    def __post_init__(self) -> None:
        if self.target_item_ids.ndim != 1:
            raise ValueError("All candidate feature tensors must be 1D.")
        expected_size = int(self.target_item_ids.numel())
        fields = (
            self.original_item_ids,
            self.left_item_ids,
            self.right_item_ids,
            self.position_indices,
            self.normalized_positions,
            self.session_lengths,
        )
        if expected_size <= 0:
            raise ValueError("CandidateFeatureTensors must contain at least one candidate.")
        for value in fields:
            if value.ndim != 1:
                raise ValueError("All candidate feature tensors must be 1D.")
            if int(value.numel()) != expected_size:
                raise ValueError("All candidate feature tensors must have the same length.")

    @property
    def candidate_count(self) -> int:
        return int(self.target_item_ids.numel())

    def to(self, device: torch.device | str) -> "CandidateFeatureTensors":
        return CandidateFeatureTensors(
            target_item_ids=self.target_item_ids.to(device=device),
            original_item_ids=self.original_item_ids.to(device=device),
            left_item_ids=self.left_item_ids.to(device=device),
            right_item_ids=self.right_item_ids.to(device=device),
            position_indices=self.position_indices.to(device=device),
            normalized_positions=self.normalized_positions.to(device=device),
            session_lengths=self.session_lengths.to(device=device),
        )


@dataclass(frozen=True)
class SessionCandidateFeatures:
    metadata: tuple[CandidateFeatureMetadata, ...]
    tensors: CandidateFeatureTensors

    def __post_init__(self) -> None:
        if not self.metadata:
            raise ValueError("SessionCandidateFeatures must contain at least one candidate.")
        if len(self.metadata) != self.tensors.candidate_count:
            raise ValueError("Feature metadata and tensors must have the same candidate count.")


def infer_max_item_id(
    fake_sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
) -> int:
    max_item_id = int(target_item)
    if max_item_id <= 0:
        raise ValueError("target_item must be a positive item id.")

    for session in fake_sessions:
        for item in session:
            item_id = int(item)
            if item_id <= 0:
                raise ValueError("All fake-session item ids must be positive.")
            if item_id > max_item_id:
                max_item_id = item_id
    return max_item_id


def build_policy_special_item_ids(max_item_id: int) -> PolicySpecialItemIds:
    normalized_max_item_id = int(max_item_id)
    if normalized_max_item_id <= 0:
        raise ValueError("max_item_id must be positive.")
    return PolicySpecialItemIds(
        left_boundary=normalized_max_item_id + 1,
        right_boundary=normalized_max_item_id + 2,
    )


def build_session_candidate_features(
    session: Sequence[int],
    candidate_positions: Sequence[int],
    *,
    target_item: int,
    special_item_ids: PolicySpecialItemIds,
) -> SessionCandidateFeatures:
    session_items = [int(item) for item in session]
    if not session_items:
        raise ValueError("session must contain at least one item.")
    if any(item <= 0 for item in session_items):
        raise ValueError("session item ids must be positive.")

    session_length = len(session_items)
    normalized_target_item = int(target_item)
    if normalized_target_item <= 0:
        raise ValueError("target_item must be a positive item id.")

    feature_rows: list[CandidateFeatureMetadata] = []
    for candidate_index, raw_position in enumerate(candidate_positions):
        position = int(raw_position)
        if position < 0 or position >= session_length:
            raise ValueError("candidate_positions contains an out-of-range position.")

        left_is_boundary = position == 0
        right_is_boundary = position == (session_length - 1)
        left_item = (
            special_item_ids.left_boundary
            if left_is_boundary
            else int(session_items[position - 1])
        )
        right_item = (
            special_item_ids.right_boundary
            if right_is_boundary
            else int(session_items[position + 1])
        )
        feature_rows.append(
            CandidateFeatureMetadata(
                candidate_index=int(candidate_index),
                position=position,
                normalized_position=float(position) / float(session_length),
                session_length=int(session_length),
                target_item=normalized_target_item,
                original_item=int(session_items[position]),
                left_item=int(left_item),
                right_item=int(right_item),
                left_is_boundary=bool(left_is_boundary),
                right_is_boundary=bool(right_is_boundary),
            )
        )

    return SessionCandidateFeatures(
        metadata=tuple(feature_rows),
        tensors=build_candidate_feature_tensors(feature_rows),
    )


def build_candidate_feature_tensors(
    feature_rows: Sequence[CandidateFeatureMetadata],
) -> CandidateFeatureTensors:
    normalized_rows = tuple(feature_rows)
    if not normalized_rows:
        raise ValueError("feature_rows must contain at least one candidate.")

    return CandidateFeatureTensors(
        target_item_ids=torch.tensor(
            [row.target_item for row in normalized_rows],
            dtype=torch.long,
        ),
        original_item_ids=torch.tensor(
            [row.original_item for row in normalized_rows],
            dtype=torch.long,
        ),
        left_item_ids=torch.tensor(
            [row.left_item for row in normalized_rows],
            dtype=torch.long,
        ),
        right_item_ids=torch.tensor(
            [row.right_item for row in normalized_rows],
            dtype=torch.long,
        ),
        position_indices=torch.tensor(
            [row.position for row in normalized_rows],
            dtype=torch.float32,
        ),
        normalized_positions=torch.tensor(
            [row.normalized_position for row in normalized_rows],
            dtype=torch.float32,
        ),
        session_lengths=torch.tensor(
            [row.session_length for row in normalized_rows],
            dtype=torch.float32,
        ),
    )


__all__ = [
    "CandidateFeatureMetadata",
    "CandidateFeatureTensors",
    "PolicySpecialItemIds",
    "SessionCandidateFeatures",
    "build_candidate_feature_tensors",
    "build_policy_special_item_ids",
    "build_session_candidate_features",
    "infer_max_item_id",
]
