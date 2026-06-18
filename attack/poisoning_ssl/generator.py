from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import PoisoningSSLSBRConfig
from attack.poisoning_ssl.dataset_bridge import SeqPoisonDatasetBundle
from attack.poisoning_ssl.model import unpad_generated_tensor
from attack.poisoning_ssl.trainer import SeqPoisonTrainer


@dataclass(frozen=True)
class GenerationRequest:
    target_item: int
    n_candidates: int
    max_seq_len: int
    seed: int
    output_dir: Path
    round_index: int
    dataset_bundle: SeqPoisonDatasetBundle | None = None
    valid_item_ids: set[int] | None = None
    config: PoisoningSSLSBRConfig | None = None
    training_seed: int | None = None


class CandidateGenerator(Protocol):
    """Generate upstream-style fake user sequences.

    The returned sequence contract is [user_id, item1, item2, ...]. The first
    token is a synthetic user id and is removed by SeqPoison-SBR postprocessing.
    Implementations must not return item-only SBR sessions.
    """

    def generate(self, request: GenerationRequest) -> list[list[int]]:
        ...


class UnimplementedSeqPoisonCandidateGenerator:
    def generate(self, request: GenerationRequest) -> list[list[int]]:
        raise NotImplementedError(
            "SeqPoison-SBR Phase 1 has no real candidate generator implementation. "
            "Mock candidates are allowed only through explicit dependency injection "
            "in tests; experiment YAML must not silently use mock fake sessions."
        )


class RealSeqPoisonCandidateGenerator:
    """Train/load local Seq-poison components and sample fake user sequences."""

    def __init__(self, trainer: SeqPoisonTrainer | None = None) -> None:
        self._trainer = trainer or SeqPoisonTrainer()
        self.last_metadata: dict[str, object] = {}

    def generate(self, request: GenerationRequest) -> list[list[int]]:
        if request.dataset_bundle is None:
            raise ValueError("RealSeqPoisonCandidateGenerator requires dataset_bundle.")
        if request.config is None:
            raise ValueError("RealSeqPoisonCandidateGenerator requires config.")
        if request.valid_item_ids is None:
            raise ValueError("RealSeqPoisonCandidateGenerator requires valid_item_ids.")
        result = self._trainer.train_or_load(
            output_dir=request.output_dir,
            dataset_bundle=request.dataset_bundle,
            config=request.config,
            target_item=int(request.target_item),
            seed=int(
                getattr(request, "training_seed", None)
                if getattr(request, "training_seed", None) is not None
                else request.seed
            ),
        )
        import torch

        torch.manual_seed(int(request.seed))
        samples = result.generator.sample(
            int(request.n_candidates),
            device=result.device,
        )
        item_sequences = unpad_generated_tensor(samples)
        candidates: list[list[int]] = []
        synthetic_user_start = 1 + (int(request.round_index) * int(request.n_candidates))
        max_candidate_length = int(request.max_seq_len) + 1
        for offset, seqpoison_sequence in enumerate(item_sequences):
            canonical_sequence = request.dataset_bundle.to_canonical_sequence(
                seqpoison_sequence
            )
            candidate = [synthetic_user_start + int(offset), *canonical_sequence]
            if len(candidate) > max_candidate_length:
                raise RuntimeError(
                    "SeqPoison-SBR real generator produced a candidate longer than "
                    "max_seq_len + 1; "
                    f"candidate_length={len(candidate)}, "
                    f"max_allowed={max_candidate_length}."
                )
            candidates.append(candidate)
        invalid_returned = sum(
            1
            for candidate in candidates
            for item in candidate[1:]
            if int(item) != 0 and int(item) not in request.valid_item_ids
        )
        if invalid_returned:
            raise RuntimeError(
                "SeqPoison-SBR real generator returned canonical item IDs outside "
                f"valid_item_ids; invalid_token_count={invalid_returned}."
            )
        self.last_metadata = {
            "generation_backend": "real",
            "real_generation_implemented": True,
            "round_index": int(request.round_index),
            "raw_candidate_count": int(len(candidates)),
            "candidate_format": "[user_id, item1, item2, ...]",
            "synthetic_user_id_start": int(synthetic_user_start),
            "classifier_checkpoint_path": str(result.classifier_checkpoint_path),
            "generator_checkpoint_path": str(result.generator_checkpoint_path),
            "discriminator_checkpoint_path": str(result.discriminator_checkpoint_path),
            "training_log_path": str(result.training_log_path),
            "generation_log_path": str(result.generation_log_path),
            **result.metadata,
        }
        save_json(
            {
                "round_index": int(request.round_index),
                "raw_candidate_count": int(len(candidates)),
                "candidate_format": "[user_id, item1, item2, ...]",
            },
            result.generation_log_path,
        )
        return candidates


class StaticCandidateGenerator:
    """Test helper for explicit dependency injection only."""

    def __init__(self, rounds: Sequence[Sequence[Sequence[int]]]) -> None:
        self._rounds = [
            [[int(item) for item in session] for session in round_sessions]
            for round_sessions in rounds
        ]

    def generate(self, request: GenerationRequest) -> list[list[int]]:
        index = int(request.round_index)
        if index >= len(self._rounds):
            return []
        return [list(session) for session in self._rounds[index]]


__all__ = [
    "CandidateGenerator",
    "GenerationRequest",
    "RealSeqPoisonCandidateGenerator",
    "StaticCandidateGenerator",
    "UnimplementedSeqPoisonCandidateGenerator",
]
