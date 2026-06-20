from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Protocol, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import PoisoningSSLSBRConfig
from attack.poisoning_ssl.dataset_bridge import SeqPoisonDatasetBundle
from attack.poisoning_ssl.model import sample_sequences_in_chunks, unpad_generated_tensor
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
        generation_start = time.perf_counter()
        first_step_target_mask = bool(request.config.first_step_target_mask)
        target_logit_bias = float(request.config.target_logit_bias_after_first_step)
        seqpoison_target_id = int(request.dataset_bundle.seqpoison_target_item)
        sample_batch_size = int(request.config.generation_sample_batch_size)
        sample_chunk_count = (
            int(request.n_candidates) + sample_batch_size - 1
        ) // sample_batch_size
        samples = sample_sequences_in_chunks(
            result.generator,
            int(request.n_candidates),
            batch_size=sample_batch_size,
            device=result.device,
            stage_name=f"final generation round {int(request.round_index) + 1}",
            log_fn=lambda message: print(
                f"[SeqPoison-SBR][target={int(request.target_item)}] {message}",
                flush=True,
            ),
            output_device=torch.device("cpu"),
            first_step_target_mask=first_step_target_mask,
            first_step_mask_target_id=(
                seqpoison_target_id
                if first_step_target_mask or target_logit_bias != 0.0
                else None
            ),
            target_logit_bias_after_first_step=target_logit_bias,
        )
        item_sequences = unpad_generated_tensor(samples)
        unexpected_seqpoison_pos0 = sum(
            1
            for sequence in item_sequences
            if sequence and int(sequence[0]) == seqpoison_target_id
        )
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
            "generation_round_duration_sec": float(time.perf_counter() - generation_start),
            "candidate_format": "[user_id, item1, item2, ...]",
            "synthetic_user_id_start": int(synthetic_user_start),
            "first_step_target_mask": first_step_target_mask,
            "first_step_target_mask_applied": first_step_target_mask,
            "first_step_target_mask_target_id_canonical": int(request.target_item),
            "first_step_target_mask_target_id_seqpoison": seqpoison_target_id,
            "unexpected_pos0_after_mask_candidate_count": (
                int(unexpected_seqpoison_pos0) if first_step_target_mask else 0
            ),
            "target_logit_bias_after_first_step": target_logit_bias,
            "target_logit_bias_after_first_step_applied": target_logit_bias != 0.0,
            "target_logit_bias_target_id_canonical": int(request.target_item),
            "target_logit_bias_target_id_seqpoison": seqpoison_target_id,
            "target_logit_bias_positions": (
                "positions>=1" if target_logit_bias != 0.0 else "none"
            ),
            "generation_sample_batch_size": sample_batch_size,
            "sample_chunk_count": int(sample_chunk_count),
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
                "first_step_target_mask": first_step_target_mask,
                "first_step_target_mask_applied": first_step_target_mask,
                "first_step_target_mask_target_id_canonical": int(request.target_item),
                "first_step_target_mask_target_id_seqpoison": seqpoison_target_id,
                "target_logit_bias_after_first_step": target_logit_bias,
                "target_logit_bias_after_first_step_applied": target_logit_bias != 0.0,
                "target_logit_bias_target_id_canonical": int(request.target_item),
                "target_logit_bias_target_id_seqpoison": seqpoison_target_id,
                "target_logit_bias_positions": (
                    "positions>=1" if target_logit_bias != 0.0 else "none"
                ),
                "generation_sample_batch_size": sample_batch_size,
                "sample_chunk_count": int(sample_chunk_count),
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
