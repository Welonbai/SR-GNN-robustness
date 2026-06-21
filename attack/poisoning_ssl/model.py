from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SeqPoisonModelSpec:
    component: str
    upstream_source: str
    status: str = "phase2_migrated"


GENERATOR_MODEL_SPEC = SeqPoisonModelSpec(
    component="generator",
    upstream_source="Seq-poison/generator.py",
)
DISCRIMINATOR_MODEL_SPEC = SeqPoisonModelSpec(
    component="discriminator",
    upstream_source="Seq-poison/discriminator.py",
)
CLASSIFIER_MODEL_SPEC = SeqPoisonModelSpec(
    component="classifier",
    upstream_source="Seq-poison/classify.py",
)


class Generator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = int(vocab_size)
        self.embeddings = nn.Embedding(int(vocab_size), int(embedding_dim))
        self.gru = nn.GRU(int(embedding_dim), int(hidden_dim))
        self.gru2out = nn.Linear(int(hidden_dim), int(vocab_size))

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, int(batch_size), self.hidden_dim, device=device)

    def forward(
        self,
        inp: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embeddings(inp).view(1, -1, self.embedding_dim)
        out, hidden = self.gru(emb, hidden)
        out = self.gru2out(out.view(-1, self.hidden_dim))
        return F.log_softmax(out, dim=1), hidden

    def sample(
        self,
        num_samples: int,
        *,
        start_letter: int = 0,
        device: torch.device | None = None,
        first_step_target_mask: bool = False,
        first_step_mask_target_id: int | None = None,
        target_logit_bias_after_first_step: float = 0.0,
    ) -> torch.Tensor:
        actual_device = device or next(self.parameters()).device
        target_bias = float(target_logit_bias_after_first_step)
        if bool(first_step_target_mask) or target_bias != 0.0:
            if first_step_mask_target_id is None:
                raise ValueError(
                    "first_step_mask_target_id is required when "
                    "first_step_target_mask or target_logit_bias_after_first_step "
                    "is enabled."
                )
            target_id = int(first_step_mask_target_id)
            if target_id <= 0 or target_id >= self.vocab_size:
                raise ValueError(
                    "first_step_mask_target_id must be a positive item id inside "
                    f"the generator vocabulary; got {target_id}, "
                    f"vocab_size={self.vocab_size}."
                )
        samples = torch.zeros(
            int(num_samples),
            self.max_seq_len,
            dtype=torch.long,
            device=actual_device,
        )
        hidden = self.init_hidden(int(num_samples), actual_device)
        inp = torch.full(
            (int(num_samples),),
            int(start_letter),
            dtype=torch.long,
            device=actual_device,
        )
        for index in range(self.max_seq_len):
            out, hidden = self.forward(inp, hidden)
            adjusted = out.clone()
            if bool(first_step_target_mask) and index == 0:
                adjusted[:, int(first_step_mask_target_id)] = float("-inf")
            elif index >= 1 and target_bias != 0.0:
                adjusted[:, int(first_step_mask_target_id)] += target_bias
            sampled = torch.multinomial(torch.softmax(adjusted, dim=1), 1).view(-1)
            samples[:, index] = sampled
            inp = sampled
        return samples

    def batch_nll_loss(self, inp: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp_t = inp.permute(1, 0)
        target_t = target.permute(1, 0)
        hidden = self.init_hidden(batch_size, inp.device)
        loss = torch.zeros((), device=inp.device)
        for index in range(seq_len):
            out, hidden = self.forward(inp_t[index], hidden)
            loss = loss + loss_fn(out, target_t[index])
        return loss

    def batch_pg_loss(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = inp.size()
        if reward.dim() != 1:
            reward = reward.view(-1)
        if int(reward.numel()) != int(batch_size):
            raise RuntimeError(
                "Generator.batch_pg_loss reward batch mismatch: "
                f"input_batch={int(inp.size(0))} target_batch={int(target.size(0))} "
                f"reward_batch={int(reward.numel())}"
            )
        inp_t = inp.permute(1, 0)
        target_t = target.permute(1, 0)
        hidden = self.init_hidden(batch_size, inp.device)
        loss = torch.zeros((), device=inp.device)
        for index in range(seq_len):
            out, hidden = self.forward(inp_t[index], hidden)
            token_logp = out.gather(1, target_t[index].view(-1, 1)).view(-1)
            loss = loss + (-(token_logp * reward).sum())
        return loss / max(1, int(batch_size))

    def batch_target_loss(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        *,
        attack_item: int,
        target_probability: float = 0.9,
    ) -> torch.Tensor:
        batch_size, seq_len = target.size()
        attack_target = target.detach().clone()
        nonpad = attack_target != 0
        random_mask = torch.rand(
            attack_target.size(),
            device=attack_target.device,
        ) < float(target_probability)
        attack_target[nonpad & random_mask] = int(attack_item)
        inp_t = inp.permute(1, 0)
        attack_t = attack_target.permute(1, 0)
        hidden = self.init_hidden(batch_size, inp.device)
        loss = torch.zeros((), device=inp.device)
        for index in range(seq_len):
            out, hidden = self.forward(inp_t[index], hidden)
            token_logp = out.gather(1, attack_t[index].view(-1, 1)).view(-1)
            loss = loss + (-token_logp.sum())
        return loss / max(1, int(batch_size))

    def batch_classifier_loss(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        classifier_reward: torch.Tensor,
    ) -> torch.Tensor:
        return self.batch_pg_loss(inp, target, classifier_reward)


def sample_sequences_in_chunks(
    generator: Any,
    total_count: int,
    *,
    batch_size: int,
    device: torch.device,
    stage_name: str | None = None,
    log_fn: Callable[[str], None] | None = None,
    output_device: torch.device | str | None = None,
    **sample_kwargs: Any,
) -> torch.Tensor:
    requested = int(total_count)
    chunk_size = int(batch_size)
    if requested <= 0:
        raise ValueError("total_count must be positive.")
    if chunk_size <= 0:
        raise ValueError("batch_size must be positive.")
    chunk_count = (requested + chunk_size - 1) // chunk_size
    if log_fn is not None and chunk_count > 1:
        label = stage_name or "generator.sample"
        log_fn(
            f"{label} chunked sampling total_requested={requested} "
            f"chunk_size={chunk_size} chunks={chunk_count}"
        )
    chunks: list[torch.Tensor] = []
    remaining = requested
    with torch.no_grad():
        while remaining > 0:
            current = min(chunk_size, remaining)
            sampled = generator.sample(current, device=device, **sample_kwargs)
            sampled = sampled.detach()
            if output_device is not None:
                sampled = sampled.to(output_device)
            chunks.append(sampled)
            remaining -= current
            del sampled
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


class Discriminator(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)
        self.embeddings = nn.Embedding(int(vocab_size), int(embedding_dim))
        self.gru = nn.GRU(
            int(embedding_dim),
            int(hidden_dim),
            num_layers=2,
            bidirectional=True,
            dropout=float(dropout),
        )
        self.gru2hidden = nn.Linear(4 * int(hidden_dim), int(hidden_dim))
        self.dropout_linear = nn.Dropout(p=float(dropout))
        self.hidden2out = nn.Linear(int(hidden_dim), 1)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(4, int(batch_size), self.hidden_dim, device=device)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        emb = self.embeddings(input).permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))
        out = torch.tanh(out)
        out = self.dropout_linear(out)
        out = self.hidden2out(out)
        return torch.sigmoid(out)

    def batch_classify(self, inp: torch.Tensor) -> torch.Tensor:
        hidden = self.init_hidden(inp.size(0), inp.device)
        return self.forward(inp, hidden).view(-1)


class Classify(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        emb_dim: int,
        filter_sizes: list[int],
        num_filters: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(vocab_size), int(emb_dim))
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, int(n), (int(f), int(emb_dim)))
                for n, f in zip(num_filters, filter_sizes)
            ]
        )
        total_filters = int(sum(num_filters))
        self.highway = nn.Linear(total_filters, total_filters)
        self.dropout = nn.Dropout(p=float(dropout))
        self.lin = nn.Linear(total_filters, int(num_classes))
        self.softmax = nn.Softmax(dim=1)
        self.init_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        gate = torch.sigmoid(highway)
        pred = gate * F.relu(highway) + (1.0 - gate) * pred
        return self.softmax(self.lin(self.dropout(pred)))

    def init_parameters(self) -> None:
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)


def prepare_generator_batch(
    samples: torch.Tensor,
    *,
    start_letter: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = samples.size()
    inp = torch.zeros(
        int(batch_size),
        int(seq_len),
        dtype=torch.long,
        device=samples.device,
    )
    target = samples.long()
    inp[:, 0] = int(start_letter)
    inp[:, 1:] = target[:, : int(seq_len) - 1]
    return inp, target


def padded_sequences(
    sequences: list[list[int]],
    *,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    padded: list[list[int]] = []
    for sequence in sequences:
        if len(sequence) > int(max_seq_len):
            raise ValueError("SeqPoison training sequence exceeded max_seq_len.")
        padded.append(list(sequence) + [0] * (int(max_seq_len) - len(sequence)))
    return torch.tensor(padded, dtype=torch.long, device=device)


def unpad_generated_tensor(samples: torch.Tensor) -> list[list[int]]:
    result: list[list[int]] = []
    for row in samples.detach().cpu().tolist():
        sequence: list[int] = []
        for index, item in enumerate(row):
            item_id = int(item)
            if item_id != 0:
                sequence.append(item_id)
                continue
            if index == len(row) - 1 or int(row[index + 1]) == 0:
                break
        result.append(sequence)
    return result


def classifier_filter_config(input_len: int) -> tuple[list[int], list[int]]:
    upstream_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    upstream_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
    pairs = [
        (size, filters)
        for size, filters in zip(upstream_sizes, upstream_filters)
        if int(size) <= int(input_len)
    ]
    if not pairs:
        pairs = [(1, 100)]
    return [size for size, _ in pairs], [filters for _, filters in pairs]


__all__ = [
    "CLASSIFIER_MODEL_SPEC",
    "DISCRIMINATOR_MODEL_SPEC",
    "GENERATOR_MODEL_SPEC",
    "Classify",
    "Discriminator",
    "Generator",
    "SeqPoisonModelSpec",
    "classifier_filter_config",
    "padded_sequences",
    "prepare_generator_batch",
    "unpad_generated_tensor",
]
