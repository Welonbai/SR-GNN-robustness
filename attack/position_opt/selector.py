from __future__ import annotations

import torch
import torch.nn.functional as F


def select_position_train(logits: torch.Tensor, temperature: float) -> tuple[int, torch.Tensor]:
    """Sample a hard position with ST-Gumbel while keeping soft backward gradients."""

    _validate_logits(logits)
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive.")

    selection = F.gumbel_softmax(logits, tau=float(temperature), hard=True, dim=0)
    selected_index = int(torch.argmax(selection).item())
    return selected_index, selection


def select_position_eval(logits: torch.Tensor) -> int:
    _validate_logits(logits)
    return int(torch.argmax(logits, dim=0).item())


def sample_position_reinforce(
    logits: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Sample one categorical action and return its log-probability.

    The returned log-probability is intended for REINFORCE-style policy-gradient
    updates, where one sampled position is taken per fake session.
    """

    _validate_logits(logits)
    distribution = torch.distributions.Categorical(logits=logits)
    sampled_index = distribution.sample()
    log_prob = distribution.log_prob(sampled_index)
    entropy = distribution.entropy()
    return int(sampled_index.item()), log_prob, entropy


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor.")
    if logits.numel() == 0:
        raise ValueError("logits must contain at least one candidate.")


__all__ = ["sample_position_reinforce", "select_position_eval", "select_position_train"]
