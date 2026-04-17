from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PositionOptObjectiveResult:
    reward: torch.Tensor
    loss: torch.Tensor
    target_utility: torch.Tensor
    gt_penalty: torch.Tensor
    gt_drop: torch.Tensor


def compute_position_opt_objective(
    target_utility: torch.Tensor | float,
    *,
    clean_gt_utility: torch.Tensor | float | None = None,
    poisoned_gt_utility: torch.Tensor | float | None = None,
    enable_gt_penalty: bool = False,
    gt_penalty_weight: float = 0.0,
    gt_tolerance: float = 0.0,
) -> PositionOptObjectiveResult:
    target_tensor = _as_tensor(target_utility)
    zero = target_tensor.new_tensor(0.0)
    gt_penalty = zero
    gt_drop = zero

    if enable_gt_penalty:
        if clean_gt_utility is None or poisoned_gt_utility is None:
            raise ValueError(
                "clean_gt_utility and poisoned_gt_utility are required when "
                "enable_gt_penalty is true."
            )
        gt_drop, gt_penalty = compute_asymmetric_gt_penalty(
            clean_gt_utility,
            poisoned_gt_utility,
            tolerance=float(gt_tolerance),
            reference_tensor=target_tensor,
        )

    reward = target_tensor - target_tensor.new_tensor(float(gt_penalty_weight)) * gt_penalty
    loss = -reward
    return PositionOptObjectiveResult(
        reward=reward,
        loss=loss,
        target_utility=target_tensor,
        gt_penalty=gt_penalty,
        gt_drop=gt_drop,
    )


def compute_asymmetric_gt_penalty(
    clean_gt_utility: torch.Tensor | float,
    poisoned_gt_utility: torch.Tensor | float,
    *,
    tolerance: float = 0.0,
    reference_tensor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    clean_gt_tensor = _as_tensor(clean_gt_utility, reference_tensor=reference_tensor)
    poisoned_gt_tensor = _as_tensor(poisoned_gt_utility, reference_tensor=clean_gt_tensor)
    gt_drop = clean_gt_tensor - poisoned_gt_tensor
    penalty = torch.relu(gt_drop - clean_gt_tensor.new_tensor(float(tolerance)))
    return gt_drop, penalty


def _as_tensor(
    value: torch.Tensor | float,
    *,
    reference_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if reference_tensor is not None:
        return reference_tensor.new_tensor(float(value))
    return torch.tensor(float(value), dtype=torch.float32)


__all__ = [
    "PositionOptObjectiveResult",
    "compute_asymmetric_gt_penalty",
    "compute_position_opt_objective",
]
