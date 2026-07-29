from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import load_config
from attack.models._srgnn_base import SRGNNBaseRunner
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from pytorch_code.model import forward as srg_forward, validate_session_mask_array
from pytorch_code.utils import Data


def test_srgnn_score_session_matches_forward_after_representation_refactor() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    runner = SRGNNBaseRunner(config, n_node=20)
    runner.build_model(build_srgnn_opt_from_train_config(config.attack.poison_model.params["train"]))
    session = [1, 2, 3]
    score_session_scores = runner.score_session(session)
    data = Data(([session], [0]), shuffle=False)
    with torch.no_grad():
        _targets, forward_scores = srg_forward(runner.model, np.array([0]), data)
    assert torch.allclose(score_session_scores, forward_scores.squeeze(0).cpu())


def test_srgnn_compute_scores_reuses_same_representation_math() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    runner = SRGNNBaseRunner(config, n_node=20)
    model = runner.build_model(
        build_srgnn_opt_from_train_config(config.attack.poison_model.params["train"])
    )
    device = model.embedding.weight.device
    hidden = torch.randn(2, 4, model.hidden_size, device=device)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], device=device)
    representation = model.compute_session_representation(hidden, mask)
    last_hidden = hidden[
        torch.arange(2, dtype=torch.long, device=device),
        torch.tensor([2, 1], dtype=torch.long, device=device),
    ]
    q1 = model.linear_one(last_hidden).view(2, 1, model.hidden_size)
    q2 = model.linear_two(hidden)
    alpha = model.linear_three(torch.sigmoid(q1 + q2))
    expected_representation = torch.sum(alpha * hidden * mask.unsqueeze(-1).float(), 1)
    if not model.nonhybrid:
        expected_representation = model.linear_transform(
            torch.cat([expected_representation, last_hidden], 1)
        )
    expected_scores = torch.matmul(representation, model.embedding.weight[1:].transpose(1, 0))
    assert torch.allclose(representation, expected_representation)
    assert torch.allclose(model.compute_scores(hidden, mask), expected_scores)


def test_srgnn_representation_rejects_mismatched_mask_shape() -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    runner = SRGNNBaseRunner(config, n_node=20)
    model = runner.build_model(
        build_srgnn_opt_from_train_config(config.attack.poison_model.params["train"])
    )
    device = model.embedding.weight.device
    hidden = torch.randn(2, 3, model.hidden_size, device=device)
    mask = torch.ones(2, 4, dtype=torch.long, device=device)
    with pytest.raises(ValueError, match="share batch and sequence dimensions"):
        model.compute_session_representation(hidden, mask)


def test_srgnn_mask_validation_rejects_empty_sessions_before_cuda_transfer() -> None:
    with pytest.raises(ValueError, match="empty session"):
        validate_session_mask_array([[0, 0], [1, 1]])


def test_srgnn_adapter_batch_target_scores_match_score_session() -> None:
    from attack.creat.srgnn_adapter import SRGNNRepresentationAdapter

    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    runner = SRGNNBaseRunner(config, n_node=20)
    runner.build_model(build_srgnn_opt_from_train_config(config.attack.poison_model.params["train"]))
    adapter = SRGNNRepresentationAdapter(runner)
    prefixes = [[1], [1, 2], [3, 4, 5]]
    target_item = 7
    batched = adapter.target_scores_for_prefixes(prefixes, target_item)
    expected = [
        float(runner.score_session(prefix)[target_item - 1].item())
        for prefix in prefixes
    ]
    assert np.allclose(batched, expected)
