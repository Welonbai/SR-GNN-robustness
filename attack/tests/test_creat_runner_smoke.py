from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import ArtifactsConfig, load_config
from attack.creat.candidates import sessions_sha1, valid_position_mask
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs import run_creat_additive_sbr as creat_runner


class _FakeAdapter:
    embedding_dim = 2
    max_item_id = 20

    def __init__(self, runner):
        self.runner = runner

    def encode_session(self, session):
        return torch.ones(2)

    def item_embeddings(self, session):
        return torch.ones(len(session), 2)

    def valid_position_mask(
        self,
        session,
        target_item,
        topk_ratio,
        nonzero_when_possible=True,
    ):
        return torch.tensor(
            [
                is_valid and int(item) != int(target_item)
                for is_valid, item in zip(
                    valid_position_mask(
                        len(session),
                        topk_ratio,
                        nonzero_when_possible=nonzero_when_possible,
                    ),
                    session,
                )
            ]
        )


class _FakeMasker:
    def eval(self):
        return self

    def __call__(self, session_rep, item_embeddings, valid_mask):
        logits = torch.arange(len(valid_mask), dtype=torch.float32)
        return logits.masked_fill(~valid_mask, -1.0e38)


class _FakeTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def train(self, *, target_item, template_sessions):
        assert template_sessions == [[1, 2], [3, 4, 5]]
        return SimpleNamespace(
            masker=_FakeMasker(),
            history={
                "target_item": int(target_item),
                "epochs": [
                    {
                        "attack_reward": 1.0,
                        "stealth_reward": 0.0,
                        "local_reward": 0.0,
                        "entropy": 0.0,
                        "total_reward": 1.0,
                    }
                ],
            },
        )


def test_creat_runner_filters_after_hashing_and_calls_orchestrator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_creat_additive_sbr_ratio1_sample10.yaml"
    )
    config = replace(
        config,
        artifacts=ArtifactsConfig(
            root=str(tmp_path),
            shared_dir="shared",
            runs_dir="runs",
            cleanup_victim_intermediates=False,
        ),
    )
    templates = [[9], [1, 2], [3, 4, 5]]
    attack_shared_dir = tmp_path / "shared_attack"
    attack_shared_dir.mkdir(parents=True)
    with (attack_shared_dir / "fake_session_source_summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump({"template_sessions_sha1": sessions_sha1(templates)}, handle)
    shared = SharedAttackArtifacts(
        stats=SimpleNamespace(item_counts={item: 1 for item in range(1, 20)}),
        clean_sessions=[[1], [1, 2]],
        clean_labels=[2, 3],
        canonical_dataset=SimpleNamespace(),
        export_paths={},
        template_sessions=templates,
        poison_runner=SimpleNamespace(model=object()),
        fake_session_count=len(templates),
        shared_paths={
            "fake_sessions": tmp_path / "fake_sessions.pkl",
            "attack_shared_dir": attack_shared_dir,
        },
    )
    monkeypatch.setattr(creat_runner, "prepare_shared_attack_artifacts", lambda *args, **kwargs: shared)
    monkeypatch.setattr(creat_runner, "SRGNNRepresentationAdapter", _FakeAdapter)
    monkeypatch.setattr(creat_runner, "CreatAdditiveSBRTrainer", _FakeTrainer)

    captured = {}

    def fake_run_targets_and_victims(*args, **kwargs):
        output = kwargs["build_poisoned"](11)
        captured["metadata"] = output.metadata
        captured["poisoned"] = output.poisoned
        return {"status": "ok"}

    monkeypatch.setattr(creat_runner, "run_targets_and_victims", fake_run_targets_and_victims)

    summary = creat_runner.run_creat_additive_sbr(config, config_path=None)

    assert summary == {"status": "ok"}
    metadata = captured["metadata"]
    assert metadata["base_template_hash"] == sessions_sha1(templates)
    assert metadata["shared_template_sessions_sha1"] == sessions_sha1(templates)
    assert metadata["effective_template_hash"] == sessions_sha1([[1, 2], [3, 4, 5]])
    assert metadata["original_template_count"] == 3
    assert metadata["filtered_template_count"] == 1
    assert metadata["filtered_no_valid_candidate_count"] == 0
    assert metadata["effective_poisoned_copied_session_count"] == 2
    assert metadata["effective_budget_ratio"] == 2 / 3
    assert metadata["expanded_poisoned_prefix_label_pair_count"] == 3
    assert metadata["target_label_poisoned_pair_count"] == 2
    assert metadata["selected_replacement_target_pair_count"] == 2
    assert metadata["expanded_target_label_pair_count"] == 2
    assert metadata["pre_existing_target_session_count"] == 0
    assert metadata["pre_existing_target_item_count"] == 0
    assert metadata["pre_existing_target_label_pair_count"] == 0
    assert metadata["post_poison_target_label_pair_count"] == 2
    assert metadata["new_target_label_pair_count"] == 2
    assert metadata["candidate_reward_stats"] is None
    assert metadata["selected_reward_stats"] is None
    assert metadata["candidate_composed_reward_stats"] is None
    assert metadata["selected_composed_reward_stats"] is None
    assert "position_entropy" in metadata
    assert metadata["creat_fidelity"]["variant"] == "v1"
    assert captured["poisoned"].clean_count == 2
    assert captured["poisoned"].fake_count == 2


def test_creat_runner_requires_train_template_source(tmp_path: Path) -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10.yaml"
    )
    config = replace(
        config,
        artifacts=ArtifactsConfig(
            root=str(tmp_path),
            shared_dir="shared",
            runs_dir="runs",
            cleanup_victim_intermediates=False,
        ),
    )
    try:
        creat_runner.run_creat_additive_sbr(config, config_path=None)
    except ValueError as exc:
        assert "train_template_clean_exact_length_matched" in str(exc)
    else:
        raise AssertionError("Expected CREAT runner to reject generated fake-session source.")
