from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from attack.common.config import load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
    attack_key_payload,
    classify_victim_training_run_type,
    shared_attack_identity_requires_poison_runner,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.runs import run_pts_direct_action_uniform as uniform_runner
from attack.pts.direct_action_policy import (
    DIRECT_ACTION_MLP_H2_PARAMETER_NAMES,
    DIRECT_ACTION_POLICY_MLP_H2,
    enumerate_valid_direct_actions,
    score_direct_action,
    stable_softmax,
)
from attack.pts.executor import PTSConstructionBatchResult


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "attack" / "configs"
UNIFORM_CONFIGS = tuple(sorted(CONFIG_DIR.glob("*ptsuniform*budget0p01*all_victims.yaml")))


@pytest.mark.parametrize("config_path", UNIFORM_CONFIGS)
def test_formal_uniform_configs_match_main_table_scope(config_path: Path) -> None:
    config = load_config(config_path)
    pts = config.attack.pts_construction

    assert pts is not None
    assert pts.enabled is True
    assert pts.method == "direct_action_mlp_uniform"
    assert config.attack.size == pytest.approx(0.01)
    assert config.targets.count == 10
    assert config.targets.reuse_saved_targets is True
    assert config.targets.bucket in {"popular", "unpopular"}
    assert config.data.dataset_name in {"yoochoose1_64", "diginetica"}
    assert config.victims.enabled == (
        "freqrec",
        "srgnn",
        "miasrec",
        "tron",
        "mdhg",
        "wearec",
    )
    assert pts.artifacts.save_cem_trace is False
    assert pts.artifacts.save_candidate_sessions is False
    assert pts.artifacts.save_top_candidate_sessions is False


@pytest.mark.parametrize("residual_suffix_len", [1, 2, 5, 11])
def test_zero_mlp_policy_is_uniform_over_all_valid_atomic_actions(
    residual_suffix_len: int,
) -> None:
    actions = enumerate_valid_direct_actions(residual_suffix_len)
    theta = [0.0] * len(DIRECT_ACTION_MLP_H2_PARAMETER_NAMES)
    scores = [
        score_direct_action(
            policy_variant=DIRECT_ACTION_POLICY_MLP_H2,
            theta=theta,
            action=action,
            residual_suffix_len=residual_suffix_len,
            length_feature_mode="z_score",
            mean_residual_suffix_len=float(residual_suffix_len),
            std_residual_suffix_len=1.0,
        )
        for action in actions
    ]
    probabilities = stable_softmax(scores)

    assert scores == [0.0] * (2 * residual_suffix_len + 1)
    assert probabilities == pytest.approx(
        [1.0 / (2 * residual_suffix_len + 1)] * (2 * residual_suffix_len + 1)
    )


def test_uniform_identity_is_fixed_and_classified_as_poisoned() -> None:
    config = load_config(UNIFORM_CONFIGS[0])
    payload = attack_key_payload(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE,
    )
    pts_payload = payload["attack"]["pts_construction"]

    assert classify_victim_training_run_type(
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE
    ) == "poisoned"
    assert shared_attack_identity_requires_poison_runner(
        PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE
    )
    assert pts_payload["method"] == "direct_action_mlp_uniform"
    assert pts_payload["direct_action_policy"]["fixed_policy"] == {
        "mode": "zero_logits_uniform_atomic_actions",
        "parameter_vector": "all_zero",
    }
    assert "cem" not in pts_payload
    assert "reward" not in pts_payload
    assert "final_selection" not in pts_payload


def test_uniform_validation_allows_preexisting_target_at_session_start() -> None:
    uniform_runner._validate_constructed_sessions(
        sessions=[[50, 50, 2]],
        session_contexts=[SimpleNamespace(anchor_position=1)],
        target_item=50,
        expected_count=1,
        max_item=100,
    )


def test_uniform_validation_requires_inserted_target_at_internal_anchor() -> None:
    with pytest.raises(ValueError, match="inserted target at its internal anchor"):
        uniform_runner._validate_constructed_sessions(
            sessions=[[50, 2, 3]],
            session_contexts=[SimpleNamespace(anchor_position=1)],
            target_item=50,
            expected_count=1,
            max_item=100,
        )


def test_runner_materializes_one_zero_policy_batch_without_cem_or_surrogate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path = REPO_ROOT / "outputs" / f"tmp_pts_uniform_test_{uuid4().hex}"
    tmp_path.mkdir(parents=True)
    config = load_config(UNIFORM_CONFIGS[0])
    stats = compute_session_stats([[1, 2, 99], [4, 5, 98]])
    shared = SimpleNamespace(
        stats=stats,
        clean_sessions=[[1], [4]],
        clean_labels=[2, 5],
        canonical_dataset=CanonicalDataset(
            train_sub=[[1, 2, 99], [4, 5, 98]],
            valid=[],
            test=[],
            item_map={},
            metadata={},
        ),
        export_paths={},
        template_sessions=[[1, 2, 3], [4, 5, 6]],
        poison_runner=object(),
        fake_session_count=2,
        shared_paths={"fake_sessions": tmp_path / "fake_sessions.pkl"},
    )
    captured: dict[str, object] = {"construction_calls": 0}

    def fake_prepare(_config, **kwargs):
        captured["prepare"] = kwargs
        return shared

    def fake_apply(**kwargs):
        captured["construction_calls"] = int(captured["construction_calls"]) + 1
        captured["policy_vector"] = kwargs["policy"].to_vector()
        captured["candidate_key"] = kwargs["candidate_key"]
        captured["iteration"] = kwargs["iteration"]
        captured["poison_runner"] = kwargs["poison_runner"]
        final_sessions = [
            [*context.prefix, 50, *context.residual_suffix]
            for context in kwargs["session_contexts"]
        ]
        captured["final_sessions"] = final_sessions
        return PTSConstructionBatchResult(
            final_sessions=final_sessions,
            per_session_records=[{"index": 0}, {"index": 1}],
            summary={"session_count": 2, "action_counts": {"stop": 2}},
        )

    def fake_run_targets(_config, **kwargs):
        captured["run_type"] = kwargs["run_type"]
        output = kwargs["build_poisoned"](50)
        captured["output"] = output
        return {"status": "completed", "targets": [50]}

    monkeypatch.setattr(uniform_runner, "prepare_shared_attack_artifacts", fake_prepare)
    monkeypatch.setattr(
        uniform_runner,
        "apply_pts_direct_action_construction_batch",
        fake_apply,
    )
    monkeypatch.setattr(uniform_runner, "run_targets_and_victims", fake_run_targets)
    monkeypatch.setattr(
        uniform_runner,
        "target_dir",
        lambda *_args, **_kwargs: tmp_path / "target_50",
    )

    result = uniform_runner.run_pts_direct_action_uniform(config)

    assert result == {"status": "completed", "targets": [50]}
    assert captured["prepare"]["require_poison_runner"] is True
    assert captured["construction_calls"] == 1
    assert captured["policy_vector"] == [0.0] * len(
        DIRECT_ACTION_MLP_H2_PARAMETER_NAMES
    )
    assert captured["candidate_key"] == "fixed_zero_policy"
    assert captured["iteration"] == 0
    assert captured["poison_runner"] is shared.poison_runner
    assert captured["run_type"] == PTS_CONSTRUCTION_DIRECT_ACTION_MLP_UNIFORM_RUN_TYPE

    output = captured["output"]
    assert output.raw_fake_sessions == captured["final_sessions"]
    assert output.poisoned.fake_count == 2
    assert output.metadata["pts_uniform_cem_enabled"] is False
    assert output.metadata["pts_uniform_surrogate_evaluation_count"] == 0

    artifact_dir = tmp_path / "target_50" / "pts_direct_action_uniform"
    marker = json.loads(
        (artifact_dir / "construction_complete.json").read_text(encoding="utf-8")
    )
    policy = json.loads((artifact_dir / "policy.json").read_text(encoding="utf-8"))
    assert marker["cem_enabled"] is False
    assert marker["surrogate_evaluation_count"] == 0
    assert marker["parameter_vector"] == [0.0] * len(
        DIRECT_ACTION_MLP_H2_PARAMETER_NAMES
    )
    assert policy["mode"] == "zero_logits_uniform_atomic_actions"
    assert policy["parameter_vector"] == marker["parameter_vector"]
    assert not (artifact_dir / "session_records.jsonl").exists()
