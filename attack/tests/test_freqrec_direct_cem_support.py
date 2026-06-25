from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from attack.common.config import load_config
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.session_stats import compute_session_stats
from attack.inner_train.freqrec_full_retrain_fixed_last import (
    FreqRecFullRetrainFixedLastInnerTrainer,
)
from attack.models.freqrec_core import (
    FREQREC_ADAPTER_VERSION,
    FREQREC_TRAIN_DATA_CONSTRUCTION_MODE,
    FreqRecInProcessModel,
    _resolve_freqrec_device,
    build_freqrec_train_rows,
)
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs.run_pts_construction_cem import (
    PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST,
    _build_candidate_evaluator_context,
    _direct_cem_model_metadata,
    _evaluate_candidate_retrain_validation_reward,
    _validate_pts_construction_run_config,
    build_pts_cem_shared_cache_identity,
    pts_cem_surrogate_seed_alignment_metadata,
    run_pts_construction_grouped_cem,
)
from attack.surrogate.freqrec_backend import FreqRecBackend, FreqRecModelHandle
from attack.tests.freqrec_test_utils import freqrec_train


COPY_CONFIG = "attack/configs/diginetica_valbest_attack_ptscem_direct_freqrec_surrogate_copy_source.yaml"
GENERATED_CONFIG = "attack/configs/diginetica_valbest_attack_ptscem_direct_freqrec_generated.yaml"
LEGACY_CONFIG = "attack/configs/ssh_diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample_unpopular_copy_train_freqrec_only.yaml"


def _tiny_train(**overrides):
    values = dict(
        epochs=1,
        batch_size=2,
        max_seq_length=3,
        hidden_size=4,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    values.update(overrides)
    return freqrec_train(**values)


def _with_tiny_freqrec_surrogate(config):
    pts = config.attack.pts_construction
    runtime = dict(config.victims.runtime or {})
    runtime["freqrec"] = {
        **dict(runtime.get("freqrec", {})),
        "device": {"use_gpu": False, "gpu_id": "0"},
        "dataloader": {"num_workers": 0},
    }
    return replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_model=replace(
                        pts.cem.surrogate_model,
                        params={"train": _tiny_train()},
                    ),
                ),
            ),
        ),
        victims=replace(config.victims, runtime=runtime),
    )


def _tiny_shared_artifacts() -> SharedAttackArtifacts:
    canonical = CanonicalDataset(
        train_sub=[[1, 2, 3], [2, 3, 4]],
        valid=[[1, 2, 3], [2, 3, 4]],
        test=[],
        item_map={item: item for item in range(1, 6)},
        metadata={"item_count": 5},
    )
    return SharedAttackArtifacts(
        stats=compute_session_stats(canonical.train_sub),
        clean_sessions=[[1], [1, 2], [2], [2, 3]],
        clean_labels=[2, 3, 3, 4],
        canonical_dataset=canonical,
        export_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=0,
        shared_paths={},
    )


def test_freqrec_score_session_contract() -> None:
    model = FreqRecInProcessModel(
        train_config=_tiny_train(),
        item_count=5,
        seed=7,
        use_gpu=False,
        num_workers=0,
    )
    scores = model.score_session([1, 2, 3])
    assert tuple(scores.shape) == (5,)
    assert scores.isfinite().all().item()

    overlong_scores = model.score_session([1, 2, 3, 4, 5])
    suffix_scores = model.score_session([3, 4, 5])
    assert tuple(overlong_scores.shape) == (5,)
    assert overlong_scores.tolist() == pytest.approx(suffix_scores.tolist())

    with pytest.raises(ValueError, match="empty"):
        model.score_session([])
    with pytest.raises(ValueError, match="outside canonical item range"):
        model.score_session([0])
    with pytest.raises(ValueError, match="outside canonical item range"):
        model.score_session([6])


def test_freqrec_device_resolution_respects_gpu_id(monkeypatch) -> None:
    assert str(_resolve_freqrec_device(use_gpu=False, gpu_id="1")) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(_resolve_freqrec_device(use_gpu=True, gpu_id="1")) == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert str(_resolve_freqrec_device(use_gpu=True, gpu_id="1")) == "cuda:1"

    with pytest.raises(ValueError, match="gpu_id"):
        _resolve_freqrec_device(use_gpu=True, gpu_id="not-an-int")
    with pytest.raises(ValueError, match="unavailable"):
        _resolve_freqrec_device(use_gpu=True, gpu_id="2")


def test_freqrec_tiny_train_save_load_and_score(tmp_path) -> None:
    model = FreqRecInProcessModel(
        train_config=_tiny_train(epochs=1),
        item_count=5,
        seed=11,
        use_gpu=False,
        num_workers=0,
    )
    losses = model.train_pairs([[1], [1, 2], [2]], [2, 3, 4], epochs=1)
    assert len(losses) == 1
    assert losses[0] == pytest.approx(losses[0])

    checkpoint = tmp_path / "freqrec.pt"
    model.save_model(checkpoint)
    assert checkpoint.exists()

    loaded = FreqRecInProcessModel(
        train_config=_tiny_train(epochs=1),
        item_count=5,
        seed=11,
        use_gpu=False,
        num_workers=0,
    )
    loaded.load_model(checkpoint)
    scores = loaded.score_session([1, 2])
    assert tuple(scores.shape) == (5,)
    assert scores.isfinite().all().item()


def test_freqrec_train_rows_are_candidate_specific_and_preserve_duplicates() -> None:
    clean_sessions = [[1], [2]]
    clean_labels = [2, 3]
    candidate_a_sessions = clean_sessions + [[1, 2], [1, 2]]
    candidate_a_labels = clean_labels + [3, 3]
    candidate_b_sessions = clean_sessions + [[2, 3]]
    candidate_b_labels = clean_labels + [4]

    rows_a = build_freqrec_train_rows(candidate_a_sessions, candidate_a_labels, item_count=5)
    rows_b = build_freqrec_train_rows(candidate_b_sessions, candidate_b_labels, item_count=5)

    assert rows_a.row_fingerprint != rows_b.row_fingerprint
    assert rows_a.row_fingerprint.count(((1, 2), 3)) == 2


def test_freqrec_backend_score_gt_validates_lengths_and_item_ids() -> None:
    config = _with_tiny_freqrec_surrogate(load_config(COPY_CONFIG))
    backend = FreqRecBackend(config, train_config=_tiny_train(), item_count=5, seed=7)

    class DummyModel:
        def score_sessions_topk(self, sessions, *, topk):
            del topk
            return [[2] for _session in sessions]

    handle = FreqRecModelHandle(model=DummyModel())
    with pytest.raises(ValueError, match="one label per validation session"):
        backend.score_gt(handle, [[1], [2]], [2])
    with pytest.raises(ValueError, match="outside canonical item range"):
        backend.score_gt(handle, [[1]], [6])


def test_freqrec_direct_cem_configs_load_and_validate() -> None:
    copy_config = load_config(COPY_CONFIG)
    generated_config = load_config(GENERATED_CONFIG)

    _validate_pts_construction_run_config(copy_config)
    _validate_pts_construction_run_config(generated_config)

    assert copy_config.attack.pts_construction.cem.surrogate_model.name == "freqrec"
    assert generated_config.attack.poison_model.name == "freqrec"
    assert generated_config.attack.pts_construction.cem.surrogate_model.name == "freqrec"


def test_freqrec_direct_cem_rejects_seed_mismatch() -> None:
    config = load_config(COPY_CONFIG)
    bad = replace(
        config,
        seeds=replace(config.seeds, surrogate_train_seed=config.seeds.victim_train_seed + 1),
    )
    with pytest.raises(ValueError, match="surrogate_train_seed"):
        _validate_pts_construction_run_config(bad)


def test_freqrec_surrogate_seed_matches_resolved_freqrec_victim_seed() -> None:
    config = load_config(COPY_CONFIG)
    metadata = pts_cem_surrogate_seed_alignment_metadata(config, target_item=123)
    assert metadata["cem_surrogate_model"] == "freqrec"
    assert metadata["resolved_surrogate_effective_seed"] == metadata["resolved_victim_effective_seed"]
    assert metadata["cem_surrogate_seed"] == metadata["resolved_victim_effective_seed"]


def test_generated_freqrec_direct_cem_rejects_seed_mismatch() -> None:
    config = load_config(GENERATED_CONFIG)
    bad = replace(
        config,
        seeds=replace(config.seeds, surrogate_train_seed=config.seeds.victim_train_seed + 1),
    )
    with pytest.raises(ValueError, match="surrogate_train_seed"):
        _validate_pts_construction_run_config(bad)


def test_freqrec_direct_cem_rejects_unsupported_retrain_protocol() -> None:
    config = load_config(COPY_CONFIG)
    pts = config.attack.pts_construction
    bad = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=replace(
                        pts.cem.surrogate_retrain,
                        checkpoint_protocol=PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST,
                        validation_enabled=True,
                        reward_checkpoint="best",
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="fixed_last"):
        _validate_pts_construction_run_config(bad)


def test_legacy_and_explicit_srgnn_surrogate_configs_are_supported() -> None:
    legacy = load_config(LEGACY_CONFIG)
    _validate_pts_construction_run_config(legacy)
    assert legacy.attack.pts_construction.cem.surrogate_model.name == "srgnn"
    assert legacy.attack.pts_construction.cem.surrogate_model.params is None

    pts = legacy.attack.pts_construction
    explicit = replace(
        legacy,
        attack=replace(
            legacy.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_model=replace(
                        pts.cem.surrogate_model,
                        params={"train": dict(legacy.victims.params["srgnn"]["train"])},
                    ),
                ),
            ),
        ),
    )
    _validate_pts_construction_run_config(explicit)
    assert explicit.attack.pts_construction.cem.surrogate_model.params["train"]["epochs"] == 4


def test_build_candidate_evaluator_context_returns_freqrec_dispatch() -> None:
    config = _with_tiny_freqrec_surrogate(load_config(COPY_CONFIG))
    context = _build_candidate_evaluator_context(
        config,
        _tiny_shared_artifacts(),
        target_item=3,
    )

    assert isinstance(context, dict)
    assert isinstance(context["backend"], FreqRecBackend)
    assert isinstance(context["inner_trainer"], FreqRecFullRetrainFixedLastInnerTrainer)
    assert context["cem_surrogate_model"] == "freqrec"
    assert context["surrogate_model"] == "freqrec"
    assert context["surrogate_retrain_protocol"] == "fixed_last"
    assert context["freqrec_adapter_version"] == FREQREC_ADAPTER_VERSION
    assert context["seed_alignment"]["cem_surrogate_model"] == "freqrec"


def test_freqrec_candidate_evaluation_smoke_uses_candidate_training_data() -> None:
    config = _with_tiny_freqrec_surrogate(load_config(COPY_CONFIG))
    context = _build_candidate_evaluator_context(
        config,
        _tiny_shared_artifacts(),
        target_item=3,
    )
    trainer = context["inner_trainer"]
    assert isinstance(trainer, FreqRecFullRetrainFixedLastInnerTrainer)

    result = _evaluate_candidate_retrain_validation_reward(
        config=config,
        evaluator_context=context,
        candidate_sessions=[[1, 2, 3], [1, 2, 3]],
        target_item=3,
        iteration=0,
        population_size=1,
        candidate_id=0,
        candidate_seed=123,
    )

    assert result.reward == pytest.approx(result.reward)
    assert "targeted_mrr@10" in result.reward_metrics
    assert "targeted_recall@20" in result.reward_metrics
    assert trainer.last_train_rows_fingerprint is not None
    assert trainer.last_train_rows_fingerprint.count(((1, 2), 3)) >= 2


def test_freqrec_cache_identity_includes_copy_source_suffix_generator(tmp_path) -> None:
    copy_config = load_config(COPY_CONFIG)
    generated_config = load_config(GENERATED_CONFIG)
    fake_sessions = tmp_path / "fake_sessions.pkl"
    poison_checkpoint = tmp_path / "poison.pt"
    import pickle

    with fake_sessions.open("wb") as handle:
        pickle.dump([[1, 2, 3]], handle)
    poison_checkpoint.write_bytes(b"freqrec poison checkpoint")

    copy_identity = build_pts_cem_shared_cache_identity(
        copy_config,
        target_item=123,
        fake_sessions_path=fake_sessions,
        poison_model_path=poison_checkpoint,
    )
    generated_identity = build_pts_cem_shared_cache_identity(
        generated_config,
        target_item=123,
        fake_sessions_path=fake_sessions,
        poison_model_path=poison_checkpoint,
    )

    assert copy_identity["surrogate_reward"]["surrogate_model"] == "freqrec"
    assert copy_identity["fake_session_source"]["generator_model"] is None
    assert copy_identity["direct_action_suffix_generator_model"] == "freqrec"
    assert copy_identity["poison_model"]["role"] == "direct_action_suffix_generator"
    assert copy_identity["poison_model"]["freqrec_adapter_version"] == FREQREC_ADAPTER_VERSION
    assert "final_victim_model" not in copy_identity
    assert "final_victim_seed" not in copy_identity
    assert copy_identity["surrogate_reward"]["freqrec_adapter_version"] == FREQREC_ADAPTER_VERSION
    assert (
        copy_identity["surrogate_reward"]["freqrec_train_data_construction_mode"]
        == FREQREC_TRAIN_DATA_CONSTRUCTION_MODE
    )

    assert generated_identity["fake_session_source"]["generator_model"] == "freqrec"
    assert generated_identity["direct_action_suffix_generator_model"] == "freqrec"
    assert (
        generated_identity["poison_model"]["role"]
        == "base_fake_session_and_direct_action_suffix_generator"
    )
    assert generated_identity["poison_model"]["freqrec_adapter_version"] == FREQREC_ADAPTER_VERSION
    assert (
        generated_identity["poison_model"]["seed_source"]
        == "configured_victim_train_seed_shared_poison_model"
    )

    changed_poison_config = replace(
        copy_config,
        attack=replace(
            copy_config.attack,
            poison_model=replace(
                copy_config.attack.poison_model,
                params={
                    **copy_config.attack.poison_model.params,
                    "train": {
                        **copy_config.attack.poison_model.params["train"],
                        "epochs": int(copy_config.attack.poison_model.params["train"]["epochs"]) + 1,
                    },
                },
            ),
        ),
    )
    changed_poison_identity = build_pts_cem_shared_cache_identity(
        changed_poison_config,
        target_item=123,
        fake_sessions_path=fake_sessions,
        poison_model_path=poison_checkpoint,
    )
    assert copy_identity != changed_poison_identity

    other_target_identity = build_pts_cem_shared_cache_identity(
        copy_config,
        target_item=124,
        fake_sessions_path=fake_sessions,
        poison_model_path=poison_checkpoint,
    )
    assert copy_identity != other_target_identity


def test_direct_cem_model_metadata_distinguishes_source_generator_and_surrogate() -> None:
    copy_config = load_config(COPY_CONFIG)
    generated_config = load_config(GENERATED_CONFIG)

    copy_metadata = _direct_cem_model_metadata(copy_config, target_item=123)
    generated_metadata = _direct_cem_model_metadata(generated_config, target_item=123)

    assert copy_metadata["fake_session_generator_model"] is None
    assert copy_metadata["direct_action_suffix_generator_model"] == "freqrec"
    assert copy_metadata["direct_action_suffix_generator_seed"] == copy_config.seeds.victim_train_seed
    assert copy_metadata["cem_surrogate_model"] == "freqrec"
    assert "fake_session_generator_seed" not in copy_metadata

    assert generated_metadata["fake_session_generator_model"] == "freqrec"
    assert generated_metadata["direct_action_suffix_generator_model"] == "freqrec"
    assert generated_metadata["direct_action_suffix_generator_seed"] == generated_config.seeds.victim_train_seed
    assert generated_metadata["fake_session_generator_seed"] == generated_config.seeds.victim_train_seed
    assert (
        generated_metadata["fake_session_generator_seed_source"]
        == "configured_victim_train_seed_shared_poison_model"
    )
    assert generated_metadata["cem_surrogate_model"] == "freqrec"


def test_copy_source_direct_action_cem_requests_freqrec_suffix_poison_runner(monkeypatch) -> None:
    config = load_config(COPY_CONFIG)
    calls = {"prepare": 0}

    def fake_prepare_shared_attack_artifacts(config_arg, *, run_type, require_poison_runner, config_path=None):
        del config_arg, run_type, config_path
        calls["prepare"] += 1
        assert require_poison_runner is True
        raise RuntimeError("stop after Direct CEM poison-runner requirement check")

    monkeypatch.setattr(
        "attack.pipeline.runs.run_pts_construction_cem.prepare_shared_attack_artifacts",
        fake_prepare_shared_attack_artifacts,
    )

    with pytest.raises(RuntimeError, match="stop after Direct CEM poison-runner requirement check"):
        run_pts_construction_grouped_cem(config, config_path=Path(COPY_CONFIG))

    assert calls == {"prepare": 1}
