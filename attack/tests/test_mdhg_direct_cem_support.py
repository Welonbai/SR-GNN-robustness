from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
import sys
from types import ModuleType

import pytest

from attack.common.config import load_config
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.data.session_stats import compute_session_stats
from attack.inner_train.mdhg_full_retrain_fixed_last import (
    MDHGFullRetrainFixedLastInnerTrainer,
)
from attack.models.mdhg_constants import (
    MDHG_ADAPTER_VERSION,
    MDHG_TRAIN_DATA_CONSTRUCTION_MODE,
)
from attack.models.mdhg_core import (
    MDHGInProcessModel,
    _import_mdhg_module,
    build_mdhg_train_rows,
)
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs.run_pts_construction_cem import (
    PTS_CEM_SURROGATE_RETRAIN_VALIDATION_BEST,
    _build_candidate_evaluator_context,
    _direct_cem_model_metadata,
    _validate_pts_construction_run_config,
    build_pts_cem_shared_cache_identity,
    pts_cem_surrogate_seed_alignment_metadata,
)
from attack.surrogate.mdhg_backend import (
    MDHGBackend,
    coerce_mdhg_poisoned_train_data,
)


BASE_CONFIG = (
    "attack/configs/"
    "ssh_diginetica_valbest_attack_ptscem_direct_freqrec_generated_popular_all_victims.yaml"
)


def _tiny_mdhg_train(**overrides):
    values = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 0.001,
        "checkpoint_protocol": "fixed_epoch",
        "validation_enabled": False,
        "export_model": "last",
    }
    values.update(overrides)
    return values


def _with_mdhg_generated_surrogate(config):
    train = _tiny_mdhg_train()
    pts = config.attack.pts_construction
    runtime = dict(config.victims.runtime or {})
    runtime["mdhg"] = {
        **dict(runtime.get("mdhg", {})),
        "device": {"use_gpu": True, "gpu_id": "0"},
    }
    params = dict(config.victims.params)
    params["mdhg"] = {"train": train}
    return replace(
        config,
        attack=replace(
            config.attack,
            poison_model=replace(
                config.attack.poison_model,
                name="mdhg",
                params={"train": train},
            ),
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_model=replace(
                        pts.cem.surrogate_model,
                        name="mdhg",
                        params={"train": train},
                    ),
                ),
            ),
        ),
        victims=replace(
            config.victims,
            enabled=["mdhg"],
            params=params,
            runtime=runtime,
        ),
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


def test_mdhg_loader_isolated_from_existing_top_level_model_and_util(monkeypatch) -> None:
    fake_model = ModuleType("model")
    fake_model.__file__ = str(
        Path("third_party/freqrec/src/model/__init__.py").resolve()
    )
    fake_util = ModuleType("util")
    fake_util.__file__ = str(Path("third_party/freqrec/src/util.py").resolve())
    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "util", fake_util)

    aliases = ("_sbr_mdhg_model", "_sbr_mdhg_util")
    sentinel = object()
    previous_aliases = {
        alias: sys.modules.get(alias, sentinel)
        for alias in aliases
    }
    for alias, previous in previous_aliases.items():
        if previous is sentinel:
            sys.modules.pop(alias, None)

    try:
        model_module = _import_mdhg_module("model")
        util_module = _import_mdhg_module("util")

        repo_root = Path(__file__).resolve().parents[2]
        assert Path(model_module.__file__).resolve() == (
            repo_root / "third_party" / "mdhg" / "model.py"
        ).resolve()
        assert Path(util_module.__file__).resolve() == (
            repo_root / "third_party" / "mdhg" / "util.py"
        ).resolve()
        assert sys.modules["model"] is fake_model
        assert sys.modules["util"] is fake_util
        assert sys.modules["_sbr_mdhg_model"] is model_module
        assert sys.modules["_sbr_mdhg_util"] is util_module
        assert _import_mdhg_module("model") is model_module
        assert _import_mdhg_module("util") is util_module
        with pytest.raises(ValueError, match="Unsupported MDHG module"):
            _import_mdhg_module("not_supported")
    finally:
        for alias, previous in previous_aliases.items():
            if previous is sentinel:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = previous


def test_mdhg_generated_direct_cem_config_validates_and_dispatches() -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))

    _validate_pts_construction_run_config(config)
    context = _build_candidate_evaluator_context(
        config,
        _tiny_shared_artifacts(),
        target_item=3,
    )

    assert config.attack.poison_model.name == "mdhg"
    assert config.attack.pts_construction.cem.surrogate_model.name == "mdhg"
    assert isinstance(context["backend"], MDHGBackend)
    assert isinstance(context["inner_trainer"], MDHGFullRetrainFixedLastInnerTrainer)
    assert context["cem_surrogate_model"] == "mdhg"
    assert context["mdhg_adapter_version"] == MDHG_ADAPTER_VERSION
    assert (
        context["mdhg_train_data_construction_mode"]
        == MDHG_TRAIN_DATA_CONSTRUCTION_MODE
    )


def test_mdhg_direct_cem_rejects_unsupported_retrain_protocol() -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))
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


def test_mdhg_generated_direct_cem_rejects_non_mdhg_poison_model() -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))
    bad = replace(
        config,
        attack=replace(
            config.attack,
            poison_model=replace(
                config.attack.poison_model,
                name="freqrec",
                params=load_config(BASE_CONFIG).attack.poison_model.params,
            ),
        ),
    )

    with pytest.raises(ValueError, match="poison_model.name == 'mdhg'"):
        _validate_pts_construction_run_config(bad)


def test_mdhg_direct_cem_rejects_seed_mismatch() -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))
    bad = replace(
        config,
        seeds=replace(
            config.seeds,
            surrogate_train_seed=int(config.seeds.victim_train_seed) + 1,
        ),
    )

    with pytest.raises(ValueError, match="surrogate_train_seed"):
        _validate_pts_construction_run_config(bad)


def test_mdhg_surrogate_seed_matches_resolved_mdhg_victim_seed() -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))
    metadata = pts_cem_surrogate_seed_alignment_metadata(config, target_item=123)

    assert metadata["cem_surrogate_model"] == "mdhg"
    assert (
        metadata["resolved_surrogate_effective_seed"]
        == metadata["resolved_victim_effective_seed"]
    )
    assert metadata["cem_surrogate_seed"] == metadata["resolved_victim_effective_seed"]


def test_mdhg_training_rows_are_candidate_specific_and_preserve_duplicates() -> None:
    shared = _tiny_shared_artifacts()
    candidate_a = [[1, 4, 5], [1, 4, 5]]
    candidate_b = [[2, 4, 5]]
    poisoned_a = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        candidate_a,
    )
    poisoned_b = build_poisoned_dataset(
        shared.clean_sessions,
        shared.clean_labels,
        candidate_b,
    )
    sessions_a, labels_a, raw_a, _fake_a = coerce_mdhg_poisoned_train_data(
        poisoned_a,
        clean_train_raw_sessions=shared.canonical_dataset.train_sub,
    )
    sessions_b, labels_b, raw_b, _fake_b = coerce_mdhg_poisoned_train_data(
        poisoned_b,
        clean_train_raw_sessions=shared.canonical_dataset.train_sub,
    )

    rows_a = build_mdhg_train_rows(
        sessions_a,
        labels_a,
        raw_train_sessions=raw_a,
        item_count=5,
    )
    rows_b = build_mdhg_train_rows(
        sessions_b,
        labels_b,
        raw_train_sessions=raw_b,
        item_count=5,
    )

    assert rows_a.row_fingerprint != rows_b.row_fingerprint
    assert rows_a.raw_session_fingerprint != rows_b.raw_session_fingerprint
    assert rows_a.row_fingerprint.count(((1, 4), 5)) == 2
    assert rows_a.raw_session_fingerprint.count((1, 4, 5)) == 2


def test_mdhg_score_session_rejects_empty_and_padding_item() -> None:
    model = MDHGInProcessModel(
        train_config=_tiny_mdhg_train(),
        item_count=5,
        seed=7,
        dataset_name="toy",
        use_gpu=False,
    )
    model.model = object()

    with pytest.raises(ValueError, match="empty"):
        model.score_session([])
    with pytest.raises(ValueError, match="outside canonical item range"):
        model.score_session([0])
    with pytest.raises(ValueError, match="outside canonical item range"):
        model.score_session([6])


def test_mdhg_cache_identity_and_metadata_distinguish_generator_and_surrogate(tmp_path) -> None:
    config = _with_mdhg_generated_surrogate(load_config(BASE_CONFIG))
    fake_sessions = tmp_path / "fake_sessions.pkl"
    poison_checkpoint = tmp_path / "poison.pt"
    with fake_sessions.open("wb") as handle:
        pickle.dump([[1, 2, 3]], handle)
    poison_checkpoint.write_bytes(b"mdhg poison checkpoint")

    identity = build_pts_cem_shared_cache_identity(
        config,
        target_item=123,
        fake_sessions_path=fake_sessions,
        poison_model_path=poison_checkpoint,
    )
    metadata = _direct_cem_model_metadata(config, target_item=123)

    assert identity["fake_session_source"]["generator_model"] == "mdhg"
    assert identity["direct_action_suffix_generator_model"] == "mdhg"
    assert identity["surrogate_reward"]["surrogate_model"] == "mdhg"
    assert identity["surrogate_reward"]["mdhg_adapter_version"] == MDHG_ADAPTER_VERSION
    assert (
        identity["surrogate_reward"]["mdhg_train_data_construction_mode"]
        == MDHG_TRAIN_DATA_CONSTRUCTION_MODE
    )
    assert identity["poison_model"]["mdhg_adapter_version"] == MDHG_ADAPTER_VERSION
    assert identity["poison_model"]["configured_seed"] == config.seeds.victim_train_seed
    assert "final_victim_model" not in identity
    assert "final_victim_seed" not in identity

    assert metadata["fake_session_generator_model"] == "mdhg"
    assert metadata["direct_action_suffix_generator_model"] == "mdhg"
    assert metadata["cem_surrogate_model"] == "mdhg"
    assert metadata["fake_session_generator_seed"] == config.seeds.victim_train_seed
    assert metadata["direct_action_suffix_generator_seed"] == config.seeds.victim_train_seed
    assert metadata["mdhg_adapter_version"] == MDHG_ADAPTER_VERSION


def test_mdhg_direct_cem_formal_ssh_configs_load_and_validate() -> None:
    paths = [
        "attack/configs/ssh_diginetica_valbest_attack_ptscem_direct_mdhg_generated_popular_all_victims.yaml",
        "attack/configs/ssh_diginetica_valbest_attack_ptscem_direct_mdhg_generated_unpopular_all_victims.yaml",
        "attack/configs/ssh_diginetica_valbest_attack_ptscem_direct_mdhg_surrogate_copy_source_popular_all_victims.yaml",
        "attack/configs/ssh_diginetica_valbest_attack_ptscem_direct_mdhg_surrogate_copy_source_unpopular_all_victims.yaml",
        "attack/configs/ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_generated_popular_all_victims.yaml",
        "attack/configs/ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_generated_unpopular_all_victims.yaml",
        "attack/configs/ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_surrogate_copy_source_popular_all_victims.yaml",
        "attack/configs/ssh_yoochoose1_64_valbest_attack_ptscem_direct_mdhg_surrogate_copy_source_unpopular_all_victims.yaml",
    ]
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        assert "D:\\" not in text
        assert "C:\\" not in text
        config = load_config(path)
        _validate_pts_construction_run_config(config)
        assert config.attack.poison_model.name == "mdhg"
        assert config.attack.pts_construction.cem.surrogate_model.name == "mdhg"
        assert tuple(config.victims.enabled) == (
            "freqrec",
            "srgnn",
            "miasrec",
            "tron",
            "mdhg",
            "wearec",
        )
        runtime = config.victims.runtime or {}
        for victim_runtime in runtime.values():
            device = victim_runtime.get("device")
            if isinstance(device, dict):
                assert str(device.get("gpu_id")) == "0"
