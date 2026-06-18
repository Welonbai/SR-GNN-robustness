from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import PoisoningSSLSBRConfig, load_config
from attack.common.paths import (
    POISONING_SSL_SBR_RUN_TYPE,
    attack_key_payload,
    classify_victim_training_run_type,
    shared_attack_artifact_key_payload,
    shared_attack_identity_requires_poison_runner,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.poisoning_ssl.diagnostics import (
    budget_diagnostics,
    duplicate_diagnostics,
    length_stats,
    nearest_rank_percentile,
    target_diagnostics,
    target_label_pair_count,
)
from attack.poisoning_ssl.dataset_bridge import export_pseudo_user_sequences
from attack.poisoning_ssl.generator import RealSeqPoisonCandidateGenerator, StaticCandidateGenerator
from attack.poisoning_ssl.pipeline import (
    compute_seqpoison_max_seq_len,
    generate_poisoning_ssl_sbr_target,
)
from attack.poisoning_ssl.postprocess import postprocess_fake_user_sequences


CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1.yaml"
)


def _toy_dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train_sub=[[1, 2, 3], [2, 9], [3, 4, 5, 9]],
        valid=[[1, 9]],
        test=[[2, 3]],
        item_map={str(item): item for item in [1, 2, 3, 4, 5, 9]},
        metadata={"dataset_name": "toy"},
    )


def _toy_shared(fake_session_count: int = 2):
    return SimpleNamespace(
        canonical_dataset=_toy_dataset(),
        stats=None,
        clean_sessions=[[1], [1, 2], [3]],
        clean_labels=[2, 3, 9],
        export_paths={},
        shared_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=int(fake_session_count),
    )


def _config(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(tmp_path / "outputs")),
        targets=replace(base.targets, mode="explicit_list", explicit_list=(9,), count=1),
        victims=replace(base.victims, enabled=("srgnn",)),
    )


def test_poisoning_ssl_sbr_config_parses_defaults_and_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    poisoning = config.attack.poisoning_ssl_sbr
    assert poisoning is not None
    assert poisoning.enabled is True
    assert poisoning.max_seq_len_policy == "train_sub_p99"
    assert poisoning.original_max_seq_len_cap == 50
    assert poisoning.max_seq_len_override is None
    assert poisoning.enforce_nonzero_target_position is False
    assert poisoning.candidate_multiplier == 1
    assert poisoning.max_generation_rounds == 10
    assert poisoning.generation_backend == "real"
    primitive = config.to_primitive()
    assert primitive["attack"]["poisoning_ssl_sbr"]["enabled"] is True
    assert primitive["attack"]["poisoning_ssl_sbr"]["generation_backend"] == "real"

    attack_payload = attack_key_payload(config, run_type=POISONING_SSL_SBR_RUN_TYPE)
    shared_payload = shared_attack_artifact_key_payload(
        config,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
    )
    assert "poison_model" not in attack_payload["attack"]
    assert "replacement_topk_ratio" not in attack_payload["attack"]
    assert "fake_session_source" not in attack_payload["attack"]
    assert "poison_model" not in shared_payload["attack_generation"]
    identity = attack_payload["attack"]["poisoning_ssl_sbr"]
    assert "enabled" not in identity
    assert "save_generated_candidates" not in identity
    assert "length_diagnostics" not in identity
    assert "reuse_existing_artifacts" not in identity
    assert identity["candidate_multiplier"] == 1
    assert identity["max_generation_rounds"] == 10
    assert identity["generation_backend"] == "real"
    assert classify_victim_training_run_type(POISONING_SSL_SBR_RUN_TYPE) == "poisoned"
    assert not shared_attack_identity_requires_poison_runner(POISONING_SSL_SBR_RUN_TYPE)


def test_poisoning_ssl_sbr_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="max_seq_len_policy"):
        PoisoningSSLSBRConfig(max_seq_len_policy="bad")
    with pytest.raises(ValueError, match="original_max_seq_len_cap"):
        PoisoningSSLSBRConfig(original_max_seq_len_cap=0)
    with pytest.raises(ValueError, match="max_seq_len_override"):
        PoisoningSSLSBRConfig(max_seq_len_override=0)
    with pytest.raises(ValueError, match="candidate_multiplier"):
        PoisoningSSLSBRConfig(candidate_multiplier=0)
    with pytest.raises(ValueError, match="max_generation_rounds"):
        PoisoningSSLSBRConfig(max_generation_rounds=0)
    with pytest.raises(ValueError, match="max_seq_len_override"):
        PoisoningSSLSBRConfig(max_seq_len_policy="fixed")
    with pytest.raises(ValueError, match="generation_backend"):
        PoisoningSSLSBRConfig(generation_backend="mock")
    with pytest.raises(ValueError, match="classifier_epochs"):
        PoisoningSSLSBRConfig(classifier_epochs=0)
    with pytest.raises(ValueError, match="max_train_sequences"):
        PoisoningSSLSBRConfig(max_train_sequences=0)
    with pytest.raises(ValueError, match="target_probability"):
        PoisoningSSLSBRConfig(target_probability=1.1)


def test_compute_seqpoison_max_seq_len_policies() -> None:
    sessions = [[1] * length for length in [2, 3, 4, 5, 100]]
    assert nearest_rank_percentile([2, 3, 4, 5, 100], 99) == 100
    assert compute_seqpoison_max_seq_len(sessions, PoisoningSSLSBRConfig()) == 50
    assert compute_seqpoison_max_seq_len(
        sessions,
        PoisoningSSLSBRConfig(max_seq_len_override=17),
    ) == 17
    assert compute_seqpoison_max_seq_len(
        sessions,
        PoisoningSSLSBRConfig(max_seq_len_policy="fixed", max_seq_len_override=13),
    ) == 13


def test_postprocess_filters_and_allows_pos0() -> None:
    result = postprocess_fake_user_sequences(
        [
            [100, 9, 1, 0, 0],
            [101, 1, 9],
            [102, 1, 2],
            [103, 9],
            [104, 1, 99, 9],
            [105, 9, 1, 9],
            [106, 2, 9],
        ],
        target_item=9,
        valid_item_ids={1, 2, 9},
        n_fake=2,
        enforce_single_target=True,
    )
    assert result.final_sessions == [[9, 1], [1, 9]]
    assert result.counts["no_target_count"] == 1
    assert result.counts["filtered_short_session_count"] == 1
    assert result.counts["invalid_item_count"] == 1
    assert result.counts["multi_target_count"] == 1
    assert result.counts["n_after_filtering"] == 3
    assert (
        result.counts["target_containing_candidate_count_before_single_target_filter"]
        == 4
    )
    assert result.counts[
        "target_containing_candidate_ratio_before_single_target_filter"
    ] == pytest.approx(4 / 7)
    diag = target_diagnostics(result.final_sessions, target_item=9)
    assert diag["target_pos0_count"] == 1
    assert diag["target_position_distribution"] == {0: 1, 1: 1}


def test_postprocess_auto_user_id_keeps_plain_sessions() -> None:
    result = postprocess_fake_user_sequences(
        [[9, 1], [1, 9]],
        target_item=9,
        valid_item_ids={1, 9},
        n_fake=2,
    )
    assert result.final_sessions == [[9, 1], [1, 9]]


def test_dataset_bridge_filters_too_long_sessions_and_records_metadata(tmp_path: Path) -> None:
    bundle = export_pseudo_user_sequences(
        [[10, 20], [20, 30, 40], [10, 20, 30, 40]],
        target_item=40,
        output_dir=tmp_path,
        valid_item_ids={10, 20, 30, 40},
        max_seq_len=3,
        max_train_sequences=1,
    )
    assert bundle.remap_used is True
    assert bundle.train_sequences == [[1, 2]]
    assert bundle.to_canonical_sequence([1, 2, 0]) == [10, 20, 0]
    assert bundle.metadata["train_session_count_before_length_filter"] == 3
    assert bundle.metadata["train_session_count_after_length_filter"] == 2
    assert bundle.metadata["excluded_train_session_count"] == 1
    assert bundle.metadata["excluded_train_session_ratio"] == pytest.approx(1 / 3)
    assert bundle.metadata["max_seq_len_value"] == 3
    assert bundle.metadata["diagnostic_max_train_sequences"] == 1
    assert bundle.metadata["train_sequence_count_used_for_training"] == 1
    assert bundle.item_id_mapping_path is not None


def test_diagnostics_budget_and_duplicates() -> None:
    sessions = [[9, 1], [1, 9, 2], [1, 9, 2]]
    stats = length_stats(sessions)
    assert stats["count"] == 3
    assert stats["length_count_by_length"] == {2: 1, 3: 2}
    assert stats["p50"] == 3
    assert target_label_pair_count(sessions, target_item=9) == 2
    budget = budget_diagnostics(sessions, target_item=9, clean_label_count=10)
    assert budget["expanded_pair_count_added"] == 5
    assert budget["effective_expanded_budget_ratio"] == 0.5
    assert budget["target_label_pair_count_added"] == 2
    assert budget["target_label_pair_ratio_added"] == 0.2
    duplicates = duplicate_diagnostics(sessions)
    assert duplicates["duplicate_session_count"] == 1
    assert duplicates["duplicate_session_ratio"] == pytest.approx(1 / 3)


def test_pipeline_with_explicit_mock_generator_writes_contract(tmp_path: Path) -> None:
    config = _config(tmp_path)
    generator = StaticCandidateGenerator(
        rounds=[
            [
                [100, 9, 1],
                [101, 1, 9],
                [102, 1, 2],
            ]
        ]
    )
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=generator,
    )
    assert result.raw_fake_sessions == [[9, 1], [1, 9]]
    assert result.metadata["n_final_injected"] == 2
    assert result.metadata["n_generated_candidates"] == 3
    assert result.metadata["target_pos0_count"] == 1
    assert result.metadata["target_label_pair_count_added"] == 1
    assert result.metadata["phase1_interface_mock_only"] is False
    assert result.metadata["real_generation_implemented"] is False
    assert (
        result.metadata["target_containing_candidate_count_before_single_target_filter"]
        == 2
    )
    target_root = (
        Path(config.artifacts.root)
        / config.artifacts.runs_dir
        / config.data.dataset_name
        / config.experiment.name
    )
    assert list(target_root.rglob("poisoning_ssl_sbr_metadata.json"))


def test_pipeline_without_generator_selects_real_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    called = {"value": False}

    class FakeRealGenerator:
        last_metadata = {
            "generation_backend": "real",
            "real_generation_implemented": True,
        }

        def generate(self, request):
            called["value"] = True
            assert request.dataset_bundle is not None
            assert request.config.generation_backend == "real"
            return [[100, 9, 1], [101, 1, 9]]

    import attack.poisoning_ssl.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "RealSeqPoisonCandidateGenerator",
        FakeRealGenerator,
    )
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
    )
    assert called["value"] is True
    assert result.metadata["real_generation_implemented"] is True


def test_real_generator_prepends_synthetic_user_id(tmp_path: Path) -> None:
    bundle = export_pseudo_user_sequences(
        [[1, 9], [2, 9]],
        target_item=9,
        output_dir=tmp_path / "bridge",
        valid_item_ids={1, 2, 9},
        max_seq_len=3,
    )

    class FakeSampleGenerator:
        def sample(self, n, *, device):
            assert n == 2
            return torch.tensor([[1, 3, 0], [2, 3, 0]], dtype=torch.long)

    class FakeTrainer:
        def train_or_load(self, **kwargs):
            return SimpleNamespace(
                classifier_checkpoint_path=tmp_path / "classifier.pt",
                generator_checkpoint_path=tmp_path / "generator.pt",
                discriminator_checkpoint_path=tmp_path / "discriminator.pt",
                training_log_path=tmp_path / "training_log.json",
                generation_log_path=tmp_path / "generation_log.json",
                metadata={
                    "enabled_reward_components": [
                        "target_related_reward",
                        "bi_classifier_reward",
                        "gan_discriminator_reward",
                    ],
                    "training_epochs": {},
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "embedding_dim": 4,
                    "hidden_dim": 4,
                    "device": "cpu",
                },
                generator=FakeSampleGenerator(),
                device=torch.device("cpu"),
            )

    generator = RealSeqPoisonCandidateGenerator(trainer=FakeTrainer())
    request = SimpleNamespace(
        target_item=9,
        n_candidates=2,
        max_seq_len=3,
        seed=123,
        output_dir=tmp_path,
        round_index=0,
        dataset_bundle=bundle,
        valid_item_ids={1, 2, 9},
        config=PoisoningSSLSBRConfig(),
    )
    candidates = generator.generate(request)
    assert candidates == [[1, 1, 9], [2, 2, 9]]
    assert generator.last_metadata["real_generation_implemented"] is True


def test_pipeline_rejects_final_sessions_longer_than_max_seq_len(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        attack=replace(
            _config(tmp_path).attack,
            poisoning_ssl_sbr=replace(
                _config(tmp_path).attack.poisoning_ssl_sbr,
                max_seq_len_override=2,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="max_observed_length=3.*violation_count=1"):
        generate_poisoning_ssl_sbr_target(
            config=config,
            shared=_toy_shared(fake_session_count=1),
            target_item=9,
            run_type=POISONING_SSL_SBR_RUN_TYPE,
            n_fake_requested=1,
            candidate_generator=StaticCandidateGenerator(
                rounds=[[[100, 1, 9, 2]]]
            ),
        )


def test_runner_build_poisoned_with_mock_generator_without_victim_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from attack.pipeline.runs import run_poisoning_ssl_sbr as runner

    config = _config(tmp_path)
    monkeypatch.setattr(
        runner,
        "prepare_lightweight_attack_artifacts",
        lambda *args, **kwargs: _toy_shared(fake_session_count=2),
    )
    captured: dict[str, object] = {}

    def fake_run_targets_and_victims(*args, **kwargs):
        payload = kwargs["build_poisoned"](9)
        captured["payload"] = payload
        return {"victim_execution_invoked": False}

    monkeypatch.setattr(runner, "run_targets_and_victims", fake_run_targets_and_victims)
    summary = runner.run_poisoning_ssl_sbr(
        config,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    assert summary == {"victim_execution_invoked": False}
    payload = captured["payload"]
    assert payload.raw_fake_sessions == [[9, 1], [1, 9]]
    assert isinstance(payload.poisoned, PoisonedDataset)
    assert payload.metadata["n_final_injected"] == 2
