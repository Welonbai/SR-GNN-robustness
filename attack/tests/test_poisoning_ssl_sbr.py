from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import PoisoningSSLSBRConfig, load_config
from attack.common.artifact_io import load_fake_sessions, load_json, save_fake_sessions, save_json
from attack.common.paths import (
    POISONING_SSL_SBR_RUN_TYPE,
    attack_key_payload,
    classify_victim_training_run_type,
    shared_attack_artifact_key_payload,
    shared_attack_identity_requires_poison_runner,
    target_dir,
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
from attack.poisoning_ssl.model import Generator
from attack.poisoning_ssl.pipeline import (
    compute_seqpoison_max_seq_len,
    generate_poisoning_ssl_sbr_target,
    _shared_fake_session_cache_root,
)
from attack.poisoning_ssl.postprocess import postprocess_fake_user_sequences
from attack.poisoning_ssl.trainer import (
    EffectiveSeqPoisonTrainingConfig,
    _checkpoint_identity,
)


CONFIG_PATH = (
    "attack/configs/diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1.yaml"
)
VERSION_A_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_diag.yaml"
)
VERSION_A_FIRSTSTEPMASK_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_firststepmask_diag.yaml"
)
VERSION_A_FIRSTSTEPMASK_BIAS_CONFIG_PATHS = (
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_firststepmask_bias1_diag.yaml",
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_firststepmask_bias2_diag.yaml",
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_versionA_firststepmask_bias3_diag.yaml",
)
FORMAL_ADV100_BIAS2_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_poisoning_ssl_sbr_popular_count1_formal_adv100_bias2_diag.yaml"
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
    assert poisoning.first_step_target_mask is False
    assert poisoning.target_logit_bias_after_first_step == 0.0
    assert poisoning.candidate_multiplier == 1
    assert poisoning.max_generation_rounds == 10
    assert poisoning.generation_backend == "real"
    assert poisoning.candidate_save_policy == "sample"
    assert PoisoningSSLSBRConfig().candidate_save_policy == "summary_only"
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
    assert identity["first_step_target_mask"] is False
    assert identity["target_logit_bias_after_first_step"] == 0.0
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
    with pytest.raises(ValueError, match="candidate_save_policy"):
        PoisoningSSLSBRConfig(candidate_save_policy="bad")
    with pytest.raises(ValueError, match="max_saved_candidates"):
        PoisoningSSLSBRConfig(max_saved_candidates=-1)
    with pytest.raises(ValueError, match="acceptance_eval_interval_epochs"):
        PoisoningSSLSBRConfig(acceptance_eval_enabled=True)


def test_poisoning_ssl_sbr_version_a_config_parses() -> None:
    config = load_config(VERSION_A_CONFIG_PATH)
    poisoning = config.attack.poisoning_ssl_sbr
    assert poisoning is not None
    assert poisoning.generation_backend == "real"
    assert poisoning.candidate_save_policy == "summary_only"
    assert poisoning.acceptance_eval_enabled is True
    assert poisoning.acceptance_eval_interval_epochs == 10
    assert poisoning.acceptance_eval_candidate_count == 200


def test_poisoning_ssl_sbr_version_a_firststepmask_config_parses() -> None:
    config = load_config(VERSION_A_FIRSTSTEPMASK_CONFIG_PATH)
    poisoning = config.attack.poisoning_ssl_sbr
    assert poisoning is not None
    assert poisoning.generation_backend == "real"
    assert poisoning.first_step_target_mask is True
    assert poisoning.candidate_save_policy == "summary_only"
    assert poisoning.reuse_existing_artifacts is True


def test_poisoning_ssl_sbr_version_a_firststepmask_bias_configs_parse() -> None:
    for expected_bias, path in zip(
        (1.0, 2.0, 3.0),
        VERSION_A_FIRSTSTEPMASK_BIAS_CONFIG_PATHS,
    ):
        config = load_config(path)
        poisoning = config.attack.poisoning_ssl_sbr
        assert poisoning is not None
        assert poisoning.first_step_target_mask is True
        assert poisoning.target_logit_bias_after_first_step == expected_bias
        assert poisoning.candidate_save_policy == "summary_only"
        assert poisoning.reuse_existing_artifacts is True


def test_poisoning_ssl_sbr_formal_adv100_bias2_config_parses() -> None:
    config = load_config(FORMAL_ADV100_BIAS2_CONFIG_PATH)
    poisoning = config.attack.poisoning_ssl_sbr
    assert poisoning is not None
    assert config.attack.size == pytest.approx(0.0001)
    assert poisoning.first_step_target_mask is True
    assert poisoning.target_logit_bias_after_first_step == 2.0
    assert poisoning.enforce_single_target is True
    assert poisoning.classifier_epochs == 20
    assert poisoning.mle_epochs == 20
    assert poisoning.adversarial_epochs == 100
    assert poisoning.candidate_multiplier == 20
    assert poisoning.max_generation_rounds == 50
    assert poisoning.candidate_save_policy == "summary_only"
    assert poisoning.reuse_existing_artifacts is True


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


def test_first_step_target_mask_masks_only_first_generated_position() -> None:
    generator = Generator(embedding_dim=4, hidden_dim=4, vocab_size=3, max_seq_len=3)
    for param in generator.parameters():
        param.data.zero_()
    generator.gru2out.bias.data[:] = torch.tensor([-100.0, 0.0, 100.0])
    samples = generator.sample(
        4,
        device=torch.device("cpu"),
        first_step_target_mask=True,
        first_step_mask_target_id=2,
    )
    assert samples[:, 0].tolist() == [1, 1, 1, 1]
    assert samples[:, 1].tolist() == [2, 2, 2, 2]
    assert samples[:, 2].tolist() == [2, 2, 2, 2]


def test_target_logit_bias_applies_only_after_first_step() -> None:
    generator = Generator(embedding_dim=4, hidden_dim=4, vocab_size=3, max_seq_len=3)
    for param in generator.parameters():
        param.data.zero_()
    generator.gru2out.bias.data[:] = torch.tensor([-100.0, 100.0, 0.0])
    samples = generator.sample(
        4,
        device=torch.device("cpu"),
        first_step_mask_target_id=2,
        target_logit_bias_after_first_step=200.0,
    )
    assert samples[:, 0].tolist() == [1, 1, 1, 1]
    assert samples[:, 1].tolist() == [2, 2, 2, 2]
    assert samples[:, 2].tolist() == [2, 2, 2, 2]


def test_first_step_mask_has_priority_over_target_logit_bias() -> None:
    generator = Generator(embedding_dim=4, hidden_dim=4, vocab_size=3, max_seq_len=2)
    for param in generator.parameters():
        param.data.zero_()
    generator.gru2out.bias.data[:] = torch.tensor([-100.0, 0.0, 100.0])
    samples = generator.sample(
        4,
        device=torch.device("cpu"),
        first_step_target_mask=True,
        first_step_mask_target_id=2,
        target_logit_bias_after_first_step=200.0,
    )
    assert samples[:, 0].tolist() == [1, 1, 1, 1]
    assert samples[:, 1].tolist() == [2, 2, 2, 2]


def test_target_logit_bias_increases_later_target_probability() -> None:
    generator = Generator(embedding_dim=4, hidden_dim=4, vocab_size=3, max_seq_len=2)
    for param in generator.parameters():
        param.data.zero_()
    generator.gru2out.bias.data[:] = torch.tensor([-100.0, 0.0, 0.0])
    torch.manual_seed(7)
    unbiased = generator.sample(
        1000,
        device=torch.device("cpu"),
        first_step_mask_target_id=2,
    )
    torch.manual_seed(7)
    biased = generator.sample(
        1000,
        device=torch.device("cpu"),
        first_step_mask_target_id=2,
        target_logit_bias_after_first_step=3.0,
    )
    unbiased_rate = float((unbiased[:, 1] == 2).float().mean().item())
    biased_rate = float((biased[:, 1] == 2).float().mean().item())
    assert biased_rate > unbiased_rate + 0.25


def test_real_generator_first_step_mask_uses_remapped_target_id(tmp_path: Path) -> None:
    bundle = export_pseudo_user_sequences(
        [[10, 40], [20, 40]],
        target_item=40,
        output_dir=tmp_path / "bridge",
        valid_item_ids={10, 20, 40},
        max_seq_len=3,
    )
    assert bundle.remap_used is True
    assert bundle.seqpoison_target_item == bundle.canonical_to_seqpoison[40]
    captured: dict[str, object] = {}

    class FakeSampleGenerator:
        def sample(self, n, *, device, **kwargs):
            captured.update(kwargs)
            return torch.tensor(
                [[bundle.canonical_to_seqpoison[10], bundle.seqpoison_target_item, 0]],
                dtype=torch.long,
            )

    class FakeTrainer:
        def train_or_load(self, **kwargs):
            return SimpleNamespace(
                classifier_checkpoint_path=tmp_path / "classifier.pt",
                generator_checkpoint_path=tmp_path / "generator.pt",
                discriminator_checkpoint_path=tmp_path / "discriminator.pt",
                training_log_path=tmp_path / "training_log.json",
                generation_log_path=tmp_path / "generation_log.json",
                metadata={
                    "training_checkpoint_reused": True,
                    "training_checkpoint_path": str(tmp_path / "checkpoints"),
                    "training_checkpoint_identity": {"token": "same"},
                    "training_checkpoint_identity_hash": "same",
                    "enabled_reward_components": [],
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
    candidates = generator.generate(
        SimpleNamespace(
            target_item=40,
            n_candidates=1,
            max_seq_len=3,
            seed=123,
            output_dir=tmp_path,
            round_index=0,
            dataset_bundle=bundle,
            valid_item_ids={10, 20, 40},
            config=PoisoningSSLSBRConfig(first_step_target_mask=True),
        )
    )
    assert captured["first_step_target_mask"] is True
    assert captured["first_step_mask_target_id"] == bundle.seqpoison_target_item
    assert captured["target_logit_bias_after_first_step"] == 0.0
    assert candidates == [[1, 10, 40]]
    assert generator.last_metadata["first_step_target_mask_applied"] is True
    assert (
        generator.last_metadata["first_step_target_mask_target_id_seqpoison"]
        == bundle.seqpoison_target_item
    )


def test_real_generator_bias_uses_remapped_target_id(tmp_path: Path) -> None:
    bundle = export_pseudo_user_sequences(
        [[10, 40], [20, 40]],
        target_item=40,
        output_dir=tmp_path / "bridge_bias",
        valid_item_ids={10, 20, 40},
        max_seq_len=3,
    )
    captured: dict[str, object] = {}

    class FakeSampleGenerator:
        def sample(self, n, *, device, **kwargs):
            captured.update(kwargs)
            return torch.tensor(
                [[bundle.canonical_to_seqpoison[10], bundle.seqpoison_target_item, 0]],
                dtype=torch.long,
            )

    class FakeTrainer:
        def train_or_load(self, **kwargs):
            return SimpleNamespace(
                classifier_checkpoint_path=tmp_path / "classifier.pt",
                generator_checkpoint_path=tmp_path / "generator.pt",
                discriminator_checkpoint_path=tmp_path / "discriminator.pt",
                training_log_path=tmp_path / "training_log.json",
                generation_log_path=tmp_path / "generation_log.json",
                metadata={
                    "training_checkpoint_reused": True,
                    "training_checkpoint_path": str(tmp_path / "checkpoints"),
                    "training_checkpoint_identity": {"token": "same"},
                    "training_checkpoint_identity_hash": "same",
                    "enabled_reward_components": [],
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
    candidates = generator.generate(
        SimpleNamespace(
            target_item=40,
            n_candidates=1,
            max_seq_len=3,
            seed=123,
            output_dir=tmp_path,
            round_index=0,
            dataset_bundle=bundle,
            valid_item_ids={10, 20, 40},
            config=PoisoningSSLSBRConfig(target_logit_bias_after_first_step=2.0),
        )
    )
    assert captured["first_step_target_mask"] is False
    assert captured["first_step_mask_target_id"] == bundle.seqpoison_target_item
    assert captured["target_logit_bias_after_first_step"] == 2.0
    assert candidates == [[1, 10, 40]]
    assert generator.last_metadata["target_logit_bias_after_first_step_applied"] is True
    assert (
        generator.last_metadata["target_logit_bias_target_id_seqpoison"]
        == bundle.seqpoison_target_item
    )


def test_training_checkpoint_identity_excludes_first_step_target_mask(tmp_path: Path) -> None:
    bundle = export_pseudo_user_sequences(
        [[1, 9], [2, 9]],
        target_item=9,
        output_dir=tmp_path / "bridge",
        valid_item_ids={1, 2, 9},
        max_seq_len=3,
    )
    effective_a = EffectiveSeqPoisonTrainingConfig.from_config(PoisoningSSLSBRConfig())
    effective_b = EffectiveSeqPoisonTrainingConfig.from_config(
        PoisoningSSLSBRConfig(first_step_target_mask=True)
    )
    effective_c = EffectiveSeqPoisonTrainingConfig.from_config(
        PoisoningSSLSBRConfig(target_logit_bias_after_first_step=3.0)
    )
    assert effective_a == effective_b
    assert effective_a == effective_c
    identity_a = _checkpoint_identity(
        dataset_bundle=bundle,
        target_item=9,
        seed=123,
        effective=effective_a,
    )
    identity_b = _checkpoint_identity(
        dataset_bundle=bundle,
        target_item=9,
        seed=123,
        effective=effective_b,
    )
    identity_c = _checkpoint_identity(
        dataset_bundle=bundle,
        target_item=9,
        seed=123,
        effective=effective_c,
    )
    assert identity_a == identity_b
    assert identity_a == identity_c


def test_training_checkpoint_identity_excludes_fake_session_cache_and_decoding_fields(
    tmp_path: Path,
) -> None:
    bundle = export_pseudo_user_sequences(
        [[1, 9], [2, 9]],
        target_item=9,
        output_dir=tmp_path / "bridge_cache_identity",
        valid_item_ids={1, 2, 9},
        max_seq_len=3,
    )
    base = EffectiveSeqPoisonTrainingConfig.from_config(PoisoningSSLSBRConfig())
    changed = EffectiveSeqPoisonTrainingConfig.from_config(
        PoisoningSSLSBRConfig(
            first_step_target_mask=True,
            target_logit_bias_after_first_step=2.0,
            candidate_multiplier=20,
            max_generation_rounds=50,
            candidate_save_policy="summary_only",
            max_saved_candidates=1,
        )
    )
    assert base == changed
    assert _checkpoint_identity(
        dataset_bundle=bundle,
        target_item=9,
        seed=123,
        effective=base,
    ) == _checkpoint_identity(
        dataset_bundle=bundle,
        target_item=9,
        seed=123,
        effective=changed,
    )


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
    assert duplicates["unique_fake_session_count"] == 2
    assert duplicates["unique_fake_session_ratio"] == pytest.approx(2 / 3)
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
    assert "total_start_time" in result.metadata
    assert "total_end_time" in result.metadata
    assert result.metadata["raw_candidates_generated_total"] == 3
    assert result.metadata["valid_sessions_generated_total"] == 2
    assert result.metadata["generation_round_durations_sec"] == [0.0]
    assert result.metadata["first_step_target_mask"] is False
    assert result.metadata["first_step_target_mask_applied"] is False
    assert result.metadata["unexpected_pos0_after_mask_count"] == 0
    assert result.metadata["target_logit_bias_after_first_step"] == 0.0
    assert result.metadata["target_logit_bias_after_first_step_applied"] is False
    assert result.metadata["target_logit_bias_positions"] == "none"
    assert "target_label_candidate_rate" in result.metadata
    assert "estimated_candidates_needed_for_target_label_budget" in result.metadata
    assert "generation_identity" in result.metadata
    assert "generation_identity_hash" in result.metadata
    assert result.metadata["generation_identity"]["attack_size"] == pytest.approx(
        config.attack.size
    )
    assert result.metadata["generation_identity"]["n_fake_requested"] == 2
    assert result.metadata["fake_session_cache_enabled"] is True
    assert result.metadata["fake_session_cache_hit"] is False
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
    generation_logs = list(target_root.rglob("generation_log.json"))
    assert generation_logs
    generation_log = load_json(generation_logs[0])
    assert generation_log["raw_candidate_count_by_round"] == [3]
    assert generation_log["valid_count_by_round"] == [2]
    manifests = list(target_root.rglob("seqpoison_sbr_manifest.json"))
    assert manifests
    manifest = load_json(manifests[0])
    assert manifest["target_item"] == 9
    assert manifest["n_final_injected"] == 2
    assert manifest["generation_identity_hash"] == result.metadata["generation_identity_hash"]
    assert manifest["raw_fake_sessions_path"].endswith("raw_fake_sessions.pkl")
    summaries = list(target_root.rglob("fake_session_sanity_summary.json"))
    assert summaries
    sanity = load_json(summaries[0])
    assert sanity["n_fake_requested"] == 2
    assert sanity["n_final_injected"] == 2
    assert sanity["target_position_distribution"] == {"0": 1, "1": 1}
    assert sanity["fake_length_p50"] == 2
    assert sanity["train_sub_length_p99"] == 4


def test_fake_session_cache_hit_loads_sessions_and_skips_generator(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    first = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )

    class FailingGenerator:
        def generate(self, request):
            raise AssertionError("cache hit should skip candidate generation")

    second = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=FailingGenerator(),
    )
    assert second.raw_fake_sessions == first.raw_fake_sessions
    assert second.metadata["fake_session_cache_hit"] is True
    assert second.metadata["generation_identity_hash"] == first.metadata[
        "generation_identity_hash"
    ]
    root = target_dir(config, 9, run_type=POISONING_SSL_SBR_RUN_TYPE)
    manifest = load_json(root / "seqpoison_sbr_manifest.json")
    assert manifest["cache_hit"] is True
    assert manifest["fake_session_cache_scope"] == "shared"


def test_shared_cache_path_is_independent_of_experiment_name(tmp_path: Path) -> None:
    config_a = _config(tmp_path)
    config_b = replace(
        config_a,
        experiment=replace(config_a.experiment, name="same_attack_different_name"),
    )
    result_a = generate_poisoning_ssl_sbr_target(
        config=config_a,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    shared_a = _shared_fake_session_cache_root(
        config=config_a,
        target_item=9,
        generation_identity_hash=result_a.metadata["generation_identity_hash"],
    )
    shared_b = _shared_fake_session_cache_root(
        config=config_b,
        target_item=9,
        generation_identity_hash=result_a.metadata["generation_identity_hash"],
    )
    assert shared_a == shared_b
    assert str(shared_a).endswith(
        "shared\\diginetica\\poisoning_ssl_sbr_fake_sessions\\9\\"
        + result_a.metadata["generation_identity_hash"]
    )


def test_same_generation_identity_different_experiment_hits_shared_cache(
    tmp_path: Path,
) -> None:
    config_a = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    first = generate_poisoning_ssl_sbr_target(
        config=config_a,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    config_b = replace(
        config_a,
        experiment=replace(config_a.experiment, name="different_experiment_name"),
    )

    class FailingGenerator:
        def generate(self, request):
            raise AssertionError("shared cache hit should skip candidate generation")

    second = generate_poisoning_ssl_sbr_target(
        config=config_b,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=FailingGenerator(),
    )
    assert second.raw_fake_sessions == first.raw_fake_sessions
    assert second.metadata["fake_session_cache_hit"] is True
    assert second.metadata["fake_session_cache_scope"] == "shared"
    local_root_b = target_dir(config_b, 9, run_type=POISONING_SSL_SBR_RUN_TYPE)
    assert (local_root_b / "raw_fake_sessions.pkl").exists()
    assert (local_root_b / "seqpoison_sbr_manifest.json").exists()
    manifest_b = load_json(local_root_b / "seqpoison_sbr_manifest.json")
    assert manifest_b["fake_session_cache_scope"] == "shared"
    assert manifest_b["local_target_root"] == str(local_root_b)


def test_shared_cache_hit_takes_precedence_over_local_cache(tmp_path: Path) -> None:
    config_a = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    first = generate_poisoning_ssl_sbr_target(
        config=config_a,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    config_b = replace(
        config_a,
        experiment=replace(config_a.experiment, name="local_cache_conflict"),
    )
    local_root_b = target_dir(config_b, 9, run_type=POISONING_SSL_SBR_RUN_TYPE)
    local_root_b.mkdir(parents=True, exist_ok=True)
    save_fake_sessions([[1, 9], [2, 9]], local_root_b / "raw_fake_sessions.pkl")
    local_metadata = dict(first.metadata)
    local_metadata["n_final_injected"] = 2
    save_json(local_metadata, local_root_b / "poisoning_ssl_sbr_metadata.json")

    class FailingGenerator:
        def generate(self, request):
            raise AssertionError("shared cache should take precedence")

    result = generate_poisoning_ssl_sbr_target(
        config=config_b,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=FailingGenerator(),
    )
    assert result.raw_fake_sessions == first.raw_fake_sessions
    assert result.raw_fake_sessions != [[1, 9], [2, 9]]
    assert result.metadata["fake_session_cache_scope"] == "shared"


def test_local_cache_used_when_shared_cache_misses(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    first = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    shared_cache_root = _shared_fake_session_cache_root(
        config=config,
        target_item=9,
        generation_identity_hash=first.metadata["generation_identity_hash"],
    )
    shutil.rmtree(shared_cache_root)

    class FailingGenerator:
        def generate(self, request):
            raise AssertionError("local cache hit should skip generation")

    second = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=FailingGenerator(),
    )
    assert second.raw_fake_sessions == first.raw_fake_sessions
    assert second.metadata["fake_session_cache_hit"] is True
    assert second.metadata["fake_session_cache_scope"] == "local"
    assert (shared_cache_root / "raw_fake_sessions.pkl").exists()


def test_fake_session_cache_miss_when_n_fake_requested_differs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    called = {"value": False}

    class OneSessionGenerator:
        def generate(self, request):
            called["value"] = True
            return [[200, 9, 2]]

    second = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=1),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=1,
        candidate_generator=OneSessionGenerator(),
    )
    assert called["value"] is True
    assert second.raw_fake_sessions == [[9, 2]]
    assert second.metadata["fake_session_cache_hit"] is False
    assert "n_fake_requested" in second.metadata["fake_session_cache_mismatch_fields"]


def test_fake_session_cache_miss_when_generation_identity_differs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    shared = _toy_shared(fake_session_count=2)
    generate_poisoning_ssl_sbr_target(
        config=config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 9, 1], [101, 1, 9]]]
        ),
    )
    changed_config = replace(
        config,
        attack=replace(
            config.attack,
            poisoning_ssl_sbr=replace(
                config.attack.poisoning_ssl_sbr,
                target_logit_bias_after_first_step=2.0,
            ),
        ),
    )
    called = {"value": False}

    class FreshGenerator:
        def generate(self, request):
            called["value"] = True
            return [[300, 9, 1], [301, 1, 9]]

    result = generate_poisoning_ssl_sbr_target(
        config=changed_config,
        shared=shared,
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=FreshGenerator(),
    )
    assert called["value"] is True
    assert result.metadata["fake_session_cache_hit"] is False
    assert result.metadata["generation_identity"]["target_logit_bias_after_first_step"] == 2.0


def test_pipeline_first_step_mask_metadata_and_generation_identity(tmp_path: Path) -> None:
    base = _config(tmp_path)
    poisoning = replace(
        base.attack.poisoning_ssl_sbr,
        first_step_target_mask=True,
    )
    config = replace(base, attack=replace(base.attack, poisoning_ssl_sbr=poisoning))
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 1, 9], [101, 2, 9]]]
        ),
    )
    assert result.raw_fake_sessions == [[1, 9], [2, 9]]
    assert result.metadata["first_step_target_mask"] is True
    assert result.metadata["target_pos0_count"] == 0
    assert result.metadata["unexpected_pos0_after_mask_count"] == 0
    assert result.metadata["target_label_pair_count_added"] == 2
    assert result.metadata["target_label_candidate_rate"] == pytest.approx(1.0)
    assert result.metadata["generation_identity"]["first_step_target_mask"] is True
    assert result.metadata["generation_identity"]["target_logit_bias_after_first_step"] == 0.0

    unmasked = _config(tmp_path / "unmasked")
    masked_payload = attack_key_payload(config, run_type=POISONING_SSL_SBR_RUN_TYPE)
    unmasked_payload = attack_key_payload(unmasked, run_type=POISONING_SSL_SBR_RUN_TYPE)
    assert (
        masked_payload["attack"]["poisoning_ssl_sbr"]["first_step_target_mask"]
        is True
    )
    assert (
        unmasked_payload["attack"]["poisoning_ssl_sbr"]["first_step_target_mask"]
        is False
    )
    assert masked_payload != unmasked_payload


def test_pipeline_target_logit_bias_metadata_and_generation_identity(tmp_path: Path) -> None:
    base = _config(tmp_path)
    poisoning = replace(
        base.attack.poisoning_ssl_sbr,
        first_step_target_mask=True,
        target_logit_bias_after_first_step=2.0,
    )
    config = replace(base, attack=replace(base.attack, poisoning_ssl_sbr=poisoning))
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 1, 9], [101, 2, 9]]]
        ),
    )
    assert result.metadata["first_step_target_mask"] is True
    assert result.metadata["target_logit_bias_after_first_step"] == 2.0
    assert result.metadata["target_logit_bias_after_first_step_applied"] is True
    assert result.metadata["target_logit_bias_target_id_canonical"] == 9
    assert result.metadata["target_logit_bias_positions"] == "positions>=1"
    assert result.metadata["target_label_pair_count_added"] == 2
    assert result.metadata["generation_identity"]["target_logit_bias_after_first_step"] == 2.0

    no_bias_config = replace(
        base,
        artifacts=replace(base.artifacts, root=str(tmp_path / "no_bias_outputs")),
        attack=replace(
            base.attack,
            poisoning_ssl_sbr=replace(
                base.attack.poisoning_ssl_sbr,
                first_step_target_mask=True,
                target_logit_bias_after_first_step=0.0,
            ),
        ),
    )
    no_bias_result = generate_poisoning_ssl_sbr_target(
        config=no_bias_config,
        shared=_toy_shared(fake_session_count=2),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=2,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 1, 9], [101, 2, 9]]]
        ),
    )
    assert (
        result.metadata["generation_identity_hash"]
        != no_bias_result.metadata["generation_identity_hash"]
    )


def test_summary_only_does_not_save_rejected_raw_candidates(tmp_path: Path) -> None:
    base = _config(tmp_path)
    poisoning = replace(
        base.attack.poisoning_ssl_sbr,
        candidate_save_policy="summary_only",
        save_generated_candidates=False,
    )
    config = replace(base, attack=replace(base.attack, poisoning_ssl_sbr=poisoning))
    result = generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=1),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=1,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 1, 2], [101, 9, 1]]]
        ),
    )
    assert result.raw_fake_sessions == [[9, 1]]
    target_root = (
        Path(config.artifacts.root)
        / config.artifacts.runs_dir
        / config.data.dataset_name
        / config.experiment.name
    )
    assert not list(target_root.rglob("generated_candidates.pkl"))
    assert list(target_root.rglob("raw_fake_sessions.pkl"))


def test_sample_candidate_save_policy_respects_max_saved_candidates(tmp_path: Path) -> None:
    base = _config(tmp_path)
    poisoning = replace(
        base.attack.poisoning_ssl_sbr,
        candidate_save_policy="sample",
        max_saved_candidates=1,
    )
    config = replace(base, attack=replace(base.attack, poisoning_ssl_sbr=poisoning))
    generate_poisoning_ssl_sbr_target(
        config=config,
        shared=_toy_shared(fake_session_count=1),
        target_item=9,
        run_type=POISONING_SSL_SBR_RUN_TYPE,
        n_fake_requested=1,
        candidate_generator=StaticCandidateGenerator(
            rounds=[[[100, 1, 2], [101, 9, 1], [102, 1, 9]]]
        ),
    )
    target_root = (
        Path(config.artifacts.root)
        / config.artifacts.runs_dir
        / config.data.dataset_name
        / config.experiment.name
    )
    saved_paths = list(target_root.rglob("generated_candidates.pkl"))
    assert saved_paths
    assert load_fake_sessions(saved_paths[0]) == [[100, 1, 2]]


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
        def sample(self, n, *, device, **kwargs):
            assert n == 2
            assert kwargs["first_step_target_mask"] is False
            assert kwargs["target_logit_bias_after_first_step"] == 0.0
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
    assert generator.last_metadata["first_step_target_mask"] is False
    assert generator.last_metadata["target_logit_bias_after_first_step"] == 0.0


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
