from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from attack.common.artifact_io import save_json
from attack.common.config import PoisoningSSLSBRConfig
from attack.poisoning_ssl.dataset_bridge import SeqPoisonDatasetBundle
from attack.poisoning_ssl.model import (
    Classify,
    Discriminator,
    Generator,
    classifier_filter_config,
    padded_sequences,
    prepare_generator_batch,
    sample_sequences_in_chunks,
    unpad_generated_tensor,
)
from attack.poisoning_ssl.diagnostics import (
    budget_diagnostics,
    length_stats,
    target_diagnostics,
)
from attack.poisoning_ssl.postprocess import postprocess_fake_user_sequences
from attack.poisoning_ssl.provenance import UPSTREAM_COMMIT


@dataclass(frozen=True)
class EffectiveSeqPoisonTrainingConfig:
    classifier_epochs: int = 20
    mle_epochs: int = 20
    adversarial_epochs: int = 100
    discriminator_pretrain_steps: int = 20
    discriminator_pretrain_epochs: int = 3
    discriminator_adversarial_steps: int = 3
    discriminator_adversarial_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    classifier_learning_rate: float = 0.01
    embedding_dim: int = 64
    hidden_dim: int = 64
    discriminator_embedding_dim: int = 64
    discriminator_hidden_dim: int = 64
    classifier_embedding_dim: int = 64
    pos_neg_samples: int = 10000
    reward_target_weight: float = 0.2
    reward_classifier_weight: float = 0.3
    reward_discriminator_weight: float = 0.5
    classifier_dropout: float = 0.75
    start_letter: int = 0
    target_probability: float = 0.9
    generation_sample_batch_size: int = 256

    @classmethod
    def from_config(cls, config: PoisoningSSLSBRConfig) -> "EffectiveSeqPoisonTrainingConfig":
        defaults = cls()
        values: dict[str, Any] = {}
        for field_name in defaults.__dataclass_fields__:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                values[field_name] = getattr(defaults, field_name) if value is None else value
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class SeqPoisonTrainerResult:
    output_dir: Path
    checkpoint_dir: Path
    classifier_checkpoint_path: Path
    generator_checkpoint_path: Path
    discriminator_checkpoint_path: Path
    training_log_path: Path
    generation_log_path: Path
    metadata: dict[str, object]
    generator: Generator
    discriminator: Discriminator
    classifier: Classify
    train_samples: torch.Tensor
    effective_config: EffectiveSeqPoisonTrainingConfig
    device: torch.device


class _ClassifierPretrainDataset(Dataset):
    def __init__(
        self,
        *,
        user_sequences: list[list[int]],
        max_seq_len: int,
        mask_id: int,
        mode: str,
        seed: int,
    ) -> None:
        rng = random.Random(int(seed))
        pairs: list[tuple[list[int], list[int], int]] = []
        masked_sequences: list[list[int]] = []
        anti_masked_sequences: list[list[int]] = []
        for sequence in user_sequences:
            if len(sequence) < 2:
                masked = list(sequence)
                anti = [int(mask_id)] * len(sequence)
            else:
                sample_length = rng.randint(1, max(1, len(sequence) // 2))
                start = rng.randint(0, len(sequence) - sample_length)
                masked = (
                    sequence[:start]
                    + [int(mask_id)] * sample_length
                    + sequence[start + sample_length :]
                )
                anti = (
                    [int(mask_id)] * len(sequence[:start])
                    + sequence[start : start + sample_length]
                    + [int(mask_id)] * len(sequence[start + sample_length :])
                )
            masked = _pad(masked, max_seq_len)
            anti = _pad(anti, max_seq_len)
            masked_sequences.append(masked)
            anti_masked_sequences.append(anti)
            pairs.append((masked, anti, 1))
        for index, masked in enumerate(masked_sequences):
            if len(masked_sequences) <= 1:
                other = index
            else:
                other = rng.randrange(0, len(masked_sequences))
                while other == index:
                    other = rng.randrange(0, len(masked_sequences))
            pairs.append((masked, anti_masked_sequences[other], 0))
        half = len(pairs) // 2
        train_cut = int(0.9 * half)
        if mode == "train":
            pairs = pairs[:train_cut] + pairs[half : half + train_cut]
        else:
            pairs = pairs[train_cut:half] + pairs[half + train_cut :]
            if not pairs:
                pairs = pairs[:1]
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        masked, anti, label = self._pairs[int(index)]
        data = torch.cat(
            (
                torch.tensor(masked, dtype=torch.long),
                torch.tensor(anti, dtype=torch.long),
            ),
            dim=0,
        )
        return data, torch.tensor(label, dtype=torch.long)


class _ClassifierRewardDataset(Dataset):
    def __init__(
        self,
        *,
        real_sequences: list[list[int]],
        fake_samples: torch.Tensor,
        max_seq_len: int,
        mask_id: int,
        seed: int,
    ) -> None:
        rng = random.Random(int(seed))
        real = _unpad_rows(fake_or_real=real_sequences)
        fake = _unpad_rows(
            fake_or_real=fake_samples.detach().cpu().tolist(),
            keep_empty=True,
        )
        pairs: list[tuple[list[int], list[int], int]] = []
        for fake_data in fake:
            if len(fake_data) < 2:
                masked = list(fake_data)
                anti = [int(mask_id)] * len(fake_data)
            else:
                real_sample = real[rng.randrange(0, len(real))]
                min_len = min(len(fake_data), len(real_sample))
                sample_length = rng.randint(1, max(1, min_len // 2))
                start = rng.randint(0, min_len - sample_length)
                masked = (
                    [int(mask_id)] * len(real_sample[:start])
                    + fake_data[start : start + sample_length]
                    + [int(mask_id)] * len(real_sample[start + sample_length :])
                )
                anti = (
                    real_sample[:start]
                    + [int(mask_id)] * sample_length
                    + real_sample[start + sample_length :]
                )
            pairs.append((_pad(masked, max_seq_len), _pad(anti, max_seq_len), 0))
        self._pairs = pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        masked, anti, label = self._pairs[int(index)]
        data = torch.cat(
            (
                torch.tensor(masked, dtype=torch.long),
                torch.tensor(anti, dtype=torch.long),
            ),
            dim=0,
        )
        return data, torch.tensor(label, dtype=torch.long)


class SeqPoisonTrainer:
    def train_or_load(
        self,
        *,
        output_dir: str | Path,
        dataset_bundle: SeqPoisonDatasetBundle,
        config: PoisoningSSLSBRConfig,
        target_item: int,
        seed: int,
    ) -> SeqPoisonTrainerResult:
        total_start_time = _now_iso()
        total_start_perf = time.perf_counter()
        effective = EffectiveSeqPoisonTrainingConfig.from_config(config)
        sample_batch_size = int(config.generation_sample_batch_size)
        device = _resolve_device(config)
        output = Path(output_dir)
        identity = _checkpoint_identity(
            dataset_bundle=dataset_bundle,
            target_item=target_item,
            seed=seed,
            effective=effective,
        )
        preferred_checkpoint_dir = _checkpoint_dir(
            output,
            config=config,
            dataset_bundle=dataset_bundle,
            target_item=target_item,
            seed=seed,
            effective=effective,
        )
        checkpoint_dir = preferred_checkpoint_dir
        discovered_checkpoint_dir = (
            None
            if config.checkpoint_dir or not bool(config.reuse_existing_artifacts)
            else _find_existing_checkpoint_dir(
                output=output,
                target_item=int(target_item),
                identity=identity,
            )
        )
        if discovered_checkpoint_dir is not None:
            checkpoint_dir = discovered_checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        identity_path = checkpoint_dir / "checkpoint_identity.json"
        classifier_path = checkpoint_dir / "classifier.pt"
        generator_path = checkpoint_dir / "generator.pt"
        discriminator_path = checkpoint_dir / "discriminator.pt"
        training_log_path = output / "training_log.json"
        generation_log_path = output / "generation_log.json"

        train_samples = padded_sequences(
            dataset_bundle.train_sequences,
            max_seq_len=dataset_bundle.max_seq_len,
            device=device,
        )
        classifier = _make_classifier(dataset_bundle, effective).to(device)
        generator = Generator(
            effective.embedding_dim,
            effective.hidden_dim,
            dataset_bundle.vocab_size,
            dataset_bundle.max_seq_len,
        ).to(device)
        discriminator = Discriminator(
            effective.discriminator_embedding_dim,
            effective.discriminator_hidden_dim,
            dataset_bundle.vocab_size,
            dataset_bundle.max_seq_len,
        ).to(device)
        can_reuse = (
            bool(config.reuse_existing_artifacts)
            and identity_path.exists()
            and classifier_path.exists()
            and generator_path.exists()
            and discriminator_path.exists()
            and _load_json(identity_path) == identity
        )
        _progress(
            target_item,
            "Trainer setup: "
            f"device={device} checkpoint_dir={checkpoint_dir} "
            f"preferred_checkpoint_dir={preferred_checkpoint_dir} "
            f"reuse_existing_artifacts={bool(config.reuse_existing_artifacts)} "
            f"reuse_hit={bool(can_reuse)} "
            f"first_step_target_mask={bool(config.first_step_target_mask)} "
            f"target_logit_bias_after_first_step="
            f"{float(config.target_logit_bias_after_first_step):.6g}",
        )
        if can_reuse:
            reuse_start = time.perf_counter()
            _progress(target_item, "Stage start: checkpoint load")
            classifier.load_state_dict(torch.load(classifier_path, map_location=device))
            generator.load_state_dict(torch.load(generator_path, map_location=device))
            discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
            _progress(
                target_item,
                "Stage done: checkpoint load "
                f"duration={_fmt_duration(time.perf_counter() - reuse_start)}",
            )
            training_log = {
                "reused_checkpoints": True,
                "checkpoint_dir": str(checkpoint_dir),
                "training_start_time": total_start_time,
                "training_end_time": _now_iso(),
                "training_duration_sec": 0.0,
                "classifier_training_duration_sec": 0.0,
                "mle_pretraining_duration_sec": 0.0,
                "discriminator_pretraining_duration_sec": 0.0,
                "adversarial_training_duration_sec": 0.0,
                "classifier_epoch_durations_sec": [],
                "mle_epoch_durations_sec": [],
                "adversarial_epoch_durations_sec": [],
                "discriminator_update_durations_sec": [],
                "acceptance_evaluations": [],
                "generation_sample_batch_size": sample_batch_size,
            }
        else:
            _seed_everything(seed)
            training_start_time = _now_iso()
            training_start_perf = time.perf_counter()
            training_log = self._train(
                classifier=classifier,
                generator=generator,
                discriminator=discriminator,
                train_samples=train_samples,
                dataset_bundle=dataset_bundle,
                effective=effective,
                config=config,
                device=device,
                seed=seed,
                target_item=int(target_item),
                progress_path=output / "training_progress.json",
            )
            training_log["training_start_time"] = training_start_time
            training_log["training_end_time"] = _now_iso()
            training_log["training_duration_sec"] = float(
                time.perf_counter() - training_start_perf
            )
            training_log["reused_checkpoints"] = False
            if bool(config.save_checkpoints):
                torch.save(classifier.cpu().state_dict(), classifier_path)
                torch.save(generator.cpu().state_dict(), generator_path)
                torch.save(discriminator.cpu().state_dict(), discriminator_path)
                classifier.to(device)
                generator.to(device)
                discriminator.to(device)
            save_json(identity, identity_path)
        training_log["total_start_time"] = total_start_time
        training_log["total_end_time"] = _now_iso()
        training_log["total_duration_sec"] = float(time.perf_counter() - total_start_perf)
        save_json(training_log, training_log_path)
        metadata = {
            "checkpoint_dir": str(checkpoint_dir),
            "preferred_checkpoint_dir": str(preferred_checkpoint_dir),
            "discovered_checkpoint_dir": (
                None if discovered_checkpoint_dir is None else str(discovered_checkpoint_dir)
            ),
            "training_checkpoint_path": str(checkpoint_dir),
            "training_checkpoint_identity": identity,
            "training_checkpoint_identity_hash": _hash_json(identity),
            "training_checkpoint_reused": bool(can_reuse),
            "classifier_checkpoint_path": str(classifier_path),
            "generator_checkpoint_path": str(generator_path),
            "discriminator_checkpoint_path": str(discriminator_path),
            "training_log_path": str(training_log_path),
            "generation_log_path": str(generation_log_path),
            "training_epochs": {
                "classifier_epochs": effective.classifier_epochs,
                "mle_epochs": effective.mle_epochs,
                "adversarial_epochs": effective.adversarial_epochs,
                "discriminator_pretrain_steps": effective.discriminator_pretrain_steps,
                "discriminator_pretrain_epochs": effective.discriminator_pretrain_epochs,
                "discriminator_adversarial_steps": effective.discriminator_adversarial_steps,
                "discriminator_adversarial_epochs": effective.discriminator_adversarial_epochs,
            },
            "batch_size": effective.batch_size,
            "generation_sample_batch_size": sample_batch_size,
            "learning_rate": effective.learning_rate,
            "classifier_learning_rate": effective.classifier_learning_rate,
            "embedding_dim": effective.embedding_dim,
            "hidden_dim": effective.hidden_dim,
            "device": str(device),
            "enabled_reward_components": [
                "target_related_reward",
                "bi_classifier_reward",
                "gan_discriminator_reward",
            ],
            "effective_training_config": effective.to_dict(),
            "training_start_time": training_log.get("training_start_time"),
            "training_end_time": training_log.get("training_end_time"),
            "training_duration_sec": training_log.get("training_duration_sec"),
            "classifier_training_duration_sec": training_log.get(
                "classifier_training_duration_sec"
            ),
            "mle_pretraining_duration_sec": training_log.get(
                "mle_pretraining_duration_sec"
            ),
            "discriminator_pretraining_duration_sec": training_log.get(
                "discriminator_pretraining_duration_sec"
            ),
            "adversarial_training_duration_sec": training_log.get(
                "adversarial_training_duration_sec"
            ),
            "classifier_epoch_durations_sec": training_log.get(
                "classifier_epoch_durations_sec",
                [],
            ),
            "mle_epoch_durations_sec": training_log.get("mle_epoch_durations_sec", []),
            "adversarial_epoch_durations_sec": training_log.get(
                "adversarial_epoch_durations_sec",
                [],
            ),
            "discriminator_update_durations_sec": training_log.get(
                "discriminator_update_durations_sec",
                [],
            ),
            "acceptance_evaluations": training_log.get("acceptance_evaluations", []),
        }
        return SeqPoisonTrainerResult(
            output_dir=output,
            checkpoint_dir=checkpoint_dir,
            classifier_checkpoint_path=classifier_path,
            generator_checkpoint_path=generator_path,
            discriminator_checkpoint_path=discriminator_path,
            training_log_path=training_log_path,
            generation_log_path=generation_log_path,
            metadata=metadata,
            generator=generator,
            discriminator=discriminator,
            classifier=classifier,
            train_samples=train_samples,
            effective_config=effective,
            device=device,
        )

    def _train(
        self,
        *,
        classifier: Classify,
        generator: Generator,
        discriminator: Discriminator,
        train_samples: torch.Tensor,
        dataset_bundle: SeqPoisonDatasetBundle,
        effective: EffectiveSeqPoisonTrainingConfig,
        config: PoisoningSSLSBRConfig,
        device: torch.device,
        seed: int,
        target_item: int,
        progress_path: Path,
    ) -> dict[str, object]:
        log: dict[str, object] = {
            "classifier": [],
            "mle": [],
            "discriminator_pretrain": [],
            "adversarial": [],
            "classifier_epoch_durations_sec": [],
            "mle_epoch_durations_sec": [],
            "adversarial_epoch_durations_sec": [],
            "discriminator_update_durations_sec": [],
            "acceptance_evaluations": [],
            "generation_sample_batch_size": int(config.generation_sample_batch_size),
        }
        stage_start = time.perf_counter()
        _progress(
            target_item,
            f"Stage start: classifier training epochs={effective.classifier_epochs}",
        )
        self._train_classifier(
            classifier=classifier,
            dataset_bundle=dataset_bundle,
            effective=effective,
            device=device,
            seed=seed,
            log=log,
            target_item=target_item,
            progress_path=progress_path,
        )
        log["classifier_training_duration_sec"] = float(time.perf_counter() - stage_start)
        _progress(
            target_item,
            "Stage done: classifier training "
            f"duration={_fmt_duration(log['classifier_training_duration_sec'])}",
        )
        gen_optimizer = optim.Adam(generator.parameters(), lr=effective.learning_rate)
        dis_optimizer = optim.Adagrad(discriminator.parameters(), lr=effective.learning_rate)
        stage_start = time.perf_counter()
        _progress(
            target_item,
            f"Stage start: generator MLE pretraining epochs={effective.mle_epochs}",
        )
        self._train_generator_mle(
            generator=generator,
            optimizer=gen_optimizer,
            train_samples=train_samples,
            effective=effective,
            log=log,
            target_item=target_item,
            progress_path=progress_path,
        )
        log["mle_pretraining_duration_sec"] = float(time.perf_counter() - stage_start)
        _progress(
            target_item,
            "Stage done: generator MLE pretraining "
            f"duration={_fmt_duration(log['mle_pretraining_duration_sec'])}",
        )
        stage_start = time.perf_counter()
        _progress(
            target_item,
            "Stage start: discriminator pretraining "
            f"steps={effective.discriminator_pretrain_steps} "
            f"epochs_per_step={effective.discriminator_pretrain_epochs}",
        )
        self._train_discriminator(
            discriminator=discriminator,
            optimizer=dis_optimizer,
            train_samples=train_samples,
            generator=generator,
            d_steps=effective.discriminator_pretrain_steps,
            epochs=effective.discriminator_pretrain_epochs,
            effective=effective,
            log_key="discriminator_pretrain",
            log=log,
            target_item=target_item,
            progress_path=progress_path,
            sample_batch_size=int(config.generation_sample_batch_size),
        )
        log["discriminator_pretraining_duration_sec"] = float(
            time.perf_counter() - stage_start
        )
        _progress(
            target_item,
            "Stage done: discriminator pretraining "
            f"duration={_fmt_duration(log['discriminator_pretraining_duration_sec'])}",
        )
        adversarial_start = time.perf_counter()
        _progress(
            target_item,
            f"Stage start: adversarial generator training epochs={effective.adversarial_epochs}",
        )
        for epoch in range(effective.adversarial_epochs):
            epoch_start = time.perf_counter()
            update_start = time.perf_counter()
            adv = self._train_generator_pg(
                generator=generator,
                optimizer=gen_optimizer,
                classifier=classifier,
                discriminator=discriminator,
                train_samples=train_samples,
                dataset_bundle=dataset_bundle,
                effective=effective,
                sample_batch_size=int(config.generation_sample_batch_size),
                seed=seed + epoch + 1000,
            )
            adv["generator_update_duration_sec"] = float(time.perf_counter() - update_start)
            self._train_discriminator(
                discriminator=discriminator,
                optimizer=dis_optimizer,
                train_samples=train_samples,
                generator=generator,
                d_steps=effective.discriminator_adversarial_steps,
                epochs=effective.discriminator_adversarial_epochs,
                effective=effective,
                log_key="adversarial_discriminator",
                log=log,
                target_item=target_item,
                progress_path=progress_path,
                sample_batch_size=int(config.generation_sample_batch_size),
            )
            epoch_duration = float(time.perf_counter() - epoch_start)
            log["adversarial_epoch_durations_sec"].append(epoch_duration)
            log["adversarial"].append(
                {"epoch": epoch + 1, "duration_sec": epoch_duration, **adv}
            )
            _progress(
                target_item,
                "adv epoch "
                f"{epoch + 1}/{effective.adversarial_epochs} "
                f"loss={adv['loss']:.4g} "
                f"loss_a={adv['loss_target']:.4g} "
                f"loss_b={adv['loss_classifier']:.4g} "
                f"loss_c={adv['loss_discriminator']:.4g} "
                f"duration={_fmt_duration(epoch_duration)}",
            )
            _write_training_progress(
                progress_path,
                target_item=target_item,
                stage="adversarial",
                current_epoch=epoch + 1,
                total_epochs=effective.adversarial_epochs,
                latest_metrics=adv,
                elapsed_sec=float(time.perf_counter() - adversarial_start),
            )
            if _acceptance_eval_due(config, epoch + 1):
                eval_payload = _acceptance_eval(
                    generator=generator,
                    dataset_bundle=dataset_bundle,
                    config=config,
                    epoch=epoch + 1,
                    device=train_samples.device,
                    seed=seed + epoch + 2000,
                )
                log["acceptance_evaluations"].append(eval_payload)
                _progress(
                    target_item,
                    "adv eval epoch "
                    f"{epoch + 1} candidates={eval_payload['eval_candidate_count']} "
                    f"target_ratio={float(eval_payload['target_containing_candidate_ratio']):.6g} "
                    f"valid={eval_payload['n_after_filtering']} "
                    f"duration={_fmt_duration(eval_payload['eval_duration_sec'])}",
                )
        log["adversarial_training_duration_sec"] = float(
            time.perf_counter() - adversarial_start
        )
        _progress(
            target_item,
            "Stage done: adversarial generator training "
            f"duration={_fmt_duration(log['adversarial_training_duration_sec'])}",
        )
        return log

    def _train_classifier(
        self,
        *,
        classifier: Classify,
        dataset_bundle: SeqPoisonDatasetBundle,
        effective: EffectiveSeqPoisonTrainingConfig,
        device: torch.device,
        seed: int,
        log: dict[str, object],
        target_item: int,
        progress_path: Path,
    ) -> None:
        dataset = _ClassifierPretrainDataset(
            user_sequences=dataset_bundle.train_sequences,
            max_seq_len=dataset_bundle.max_seq_len,
            mask_id=dataset_bundle.mask_id,
            mode="train",
            seed=seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=effective.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed)),
        )
        optimizer = optim.SGD(
            classifier.parameters(),
            lr=effective.classifier_learning_rate,
            momentum=0.9,
        )
        for epoch in range(effective.classifier_epochs):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            total_acc = 0.0
            batches = 0
            for data, target in loader:
                data = data.to(device)
                target = target.to(device)
                output = classifier(data)
                loss = -torch.log(output.gather(1, target.view(-1, 1)).clamp_min(1.0e-12)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pred = output.argmax(dim=1)
                total_loss += float(loss.detach().cpu())
                total_acc += float((pred == target).float().mean().detach().cpu())
                batches += 1
            duration = float(time.perf_counter() - epoch_start)
            loss_value = total_loss / max(1, batches)
            log["classifier_epoch_durations_sec"].append(duration)
            log["classifier"].append(
                {
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "accuracy": total_acc / max(1, batches),
                    "duration_sec": duration,
                }
            )
            _progress(
                target_item,
                "classifier epoch "
                f"{epoch + 1}/{effective.classifier_epochs} "
                f"loss={loss_value:.4g} duration={_fmt_duration(duration)}",
            )
            _write_training_progress(
                progress_path,
                target_item=target_item,
                stage="classifier",
                current_epoch=epoch + 1,
                total_epochs=effective.classifier_epochs,
                latest_metrics={"loss": loss_value},
            )

    def _train_generator_mle(
        self,
        *,
        generator: Generator,
        optimizer: optim.Optimizer,
        train_samples: torch.Tensor,
        effective: EffectiveSeqPoisonTrainingConfig,
        log: dict[str, object],
        target_item: int,
        progress_path: Path,
    ) -> None:
        train_len = int(train_samples.size(0))
        for epoch in range(effective.mle_epochs):
            epoch_start = time.perf_counter()
            total_loss = 0.0
            batches = 0
            for start in range(0, train_len, effective.batch_size):
                samples = train_samples[start : start + effective.batch_size]
                inp, target = prepare_generator_batch(
                    samples,
                    start_letter=effective.start_letter,
                )
                optimizer.zero_grad()
                loss = generator.batch_nll_loss(inp, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
            duration = float(time.perf_counter() - epoch_start)
            loss_value = (
                total_loss
                / max(1, batches)
                / max(1, int(train_samples.size(1)))
            )
            log["mle_epoch_durations_sec"].append(duration)
            log["mle"].append(
                {
                    "epoch": epoch + 1,
                    "average_train_nll": loss_value,
                    "duration_sec": duration,
                }
            )
            _progress(
                target_item,
                "mle epoch "
                f"{epoch + 1}/{effective.mle_epochs} "
                f"loss={loss_value:.4g} duration={_fmt_duration(duration)}",
            )
            _write_training_progress(
                progress_path,
                target_item=target_item,
                stage="mle",
                current_epoch=epoch + 1,
                total_epochs=effective.mle_epochs,
                latest_metrics={"average_train_nll": loss_value},
            )

    def _train_discriminator(
        self,
        *,
        discriminator: Discriminator,
        optimizer: optim.Optimizer,
        train_samples: torch.Tensor,
        generator: Generator,
        d_steps: int,
        epochs: int,
        effective: EffectiveSeqPoisonTrainingConfig,
        log_key: str,
        log: dict[str, object],
        target_item: int,
        progress_path: Path,
        sample_batch_size: int,
    ) -> None:
        device = train_samples.device
        if log_key not in log:
            log[log_key] = []
        for step in range(int(d_steps)):
            sample_count = min(
                int(effective.pos_neg_samples),
                max(int(effective.batch_size), int(train_samples.size(0))),
            )
            fake = sample_sequences_in_chunks(
                generator,
                sample_count,
                batch_size=int(sample_batch_size),
                device=device,
                stage_name=f"{log_key} step {step + 1}",
                log_fn=lambda message: _progress(target_item, message),
            )
            real_indices = torch.randint(
                0,
                int(train_samples.size(0)),
                (sample_count,),
                device=device,
            )
            real = train_samples[real_indices]
            data = torch.cat((real, fake), dim=0).long()
            labels = torch.cat(
                (
                    torch.ones(real.size(0), device=device),
                    torch.zeros(fake.size(0), device=device),
                ),
                dim=0,
            )
            perm = torch.randperm(data.size(0), device=device)
            data = data[perm]
            labels = labels[perm]
            for epoch in range(int(epochs)):
                update_start = time.perf_counter()
                total_loss = 0.0
                total_acc = 0.0
                batches = 0
                for start in range(0, data.size(0), effective.batch_size):
                    inp = data[start : start + effective.batch_size]
                    target = labels[start : start + effective.batch_size]
                    optimizer.zero_grad()
                    out = discriminator.batch_classify(inp)
                    loss = nn.BCELoss()(out, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach().cpu())
                    total_acc += float(((out > 0.5) == (target > 0.5)).float().mean().detach().cpu())
                    batches += 1
                duration = float(time.perf_counter() - update_start)
                loss_value = total_loss / max(1, batches)
                acc_value = total_acc / max(1, batches)
                log["discriminator_update_durations_sec"].append(duration)
                log[log_key].append(
                    {
                        "step": step + 1,
                        "epoch": epoch + 1,
                        "loss": loss_value,
                        "accuracy": acc_value,
                        "duration_sec": duration,
                    }
                )
                _progress(
                    target_item,
                    "discriminator "
                    f"{log_key} step {step + 1}/{int(d_steps)} "
                    f"epoch {epoch + 1}/{int(epochs)} "
                    f"loss={loss_value:.4g} duration={_fmt_duration(duration)}",
                )
                _write_training_progress(
                    progress_path,
                    target_item=target_item,
                    stage=log_key,
                    current_step=step + 1,
                    total_steps=int(d_steps),
                    current_epoch=epoch + 1,
                    total_epochs=int(epochs),
                    latest_metrics={"loss": loss_value, "accuracy": acc_value},
                )

    def _train_generator_pg(
        self,
        *,
        generator: Generator,
        optimizer: optim.Optimizer,
        classifier: Classify,
        discriminator: Discriminator,
        train_samples: torch.Tensor,
        dataset_bundle: SeqPoisonDatasetBundle,
        effective: EffectiveSeqPoisonTrainingConfig,
        sample_batch_size: int,
        seed: int,
    ) -> dict[str, float]:
        batch_count = int(effective.batch_size * 2)
        samples = sample_sequences_in_chunks(
            generator,
            batch_count,
            batch_size=int(sample_batch_size),
            device=train_samples.device,
            stage_name="adversarial generator policy-gradient",
        )
        inp, target = prepare_generator_batch(samples, start_letter=effective.start_letter)
        rewards = discriminator.batch_classify(target).detach()
        classifier_dataset = _ClassifierRewardDataset(
            real_sequences=dataset_bundle.train_sequences,
            fake_samples=samples,
            max_seq_len=dataset_bundle.max_seq_len,
            mask_id=dataset_bundle.mask_id,
            seed=seed,
        )
        classifier_loader = DataLoader(
            classifier_dataset,
            batch_size=batch_count,
            shuffle=False,
        )
        reward_batches = []
        with torch.no_grad():
            for data, _label in classifier_loader:
                data = data.to(train_samples.device)
                reward_batches.append(classifier(data)[:, 1])
        classifier_reward = (
            torch.cat(reward_batches, dim=0).detach()
            if reward_batches
            else torch.zeros(target.size(0), device=train_samples.device)
        )
        expected = int(target.size(0))
        if int(classifier_reward.numel()) != expected:
            raise RuntimeError(
                "SeqPoison-SBR classifier reward batch mismatch: "
                f"samples={int(samples.size(0))} input_batch={int(inp.size(0))} "
                f"target_batch={expected} "
                f"classifier_reward={int(classifier_reward.numel())}"
            )
        if int(rewards.numel()) != expected:
            raise RuntimeError(
                "SeqPoison-SBR discriminator reward batch mismatch: "
                f"target_batch={expected} discriminator_reward={int(rewards.numel())}"
            )
        loss_a = generator.batch_target_loss(
            inp,
            target,
            attack_item=dataset_bundle.seqpoison_target_item,
            target_probability=effective.target_probability,
        )
        loss_b = generator.batch_classifier_loss(inp, target, classifier_reward)
        loss_c = generator.batch_pg_loss(inp, target, rewards)
        loss = (
            float(effective.reward_discriminator_weight) * loss_c
            + float(effective.reward_classifier_weight) * loss_b
            + float(effective.reward_target_weight) * loss_a
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "loss_target": float(loss_a.detach().cpu()),
            "loss_classifier": float(loss_b.detach().cpu()),
            "loss_discriminator": float(loss_c.detach().cpu()),
        }


def _make_classifier(
    dataset_bundle: SeqPoisonDatasetBundle,
    effective: EffectiveSeqPoisonTrainingConfig,
) -> Classify:
    filter_sizes, num_filters = classifier_filter_config(2 * int(dataset_bundle.max_seq_len))
    return Classify(
        2,
        int(dataset_bundle.max_item_id + 2),
        effective.classifier_embedding_dim,
        filter_sizes,
        num_filters,
        effective.classifier_dropout,
    )


def _pad(sequence: list[int], max_seq_len: int) -> list[int]:
    return (list(sequence) + [0] * max(0, int(max_seq_len) - len(sequence)))[: int(max_seq_len)]


def _unpad_rows(
    fake_or_real: list[list[int]],
    *,
    keep_empty: bool = False,
) -> list[list[int]]:
    result: list[list[int]] = []
    for row in fake_or_real:
        sequence: list[int] = []
        for index, item in enumerate(row):
            item_id = int(item)
            if item_id != 0:
                sequence.append(item_id)
                continue
            if index == len(row) - 1 or int(row[index + 1]) == 0:
                break
        if sequence or keep_empty:
            result.append(sequence)
    return result or [[1]]


def _resolve_device(config: PoisoningSSLSBRConfig) -> torch.device:
    if config.device:
        requested = str(config.device)
    elif config.gpu_id is not None and torch.cuda.is_available():
        requested = f"cuda:{config.gpu_id}"
    else:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "SeqPoison-SBR requested CUDA device but torch.cuda.is_available() is false."
        )
    return torch.device(requested)


def _checkpoint_dir(
    output: Path,
    *,
    config: PoisoningSSLSBRConfig,
    dataset_bundle: SeqPoisonDatasetBundle,
    target_item: int,
    seed: int,
    effective: EffectiveSeqPoisonTrainingConfig,
) -> Path:
    root = Path(config.checkpoint_dir) if config.checkpoint_dir else output / "checkpoints"
    token = _hash_json(
        _checkpoint_identity(
            dataset_bundle=dataset_bundle,
            target_item=target_item,
            seed=seed,
            effective=effective,
        )
    )
    return root / f"target_{int(target_item)}_seqpoison_{token}"


def _find_existing_checkpoint_dir(
    *,
    output: Path,
    target_item: int,
    identity: dict[str, object],
) -> Path | None:
    token = _hash_json(identity)
    expected_name = f"target_{int(target_item)}_seqpoison_{token}"
    dataset_runs_dir = _dataset_runs_dir_from_output(output)
    if dataset_runs_dir is None or not dataset_runs_dir.exists():
        return None
    candidates = sorted(
        dataset_runs_dir.rglob(expected_name),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        identity_path = candidate / "checkpoint_identity.json"
        if not identity_path.exists():
            continue
        if _load_json(identity_path) != identity:
            continue
        if not all(
            (candidate / filename).exists()
            for filename in ("classifier.pt", "generator.pt", "discriminator.pt")
        ):
            continue
        return candidate
    return None


def _dataset_runs_dir_from_output(output: Path) -> Path | None:
    resolved = Path(output)
    parts = list(resolved.parts)
    if "runs" not in parts:
        return None
    run_index = len(parts) - 1 - list(reversed(parts)).index("runs")
    if len(parts) <= run_index + 1:
        return None
    return Path(*parts[: run_index + 2])


def _checkpoint_identity(
    *,
    dataset_bundle: SeqPoisonDatasetBundle,
    target_item: int,
    seed: int,
    effective: EffectiveSeqPoisonTrainingConfig,
) -> dict[str, object]:
    return {
        "upstream_commit": UPSTREAM_COMMIT,
        "target_item": int(target_item),
        "seqpoison_target_item": int(dataset_bundle.seqpoison_target_item),
        "max_seq_len": int(dataset_bundle.max_seq_len),
        "seed": int(seed),
        "remap_used": bool(dataset_bundle.remap_used),
        "train_sequence_count": int(len(dataset_bundle.train_sequences)),
        "max_item_id": int(dataset_bundle.max_item_id),
        "effective_training_config": effective.to_dict(),
    }


def _hash_json(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(target_item: int, message: str) -> None:
    print(f"[SeqPoison-SBR][target={int(target_item)}] {message}", flush=True)


def _fmt_duration(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "n/a"


def _write_training_progress(
    path: Path,
    *,
    target_item: int,
    stage: str,
    current_epoch: int | None = None,
    total_epochs: int | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    latest_metrics: dict[str, object] | None = None,
    elapsed_sec: float | None = None,
) -> None:
    save_json(
        {
            "target_item": int(target_item),
            "current_stage": str(stage),
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "current_step": current_step,
            "total_steps": total_steps,
            "elapsed_sec": elapsed_sec,
            "latest_metrics": dict(latest_metrics or {}),
            "updated_at": _now_iso(),
        },
        path,
    )


def _acceptance_eval_due(config: PoisoningSSLSBRConfig, epoch: int) -> bool:
    if not bool(config.acceptance_eval_enabled):
        return False
    interval = config.acceptance_eval_interval_epochs
    if interval is None:
        return False
    return int(epoch) % int(interval) == 0


def _acceptance_eval(
    *,
    generator: Generator,
    dataset_bundle: SeqPoisonDatasetBundle,
    config: PoisoningSSLSBRConfig,
    epoch: int,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    started = time.perf_counter()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(seed))
        samples = sample_sequences_in_chunks(
            generator,
            int(config.acceptance_eval_candidate_count),
            batch_size=int(config.generation_sample_batch_size),
            device=device,
            stage_name=f"acceptance eval epoch {int(epoch)}",
            log_fn=lambda message: print(f"[SeqPoison-SBR] {message}", flush=True),
            output_device=torch.device("cpu"),
        )
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
    item_sequences = unpad_generated_tensor(samples)
    candidates = [
        [index + 1, *dataset_bundle.to_canonical_sequence(sequence)]
        for index, sequence in enumerate(item_sequences)
    ]
    result = postprocess_fake_user_sequences(
        candidates,
        target_item=int(dataset_bundle.target_item),
        valid_item_ids=dataset_bundle.valid_item_ids,
        n_fake=int(len(candidates)),
        enforce_single_target=bool(config.enforce_single_target),
        filter_no_target=bool(config.filter_no_target),
        filter_short_sessions=bool(config.filter_short_sessions),
        remove_user_id=True,
    )
    target_containing = int(
        result.counts.get(
            "target_containing_candidate_count_before_single_target_filter",
            0,
        )
    )
    target_stats = target_diagnostics(
        result.valid_sessions,
        target_item=int(dataset_bundle.target_item),
    )
    budget = budget_diagnostics(
        result.valid_sessions,
        target_item=int(dataset_bundle.target_item),
        clean_label_count=1,
    )
    return {
        "phase": "adversarial",
        "epoch": int(epoch),
        "eval_candidate_count": int(len(candidates)),
        "target_containing_candidate_count": target_containing,
        "target_containing_candidate_ratio": (
            0.0 if not candidates else float(target_containing / len(candidates))
        ),
        "n_after_filtering": int(result.counts.get("n_after_filtering", 0)),
        "target_label_pair_count_added": int(
            budget["target_label_pair_count_added"]
        ),
        "target_position_distribution": target_stats["target_position_distribution"],
        "generated_candidate_length_stats": length_stats(candidates),
        "eval_duration_sec": float(time.perf_counter() - started),
    }


__all__ = [
    "EffectiveSeqPoisonTrainingConfig",
    "SeqPoisonTrainer",
    "SeqPoisonTrainerResult",
]
