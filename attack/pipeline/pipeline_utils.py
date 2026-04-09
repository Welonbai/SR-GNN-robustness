from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace

from attack.common.artifact_io import (
    load_fake_sessions,
    load_poison_model,
    load_target_info,
    save_fake_sessions,
    save_poison_model,
    save_target_info,
)
from attack.common.paths import shared_artifact_paths
from attack.data.dataset_serializer import load_srg_nn_train
from attack.data.session_stats import SessionStats, compute_session_stats
from attack.data.target_selector import sample_one_from_popular, sample_one_from_unpopular
from attack.generation.fake_session_generator import FakeSessionGenerator
from attack.generation.fake_session_parameter_sampler import FakeSessionParameterSampler
from attack.models.srgnn_runner import SRGNNRunner

from attack.common.config import Config


def build_default_opt(epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=100,
        hiddenSize=100,
        epoch=epochs,
        lr=0.001,
        lr_dc=0.1,
        lr_dc_step=3,
        l2=1e-5,
        step=1,
        patience=10,
        nonhybrid=False,
    )


def _fake_session_count(ratio: float, clean_count: int) -> int:
    if ratio <= 0:
        return 0
    count = int(round(clean_count * ratio))
    return max(1, count)


def _resolve_target_item(stats: SessionStats, config: Config) -> int:
    mode = config.attack.target_selection_mode
    if mode == "sample_one_from_popular":
        return sample_one_from_popular(stats, seed=config.experiment.seed)
    if mode == "sample_one_from_unpopular":
        return sample_one_from_unpopular(stats, seed=config.experiment.seed)
    raise ValueError("Unsupported target_selection_mode.")


def _load_or_train_poison_runner(
    config: Config,
    *,
    poison_epochs: int,
    target_item: int,
    shared_paths: dict[str, Path],
) -> SRGNNRunner:
    runner = SRGNNRunner(config)
    runner.build_model(build_default_opt(poison_epochs))
    if load_poison_model(runner, shared_paths["poison_model"]):
        print(f"Loaded poison model checkpoint from {shared_paths['poison_model']}")
        return runner

    train_data, test_data = runner.load_dataset()
    if poison_epochs > 0:
        runner.train(
            train_data,
            test_data,
            poison_epochs,
            target_item=target_item,
            topk=config.evaluation.topk,
        )
    save_poison_model(runner, shared_paths["poison_model"])
    return runner


@dataclass(frozen=True)
class SharedAttackArtifacts:
    target_item: int
    stats: SessionStats
    clean_sessions: list[list[int]]
    clean_labels: list[int]
    template_sessions: list[list[int]]
    poison_runner: SRGNNRunner | None
    fake_session_count: int
    shared_paths: dict[str, Path]


def prepare_shared_attack_artifacts(
    config: Config,
    *,
    poison_epochs: int,
    require_poison_runner: bool,
    config_path: str | Path | None = None,
) -> SharedAttackArtifacts:
    shared_paths = shared_artifact_paths(config)
    shared_paths["shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        snapshot_path = shared_paths["config_snapshot"]
        if not snapshot_path.exists():
            shutil.copyfile(config_path, snapshot_path)

    clean_sessions, clean_labels = load_srg_nn_train(config.dataset.train)
    stats = compute_session_stats(clean_sessions)

    target_info = load_target_info(shared_paths["target_info"])
    if target_info is None:
        target_item = _resolve_target_item(stats, config)
        save_target_info(
            shared_paths["target_info"],
            target_item=target_item,
            target_selection_mode=config.attack.target_selection_mode,
            seed=config.experiment.seed,
        )
    else:
        target_item = int(target_info["target_item"])

    template_sessions = load_fake_sessions(shared_paths["fake_sessions"])
    poison_runner = None
    if template_sessions is None:
        poison_runner = _load_or_train_poison_runner(
            config,
            poison_epochs=poison_epochs,
            target_item=target_item,
            shared_paths=shared_paths,
        )
        sampler = FakeSessionParameterSampler(stats)
        generator = FakeSessionGenerator(
            poison_runner,
            sampler,
            topk=config.attack.fake_session_generation_topk,
        )
        fake_count = _fake_session_count(config.attack.size, len(clean_sessions))
        template_sessions = [s.items for s in generator.generate_many(fake_count)]
        save_fake_sessions(template_sessions, shared_paths["fake_sessions"])
    else:
        print(f"Loaded fake sessions from {shared_paths['fake_sessions']}")
        fake_count = len(template_sessions)

    if require_poison_runner and poison_runner is None:
        poison_runner = _load_or_train_poison_runner(
            config,
            poison_epochs=poison_epochs,
            target_item=target_item,
            shared_paths=shared_paths,
        )

    return SharedAttackArtifacts(
        target_item=int(target_item),
        stats=stats,
        clean_sessions=clean_sessions,
        clean_labels=clean_labels,
        template_sessions=template_sessions,
        poison_runner=poison_runner,
        fake_session_count=fake_count,
        shared_paths=shared_paths,
    )


__all__ = [
    "SharedAttackArtifacts",
    "build_default_opt",
    "prepare_shared_attack_artifacts",
]
