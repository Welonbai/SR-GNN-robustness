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
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.data.session_stats import SessionStats, compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset
from attack.data.target_selector import (
    sample_many_from_all,
    sample_many_from_popular,
    sample_many_from_unpopular,
)
from attack.generation.fake_session_generator import FakeSessionGenerator
from attack.generation.fake_session_parameter_sampler import FakeSessionParameterSampler
from attack.models.poison.srgnn_poison_runner import SRGNNPoisonRunner

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


def build_clean_pairs(canonical_dataset: CanonicalDataset) -> tuple[list[list[int]], list[int]]:
    clean_sessions: list[list[int]] = []
    clean_labels: list[int] = []
    for session in canonical_dataset.train_sub:
        prefixes, labels = expand_session_to_samples(session)
        clean_sessions.extend(prefixes)
        clean_labels.extend(labels)
    return clean_sessions, clean_labels


def _resolve_target_items(stats: SessionStats, config: Config) -> list[int]:
    mode = config.targets.mode
    if mode == "explicit_list":
        return [int(item) for item in config.targets.explicit_list]
    if mode == "sampled":
        seed = config.seeds.target_selection_seed
        count = config.targets.count
        if config.targets.bucket == "popular":
            return sample_many_from_popular(stats, seed=seed, count=count)
        if config.targets.bucket == "unpopular":
            return sample_many_from_unpopular(stats, seed=seed, count=count)
        if config.targets.bucket == "all":
            return sample_many_from_all(stats, seed=seed, count=count)
        raise ValueError("Unsupported targets.bucket.")
    raise ValueError("Unsupported targets.mode.")


def resolve_target_items(
    stats: SessionStats,
    config: Config,
    *,
    shared_paths: dict[str, Path] | None = None,
) -> list[int]:
    if shared_paths is None:
        return _resolve_target_items(stats, config)

    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    target_info = None
    if config.targets.reuse_saved_targets:
        target_info = load_target_info(shared_paths["target_info"])
    if target_info is None:
        target_items = _resolve_target_items(stats, config)
        save_target_info(
            shared_paths["target_info"],
            target_items=target_items,
            target_selection_mode=config.targets.mode,
            seed=config.seeds.target_selection_seed,
            bucket=config.targets.bucket if config.targets.mode == "sampled" else None,
            count=config.targets.count if config.targets.mode == "sampled" else None,
            explicit_list=list(config.targets.explicit_list),
        )
    else:
        if "target_items" in target_info:
            target_items = [int(item) for item in target_info["target_items"]]
        elif "target_item" in target_info:
            target_items = [int(target_info["target_item"])]
        else:
            raise ValueError("target_info.json is missing target_items.")
    return [int(item) for item in target_items]


def _export_srg_nn_dataset(
    *,
    dataset: CanonicalDataset,
    export_dir: Path,
) -> dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    train_path = export_dir / "train.txt"
    valid_path = export_dir / "valid.txt"
    test_path = export_dir / "test.txt"
    if train_path.exists() and valid_path.exists() and test_path.exists():
        return {"train": train_path, "valid": valid_path, "test": test_path}
    exporter = SRGNNExporter()
    result = exporter.export(dataset, export_dir)
    return result.files


def _load_or_train_poison_runner(
    config: Config,
    *,
    poison_epochs: int,
    shared_paths: dict[str, Path],
    export_paths: dict[str, Path],
) -> SRGNNPoisonRunner:
    runner = SRGNNPoisonRunner(config)
    runner.build_model(build_default_opt(poison_epochs))
    if load_poison_model(runner, shared_paths["poison_model"]):
        print(f"Loaded poison model checkpoint from {shared_paths['poison_model']}")
        return runner

    print(f"No poison model checkpoint found. Training new poison model for {poison_epochs} epochs.")
    train_data, test_data = runner.load_dataset(
        train_path=export_paths["train"],
        test_path=export_paths["valid"],
    )
    if poison_epochs > 0:
        runner.train(
            train_data,
            test_data,
            poison_epochs,
            topk=config.evaluation.topk,
        )
    save_poison_model(runner, shared_paths["poison_model"])
    print(f"Saved poison model checkpoint to {shared_paths['poison_model']}")
    return runner


@dataclass(frozen=True)
class SharedAttackArtifacts:
    stats: SessionStats
    clean_sessions: list[list[int]]
    clean_labels: list[int]
    canonical_dataset: CanonicalDataset
    export_paths: dict[str, Path]
    template_sessions: list[list[int]]
    poison_runner: SRGNNPoisonRunner | None
    fake_session_count: int
    shared_paths: dict[str, Path]


def prepare_shared_attack_artifacts(
    config: Config,
    *,
    poison_epochs: int,
    require_poison_runner: bool,
    config_path: str | Path | None = None,
) -> SharedAttackArtifacts:
    canonical_dataset = ensure_canonical_dataset(config)
    shared_paths = shared_artifact_paths(config)
    shared_paths["attack_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        snapshot_path = shared_paths["attack_config_snapshot"]
        if not snapshot_path.exists():
            shutil.copyfile(config_path, snapshot_path)

    stats = compute_session_stats(canonical_dataset.train_sub)
    clean_sessions, clean_labels = build_clean_pairs(canonical_dataset)

    export_dir = shared_paths["attack_shared_dir"] / "export"
    export_paths = _export_srg_nn_dataset(
        dataset=canonical_dataset,
        export_dir=export_dir,
    )

    template_sessions = load_fake_sessions(shared_paths["fake_sessions"])
    poison_runner = None
    if template_sessions is None:
        print("No fake sessions found. Generating new fake sessions.")
        poison_runner = _load_or_train_poison_runner(
            config,
            poison_epochs=poison_epochs,
            shared_paths=shared_paths,
            export_paths=export_paths,
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
        print(f"Saved fake sessions to {shared_paths['fake_sessions']}")
    else:
        print(f"Loaded fake sessions from {shared_paths['fake_sessions']}")
        fake_count = len(template_sessions)

    if require_poison_runner and poison_runner is None:
        poison_runner = _load_or_train_poison_runner(
            config,
            poison_epochs=poison_epochs,
            shared_paths=shared_paths,
            export_paths=export_paths,
        )

    return SharedAttackArtifacts(
        stats=stats,
        clean_sessions=clean_sessions,
        clean_labels=clean_labels,
        canonical_dataset=canonical_dataset,
        export_paths=export_paths,
        template_sessions=template_sessions,
        poison_runner=poison_runner,
        fake_session_count=fake_count,
        shared_paths=shared_paths,
    )


__all__ = [
    "SharedAttackArtifacts",
    "build_default_opt",
    "build_clean_pairs",
    "prepare_shared_attack_artifacts",
    "resolve_target_items",
]
