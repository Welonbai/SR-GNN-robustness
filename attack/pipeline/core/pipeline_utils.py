from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Mapping

from attack.common.artifact_io import (
    load_json,
    load_fake_sessions,
    load_poison_model,
    load_selected_targets,
    load_target_info,
    save_fake_sessions,
    save_poison_model,
    save_selected_targets,
    save_target_selection_meta,
    save_target_info,
)
from attack.common.paths import shared_artifact_paths, split_key, target_selection_key
from attack.common.seed import derive_seed, set_seed
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
from attack.pipeline.core.train_history import save_train_history

from attack.common.config import Config


def build_srgnn_opt_from_train_config(train_config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=int(train_config["batch_size"]),
        hiddenSize=int(train_config["hidden_size"]),
        epoch=int(train_config["epochs"]),
        lr=float(train_config["lr"]),
        lr_dc=float(train_config["lr_dc"]),
        lr_dc_step=int(train_config["lr_dc_step"]),
        l2=float(train_config["l2"]),
        step=int(train_config["step"]),
        patience=int(train_config["patience"]),
        nonhybrid=bool(train_config["nonhybrid"]),
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


def _popular_pool(stats: SessionStats) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    avg_count = stats.total_items / float(len(stats.item_counts))
    pool = [int(item) for item, count in stats.item_counts.items() if count > avg_count]
    if not pool:
        raise ValueError("Popular pool is empty under item_count > average_count.")
    return pool


def _unpopular_pool(stats: SessionStats, *, threshold: int = 10) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    pool = [int(item) for item, count in stats.item_counts.items() if count < threshold]
    if not pool:
        raise ValueError("Unpopular pool is empty under item_count < 10.")
    return pool


def _all_pool(stats: SessionStats) -> list[int]:
    if not stats.item_counts:
        raise ValueError("item_counts is empty; cannot select target items.")
    return [int(item) for item in stats.item_counts]


def _target_candidate_pool(stats: SessionStats, config: Config) -> list[int]:
    if config.targets.mode == "explicit_list":
        return [int(item) for item in config.targets.explicit_list]
    if config.targets.bucket == "popular":
        return _popular_pool(stats)
    if config.targets.bucket == "unpopular":
        return _unpopular_pool(stats)
    if config.targets.bucket == "all":
        return _all_pool(stats)
    raise ValueError("Unsupported targets.bucket.")


def _candidate_pool_summary(stats: SessionStats, pool: list[int]) -> dict[str, Any]:
    counts = [int(stats.item_counts[item]) for item in pool if item in stats.item_counts]
    summary: dict[str, Any] = {
        "preview": [int(item) for item in sorted(pool)[:10]],
    }
    if counts:
        summary.update(
            {
                "min_item_count": int(min(counts)),
                "max_item_count": int(max(counts)),
                "avg_item_count": float(sum(counts) / len(counts)),
            }
        )
    return summary


def _target_selection_meta_payload(
    stats: SessionStats,
    config: Config,
    *,
    candidate_pool: list[int],
) -> dict[str, Any]:
    explicit_list = [int(item) for item in config.targets.explicit_list]
    bucket = config.targets.bucket if config.targets.mode == "sampled" else None
    count = int(config.targets.count) if config.targets.mode == "sampled" else None
    return {
        "target_selection_seed": int(config.seeds.target_selection_seed),
        "targets": {
            "mode": config.targets.mode,
            "bucket": bucket,
            "count": count,
            "explicit_list": explicit_list,
        },
        "bucket": bucket,
        "count": count,
        "explicit_list": explicit_list,
        "candidate_pool_size": int(len(candidate_pool)),
        "candidate_pool_summary": _candidate_pool_summary(stats, candidate_pool),
        "target_selection_key": target_selection_key(config),
        "split_key": split_key(config),
    }


def resolve_target_items(
    stats: SessionStats,
    config: Config,
    *,
    shared_paths: dict[str, Path] | None = None,
) -> list[int]:
    if shared_paths is None:
        return _resolve_target_items(stats, config)

    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    selected_targets = None
    if config.targets.reuse_saved_targets:
        selected_targets = load_selected_targets(shared_paths["selected_targets"])
        if selected_targets is None:
            legacy_target_info = load_target_info(shared_paths["target_info"])
            if legacy_target_info is not None:
                raw_target_items = legacy_target_info.get("target_items")
                if isinstance(raw_target_items, list):
                    selected_targets = [int(item) for item in raw_target_items]
                    save_selected_targets(shared_paths["selected_targets"], selected_targets)

    candidate_pool = _target_candidate_pool(stats, config)
    meta_payload = _target_selection_meta_payload(
        stats,
        config,
        candidate_pool=candidate_pool,
    )

    if selected_targets is None:
        target_items = _resolve_target_items(stats, config)
        save_selected_targets(shared_paths["selected_targets"], target_items)
        save_target_selection_meta(shared_paths["target_selection_meta"], meta_payload)
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
        target_items = [int(item) for item in selected_targets]
        if load_json(shared_paths["target_selection_meta"]) is None:
            save_target_selection_meta(shared_paths["target_selection_meta"], meta_payload)
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
    shared_paths: dict[str, Path],
    export_paths: dict[str, Path],
) -> SRGNNPoisonRunner:
    poison_train_config = _poison_train_config(config)
    configured_poison_epochs = int(poison_train_config["epochs"])
    runner = SRGNNPoisonRunner(config)
    runner.build_model(build_srgnn_opt_from_train_config(poison_train_config))
    if load_poison_model(runner, shared_paths["poison_model"]):
        print(f"Loaded poison model checkpoint from {shared_paths['poison_model']}")
        return runner

    print(
        "No poison model checkpoint found. "
        f"Training new poison model for {configured_poison_epochs} epochs."
    )
    train_data, test_data = runner.load_dataset(
        train_path=export_paths["train"],
        test_path=export_paths["valid"],
    )
    if configured_poison_epochs > 0:
        runner.train(
            train_data,
            test_data,
            configured_poison_epochs,
            topk=max(config.evaluation.topk),
        )
    save_poison_model(runner, shared_paths["poison_model"])
    print(f"Saved poison model checkpoint to {shared_paths['poison_model']}")
    if runner.train_loss_history:
        save_train_history(
            shared_paths["poison_train_history"],
            role="poison",
            model="srgnn",
            epochs=len(runner.train_loss_history),
            train_loss=runner.train_loss_history,
            valid_loss=[None] * len(runner.train_loss_history),
            notes="valid_loss not available for SRGNN poison training.",
        )
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
    run_type: str,
    require_poison_runner: bool,
    config_path: str | Path | None = None,
) -> SharedAttackArtifacts:
    generation_seed = int(config.seeds.fake_session_seed)
    canonical_dataset = ensure_canonical_dataset(config)
    shared_paths = shared_artifact_paths(config, run_type=run_type)
    shared_paths["attack_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        snapshot_path = shared_paths["attack_config_snapshot"]
        if not snapshot_path.exists():
            shutil.copyfile(config_path, snapshot_path)
    shared_paths["target_shared_dir"].mkdir(parents=True, exist_ok=True)
    if config_path:
        target_snapshot_path = shared_paths["target_config_snapshot"]
        if not target_snapshot_path.exists():
            shutil.copyfile(config_path, target_snapshot_path)

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
        set_seed(derive_seed(generation_seed, "poison_model_generation"))
        poison_runner = _load_or_train_poison_runner(
            config,
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
        set_seed(derive_seed(generation_seed, "fake_session_generation"))
        template_sessions = [s.items for s in generator.generate_many(fake_count)]
        save_fake_sessions(template_sessions, shared_paths["fake_sessions"])
        print(f"Saved fake sessions to {shared_paths['fake_sessions']}")
    else:
        print(f"Loaded fake sessions from {shared_paths['fake_sessions']}")
        fake_count = len(template_sessions)

    if require_poison_runner and poison_runner is None:
        set_seed(derive_seed(generation_seed, "poison_model_generation"))
        poison_runner = _load_or_train_poison_runner(
            config,
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


def _poison_train_config(config: Config) -> dict[str, Any]:
    return dict(config.attack.poison_model.params["train"])


__all__ = [
    "SharedAttackArtifacts",
    "build_srgnn_opt_from_train_config",
    "build_clean_pairs",
    "prepare_shared_attack_artifacts",
    "resolve_target_items",
]
