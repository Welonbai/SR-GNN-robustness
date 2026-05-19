from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import pickle

from attack.common.config import (
    AnchorConstructionConfig,
    ArtifactsConfig,
    AttackConfig,
    CanonicalSplitConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    PoisonModelConfig,
    SeedsConfig,
    TargetsConfig,
    VictimsConfig,
)
from attack.common.paths import canonical_split_paths
from attack.data.dataset_specs import resolve_dataset_spec
from attack.data.unified_split import (
    SplitConfig,
    _load_raw_sessions,
    build_canonical_dataset,
)
from attack.models._srgnn_base import _infer_n_node


def _config(tmp_path: Path, *, dataset_name: str, test_days: int = 1) -> Config:
    return Config(
        experiment=ExperimentConfig(name="test"),
        data=DataConfig(
            dataset_name=dataset_name,
            split_protocol="unified",
            poison_train_only=True,
            canonical_split=CanonicalSplitConfig(
                min_item_count=1,
                min_session_len=2,
                valid_ratio=0.1,
                test_days=test_days,
            ),
        ),
        seeds=SeedsConfig(
            fake_session_seed=1,
            target_selection_seed=1,
            position_opt_seed=1,
            surrogate_train_seed=1,
            victim_train_seed=1,
        ),
        attack=AttackConfig(
            size=0.01,
            fake_session_generation_topk=10,
            replacement_topk_ratio=1.0,
            poison_model=PoisonModelConfig(
                name="srgnn",
                params={
                    "train": {
                        "epochs": 1,
                        "batch_size": 2,
                        "hidden_size": 4,
                        "lr": 0.001,
                        "lr_dc": 0.1,
                        "lr_dc_step": 3,
                        "l2": 0.0,
                        "step": 1,
                        "patience": 1,
                        "nonhybrid": False,
                    }
                },
            ),
        ),
        anchor_construction=AnchorConstructionConfig(),
        targets=TargetsConfig(
            mode="sampled",
            explicit_list=(),
            bucket="popular",
            count=1,
            reuse_saved_targets=True,
        ),
        victims=VictimsConfig(enabled=("srgnn",), params={"srgnn": {"train": {}}}),
        evaluation=EvaluationConfig(
            topk=(5,),
            targeted_metrics=("recall",),
            ground_truth_metrics=("recall",),
        ),
        artifacts=ArtifactsConfig(
            root=str(tmp_path / "outputs"),
            shared_dir="shared",
            runs_dir="runs",
        ),
    )


def _write_yoochoose_raw(root: Path, rows: list[tuple[str, str, str, str]]) -> None:
    dataset_dir = root / "yoochoose"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "yoochoose-clicks.dat").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(row) + "\n")


def _large_yoochoose_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    item_ids = [str(i) for i in range(1, 11)]
    for idx in range(130):
        session_id = f"train_{idx:03d}"
        day = 1 + (idx // 20)
        rows.append((session_id, f"2014-04-{day:02d}T10:00:00.000Z", item_ids[idx % 10], "0"))
        rows.append((session_id, f"2014-04-{day:02d}T10:00:00.100Z", item_ids[(idx + 1) % 10], "0"))
    for idx in range(2):
        session_id = f"test_{idx:03d}"
        rows.append((session_id, "2014-04-10T10:00:00.000Z", item_ids[idx], "0"))
        rows.append((session_id, "2014-04-10T10:00:00.100Z", item_ids[idx + 1], "0"))
    return rows


def test_yoochoose_no_header_parser_and_timestamp_units(tmp_path: Path) -> None:
    _write_yoochoose_raw(
        tmp_path,
        [
            ("s1", "2014-04-07T10:51:09.277Z", "10", "0"),
            ("s1", "2014-04-07T10:51:09.100Z", "20", "0"),
        ],
    )
    spec = resolve_dataset_spec("yoochoose1_64", tmp_path)
    assert spec.has_header is False
    assert spec.fieldnames == ["session_id", "timestamp", "item_id", "category"]

    row = {
        "session_id": "s1",
        "timestamp": "2014-04-07T10:51:09.277Z",
        "item_id": "10",
        "category": "0",
    }
    event_seconds = spec.parse_event_date(row)
    _, _, sort_key = spec.extract_session_item(row)
    assert 1_396_000_000 < event_seconds < 1_397_000_000
    assert sort_key == int(event_seconds * 1000)

    sessions, _ = _load_raw_sessions(spec)
    assert sessions["s1"] == ["20", "10"]


def test_diginetica_header_loader_regression(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "diginetica"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "train-item-views.csv").open("w", encoding="utf-8") as handle:
        handle.write("sessionId;userId;itemId;timeframe;eventdate\n")
        handle.write("s1;u1;20;2;2016-05-01\n")
        handle.write("s1;u1;10;1;2016-05-01\n")

    spec = resolve_dataset_spec("diginetica", tmp_path)
    assert spec.has_header is True
    sessions, _ = _load_raw_sessions(spec)
    assert sessions["s1"] == ["10", "20"]


def test_yoochoose_time_split_and_recent_fraction_tail(tmp_path: Path) -> None:
    _write_yoochoose_raw(tmp_path, _large_yoochoose_rows())

    full = build_canonical_dataset(
        _config(tmp_path, dataset_name="yoochoose", test_days=1),
        split_config=SplitConfig(
            min_item_count=1,
            min_session_len=2,
            valid_ratio=0.1,
            test_days=1,
        ),
        dataset_root=tmp_path,
    )
    variant = build_canonical_dataset(
        _config(tmp_path, dataset_name="yoochoose1_64", test_days=1),
        split_config=SplitConfig(
            min_item_count=1,
            min_session_len=2,
            valid_ratio=0.1,
            test_days=1,
        ),
        dataset_root=tmp_path,
    )

    full_train = full.train_sub + full.valid
    variant_train = variant.train_sub + variant.valid
    assert len(variant_train) < len(full_train)
    assert variant_train == full_train[-len(variant_train) :]
    assert variant.metadata["variant"] == "1_64"
    assert variant.metadata["train_tail_fraction"] == 1.0 / 64.0
    assert variant.metadata["train_sessions_before_variant"] == len(full_train)
    assert variant.metadata["train_sessions_after_variant"] == len(variant_train)
    assert len(variant.test) > 0


def test_yoochoose_time_split_uses_configured_test_days(tmp_path: Path) -> None:
    _write_yoochoose_raw(tmp_path, _large_yoochoose_rows())

    one_day = build_canonical_dataset(
        _config(tmp_path, dataset_name="yoochoose", test_days=1),
        split_config=SplitConfig(
            min_item_count=1,
            min_session_len=2,
            valid_ratio=0.1,
            test_days=1,
        ),
        dataset_root=tmp_path,
    )
    four_days = build_canonical_dataset(
        _config(tmp_path, dataset_name="yoochoose", test_days=4),
        split_config=SplitConfig(
            min_item_count=1,
            min_session_len=2,
            valid_ratio=0.1,
            test_days=4,
        ),
        dataset_root=tmp_path,
    )

    assert len(four_days.test) > len(one_day.test)


def test_item_map_metadata_allows_prevariant_item_space(tmp_path: Path) -> None:
    _write_yoochoose_raw(tmp_path, _large_yoochoose_rows())
    dataset = build_canonical_dataset(
        _config(tmp_path, dataset_name="yoochoose1_64"),
        split_config=SplitConfig(
            min_item_count=1,
            min_session_len=2,
            valid_ratio=0.1,
            test_days=1,
        ),
        dataset_root=tmp_path,
    )

    values = sorted(int(item) for item in dataset.item_map.values())
    assert values == list(range(1, len(values) + 1))
    assert dataset.metadata["counts"]["items"] == len(dataset.item_map)
    observed_max = dataset.metadata["observed"]["max_item_id"]
    assert observed_max <= dataset.metadata["counts"]["items"]


def test_srgnn_n_node_prefers_canonical_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path, dataset_name="yoochoose1_64")
    paths = canonical_split_paths(config)
    paths["canonical_dir"].mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump({"item_count": 99, "counts": {"items": 99}}, handle)

    assert _infer_n_node(config, tmp_path / "missing_train.txt") == 100


def test_srgnn_n_node_scans_pickles_when_metadata_missing(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path, dataset_name="synthetic"),
        artifacts=ArtifactsConfig(
            root=str(tmp_path / "empty_outputs"),
            shared_dir="shared",
            runs_dir="runs",
        ),
    )
    train_path = tmp_path / "export" / "train.txt"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with train_path.open("wb") as handle:
        pickle.dump(([[1, 7], [2]], [8, 3]), handle)
    with (train_path.parent / "valid.txt").open("wb") as handle:
        pickle.dump(([[4]], [9]), handle)

    assert _infer_n_node(config, train_path) == 10
