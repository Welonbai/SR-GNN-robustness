from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import load_config
from attack.common.paths import canonical_split_paths, split_key
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.session_stats import compute_session_stats
from attack.data.unified_split import ensure_canonical_dataset


def _expanded_pair_count(sessions: Sequence[Sequence[int]]) -> int:
    return sum(max(0, len(session) - 1) for session in sessions)


def _average_length(sessions: Sequence[Sequence[int]]) -> float:
    if not sessions:
        return 0.0
    return float(mean(len(session) for session in sessions))


def _max_observed_item_id(dataset: CanonicalDataset) -> int:
    max_item = 0
    for sessions in (dataset.train_sub, dataset.valid, dataset.test):
        for session in sessions:
            if session:
                max_item = max(max_item, max(int(item) for item in session))
    return int(max_item)


def _popular_pool_size(dataset: CanonicalDataset) -> int:
    stats = compute_session_stats(dataset.train_sub)
    if not stats.item_counts:
        return 0
    avg_count = stats.total_items / float(len(stats.item_counts))
    return sum(1 for count in stats.item_counts.values() if int(count) > avg_count)


def _unpopular_pool_size(dataset: CanonicalDataset, *, threshold: int = 10) -> int:
    stats = compute_session_stats(dataset.train_sub)
    return sum(1 for count in stats.item_counts.values() if int(count) < int(threshold))


def build_diagnostic_payload(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    dataset = ensure_canonical_dataset(config)
    metadata = dataset.metadata
    paths = canonical_split_paths(config, split_key=split_key(config))
    item_count = int(len(dataset.item_map))
    max_item_id = int(max(dataset.item_map.values(), default=0))

    return {
        "canonical_path": str(paths["canonical_dir"]),
        "dataset_name": metadata.get("dataset_name", config.data.dataset_name),
        "source_dataset": metadata.get("source_dataset"),
        "variant": metadata.get("variant"),
        "train_tail_fraction": metadata.get("train_tail_fraction"),
        "raw_session_count": metadata.get("raw_session_count"),
        "filtered_session_count": metadata.get("filtered_session_count"),
        "train_sessions_before_variant": metadata.get("train_sessions_before_variant"),
        "train_sessions_after_variant": metadata.get("train_sessions_after_variant"),
        "expanded_pairs_before_variant": metadata.get("expanded_pairs_before_variant"),
        "expanded_pairs_after_variant": metadata.get("expanded_pairs_after_variant"),
        "train_sub_session_count": len(dataset.train_sub),
        "valid_session_count": len(dataset.valid),
        "test_session_count": len(dataset.test),
        "expanded_train_sub_pair_count": _expanded_pair_count(dataset.train_sub),
        "expanded_valid_pair_count": _expanded_pair_count(dataset.valid),
        "expanded_test_pair_count": _expanded_pair_count(dataset.test),
        "item_count": item_count,
        "max_item_id": max_item_id,
        "max_exported_item_id": _max_observed_item_id(dataset),
        "average_train_sub_session_length": _average_length(dataset.train_sub),
        "average_valid_session_length": _average_length(dataset.valid),
        "average_test_session_length": _average_length(dataset.test),
        "popular_target_pool_size": _popular_pool_size(dataset),
        "unpopular_target_pool_size": _unpopular_pool_size(dataset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    payload = build_diagnostic_payload(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
