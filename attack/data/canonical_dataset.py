from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import pickle


@dataclass(frozen=True)
class CanonicalDataset:
    train_sub: list[list[int]]
    valid: list[list[int]]
    test: list[list[int]]
    item_map: dict[Any, int]
    metadata: dict[str, Any]


def save_canonical_dataset(dataset: CanonicalDataset, paths: dict[str, Path]) -> None:
    base = paths["canonical_dir"]
    base.mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(dataset.metadata, handle, indent=2, sort_keys=True)
    with paths["item_map"].open("wb") as handle:
        pickle.dump(dataset.item_map, handle)
    with paths["train_sub"].open("wb") as handle:
        pickle.dump(dataset.train_sub, handle)
    with paths["valid"].open("wb") as handle:
        pickle.dump(dataset.valid, handle)
    with paths["test"].open("wb") as handle:
        pickle.dump(dataset.test, handle)


def load_canonical_dataset(paths: dict[str, Path]) -> CanonicalDataset:
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with paths["item_map"].open("rb") as handle:
        item_map = pickle.load(handle)
    with paths["train_sub"].open("rb") as handle:
        train_sub = pickle.load(handle)
    with paths["valid"].open("rb") as handle:
        valid = pickle.load(handle)
    with paths["test"].open("rb") as handle:
        test = pickle.load(handle)
    return CanonicalDataset(
        train_sub=train_sub,
        valid=valid,
        test=test,
        item_map=item_map,
        metadata=metadata,
    )


def canonical_dataset_exists(paths: dict[str, Path]) -> bool:
    return (
        paths["metadata"].exists()
        and paths["item_map"].exists()
        and paths["train_sub"].exists()
        and paths["valid"].exists()
        and paths["test"].exists()
    )


__all__ = [
    "CanonicalDataset",
    "save_canonical_dataset",
    "load_canonical_dataset",
    "canonical_dataset_exists",
]
