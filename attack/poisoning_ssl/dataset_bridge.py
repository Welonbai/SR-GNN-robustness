from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from attack.common.artifact_io import save_json


@dataclass(frozen=True)
class SeqPoisonTrainingData:
    train_path: Path
    user_sequence_count: int
    max_item_id: int
    metadata: dict[str, object]


def export_pseudo_user_sequences(
    train_sub: Sequence[Sequence[int]],
    *,
    target_item: int,
    output_dir: str | Path,
) -> SeqPoisonTrainingData:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "seqpoison_train.txt"
    max_item_id = 0
    with train_path.open("w", encoding="utf-8") as handle:
        for index, session in enumerate(train_sub, start=1):
            items = [int(item) for item in session]
            if items:
                max_item_id = max(max_item_id, max(items))
            handle.write(" ".join([str(index), *[str(item) for item in items]]) + "\n")
    metadata = {
        "target_item": int(target_item),
        "user_sequence_count": int(len(train_sub)),
        "max_item_id": int(max_item_id),
        "item_id_space": "canonical_internal_item_ids",
        "padding_id": 0,
        "synthetic_user_id_start": 1,
        "remap_applied": False,
    }
    save_json(metadata, output / "dataset_bridge_metadata.json")
    return SeqPoisonTrainingData(
        train_path=train_path,
        user_sequence_count=int(len(train_sub)),
        max_item_id=int(max_item_id),
        metadata=metadata,
    )


__all__ = ["SeqPoisonTrainingData", "export_pseudo_user_sequences"]
