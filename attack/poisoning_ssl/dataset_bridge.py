from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from attack.common.artifact_io import save_json


@dataclass(frozen=True)
class SeqPoisonDatasetBundle:
    train_path: Path
    train_sequences: list[list[int]]
    canonical_train_sequences: list[list[int]]
    target_item: int
    seqpoison_target_item: int
    valid_item_ids: set[int]
    seqpoison_valid_item_ids: set[int]
    max_item_id: int
    vocab_size: int
    mask_id: int
    max_seq_len: int
    remap_used: bool
    canonical_to_seqpoison: dict[int, int]
    seqpoison_to_canonical: dict[int, int]
    item_id_mapping_path: Path | None
    metadata: dict[str, object]

    @property
    def user_sequence_count(self) -> int:
        return int(len(self.train_sequences))

    def to_canonical_sequence(self, sequence: Sequence[int]) -> list[int]:
        if not self.remap_used:
            return [int(item) for item in sequence]
        return [
            0 if int(item) == 0 else int(self.seqpoison_to_canonical[int(item)])
            for item in sequence
        ]


SeqPoisonTrainingData = SeqPoisonDatasetBundle


def export_pseudo_user_sequences(
    train_sub: Sequence[Sequence[int]],
    *,
    target_item: int,
    output_dir: str | Path,
    valid_item_ids: set[int] | None = None,
    max_seq_len: int | None = None,
    max_train_sequences: int | None = None,
) -> SeqPoisonDatasetBundle:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    max_len = None if max_seq_len is None else int(max_seq_len)
    canonical_sequences_all = [
        [int(item) for item in session if int(item) > 0]
        for session in train_sub
        if session
    ]
    canonical_sequences = [
        list(session)
        for session in canonical_sequences_all
        if max_len is None or len(session) <= max_len
    ]
    length_valid_count = int(len(canonical_sequences))
    if max_train_sequences is not None:
        canonical_sequences = canonical_sequences[: int(max_train_sequences)]
    if not canonical_sequences:
        raise ValueError(
            "SeqPoison-SBR dataset bridge has no train_sub sessions after "
            "length filtering."
        )
    valid_items = (
        {int(item) for item in valid_item_ids if int(item) > 0}
        if valid_item_ids is not None
        else {item for session in canonical_sequences_all for item in session}
    )
    valid_items.add(int(target_item))
    remap_used = _requires_remap(valid_items)
    if remap_used:
        canonical_to_seqpoison = {
            canonical: index
            for index, canonical in enumerate(sorted(valid_items), start=1)
        }
        seqpoison_to_canonical = {
            seqpoison: canonical
            for canonical, seqpoison in canonical_to_seqpoison.items()
        }
    else:
        canonical_to_seqpoison = {item: item for item in sorted(valid_items)}
        seqpoison_to_canonical = {item: item for item in sorted(valid_items)}

    train_sequences = [
        [canonical_to_seqpoison[int(item)] for item in session]
        for session in canonical_sequences
    ]
    seqpoison_target = int(canonical_to_seqpoison[int(target_item)])
    seqpoison_valid_ids = {int(value) for value in canonical_to_seqpoison.values()}
    max_item_id = int(max(seqpoison_valid_ids))
    mask_id = int(max_item_id + 1)
    train_path = output / "seqpoison_train.txt"
    with train_path.open("w", encoding="utf-8") as handle:
        for index, session in enumerate(train_sequences, start=1):
            handle.write(" ".join([str(index), *[str(item) for item in session]]) + "\n")

    mapping_path: Path | None = None
    if remap_used:
        mapping_path = output / "item_id_mapping.json"
        save_json(
            {
                "canonical_to_seqpoison": {
                    str(key): int(value) for key, value in canonical_to_seqpoison.items()
                },
                "seqpoison_to_canonical": {
                    str(key): int(value) for key, value in seqpoison_to_canonical.items()
                },
            },
            mapping_path,
        )

    before = int(len(canonical_sequences_all))
    after_length_filter = int(length_valid_count)
    excluded = int(before - after_length_filter)
    metadata = {
        "target_item": int(target_item),
        "seqpoison_target_item": int(seqpoison_target),
        "user_sequence_count": int(len(canonical_sequences)),
        "max_item_id": max_item_id,
        "vocab_size": int(max_item_id + 1),
        "mask_id": mask_id,
        "item_id_space": (
            "dense_seqpoison_item_ids" if remap_used else "canonical_internal_item_ids"
        ),
        "padding_id": 0,
        "start_letter": 0,
        "synthetic_user_id_start": 1,
        "remap_applied": bool(remap_used),
        "item_id_mapping_path": None if mapping_path is None else str(mapping_path),
        "train_session_count_before_length_filter": before,
        "train_session_count_after_length_filter": after_length_filter,
        "excluded_train_session_count": excluded,
        "excluded_train_session_ratio": 0.0 if before <= 0 else float(excluded / before),
        "max_seq_len_value": None if max_len is None else int(max_len),
        "diagnostic_max_train_sequences": (
            None if max_train_sequences is None else int(max_train_sequences)
        ),
        "train_sequence_count_used_for_training": int(len(canonical_sequences)),
    }
    save_json(metadata, output / "dataset_bridge_metadata.json")
    return SeqPoisonDatasetBundle(
        train_path=train_path,
        train_sequences=train_sequences,
        canonical_train_sequences=canonical_sequences,
        target_item=int(target_item),
        seqpoison_target_item=seqpoison_target,
        valid_item_ids=valid_items,
        seqpoison_valid_item_ids=seqpoison_valid_ids,
        max_item_id=max_item_id,
        vocab_size=int(max_item_id + 1),
        mask_id=mask_id,
        max_seq_len=0 if max_len is None else int(max_len),
        remap_used=bool(remap_used),
        canonical_to_seqpoison=canonical_to_seqpoison,
        seqpoison_to_canonical=seqpoison_to_canonical,
        item_id_mapping_path=mapping_path,
        metadata=metadata,
    )


def _requires_remap(valid_item_ids: set[int]) -> bool:
    if not valid_item_ids:
        return False
    positives = sorted(int(item) for item in valid_item_ids if int(item) > 0)
    return positives != list(range(1, int(max(positives)) + 1))


__all__ = [
    "SeqPoisonDatasetBundle",
    "SeqPoisonTrainingData",
    "export_pseudo_user_sequences",
]
