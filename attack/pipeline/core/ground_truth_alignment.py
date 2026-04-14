from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from attack.common.config import Config
from attack.data.canonical_dataset import CanonicalDataset


def resolve_ground_truth_labels(
    config: Config,
    *,
    victim_name: str,
    canonical_dataset: CanonicalDataset,
    predictions: Sequence[Sequence[int]] | None,
) -> list[int] | None:
    """Rebuild ground-truth labels aligned with one victim's exported predictions."""
    if predictions is None:
        return None

    if victim_name == "srgnn":
        labels = _sequence_prefix_labels(canonical_dataset.test)
    elif victim_name == "miasrec":
        labels = _sequence_prefix_labels(canonical_dataset.test)
    elif victim_name == "tron":
        labels = _resolve_tron_ground_truth_labels(config, canonical_dataset)
    else:
        raise ValueError(f"Unsupported victim model for ground-truth alignment: {victim_name}")

    prediction_count = len(predictions)
    label_count = len(labels)
    if label_count != prediction_count:
        raise ValueError(
            f"Ground-truth label alignment mismatch for victim '{victim_name}': "
            f"{label_count} labels vs {prediction_count} predictions."
        )
    return labels


def _sequence_prefix_labels(sequences: Sequence[Sequence[int]]) -> list[int]:
    labels: list[int] = []
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        for suffix_length in range(1, len(sequence)):
            labels.append(int(sequence[-suffix_length]))
    return labels


def _resolve_tron_ground_truth_labels(
    config: Config,
    canonical_dataset: CanonicalDataset,
) -> list[int]:
    tron_config = _load_tron_runtime_config(config)
    batch_size = int(tron_config["batch_size"])
    max_session_length = int(tron_config["max_session_length"])
    if batch_size <= 0:
        raise ValueError("TRON batch_size must be positive for ground-truth alignment.")
    if max_session_length <= 0:
        raise ValueError("TRON max_session_length must be positive for ground-truth alignment.")

    session_samples = [
        _build_tron_session_sample(sequence, max_session_length=max_session_length)
        for sequence in canonical_dataset.test
    ]
    session_samples = [sample for sample in session_samples if sample is not None]

    aligned_labels: list[int] = []
    for start in range(0, len(session_samples), batch_size):
        batch = session_samples[start : start + batch_size]
        padded_labels: list[list[int]] = []
        padded_masks: list[list[int]] = []
        for labels in batch:
            session_len = len(labels)
            pad_len = max_session_length - session_len
            padded_labels.append(([0] * pad_len) + labels)
            padded_masks.append(([0] * pad_len) + ([1] * session_len))

        for time_index in range(max_session_length):
            for row_index in range(len(batch)):
                if padded_masks[row_index][time_index] <= 0:
                    continue
                aligned_labels.append(int(padded_labels[row_index][time_index]))
    return aligned_labels


def _build_tron_session_sample(
    sequence: Sequence[int],
    *,
    max_session_length: int,
) -> list[int] | None:
    if len(sequence) < 2:
        return None
    truncated = list(sequence[-(max_session_length + 1) :])
    labels = truncated[1:]
    if not labels:
        return None
    return [int(label) for label in labels]


def _load_tron_runtime_config(config: Config) -> dict[str, object]:
    runtime = (config.victims.runtime or {}).get("tron")
    if not isinstance(runtime, dict):
        raise ValueError("Missing victims.runtime.tron configuration.")
    repo_root = runtime.get("repo_root")
    if not isinstance(repo_root, str) or not repo_root.strip():
        raise ValueError("Missing victims.runtime.tron.repo_root configuration.")

    config_path = (
        Path(repo_root).resolve()
        / "configs"
        / "tron"
        / f"{config.data.dataset_name}.json"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"TRON config not found for ground-truth alignment: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"TRON config must be a JSON object: {config_path}")
    missing = [key for key in ("batch_size", "max_session_length") if key not in payload]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"TRON config missing required keys for alignment: {joined}")
    return payload


__all__ = ["resolve_ground_truth_labels"]
