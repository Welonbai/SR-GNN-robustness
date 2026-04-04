from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PoisonedDataset:
    sessions: list[list[int]]
    labels: list[int]
    clean_count: int
    fake_count: int


def expand_session_to_samples(session: Sequence[int]) -> tuple[list[list[int]], list[int]]:
    if len(session) < 2:
        return [], []
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for i in range(1, len(session)):
        target = session[-i]
        prefix = list(session[:-i])
        prefixes.append(prefix)
        labels.append(int(target))
    return prefixes, labels


def build_poisoned_dataset(
    clean_sessions: Sequence[Sequence[int]],
    clean_labels: Sequence[int],
    fake_sessions: Iterable[Sequence[int]],
) -> PoisonedDataset:
    clean_sessions_list = [list(s) for s in clean_sessions]
    clean_labels_list = list(clean_labels)
    fake_sessions_list = [list(s) for s in fake_sessions]

    if len(clean_sessions_list) != len(clean_labels_list):
        raise ValueError("clean_sessions and clean_labels must be the same length.")

    fake_sample_sessions: list[list[int]] = []
    fake_sample_labels: list[int] = []
    for session in fake_sessions_list:
        prefixes, labels = expand_session_to_samples(session)
        fake_sample_sessions.extend(prefixes)
        fake_sample_labels.extend(labels)

    poisoned_sessions = clean_sessions_list + fake_sample_sessions
    poisoned_labels = clean_labels_list + fake_sample_labels

    return PoisonedDataset(
        sessions=poisoned_sessions,
        labels=poisoned_labels,
        clean_count=len(clean_sessions_list),
        fake_count=len(fake_sessions_list),
    )


__all__ = ["PoisonedDataset", "build_poisoned_dataset", "expand_session_to_samples"]
