from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import pickle


def load_srg_nn_train(path: str | Path) -> tuple[list[list[int]], list[int]]:
    data = pickle.load(Path(path).open("rb"))
    if not (isinstance(data, (list, tuple)) and len(data) == 2):
        raise ValueError("Expected SR-GNN train format: (sessions, labels).")
    sessions, labels = data
    return [list(s) for s in sessions], [int(l) for l in labels]


def save_srg_nn_train(path: str | Path, sessions: Sequence[Sequence[int]], labels: Sequence[int]) -> None:
    if len(sessions) != len(labels):
        raise ValueError("sessions and labels must be the same length.")
    data = (list(map(list, sessions)), list(map(int, labels)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(data, path.open("wb"))


__all__ = ["load_srg_nn_train", "save_srg_nn_train"]
