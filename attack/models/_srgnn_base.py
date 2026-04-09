from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pickle

import numpy as np
import torch

from attack.common.config import Config
from attack.common.paths import dataset_paths
from pytorch_code.model import SessionGraph, train_test, trans_to_cpu, trans_to_cuda, forward as srg_forward
from pytorch_code.utils import Data


def _infer_n_node(dataset_path: Path) -> int:
    path_str = str(dataset_path).lower()
    if "diginetica" in path_str:
        return 43098
    if "yoochoose1_64" in path_str or "yoochoose1_4" in path_str:
        return 37484
    return 310


class SRGNNBaseRunner:
    def __init__(self, config: Config, base_dir: str | Path | None = None, n_node: int | None = None) -> None:
        self.config = config
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.n_node = n_node or _infer_n_node(self._resolve_path(dataset_paths(config)["train"]))
        self.model: SessionGraph | None = None
        self.opt = None

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (self.base_dir / path)

    def build_model(self, opt) -> SessionGraph:
        self.opt = opt
        self.model = trans_to_cuda(SessionGraph(opt, self.n_node))
        return self.model

    def load_dataset(
        self,
        train_path: str | Path | None = None,
        test_path: str | Path | None = None,
        shuffle_train: bool = True,
    ) -> tuple[Data, Data]:
        paths = dataset_paths(self.config)
        train_path = self._resolve_path(train_path or paths["train"])
        test_path = self._resolve_path(test_path or paths["test"])
        train_data = pickle.load(train_path.open("rb"))
        test_data = pickle.load(test_path.open("rb"))
        return Data(train_data, shuffle=shuffle_train), Data(test_data, shuffle=False)

    def train(
        self,
        train_data: Data,
        test_data: Data,
        epochs: int,
        target_item: int | None = None,
        topk: int = 20,
    ) -> list[tuple[float, float]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        history: list[tuple[float, float]] = []
        if target_item is not None:
            from attack.pipeline.evaluator import evaluate_targeted_precision_at_k
        for epoch in range(epochs):
            print(f"epoch {epoch + 1}/{epochs}")
            hit, mrr = train_test(self.model, train_data, test_data)
            if target_item is not None:
                targeted = evaluate_targeted_precision_at_k(
                    self, test_data, target_item=target_item, topk=topk
                )
                print(
                    f"epoch {epoch + 1}/{epochs} targeted_p@{topk}={targeted:.4f}"
                )
            print(f"epoch {epoch + 1}/{epochs} metrics: hit={hit:.4f} mrr={mrr:.4f}")
            history.append((hit, mrr))
        return history

    def evaluate(self, test_data: Data, topk: int = 20) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        self.model.eval()
        hit, mrr = [], []
        slices = test_data.generate_batch(self.model.batch_size)
        with torch.no_grad():
            for i in slices:
                targets, scores = srg_forward(self.model, i, test_data)
                sub_scores = scores.topk(topk)[1]
                sub_scores = trans_to_cpu(sub_scores).detach().numpy()
                for score, target in zip(sub_scores, targets):
                    hit.append(np.isin(target - 1, score))
                    if len(np.where(score == target - 1)[0]) == 0:
                        mrr.append(0)
                    else:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
        return float(np.mean(hit) * 100), float(np.mean(mrr) * 100)

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        if not session:
            raise ValueError("Session must contain at least one item.")
        data = Data(([list(session)], [0]), shuffle=False)
        with torch.no_grad():
            _, scores = srg_forward(self.model, np.array([0]), data)
        return trans_to_cpu(scores.squeeze(0).detach())

    def load_model(self, checkpoint_path: str | Path, map_location: str | None = None) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        state = torch.load(self._resolve_path(checkpoint_path), map_location=map_location)
        self.model.load_state_dict(state)

    def save_model(self, checkpoint_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        path = self._resolve_path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


__all__ = ["SRGNNBaseRunner"]
