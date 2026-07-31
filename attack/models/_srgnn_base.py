from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import json
import pickle

import numpy as np
import torch

from attack.common.config import Config
from attack.common.paths import canonical_split_paths, dataset_paths
from pytorch_code.model import SessionGraph, train_test, trans_to_cpu, trans_to_cuda, forward as srg_forward
from pytorch_code.utils import Data


def _legacy_infer_n_node(dataset_path: Path) -> int:
    path_str = str(dataset_path).lower()
    if "diginetica" in path_str:
        return 43098
    if "yoochoose1_64" in path_str or "yoochoose1_4" in path_str:
        return 37484
    return 310


def _positive_int(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    if integer <= 0:
        return None
    return integer


def _infer_n_node_from_canonical(config: Config) -> int | None:
    paths = canonical_split_paths(config)
    metadata_path = paths["metadata"]
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        item_count = _positive_int(metadata.get("item_count"))
        if item_count is None:
            counts = metadata.get("counts")
            if isinstance(counts, dict):
                item_count = _positive_int(counts.get("items"))
        if item_count is not None:
            return int(item_count) + 1

    item_map_path = paths["item_map"]
    if item_map_path.exists():
        with item_map_path.open("rb") as handle:
            item_map = pickle.load(handle)
        if isinstance(item_map, dict) and item_map:
            return int(max(int(item) for item in item_map.values())) + 1
    return None


def _infer_n_node_from_export_metadata(dataset_path: Path) -> int | None:
    metadata_path = dataset_path.parent / "export_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    item_count = _positive_int(metadata.get("item_count"))
    if item_count is not None:
        return int(item_count) + 1
    max_item_id = _positive_int(metadata.get("max_item_id"))
    if max_item_id is not None:
        return int(max_item_id) + 1
    return None


def _max_item_id_from_srg_nn_pickle(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if isinstance(data, (list, tuple)) and len(data) == 2:
        sessions, labels = data
    else:
        sessions, labels = data, []
    max_item = 0
    for session in sessions:
        if session:
            max_item = max(max_item, max(int(item) for item in session))
    for label in labels:
        max_item = max(max_item, int(label))
    return int(max_item)


def _infer_n_node_from_pickles(dataset_path: Path) -> int | None:
    candidates = [dataset_path]
    for name in ("valid.txt", "test.txt"):
        candidate = dataset_path.parent / name
        if candidate != dataset_path:
            candidates.append(candidate)
    max_item = 0
    for candidate in candidates:
        max_item = max(max_item, _max_item_id_from_srg_nn_pickle(candidate))
    if max_item <= 0:
        return None
    return int(max_item) + 1


def _infer_n_node(config: Config, dataset_path: Path) -> int:
    return (
        _infer_n_node_from_canonical(config)
        or _infer_n_node_from_export_metadata(dataset_path)
        or _infer_n_node_from_pickles(dataset_path)
        or _legacy_infer_n_node(dataset_path)
    )


class SRGNNBaseRunner:
    def __init__(self, config: Config, base_dir: str | Path | None = None, n_node: int | None = None) -> None:
        self.config = config
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.n_node = n_node or _infer_n_node(
            config,
            self._resolve_path(dataset_paths(config)["train"]),
        )
        self.model: SessionGraph | None = None
        self.opt = None
        self.train_loss_history: list[float] = []

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
        train_losses: list[float] = []
        if target_item is not None:
            from attack.pipeline.core.evaluator import evaluate_targeted_metrics
        for epoch in range(epochs):
            print(f"epoch {epoch + 1}/{epochs}")
            hit, mrr, avg_loss = train_test(
                self.model,
                train_data,
                test_data,
                log_batches=False,
            )
            train_losses.append(float(avg_loss))
            if target_item is not None:
                rankings = self.predict_topk(test_data, topk=topk)
                metrics, _ = evaluate_targeted_metrics(
                    rankings,
                    target_item=target_item,
                    metrics=["precision"],
                    topk=[topk],
                )
                targeted = metrics.get(f"targeted_precision@{topk}", 0.0)
                print(f"epoch {epoch + 1}/{epochs} targeted_p@{topk}={targeted:.4f}")
            print(f"epoch {epoch + 1}/{epochs} metrics: hit={hit:.4f} mrr={mrr:.4f}")
            history.append((hit, mrr))
        self.train_loss_history = train_losses
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

    def predict_topk(self, test_data: Data, topk: int = 20) -> list[list[int]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.model.eval()
        rankings: list[list[int]] = []
        slices = test_data.generate_batch(self.model.batch_size)
        with torch.no_grad():
            for i in slices:
                _, scores = srg_forward(self.model, i, test_data)
                k = min(topk, scores.shape[1])
                topk_indices = scores.topk(k)[1]
                topk_indices = trans_to_cpu(topk_indices).detach().numpy()
                for row in topk_indices:
                    rankings.append([int(item) + 1 for item in row.tolist()])
        return rankings

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call build_model() first.")
        if not session:
            raise ValueError("Session must contain at least one item.")
        # The label is unused for scoring, but keep it inside SR-GNN's valid
        # one-based item range so batch validation remains strict everywhere.
        data = Data(([list(session)], [1]), shuffle=False)
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
