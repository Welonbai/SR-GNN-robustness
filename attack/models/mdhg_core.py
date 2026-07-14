from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import sys

import numpy as np
import torch
from torch import nn

from attack.common.seed import set_seed
from attack.models.mdhg_constants import (
    MDHG_ADAPTER_VERSION,
    MDHG_TRAIN_DATA_CONSTRUCTION_MODE,
)


def _mdhg_src_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "mdhg"


def _ensure_mdhg_import_path() -> None:
    src = str(_mdhg_src_dir())
    if src not in sys.path:
        sys.path.insert(0, src)


def _import_mdhg_module(module_name: str):
    _ensure_mdhg_import_path()
    module = importlib.import_module(module_name)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    src_dir = _mdhg_src_dir().resolve()
    if src_dir not in module_path.parents and module_path != src_dir:
        raise ImportError(
            f"Imported {module_name!r} from {module_path}, expected it under {src_dir}."
        )
    return module


@dataclass(frozen=True)
class MDHGTrainRows:
    sessions: list[list[int]]
    labels: list[int]
    raw_train_sessions: list[list[int]]
    row_fingerprint: tuple[tuple[tuple[int, ...], int], ...]
    raw_session_fingerprint: tuple[tuple[int, ...], ...]


class MDHGInProcessModel:
    """In-process wrapper for the original CUDA-only MDHG implementation.

    MDHG constructs several tensors with torch.cuda.FloatTensor directly, so this
    adapter deliberately fails when CUDA is not available instead of pretending
    a CPU fallback exists.
    """

    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        item_count: int,
        seed: int,
        dataset_name: str,
        use_gpu: bool,
        gpu_id: str | int = "0",
    ) -> None:
        self.train_config = dict(train_config)
        self.item_count = _positive_int(item_count, "item_count")
        self.seed = int(seed)
        self.dataset_name = str(dataset_name)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = str(gpu_id)
        self.args = self._build_args()
        self.model: nn.Module | None = None
        self.raw_train_sessions: list[list[int]] | None = None
        self.train_loss_history: list[float] = []

    @property
    def batch_size(self) -> int:
        return int(self.train_config["batch_size"])

    def train_pairs(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        *,
        raw_train_sessions: Sequence[Sequence[int]],
        epochs: int | None = None,
    ) -> list[float]:
        rows = build_mdhg_train_rows(
            sessions,
            labels,
            raw_train_sessions=raw_train_sessions,
            item_count=self.item_count,
        )
        if not rows.sessions:
            raise ValueError("MDHG training data must not be empty.")
        epoch_count = int(epochs if epochs is not None else self.train_config["epochs"])
        if epoch_count <= 0:
            raise ValueError("MDHG epochs must be positive.")

        self._build_model_from_raw_sessions(rows.raw_train_sessions)
        assert self.model is not None
        model_module = _import_mdhg_module("model")
        train_data = _build_mdhg_data(
            (rows.sessions, rows.labels),
            rows.raw_train_sessions,
            n_node=self.item_count,
        )
        losses: list[float] = []
        for epoch in range(epoch_count):
            self.model.train()
            total = 0.0
            batches = 0
            for batch_indices, real_count in _full_batch_indices(
                len(rows.sessions),
                self.batch_size,
            ):
                if real_count <= 0:
                    continue
                tar, _scores, con_loss, loss_item, loss_diff = model_module.forward(
                    self.model,
                    batch_indices,
                    train_data,
                    epoch,
                    True,
                )
                del tar, _scores
                loss = loss_item + con_loss + loss_diff
                if not torch.isfinite(loss).all().item():
                    raise RuntimeError("MDHG training produced non-finite loss.")
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                total += float(loss.item())
                batches += 1
            if batches <= 0:
                raise RuntimeError("MDHG training produced no batches.")
            losses.append(float(total / batches))
        self.train_loss_history = losses
        return losses

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("MDHG model is not initialized.")
        normalized = _normalize_score_session(session, item_count=self.item_count)
        batch_sessions = _pad_sessions_to_batch([normalized], self.batch_size)
        batch_labels = [1 for _session in batch_sessions]
        score_data = _MDHGSliceData(batch_sessions, batch_labels)
        model_module = _import_mdhg_module("model")
        self.model.eval()
        with torch.no_grad():
            _tar, scores, _con_loss, _loss_item, _loss_diff = model_module.forward(
                self.model,
                np.arange(self.batch_size),
                score_data,
                0,
                False,
            )
            canonical_scores = scores[0, : self.item_count]
            if canonical_scores.shape != (self.item_count,):
                raise RuntimeError("MDHG score vector has unexpected shape.")
            if not torch.isfinite(canonical_scores).all().item():
                raise RuntimeError("MDHG score_session produced non-finite scores.")
            return canonical_scores.detach().cpu()

    def score_sessions_topk(
        self,
        sessions: Sequence[Sequence[int]],
        *,
        topk: int,
    ) -> list[list[int]]:
        if self.model is None:
            raise RuntimeError("MDHG model is not initialized.")
        normalized = [
            _normalize_score_session(session, item_count=self.item_count)
            for session in sessions
        ]
        k = min(int(topk), self.item_count)
        if k <= 0:
            raise ValueError("topk must be positive.")
        rankings: list[list[int]] = []
        model_module = _import_mdhg_module("model")
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                real_sessions = normalized[start : start + self.batch_size]
                if not real_sessions:
                    continue
                batch_sessions = _pad_sessions_to_batch(real_sessions, self.batch_size)
                batch_labels = [1 for _session in batch_sessions]
                score_data = _MDHGSliceData(batch_sessions, batch_labels)
                _tar, scores, _con_loss, _loss_item, _loss_diff = model_module.forward(
                    self.model,
                    np.arange(self.batch_size),
                    score_data,
                    0,
                    False,
                )
                canonical_scores = scores[: len(real_sessions), : self.item_count]
                if not torch.isfinite(canonical_scores).all().item():
                    raise RuntimeError("MDHG batched scoring produced non-finite scores.")
                top_indices = torch.topk(
                    canonical_scores,
                    k=k,
                    dim=1,
                    largest=True,
                    sorted=True,
                ).indices
                rankings.extend(
                    [[int(index) + 1 for index in row] for row in top_indices.detach().cpu().tolist()]
                )
        if len(rankings) != len(normalized):
            raise RuntimeError("MDHG scoring missed one or more sessions.")
        return rankings

    def save_model(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("MDHG model is not initialized.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
        }
        torch.save(state, destination)

    def load_model(
        self,
        path: str | Path,
        *,
        raw_train_sessions: Sequence[Sequence[int]],
    ) -> None:
        rows = build_mdhg_train_rows(
            _minimal_training_prefixes(raw_train_sessions),
            _minimal_training_labels(raw_train_sessions),
            raw_train_sessions=raw_train_sessions,
            item_count=self.item_count,
        )
        self._build_model_from_raw_sessions(rows.raw_train_sessions)
        assert self.model is not None
        state = torch.load(Path(path), map_location="cpu")
        if not isinstance(state, Mapping):
            raise TypeError("MDHG checkpoint must contain a state_dict mapping.")
        self.model.load_state_dict(state)

    def cleanup(self) -> None:
        if self.model is not None:
            optimizer = getattr(self.model, "optimizer", None)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            self.model.cpu()
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_model_from_raw_sessions(
        self,
        raw_train_sessions: Sequence[Sequence[int]],
    ) -> None:
        self._require_cuda_device()
        raw_rows = [
            _normalize_raw_session(
                session,
                item_count=self.item_count,
                field=f"raw_train_sessions[{index}]",
            )
            for index, session in enumerate(raw_train_sessions)
        ]
        if not raw_rows:
            raise ValueError("MDHG raw_train_sessions must not be empty.")
        set_seed(self.seed)
        util_module = _import_mdhg_module("util")
        model_module = _import_mdhg_module("model")
        graph_data = util_module.Data(
            (_minimal_training_prefixes(raw_rows), _minimal_training_labels(raw_rows)),
            raw_rows,
            shuffle=False,
            n_node=self.item_count,
        )
        self.model = model_module.trans_to_cuda(
            model_module.MDHG(
                R=graph_data.R,
                adj1=graph_data.adj1,
                adj2=graph_data.adj2,
                adjacency=graph_data.adjacency,
                adjacency_T=graph_data.adjacency_T,
                adjacency1=graph_data.adjacency1,
                R1=graph_data.R1,
                n_node=self.item_count,
                lr=float(self.args.lr),
                l2=float(self.args.l2),
                beta=float(self.args.beta),
                lam=float(self.args.lam),
                eps=float(self.args.eps),
                layers=int(self.args.layer),
                emb_size=int(self.args.embSize),
                batch_size=self.batch_size,
                dataset=self.dataset_name,
                K1=int(self.args.K1),
                K2=int(self.args.K2),
                K3=int(self.args.K3),
                dropout=float(self.args.dropout),
                alpha=float(self.args.alpha),
            )
        )
        _reset_parameters_like_mdhg_main(self.model)
        self.raw_train_sessions = raw_rows

    def _require_cuda_device(self) -> None:
        if not self.use_gpu:
            raise RuntimeError("MDHG in-process adapter is CUDA-only; use_gpu must be true.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "MDHG in-process adapter requires CUDA because the original model "
                "constructs torch.cuda.FloatTensor values directly."
            )
        try:
            device_index = int(self.gpu_id)
        except ValueError as exc:
            raise ValueError(f"MDHG gpu_id must be an integer, got {self.gpu_id!r}.") from exc
        if device_index < 0:
            raise ValueError("MDHG gpu_id must be non-negative.")
        device_count = int(torch.cuda.device_count())
        if device_count > 0 and device_index >= device_count:
            raise ValueError(
                f"MDHG gpu_id {device_index} is unavailable; CUDA device_count={device_count}."
            )
        torch.cuda.set_device(device_index)

    def _build_args(self) -> SimpleNamespace:
        train = self.train_config
        return SimpleNamespace(
            epoch=int(train["epochs"]),
            batchSize=int(train["batch_size"]),
            embSize=int(train.get("embSize", train.get("emb_size", 100))),
            l2=float(train.get("l2", 1e-5)),
            lr=float(train["lr"]),
            layer=int(train.get("layer", train.get("layers", 2))),
            beta=float(train.get("beta", 0.005)),
            lam=float(train.get("lam", 0.0001)),
            eps=float(train.get("eps", 0.2)),
            K1=int(train.get("K1", 80)),
            K2=int(train.get("K2", 50)),
            K3=int(train.get("K3", 20)),
            dropout=float(train.get("dropout", 0.5)),
            alpha=float(train.get("alpha", 0.2)),
        )


class _MDHGSliceData:
    def __init__(self, sessions: Sequence[Sequence[int]], labels: Sequence[int]) -> None:
        self.raw = np.asarray([list(session) for session in sessions], dtype=object)
        self.targets = np.asarray([int(label) for label in labels])

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = int(np.max(num_node))
        session_len = []
        reversed_sess_item = []
        mask = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            row = list(session)
            items.append(row + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(row)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask


def build_mdhg_train_rows(
    sessions: Sequence[Sequence[int]],
    labels: Sequence[int],
    *,
    raw_train_sessions: Sequence[Sequence[int]],
    item_count: int,
) -> MDHGTrainRows:
    raw_sessions_list = list(sessions)
    raw_labels_list = list(labels)
    if len(raw_sessions_list) != len(raw_labels_list):
        raise ValueError("MDHG sessions and labels must have equal lengths.")
    normalized_sessions = [
        _normalize_train_session(
            session,
            item_count=item_count,
            field=f"train.sessions[{index}]",
        )
        for index, session in enumerate(raw_sessions_list)
    ]
    normalized_labels = [
        _item_id(label, item_count=item_count, field=f"train.labels[{index}]")
        for index, label in enumerate(raw_labels_list)
    ]
    normalized_raw = [
        _normalize_raw_session(
            session,
            item_count=item_count,
            field=f"raw_train_sessions[{index}]",
        )
        for index, session in enumerate(raw_train_sessions)
    ]
    if not normalized_sessions:
        raise ValueError("MDHG train rows must not be empty.")
    if not normalized_raw:
        raise ValueError("MDHG raw train sessions must not be empty.")
    return MDHGTrainRows(
        sessions=normalized_sessions,
        labels=normalized_labels,
        raw_train_sessions=normalized_raw,
        row_fingerprint=tuple(
            (tuple(session), int(label))
            for session, label in zip(normalized_sessions, normalized_labels)
        ),
        raw_session_fingerprint=tuple(tuple(session) for session in normalized_raw),
    )


def _build_mdhg_data(
    data: tuple[Sequence[Sequence[int]], Sequence[int]],
    raw_train_sessions: Sequence[Sequence[int]],
    *,
    n_node: int,
):
    util_module = _import_mdhg_module("util")
    return util_module.Data(data, raw_train_sessions, shuffle=False, n_node=int(n_node))


def _full_batch_indices(length: int, batch_size: int) -> list[tuple[np.ndarray, int]]:
    length = int(length)
    batch_size = int(batch_size)
    if length <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    order = np.arange(length)
    batches: list[tuple[np.ndarray, int]] = []
    for start in range(0, length, batch_size):
        indices = order[start : start + batch_size]
        real_count = len(indices)
        if real_count < batch_size:
            if length >= batch_size:
                indices = order[-batch_size:]
            else:
                repeats = np.resize(order, batch_size)
                indices = repeats
        batches.append((np.asarray(indices, dtype=np.int64), int(real_count)))
    return batches


def _pad_sessions_to_batch(
    sessions: Sequence[Sequence[int]],
    batch_size: int,
) -> list[list[int]]:
    rows = [list(session) for session in sessions]
    if not rows:
        raise ValueError("MDHG scoring sessions must not be empty.")
    while len(rows) < int(batch_size):
        rows.append(list(rows[-1]))
    return rows


def _minimal_training_prefixes(raw_train_sessions: Sequence[Sequence[int]]) -> list[list[int]]:
    prefixes: list[list[int]] = []
    for session in raw_train_sessions:
        row = list(session)
        prefixes.append(row[:-1] if len(row) >= 2 else [row[0]])
    return prefixes


def _minimal_training_labels(raw_train_sessions: Sequence[Sequence[int]]) -> list[int]:
    labels: list[int] = []
    for session in raw_train_sessions:
        row = list(session)
        labels.append(int(row[-1]))
    return labels


def _normalize_score_session(
    session: Sequence[int],
    *,
    item_count: int,
) -> list[int]:
    if isinstance(session, (str, bytes)) or not isinstance(session, Sequence):
        raise TypeError("MDHG score_session input must be a sequence of item IDs.")
    if not session:
        raise ValueError("MDHG score_session does not support empty sessions.")
    return [
        _item_id(item, item_count=item_count, field=f"score_session[{index}]")
        for index, item in enumerate(session)
    ]


def _normalize_train_session(
    session: Sequence[int],
    *,
    item_count: int,
    field: str,
) -> list[int]:
    if isinstance(session, (str, bytes)) or not isinstance(session, Sequence):
        raise TypeError(f"MDHG {field} must be a sequence.")
    if not session:
        raise ValueError(f"MDHG {field} must not be empty.")
    return [
        _item_id(item, item_count=item_count, field=f"{field}[{index}]")
        for index, item in enumerate(session)
    ]


def _normalize_raw_session(
    session: Sequence[int],
    *,
    item_count: int,
    field: str,
) -> list[int]:
    return _normalize_train_session(session, item_count=item_count, field=field)


def _item_id(value: Any, *, item_count: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"MDHG {field} must be an integer.")
    item = int(value)
    if item < 1 or item > int(item_count):
        raise ValueError(
            f"MDHG {field}={item} is outside canonical item range 1..{item_count}; "
            "item 0 is reserved for padding."
        )
    return item


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"MDHG {field} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"MDHG {field} must be positive.")
    return int(value)


def _reset_parameters_like_mdhg_main(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.xavier_uniform_(layer.weight)


__all__ = [
    "MDHG_ADAPTER_VERSION",
    "MDHG_TRAIN_DATA_CONSTRUCTION_MODE",
    "MDHGInProcessModel",
    "MDHGTrainRows",
    "build_mdhg_train_rows",
]
