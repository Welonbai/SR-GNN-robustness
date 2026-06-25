from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from attack.common.seed import set_seed
from attack.data.poisoned_dataset_builder import PoisonedDataset


FREQREC_ADAPTER_VERSION = 1
FREQREC_TRAIN_DATA_CONSTRUCTION_MODE = "canonical_prefix_label_pairs_v1"


def _freqrec_src_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "third_party" / "freqrec" / "src"


def _ensure_freqrec_import_path() -> None:
    src = str(_freqrec_src_dir())
    if src not in sys.path:
        sys.path.insert(0, src)


def _import_freqrec_module(module_name: str):
    _ensure_freqrec_import_path()
    module = importlib.import_module(module_name)
    module_path = Path(getattr(module, "__file__", "")).resolve()
    src_dir = _freqrec_src_dir().resolve()
    if src_dir not in module_path.parents and module_path != src_dir:
        raise ImportError(
            f"Imported {module_name!r} from {module_path}, expected it under {src_dir}."
        )
    return module


_dataset_module = _import_freqrec_module("dataset")
_canonical_io_module = _import_freqrec_module("canonical_io")
_model_module = _import_freqrec_module("model")

CanonicalRecDataset = _dataset_module.CanonicalRecDataset
CanonicalRecord = _canonical_io_module.CanonicalRecord
MODEL_DICT = _model_module.MODEL_DICT


@dataclass(frozen=True)
class FreqRecTrainRows:
    sessions: list[list[int]]
    labels: list[int]
    row_fingerprint: tuple[tuple[tuple[int, ...], int], ...]


class FreqRecInProcessModel:
    def __init__(
        self,
        *,
        train_config: Mapping[str, Any],
        item_count: int,
        seed: int,
        use_gpu: bool,
        gpu_id: str | int = "0",
        num_workers: int = 0,
    ) -> None:
        self.train_config = dict(train_config)
        self.item_count = _positive_int(item_count, "item_count")
        self.seed = int(seed)
        self.use_gpu = bool(use_gpu)
        self.gpu_id = str(gpu_id)
        self.num_workers = int(num_workers)
        if self.num_workers < 0:
            raise ValueError("FreqRec num_workers must be non-negative.")
        self.args = self._build_args()
        self.device = _resolve_freqrec_device(use_gpu=self.use_gpu, gpu_id=self.gpu_id)
        set_seed(self.seed)
        self.model = MODEL_DICT["freqrec"](args=self.args).to(self.device)
        betas = (
            float(self.train_config["adam_beta1"]),
            float(self.train_config["adam_beta2"]),
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(self.train_config["lr"]),
            betas=betas,
            weight_decay=float(self.train_config["weight_decay"]),
        )
        self.train_loss_history: list[float] = []

    @property
    def max_seq_length(self) -> int:
        return int(self.train_config["max_seq_length"])

    def train_pairs(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        *,
        epochs: int | None = None,
    ) -> list[float]:
        rows = build_freqrec_train_rows(
            sessions,
            labels,
            item_count=self.item_count,
        )
        if not rows.sessions:
            raise ValueError("FreqRec training data must not be empty.")
        epoch_count = int(epochs if epochs is not None else self.train_config["epochs"])
        if epoch_count <= 0:
            raise ValueError("FreqRec epochs must be positive.")
        dataset = _PrefixLabelDataset(rows.sessions, rows.labels, self.max_seq_length)
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset,
            sampler=RandomSampler(dataset, generator=generator),
            batch_size=int(self.train_config["batch_size"]),
            num_workers=self.num_workers,
            drop_last=False,
        )
        losses: list[float] = []
        for _epoch in range(epoch_count):
            self.model.train()
            total = 0.0
            batches = 0
            for batch in loader:
                batch = tuple(t.to(self.device) for t in batch)
                example_ids, input_ids, answers, neg_answer, same_target = batch
                loss = self.model.calculate_loss(
                    input_ids,
                    answers,
                    neg_answer,
                    same_target,
                    example_ids,
                )
                if not torch.isfinite(loss).all().item():
                    raise RuntimeError("FreqRec training produced non-finite loss.")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += float(loss.item())
                batches += 1
            if batches <= 0:
                raise RuntimeError("FreqRec training loader produced no batches.")
            losses.append(float(total / batches))
        self.train_loss_history = losses
        return losses

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        normalized = _normalize_score_session(
            session,
            item_count=self.item_count,
            max_seq_length=self.max_seq_length,
        )
        pad_len = self.max_seq_length - len(normalized)
        input_ids = ([0] * pad_len) + normalized
        tensor = torch.as_tensor([input_ids], dtype=torch.long, device=self.device)
        self.model.eval()
        with torch.no_grad():
            sequence_output, _ = self.model.predict(
                tensor,
                torch.as_tensor([0], dtype=torch.long, device=self.device),
            )
            final = sequence_output[:, -1, :]
            scores = torch.matmul(final, self.model.item_embeddings.weight.transpose(0, 1))
            canonical_scores = scores[:, 1 : self.item_count + 1].squeeze(0)
            if canonical_scores.shape != (self.item_count,):
                raise RuntimeError("FreqRec score vector has unexpected shape.")
            if not torch.isfinite(canonical_scores).all().item():
                raise RuntimeError("FreqRec score_session produced non-finite scores.")
            return canonical_scores.detach().cpu()

    def score_sessions_topk(
        self,
        sessions: Sequence[Sequence[int]],
        *,
        topk: int,
    ) -> list[list[int]]:
        normalized = [
            _normalize_score_session(
                session,
                item_count=self.item_count,
                max_seq_length=self.max_seq_length,
            )
            for session in sessions
        ]
        dataset = _PrefixOnlyDataset(normalized, self.max_seq_length)
        loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=int(self.train_config["batch_size"]),
            num_workers=self.num_workers,
            drop_last=False,
        )
        k = min(int(topk), self.item_count)
        if k <= 0:
            raise ValueError("topk must be positive.")
        rankings: list[list[int] | None] = [None] * len(normalized)
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                example_ids, input_ids = (t.to(self.device) for t in batch)
                sequence_output, _ = self.model.predict(input_ids, example_ids)
                final = sequence_output[:, -1, :]
                scores = torch.matmul(final, self.model.item_embeddings.weight.transpose(0, 1))
                canonical_scores = scores[:, 1 : self.item_count + 1]
                if not torch.isfinite(canonical_scores).all().item():
                    raise RuntimeError("FreqRec batched scoring produced non-finite scores.")
                indices = torch.topk(canonical_scores, k=k, dim=1, largest=True, sorted=True).indices
                for example_id, row in zip(example_ids.detach().cpu().tolist(), indices.detach().cpu().tolist()):
                    rankings[int(example_id)] = [int(index) + 1 for index in row]
        if any(row is None for row in rankings):
            raise RuntimeError("FreqRec scoring missed one or more sessions.")
        return [list(row) for row in rankings if row is not None]

    def save_model(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.cpu().state_dict(), destination)
        self.model.to(self.device)

    def load_model(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location=self.device)
        if not isinstance(state, Mapping):
            raise TypeError("FreqRec checkpoint must contain a state_dict mapping.")
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def cleanup(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_args(self) -> SimpleNamespace:
        train = self.train_config
        return SimpleNamespace(
            model_type="freqrec",
            item_count=self.item_count,
            item_size=self.item_count + 1,
            num_users=1,
            max_seq_length=int(train["max_seq_length"]),
            batch_size=int(train["batch_size"]),
            hidden_size=int(train["hidden_size"]),
            num_hidden_layers=int(train["num_hidden_layers"]),
            num_attention_heads=int(train["num_attention_heads"]),
            hidden_act=str(train["hidden_act"]),
            attention_probs_dropout_prob=float(train["attention_probs_dropout_prob"]),
            hidden_dropout_prob=float(train["hidden_dropout_prob"]),
            initializer_range=float(train["initializer_range"]),
            alpha=float(train["alpha"]),
            gama=float(train["gama"]),
            alpha_loss=float(train["alpha_loss"]),
            fft_loss_type=str(train["fft_loss_type"]),
            chux=str(train["chux"]),
            fourier_loss=bool(train["fourier_loss"]),
            adam_beta1=float(train["adam_beta1"]),
            adam_beta2=float(train["adam_beta2"]),
            weight_decay=float(train["weight_decay"]),
            lr=float(train["lr"]),
            fre=float(train["fre"]),
            no_cuda=not self.use_gpu,
            gpu_id=self.gpu_id,
            seed=self.seed,
            num_workers=self.num_workers,
            log_freq=1,
        )


class _PrefixLabelDataset(CanonicalRecDataset):
    def __init__(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        max_seq_length: int,
    ) -> None:
        records = [
            CanonicalRecord(
                example_id=int(index),
                input_prefix=tuple(int(item) for item in session),
                label=int(label),
            )
            for index, (session, label) in enumerate(zip(sessions, labels))
        ]
        super().__init__(records, max_seq_length)


class _PrefixOnlyDataset(torch.utils.data.Dataset):
    def __init__(self, sessions: Sequence[Sequence[int]], max_seq_length: int) -> None:
        self.sessions = [list(session) for session in sessions]
        self.max_seq_length = int(max_seq_length)

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(self, index: int):
        session = self.sessions[int(index)]
        input_ids = session[-self.max_seq_length :]
        input_ids = ([0] * (self.max_seq_length - len(input_ids))) + input_ids
        return (
            torch.tensor(int(index), dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
        )


def build_freqrec_train_rows(
    sessions: Sequence[Sequence[int]] | PoisonedDataset,
    labels: Sequence[int] | None = None,
    *,
    item_count: int,
) -> FreqRecTrainRows:
    if isinstance(sessions, PoisonedDataset):
        if labels is not None:
            raise ValueError("labels must be omitted when sessions is a PoisonedDataset.")
        raw_sessions = sessions.sessions
        raw_labels = sessions.labels
    else:
        if labels is None:
            raise ValueError("labels are required for FreqRec training rows.")
        raw_sessions = sessions
        raw_labels = labels
    raw_labels_list = list(raw_labels)
    raw_sessions_list = list(raw_sessions)
    if len(raw_sessions_list) != len(raw_labels_list):
        raise ValueError("FreqRec sessions and labels must have equal lengths.")
    normalized_sessions: list[list[int]] = []
    normalized_labels: list[int] = []
    for row_index, (session, label) in enumerate(zip(raw_sessions_list, raw_labels_list)):
        normalized_session = _normalize_train_session(
            session,
            item_count=item_count,
            field=f"train.sessions[{row_index}]",
        )
        label_id = _item_id(label, item_count=item_count, field=f"train.labels[{row_index}]")
        normalized_sessions.append(normalized_session)
        normalized_labels.append(label_id)
    fingerprint = tuple(
        (tuple(session), int(label))
        for session, label in zip(normalized_sessions, normalized_labels)
    )
    return FreqRecTrainRows(
        sessions=normalized_sessions,
        labels=normalized_labels,
        row_fingerprint=fingerprint,
    )


def _normalize_score_session(
    session: Sequence[int],
    *,
    item_count: int,
    max_seq_length: int,
) -> list[int]:
    if isinstance(session, (str, bytes)) or not isinstance(session, Sequence):
        raise TypeError("FreqRec score_session input must be a sequence of item IDs.")
    if not session:
        raise ValueError("FreqRec score_session does not support empty sessions.")
    normalized = [
        _item_id(item, item_count=item_count, field=f"score_session[{index}]")
        for index, item in enumerate(session)
    ]
    return normalized[-int(max_seq_length) :]


def _normalize_train_session(
    session: Sequence[int],
    *,
    item_count: int,
    field: str,
) -> list[int]:
    if isinstance(session, (str, bytes)) or not isinstance(session, Sequence):
        raise TypeError(f"FreqRec {field} must be a sequence.")
    if not session:
        raise ValueError(f"FreqRec {field} must not be empty.")
    return [
        _item_id(item, item_count=item_count, field=f"{field}[{index}]")
        for index, item in enumerate(session)
    ]


def _item_id(value: Any, *, item_count: int, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"FreqRec {field} must be an integer.")
    item = int(value)
    if item < 1 or item > int(item_count):
        raise ValueError(
            f"FreqRec {field}={item} is outside canonical item range 1..{item_count}."
        )
    return item


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"FreqRec {field} must be an integer.")
    if int(value) <= 0:
        raise ValueError(f"FreqRec {field} must be positive.")
    return int(value)


def _resolve_freqrec_device(*, use_gpu: bool, gpu_id: str | int | None) -> torch.device:
    if not bool(use_gpu):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    raw_gpu_id = "0" if gpu_id is None or str(gpu_id).strip() == "" else str(gpu_id)
    try:
        device_index = int(raw_gpu_id)
    except ValueError as exc:
        raise ValueError(f"FreqRec gpu_id must be an integer, got {gpu_id!r}.") from exc
    if device_index < 0:
        raise ValueError(f"FreqRec gpu_id must be non-negative, got {device_index}.")
    device_count = int(torch.cuda.device_count())
    if device_count > 0 and device_index >= device_count:
        raise ValueError(
            f"FreqRec gpu_id {device_index} is unavailable; CUDA device_count={device_count}."
        )
    return torch.device(f"cuda:{device_index}")


__all__ = [
    "FREQREC_ADAPTER_VERSION",
    "FREQREC_TRAIN_DATA_CONSTRUCTION_MODE",
    "FreqRecInProcessModel",
    "FreqRecTrainRows",
    "build_freqrec_train_rows",
    "_resolve_freqrec_device",
]
