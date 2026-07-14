from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from attack.common.config import Config
from attack.models.mdhg_core import MDHGInProcessModel


class MDHGPoisonRunner:
    name = "mdhg"

    def __init__(
        self,
        config: Config,
        *,
        item_count: int,
        seed: int,
        raw_train_sessions: Sequence[Sequence[int]],
    ) -> None:
        self.config = config
        self.item_count = int(item_count)
        self.seed = int(seed)
        self.raw_train_sessions = [list(session) for session in raw_train_sessions]
        self.train_config = dict(config.attack.poison_model.params["train"])
        runtime = (config.victims.runtime or {}).get("mdhg", {})
        device = runtime.get("device", {}) if isinstance(runtime, Mapping) else {}
        self.use_gpu = bool(device.get("use_gpu", False))
        self.gpu_id = device.get("gpu_id", "0")
        self.model: MDHGInProcessModel | None = None
        self.train_loss_history: list[float] = []

    def build_model(self, opt=None) -> MDHGInProcessModel:
        del opt
        self.model = MDHGInProcessModel(
            train_config=self.train_config,
            item_count=self.item_count,
            seed=self.seed,
            dataset_name=self.config.data.dataset_name,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
        )
        return self.model

    def train_pairs(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        *,
        raw_train_sessions: Sequence[Sequence[int]] | None = None,
        epochs: int | None = None,
    ) -> list[float]:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        raw_sessions = (
            self.raw_train_sessions
            if raw_train_sessions is None
            else [list(session) for session in raw_train_sessions]
        )
        losses = self.model.train_pairs(
            sessions,
            labels,
            raw_train_sessions=raw_sessions,
            epochs=epochs,
        )
        self.train_loss_history = list(losses)
        return losses

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("MDHG poison model is not initialized.")
        return self.model.score_session(session)

    def save_model(self, checkpoint_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("MDHG poison model is not initialized.")
        self.model.save_model(checkpoint_path)

    def load_model(self, checkpoint_path: str | Path, map_location: str | None = None) -> None:
        del map_location
        if self.model is None:
            self.build_model()
        assert self.model is not None
        self.model.load_model(checkpoint_path, raw_train_sessions=self.raw_train_sessions)

    def cleanup(self) -> None:
        if self.model is not None:
            self.model.cleanup()


__all__ = ["MDHGPoisonRunner"]
