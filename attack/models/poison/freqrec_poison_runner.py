from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from attack.common.config import Config
from attack.models.freqrec_core import FreqRecInProcessModel


class FreqRecPoisonRunner:
    name = "freqrec"

    def __init__(
        self,
        config: Config,
        *,
        item_count: int,
        seed: int,
    ) -> None:
        self.config = config
        self.item_count = int(item_count)
        self.seed = int(seed)
        self.train_config = dict(config.attack.poison_model.params["train"])
        runtime = (config.victims.runtime or {}).get("freqrec", {})
        device = runtime.get("device", {}) if isinstance(runtime, dict) else {}
        dataloader = runtime.get("dataloader", {}) if isinstance(runtime, dict) else {}
        self.use_gpu = bool(device.get("use_gpu", False))
        self.gpu_id = device.get("gpu_id", "0")
        self.num_workers = int(dataloader.get("num_workers", 0))
        self.model: FreqRecInProcessModel | None = None
        self.train_loss_history: list[float] = []

    def build_model(self, opt=None) -> FreqRecInProcessModel:
        del opt
        self.model = FreqRecInProcessModel(
            train_config=self.train_config,
            item_count=self.item_count,
            seed=self.seed,
            use_gpu=self.use_gpu,
            gpu_id=self.gpu_id,
            num_workers=self.num_workers,
        )
        return self.model

    def train_pairs(
        self,
        sessions: Sequence[Sequence[int]],
        labels: Sequence[int],
        *,
        epochs: int | None = None,
    ) -> list[float]:
        if self.model is None:
            self.build_model()
        assert self.model is not None
        losses = self.model.train_pairs(sessions, labels, epochs=epochs)
        self.train_loss_history = list(losses)
        return losses

    def score_session(self, session: Sequence[int]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("FreqRec poison model is not initialized.")
        return self.model.score_session(session)

    def save_model(self, checkpoint_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("FreqRec poison model is not initialized.")
        self.model.save_model(checkpoint_path)

    def load_model(self, checkpoint_path: str | Path, map_location: str | None = None) -> None:
        del map_location
        if self.model is None:
            self.build_model()
        assert self.model is not None
        self.model.load_model(checkpoint_path)


__all__ = ["FreqRecPoisonRunner"]
