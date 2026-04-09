from __future__ import annotations

from abc import ABC, abstractmethod


class VictimRunnerBase(ABC):
    name: str

    @abstractmethod
    def build_model(self, opt):
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def score_session(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save_model(self, *args, **kwargs):
        raise NotImplementedError


__all__ = ["VictimRunnerBase"]
