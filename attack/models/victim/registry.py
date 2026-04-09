from __future__ import annotations

from typing import Dict, Type

from .base_runner import VictimRunnerBase


_REGISTRY: Dict[str, Type[VictimRunnerBase]] = {}


def register_victim(name: str, runner_cls: Type[VictimRunnerBase]) -> None:
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not runner_cls:
        raise ValueError(f"Victim runner already registered for '{name}'.")
    _REGISTRY[name] = runner_cls


def get_victim_runner(name: str) -> Type[VictimRunnerBase]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown victim runner '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_victims() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["register_victim", "get_victim_runner", "available_victims"]
