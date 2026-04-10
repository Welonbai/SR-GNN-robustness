from __future__ import annotations

from typing import Dict, Type
import importlib

from .base_runner import VictimRunnerBase


_REGISTRY: Dict[str, Type[VictimRunnerBase]] = {}
_DEFAULT_VICTIM_MODULES = (
    "attack.models.victim.srgnn_runner",
    "attack.models.victim.miasrec_runner",
)
_DEFAULTS_LOADED = False


def _ensure_default_victims() -> None:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    for module_name in _DEFAULT_VICTIM_MODULES:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
    _DEFAULTS_LOADED = True


def register_victim(name: str, runner_cls: Type[VictimRunnerBase]) -> None:
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not runner_cls:
        raise ValueError(f"Victim runner already registered for '{name}'.")
    _REGISTRY[name] = runner_cls


def get_victim_runner(name: str) -> Type[VictimRunnerBase]:
    _ensure_default_victims()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown victim runner '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_victims() -> list[str]:
    _ensure_default_victims()
    return sorted(_REGISTRY)


__all__ = ["register_victim", "get_victim_runner", "available_victims"]
