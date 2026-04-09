from __future__ import annotations

from attack.models._srgnn_base import SRGNNBaseRunner


class SRGNNPoisonRunner(SRGNNBaseRunner):
    name = "srgnn"
    role = "poison"


__all__ = ["SRGNNPoisonRunner"]
