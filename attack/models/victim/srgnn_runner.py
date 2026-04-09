from __future__ import annotations

from attack.models._srgnn_base import SRGNNBaseRunner
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim


class SRGNNVictimRunner(SRGNNBaseRunner, VictimRunnerBase):
    name = "srgnn"
    role = "victim"


register_victim(SRGNNVictimRunner.name, SRGNNVictimRunner)


__all__ = ["SRGNNVictimRunner"]
