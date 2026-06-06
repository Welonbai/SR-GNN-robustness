from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.creat.diagnostics import creat_fidelity_metadata, position_collapse_summary


def test_position_collapse_summary_uses_final_position_distribution() -> None:
    collapsed = position_collapse_summary([1] * 9 + [2])
    assert collapsed["position_top1_index"] == 1
    assert collapsed["position_top1_ratio"] == 0.9
    assert collapsed["is_position_collapsed"] is True


def test_creat_fidelity_reports_disabled_and_enabled_dpp() -> None:
    disabled = creat_fidelity_metadata(variant="v2", dpp_reward_weight=0.0)
    enabled = creat_fidelity_metadata(variant="v2", dpp_reward_weight=0.1)
    assert (
        disabled["original_creat_components"]["pattern_diversity_dpp"]
        == "implemented_disabled"
    )
    assert (
        enabled["original_creat_components"]["pattern_diversity_dpp"]
        == "implemented_enabled"
    )
    assert (
        disabled["original_creat_components"]["unbalanced_co_optimal_transport"]
        == "not_implemented"
    )
