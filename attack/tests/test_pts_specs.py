from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.pts.specs import get_default_pts_v1_specs, lookup_spec_by_name


def test_default_pts_v1_specs_contain_expected_names() -> None:
    specs = get_default_pts_v1_specs()

    assert [spec.name for spec in specs] == [
        "keep_residual_suffix",
        "regenerate_residual_suffix",
        "consume_one_keep_rest",
        "consume_all_stop",
    ]


def test_lookup_spec_by_name_returns_single_match() -> None:
    specs = get_default_pts_v1_specs()

    spec = lookup_spec_by_name(specs, "consume_all_stop")

    assert spec.name == "consume_all_stop"
