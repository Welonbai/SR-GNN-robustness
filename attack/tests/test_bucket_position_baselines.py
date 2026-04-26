from __future__ import annotations

import json
import random
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import save_json
from attack.common.config import load_config
from attack.common.paths import (
    attack_key,
    run_group_key,
    shared_attack_artifact_key,
    victim_prediction_key,
)
from attack.pipeline.runs.run_bucket_position_baseline import (
    _validate_bucket_target_cohort,
)
from attack.position_opt.bucket_diagnostics import (
    build_bucket_diagnostics,
    build_bucket_position_summary,
    write_selected_positions_jsonl,
)
from attack.position_opt.bucket_selector import (
    BUCKET_ABS_POS2,
    BUCKET_ABS_POS3PLUS,
    BUCKET_FIRST_NONZERO,
    BUCKET_METHODS,
    BUCKET_NONFIRST_NONZERO,
    _resolve_mode_candidates,
    selection_record_to_jsonable,
    select_bucket_session_position,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "attack"
    / "configs"
    / "diginetica_attack_bucket_position_baselines_ratio1.yaml"
)


def _bucket_config():
    return load_config(CONFIG_PATH)


def _session(length: int) -> list[int]:
    return list(range(1, length + 1))


def _artifact_temp_dir() -> Path:
    path = REPO_ROOT / "outputs" / ".pytest_bucket_artifacts" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_case1_single_nonzero_candidate_uses_expected_fallbacks() -> None:
    session = _session(2)

    first = select_bucket_session_position(
        method_name=BUCKET_FIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(7),
    )
    abs_pos2 = select_bucket_session_position(
        method_name=BUCKET_ABS_POS2,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(7),
    )
    nonfirst = select_bucket_session_position(
        method_name=BUCKET_NONFIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(7),
    )
    pos3plus = select_bucket_session_position(
        method_name=BUCKET_ABS_POS3PLUS,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(7),
    )

    assert first.selected_position == 1
    assert first.fallback_used is False
    assert abs_pos2.selected_position == 1
    assert abs_pos2.fallback_used is True
    assert abs_pos2.fallback_reason == "no_abs_pos2_candidate"
    assert nonfirst.selected_position == 1
    assert nonfirst.fallback_used is True
    assert nonfirst.fallback_reason == "no_nonfirst_nonzero_candidate"
    assert pos3plus.selected_position == 1
    assert pos3plus.fallback_used is True
    assert pos3plus.fallback_reason == "no_pos3plus_candidate"


def test_first_nonzero_uses_min_position_even_if_candidates_are_unsorted() -> None:
    mode_candidates, fallback_reason = _resolve_mode_candidates(
        BUCKET_FIRST_NONZERO,
        (4, 2, 1, 3),
        fallback_to_pos0_only=False,
    )

    assert mode_candidates == (1,)
    assert fallback_reason is None


def test_case2_two_nonzero_candidates_matches_expected_behavior() -> None:
    session = _session(3)

    first = select_bucket_session_position(
        method_name=BUCKET_FIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(11),
    )
    abs_pos2 = select_bucket_session_position(
        method_name=BUCKET_ABS_POS2,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(11),
    )
    nonfirst = select_bucket_session_position(
        method_name=BUCKET_NONFIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(11),
    )
    pos3plus = select_bucket_session_position(
        method_name=BUCKET_ABS_POS3PLUS,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(11),
    )

    assert first.selected_position == 1
    assert abs_pos2.selected_position == 2
    assert nonfirst.selected_position == 2
    assert nonfirst.mode_candidate_positions == (2,)
    assert pos3plus.selected_position in {1, 2}
    assert pos3plus.fallback_used is True
    assert pos3plus.nonzero_candidate_positions == (1, 2)


def test_case3_tail_buckets_only_sample_from_allowed_positions() -> None:
    session = _session(5)

    first = select_bucket_session_position(
        method_name=BUCKET_FIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(13),
    )
    abs_pos2 = select_bucket_session_position(
        method_name=BUCKET_ABS_POS2,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(13),
    )
    nonfirst = select_bucket_session_position(
        method_name=BUCKET_NONFIRST_NONZERO,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(13),
    )
    pos3plus = select_bucket_session_position(
        method_name=BUCKET_ABS_POS3PLUS,
        fake_session_index=0,
        session=session,
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(13),
    )

    assert first.selected_position == 1
    assert abs_pos2.selected_position == 2
    assert nonfirst.selected_position in {2, 3, 4}
    assert nonfirst.mode_candidate_positions == (2, 3, 4)
    assert pos3plus.selected_position in {3, 4}
    assert pos3plus.mode_candidate_positions == (3, 4)


def test_pos0_is_never_selected_when_nonzero_candidates_exist() -> None:
    rng = random.Random(17)
    sessions = (_session(3), _session(5), _session(8))
    for method_name in BUCKET_METHODS:
        for index, session in enumerate(sessions):
            record = select_bucket_session_position(
                method_name=method_name,
                fake_session_index=index,
                session=session,
                target_item=5334,
                replacement_topk_ratio=1.0,
                nonzero_action_when_possible=True,
                rng=rng,
            )
            assert record.selected_position > 0


def test_same_seed_reproduces_identical_selected_positions() -> None:
    sessions = [_session(3), _session(5), _session(8), _session(9)]

    def collect(method_name: str, seed: int) -> list[int]:
        rng = random.Random(seed)
        return [
            select_bucket_session_position(
                method_name=method_name,
                fake_session_index=index,
                session=session,
                target_item=5334,
                replacement_topk_ratio=1.0,
                nonzero_action_when_possible=True,
                rng=rng,
            ).selected_position
            for index, session in enumerate(sessions)
        ]

    assert collect(BUCKET_ABS_POS3PLUS, 20260405) == collect(
        BUCKET_ABS_POS3PLUS,
        20260405,
    )
    assert collect(BUCKET_NONFIRST_NONZERO, 20260405) == collect(
        BUCKET_NONFIRST_NONZERO,
        20260405,
    )


def test_different_seed_can_change_stochastic_selected_positions() -> None:
    sessions = [_session(5), _session(6), _session(7), _session(8), _session(9)]

    def collect(seed: int) -> list[int]:
        rng = random.Random(seed)
        return [
            select_bucket_session_position(
                method_name=BUCKET_ABS_POS3PLUS,
                fake_session_index=index,
                session=session,
                target_item=5334,
                replacement_topk_ratio=1.0,
                nonzero_action_when_possible=True,
                rng=rng,
            ).selected_position
            for index, session in enumerate(sessions)
        ]

    assert collect(1) != collect(2)


def test_bucket_run_identity_is_isolated_while_shared_fake_sessions_are_reused() -> None:
    config = _bucket_config()

    attack_keys = {
        method_name: attack_key(config, run_type=method_name)
        for method_name in BUCKET_METHODS
    }
    run_group_keys = {
        method_name: run_group_key(config, run_type=method_name)
        for method_name in BUCKET_METHODS
    }
    victim_keys = {
        method_name: victim_prediction_key(
            config,
            "srgnn",
            run_type=method_name,
        )
        for method_name in BUCKET_METHODS
    }
    shared_keys = {
        method_name: shared_attack_artifact_key(config, run_type=method_name)
        for method_name in BUCKET_METHODS
    }

    assert len(set(attack_keys.values())) == len(BUCKET_METHODS)
    assert len(set(run_group_keys.values())) == len(BUCKET_METHODS)
    assert len(set(victim_keys.values())) == len(BUCKET_METHODS)
    assert len(set(shared_keys.values())) == 1


def test_bucket_target_cohort_guard_accepts_expected_registry() -> None:
    config = _bucket_config()
    target_registry = {
        "target_cohort_key": "target_cohort_8be070ab82",
        "ordered_targets": [11103, 39588, 5334, 5418],
    }

    _, resolved_prefix, validation = _validate_bucket_target_cohort(
        config,
        target_registry=target_registry,
    )

    assert resolved_prefix == [11103, 39588, 5334]
    assert validation["passed"] is True


def test_bucket_target_cohort_guard_rejects_mismatched_registry() -> None:
    config = _bucket_config()
    with pytest.raises(RuntimeError, match="unexpected target cohort key"):
        _validate_bucket_target_cohort(
            config,
            target_registry={
                "target_cohort_key": "target_cohort_wrong",
                "ordered_targets": [11103, 39588, 5334],
            },
        )

    with pytest.raises(RuntimeError, match="unexpected target prefix"):
        _validate_bucket_target_cohort(
            config,
            target_registry={
                "target_cohort_key": "target_cohort_8be070ab82",
                "ordered_targets": [5334, 11103, 39588],
            },
        )


def test_bucket_artifacts_include_required_fields() -> None:
    tmp_path = _artifact_temp_dir()
    rng = random.Random(23)
    try:
        records = [
            select_bucket_session_position(
                method_name=BUCKET_ABS_POS2,
                fake_session_index=index,
                session=session,
                target_item=5334,
                replacement_topk_ratio=1.0,
                nonzero_action_when_possible=True,
                rng=rng,
            )
            for index, session in enumerate((_session(2), _session(3), _session(5)))
        ]

        jsonl_path = write_selected_positions_jsonl(tmp_path / "selected_positions.jsonl", records)
        summary = build_bucket_position_summary(
            records,
            method_name=BUCKET_ABS_POS2,
            target_item=5334,
            seed=20260405,
            seed_source="position_opt_seed",
            replacement_topk_ratio=1.0,
            nonzero_action_when_possible=True,
        )
        summary_path = tmp_path / "position_summary.json"
        save_json(summary, summary_path)
        diagnostics = build_bucket_diagnostics(
            records,
            method_name=BUCKET_ABS_POS2,
            target_item=5334,
            seed=20260405,
            seed_source="position_opt_seed",
            replacement_topk_ratio=1.0,
            nonzero_action_when_possible=True,
            shared_fake_sessions_path="outputs/shared/diginetica/attack/attack_shared_1c4345bfa3/fake_sessions.pkl",
            target_cohort_key="target_cohort_8be070ab82",
            resolved_target_prefix=[11103, 39588, 5334],
            cohort_validation={"passed": True},
        )
        diagnostics_path = tmp_path / "bucket_diagnostics.json"
        save_json(diagnostics, diagnostics_path)

        jsonl_lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        parsed_lines = [json.loads(line) for line in jsonl_lines]
        parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        parsed_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

        assert parsed_lines
        assert {
            "fake_session_index",
            "session_length",
            "target_item",
            "base_candidate_positions",
            "nonzero_candidate_positions",
            "selected_position",
            "selected_mode",
            "fallback_used",
            "fallback_reason",
            "candidate_count",
            "mode_candidate_count",
        } <= set(parsed_lines[0])
        assert parsed_summary["total_fake_sessions"] == len(records)
        assert parsed_summary["method_name"] == BUCKET_ABS_POS2
        assert parsed_summary["seed_source"] == "position_opt_seed"
        assert parsed_diagnostics["selected_pos2_count"] == 2
        assert parsed_diagnostics["summary"]["fallback_count"] == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_selection_record_jsonable_preserves_expected_schema() -> None:
    record = select_bucket_session_position(
        method_name=BUCKET_FIRST_NONZERO,
        fake_session_index=0,
        session=_session(3),
        target_item=5334,
        replacement_topk_ratio=1.0,
        nonzero_action_when_possible=True,
        rng=random.Random(29),
    )

    payload = selection_record_to_jsonable(record)

    assert payload["base_candidate_positions"] == [0, 1, 2]
    assert payload["nonzero_candidate_positions"] == [1, 2]
    assert payload["mode_candidate_positions"] == [1]
    assert payload["selected_position"] == 1
