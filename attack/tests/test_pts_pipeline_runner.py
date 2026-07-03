from __future__ import annotations

from dataclasses import fields, replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest

import attack.common.paths as paths

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    PTSCEMEpochRewardDiagnosticsRuntimeConfig,
    PTSCEMRuntimeConfig,
    PTSArtifactsConfig,
    PTSRewardConfig,
    load_config,
)
from attack.common.artifact_io import save_fake_sessions, save_json
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    attack_key,
    run_group_key,
    shared_attack_artifact_key,
    target_cohort_key,
    target_selection_key,
    victim_prediction_key,
)
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs import run_pts_construction_cem
from attack.pipeline.runs.run_pts_construction_cem import (
    PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION,
    PTS_CEM_SHARED_CACHE_SCHEMA_VERSION,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    build_pts_cem_shared_cache_identity,
    pts_cem_shared_cache_dir,
    pts_cem_shared_cache_key,
    _load_json_sessions,
    _materialize_shared_pts_cem_cache,
    _resolve_pts_cem_base_seed,
    _try_load_shared_pts_cem_cache,
    _try_load_cached_pts_best_candidate,
    _validate_pts_construction_run_config,
    _write_shared_pts_cem_cache,
    _write_pts_construction_complete_marker,
)
from attack.tools.invalidate_victim_cells import invalidate_victim_cells


CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "tests"
    / "fixtures"
    / "configs"
    / "diginetica_valbest_attack_pts_construction_grouped_cem_space_filling_ratio1_srgnn_partial4_target5334.yaml"
)
def _with_pts(config, pts_config):
    return replace(
        config,
        attack=replace(config.attack, pts_construction=pts_config),
    )


def _make_test_output_dir() -> Path:
    path = REPO_ROOT / "outputs" / f"tmp_pts_pipeline_runner_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    return path


def _with_output_root(config, root: Path):
    return replace(
        config,
        artifacts=replace(config.artifacts, root=str(root)),
    )


def _write_identity_inputs(work_dir: Path, *, fake_payload: bytes = b"fake") -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    fake_sessions = work_dir / "fake_sessions.pkl"
    poison_model = work_dir / "poison_model.pt"
    fake_sessions.write_bytes(fake_payload)
    poison_model.write_bytes(b"poison")
    return fake_sessions, poison_model


def _shared_identity_for(
    config,
    *,
    target_item: int = 5334,
    fake_payload: bytes = b"fake",
):
    work_dir = _make_test_output_dir()
    fake_sessions, poison_model = _write_identity_inputs(
        work_dir,
        fake_payload=fake_payload,
    )
    return (
        work_dir,
        build_pts_cem_shared_cache_identity(
            config,
            target_item=target_item,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        ),
    )


def _write_minimal_pts_artifacts(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pts_cem_trace.jsonl").write_text("{}", encoding="utf-8")
    save_json([], root / "pts_policy_history.json")
    save_json({"candidate_key": "iter0_cand0", "policy": {}}, root / "pts_best_policy.json")
    save_json({}, root / "pts_final_policy.json")
    save_json(
        {
            "top_k": 1,
            "candidates": [
                {
                    "rank": 1,
                    "candidate_key": "iter0_cand0",
                    "policy": {},
                }
            ],
        },
        root / "pts_top_candidate_policies.json",
    )
    rank1 = root / "top_candidates" / "rank_1"
    rank1.mkdir(parents=True, exist_ok=True)
    save_json({"policy": "rank1"}, rank1 / "policy.json")
    save_json([[1, 2, 5334]], rank1 / "sessions.json")
    (rank1 / "session_records.jsonl").write_text("{}", encoding="utf-8")
    save_json(
        {
            "rank": 1,
            "candidate_key": "iter0_cand0",
            "iteration": 0,
            "candidate_id": 0,
            "candidate_seed": 20260405,
            "reward": 0.5,
            "reward_metrics": {"reward": 0.5},
            "target_item": 5334,
        },
        rank1 / "metadata.json",
    )
    save_json(
        {
            "candidates": [
                {
                    "rank": 1,
                    "candidate_key": "iter0_cand0",
                    "policy_path": str(rank1 / "policy.json"),
                    "sessions_path": str(rank1 / "sessions.json"),
                    "session_records_path": str(rank1 / "session_records.jsonl"),
                    "metadata_path": str(rank1 / "metadata.json"),
                }
            ]
        },
        root / "pts_top_candidates.json",
    )
    return {
        "pts_cem_trace": str(root / "pts_cem_trace.jsonl"),
        "pts_policy_history": str(root / "pts_policy_history.json"),
        "pts_best_policy": str(root / "pts_best_policy.json"),
        "pts_final_policy": str(root / "pts_final_policy.json"),
        "pts_top_candidates": str(root / "pts_top_candidates.json"),
        "pts_top_candidate_policies": str(root / "pts_top_candidate_policies.json"),
        "top_candidate_rank_1_policy": str(rank1 / "policy.json"),
        "top_candidate_rank_1_sessions": str(rank1 / "sessions.json"),
        "top_candidate_rank_1_session_records": str(rank1 / "session_records.jsonl"),
        "top_candidate_rank_1_metadata": str(rank1 / "metadata.json"),
    }


def _add_optional_rank2_artifacts(root: Path) -> None:
    rank2 = root / "top_candidates" / "rank_2"
    rank2.mkdir(parents=True, exist_ok=True)
    save_json({"policy": "rank2"}, rank2 / "policy.json")
    save_json([[4, 5, 5334]], rank2 / "sessions.json")
    (rank2 / "session_records.jsonl").write_text("{}", encoding="utf-8")
    save_json(
        {
            "rank": 2,
            "candidate_key": "iter0_cand1",
            "iteration": 0,
            "candidate_id": 1,
            "candidate_seed": 20261405,
            "reward": 0.4,
            "reward_metrics": {"reward": 0.4},
            "target_item": 5334,
        },
        rank2 / "metadata.json",
    )
    top_candidates = json.loads((root / "pts_top_candidates.json").read_text(encoding="utf-8"))
    top_candidates["candidates"].append(
        {
            "rank": 2,
            "candidate_key": "iter0_cand1",
            "policy_path": str(rank2 / "policy.json"),
            "sessions_path": str(rank2 / "sessions.json"),
            "session_records_path": str(rank2 / "session_records.jsonl"),
            "metadata_path": str(rank2 / "metadata.json"),
        }
    )
    save_json(top_candidates, root / "pts_top_candidates.json")
    top_policies = json.loads(
        (root / "pts_top_candidate_policies.json").read_text(encoding="utf-8")
    )
    top_policies["top_k"] = 2
    top_policies["candidates"].append(
        {
            "rank": 2,
            "candidate_key": "iter0_cand1",
            "policy": {},
        }
    )
    save_json(top_policies, root / "pts_top_candidate_policies.json")


class FakeBestCandidate:
    iteration = 0
    candidate_id = 0
    candidate_seed = 20260405
    reward = 0.5
    reward_metrics = {"reward": 0.5}


def test_runner_imports_without_training() -> None:
    assert run_pts_construction_cem.DEFAULT_PTS_CONSTRUCTION_CEM_CONFIG_PATH
    assert callable(run_pts_construction_cem.run_pts_construction_grouped_cem)


def test_pts_runner_runtime_still_explicitly_requires_poison_runner(monkeypatch) -> None:
    config = load_config(CONFIG_PATH)
    captured: dict[str, object] = {}

    def fake_prepare_shared_attack_artifacts(config, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after runtime preparation request")

    monkeypatch.setattr(
        run_pts_construction_cem,
        "prepare_shared_attack_artifacts",
        fake_prepare_shared_attack_artifacts,
    )

    with pytest.raises(RuntimeError, match="stop after runtime preparation request"):
        run_pts_construction_cem.run_pts_construction_grouped_cem(config)

    assert captured["require_poison_runner"] is True


def test_generated_source_pts_shared_attack_identity_is_unchanged_by_runtime_requirement() -> None:
    config = load_config(
        REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample.yaml"
    )

    assert shared_attack_artifact_key(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    ) == shared_attack_artifact_key(
        config,
        run_type=PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
        require_poison_runner=True,
    )


def test_tron_semantics_version_does_not_change_pts_cem_identities(
    monkeypatch,
) -> None:
    config = load_config(
        REPO_ROOT
        / "attack"
        / "configs"
        / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample.yaml"
    )
    run_type = PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE
    attack_identity_context = (
        run_pts_construction_cem.build_pts_construction_attack_identity_context(config)
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions = work_dir / "fake_sessions.pkl"
        poison_model = work_dir / "poison_model.pt"
        save_fake_sessions([[1, 2, 3]], fake_sessions)
        poison_model.write_bytes(b"poison")
        shared_identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        before = {
            "attack": attack_key(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "run_group": run_group_key(
                config,
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "target_cohort": target_cohort_key(config),
            "target_selection": target_selection_key(config),
            "shared_cem": pts_cem_shared_cache_key(shared_identity),
            "srgnn": victim_prediction_key(
                config,
                "srgnn",
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "miasrec": victim_prediction_key(
                config,
                "miasrec",
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
            "tron": victim_prediction_key(
                config,
                "tron",
                run_type=run_type,
                attack_identity_context=attack_identity_context,
            ),
        }

        monkeypatch.setattr(
            paths,
            "TRON_VICTIM_DATA_SEMANTICS",
            "tron_raw_session_export_v2_test",
        )

        assert attack_key(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ) == before["attack"]
        assert run_group_key(
            config,
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ) == before["run_group"]
        assert target_cohort_key(config) == before["target_cohort"]
        assert target_selection_key(config) == before["target_selection"]
        assert pts_cem_shared_cache_key(
            build_pts_cem_shared_cache_identity(
                config,
                target_item=5334,
                fake_sessions_path=fake_sessions,
                poison_model_path=poison_model,
            )
        ) == before["shared_cem"]
        assert victim_prediction_key(
            config,
            "srgnn",
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ) == before["srgnn"]
        assert victim_prediction_key(
            config,
            "miasrec",
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ) == before["miasrec"]
        assert victim_prediction_key(
            config,
            "tron",
            run_type=run_type,
            attack_identity_context=attack_identity_context,
        ) != before["tron"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_runner_help_exits_successfully() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "attack.pipeline.runs.run_pts_construction_cem",
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "PTS-CEM construction" in completed.stdout
    assert "--force-recompute-pts-cem" in completed.stdout


def test_runner_validation_rejects_save_candidate_sessions() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    artifacts = object.__new__(PTSArtifactsConfig)
    object.__setattr__(artifacts, "save_cem_trace", True)
    object.__setattr__(artifacts, "save_best_policy", True)
    object.__setattr__(artifacts, "save_final_policy", True)
    object.__setattr__(artifacts, "save_per_session_records", True)
    object.__setattr__(artifacts, "save_candidate_sessions", True)
    object.__setattr__(artifacts, "save_best_sessions", True)
    object.__setattr__(artifacts, "save_top_candidate_sessions", True)
    bad_config = _with_pts(
        config,
        replace(
            pts,
            artifacts=artifacts,
        ),
    )

    with pytest.raises(ValueError, match="save_candidate_sessions"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_disabled_top_candidate_sessions() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    artifacts = object.__new__(PTSArtifactsConfig)
    object.__setattr__(artifacts, "save_cem_trace", True)
    object.__setattr__(artifacts, "save_best_policy", True)
    object.__setattr__(artifacts, "save_final_policy", True)
    object.__setattr__(artifacts, "save_per_session_records", True)
    object.__setattr__(artifacts, "save_candidate_sessions", False)
    object.__setattr__(artifacts, "save_best_sessions", True)
    object.__setattr__(artifacts, "save_top_candidate_sessions", False)
    bad_config = _with_pts(config, replace(pts, artifacts=artifacts))

    with pytest.raises(ValueError, match="save_top_candidate_sessions"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_unsupported_penalties() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    reward = object.__new__(PTSRewardConfig)
    object.__setattr__(reward, "target_summary", "raw_lowk_mrr_recall_10_20")
    object.__setattr__(reward, "enable_gt_penalty", True)
    object.__setattr__(reward, "gt_penalty_weight", 1.0)
    object.__setattr__(reward, "enable_length_penalty", False)
    object.__setattr__(reward, "length_penalty_weight", 0.0)
    bad_config = _with_pts(config, replace(pts, reward=reward))

    with pytest.raises(NotImplementedError, match="GT penalty"):
        _validate_pts_construction_run_config(bad_config)


def test_runner_validation_rejects_unsupported_seed_source() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    cem = object.__new__(PTSCEMRuntimeConfig)
    for field in fields(PTSCEMRuntimeConfig):
        object.__setattr__(cem, field.name, getattr(pts.cem, field.name))
    object.__setattr__(cem, "seed_source", "fake_session_seed")
    bad_config = _with_pts(config, replace(pts, cem=cem))

    with pytest.raises(ValueError, match="position_opt_seed"):
        _resolve_pts_cem_base_seed(bad_config)


def test_shared_pts_cem_key_excludes_cohort_and_experiment_identity() -> None:
    config = load_config(CONFIG_PATH)
    sampled = replace(
        config,
        experiment=replace(config.experiment, name="sampled_formal_run"),
        targets=replace(
            config.targets,
            mode="sampled",
            explicit_list=(),
            count=20,
        ),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        explicit_identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        sampled_identity = build_pts_cem_shared_cache_identity(
            sampled,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )

        assert target_cohort_key(config) != target_cohort_key(sampled)
        assert pts_cem_shared_cache_key(explicit_identity) == (
            pts_cem_shared_cache_key(sampled_identity)
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_excludes_victims_enabled() -> None:
    config = load_config(CONFIG_PATH)
    expanded = replace(
        config,
        victims=replace(config.victims, enabled=("srgnn", "miasrec", "tron")),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        base_identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        expanded_identity = build_pts_cem_shared_cache_identity(
            expanded,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )

        assert pts_cem_shared_cache_key(base_identity) == (
            pts_cem_shared_cache_key(expanded_identity)
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_excludes_mdhg_runtime_diagnostics() -> None:
    config = load_config(CONFIG_PATH)
    runtime = dict(config.victims.runtime or {})
    runtime["mdhg"] = {
        "python_executable": "python",
        "repo_root": "third_party/mdhg",
        "working_dir": "third_party/mdhg",
        "device": {"use_gpu": True, "gpu_id": "0"},
    }
    base = replace(config, victims=replace(config.victims, runtime=runtime))
    diagnostic_runtime = dict(runtime)
    diagnostic_runtime["mdhg"] = {
        **runtime["mdhg"],
        "diagnostics": {
            "epoch_metrics": True,
            "per_epoch_predictions": True,
        },
    }
    diagnostic = replace(
        base,
        victims=replace(base.victims, runtime=diagnostic_runtime),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        base_identity = build_pts_cem_shared_cache_identity(
            base,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        diagnostic_identity = build_pts_cem_shared_cache_identity(
            diagnostic,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        assert pts_cem_shared_cache_key(base_identity) == pts_cem_shared_cache_key(
            diagnostic_identity
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_includes_target_item() -> None:
    config = load_config(CONFIG_PATH)
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        target_5334 = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        target_11103 = build_pts_cem_shared_cache_identity(
            config,
            target_item=11103,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )

        assert pts_cem_shared_cache_key(target_5334) != (
            pts_cem_shared_cache_key(target_11103)
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_includes_pts_cem_config_and_seed() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    uniform = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                init={"mode": "uniform"},
            ),
        ),
    )
    schedule_config = _with_pts(
        config,
        replace(pts, cem=replace(pts.cem, population_schedule=(8, 8, 8))),
    )
    action_config = _with_pts(
        config,
        replace(
            pts,
            actions=replace(
                pts.actions,
                enabled=(
                    "keep_residual_suffix",
                    "consume_one_generate_continuation",
                    "consume_all_stop",
                ),
            ),
        ),
    )
    seed_config = replace(
        config,
        seeds=replace(config.seeds, position_opt_seed=20260406),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)

        def key_for(candidate_config) -> str:
            return pts_cem_shared_cache_key(
                build_pts_cem_shared_cache_identity(
                    candidate_config,
                    target_item=5334,
                    fake_sessions_path=fake_sessions,
                    poison_model_path=poison_model,
                )
            )

        base_key = key_for(config)
        assert key_for(uniform) != base_key
        assert key_for(schedule_config) != base_key
        assert key_for(action_config) != base_key
        assert key_for(seed_config) != base_key
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_excludes_epoch_reward_diagnostics() -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    diagnostic_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                    enabled=True,
                    epochs=(2, 3),
                ),
            ),
        ),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        base_key = pts_cem_shared_cache_key(
            build_pts_cem_shared_cache_identity(
                config,
                target_item=5334,
                fake_sessions_path=fake_sessions,
                poison_model_path=poison_model,
            )
        )
        diagnostic_key = pts_cem_shared_cache_key(
            build_pts_cem_shared_cache_identity(
                diagnostic_config,
                target_item=5334,
                fake_sessions_path=fake_sessions,
                poison_model_path=poison_model,
            )
        )

        assert diagnostic_key == base_key
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_key_includes_surrogate_alignment_and_fake_hash() -> None:
    config = load_config(CONFIG_PATH)
    changed_victim_seed = replace(
        config,
        seeds=replace(config.seeds, victim_train_seed=20260406),
    )
    work_dir = _make_test_output_dir()
    try:
        fake_sessions, poison_model = _write_identity_inputs(work_dir)
        other_fake_sessions, _ = _write_identity_inputs(
            work_dir / "other",
            fake_payload=b"different-fake-sessions",
        )

        def key_for(candidate_config, fake_path=fake_sessions) -> str:
            return pts_cem_shared_cache_key(
                build_pts_cem_shared_cache_identity(
                    candidate_config,
                    target_item=5334,
                    fake_sessions_path=fake_path,
                    poison_model_path=poison_model,
                )
            )

        base_key = key_for(config)
        assert key_for(changed_victim_seed) != base_key
        assert key_for(config, other_fake_sessions) != base_key
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_load_json_sessions_validates_shape() -> None:
    work_dir = _make_test_output_dir()
    try:
        sessions_path = work_dir / "sessions.json"
        save_json([[1, "2", 3.0], [4, 5]], sessions_path)
        assert _load_json_sessions(sessions_path) == [[1, 2, 3], [4, 5]]

        save_json({"not": "sessions"}, sessions_path)
        with pytest.raises(ValueError, match="JSON list"):
            _load_json_sessions(sessions_path)

        save_json([[1, 2], "bad-row"], sessions_path)
        with pytest.raises(ValueError, match="session lists"):
            _load_json_sessions(sessions_path)

        save_json([[1, "not-an-int"]], sessions_path)
        with pytest.raises(ValueError, match="int-like"):
            _load_json_sessions(sessions_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_legacy_fallback_loads_rank1_sessions() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99], [3, 4, 99]], rank1 / "sessions.json")
        save_json(
            {
                "rank": 1,
                "iteration": 2,
                "candidate_id": 3,
                "candidate_seed": 202,
                "reward": 0.25,
                "target_item": 99,
            },
            rank1 / "metadata.json",
        )
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")

        cached = _try_load_cached_pts_best_candidate(
            artifact_dir=root,
            target_item=99,
        )

        assert cached is not None
        assert cached.sessions == [[1, 2, 99], [3, 4, 99]]
        assert cached.cache_mode == "legacy_top_candidate_files"
        assert cached.cache_marker_missing is True
        assert cached.complete_marker_path is None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cached_runner_propagates_cached_final_raw_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = _with_output_root(load_config(CONFIG_PATH), tmp_path)
    cached_sessions = [[1, 2, 5334], [3, 4, 5334]]
    cached = run_pts_construction_cem.CachedPTSBestCandidate(
        sessions=cached_sessions,
        metadata={"target_item": 5334, "reward": 0.5},
        sessions_path=tmp_path / "sessions.json",
        metadata_path=tmp_path / "metadata.json",
        top_candidates_path=None,
        complete_marker_path=None,
        cache_mode="complete_marker",
        cache_marker_missing=False,
    )
    shared = SharedAttackArtifacts(
        stats=SimpleNamespace(),
        clean_sessions=[[1]],
        clean_labels=[2],
        canonical_dataset=object(),
        export_paths={},
        template_sessions=[],
        poison_runner=None,
        fake_session_count=0,
        shared_paths={"fake_sessions": tmp_path / "fake_sessions.pkl"},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_pts_construction_cem,
        "prepare_shared_attack_artifacts",
        lambda *args, **kwargs: shared,
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_pts_construction_artifact_dir",
        lambda *args, **kwargs: tmp_path / "pts_construction_cem",
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "build_pts_cem_shared_cache_identity",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "pts_cem_shared_cache_key",
        lambda identity: "shared-key",
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "pts_cem_shared_cache_dir",
        lambda *args, **kwargs: tmp_path / "shared-cache",
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_current_pts_construction_cache_identity",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_try_load_cached_pts_best_candidate",
        lambda *args, **kwargs: cached,
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "_target_metadata_from_cache",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "PTSDirectActionMLPCEMTrainer",
        lambda *args, **kwargs: pytest.fail("CEM trainer must not be built on cache reuse"),
    )
    monkeypatch.setattr(
        run_pts_construction_cem,
        "PTSGroupedCEMTrainer",
        lambda *args, **kwargs: pytest.fail("CEM trainer must not be built on cache reuse"),
    )

    def fake_run_targets_and_victims(*args, **kwargs):
        payload = kwargs["build_poisoned"](5334)
        captured["raw_fake_sessions"] = payload.raw_fake_sessions
        return {"status": "ok"}

    monkeypatch.setattr(
        run_pts_construction_cem,
        "run_targets_and_victims",
        fake_run_targets_and_victims,
    )

    assert run_pts_construction_cem.run_pts_construction_grouped_cem(config) == {
        "status": "ok"
    }
    assert captured["raw_fake_sessions"] == cached_sessions


def test_tron_only_invalidation_preserves_resolvable_pts_cem_cache(
    tmp_path: Path,
) -> None:
    config = _with_output_root(load_config(CONFIG_PATH), tmp_path)
    attack_identity_context = (
        run_pts_construction_cem.build_pts_construction_attack_identity_context(config)
    )
    run_type = PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
    original_run_group = run_group_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    )
    run_dir = tmp_path / "runs" / "diginetica" / "test" / original_run_group
    artifact_dir = run_dir / "targets" / "5334" / "pts_construction_cem"
    _write_minimal_pts_artifacts(artifact_dir)
    current_identity = run_pts_construction_cem._current_pts_construction_cache_identity(
        config,
        attack_identity_context=attack_identity_context,
        target_item=5334,
    )
    save_json(
        {
            "status": "completed",
            "run_type": run_type,
            "target_item": 5334,
            "identity": current_identity,
            "best_candidate": {
                "sessions_path": "top_candidates/rank_1/sessions.json",
                "metadata_path": "top_candidates/rank_1/metadata.json",
            },
        },
        artifact_dir / "pts_construction_complete.json",
    )
    (run_dir / "targets" / "5334" / "victims" / "tron").mkdir(parents=True)
    save_json(
        {
            "run_group_key": original_run_group,
            "target_cohort_key": target_cohort_key(config),
            "targets_order": [5334],
            "victims": {
                "srgnn": {"victim_prediction_key": "srgnn-key"},
                "miasrec": {"victim_prediction_key": "miasrec-key"},
                "tron": {"victim_prediction_key": "old-tron-key"},
            },
            "cells": {
                "5334": {
                    "srgnn": {"status": "completed"},
                    "miasrec": {"status": "completed"},
                    "tron": {"status": "completed"},
                }
            },
            "created_at": "old",
            "updated_at": "old",
        },
        run_dir / "run_coverage.json",
    )

    invalidate_victim_cells(
        [run_dir],
        victim="tron",
        allowed_roots=[tmp_path],
    )

    assert run_group_key(
        config,
        run_type=run_type,
        attack_identity_context=attack_identity_context,
    ) == original_run_group
    assert artifact_dir.is_dir()
    cached = _try_load_cached_pts_best_candidate(
        artifact_dir=artifact_dir,
        target_item=5334,
        current_identity=current_identity,
    )
    assert cached is not None
    assert cached.sessions == [[1, 2, 5334]]
    coverage = json.loads((run_dir / "run_coverage.json").read_text(encoding="utf-8"))
    assert coverage["cells"]["5334"]["srgnn"]["status"] == "completed"
    assert coverage["cells"]["5334"]["miasrec"]["status"] == "completed"
    assert coverage["cells"]["5334"]["tron"]["status"] == "requested"


def test_epoch_reward_diagnostics_cache_reuse_warning(capsys) -> None:
    config = load_config(CONFIG_PATH)
    pts = config.attack.pts_construction
    assert pts is not None
    diagnostic_config = _with_pts(
        config,
        replace(
            pts,
            cem=replace(
                pts.cem,
                epoch_reward_diagnostics=PTSCEMEpochRewardDiagnosticsRuntimeConfig(
                    enabled=True,
                    epochs=(2, 3),
                ),
            ),
        ),
    )
    cached = run_pts_construction_cem.CachedPTSBestCandidate(
        sessions=[[1, 2, 5334]],
        metadata={"target_item": 5334, "reward": 0.5},
        sessions_path=Path("sessions.json"),
        metadata_path=Path("metadata.json"),
        top_candidates_path=None,
        complete_marker_path=None,
        cache_mode="complete_marker",
        cache_marker_missing=False,
    )

    run_pts_construction_cem._warn_if_reused_cache_missing_epoch_diagnostics(
        diagnostic_config,
        cached,
    )

    captured = capsys.readouterr()
    assert (
        "Epoch reward diagnostics requested, but reused PTS-CEM cache does not "
        "contain diagnostics."
    ) in captured.out


def test_pts_cache_complete_marker_resolves_relative_paths() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 99}, rank1 / "metadata.json")
        save_json(
            {
                "schema_version": "pts_construction_cache_v1",
                "status": "completed",
                "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                "target_item": 99,
                "cache_mode": "fresh_cem",
                "best_candidate": {
                    "rank": 1,
                    "iteration": 1,
                    "candidate_id": 2,
                    "candidate_seed": 3,
                    "reward": 0.5,
                    "reward_metrics": {"reward": 0.5},
                    "sessions_path": "top_candidates/rank_1/sessions.json",
                    "metadata_path": "top_candidates/rank_1/metadata.json",
                    "policy_path": "top_candidates/rank_1/policy.json",
                },
            },
            root / "pts_construction_complete.json",
        )

        cached = _try_load_cached_pts_best_candidate(
            artifact_dir=root,
            target_item=99,
        )

        assert cached is not None
        assert cached.cache_mode == "complete_marker"
        assert cached.cache_marker_missing is False
        assert cached.sessions_path == rank1 / "sessions.json"
        assert cached.metadata["reward"] == 0.5
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_rejects_target_mismatch() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 99]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 100}, rank1 / "metadata.json")
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")

        with pytest.raises(ValueError, match="target_item mismatch"):
            _try_load_cached_pts_best_candidate(
                artifact_dir=root,
                target_item=99,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_cache_complete_marker_rejects_target_mismatch() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        save_json(
            {
                "status": "completed",
                "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                "target_item": 100,
                "best_candidate": {
                    "sessions_path": "top_candidates/rank_1/sessions.json",
                    "metadata_path": "top_candidates/rank_1/metadata.json",
                },
            },
            root / "pts_construction_complete.json",
        )

        with pytest.raises(ValueError, match="target_item mismatch"):
            _try_load_cached_pts_best_candidate(
                artifact_dir=root,
                target_item=99,
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_cache_materializes_into_local_run_group() -> None:
    work_dir = _make_test_output_dir()
    try:
        config = _with_output_root(load_config(CONFIG_PATH), work_dir / "outputs")
        pts = config.attack.pts_construction
        lean_config = _with_pts(
            config,
            replace(
                pts,
                cem=replace(pts.cem, save_top_k_candidates=1),
                artifacts=replace(pts.artifacts, save_per_session_records=False),
            ),
        )
        fake_sessions, poison_model = _write_identity_inputs(work_dir / "identity")
        identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        shared_key = pts_cem_shared_cache_key(identity)
        shared_dir = pts_cem_shared_cache_dir(config, shared_key)
        source_dir = work_dir / "source" / "pts_construction_cem"
        artifact_paths = _write_minimal_pts_artifacts(source_dir)
        _add_optional_rank2_artifacts(source_dir)
        attack_identity_context = (
            run_pts_construction_cem.build_pts_construction_attack_identity_context(
                config
            )
        )
        _write_shared_pts_cem_cache(
            config=config,
            target_item=5334,
            local_artifact_dir=source_dir,
            artifact_paths=artifact_paths,
            best_candidate=FakeBestCandidate(),
            shared_cache_dir=shared_dir,
            shared_cache_key=shared_key,
            shared_cache_identity=identity,
            attack_identity_context=attack_identity_context,
        )

        shared_cached = _try_load_shared_pts_cem_cache(
            shared_cache_dir=shared_dir,
            target_item=5334,
            shared_cache_key=shared_key,
            shared_cache_identity=identity,
        )
        assert shared_cached is not None
        local_dir = work_dir / "local_run_group" / "targets" / "5334" / "pts_construction_cem"
        current_identity = run_pts_construction_cem._current_pts_construction_cache_identity(
            lean_config,
            attack_identity_context=attack_identity_context,
            target_item=5334,
        )
        cached = _materialize_shared_pts_cem_cache(
            config=lean_config,
            target_item=5334,
            local_artifact_dir=local_dir,
            shared_cache_dir=shared_dir,
            shared_cached=shared_cached,
            shared_cache_key=shared_key,
            attack_identity_context=attack_identity_context,
            current_identity=current_identity,
        )

        assert cached.reused_shared_pts_cem is True
        assert cached.local_materialized_from_shared is True
        assert (local_dir / "pts_cem_trace.jsonl").exists()
        assert (local_dir / "top_candidates" / "rank_1" / "sessions.json").exists()
        assert not (local_dir / "top_candidates" / "rank_1" / "session_records.jsonl").exists()
        assert not (local_dir / "top_candidates" / "rank_2").exists()
        assert not (local_dir / "pts_cem_shared_complete.json").exists()
        with (local_dir / "pts_construction_complete.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            marker = json.load(handle)
        assert marker["run_group_key"] == run_group_key(
            config,
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            attack_identity_context=attack_identity_context,
        )
        assert marker["target_cohort_key"] == target_cohort_key(config)
        assert marker["shared_pts_cem_cache_key"] == shared_key
        assert marker["reused_shared_pts_cem"] is True
        assert marker["local_materialized_from_shared"] is True
        with (local_dir / "pts_top_candidates.json").open(
            "r",
            encoding="utf-8",
        ) as handle:
            top_candidates = json.load(handle)
        assert len(top_candidates["candidates"]) == 1
        row = top_candidates["candidates"][0]
        assert str(local_dir) in row["policy_path"]
        assert str(local_dir) in row["sessions_path"]
        assert row["session_records_path"] is None
        assert str(source_dir) not in row["policy_path"]
        assert str(shared_dir) not in row["policy_path"]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_shared_pts_cem_cache_miss_for_missing_or_invalid_marker() -> None:
    work_dir = _make_test_output_dir()
    try:
        config = _with_output_root(load_config(CONFIG_PATH), work_dir / "outputs")
        fake_sessions, poison_model = _write_identity_inputs(work_dir / "identity")
        identity = build_pts_cem_shared_cache_identity(
            config,
            target_item=5334,
            fake_sessions_path=fake_sessions,
            poison_model_path=poison_model,
        )
        shared_key = pts_cem_shared_cache_key(identity)
        shared_dir = pts_cem_shared_cache_dir(config, shared_key)

        assert _try_load_shared_pts_cem_cache(
            shared_cache_dir=shared_dir,
            target_item=5334,
            shared_cache_key=shared_key,
            shared_cache_identity=identity,
        ) is None

        shared_dir.mkdir(parents=True)
        save_json(
            {
                "schema_version": PTS_CEM_SHARED_CACHE_SCHEMA_VERSION,
                "status": "completed",
                "shared_pts_cem_cache_key": "wrong",
                "target_item": 5334,
                "construction_identity": identity,
                "required_artifact_files": [],
            },
            shared_dir / "pts_cem_shared_complete.json",
        )
        assert _try_load_shared_pts_cem_cache(
            shared_cache_dir=shared_dir,
            target_item=5334,
            shared_cache_key=shared_key,
            shared_cache_identity=identity,
        ) is None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_force_recompute_resets_local_without_using_shared_cache() -> None:
    work_dir = _make_test_output_dir()
    try:
        config = _with_output_root(load_config(CONFIG_PATH), work_dir / "outputs")
        attack_identity_context = (
            run_pts_construction_cem.build_pts_construction_attack_identity_context(
                config
            )
        )
        artifact_dir = run_pts_construction_cem._pts_construction_artifact_dir(
            config,
            5334,
            attack_identity_context=attack_identity_context,
        )
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "stale.txt").write_text("stale", encoding="utf-8")

        run_pts_construction_cem._reset_pts_artifact_dir_for_force(
            artifact_dir=artifact_dir,
            config=config,
            target_item=5334,
            attack_identity_context=attack_identity_context,
        )

        assert not artifact_dir.exists()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_incompatible_local_pts_cem_marker_raises_clear_error() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 5334]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 5334}, rank1 / "metadata.json")
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")

        with pytest.raises(ValueError, match="Remove this pts_construction_cem folder"):
            _try_load_cached_pts_best_candidate(
                artifact_dir=root,
                target_item=5334,
                current_shared_cache_key="pts_cem_shared_expected",
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_fresh_local_pts_cem_marker_allows_shared_key_schema_drift() -> None:
    work_dir = _make_test_output_dir()
    try:
        root = work_dir / "pts_construction_cem"
        rank1 = root / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        save_json([[1, 2, 5334]], rank1 / "sessions.json")
        save_json({"rank": 1, "target_item": 5334}, rank1 / "metadata.json")
        save_json({"type": "policy"}, rank1 / "policy.json")
        save_json({"candidates": [{"rank": 1}]}, root / "pts_top_candidates.json")
        current_identity = {
            "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            "attack_key": "attack_same",
            "run_group_key": "run_group_same",
        }
        save_json(
            {
                "schema_version": "pts_construction_cache_v1",
                "local_artifact_schema_version": PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION,
                "status": "completed",
                "run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
                "target_item": 5334,
                "cache_mode": "fresh_cem",
                "shared_pts_cem_cache_key": "pts_cem_shared_old_schema",
                "reused_shared_pts_cem": False,
                "local_materialized_from_shared": False,
                "identity": current_identity,
                "best_candidate": {
                    "rank": 1,
                    "iteration": 0,
                    "candidate_id": 0,
                    "candidate_seed": 123,
                    "reward": 0.1,
                    "reward_metrics": {"targeted_recall@20": 0.1},
                    "sessions_path": "top_candidates/rank_1/sessions.json",
                    "metadata_path": "top_candidates/rank_1/metadata.json",
                    "policy_path": "top_candidates/rank_1/policy.json",
                },
            },
            root / "pts_construction_complete.json",
        )

        cached = _try_load_cached_pts_best_candidate(
            artifact_dir=root,
            target_item=5334,
            current_identity=current_identity,
            current_shared_cache_key="pts_cem_shared_new_schema",
        )

        assert cached is not None
        assert cached.cache_mode == "complete_marker"
        assert cached.sessions == [[1, 2, 5334]]
        assert cached.shared_pts_cem_cache_key == "pts_cem_shared_old_schema"
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_complete_marker_writer_creates_candidate_fields() -> None:
    work_dir = _make_test_output_dir()
    try:
        config = load_config(CONFIG_PATH)
        artifact_dir = work_dir / "pts_construction_cem"
        rank1 = artifact_dir / "top_candidates" / "rank_1"
        rank1.mkdir(parents=True)
        artifact_paths = {
            "top_candidate_rank_1_sessions": str(rank1 / "sessions.json"),
            "top_candidate_rank_1_metadata": str(rank1 / "metadata.json"),
            "top_candidate_rank_1_policy": str(rank1 / "policy.json"),
        }

        class FakeBestCandidate:
            iteration = 2
            candidate_id = 5
            candidate_seed = 20260405
            reward = 0.75
            reward_metrics = {"reward": 0.75}

        marker_path = _write_pts_construction_complete_marker(
            config=config,
            target_item=5334,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
            best_candidate=FakeBestCandidate(),
            attack_identity_context=(
                run_pts_construction_cem.build_pts_construction_attack_identity_context(
                    config
                )
            ),
        )

        with marker_path.open("r", encoding="utf-8") as handle:
            marker = json.load(handle)
        assert marker["status"] == "completed"
        assert (
            marker["local_artifact_schema_version"]
            == PTS_CEM_LOCAL_ARTIFACT_SCHEMA_VERSION
        )
        assert marker["run_type"] == PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
        assert marker["run_group_key"]
        assert marker["target_cohort_key"] == target_cohort_key(config)
        assert marker["target_item"] == 5334
        assert marker["reused_shared_pts_cem"] is False
        assert marker["local_materialized_from_shared"] is False
        assert (
            marker["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            marker["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert marker["configured_surrogate_train_seed"] == 20260405
        assert marker["configured_victim_train_seed"] == 20260405
        assert marker["resolved_surrogate_effective_seed"] == 1386226870
        assert marker["resolved_victim_effective_seed"] == 1386226870
        assert marker["surrogate_victim_seed_aligned"] is True
        assert marker["identity"]["target_item"] == 5334
        assert (
            marker["identity"]["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            marker["identity"]["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert marker["identity"]["resolved_surrogate_effective_seed"] == 1386226870
        assert marker["identity"]["resolved_victim_effective_seed"] == 1386226870
        assert marker["best_candidate"]["rank"] == 1
        assert marker["best_candidate"]["candidate_id"] == 5
        assert marker["best_candidate"]["sessions_path"] == (
            "top_candidates/rank_1/sessions.json"
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def test_pts_target_metadata_uses_aligned_surrogate_retrain_seed() -> None:
    config = load_config(CONFIG_PATH)
    pts_config = config.attack.pts_construction
    assert pts_config is not None
    cem_config = pts_config.cem
    artifact_dir = Path("outputs/test_pts_metadata/pts_construction_cem")
    artifact_paths = {
        "pts_cem_trace": str(artifact_dir / "pts_cem_trace.jsonl"),
        "pts_top_candidates": str(artifact_dir / "pts_top_candidates.json"),
        "top_candidate_rank_1_sessions": str(
            artifact_dir / "top_candidates" / "rank_1" / "sessions.json"
        ),
        "top_candidate_rank_1_metadata": str(
            artifact_dir / "top_candidates" / "rank_1" / "metadata.json"
        ),
    }

    class FakeBestCandidate:
        iteration = 2
        candidate_id = 5
        candidate_key = "iter2_cand5"
        candidate_seed = 20260405
        reward = 0.75
        reward_metrics = {"reward": 0.75}
        final_sessions = [[1, 2, 5334]]

    metadata = run_pts_construction_cem._target_metadata(
        config=config,
        pts_config=pts_config,
        cem_config=cem_config,
        artifact_dir=artifact_dir,
        artifact_paths=artifact_paths,
        best_candidate=FakeBestCandidate(),
        target_item=5334,
        complete_marker_path=artifact_dir / "pts_construction_complete.json",
    )
    cached = run_pts_construction_cem.CachedPTSBestCandidate(
        sessions=[[1, 2, 5334]],
        metadata={
            "iteration": 2,
            "candidate_id": 5,
            "candidate_seed": 20260405,
            "reward": 0.75,
            "reward_metrics": {"reward": 0.75},
        },
        sessions_path=Path(artifact_paths["top_candidate_rank_1_sessions"]),
        metadata_path=Path(artifact_paths["top_candidate_rank_1_metadata"]),
        top_candidates_path=Path(artifact_paths["pts_top_candidates"]),
        complete_marker_path=None,
        cache_mode="complete_marker",
        cache_marker_missing=False,
    )
    cached_metadata = run_pts_construction_cem._target_metadata_from_cache(
        config=config,
        pts_config=pts_config,
        cem_config=cem_config,
        artifact_dir=artifact_dir,
        target_item=5334,
        cached=cached,
    )

    for payload in (metadata, cached_metadata):
        assert payload["target_item"] == 5334
        assert payload["pts_candidate_retrain_seed"] == 1386226870
        assert (
            payload["pts_cem_surrogate_seed_alignment_mode"]
            == "victim_effective_seed"
        )
        assert (
            payload["pts_cem_surrogate_seed_alignment_target_victim_name"]
            == "srgnn"
        )
        assert payload["configured_surrogate_train_seed"] == 20260405
        assert payload["configured_victim_train_seed"] == 20260405
        assert payload["resolved_surrogate_effective_seed"] == 1386226870
        assert payload["resolved_victim_effective_seed"] == 1386226870
        assert payload["surrogate_victim_seed_aligned"] is True
