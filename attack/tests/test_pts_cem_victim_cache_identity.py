from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import shutil
import sys
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json, save_json
from attack.common.config import PTSCEMSurrogateRetrainRuntimeConfig, load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    shared_artifact_paths,
    victim_prediction_key,
    victim_prediction_key_payload,
)
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.data.session_stats import compute_session_stats
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    _pts_cem_victim_attack_identity_context,
    _pts_cem_victim_identity_metrics_payload,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import build_ordered_target_cohort
from attack.pipeline.core.victim_execution import VictimExecutionResult


CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_internal_sample.yaml"
)
DIRECT_ACTION_CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_sample.yaml"
)
RUN_TYPE = PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE
DIRECT_ACTION_RUN_TYPE = PTS_CONSTRUCTION_DIRECT_ACTION_MLP_CEM_RUN_TYPE


def _base_config():
    return load_config(CONFIG_PATH)


def _direct_action_config():
    return load_config(DIRECT_ACTION_CONFIG_PATH)


def _metadata(
    *,
    target_item: int = 39588,
    shared_cache_key: str = "pts_cem_shared_same",
    sessions_sha1: str = "a" * 40,
    source_candidate_key: str = "iter1_cand5",
) -> dict[str, object]:
    return {
        "pts_construction_method": "grouped_cem_v1",
        "shared_pts_cem_cache_key": shared_cache_key,
        "selected_pts_cem_sessions_sha1": sessions_sha1,
        "source_candidate_rank": 1,
        "source_candidate_key": source_candidate_key,
        "pts_cem_surrogate_seed_alignment_mode": "victim_effective_seed",
        "pts_cem_surrogate_seed_alignment_target_victim_name": "srgnn",
        "configured_surrogate_train_seed": 20260405,
        "configured_victim_train_seed": 20260405,
        "resolved_surrogate_effective_seed": 302235205,
        "resolved_victim_effective_seed": 302235205,
        "surrogate_victim_seed_aligned": True,
        "target_item": int(target_item),
    }


def _direct_action_metadata(
    *,
    target_item: int = 11103,
    shared_cache_key: str = "pts_cem_shared_direct",
    sessions_sha1: str = "d" * 40,
) -> dict[str, object]:
    return {
        **_metadata(
            target_item=target_item,
            shared_cache_key=shared_cache_key,
            sessions_sha1=sessions_sha1,
            source_candidate_key="iter0_cand12",
        ),
        "pts_construction_method": "direct_action_mlp_cem",
    }


def _identity(
    config, *, target_item: int = 39588, metadata: dict[str, object] | None = None
):
    return _pts_cem_victim_attack_identity_context(
        config,
        run_type=RUN_TYPE,
        target_item=target_item,
        target_metadata=(
            _metadata(target_item=target_item) if metadata is None else metadata
        ),
    )


def _direct_action_identity(
    config,
    *,
    target_item: int = 11103,
    metadata: dict[str, object] | None = None,
):
    return _pts_cem_victim_attack_identity_context(
        config,
        run_type=DIRECT_ACTION_RUN_TYPE,
        target_item=target_item,
        target_metadata=(
            _direct_action_metadata(target_item=target_item)
            if metadata is None
            else metadata
        ),
    )


def test_pts_cem_victim_identity_excludes_target_cohort_mode_and_count() -> None:
    base = _base_config()
    explicit = replace(
        base,
        targets=replace(
            base.targets,
            mode="explicit_list",
            explicit_list=(39588,),
            bucket="popular",
            count=1,
        ),
    )
    sampled = replace(
        base,
        targets=replace(
            base.targets,
            mode="sampled",
            explicit_list=(),
            bucket="popular",
            count=2,
        ),
    )

    assert _identity(explicit) == _identity(sampled)


def test_direct_action_pts_cem_victim_identity_excludes_count_and_artifact_knobs() -> (
    None
):
    base = _direct_action_config()
    pts = base.attack.pts_construction
    assert pts is not None
    old_artifact_shape = replace(
        base,
        targets=replace(base.targets, count=5),
        attack=replace(
            base.attack,
            pts_construction=replace(
                pts,
                artifacts=replace(pts.artifacts, save_per_session_records=True),
                cem=replace(pts.cem, save_top_k_candidates=3),
            ),
        ),
    )
    lean_artifact_shape = replace(
        base,
        targets=replace(base.targets, count=6),
        attack=replace(
            base.attack,
            pts_construction=replace(
                pts,
                artifacts=replace(pts.artifacts, save_per_session_records=False),
                cem=replace(pts.cem, save_top_k_candidates=1),
            ),
        ),
    )

    old_identity = _direct_action_identity(old_artifact_shape)
    lean_identity = _direct_action_identity(lean_artifact_shape)

    assert old_identity == lean_identity
    assert victim_prediction_key(
        old_artifact_shape,
        "srgnn",
        run_type=DIRECT_ACTION_RUN_TYPE,
        victim_attack_identity_context=old_identity,
        victim_effective_train_seed=123,
    ) == victim_prediction_key(
        lean_artifact_shape,
        "srgnn",
        run_type=DIRECT_ACTION_RUN_TYPE,
        victim_attack_identity_context=lean_identity,
        victim_effective_train_seed=123,
    )


def test_pts_cem_victim_identity_excludes_experiment_name() -> None:
    base = _base_config()
    renamed = replace(
        base, experiment=replace(base.experiment, name="other_experiment")
    )

    assert _identity(base) == _identity(renamed)


def test_pts_cem_victim_identity_includes_selected_sessions_hash() -> None:
    config = _base_config()

    assert _identity(config, metadata=_metadata(sessions_sha1="a" * 40)) != _identity(
        config,
        metadata=_metadata(sessions_sha1="b" * 40),
    )


def test_pts_cem_victim_identity_includes_target_item() -> None:
    config = _base_config()

    assert _identity(config, target_item=39588) != _identity(config, target_item=11103)


def test_pts_cem_victim_identity_excludes_evaluation_rendering_config() -> None:
    config = _base_config()
    changed_evaluation = replace(
        config,
        evaluation=replace(
            config.evaluation,
            topk=(10,),
            targeted_metrics=("recall",),
            ground_truth_metrics=("ndcg",),
        ),
    )
    identity = _identity(config)
    changed_identity = _identity(changed_evaluation)

    assert "evaluation" not in identity
    assert identity == changed_identity
    assert victim_prediction_key(
        config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=123,
    ) == victim_prediction_key(
        changed_evaluation,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=changed_identity,
        victim_effective_train_seed=123,
    )


def test_pts_cem_surrogate_retrain_protocol_excluded_from_victim_identity() -> None:
    config = _base_config()
    pts = config.attack.pts_construction
    assert pts is not None
    fixed_config = replace(
        config,
        attack=replace(
            config.attack,
            pts_construction=replace(
                pts,
                cem=replace(
                    pts.cem,
                    surrogate_retrain=PTSCEMSurrogateRetrainRuntimeConfig(
                        checkpoint_protocol="fixed_last",
                        validation_enabled=False,
                        reward_checkpoint="last",
                    ),
                ),
            ),
        ),
    )
    metadata = _metadata()
    identity = _identity(config, metadata=metadata)
    fixed_identity = _identity(fixed_config, metadata=metadata)

    assert identity == fixed_identity
    assert victim_prediction_key(
        config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=123,
    ) == victim_prediction_key(
        fixed_config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=fixed_identity,
        victim_effective_train_seed=123,
    )


def test_pts_cem_victim_metrics_payload_records_evaluation_config() -> None:
    config = _base_config()
    payload = _pts_cem_victim_identity_metrics_payload(
        config,
        run_type=RUN_TYPE,
        attack_identity_context=None,
        victim_attack_identity_context=_identity(config),
    )

    assert payload["evaluation"] == {
        "topk": [int(k) for k in config.evaluation.topk],
        "targeted_metrics": list(config.evaluation.targeted_metrics),
        "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
    }


def test_pts_cem_victim_key_includes_victim_model() -> None:
    config = _base_config()
    miasrec_params = {"train": {"epochs": 10, "hidden_size": 64, "dropout": 0.0}}
    with_miasrec = replace(
        config,
        victims=replace(
            config.victims,
            params={**config.victims.params, "miasrec": miasrec_params},
        ),
    )
    identity = _identity(config)

    assert victim_prediction_key(
        config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=123,
    ) != victim_prediction_key(
        with_miasrec,
        "miasrec",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=123,
    )


def test_pts_cem_victim_key_includes_victim_effective_seed() -> None:
    config = _base_config()
    identity = _identity(config)

    assert victim_prediction_key(
        config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=123,
    ) != victim_prediction_key(
        config,
        "srgnn",
        run_type=RUN_TYPE,
        victim_attack_identity_context=identity,
        victim_effective_train_seed=456,
    )


def test_non_pts_victim_prediction_key_keeps_run_level_attack_identity() -> None:
    config = _base_config()
    payload = victim_prediction_key_payload(config, "srgnn", run_type="attack")

    assert "attack_key" in payload
    assert "victim_attack_identity" not in payload
    assert "victim_effective_train_seed" not in payload


@contextmanager
def _temp_root():
    temp_root = REPO_ROOT / "outputs" / ".pytest_pts_cem_victim_cache" / uuid4().hex
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_root
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _stats():
    return compute_session_stats(
        [
            [101, 202, 303],
            [202, 303, 404],
            [303, 404, 505],
            [404, 505, 606],
        ]
    )


def _context(config) -> RunContext:
    return RunContext(
        canonical_dataset=object(),
        stats=_stats(),
        clean_sessions=[[1, 2]],
        clean_labels=[3],
        export_paths={},
        shared_paths=shared_artifact_paths(config, run_type=RUN_TYPE),
        fake_session_count=1,
    )


def _config_for_temp_root(
    temp_root: Path, *, sampled: bool, target_item: int | None = None
):
    base = _base_config()
    targets = (
        replace(base.targets, mode="sampled", explicit_list=(), bucket="all", count=1)
        if sampled
        else replace(
            base.targets,
            mode="explicit_list",
            explicit_list=(int(target_item),),
            bucket="all",
            count=1,
        )
    )
    return replace(
        base,
        artifacts=replace(base.artifacts, root=str(temp_root)),
        experiment=replace(
            base.experiment,
            name=("pts_cem_sampled_reuse" if sampled else "pts_cem_explicit_seed"),
        ),
        targets=targets,
    )


def _install_fake_victim(monkeypatch, execute_calls: list[dict[str, object]]) -> None:
    def fake_execute_single_victim(config, **kwargs):
        target_item = int(kwargs["target_item"])
        victim_name = str(kwargs["victim_name"])
        predictions_path = kwargs["predictions_path"]
        save_json(
            {
                "available": True,
                "count": 1,
                "rankings": [[7, 8, 9]],
                "target_item": target_item,
                "victim": victim_name,
            },
            predictions_path,
        )
        execute_calls.append({"target_item": target_item, "victim_name": victim_name})
        return VictimExecutionResult(
            predictions=[[7, 8, 9]],
            predictions_path=predictions_path,
            extra={},
            poisoned_train_path=None,
        )

    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.execute_single_victim",
        fake_execute_single_victim,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.resolve_ground_truth_labels",
        lambda *args, **kwargs: [1],
    )
    monkeypatch.setattr(
        "attack.pipeline.core.orchestrator.evaluate_prediction_metrics",
        lambda *args, **kwargs: ({"targeted_recall@10": 1.0}, True),
    )


def test_explicit_to_sampled_pts_cem_victim_prediction_cache_reuse(monkeypatch) -> None:
    with _temp_root() as temp_root:
        sampled_config = _config_for_temp_root(temp_root, sampled=True)
        sampled_target = int(
            build_ordered_target_cohort(_stats(), sampled_config)["ordered_targets"][0]
        )
        explicit_config = _config_for_temp_root(
            temp_root,
            sampled=False,
            target_item=sampled_target,
        )
        sessions_path = temp_root / "selected_sessions.json"
        save_json([[11, 22, sampled_target]], sessions_path)
        selected_sessions_sha1 = _sha1_file(sessions_path)
        execute_calls: list[dict[str, object]] = []
        _install_fake_victim(monkeypatch, execute_calls)

        def build_poisoned(target_item: int) -> TargetPoisonOutput:
            assert int(target_item) == sampled_target
            return TargetPoisonOutput(
                poisoned=PoisonedDataset(
                    sessions=[[1, 2], [11, 22]],
                    labels=[3, int(target_item)],
                    clean_count=1,
                    fake_count=1,
                ),
                metadata={
                    **_metadata(
                        target_item=int(target_item),
                        shared_cache_key="pts_cem_shared_cross_cohort",
                        sessions_sha1=selected_sessions_sha1,
                    ),
                    "pts_best_candidate_sessions_path": str(sessions_path),
                },
            )

        run_targets_and_victims(
            explicit_config,
            config_path=None,
            context=_context(explicit_config),
            run_type=RUN_TYPE,
            build_poisoned=build_poisoned,
        )
        run_targets_and_victims(
            sampled_config,
            config_path=None,
            context=_context(sampled_config),
            run_type=RUN_TYPE,
            build_poisoned=build_poisoned,
        )

        sampled_metrics = load_json(
            next(
                (temp_root / "runs" / "diginetica" / "pts_cem_sampled_reuse").rglob(
                    "targets/*/victims/srgnn/metrics.json"
                )
            )
        )

    assert execute_calls == [
        {"target_item": sampled_target, "victim_name": victim_name}
        for victim_name in explicit_config.victims.enabled
    ]
    assert sampled_metrics["reused_predictions"] is True
    assert (
        sampled_metrics["victim_prediction_attack_identity_mode"]
        == "pts_cem_target_level_construction"
    )


def _sha1_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read())
    return digest.hexdigest()
