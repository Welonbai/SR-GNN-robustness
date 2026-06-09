from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
    PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.runs.run_pts_construction_cem import (
    _load_json_dict,
    _load_json_sessions,
    build_pts_construction_attack_identity_context,
)


DEFAULT_PTS_CONSTRUCTION_CANDIDATE_REPLAY_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_pts_construction_grouped_cem_ratio1_srgnn_partial4.yaml"
)
_LOG_PREFIX = "[pts-cem-candidate-replay]"


def run_pts_construction_candidate_replay(
    config: Config,
    *,
    candidate_rank: int,
    target_item: int | None = None,
    experiment_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    candidate_rank = _validate_candidate_rank(candidate_rank)
    source_config = config
    replay_config = _with_replay_overrides(
        config,
        candidate_rank=candidate_rank,
        target_item=target_item,
        experiment_name=experiment_name,
    )
    _validate_replay_config(replay_config)
    replay_attack_identity_context = build_pts_candidate_replay_run_identity_context(
        source_config,
        candidate_rank=candidate_rank,
    )

    shared = prepare_shared_attack_artifacts(
        replay_config,
        run_type=PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )
    context = RunContext.from_shared(shared)

    print(
        f"{_LOG_PREFIX} candidate_rank={candidate_rank} "
        f"targets={list(replay_config.targets.explicit_list or []) or replay_config.targets.mode} "
        f"victims={list(replay_config.victims.enabled)}"
    )

    def build_poisoned(requested_target_item: int) -> TargetPoisonOutput:
        requested_target = int(requested_target_item)
        source = load_pts_cem_top_candidate_source(
            source_config,
            target_item=requested_target,
            candidate_rank=candidate_rank,
        )
        _validate_replay_sessions_count(
            source.sessions,
            template_session_count=len(shared.template_sessions),
            target_item=requested_target,
            candidate_rank=candidate_rank,
        )
        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            source.sessions,
        )
        replay_root = (
            target_dir(
                replay_config,
                requested_target,
                run_type=PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
                attack_identity_context=replay_attack_identity_context,
            )
            / "pts_candidate_replay"
            / f"rank_{candidate_rank}"
        )
        replay_metadata = _write_replay_artifacts(
            replay_root=replay_root,
            source=source,
            target_item=requested_target,
            candidate_rank=candidate_rank,
            replay_victims=replay_config.victims.enabled,
        )
        print(
            f"{_LOG_PREFIX} target={requested_target} replaying rank={candidate_rank} "
            f"sessions={source.sessions_path}"
        )
        return TargetPoisonOutput(
            poisoned=poisoned,
            raw_fake_sessions=source.sessions,
            metadata=_target_replay_metadata(
                source=source,
                replay_metadata=replay_metadata,
                target_item=requested_target,
                candidate_rank=candidate_rank,
            ),
        )

    summary = run_targets_and_victims(
        replay_config,
        config_path=config_path,
        context=context,
        run_type=PTS_CONSTRUCTION_CANDIDATE_REPLAY_RUN_TYPE,
        build_poisoned=build_poisoned,
        attack_identity_context=replay_attack_identity_context,
    )
    print(f"{_LOG_PREFIX} candidate replay completed.")
    return summary


class PTSCEMTopCandidateSource:
    def __init__(
        self,
        *,
        target_item: int,
        candidate_rank: int,
        source_artifact_dir: Path,
        rank_dir: Path,
        sessions_path: Path,
        metadata_path: Path,
        policy_path: Path,
        session_records_path: Path | None,
        top_candidates_path: Path | None,
        top_candidate_policies_path: Path | None,
        cem_trace_path: Path | None,
        sessions: list[list[int]],
        metadata: dict[str, object],
        policy: dict[str, object],
        attack_identity_context: dict[str, object],
    ) -> None:
        self.target_item = int(target_item)
        self.candidate_rank = int(candidate_rank)
        self.source_artifact_dir = Path(source_artifact_dir)
        self.rank_dir = Path(rank_dir)
        self.sessions_path = Path(sessions_path)
        self.metadata_path = Path(metadata_path)
        self.policy_path = Path(policy_path)
        self.session_records_path = session_records_path
        self.top_candidates_path = top_candidates_path
        self.top_candidate_policies_path = top_candidate_policies_path
        self.cem_trace_path = cem_trace_path
        self.sessions = sessions
        self.metadata = metadata
        self.policy = policy
        self.attack_identity_context = attack_identity_context


def load_pts_cem_top_candidate_source(
    config: Config,
    *,
    target_item: int,
    candidate_rank: int,
) -> PTSCEMTopCandidateSource:
    candidate_rank = _validate_candidate_rank(candidate_rank)
    source_artifact_dir = _source_pts_cem_artifact_dir(
        config,
        target_item=int(target_item),
    )
    paths = resolve_pts_cem_top_candidate_paths(
        source_artifact_dir,
        candidate_rank=candidate_rank,
    )
    sessions_path = paths["sessions"]
    if not sessions_path.exists():
        raise FileNotFoundError(
            f"PTS-CEM candidate rank {candidate_rank} sessions not found. "
            f"Run PTS-CEM with save_top_k_candidates >= {candidate_rank} and "
            f"save_top_candidate_sessions=true first. Missing: {sessions_path}"
        )
    metadata_path = paths["metadata"]
    policy_path = paths["policy"]
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PTS-CEM candidate rank {candidate_rank} metadata not found: {metadata_path}"
        )
    if not policy_path.exists():
        raise FileNotFoundError(
            f"PTS-CEM candidate rank {candidate_rank} policy not found: {policy_path}"
        )
    sessions = _load_json_sessions(sessions_path)
    metadata = _load_json_dict(metadata_path)
    policy = _load_json_dict(policy_path)
    _validate_source_metadata_target(
        metadata,
        target_item=int(target_item),
        label=str(metadata_path),
    )
    attack_identity_context = build_pts_candidate_replay_attack_identity_context(
        target_item=int(target_item),
        candidate_rank=candidate_rank,
        source_artifact_dir=source_artifact_dir,
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        policy_path=policy_path,
        metadata=metadata,
    )
    return PTSCEMTopCandidateSource(
        target_item=int(target_item),
        candidate_rank=candidate_rank,
        source_artifact_dir=source_artifact_dir,
        rank_dir=paths["rank_dir"],
        sessions_path=sessions_path,
        metadata_path=metadata_path,
        policy_path=policy_path,
        session_records_path=(
            paths["session_records"] if paths["session_records"].exists() else None
        ),
        top_candidates_path=(
            paths["top_candidates"] if paths["top_candidates"].exists() else None
        ),
        top_candidate_policies_path=(
            paths["top_candidate_policies"]
            if paths["top_candidate_policies"].exists()
            else None
        ),
        cem_trace_path=paths["cem_trace"] if paths["cem_trace"].exists() else None,
        sessions=sessions,
        metadata=metadata,
        policy=policy,
        attack_identity_context=attack_identity_context,
    )


def resolve_pts_cem_top_candidate_paths(
    artifact_dir: Path,
    *,
    candidate_rank: int,
) -> dict[str, Path]:
    candidate_rank = _validate_candidate_rank(candidate_rank)
    root = Path(artifact_dir)
    rank_dir = root / "top_candidates" / f"rank_{candidate_rank}"
    return {
        "artifact_dir": root,
        "rank_dir": rank_dir,
        "sessions": rank_dir / "sessions.json",
        "metadata": rank_dir / "metadata.json",
        "policy": rank_dir / "policy.json",
        "session_records": rank_dir / "session_records.jsonl",
        "top_candidates": root / "pts_top_candidates.json",
        "top_candidate_policies": root / "pts_top_candidate_policies.json",
        "cem_trace": root / "pts_cem_trace.jsonl",
    }


def build_pts_candidate_replay_attack_identity_context(
    *,
    target_item: int,
    candidate_rank: int,
    source_artifact_dir: Path,
    sessions_path: Path,
    metadata_path: Path,
    policy_path: Path,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "pts_candidate_replay_source_candidate": {
            "candidate_rank": int(candidate_rank),
            "source_run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
        },
        "target_item": int(target_item),
        "source_pts_cem_artifact_dir": str(source_artifact_dir),
        "source_candidate": {
            "rank": int(candidate_rank),
            "iteration": _optional_int(metadata.get("iteration")),
            "candidate_id": _optional_int(metadata.get("candidate_id")),
            "candidate_seed": _optional_int(metadata.get("candidate_seed")),
            "reward": _optional_float(metadata.get("reward")),
        },
        "source_hashes": {
            "sessions_sha1": _sha1_file(sessions_path),
            "metadata_sha1": _sha1_file(metadata_path),
            "policy_sha1": _sha1_file(policy_path),
        },
    }


def build_pts_candidate_replay_run_identity_context(
    config: Config,
    *,
    candidate_rank: int,
) -> dict[str, object]:
    return {
        "pts_candidate_replay": {
            "candidate_rank": int(candidate_rank),
            "source_run_type": PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            "source_experiment_name": config.experiment.name,
            "source_pts_construction_identity": (
                build_pts_construction_attack_identity_context(config)
            ),
        }
    }


def _source_pts_cem_artifact_dir(
    config: Config,
    *,
    target_item: int,
) -> Path:
    return (
        target_dir(
            config,
            int(target_item),
            run_type=PTS_CONSTRUCTION_GROUPED_CEM_RUN_TYPE,
            attack_identity_context=build_pts_construction_attack_identity_context(config),
        )
        / "pts_construction_cem"
    )


def _with_replay_overrides(
    config: Config,
    *,
    candidate_rank: int,
    target_item: int | None,
    experiment_name: str | None,
) -> Config:
    if experiment_name is None or not str(experiment_name).strip():
        experiment_name = (
            f"{config.experiment.name}_pts_candidate_replay_rank{int(candidate_rank)}"
        )
    targets = config.targets
    if target_item is not None:
        targets = replace(
            targets,
            mode="explicit_list",
            explicit_list=(int(target_item),),
            count=1,
        )
    return replace(
        config,
        experiment=replace(config.experiment, name=str(experiment_name)),
        targets=targets,
    )


def _validate_replay_config(config: Config) -> None:
    if not bool(config.data.poison_train_only):
        raise ValueError("PTS-CEM candidate replay requires data.poison_train_only=true.")
    if config.attack.pts_construction is None:
        raise ValueError("PTS-CEM candidate replay requires attack.pts_construction.")


def _validate_candidate_rank(candidate_rank: int) -> int:
    rank = int(candidate_rank)
    if rank <= 0:
        raise ValueError("candidate_rank must be positive.")
    return rank


def _validate_replay_sessions_count(
    sessions: Sequence[Sequence[int]],
    *,
    template_session_count: int,
    target_item: int,
    candidate_rank: int,
) -> None:
    if len(sessions) != int(template_session_count):
        raise ValueError(
            "PTS-CEM candidate replay sessions count does not match shared fake "
            f"sessions for target {int(target_item)} rank {int(candidate_rank)}: "
            f"expected {int(template_session_count)}, found {len(sessions)}."
        )


def _validate_source_metadata_target(
    metadata: Mapping[str, object],
    *,
    target_item: int,
    label: str,
) -> None:
    if "target_item" not in metadata:
        return
    saved = _optional_int(metadata.get("target_item"))
    if saved != int(target_item):
        raise ValueError(
            "PTS-CEM candidate metadata target_item mismatch: "
            f"expected {int(target_item)}, found {saved}. Source={label}"
        )


def _write_replay_artifacts(
    *,
    replay_root: Path,
    source: PTSCEMTopCandidateSource,
    target_item: int,
    candidate_rank: int,
    replay_victims: Sequence[str],
) -> dict[str, object]:
    replay_root.mkdir(parents=True, exist_ok=True)
    source_policy_path = replay_root / "replay_source_policy.json"
    source_metadata_path = replay_root / "replay_candidate_metadata.json"
    source_sessions_path_txt = replay_root / "replay_source_sessions_path.txt"
    comparison_path = replay_root / "comparison_summary.json"
    save_json(source.policy, source_policy_path)
    replay_metadata = _source_replay_metadata_payload(
        source=source,
        target_item=target_item,
        candidate_rank=candidate_rank,
        replay_victims=replay_victims,
        replay_root=replay_root,
        source_policy_path=source_policy_path,
        source_metadata_path=source_metadata_path,
        source_sessions_path_txt=source_sessions_path_txt,
        comparison_path=comparison_path,
    )
    save_json(replay_metadata, source_metadata_path)
    source_sessions_path_txt.write_text(str(source.sessions_path), encoding="utf-8")
    save_json(_comparison_summary_payload(replay_metadata), comparison_path)
    return replay_metadata


def _source_replay_metadata_payload(
    *,
    source: PTSCEMTopCandidateSource,
    target_item: int,
    candidate_rank: int,
    replay_victims: Sequence[str],
    replay_root: Path,
    source_policy_path: Path,
    source_metadata_path: Path,
    source_sessions_path_txt: Path,
    comparison_path: Path,
) -> dict[str, object]:
    reward_metrics = source.metadata.get("reward_metrics")
    if not isinstance(reward_metrics, Mapping):
        reward_metrics = {}
    payload = {
        "target_item": int(target_item),
        "candidate_rank": int(candidate_rank),
        "candidate_iteration": _optional_int(source.metadata.get("iteration")),
        "candidate_id": _optional_int(source.metadata.get("candidate_id")),
        "candidate_seed": _optional_int(source.metadata.get("candidate_seed")),
        "candidate_validation_reward": _optional_float(source.metadata.get("reward")),
        "candidate_reward_metrics": dict(reward_metrics),
        "replay_victims": list(replay_victims),
        "source_paths": {
            "artifact_dir": str(source.source_artifact_dir),
            "rank_dir": str(source.rank_dir),
            "sessions": str(source.sessions_path),
            "metadata": str(source.metadata_path),
            "policy": str(source.policy_path),
            "session_records": (
                str(source.session_records_path)
                if source.session_records_path is not None
                else None
            ),
            "top_candidates": (
                str(source.top_candidates_path)
                if source.top_candidates_path is not None
                else None
            ),
            "top_candidate_policies": (
                str(source.top_candidate_policies_path)
                if source.top_candidate_policies_path is not None
                else None
            ),
            "cem_trace": str(source.cem_trace_path) if source.cem_trace_path else None,
        },
        "replay_paths": {
            "replay_root": str(replay_root),
            "replay_candidate_metadata": str(source_metadata_path),
            "replay_source_policy": str(source_policy_path),
            "replay_source_sessions_path": str(source_sessions_path_txt),
            "comparison_summary": str(comparison_path),
        },
        "source_hashes": {
            "sessions_sha1": _sha1_file(source.sessions_path),
            "metadata_sha1": _sha1_file(source.metadata_path),
            "policy_sha1": _sha1_file(source.policy_path),
        },
        "source_candidate_metadata": source.metadata,
        "note": "Victim metrics are stored in the standard replay run victim outputs.",
    }
    return _to_jsonable(payload)


def _comparison_summary_payload(
    replay_metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        "target_item": replay_metadata.get("target_item"),
        "candidate_rank": replay_metadata.get("candidate_rank"),
        "source_candidate": {
            "iteration": replay_metadata.get("candidate_iteration"),
            "candidate_id": replay_metadata.get("candidate_id"),
            "candidate_seed": replay_metadata.get("candidate_seed"),
            "validation_reward": replay_metadata.get("candidate_validation_reward"),
            "reward_metrics": replay_metadata.get("candidate_reward_metrics", {}),
        },
        "source_paths": replay_metadata.get("source_paths", {}),
        "replay_victims": replay_metadata.get("replay_victims", []),
        "note": "Victim metrics are stored in the standard replay run victim outputs.",
    }


def _target_replay_metadata(
    *,
    source: PTSCEMTopCandidateSource,
    replay_metadata: Mapping[str, object],
    target_item: int,
    candidate_rank: int,
) -> dict[str, object]:
    return {
        "pts_candidate_replay": True,
        "pts_replay_candidate_rank": int(candidate_rank),
        "pts_replay_source_sessions_path": str(source.sessions_path),
        "pts_replay_source_metadata_path": str(source.metadata_path),
        "pts_replay_source_policy_path": str(source.policy_path),
        "pts_replay_metadata_path": replay_metadata["replay_paths"][
            "replay_candidate_metadata"
        ],
        "pts_replay_comparison_summary_path": replay_metadata["replay_paths"][
            "comparison_summary"
        ],
        "pts_candidate_iteration": replay_metadata.get("candidate_iteration"),
        "pts_candidate_id": replay_metadata.get("candidate_id"),
        "pts_candidate_seed": replay_metadata.get("candidate_seed"),
        "pts_candidate_validation_reward": replay_metadata.get(
            "candidate_validation_reward"
        ),
        "pts_candidate_reward_metrics": replay_metadata.get(
            "candidate_reward_metrics", {}
        ),
        "pts_replay_source_candidate_metadata": source.metadata,
        "pts_cem_reused": True,
        "pts_cem_cache_mode": f"candidate_replay_rank_{int(candidate_rank)}",
        "pts_final_selection_mode": "candidate_replay",
        "target_item": int(target_item),
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Replay a saved PTS-CEM top candidate through victim evaluation."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PTS_CONSTRUCTION_CANDIDATE_REPLAY_CONFIG_PATH,
        help="Path to the PTS-CEM YAML config used for source artifacts.",
    )
    parser.add_argument(
        "--candidate-rank",
        type=int,
        required=True,
        help="Saved PTS-CEM top-candidate rank to replay, e.g. 2.",
    )
    parser.add_argument(
        "--target-item",
        type=int,
        default=None,
        help="Optional single target item to replay without editing YAML.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional replay experiment name override.",
    )
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    config = load_config(config_path)
    run_pts_construction_candidate_replay(
        config,
        candidate_rank=int(args.candidate_rank),
        target_item=(None if args.target_item is None else int(args.target_item)),
        experiment_name=args.experiment_name,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PTS_CONSTRUCTION_CANDIDATE_REPLAY_CONFIG_PATH",
    "PTSCEMTopCandidateSource",
    "build_pts_candidate_replay_attack_identity_context",
    "build_pts_candidate_replay_run_identity_context",
    "load_pts_cem_top_candidate_source",
    "main",
    "resolve_pts_cem_top_candidate_paths",
    "run_pts_construction_candidate_replay",
    "_source_pts_cem_artifact_dir",
    "_target_replay_metadata",
    "_validate_candidate_rank",
]
