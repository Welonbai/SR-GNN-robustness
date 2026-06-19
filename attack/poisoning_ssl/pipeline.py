from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Sequence

from attack.common.artifact_io import save_fake_sessions, save_json
from attack.common.config import (
    Config,
    PoisoningSSLSBRConfig,
    POISONING_SSL_SBR_GENERATION_BACKEND_REAL,
    POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_FIXED,
    POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_TRAIN_SUB_P99,
)
from attack.common.paths import target_dir
from attack.common.seed import derive_seed
from attack.poisoning_ssl import ADAPTED_METHOD_NAME, METHOD_NAME, ORIGINAL_METHOD_NAME
from attack.poisoning_ssl.dataset_bridge import export_pseudo_user_sequences
from attack.poisoning_ssl.diagnostics import (
    budget_diagnostics,
    duplicate_diagnostics,
    length_stats,
    length_stats_from_lengths,
    nearest_rank_percentile,
    stringify_mapping_keys,
    target_diagnostics,
)
from attack.poisoning_ssl.generator import (
    CandidateGenerator,
    GenerationRequest,
    RealSeqPoisonCandidateGenerator,
)
from attack.poisoning_ssl.postprocess import postprocess_fake_user_sequences
from attack.poisoning_ssl.provenance import (
    UPSTREAM_COMMIT,
    UPSTREAM_URL,
    provenance_payload,
)


@dataclass(frozen=True)
class PoisoningSSLSBRTargetResult:
    raw_fake_sessions: list[list[int]]
    metadata: dict[str, object]


def compute_seqpoison_max_seq_len(
    train_sub_sessions: Sequence[Sequence[int]],
    poisoning_ssl_config: PoisoningSSLSBRConfig,
) -> int:
    if poisoning_ssl_config.max_seq_len_override is not None:
        return int(poisoning_ssl_config.max_seq_len_override)
    if (
        poisoning_ssl_config.max_seq_len_policy
        == POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_FIXED
    ):
        raise ValueError(
            "max_seq_len_override is required when max_seq_len_policy == 'fixed'."
        )
    if (
        poisoning_ssl_config.max_seq_len_policy
        != POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_TRAIN_SUB_P99
    ):
        raise ValueError(
            "Unsupported SeqPoison-SBR max_seq_len_policy: "
            f"{poisoning_ssl_config.max_seq_len_policy!r}."
        )
    lengths = [len(session) for session in train_sub_sessions]
    p99 = nearest_rank_percentile(lengths, 99)
    return int(min(int(poisoning_ssl_config.original_max_seq_len_cap), int(p99)))


def generate_poisoning_ssl_sbr_target(
    *,
    config: Config,
    shared,
    target_item: int,
    run_type: str,
    n_fake_requested: int,
    candidate_generator: CandidateGenerator | None = None,
) -> PoisoningSSLSBRTargetResult:
    total_start_time = _now_iso()
    total_start_perf = time.perf_counter()
    poisoning_config = config.attack.poisoning_ssl_sbr
    if poisoning_config is None:
        raise ValueError("SeqPoison-SBR requires attack.poisoning_ssl_sbr.")
    if bool(poisoning_config.enforce_nonzero_target_position):
        raise NotImplementedError(
            "SeqPoison-SBR Phase 1 does not implement nonzero target-position "
            "constraints. Set attack.poisoning_ssl_sbr.enforce_nonzero_target_position "
            "to false."
        )
    target = int(target_item)
    n_fake = int(n_fake_requested)
    if n_fake <= 0:
        raise ValueError("n_fake_requested must be positive.")

    target_root = target_dir(config, target, run_type=run_type)
    target_root.mkdir(parents=True, exist_ok=True)
    generation_dir = target_root / "poisoning_ssl_sbr_generation"
    max_seq_len = compute_seqpoison_max_seq_len(
        shared.canonical_dataset.train_sub,
        poisoning_config,
    )
    valid_item_ids = _valid_item_ids(shared.canonical_dataset)
    bridge = export_pseudo_user_sequences(
        shared.canonical_dataset.train_sub,
        target_item=target,
        output_dir=generation_dir / "dataset_bridge",
        valid_item_ids=valid_item_ids,
        max_seq_len=max_seq_len,
        max_train_sequences=poisoning_config.max_train_sequences,
    )
    generation_seed = derive_seed(
        int(config.seeds.fake_session_seed) + int(poisoning_config.generation_seed_offset),
        "poisoning_ssl_sbr",
        target,
    )
    postprocess_seed = derive_seed(generation_seed, "postprocess")
    generator = candidate_generator or _default_candidate_generator(poisoning_config)

    candidate_save_policy = str(poisoning_config.candidate_save_policy)
    max_saved_candidates = int(poisoning_config.max_saved_candidates)
    saved_candidates: list[list[int]] = []
    candidate_lengths: list[int] = []
    valid_sessions: list[list[int]] = []
    cumulative_counts: dict[str, int | float] = {
        "n_generated_candidates": 0,
        "invalid_item_count": 0,
        "filtered_short_session_count": 0,
        "no_target_count": 0,
        "multi_target_count": 0,
        "target_containing_candidate_count_before_single_target_filter": 0,
    }
    generation_round_metadata: list[dict[str, object]] = []
    raw_candidate_count_by_round: list[int] = []
    valid_count_by_round: list[int] = []
    filter_count_by_round: list[int] = []
    generation_round_durations_sec: list[float] = []
    generation_start_time = _now_iso()
    generation_start_perf = time.perf_counter()
    for round_index in range(int(poisoning_config.max_generation_rounds)):
        request = GenerationRequest(
            target_item=target,
            n_candidates=max(1, n_fake * int(poisoning_config.candidate_multiplier)),
            max_seq_len=max_seq_len,
            seed=derive_seed(generation_seed, "round", round_index),
            output_dir=generation_dir,
            round_index=round_index,
            dataset_bundle=bridge,
            valid_item_ids=valid_item_ids,
            config=poisoning_config,
            training_seed=generation_seed,
        )
        round_candidates = generator.generate(request)
        raw_candidate_count_by_round.append(int(len(round_candidates)))
        normalized_round_candidates = [
            [int(item) for item in session] for session in round_candidates
        ]
        candidate_lengths.extend(int(len(session)) for session in normalized_round_candidates)
        _retain_candidates_for_policy(
            saved_candidates=saved_candidates,
            round_candidates=normalized_round_candidates,
            candidate_save_policy=candidate_save_policy,
            max_saved_candidates=max_saved_candidates,
        )
        round_postprocess = postprocess_fake_user_sequences(
            normalized_round_candidates,
            target_item=target,
            valid_item_ids=valid_item_ids,
            n_fake=max(1, len(normalized_round_candidates)),
            enforce_single_target=bool(poisoning_config.enforce_single_target),
            filter_no_target=bool(poisoning_config.filter_no_target),
            filter_short_sessions=bool(poisoning_config.filter_short_sessions),
            remove_user_id=True,
        )
        _merge_postprocess_counts(cumulative_counts, round_postprocess.counts)
        valid_sessions.extend([list(session) for session in round_postprocess.valid_sessions])
        final_sessions = [list(session) for session in valid_sessions[:n_fake]]
        cumulative_counts["n_after_filtering"] = int(len(valid_sessions))
        cumulative_counts["n_final_injected"] = int(len(final_sessions))
        total_generated_so_far = int(cumulative_counts["n_generated_candidates"])
        target_containing = int(
            cumulative_counts[
                "target_containing_candidate_count_before_single_target_filter"
            ]
        )
        cumulative_counts[
            "target_containing_candidate_ratio_before_single_target_filter"
        ] = (
            0.0
            if total_generated_so_far <= 0
            else float(target_containing / total_generated_so_far)
        )
        valid_count_by_round.append(int(cumulative_counts.get("n_after_filtering", 0)))
        filter_count_by_round.append(
            int(cumulative_counts["n_generated_candidates"])
            - int(cumulative_counts.get("n_after_filtering", 0))
        )
        last_metadata = getattr(generator, "last_metadata", None)
        if isinstance(last_metadata, dict) and last_metadata:
            generation_round_metadata.append(dict(last_metadata))
            generation_round_durations_sec.append(
                float(last_metadata.get("generation_round_duration_sec", 0.0))
            )
        else:
            generation_round_durations_sec.append(0.0)
        if len(final_sessions) == n_fake:
            break

    generation_end_time = _now_iso()
    generation_duration_sec = float(time.perf_counter() - generation_start_perf)
    if not raw_candidate_count_by_round:
        raise RuntimeError("SeqPoison-SBR generation did not execute any rounds.")
    final_sessions = [list(session) for session in valid_sessions[:n_fake]]

    metadata = _metadata(
        config=config,
        shared=shared,
        target_item=target,
        n_fake_requested=n_fake,
        max_seq_len=max_seq_len,
        generation_seed=generation_seed,
        postprocess_seed=postprocess_seed,
        candidate_lengths=candidate_lengths,
        final_sessions=final_sessions,
        postprocess_counts=cumulative_counts,
        bridge_metadata=bridge.metadata,
        generation_round_metadata=generation_round_metadata,
        raw_candidate_count_by_round=raw_candidate_count_by_round,
        valid_count_by_round=valid_count_by_round,
        filter_count_by_round=filter_count_by_round,
        generation_round_durations_sec=generation_round_durations_sec,
        total_start_time=total_start_time,
        total_end_time=_now_iso(),
        total_duration_sec=float(time.perf_counter() - total_start_perf),
        generation_start_time=generation_start_time,
        generation_end_time=generation_end_time,
        generation_duration_sec=generation_duration_sec,
        saved_candidate_count=len(saved_candidates),
    )
    max_observed_length, max_seq_len_violation_count = _max_seq_len_violations(
        final_sessions,
        max_seq_len=int(max_seq_len),
    )
    metadata["max_observed_final_length"] = int(max_observed_length)
    metadata["max_seq_len_violation_count"] = int(max_seq_len_violation_count)
    _write_artifacts(
        target_root=target_root,
        saved_candidates=saved_candidates,
        final_sessions=final_sessions,
        metadata=metadata,
        candidate_save_policy=candidate_save_policy,
    )
    if len(final_sessions) != n_fake:
        raise RuntimeError(
            "SeqPoison-SBR could not generate enough valid fake sessions: "
            f"requested={n_fake}, final={len(final_sessions)}, "
            f"candidates={int(cumulative_counts['n_generated_candidates'])}. "
            "See poisoning_ssl_sbr_metadata.json."
        )
    if max_seq_len_violation_count > 0:
        raise RuntimeError(
            "SeqPoison-SBR generated final sessions longer than max_seq_len; "
            f"max_seq_len={int(max_seq_len)}, "
            f"max_observed_length={int(max_observed_length)}, "
            f"violation_count={int(max_seq_len_violation_count)}. "
            "Phase 1 does not crop or truncate non-padding tokens."
        )
    return PoisoningSSLSBRTargetResult(
        raw_fake_sessions=[list(session) for session in final_sessions],
        metadata=metadata,
    )


def _default_candidate_generator(
    poisoning_config: PoisoningSSLSBRConfig,
) -> CandidateGenerator:
    if poisoning_config.generation_backend == POISONING_SSL_SBR_GENERATION_BACKEND_REAL:
        return RealSeqPoisonCandidateGenerator()
    raise ValueError(
        "Unsupported SeqPoison-SBR generation_backend: "
        f"{poisoning_config.generation_backend!r}."
    )


def _metadata(
    *,
    config: Config,
    shared,
    target_item: int,
    n_fake_requested: int,
    max_seq_len: int,
    generation_seed: int,
    postprocess_seed: int,
    candidate_lengths: list[int],
    final_sessions: list[list[int]],
    postprocess_counts: dict[str, int | float],
    bridge_metadata: dict[str, object],
    generation_round_metadata: list[dict[str, object]],
    raw_candidate_count_by_round: list[int],
    valid_count_by_round: list[int],
    filter_count_by_round: list[int],
    generation_round_durations_sec: list[float],
    total_start_time: str,
    total_end_time: str,
    total_duration_sec: float,
    generation_start_time: str,
    generation_end_time: str,
    generation_duration_sec: float,
    saved_candidate_count: int,
) -> dict[str, object]:
    target_stats = target_diagnostics(final_sessions, target_item=int(target_item))
    duplicate_stats = duplicate_diagnostics(final_sessions)
    budget_stats = budget_diagnostics(
        final_sessions,
        target_item=int(target_item),
        clean_label_count=int(len(shared.clean_labels)),
    )
    n_generated = int(postprocess_counts.get("n_generated_candidates", 0))
    n_after_filtering = int(postprocess_counts.get("n_after_filtering", 0))
    n_final = int(len(final_sessions))
    poisoning_config = config.attack.poisoning_ssl_sbr
    latest_generation_metadata = (
        generation_round_metadata[-1] if generation_round_metadata else {}
    )
    metadata = {
        "method_name": METHOD_NAME,
        "original_method_name": ORIGINAL_METHOD_NAME,
        "adapted_method_name": ADAPTED_METHOD_NAME,
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "target_item": int(target_item),
        "n_fake_requested": int(n_fake_requested),
        "n_generated_candidates": n_generated,
        "n_after_filtering": n_after_filtering,
        "n_final_injected": n_final,
        "acceptance_rate": 0.0 if n_generated <= 0 else float(n_after_filtering / n_generated),
        "total_start_time": total_start_time,
        "total_end_time": total_end_time,
        "total_duration_sec": float(total_duration_sec),
        "training_start_time": latest_generation_metadata.get("training_start_time"),
        "training_end_time": latest_generation_metadata.get("training_end_time"),
        "training_duration_sec": latest_generation_metadata.get("training_duration_sec"),
        "generation_start_time": generation_start_time,
        "generation_end_time": generation_end_time,
        "generation_duration_sec": float(generation_duration_sec),
        "max_seq_len_policy": (
            None if poisoning_config is None else poisoning_config.max_seq_len_policy
        ),
        "max_seq_len_value": int(max_seq_len),
        "clean_train_length_stats": length_stats(shared.canonical_dataset.train_sub),
        "generated_candidate_length_stats": length_stats_from_lengths(candidate_lengths),
        "final_fake_length_stats": length_stats(final_sessions),
        "generation_seed": int(generation_seed),
        "postprocess_seed": int(postprocess_seed),
        "invalid_item_count": int(postprocess_counts.get("invalid_item_count", 0)),
        "filtered_short_session_count": int(
            postprocess_counts.get("filtered_short_session_count", 0)
        ),
        "no_target_count": int(postprocess_counts.get("no_target_count", 0)),
        "multi_target_count": int(postprocess_counts.get("multi_target_count", 0)),
        "target_containing_candidate_count_before_single_target_filter": int(
            postprocess_counts.get(
                "target_containing_candidate_count_before_single_target_filter",
                0,
            )
        ),
        "target_containing_candidate_ratio_before_single_target_filter": float(
            postprocess_counts.get(
                "target_containing_candidate_ratio_before_single_target_filter",
                0.0,
            )
        ),
        "target_acceptance_failure_reason": _target_acceptance_failure_reason(
            n_generated=n_generated,
            n_after_filtering=n_after_filtering,
            postprocess_counts=postprocess_counts,
        ),
        "single_target_count": int(target_stats["single_target_count"]),
        "target_occurrence_stats": target_stats["target_occurrence_stats"],
        "target_position_distribution": target_stats["target_position_distribution"],
        "target_pos0_count": int(target_stats["target_pos0_count"]),
        "target_pos0_ratio": float(target_stats["target_pos0_ratio"]),
        "target_nonzero_count": int(target_stats["target_nonzero_count"]),
        "target_nonzero_ratio": float(target_stats["target_nonzero_ratio"]),
        "dataset_bridge": bridge_metadata,
        "generation_backend": (
            None if poisoning_config is None else poisoning_config.generation_backend
        ),
        "real_generation_implemented": bool(
            latest_generation_metadata.get("real_generation_implemented", False)
        ),
        "upstream_component_map": provenance_payload()["upstream_migration_map"],
        "enabled_reward_components": latest_generation_metadata.get(
            "enabled_reward_components",
            [],
        ),
        "classifier_checkpoint_path": latest_generation_metadata.get(
            "classifier_checkpoint_path"
        ),
        "generator_checkpoint_path": latest_generation_metadata.get(
            "generator_checkpoint_path"
        ),
        "discriminator_checkpoint_path": latest_generation_metadata.get(
            "discriminator_checkpoint_path"
        ),
        "training_epochs": latest_generation_metadata.get("training_epochs", {}),
        "classifier_training_duration_sec": latest_generation_metadata.get(
            "classifier_training_duration_sec"
        ),
        "mle_pretraining_duration_sec": latest_generation_metadata.get(
            "mle_pretraining_duration_sec"
        ),
        "discriminator_pretraining_duration_sec": latest_generation_metadata.get(
            "discriminator_pretraining_duration_sec"
        ),
        "adversarial_training_duration_sec": latest_generation_metadata.get(
            "adversarial_training_duration_sec"
        ),
        "classifier_epoch_durations_sec": latest_generation_metadata.get(
            "classifier_epoch_durations_sec",
            [],
        ),
        "mle_epoch_durations_sec": latest_generation_metadata.get(
            "mle_epoch_durations_sec",
            [],
        ),
        "adversarial_epoch_durations_sec": latest_generation_metadata.get(
            "adversarial_epoch_durations_sec",
            [],
        ),
        "discriminator_update_durations_sec": latest_generation_metadata.get(
            "discriminator_update_durations_sec",
            [],
        ),
        "acceptance_evaluations": latest_generation_metadata.get(
            "acceptance_evaluations",
            [],
        ),
        "batch_size": latest_generation_metadata.get("batch_size"),
        "learning_rate": latest_generation_metadata.get("learning_rate"),
        "embedding_dim": latest_generation_metadata.get("embedding_dim"),
        "hidden_dim": latest_generation_metadata.get("hidden_dim"),
        "device": latest_generation_metadata.get("device"),
        "generation_rounds_used": int(len(raw_candidate_count_by_round)),
        "candidate_multiplier": (
            None if poisoning_config is None else int(poisoning_config.candidate_multiplier)
        ),
        "max_generation_rounds": (
            None if poisoning_config is None else int(poisoning_config.max_generation_rounds)
        ),
        "raw_candidate_count_by_round": list(raw_candidate_count_by_round),
        "valid_count_by_round": list(valid_count_by_round),
        "filter_count_by_round": list(filter_count_by_round),
        "generation_round_durations_sec": list(generation_round_durations_sec),
        "raw_candidates_generated_total": n_generated,
        "valid_sessions_generated_total": n_after_filtering,
        "candidates_per_second": (
            0.0
            if generation_duration_sec <= 0.0
            else float(n_generated / generation_duration_sec)
        ),
        "valid_sessions_per_second": (
            0.0
            if generation_duration_sec <= 0.0
            else float(n_after_filtering / generation_duration_sec)
        ),
        "candidate_save_policy": (
            None if poisoning_config is None else poisoning_config.candidate_save_policy
        ),
        "max_saved_candidates": (
            None if poisoning_config is None else int(poisoning_config.max_saved_candidates)
        ),
        "saved_generated_candidate_count": int(saved_candidate_count),
        "generation_round_metadata": generation_round_metadata,
        "remap_used": bool(bridge_metadata.get("remap_applied", False)),
        "item_id_mapping_path": bridge_metadata.get("item_id_mapping_path"),
        **duplicate_stats,
        **budget_stats,
        "provenance": provenance_payload(),
        "phase1_interface_mock_only": False,
        "reportable_baseline": False,
    }
    return stringify_mapping_keys(metadata)  # stable JSON shape for int-keyed histograms


def _target_acceptance_failure_reason(
    *,
    n_generated: int,
    n_after_filtering: int,
    postprocess_counts: dict[str, int],
) -> str | None:
    if int(n_generated) <= 0 or int(n_after_filtering) > 0:
        return None
    if int(postprocess_counts.get("no_target_count", 0)) == int(n_generated):
        return "no_target_item_appeared_in_any_candidate"
    if int(postprocess_counts.get("invalid_item_count", 0)) == int(n_generated):
        return "all_candidates_had_invalid_items"
    if int(postprocess_counts.get("filtered_short_session_count", 0)) == int(n_generated):
        return "all_candidates_were_too_short"
    if int(postprocess_counts.get("multi_target_count", 0)) == int(n_generated):
        return "all_target_candidates_had_multiple_target_occurrences"
    return "mixed_filter_reasons"


def _merge_postprocess_counts(
    cumulative: dict[str, int | float],
    current: dict[str, int | float],
) -> None:
    additive_keys = (
        "n_generated_candidates",
        "invalid_item_count",
        "filtered_short_session_count",
        "no_target_count",
        "multi_target_count",
        "target_containing_candidate_count_before_single_target_filter",
    )
    for key in additive_keys:
        cumulative[key] = int(cumulative.get(key, 0)) + int(current.get(key, 0))


def _retain_candidates_for_policy(
    *,
    saved_candidates: list[list[int]],
    round_candidates: list[list[int]],
    candidate_save_policy: str,
    max_saved_candidates: int,
) -> None:
    if candidate_save_policy == "all":
        saved_candidates.extend([list(candidate) for candidate in round_candidates])
        return
    if candidate_save_policy != "sample":
        return
    remaining = max(0, int(max_saved_candidates) - len(saved_candidates))
    if remaining <= 0:
        return
    saved_candidates.extend([list(candidate) for candidate in round_candidates[:remaining]])


def _write_artifacts(
    *,
    target_root: Path,
    saved_candidates: list[list[int]],
    final_sessions: list[list[int]],
    metadata: dict[str, object],
    candidate_save_policy: str,
) -> None:
    if candidate_save_policy in {"sample", "all"}:
        save_fake_sessions(saved_candidates, target_root / "generated_candidates.pkl")
    save_fake_sessions(final_sessions, target_root / "raw_fake_sessions.pkl")
    save_json(metadata, target_root / "poisoning_ssl_sbr_metadata.json")
    save_json(
        {
            "clean_train_length_stats": metadata["clean_train_length_stats"],
            "generated_candidate_length_stats": metadata["generated_candidate_length_stats"],
            "final_fake_length_stats": metadata["final_fake_length_stats"],
        },
        target_root / "length_distribution.json",
    )
    save_json(provenance_payload(), target_root / "provenance.json")
    save_json(
        {
            "generation_backend": metadata.get("generation_backend"),
            "generation_rounds_used": metadata.get("generation_rounds_used"),
            "raw_candidate_count_by_round": metadata.get("raw_candidate_count_by_round"),
            "valid_count_by_round": metadata.get("valid_count_by_round"),
            "filter_count_by_round": metadata.get("filter_count_by_round"),
            "n_generated_candidates": metadata.get("n_generated_candidates"),
            "n_after_filtering": metadata.get("n_after_filtering"),
            "n_final_injected": metadata.get("n_final_injected"),
            "no_target_count": metadata.get("no_target_count"),
            "multi_target_count": metadata.get("multi_target_count"),
            "invalid_item_count": metadata.get("invalid_item_count"),
            "filtered_short_session_count": metadata.get("filtered_short_session_count"),
            "target_containing_candidate_count_before_single_target_filter": metadata.get(
                "target_containing_candidate_count_before_single_target_filter"
            ),
            "target_containing_candidate_ratio_before_single_target_filter": metadata.get(
                "target_containing_candidate_ratio_before_single_target_filter"
            ),
            "target_acceptance_failure_reason": metadata.get(
                "target_acceptance_failure_reason"
            ),
            "target_position_distribution": metadata.get("target_position_distribution"),
            "generation_round_durations_sec": metadata.get(
                "generation_round_durations_sec"
            ),
            "raw_candidates_generated_total": metadata.get(
                "raw_candidates_generated_total"
            ),
            "valid_sessions_generated_total": metadata.get(
                "valid_sessions_generated_total"
            ),
            "candidates_per_second": metadata.get("candidates_per_second"),
            "valid_sessions_per_second": metadata.get("valid_sessions_per_second"),
            "candidate_save_policy": metadata.get("candidate_save_policy"),
            "saved_generated_candidate_count": metadata.get(
                "saved_generated_candidate_count"
            ),
        },
        target_root / "generation_log.json",
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_seq_len_violations(
    sessions: Sequence[Sequence[int]],
    *,
    max_seq_len: int,
) -> tuple[int, int]:
    lengths = [int(len(session)) for session in sessions]
    if not lengths:
        return 0, 0
    violation_count = int(sum(length > int(max_seq_len) for length in lengths))
    return int(max(lengths)), violation_count


def _valid_item_ids(canonical_dataset) -> set[int]:
    item_map = getattr(canonical_dataset, "item_map", None)
    if isinstance(item_map, dict) and item_map:
        values = {int(value) for value in item_map.values() if int(value) > 0}
        if values:
            return values
    valid: set[int] = set()
    for split_name in ("train_sub", "valid", "test"):
        for session in getattr(canonical_dataset, split_name):
            valid.update(int(item) for item in session if int(item) > 0)
    return valid


__all__ = [
    "PoisoningSSLSBRTargetResult",
    "compute_seqpoison_max_seq_len",
    "generate_poisoning_ssl_sbr_target",
]
