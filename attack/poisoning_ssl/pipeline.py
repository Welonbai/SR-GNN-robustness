from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from attack.common.artifact_io import load_fake_sessions, load_json, save_fake_sessions, save_json
from attack.common.config import (
    Config,
    PoisoningSSLSBRConfig,
    POISONING_SSL_SBR_GENERATION_BACKEND_REAL,
    POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_FIXED,
    POISONING_SSL_SBR_MAX_SEQ_LEN_POLICY_TRAIN_SUB_P99,
)
from attack.common.paths import shared_root, target_dir
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
from attack.poisoning_ssl.trainer import (
    EffectiveSeqPoisonTrainingConfig,
    _checkpoint_identity,
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
    config_path: str | Path | None = None,
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
    _progress(
        target,
        "Start generation diagnostic: "
        f"method={METHOD_NAME} dataset={config.data.dataset_name} "
        f"target={target} config={config_path if config_path is not None else '<none>'} "
        f"n_fake={n_fake} max_seq_len={max_seq_len} "
        f"backend={poisoning_config.generation_backend} "
        f"device={_configured_device_label(poisoning_config)} "
        f"reuse_existing_artifacts={bool(poisoning_config.reuse_existing_artifacts)} "
        f"first_step_target_mask={bool(poisoning_config.first_step_target_mask)} "
        f"target_logit_bias_after_first_step="
        f"{float(poisoning_config.target_logit_bias_after_first_step):.6g} "
        f"output={target_root}",
    )
    valid_item_ids = _valid_item_ids(shared.canonical_dataset)
    bridge_start = time.perf_counter()
    _progress(target, "Stage start: dataset bridge / train data export")
    bridge = export_pseudo_user_sequences(
        shared.canonical_dataset.train_sub,
        target_item=target,
        output_dir=generation_dir / "dataset_bridge",
        valid_item_ids=valid_item_ids,
        max_seq_len=max_seq_len,
        max_train_sequences=poisoning_config.max_train_sequences,
    )
    _progress(
        target,
        "Stage done: dataset bridge / train data export "
        f"duration={_fmt_duration(time.perf_counter() - bridge_start)} "
        f"train_used={bridge.metadata.get('train_sequence_count_used_for_training')}",
    )
    generation_seed = derive_seed(
        int(config.seeds.fake_session_seed) + int(poisoning_config.generation_seed_offset),
        "poisoning_ssl_sbr",
        target,
    )
    postprocess_seed = derive_seed(generation_seed, "postprocess")
    expected_training_identity = _checkpoint_identity(
        dataset_bundle=bridge,
        target_item=target,
        seed=int(generation_seed),
        effective=EffectiveSeqPoisonTrainingConfig.from_config(poisoning_config),
    )
    expected_training_identity_hash = _hash_json(expected_training_identity)
    expected_generation_identity = _generation_identity(
        config=config,
        target_item=target,
        n_fake_requested=n_fake,
        max_seq_len=max_seq_len,
        generation_seed=int(generation_seed),
        poisoning_config=poisoning_config,
        training_checkpoint_identity_hash=expected_training_identity_hash,
    )
    expected_generation_identity_hash = _hash_json(expected_generation_identity)
    shared_cache_root = _shared_fake_session_cache_root(
        config=config,
        target_item=target,
        generation_identity_hash=expected_generation_identity_hash,
    )
    shared_cache_probe = _load_fake_session_cache(
        cache_root=shared_cache_root,
        expected_generation_identity=expected_generation_identity,
        n_fake_requested=n_fake,
        max_seq_len=max_seq_len,
        reuse_existing_artifacts=bool(poisoning_config.reuse_existing_artifacts),
        scope="shared",
        strict_identity_collision=True,
    )
    local_cache_probe = {"hit": False, "reason": ""}
    if shared_cache_probe.get("collision"):
        raise RuntimeError(str(shared_cache_probe["reason"]))
    if shared_cache_probe["hit"]:
        cache_probe = shared_cache_probe
    else:
        local_cache_probe = _load_fake_session_cache(
            cache_root=target_root,
            expected_generation_identity=expected_generation_identity,
            n_fake_requested=n_fake,
            max_seq_len=max_seq_len,
            reuse_existing_artifacts=bool(poisoning_config.reuse_existing_artifacts),
            scope="local",
            strict_identity_collision=False,
        )
        cache_probe = local_cache_probe if local_cache_probe["hit"] else shared_cache_probe
        if not local_cache_probe["hit"] and local_cache_probe["reason"]:
            cache_probe = local_cache_probe
    if cache_probe["hit"]:
        cache_root = Path(str(cache_probe["cache_root"]))
        cache_scope = str(cache_probe["scope"])
        cache_path = cache_root / "raw_fake_sessions.pkl"
        cached_sessions = cache_probe["sessions"]
        assert isinstance(cached_sessions, list)
        metadata = dict(cache_probe["metadata"])
        metadata.update(
            _cache_metadata_updates(
                target_root=target_root,
                shared_cache_root=shared_cache_root,
                cache_root=cache_root,
                cache_scope=cache_scope,
                n_fake=n_fake,
                cached_session_count=len(cached_sessions),
                generation_identity=expected_generation_identity,
                generation_identity_hash=expected_generation_identity_hash,
                total_start_time=total_start_time,
                total_duration_sec=float(time.perf_counter() - total_start_perf),
            )
        )
        _progress(
            target,
            f"{cache_scope} fake-session cache hit: {cache_path} "
            f"identity={metadata.get('generation_identity_hash')}",
        )
        max_observed_length, max_seq_len_violation_count = _max_seq_len_violations(
            cached_sessions,
            max_seq_len=int(max_seq_len),
        )
        metadata["max_observed_final_length"] = int(max_observed_length)
        metadata["max_seq_len_violation_count"] = int(max_seq_len_violation_count)
        if len(cached_sessions) != n_fake:
            raise RuntimeError(
                "SeqPoison-SBR fake-session cache count mismatch after load: "
                f"requested={n_fake}, cached={len(cached_sessions)}."
            )
        if max_seq_len_violation_count > 0:
            raise RuntimeError(
                "SeqPoison-SBR cached final sessions are longer than max_seq_len; "
                f"max_seq_len={int(max_seq_len)}, "
                f"max_observed_length={int(max_observed_length)}, "
                f"violation_count={int(max_seq_len_violation_count)}."
            )
        _write_artifacts(
            target_root=target_root,
            saved_candidates=[],
            final_sessions=cached_sessions,
            metadata=stringify_mapping_keys(metadata),
            candidate_save_policy=str(poisoning_config.candidate_save_policy),
            config=config,
            manifest_target_root=target_root,
        )
        if cache_scope == "local":
            _write_shared_fake_session_cache(
                config=config,
                shared_cache_root=shared_cache_root,
                final_sessions=cached_sessions,
                metadata=stringify_mapping_keys(metadata),
                candidate_save_policy=str(poisoning_config.candidate_save_policy),
                source_target_root=target_root,
            )
        return PoisoningSSLSBRTargetResult(
            raw_fake_sessions=[list(session) for session in cached_sessions],
            metadata=stringify_mapping_keys(metadata),
        )
    if shared_cache_probe["reason"]:
        fields = shared_cache_probe.get("mismatch_fields", [])
        suffix = (
            f" mismatch fields={fields}"
            if fields
            else f" reason={shared_cache_probe['reason']}"
        )
        _progress(target, f"shared fake-session cache miss:{suffix}")
    if local_cache_probe["reason"]:
        fields = local_cache_probe.get("mismatch_fields", [])
        suffix = (
            f" mismatch fields={fields}"
            if fields
            else f" reason={local_cache_probe['reason']}"
        )
        _progress(target, f"local fake-session cache miss:{suffix}")
    cache_mismatch_fields = list(
        cache_probe.get("mismatch_fields", [])
        or shared_cache_probe.get("mismatch_fields", [])
    )
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
    _progress(
        target,
        "Stage start: candidate generation "
        f"rounds={int(poisoning_config.max_generation_rounds)} "
        f"candidate_multiplier={int(poisoning_config.candidate_multiplier)}",
    )
    for round_index in range(int(poisoning_config.max_generation_rounds)):
        round_start = time.perf_counter()
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
        postprocess_start = time.perf_counter()
        _progress(target, f"Stage start: postprocess round {round_index + 1}")
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
        postprocess_duration = time.perf_counter() - postprocess_start
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
        round_duration = time.perf_counter() - round_start
        round_valid = int(round_postprocess.counts.get("n_after_filtering", 0))
        round_target_candidates = int(
            round_postprocess.counts.get(
                "target_containing_candidate_count_before_single_target_filter",
                0,
            )
        )
        acceptance_so_far = (
            0.0
            if int(cumulative_counts["n_generated_candidates"]) <= 0
            else float(
                int(cumulative_counts.get("n_after_filtering", 0))
                / int(cumulative_counts["n_generated_candidates"])
            )
        )
        _progress(
            target,
            "Stage done: postprocess round "
            f"{round_index + 1} duration={_fmt_duration(postprocess_duration)}",
        )
        _progress(
            target,
            "generation round "
            f"{round_index + 1}/{int(poisoning_config.max_generation_rounds)} "
            f"raw={len(round_candidates)} valid={round_valid} "
            f"total_valid={int(cumulative_counts.get('n_after_filtering', 0))} "
            f"target_candidates={round_target_candidates} "
            f"acceptance={acceptance_so_far:.6g} "
            f"duration={_fmt_duration(round_duration)}",
        )
        save_json(
            {
                "current_stage": "generation",
                "current_round": int(round_index + 1),
                "max_generation_rounds": int(poisoning_config.max_generation_rounds),
                "elapsed_sec": float(time.perf_counter() - generation_start_perf),
                "raw_candidate_count_by_round": list(raw_candidate_count_by_round),
                "valid_count_by_round": list(valid_count_by_round),
                "filter_count_by_round": list(filter_count_by_round),
                "n_generated_candidates": int(cumulative_counts["n_generated_candidates"]),
                "n_after_filtering": int(cumulative_counts.get("n_after_filtering", 0)),
                "target_containing_candidate_count_before_single_target_filter": int(
                    cumulative_counts[
                        "target_containing_candidate_count_before_single_target_filter"
                    ]
                ),
                "acceptance_rate_so_far": acceptance_so_far,
                "first_step_target_mask": bool(poisoning_config.first_step_target_mask),
                "target_logit_bias_after_first_step": float(
                    poisoning_config.target_logit_bias_after_first_step
                ),
            },
            target_root / "generation_progress.json",
        )
        if len(final_sessions) == n_fake:
            break

    generation_end_time = _now_iso()
    generation_duration_sec = float(time.perf_counter() - generation_start_perf)
    _progress(
        target,
        "Stage done: candidate generation "
        f"duration={_fmt_duration(generation_duration_sec)}",
    )
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
        expected_training_checkpoint_identity=expected_training_identity,
        expected_training_checkpoint_identity_hash=expected_training_identity_hash,
    )
    metadata["fake_session_cache_enabled"] = bool(poisoning_config.reuse_existing_artifacts)
    metadata["fake_session_cache_hit"] = False
    metadata["fake_session_cache_scope"] = "miss"
    metadata["fake_session_cache_path"] = None
    metadata["shared_fake_session_cache_path"] = str(shared_cache_root)
    metadata["local_target_root"] = str(target_root)
    metadata["fake_session_cache_mismatch_fields"] = cache_mismatch_fields
    max_observed_length, max_seq_len_violation_count = _max_seq_len_violations(
        final_sessions,
        max_seq_len=int(max_seq_len),
    )
    metadata["max_observed_final_length"] = int(max_observed_length)
    metadata["max_seq_len_violation_count"] = int(max_seq_len_violation_count)
    artifact_start = time.perf_counter()
    _progress(
        target,
        "Generation identity: "
        f"hash={metadata.get('generation_identity_hash')} "
        f"training_checkpoint_reused={metadata.get('training_checkpoint_reused')} "
        f"checkpoint={metadata.get('training_checkpoint_path')} "
        f"first_step_target_mask={metadata.get('first_step_target_mask')} "
        f"target_logit_bias_after_first_step="
        f"{float(metadata.get('target_logit_bias_after_first_step', 0.0)):.6g} "
        f"output={target_root}",
    )
    _progress(target, "Stage start: artifact writing")
    _write_artifacts(
        target_root=target_root,
        saved_candidates=saved_candidates,
        final_sessions=final_sessions,
        metadata=metadata,
        candidate_save_policy=candidate_save_policy,
        config=config,
        manifest_target_root=target_root,
    )
    metadata_path = target_root / "poisoning_ssl_sbr_metadata.json"
    _progress(
        target,
        "Stage done: artifact writing "
        f"duration={_fmt_duration(time.perf_counter() - artifact_start)}",
    )
    if len(final_sessions) != n_fake:
        _progress(
            target,
            "Failed: insufficient valid fake sessions "
            f"generated={int(cumulative_counts['n_generated_candidates'])} "
            f"valid={int(cumulative_counts.get('n_after_filtering', 0))} "
            f"injected={len(final_sessions)} metadata={metadata_path}",
        )
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
    _write_shared_fake_session_cache(
        config=config,
        shared_cache_root=shared_cache_root,
        final_sessions=final_sessions,
        metadata=metadata,
        candidate_save_policy=candidate_save_policy,
        source_target_root=target_root,
    )
    _progress(
        target,
        "Done: "
        f"generated={metadata.get('n_generated_candidates')} "
        f"valid={metadata.get('n_after_filtering')} "
        f"injected={metadata.get('n_final_injected')} "
        f"acceptance={float(metadata.get('acceptance_rate', 0.0)):.6g} "
        f"target_label_pairs={metadata.get('target_label_pair_count_added')} "
        f"target_pos0_ratio={float(metadata.get('target_pos0_ratio', 0.0)):.6g} "
        f"target_logit_bias_after_first_step="
        f"{float(metadata.get('target_logit_bias_after_first_step', 0.0)):.6g} "
        f"generation_identity_hash={metadata.get('generation_identity_hash')} "
        f"training_duration={_fmt_duration(metadata.get('training_duration_sec'))} "
        f"generation_duration={_fmt_duration(metadata.get('generation_duration_sec'))} "
        f"total_duration={_fmt_duration(metadata.get('total_duration_sec'))} "
        f"metadata={metadata_path}",
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
    expected_training_checkpoint_identity: dict[str, object],
    expected_training_checkpoint_identity_hash: str,
) -> dict[str, object]:
    target_stats = target_diagnostics(final_sessions, target_item=int(target_item))
    duplicate_stats = duplicate_diagnostics(final_sessions)
    repeated_stats = _repeated_item_stats(final_sessions, target_item=int(target_item))
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
    training_epochs = latest_generation_metadata.get("training_epochs", {})
    if not isinstance(training_epochs, dict):
        training_epochs = {}
    first_generation_metadata = (
        generation_round_metadata[0] if generation_round_metadata else {}
    )
    training_checkpoint_identity = latest_generation_metadata.get(
        "training_checkpoint_identity"
    ) or expected_training_checkpoint_identity
    training_checkpoint_identity_hash = latest_generation_metadata.get(
        "training_checkpoint_identity_hash"
    ) or expected_training_checkpoint_identity_hash
    generation_identity = _generation_identity(
        config=config,
        target_item=int(target_item),
        n_fake_requested=int(n_fake_requested),
        max_seq_len=int(max_seq_len),
        generation_seed=int(generation_seed),
        poisoning_config=poisoning_config,
        training_checkpoint_identity_hash=training_checkpoint_identity_hash,
    )
    target_label_candidate_rate = (
        0.0
        if n_generated <= 0
        else float(budget_stats["target_label_pair_count_added"] / n_generated)
    )
    estimated_candidates_needed = float(
        int(n_fake_requested) / max(target_label_candidate_rate, 1.0e-12)
    )
    first_step_target_mask = (
        False if poisoning_config is None else bool(poisoning_config.first_step_target_mask)
    )
    target_logit_bias_after_first_step = (
        0.0
        if poisoning_config is None
        else float(poisoning_config.target_logit_bias_after_first_step)
    )
    first_step_seqpoison_target = int(
        bridge_metadata.get("seqpoison_target_item", target_item)
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
        "first_step_target_mask": first_step_target_mask,
        "first_step_target_mask_applied": bool(
            first_step_target_mask
            and latest_generation_metadata.get(
                "first_step_target_mask_applied",
                False,
            )
        ),
        "first_step_target_mask_target_id_canonical": int(target_item),
        "first_step_target_mask_target_id_seqpoison": first_step_seqpoison_target,
        "unexpected_pos0_after_mask_count": (
            int(target_stats["target_pos0_count"]) if first_step_target_mask else 0
        ),
        "unexpected_pos0_after_mask_ratio": (
            float(target_stats["target_pos0_ratio"]) if first_step_target_mask else 0.0
        ),
        "target_logit_bias_after_first_step": target_logit_bias_after_first_step,
        "target_logit_bias_after_first_step_applied": (
            target_logit_bias_after_first_step != 0.0
        ),
        "target_logit_bias_target_id_canonical": int(target_item),
        "target_logit_bias_target_id_seqpoison": first_step_seqpoison_target,
        "target_logit_bias_positions": (
            "positions>=1"
            if target_logit_bias_after_first_step != 0.0
            else "none"
        ),
        "target_label_candidate_rate": target_label_candidate_rate,
        "estimated_candidates_needed_for_target_label_budget": estimated_candidates_needed,
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
        "training_checkpoint_reused": bool(
            first_generation_metadata.get("training_checkpoint_reused", False)
        ),
        "training_checkpoint_path": latest_generation_metadata.get(
            "training_checkpoint_path"
        )
        or latest_generation_metadata.get("checkpoint_dir"),
        "training_checkpoint_identity": training_checkpoint_identity,
        "training_checkpoint_identity_hash": training_checkpoint_identity_hash,
        "generation_identity": generation_identity,
        "generation_identity_hash": _hash_json(generation_identity),
        "fake_session_cache_enabled": (
            False
            if poisoning_config is None
            else bool(poisoning_config.reuse_existing_artifacts)
        ),
        "fake_session_cache_hit": False,
        "fake_session_cache_path": None,
        "fake_session_cache_mismatch_fields": [],
        "training_epochs": training_epochs,
        "classifier_epochs": training_epochs.get("classifier_epochs"),
        "mle_epochs": training_epochs.get("mle_epochs"),
        "adversarial_epochs": training_epochs.get("adversarial_epochs"),
        "discriminator_pretrain_steps": training_epochs.get(
            "discriminator_pretrain_steps"
        ),
        "discriminator_pretrain_epochs": training_epochs.get(
            "discriminator_pretrain_epochs"
        ),
        "discriminator_adversarial_steps": training_epochs.get(
            "discriminator_adversarial_steps"
        ),
        "discriminator_adversarial_epochs": training_epochs.get(
            "discriminator_adversarial_epochs"
        ),
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
        **repeated_stats,
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


def _generation_identity(
    *,
    config: Config,
    target_item: int,
    n_fake_requested: int,
    max_seq_len: int,
    generation_seed: int,
    poisoning_config: PoisoningSSLSBRConfig | None,
    training_checkpoint_identity_hash: object,
) -> dict[str, object]:
    return {
        "dataset": str(config.data.dataset_name),
        "split_identity": _split_identity_payload(config),
        "target_item": int(target_item),
        "attack_size": float(config.attack.size),
        "n_fake_requested": int(n_fake_requested),
        "max_seq_len": int(max_seq_len),
        "generation_seed": int(generation_seed),
        "training_checkpoint_identity_hash": training_checkpoint_identity_hash,
        "enforce_single_target": (
            True
            if poisoning_config is None
            else bool(poisoning_config.enforce_single_target)
        ),
        "first_step_target_mask": (
            False
            if poisoning_config is None
            else bool(poisoning_config.first_step_target_mask)
        ),
        "target_logit_bias_after_first_step": (
            0.0
            if poisoning_config is None
            else float(poisoning_config.target_logit_bias_after_first_step)
        ),
        "candidate_multiplier": (
            None if poisoning_config is None else int(poisoning_config.candidate_multiplier)
        ),
        "max_generation_rounds": (
            None
            if poisoning_config is None
            else int(poisoning_config.max_generation_rounds)
        ),
        "candidate_save_policy": (
            None if poisoning_config is None else str(poisoning_config.candidate_save_policy)
        ),
        "max_saved_candidates": (
            None if poisoning_config is None else int(poisoning_config.max_saved_candidates)
        ),
    }


def _hash_json(payload: dict[str, object]) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:12]


def _split_identity_payload(config: Config) -> dict[str, object]:
    canonical = getattr(config.data, "canonical_split", None)
    canonical_payload = (
        dict(canonical.__dict__) if hasattr(canonical, "__dict__") else canonical
    )
    return {
        "dataset_name": str(config.data.dataset_name),
        "split_protocol": str(config.data.split_protocol),
        "poison_train_only": bool(config.data.poison_train_only),
        "canonical_split": canonical_payload,
    }


def _shared_fake_session_cache_root(
    *,
    config: Config,
    target_item: int,
    generation_identity_hash: str,
) -> Path:
    return (
        shared_root(config)
        / "poisoning_ssl_sbr_fake_sessions"
        / str(int(target_item))
        / str(generation_identity_hash)
    )


def _cache_metadata_updates(
    *,
    target_root: Path,
    shared_cache_root: Path,
    cache_root: Path,
    cache_scope: str,
    n_fake: int,
    cached_session_count: int,
    generation_identity: dict[str, object],
    generation_identity_hash: str,
    total_start_time: str,
    total_duration_sec: float,
) -> dict[str, object]:
    return {
        "fake_session_cache_enabled": True,
        "fake_session_cache_hit": True,
        "fake_session_cache_scope": cache_scope,
        "fake_session_cache_path": str(cache_root),
        "shared_fake_session_cache_path": str(shared_cache_root),
        "local_target_root": str(target_root),
        "fake_session_cache_mismatch_fields": [],
        "n_fake_requested": int(n_fake),
        "n_final_injected": int(cached_session_count),
        "generation_identity": generation_identity,
        "generation_identity_hash": generation_identity_hash,
        "total_start_time": total_start_time,
        "total_end_time": _now_iso(),
        "total_duration_sec": float(total_duration_sec),
    }


def _load_fake_session_cache(
    *,
    cache_root: Path,
    expected_generation_identity: dict[str, object],
    n_fake_requested: int,
    max_seq_len: int,
    reuse_existing_artifacts: bool,
    scope: str,
    strict_identity_collision: bool,
) -> dict[str, object]:
    raw_path = cache_root / "raw_fake_sessions.pkl"
    metadata_path = cache_root / "poisoning_ssl_sbr_metadata.json"
    if not reuse_existing_artifacts:
        return {
            "hit": False,
            "reason": "reuse_existing_artifacts=false",
            "scope": scope,
            "cache_root": str(cache_root),
        }
    if not raw_path.exists() and not metadata_path.exists():
        return {"hit": False, "reason": "", "scope": scope, "cache_root": str(cache_root)}
    if not raw_path.exists():
        return {
            "hit": False,
            "reason": "raw_fake_sessions.pkl missing",
            "scope": scope,
            "cache_root": str(cache_root),
        }
    if not metadata_path.exists():
        return {
            "hit": False,
            "reason": "poisoning_ssl_sbr_metadata.json missing",
            "scope": scope,
            "cache_root": str(cache_root),
        }
    metadata = load_json(metadata_path)
    if not isinstance(metadata, dict):
        return {
            "hit": False,
            "reason": "metadata is not a JSON object",
            "scope": scope,
            "cache_root": str(cache_root),
        }
    observed_identity = metadata.get("generation_identity")
    if not isinstance(observed_identity, dict):
        return {
            "hit": False,
            "reason": "generation_identity missing",
            "mismatch_fields": ["generation_identity"],
            "scope": scope,
            "cache_root": str(cache_root),
        }
    mismatch_fields = _identity_mismatch_fields(
        observed_identity,
        expected_generation_identity,
    )
    if mismatch_fields:
        if strict_identity_collision:
            return {
                "hit": False,
                "collision": True,
                "reason": (
                    "SeqPoison-SBR shared fake-session cache identity mismatch "
                    f"at hash path {cache_root}; mismatch_fields={mismatch_fields}."
                ),
                "mismatch_fields": mismatch_fields,
                "scope": scope,
                "cache_root": str(cache_root),
            }
        return {
            "hit": False,
            "reason": "generation_identity mismatch",
            "mismatch_fields": mismatch_fields,
            "scope": scope,
            "cache_root": str(cache_root),
        }
    sessions = load_fake_sessions(raw_path)
    if sessions is None:
        return {
            "hit": False,
            "reason": "raw_fake_sessions.pkl unreadable",
            "scope": scope,
            "cache_root": str(cache_root),
        }
    sessions = [[int(item) for item in session] for session in sessions]
    if len(sessions) != int(n_fake_requested):
        return {
            "hit": False,
            "reason": "n_final_injected mismatch",
            "mismatch_fields": ["n_fake_requested"],
            "scope": scope,
            "cache_root": str(cache_root),
        }
    _max_observed, violation_count = _max_seq_len_violations(
        sessions,
        max_seq_len=int(max_seq_len),
    )
    if violation_count > 0:
        return {
            "hit": False,
            "reason": "max_seq_len violation",
            "mismatch_fields": ["max_seq_len"],
            "scope": scope,
            "cache_root": str(cache_root),
        }
    return {
        "hit": True,
        "reason": "",
        "sessions": sessions,
        "metadata": metadata,
        "mismatch_fields": [],
        "scope": scope,
        "cache_root": str(cache_root),
    }


def _identity_mismatch_fields(
    observed: dict[str, object],
    expected: dict[str, object],
) -> list[str]:
    fields: list[str] = []
    for key in sorted(set(observed) | set(expected)):
        if observed.get(key) != expected.get(key):
            fields.append(str(key))
    return fields


def _repeated_item_stats(
    sessions: Sequence[Sequence[int]],
    *,
    target_item: int,
) -> dict[str, object]:
    count = len(sessions)
    repeated_items = sum(1 for session in sessions if len(set(session)) < len(session))
    repeated_target = sum(
        1 for session in sessions if list(session).count(int(target_item)) > 1
    )
    return {
        "sessions_with_repeated_items": int(repeated_items),
        "sessions_with_repeated_items_ratio": (
            0.0 if count <= 0 else float(repeated_items / count)
        ),
        "sessions_with_repeated_target": int(repeated_target),
        "sessions_with_repeated_target_ratio": (
            0.0 if count <= 0 else float(repeated_target / count)
        ),
    }


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
    config: Config,
    manifest_target_root: Path,
) -> None:
    if candidate_save_policy in {"sample", "all"} and saved_candidates:
        save_fake_sessions(saved_candidates, target_root / "generated_candidates.pkl")
    save_fake_sessions(final_sessions, target_root / "raw_fake_sessions.pkl")
    save_json(metadata, target_root / "poisoning_ssl_sbr_metadata.json")
    save_json(
        _fake_session_sanity_summary(metadata),
        target_root / "fake_session_sanity_summary.json",
    )
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
            "first_step_target_mask": metadata.get("first_step_target_mask"),
            "first_step_target_mask_applied": metadata.get(
                "first_step_target_mask_applied"
            ),
            "first_step_target_mask_target_id_canonical": metadata.get(
                "first_step_target_mask_target_id_canonical"
            ),
            "first_step_target_mask_target_id_seqpoison": metadata.get(
                "first_step_target_mask_target_id_seqpoison"
            ),
            "unexpected_pos0_after_mask_count": metadata.get(
                "unexpected_pos0_after_mask_count"
            ),
            "unexpected_pos0_after_mask_ratio": metadata.get(
                "unexpected_pos0_after_mask_ratio"
            ),
            "target_logit_bias_after_first_step": metadata.get(
                "target_logit_bias_after_first_step"
            ),
            "target_logit_bias_after_first_step_applied": metadata.get(
                "target_logit_bias_after_first_step_applied"
            ),
            "target_logit_bias_target_id_canonical": metadata.get(
                "target_logit_bias_target_id_canonical"
            ),
            "target_logit_bias_target_id_seqpoison": metadata.get(
                "target_logit_bias_target_id_seqpoison"
            ),
            "target_logit_bias_positions": metadata.get("target_logit_bias_positions"),
            "target_label_candidate_rate": metadata.get("target_label_candidate_rate"),
            "estimated_candidates_needed_for_target_label_budget": metadata.get(
                "estimated_candidates_needed_for_target_label_budget"
            ),
            "training_checkpoint_reused": metadata.get("training_checkpoint_reused"),
            "training_checkpoint_path": metadata.get("training_checkpoint_path"),
            "training_checkpoint_identity_hash": metadata.get(
                "training_checkpoint_identity_hash"
            ),
            "generation_identity_hash": metadata.get("generation_identity_hash"),
            "fake_session_cache_enabled": metadata.get("fake_session_cache_enabled"),
            "fake_session_cache_hit": metadata.get("fake_session_cache_hit"),
            "fake_session_cache_scope": metadata.get("fake_session_cache_scope"),
            "fake_session_cache_path": metadata.get("fake_session_cache_path"),
            "shared_fake_session_cache_path": metadata.get(
                "shared_fake_session_cache_path"
            ),
            "local_target_root": metadata.get("local_target_root"),
            "fake_session_cache_mismatch_fields": metadata.get(
                "fake_session_cache_mismatch_fields"
            ),
        },
        target_root / "generation_log.json",
    )
    save_json(
        _target_manifest(
            config=config,
            target_root=target_root,
            metadata=metadata,
            manifest_target_root=manifest_target_root,
        ),
        target_root / "seqpoison_sbr_manifest.json",
    )


def _write_shared_fake_session_cache(
    *,
    config: Config,
    shared_cache_root: Path,
    final_sessions: list[list[int]],
    metadata: dict[str, object],
    candidate_save_policy: str,
    source_target_root: Path,
) -> None:
    existing_metadata = load_json(shared_cache_root / "poisoning_ssl_sbr_metadata.json")
    if isinstance(existing_metadata, dict):
        existing_identity = existing_metadata.get("generation_identity")
        current_identity = metadata.get("generation_identity")
        if isinstance(existing_identity, dict) and existing_identity != current_identity:
            mismatch_fields = _identity_mismatch_fields(
                existing_identity,
                current_identity if isinstance(current_identity, dict) else {},
            )
            raise RuntimeError(
                "SeqPoison-SBR shared fake-session cache hash collision or "
                "corrupt cache: "
                f"path={shared_cache_root}, mismatch_fields={mismatch_fields}."
            )
    shared_metadata = dict(metadata)
    shared_metadata["shared_fake_session_cache_path"] = str(shared_cache_root)
    shared_metadata["local_target_root"] = str(source_target_root)
    _write_artifacts(
        target_root=shared_cache_root,
        saved_candidates=[],
        final_sessions=final_sessions,
        metadata=stringify_mapping_keys(shared_metadata),
        candidate_save_policy=candidate_save_policy,
        config=config,
        manifest_target_root=source_target_root,
    )


def _fake_session_sanity_summary(metadata: dict[str, object]) -> dict[str, object]:
    fake_stats = _stats_object(metadata.get("final_fake_length_stats"))
    train_stats = _stats_object(metadata.get("clean_train_length_stats"))
    return {
        "n_fake_requested": metadata.get("n_fake_requested"),
        "n_final_injected": metadata.get("n_final_injected"),
        "n_generated_candidates": metadata.get("n_generated_candidates"),
        "acceptance_rate": metadata.get("acceptance_rate"),
        "target_position_distribution": metadata.get("target_position_distribution"),
        "target_pos0_ratio": metadata.get("target_pos0_ratio"),
        "target_label_pair_count_added": metadata.get("target_label_pair_count_added"),
        "expanded_pair_count_added": metadata.get("expanded_pair_count_added"),
        "no_target_count": metadata.get("no_target_count"),
        "multi_target_count": metadata.get("multi_target_count"),
        "filtered_short_session_count": metadata.get("filtered_short_session_count"),
        "invalid_item_count": metadata.get("invalid_item_count"),
        "max_seq_len": metadata.get("max_seq_len_value"),
        "max_seq_len_violation_count": metadata.get("max_seq_len_violation_count"),
        "fake_length_mean": fake_stats.get("mean"),
        "fake_length_p50": fake_stats.get("p50"),
        "fake_length_p75": fake_stats.get("p75"),
        "fake_length_p90": fake_stats.get("p90"),
        "fake_length_p95": fake_stats.get("p95"),
        "fake_length_p99": fake_stats.get("p99"),
        "fake_length_max": fake_stats.get("max"),
        "train_sub_length_mean": train_stats.get("mean"),
        "train_sub_length_p50": train_stats.get("p50"),
        "train_sub_length_p75": train_stats.get("p75"),
        "train_sub_length_p90": train_stats.get("p90"),
        "train_sub_length_p95": train_stats.get("p95"),
        "train_sub_length_p99": train_stats.get("p99"),
        "train_sub_length_max": train_stats.get("max"),
        "unique_fake_session_count": metadata.get("unique_fake_session_count"),
        "unique_fake_session_ratio": metadata.get("unique_fake_session_ratio"),
        "duplicate_fake_session_count": metadata.get("duplicate_session_count"),
        "sessions_with_repeated_items": metadata.get("sessions_with_repeated_items"),
        "sessions_with_repeated_items_ratio": metadata.get(
            "sessions_with_repeated_items_ratio"
        ),
        "sessions_with_repeated_target": metadata.get("sessions_with_repeated_target"),
        "sessions_with_repeated_target_ratio": metadata.get(
            "sessions_with_repeated_target_ratio"
        ),
    }


def _target_manifest(
    *,
    config: Config,
    target_root: Path,
    metadata: dict[str, object],
    manifest_target_root: Path,
) -> dict[str, object]:
    poisoning_config = config.attack.poisoning_ssl_sbr
    return {
        "dataset": str(config.data.dataset_name),
        "config_name": str(config.experiment.name),
        "run_name": str(config.experiment.name),
        "target_item": metadata.get("target_item"),
        "target_group": getattr(config.targets, "bucket", None),
        "attack_size": float(config.attack.size),
        "n_fake_requested": metadata.get("n_fake_requested"),
        "n_final_injected": metadata.get("n_final_injected"),
        "seed": metadata.get("generation_seed"),
        "max_seq_len": metadata.get("max_seq_len_value"),
        "training_checkpoint_identity_hash": metadata.get(
            "training_checkpoint_identity_hash"
        ),
        "training_checkpoint_path": metadata.get("training_checkpoint_path"),
        "classifier_checkpoint_path": metadata.get("classifier_checkpoint_path"),
        "generator_checkpoint_path": metadata.get("generator_checkpoint_path"),
        "discriminator_checkpoint_path": metadata.get("discriminator_checkpoint_path"),
        "generation_identity_hash": metadata.get("generation_identity_hash"),
        "raw_fake_sessions_path": str(target_root / "raw_fake_sessions.pkl"),
        "metadata_path": str(target_root / "poisoning_ssl_sbr_metadata.json"),
        "generation_log_path": str(target_root / "generation_log.json"),
        "length_distribution_path": str(target_root / "length_distribution.json"),
        "fake_session_sanity_summary_path": str(
            target_root / "fake_session_sanity_summary.json"
        ),
        "first_step_target_mask": metadata.get("first_step_target_mask"),
        "target_logit_bias_after_first_step": metadata.get(
            "target_logit_bias_after_first_step"
        ),
        "adversarial_epochs": metadata.get("adversarial_epochs"),
        "candidate_multiplier": metadata.get("candidate_multiplier"),
        "max_generation_rounds": metadata.get("max_generation_rounds"),
        "candidate_save_policy": metadata.get("candidate_save_policy"),
        "shared_fake_session_cache_path": metadata.get(
            "shared_fake_session_cache_path"
        ),
        "local_target_root": str(manifest_target_root),
        "fake_session_cache_hit": metadata.get("fake_session_cache_hit"),
        "fake_session_cache_scope": metadata.get("fake_session_cache_scope"),
        "fake_session_cache_path": metadata.get("fake_session_cache_path"),
        "created_at": _now_iso(),
        "cache_hit": metadata.get("fake_session_cache_hit"),
    }


def _stats_object(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _progress(target_item: int, message: str) -> None:
    print(f"[SeqPoison-SBR][target={int(target_item)}] {message}", flush=True)


def _fmt_duration(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "n/a"


def _configured_device_label(config: PoisoningSSLSBRConfig) -> str:
    if config.device:
        return str(config.device)
    if config.gpu_id is not None:
        return f"cuda:{config.gpu_id}"
    return "auto"


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
