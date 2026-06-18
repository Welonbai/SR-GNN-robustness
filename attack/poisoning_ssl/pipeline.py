from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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

    all_candidates: list[list[int]] = []
    postprocess_result = None
    generation_round_metadata: list[dict[str, object]] = []
    raw_candidate_count_by_round: list[int] = []
    valid_count_by_round: list[int] = []
    filter_count_by_round: list[int] = []
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
        all_candidates.extend([[int(item) for item in session] for session in round_candidates])
        postprocess_result = postprocess_fake_user_sequences(
            all_candidates,
            target_item=target,
            valid_item_ids=valid_item_ids,
            n_fake=n_fake,
            enforce_single_target=bool(poisoning_config.enforce_single_target),
            filter_no_target=bool(poisoning_config.filter_no_target),
            filter_short_sessions=bool(poisoning_config.filter_short_sessions),
            remove_user_id=True,
        )
        valid_count_by_round.append(int(postprocess_result.counts.get("n_after_filtering", 0)))
        filter_count_by_round.append(
            int(len(all_candidates)) - int(postprocess_result.counts.get("n_after_filtering", 0))
        )
        last_metadata = getattr(generator, "last_metadata", None)
        if isinstance(last_metadata, dict) and last_metadata:
            generation_round_metadata.append(dict(last_metadata))
        if len(postprocess_result.final_sessions) == n_fake:
            break

    if postprocess_result is None:
        raise RuntimeError("SeqPoison-SBR generation did not execute any rounds.")

    metadata = _metadata(
        config=config,
        shared=shared,
        target_item=target,
        n_fake_requested=n_fake,
        max_seq_len=max_seq_len,
        generation_seed=generation_seed,
        postprocess_seed=postprocess_seed,
        candidates=all_candidates,
        final_sessions=postprocess_result.final_sessions,
        postprocess_counts=postprocess_result.counts,
        bridge_metadata=bridge.metadata,
        generation_round_metadata=generation_round_metadata,
        raw_candidate_count_by_round=raw_candidate_count_by_round,
        valid_count_by_round=valid_count_by_round,
        filter_count_by_round=filter_count_by_round,
    )
    max_observed_length, max_seq_len_violation_count = _max_seq_len_violations(
        postprocess_result.final_sessions,
        max_seq_len=int(max_seq_len),
    )
    metadata["max_observed_final_length"] = int(max_observed_length)
    metadata["max_seq_len_violation_count"] = int(max_seq_len_violation_count)
    _write_artifacts(
        target_root=target_root,
        candidates=all_candidates,
        final_sessions=postprocess_result.final_sessions,
        metadata=metadata,
        save_generated_candidates=bool(poisoning_config.save_generated_candidates),
    )
    if len(postprocess_result.final_sessions) != n_fake:
        raise RuntimeError(
            "SeqPoison-SBR could not generate enough valid fake sessions: "
            f"requested={n_fake}, final={len(postprocess_result.final_sessions)}, "
            f"candidates={len(all_candidates)}. See poisoning_ssl_sbr_metadata.json."
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
        raw_fake_sessions=[list(session) for session in postprocess_result.final_sessions],
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
    candidates: list[list[int]],
    final_sessions: list[list[int]],
    postprocess_counts: dict[str, int],
    bridge_metadata: dict[str, object],
    generation_round_metadata: list[dict[str, object]],
    raw_candidate_count_by_round: list[int],
    valid_count_by_round: list[int],
    filter_count_by_round: list[int],
) -> dict[str, object]:
    target_stats = target_diagnostics(final_sessions, target_item=int(target_item))
    duplicate_stats = duplicate_diagnostics(final_sessions)
    budget_stats = budget_diagnostics(
        final_sessions,
        target_item=int(target_item),
        clean_label_count=int(len(shared.clean_labels)),
    )
    n_generated = int(len(candidates))
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
        "max_seq_len_policy": (
            None if poisoning_config is None else poisoning_config.max_seq_len_policy
        ),
        "max_seq_len_value": int(max_seq_len),
        "clean_train_length_stats": length_stats(shared.canonical_dataset.train_sub),
        "generated_candidate_length_stats": length_stats(candidates),
        "final_fake_length_stats": length_stats(final_sessions),
        "generation_seed": int(generation_seed),
        "postprocess_seed": int(postprocess_seed),
        "invalid_item_count": int(postprocess_counts.get("invalid_item_count", 0)),
        "filtered_short_session_count": int(
            postprocess_counts.get("filtered_short_session_count", 0)
        ),
        "no_target_count": int(postprocess_counts.get("no_target_count", 0)),
        "multi_target_count": int(postprocess_counts.get("multi_target_count", 0)),
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


def _write_artifacts(
    *,
    target_root: Path,
    candidates: list[list[int]],
    final_sessions: list[list[int]],
    metadata: dict[str, object],
    save_generated_candidates: bool,
) -> None:
    if save_generated_candidates:
        save_fake_sessions(candidates, target_root / "generated_candidates.pkl")
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
