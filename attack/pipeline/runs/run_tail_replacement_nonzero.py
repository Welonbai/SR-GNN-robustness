from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    poison_model_key,
    poison_model_key_payload,
    shared_attack_artifact_key,
    target_dir,
)
from attack.data.poisoned_dataset_builder import build_poisoned_dataset
from attack.insertion.tail_replacement import TailReplacementPolicy, TailReplacementResult
from attack.pipeline.core.orchestrator import (
    RunContext,
    TargetPoisonOutput,
    run_targets_and_victims,
)
from attack.pipeline.core.pipeline_utils import prepare_shared_attack_artifacts
from attack.pipeline.core.position_stats import save_position_stats


DEFAULT_TAIL_REPLACEMENT_CONFIG_PATH = (
    "attack/configs/"
    "diginetica_valbest_attack_tail_replacement_nonzero_when_possible_ratio1_"
    "srgnn_target11103_partial5.yaml"
)
RANDOM_NZ_RUN_TYPE = "random_nonzero_when_possible"


def run_tail_replacement_nonzero(
    config: Config,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    if not config.data.poison_train_only:
        raise ValueError("Tail-Replacement-NZ expects data.poison_train_only to be true.")
    shared = prepare_shared_attack_artifacts(
        config,
        run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        require_poison_runner=False,
        config_path=config_path,
    )

    context = RunContext.from_shared(shared)

    def build_poisoned(target_item: int) -> TargetPoisonOutput:
        policy = TailReplacementPolicy()
        results: list[TailReplacementResult] = [
            policy.apply_with_metadata(session, int(target_item))
            for session in shared.template_sessions
        ]
        fake_sessions = [result.session for result in results]
        replacement_positions = [int(result.replacement_position) for result in results]

        _validate_tail_replaced_sessions(
            template_sessions=shared.template_sessions,
            replaced_sessions=fake_sessions,
            replacement_positions=replacement_positions,
            target_item=int(target_item),
        )

        max_item = max(shared.stats.item_counts)
        if any(max(session) > max_item for session in fake_sessions):
            raise ValueError("Generated fake sessions contain invalid item IDs.")

        poisoned = build_poisoned_dataset(
            shared.clean_sessions,
            shared.clean_labels,
            fake_sessions,
        )

        target_root = target_dir(
            config,
            int(target_item),
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        )
        target_root.mkdir(parents=True, exist_ok=True)

        position_stats_path = save_position_stats(
            target_root / "position_stats.json",
            sessions=shared.template_sessions,
            positions=replacement_positions,
            run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
            target_item=int(target_item),
            note=(
                "Tail-Replacement-NZ deterministically replaces the final nonzero "
                "position; tail_position is an overlapping summary group."
            ),
        )

        metadata = build_tail_replacement_metadata(
            config=config,
            target_item=int(target_item),
            template_sessions=shared.template_sessions,
            replaced_sessions=fake_sessions,
            replacement_results=results,
            clean_train_sessions=shared.canonical_dataset.train_sub,
            template_fake_sessions_path=shared.shared_paths["fake_sessions"],
            poison_model_checkpoint_path=_existing_path_or_none(
                shared.shared_paths.get("poison_model")
            ),
        )
        metadata_path = target_root / "tail_replacement_position_metadata.json"
        save_json(metadata, metadata_path)

        return TargetPoisonOutput(
            poisoned=poisoned,
            metadata={
                "position_stats_path": str(position_stats_path),
                "tail_replacement_position_metadata_path": str(metadata_path),
                "template_fake_sessions_path": str(shared.shared_paths["fake_sessions"]),
                "poison_model_checkpoint_path": _existing_path_or_none(
                    shared.shared_paths.get("poison_model")
                ),
                "shared_fake_sessions_key": shared_attack_artifact_key(
                    config,
                    run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
                ),
            },
        )

    return run_targets_and_victims(
        config,
        config_path=config_path,
        context=context,
        run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        build_poisoned=build_poisoned,
    )


def build_tail_replacement_metadata(
    *,
    config: Config,
    target_item: int,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    replacement_results: Sequence[TailReplacementResult],
    clean_train_sessions: Sequence[Sequence[int]],
    template_fake_sessions_path: str | Path,
    poison_model_checkpoint_path: str | Path | None,
    preview_limit: int = 20,
) -> dict[str, object]:
    if len(template_sessions) != len(replaced_sessions):
        raise ValueError("template_sessions and replaced_sessions must have the same length.")
    if len(template_sessions) != len(replacement_results):
        raise ValueError("replacement_results must align 1:1 with template_sessions.")

    replacement_positions = [
        int(result.replacement_position) for result in replacement_results
    ]
    position_counts: Counter[int] = Counter(replacement_positions)
    position_group_counts = _position_group_counts(
        template_sessions,
        replacement_positions,
    )
    total = int(len(replacement_positions))
    target_counts = [
        sum(1 for item in session if int(item) == int(target_item))
        for session in replaced_sessions
    ]
    length_deltas = [
        int(len(replaced)) - int(len(original))
        for original, replaced in zip(template_sessions, replaced_sessions)
    ]
    shared_key = shared_attack_artifact_key(
        config,
        run_type=TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    )
    return {
        "run_type": TAIL_REPLACEMENT_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        "operation": "replacement",
        "replacement_policy": "tail",
        "target_item": int(target_item),
        "fake_session_count": int(len(replaced_sessions)),
        "replacement_topk_ratio": float(config.attack.replacement_topk_ratio),
        "replacement_topk_ratio_used": False,
        "topk_sampling_note": (
            "Tail-Replacement-NZ is deterministic and does not use top-k sampling; "
            "attack.replacement_topk_ratio is recorded only for comparability."
        ),
        "template_fake_sessions_path": str(template_fake_sessions_path),
        "poison_model_checkpoint_path": (
            None
            if poison_model_checkpoint_path is None
            else str(poison_model_checkpoint_path)
        ),
        "poison_model_key": poison_model_key(config),
        "poison_model_identity": poison_model_key_payload(config),
        "shared_fake_sessions_key": shared_key,
        "random_nz_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_NZ_RUN_TYPE,
        ),
        "random_insertion_shared_fake_sessions_key": shared_attack_artifact_key(
            config,
            run_type=RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
        ),
        "same_shared_fake_sessions_as_random_nz_expected": True,
        "original_length_distribution": _length_distribution(template_sessions),
        "tail_replaced_length_distribution": _length_distribution(replaced_sessions),
        "clean_train_length_distribution": _length_distribution(clean_train_sessions),
        "length_shift_summary": _numeric_summary(length_deltas),
        "replacement_position_counts": _stringify_counts(position_counts),
        "replacement_position_ratios": _stringify_ratios(position_counts, total=total),
        "replacement_position_group_counts": _stringify_named_counts(
            position_group_counts
        ),
        "replacement_position_group_ratios": _stringify_named_ratios(
            position_group_counts,
            total=total,
        ),
        "tail_position_count": int(position_group_counts["tail_position"]),
        "tail_position_ratio": (
            float(position_group_counts["tail_position"]) / float(total)
            if total
            else 0.0
        ),
        "tail_position_is_overlapping_group": True,
        "pre_existing_target_in_template_sessions_count": int(
            sum(
                1
                for session in template_sessions
                if int(target_item) in {int(item) for item in session}
            )
        ),
        "noop_tail_replacement_count": int(
            sum(1 for result in replacement_results if result.was_noop)
        ),
        "single_item_tail_pos0_count": 0,
        "injected_sessions_containing_target_count": int(
            sum(1 for count in target_counts if count > 0)
        ),
        "all_injected_sessions_contain_target": bool(
            all(count > 0 for count in target_counts)
        ),
        "target_occurrence_count_after_replacement": _numeric_summary(target_counts),
        "sessions_with_multiple_target_occurrences_count": int(
            sum(1 for count in target_counts if count > 1)
        ),
        "previews": _tail_replacement_previews(
            template_sessions=template_sessions,
            replaced_sessions=replaced_sessions,
            replacement_results=replacement_results,
            limit=preview_limit,
        ),
    }


def _validate_tail_replaced_sessions(
    *,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    replacement_positions: Sequence[int],
    target_item: int,
) -> None:
    if len(template_sessions) != len(replaced_sessions):
        raise RuntimeError("Injected fake-session count does not equal template count.")
    if len(template_sessions) != len(replacement_positions):
        raise RuntimeError("Replacement position count does not equal template count.")
    for original, replaced, position in zip(
        template_sessions,
        replaced_sessions,
        replacement_positions,
    ):
        original_list = [int(item) for item in original]
        replaced_list = [int(item) for item in replaced]
        replacement_position = int(position)
        if len(original_list) < 2:
            raise RuntimeError(
                "Tail-Replacement-NZ requires template session length >= 2."
            )
        if int(target_item) not in set(replaced_list):
            raise RuntimeError("Tail-Replacement-NZ injected session is missing target item.")
        if len(replaced_list) != len(original_list):
            raise RuntimeError("Tail-Replacement-NZ injected session length changed.")
        if replacement_position != len(original_list) - 1:
            raise RuntimeError("Tail-Replacement-NZ did not select the tail position.")
        expected = original_list[:-1] + [int(target_item)]
        if replaced_list != expected:
            raise RuntimeError("Tail-Replacement-NZ did not preserve the original prefix.")


def _existing_path_or_none(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    return str(path_obj) if path_obj.exists() else None


def _length_distribution(sessions: Sequence[Sequence[int]]) -> dict[str, int]:
    counts: Counter[int] = Counter(int(len(session)) for session in sessions)
    return {f"len{length}": int(count) for length, count in sorted(counts.items())}


def _numeric_summary(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    normalized = [int(value) for value in values]
    return {
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
    }


def _position_group_counts(
    sessions: Sequence[Sequence[int]],
    positions: Sequence[int],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for session, position in zip(sessions, positions):
        for group in _position_groups(int(position), int(len(session))):
            counts[group] += 1
    for group in _POSITION_GROUP_ORDER:
        counts.setdefault(group, 0)
    return counts


def _position_groups(position: int, session_length: int) -> tuple[str, ...]:
    groups: list[str] = []
    if position == 1:
        groups.append("pos1")
    elif position == 2:
        groups.append("pos2")
    elif position == 3:
        groups.append("pos3")
    elif 4 <= position <= 5:
        groups.append("pos4_5")
    elif position >= 6:
        groups.append("pos6_plus")
    if position == session_length - 1:
        groups.append("tail_position")
    return tuple(groups)


_POSITION_GROUP_ORDER = (
    "pos1",
    "pos2",
    "pos3",
    "pos4_5",
    "pos6_plus",
    "tail_position",
)


def _stringify_counts(counter: Counter[int]) -> dict[str, int]:
    return {str(position): int(count) for position, count in sorted(counter.items())}


def _stringify_ratios(counter: Counter[int], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {str(position): 0.0 for position, _ in sorted(counter.items())}
    return {
        str(position): float(count) / float(total)
        for position, count in sorted(counter.items())
    }


def _stringify_named_counts(counter: Counter[str]) -> dict[str, int]:
    return {group: int(counter.get(group, 0)) for group in _POSITION_GROUP_ORDER}


def _stringify_named_ratios(counter: Counter[str], *, total: int) -> dict[str, float]:
    if total <= 0:
        return {group: 0.0 for group in _POSITION_GROUP_ORDER}
    return {
        group: float(counter.get(group, 0)) / float(total)
        for group in _POSITION_GROUP_ORDER
    }


def _tail_replacement_previews(
    *,
    template_sessions: Sequence[Sequence[int]],
    replaced_sessions: Sequence[Sequence[int]],
    replacement_results: Sequence[TailReplacementResult],
    limit: int,
) -> list[dict[str, object]]:
    previews: list[dict[str, object]] = []
    for original, replaced, result in zip(
        template_sessions[:limit],
        replaced_sessions[:limit],
        replacement_results[:limit],
    ):
        previews.append(
            {
                "original_session": [int(item) for item in original],
                "tail_replaced_session": [int(item) for item in replaced],
                "replacement_position": int(result.replacement_position),
                "original_tail_item": int(result.original_item),
                "was_noop": bool(result.was_noop),
                "index_base": "zero_based",
            }
        )
    return previews


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_TAIL_REPLACEMENT_CONFIG_PATH,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_tail_replacement_nonzero(
        config,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
