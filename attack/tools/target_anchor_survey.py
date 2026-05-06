from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from attack.common.config import Config, load_config
from attack.common.paths import (
    INTERNAL_RANDOM_INSERTION_NONZERO_WHEN_POSSIBLE_RUN_TYPE,
    shared_artifact_paths,
)
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.data.unified_split import ensure_canonical_dataset
from attack.insertion.internal_random_insertion_nonzero_when_possible import (
    InternalRandomInsertionNonzeroWhenPossiblePolicy,
)
from attack.models._srgnn_base import SRGNNBaseRunner
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from pytorch_code.model import forward as srg_forward
from pytorch_code.model import trans_to_cpu
from pytorch_code.utils import Data


RANK_CONVENTION = "rank = 1 + count(scores > target_score); lower is better"
DEFAULT_CONFIG = (
    "attack/configs/"
    "diginetica_valbest_attack_random_nonzero_when_possible_ratio1_srgnn_sample3.yaml"
)


@dataclass
class RunningRows:
    ranks: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def add(self, rank: int, score: float) -> None:
        self.ranks.append(int(rank))
        self.scores.append(float(score))


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else (Path.cwd() / path_obj)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    normalized = [float(value) for value in values]
    return {
        "count": int(len(normalized)),
        "min": float(min(normalized)),
        "max": float(max(normalized)),
        "mean": float(sum(normalized) / len(normalized)),
        "median": _percentile(normalized, 0.5),
        "q25": _percentile(normalized, 0.25),
        "q75": _percentile(normalized, 0.75),
    }


def _top_counter(counter: Counter[int], limit: int) -> list[dict[str, int]]:
    return [
        {"item": int(item), "count": int(count)}
        for item, count in counter.most_common(int(limit))
    ]


def item_counts(sessions: Sequence[Sequence[int]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for session in sessions:
        counts.update(int(item) for item in session)
    return counts


def frequency_ranks(counts: Mapping[int, int]) -> dict[int, int]:
    ordered = sorted(counts.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))
    return {int(item): index + 1 for index, (item, _) in enumerate(ordered)}


def expanded_cases(sessions: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    prefixes: list[list[int]] = []
    labels: list[int] = []
    for session in sessions:
        session_prefixes, session_labels = expand_session_to_samples(session)
        prefixes.extend(session_prefixes)
        labels.extend(int(label) for label in session_labels)
    return prefixes, labels


def compute_target_split_stats(
    sessions: Sequence[Sequence[int]],
    target: int,
    *,
    counts: Mapping[int, int] | None = None,
    ranks: Mapping[int, int] | None = None,
    include_position_distribution: bool,
    expanded_label_count: bool,
) -> dict[str, Any]:
    target = int(target)
    counts = counts or item_counts(sessions)
    ranks = ranks or frequency_ranks(counts)
    positions: list[int] = []
    target_sessions = 0
    position_buckets: Counter[str] = Counter()
    for session in sessions:
        normalized = [int(item) for item in session]
        session_positions = [index for index, item in enumerate(normalized) if item == target]
        if session_positions:
            target_sessions += 1
        for position in session_positions:
            positions.append(position)
            if position == 0:
                bucket = "pos0"
            elif position == 1:
                bucket = "pos1"
            elif position == 2:
                bucket = "pos2"
            elif position == 3:
                bucket = "pos3"
            elif position in {4, 5}:
                bucket = "pos4_5"
            else:
                bucket = "pos6_plus"
            position_buckets[bucket] += 1
            if position == len(normalized) - 1:
                position_buckets["tail_position"] += 1

    label_count = None
    if expanded_label_count:
        _, labels = expanded_cases(sessions)
        label_count = sum(1 for label in labels if int(label) == target)

    item_count_values = list(int(value) for value in counts.values())
    occurrence = int(counts.get(target, 0))
    rank = ranks.get(target)
    popularity_percentile = None
    if rank is not None and len(ranks) > 1:
        popularity_percentile = 1.0 - ((int(rank) - 1) / float(len(ranks) - 1))
    q25 = _percentile(item_count_values, 0.25) or 0.0
    q75 = _percentile(item_count_values, 0.75) or 0.0
    if occurrence <= q25:
        density = "sparse"
    elif occurrence >= q75:
        density = "dense"
    else:
        density = "medium"

    payload: dict[str, Any] = {
        "target_occurrence_count": occurrence,
        "sessions_containing_target": int(target_sessions),
        "target_item_frequency_rank": None if rank is None else int(rank),
        "target_popularity_percentile": popularity_percentile,
        "density_relative_to_dataset": density,
        "avg_position": (
            None if not positions else float(sum(positions) / len(positions))
        ),
        "position_summary": _summary(positions),
    }
    if include_position_distribution:
        payload["position_distribution"] = {
            key: int(position_buckets.get(key, 0))
            for key in (
                "pos0",
                "pos1",
                "pos2",
                "pos3",
                "pos4_5",
                "pos6_plus",
                "tail_position",
            )
        }
    if expanded_label_count:
        payload["target_label_count_expanded_prefix_cases"] = int(label_count or 0)
    return payload


def compute_neighbor_stats(
    sessions: Sequence[Sequence[int]],
    target: int,
    *,
    top_k: int,
) -> dict[str, Any]:
    target = int(target)
    counts = item_counts(sessions)
    transition_left_count: Counter[int] = Counter()
    transition_right_count: Counter[int] = Counter()
    predecessor_counts: Counter[int] = Counter()
    successor_counts: Counter[int] = Counter()
    predecessor_sessions: defaultdict[int, set[int]] = defaultdict(set)
    successor_sessions: defaultdict[int, set[int]] = defaultdict(set)
    cooccur_occurrences: Counter[int] = Counter()
    cooccur_sessions: Counter[int] = Counter()
    target_session_count = 0
    total_transitions = 0

    for session_index, raw_session in enumerate(sessions):
        session = [int(item) for item in raw_session]
        if target in session:
            target_session_count += 1
            session_item_counts = Counter(session)
            for item, count in session_item_counts.items():
                if int(item) == target:
                    continue
                cooccur_occurrences[int(item)] += int(count)
                cooccur_sessions[int(item)] += 1
        for left, right in zip(session, session[1:]):
            left = int(left)
            right = int(right)
            transition_left_count[left] += 1
            transition_right_count[right] += 1
            total_transitions += 1
            if right == target:
                predecessor_counts[left] += 1
                predecessor_sessions[left].add(session_index)
            if left == target:
                successor_counts[right] += 1
                successor_sessions[right].add(session_index)

    base_rate_target = (
        float(sum(predecessor_counts.values())) / float(total_transitions)
        if total_transitions
        else 0.0
    )
    predecessors: list[dict[str, Any]] = []
    for item, count in predecessor_counts.most_common(top_k):
        context_count = int(transition_left_count.get(item, 0))
        confidence = float(count) / float(context_count) if context_count else 0.0
        predecessors.append(
            {
                "predecessor_item": int(item),
                "count_i_to_target": int(count),
                "item_frequency": int(counts.get(item, 0)),
                "predecessor_frequency": context_count,
                "confidence_i_to_target": confidence,
                "base_rate_target_as_next_item": base_rate_target,
                "lift_i_to_target": (
                    confidence / base_rate_target if base_rate_target else None
                ),
                "support_session_count": int(len(predecessor_sessions[item])),
                "normalized_support": (
                    float(len(predecessor_sessions[item])) / float(len(sessions))
                    if sessions
                    else 0.0
                ),
            }
        )

    target_as_left = int(transition_left_count.get(target, 0))
    successors: list[dict[str, Any]] = []
    for item, count in successor_counts.most_common(top_k):
        confidence = float(count) / float(target_as_left) if target_as_left else 0.0
        base_rate_item = (
            float(transition_right_count.get(item, 0)) / float(total_transitions)
            if total_transitions
            else 0.0
        )
        successors.append(
            {
                "successor_item": int(item),
                "count_target_to_j": int(count),
                "item_frequency": int(counts.get(item, 0)),
                "confidence_target_to_j": confidence,
                "lift_target_to_j": confidence / base_rate_item if base_rate_item else None,
                "support_session_count": int(len(successor_sessions[item])),
            }
        )

    cooccurrence: list[dict[str, Any]] = []
    session_count = len(sessions)
    p_target = float(target_session_count) / float(session_count) if session_count else 0.0
    for item, session_support in cooccur_sessions.most_common(top_k):
        p_item = (
            float(sum(1 for session in sessions if int(item) in set(map(int, session))))
            / float(session_count)
            if session_count
            else 0.0
        )
        p_both = float(session_support) / float(session_count) if session_count else 0.0
        lift = p_both / (p_item * p_target) if p_item and p_target else None
        pmi = math.log(lift) if lift and lift > 0 else None
        cooccurrence.append(
            {
                "cooccur_item": int(item),
                "cooccur_count": int(cooccur_occurrences[item]),
                "cooccur_session_count": int(session_support),
                "item_frequency": int(counts.get(item, 0)),
                "lift": lift,
                "pmi": pmi,
            }
        )

    return {
        "predecessors": predecessors,
        "successors": successors,
        "cooccurrence": cooccurrence,
        "base_rate_target_as_next_item": base_rate_target,
        "total_transitions": int(total_transitions),
    }


def filter_vulnerable_cases(
    ranks: Sequence[int],
    *,
    rank_min: int,
    rank_max: int,
) -> list[int]:
    return [
        index
        for index, rank in enumerate(ranks)
        if int(rank_min) < int(rank) <= int(rank_max)
    ]


def anchor_score(
    *,
    vulnerable_coverage: float,
    fake_session_count_with_anchor: int,
    avg_vulnerable_target_rank: float | None,
) -> float:
    if avg_vulnerable_target_rank is None or avg_vulnerable_target_rank <= 0:
        return 0.0
    rank_closeness_weight = 1.0 / math.log2(float(avg_vulnerable_target_rank) + 1.0)
    return (
        float(vulnerable_coverage)
        * math.log1p(int(fake_session_count_with_anchor))
        * rank_closeness_weight
    )


def fake_session_anchor_availability(
    fake_sessions: Sequence[Sequence[int]],
    anchors: Iterable[int],
) -> dict[int, dict[str, Any]]:
    anchor_set = {int(anchor) for anchor in anchors}
    result: dict[int, dict[str, Any]] = {}
    total_sessions = len(fake_sessions)
    for anchor in anchor_set:
        session_count = 0
        occurrence_count = 0
        feasible_count = 0
        positions: list[int] = []
        for raw_session in fake_sessions:
            session = [int(item) for item in raw_session]
            found = False
            for pos, item in enumerate(session):
                if item != anchor:
                    continue
                found = True
                occurrence_count += 1
                positions.append(int(pos))
                if pos < len(session) - 1:
                    feasible_count += 1
            if found:
                session_count += 1
        result[anchor] = {
            "anchor_item": int(anchor),
            "fake_session_count_with_anchor": int(session_count),
            "fake_session_coverage": (
                float(session_count) / float(total_sessions) if total_sessions else 0.0
            ),
            "total_anchor_occurrences_in_fake_sessions": int(occurrence_count),
            "avg_position_in_fake_sessions": (
                None if not positions else float(sum(positions) / len(positions))
            ),
            "anchor_after_insertion_feasible_count": int(feasible_count),
            "internal_insertion_feasible_ratio": (
                float(feasible_count) / float(occurrence_count)
                if occurrence_count
                else 0.0
            ),
        }
    return result


def score_validation_prefixes(
    config: Config,
    *,
    checkpoint_path: Path,
    prefixes: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> dict[int, dict[str, list[float | int]]]:
    train_config = dict(config.attack.poison_model.params["train"])
    runner = SRGNNBaseRunner(config)
    runner.build_model(build_srgnn_opt_from_train_config(train_config))
    runner.load_model(checkpoint_path, map_location="cpu")
    if runner.model is None:
        raise RuntimeError("SR-GNN model failed to initialize.")

    normalized_prefixes = [list(map(int, prefix)) for prefix in prefixes]
    data = Data((normalized_prefixes, [1] * len(normalized_prefixes)), shuffle=False)
    targets_list = [int(target) for target in targets]
    output = {
        int(target): {"ranks": [], "scores": []}
        for target in targets_list
    }
    target_tensor = None
    runner.model.eval()
    with torch.no_grad():
        for batch_indices in data.generate_batch(runner.model.batch_size):
            _, scores = srg_forward(runner.model, batch_indices, data)
            probabilities = torch.softmax(scores, dim=1)
            if target_tensor is None or target_tensor.device != scores.device:
                target_tensor = torch.as_tensor(
                    [target - 1 for target in targets_list],
                    dtype=torch.long,
                    device=scores.device,
                )
            for target_position, target in enumerate(targets_list):
                item_index = target_tensor[target_position]
                target_scores = scores[:, item_index]
                ranks = 1 + torch.sum(scores > target_scores.unsqueeze(1), dim=1)
                target_probs = probabilities[:, item_index]
                output[target]["ranks"].extend(
                    int(value) for value in trans_to_cpu(ranks).tolist()
                )
                output[target]["scores"].extend(
                    float(value) for value in trans_to_cpu(target_probs).tolist()
                )
    return output


def vulnerable_anchor_analysis(
    prefixes: Sequence[Sequence[int]],
    ranks: Sequence[int],
    scores: Sequence[float],
    *,
    rank_min: int,
    rank_max: int,
    train_counts: Mapping[int, int],
    train_ranks: Mapping[int, int],
    train_predecessor_by_item: Mapping[int, Mapping[str, Any]],
    train_cooccur_by_item: Mapping[int, Mapping[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    vulnerable_indices = filter_vulnerable_cases(
        ranks,
        rank_min=rank_min,
        rank_max=rank_max,
    )
    last_items: dict[int, RunningRows] = defaultdict(RunningRows)
    last_pairs: dict[tuple[int, int], RunningRows] = defaultdict(RunningRows)
    length_buckets: Counter[str] = Counter()
    lengths: list[int] = []
    for index in vulnerable_indices:
        prefix = [int(item) for item in prefixes[index]]
        if not prefix:
            continue
        length = len(prefix)
        lengths.append(length)
        if length == 1:
            length_buckets["len1"] += 1
        elif length == 2:
            length_buckets["len2"] += 1
        elif length == 3:
            length_buckets["len3"] += 1
        elif length == 4:
            length_buckets["len4"] += 1
        else:
            length_buckets["len5_plus"] += 1
        last_items[prefix[-1]].add(int(ranks[index]), float(scores[index]))
        if len(prefix) >= 2:
            last_pairs[(prefix[-2], prefix[-1])].add(
                int(ranks[index]),
                float(scores[index]),
            )

    vulnerable_count = len(vulnerable_indices)
    last_rows: list[dict[str, Any]] = []
    for item, rows in last_items.items():
        predecessor = train_predecessor_by_item.get(item, {})
        cooccur = train_cooccur_by_item.get(item, {})
        last_rows.append(
            {
                "anchor_item": int(item),
                "vulnerable_count": int(len(rows.ranks)),
                "vulnerable_coverage": (
                    float(len(rows.ranks)) / float(vulnerable_count)
                    if vulnerable_count
                    else 0.0
                ),
                "avg_target_rank": float(sum(rows.ranks) / len(rows.ranks)),
                "median_target_rank": _percentile(rows.ranks, 0.5),
                "avg_target_score": float(sum(rows.scores) / len(rows.scores)),
                "item_frequency_in_train_sub": int(train_counts.get(item, 0)),
                "item_frequency_rank": train_ranks.get(item),
                "appears_in_train_predecessors": bool(predecessor),
                "train_predecessor_count_to_target": int(
                    predecessor.get("count_i_to_target", 0)
                ),
                "train_cooccur_count_with_target": int(
                    cooccur.get("cooccur_count", 0)
                ),
            }
        )
    last_rows.sort(key=lambda row: (-int(row["vulnerable_count"]), int(row["anchor_item"])))

    pair_rows: list[dict[str, Any]] = []
    for (item_a, item_b), rows in last_pairs.items():
        pair_rows.append(
            {
                "item_a": int(item_a),
                "item_b": int(item_b),
                "vulnerable_count": int(len(rows.ranks)),
                "vulnerable_coverage": (
                    float(len(rows.ranks)) / float(vulnerable_count)
                    if vulnerable_count
                    else 0.0
                ),
                "avg_target_rank": float(sum(rows.ranks) / len(rows.ranks)),
                "avg_target_score": float(sum(rows.scores) / len(rows.scores)),
            }
        )
    pair_rows.sort(
        key=lambda row: (-int(row["vulnerable_count"]), int(row["item_a"]), int(row["item_b"]))
    )

    return {
        "rank_min": int(rank_min),
        "rank_max": int(rank_max),
        "rank_convention": RANK_CONVENTION,
        "total_validation_prefix_count": int(len(prefixes)),
        "vulnerable_prefix_count": int(vulnerable_count),
        "vulnerable_prefix_ratio": (
            float(vulnerable_count) / float(len(prefixes)) if prefixes else 0.0
        ),
        "target_rank_summary": _summary(ranks),
        "target_score_summary": _summary(scores),
        "vulnerable_rank_summary": _summary([ranks[i] for i in vulnerable_indices]),
        "vulnerable_score_summary": _summary([scores[i] for i in vulnerable_indices]),
        "last_item_anchors": last_rows[:top_k],
        "last_2_pair_anchors": pair_rows[:top_k],
        "prefix_length_distribution": {
            "len1": int(length_buckets.get("len1", 0)),
            "len2": int(length_buckets.get("len2", 0)),
            "len3": int(length_buckets.get("len3", 0)),
            "len4": int(length_buckets.get("len4", 0)),
            "len5_plus": int(length_buckets.get("len5_plus", 0)),
            "mean_length": None if not lengths else float(sum(lengths) / len(lengths)),
            "median_length": _percentile(lengths, 0.5),
        },
    }


def resolve_existing_poison_checkpoint(config: Config) -> Path | None:
    paths = shared_artifact_paths(config, run_type="random_nonzero_when_possible")
    candidates = [
        paths.get("poison_model"),
        paths.get("legacy_attack_poison_model"),
        Path("outputs/shared/diginetica/poison_models/poison_model_3bd9126448/poison_model.pt"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        path = _repo_path(candidate)
        if path.exists():
            return path
    return None


def resolve_fake_sessions(
    config: Config,
    *,
    explicit_path: str | Path | None,
) -> tuple[list[list[int]] | None, Path | None]:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(_repo_path(explicit_path))
    paths = shared_artifact_paths(config, run_type="random_nonzero_when_possible")
    candidates.append(_repo_path(paths["fake_sessions"]))
    candidates.append(
        Path("outputs/shared/diginetica/attack/attack_shared_e54e7448f5/fake_sessions.pkl")
    )
    for candidate in candidates:
        if candidate.exists():
            sessions = _load_pickle(candidate)
            return [list(map(int, session)) for session in sessions], candidate
    return None, None


def resolve_p5_metadata_paths(
    targets: Sequence[int],
    explicit_paths: Sequence[str | Path],
) -> dict[int, Path]:
    resolved: dict[int, Path] = {}
    for raw_path in explicit_paths:
        path = _repo_path(raw_path)
        payload = _read_json(path)
        if payload is None:
            continue
        target = payload.get("target_item")
        if target is not None:
            resolved[int(target)] = path
    for target in targets:
        if int(target) in resolved:
            continue
        pattern = (
            "outputs/runs/diginetica/"
            f"**/targets/{int(target)}/internal_random_insertion_metadata.json"
        )
        matches = sorted(Path.cwd().glob(pattern), key=lambda path: str(path))
        if matches:
            resolved[int(target)] = matches[-1]
    return resolved


def compute_p5_survey(
    *,
    config: Config,
    target: int,
    metadata_path: Path | None,
    fake_sessions: Sequence[Sequence[int]] | None,
    vulnerable_last_items: Sequence[Mapping[str, Any]],
    vulnerable_pairs: Sequence[Mapping[str, Any]],
    train_predecessors: Sequence[Mapping[str, Any]],
    train_successors: Sequence[Mapping[str, Any]],
    train_cooccurrence: Sequence[Mapping[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    if metadata_path is None or not metadata_path.exists():
        return {
            "available": False,
            "message": f"P5 metadata unavailable for target {int(target)}; overlap analysis skipped.",
        }
    metadata = _read_json(metadata_path) or {}
    if fake_sessions is None:
        fake_path = metadata.get("template_fake_sessions_path")
        if isinstance(fake_path, str) and fake_path.strip() and _repo_path(fake_path).exists():
            fake_sessions = _load_pickle(_repo_path(fake_path))
    if fake_sessions is None:
        return {
            "available": False,
            "metadata_path": str(metadata_path),
            "message": f"P5 metadata found for target {int(target)}, but fake sessions were unavailable; overlap analysis skipped.",
        }

    topk_ratio = float(metadata.get("internal_insertion_slot_topk_ratio", config.attack.replacement_topk_ratio))
    policy = InternalRandomInsertionNonzeroWhenPossiblePolicy(
        topk_ratio=topk_ratio,
        rng=random.Random(int(config.seeds.fake_session_seed)),
    )
    results = [policy.apply_with_metadata(session, int(target)) for session in fake_sessions]
    left_counts = Counter(int(result.left_item) for result in results)
    right_counts = Counter(int(result.right_item) for result in results)
    pair_counts = Counter((int(result.left_item), int(result.right_item)) for result in results)
    slot_counts = Counter(int(result.insertion_slot) for result in results)
    length_counts = Counter(len(result.session) for result in results)

    vulnerable_left_set = {int(row["anchor_item"]) for row in vulnerable_last_items}
    vulnerable_pair_set = {
        (int(row["item_a"]), int(row["item_b"])) for row in vulnerable_pairs
    }
    predecessor_set = {int(row["predecessor_item"]) for row in train_predecessors}
    successor_set = {int(row["successor_item"]) for row in train_successors}
    cooccur_set = {int(row["cooccur_item"]) for row in train_cooccurrence}
    left_set = set(left_counts)
    right_set = set(right_counts)
    pair_set = set(pair_counts)

    def overlap_payload(source: set[Any], anchor_set: set[Any], label: str) -> dict[str, Any]:
        overlap = source & anchor_set
        top = sorted(
            overlap,
            key=lambda item: (
                -(
                    left_counts.get(item, 0)
                    if not isinstance(item, tuple)
                    else pair_counts.get(item, 0)
                ),
                item,
            ),
        )[:top_k]
        return {
            "name": label,
            "overlap_count": int(len(overlap)),
            "overlap_ratio_relative_to_p5_unique": (
                float(len(overlap)) / float(len(source)) if source else 0.0
            ),
            "overlap_ratio_relative_to_anchor_set": (
                float(len(overlap)) / float(len(anchor_set)) if anchor_set else 0.0
            ),
            "top_overlapping_anchors": [
                list(item) if isinstance(item, tuple) else int(item) for item in top
            ],
        }

    metrics_path = metadata_path.parent / "victims" / "srgnn" / "metrics.json"
    metrics = _read_json(metrics_path) or {}
    metric_values = metrics.get("metrics", {}) if isinstance(metrics.get("metrics"), dict) else {}
    raw_lowk = None
    lowk_keys = (
        "targeted_mrr@10",
        "targeted_mrr@20",
        "targeted_recall@10",
        "targeted_recall@20",
    )
    if all(key in metric_values for key in lowk_keys):
        raw_lowk = float(sum(float(metric_values[key]) for key in lowk_keys) / len(lowk_keys))

    return {
        "available": True,
        "metadata_path": str(metadata_path),
        "p5_raw_lowk": raw_lowk,
        "unique_left_item_count": int(len(left_counts)),
        "unique_right_item_count": int(len(right_counts)),
        "unique_left_right_pair_count": int(len(pair_counts)),
        "top_left_items": _top_counter(left_counts, top_k),
        "top_right_items": _top_counter(right_counts, top_k),
        "top_left_right_pairs": [
            {"item_a": int(pair[0]), "item_b": int(pair[1]), "count": int(count)}
            for pair, count in pair_counts.most_common(top_k)
        ],
        "insertion_slot_distribution": {
            str(slot): int(count) for slot, count in sorted(slot_counts.items())
        },
        "session_length_distribution": {
            f"len{length}": int(count) for length, count in sorted(length_counts.items())
        },
        "overlap": {
            "p5_left_items_intersect_vulnerable_last_item_anchors": overlap_payload(
                left_set,
                vulnerable_left_set,
                "P5 left items ∩ vulnerable last-item anchors",
            ),
            "p5_right_items_intersect_target_successors": overlap_payload(
                right_set,
                successor_set,
                "P5 right items ∩ target successors",
            ),
            "p5_left_right_pairs_intersect_vulnerable_last_2_prefix_pairs": overlap_payload(
                pair_set,
                vulnerable_pair_set,
                "P5 left-right pairs ∩ vulnerable last-2 prefix pairs",
            ),
            "p5_left_items_intersect_train_predecessors": overlap_payload(
                left_set,
                predecessor_set,
                "P5 left items ∩ train predecessors",
            ),
            "p5_left_items_intersect_cooccurrence_items": overlap_payload(
                left_set,
                cooccur_set,
                "P5 left items ∩ co-occurrence items",
            ),
        },
        "left_counts": {str(item): int(count) for item, count in left_counts.items()},
        "right_counts": {str(item): int(count) for item, count in right_counts.items()},
    }


def candidate_rows(
    *,
    target: int,
    train_predecessors: Sequence[Mapping[str, Any]],
    vulnerable_last_items: Sequence[Mapping[str, Any]],
    cooccurrence: Sequence[Mapping[str, Any]],
    fake_availability: Mapping[int, Mapping[str, Any]],
    p5: Mapping[str, Any],
) -> list[dict[str, Any]]:
    predecessor_by_item = {int(row["predecessor_item"]): row for row in train_predecessors}
    vulnerable_by_item = {int(row["anchor_item"]): row for row in vulnerable_last_items}
    cooccur_by_item = {int(row["cooccur_item"]): row for row in cooccurrence}
    p5_left_counts = {
        int(item): int(count)
        for item, count in (p5.get("left_counts", {}) if p5.get("available") else {}).items()
    }
    p5_right_counts = {
        int(item): int(count)
        for item, count in (p5.get("right_counts", {}) if p5.get("available") else {}).items()
    }
    anchors = (
        set(predecessor_by_item)
        | set(vulnerable_by_item)
        | set(cooccur_by_item)
        | set(p5_left_counts)
    )
    rows: list[dict[str, Any]] = []
    for anchor in sorted(anchors):
        pred = predecessor_by_item.get(anchor, {})
        vuln = vulnerable_by_item.get(anchor, {})
        cooccur = cooccur_by_item.get(anchor, {})
        fake = fake_availability.get(anchor, {})
        avg_rank = vuln.get("avg_target_rank")
        score = anchor_score(
            vulnerable_coverage=float(vuln.get("vulnerable_coverage", 0.0) or 0.0),
            fake_session_count_with_anchor=int(fake.get("fake_session_count_with_anchor", 0) or 0),
            avg_vulnerable_target_rank=(
                None if avg_rank is None else float(avg_rank)
            ),
        )
        overlap_flags = []
        if anchor in p5_left_counts and anchor in vulnerable_by_item:
            overlap_flags.append("p5_left_and_vulnerable_last_item")
        if anchor in p5_left_counts and anchor in predecessor_by_item:
            overlap_flags.append("p5_left_and_train_predecessor")
        if anchor in p5_left_counts and anchor in cooccur_by_item:
            overlap_flags.append("p5_left_and_cooccur")
        rows.append(
            {
                "target_item": int(target),
                "anchor_item": int(anchor),
                "is_train_predecessor": bool(anchor in predecessor_by_item),
                "is_vulnerable_last_item": bool(anchor in vulnerable_by_item),
                "is_cooccur_item": bool(anchor in cooccur_by_item),
                "is_p5_left_item": bool(anchor in p5_left_counts),
                "train_predecessor_count_to_target": int(pred.get("count_i_to_target", 0) or 0),
                "train_predecessor_confidence": pred.get("confidence_i_to_target", 0.0),
                "train_predecessor_lift": pred.get("lift_i_to_target"),
                "vulnerable_count": int(vuln.get("vulnerable_count", 0) or 0),
                "vulnerable_coverage": float(vuln.get("vulnerable_coverage", 0.0) or 0.0),
                "avg_vulnerable_target_rank": avg_rank,
                "avg_vulnerable_target_score": vuln.get("avg_target_score"),
                "cooccur_count": int(cooccur.get("cooccur_count", 0) or 0),
                "fake_session_count_with_anchor": int(fake.get("fake_session_count_with_anchor", 0) or 0),
                "fake_session_coverage": float(fake.get("fake_session_coverage", 0.0) or 0.0),
                "internal_insertion_feasible_count": int(fake.get("anchor_after_insertion_feasible_count", 0) or 0),
                "internal_insertion_feasible_ratio": float(fake.get("internal_insertion_feasible_ratio", 0.0) or 0.0),
                "p5_left_count": int(p5_left_counts.get(anchor, 0)),
                "p5_right_count": int(p5_right_counts.get(anchor, 0)),
                "p5_overlap_flags": ";".join(overlap_flags),
                "anchor_score": score,
            }
        )
    rows.sort(key=lambda row: (-float(row["anchor_score"]), -int(row["vulnerable_count"]), int(row["anchor_item"])))
    return rows


def _markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_target_markdown(path: Path, payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> None:
    target = payload["target_item"]
    train = payload["target_sparsity"]["train_sub"]
    vulnerable = payload["vulnerable_validation_prefixes"]
    neighbors = payload["historical_neighbors_train_sub"]
    p5 = payload["p5_internal_random_insertion_local_pair_survey"]
    top_vuln = vulnerable["last_item_anchors"][:10]
    top_pred = neighbors["predecessors"][:10]
    top_candidates = candidates[:10]
    lines = [
        f"# Target Anchor Survey: {target}",
        "",
        "This is an analysis-only survey. No attack method was implemented or modified.",
        f"Rank convention: `{RANK_CONVENTION}`.",
        "Vulnerable anchors represent validation contexts where the target is not top-20 yet but is close enough to be potentially promoted.",
        "Co-occurrence is auxiliary because SBR is next-item prediction.",
        "anchor_score is exploratory and not yet validated as a poisoning objective.",
        "",
        "## Target Sparsity",
        _markdown_table(
            [[
                "train_sub",
                train["target_occurrence_count"],
                train["sessions_containing_target"],
                train["target_item_frequency_rank"],
                _fmt(train["target_popularity_percentile"]),
                train["density_relative_to_dataset"],
            ]],
            ["split", "occurrences", "sessions", "frequency_rank", "popularity_percentile", "density"],
        ),
        "",
        f"Position distribution: `{train.get('position_distribution', {})}`",
        "",
        "## Historical Neighbors",
        _markdown_table(
            [
                [
                    row["predecessor_item"],
                    row["count_i_to_target"],
                    _fmt(row["confidence_i_to_target"]),
                    _fmt(row["lift_i_to_target"]),
                    row["support_session_count"],
                ]
                for row in top_pred
            ],
            ["predecessor", "count", "confidence", "lift", "support_sessions"],
        ),
        "",
        "## Vulnerable Validation Anchors",
        f"Range: `{vulnerable['rank_min']} < rank <= {vulnerable['rank_max']}`. Vulnerable prefixes: `{vulnerable['vulnerable_prefix_count']}` / `{vulnerable['total_validation_prefix_count']}`.",
        _markdown_table(
            [
                [
                    row["anchor_item"],
                    row["vulnerable_count"],
                    _fmt(row["vulnerable_coverage"]),
                    _fmt(row["avg_target_rank"], 2),
                    _fmt(row["avg_target_score"], 6),
                    row["train_predecessor_count_to_target"],
                ]
                for row in top_vuln
            ],
            ["anchor", "vuln_count", "coverage", "avg_rank", "avg_score", "train_pred_count"],
        ),
        "",
        "## Fake-Session Availability",
        _markdown_table(
            [
                [
                    row["anchor_item"],
                    row["fake_session_count_with_anchor"],
                    _fmt(row["fake_session_coverage"]),
                    row["internal_insertion_feasible_count"],
                    _fmt(row["internal_insertion_feasible_ratio"]),
                ]
                for row in top_candidates
            ],
            ["anchor", "fake_sessions", "fake_coverage", "feasible_occurrences", "feasible_ratio"],
        ),
        "",
        "## P5 Internal-Random-Insertion Survey",
    ]
    if p5.get("available"):
        lines.extend(
            [
                f"Metadata: `{p5.get('metadata_path')}`",
                f"Unique left/right/pairs: `{p5.get('unique_left_item_count')}` / `{p5.get('unique_right_item_count')}` / `{p5.get('unique_left_right_pair_count')}`.",
                f"P5 raw_lowk: `{_fmt(p5.get('p5_raw_lowk'))}`",
            ]
        )
    else:
        lines.append(str(p5.get("message")))
    lines.extend(
        [
            "",
            "## Top Candidate Anchors",
            _markdown_table(
                [
                    [
                        row["anchor_item"],
                        _fmt(row["anchor_score"]),
                        row["vulnerable_count"],
                        _fmt(row["vulnerable_coverage"]),
                        row["fake_session_count_with_anchor"],
                        row["train_predecessor_count_to_target"],
                        row["p5_left_count"],
                    ]
                    for row in top_candidates
                ],
                ["anchor", "anchor_score", "vuln_count", "vuln_cov", "fake_sessions", "train_pred", "p5_left"],
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recommend_next_experiment(summary_row: Mapping[str, Any]) -> str:
    vulnerable_count = int(summary_row.get("vulnerable_prefix_count", 0) or 0)
    top_cov = float(summary_row.get("top_vulnerable_anchor_coverage", 0.0) or 0.0)
    fake_cov = float(summary_row.get("fake_session_coverage_top20_vulnerable_anchors", 0.0) or 0.0)
    pred_count = int(summary_row.get("top_train_predecessor_count", 0) or 0)
    if vulnerable_count <= 0:
        return "Not enough anchor signal"
    if top_cov >= 0.02 and fake_cov >= 0.05:
        return "Existing-Anchor Internal Insertion"
    if top_cov >= 0.01 and fake_cov < 0.05:
        return "Anchor-Replacement + Internal Insertion"
    if pred_count <= 1 and fake_cov >= 0.20:
        return "Continue with Internal-Random-Insertion only"
    return "Anchor-Replacement + Internal Insertion"


def summarize_target(
    *,
    target_payload: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    fake_sessions: Sequence[Sequence[int]] | None,
) -> dict[str, Any]:
    target = int(target_payload["target_item"])
    train = target_payload["target_sparsity"]["train_sub"]
    vulnerable = target_payload["vulnerable_validation_prefixes"]
    top_vuln = vulnerable["last_item_anchors"][:20]
    top_vuln_set = {int(row["anchor_item"]) for row in top_vuln}
    fake_coverage_top20 = 0.0
    if fake_sessions and top_vuln_set:
        count = sum(
            1
            for session in fake_sessions
            if any(int(item) in top_vuln_set for item in session)
        )
        fake_coverage_top20 = float(count) / float(len(fake_sessions))
    neighbors = target_payload["historical_neighbors_train_sub"]
    p5 = target_payload["p5_internal_random_insertion_local_pair_survey"]
    row = {
        "target": target,
        "train_occurrence_count": int(train["target_occurrence_count"]),
        "vulnerable_prefix_count": int(vulnerable["vulnerable_prefix_count"]),
        "top_vulnerable_anchor_coverage": (
            None if not vulnerable["last_item_anchors"] else float(vulnerable["last_item_anchors"][0]["vulnerable_coverage"])
        ),
        "top_train_predecessor_count": (
            0 if not neighbors["predecessors"] else int(neighbors["predecessors"][0]["count_i_to_target"])
        ),
        "fake_session_coverage_top20_vulnerable_anchors": fake_coverage_top20,
        "p5_available": bool(p5.get("available")),
        "p5_raw_lowk": p5.get("p5_raw_lowk") if p5.get("available") else None,
        "top5_vulnerable_anchors": ", ".join(
            str(row["anchor_item"]) for row in vulnerable["last_item_anchors"][:5]
        ),
        "top5_train_predecessors": ", ".join(
            str(row["predecessor_item"]) for row in neighbors["predecessors"][:5]
        ),
        "top5_candidate_anchors": ", ".join(str(row["anchor_item"]) for row in candidates[:5]),
    }
    row["recommendation"] = recommend_next_experiment(row)
    if row["vulnerable_prefix_count"] <= 0 or fake_coverage_top20 <= 0.0:
        feasible = "no"
    elif fake_coverage_top20 < 0.05:
        feasible = "limited"
    else:
        feasible = "yes"
    row["existing_anchor_insertion_feasible"] = feasible
    return row


def write_summary_markdown(path: Path, rows: Sequence[Mapping[str, Any]], payloads: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Target Anchor Survey Summary",
        "",
        "This survey uses train_sub and validation for anchor analysis. Test data is excluded unless explicitly enabled as post-hoc diagnostics.",
        "anchor_score is exploratory and not yet validated as a poisoning objective.",
        "",
        "## Executive Summary",
        _markdown_table(
            [
                [
                    row["target"],
                    row["train_occurrence_count"],
                    row["vulnerable_prefix_count"],
                    row["top5_vulnerable_anchors"],
                    row["top5_train_predecessors"],
                    _fmt(row["fake_session_coverage_top20_vulnerable_anchors"]),
                    row["existing_anchor_insertion_feasible"],
                ]
                for row in rows
            ],
            ["target", "train_count", "vuln_prefixes", "top5_vuln_anchors", "top5_train_preds", "fake_cov_top20", "feasible"],
        ),
        "",
        "## Cross-Target Comparison",
        _markdown_table(
            [
                [
                    row["target"],
                    row["train_occurrence_count"],
                    row["vulnerable_prefix_count"],
                    _fmt(row["top_vulnerable_anchor_coverage"]),
                    row["top_train_predecessor_count"],
                    _fmt(row["fake_session_coverage_top20_vulnerable_anchors"]),
                    row["p5_available"],
                    _fmt(row["p5_raw_lowk"]),
                    row["recommendation"],
                ]
                for row in rows
            ],
            [
                "target",
                "train_occurrence_count",
                "vulnerable_prefix_count",
                "top_vulnerable_anchor_coverage",
                "top_train_predecessor_count",
                "fake_session_coverage_top20_vulnerable_anchors",
                "p5_available",
                "p5_raw_lowk",
                "recommendation",
            ],
        ),
        "",
        "## Interpretation",
    ]
    sparse_targets = [
        int(payload["target_item"])
        for payload in payloads
        if payload["target_sparsity"]["train_sub"]["density_relative_to_dataset"] == "sparse"
    ]
    rows_by_target = {int(row["target"]): row for row in rows}
    best = max(
        rows,
        key=lambda row: (
            int(row["vulnerable_prefix_count"]),
            float(row["fake_session_coverage_top20_vulnerable_anchors"] or 0.0),
        ),
    ) if rows else None
    lines.extend(
        [
            f"- Historical transitions are sparse for: {', '.join(map(str, sparse_targets)) if sparse_targets else 'none by item-frequency quartile'}; predecessor counts should be treated cautiously when they are low.",
            "- Vulnerable validation anchors can differ from historical predecessors; the per-target CSVs expose both flags side by side.",
            "- Useful anchors are available in fake sessions when fake coverage for top vulnerable anchors is nonzero; low coverage points to anchor replacement before insertion.",
            "- P5 should be interpreted as broad-diversity coverage when P5-left overlap with vulnerable or predecessor anchors is low.",
            f"- Best next target for anchor-aware internal insertion: {best['target'] if best else ''}.",
            "",
            "## Recommended Next Experiments",
        ]
    )
    for row in rows:
        lines.append(f"- Target {row['target']}: {row['recommendation']}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_survey(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_config(config_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [int(target) for target in args.targets]
    top_k = int(args.top_k)

    dataset = ensure_canonical_dataset(config)
    train_counts = item_counts(dataset.train_sub)
    train_ranks = frequency_ranks(train_counts)
    valid_prefixes, valid_labels = expanded_cases(dataset.valid)
    checkpoint_path = resolve_existing_poison_checkpoint(config)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No existing validation-best SR-GNN checkpoint was found; survey does not train a replacement."
        )
    fake_sessions, fake_sessions_path = resolve_fake_sessions(
        config,
        explicit_path=args.fake_sessions_path,
    )
    p5_paths = resolve_p5_metadata_paths(targets, args.p5_metadata_paths or [])

    print(f"[survey] Scoring {len(valid_prefixes)} validation prefixes with {checkpoint_path}")
    validation_scores = score_validation_prefixes(
        config,
        checkpoint_path=checkpoint_path,
        prefixes=valid_prefixes,
        targets=targets,
    )

    target_payloads: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_candidate_rows: dict[int, list[dict[str, Any]]] = {}
    for target in targets:
        print(f"[survey] Building survey outputs for target {target}")
        sparsity = {
            "train_sub": compute_target_split_stats(
                dataset.train_sub,
                target,
                counts=train_counts,
                ranks=train_ranks,
                include_position_distribution=True,
                expanded_label_count=False,
            ),
            "validation": compute_target_split_stats(
                dataset.valid,
                target,
                include_position_distribution=False,
                expanded_label_count=True,
            ),
        }
        if bool(args.include_test_posthoc):
            sparsity["test_posthoc"] = compute_target_split_stats(
                dataset.test,
                target,
                include_position_distribution=False,
                expanded_label_count=True,
            )
            sparsity["test_posthoc"]["test_posthoc_only"] = True

        neighbors = compute_neighbor_stats(dataset.train_sub, target, top_k=top_k)
        predecessor_by_item = {
            int(row["predecessor_item"]): row for row in neighbors["predecessors"]
        }
        cooccur_by_item = {int(row["cooccur_item"]): row for row in neighbors["cooccurrence"]}
        ranks = [int(value) for value in validation_scores[target]["ranks"]]
        scores = [float(value) for value in validation_scores[target]["scores"]]
        vulnerable = vulnerable_anchor_analysis(
            valid_prefixes,
            ranks,
            scores,
            rank_min=int(args.rank_min),
            rank_max=int(args.rank_max),
            train_counts=train_counts,
            train_ranks=train_ranks,
            train_predecessor_by_item=predecessor_by_item,
            train_cooccur_by_item=cooccur_by_item,
            top_k=top_k,
        )
        availability_anchors = set()
        availability_anchors.update(
            int(row["anchor_item"]) for row in vulnerable["last_item_anchors"][:top_k]
        )
        availability_anchors.update(
            int(row["predecessor_item"]) for row in neighbors["predecessors"][:top_k]
        )
        availability_anchors.update(
            int(row["cooccur_item"]) for row in neighbors["cooccurrence"][:top_k]
        )
        fake_availability = (
            fake_session_anchor_availability(fake_sessions, availability_anchors)
            if fake_sessions is not None
            else {}
        )
        fake_payload = {
            "available": fake_sessions is not None,
            "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
            "total_fake_sessions": 0 if fake_sessions is None else int(len(fake_sessions)),
            "anchors": list(fake_availability.values()),
        }

        p5 = compute_p5_survey(
            config=config,
            target=target,
            metadata_path=p5_paths.get(target),
            fake_sessions=fake_sessions,
            vulnerable_last_items=vulnerable["last_item_anchors"],
            vulnerable_pairs=vulnerable["last_2_pair_anchors"],
            train_predecessors=neighbors["predecessors"],
            train_successors=neighbors["successors"],
            train_cooccurrence=neighbors["cooccurrence"],
            top_k=top_k,
        )
        rows = candidate_rows(
            target=target,
            train_predecessors=neighbors["predecessors"],
            vulnerable_last_items=vulnerable["last_item_anchors"],
            cooccurrence=neighbors["cooccurrence"],
            fake_availability=fake_availability,
            p5=p5,
        )
        payload = {
            "target_item": int(target),
            "metadata": {
                "config_path": str(config_path),
                "config_experiment_name": config.experiment.name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_role": "validation-best SR-GNN poison-model checkpoint used for fake-session generation",
                "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
                "rank_convention": RANK_CONVENTION,
                "rank_min": int(args.rank_min),
                "rank_max": int(args.rank_max),
                "top_k": int(top_k),
                "test_used_for_anchor_selection": False,
            },
            "target_sparsity": sparsity,
            "historical_neighbors_train_sub": neighbors,
            "vulnerable_validation_prefixes": vulnerable,
            "fake_session_anchor_availability": fake_payload,
            "p5_internal_random_insertion_local_pair_survey": p5,
        }
        _write_json(output_dir / f"target_anchor_survey_{target}.json", payload)
        candidate_fields = [
            "target_item",
            "anchor_item",
            "is_train_predecessor",
            "is_vulnerable_last_item",
            "is_cooccur_item",
            "is_p5_left_item",
            "train_predecessor_count_to_target",
            "train_predecessor_confidence",
            "train_predecessor_lift",
            "vulnerable_count",
            "vulnerable_coverage",
            "avg_vulnerable_target_rank",
            "avg_vulnerable_target_score",
            "cooccur_count",
            "fake_session_count_with_anchor",
            "fake_session_coverage",
            "internal_insertion_feasible_count",
            "internal_insertion_feasible_ratio",
            "p5_left_count",
            "p5_right_count",
            "p5_overlap_flags",
            "anchor_score",
        ]
        _write_csv(
            output_dir / f"target_anchor_candidates_{target}.csv",
            rows,
            candidate_fields,
        )
        write_target_markdown(
            output_dir / f"target_anchor_survey_{target}.md",
            payload,
            rows,
        )
        target_payloads.append(payload)
        all_candidate_rows[target] = rows
        summary_rows.append(
            summarize_target(
                target_payload=payload,
                candidates=rows,
                fake_sessions=fake_sessions,
            )
        )

    for row in summary_rows:
        row["recommendation"] = recommend_next_experiment(row)
    summary_payload = {
        "metadata": {
            "config_path": str(config_path),
            "config_experiment_name": config.experiment.name,
            "targets": targets,
            "rank_convention": RANK_CONVENTION,
            "rank_min": int(args.rank_min),
            "rank_max": int(args.rank_max),
            "checkpoint_path": str(checkpoint_path),
            "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
            "test_used_for_anchor_selection": False,
        },
        "targets": target_payloads,
        "summary_rows": summary_rows,
    }
    _write_json(output_dir / "target_anchor_survey_summary.json", summary_payload)
    _write_csv(
        output_dir / "target_anchor_survey_summary.csv",
        summary_rows,
        [
            "target",
            "train_occurrence_count",
            "vulnerable_prefix_count",
            "top_vulnerable_anchor_coverage",
            "top_train_predecessor_count",
            "fake_session_coverage_top20_vulnerable_anchors",
            "p5_available",
            "p5_raw_lowk",
            "recommendation",
            "existing_anchor_insertion_feasible",
            "top5_vulnerable_anchors",
            "top5_train_predecessors",
            "top5_candidate_anchors",
        ],
    )
    write_summary_markdown(
        output_dir / "target_anchor_survey_summary.md",
        summary_rows,
        target_payloads,
    )
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Target anchor dataset survey.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--targets", nargs="+", type=int, required=True)
    parser.add_argument("--rank-min", type=int, default=20)
    parser.add_argument("--rank-max", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/analysis/target_anchor_survey")
    parser.add_argument("--fake-sessions-path", default=None)
    parser.add_argument("--p5-metadata-paths", nargs="*", default=[])
    parser.add_argument(
        "--include-test-posthoc",
        action="store_true",
        help="Include test statistics as a clearly marked post-hoc section.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_survey(args)
    output_dir = Path(args.output_dir)
    print(f"[survey] Wrote summary to {output_dir / 'target_anchor_survey_summary.md'}")
    print(
        "[survey] Recommendations: "
        + "; ".join(
            f"{row['target']}={row['recommendation']}"
            for row in payload["summary_rows"]
        )
    )


if __name__ == "__main__":
    main()
