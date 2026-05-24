from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from attack.common.config import Config, load_config
from attack.common.paths import shared_artifact_paths
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.data.unified_split import ensure_canonical_dataset
from attack.models._srgnn_base import SRGNNBaseRunner
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from pytorch_code.model import forward as srg_forward
from pytorch_code.model import trans_to_cpu
from pytorch_code.utils import Data


RANK_CONVENTION = "rank = 1 + count(scores > target_score); lower is better"
DEFAULT_CONFIG = (
    "attack/configs/"
    "diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10.yaml"
)
DEFAULT_OUTPUT_DIR = "outputs/analysis/target_action_feature_survey"


SUMMARY_COLUMNS = [
    "target_item",
    "train_target_occurrence_count",
    "train_target_session_count",
    "train_target_label_count",
    "valid_target_occurrence_count",
    "valid_target_session_count",
    "valid_target_label_count",
    "target_frequency_rank",
    "target_popularity_percentile",
    "target_density_bucket",
    "num_unique_predecessors",
    "num_unique_successors",
    "num_cooccurrence_items",
    "top1_predecessor_count",
    "top5_predecessor_count",
    "top1_predecessor_share",
    "top5_predecessor_share",
    "predecessor_entropy",
    "predecessor_effective_count",
    "top1_successor_count",
    "top5_successor_count",
    "top1_successor_share",
    "top5_successor_share",
    "successor_entropy",
    "successor_effective_count",
    "top1_cooccurrence_count",
    "top5_cooccurrence_count",
    "top1_cooccurrence_share",
    "top5_cooccurrence_share",
    "cooccurrence_entropy",
    "cooccurrence_effective_count",
    "top5_predecessor_items",
    "top5_successor_items",
    "top5_cooccurrence_items",
    "validation_prefix_count",
    "near_top_prefix_count",
    "near_top_prefix_ratio",
    "clean_target_rank_mean",
    "clean_target_rank_median",
    "clean_target_rank_q25",
    "clean_target_rank_q75",
    "clean_target_rank_min",
    "clean_target_rank_max",
    "near_top_rank_mean",
    "near_top_rank_median",
    "near_top_rank_q25",
    "near_top_rank_q75",
    "rank_1_20_count",
    "rank_21_50_count",
    "rank_51_100_count",
    "rank_101_200_count",
    "rank_above_200_count",
    "target_score_mean",
    "target_score_median",
    "margin_to_top20_mean",
    "margin_to_top20_median",
    "num_near_top_last_item_anchors",
    "top1_near_top_anchor_count",
    "top1_near_top_anchor_coverage",
    "top5_near_top_anchor_coverage",
    "top10_near_top_anchor_coverage",
    "top20_near_top_anchor_coverage",
    "near_top_anchor_entropy",
    "near_top_anchor_effective_count",
    "top5_near_top_anchor_items",
    "top20_near_top_anchor_items",
    "top5_near_top_anchor_train_predecessor_overlap_count",
    "top20_near_top_anchor_train_predecessor_overlap_count",
    "top20_near_top_anchor_train_predecessor_overlap_ratio",
    "fake_session_coverage_count_top20_near_top_anchors",
    "fake_session_coverage_ratio_top20_near_top_anchors",
    "fake_session_occurrence_count_top20_near_top_anchors",
    "fake_session_coverage_count_top20_train_predecessors",
    "fake_session_coverage_ratio_top20_train_predecessors",
    "fake_session_occurrence_count_top20_train_predecessors",
    "insertion_exposure_source",
    "insertion_metadata_available",
    "insertion_fake_session_count",
    "insertion_length_shift_min",
    "insertion_length_shift_max",
    "insertion_length_shift_mean",
    "insertion_tail_slot_ratio",
    "insertion_unique_left_item_count",
    "insertion_unique_right_item_count",
    "insertion_unique_left_right_pair_count",
    "insertion_candidate_unique_left_item_count",
    "insertion_candidate_unique_right_item_count",
    "insertion_candidate_unique_left_right_pair_count",
    "insertion_candidate_left_entropy",
    "insertion_candidate_right_entropy",
    "insertion_candidate_pair_entropy",
    "insertion_sampled_unique_left_item_count",
    "insertion_sampled_unique_right_item_count",
    "insertion_sampled_unique_left_right_pair_count",
    "insertion_every_target_has_left_neighbor",
    "insertion_every_target_has_right_neighbor",
    "insertion_left_entropy",
    "insertion_right_entropy",
    "insertion_pair_entropy",
    "insertion_left_overlap_count_with_top20_near_top_anchors",
    "insertion_left_overlap_ratio_relative_to_unique_left",
    "insertion_left_overlap_ratio_relative_to_top20_near_top_anchors",
    "replacement_exposure_source",
    "replacement_metadata_available",
    "replacement_fake_session_count",
    "replacement_length_shift_min",
    "replacement_length_shift_max",
    "replacement_length_shift_mean",
    "replacement_internal_replacement_count",
    "replacement_internal_replacement_ratio",
    "replacement_tail_fallback_count",
    "replacement_tail_fallback_ratio",
    "replacement_unique_left_item_count",
    "replacement_unique_right_item_count",
    "replacement_unique_left_right_pair_count",
    "replacement_candidate_unique_left_item_count",
    "replacement_candidate_unique_right_item_count",
    "replacement_candidate_unique_left_right_pair_count",
    "replacement_candidate_left_entropy",
    "replacement_candidate_right_entropy",
    "replacement_candidate_pair_entropy",
    "replacement_sampled_unique_left_item_count",
    "replacement_sampled_unique_right_item_count",
    "replacement_sampled_unique_left_right_pair_count",
    "replacement_every_internal_target_has_left_neighbor",
    "replacement_every_internal_target_has_right_neighbor",
    "replacement_every_target_has_left_neighbor",
    "replacement_every_target_has_right_neighbor",
    "replacement_pos1_ratio",
    "replacement_pos2_ratio",
    "replacement_pos3_ratio",
    "replacement_pos4_5_ratio",
    "replacement_pos6_plus_ratio",
    "replacement_tail_position_ratio",
    "replacement_left_entropy",
    "replacement_right_entropy",
    "replacement_pair_entropy",
    "replacement_left_overlap_count_with_top20_near_top_anchors",
    "replacement_left_overlap_ratio_relative_to_unique_left",
    "replacement_left_overlap_ratio_relative_to_top20_near_top_anchors",
]


SUMMARY_MD_COLUMNS = [
    "target_item",
    "train_target_occurrence_count",
    "target_frequency_rank",
    "target_popularity_percentile",
    "num_unique_predecessors",
    "predecessor_effective_count",
    "num_unique_successors",
    "successor_effective_count",
    "near_top_prefix_count",
    "near_top_prefix_ratio",
    "top1_near_top_anchor_coverage",
    "top5_near_top_anchor_coverage",
    "near_top_anchor_effective_count",
    "fake_session_coverage_ratio_top20_near_top_anchors",
    "insertion_exposure_source",
    "insertion_tail_slot_ratio",
    "insertion_candidate_unique_left_item_count",
    "insertion_candidate_unique_left_right_pair_count",
    "replacement_exposure_source",
    "replacement_tail_fallback_ratio",
    "replacement_candidate_unique_left_item_count",
    "replacement_candidate_unique_left_right_pair_count",
]


def _repo_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else Path.cwd() / path_obj


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Counter):
        return {str(key): int(count) for key, count in value.items()}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)


def _load_json_or_none(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path_obj = _repo_path(path)
    if not path_obj.exists():
        return None
    with path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(_to_jsonable(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _percentile(values: Sequence[float | int], q: float) -> float | None:
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


def _safe_get_nested(dct: Mapping[str, Any] | None, path: Sequence[str], default: Any = None) -> Any:
    current: Any = dct
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _shannon_entropy(counter: Mapping[Any, int]) -> float:
    total = sum(int(count) for count in counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        count = int(count)
        if count <= 0:
            continue
        p = float(count) / float(total)
        entropy -= p * math.log(p)
    return float(entropy)


def _effective_count(counter: Mapping[Any, int]) -> float:
    total = sum(int(count) for count in counter.values())
    if total <= 0:
        return 0.0
    return float(math.exp(_shannon_entropy(counter)))


def _top_counter(counter: Mapping[int, int], limit: int) -> list[dict[str, int]]:
    return [
        {"item": int(item), "count": int(count)}
        for item, count in Counter(counter).most_common(int(limit))
    ]


def _top_pair_counter(counter: Mapping[tuple[int, int], int], limit: int) -> list[dict[str, int]]:
    return [
        {"left_item": int(pair[0]), "right_item": int(pair[1]), "count": int(count)}
        for pair, count in Counter(counter).most_common(int(limit))
    ]


def _item_list(rows: Sequence[Mapping[str, Any]], key: str = "item") -> list[int]:
    return [int(row[key]) for row in rows]


def _top_items_string(rows: Sequence[Mapping[str, Any]], key: str = "item") -> str:
    return ";".join(str(int(row[key])) for row in rows)


def _coverage_for_top_k(counter: Mapping[Any, int], k: int, total: int | None = None) -> float:
    if total is None:
        total = sum(int(count) for count in counter.values())
    if not total:
        return 0.0
    return float(sum(int(count) for _, count in Counter(counter).most_common(int(k)))) / float(total)


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
        prefixes.extend([list(map(int, prefix)) for prefix in session_prefixes])
        labels.extend(int(label) for label in session_labels)
    return prefixes, labels


def _label_counts(sessions: Sequence[Sequence[int]]) -> Counter[int]:
    _, labels = expanded_cases(sessions)
    return Counter(int(label) for label in labels)


def _split_target_counts(
    sessions: Sequence[Sequence[int]],
    label_counts: Mapping[int, int],
    target: int,
) -> dict[str, int]:
    target = int(target)
    occurrence = 0
    session_count = 0
    for raw_session in sessions:
        session = [int(item) for item in raw_session]
        count = sum(1 for item in session if item == target)
        occurrence += count
        if count:
            session_count += 1
    return {
        "occurrence_count": int(occurrence),
        "session_count": int(session_count),
        "label_count": int(label_counts.get(target, 0)),
    }


def _density_bucket(occurrence: int, all_train_counts: Mapping[int, int]) -> tuple[str, dict[str, Any]]:
    values = list(int(value) for value in all_train_counts.values())
    q25 = _percentile(values, 0.25) or 0.0
    q75 = _percentile(values, 0.75) or 0.0
    if int(occurrence) <= q25:
        bucket = "sparse"
    elif int(occurrence) >= q75:
        bucket = "dense"
    else:
        bucket = "medium"
    return bucket, {
        "rule": "sparse if train occurrence <= all-train item q25; dense if >= q75; otherwise medium",
        "q25": q25,
        "q75": q75,
    }


def compute_target_popularity(
    *,
    train_sessions: Sequence[Sequence[int]],
    valid_sessions: Sequence[Sequence[int]],
    train_counts: Mapping[int, int],
    train_ranks: Mapping[int, int],
    train_label_counts: Mapping[int, int],
    valid_label_counts: Mapping[int, int],
    target: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = int(target)
    train = _split_target_counts(train_sessions, train_label_counts, target)
    valid = _split_target_counts(valid_sessions, valid_label_counts, target)
    rank = train_ranks.get(target)
    percentile = None
    if rank is not None and len(train_ranks) > 1:
        percentile = 1.0 - ((int(rank) - 1) / float(len(train_ranks) - 1))
    density, density_metadata = _density_bucket(train["occurrence_count"], train_counts)
    flat = {
        "target_item": target,
        "train_target_occurrence_count": train["occurrence_count"],
        "train_target_session_count": train["session_count"],
        "train_target_label_count": train["label_count"],
        "valid_target_occurrence_count": valid["occurrence_count"],
        "valid_target_session_count": valid["session_count"],
        "valid_target_label_count": valid["label_count"],
        "target_frequency_rank": None if rank is None else int(rank),
        "target_popularity_percentile": percentile,
        "target_density_bucket": density,
    }
    grouped = {
        "train_sub": train,
        "validation": valid,
        "target_frequency_rank": None if rank is None else int(rank),
        "target_popularity_percentile": percentile,
        "target_density_bucket": density,
        "density_bucket_metadata": density_metadata,
    }
    return grouped, flat


def _counter_distribution_profile(counter: Counter[int], *, prefix: str, top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    total = sum(int(count) for count in counter.values())
    top5 = _top_counter(counter, 5)
    top20 = _top_counter(counter, min(20, top_k))
    entropy = _shannon_entropy(counter)
    effective = _effective_count(counter)
    flat = {
        f"num_{prefix if prefix == 'cooccurrence_items' else 'unique_' + prefix}": int(len(counter)),
        f"top1_{prefix[:-1] if prefix.endswith('s') else prefix}_count": (
            int(counter.most_common(1)[0][1]) if counter else 0
        ),
        f"top5_{prefix[:-1] if prefix.endswith('s') else prefix}_count": (
            int(sum(count for _, count in counter.most_common(5)))
        ),
        f"top1_{prefix[:-1] if prefix.endswith('s') else prefix}_share": _coverage_for_top_k(counter, 1, total),
        f"top5_{prefix[:-1] if prefix.endswith('s') else prefix}_share": _coverage_for_top_k(counter, 5, total),
        f"{prefix[:-1] if prefix.endswith('s') else prefix}_entropy": entropy,
        f"{prefix[:-1] if prefix.endswith('s') else prefix}_effective_count": effective,
        f"top5_{prefix[:-1] if prefix.endswith('s') else prefix}_items": _top_items_string(top5),
    }
    grouped = {
        "total_count": int(total),
        "num_unique_items": int(len(counter)),
        "top_items": _top_counter(counter, top_k),
        "top5_items": top5,
        "top20_items": top20,
        "entropy": entropy,
        "effective_count": effective,
        "empty_counter_behavior": "count/share/entropy/effective_count are 0 when no observations exist",
    }
    return grouped, flat


def compute_natural_transition_profile(
    sessions: Sequence[Sequence[int]],
    target: int,
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = int(target)
    predecessors: Counter[int] = Counter()
    successors: Counter[int] = Counter()
    cooccurrence: Counter[int] = Counter()
    for raw_session in sessions:
        session = [int(item) for item in raw_session]
        if target in session:
            session_counts = Counter(session)
            for item, count in session_counts.items():
                if int(item) != target:
                    cooccurrence[int(item)] += int(count)
        for index, item in enumerate(session):
            if int(item) != target:
                continue
            if index > 0:
                predecessors[int(session[index - 1])] += 1
            if index < len(session) - 1:
                successors[int(session[index + 1])] += 1

    pred_group, pred_flat_raw = _counter_distribution_profile(predecessors, prefix="predecessors", top_k=top_k)
    succ_group, succ_flat_raw = _counter_distribution_profile(successors, prefix="successors", top_k=top_k)
    co_group, co_flat_raw = _counter_distribution_profile(cooccurrence, prefix="cooccurrence_items", top_k=top_k)
    flat = {
        "num_unique_predecessors": pred_flat_raw["num_unique_predecessors"],
        "num_unique_successors": succ_flat_raw["num_unique_successors"],
        "num_cooccurrence_items": co_flat_raw["num_cooccurrence_items"],
        "top1_predecessor_count": pred_flat_raw["top1_predecessor_count"],
        "top5_predecessor_count": pred_flat_raw["top5_predecessor_count"],
        "top1_predecessor_share": pred_flat_raw["top1_predecessor_share"],
        "top5_predecessor_share": pred_flat_raw["top5_predecessor_share"],
        "predecessor_entropy": pred_flat_raw["predecessor_entropy"],
        "predecessor_effective_count": pred_flat_raw["predecessor_effective_count"],
        "top1_successor_count": succ_flat_raw["top1_successor_count"],
        "top5_successor_count": succ_flat_raw["top5_successor_count"],
        "top1_successor_share": succ_flat_raw["top1_successor_share"],
        "top5_successor_share": succ_flat_raw["top5_successor_share"],
        "successor_entropy": succ_flat_raw["successor_entropy"],
        "successor_effective_count": succ_flat_raw["successor_effective_count"],
        "top1_cooccurrence_count": co_flat_raw["top1_cooccurrence_item_count"],
        "top5_cooccurrence_count": co_flat_raw["top5_cooccurrence_item_count"],
        "top1_cooccurrence_share": co_flat_raw["top1_cooccurrence_item_share"],
        "top5_cooccurrence_share": co_flat_raw["top5_cooccurrence_item_share"],
        "cooccurrence_entropy": co_flat_raw["cooccurrence_item_entropy"],
        "cooccurrence_effective_count": co_flat_raw["cooccurrence_item_effective_count"],
        "top5_predecessor_items": pred_flat_raw["top5_predecessor_items"],
        "top5_successor_items": succ_flat_raw["top5_successor_items"],
        "top5_cooccurrence_items": co_flat_raw["top5_cooccurrence_item_items"],
    }
    grouped = {
        "predecessors": pred_group,
        "successors": succ_group,
        "cooccurrence": co_group,
        "predecessor_counts": predecessors,
        "successor_counts": successors,
        "cooccurrence_counts": cooccurrence,
    }
    return grouped, flat


def near_top_indices(
    ranks: Sequence[int],
    *,
    rank_min: int,
    rank_max: int,
) -> list[int]:
    return [
        index
        for index, rank in enumerate(ranks)
        if int(rank_min) <= int(rank) <= int(rank_max)
    ]


def resolve_clean_poison_checkpoint(config: Config) -> Path | None:
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


def score_validation_prefixes(
    config: Config,
    *,
    checkpoint_path: Path,
    prefixes: Sequence[Sequence[int]],
    targets: Sequence[int],
) -> dict[int, dict[str, list[float | int | None]]]:
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
        int(target): {"ranks": [], "scores": [], "margin_to_top20": []}
        for target in targets_list
    }
    target_tensor = None
    runner.model.eval()
    with torch.no_grad():
        for batch_indices in data.generate_batch(runner.model.batch_size):
            _, scores = srg_forward(runner.model, batch_indices, data)
            if target_tensor is None or target_tensor.device != scores.device:
                target_tensor = torch.as_tensor(
                    [target - 1 for target in targets_list],
                    dtype=torch.long,
                    device=scores.device,
                )
            top20_cutoff = None
            if scores.shape[1] >= 20:
                top20_cutoff = torch.topk(scores, k=20, dim=1).values[:, -1]
            for target_position, target in enumerate(targets_list):
                item_index = target_tensor[target_position]
                target_scores = scores[:, item_index]
                ranks = 1 + torch.sum(scores > target_scores.unsqueeze(1), dim=1)
                output[target]["ranks"].extend(
                    int(value) for value in trans_to_cpu(ranks).tolist()
                )
                output[target]["scores"].extend(
                    float(value) for value in trans_to_cpu(target_scores).tolist()
                )
                if top20_cutoff is None:
                    output[target]["margin_to_top20"].extend([None] * int(scores.shape[0]))
                else:
                    margins = top20_cutoff - target_scores
                    output[target]["margin_to_top20"].extend(
                        float(value) for value in trans_to_cpu(margins).tolist()
                    )
    return output


def compute_clean_rank_susceptibility_profile(
    ranks: Sequence[int],
    scores: Sequence[float],
    margins: Sequence[float | None],
    *,
    rank_min: int,
    rank_max: int,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    ranks_int = [int(rank) for rank in ranks]
    scores_float = [float(score) for score in scores]
    margin_values = [float(value) for value in margins if value is not None]
    near_indices = near_top_indices(ranks_int, rank_min=rank_min, rank_max=rank_max)
    near_ranks = [ranks_int[index] for index in near_indices]
    rank_summary = _summary(ranks_int)
    near_summary = _summary(near_ranks)
    score_summary = _summary(scores_float)
    margin_summary = _summary(margin_values)
    flat = {
        "validation_prefix_count": int(len(ranks_int)),
        "near_top_prefix_count": int(len(near_indices)),
        "near_top_prefix_ratio": float(len(near_indices)) / float(len(ranks_int)) if ranks_int else 0.0,
        "clean_target_rank_mean": rank_summary["mean"],
        "clean_target_rank_median": rank_summary["median"],
        "clean_target_rank_q25": rank_summary["q25"],
        "clean_target_rank_q75": rank_summary["q75"],
        "clean_target_rank_min": rank_summary["min"],
        "clean_target_rank_max": rank_summary["max"],
        "near_top_rank_mean": near_summary["mean"],
        "near_top_rank_median": near_summary["median"],
        "near_top_rank_q25": near_summary["q25"],
        "near_top_rank_q75": near_summary["q75"],
        "rank_1_20_count": sum(1 for rank in ranks_int if 1 <= rank <= 20),
        "rank_21_50_count": sum(1 for rank in ranks_int if 21 <= rank <= 50),
        "rank_51_100_count": sum(1 for rank in ranks_int if 51 <= rank <= 100),
        "rank_101_200_count": sum(1 for rank in ranks_int if 101 <= rank <= 200),
        "rank_above_200_count": sum(1 for rank in ranks_int if rank > 200),
        "target_score_mean": score_summary["mean"],
        "target_score_median": score_summary["median"],
        "margin_to_top20_mean": margin_summary["mean"],
        "margin_to_top20_median": margin_summary["median"],
    }
    grouped = {
        "rank_convention": RANK_CONVENTION,
        "near_top_definition": {
            "rank_min": int(rank_min),
            "rank_max": int(rank_max),
            "description": (
                "validation prefixes where the target is not in top-20 but is "
                "within top-200 under the clean model when defaults are used"
            ),
        },
        "validation_prefix_count": flat["validation_prefix_count"],
        "near_top_prefix_count": flat["near_top_prefix_count"],
        "near_top_prefix_ratio": flat["near_top_prefix_ratio"],
        "rank_summary": rank_summary,
        "near_top_rank_summary": near_summary,
        "target_score_summary": score_summary,
        "margin_to_top20_summary": margin_summary,
        "rank_buckets": {
            "rank_1_20_count": flat["rank_1_20_count"],
            "rank_21_50_count": flat["rank_21_50_count"],
            "rank_51_100_count": flat["rank_51_100_count"],
            "rank_101_200_count": flat["rank_101_200_count"],
            "rank_above_200_count": flat["rank_above_200_count"],
        },
    }
    return grouped, flat, near_indices


def compute_near_top_anchor_concentration(
    prefixes: Sequence[Sequence[int]],
    near_indices: Sequence[int],
    train_predecessor_counts: Mapping[int, int],
    *,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
    anchor_counts: Counter[int] = Counter()
    for index in near_indices:
        prefix = [int(item) for item in prefixes[index]]
        if prefix:
            anchor_counts[int(prefix[-1])] += 1
    near_count = len(near_indices)
    top5 = _top_counter(anchor_counts, 5)
    top20 = _top_counter(anchor_counts, 20)
    top20_items = _item_list(top20)
    train_predecessor_set = {int(item) for item in train_predecessor_counts}
    top5_overlap = len(set(_item_list(top5)) & train_predecessor_set)
    top20_overlap = len(set(top20_items) & train_predecessor_set)
    flat = {
        "num_near_top_last_item_anchors": int(len(anchor_counts)),
        "top1_near_top_anchor_count": int(anchor_counts.most_common(1)[0][1]) if anchor_counts else 0,
        "top1_near_top_anchor_coverage": _coverage_for_top_k(anchor_counts, 1, near_count),
        "top5_near_top_anchor_coverage": _coverage_for_top_k(anchor_counts, 5, near_count),
        "top10_near_top_anchor_coverage": _coverage_for_top_k(anchor_counts, 10, near_count),
        "top20_near_top_anchor_coverage": _coverage_for_top_k(anchor_counts, 20, near_count),
        "near_top_anchor_entropy": _shannon_entropy(anchor_counts),
        "near_top_anchor_effective_count": _effective_count(anchor_counts),
        "top5_near_top_anchor_items": _top_items_string(top5),
        "top20_near_top_anchor_items": _top_items_string(top20),
        "top5_near_top_anchor_train_predecessor_overlap_count": int(top5_overlap),
        "top20_near_top_anchor_train_predecessor_overlap_count": int(top20_overlap),
        "top20_near_top_anchor_train_predecessor_overlap_ratio": (
            float(top20_overlap) / float(len(top20_items)) if top20_items else 0.0
        ),
    }
    grouped = {
        "definition": "last item of a near-top validation context",
        "near_top_prefix_count": int(near_count),
        "anchor_counts": anchor_counts,
        "top_anchors": _top_counter(anchor_counts, top_k),
        "top20_anchor_items": top20_items,
        "entropy": flat["near_top_anchor_entropy"],
        "effective_count": flat["near_top_anchor_effective_count"],
        "train_predecessor_overlap": {
            "top5_count": top5_overlap,
            "top20_count": top20_overlap,
            "top20_ratio": flat["top20_near_top_anchor_train_predecessor_overlap_ratio"],
        },
    }
    return grouped, flat, top20_items


def fake_session_item_availability(
    fake_sessions: Sequence[Sequence[int]] | None,
    items: Iterable[int],
) -> dict[str, Any]:
    item_set = {int(item) for item in items}
    if fake_sessions is None:
        return {
            "available": False,
            "coverage_count": None,
            "coverage_ratio": None,
            "occurrence_count": None,
            "item_count": int(len(item_set)),
        }
    item_occurrences: Counter[int] = Counter()
    for raw_session in fake_sessions:
        for item in raw_session:
            item = int(item)
            if item in item_set:
                item_occurrences[item] += 1
    coverage_count = sum(1 for item in item_set if item_occurrences.get(item, 0) > 0)
    return {
        "available": True,
        "coverage_count": int(coverage_count),
        "coverage_ratio": float(coverage_count) / float(len(item_set)) if item_set else 0.0,
        "occurrence_count": int(sum(item_occurrences.values())),
        "item_count": int(len(item_set)),
        "covered_items": sorted(int(item) for item in item_set if item_occurrences.get(item, 0) > 0),
    }


def compute_fake_session_availability(
    fake_sessions: Sequence[Sequence[int]] | None,
    *,
    fake_sessions_path: Path | None,
    top20_near_top_anchors: Sequence[int],
    top20_train_predecessors: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor_payload = fake_session_item_availability(fake_sessions, top20_near_top_anchors)
    predecessor_payload = fake_session_item_availability(fake_sessions, top20_train_predecessors)
    flat = {
        "fake_session_coverage_count_top20_near_top_anchors": anchor_payload["coverage_count"],
        "fake_session_coverage_ratio_top20_near_top_anchors": anchor_payload["coverage_ratio"],
        "fake_session_occurrence_count_top20_near_top_anchors": anchor_payload["occurrence_count"],
        "fake_session_coverage_count_top20_train_predecessors": predecessor_payload["coverage_count"],
        "fake_session_coverage_ratio_top20_train_predecessors": predecessor_payload["coverage_ratio"],
        "fake_session_occurrence_count_top20_train_predecessors": predecessor_payload["occurrence_count"],
    }
    grouped = {
        "available": fake_sessions is not None,
        "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
        "total_fake_sessions": None if fake_sessions is None else int(len(fake_sessions)),
        "top20_near_top_anchors": anchor_payload,
        "top20_train_predecessors": predecessor_payload,
        "coverage_definition": "coverage_count is the number of top20 items appearing at least once in fake sessions; coverage_ratio divides by the number of top20 items",
    }
    return grouped, flat


def _normalize_item_counter(value: Any) -> Counter[int] | None:
    if not isinstance(value, Mapping):
        return None
    counter: Counter[int] = Counter()
    for raw_key, raw_count in value.items():
        try:
            counter[int(raw_key)] += int(raw_count)
        except (TypeError, ValueError):
            return None
    return counter


def _parse_pair_key(raw_key: Any) -> tuple[int, int] | None:
    if isinstance(raw_key, (list, tuple)) and len(raw_key) == 2:
        return int(raw_key[0]), int(raw_key[1])
    text = str(raw_key)
    for separator in (",", "|", ":", "_"):
        if separator in text:
            left, right = text.split(separator, 1)
            return int(left.strip()), int(right.strip())
    return None


def _normalize_pair_counter(value: Any) -> Counter[tuple[int, int]] | None:
    if not isinstance(value, Mapping):
        return None
    counter: Counter[tuple[int, int]] = Counter()
    for raw_key, raw_count in value.items():
        try:
            pair = _parse_pair_key(raw_key)
            if pair is None:
                return None
            counter[pair] += int(raw_count)
        except (TypeError, ValueError):
            return None
    return counter


def _first_counter(metadata: Mapping[str, Any], keys: Sequence[str], *, pair: bool = False) -> Counter[Any] | None:
    for key in keys:
        value = metadata.get(key)
        counter = _normalize_pair_counter(value) if pair else _normalize_item_counter(value)
        if counter is not None:
            return counter
    return None


def _full_records_from_metadata(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]] | None:
    fake_session_count = metadata.get("fake_session_count")
    try:
        expected = int(fake_session_count)
    except (TypeError, ValueError):
        return None
    for key in ("records", "session_records", "operation_records", "per_session_records", "previews"):
        value = metadata.get(key)
        if isinstance(value, list) and len(value) == expected and all(isinstance(row, Mapping) for row in value):
            return value
    return None


def _extract_preview_item_sets(metadata: Mapping[str, Any], operation_type: str) -> dict[str, Counter[Any] | None]:
    left = _first_counter(
        metadata,
        ("left_item_counts", "left_counts", "left_item_count_distribution"),
    )
    right = _first_counter(
        metadata,
        ("right_item_counts", "right_counts", "right_item_count_distribution"),
    )
    pair = _first_counter(
        metadata,
        ("left_right_pair_counts", "pair_counts", "left_right_counts"),
        pair=True,
    )
    if left is not None or right is not None or pair is not None:
        return {"left": left, "right": right, "pair": pair}

    records = _full_records_from_metadata(metadata)
    if records is None:
        return {"left": None, "right": None, "pair": None}
    left_counter: Counter[int] = Counter()
    right_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[int, int]] = Counter()
    for record in records:
        left_item = record.get("left_item")
        right_item = record.get("right_item")
        if left_item is not None:
            left_counter[int(left_item)] += 1
        if right_item is not None:
            right_counter[int(right_item)] += 1
        if left_item is not None and right_item is not None:
            pair_counter[(int(left_item), int(right_item))] += 1
    return {
        "left": left_counter,
        "right": right_counter,
        "pair": pair_counter if operation_type in {"insertion", "replacement"} else None,
    }


def _flatten_length_shift_summary(prefix: str, metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary = metadata.get("length_shift_summary")
    if not isinstance(summary, Mapping):
        return {
            f"{prefix}_length_shift_min": None,
            f"{prefix}_length_shift_max": None,
            f"{prefix}_length_shift_mean": None,
        }
    return {
        f"{prefix}_length_shift_min": summary.get("min"),
        f"{prefix}_length_shift_max": summary.get("max"),
        f"{prefix}_length_shift_mean": summary.get("mean"),
    }


def _extract_position_ratio(metadata: Mapping[str, Any], field: str) -> Any:
    group_ratios = metadata.get("replacement_position_group_ratios")
    if isinstance(group_ratios, Mapping) and field in group_ratios:
        return group_ratios[field]
    ratios = metadata.get("replacement_position_ratios")
    if not isinstance(ratios, Mapping):
        return None
    if field == "pos4_5":
        return float(ratios.get("4", 0.0) or 0.0) + float(ratios.get("5", 0.0) or 0.0)
    if field == "pos6_plus":
        total = 0.0
        for key, value in ratios.items():
            try:
                if int(key) >= 6:
                    total += float(value)
            except (TypeError, ValueError):
                continue
        return total
    if field == "tail_position":
        return metadata.get("tail_position_ratio", metadata.get("tail_fallback_ratio"))
    if field.startswith("pos"):
        return ratios.get(field[3:])
    return None


def _insertion_tail_slot_ratio(metadata: Mapping[str, Any]) -> Any:
    # For Internal Random Insertion-NZ, "tail slot" means appending after the
    # final original item. The method's valid slots are 1..L-1, so appending at
    # slot L is not part of the action space.
    records = _full_records_from_metadata(metadata)
    if records is None:
        return 0.0
    total = 0
    append_tail = 0
    for record in records:
        if record.get("insertion_slot") is None or record.get("original_length") is None:
            return None
        slot = int(record["insertion_slot"])
        original_length = int(record["original_length"])
        total += 1
        if slot == original_length:
            append_tail += 1
    return float(append_tail) / float(total) if total else 0.0


def _replacement_position_ratio_from_complete_records(
    metadata: Mapping[str, Any],
    field: str,
) -> Any:
    extracted = _extract_position_ratio(metadata, field)
    if extracted is not None:
        return extracted
    records = _full_records_from_metadata(metadata)
    if records is None:
        return None
    counts: Counter[str] = Counter()
    for record in records:
        if record.get("replacement_position") is None:
            return None
        if bool(record.get("used_tail_fallback", False)):
            group = "tail_position"
        else:
            group = _position_group(int(record["replacement_position"]))
        counts[group] += 1
    total = sum(counts.values())
    return float(counts[field]) / float(total) if total else 0.0


def _metadata_base(prefix: str, *, source: str, metadata_available: bool) -> dict[str, Any]:
    return {
        f"{prefix}_exposure_source": source,
        f"{prefix}_metadata_available": bool(metadata_available),
        f"{prefix}_fake_session_count": None,
        f"{prefix}_length_shift_min": None,
        f"{prefix}_length_shift_max": None,
        f"{prefix}_length_shift_mean": None,
        f"{prefix}_unique_left_item_count": None,
        f"{prefix}_unique_right_item_count": None,
        f"{prefix}_unique_left_right_pair_count": None,
        f"{prefix}_candidate_unique_left_item_count": None,
        f"{prefix}_candidate_unique_right_item_count": None,
        f"{prefix}_candidate_unique_left_right_pair_count": None,
        f"{prefix}_candidate_left_entropy": None,
        f"{prefix}_candidate_right_entropy": None,
        f"{prefix}_candidate_pair_entropy": None,
        f"{prefix}_sampled_unique_left_item_count": None,
        f"{prefix}_sampled_unique_right_item_count": None,
        f"{prefix}_sampled_unique_left_right_pair_count": None,
        f"{prefix}_left_entropy": None,
        f"{prefix}_right_entropy": None,
        f"{prefix}_pair_entropy": None,
        f"{prefix}_left_overlap_count_with_top20_near_top_anchors": None,
        f"{prefix}_left_overlap_ratio_relative_to_unique_left": None,
        f"{prefix}_left_overlap_ratio_relative_to_top20_near_top_anchors": None,
    }


def _counter_entropy_or_none(counter: Counter[Any] | None) -> float | None:
    return None if counter is None else _shannon_entropy(counter)


def _overlap_from_left_counter(
    left_counter: Counter[int] | None,
    top20_near_top_anchors: Sequence[int],
) -> dict[str, Any]:
    if left_counter is None:
        return {
            "count": None,
            "ratio_unique_left": None,
            "ratio_top20_near_top_anchors": None,
        }
    left_set = {int(item) for item in left_counter}
    anchor_set = {int(item) for item in top20_near_top_anchors}
    overlap_count = len(left_set & anchor_set)
    return {
        "count": int(overlap_count),
        "ratio_unique_left": float(overlap_count) / float(len(left_set)) if left_set else 0.0,
        "ratio_top20_near_top_anchors": (
            float(overlap_count) / float(len(anchor_set)) if anchor_set else 0.0
        ),
    }


def _exposure_counts_from_counters(
    metadata: Mapping[str, Any],
    operation_type: str,
) -> tuple[Counter[int] | None, Counter[int] | None, Counter[tuple[int, int]] | None]:
    counters = _extract_preview_item_sets(metadata, operation_type)
    return counters["left"], counters["right"], counters["pair"]


def parse_insertion_exposure_metadata(
    metadata: Mapping[str, Any],
    *,
    top20_near_top_anchors: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_counter, right_counter, pair_counter = _exposure_counts_from_counters(metadata, "insertion")
    overlap = _overlap_from_left_counter(left_counter, top20_near_top_anchors)
    flat = _metadata_base("insertion", source="metadata", metadata_available=True)
    flat.update(_flatten_length_shift_summary("insertion", metadata))
    flat.update(
        {
            "insertion_fake_session_count": metadata.get("fake_session_count"),
            "insertion_tail_slot_ratio": _insertion_tail_slot_ratio(metadata),
            "insertion_unique_left_item_count": metadata.get(
                "unique_left_item_count",
                len(left_counter) if left_counter is not None else None,
            ),
            "insertion_unique_right_item_count": metadata.get(
                "unique_right_item_count",
                len(right_counter) if right_counter is not None else None,
            ),
            "insertion_unique_left_right_pair_count": metadata.get(
                "unique_left_right_pair_count",
                len(pair_counter) if pair_counter is not None else None,
            ),
            "insertion_sampled_unique_left_item_count": metadata.get(
                "unique_left_item_count",
                len(left_counter) if left_counter is not None else None,
            ),
            "insertion_sampled_unique_right_item_count": metadata.get(
                "unique_right_item_count",
                len(right_counter) if right_counter is not None else None,
            ),
            "insertion_sampled_unique_left_right_pair_count": metadata.get(
                "unique_left_right_pair_count",
                len(pair_counter) if pair_counter is not None else None,
            ),
            "insertion_every_target_has_left_neighbor": metadata.get(
                "every_inserted_target_has_left_neighbor"
            ),
            "insertion_every_target_has_right_neighbor": metadata.get(
                "every_inserted_target_has_right_neighbor"
            ),
            "insertion_left_entropy": _counter_entropy_or_none(left_counter),
            "insertion_right_entropy": _counter_entropy_or_none(right_counter),
            "insertion_pair_entropy": _counter_entropy_or_none(pair_counter),
            "insertion_left_overlap_count_with_top20_near_top_anchors": overlap["count"],
            "insertion_left_overlap_ratio_relative_to_unique_left": overlap["ratio_unique_left"],
            "insertion_left_overlap_ratio_relative_to_top20_near_top_anchors": overlap[
                "ratio_top20_near_top_anchors"
            ],
        }
    )
    grouped = {
        "available": True,
        "source": "metadata",
        "metadata_fields_used_for_entropy_and_overlap": left_counter is not None,
        "flat": flat,
        "top_left_items": [] if left_counter is None else _top_counter(left_counter, 20),
        "top_right_items": [] if right_counter is None else _top_counter(right_counter, 20),
        "top_left_right_pairs": [] if pair_counter is None else _top_pair_counter(pair_counter, 20),
    }
    return grouped, flat


def parse_replacement_exposure_metadata(
    metadata: Mapping[str, Any],
    *,
    top20_near_top_anchors: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_counter, right_counter, pair_counter = _exposure_counts_from_counters(metadata, "replacement")
    overlap = _overlap_from_left_counter(left_counter, top20_near_top_anchors)
    flat = _metadata_base("replacement", source="metadata", metadata_available=True)
    flat.update(_flatten_length_shift_summary("replacement", metadata))
    flat.update(
        {
            "replacement_fake_session_count": metadata.get("fake_session_count"),
            "replacement_internal_replacement_count": metadata.get("internal_replacement_count"),
            "replacement_internal_replacement_ratio": metadata.get("internal_replacement_ratio"),
            "replacement_tail_fallback_count": metadata.get("tail_fallback_count"),
            "replacement_tail_fallback_ratio": metadata.get("tail_fallback_ratio"),
            "replacement_unique_left_item_count": metadata.get(
                "unique_left_item_count",
                len(left_counter) if left_counter is not None else None,
            ),
            "replacement_unique_right_item_count": metadata.get(
                "unique_right_item_count",
                len(right_counter) if right_counter is not None else None,
            ),
            "replacement_unique_left_right_pair_count": metadata.get(
                "unique_left_right_pair_count",
                len(pair_counter) if pair_counter is not None else None,
            ),
            "replacement_sampled_unique_left_item_count": metadata.get(
                "unique_left_item_count",
                len(left_counter) if left_counter is not None else None,
            ),
            "replacement_sampled_unique_right_item_count": metadata.get(
                "unique_right_item_count",
                len(right_counter) if right_counter is not None else None,
            ),
            "replacement_sampled_unique_left_right_pair_count": metadata.get(
                "unique_left_right_pair_count",
                len(pair_counter) if pair_counter is not None else None,
            ),
            "replacement_every_internal_target_has_left_neighbor": metadata.get(
                "every_internal_replaced_target_has_left_neighbor"
            ),
            "replacement_every_internal_target_has_right_neighbor": metadata.get(
                "every_internal_replaced_target_has_right_neighbor"
            ),
            "replacement_every_target_has_left_neighbor": metadata.get(
                "every_replaced_target_has_left_neighbor"
            ),
            "replacement_every_target_has_right_neighbor": metadata.get(
                "every_replaced_target_has_right_neighbor"
            ),
            "replacement_pos1_ratio": _replacement_position_ratio_from_complete_records(metadata, "pos1"),
            "replacement_pos2_ratio": _replacement_position_ratio_from_complete_records(metadata, "pos2"),
            "replacement_pos3_ratio": _replacement_position_ratio_from_complete_records(metadata, "pos3"),
            "replacement_pos4_5_ratio": _replacement_position_ratio_from_complete_records(metadata, "pos4_5"),
            "replacement_pos6_plus_ratio": _replacement_position_ratio_from_complete_records(metadata, "pos6_plus"),
            "replacement_tail_position_ratio": _replacement_position_ratio_from_complete_records(metadata, "tail_position"),
            "replacement_left_entropy": _counter_entropy_or_none(left_counter),
            "replacement_right_entropy": _counter_entropy_or_none(right_counter),
            "replacement_pair_entropy": _counter_entropy_or_none(pair_counter),
            "replacement_left_overlap_count_with_top20_near_top_anchors": overlap["count"],
            "replacement_left_overlap_ratio_relative_to_unique_left": overlap["ratio_unique_left"],
            "replacement_left_overlap_ratio_relative_to_top20_near_top_anchors": overlap[
                "ratio_top20_near_top_anchors"
            ],
        }
    )
    grouped = {
        "available": True,
        "source": "metadata",
        "metadata_fields_used_for_entropy_and_overlap": left_counter is not None,
        "flat": flat,
        "top_left_items": [] if left_counter is None else _top_counter(left_counter, 20),
        "top_right_items": [] if right_counter is None else _top_counter(right_counter, 20),
        "top_left_right_pairs": [] if pair_counter is None else _top_pair_counter(pair_counter, 20),
    }
    return grouped, flat


def simulate_insertion_exposure(
    fake_sessions: Sequence[Sequence[int]] | None,
    *,
    top20_near_top_anchors: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if fake_sessions is None:
        flat = _metadata_base("insertion", source="missing", metadata_available=False)
        flat.update(
            {
                "insertion_tail_slot_ratio": None,
                "insertion_every_target_has_left_neighbor": None,
                "insertion_every_target_has_right_neighbor": None,
            }
        )
        return {"available": False, "source": "missing", "flat": flat}, flat
    left_counter: Counter[int] = Counter()
    right_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[int, int]] = Counter()
    valid_slot_count = 0
    for raw_session in fake_sessions:
        session = [int(item) for item in raw_session]
        if len(session) < 2:
            continue
        for slot in range(1, len(session)):
            left = int(session[slot - 1])
            right = int(session[slot])
            left_counter[left] += 1
            right_counter[right] += 1
            pair_counter[(left, right)] += 1
            valid_slot_count += 1
    overlap = _overlap_from_left_counter(left_counter, top20_near_top_anchors)
    flat = _metadata_base("insertion", source="simulated_from_fake_sessions", metadata_available=False)
    flat.update(
        {
            "insertion_fake_session_count": int(len(fake_sessions)),
            "insertion_length_shift_min": 1.0,
            "insertion_length_shift_max": 1.0,
            "insertion_length_shift_mean": 1.0,
            "insertion_tail_slot_ratio": 0.0,
            "insertion_unique_left_item_count": int(len(left_counter)),
            "insertion_unique_right_item_count": int(len(right_counter)),
            "insertion_unique_left_right_pair_count": int(len(pair_counter)),
            "insertion_candidate_unique_left_item_count": int(len(left_counter)),
            "insertion_candidate_unique_right_item_count": int(len(right_counter)),
            "insertion_candidate_unique_left_right_pair_count": int(len(pair_counter)),
            "insertion_every_target_has_left_neighbor": bool(valid_slot_count > 0),
            "insertion_every_target_has_right_neighbor": bool(valid_slot_count > 0),
            "insertion_left_entropy": _shannon_entropy(left_counter),
            "insertion_right_entropy": _shannon_entropy(right_counter),
            "insertion_pair_entropy": _shannon_entropy(pair_counter),
            "insertion_candidate_left_entropy": _shannon_entropy(left_counter),
            "insertion_candidate_right_entropy": _shannon_entropy(right_counter),
            "insertion_candidate_pair_entropy": _shannon_entropy(pair_counter),
            "insertion_left_overlap_count_with_top20_near_top_anchors": overlap["count"],
            "insertion_left_overlap_ratio_relative_to_unique_left": overlap["ratio_unique_left"],
            "insertion_left_overlap_ratio_relative_to_top20_near_top_anchors": overlap[
                "ratio_top20_near_top_anchors"
            ],
        }
    )
    grouped = {
        "available": True,
        "source": "simulated_from_fake_sessions",
        "candidate_space_definition": "valid insertion slots are 1..L-1, excluding prepend slot 0 and append-after-tail slot L",
        "valid_slot_count": int(valid_slot_count),
        "flat": flat,
        "top_left_items": _top_counter(left_counter, 20),
        "top_right_items": _top_counter(right_counter, 20),
        "top_left_right_pairs": _top_pair_counter(pair_counter, 20),
    }
    return grouped, flat


def _position_group(position: int) -> str:
    if position == 1:
        return "pos1"
    if position == 2:
        return "pos2"
    if position == 3:
        return "pos3"
    if position in {4, 5}:
        return "pos4_5"
    return "pos6_plus"


def simulate_replacement_exposure(
    fake_sessions: Sequence[Sequence[int]] | None,
    *,
    top20_near_top_anchors: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if fake_sessions is None:
        flat = _metadata_base("replacement", source="missing", metadata_available=False)
        flat.update(
            {
                "replacement_internal_replacement_count": None,
                "replacement_internal_replacement_ratio": None,
                "replacement_tail_fallback_count": None,
                "replacement_tail_fallback_ratio": None,
                "replacement_every_internal_target_has_left_neighbor": None,
                "replacement_every_internal_target_has_right_neighbor": None,
                "replacement_every_target_has_left_neighbor": None,
                "replacement_every_target_has_right_neighbor": None,
                "replacement_pos1_ratio": None,
                "replacement_pos2_ratio": None,
                "replacement_pos3_ratio": None,
                "replacement_pos4_5_ratio": None,
                "replacement_pos6_plus_ratio": None,
                "replacement_tail_position_ratio": None,
            }
        )
        return {"available": False, "source": "missing", "flat": flat}, flat
    left_counter: Counter[int] = Counter()
    right_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[int, int]] = Counter()
    group_counts: Counter[str] = Counter()
    tail_fallback_count = 0
    internal_session_count = 0
    valid_position_count = 0
    for raw_session in fake_sessions:
        session = [int(item) for item in raw_session]
        if len(session) < 2:
            continue
        if len(session) >= 3:
            positions = list(range(1, len(session) - 1))
            internal_session_count += 1
            tail_fallback = False
        else:
            positions = [1]
            tail_fallback = True
            tail_fallback_count += 1
        for position in positions:
            left = int(session[position - 1])
            right = None if tail_fallback else int(session[position + 1])
            left_counter[left] += 1
            if right is not None:
                right_counter[right] += 1
                pair_counter[(left, right)] += 1
            group_counts["tail_position" if tail_fallback else _position_group(position)] += 1
            valid_position_count += 1
    overlap = _overlap_from_left_counter(left_counter, top20_near_top_anchors)
    denominator = float(valid_position_count) if valid_position_count else 0.0
    flat = _metadata_base("replacement", source="simulated_from_fake_sessions", metadata_available=False)
    flat.update(
        {
            "replacement_fake_session_count": int(len(fake_sessions)),
            "replacement_length_shift_min": 0.0,
            "replacement_length_shift_max": 0.0,
            "replacement_length_shift_mean": 0.0,
            "replacement_internal_replacement_count": int(internal_session_count),
            "replacement_internal_replacement_ratio": (
                float(internal_session_count) / float(len(fake_sessions)) if fake_sessions else 0.0
            ),
            "replacement_tail_fallback_count": int(tail_fallback_count),
            "replacement_tail_fallback_ratio": (
                float(tail_fallback_count) / float(len(fake_sessions)) if fake_sessions else 0.0
            ),
            "replacement_unique_left_item_count": int(len(left_counter)),
            "replacement_unique_right_item_count": int(len(right_counter)),
            "replacement_unique_left_right_pair_count": int(len(pair_counter)),
            "replacement_candidate_unique_left_item_count": int(len(left_counter)),
            "replacement_candidate_unique_right_item_count": int(len(right_counter)),
            "replacement_candidate_unique_left_right_pair_count": int(len(pair_counter)),
            "replacement_every_internal_target_has_left_neighbor": bool(internal_session_count > 0),
            "replacement_every_internal_target_has_right_neighbor": bool(internal_session_count > 0),
            "replacement_every_target_has_left_neighbor": bool(valid_position_count > 0),
            "replacement_every_target_has_right_neighbor": bool(tail_fallback_count == 0 and valid_position_count > 0),
            "replacement_pos1_ratio": float(group_counts["pos1"]) / denominator if denominator else 0.0,
            "replacement_pos2_ratio": float(group_counts["pos2"]) / denominator if denominator else 0.0,
            "replacement_pos3_ratio": float(group_counts["pos3"]) / denominator if denominator else 0.0,
            "replacement_pos4_5_ratio": float(group_counts["pos4_5"]) / denominator if denominator else 0.0,
            "replacement_pos6_plus_ratio": float(group_counts["pos6_plus"]) / denominator if denominator else 0.0,
            "replacement_tail_position_ratio": float(group_counts["tail_position"]) / denominator if denominator else 0.0,
            "replacement_left_entropy": _shannon_entropy(left_counter),
            "replacement_right_entropy": _shannon_entropy(right_counter),
            "replacement_pair_entropy": _shannon_entropy(pair_counter),
            "replacement_candidate_left_entropy": _shannon_entropy(left_counter),
            "replacement_candidate_right_entropy": _shannon_entropy(right_counter),
            "replacement_candidate_pair_entropy": _shannon_entropy(pair_counter),
            "replacement_left_overlap_count_with_top20_near_top_anchors": overlap["count"],
            "replacement_left_overlap_ratio_relative_to_unique_left": overlap["ratio_unique_left"],
            "replacement_left_overlap_ratio_relative_to_top20_near_top_anchors": overlap[
                "ratio_top20_near_top_anchors"
            ],
        }
    )
    grouped = {
        "available": True,
        "source": "simulated_from_fake_sessions",
        "candidate_space_definition": "valid positions are 1..L-2 for L>=3; L==2 uses fallback position 1",
        "valid_position_count": int(valid_position_count),
        "flat": flat,
        "top_left_items": _top_counter(left_counter, 20),
        "top_right_items": _top_counter(right_counter, 20),
        "top_left_right_pairs": _top_pair_counter(pair_counter, 20),
    }
    return grouped, flat


def _metadata_target_from_json(path: Path) -> int:
    payload = _load_json_or_none(path)
    if payload is None:
        raise ValueError(f"Metadata file is missing or not a JSON object: {path}")
    target = payload.get("target_item")
    if target is None:
        raise ValueError(f"Metadata file does not contain target_item: {path}")
    return int(target)


def resolve_exposure_metadata_paths(
    *,
    explicit_paths: Sequence[str | Path],
    metadata_dir: str | Path | None,
    filename: str,
    method_name: str,
) -> dict[int, Path]:
    candidates: list[Path] = []
    if explicit_paths:
        candidates = [_repo_path(path) for path in explicit_paths]
    elif metadata_dir is not None:
        root = _repo_path(metadata_dir)
        if root.exists():
            candidates = sorted(root.rglob(filename), key=lambda path: str(path))
    resolved: dict[int, Path] = {}
    for path in candidates:
        if not path.exists():
            raise FileNotFoundError(f"{method_name} metadata path does not exist: {path}")
        target = _metadata_target_from_json(path)
        if target in resolved:
            raise ValueError(
                f"Multiple {method_name} metadata files found for target {target}: "
                f"{resolved[target]} and {path}. Provide one explicit path list to disambiguate."
            )
        resolved[target] = path
    return resolved


def _load_exposure_metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = _load_json_or_none(path)
    if payload is None:
        raise ValueError(f"Exposure metadata file is missing or invalid: {path}")
    return payload


def build_summary_row(*parts: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in SUMMARY_COLUMNS}
    for part in parts:
        for key, value in part.items():
            if key in row:
                row[key] = value
    missing = [column for column in SUMMARY_COLUMNS if column not in row]
    if missing:
        raise RuntimeError(f"Summary row missing columns: {missing}")
    return row


def _markdown_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_to_jsonable(value), sort_keys=True)
    return str(value)


def write_target_markdown(path: Path, payload: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    target = int(payload["target_item"])
    notes = payload.get("notes", [])
    lines = [
        f"# Target Action Feature Survey: {target}",
        "",
        "This analysis describes data-side and clean-model characteristics only. It does not compare attack outcomes or select attack methods.",
        "",
        "## Target popularity and sparsity",
        _markdown_table(
            [[
                row["train_target_occurrence_count"],
                row["train_target_session_count"],
                row["train_target_label_count"],
                row["valid_target_occurrence_count"],
                row["valid_target_session_count"],
                row["valid_target_label_count"],
                row["target_frequency_rank"],
                row["target_popularity_percentile"],
                row["target_density_bucket"],
            ]],
            ["train_occ", "train_sessions", "train_labels", "valid_occ", "valid_sessions", "valid_labels", "freq_rank", "pop_percentile", "density"],
        ),
        "",
        "## Natural transition profile",
        _markdown_table(
            [[
                row["num_unique_predecessors"],
                row["predecessor_effective_count"],
                row["top5_predecessor_items"],
                row["num_unique_successors"],
                row["successor_effective_count"],
                row["top5_successor_items"],
                row["num_cooccurrence_items"],
                row["cooccurrence_effective_count"],
                row["top5_cooccurrence_items"],
            ]],
            ["pred_unique", "pred_eff", "top5_pred", "succ_unique", "succ_eff", "top5_succ", "co_unique", "co_eff", "top5_co"],
        ),
        "",
        "## Clean-rank susceptibility profile",
        "Near-top validation contexts are validation prefixes where the target is not in top-20 but is within top-200 under the clean model when defaults are used.",
        _markdown_table(
            [[
                row["validation_prefix_count"],
                row["near_top_prefix_count"],
                row["near_top_prefix_ratio"],
                row["clean_target_rank_median"],
                row["near_top_rank_median"],
                row["rank_1_20_count"],
                row["rank_21_50_count"],
                row["rank_51_100_count"],
                row["rank_101_200_count"],
                row["rank_above_200_count"],
            ]],
            ["valid_prefixes", "near_top", "near_top_ratio", "rank_median", "near_top_median", "r1_20", "r21_50", "r51_100", "r101_200", "r200_plus"],
        ),
        "",
        "## Near-top anchor concentration",
        "Near-top anchors are the last items of near-top validation contexts and are reported descriptively, not as attack candidates.",
        _markdown_table(
            [[
                row["num_near_top_last_item_anchors"],
                row["top1_near_top_anchor_count"],
                row["top1_near_top_anchor_coverage"],
                row["top5_near_top_anchor_coverage"],
                row["near_top_anchor_effective_count"],
                row["top20_near_top_anchor_train_predecessor_overlap_ratio"],
                row["top5_near_top_anchor_items"],
            ]],
            ["unique", "top1_count", "top1_cov", "top5_cov", "effective", "pred_overlap20", "top5"],
        ),
        "",
        "## Fake-session availability",
        _markdown_table(
            [[
                row["fake_session_coverage_count_top20_near_top_anchors"],
                row["fake_session_coverage_ratio_top20_near_top_anchors"],
                row["fake_session_occurrence_count_top20_near_top_anchors"],
                row["fake_session_coverage_count_top20_train_predecessors"],
                row["fake_session_coverage_ratio_top20_train_predecessors"],
                row["fake_session_occurrence_count_top20_train_predecessors"],
            ]],
            ["anchor_cov_count", "anchor_cov_ratio", "anchor_occ", "pred_cov_count", "pred_cov_ratio", "pred_occ"],
        ),
        "",
        "## Internal insertion exposure compatibility",
        (
            "These fields describe the simulated candidate space of Internal Random Insertion-NZ, not sampled attack outcomes."
            if row["insertion_exposure_source"] == "simulated_from_fake_sessions"
            else (
                "These fields are parsed from sampled Internal Random Insertion-NZ metadata."
                if row["insertion_exposure_source"] == "metadata"
                else "Internal Random Insertion-NZ exposure metadata and fake-session simulation inputs are unavailable."
            )
        ),
        _markdown_table(
            [[
                row["insertion_exposure_source"],
                row["insertion_fake_session_count"],
                row["insertion_candidate_unique_left_item_count"],
                row["insertion_candidate_unique_right_item_count"],
                row["insertion_candidate_unique_left_right_pair_count"],
                row["insertion_tail_slot_ratio"],
                row["insertion_left_overlap_ratio_relative_to_top20_near_top_anchors"],
            ]],
            ["source", "fake_sessions", "candidate_left_unique", "candidate_right_unique", "candidate_pair_unique", "tail_slot_ratio", "left_near_top_overlap"],
        ),
        "",
        "## Internal replacement exposure compatibility",
        (
            "These fields describe the simulated candidate space of Internal Random Replacement-NZ, not sampled attack outcomes."
            if row["replacement_exposure_source"] == "simulated_from_fake_sessions"
            else (
                "These fields are parsed from sampled Internal Random Replacement-NZ metadata."
                if row["replacement_exposure_source"] == "metadata"
                else "Internal Random Replacement-NZ exposure metadata and fake-session simulation inputs are unavailable."
            )
        ),
        _markdown_table(
            [[
                row["replacement_exposure_source"],
                row["replacement_fake_session_count"],
                row["replacement_tail_fallback_ratio"],
                row["replacement_candidate_unique_left_item_count"],
                row["replacement_candidate_unique_right_item_count"],
                row["replacement_candidate_unique_left_right_pair_count"],
                row["replacement_left_overlap_ratio_relative_to_top20_near_top_anchors"],
            ]],
            ["source", "fake_sessions", "tail_fallback", "candidate_left_unique", "candidate_right_unique", "candidate_pair_unique", "left_near_top_overlap"],
        ),
        "",
        "## Notes",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- No additional notes.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Target Action Feature Survey Summary",
        "",
        "This survey describes target/item characteristics and exposure compatibility features. It does not read or compare attack outcome metrics.",
        "",
        _markdown_table(
            [[row.get(column) for column in SUMMARY_MD_COLUMNS] for row in rows],
            SUMMARY_MD_COLUMNS,
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_survey(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _repo_path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = load_config(config_path)
    output_dir = _repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [int(target) for target in args.targets]
    top_k = int(args.top_k)
    near_top_rank_min = int(args.near_top_rank_min)
    near_top_rank_max = int(args.near_top_rank_max)

    dataset = ensure_canonical_dataset(config)
    train_counts = item_counts(dataset.train_sub)
    train_ranks = frequency_ranks(train_counts)
    train_label_counts = _label_counts(dataset.train_sub)
    valid_label_counts = _label_counts(dataset.valid)
    valid_prefixes, _ = expanded_cases(dataset.valid)

    checkpoint_path = resolve_clean_poison_checkpoint(config)
    if checkpoint_path is None:
        raise FileNotFoundError(
            "No clean SR-GNN poison-model checkpoint trained on train_sub was found; "
            "the survey will not use poisoned victim or attack outcome checkpoints."
        )
    fake_sessions, fake_sessions_path = resolve_fake_sessions(
        config,
        explicit_path=args.fake_sessions_path,
    )
    insertion_paths = resolve_exposure_metadata_paths(
        explicit_paths=args.internal_insertion_metadata_paths or [],
        metadata_dir=args.internal_insertion_metadata_dir,
        filename="internal_random_insertion_metadata.json",
        method_name="internal insertion",
    )
    replacement_paths = resolve_exposure_metadata_paths(
        explicit_paths=args.internal_replacement_metadata_paths or [],
        metadata_dir=args.internal_replacement_metadata_dir,
        filename="internal_random_replacement_metadata.json",
        method_name="internal replacement",
    )

    print(f"[survey] Scoring {len(valid_prefixes)} validation prefixes with clean checkpoint {checkpoint_path}")
    validation_scores = score_validation_prefixes(
        config,
        checkpoint_path=checkpoint_path,
        prefixes=valid_prefixes,
        targets=targets,
    )

    rows: list[dict[str, Any]] = []
    target_payloads: list[dict[str, Any]] = []
    for target in targets:
        print(f"[survey] Building target/item feature survey for target {target}")
        popularity_group, popularity_flat = compute_target_popularity(
            train_sessions=dataset.train_sub,
            valid_sessions=dataset.valid,
            train_counts=train_counts,
            train_ranks=train_ranks,
            train_label_counts=train_label_counts,
            valid_label_counts=valid_label_counts,
            target=target,
        )
        natural_group, natural_flat = compute_natural_transition_profile(
            dataset.train_sub,
            target,
            top_k=top_k,
        )
        ranks = [int(value) for value in validation_scores[target]["ranks"]]
        scores = [float(value) for value in validation_scores[target]["scores"]]
        margins = validation_scores[target]["margin_to_top20"]
        clean_group, clean_flat, near_indices = compute_clean_rank_susceptibility_profile(
            ranks,
            scores,
            margins,
            rank_min=near_top_rank_min,
            rank_max=near_top_rank_max,
        )
        near_anchor_group, near_anchor_flat, top20_near_top_anchors = compute_near_top_anchor_concentration(
            valid_prefixes,
            near_indices,
            natural_group["predecessor_counts"],
            top_k=top_k,
        )
        top20_train_predecessors = _item_list(natural_group["predecessors"]["top20_items"])
        fake_group, fake_flat = compute_fake_session_availability(
            fake_sessions,
            fake_sessions_path=fake_sessions_path,
            top20_near_top_anchors=top20_near_top_anchors,
            top20_train_predecessors=top20_train_predecessors,
        )

        insertion_metadata = _load_exposure_metadata(insertion_paths.get(target))
        if insertion_metadata is None:
            insertion_group, insertion_flat = simulate_insertion_exposure(
                fake_sessions,
                top20_near_top_anchors=top20_near_top_anchors,
            )
        else:
            insertion_group, insertion_flat = parse_insertion_exposure_metadata(
                insertion_metadata,
                top20_near_top_anchors=top20_near_top_anchors,
            )
            insertion_group["metadata_path"] = str(insertion_paths[target])

        replacement_metadata = _load_exposure_metadata(replacement_paths.get(target))
        if replacement_metadata is None:
            replacement_group, replacement_flat = simulate_replacement_exposure(
                fake_sessions,
                top20_near_top_anchors=top20_near_top_anchors,
            )
        else:
            replacement_group, replacement_flat = parse_replacement_exposure_metadata(
                replacement_metadata,
                top20_near_top_anchors=top20_near_top_anchors,
            )
            replacement_group["metadata_path"] = str(replacement_paths[target])

        notes = [
            "Clean-rank scoring uses validation prefixes and the clean SR-GNN poison-model checkpoint trained on train_sub before poisoning.",
            "Test data is not used in the main survey.",
        ]
        if insertion_metadata is None and fake_sessions is not None:
            notes.append("Insertion exposure compatibility was simulated from shared fake-session candidate spaces because metadata was not provided.")
        if replacement_metadata is None and fake_sessions is not None:
            notes.append("Replacement exposure compatibility was simulated from shared fake-session candidate spaces because metadata was not provided.")
        if fake_sessions is None:
            notes.append("Shared fake sessions were unavailable; fake-session and simulated exposure fields are null.")

        row = build_summary_row(
            popularity_flat,
            natural_flat,
            clean_flat,
            near_anchor_flat,
            fake_flat,
            insertion_flat,
            replacement_flat,
        )
        payload = {
            "target_item": int(target),
            "metadata": {
                "config_path": str(config_path),
                "dataset_name": config.data.dataset_name,
                "split_protocol": config.data.split_protocol,
                "clean_checkpoint_path": str(checkpoint_path),
                "rank_convention": RANK_CONVENTION,
                "near_top_rank_min": near_top_rank_min,
                "near_top_rank_max": near_top_rank_max,
                "test_used_in_main_summary": False,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "target_popularity": popularity_group,
            "natural_transition_profile": natural_group,
            "clean_rank_susceptibility_profile": clean_group,
            "near_top_anchor_concentration": near_anchor_group,
            "fake_session_availability": fake_group,
            "insertion_exposure_compatibility": insertion_group,
            "replacement_exposure_compatibility": replacement_group,
            "notes": notes,
        }
        _write_json(output_dir / f"target_action_feature_survey_{target}.json", payload)
        write_target_markdown(
            output_dir / f"target_action_feature_survey_{target}.md",
            payload,
            row,
        )
        rows.append(row)
        target_payloads.append(payload)

    summary_payload = {
        "config_path": str(config_path),
        "dataset_name": config.data.dataset_name,
        "split_protocol": config.data.split_protocol,
        "target_list": targets,
        "near_top_rank_definition": {
            "rank_min": near_top_rank_min,
            "rank_max": near_top_rank_max,
            "rank_convention": RANK_CONVENTION,
            "description": (
                "validation prefixes where the target is not in top-20 but is "
                "within top-200 under the clean model when defaults are used"
            ),
        },
        "clean_checkpoint_path": str(checkpoint_path),
        "fake_sessions_path": None if fake_sessions_path is None else str(fake_sessions_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "targets": target_payloads,
    }
    _write_json(output_dir / "target_action_feature_survey_summary.json", summary_payload)
    _write_csv(output_dir / "target_action_feature_survey_summary.csv", rows, SUMMARY_COLUMNS)
    write_summary_markdown(output_dir / "target_action_feature_survey_summary.md", rows)
    return summary_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-free target/item feature survey for SBR robustness analysis."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--targets", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--fake-sessions-path", default=None)
    parser.add_argument("--near-top-rank-min", type=int, default=21)
    parser.add_argument("--near-top-rank-max", type=int, default=200)
    parser.add_argument("--rank-min", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rank-max", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--internal-insertion-metadata-dir", default=None)
    parser.add_argument("--internal-replacement-metadata-dir", default=None)
    parser.add_argument("--internal-insertion-metadata-paths", nargs="*", default=[])
    parser.add_argument("--internal-replacement-metadata-paths", nargs="*", default=[])
    parser.add_argument(
        "--include-test-posthoc",
        action="store_true",
        help="Reserved for explicitly enabled post-hoc diagnostics; test fields are not included in the main summary.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rank_min is not None:
        args.near_top_rank_min = int(args.rank_min)
    if args.rank_max is not None:
        args.near_top_rank_max = int(args.rank_max)
    payload = run_survey(args)
    output_dir = _repo_path(args.output_dir)
    print(f"[survey] Wrote summary to {output_dir / 'target_action_feature_survey_summary.md'}")
    print(f"[survey] Surveyed {len(payload['rows'])} targets without reading attack outcome metrics.")


if __name__ == "__main__":
    main()
