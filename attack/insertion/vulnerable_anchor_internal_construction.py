from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class VulnerableAnchorInternalConstructionResult:
    session: list[int]
    anchor_item: int
    target_item: int
    target_insertion_slot: int
    anchor_replace_position: int
    original_replaced_item: int
    right_item: int
    was_anchor_replacement_noop: bool
    anchor_already_in_original_session: bool
    original_length: int
    final_length: int
    pre_existing_target_count: int
    target_occurrence_count_after_construction: int
    final_target_positions: list[int]


@dataclass(frozen=True)
class LoadedVulnerableAnchorPool:
    target_item: int
    anchor_pool: list[int]
    top_anchor_rows: list[dict[str, Any]]
    survey_file_path: str
    survey_file_hash: str
    source_format: str
    rank_min: int | None = None
    rank_max: int | None = None

    @property
    def selected_anchor_pool_hash(self) -> str:
        return stable_anchor_pool_hash(self.anchor_pool)


class VulnerableAnchorInternalConstructionPolicy:
    def __init__(
        self,
        anchor_pool: Sequence[int],
        topk_ratio: float,
        rng: random.Random | None = None,
        anchor_assignment_strategy: str = "round_robin",
    ) -> None:
        if not anchor_pool:
            raise ValueError("anchor_pool must contain at least one anchor item.")
        if not (0.0 < topk_ratio <= 1.0):
            raise ValueError("topk_ratio must be within (0, 1].")
        strategy = str(anchor_assignment_strategy).strip().lower()
        if strategy != "round_robin":
            raise ValueError(
                "Vulnerable-Anchor Internal Construction v1 supports only "
                "anchor_assignment_strategy='round_robin'."
            )
        self.anchor_pool = [int(item) for item in anchor_pool]
        self.topk_ratio = float(topk_ratio)
        self.rng = rng or random.Random()
        self.anchor_assignment_strategy = strategy

    def apply(
        self,
        session: Sequence[int],
        target_item: int,
        session_index: int,
    ) -> list[int]:
        return self.apply_with_metadata(session, target_item, session_index).session

    def apply_with_metadata(
        self,
        session: Sequence[int],
        target_item: int,
        session_index: int,
    ) -> VulnerableAnchorInternalConstructionResult:
        original = [int(item) for item in session]
        if len(original) < 2:
            raise ValueError(
                "Vulnerable-Anchor Internal Construction requires session length >= 2; "
                "there is no valid internal insertion slot with both neighbors."
            )

        target = int(target_item)
        anchor = int(self.anchor_pool[int(session_index) % len(self.anchor_pool)])
        length = len(original)
        valid_internal_slot_count = length - 1
        topk_count = max(
            1,
            int(math.ceil(valid_internal_slot_count * self.topk_ratio)),
        )
        max_slot = min(topk_count, valid_internal_slot_count)
        target_insertion_slot = int(self.rng.randint(1, max_slot))
        if target_insertion_slot < 1 or target_insertion_slot > length - 1:
            raise RuntimeError("Sampled target insertion slot is not internal.")

        anchor_replace_position = int(target_insertion_slot - 1)
        original_replaced_item = int(original[anchor_replace_position])
        right_item = int(original[target_insertion_slot])

        constructed = list(original)
        constructed[anchor_replace_position] = anchor
        constructed.insert(target_insertion_slot, target)

        final_target_positions = [
            int(index)
            for index, item in enumerate(constructed)
            if int(item) == target
        ]
        result = VulnerableAnchorInternalConstructionResult(
            session=constructed,
            anchor_item=anchor,
            target_item=target,
            target_insertion_slot=target_insertion_slot,
            anchor_replace_position=anchor_replace_position,
            original_replaced_item=original_replaced_item,
            right_item=right_item,
            was_anchor_replacement_noop=bool(original_replaced_item == anchor),
            anchor_already_in_original_session=bool(anchor in set(original)),
            original_length=length,
            final_length=len(constructed),
            pre_existing_target_count=int(sum(1 for item in original if item == target)),
            target_occurrence_count_after_construction=int(
                sum(1 for item in constructed if item == target)
            ),
            final_target_positions=final_target_positions,
        )
        _validate_result(original, result)
        return result


def load_vulnerable_anchor_pool(
    *,
    survey_output_dir: str | Path,
    target_item: int,
    anchor_top_m: int,
    require_survey_file: bool = True,
) -> LoadedVulnerableAnchorPool:
    if anchor_top_m < 1:
        raise ValueError("anchor_top_m must be >= 1.")
    survey_dir = Path(survey_output_dir)
    csv_path = survey_dir / f"target_anchor_candidates_{int(target_item)}.csv"
    json_path = survey_dir / f"target_anchor_survey_{int(target_item)}.json"
    if csv_path.exists():
        return _load_anchor_pool_from_csv(
            csv_path,
            companion_json_path=json_path,
            target_item=int(target_item),
            anchor_top_m=int(anchor_top_m),
        )
    if json_path.exists():
        return _load_anchor_pool_from_json(
            json_path,
            target_item=int(target_item),
            anchor_top_m=int(anchor_top_m),
        )
    message = (
        "Vulnerable-Anchor Internal Construction requires target-anchor survey "
        f"output for target {int(target_item)}. Expected one of: "
        f"{csv_path}, {json_path}."
    )
    if require_survey_file:
        raise FileNotFoundError(message)
    raise FileNotFoundError(message)


def stable_anchor_pool_hash(anchor_pool: Sequence[int]) -> str:
    payload = json.dumps(
        [int(item) for item in anchor_pool],
        sort_keys=False,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def file_sha1(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_anchor_pool_from_csv(
    path: Path,
    *,
    companion_json_path: Path | None = None,
    target_item: int,
    anchor_top_m: int,
) -> LoadedVulnerableAnchorPool:
    required = {
        "target_item",
        "anchor_item",
        "is_vulnerable_last_item",
        "vulnerable_count",
        "vulnerable_coverage",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                f"Anchor candidate CSV is missing required columns: {missing}. "
                f"path={path}"
            )
        rows = [dict(row) for row in reader]

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if _to_int(row.get("target_item"), "target_item") != int(target_item):
            continue
        anchor_item = _to_int(row.get("anchor_item"), "anchor_item")
        if anchor_item == int(target_item):
            continue
        if not _parse_bool(row.get("is_vulnerable_last_item")):
            continue
        vulnerable_count = _to_int(row.get("vulnerable_count"), "vulnerable_count")
        if vulnerable_count <= 0:
            continue
        normalized = _normalize_candidate_row(row)
        normalized["target_item"] = int(target_item)
        normalized["anchor_item"] = int(anchor_item)
        normalized["vulnerable_count"] = int(vulnerable_count)
        normalized["is_vulnerable_last_item"] = True
        candidates.append(normalized)

    rank_min = None
    rank_max = None
    if companion_json_path is not None and companion_json_path.exists():
        rank_min, rank_max = _rank_bounds_from_json(companion_json_path)

    return _finalize_loaded_pool(
        path=path,
        target_item=int(target_item),
        anchor_top_m=int(anchor_top_m),
        source_format="csv",
        rows=candidates,
        rank_min=rank_min,
        rank_max=rank_max,
    )


def _load_anchor_pool_from_json(
    path: Path,
    *,
    target_item: int,
    anchor_top_m: int,
) -> LoadedVulnerableAnchorPool:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Anchor survey JSON root must be an object: {path}")
    rows = _find_first_list(payload, "last_item_anchors")
    if rows is None:
        raise ValueError(
            f"Anchor survey JSON is missing last_item_anchors for target {target_item}: {path}"
        )
    candidates: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        anchor_item = _to_int(raw.get("anchor_item"), "anchor_item")
        if anchor_item == int(target_item):
            continue
        vulnerable_count = _to_int(raw.get("vulnerable_count"), "vulnerable_count")
        if vulnerable_count <= 0:
            continue
        normalized = _normalize_candidate_row(raw)
        normalized["target_item"] = int(target_item)
        normalized["anchor_item"] = int(anchor_item)
        normalized["vulnerable_count"] = int(vulnerable_count)
        normalized["is_vulnerable_last_item"] = True
        candidates.append(normalized)

    survey_meta = payload.get("survey_config", {})
    if not isinstance(survey_meta, Mapping):
        survey_meta = payload
    rank_min = _optional_int(survey_meta.get("rank_min"))
    rank_max = _optional_int(survey_meta.get("rank_max"))
    return _finalize_loaded_pool(
        path=path,
        target_item=int(target_item),
        anchor_top_m=int(anchor_top_m),
        source_format="json",
        rows=candidates,
        rank_min=rank_min,
        rank_max=rank_max,
    )


def _rank_bounds_from_json(path: Path) -> tuple[int | None, int | None]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        return None, None
    survey_meta = payload.get("survey_config")
    if not isinstance(survey_meta, Mapping):
        survey_meta = payload.get("metadata")
    if not isinstance(survey_meta, Mapping):
        survey_meta = _find_rank_bounds_mapping(payload)
    if not isinstance(survey_meta, Mapping):
        return None, None
    return (
        _optional_int(survey_meta.get("rank_min")),
        _optional_int(survey_meta.get("rank_max")),
    )


def _find_rank_bounds_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        if "rank_min" in value and "rank_max" in value:
            return value
        for inner in value.values():
            found = _find_rank_bounds_mapping(inner)
            if found is not None:
                return found
    if isinstance(value, list):
        for inner in value:
            found = _find_rank_bounds_mapping(inner)
            if found is not None:
                return found
    return None


def _finalize_loaded_pool(
    *,
    path: Path,
    target_item: int,
    anchor_top_m: int,
    source_format: str,
    rows: list[dict[str, Any]],
    rank_min: int | None,
    rank_max: int | None,
) -> LoadedVulnerableAnchorPool:
    rows.sort(
        key=lambda row: (
            -int(row.get("vulnerable_count", 0)),
            -float(row.get("vulnerable_coverage", 0.0) or 0.0),
            -float(row.get("anchor_score", 0.0) or 0.0),
            int(row["anchor_item"]),
        )
    )
    selected = rows[: int(anchor_top_m)]
    anchor_pool = [int(row["anchor_item"]) for row in selected]
    if not anchor_pool:
        raise ValueError(
            "No vulnerable validation last-item anchors were available after filtering "
            f"for target {int(target_item)} from {path}."
        )
    return LoadedVulnerableAnchorPool(
        target_item=int(target_item),
        anchor_pool=anchor_pool,
        top_anchor_rows=selected,
        survey_file_path=str(path),
        survey_file_hash=file_sha1(path),
        source_format=source_format,
        rank_min=rank_min,
        rank_max=rank_max,
    )


def _normalize_candidate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "anchor_item",
        "vulnerable_count",
        "vulnerable_coverage",
        "avg_target_rank",
        "avg_vulnerable_target_rank",
        "train_predecessor_count_to_target",
        "fake_session_count_with_anchor",
        "anchor_score",
    )
    normalized: dict[str, Any] = {}
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value == "":
            normalized[key] = None
        elif key in {
            "anchor_item",
            "vulnerable_count",
            "train_predecessor_count_to_target",
            "fake_session_count_with_anchor",
        }:
            normalized[key] = _optional_int(value)
        else:
            normalized[key] = _optional_float(value)
    if "avg_target_rank" not in normalized and "avg_vulnerable_target_rank" in normalized:
        normalized["avg_target_rank"] = normalized["avg_vulnerable_target_rank"]
    return normalized


def _validate_result(
    original: Sequence[int],
    result: VulnerableAnchorInternalConstructionResult,
) -> None:
    final = result.session
    slot = int(result.target_insertion_slot)
    if len(final) != len(original) + 1:
        raise RuntimeError("Constructed session length delta is not +1.")
    if int(result.target_item) not in set(final):
        raise RuntimeError("Constructed session is missing target item.")
    if slot < 1 or slot > len(original) - 1:
        raise RuntimeError("Constructed target slot is slot0 or tail.")
    if slot + 1 >= len(final):
        raise RuntimeError("Constructed target has no right neighbor.")
    if final[slot - 1] != int(result.anchor_item):
        raise RuntimeError("Constructed target left neighbor is not the selected anchor.")
    if final[slot] != int(result.target_item):
        raise RuntimeError("Constructed target is not at the recorded insertion slot.")
    if final[slot + 1] != int(result.right_item):
        raise RuntimeError("Constructed target right neighbor metadata is invalid.")
    expected = (
        [int(item) for item in original[: slot - 1]]
        + [int(result.anchor_item), int(result.target_item)]
        + [int(item) for item in original[slot:]]
    )
    if final != expected:
        raise RuntimeError(
            "Constructed session does not preserve original order except the "
            "anchor replacement and inserted target."
        )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value) != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f", ""}:
            return False
    raise ValueError(f"Unable to parse boolean value: {value!r}")


def _to_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Expected {field_name} to be an int, got bool.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unable to parse {field_name} as int: {value!r}") from exc


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _find_first_list(value: Any, key: str) -> list[Any] | None:
    if isinstance(value, Mapping):
        candidate = value.get(key)
        if isinstance(candidate, list):
            return candidate
        for inner in value.values():
            found = _find_first_list(inner, key)
            if found is not None:
                return found
    if isinstance(value, list):
        for inner in value:
            found = _find_first_list(inner, key)
            if found is not None:
                return found
    return None


__all__ = [
    "LoadedVulnerableAnchorPool",
    "VulnerableAnchorInternalConstructionPolicy",
    "VulnerableAnchorInternalConstructionResult",
    "file_sha1",
    "load_vulnerable_anchor_pool",
    "stable_anchor_pool_hash",
]
