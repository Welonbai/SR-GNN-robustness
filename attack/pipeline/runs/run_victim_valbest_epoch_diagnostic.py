from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import Config, load_config
from attack.common.paths import (
    freqrec_diagnostic_key,
    run_group_key,
    runs_root,
    target_dir,
)
from attack.data.exporters.miasrec_exporter import MiaSRecExporter
from attack.data.exporters.tron_exporter import TRONExporter
from attack.data.exporters.freqrec_exporter import FreqRecExporter
from attack.data.poisoned_dataset_builder import PoisonedDataset, build_poisoned_dataset
from attack.data.unified_split import ensure_canonical_dataset
from attack.models.victim.miasrec_runner import MiaSRecRunner
from attack.models.victim.tron_runner import TRONRunner
from attack.models.victim.freqrec_runner import FreqRecRunner
from attack.models.victim.freqrec_diagnostics import (
    summarize_freqrec_epoch_diagnostics,
)
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.pipeline.core.pipeline_utils import build_clean_pairs
from attack.pipeline.core.victim_execution import victim_effective_train_seed


VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE = "victim_valbest_epoch_diagnostic"
SOURCE_METHOD_NOTE = "5-action space-filling seed-aligned PTS-CEM"
EXPECTED_ACTIONS = (
    "keep_residual_suffix",
    "regenerate_residual_suffix",
    "consume_one_keep_rest",
    "consume_one_generate_continuation",
    "consume_all_stop",
)
DEFAULT_EXPECTED_CANDIDATE_KEY = "iter1_cand5"
DEFAULT_TARGET_ITEM = 39588


@dataclass(frozen=True)
class SourcePTSArtifact:
    target_item: int
    candidate_rank: int
    artifact_dir: Path
    source_run: Path
    sessions_path: Path
    metadata_path: Path | None
    policy_path: Path | None
    complete_marker_path: Path | None
    top_candidates_path: Path | None
    sessions: list[list[int]]
    metadata: dict[str, Any]
    policy: dict[str, Any]
    complete_marker: dict[str, Any]
    sessions_sha1: str
    source_pts_cem_cache_key: str | None
    source_candidate_key: str | None
    artifact_raw_lowk: float | None
    manual_raw_lowk: float | None
    validation_warnings: list[str]


def run_diagnostic(
    config_path: str | Path,
    *,
    target_item: int | None = None,
    victim: str = "all",
    source_pts_cem_run: str | Path | None = None,
    candidate_rank: int | None = None,
    experiment_name: str | None = None,
    max_epochs: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    raw_config = _load_yaml_mapping(config_path)
    diagnostic_config = _diagnostic_options(raw_config)
    selected_victims = _selected_victims(config, victim)
    if experiment_name:
        from dataclasses import replace

        config = replace(config, experiment=replace(config.experiment, name=str(experiment_name)))

    summaries: list[dict[str, Any]] = []
    out_dir: Path | None = None
    if "freqrec" in selected_victims:
        effective_freqrec_epochs = int(
            max_epochs
            if max_epochs is not None
            else config.victims.params["freqrec"]["train"]["epochs"]
        )
        freqrec_out_dir = _freqrec_diagnostic_dir(
            config,
            effective_epochs=effective_freqrec_epochs,
        )
        out_dir = freqrec_out_dir
        if freqrec_out_dir.exists() and force:
            shutil.rmtree(freqrec_out_dir)
        freqrec_out_dir.mkdir(parents=True, exist_ok=True)
        summaries.append(
            _run_freqrec_diagnostic(
                config,
                out_dir=freqrec_out_dir,
                effective_epochs=effective_freqrec_epochs,
            )
        )

    source_victims = [
        victim_name
        for victim_name in selected_victims
        if victim_name in {"miasrec", "tron"}
    ]
    source = None
    target = None
    if source_victims:
        source_config = _diagnostic_source_config(raw_config)
        target = int(
            target_item
            if target_item is not None
            else source_config.get("target_item", _target_from_config(config))
        )
        rank = int(
            candidate_rank
            if candidate_rank is not None
            else source_config.get("candidate_rank", 1)
        )
        requested_source = (
            source_pts_cem_run
            if source_pts_cem_run is not None
            else source_config.get("source_run")
        )
        source = resolve_source_pts_artifact(
            config,
            target_item=target,
            candidate_rank=rank,
            source_run=requested_source,
            expected_candidate_key=str(
                source_config.get(
                    "expected_candidate_key", DEFAULT_EXPECTED_CANDIDATE_KEY
                )
            ),
            manual_raw_lowk=_optional_float(
                source_config.get("final_target_raw_lowk")
            ),
        )
        poisoned = _build_poisoned_train(config, source.sessions)
        out_dir = _diagnostic_target_dir(config, source)
        if out_dir.exists() and force:
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(source_identity_payload(source), out_dir / "source_pts_cem_identity.json")

    for victim_name in source_victims:
        if victim_name == "miasrec":
            summary = _run_miasrec_diagnostic(
                config,
                source=source,
                poisoned=poisoned,
                out_dir=out_dir,
                max_epochs=max_epochs,
                diagnostic_config=diagnostic_config,
            )
        elif victim_name == "tron":
            summary = _run_tron_diagnostic(
                config,
                source=source,
                poisoned=poisoned,
                out_dir=out_dir,
                max_epochs=max_epochs,
                diagnostic_config=diagnostic_config,
            )
        else:
            raise ValueError(f"Unsupported diagnostic victim: {victim_name}")
        summaries.append(summary)

    if out_dir is None:
        raise ValueError("No supported diagnostic victims were selected.")
    combined = {
        "dataset": config.data.dataset_name,
        "target_item": None if target is None else int(target),
        "source": None if source is None else source_identity_payload(source),
        "victims": summaries,
    }
    save_json(combined, out_dir / "victim_valbest_epoch_summary.json")
    _write_summary_csv(summaries, out_dir / "victim_valbest_epoch_summary.csv")
    return combined


def resolve_source_pts_artifact(
    config: Config,
    *,
    target_item: int,
    candidate_rank: int,
    source_run: str | Path | None,
    expected_candidate_key: str = DEFAULT_EXPECTED_CANDIDATE_KEY,
    manual_raw_lowk: float | None = None,
) -> SourcePTSArtifact:
    rank = _validate_candidate_rank(candidate_rank)
    candidate_dirs = _candidate_artifact_dirs(
        config,
        target_item=int(target_item),
        source_run=source_run,
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"PTS-CEM best candidate sessions not found for target {int(target_item)}. "
            "Run PTS-CEM first or provide source path."
        )
    if len(candidate_dirs) > 1:
        raise ValueError(
            "Multiple compatible PTS-CEM artifacts found for target "
            f"{int(target_item)}. Provide --source-pts-cem-run or "
            "source_attack.source_run."
        )

    artifact_dir = candidate_dirs[0]
    complete_marker = _load_json_object(artifact_dir / "pts_construction_complete.json")
    sessions_path = artifact_dir / "top_candidates" / f"rank_{rank}" / "sessions.json"
    metadata_path = artifact_dir / "top_candidates" / f"rank_{rank}" / "metadata.json"
    policy_path = artifact_dir / "top_candidates" / f"rank_{rank}" / "policy.json"
    if complete_marker:
        best_candidate = complete_marker.get("best_candidate")
        if isinstance(best_candidate, Mapping) and int(best_candidate.get("rank", rank)) == rank:
            sessions_path = _resolve_relative_artifact_path(
                artifact_dir,
                best_candidate.get("sessions_path"),
                fallback=sessions_path,
            )
            metadata_path = _resolve_relative_artifact_path(
                artifact_dir,
                best_candidate.get("metadata_path"),
                fallback=metadata_path,
            )
            policy_path = _resolve_relative_artifact_path(
                artifact_dir,
                best_candidate.get("policy_path"),
                fallback=policy_path,
            )

    if not sessions_path.exists():
        raise FileNotFoundError(
            f"PTS-CEM best candidate sessions not found for target {int(target_item)}. "
            "Run PTS-CEM first or provide source path. "
            f"Missing: {sessions_path}"
        )

    metadata = _load_json_object(metadata_path)
    policy = _load_json_object(policy_path)
    top_candidates_path = artifact_dir / "pts_top_candidates.json"
    top_candidates = _load_json_object(top_candidates_path)
    sessions = _load_json_sessions(sessions_path)
    sessions_sha1 = _sha1_file(sessions_path)

    merged_metadata = _merge_source_metadata(
        rank=rank,
        metadata=metadata,
        policy=policy,
        complete_marker=complete_marker,
        top_candidates=top_candidates,
    )
    artifact_raw_lowk = _extract_raw_lowk(merged_metadata)
    warnings = validate_source_metadata(
        merged_metadata,
        target_item=int(target_item),
        candidate_rank=rank,
        expected_candidate_key=expected_candidate_key,
        manual_raw_lowk=manual_raw_lowk,
        artifact_raw_lowk=artifact_raw_lowk,
    )

    return SourcePTSArtifact(
        target_item=int(target_item),
        candidate_rank=rank,
        artifact_dir=artifact_dir,
        source_run=_source_run_root(artifact_dir),
        sessions_path=sessions_path,
        metadata_path=metadata_path if metadata_path.exists() else None,
        policy_path=policy_path if policy_path.exists() else None,
        complete_marker_path=(
            artifact_dir / "pts_construction_complete.json"
            if complete_marker
            else None
        ),
        top_candidates_path=top_candidates_path if top_candidates_path.exists() else None,
        sessions=sessions,
        metadata=merged_metadata,
        policy=policy,
        complete_marker=complete_marker,
        sessions_sha1=sessions_sha1,
        source_pts_cem_cache_key=_optional_str(
            complete_marker.get("shared_pts_cem_cache_key")
        ),
        source_candidate_key=_optional_str(merged_metadata.get("candidate_key")),
        artifact_raw_lowk=artifact_raw_lowk,
        manual_raw_lowk=manual_raw_lowk,
        validation_warnings=warnings,
    )


def validate_source_metadata(
    metadata: Mapping[str, Any],
    *,
    target_item: int,
    candidate_rank: int,
    expected_candidate_key: str,
    manual_raw_lowk: float | None,
    artifact_raw_lowk: float | None,
) -> list[str]:
    warnings: list[str] = []
    _check_optional_equal(
        metadata,
        "target_item",
        int(target_item),
        warnings,
        missing="source metadata missing target_item",
    )
    _check_optional_equal(
        metadata,
        "rank",
        int(candidate_rank),
        warnings,
        missing="source metadata missing candidate rank",
    )
    if metadata.get("candidate_key") is None:
        warnings.append("source metadata missing candidate_key")
    elif str(metadata["candidate_key"]) != str(expected_candidate_key):
        raise ValueError(
            "PTS-CEM source candidate_key mismatch: "
            f"expected {expected_candidate_key}, found {metadata['candidate_key']}."
        )

    actions = _extract_enabled_actions(metadata)
    if actions is None:
        warnings.append("source metadata missing enabled PTS-CEM actions")
    elif set(actions) != set(EXPECTED_ACTIONS):
        raise ValueError(
            "PTS-CEM source enabled actions mismatch: "
            f"expected {list(EXPECTED_ACTIONS)}, found {sorted(actions)}."
        )

    prefix_range = _nested_get(metadata, ("prefix_selector", "range"))
    if prefix_range is None:
        warnings.append("source metadata missing prefix_selector.range")
    elif str(prefix_range).strip().lower() != "internal":
        raise ValueError(
            "PTS-CEM source prefix_selector.range mismatch: "
            f"expected internal, found {prefix_range}."
        )

    init_mode = metadata.get("init_mode") or _nested_get(metadata, ("sample_metadata", "init_mode"))
    if init_mode is None:
        warnings.append("source metadata missing CEM init mode")
    else:
        init_text = str(init_mode).strip().lower()
        if "vertex" not in init_text or "space_filling" not in init_text:
            raise ValueError(
                "PTS-CEM source init mode mismatch: "
                f"expected vertex/space-filling, found {init_mode}."
            )

    origin = metadata.get("sample_origin") or _nested_get(metadata, ("sample_metadata", "sample_origin"))
    if origin is None:
        warnings.append("source metadata missing best candidate origin")
    elif str(origin).strip().lower() != "elite_centered":
        raise ValueError(
            "PTS-CEM source candidate origin mismatch: "
            f"expected elite_centered, found {origin}."
        )

    aligned = metadata.get("surrogate_victim_seed_aligned")
    if aligned is None:
        warnings.append("source metadata missing surrogate_victim_seed_aligned")
    elif bool(aligned) is not True:
        raise ValueError("PTS-CEM source is not surrogate/victim seed aligned.")

    if manual_raw_lowk is not None and artifact_raw_lowk is not None:
        if abs(float(manual_raw_lowk) - float(artifact_raw_lowk)) > 1e-9:
            warnings.append(
                "manual source_final_target_raw_lowk differs from artifact "
                f"raw_lowk_mrr_recall_10_20: manual={manual_raw_lowk}, "
                f"artifact={artifact_raw_lowk}"
            )
    return warnings


def source_identity_payload(source: SourcePTSArtifact) -> dict[str, Any]:
    return {
        "source_pts_cem_run": str(source.source_run),
        "source_pts_cem_artifact_dir": str(source.artifact_dir),
        "source_pts_cem_cache_key": source.source_pts_cem_cache_key,
        "source_candidate_rank": int(source.candidate_rank),
        "source_candidate_key": source.source_candidate_key,
        "source_sessions_path": str(source.sessions_path),
        "source_sessions_sha1": source.sessions_sha1,
        "source_target_item": int(source.target_item),
        "source_method_note": SOURCE_METHOD_NOTE,
        "source_artifact_raw_lowk": source.artifact_raw_lowk,
        "source_manual_raw_lowk": source.manual_raw_lowk,
        "source_validation_warnings": list(source.validation_warnings),
    }


def summarize_epoch_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    victim_name: str,
    primary_metric: str,
    checkpoint_path: str | None = None,
    checkpoint_selection_mode: str,
    source: SourcePTSArtifact | None = None,
    max_epochs: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"Cannot summarize empty epoch metrics for {victim_name}.")
    normalized = [_normalize_metric_row(row) for row in rows]
    last_row = max(normalized, key=lambda row: int(row["epoch"]))
    best_primary = _best_row(normalized, primary_metric)
    best_mrr = _best_row(normalized, "mrr@20")
    best_recall = _best_row(normalized, "recall@20")
    primary_last = _optional_float(last_row.get(primary_metric))
    primary_best_value = _optional_float(best_primary.get(primary_metric)) if best_primary else None
    payload: dict[str, Any] = {
        "victim_name": victim_name,
        "max_epochs": int(max_epochs if max_epochs is not None else max(row["epoch"] for row in normalized)),
        "primary_metric": primary_metric,
        "best_epoch": None if best_primary is None else int(best_primary["epoch"]),
        "best_metric_value": primary_best_value,
        "best_epoch_by_mrr20": None if best_mrr is None else int(best_mrr["epoch"]),
        "best_valid_mrr20": None if best_mrr is None else _optional_float(best_mrr.get("mrr@20")),
        "best_epoch_by_recall20": None if best_recall is None else int(best_recall["epoch"]),
        "best_valid_recall20": None if best_recall is None else _optional_float(best_recall.get("recall@20")),
        "last_epoch": int(last_row["epoch"]),
        "last_epoch_metric_value": primary_last,
        "best_vs_last_delta": (
            None
            if primary_best_value is None or primary_last is None
            else float(primary_best_value) - float(primary_last)
        ),
        "selected_checkpoint_path": checkpoint_path,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "recommended_fixed_epoch_by_primary_metric": (
            None if best_primary is None else int(best_primary["epoch"])
        ),
        "recommendation_basis": "single_target_single_attack_diagnostic",
    }
    if source is not None:
        payload.update(source_identity_payload(source))
        payload["dataset"] = None
        payload["target_item"] = int(source.target_item)
    if extra:
        payload.update(dict(extra))
    return payload


def load_miasrec_epoch_metrics(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            row = {
                "epoch": int(payload["epoch"]),
                "valid_score": _optional_float(payload.get("valid_score")),
                "train_loss": _optional_float(payload.get("train_loss")),
                "valid_loss": _optional_float(payload.get("valid_loss")),
                "mrr@20": _metric_from_mapping(payload, ("valid_result",), "mrr@20"),
                "recall@20": _metric_from_mapping(payload, ("valid_result",), "recall@20"),
            }
            rows.append(row)
    return rows


def load_tron_epoch_metrics(
    log_dir: str | Path,
    *,
    max_epochs: int | None = None,
) -> list[dict[str, Any]]:
    metrics_path = _find_latest_metrics_csv(Path(log_dir))
    if metrics_path is None:
        return []
    by_epoch: dict[int, dict[str, Any]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            epoch_value = _first_float_value(raw_row, ["epoch"])
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            if max_epochs is not None and epoch >= int(max_epochs):
                continue
            row = by_epoch.setdefault(epoch, {"epoch": epoch})
            for key, value in raw_row.items():
                if value is None or value == "":
                    continue
                parsed = _try_float(value)
                if parsed is not None:
                    row[key] = parsed
    rows: list[dict[str, Any]] = []
    for epoch in sorted(by_epoch):
        raw = by_epoch[epoch]
        rows.append(
            {
                "epoch": int(epoch) + 1,
                "train_loss": _optional_float(raw.get("train_loss")),
                "valid_loss": _optional_float(raw.get("test_loss")),
                "recall@20": _optional_float(raw.get("recall_cutoff_20")),
                "mrr@20": _optional_float(raw.get("mrr_cutoff_20")),
                "recall_cutoff_20": _optional_float(raw.get("recall_cutoff_20")),
                "mrr_cutoff_20": _optional_float(raw.get("mrr_cutoff_20")),
                "test_loss": _optional_float(raw.get("test_loss")),
            }
        )
    return rows


def _run_miasrec_diagnostic(
    config: Config,
    *,
    source: SourcePTSArtifact,
    poisoned: PoisonedDataset,
    out_dir: Path,
    max_epochs: int | None,
    diagnostic_config: Mapping[str, Any],
) -> dict[str, Any]:
    victim_dir = out_dir / "miasrec"
    export_root = victim_dir / "export"
    epoch_metrics_path = out_dir / "miasrec_epoch_metrics.jsonl"
    diagnostic_summary_path = victim_dir / "miasrec_diagnostic_summary.json"
    raw_topk_path = victim_dir / "miasrec_topk_raw.json"
    exporter = MiaSRecExporter()
    exporter.export_with_poisoned_train(
        ensure_canonical_dataset(config),
        poisoned_sessions=poisoned.sessions,
        poisoned_labels=poisoned.labels,
        output_dir=export_root,
        dataset_name=config.data.dataset_name,
    )
    runner = MiaSRecRunner(config)
    train_config = dict(config.victims.params["miasrec"]["train"])
    epochs = int(max_epochs if max_epochs is not None else train_config["epochs"])
    run_info = runner.run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=victim_dir,
        export_topk_path=raw_topk_path,
        topk=max(config.evaluation.topk),
        max_epochs=epochs,
        victim_train_seed=victim_effective_train_seed(
            config,
            victim_name="miasrec",
            run_type=VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE,
            target_item=source.target_item,
        ),
        diagnostic_epoch_metrics_path=epoch_metrics_path,
        diagnostic_summary_path=diagnostic_summary_path,
    )
    rows = load_miasrec_epoch_metrics(epoch_metrics_path)
    _write_json_and_csv(rows, out_dir / "miasrec_epoch_metrics")
    diag_summary = _load_json_object(diagnostic_summary_path)
    checkpoint_path = _optional_str(diag_summary.get("selected_checkpoint_path"))
    summary = summarize_epoch_metrics(
        rows,
        victim_name="miasrec",
        primary_metric=str(diagnostic_config.get("miasrec_primary_metric", "mrr@20")),
        checkpoint_path=checkpoint_path,
        checkpoint_selection_mode="recbole_validation_best",
        source=source,
        max_epochs=epochs,
        extra={
            "dataset": config.data.dataset_name,
            "valid_metric": diag_summary.get("valid_metric", "MRR@20"),
            "load_best_model_for_final_evaluation": bool(
                diag_summary.get("load_best_model", True)
            ),
            "used_best_checkpoint_for_export": True,
            "run_info": run_info,
        },
    )
    save_json(summary, out_dir / "miasrec_valbest_summary.json")
    return summary


def _run_tron_diagnostic(
    config: Config,
    *,
    source: SourcePTSArtifact,
    poisoned: PoisonedDataset,
    out_dir: Path,
    max_epochs: int | None,
    diagnostic_config: Mapping[str, Any],
) -> dict[str, Any]:
    victim_dir = out_dir / "tron"
    export_root = victim_dir / "export"
    diagnostic_summary_path = victim_dir / "tron_diagnostic_summary.json"
    raw_topk_path = victim_dir / "tron_topk_raw.json"
    exporter = TRONExporter()
    exporter.export_with_raw_poisoned_train(
        ensure_canonical_dataset(config),
        raw_fake_sessions=source.sessions,
        output_dir=export_root,
        dataset_name=config.data.dataset_name,
    )
    runner = TRONRunner(config)
    train_config = dict(config.victims.params["tron"]["train"])
    epochs = int(max_epochs if max_epochs is not None else train_config["max_epochs"])
    run_info = runner.run(
        export_root=export_root,
        dataset_name=config.data.dataset_name,
        run_dir=victim_dir,
        export_topk_path=raw_topk_path,
        topk=max(config.evaluation.topk),
        max_epochs=epochs,
        victim_train_seed=victim_effective_train_seed(
            config,
            victim_name="tron",
            run_type=VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE,
            target_item=source.target_item,
        ),
        diagnostic_summary_path=diagnostic_summary_path,
    )
    rows = load_tron_epoch_metrics(run_info["log_dir"], max_epochs=epochs)
    _write_json_and_csv(rows, out_dir / "tron_epoch_metrics")
    diag_summary = _load_json_object(diagnostic_summary_path)
    checkpoint_path = _optional_str(diag_summary.get("best_model_path"))
    compared = bool(diag_summary.get("best_checkpoint_validation"))
    summary = summarize_epoch_metrics(
        rows,
        victim_name="tron",
        primary_metric=str(diagnostic_config.get("tron_primary_metric", "recall@20")),
        checkpoint_path=checkpoint_path,
        checkpoint_selection_mode="lightning_model_checkpoint_recall_cutoff_20",
        source=source,
        max_epochs=epochs,
        extra={
            "dataset": config.data.dataset_name,
            "checkpoint_callback_best_model_score": _optional_float(
                diag_summary.get("best_model_score")
            ),
            "formal_export_behavior": "last_model",
            "diagnostic_compared_best_checkpoint": compared,
            "used_best_checkpoint_for_formal_export": False,
            "run_info": run_info,
        },
    )
    save_json(summary, out_dir / "tron_valbest_summary.json")
    return summary


def _run_freqrec_diagnostic(
    config: Config,
    *,
    out_dir: Path,
    effective_epochs: int,
) -> dict[str, Any]:
    requested_topk = max(max(config.evaluation.topk), 20)
    train_config = dict(config.victims.params["freqrec"]["train"])
    metric_cutoffs = sorted(set(int(k) for k in train_config["metric_cutoffs"]) | {20})
    if requested_topk < max(metric_cutoffs):
        raise ValueError(
            "FreqRec diagnostic top-k must cover all parent diagnostic metric cutoffs."
        )
    runtime = (config.victims.runtime or {}).get("freqrec", {})
    diagnostics = runtime.get("diagnostics", {}) if isinstance(runtime, Mapping) else {}
    if not (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("epoch_metrics") is True
        and diagnostics.get("per_epoch_predictions") is True
    ):
        raise ValueError(
            "FreqRec diagnostic execution requires victims.runtime.freqrec.diagnostics "
            "epoch_metrics=true and per_epoch_predictions=true."
        )
    canonical = ensure_canonical_dataset(config)
    clean_prefixes, clean_labels = build_clean_pairs(canonical)
    victim_dir = out_dir
    export = FreqRecExporter().export_with_train_pairs(
        canonical,
        train_prefixes=clean_prefixes,
        train_labels=clean_labels,
        output_dir=victim_dir / "export",
        dataset_name=config.data.dataset_name,
        max_seq_length=int(train_config["max_seq_length"]),
        mode="clean",
    )
    epochs = int(effective_epochs)
    seed = victim_effective_train_seed(
        config,
        victim_name="freqrec",
        run_type="clean",
        target_item=0,
    )
    runner = FreqRecRunner(config)
    run_info = runner.run(
        train_path=export.files["train"],
        valid_path=export.files["valid"],
        test_path=export.files["test"],
        metadata_path=export.files["metadata"],
        item_count=export.item_count,
        expected_test_count=export.test_example_count,
        run_dir=victim_dir,
        prediction_output_path=victim_dir / "freqrec_topk_raw.json",
        requested_topk=requested_topk,
        epochs=epochs,
        victim_train_seed=seed,
        target_item=None,
    )
    validation_labels: list[int] = []
    for session in canonical.valid:
        validation_labels.extend(expand_session_to_samples(session)[1])
    rows = summarize_freqrec_epoch_diagnostics(
        runner=runner,
        epoch_metrics_path=Path(run_info["epoch_metrics_output_path"]),
        per_epoch_prediction_dir=Path(run_info["per_epoch_prediction_dir"]),
        validation_labels=validation_labels,
        item_count=export.item_count,
        requested_topk=requested_topk,
        configured_epochs=epochs,
        seed=seed,
        metric_cutoffs=metric_cutoffs,
    )
    summary = {
        "victim_name": "freqrec",
        "victim": "freqrec",
        "dataset": config.data.dataset_name,
        "target_item": None,
        "diagnostic_scope": "dataset_victim_clean",
        "epochs": rows,
        "run_info": run_info,
        "selection_uses_test_metrics": False,
    }
    save_json(summary, out_dir / "freqrec_valbest_summary.json")
    return summary


def _freqrec_diagnostic_dir(config: Config, *, effective_epochs: int) -> Path:
    return (
        runs_root(config)
        / VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE
        / freqrec_diagnostic_key(config, effective_epochs=effective_epochs)
        / "dataset_victim"
        / "freqrec"
    )


def _build_poisoned_train(config: Config, candidate_sessions: Sequence[Sequence[int]]) -> PoisonedDataset:
    canonical = ensure_canonical_dataset(config)
    clean_sessions, clean_labels = build_clean_pairs(canonical)
    return build_poisoned_dataset(clean_sessions, clean_labels, candidate_sessions)


def _diagnostic_target_dir(config: Config, source: SourcePTSArtifact) -> Path:
    context = {
        "victim_valbest_epoch_diagnostic": {
            "source_sessions_sha1": source.sessions_sha1,
            "source_candidate_rank": int(source.candidate_rank),
            "source_candidate_key": source.source_candidate_key,
        }
    }
    return (
        target_dir(
            config,
            source.target_item,
            run_type=VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE,
            attack_identity_context=context,
        )
        / "victim_valbest_epoch_diagnostic"
    )


def _candidate_artifact_dirs(
    config: Config,
    *,
    target_item: int,
    source_run: str | Path | None,
) -> list[Path]:
    if source_run is not None and str(source_run).strip():
        source = Path(str(source_run))
        if not source.exists():
            source = Path(config.artifacts.root) / config.artifacts.runs_dir / config.data.dataset_name / str(source_run)
        return _find_artifact_dirs_under(source, target_item=target_item)
    experiment_root = Path(config.artifacts.root) / config.artifacts.runs_dir / config.data.dataset_name / config.experiment.name
    return _find_artifact_dirs_under(experiment_root, target_item=target_item)


def _find_artifact_dirs_under(root: Path, *, target_item: int) -> list[Path]:
    if not root.exists():
        return []
    candidates: set[Path] = set()
    direct = root / "top_candidates"
    if root.name == "pts_construction_cem" and direct.exists():
        candidates.add(root.resolve())
    target_relative = root / "targets" / str(int(target_item)) / "pts_construction_cem"
    if target_relative.exists():
        candidates.add(target_relative.resolve())
    for path in root.glob(f"*/targets/{int(target_item)}/pts_construction_cem"):
        if path.exists():
            candidates.add(path.resolve())
    for path in root.glob(f"**/targets/{int(target_item)}/pts_construction_cem"):
        if path.exists():
            candidates.add(path.resolve())
    return sorted(candidates, key=lambda path: str(path))


def _merge_source_metadata(
    *,
    rank: int,
    metadata: Mapping[str, Any],
    policy: Mapping[str, Any],
    complete_marker: Mapping[str, Any],
    top_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(metadata)
    if "rank" not in merged:
        merged["rank"] = int(rank)
    best_candidate = complete_marker.get("best_candidate")
    if isinstance(best_candidate, Mapping):
        for key in ("rank", "candidate_id", "candidate_seed", "iteration", "reward", "reward_metrics"):
            if key not in merged and key in best_candidate:
                merged[key] = best_candidate[key]
    for candidate in top_candidates.get("candidates", []) if isinstance(top_candidates.get("candidates"), list) else []:
        if not isinstance(candidate, Mapping):
            continue
        if int(candidate.get("rank", -1)) == int(rank) or candidate.get("candidate_key") == merged.get("candidate_key"):
            for key, value in candidate.items():
                merged.setdefault(key, value)
            break
    if policy and "policy" not in merged:
        merged["policy"] = dict(policy)
    if complete_marker:
        merged.setdefault("target_item", complete_marker.get("target_item"))
        merged.setdefault("surrogate_victim_seed_aligned", complete_marker.get("surrogate_victim_seed_aligned"))
        identity = complete_marker.get("identity")
        if isinstance(identity, Mapping):
            merged.setdefault("identity", dict(identity))
    return merged


def _extract_raw_lowk(metadata: Mapping[str, Any]) -> float | None:
    for container in (metadata.get("reward_metrics"), metadata):
        if isinstance(container, Mapping):
            value = container.get("raw_lowk_mrr_recall_10_20")
            parsed = _optional_float(value)
            if parsed is not None:
                return parsed
    return None


def _extract_enabled_actions(metadata: Mapping[str, Any]) -> list[str] | None:
    candidates = [
        _nested_get(metadata, ("policy", "enabled_actions")),
        metadata.get("enabled_actions"),
        _nested_get(metadata, ("sample_metadata", "policy", "enabled_actions")),
    ]
    for value in candidates:
        if isinstance(value, list):
            return [str(item) for item in value]
    return None


def _write_json_and_csv(rows: Sequence[Mapping[str, Any]], path_without_suffix: Path) -> None:
    save_json([dict(row) for row in rows], path_without_suffix.with_suffix(".json"))
    _write_rows_csv(rows, path_without_suffix.with_suffix(".csv"))


def _write_rows_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    preferred = [
        "dataset",
        "target_item",
        "victim_name",
        "source_pts_cem_run",
        "source_candidate_rank",
        "source_candidate_key",
        "source_sessions_sha1",
        "max_epochs",
        "primary_metric",
        "best_epoch",
        "best_metric_value",
        "best_epoch_by_mrr20",
        "best_epoch_by_recall20",
        "last_epoch",
        "last_epoch_metric_value",
        "best_vs_last_delta",
        "selected_checkpoint_path",
        "checkpoint_selection_mode",
        "formal_export_behavior",
        "diagnostic_compared_best_checkpoint",
        "used_best_checkpoint_for_formal_export",
    ]
    flat_rows = [_flatten_summary(row) for row in rows]
    all_keys = list(preferred)
    for row in flat_rows:
        for key in row:
            if key not in all_keys:
                all_keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in flat_rows:
            writer.writerow({key: row.get(key) for key in all_keys})


def _flatten_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list)):
            flat[key] = json.dumps(value, sort_keys=True)
        else:
            flat[key] = value
    return flat


def _normalize_metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["epoch"] = int(row["epoch"])
    if "mrr@20" not in normalized:
        normalized["mrr@20"] = _optional_float(row.get("mrr_cutoff_20"))
    if "recall@20" not in normalized:
        normalized["recall@20"] = _optional_float(row.get("recall_cutoff_20"))
    return normalized


def _best_row(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any] | None:
    available = [row for row in rows if _optional_float(row.get(metric)) is not None]
    if not available:
        return None
    return max(available, key=lambda row: (_optional_float(row.get(metric)), -int(row["epoch"])))


def _metric_from_mapping(payload: Mapping[str, Any], path: Sequence[str], key: str) -> float | None:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    if not isinstance(current, Mapping):
        return None
    normalized_key = key.lower()
    for item_key, value in current.items():
        if str(item_key).lower() == normalized_key:
            return _optional_float(value)
    return None


def _find_latest_metrics_csv(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = list(log_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _first_float_value(row: Mapping[str, str], keys: Sequence[str]) -> float | None:
    lower = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        parsed = _try_float(lower.get(str(key).lower()))
        if parsed is not None:
            return parsed
    return None


def _try_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    return _try_float(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _validate_candidate_rank(candidate_rank: int) -> int:
    rank = int(candidate_rank)
    if rank <= 0:
        raise ValueError("candidate_rank must be positive.")
    return rank


def _check_optional_equal(
    metadata: Mapping[str, Any],
    key: str,
    expected: Any,
    warnings: list[str],
    *,
    missing: str,
) -> None:
    if metadata.get(key) is None:
        warnings.append(missing)
        return
    actual = metadata[key]
    if isinstance(expected, int):
        try:
            actual = int(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PTS-CEM source {key} is invalid: {metadata[key]!r}") from exc
    if actual != expected:
        raise ValueError(
            f"PTS-CEM source {key} mismatch: expected {expected}, found {actual}."
        )


def _nested_get(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _resolve_relative_artifact_path(root: Path, value: Any, *, fallback: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_json_sessions(path: str | Path) -> list[list[int]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"PTS-CEM sessions must be a list: {path}")
    sessions: list[list[int]] = []
    for session in payload:
        if not isinstance(session, list):
            raise ValueError(f"PTS-CEM sessions must contain lists: {path}")
        sessions.append([int(item) for item in session])
    return sessions


def _sha1_file(path: str | Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_run_root(artifact_dir: Path) -> Path:
    parts = artifact_dir.parts
    if "targets" in parts:
        index = parts.index("targets")
        return Path(*parts[:index])
    return artifact_dir


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping.")
    return payload


def _diagnostic_source_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    source = raw_config.get("source_attack", {})
    if source is None:
        return {}
    if not isinstance(source, Mapping):
        raise TypeError("source_attack must be a mapping.")
    return dict(source)


def _diagnostic_options(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = raw_config.get("diagnostic", {})
    if diagnostic is None:
        return {}
    if not isinstance(diagnostic, Mapping):
        raise TypeError("diagnostic must be a mapping.")
    result = dict(diagnostic)
    primary = result.get("primary_metric")
    if isinstance(primary, Mapping):
        result["miasrec_primary_metric"] = str(primary.get("miasrec", "mrr@20"))
        result["tron_primary_metric"] = str(primary.get("tron", "recall@20"))
    return result


def _target_from_config(config: Config) -> int:
    if config.targets.explicit_list:
        return int(config.targets.explicit_list[0])
    return DEFAULT_TARGET_ITEM


def _selected_victims(config: Config, victim: str) -> list[str]:
    requested = str(victim).strip().lower()
    if requested == "all":
        selected = [
            name
            for name in config.victims.enabled
            if name in {"miasrec", "tron", "freqrec"}
        ]
    elif requested in {"miasrec", "tron", "freqrec"}:
        selected = [requested]
    else:
        raise ValueError("victim must be one of: miasrec, tron, freqrec, all")
    if not selected:
        raise ValueError("No supported victims selected for diagnostic.")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-item", type=int, default=None)
    parser.add_argument(
        "--victim", choices=["miasrec", "tron", "freqrec", "all"], default="all"
    )
    parser.add_argument("--source-pts-cem-run", default=None)
    parser.add_argument("--candidate-rank", type=int, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_diagnostic(
        args.config,
        target_item=args.target_item,
        victim=args.victim,
        source_pts_cem_run=args.source_pts_cem_run,
        candidate_rank=args.candidate_rank,
        experiment_name=args.experiment_name,
        max_epochs=args.max_epochs,
        force=bool(args.force),
    )
    print(json.dumps(result["source"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_EXPECTED_CANDIDATE_KEY",
    "EXPECTED_ACTIONS",
    "SOURCE_METHOD_NOTE",
    "SourcePTSArtifact",
    "VICTIM_VALBEST_EPOCH_DIAGNOSTIC_RUN_TYPE",
    "load_miasrec_epoch_metrics",
    "load_tron_epoch_metrics",
    "resolve_source_pts_artifact",
    "run_diagnostic",
    "source_identity_payload",
    "summarize_epoch_metrics",
    "validate_source_metadata",
]
