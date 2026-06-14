from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.artifact_io import save_json
from attack.common.config import Config
from attack.common.seed import derive_seed, set_seed
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.miasrec_exporter import MiaSRecExporter
from attack.data.exporters.mdhg_exporter import MDHGExporter
from attack.data.exporters.freqrec_exporter import FreqRecExporter
from attack.data.exporters.wearec_exporter import WEARecExporter, WEARecExportResult
from attack.data.canonical_fingerprints import (
    CANONICAL_FINGERPRINT_SEMANTICS,
    ITEM_VOCABULARY_FINGERPRINT_SEMANTICS,
    fingerprint_exported_jsonl,
    fingerprint_item_vocabulary,
    load_exported_canonical_labels,
    resolve_wearec_repository_provenance,
)
from attack.common.paths import attack_key_payload, classify_victim_training_run_type
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.exporters.tron_exporter import TRONExporter
from attack.common.srgnn_training_protocol import srgnn_validation_best_enabled
from attack.models.srgnn_validation_training import (
    srgnn_validation_train_history_extra,
    train_srgnn_validation_best,
)
from attack.models.victim.registry import get_victim_runner
from attack.models.victim.mdhg_diagnostics import summarize_mdhg_epoch_diagnostics
from attack.models.victim.wearec_runner import (
    WEAREC_ARTIFACT_CONTRACT_VERSION,
    WEAREC_RUNNER_SEMANTICS_VERSION,
    effective_wearec_config,
)
from attack.pipeline.core.evaluator import save_predictions
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from attack.pipeline.core.train_history import save_train_history


@dataclass(frozen=True)
class VictimExecutionResult:
    predictions: list[list[int]] | None
    predictions_path: Path | None
    extra: dict[str, object]
    poisoned_train_path: Path | None


def execute_single_victim(
    config: Config,
    *,
    run_type: str,
    victim_name: str,
    canonical_dataset: CanonicalDataset,
    poisoned_sessions: Sequence[Sequence[int]],
    poisoned_labels: Sequence[int],
    raw_fake_sessions: Sequence[Sequence[int]],
    run_dir: Path,
    poisoned_train_path: Path,
    target_item: int,
    eval_topk: Sequence[int],
    srg_nn_export_paths: dict[str, Path] | None = None,
    predictions_path: Path | None = None,
    prepared_wearec: Mapping[str, Any] | None = None,
) -> VictimExecutionResult:
    victim_stage_seed = _victim_stage_seed(
        config,
        victim_name=victim_name,
        run_type=run_type,
        target_item=target_item,
    )
    set_seed(victim_stage_seed)
    max_topk = max(eval_topk)
    if victim_name == "srgnn":
        victim_train_config = _require_victim_train_config(config, victim_name)
        victim_epochs = int(victim_train_config["epochs"])
        if srg_nn_export_paths is None:
            raise ValueError("SRGNN execution requires clean export paths for valid/test.")
        exporter = SRGNNExporter()
        poisoned_train_path = exporter.export_train_pairs(
            poisoned_sessions,
            poisoned_labels,
            poisoned_train_path,
        )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={
                "export_topk_k": int(max_topk),
                "predictions_path": predictions_path,
                "poisoned_train_path": poisoned_train_path,
                "run_dir": run_dir,
                "victim_train_seed": int(victim_stage_seed),
            },
        )

        victim_cls = get_victim_runner(victim_name)
        attacked_runner = victim_cls(config)
        attacked_runner.build_model(build_srgnn_opt_from_train_config(victim_train_config))
        attacked_train_data, attacked_valid_data = attacked_runner.load_dataset(
            train_path=poisoned_train_path,
            test_path=srg_nn_export_paths["valid"],
        )
        if srgnn_validation_best_enabled(victim_train_config):
            result = train_srgnn_validation_best(
                attacked_runner,
                attacked_train_data,
                attacked_valid_data,
                train_config=victim_train_config,
                max_epochs=victim_epochs,
                patience=int(victim_train_config["patience"]),
                best_checkpoint_path=run_dir / "best_validation.pt",
                log_prefix="[victim:srgnn-validation-best]",
            )
            save_train_history(
                run_dir / "train_history.json",
                role="victim",
                model="srgnn",
                epochs=len(result.rows),
                train_loss=[float(row["train_loss"]) for row in result.rows],
                valid_loss=[None] * len(result.rows),
                notes=(
                    "SRGNN victim training selected the checkpoint with highest "
                    "validation ground-truth MRR@20. Test metrics were not used."
                ),
                extra=srgnn_validation_train_history_extra(result),
            )
        elif victim_epochs > 0:
            attacked_runner.train(
                attacked_train_data,
                attacked_valid_data,
                victim_epochs,
                target_item=(None if run_type == "clean" else target_item),
                topk=max_topk,
            )
            if attacked_runner.train_loss_history:
                save_train_history(
                    run_dir / "train_history.json",
                    role="victim",
                    model="srgnn",
                    epochs=len(attacked_runner.train_loss_history),
                    train_loss=attacked_runner.train_loss_history,
                    valid_loss=[None] * len(attacked_runner.train_loss_history),
                    notes="valid_loss not available for SRGNN victim training.",
                )

        _, attacked_test_data = attacked_runner.load_dataset(
            train_path=poisoned_train_path,
            test_path=srg_nn_export_paths["test"],
            shuffle_train=False,
        )
        rankings = attacked_runner.predict_topk(attacked_test_data, topk=max_topk)
        if predictions_path is not None:
            save_predictions(
                predictions_path,
                topk=max_topk,
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={},
            poisoned_train_path=poisoned_train_path,
        )

    if victim_name == "miasrec":
        export_root = run_dir / "export" / "miasrec"
        miasrec_export = MiaSRecExporter()
        export_result = miasrec_export.export_with_poisoned_train(
            canonical_dataset,
            poisoned_sessions=poisoned_sessions,
            poisoned_labels=poisoned_labels,
            output_dir=export_root,
            dataset_name=config.data.dataset_name,
        )
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "miasrec_topk_raw.json"
        victim_train_config = _require_victim_train_config(config, victim_name)
        pipeline_injected = {
            "export_root": export_root,
            "export_topk_k": int(max_topk),
            "export_topk_path": raw_predictions_path,
            "run_dir": run_dir,
            "checkpoint_dir": run_dir / "miasrec_checkpoints",
            "log_path": run_dir / "miasrec_stdout.log",
            "victim_train_seed": int(victim_stage_seed),
        }
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected=pipeline_injected,
        )
        run_info = runner.run(
            export_root=export_root,
            dataset_name=config.data.dataset_name,
            run_dir=run_dir,
            export_topk_path=raw_predictions_path,
            topk=max_topk,
            max_epochs=int(victim_train_config["epochs"]),
            victim_train_seed=int(victim_stage_seed),
            target_item=int(target_item),
        )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={**pipeline_injected, **run_info},
        )
        _save_miasrec_history(run_dir, Path(run_info["log_path"]))
        rankings = runner.predict_topk(predictions_path=raw_predictions_path, topk=max_topk)
        if predictions_path is not None:
            save_predictions(
                predictions_path,
                topk=max_topk,
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={
                "miasrec": run_info,
                "miasrec_export": {key: str(path) for key, path in export_result.files.items()},
            },
            poisoned_train_path=None,
        )

    if victim_name == "tron":
        export_root = run_dir / "export" / "tron"
        tron_export = TRONExporter()
        export_result = tron_export.export_with_raw_poisoned_train(
            canonical_dataset,
            raw_fake_sessions=raw_fake_sessions,
            output_dir=export_root,
            dataset_name=config.data.dataset_name,
        )
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "tron_topk_raw.json"
        victim_train_config = _require_victim_train_config(config, victim_name)
        pipeline_injected = {
            "export_root": export_root,
            "export_topk_k": int(max_topk),
            "export_topk_path": raw_predictions_path,
            "run_dir": run_dir,
            "config_dir": run_dir / "tron_config",
            "log_path": run_dir / "tron_stdout.log",
            "log_dir": run_dir / "tron_logs",
            "victim_train_seed": int(victim_stage_seed),
        }
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected=pipeline_injected,
        )
        run_info = runner.run(
            export_root=export_root,
            dataset_name=config.data.dataset_name,
            run_dir=run_dir,
            export_topk_path=raw_predictions_path,
            topk=max_topk,
            max_epochs=int(victim_train_config["max_epochs"]),
            victim_train_seed=int(victim_stage_seed),
            target_item=int(target_item),
        )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={**pipeline_injected, **run_info},
        )
        _save_tron_history(run_dir, Path(run_info["log_dir"]))
        rankings = runner.predict_topk(predictions_path=raw_predictions_path, topk=max_topk)
        if predictions_path is not None:
            save_predictions(
                predictions_path,
                topk=max_topk,
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={
                "tron": run_info,
                "tron_export": {key: str(path) for key, path in export_result.files.items()},
            },
            poisoned_train_path=None,
        )

    if victim_name == "mdhg":
        export_root = run_dir / "export" / "mdhg"
        export_result = MDHGExporter().export_with_poisoned_train(
            canonical_dataset,
            poisoned_sessions=poisoned_sessions,
            poisoned_labels=poisoned_labels,
            raw_fake_sessions=raw_fake_sessions,
            output_dir=export_root,
            dataset_name=config.data.dataset_name,
        )
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "mdhg_topk_raw.json"
        epoch_pipeline_metrics_path = run_dir / "mdhg_epoch_pipeline_metrics.jsonl"
        victim_train_config = _require_victim_train_config(config, victim_name)
        pipeline_injected = {
            "data_dir": export_result.data_dir,
            "n_node": int(export_result.n_node),
            "expected_test_count": int(export_result.test_example_count),
            "export_topk_k": int(max_topk),
            "export_topk_path": raw_predictions_path,
            "run_dir": run_dir,
            "log_path": run_dir / "mdhg_stdout.log",
            "victim_train_seed": int(victim_stage_seed),
            "train_pairs_match_raw_expansion": export_result.train_pairs_match_raw_expansion,
            "target_item": int(target_item),
            "evaluation_topk": [int(k) for k in eval_topk],
            "targeted_metrics": list(config.evaluation.targeted_metrics),
            "ground_truth_metrics": list(config.evaluation.ground_truth_metrics),
            "mdhg_test_data_path": export_result.files["test"].resolve(),
            "epoch_pipeline_metrics_output_path": epoch_pipeline_metrics_path.resolve(),
        }
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected=pipeline_injected,
        )
        run_info = runner.run(
            data_dir=export_result.data_dir,
            dataset_name=config.data.dataset_name,
            n_node=export_result.n_node,
            expected_test_count=export_result.test_example_count,
            run_dir=run_dir,
            export_topk_path=raw_predictions_path,
            topk=max_topk,
            max_epochs=int(victim_train_config["epochs"]),
            victim_train_seed=int(victim_stage_seed),
            target_item=int(target_item),
        )
        per_epoch_prediction_dir = run_info.get("per_epoch_prediction_dir")
        if per_epoch_prediction_dir is not None:
            summarize_mdhg_epoch_diagnostics(
                run_dir,
                target_item=int(target_item),
                evaluation_topk=eval_topk,
                targeted_metrics=config.evaluation.targeted_metrics,
                ground_truth_metrics=config.evaluation.ground_truth_metrics,
                test_data_path=export_result.files["test"],
                expected_test_count=export_result.test_example_count,
                n_node=export_result.n_node,
                requested_topk=max_topk,
                per_epoch_prediction_dir=Path(per_epoch_prediction_dir),
                output_path=epoch_pipeline_metrics_path,
            )
            run_info["epoch_pipeline_metrics_output_path"] = str(
                epoch_pipeline_metrics_path
            )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={**pipeline_injected, **run_info},
        )
        _save_mdhg_history(run_dir, Path(run_info["log_path"]))
        rankings = runner.predict_topk(
            predictions_path=raw_predictions_path,
            expected_test_count=export_result.test_example_count,
            n_node=export_result.n_node,
            requested_topk=max_topk,
            topk=max_topk,
        )
        if predictions_path is not None:
            effective_topk = len(rankings[0]) if rankings else min(max_topk, export_result.n_node)
            save_predictions(
                predictions_path,
                topk=effective_topk,
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        export_metadata = {
            "data_dir": str(export_result.data_dir),
            "n_node": export_result.n_node,
            "train_example_count": export_result.train_example_count,
            "test_example_count": export_result.test_example_count,
            "raw_train_session_count": export_result.raw_train_session_count,
            "observed_max_item_id": export_result.observed_max_item_id,
            "expected_raw_expanded_pair_count": export_result.expected_raw_expanded_pair_count,
            "train_pairs_match_raw_expansion": export_result.train_pairs_match_raw_expansion,
            "files": {key: str(path) for key, path in export_result.files.items()},
        }
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={"mdhg": run_info, "mdhg_export": export_metadata},
            poisoned_train_path=None,
        )

    if victim_name == "freqrec":
        victim_train_config = _require_victim_train_config(config, victim_name)
        export_root = run_dir / "export" / "freqrec"
        export_result = FreqRecExporter().export_with_train_pairs(
            canonical_dataset,
            train_prefixes=poisoned_sessions,
            train_labels=poisoned_labels,
            output_dir=export_root,
            dataset_name=config.data.dataset_name,
            max_seq_length=int(victim_train_config["max_seq_length"]),
            mode=("clean" if run_type == "clean" else "poisoned"),
        )
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "freqrec_topk_raw.json"
        pipeline_injected = {
            "data_dir": export_result.data_dir,
            "item_count": export_result.item_count,
            "expected_test_count": export_result.test_example_count,
            "export_topk_k": int(max_topk),
            "export_topk_path": raw_predictions_path,
            "run_dir": run_dir,
            "log_path": run_dir / "freqrec_stdout.log",
            "victim_train_seed": int(victim_stage_seed),
        }
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected=pipeline_injected,
        )
        run_info = runner.run(
            train_path=export_result.files["train"],
            valid_path=export_result.files["valid"],
            test_path=export_result.files["test"],
            metadata_path=export_result.files["metadata"],
            item_count=export_result.item_count,
            expected_test_count=export_result.test_example_count,
            run_dir=run_dir,
            prediction_output_path=raw_predictions_path,
            requested_topk=max_topk,
            epochs=int(victim_train_config["epochs"]),
            victim_train_seed=int(victim_stage_seed),
            target_item=(None if run_type == "clean" else int(target_item)),
        )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={**pipeline_injected, **run_info},
        )
        rankings = runner.predict_topk(
            predictions_path=raw_predictions_path,
            item_count=export_result.item_count,
            expected_example_count=export_result.test_example_count,
            requested_topk=max_topk,
            configured_epochs=int(victim_train_config["epochs"]),
            seed=int(victim_stage_seed),
        )
        if predictions_path is not None:
            save_predictions(
                predictions_path,
                topk=min(max_topk, export_result.item_count),
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        export_metadata = {
            "data_dir": str(export_result.data_dir),
            "item_count": export_result.item_count,
            "max_seq_length": export_result.max_seq_length,
            "train_example_count": export_result.train_example_count,
            "valid_example_count": export_result.valid_example_count,
            "test_example_count": export_result.test_example_count,
            "observed_max_item_id": export_result.observed_max_item_id,
            "files": {key: str(path) for key, path in export_result.files.items()},
        }
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={"freqrec": run_info, "freqrec_export": export_metadata},
            poisoned_train_path=None,
        )

    if victim_name == "wearec":
        prepared = dict(prepared_wearec or {})
        export_result = prepared.get("export_result")
        identity = prepared.get("identity")
        if not isinstance(export_result, WEARecExportResult) or not isinstance(
            identity, Mapping
        ):
            raise ValueError("WEARec execution requires a prepared export and identity.")
        effective = identity["effective_config"]
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "wearec_topk_raw.json"
        run_info = runner.run(
            train_path=export_result.files["train"],
            valid_path=export_result.files["valid"],
            test_path=export_result.files["test"],
            metadata_path=export_result.files["metadata"],
            item_count=export_result.item_count,
            expected_test_count=export_result.test_example_count,
            run_dir=run_dir,
            prediction_output_path=raw_predictions_path,
            requested_topk=int(max_topk),
            epochs=int(effective["epochs"]),
            victim_train_seed=int(effective["seed"]),
            target_item=(
                None if identity["training_mode"] == "clean" else int(target_item)
            ),
            training_mode=str(identity["training_mode"]),
            dataset_name=str(identity["dataset_name"]),
        )
        _write_victim_resolved_config(
            config,
            victim_name,
            run_dir,
            pipeline_injected={
                **run_info,
                "scientific_identity": dict(identity),
                "data_dir": export_result.data_dir,
            },
        )
        rankings = runner.predict_topk(
            predictions_path=raw_predictions_path,
            item_count=export_result.item_count,
            expected_labels=prepared["test_labels"],
            requested_topk=int(max_topk),
            configured_epochs=int(effective["epochs"]),
            seed=int(effective["seed"]),
            expected_training_mode=str(identity["training_mode"]),
            expected_dataset_name=str(identity["dataset_name"]),
        )
        if predictions_path is not None:
            save_predictions(
                predictions_path,
                topk=int(max_topk),
                rankings=rankings,
                victim=victim_name,
                target_item=target_item,
            )
        return VictimExecutionResult(
            predictions=rankings,
            predictions_path=predictions_path,
            extra={
                "wearec": {**run_info, "scientific_identity": dict(identity)},
                "wearec_export": {
                    "data_dir": str(export_result.data_dir),
                    "item_count": export_result.item_count,
                    "max_seq_length": export_result.max_seq_length,
                    "train_example_count": export_result.train_example_count,
                    "valid_example_count": export_result.valid_example_count,
                    "test_example_count": export_result.test_example_count,
                    "files": {
                        key: str(path) for key, path in export_result.files.items()
                    },
                },
            },
            poisoned_train_path=None,
        )

    raise ValueError(f"Unsupported victim model: {victim_name}")


def victim_effective_train_seed(
    config: Config,
    *,
    victim_name: str,
    run_type: str,
    target_item: int,
) -> int:
    if run_type == "clean":
        return derive_seed(
            config.seeds.victim_train_seed,
            "victim_train",
            victim_name,
        )
    return derive_seed(
        config.seeds.victim_train_seed,
        "victim_train",
        victim_name,
        int(target_item),
    )


def prepare_wearec_execution(
    config: Config,
    *,
    run_type: str,
    canonical_dataset: CanonicalDataset,
    train_prefixes: Sequence[Sequence[int]],
    train_labels: Sequence[int],
    run_dir: Path,
    requested_topk: int,
    target_item: int,
    attack_identity_context: Mapping[str, Any] | None,
    provenance_resolver=resolve_wearec_repository_provenance,
) -> dict[str, Any]:
    training_mode = classify_victim_training_run_type(run_type)
    if training_mode == "unsupported":
        raise ValueError(f"Unsupported WEARec victim-training run type: {run_type}")
    seed = victim_effective_train_seed(
        config,
        victim_name="wearec",
        run_type=run_type,
        target_item=target_item,
    )
    effective = effective_wearec_config(
        config,
        seed=seed,
        requested_topk=requested_topk,
    )
    export_result = WEARecExporter().export_with_train_pairs(
        canonical_dataset,
        train_prefixes=train_prefixes,
        train_labels=train_labels,
        output_dir=run_dir / "export" / "wearec",
        dataset_name=config.data.dataset_name,
        max_seq_length=int(effective["max_seq_length"]),
        mode=training_mode,
    )
    if requested_topk > export_result.item_count:
        raise ValueError("WEARec requested_topk must not exceed exported item_count.")
    if any(value > export_result.item_count for value in effective["metric_cutoffs"]):
        raise ValueError("WEARec metric cutoffs must not exceed exported item_count.")
    valid_labels = load_exported_canonical_labels(export_result.files["valid"])
    test_labels = load_exported_canonical_labels(export_result.files["test"])
    if len(valid_labels) != export_result.valid_example_count:
        raise ValueError("WEARec validation count does not match exported JSONL.")
    if len(test_labels) != export_result.test_example_count:
        raise ValueError("WEARec test count does not match exported JSONL.")
    provenance = provenance_resolver(
        Path(__file__).resolve().parents[3],
        Path((config.victims.runtime or {})["wearec"]["repo_root"]),
    )
    identity: dict[str, Any] = {
        "dataset_name": config.data.dataset_name,
        "dataset_variant": canonical_dataset.metadata.get("variant", "full"),
        "ordered_exported_train_jsonl_sha256": fingerprint_exported_jsonl(
            export_result.files["train"]
        ),
        "ordered_exported_valid_jsonl_sha256": fingerprint_exported_jsonl(
            export_result.files["valid"]
        ),
        "ordered_exported_test_jsonl_sha256": fingerprint_exported_jsonl(
            export_result.files["test"]
        ),
        "item_vocabulary_fingerprint": fingerprint_item_vocabulary(
            canonical_dataset.item_map
        ),
        "fingerprint_semantics": CANONICAL_FINGERPRINT_SEMANTICS,
        "item_vocabulary_fingerprint_semantics": ITEM_VOCABULARY_FINGERPRINT_SEMANTICS,
        "item_count": export_result.item_count,
        "victim": "wearec",
        "training_mode": training_mode,
        "checkpoint_protocol": "fixed_epoch",
        "effective_config": effective,
        "canonical_exporter_semantics": export_result.exporter_semantics,
        **provenance,
        "wearec_runner_semantics_version": WEAREC_RUNNER_SEMANTICS_VERSION,
        "wearec_artifact_contract_version": WEAREC_ARTIFACT_CONTRACT_VERSION,
    }
    if training_mode == "poisoned":
        identity.update(
            {
                "attack_identity": attack_key_payload(
                    config,
                    run_type=run_type,
                    attack_identity_context=attack_identity_context,
                ),
                "target_item": int(target_item),
                "poison_budget": float(config.attack.size),
                "effective_poisoned_seed": int(seed),
            }
        )
    return {
        "export_result": export_result,
        "identity": identity,
        "valid_labels": valid_labels,
        "test_labels": test_labels,
    }


def _victim_stage_seed(
    config: Config,
    *,
    victim_name: str,
    run_type: str,
    target_item: int,
) -> int:
    return victim_effective_train_seed(
        config,
        victim_name=victim_name,
        run_type=run_type,
        target_item=target_item,
    )


def _require_victim_train_config(config: Config, victim_name: str) -> dict[str, object]:
    params = config.victims.params.get(victim_name)
    if params is None:
        raise ValueError(f"Missing victims.params.{victim_name} configuration.")
    train = params.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"Missing victims.params.{victim_name}.train configuration.")
    return dict(train)


def _write_victim_resolved_config(
    config: Config,
    victim_name: str,
    run_dir: Path,
    *,
    pipeline_injected: dict[str, object],
) -> None:
    runtime = (config.victims.runtime or {}).get(victim_name, {})
    payload = {
        "victim_name": victim_name,
        "seeds": {
            "victim_train_seed": int(config.seeds.victim_train_seed),
        },
        "params": _primitive_value(config.victims.params.get(victim_name, {})),
        "runtime": _primitive_value(runtime),
        "pipeline_injected": _primitive_value(pipeline_injected),
    }
    save_json(payload, run_dir / "resolved_config.json")


def _primitive_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _primitive_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive_value(item) for item in value]
    return value


def _save_miasrec_history(run_dir: Path, log_path: Path) -> None:
    try:
        history = _extract_loss_from_log(log_path)
    except OSError:
        return
    if history["epochs"] == 0:
        return
    save_train_history(
        run_dir / "train_history.json",
        role="victim",
        model="miasrec",
        epochs=history["epochs"],
        train_loss=history["train_loss"],
        valid_loss=history["valid_loss"],
        notes=history.get("notes"),
    )


def _save_tron_history(run_dir: Path, log_dir: Path) -> None:
    history = _extract_loss_from_metrics_csv(log_dir)
    if history["epochs"] == 0:
        return
    save_train_history(
        run_dir / "train_history.json",
        role="victim",
        model="tron",
        epochs=history["epochs"],
        train_loss=history["train_loss"],
        valid_loss=history["valid_loss"],
        notes=history.get("notes"),
    )


def _save_mdhg_history(run_dir: Path, log_path: Path) -> None:
    history = _extract_loss_from_log(log_path)
    if history["epochs"] == 0:
        return
    save_train_history(
        run_dir / "train_history.json",
        role="victim",
        model="mdhg",
        epochs=history["epochs"],
        train_loss=history["train_loss"],
        valid_loss=history["valid_loss"],
        notes=history.get("notes"),
    )


def _extract_loss_from_log(log_path: Path) -> dict[str, object]:
    import re

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    train_pattern = re.compile(rf"\btrain[_ ]loss\b\s*[:=]\s*({number})", re.IGNORECASE)
    valid_pattern = re.compile(
        rf"\b(?:valid|validation|val|eval|test)[_ ]loss\b\s*[:=]\s*({number})",
        re.IGNORECASE,
    )
    train_loss: list[float] = []
    valid_loss: list[float] = []
    if not log_path.exists():
        return {"epochs": 0, "train_loss": [], "valid_loss": [], "notes": "log not found"}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = train_pattern.search(line)
            if match:
                train_loss.append(float(match.group(1)))
            match = valid_pattern.search(line)
            if match:
                valid_loss.append(float(match.group(1)))

    epochs = max(len(train_loss), len(valid_loss))
    if epochs == 0:
        return {
            "epochs": 0,
            "train_loss": [],
            "valid_loss": [],
            "notes": "loss not found in log",
        }
    if len(train_loss) < epochs:
        train_loss.extend([None] * (epochs - len(train_loss)))
    if len(valid_loss) < epochs:
        valid_loss.extend([None] * (epochs - len(valid_loss)))
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }


def _extract_loss_from_metrics_csv(log_dir: Path) -> dict[str, object]:
    import csv

    metrics_path = _find_latest_metrics_csv(log_dir)
    if metrics_path is None:
        return {
            "epochs": 0,
            "train_loss": [],
            "valid_loss": [],
            "notes": "metrics.csv not found",
        }
    train_by_epoch: dict[int, float] = {}
    valid_by_epoch: dict[int, float] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {
                "epochs": 0,
                "train_loss": [],
                "valid_loss": [],
                "notes": "metrics.csv missing header",
            }
        field_map = {name.lower(): name for name in reader.fieldnames}

        def _first_float(row: dict[str, str], keys: list[str]) -> float | None:
            for key in keys:
                original = field_map.get(key)
                if original is None:
                    continue
                value = row.get(original)
                if value is None or value == "":
                    continue
                try:
                    return float(value)
                except ValueError:
                    continue
            return None

        for row in reader:
            epoch_value = _first_float(row, ["epoch"])
            if epoch_value is None:
                continue
            epoch = int(epoch_value)
            train_value = _first_float(
                row,
                ["train_loss", "train/loss", "loss/train", "training_loss"],
            )
            if train_value is not None:
                train_by_epoch[epoch] = train_value
            valid_value = _first_float(
                row,
                [
                    "val_loss",
                    "valid_loss",
                    "test_loss",
                    "val/loss",
                    "loss/val",
                    "valid/loss",
                    "loss/valid",
                    "test/loss",
                    "loss/test",
                ],
            )
            if valid_value is not None:
                valid_by_epoch[epoch] = valid_value

    epochs = 0
    if train_by_epoch:
        epochs = max(epochs, max(train_by_epoch) + 1)
    if valid_by_epoch:
        epochs = max(epochs, max(valid_by_epoch) + 1)
    if epochs == 0:
        return {
            "epochs": 0,
            "train_loss": [],
            "valid_loss": [],
            "notes": "no loss values in metrics.csv",
        }
    train_loss = [train_by_epoch.get(i) for i in range(epochs)]
    valid_loss = [valid_by_epoch.get(i) for i in range(epochs)]
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "valid_loss": valid_loss,
    }


def _find_latest_metrics_csv(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = list(log_dir.rglob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


__all__ = [
    "VictimExecutionResult",
    "execute_single_victim",
    "prepare_wearec_execution",
    "victim_effective_train_seed",
]
