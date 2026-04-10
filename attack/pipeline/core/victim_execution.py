from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from attack.common.config import Config
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.miasrec_exporter import MiaSRecExporter
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.data.exporters.tron_exporter import TRONExporter
from attack.models.victim.registry import get_victim_runner
from attack.pipeline.core.evaluator import save_predictions
from attack.pipeline.core.pipeline_utils import build_default_opt


@dataclass(frozen=True)
class VictimExecutionResult:
    predictions: list[list[int]] | None
    predictions_path: Path | None
    extra: dict[str, object]
    poisoned_train_path: Path | None


def execute_single_victim(
    config: Config,
    *,
    victim_name: str,
    canonical_dataset: CanonicalDataset,
    poisoned_sessions: Sequence[Sequence[int]],
    poisoned_labels: Sequence[int],
    run_dir: Path,
    poisoned_train_path: Path,
    target_item: int,
    attack_epochs: int,
    eval_topk: Sequence[int],
    srg_nn_export_paths: dict[str, Path] | None = None,
    predictions_path: Path | None = None,
) -> VictimExecutionResult:
    max_topk = max(eval_topk)
    if victim_name == "srgnn":
        if srg_nn_export_paths is None:
            raise ValueError("SRGNN execution requires clean export paths for valid/test.")
        exporter = SRGNNExporter()
        poisoned_train_path = exporter.export_train_pairs(
            poisoned_sessions,
            poisoned_labels,
            poisoned_train_path,
        )

        victim_cls = get_victim_runner(victim_name)
        attacked_runner = victim_cls(config)
        attacked_runner.build_model(build_default_opt(attack_epochs))
        attacked_train_data, attacked_valid_data = attacked_runner.load_dataset(
            train_path=poisoned_train_path,
            test_path=srg_nn_export_paths["valid"],
        )
        if attack_epochs > 0:
            attacked_runner.train(
                attacked_train_data,
                attacked_valid_data,
                attack_epochs,
                target_item=target_item,
                topk=max_topk,
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
        print(f"[victim:miasrec] Exporting dataset to {export_root}")
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
        print(f"[victim:miasrec] Running MiaSRec, log at {run_dir / 'miasrec_stdout.log'}")
        run_info = runner.run(
            export_root=export_root,
            dataset_name=config.data.dataset_name,
            run_dir=run_dir,
            export_topk_path=raw_predictions_path,
            topk=max_topk,
        )
        rankings = runner.predict_topk(predictions_path=raw_predictions_path, topk=max_topk)
        print(f"[victim:miasrec] Completed. Predictions: {raw_predictions_path}")
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
        print(f"[victim:tron] Exporting dataset to {export_root}")
        tron_export = TRONExporter()
        export_result = tron_export.export_with_poisoned_train(
            canonical_dataset,
            poisoned_sessions=poisoned_sessions,
            poisoned_labels=poisoned_labels,
            output_dir=export_root,
            dataset_name=config.data.dataset_name,
        )
        runner = get_victim_runner(victim_name)(config)
        raw_predictions_path = run_dir / "tron_topk_raw.json"
        print(f"[victim:tron] Running TRON, log at {run_dir / 'tron_stdout.log'}")
        run_info = runner.run(
            export_root=export_root,
            dataset_name=config.data.dataset_name,
            run_dir=run_dir,
            export_topk_path=raw_predictions_path,
            topk=max_topk,
            max_epochs=attack_epochs,
        )
        rankings = runner.predict_topk(predictions_path=raw_predictions_path, topk=max_topk)
        print(f"[victim:tron] Completed. Predictions: {raw_predictions_path}")
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

    raise ValueError(f"Unsupported victim model: {victim_name}")


__all__ = ["VictimExecutionResult", "execute_single_victim"]
