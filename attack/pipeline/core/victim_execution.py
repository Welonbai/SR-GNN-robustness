from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from attack.common.config import Config
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.miasrec_exporter import MiaSRecExporter
from attack.data.exporters.srgnn_exporter import SRGNNExporter
from attack.models.victim.registry import get_victim_runner
from attack.pipeline.core.evaluator import evaluate_runner, evaluate_targeted_precision_at_k
from attack.pipeline.core.pipeline_utils import build_default_opt


@dataclass(frozen=True)
class VictimExecutionResult:
    metrics: dict[str, object] | None
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
    eval_topk: int,
    srg_nn_export_paths: dict[str, Path] | None = None,
) -> VictimExecutionResult:
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
                topk=eval_topk,
            )

        _, attacked_test_data = attacked_runner.load_dataset(
            train_path=poisoned_train_path,
            test_path=srg_nn_export_paths["test"],
            shuffle_train=False,
        )
        metrics = evaluate_runner(attacked_runner, attacked_test_data, topk=eval_topk)
        targeted = evaluate_targeted_precision_at_k(
            attacked_runner,
            attacked_test_data,
            target_item=target_item,
            topk=eval_topk,
        )
        metrics["targeted_precision_at_k"] = float(targeted)
        return VictimExecutionResult(
            metrics=metrics,
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
        run_info = runner.run(
            export_root=export_root,
            dataset_name=config.data.dataset_name,
            run_dir=run_dir,
        )
        return VictimExecutionResult(
            metrics=None,
            extra={
                "miasrec": run_info,
                "miasrec_export": {key: str(path) for key, path in export_result.files.items()},
            },
            poisoned_train_path=None,
        )

    raise ValueError(f"Unsupported victim model: {victim_name}")


__all__ = ["VictimExecutionResult", "execute_single_victim"]
