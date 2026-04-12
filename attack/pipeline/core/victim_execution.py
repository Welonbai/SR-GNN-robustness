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
            max_epochs=attack_epochs,
        )
        _save_miasrec_history(run_dir, Path(run_info["log_path"]))
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
        _save_tron_history(run_dir, Path(run_info["log_dir"]))
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


def _extract_loss_from_log(log_path: Path) -> dict[str, object]:
    import re

    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    train_pattern = re.compile(rf"\\btrain[_ ]loss\\b\\s*[:=]\\s*({number})", re.IGNORECASE)
    valid_pattern = re.compile(
        rf"\\b(?:valid|validation|val|eval|test)[_ ]loss\\b\\s*[:=]\\s*({number})",
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


__all__ = ["VictimExecutionResult", "execute_single_victim"]
