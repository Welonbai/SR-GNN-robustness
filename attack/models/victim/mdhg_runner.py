from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim
from attack.models.victim.subprocess_progress import (
    resolve_subprocess_gpu_selector,
    run_subprocess_with_epoch_progress,
)


class MDHGRunner(VictimRunnerBase):
    name = "mdhg"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        runtime = _require_runtime_config(config, self.name)
        self.train_config = _require_train_config(config, self.name)
        self.python_executable = str(runtime["python_executable"])
        self.repo_root = Path(repo_root) if repo_root is not None else Path(runtime["repo_root"])
        self.working_dir = Path(runtime["working_dir"])
        self.device_config = dict(runtime["device"])
        diagnostics = runtime.get("diagnostics", {})
        self.diagnostics_config = dict(diagnostics) if isinstance(diagnostics, dict) else {}

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("MDHG datasets are loaded by the subprocess.")

    def train(self, *args, **kwargs):
        return self.run(**kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("MDHG evaluation is handled inside the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("MDHG does not expose per-session scoring.")

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("MDHG model loading is not supported.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("MDHG model saving is not supported.")

    def run(
        self,
        *,
        data_dir: Path,
        dataset_name: str,
        n_node: int,
        expected_test_count: int,
        run_dir: Path,
        export_topk_path: Path,
        topk: int,
        max_epochs: int | None = None,
        victim_train_seed: int | None = None,
        target_item: int | None = None,
    ) -> dict[str, str | int | bool | None]:
        main_path = self.repo_root / "main.py"
        for label, path in (
            ("repository", self.repo_root),
            ("working directory", self.working_dir),
            ("entrypoint", main_path),
            ("dataset directory", Path(data_dir)),
        ):
            if not path.exists():
                raise FileNotFoundError(f"MDHG {label} not found: {path}")

        effective_epochs = int(max_epochs or self.train_config["epochs"])
        effective_seed = int(
            victim_train_seed
            if victim_train_seed is not None
            else self.config.seeds.victim_train_seed
        )
        gpu_id = str(self.device_config["gpu_id"]).strip()
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(effective_seed)
        subprocess_gpu_id = resolve_subprocess_gpu_selector(gpu_id, env)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "mdhg_stdout.log"
        diagnostics_dir = run_dir / "diagnostics"
        epoch_metrics_path = diagnostics_dir / "mdhg_epoch_metrics.jsonl"
        per_epoch_prediction_dir = diagnostics_dir / "per_epoch_predictions"
        epoch_metrics_enabled = bool(self.diagnostics_config.get("epoch_metrics", False))
        per_epoch_predictions_enabled = bool(
            self.diagnostics_config.get("per_epoch_predictions", False)
        )

        cmd = [
            self.python_executable,
            str(main_path.resolve()),
            "--dataset",
            dataset_name,
            "--data_dir",
            str(Path(data_dir).resolve()),
            "--n_node",
            str(int(n_node)),
            "--epoch",
            str(effective_epochs),
            "--batchSize",
            str(int(self.train_config["batch_size"])),
            "--lr",
            str(float(self.train_config["lr"])),
            "--gpu_id",
            subprocess_gpu_id,
            "--seed",
            str(effective_seed),
            "--topk",
            str(int(topk)),
            "--prediction_output_path",
            str(export_topk_path.resolve()),
        ]
        if epoch_metrics_enabled:
            cmd.extend(
                [
                    "--epoch_metrics_output_path",
                    str(epoch_metrics_path.resolve()),
                ]
            )
        if per_epoch_predictions_enabled:
            cmd.extend(
                [
                    "--per_epoch_prediction_dir",
                    str(per_epoch_prediction_dir.resolve()),
                ]
            )
        result = run_subprocess_with_epoch_progress(
            cmd,
            cwd=self.working_dir,
            env=env,
            log_path=log_path,
            model_name=self.name,
            target_item=target_item,
            total_epochs=effective_epochs,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MDHG subprocess failed with code {result.returncode}. See log: {log_path}"
            )
        rankings = self.predict_topk(
            predictions_path=export_topk_path,
            expected_test_count=expected_test_count,
            n_node=n_node,
            requested_topk=topk,
        )
        effective_topk = len(rankings[0]) if rankings else min(int(topk), int(n_node))
        run_info: dict[str, str | int | bool | None] = {
            "returncode": int(result.returncode),
            "log_path": str(log_path),
            "export_topk_path": str(export_topk_path),
            "python_executable": self.python_executable,
            "repo_root": str(self.repo_root),
            "working_dir": str(self.working_dir),
            "data_dir": str(Path(data_dir)),
            "n_node": int(n_node),
            "requested_topk": int(topk),
            "effective_topk": int(effective_topk),
            "expected_test_count": int(expected_test_count),
            "gpu_id": gpu_id,
            "victim_train_seed": effective_seed,
            "victim_name": self.name,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
            "epochs_configured": effective_epochs,
            "epochs_completed": effective_epochs,
            "used_best_checkpoint_for_export": False,
            "selected_checkpoint_epoch": effective_epochs,
            "selected_checkpoint_path": None,
            "validation_metrics_recorded": False,
            "prediction_count": len(rankings),
        }
        if epoch_metrics_enabled:
            run_info["epoch_metrics_output_path"] = str(epoch_metrics_path.resolve())
        if per_epoch_predictions_enabled:
            run_info["per_epoch_prediction_dir"] = str(per_epoch_prediction_dir.resolve())
        return run_info

    def predict_topk(
        self,
        *,
        predictions_path: Path,
        expected_test_count: int,
        n_node: int,
        requested_topk: int,
        topk: int | None = None,
    ) -> list[list[int]]:
        payload = _load_prediction_payload(
            predictions_path,
            expected_test_count=expected_test_count,
            n_node=n_node,
            requested_topk=requested_topk,
        )
        rankings = payload["rankings"]
        if topk is not None:
            return [row[: int(topk)] for row in rankings]
        return rankings


def _load_prediction_payload(
    predictions_path: Path,
    *,
    expected_test_count: int,
    n_node: int,
    requested_topk: int,
) -> dict[str, Any]:
    path = Path(predictions_path)
    if not path.exists():
        raise RuntimeError(f"MDHG did not export top-k predictions. Missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rankings = payload.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("MDHG predictions JSON must contain a rankings list.")
    if len(rankings) != int(expected_test_count):
        raise ValueError(
            f"MDHG prediction count mismatch: {len(rankings)} vs {expected_test_count} expected."
        )
    actual_topk = payload.get("topk")
    if not isinstance(actual_topk, int) or actual_topk <= 0:
        raise ValueError("MDHG predictions JSON topk must be a positive integer.")
    expected_effective_topk = min(int(requested_topk), int(n_node))
    if actual_topk != expected_effective_topk:
        raise ValueError(
            f"MDHG JSON topk {actual_topk} does not match requested effective topk "
            f"{expected_effective_topk}."
        )
    if payload.get("requested_topk") != int(requested_topk):
        raise ValueError("MDHG JSON requested_topk does not match the pipeline request.")
    if payload.get("n_node") != int(n_node):
        raise ValueError("MDHG JSON n_node does not match the exported canonical universe.")

    normalized: list[list[int]] = []
    for row_index, row in enumerate(rankings):
        if not isinstance(row, list) or len(row) != actual_topk:
            raise ValueError(
                f"MDHG ranking row {row_index} length does not match JSON topk {actual_topk}."
            )
        normalized_row = [int(item) for item in row]
        if any(item < 1 or item > int(n_node) for item in normalized_row):
            raise ValueError(
                f"MDHG ranking row {row_index} contains item IDs outside 1..{n_node}."
            )
        normalized.append(normalized_row)
    payload["rankings"] = normalized
    return payload


def _require_runtime_config(config: Config, victim_name: str) -> dict[str, Any]:
    runtime = (config.victims.runtime or {}).get(victim_name)
    if not isinstance(runtime, dict):
        raise ValueError(f"Missing victims.runtime.{victim_name} configuration.")
    return dict(runtime)


def _require_train_config(config: Config, victim_name: str) -> dict[str, Any]:
    params = config.victims.params.get(victim_name)
    train = params.get("train") if isinstance(params, dict) else None
    if not isinstance(train, dict):
        raise ValueError(f"Missing victims.params.{victim_name}.train configuration.")
    return dict(train)


register_victim(MDHGRunner.name, MDHGRunner)


__all__ = ["MDHGRunner"]
