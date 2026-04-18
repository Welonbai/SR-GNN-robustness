from __future__ import annotations

import os
from pathlib import Path
import json
import shutil
import subprocess
from typing import Any

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim


class MiaSRecRunner(VictimRunnerBase):
    name = "miasrec"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        runtime = _require_runtime_config(config, self.name)
        train_config = _require_train_config(config, self.name)
        self.python_executable = runtime["python_executable"]
        self.repo_root = Path(repo_root) if repo_root is not None else Path(runtime["repo_root"])
        self.working_dir = Path(runtime["working_dir"])
        self.train_config = train_config
        self.device_config = dict(runtime["device"])
        self.logging_config = dict(runtime["logging"])

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec runner does not load datasets directly.")

    def train(self, *args, **kwargs):
        export_root = kwargs.get("export_root")
        dataset_name = kwargs.get("dataset_name")
        run_dir = kwargs.get("run_dir")
        export_topk_path = kwargs.get("export_topk_path")
        topk = kwargs.get("topk")
        max_epochs = kwargs.get("max_epochs")
        victim_train_seed = kwargs.get("victim_train_seed")
        if (
            export_root is None
            or dataset_name is None
            or run_dir is None
            or export_topk_path is None
            or topk is None
        ):
            raise ValueError(
                "train() requires export_root, dataset_name, run_dir, export_topk_path, and topk."
            )
        return self.run(
            export_root=Path(export_root),
            dataset_name=str(dataset_name),
            run_dir=Path(run_dir),
            export_topk_path=Path(export_topk_path),
            topk=int(topk),
            max_epochs=int(max_epochs) if max_epochs is not None else None,
            victim_train_seed=(
                int(victim_train_seed) if victim_train_seed is not None else None
            ),
        )

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec evaluation is handled inside the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec does not expose per-session scoring.")

    def predict_topk(self, *, predictions_path: Path, topk: int | None = None) -> list[list[int]]:
        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"MiaSRec predictions not found: {predictions_path}")
        with predictions_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rankings = payload.get("rankings")
        if rankings is None:
            raise ValueError("MiaSRec predictions file missing rankings.")
        if topk is not None:
            rankings = [list(map(int, row[:topk])) for row in rankings]
        else:
            rankings = [list(map(int, row)) for row in rankings]
        return rankings

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec model loading is not supported.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec model saving is not supported.")

    def run(
        self,
        *,
        export_root: Path,
        dataset_name: str,
        run_dir: Path,
        export_topk_path: Path,
        topk: int,
        max_epochs: int | None = None,
        victim_train_seed: int | None = None,
    ) -> dict[str, str | int]:
        if not self.repo_root.exists():
            raise FileNotFoundError(f"MiaSRec repository not found: {self.repo_root}")
        if not self.working_dir.exists():
            raise FileNotFoundError(f"MiaSRec working directory not found: {self.working_dir}")
        main_path = self.repo_root / "main.py"
        if not main_path.exists():
            raise FileNotFoundError(f"MiaSRec entrypoint missing: {main_path}")

        if not export_root.exists():
            raise FileNotFoundError(f"MiaSRec export root missing: {export_root}")
        dataset_dir = export_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"MiaSRec dataset directory missing: {dataset_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "miasrec_stdout.log"
        checkpoint_dir = run_dir / "miasrec_checkpoints"
        log_root = self.repo_root / "log"
        tensorboard_root = self.repo_root / "log_tensorboard"
        saved_root = self.repo_root / "saved"
        log_snapshot = _snapshot_files(log_root)
        tb_snapshot = _snapshot_files(tensorboard_root)
        saved_snapshot = _snapshot_files(saved_root)
        effective_epochs = int(max_epochs) if max_epochs is not None else int(self.train_config["epochs"])
        requested_gpu_id = _resolve_requested_gpu_id(self.device_config)
        effective_seed = (
            int(victim_train_seed)
            if victim_train_seed is not None
            else int(self.config.seeds.victim_train_seed)
        )

        override_path = run_dir / "miasrec_override.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        with override_path.open("w", encoding="utf-8") as handle:
            handle.write(f"epochs: {effective_epochs}\n")
            handle.write(f"use_gpu: {json.dumps(bool(self.device_config['use_gpu']))}\n")
            handle.write(f"gpu_id: {json.dumps(requested_gpu_id)}\n")
            handle.write(f"seed: {effective_seed}\n")
            handle.write("reproducibility: true\n")
            handle.write(
                f"show_progress: {json.dumps(bool(self.logging_config['show_progress']))}\n"
            )
            handle.write(
                f"train_batch_size: {int(self.train_config['train_batch_size'])}\n"
            )
            handle.write(
                f"eval_batch_size: {int(self.train_config['eval_batch_size'])}\n"
            )
            handle.write(f"export_topk_k: {int(topk)}\n")
            handle.write(f"export_topk_path: {json.dumps(str(export_topk_path.resolve()))}\n")

        cmd = [
            self.python_executable,
            str(main_path.resolve()),
            "--model",
            "miasrec",
            "--dataset",
            dataset_name,
            "--config2",
            str(override_path.resolve()),
            "--data_path",
            str(dataset_dir.resolve()),
            "--checkpoint_dir",
            str(checkpoint_dir.resolve()),
        ]
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(effective_seed)
        if bool(self.device_config["use_gpu"]):
            env["CUDA_VISIBLE_DEVICES"] = requested_gpu_id
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        print(f"[VictimRunner] launching {self.name}")
        print(f"python_executable={self.python_executable}")
        print(f"repo_root={self.repo_root}")
        print(f"working_dir={self.working_dir}")
        print(f"[miasrec] Starting subprocess. Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        _move_new_files(log_root, log_snapshot, run_dir / "miasrec_logs")
        _move_new_files(tensorboard_root, tb_snapshot, run_dir / "miasrec_tensorboard")
        _move_new_files(saved_root, saved_snapshot, run_dir / "miasrec_saved")

        if result.returncode != 0:
            raise RuntimeError(
                f"MiaSRec subprocess failed with code {result.returncode}. "
                f"See log: {log_path}"
            )
        if not export_topk_path.exists():
            raise RuntimeError(
                "MiaSRec did not export top-k predictions. "
                f"Missing: {export_topk_path}"
            )

        print(f"[miasrec] Completed. Predictions: {export_topk_path}")
        return {
            "returncode": int(result.returncode),
            "log_path": str(log_path),
            "config_path": str(override_path),
            "checkpoint_dir": str(checkpoint_dir),
            "export_topk_path": str(export_topk_path),
            "victim_train_seed": int(effective_seed),
        }


def _require_runtime_config(config: Config, victim_name: str) -> dict[str, Any]:
    runtime = (config.victims.runtime or {}).get(victim_name)
    if runtime is None:
        raise ValueError(f"Missing victims.runtime.{victim_name} configuration.")
    missing = [key for key in ("python_executable", "repo_root", "working_dir") if not runtime.get(key)]
    if missing:
        joined = ", ".join(f"victims.runtime.{victim_name}.{key}" for key in missing)
        raise ValueError(f"Missing required runtime configuration: {joined}")
    return dict(runtime)


def _require_train_config(config: Config, victim_name: str) -> dict[str, Any]:
    params = config.victims.params.get(victim_name)
    if params is None:
        raise ValueError(f"Missing victims.params.{victim_name} configuration.")
    train = params.get("train")
    if not isinstance(train, dict):
        raise ValueError(f"Missing victims.params.{victim_name}.train configuration.")
    return dict(train)


def _resolve_requested_gpu_id(device_config: dict[str, Any]) -> str:
    if not bool(device_config.get("use_gpu", False)):
        return ""
    return str(device_config["gpu_id"]).strip()


def _snapshot_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


def _move_new_files(root: Path, before: set[Path], dest_root: Path) -> None:
    if not root.exists():
        return
    after = {path for path in root.rglob("*") if path.is_file()}
    new_files = [path for path in after if path not in before]
    if not new_files:
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    for path in new_files:
        relative = path.relative_to(root)
        target = dest_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    _prune_empty_dirs(root)


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                continue


register_victim(MiaSRecRunner.name, MiaSRecRunner)


__all__ = ["MiaSRecRunner"]
