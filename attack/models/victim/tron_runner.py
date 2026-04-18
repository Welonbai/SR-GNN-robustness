from __future__ import annotations

from pathlib import Path
import json
import os
import subprocess
from typing import Any

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim


class TRONRunner(VictimRunnerBase):
    name = "tron"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        runtime = _require_runtime_config(config, self.name)
        train_config = _require_train_config(config, self.name)
        self.python_executable = runtime["python_executable"]
        self.repo_root = Path(repo_root) if repo_root is not None else Path(runtime["repo_root"])
        self.working_dir = Path(runtime["working_dir"])
        self.train_config = train_config
        self.device_config = dict(runtime["device"])
        self.dataloader_config = dict(runtime["dataloader"])

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("TRON runner does not load datasets directly.")

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
        raise NotImplementedError("TRON evaluation is handled inside the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("TRON does not expose per-session scoring.")

    def predict_topk(self, *, predictions_path: Path, topk: int | None = None) -> list[list[int]]:
        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"TRON predictions not found: {predictions_path}")
        with predictions_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rankings = payload.get("rankings")
        if rankings is None:
            raise ValueError("TRON predictions file missing rankings.")
        if topk is not None:
            rankings = [list(map(int, row[:topk])) for row in rankings]
        else:
            rankings = [list(map(int, row)) for row in rankings]
        return rankings

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("TRON model loading is not supported.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("TRON model saving is not supported.")

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
            raise FileNotFoundError(f"TRON repository not found: {self.repo_root}")
        if not self.working_dir.exists():
            raise FileNotFoundError(f"TRON working directory not found: {self.working_dir}")
        dataset_dir = export_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"TRON dataset directory missing: {dataset_dir}")

        base_config = _resolve_base_config(self.repo_root, dataset_name)
        with base_config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)

        config["dataset"] = dataset_name
        effective_seed = (
            int(victim_train_seed)
            if victim_train_seed is not None
            else int(self.config.seeds.victim_train_seed)
        )
        config["seed"] = int(effective_seed)
        config["export_topk_path"] = str(export_topk_path.resolve())
        config["export_topk_k"] = int(topk)
        config["log_dir"] = str((run_dir / "tron_logs").resolve())
        config["num_workers"] = int(self.dataloader_config["num_workers"])
        config["max_epochs"] = (
            int(max_epochs) if max_epochs is not None else int(self.train_config["max_epochs"])
        )
        config["accelerator"] = "gpu" if bool(self.device_config["use_gpu"]) else "cpu"

        run_dir.mkdir(parents=True, exist_ok=True)
        config_dir = run_dir / "tron_config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "tron_run.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)

        cmd = [
            self.python_executable,
            "-m",
            "src",
            "--config-filename",
            config_path.stem,
            "--config-dir",
            str(config_dir.resolve()),
            "--data-dir",
            str(export_root.resolve()),
        ]

        log_path = run_dir / "tron_stdout.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = _prepend_pythonpath(env.get("PYTHONPATH"), self.repo_root)
        env["PYTHONHASHSEED"] = str(effective_seed)
        if bool(self.device_config["use_gpu"]):
            env["CUDA_VISIBLE_DEVICES"] = str(self.device_config["gpu_id"]).strip()
        else:
            env.pop("CUDA_VISIBLE_DEVICES", None)

        print(f"[VictimRunner] launching {self.name}")
        print(f"python_executable={self.python_executable}")
        print(f"repo_root={self.repo_root}")
        print(f"working_dir={self.working_dir}")
        print(f"[tron] Starting subprocess. Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=run_dir,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        _remove_checkpoints(run_dir)

        if result.returncode != 0:
            raise RuntimeError(
                f"TRON subprocess failed with code {result.returncode}. "
                f"See log: {log_path}"
            )
        if not export_topk_path.exists():
            raise RuntimeError(
                "TRON did not export top-k predictions. "
                f"Missing: {export_topk_path}"
            )

        print(f"[tron] Completed. Predictions: {export_topk_path}")
        return {
            "returncode": int(result.returncode),
            "log_path": str(log_path),
            "export_topk_path": str(export_topk_path),
            "config_path": str(config_path),
            "log_dir": config["log_dir"],
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


def _resolve_base_config(tron_root: Path, dataset_name: str) -> Path:
    config_path = tron_root / "configs" / "tron" / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"TRON config not found for dataset '{dataset_name}': {config_path}"
        )
    return config_path


def _prepend_pythonpath(current: str | None, tron_root: Path) -> str:
    tron_value = str(tron_root.resolve())
    if not current:
        return tron_value
    return tron_value + os.pathsep + current


def _remove_checkpoints(run_dir: Path) -> None:
    for suffix in (".ckpt", ".onnx"):
        for path in run_dir.rglob(f"*{suffix}"):
            try:
                path.unlink()
            except OSError:
                continue


register_victim(TRONRunner.name, TRONRunner)


__all__ = ["TRONRunner"]
