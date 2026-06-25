from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim
from attack.models.victim.subprocess_progress import run_subprocess_with_epoch_progress


class DTGATRunner(VictimRunnerBase):
    name = "dtgat"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        self.train_config = _require_train_config(config)
        runtime = _require_runtime_config(config)
        self.python_executable = str(runtime["python_executable"])
        self.repo_root = Path(repo_root) if repo_root is not None else Path(runtime["repo_root"])
        self.working_dir = Path(runtime["working_dir"])
        self.device_config = dict(runtime["device"])

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("DT-GAT datasets are loaded by the subprocess.")

    def train(self, *args, **kwargs):
        return self.run(**kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("DT-GAT evaluation is handled inside the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("DT-GAT does not expose per-session scoring.")

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("DT-GAT model loading is not supported.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("DT-GAT model saving is not supported.")

    def build_command(
        self,
        *,
        data_dir: Path,
        dataset_name: str,
        n_node: int,
        prediction_output_path: Path,
        metrics_output_path: Path,
        resolved_config_output_path: Path,
        requested_topk: int,
        epochs: int,
        seed: int,
    ) -> tuple[list[str], dict[str, str]]:
        main_path = self.repo_root / "main.py"
        gpu_id = _single_gpu_id(self.device_config["gpu_id"])
        use_gpu = bool(self.device_config["use_gpu"])
        train = self.train_config
        cmd = [
            self.python_executable,
            str(main_path.resolve()),
            "--dataset",
            str(dataset_name),
            "--data_dir",
            str(Path(data_dir).resolve()),
            "--n_node",
            str(int(n_node)),
            "--epoch",
            str(int(epochs)),
            "--batchSize",
            str(int(train["batch_size"])),
            "--embSize",
            str(int(train["emb_size"])),
            "--time_dims",
            str(int(train["time_dims"])),
            "--intent_num",
            str(int(train["intent_num"])),
            "--lr",
            str(float(train["lr"])),
            "--dropout",
            str(float(train["dropout"])),
            "--l2",
            str(float(train["l2"])),
            "--lr_dc",
            str(float(train["lr_dc"])),
            "--lr_dc_step",
            str(int(train["lr_dc_step"])),
            "--layer",
            str(int(train["layer"])),
            "--beta",
            str(float(train["beta"])),
            "--topk",
            str(int(requested_topk)),
            "--prediction_output_path",
            str(Path(prediction_output_path).resolve()),
            "--metrics_output_path",
            str(Path(metrics_output_path).resolve()),
            "--resolved_config_output_path",
            str(Path(resolved_config_output_path).resolve()),
            "--seed",
            str(int(seed)),
            "--gpu_id",
            gpu_id,
            "--cuda",
            "true" if use_gpu else "false",
        ]
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(int(seed))
        if use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        return cmd, env

    def run(
        self,
        *,
        data_dir: Path,
        dataset_name: str,
        n_node: int,
        expected_test_count: int,
        run_dir: Path,
        prediction_output_path: Path,
        requested_topk: int,
        epochs: int,
        victim_train_seed: int,
        target_item: int | None,
    ) -> dict[str, Any]:
        main_path = self.repo_root / "main.py"
        for label, path, kind in (
            ("python executable", Path(self.python_executable), "file"),
            ("repository", self.repo_root, "directory"),
            ("working directory", self.working_dir, "directory"),
            ("entrypoint", main_path, "file"),
        ):
            if not path.is_absolute():
                raise ValueError(f"DT-GAT {label} must be an absolute path: {path}")
            valid = path.is_file() if kind == "file" else path.is_dir()
            if not valid:
                raise FileNotFoundError(f"DT-GAT {label} not found: {path}")
        if not Path(data_dir).is_dir():
            raise FileNotFoundError(f"DT-GAT dataset directory not found: {data_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "dtgat_stdout.log"
        metrics_output_path = run_dir / "dtgat_third_party_metrics.json"
        resolved_config_output_path = run_dir / "dtgat_third_party_resolved_config.json"
        cmd, env = self.build_command(
            data_dir=data_dir,
            dataset_name=dataset_name,
            n_node=n_node,
            prediction_output_path=prediction_output_path,
            metrics_output_path=metrics_output_path,
            resolved_config_output_path=resolved_config_output_path,
            requested_topk=requested_topk,
            epochs=epochs,
            seed=victim_train_seed,
        )
        result = run_subprocess_with_epoch_progress(
            cmd,
            cwd=self.working_dir,
            env=env,
            log_path=log_path,
            model_name=self.name,
            target_item=target_item,
            total_epochs=epochs,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"DT-GAT subprocess failed with code {result.returncode}. See log: {log_path}"
            )
        rankings = self.predict_topk(
            predictions_path=prediction_output_path,
            expected_test_count=expected_test_count,
            n_node=n_node,
            requested_topk=requested_topk,
        )
        effective_topk = len(rankings[0]) if rankings else min(int(requested_topk), int(n_node))
        return {
            "returncode": int(result.returncode),
            "prediction_output_path": str(prediction_output_path),
            "metrics_output_path": str(metrics_output_path),
            "third_party_resolved_config_output_path": str(resolved_config_output_path),
            "log_path": str(log_path),
            "python_executable": self.python_executable,
            "repo_root": str(self.repo_root),
            "working_dir": str(self.working_dir),
            "data_dir": str(Path(data_dir)),
            "n_node": int(n_node),
            "requested_topk": int(requested_topk),
            "effective_topk": int(effective_topk),
            "expected_test_count": int(expected_test_count),
            "gpu_id": _single_gpu_id(self.device_config["gpu_id"]),
            "use_gpu": bool(self.device_config["use_gpu"]),
            "victim_train_seed": int(victim_train_seed),
            "victim_name": self.name,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
            "epochs_configured": int(epochs),
            "epochs_completed": int(epochs),
            "used_best_checkpoint_for_export": False,
            "selected_checkpoint_epoch": int(epochs),
            "selected_checkpoint_path": None,
            "validation_metrics_recorded": False,
            "prediction_count": len(rankings),
        }

    def predict_topk(
        self,
        *,
        predictions_path: Path,
        expected_test_count: int,
        n_node: int,
        requested_topk: int,
        topk: int | None = None,
    ) -> list[list[int]]:
        payload = load_dtgat_prediction_payload(
            predictions_path,
            expected_test_count=expected_test_count,
            n_node=n_node,
            requested_topk=requested_topk,
        )
        rankings = payload["rankings"]
        if topk is not None:
            return [row[: int(topk)] for row in rankings]
        return rankings


def load_dtgat_prediction_payload(
    predictions_path: Path,
    *,
    expected_test_count: int,
    n_node: int,
    requested_topk: int,
) -> dict[str, Any]:
    path = Path(predictions_path)
    if not path.is_file():
        raise RuntimeError(f"DT-GAT did not export top-k predictions. Missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("DT-GAT predictions JSON must be an object.")
    rankings = payload.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("DT-GAT predictions JSON must contain a rankings list.")
    if len(rankings) != int(expected_test_count):
        raise ValueError(
            f"DT-GAT prediction count mismatch: {len(rankings)} vs {expected_test_count} expected."
        )
    actual_topk = _exact_int(payload.get("topk"), "topk")
    if actual_topk <= 0:
        raise ValueError("DT-GAT predictions JSON topk must be a positive integer.")
    expected_effective_topk = min(int(requested_topk), int(n_node))
    if actual_topk != expected_effective_topk:
        raise ValueError(
            f"DT-GAT JSON topk {actual_topk} does not match requested effective topk "
            f"{expected_effective_topk}."
        )
    if _exact_int(payload.get("requested_topk"), "requested_topk") != int(requested_topk):
        raise ValueError("DT-GAT JSON requested_topk does not match the pipeline request.")
    if _exact_int(payload.get("n_node"), "n_node") != int(n_node):
        raise ValueError("DT-GAT JSON n_node does not match the exported item universe.")
    if "num_examples" in payload and _exact_int(payload["num_examples"], "num_examples") != int(expected_test_count):
        raise ValueError("DT-GAT JSON num_examples does not match expected test count.")

    normalized: list[list[int]] = []
    for row_index, row in enumerate(rankings):
        if not isinstance(row, list) or len(row) < actual_topk:
            raise ValueError(
                f"DT-GAT ranking row {row_index} length is shorter than JSON topk {actual_topk}."
            )
        normalized_row = [_exact_int(item, f"rankings[{row_index}][]") for item in row[:actual_topk]]
        if any(item < 1 or item > int(n_node) for item in normalized_row):
            raise ValueError(
                f"DT-GAT ranking row {row_index} contains item IDs outside 1..{n_node}."
            )
        if len(set(normalized_row)) != len(normalized_row):
            raise ValueError(f"DT-GAT ranking row {row_index} contains duplicate item IDs.")
        normalized.append(normalized_row)
    payload["rankings"] = normalized
    return payload


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"DT-GAT prediction artifact {field} must be an integer.")
    return int(value)


def _single_gpu_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("DT-GAT GPU ID must identify exactly one physical GPU.")
    text = str(value).strip()
    if not text or "," in text or not text.isdigit():
        raise ValueError("DT-GAT GPU ID must be one non-negative physical GPU integer.")
    return text


def _require_runtime_config(config: Config) -> dict[str, Any]:
    runtime = (config.victims.runtime or {}).get("dtgat")
    if not isinstance(runtime, dict):
        raise ValueError("Missing victims.runtime.dtgat configuration.")
    return dict(runtime)


def _require_train_config(config: Config) -> dict[str, Any]:
    params = config.victims.params.get("dtgat")
    train = params.get("train") if isinstance(params, dict) else None
    if not isinstance(train, dict):
        raise ValueError("Missing victims.params.dtgat.train configuration.")
    return dict(train)


register_victim(DTGATRunner.name, DTGATRunner)


__all__ = ["DTGATRunner", "load_dtgat_prediction_payload"]
