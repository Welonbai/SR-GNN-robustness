from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim
from attack.models.victim.subprocess_progress import run_subprocess_with_epoch_progress
from attack.models.victim.wearec_diagnostics import validate_wearec_metrics
from attack.data.canonical_fingerprints import (
    file_provenance,
    load_exported_canonical_labels,
)


WEAREC_RUNNER_SEMANTICS_VERSION = 1
WEAREC_ARTIFACT_CONTRACT_VERSION = 1
WEAREC_CHECKPOINT_PROTOCOL = "fixed_epoch"
_MODEL_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "hidden_act",
    "hidden_dropout_prob",
    "initializer_range",
    "num_heads",
    "alpha",
)


def effective_wearec_config(
    config: Config,
    *,
    seed: int,
    requested_topk: int,
    epochs: int | None = None,
) -> dict[str, Any]:
    train = _require_train_config(config)
    cutoffs = [int(value) for value in train["metric_cutoffs"]]
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("WEARec metric_cutoffs must not contain duplicates.")
    cutoffs = sorted(cutoffs)
    requested = int(requested_topk)
    if requested < max(cutoffs):
        raise ValueError("WEARec requested_topk must cover every metric cutoff.")
    projection = {
        "checkpoint_protocol": WEAREC_CHECKPOINT_PROTOCOL,
        "epochs": int(train["epochs"] if epochs is None else epochs),
        "batch_size": int(train["batch_size"]),
        "seed": int(seed),
        "max_seq_length": int(train["max_seq_length"]),
        "requested_topk": requested,
        "metric_cutoffs": cutoffs,
        "hidden_size": int(train["hidden_size"]),
        "num_hidden_layers": int(train["num_hidden_layers"]),
        "hidden_act": str(train["hidden_act"]).strip().lower(),
        "hidden_dropout_prob": _finite(train["hidden_dropout_prob"], "hidden_dropout_prob"),
        "initializer_range": _finite(train["initializer_range"], "initializer_range"),
        "num_heads": int(train["num_heads"]),
        "alpha": _finite(train["alpha"], "alpha"),
        "lr": _finite(train["lr"], "lr"),
        "weight_decay": _finite(train["weight_decay"], "weight_decay"),
        "adam_beta1": _finite(train["adam_beta1"], "adam_beta1"),
        "adam_beta2": _finite(train["adam_beta2"], "adam_beta2"),
    }
    if any(projection[key] <= 0 for key in ("epochs", "batch_size", "max_seq_length", "requested_topk", "hidden_size", "num_hidden_layers", "num_heads")):
        raise ValueError("WEARec integer scientific parameters must be positive.")
    if projection["max_seq_length"] % 2:
        raise ValueError("WEARec max_seq_length must be even.")
    if projection["hidden_size"] % projection["num_heads"]:
        raise ValueError("WEARec hidden_size must be divisible by num_heads.")
    return projection


class WEARecRunner(VictimRunnerBase):
    name = "wearec"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        self.train_config = _require_train_config(config)
        runtime = _require_runtime_config(config)
        self.python_executable = str(runtime["python_executable"])
        self.repo_root = Path(repo_root or runtime["repo_root"])
        self.working_dir = Path(runtime["working_dir"])
        self.device_config = dict(runtime["device"])
        self.dataloader_config = dict(runtime["dataloader"])

    def build_model(self, opt=None): return None
    def load_dataset(self, *args, **kwargs): raise NotImplementedError
    def train(self, *args, **kwargs): return self.run(**kwargs)
    def evaluate(self, *args, **kwargs): raise NotImplementedError
    def score_session(self, *args, **kwargs): raise NotImplementedError
    def load_model(self, *args, **kwargs): raise NotImplementedError
    def save_model(self, *args, **kwargs): raise NotImplementedError

    def build_command(
        self,
        *,
        train_path: Path,
        valid_path: Path,
        test_path: Path,
        metadata_path: Path,
        prediction_output_path: Path,
        checkpoint_output_path: Path,
        epoch_metrics_output_path: Path | None,
        per_epoch_prediction_dir: Path | None,
        internal_output_dir: Path,
        requested_topk: int,
        epochs: int,
        seed: int,
        per_epoch_diagnostics: bool,
    ) -> tuple[list[str], dict[str, str]]:
        effective = effective_wearec_config(
            self.config, seed=seed, requested_topk=requested_topk, epochs=epochs
        )
        if int(self.dataloader_config["num_workers"]) != 0:
            raise ValueError("WEARec Phase 2 requires num_workers == 0.")
        cmd = [
            self.python_executable,
            str((self.repo_root / "src" / "main.py").resolve()),
            "--canonical_sbr_mode",
            "--model_type", "WEARec",
            "--train_path", str(Path(train_path).resolve()),
            "--valid_path", str(Path(valid_path).resolve()),
            "--test_path", str(Path(test_path).resolve()),
            "--metadata_path", str(Path(metadata_path).resolve()),
            "--prediction_output_path", str(Path(prediction_output_path).resolve()),
            "--checkpoint_output_path", str(Path(checkpoint_output_path).resolve()),
            "--output_dir", str(Path(internal_output_dir).resolve()),
            "--train_name", "wearec_canonical",
            "--epochs", str(effective["epochs"]),
            "--batch_size", str(effective["batch_size"]),
            "--max_seq_length", str(effective["max_seq_length"]),
            "--metric_cutoffs", *[str(value) for value in effective["metric_cutoffs"]],
            "--topk", str(effective["requested_topk"]),
            "--seed", str(effective["seed"]),
            "--num_workers", "0",
            "--gpu_id", _single_gpu_id(self.device_config["gpu_id"]),
            "--hidden_size", str(effective["hidden_size"]),
            "--num_hidden_layers", str(effective["num_hidden_layers"]),
            "--hidden_act", effective["hidden_act"],
            "--hidden_dropout_prob", str(effective["hidden_dropout_prob"]),
            "--initializer_range", str(effective["initializer_range"]),
            "--num_heads", str(effective["num_heads"]),
            "--alpha", str(effective["alpha"]),
            "--lr", str(effective["lr"]),
            "--weight_decay", str(effective["weight_decay"]),
            "--adam_beta1", str(effective["adam_beta1"]),
            "--adam_beta2", str(effective["adam_beta2"]),
        ]
        if per_epoch_diagnostics:
            if epoch_metrics_output_path is None:
                raise ValueError("WEARec diagnostics require epoch_metrics_output_path.")
            cmd.extend(["--per_epoch_diagnostics", "--epoch_metrics_output_path", str(epoch_metrics_output_path.resolve())])
            if per_epoch_prediction_dir is not None:
                cmd.extend(["--per_epoch_prediction_dir", str(per_epoch_prediction_dir.resolve())])
        if not bool(self.device_config["use_gpu"]):
            cmd.append("--no_cuda")
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        if bool(self.device_config["use_gpu"]):
            env["CUDA_VISIBLE_DEVICES"] = _single_gpu_id(self.device_config["gpu_id"])
        return cmd, env

    def run(
        self,
        *,
        train_path: Path,
        valid_path: Path,
        test_path: Path,
        metadata_path: Path,
        item_count: int,
        expected_test_count: int,
        run_dir: Path,
        prediction_output_path: Path,
        requested_topk: int,
        epochs: int,
        victim_train_seed: int,
        target_item: int | None,
        training_mode: str = "clean",
        dataset_name: str | None = None,
        per_epoch_diagnostics: bool = False,
        per_epoch_predictions: bool = False,
    ) -> dict[str, Any]:
        runtime_paths = (
            ("python executable", Path(self.python_executable), "file"),
            ("repository", self.repo_root, "directory"),
            ("working directory", self.working_dir, "directory"),
            ("entrypoint", self.repo_root / "src" / "main.py", "file"),
        )
        for label, path, kind in runtime_paths:
            if not Path(path).is_absolute():
                raise ValueError(f"WEARec {label} must be an absolute path: {path}")
            valid = Path(path).is_file() if kind == "file" else Path(path).is_dir()
            if not valid:
                raise FileNotFoundError(f"WEARec {label} not found: {path}")
        for label, path in (
            ("train data", train_path), ("validation data", valid_path),
            ("test data", test_path), ("metadata", metadata_path),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(f"WEARec {label} not found: {path}")
        run_dir.mkdir(parents=True, exist_ok=True)
        raw = Path(prediction_output_path)
        checkpoint = run_dir / "wearec_checkpoint.pt"
        log = run_dir / "wearec_stdout.log"
        epoch_metrics = run_dir / "wearec_epoch_metrics.jsonl" if per_epoch_diagnostics else None
        epoch_predictions = run_dir / "wearec_per_epoch_predictions" if per_epoch_diagnostics and per_epoch_predictions else None
        effective = effective_wearec_config(
            self.config, seed=victim_train_seed, requested_topk=requested_topk, epochs=epochs
        )
        expected_labels = load_exported_canonical_labels(test_path)
        if int(expected_test_count) != len(expected_labels):
            raise ValueError(
                "WEARec exported test count does not match authoritative test JSONL."
            )
        cmd, env = self.build_command(
            train_path=train_path, valid_path=valid_path, test_path=test_path,
            metadata_path=metadata_path, prediction_output_path=raw,
            checkpoint_output_path=checkpoint, epoch_metrics_output_path=epoch_metrics,
            per_epoch_prediction_dir=epoch_predictions,
            internal_output_dir=run_dir / "wearec_internal_output",
            requested_topk=requested_topk, epochs=epochs, seed=victim_train_seed,
            per_epoch_diagnostics=per_epoch_diagnostics,
        )
        result = run_subprocess_with_epoch_progress(
            cmd, cwd=self.working_dir, env=env, log_path=log, model_name=self.name,
            target_item=target_item, total_epochs=epochs, epoch_numbers_are_one_based=True,
        )
        if result.returncode:
            raise RuntimeError(f"WEARec subprocess failed with code {result.returncode}. See {log}")
        payload = load_wearec_prediction_payload(
            raw, item_count=item_count, expected_labels=expected_labels,
            effective_config=effective,
            expected_training_mode=training_mode,
            expected_dataset_name=dataset_name,
        )
        checkpoint_provenance = file_provenance(checkpoint)
        raw_provenance = file_provenance(raw)
        log_provenance = file_provenance(log)
        return {
            "returncode": 0,
            "prediction_output_path": str(raw),
            "checkpoint_output_path": str(checkpoint),
            "log_path": str(log),
            "epoch_metrics_output_path": str(epoch_metrics) if epoch_metrics else None,
            "per_epoch_prediction_dir": str(epoch_predictions) if epoch_predictions else None,
            "effective_config": effective,
            "current_epoch": payload["current_epoch"],
            "epochs_requested": payload["epochs_requested"],
            "epochs_completed": payload["epochs_completed"],
            "final_epoch": payload["final_epoch"],
            "selected_epoch": payload["selected_epoch"],
            "test_metrics": payload["test_metrics"],
            "prediction_count": len(payload["rankings"]),
            "checkpoint_provenance": checkpoint_provenance,
            "raw_prediction_provenance": raw_provenance,
            "log_provenance": log_provenance,
            "python_executable": self.python_executable,
            "repo_root": str(self.repo_root),
            "working_dir": str(self.working_dir),
            "use_gpu": bool(self.device_config["use_gpu"]),
            "gpu_id": _single_gpu_id(self.device_config["gpu_id"]),
        }

    def predict_topk(
        self,
        *,
        predictions_path: Path,
        item_count: int,
        expected_labels: Sequence[int],
        requested_topk: int,
        configured_epochs: int,
        seed: int,
        expected_training_mode: str,
        expected_dataset_name: str,
    ) -> list[list[int]]:
        effective = effective_wearec_config(
            self.config, seed=seed, requested_topk=requested_topk, epochs=configured_epochs
        )
        payload = load_wearec_prediction_payload(
            predictions_path, item_count=item_count,
            expected_labels=expected_labels,
            effective_config=effective,
            expected_training_mode=expected_training_mode,
            expected_dataset_name=expected_dataset_name,
        )
        return [list(row["items"]) for row in payload["rankings"]]


def load_wearec_prediction_payload(
    path: str | Path,
    *,
    item_count: int,
    expected_labels: Sequence[int],
    effective_config: Mapping[str, Any],
    expected_training_mode: str | None = None,
    expected_dataset_name: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"WEARec prediction artifact is missing: {source}")
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("WEARec prediction artifact must be a JSON object.")
    forbidden = {
        "best_epoch",
        "best_metric",
        "best_checkpoint",
        "early_stopping",
    }
    present_forbidden = forbidden.intersection(payload)
    if present_forbidden:
        raise ValueError(
            "WEARec canonical artifact contains forbidden validation-best fields: "
            + ", ".join(sorted(present_forbidden))
        )
    required = {
        "schema_version", "model", "mode", "split", "training_mode", "dataset_name",
        "checkpoint_protocol", "epochs_requested", "epochs_completed", "current_epoch",
        "final_epoch", "selected_epoch", "item_count", "max_seq_length",
        "metric_cutoffs", "requested_topk", "topk", "evaluation_topk",
        "example_count", "batch_size", "batch_count", "final_batch_size",
        "num_workers", "drop_last", "train_sampler", "evaluation_sampler",
        "seed", "model_config", "rankings", "test_metrics",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError("WEARec prediction artifact missing fields: " + ", ".join(sorted(missing)))
    authoritative_labels = list(expected_labels)
    if not authoritative_labels:
        raise ValueError("WEARec authoritative expected labels must not be empty.")
    if any(type(label) is not int or label < 1 or label > item_count for label in authoritative_labels):
        raise ValueError("WEARec authoritative expected labels are invalid.")
    expected_example_count = len(authoritative_labels)
    exact_integer_fields = (
        "schema_version",
        "epochs_requested",
        "epochs_completed",
        "current_epoch",
        "final_epoch",
        "selected_epoch",
        "item_count",
        "max_seq_length",
        "requested_topk",
        "topk",
        "evaluation_topk",
        "example_count",
        "batch_size",
        "batch_count",
        "final_batch_size",
        "num_workers",
        "seed",
    )
    for key in exact_integer_fields:
        _exact_artifact_int(payload[key], key)
    for key, expected in (
        ("schema_version", 1), ("model", "wearec"), ("mode", "canonical_sbr"),
        ("split", "test"), ("checkpoint_protocol", "fixed_epoch"),
        ("item_count", item_count), ("example_count", expected_example_count),
        ("num_workers", 0),
    ):
        if payload[key] != expected:
            raise ValueError(f"WEARec artifact {key} does not match {expected!r}.")
    if type(payload["drop_last"]) is not bool or payload["drop_last"] is not False:
        raise ValueError("WEARec artifact drop_last must be false.")
    if type(payload["training_mode"]) is not str:
        raise ValueError("WEARec artifact training_mode must be a string.")
    if type(payload["dataset_name"]) is not str:
        raise ValueError("WEARec artifact dataset_name must be a string.")
    if expected_training_mode is not None and payload["training_mode"] != expected_training_mode:
        raise ValueError("WEARec artifact training_mode does not match parent request.")
    if expected_dataset_name is not None and payload["dataset_name"] != expected_dataset_name:
        raise ValueError("WEARec artifact dataset_name does not match parent request.")
    for key in ("epochs_requested", "epochs_completed", "current_epoch", "final_epoch", "selected_epoch"):
        if type(payload[key]) is not int or payload[key] != effective_config["epochs"]:
            raise ValueError(f"WEARec artifact {key} is not the configured final epoch.")
    for key in ("batch_size", "max_seq_length", "requested_topk"):
        if payload[key] != effective_config[key]:
            raise ValueError(f"WEARec artifact {key} does not match effective configuration.")
    if payload["topk"] != effective_config["requested_topk"]:
        raise ValueError("WEARec artifact topk does not match requested_topk.")
    if not isinstance(payload["metric_cutoffs"], list) or any(
        type(value) is not int for value in payload["metric_cutoffs"]
    ):
        raise ValueError("WEARec artifact metric_cutoffs must be exact integers.")
    if payload["metric_cutoffs"] != effective_config["metric_cutoffs"]:
        raise ValueError("WEARec artifact metric_cutoffs do not match.")
    if payload["evaluation_topk"] != max(effective_config["requested_topk"], max(effective_config["metric_cutoffs"])):
        raise ValueError("WEARec artifact evaluation_topk does not match.")
    if payload["seed"] != effective_config["seed"]:
        raise ValueError("WEARec artifact seed does not match.")
    if payload["train_sampler"] != "seeded_random" or payload["evaluation_sampler"] != "sequential":
        raise ValueError("WEARec artifact sampler metadata does not match canonical mode.")
    expected_batches = math.ceil(expected_example_count / effective_config["batch_size"])
    expected_final = expected_example_count - effective_config["batch_size"] * (
        expected_batches - 1
    )
    if payload["batch_count"] != expected_batches or payload["final_batch_size"] != expected_final:
        raise ValueError("WEARec artifact batch metadata is inconsistent.")
    model_config = payload["model_config"]
    if not isinstance(model_config, Mapping):
        raise ValueError("WEARec artifact model_config must be an object.")
    for key in _MODEL_CONFIG_FIELDS:
        if model_config.get(key) != effective_config[key]:
            raise ValueError(f"WEARec artifact model_config.{key} does not match.")
    rankings = payload["rankings"]
    if not isinstance(rankings, list) or len(rankings) != expected_example_count:
        raise ValueError("WEARec ranking count does not match.")
    normalized: list[dict[str, Any]] = []
    labels: list[int] = []
    for expected_id, row in enumerate(rankings):
        if not isinstance(row, Mapping) or row.get("example_id") != expected_id:
            raise ValueError("WEARec rankings are not in example_id order.")
        label = row.get("label")
        items = row.get("items")
        if type(label) is not int or not 1 <= label <= item_count:
            raise ValueError("WEARec ranking label is invalid.")
        if label != authoritative_labels[expected_id]:
            raise ValueError(
                f"WEARec ranking label for example_id {expected_id} does not match "
                "the authoritative exported test label."
            )
        if not isinstance(items, list) or len(items) != effective_config["requested_topk"]:
            raise ValueError("WEARec ranking length is invalid.")
        if any(type(item) is not int or not 1 <= item <= item_count for item in items):
            raise ValueError("WEARec ranking item is invalid.")
        if len(set(items)) != len(items):
            raise ValueError("WEARec ranking contains duplicates.")
        labels.append(authoritative_labels[expected_id])
        normalized.append({"example_id": expected_id, "label": label, "items": list(items)})
    validate_wearec_metrics(
        [row["items"] for row in normalized], labels, payload["test_metrics"],
        effective_config["metric_cutoffs"],
    )
    payload["rankings"] = normalized
    return payload


def _exact_artifact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"WEARec artifact {field} must be an exact integer.")
    return int(value)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"WEARec {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"WEARec {field} must be finite.")
    return result


def _single_gpu_id(value: Any) -> str:
    text = str(value).strip()
    if not text.isdigit() or "," in text:
        raise ValueError("WEARec GPU ID must be one non-negative integer.")
    return text


def _require_train_config(config: Config) -> dict[str, Any]:
    params = config.victims.params.get("wearec")
    train = params.get("train") if isinstance(params, Mapping) else None
    if not isinstance(train, Mapping):
        raise ValueError("Missing victims.params.wearec.train configuration.")
    return dict(train)


def _require_runtime_config(config: Config) -> dict[str, Any]:
    runtime = (config.victims.runtime or {}).get("wearec")
    if not isinstance(runtime, Mapping):
        raise ValueError("Missing victims.runtime.wearec configuration.")
    return dict(runtime)


register_victim(WEARecRunner.name, WEARecRunner)

__all__ = [
    "WEAREC_ARTIFACT_CONTRACT_VERSION",
    "WEAREC_RUNNER_SEMANTICS_VERSION",
    "WEARecRunner",
    "effective_wearec_config",
    "load_wearec_prediction_payload",
]
