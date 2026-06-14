from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim
from attack.models.victim.subprocess_progress import run_subprocess_with_epoch_progress


_METRIC_PATTERN = re.compile(r"^(hr|mrr|ndcg)@([1-9]\d*)$")


class FreqRecRunner(VictimRunnerBase):
    name = "freqrec"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        self.train_config = _require_train_config(config)
        runtime = _require_runtime_config(config)
        self.python_executable = str(runtime["python_executable"])
        self.repo_root = (
            Path(repo_root) if repo_root is not None else Path(runtime["repo_root"])
        )
        self.working_dir = Path(runtime["working_dir"])
        self.device_config = dict(runtime["device"])
        self.dataloader_config = dict(runtime["dataloader"])
        diagnostics = runtime.get("diagnostics", {})
        self.diagnostics_config = (
            dict(diagnostics) if isinstance(diagnostics, Mapping) else {}
        )

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("FreqRec canonical datasets are loaded by the subprocess.")

    def train(self, *args, **kwargs):
        return self.run(**kwargs)

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("FreqRec evaluation is handled by the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("FreqRec does not expose per-session scoring.")

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("FreqRec model loading is handled by the subprocess.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("FreqRec model saving is handled by the subprocess.")

    def build_command(
        self,
        *,
        train_path: Path,
        valid_path: Path,
        test_path: Path,
        metadata_path: Path,
        prediction_output_path: Path,
        checkpoint_output_path: Path | None,
        epoch_metrics_output_path: Path | None,
        per_epoch_prediction_dir: Path | None,
        internal_output_dir: Path,
        train_name: str,
        requested_topk: int,
        epochs: int,
        seed: int,
    ) -> tuple[list[str], dict[str, str]]:
        main_path = self.repo_root / "src" / "main.py"
        num_workers = int(self.dataloader_config["num_workers"])
        gpu_id = _single_gpu_id(self.device_config["gpu_id"])
        use_gpu = bool(self.device_config["use_gpu"])
        train = self.train_config
        cmd = [
            self.python_executable,
            str(main_path.resolve()),
            "--canonical_sbr_mode",
            "--train_path",
            str(Path(train_path).resolve()),
            "--valid_path",
            str(Path(valid_path).resolve()),
            "--test_path",
            str(Path(test_path).resolve()),
            "--metadata_path",
            str(Path(metadata_path).resolve()),
            "--checkpoint_protocol",
            str(train["checkpoint_protocol"]),
            "--validation_metric",
            str(train["validation_metric"]),
            "--metric_cutoffs",
            *[str(int(value)) for value in train["metric_cutoffs"]],
            "--topk",
            str(int(requested_topk)),
            "--prediction_output_path",
            str(Path(prediction_output_path).resolve()),
            "--output_dir",
            str(Path(internal_output_dir).resolve()),
            "--train_name",
            str(train_name),
            "--epochs",
            str(int(epochs)),
            "--batch_size",
            str(int(train["batch_size"])),
            "--max_seq_length",
            str(int(train["max_seq_length"])),
            "--lr",
            str(float(train["lr"])),
            "--seed",
            str(int(seed)),
            "--gpu_id",
            gpu_id,
            "--num_workers",
            str(num_workers),
            "--model_type",
            "freqrec",
            "--hidden_size",
            str(int(train["hidden_size"])),
            "--num_hidden_layers",
            str(int(train["num_hidden_layers"])),
            "--num_attention_heads",
            str(int(train["num_attention_heads"])),
            "--hidden_act",
            str(train["hidden_act"]),
            "--attention_probs_dropout_prob",
            str(float(train["attention_probs_dropout_prob"])),
            "--hidden_dropout_prob",
            str(float(train["hidden_dropout_prob"])),
            "--initializer_range",
            str(float(train["initializer_range"])),
            "--alpha",
            str(float(train["alpha"])),
            "--gama",
            str(float(train["gama"])),
            "--alpha_loss",
            str(float(train["alpha_loss"])),
            "--fft_loss_type",
            str(train["fft_loss_type"]),
            "--chux",
            str(train["chux"]),
            "--adam_beta1",
            str(float(train["adam_beta1"])),
            "--adam_beta2",
            str(float(train["adam_beta2"])),
            "--weight_decay",
            str(float(train["weight_decay"])),
            "--patience",
            str(int(train["patience"])),
            "--fre",
            "1.0",
        ]
        if checkpoint_output_path is not None:
            cmd.extend(
                ["--checkpoint_output_path", str(Path(checkpoint_output_path).resolve())]
            )
        if epoch_metrics_output_path is not None:
            cmd.extend(
                [
                    "--epoch_metrics_output_path",
                    str(Path(epoch_metrics_output_path).resolve()),
                ]
            )
        if per_epoch_prediction_dir is not None:
            cmd.extend(
                [
                    "--per_epoch_prediction_dir",
                    str(Path(per_epoch_prediction_dir).resolve()),
                ]
            )
        if not use_gpu:
            cmd.append("--no_cuda")

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(int(seed))
        if use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
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
    ) -> dict[str, Any]:
        main_path = self.repo_root / "src" / "main.py"
        for label, path in (
            ("repository", self.repo_root),
            ("working directory", self.working_dir),
            ("entrypoint", main_path),
            ("train data", Path(train_path)),
            ("validation data", Path(valid_path)),
            ("test data", Path(test_path)),
            ("metadata", Path(metadata_path)),
        ):
            if not path.exists():
                raise FileNotFoundError(f"FreqRec {label} not found: {path}")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "freqrec_stdout.log"
        internal_output_dir = run_dir / "freqrec_internal_output"
        train_name = "freqrec_canonical"
        internal_log_path = internal_output_dir / f"{train_name}.log"
        checkpoint_path = (
            run_dir / "freqrec_checkpoint.pt"
            if bool(self.diagnostics_config.get("save_checkpoint", False))
            else None
        )
        epoch_metrics_path = (
            run_dir / "freqrec_epoch_metrics.jsonl"
            if bool(self.diagnostics_config.get("epoch_metrics", False))
            else None
        )
        per_epoch_prediction_dir = (
            run_dir / "freqrec_per_epoch_predictions"
            if bool(self.diagnostics_config.get("per_epoch_predictions", False))
            else None
        )
        cmd, env = self.build_command(
            train_path=train_path,
            valid_path=valid_path,
            test_path=test_path,
            metadata_path=metadata_path,
            prediction_output_path=prediction_output_path,
            checkpoint_output_path=checkpoint_path,
            epoch_metrics_output_path=epoch_metrics_path,
            per_epoch_prediction_dir=per_epoch_prediction_dir,
            internal_output_dir=internal_output_dir,
            train_name=train_name,
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
                f"FreqRec subprocess failed with code {result.returncode}. See log: {log_path}"
            )
        payload = self.load_prediction_payload(
            prediction_output_path,
            split="test",
            item_count=item_count,
            expected_example_count=expected_test_count,
            requested_topk=requested_topk,
            configured_epochs=epochs,
            seed=victim_train_seed,
        )
        run_info: dict[str, Any] = {
            "returncode": result.returncode,
            "log_path": str(log_path),
            "internal_output_dir": str(internal_output_dir),
            "internal_log_path": str(internal_log_path),
            "prediction_output_path": str(prediction_output_path),
            "checkpoint_protocol": payload["checkpoint_protocol"],
            "current_epoch": payload["current_epoch"],
            "selected_epoch": payload["selected_epoch"],
            "epochs_requested": payload["epochs_requested"],
            "epochs_completed": payload["epochs_completed"],
            "best_epoch": payload["best_epoch"],
            "best_metric": payload["best_metric"],
            "validation_metric": payload["validation_metric"],
            "requested_topk": payload["requested_topk"],
            "topk": payload["topk"],
            "evaluation_topk": payload["evaluation_topk"],
            "batch_size": payload["batch_size"],
            "batch_count": payload["batch_count"],
            "final_batch_size": payload["final_batch_size"],
            "num_workers": payload["num_workers"],
            "drop_last": payload["drop_last"],
            "train_sampler": payload["train_sampler"],
            "evaluation_sampler": payload["evaluation_sampler"],
            "seed": payload["seed"],
            "prediction_count": len(payload["rankings"]),
            "python_executable": self.python_executable,
            "repo_root": str(self.repo_root),
            "working_dir": str(self.working_dir),
            "gpu_id": _single_gpu_id(self.device_config["gpu_id"]),
            "use_gpu": bool(self.device_config["use_gpu"]),
        }
        if checkpoint_path is not None:
            run_info["checkpoint_output_path"] = str(checkpoint_path)
        if epoch_metrics_path is not None:
            run_info["epoch_metrics_output_path"] = str(epoch_metrics_path)
        if per_epoch_prediction_dir is not None:
            run_info["per_epoch_prediction_dir"] = str(per_epoch_prediction_dir)
        return run_info

    def predict_topk(
        self,
        *,
        predictions_path: Path,
        item_count: int,
        expected_example_count: int,
        requested_topk: int,
        configured_epochs: int,
        seed: int,
    ) -> list[list[int]]:
        payload = self.load_prediction_payload(
            predictions_path,
            split="test",
            item_count=item_count,
            expected_example_count=expected_example_count,
            requested_topk=requested_topk,
            configured_epochs=configured_epochs,
            seed=seed,
        )
        return [list(row["items"]) for row in payload["rankings"]]

    def load_prediction_payload(
        self,
        path: Path,
        *,
        split: str,
        item_count: int,
        expected_example_count: int,
        requested_topk: int,
        configured_epochs: int,
        seed: int,
        expected_current_epoch: int | None = None,
    ) -> dict[str, Any]:
        return load_freqrec_prediction_payload(
            path,
            split=split,
            item_count=item_count,
            expected_example_count=expected_example_count,
            requested_topk=requested_topk,
            configured_epochs=configured_epochs,
            checkpoint_protocol=str(self.train_config["checkpoint_protocol"]),
            validation_metric=str(self.train_config["validation_metric"]),
            metric_cutoffs=self.train_config["metric_cutoffs"],
            batch_size=int(self.train_config["batch_size"]),
            seed=seed,
            expected_current_epoch=expected_current_epoch,
        )


def load_freqrec_prediction_payload(
    path: Path,
    *,
    split: str,
    item_count: int,
    expected_example_count: int,
    requested_topk: int,
    configured_epochs: int,
    checkpoint_protocol: str,
    validation_metric: str,
    metric_cutoffs: Sequence[int],
    batch_size: int,
    seed: int,
    expected_current_epoch: int | None = None,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"FreqRec prediction artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid FreqRec prediction JSON at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("FreqRec prediction artifact must be a JSON object.")
    required = {
        "schema_version",
        "model",
        "mode",
        "split",
        "checkpoint_protocol",
        "current_epoch",
        "selected_epoch",
        "epochs_requested",
        "epochs_completed",
        "best_epoch",
        "best_metric",
        "validation_metric",
        "requested_topk",
        "topk",
        "evaluation_topk",
        "item_count",
        "example_count",
        "batch_size",
        "batch_count",
        "final_batch_size",
        "num_workers",
        "drop_last",
        "train_sampler",
        "evaluation_sampler",
        "seed",
        "rankings",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError("FreqRec prediction artifact missing fields: " + ", ".join(missing))
    if _exact_int(payload["schema_version"], "schema_version") != 1:
        raise ValueError("Unsupported FreqRec prediction schema_version.")
    for key, expected in (
        ("model", "freqrec"),
        ("mode", "canonical_sbr"),
        ("split", split),
        ("checkpoint_protocol", checkpoint_protocol),
        ("validation_metric", validation_metric),
        ("train_sampler", "seeded_random"),
        ("evaluation_sampler", "sequential"),
    ):
        if payload[key] != expected:
            raise ValueError(f"FreqRec artifact {key}={payload[key]!r}; expected {expected!r}.")
    if payload["drop_last"] is not False:
        raise ValueError("FreqRec artifact drop_last must be false.")

    exact_values = {
        "item_count": item_count,
        "example_count": expected_example_count,
        "requested_topk": requested_topk,
        "batch_size": batch_size,
        "seed": seed,
        "epochs_requested": configured_epochs,
        "epochs_completed": (
            expected_current_epoch if split == "validation" else configured_epochs
        ),
    }
    for key, expected in exact_values.items():
        if _exact_int(payload[key], key) != int(expected):
            raise ValueError(f"FreqRec artifact {key} does not match expected value {expected}.")
    exported_topk = min(requested_topk, item_count)
    if _exact_int(payload["topk"], "topk") != exported_topk:
        raise ValueError("FreqRec artifact topk does not match min(requested_topk, item_count).")
    monitor_cutoff = _metric_cutoff(validation_metric)
    cutoffs = [_exact_int(value, "metric_cutoffs[]") for value in metric_cutoffs]
    expected_evaluation_topk = min(
        max(requested_topk, max(cutoffs), monitor_cutoff), item_count
    )
    if _exact_int(payload["evaluation_topk"], "evaluation_topk") != expected_evaluation_topk:
        raise ValueError("FreqRec artifact evaluation_topk does not match resolved configuration.")
    expected_batches = math.ceil(expected_example_count / batch_size)
    expected_final = expected_example_count - batch_size * (expected_batches - 1)
    if _exact_int(payload["batch_count"], "batch_count") != expected_batches:
        raise ValueError("FreqRec artifact batch_count is inconsistent.")
    if _exact_int(payload["final_batch_size"], "final_batch_size") != expected_final:
        raise ValueError("FreqRec artifact final_batch_size is inconsistent.")
    if _exact_int(payload["num_workers"], "num_workers") < 0:
        raise ValueError("FreqRec artifact num_workers must be non-negative.")

    current_epoch = _exact_int(payload["current_epoch"], "current_epoch")
    selected_epoch = _exact_int(payload["selected_epoch"], "selected_epoch")
    completed = _exact_int(payload["epochs_completed"], "epochs_completed")
    if split == "validation":
        if expected_current_epoch is None:
            raise ValueError("Per-epoch validation parsing requires expected_current_epoch.")
        if current_epoch != expected_current_epoch or selected_epoch != expected_current_epoch:
            raise ValueError("FreqRec per-epoch current/selected epoch is inconsistent.")
        if checkpoint_protocol == "fixed_epoch":
            if payload["best_epoch"] is not None or payload["best_metric"] is not None:
                raise ValueError(
                    "FreqRec fixed-epoch per-epoch checkpoint-selection fields must be null."
                )
        else:
            best_epoch = _exact_int(payload["best_epoch"], "best_epoch")
            _finite_number(payload["best_metric"], "best_metric")
            if not 1 <= best_epoch <= current_epoch:
                raise ValueError("FreqRec per-epoch best_epoch is invalid.")
    elif checkpoint_protocol == "fixed_epoch":
        if current_epoch != configured_epochs or selected_epoch != configured_epochs:
            raise ValueError("FreqRec fixed-epoch artifact did not export final weights.")
        if payload["best_epoch"] is not None or payload["best_metric"] is not None:
            raise ValueError("FreqRec fixed-epoch best checkpoint fields must be null.")
    else:
        best_epoch = _exact_int(payload["best_epoch"], "best_epoch")
        best_metric = _finite_number(payload["best_metric"], "best_metric")
        if completed != configured_epochs:
            raise ValueError("FreqRec validation-best must complete the full epoch budget.")
        if not (1 <= selected_epoch == best_epoch <= completed):
            raise ValueError("FreqRec validation-best selected/best epoch is invalid.")
        if current_epoch != best_epoch:
            raise ValueError("FreqRec validation-best current_epoch must be restored best_epoch.")
        payload["best_metric"] = best_metric

    rankings = payload["rankings"]
    if not isinstance(rankings, list) or len(rankings) != expected_example_count:
        raise ValueError("FreqRec ranking row count does not match example_count.")
    normalized: list[dict[str, Any]] = []
    for expected_id, row in enumerate(rankings):
        if not isinstance(row, dict):
            raise ValueError(f"FreqRec ranking row {expected_id} must be an object.")
        if set(("example_id", "items")) - row.keys():
            raise ValueError(f"FreqRec ranking row {expected_id} is missing required fields.")
        if _exact_int(row["example_id"], "rankings[].example_id") != expected_id:
            raise ValueError("FreqRec ranking rows are not in exact example_id order.")
        items = row["items"]
        if not isinstance(items, list) or len(items) != exported_topk:
            raise ValueError(f"FreqRec ranking row {expected_id} has incorrect length.")
        normalized_items = [_exact_int(item, "rankings[].items[]") for item in items]
        if any(item < 1 or item > item_count for item in normalized_items):
            raise ValueError(f"FreqRec ranking row {expected_id} contains out-of-range IDs.")
        if len(set(normalized_items)) != len(normalized_items):
            raise ValueError(f"FreqRec ranking row {expected_id} contains duplicate IDs.")
        normalized.append({"example_id": expected_id, "items": normalized_items})
    payload["rankings"] = normalized
    return payload


def _metric_cutoff(metric: str) -> int:
    match = _METRIC_PATTERN.fullmatch(metric)
    if match is None:
        raise ValueError(f"Unsupported FreqRec validation metric: {metric!r}.")
    return int(match.group(2))


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"FreqRec artifact {field} must be an integer.")
    return int(value)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"FreqRec artifact {field} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"FreqRec artifact {field} must be finite.")
    return result


def _single_gpu_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("FreqRec GPU ID must identify exactly one physical GPU.")
    text = str(value).strip()
    if not text or "," in text or not text.isdigit():
        raise ValueError("FreqRec GPU ID must be one non-negative physical GPU integer.")
    return text


def _require_runtime_config(config: Config) -> dict[str, Any]:
    runtime = (config.victims.runtime or {}).get("freqrec")
    if not isinstance(runtime, dict):
        raise ValueError("Missing victims.runtime.freqrec configuration.")
    return dict(runtime)


def _require_train_config(config: Config) -> dict[str, Any]:
    params = config.victims.params.get("freqrec")
    train = params.get("train") if isinstance(params, dict) else None
    if not isinstance(train, dict):
        raise ValueError("Missing victims.params.freqrec.train configuration.")
    return dict(train)


register_victim(FreqRecRunner.name, FreqRecRunner)


__all__ = [
    "FreqRecRunner",
    "load_freqrec_prediction_payload",
]
