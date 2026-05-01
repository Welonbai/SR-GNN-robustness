from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attack.common.config import Config, load_config
from attack.common.paths import canonical_split_paths, split_key
from attack.data.poisoned_dataset_builder import expand_session_to_samples
from attack.data.unified_split import ensure_canonical_dataset
from attack.pipeline.core.evaluator import evaluate_ground_truth_metrics


DEFAULT_CONFIG = Path(
    "attack/configs/"
    "diginetica_attack_rank_bucket_cem_lowk_ft6500_srgnn_targets11103_14514_5418.yaml"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/diagnostics/clean_srgnn_training_sanity")
BEST_METRIC = "ground_truth_mrr@20"
VALID_BEST_METRIC_KEY = "valid_ground_truth_mrr@20"
VALID_RECALL20_KEY = "valid_ground_truth_recall@20"
EPOCH8 = 8
CLOSE_RELATIVE_THRESHOLD = 0.01
TEST_METRICS_NOTE = (
    "test metrics are diagnostic only and are not used for model selection"
)
_GT_METRICS = ("recall", "mrr")
_GT_TOPK = (10, 20)
_TRAINING_DEPS: dict[str, Any] | None = None


def _load_training_deps() -> dict[str, Any]:
    global _TRAINING_DEPS
    if _TRAINING_DEPS is not None:
        return _TRAINING_DEPS
    try:
        import torch

        from attack.common.seed import set_seed
        from attack.models.victim.srgnn_runner import SRGNNVictimRunner
        from pytorch_code.model import forward as srg_forward
        from pytorch_code.model import trans_to_cuda
        from pytorch_code.utils import Data
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        raise RuntimeError(
            "Unable to import the SR-GNN training dependencies. Run this tool with "
            "the same Python environment used for SR-GNN experiments."
        ) from exc
    _TRAINING_DEPS = {
        "Data": Data,
        "SRGNNVictimRunner": SRGNNVictimRunner,
        "set_seed": set_seed,
        "srg_forward": srg_forward,
        "torch": torch,
        "trans_to_cuda": trans_to_cuda,
    }
    return _TRAINING_DEPS


def _derive_seed(base_seed: int, *components: object) -> int:
    normalized_base = int(base_seed)
    if not components:
        return normalized_base
    payload = json.dumps(
        [normalized_base, *components],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(derived % (2**31 - 1))


def _build_srgnn_opt_from_train_config(train_config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        batchSize=int(train_config["batch_size"]),
        hiddenSize=int(train_config["hidden_size"]),
        epoch=int(train_config["epochs"]),
        lr=float(train_config["lr"]),
        lr_dc=float(train_config["lr_dc"]),
        lr_dc_step=int(train_config["lr_dc_step"]),
        l2=float(train_config["l2"]),
        step=int(train_config["step"]),
        patience=int(train_config["patience"]),
        nonhybrid=bool(train_config["nonhybrid"]),
    )


def _sequences_to_pairs(sequences: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[int]]:
    sessions: list[list[int]] = []
    labels: list[int] = []
    for sequence in sequences:
        prefixes, prefix_labels = expand_session_to_samples(sequence)
        sessions.extend(prefixes)
        labels.extend(int(label) for label in prefix_labels)
    return sessions, labels


def _resolve_train_config(config: Config, source: str, *, max_epochs: int) -> dict[str, Any]:
    source_name = str(source).strip().lower()
    if source_name == "poison_model":
        train = config.attack.poison_model.params.get("train")
    elif source_name == "victim_srgnn":
        train = config.victims.params.get("srgnn", {}).get("train")
    else:
        raise ValueError("train_config_source must be 'poison_model' or 'victim_srgnn'.")
    if not isinstance(train, Mapping):
        raise ValueError(f"Missing SR-GNN train config for source: {source_name}")
    resolved = dict(train)
    resolved["epochs"] = int(max_epochs)
    return resolved


def _default_seed(config: Config, source: str) -> int:
    source_name = str(source).strip().lower()
    if source_name == "poison_model":
        return _derive_seed(config.seeds.fake_session_seed, "poison_model_generation")
    if source_name == "victim_srgnn":
        return _derive_seed(config.seeds.victim_train_seed, "victim_train", "srgnn")
    raise ValueError("train_config_source must be 'poison_model' or 'victim_srgnn'.")


def _train_one_epoch(runner: Any, train_data: Any) -> float:
    deps = _load_training_deps()
    srg_forward = deps["srg_forward"]
    torch = deps["torch"]
    trans_to_cuda = deps["trans_to_cuda"]
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    model = runner.model
    model.train()
    total_loss = 0.0
    batch_count = 0
    for batch_indices in train_data.generate_batch(model.batch_size):
        model.optimizer.zero_grad()
        targets, scores = srg_forward(model, batch_indices, train_data)
        targets_tensor = trans_to_cuda(torch.as_tensor(targets, dtype=torch.long))
        loss = model.loss_function(scores, targets_tensor - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += float(loss.item())
        batch_count += 1
    model.scheduler.step()
    avg_loss = total_loss / float(max(1, batch_count))
    runner.train_loss_history.append(float(avg_loss))
    return float(avg_loss)


def _evaluate_ground_truth_for_data(
    runner: Any,
    data: Any,
    labels: Sequence[int],
) -> dict[str, float]:
    rankings = runner.predict_topk(data, topk=max(_GT_TOPK))
    metrics, available = evaluate_ground_truth_metrics(
        rankings,
        labels=labels,
        metrics=_GT_METRICS,
        topk=_GT_TOPK,
    )
    if not available:
        raise RuntimeError("Ground-truth metrics are unavailable.")
    return {str(key): float(value) for key, value in metrics.items()}


def _prefixed_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _current_lr(runner: Any) -> float:
    if runner.model is None:
        raise RuntimeError("SR-GNN model is not initialized.")
    return float(runner.model.optimizer.param_groups[0]["lr"])


def improved_over_best(current_valid_mrr20: float, best_valid_mrr20: float) -> bool:
    return float(current_valid_mrr20) > float(best_valid_mrr20)


def epoch8_status(epoch8_epoch: int | None, best_epoch: int) -> str | None:
    if epoch8_epoch is None:
        return None
    if int(epoch8_epoch) < int(best_epoch):
        return "before_best"
    if int(epoch8_epoch) == int(best_epoch):
        return "at_best"
    return "after_best"


def _relative_delta(epoch_value: float | None, best_value: float | None) -> float | None:
    if epoch_value is None or best_value is None:
        return None
    absolute_delta = float(best_value) - float(epoch_value)
    denominator = abs(float(best_value))
    if denominator == 0.0:
        return 0.0 if absolute_delta == 0.0 else None
    return float(absolute_delta / denominator)


def _absolute_delta(epoch_value: float | None, best_value: float | None) -> float | None:
    if epoch_value is None or best_value is None:
        return None
    return float(best_value) - float(epoch_value)


def _close_to_best(relative_delta: float | None) -> bool | None:
    if relative_delta is None:
        return None
    return bool(float(relative_delta) <= CLOSE_RELATIVE_THRESHOLD)


def _best_row(rows: Sequence[Mapping[str, Any]], metric_key: str) -> Mapping[str, Any]:
    if not rows:
        raise ValueError("Cannot select a best epoch from an empty history.")
    return max(
        rows,
        key=lambda row: (float(row[metric_key]), -int(row["epoch"])),
    )


def _epoch_row(rows: Sequence[Mapping[str, Any]], epoch: int) -> Mapping[str, Any] | None:
    for row in rows:
        if int(row["epoch"]) == int(epoch):
            return row
    return None


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    config_path: str | Path,
    output_dir: str | Path,
    max_epochs: int,
    patience: int,
    train_config_source: str,
    seed: int,
    data_sources: Mapping[str, Any],
    checkpoint_paths: Mapping[str, str],
    stopped_early: bool,
    stopped_epoch: int | None,
) -> dict[str, Any]:
    best_mrr20_row = _best_row(rows, VALID_BEST_METRIC_KEY)
    best_recall20_row = _best_row(rows, VALID_RECALL20_KEY)
    epoch8_row = _epoch_row(rows, EPOCH8)

    best_mrr20 = float(best_mrr20_row[VALID_BEST_METRIC_KEY])
    best_recall20 = float(best_recall20_row[VALID_RECALL20_KEY])
    epoch8_mrr20 = (
        None if epoch8_row is None else float(epoch8_row[VALID_BEST_METRIC_KEY])
    )
    epoch8_recall20 = (
        None if epoch8_row is None else float(epoch8_row[VALID_RECALL20_KEY])
    )
    mrr20_abs_delta = _absolute_delta(epoch8_mrr20, best_mrr20)
    mrr20_rel_delta = _relative_delta(epoch8_mrr20, best_mrr20)
    recall20_abs_delta = _absolute_delta(epoch8_recall20, best_recall20)
    recall20_rel_delta = _relative_delta(epoch8_recall20, best_recall20)

    return {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "best_checkpoint_metric": VALID_BEST_METRIC_KEY,
        "improvement_rule": "improved = current_valid_mrr20 > best_valid_mrr20",
        "train_config_source": str(train_config_source),
        "seed": int(seed),
        "epochs_completed": int(len(rows)),
        "stopped_early": bool(stopped_early),
        "stopped_epoch": None if stopped_epoch is None else int(stopped_epoch),
        "best_valid_mrr20_epoch": int(best_mrr20_row["epoch"]),
        "best_valid_mrr20": best_mrr20,
        "best_valid_recall20_epoch": int(best_recall20_row["epoch"]),
        "best_valid_recall20": best_recall20,
        "epoch8_mrr20": epoch8_mrr20,
        "epoch8_recall20": epoch8_recall20,
        "epoch8_status": epoch8_status(
            None if epoch8_row is None else int(epoch8_row["epoch"]),
            int(best_mrr20_row["epoch"]),
        ),
        "epoch8_to_best_mrr20_abs_delta": mrr20_abs_delta,
        "epoch8_to_best_mrr20_rel_delta": mrr20_rel_delta,
        "epoch8_to_best_recall20_abs_delta": recall20_abs_delta,
        "epoch8_to_best_recall20_rel_delta": recall20_rel_delta,
        "epoch8_close_to_best_mrr20": _close_to_best(mrr20_rel_delta),
        "epoch8_recall20_close_to_best": _close_to_best(recall20_rel_delta),
        "relative_close_threshold": CLOSE_RELATIVE_THRESHOLD,
        "test_metrics_note": TEST_METRICS_NOTE,
        "metric_scale": "proportion_0_to_1",
        "data_sources": dict(data_sources),
        "checkpoint_paths": dict(checkpoint_paths),
    }


def render_markdown_report(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Clean SR-GNN Training Sanity Check",
        "",
        "## Verdict",
        f"- epoch8_status: `{summary.get('epoch8_status')}`",
        (
            "- epoch 8 MRR@20 delta to best: "
            f"{_format_float(summary.get('epoch8_to_best_mrr20_abs_delta'))} abs, "
            f"{_format_percent(summary.get('epoch8_to_best_mrr20_rel_delta'))} rel"
        ),
        (
            "- epoch 8 Recall@20 delta to best: "
            f"{_format_float(summary.get('epoch8_to_best_recall20_abs_delta'))} abs, "
            f"{_format_percent(summary.get('epoch8_to_best_recall20_rel_delta'))} rel"
        ),
        f"- test metrics note: {summary.get('test_metrics_note')}",
        "",
        "## Best Epochs",
        "| metric | epoch | value |",
        "|---|---:|---:|",
        (
            "| valid MRR@20 | "
            f"{summary['best_valid_mrr20_epoch']} | "
            f"{_format_float(summary['best_valid_mrr20'])} |"
        ),
        (
            "| valid Recall@20 | "
            f"{summary['best_valid_recall20_epoch']} | "
            f"{_format_float(summary['best_valid_recall20'])} |"
        ),
        (
            "| epoch 8 MRR@20 | 8 | "
            f"{_format_float(summary.get('epoch8_mrr20'))} |"
        ),
        (
            "| epoch 8 Recall@20 | 8 | "
            f"{_format_float(summary.get('epoch8_recall20'))} |"
        ),
        "",
        "## Run Setup",
        f"- config: `{summary['config_path']}`",
        f"- train_config_source: `{summary['train_config_source']}`",
        f"- seed: `{summary['seed']}`",
        f"- max_epochs: `{summary['max_epochs']}`",
        f"- patience: `{summary['patience']}`",
        f"- improvement_rule: `{summary['improvement_rule']}`",
        f"- metric_scale: `{summary['metric_scale']}`",
        "",
        "## Data Sources",
    ]
    data_sources = summary.get("data_sources", {})
    if isinstance(data_sources, Mapping):
        for key in sorted(data_sources):
            lines.append(f"- {key}: `{data_sources[key]}`")
    lines.extend(
        [
            "",
            "## Epoch History",
            (
                "| epoch | train_loss | valid MRR@20 | valid Recall@20 | "
                "test MRR@20 | test Recall@20 | best | bad_counter | lr |"
            ),
            "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {epoch} | {train_loss} | {valid_mrr20} | {valid_recall20} | "
            "{test_mrr20} | {test_recall20} | {best} | {bad_counter} | {lr} |".format(
                epoch=int(row["epoch"]),
                train_loss=_format_float(row.get("train_loss")),
                valid_mrr20=_format_float(row.get("valid_ground_truth_mrr@20")),
                valid_recall20=_format_float(row.get("valid_ground_truth_recall@20")),
                test_mrr20=_format_float(row.get("test_ground_truth_mrr@20")),
                test_recall20=_format_float(row.get("test_ground_truth_recall@20")),
                best="yes" if row.get("is_best_mrr20") else "",
                bad_counter=int(row.get("bad_counter", 0)),
                lr=_format_float(row.get("lr")),
            )
        )
    return "\n".join(lines) + "\n"


def _format_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.9f}"


def _format_percent(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:.4f}%"


def _save_json(payload: Any, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return output_path


def _data_sources_payload(
    config: Config,
    *,
    train_pair_count: int,
    valid_pair_count: int,
    test_pair_count: int,
    train_session_count: int,
    valid_session_count: int,
    test_session_count: int,
) -> dict[str, Any]:
    paths = canonical_split_paths(config, split_key=split_key(config))
    return {
        "train_source": "canonical_dataset.train_sub",
        "valid_source": "canonical_dataset.valid",
        "test_source": "canonical_dataset.test",
        "train_sub_artifact": str(paths["train_sub"]),
        "valid_artifact": str(paths["valid"]),
        "test_artifact": str(paths["test"]),
        "srgnn_train_pair_source": "in_memory_pairs_from_train_sub_not_train_plus_valid",
        "train_session_count": int(train_session_count),
        "valid_session_count": int(valid_session_count),
        "test_session_count": int(test_session_count),
        "train_pair_count": int(train_pair_count),
        "valid_pair_count": int(valid_pair_count),
        "test_pair_count": int(test_pair_count),
    }


def run_diagnostic(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path | None = None,
    max_epochs: int = 30,
    patience: int = 10,
    best_metric: str = BEST_METRIC,
    train_config_source: str = "poison_model",
    seed: int | None = None,
    include_test_diagnostics: bool = True,
) -> dict[str, Any]:
    if str(best_metric).strip() != BEST_METRIC:
        raise ValueError(f"Only {BEST_METRIC!r} is supported for best checkpoint selection.")
    if int(max_epochs) <= 0:
        raise ValueError("max_epochs must be positive.")
    if int(patience) <= 0:
        raise ValueError("patience must be positive.")

    config_path = Path(config_path)
    config = load_config(config_path)
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_ROOT / config_path.stem
    )
    checkpoint_dir = resolved_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        "best_validation": str(checkpoint_dir / "best_validation.pt"),
        "epoch_008": str(checkpoint_dir / "epoch_008.pt"),
        "final": str(checkpoint_dir / "final.pt"),
    }

    resolved_seed = int(seed) if seed is not None else _default_seed(config, train_config_source)
    deps = _load_training_deps()
    deps["set_seed"](resolved_seed)

    print(
        "[clean-srgnn-sanity] loading canonical dataset "
        f"config={config_path} train_source=canonical_dataset.train_sub",
        flush=True,
    )
    canonical_dataset = ensure_canonical_dataset(config)
    train_sessions, train_labels = _sequences_to_pairs(canonical_dataset.train_sub)
    valid_sessions, valid_labels = _sequences_to_pairs(canonical_dataset.valid)
    test_sessions, test_labels = _sequences_to_pairs(canonical_dataset.test)
    if not train_sessions:
        raise ValueError("No SR-GNN train prefixes derived from canonical_dataset.train_sub.")
    if not valid_sessions:
        raise ValueError("No SR-GNN validation prefixes derived from canonical_dataset.valid.")
    if not test_sessions:
        raise ValueError("No SR-GNN test prefixes derived from canonical_dataset.test.")

    Data = deps["Data"]
    SRGNNVictimRunner = deps["SRGNNVictimRunner"]
    train_data = Data((train_sessions, train_labels), shuffle=True)
    valid_data = Data((valid_sessions, valid_labels), shuffle=False)
    test_data = Data((test_sessions, test_labels), shuffle=False)

    train_config = _resolve_train_config(
        config,
        train_config_source,
        max_epochs=int(max_epochs),
    )
    runner = SRGNNVictimRunner(config)
    runner.build_model(_build_srgnn_opt_from_train_config(train_config))

    rows: list[dict[str, Any]] = []
    best_valid_mrr20 = float("-inf")
    bad_counter = 0
    stopped_early = False
    stopped_epoch: int | None = None
    start = time.perf_counter()

    for epoch in range(1, int(max_epochs) + 1):
        print(
            f"[clean-srgnn-sanity] epoch={epoch}/{int(max_epochs)} training...",
            flush=True,
        )
        train_loss = _train_one_epoch(runner, train_data)
        valid_metrics = _evaluate_ground_truth_for_data(runner, valid_data, valid_labels)
        test_metrics = (
            _evaluate_ground_truth_for_data(runner, test_data, test_labels)
            if include_test_diagnostics
            else {}
        )

        current_valid_mrr20 = float(valid_metrics["ground_truth_mrr@20"])
        is_best = improved_over_best(current_valid_mrr20, best_valid_mrr20)
        if is_best:
            best_valid_mrr20 = current_valid_mrr20
            bad_counter = 0
            runner.save_model(checkpoint_paths["best_validation"])
        else:
            bad_counter += 1
        if epoch == EPOCH8:
            runner.save_model(checkpoint_paths["epoch_008"])

        row: dict[str, Any] = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            **_prefixed_metrics("valid", valid_metrics),
            **_prefixed_metrics("test", test_metrics),
            "is_best_mrr20": bool(is_best),
            "bad_counter": int(bad_counter),
            "lr": _current_lr(runner),
            "elapsed_seconds": float(time.perf_counter() - start),
        }
        rows.append(row)
        print(
            "[clean-srgnn-sanity] "
            f"epoch={epoch} train_loss={train_loss:.6g} "
            f"valid_mrr20={row['valid_ground_truth_mrr@20']:.9f} "
            f"valid_recall20={row['valid_ground_truth_recall@20']:.9f} "
            f"best={is_best} bad_counter={bad_counter}",
            flush=True,
        )

        if bad_counter >= int(patience):
            stopped_early = True
            stopped_epoch = int(epoch)
            print(
                f"[clean-srgnn-sanity] early stop at epoch={epoch} patience={int(patience)}",
                flush=True,
            )
            break

    runner.save_model(checkpoint_paths["final"])

    data_sources = _data_sources_payload(
        config,
        train_pair_count=len(train_sessions),
        valid_pair_count=len(valid_sessions),
        test_pair_count=len(test_sessions),
        train_session_count=len(canonical_dataset.train_sub),
        valid_session_count=len(canonical_dataset.valid),
        test_session_count=len(canonical_dataset.test),
    )
    summary = build_summary(
        rows,
        config_path=config_path,
        output_dir=resolved_output_dir,
        max_epochs=int(max_epochs),
        patience=int(patience),
        train_config_source=train_config_source,
        seed=resolved_seed,
        data_sources=data_sources,
        checkpoint_paths=checkpoint_paths,
        stopped_early=stopped_early,
        stopped_epoch=stopped_epoch,
    )
    history_payload = {
        "config_path": str(config_path),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "best_checkpoint_metric": VALID_BEST_METRIC_KEY,
        "improvement_rule": "improved = current_valid_mrr20 > best_valid_mrr20",
        "train_config_source": str(train_config_source),
        "seed": int(resolved_seed),
        "test_metrics_note": TEST_METRICS_NOTE,
        "data_sources": data_sources,
        "rows": rows,
    }
    output_paths = {
        "history": _save_json(history_payload, resolved_output_dir / "history.json"),
        "summary": _save_json(summary, resolved_output_dir / "summary.json"),
    }
    report_path = resolved_output_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(summary, rows), encoding="utf-8")
    output_paths["report"] = report_path

    print("[clean-srgnn-sanity] completed. Outputs:", flush=True)
    for label, path in output_paths.items():
        print(f"[clean-srgnn-sanity]   {label}: {path}", flush=True)
    return {
        "summary": summary,
        "history": history_payload,
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "checkpoint_paths": checkpoint_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean SR-GNN training sanity check with validation best checkpoint diagnostics."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to outputs/diagnostics/clean_srgnn_training_sanity/<config_stem>.",
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--best-metric",
        default=BEST_METRIC,
        choices=[BEST_METRIC],
        help="Validation metric used for model selection.",
    )
    parser.add_argument(
        "--train-config-source",
        default="poison_model",
        choices=["poison_model", "victim_srgnn"],
        help="SR-GNN hyperparameter source. Default matches clean poison-model generation.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional explicit training seed.")
    parser.add_argument(
        "--skip-test-diagnostics",
        action="store_true",
        help="Skip per-epoch test metrics. Test metrics are diagnostic-only when enabled.",
    )
    args = parser.parse_args()
    result = run_diagnostic(
        config_path=args.config,
        output_dir=args.output_dir,
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        best_metric=args.best_metric,
        train_config_source=args.train_config_source,
        seed=args.seed,
        include_test_diagnostics=not bool(args.skip_test_diagnostics),
    )
    print(json.dumps(result["output_paths"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BEST_METRIC",
    "CLOSE_RELATIVE_THRESHOLD",
    "EPOCH8",
    "TEST_METRICS_NOTE",
    "build_summary",
    "epoch8_status",
    "improved_over_best",
    "render_markdown_report",
    "run_diagnostic",
]
