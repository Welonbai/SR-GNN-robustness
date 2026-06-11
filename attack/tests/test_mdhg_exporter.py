from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from attack.common.config import load_config
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.mdhg_exporter import MDHGExporter, sequences_to_pairs
from attack.pipeline.core.victim_execution import execute_single_victim


def _dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train_sub=[[1, 2, 3], [2, 4]],
        valid=[[1, 4]],
        test=[[1, 3, 4], [2, 3]],
        item_map={"a": 1, "b": 2, "c": 3, "d": 4, "unused": 5},
        metadata={"dataset_name": "toy", "counts": {"items": 5}},
    )


def _read(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def test_clean_export_writes_required_pickles_and_canonical_test_order(tmp_path) -> None:
    result = MDHGExporter().export(_dataset(), tmp_path)

    assert set(result.files) == {"train", "test", "all_train_seq"}
    assert [path.name for path in result.files.values()] == [
        "train.txt",
        "test.txt",
        "all_train_seq.txt",
    ]
    assert _read(result.files["train"]) == sequences_to_pairs(_dataset().train_sub)
    assert _read(result.files["test"]) == (
        [[1, 3], [1], [2]],
        [4, 3, 3],
    )
    assert _read(result.files["all_train_seq"]) == _dataset().train_sub
    assert result.n_node == 5
    assert result.observed_max_item_id == 4
    assert result.train_pairs_match_raw_expansion is True


def test_attack_export_keeps_authoritative_pairs_and_reports_raw_divergence(tmp_path) -> None:
    result = MDHGExporter().export_with_poisoned_train(
        _dataset(),
        poisoned_sessions=[[1], [2, 5]],
        poisoned_labels=[5, 1],
        raw_fake_sessions=[[4, 5]],
        output_dir=tmp_path,
        dataset_name="toy",
    )

    assert _read(result.files["train"]) == ([[1], [2, 5]], [5, 1])
    assert _read(result.files["all_train_seq"]) == [[1, 2, 3], [2, 4], [4, 5]]
    assert result.raw_train_session_count == 3
    assert result.expected_raw_expanded_pair_count == 4
    assert result.train_pairs_match_raw_expansion is False
    assert result.n_node == 5


def test_export_rejects_zero_and_out_of_range_ids(tmp_path) -> None:
    with pytest.raises(ValueError, match="reserved for padding"):
        MDHGExporter().export_with_poisoned_train(
            _dataset(),
            poisoned_sessions=[[0, 1]],
            poisoned_labels=[2],
            raw_fake_sessions=[],
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="exceeds canonical n_node"):
        MDHGExporter().export_with_poisoned_train(
            _dataset(),
            poisoned_sessions=[[1]],
            poisoned_labels=[6],
            raw_fake_sessions=[],
            output_dir=tmp_path,
        )


def test_clean_execution_exports_clean_pairs_and_no_fake_raw_sessions(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "configs"
        / "diginetica_attack_dpsbr.yaml"
    )
    config = load_config(config_path)
    params = dict(config.victims.params)
    params["mdhg"] = {
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    config = replace(config, victims=replace(config.victims, params=params))
    dataset = _dataset()
    clean_sessions, clean_labels = sequences_to_pairs(dataset.train_sub)

    class FakeMDHGRunner:
        def __init__(self, _config):
            pass

        def run(self, **kwargs):
            prediction_path = Path(kwargs["export_topk_path"])
            prediction_path.write_text(
                '{"topk": 1, "requested_topk": 1, "n_node": 5, '
                '"rankings": [[1], [1], [1]]}',
                encoding="utf-8",
            )
            return {"log_path": str(tmp_path / "missing.log")}

        def predict_topk(self, **kwargs):
            return [[1], [1], [1]]

    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution.get_victim_runner",
        lambda victim_name: FakeMDHGRunner,
    )
    run_dir = tmp_path / "victim"
    execute_single_victim(
        config,
        run_type="clean",
        victim_name="mdhg",
        canonical_dataset=dataset,
        poisoned_sessions=clean_sessions,
        poisoned_labels=clean_labels,
        raw_fake_sessions=[],
        run_dir=run_dir,
        poisoned_train_path=run_dir / "unused.pkl",
        target_item=1,
        eval_topk=[1],
        predictions_path=run_dir / "predictions.json",
    )

    data_dir = run_dir / "export" / "mdhg" / config.data.dataset_name
    assert _read(data_dir / "train.txt") == sequences_to_pairs(dataset.train_sub)
    assert _read(data_dir / "all_train_seq.txt") == dataset.train_sub


def test_execution_records_and_summarizes_epoch_diagnostic_context(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "attack"
        / "configs"
        / "diginetica_attack_dpsbr.yaml"
    )
    config = load_config(config_path)
    params = dict(config.victims.params)
    params["mdhg"] = {
        "train": {
            "epochs": 1,
            "batch_size": 4,
            "lr": 0.001,
            "checkpoint_protocol": "fixed_epoch",
            "validation_enabled": False,
            "export_model": "last",
        }
    }
    runtime = dict(config.victims.runtime or {})
    runtime["mdhg"] = {
        "python_executable": "python",
        "repo_root": "third_party/mdhg",
        "working_dir": "third_party/mdhg",
        "device": {"use_gpu": True, "gpu_id": "0"},
        "diagnostics": {
            "epoch_metrics": True,
            "per_epoch_predictions": True,
        },
    }
    config = replace(
        config,
        victims=replace(config.victims, params=params, runtime=runtime),
    )
    dataset = _dataset()
    clean_sessions, clean_labels = sequences_to_pairs(dataset.train_sub)

    class FakeMDHGRunner:
        def __init__(self, _config):
            pass

        def run(self, **kwargs):
            prediction_path = Path(kwargs["export_topk_path"])
            payload = {
                "topk": 1,
                "requested_topk": 1,
                "n_node": 5,
                "rankings": [[4], [3], [3]],
            }
            prediction_path.write_text(json.dumps(payload), encoding="utf-8")
            per_epoch_dir = Path(kwargs["run_dir"]) / "mdhg_per_epoch_predictions"
            per_epoch_dir.mkdir()
            (per_epoch_dir / "epoch_001_topk.json").write_text(
                json.dumps({"epoch": 1, **payload}),
                encoding="utf-8",
            )
            return {
                "log_path": str(tmp_path / "missing.log"),
                "per_epoch_prediction_dir": str(per_epoch_dir.resolve()),
            }

        def predict_topk(self, **kwargs):
            return [[4], [3], [3]]

    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution.get_victim_runner",
        lambda victim_name: FakeMDHGRunner,
    )
    run_dir = tmp_path / "victim"
    execute_single_victim(
        config,
        run_type="clean",
        victim_name="mdhg",
        canonical_dataset=dataset,
        poisoned_sessions=clean_sessions,
        poisoned_labels=clean_labels,
        raw_fake_sessions=[],
        run_dir=run_dir,
        poisoned_train_path=run_dir / "unused.pkl",
        target_item=1,
        eval_topk=[1],
        predictions_path=run_dir / "predictions.json",
    )

    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    injected = resolved["pipeline_injected"]
    assert injected["target_item"] == 1
    assert injected["evaluation_topk"] == [1]
    assert injected["targeted_metrics"] == list(config.evaluation.targeted_metrics)
    assert injected["ground_truth_metrics"] == list(config.evaluation.ground_truth_metrics)
    assert Path(injected["mdhg_test_data_path"]).name == "test.txt"
    assert Path(injected["per_epoch_prediction_dir"]).name == "mdhg_per_epoch_predictions"
    rows = [
        json.loads(line)
        for line in (run_dir / "mdhg_epoch_pipeline_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["epoch"] == 1
    assert rows[0]["metrics"]["ground_truth_recall@1"] == 1.0
