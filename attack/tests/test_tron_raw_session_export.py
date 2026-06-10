from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace

from attack.common.config import ArtifactsConfig, load_config
import attack.common.paths as paths
from attack.common.paths import (
    run_group_key,
    target_cohort_key,
    target_selection_key,
    victim_prediction_key,
    victim_prediction_key_payload,
)
from attack.data.canonical_dataset import CanonicalDataset
from attack.data.exporters.base_exporter import ExportResult
from attack.data.exporters.miasrec_exporter import MiaSRecExporter
from attack.data.exporters.srgnn_exporter import SRGNNExporter, load_srg_nn_train
from attack.data.exporters.tron_exporter import TRONExporter
from attack.pipeline.core.victim_execution import execute_single_victim
from attack.pipeline.core.pipeline_utils import SharedAttackArtifacts
from attack.pipeline.runs import run_random_nonzero as random_nonzero_runner


CONFIG_PATH = Path("attack/configs/diginetica_valbest_clean_sample10.yaml")


def _dataset() -> CanonicalDataset:
    return CanonicalDataset(
        train_sub=[[1, 2, 3, 4]],
        valid=[[8, 9, 10]],
        test=[[11, 12, 13]],
        item_map={},
        metadata={"dataset_name": "toy"},
    )


def _read_sequences(path: Path) -> list[list[int]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [
        [int(event["aid"]) for event in row["events"]]
        for row in rows
    ]


def test_tron_clean_export_keeps_one_raw_train_sequence(tmp_path: Path) -> None:
    result = TRONExporter().export(_dataset(), tmp_path)

    assert _read_sequences(result.files["train"]) == [[1, 2, 3, 4]]
    assert _read_sequences(result.files["valid"]) == [[8, 9, 10]]
    assert _read_sequences(result.files["test"]) == [[11, 12, 13]]


def test_tron_raw_poisoned_export_appends_deep_copied_raw_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    raw_fake_sessions = [[5, 6, 99, 7]]
    captured: dict[str, list[list[int]]] = {}

    def fake_export_sequences(self, dataset, *, output_dir, train_sequences, dataset_name=None):
        captured["train_sequences"] = list(train_sequences)
        return ExportResult(output_dir=tmp_path, files={})

    monkeypatch.setattr(TRONExporter, "_export_sequences", fake_export_sequences)

    TRONExporter().export_with_raw_poisoned_train(
        dataset,
        raw_fake_sessions=raw_fake_sessions,
        output_dir=tmp_path,
        dataset_name="toy",
    )
    dataset.train_sub[0][0] = 1000
    raw_fake_sessions[0][0] = 2000

    assert captured["train_sequences"] == [[1, 2, 3, 4], [5, 6, 99, 7]]


def test_tron_raw_poisoned_export_writes_clean_plus_fake_sequences(tmp_path: Path) -> None:
    result = TRONExporter().export_with_raw_poisoned_train(
        _dataset(),
        raw_fake_sessions=[[5, 6, 99, 7]],
        output_dir=tmp_path,
        dataset_name="toy",
    )

    assert _read_sequences(result.files["train"]) == [
        [1, 2, 3, 4],
        [5, 6, 99, 7],
    ]


def test_srgnn_and_miasrec_poisoned_pair_exports_are_unchanged(tmp_path: Path) -> None:
    poisoned_sessions = [[1], [1, 2]]
    poisoned_labels = [2, 3]

    srgnn_path = SRGNNExporter().export_train_pairs(
        poisoned_sessions,
        poisoned_labels,
        tmp_path / "srgnn_train.txt",
    )
    assert load_srg_nn_train(srgnn_path) == (poisoned_sessions, poisoned_labels)

    miasrec_result = MiaSRecExporter().export_with_poisoned_train(
        _dataset(),
        poisoned_sessions=poisoned_sessions,
        poisoned_labels=poisoned_labels,
        output_dir=tmp_path / "miasrec",
        dataset_name="toy",
    )
    train_lines = miasrec_result.files["train"].read_text(encoding="utf-8").splitlines()
    assert train_lines[1:] == ["1\t1\t2", "2\t1 2\t3"]


def test_formal_tron_branch_uses_raw_export_not_pair_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(CONFIG_PATH)
    raw_fake_sessions = [[5, 6, 99, 7]]
    captured: dict[str, object] = {}

    def fail_pair_export(*args, **kwargs):
        raise AssertionError("formal TRON branch must not call pair-based export")

    def fake_raw_export(self, dataset, *, raw_fake_sessions, output_dir, dataset_name=None):
        captured["raw_fake_sessions"] = [list(session) for session in raw_fake_sessions]
        return ExportResult(output_dir=Path(output_dir), files={"train": Path(output_dir) / "train"})

    class FakeTRONRunner:
        def __init__(self, config):
            pass

        def run(self, **kwargs):
            return {"log_dir": str(tmp_path / "logs")}

        def predict_topk(self, **kwargs):
            return [[1, 2, 3]]

    monkeypatch.setattr(TRONExporter, "export_with_poisoned_train", fail_pair_export)
    monkeypatch.setattr(TRONExporter, "export_with_raw_poisoned_train", fake_raw_export)
    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution.get_victim_runner",
        lambda victim_name: FakeTRONRunner,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution._write_victim_resolved_config",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "attack.pipeline.core.victim_execution._save_tron_history",
        lambda *args, **kwargs: None,
    )

    result = execute_single_victim(
        config,
        run_type="attack",
        victim_name="tron",
        canonical_dataset=_dataset(),
        poisoned_sessions=[[1], [1, 2], [5, 6, 99]],
        poisoned_labels=[2, 3, 7],
        raw_fake_sessions=raw_fake_sessions,
        run_dir=tmp_path / "victim",
        poisoned_train_path=tmp_path / "poisoned_train.txt",
        target_item=99,
        eval_topk=(10,),
        predictions_path=None,
    )

    assert captured["raw_fake_sessions"] == raw_fake_sessions
    assert result.predictions == [[1, 2, 3]]


def test_only_tron_victim_key_has_raw_session_semantics_version() -> None:
    config = load_config(CONFIG_PATH)

    tron_payload = victim_prediction_key_payload(config, "tron", run_type="clean")
    srgnn_payload = victim_prediction_key_payload(config, "srgnn", run_type="clean")
    miasrec_payload = victim_prediction_key_payload(config, "miasrec", run_type="clean")

    assert tron_payload["victim_data_semantics"] == "tron_raw_session_export_v1"
    assert "victim_data_semantics" not in srgnn_payload
    assert "victim_data_semantics" not in miasrec_payload


def test_tron_semantics_version_changes_only_tron_victim_identity(monkeypatch) -> None:
    config = load_config(CONFIG_PATH)
    run_type = "clean"
    before = {
        "run_group": run_group_key(config, run_type=run_type),
        "target_cohort": target_cohort_key(config),
        "target_selection": target_selection_key(config),
        "srgnn": victim_prediction_key(config, "srgnn", run_type=run_type),
        "miasrec": victim_prediction_key(config, "miasrec", run_type=run_type),
        "tron": victim_prediction_key(config, "tron", run_type=run_type),
    }

    monkeypatch.setattr(paths, "TRON_VICTIM_DATA_SEMANTICS", "tron_raw_session_export_v2_test")

    assert run_group_key(config, run_type=run_type) == before["run_group"]
    assert target_cohort_key(config) == before["target_cohort"]
    assert target_selection_key(config) == before["target_selection"]
    assert victim_prediction_key(config, "srgnn", run_type=run_type) == before["srgnn"]
    assert victim_prediction_key(config, "miasrec", run_type=run_type) == before["miasrec"]
    assert victim_prediction_key(config, "tron", run_type=run_type) != before["tron"]


def test_random_nonzero_returns_final_modified_raw_fake_sessions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = load_config(
        "attack/configs/diginetica_valbest_attack_random_nonzero_when_possible_ratio1_sample10.yaml"
    )
    config = replace(
        config,
        artifacts=ArtifactsConfig(
            root=str(tmp_path),
            shared_dir="shared",
            runs_dir="runs",
            cleanup_victim_intermediates=False,
        ),
    )
    shared = SharedAttackArtifacts(
        stats=SimpleNamespace(item_counts={item: 1 for item in range(1, 200)}),
        clean_sessions=[[1]],
        clean_labels=[2],
        canonical_dataset=_dataset(),
        export_paths={},
        template_sessions=[[1, 2, 3], [4, 5, 6]],
        poison_runner=None,
        fake_session_count=2,
        shared_paths={},
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        random_nonzero_runner,
        "prepare_shared_attack_artifacts",
        lambda *args, **kwargs: shared,
    )

    def fake_run_targets_and_victims(*args, **kwargs):
        payload = kwargs["build_poisoned"](99)
        captured["raw_fake_sessions"] = payload.raw_fake_sessions
        return {"status": "ok"}

    monkeypatch.setattr(
        random_nonzero_runner,
        "run_targets_and_victims",
        fake_run_targets_and_victims,
    )

    assert random_nonzero_runner.run_random_nonzero(config) == {"status": "ok"}
    assert captured["raw_fake_sessions"] == [[1, 2, 99], [4, 99, 6]]
