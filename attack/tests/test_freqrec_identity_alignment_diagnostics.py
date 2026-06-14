from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from attack.common.paths import (
    freqrec_diagnostic_key,
    run_group_key_payload,
    victim_prediction_key,
)
from attack.common.config import _normalize_freqrec_train
from attack.data.canonical_dataset import CanonicalDataset
from attack.models.victim.freqrec_diagnostics import (
    load_freqrec_epoch_metrics,
    summarize_freqrec_epoch_diagnostics,
)
from attack.models.victim.freqrec_runner import FreqRecRunner
from attack.pipeline.core.ground_truth_alignment import resolve_ground_truth_labels
from attack.tests.freqrec_test_utils import freqrec_config, prediction_payload
from attack.pipeline.runs import run_victim_valbest_epoch_diagnostic as diagnostic_module


def _epoch_row(epoch, *, train_loss, validation):
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation": validation,
        "train_runtime_seconds": 1.0,
        "validation_runtime_seconds": 0.5,
        "epoch_runtime_seconds": 1.5,
        "batch_size": 4,
        "train_example_count": 5,
        "train_batch_count": 2,
        "train_final_batch_size": 1,
        "validation_example_count": 2,
        "validation_batch_count": 1,
        "validation_final_batch_size": 2,
        "num_workers": 0,
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "checkpoint_protocol": "fixed_epoch",
        "improved": None,
        "best_epoch": None,
        "best_metric": None,
    }


def test_alignment_has_explicit_hand_written_prefix_label_order(tmp_path):
    dataset = CanonicalDataset(
        train_sub=[],
        valid=[[1, 2, 3]],
        test=[[1, 2, 3], [4, 5]],
        item_map={str(i): i for i in range(1, 6)},
        metadata={"item_count": 5},
    )
    labels = resolve_ground_truth_labels(
        freqrec_config(tmp_path),
        victim_name="freqrec",
        canonical_dataset=dataset,
        predictions=[[1], [2], [3]],
    )
    assert labels == [3, 2, 5]


def test_freqrec_identity_includes_batch_seed_epochs_and_model_but_not_runtime(tmp_path):
    base = freqrec_config(tmp_path)
    base_key = victim_prediction_key(base, "freqrec", run_type="clean")
    for overrides in (
        {"batch_size": 8},
        {"epochs": 3},
        {"hidden_size": 16},
    ):
        changed = freqrec_config(tmp_path, train_overrides=overrides)
        assert victim_prediction_key(changed, "freqrec", run_type="clean") != base_key
    changed_seed = replace(
        base, seeds=replace(base.seeds, victim_train_seed=base.seeds.victim_train_seed + 1)
    )
    assert victim_prediction_key(changed_seed, "freqrec", run_type="clean") != base_key
    runtime_changed = freqrec_config(
        tmp_path,
        runtime_overrides={
            "python_executable": "different-python",
            "device": {"gpu_id": "7"},
            "dataloader": {"num_workers": 9},
        },
    )
    assert victim_prediction_key(runtime_changed, "freqrec", run_type="clean") == base_key


def test_fixed_epoch_keeps_metric_cutoffs_but_excludes_monitor_and_patience(tmp_path):
    base = freqrec_config(tmp_path)
    monitor_changed = freqrec_config(
        tmp_path,
        train_overrides={
            "validation_metric": "mrr@20",
            "patience": 99,
        },
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        monitor_changed, "freqrec", run_type="clean"
    )
    cutoffs_changed = freqrec_config(
        tmp_path, train_overrides={"metric_cutoffs": [5, 20]}
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") != victim_prediction_key(
        cutoffs_changed, "freqrec", run_type="clean"
    )
    changed_eval = replace(
        base, evaluation=replace(base.evaluation, topk=(5, 10))
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        changed_eval, "freqrec", run_type="clean"
    )
    assert run_group_key_payload(base, run_type="clean") != run_group_key_payload(
        changed_eval, run_type="clean"
    )


def test_validation_best_monitor_changes_identity(tmp_path):
    base = freqrec_config(
        tmp_path,
        train_overrides={"checkpoint_protocol": "validation_best"},
    )
    changed = freqrec_config(
        tmp_path,
        train_overrides={
            "checkpoint_protocol": "validation_best",
            "validation_metric": "mrr@20",
        },
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") != victim_prediction_key(
        changed, "freqrec", run_type="clean"
    )
    patience_changed = freqrec_config(
        tmp_path,
        train_overrides={
            "checkpoint_protocol": "validation_best",
            "patience": 99,
        },
    )
    assert victim_prediction_key(base, "freqrec", run_type="clean") == victim_prediction_key(
        patience_changed, "freqrec", run_type="clean"
    )


def test_freqrec_diagnostic_path_uses_scientific_identity_only(tmp_path):
    base = freqrec_config(tmp_path)
    same = freqrec_config(tmp_path)
    configured_epochs = int(base.victims.params["freqrec"]["train"]["epochs"])
    assert diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=configured_epochs
    ) == diagnostic_module._freqrec_diagnostic_dir(
        same, effective_epochs=configured_epochs
    )
    assert freqrec_diagnostic_key(
        base, effective_epochs=configured_epochs
    ) == freqrec_diagnostic_key(same, effective_epochs=configured_epochs)

    assert diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=20
    ) != diagnostic_module._freqrec_diagnostic_dir(base, effective_epochs=30)
    assert freqrec_diagnostic_key(
        base, effective_epochs=20
    ) != freqrec_diagnostic_key(base, effective_epochs=30)

    batch_changed = freqrec_config(tmp_path, train_overrides={"batch_size": 8})
    profile_changed = freqrec_config(tmp_path, train_overrides={"alpha": 0.4})
    assert diagnostic_module._freqrec_diagnostic_dir(
        batch_changed, effective_epochs=configured_epochs
    ) != diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=configured_epochs
    )
    assert diagnostic_module._freqrec_diagnostic_dir(
        profile_changed, effective_epochs=configured_epochs
    ) != diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=configured_epochs
    )

    runtime_changed = freqrec_config(
        tmp_path,
        runtime_overrides={
            "python_executable": "other-python",
            "device": {"gpu_id": "7"},
            "dataloader": {"num_workers": 11},
        },
    )
    assert diagnostic_module._freqrec_diagnostic_dir(
        runtime_changed, effective_epochs=configured_epochs
    ) == diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=configured_epochs
    )

    other_dataset = replace(
        base,
        data=replace(base.data, dataset_name="other_dataset"),
    )
    assert diagnostic_module._freqrec_diagnostic_dir(
        other_dataset, effective_epochs=configured_epochs
    ) != diagnostic_module._freqrec_diagnostic_dir(
        base, effective_epochs=configured_epochs
    )


def test_freqrec_config_validation_rejects_ablation_and_invalid_fft_loss():
    from attack.tests.freqrec_test_utils import freqrec_train

    assert _normalize_freqrec_train(
        freqrec_train(), "victims.params.freqrec.train"
    )["fourier_loss"] is True
    with pytest.raises(ValueError, match="fre must be exactly 1.0"):
        _normalize_freqrec_train(
            freqrec_train(fre=0.5), "victims.params.freqrec.train"
        )
    with pytest.raises(ValueError, match="fourier_loss must be true"):
        _normalize_freqrec_train(
            freqrec_train(fourier_loss=False), "victims.params.freqrec.train"
        )
    with pytest.raises(ValueError, match="fft_loss_type"):
        _normalize_freqrec_train(
            freqrec_train(fft_loss_type="cos"), "victims.params.freqrec.train"
        )
    for field in ("alpha", "gama", "alpha_loss"):
        for value in (0.0, 1.0):
            with pytest.raises(ValueError, match=field):
                _normalize_freqrec_train(
                    freqrec_train(**{field: value}),
                    "victims.params.freqrec.train",
                )
    with pytest.raises(ValueError, match="chux"):
        _normalize_freqrec_train(
            freqrec_train(chux="x"), "victims.params.freqrec.train"
        )
    with pytest.raises(ValueError, match="hidden_act"):
        _normalize_freqrec_train(
            freqrec_train(hidden_act="softplus"), "victims.params.freqrec.train"
        )


def test_epoch_metrics_parser_requires_contiguous_finite_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                _epoch_row(
                    epoch,
                    train_loss=1.0 / epoch,
                    validation={"ndcg@20": 0.1 * epoch},
                )
            )
            for epoch in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    assert [row["epoch"] for row in load_freqrec_epoch_metrics(path)] == [1, 2]
    path.write_text(
        json.dumps(_epoch_row(2, train_loss=float("nan"), validation={})),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_freqrec_epoch_metrics(path)


def test_epoch_metrics_protocol_selection_state_validation(tmp_path):
    path = tmp_path / "metrics.jsonl"
    invalid_fixed = _epoch_row(1, train_loss=1.0, validation={"ndcg@20": 0.1})
    invalid_fixed["improved"] = True
    path.write_text(json.dumps(invalid_fixed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain checkpoint-selection"):
        load_freqrec_epoch_metrics(path)

    valid_best = _epoch_row(1, train_loss=1.0, validation={"ndcg@20": 0.1})
    valid_best.update(
        checkpoint_protocol="validation_best",
        improved=True,
        best_epoch=1,
        best_metric=0.1,
    )
    path.write_text(json.dumps(valid_best) + "\n", encoding="utf-8")
    assert load_freqrec_epoch_metrics(path)[0]["best_epoch"] == 1
    valid_best["best_epoch"] = 0
    path.write_text(json.dumps(valid_best) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="best_epoch is invalid"):
        load_freqrec_epoch_metrics(path)


def test_validation_best_epoch_metrics_sequential_state(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = _epoch_row(1, train_loss=1.0, validation={"ndcg@20": 0.1})
    first.update(
        checkpoint_protocol="validation_best",
        improved=True,
        best_epoch=1,
        best_metric=0.1,
    )
    second = _epoch_row(2, train_loss=0.9, validation={"ndcg@20": 0.09})
    second.update(
        checkpoint_protocol="validation_best",
        improved=False,
        best_epoch=1,
        best_metric=0.1,
    )
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    assert len(load_freqrec_epoch_metrics(path)) == 2

    invalid_first = dict(first, improved=False)
    path.write_text(json.dumps(invalid_first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="initial best state"):
        load_freqrec_epoch_metrics(path)

    invalid_second = dict(second, best_epoch=2)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(invalid_second) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="previous best_epoch"):
        load_freqrec_epoch_metrics(path)

    invalid_metric = dict(second, best_metric=0.11)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(invalid_metric) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="previous best_metric"):
        load_freqrec_epoch_metrics(path)

    improved_second = dict(second, improved=True, best_epoch=1, best_metric=0.2)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(improved_second) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not the current epoch"):
        load_freqrec_epoch_metrics(path)


def test_validation_best_epoch_metrics_require_strict_metric_improvement(tmp_path):
    path = tmp_path / "metrics.jsonl"
    first = _epoch_row(1, train_loss=1.0, validation={"ndcg@20": 0.1})
    first.update(
        checkpoint_protocol="validation_best",
        improved=True,
        best_epoch=1,
        best_metric=0.1,
    )
    unchanged = _epoch_row(2, train_loss=0.9, validation={"ndcg@20": 0.09})
    unchanged.update(
        checkpoint_protocol="validation_best",
        improved=False,
        best_epoch=1,
        best_metric=0.1,
    )
    improved = _epoch_row(3, train_loss=0.8, validation={"ndcg@20": 0.2})
    improved.update(
        checkpoint_protocol="validation_best",
        improved=True,
        best_epoch=3,
        best_metric=0.2,
    )
    path.write_text(
        "\n".join(json.dumps(row) for row in (first, unchanged, improved)) + "\n",
        encoding="utf-8",
    )
    assert [row["epoch"] for row in load_freqrec_epoch_metrics(path)] == [1, 2, 3]

    for invalid_metric in (0.08, 0.1):
        invalid = dict(unchanged, improved=True, best_epoch=2, best_metric=invalid_metric)
        path.write_text(
            json.dumps(first) + "\n" + json.dumps(invalid) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="did not strictly increase"):
            load_freqrec_epoch_metrics(path)

    changed_metric = dict(unchanged, best_metric=0.11)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(changed_metric) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="previous best_metric"):
        load_freqrec_epoch_metrics(path)

    changed_epoch = dict(unchanged, best_epoch=2)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(changed_epoch) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="previous best_epoch"):
        load_freqrec_epoch_metrics(path)


def test_freqrec_run_diagnostic_does_not_resolve_attack_source(monkeypatch, tmp_path):
    config = freqrec_config(tmp_path)
    config = replace(
        config,
        artifacts=replace(config.artifacts, root=str(tmp_path / "outputs")),
    )
    monkeypatch.setattr(diagnostic_module, "load_config", lambda _: config)
    monkeypatch.setattr(diagnostic_module, "_load_yaml_mapping", lambda _: {})
    monkeypatch.setattr(
        diagnostic_module,
        "resolve_source_pts_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FreqRec diagnostic must not resolve PTS-CEM source")
        ),
    )
    monkeypatch.setattr(
        diagnostic_module,
        "_build_poisoned_train",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("FreqRec diagnostic must not build poisoned training data")
        ),
    )
    observed_dir_epochs = []
    observed_run_epochs = []
    original_dir = diagnostic_module._freqrec_diagnostic_dir

    def capture_dir(config, *, effective_epochs):
        observed_dir_epochs.append(effective_epochs)
        return original_dir(config, effective_epochs=effective_epochs)

    def capture_run(config, *, out_dir, effective_epochs):
        observed_run_epochs.append(effective_epochs)
        return {
            "victim_name": "freqrec",
            "dataset": config.data.dataset_name,
            "target_item": None,
            "effective_epochs": effective_epochs,
        }

    monkeypatch.setattr(diagnostic_module, "_freqrec_diagnostic_dir", capture_dir)
    monkeypatch.setattr(
        diagnostic_module,
        "_run_freqrec_diagnostic",
        capture_run,
    )
    configured_epochs = int(config.victims.params["freqrec"]["train"]["epochs"])
    default_result = diagnostic_module.run_diagnostic(
        "unused.yaml",
        victim="freqrec",
    )
    override_result = diagnostic_module.run_diagnostic(
        "unused.yaml",
        victim="freqrec",
        max_epochs=30,
    )
    assert default_result["target_item"] is None
    assert default_result["source"] is None
    assert default_result["victims"][0]["effective_epochs"] == configured_epochs
    assert override_result["victims"][0]["victim_name"] == "freqrec"
    assert override_result["victims"][0]["effective_epochs"] == 30
    assert observed_dir_epochs == [configured_epochs, 30]
    assert observed_run_epochs == [configured_epochs, 30]


def test_freqrec_diagnostic_exports_clean_pairs_and_uses_clean_seed(
    monkeypatch, tmp_path
):
    config = freqrec_config(
        tmp_path,
        runtime_overrides={
            "diagnostics": {
                "epoch_metrics": True,
                "per_epoch_predictions": True,
            }
        },
    )
    dataset = CanonicalDataset(
        train_sub=[[1, 2, 3]],
        valid=[[1, 2, 4]],
        test=[[2, 3, 5]],
        item_map={str(i): i for i in range(1, 6)},
        metadata={"item_count": 5},
    )
    captured = {}

    class FakeExporter:
        def export_with_train_pairs(self, canonical, **kwargs):
            captured["canonical"] = canonical
            captured["export"] = kwargs
            return SimpleNamespace(
                files={
                    "train": tmp_path / "train.jsonl",
                    "valid": tmp_path / "valid.jsonl",
                    "test": tmp_path / "test.jsonl",
                    "metadata": tmp_path / "metadata.json",
                },
                item_count=5,
                test_example_count=2,
            )

    class FakeRunner:
        def __init__(self, config):
            pass

        def run(self, **kwargs):
            captured["run"] = kwargs
            return {
                "epoch_metrics_output_path": str(tmp_path / "metrics.jsonl"),
                "per_epoch_prediction_dir": str(tmp_path / "predictions"),
            }

    monkeypatch.setattr(diagnostic_module, "ensure_canonical_dataset", lambda _: dataset)
    monkeypatch.setattr(diagnostic_module, "FreqRecExporter", FakeExporter)
    monkeypatch.setattr(diagnostic_module, "FreqRecRunner", FakeRunner)
    monkeypatch.setattr(
        diagnostic_module,
        "summarize_freqrec_epoch_diagnostics",
        lambda **kwargs: [],
    )
    summary = diagnostic_module._run_freqrec_diagnostic(
        config,
        out_dir=tmp_path / "diagnostic",
        effective_epochs=2,
    )
    assert captured["export"]["train_prefixes"] == [[1, 2], [1]]
    assert captured["export"]["train_labels"] == [3, 2]
    assert captured["export"]["mode"] == "clean"
    assert captured["run"]["target_item"] is None
    assert captured["run"]["victim_train_seed"] == diagnostic_module.victim_effective_train_seed(
        config,
        victim_name="freqrec",
        run_type="clean",
        target_item=0,
    )
    assert summary["diagnostic_scope"] == "dataset_victim_clean"


def test_diagnostic_recomputes_parent_validation_metrics_from_per_epoch_artifact(
    tmp_path,
):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps(
            _epoch_row(
                1,
                train_loss=1.0,
                validation={
                    "hr@20": 1.0,
                    "mrr@20": 0.75,
                    "ndcg@20": 0.8154648767857288,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    payload = prediction_payload(
        item_count=20,
        example_count=2,
        requested_topk=20,
        evaluation_topk=20,
        epochs=2,
        split="validation",
        current_epoch=1,
        selected_epoch=1,
        epochs_completed=1,
    )
    payload["rankings"][0]["items"] = list(range(1, 21))
    payload["rankings"][1]["items"] = [2, 1, *range(3, 21)]
    (prediction_dir / "epoch_001_validation_topk.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = summarize_freqrec_epoch_diagnostics(
        runner=FreqRecRunner(freqrec_config(tmp_path)),
        epoch_metrics_path=metrics_path,
        per_epoch_prediction_dir=prediction_dir,
        validation_labels=[1, 1],
        item_count=20,
        requested_topk=20,
        configured_epochs=2,
        seed=7,
        metric_cutoffs=[20],
    )
    assert rows[0]["parent_validation"]["ground_truth_recall@20"] == 1.0
    assert rows[0]["parent_validation"]["ground_truth_mrr@20"] == 0.75
    assert abs(rows[0]["consistency_delta"]["mrr@20"]) < 1e-12


def test_diagnostic_rejects_persisted_depth_below_parent_cutoff(tmp_path):
    with pytest.raises(ValueError, match="at least 20"):
        summarize_freqrec_epoch_diagnostics(
            runner=FreqRecRunner(freqrec_config(tmp_path)),
            epoch_metrics_path=tmp_path / "missing.jsonl",
            per_epoch_prediction_dir=tmp_path,
            validation_labels=[1],
            item_count=10,
            requested_topk=10,
            configured_epochs=2,
            seed=7,
            metric_cutoffs=[20],
        )
