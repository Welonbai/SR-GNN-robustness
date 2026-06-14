from __future__ import annotations

import inspect
import json

import pytest

import attack.models.victim.wearec_runner as wearec_runner_module
from attack.models.victim.wearec_runner import (
    effective_wearec_config,
    load_wearec_prediction_payload,
)
from attack.tests.wearec_test_utils import (
    raw_prediction_payload,
    wearec_config,
)


def test_effective_projection_excludes_unused_compatibility_fields(tmp_path):
    base = wearec_config(tmp_path)
    changed = wearec_config(
        tmp_path,
        train_overrides={
            "num_attention_heads": 99,
            "attention_probs_dropout_prob": 0.9,
        },
    )
    assert effective_wearec_config(base, seed=7, requested_topk=5) == (
        effective_wearec_config(changed, seed=7, requested_topk=5)
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 8), ("lr", 0.002), ("hidden_size", 16),
        ("num_hidden_layers", 2), ("hidden_act", "relu"),
        ("hidden_dropout_prob", 0.2), ("initializer_range", 0.03),
        ("num_heads", 1), ("alpha", 0.4), ("weight_decay", 0.1),
        ("adam_beta1", 0.8), ("adam_beta2", 0.99),
    ],
)
def test_every_effective_train_field_changes_projection(tmp_path, field, value):
    base = effective_wearec_config(wearec_config(tmp_path), seed=7, requested_topk=5)
    changed = effective_wearec_config(
        wearec_config(tmp_path, train_overrides={field: value}),
        seed=7,
        requested_topk=5,
    )
    assert changed != base


def test_requested_topk_must_cover_cutoffs(tmp_path):
    with pytest.raises(ValueError, match="cover every metric cutoff"):
        effective_wearec_config(wearec_config(tmp_path), seed=7, requested_topk=3)


@pytest.mark.parametrize("requested_topk", [5, 6])
def test_requested_topk_equal_to_or_above_max_cutoff_is_accepted(
    tmp_path, requested_topk
):
    projection = effective_wearec_config(
        wearec_config(tmp_path), seed=7, requested_topk=requested_topk
    )
    assert projection["requested_topk"] == requested_topk


def test_native_raw_artifact_accepts_no_parent_provenance(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw_prediction_payload()), encoding="utf-8")
    effective = effective_wearec_config(
        wearec_config(tmp_path), seed=7, requested_topk=5
    )
    payload = load_wearec_prediction_payload(
        path, item_count=5, expected_labels=[1, 2, 3], effective_config=effective
    )
    assert payload["final_epoch"] == 2
    assert "parent_repository_commit" not in payload


def test_native_raw_artifact_rejects_batch_mismatch(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw_prediction_payload(batch_size=8)), encoding="utf-8")
    with pytest.raises(ValueError, match="batch_size"):
        load_wearec_prediction_payload(
            path,
            item_count=5,
            expected_labels=[1, 2, 3],
            effective_config=effective_wearec_config(
                wearec_config(tmp_path), seed=7, requested_topk=5
            ),
        )


@pytest.mark.parametrize(
    "labels",
    [
        [4, 2, 3],
        [2, 1, 3],
    ],
)
def test_native_raw_artifact_rejects_wrong_or_permuted_labels(tmp_path, labels):
    path = tmp_path / "raw.json"
    payload = raw_prediction_payload()
    for row, label in zip(payload["rankings"], labels):
        row["label"] = label
        row["items"] = [label] + [item for item in range(1, 6) if item != label]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authoritative exported test label"):
        load_wearec_prediction_payload(
            path,
            item_count=5,
            expected_labels=[1, 2, 3],
            effective_config=effective_wearec_config(
                wearec_config(tmp_path), seed=7, requested_topk=5
            ),
        )


def test_native_raw_artifact_rejects_mismatched_recomputed_metric(tmp_path):
    path = tmp_path / "raw.json"
    payload = raw_prediction_payload()
    payload["test_metrics"]["ndcg@3"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="ndcg@3"):
        load_wearec_prediction_payload(
            path,
            item_count=5,
            expected_labels=[1, 2, 3],
            effective_config=effective_wearec_config(
                wearec_config(tmp_path), seed=7, requested_topk=5
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "epochs_requested",
        "item_count",
        "example_count",
        "batch_size",
        "requested_topk",
        "seed",
    ],
)
def test_native_raw_artifact_rejects_boolean_integer_fields(tmp_path, field):
    path = tmp_path / "raw.json"
    payload = raw_prediction_payload()
    payload[field] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact integer"):
        load_wearec_prediction_payload(
            path,
            item_count=5,
            expected_labels=[1, 2, 3],
            effective_config=effective_wearec_config(
                wearec_config(tmp_path), seed=7, requested_topk=5
            ),
        )


@pytest.mark.parametrize(
    "field",
    ["best_epoch", "best_metric", "best_checkpoint", "early_stopping"],
)
def test_native_raw_artifact_rejects_validation_best_fields(tmp_path, field):
    path = tmp_path / "raw.json"
    payload = raw_prediction_payload()
    payload[field] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden validation-best"):
        load_wearec_prediction_payload(
            path,
            item_count=5,
            expected_labels=[1, 2, 3],
            effective_config=effective_wearec_config(
                wearec_config(tmp_path), seed=7, requested_topk=5
            ),
        )


def test_parent_wearec_validation_does_not_load_checkpoints():
    assert "torch.load" not in inspect.getsource(wearec_runner_module)
