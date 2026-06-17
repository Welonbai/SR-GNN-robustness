from __future__ import annotations

from dataclasses import replace

from attack.common.paths import victim_prediction_key_payload
from attack.data.canonical_dataset import CanonicalDataset
from attack.pipeline.core.pipeline_utils import build_clean_pairs
from attack.pipeline.core.victim_execution import prepare_wearec_execution
from attack.tests.wearec_test_utils import wearec_config


PROVENANCE = {
    "parent_repository_commit": "parent-a",
    "parent_tracked_worktree_clean": True,
    "wearec_gitlink_commit": "wearec-a",
    "wearec_submodule_commit": "wearec-a",
    "wearec_tracked_worktree_clean": True,
}


def _dataset(*, valid=None, test=None, item_map=None):
    return CanonicalDataset(
        train_sub=[[1, 2, 3]],
        valid=valid or [[1, 2, 3]],
        test=test or [[2, 3, 4]],
        item_map=item_map or {str(value): value for value in range(1, 6)},
        metadata={
            "item_count": 5,
            "counts": {"items": 5},
            "variant": "full",
        },
    )


def _prepare(tmp_path, dataset=None, prefixes=None, labels=None, config=None, provenance=None):
    dataset = dataset or _dataset()
    clean_prefixes, clean_labels = build_clean_pairs(dataset)
    return prepare_wearec_execution(
        config or wearec_config(tmp_path),
        run_type="clean",
        canonical_dataset=dataset,
        train_prefixes=prefixes or clean_prefixes,
        train_labels=labels or clean_labels,
        run_dir=tmp_path / "run",
        requested_topk=5,
        target_item=1,
        attack_identity_context=None,
        provenance_resolver=lambda *_: dict(provenance or PROVENANCE),
    )


def test_identity_is_content_and_implementation_addressed(tmp_path):
    first = _prepare(tmp_path / "a")
    second = _prepare(tmp_path / "b")
    assert first["identity"] == second["identity"]

    prefixes, labels = build_clean_pairs(_dataset())
    changed_labels = list(labels)
    changed_labels[0] = 4
    changed = _prepare(
        tmp_path / "c", prefixes=prefixes, labels=changed_labels
    )
    assert changed["identity"] != first["identity"]

    changed_valid = _prepare(
        tmp_path / "d", dataset=_dataset(valid=[[1, 3, 2]])
    )
    changed_test = _prepare(
        tmp_path / "e", dataset=_dataset(test=[[2, 4, 3]])
    )
    assert changed_valid["identity"] != first["identity"]
    assert changed_test["identity"] != first["identity"]


def test_batch_size_and_repository_commits_change_identity(tmp_path):
    base = _prepare(tmp_path / "base")
    batch = _prepare(
        tmp_path / "batch",
        config=wearec_config(tmp_path / "batch", train_overrides={"batch_size": 8}),
    )
    parent = _prepare(
        tmp_path / "parent",
        provenance={**PROVENANCE, "parent_repository_commit": "parent-b"},
    )
    submodule = _prepare(
        tmp_path / "submodule",
        provenance={
            **PROVENANCE,
            "wearec_gitlink_commit": "wearec-b",
            "wearec_submodule_commit": "wearec-b",
        },
    )
    assert batch["identity"] != base["identity"]
    assert parent["identity"] != base["identity"]
    assert submodule["identity"] != base["identity"]


def test_victim_key_uses_complete_scientific_identity(tmp_path):
    config = wearec_config(tmp_path)
    prepared = _prepare(tmp_path, config=config)
    payload = victim_prediction_key_payload(
        config,
        "wearec",
        run_type="clean",
        victim_attack_identity_context=prepared["identity"],
        victim_effective_train_seed=prepared["identity"]["effective_config"]["seed"],
    )
    assert payload["wearec_scientific_identity"] == prepared["identity"]
    assert payload["wearec_scientific_identity"]["training_mode"] == "clean"
    assert "attack_identity" not in payload["wearec_scientific_identity"]


def test_clean_identity_is_target_independent_and_poisoned_identity_is_not(tmp_path):
    dataset = _dataset()
    prefixes, labels = build_clean_pairs(dataset)
    config = wearec_config(tmp_path)

    def prepare(run_type, target_item, train_labels, attack_context):
        return prepare_wearec_execution(
            config,
            run_type=run_type,
            canonical_dataset=dataset,
            train_prefixes=prefixes,
            train_labels=train_labels,
            run_dir=tmp_path / f"{run_type}-{target_item}",
            requested_topk=5,
            target_item=target_item,
            attack_identity_context=attack_context,
            provenance_resolver=lambda *_: dict(PROVENANCE),
        )["identity"]

    clean_a = prepare("clean", 1, labels, None)
    clean_b = prepare("clean", 5, labels, None)
    poisoned_labels = list(labels)
    poisoned_labels[0] = 4
    poisoned = prepare(
        "dpsbr_baseline",
        5,
        poisoned_labels,
        {"run_type": "dpsbr_baseline", "attack_config": "test"},
    )
    assert clean_a == clean_b
    assert poisoned["training_mode"] == "poisoned"
    assert poisoned["target_item"] == 5
    assert poisoned["ordered_exported_train_jsonl_sha256"] != clean_a[
        "ordered_exported_train_jsonl_sha256"
    ]
    assert "attack_identity" in poisoned
