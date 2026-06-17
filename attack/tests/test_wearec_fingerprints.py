from __future__ import annotations

import json

import pytest

from attack.data.canonical_fingerprints import (
    fingerprint_exported_jsonl,
    fingerprint_item_vocabulary,
    normalize_source_item_id,
)


def _write(path, rows, newline="\n"):
    text = newline.join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    ) + newline
    path.write_bytes(text.encode("utf-8"))


def _rows():
    return [
        {"example_id": 0, "input_prefix": [1], "label": 2},
        {"example_id": 1, "input_prefix": [1, 2], "label": 3},
    ]


def test_exported_row_fingerprint_is_path_and_newline_independent(tmp_path):
    left = tmp_path / "left.jsonl"
    right = tmp_path / "nested" / "right.jsonl"
    right.parent.mkdir()
    _write(left, _rows(), "\n")
    _write(right, _rows(), "\r\n")
    assert fingerprint_exported_jsonl(left) == fingerprint_exported_jsonl(right)
    assert fingerprint_exported_jsonl(left) == fingerprint_exported_jsonl(left)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows[0].update(label=3),
        lambda rows: rows[1].update(input_prefix=[2, 1]),
        lambda rows: rows.reverse(),
    ],
)
def test_exported_row_fingerprint_changes_with_scientific_content(tmp_path, mutation):
    base = _rows()
    changed = _rows()
    mutation(changed)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, base)
    _write(second, changed)
    if changed[0]["example_id"] != 0:
        with pytest.raises(ValueError):
            fingerprint_exported_jsonl(second)
    else:
        assert fingerprint_exported_jsonl(first) != fingerprint_exported_jsonl(second)


def test_item_vocabulary_normalization_and_type_distinction():
    np = pytest.importorskip("numpy")
    assert normalize_source_item_id(1) == normalize_source_item_id(np.int64(1))
    assert normalize_source_item_id("1") == normalize_source_item_id(np.str_("1"))
    assert normalize_source_item_id(1) != normalize_source_item_id("1")
    assert fingerprint_item_vocabulary({"a": 1, "b": 2}) != fingerprint_item_vocabulary(
        {"a": 2, "b": 1}
    )


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), [], object()])
def test_item_vocabulary_rejects_unsupported_source_ids(value):
    with pytest.raises(TypeError):
        normalize_source_item_id(value)
