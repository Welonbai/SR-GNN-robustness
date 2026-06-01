from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.artifact_io import load_json
from attack.common.config import load_config
from attack.data.canonical_dataset import CanonicalDataset
from attack.pipeline.runs import run_train_template_source_diagnostic as diag
from attack.pipeline.runs.run_train_template_source_diagnostic import (
    allocate_exact_length_quotas,
    jensen_shannon_divergence,
    ks_statistic,
    sample_train_templates_clean_exact_length_matched,
    target_pre_existing_stats,
    validate_train_sub_raw_sessions,
)


CONFIG_PATH = (
    REPO_ROOT
    / "attack"
    / "configs"
    / "diginetica_valbest_attack_ptscem_direct_guassian_mlp_internal_unpopular_sample10_fixed_epoch.yaml"
)


def test_allocate_exact_length_quotas_sums_to_n_fake_and_tracks_distribution() -> None:
    quotas = allocate_exact_length_quotas({2: 2, 3: 1, 5: 1}, 10)

    assert sum(quotas.values()) == 10
    assert quotas == {2: 5, 3: 3, 5: 2}


def test_sampler_without_replacement_when_quota_fits_available_counts() -> None:
    sessions = [[1, 2], [3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13, 14]]

    sampled, metadata, rows = sample_train_templates_clean_exact_length_matched(
        sessions,
        n_fake=4,
        seed=7,
    )

    assert len(sampled) == 4
    assert metadata["fallback_nearest_length_count"] == 0
    assert metadata["replacement_sample_count"] == 0
    assert metadata["record_duplicate_count"] == 0
    assert {row["sampling_mode"] for row in rows} == {"exact_without_replacement"}


def test_sampler_produces_requested_count_and_keeps_target_containing_sessions() -> None:
    sessions = [[99, 1], [2, 3], [4, 5]]

    sampled, metadata, _ = sample_train_templates_clean_exact_length_matched(
        sessions,
        n_fake=3,
        seed=1,
    )
    target_rows = target_pre_existing_stats(sampled, [99])

    assert len(sampled) == 3
    assert metadata["sampled_template_count"] == 3
    assert target_rows == [
        {
            "target_item": 99,
            "template_sessions_containing_target_count": 1,
            "template_sessions_containing_target_ratio": pytest.approx(1 / 3),
            "total_target_occurrences_in_templates": 1,
        }
    ]


def test_sampler_records_nearest_length_fallback_and_shortage_length() -> None:
    sessions = [[1, 2], [3, 4, 5]]

    sampled, metadata, rows = sample_train_templates_clean_exact_length_matched(
        sessions,
        n_fake=3,
        seed=3,
    )

    assert len(sampled) == 3
    assert metadata["fallback_nearest_length_count"] == 1
    assert metadata["replacement_sample_count"] == 1
    assert metadata["shortage_by_quota_length"]
    assert 2 in {item["quota_length"] for item in metadata["shortage_by_quota_length"]}
    assert "nearest_length_fallback" in {row["sampling_mode"] for row in rows}


def test_sampler_uses_replacement_when_n_fake_exceeds_pool_size() -> None:
    sessions = [[1, 2], [3, 4, 5]]

    sampled, metadata, rows = sample_train_templates_clean_exact_length_matched(
        sessions,
        n_fake=5,
        seed=9,
    )

    assert len(sampled) == 5
    assert metadata["replacement_sample_count"] == 3
    assert metadata["record_duplicate_count"] == 3
    assert metadata["content_duplicate_count"] == 3
    assert "replacement" in {row["sampling_mode"] for row in rows}


def test_js_and_ks_are_zero_for_identical_distributions() -> None:
    assert jensen_shannon_divergence({2: 2, 3: 1}, {2: 2, 3: 1}) == pytest.approx(0.0)
    assert ks_statistic([2, 2, 3], [2, 2, 3]) == pytest.approx(0.0)


def test_validate_train_sub_rejects_expanded_shapes() -> None:
    assert validate_train_sub_raw_sessions([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]

    with pytest.raises(ValueError, match="expanded"):
        validate_train_sub_raw_sessions(([[1], [2]], [2, 3]))

    with pytest.raises(ValueError, match="expanded"):
        validate_train_sub_raw_sessions([([1, 2], 3)])


def test_cli_smoke_writes_artifacts_without_fake_session_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "diagnostic.yaml"
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    config_text = config_text.replace("  root: outputs", f"  root: {tmp_path.as_posix()}")
    config_text = config_text.replace("  reuse_saved_targets: true", "  reuse_saved_targets: false")
    config_path.write_text(config_text, encoding="utf-8")

    toy_sessions = [[item, item + 100] for item in range(1, 31)]
    toy_dataset = CanonicalDataset(
        train_sub=toy_sessions,
        valid=[[1, 2]],
        test=[[3, 4]],
        item_map={},
        metadata={"fixture": True},
    )
    monkeypatch.setattr(diag, "ensure_canonical_dataset", lambda config: toy_dataset)

    output_dir = tmp_path / "diagnostic_output"
    exit_code = diag.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary_path = output_dir / "train_template_source_summary.json"
    sessions_path = output_dir / "sampled_train_template_sessions.jsonl"
    length_path = output_dir / "length_distribution_comparison.csv"
    target_path = output_dir / "target_pre_existing_stats.csv"
    assert summary_path.exists()
    assert sessions_path.exists()
    assert length_path.exists()
    assert target_path.exists()

    summary = load_json(summary_path)
    assert summary["raw_session_representation"] == "canonical_dataset.train_sub raw sessions"
    assert summary["denominator_representation"] == "expanded prefix-label pairs"
    assert summary["target_registry_mode"] == "initialized_registry"
    assert summary["generated_fake_cache"]["loaded"] is False
    assert summary["config_path"] == str(config_path)
    assert summary["dataset"] == "diginetica"
    assert summary["experiment_name"]

    first_line = sessions_path.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(first_line)
    assert {
        "source_session_index",
        "quota_length",
        "sampled_from_length",
        "sampling_mode",
    }.issubset(row)

    with length_path.open("r", encoding="utf-8", newline="") as handle:
        length_rows = list(csv.DictReader(handle))
    assert length_rows
    assert "generated_fake_count" not in length_rows[0]
