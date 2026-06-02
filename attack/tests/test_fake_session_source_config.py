from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack.common.config import (
    FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED,
    FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED,
    load_config,
)


CONFIG_PATH = REPO_ROOT / "attack" / "configs" / "diginetica_attack_dpsbr.yaml"


def _write_config(tmp_path: Path, attack_source_block: str | None) -> Path:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("  root: outputs", f"  root: {tmp_path.as_posix()}")
    if attack_source_block:
        marker = "  poison_model:\n"
        text = text.replace(marker, attack_source_block + marker, 1)
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_fake_session_source_defaults_to_poison_model_generated(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, None))

    assert config.attack.fake_session_source.type == FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED


def test_explicit_poison_model_generated_fake_session_source_parses(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            "  fake_session_source:\n"
            "    type: poison_model_generated\n",
        )
    )

    assert config.attack.fake_session_source.type == FAKE_SESSION_SOURCE_POISON_MODEL_GENERATED


def test_train_template_fake_session_source_nested_config_parses(tmp_path: Path) -> None:
    config = load_config(
        _write_config(
            tmp_path,
            "  fake_session_source:\n"
            "    type: train_template_clean_exact_length_matched\n"
            "    train_template:\n"
            "      reference_split: train_sub\n"
            "      target_filtering: none\n"
            "      replacement: false\n"
            "      fallback:\n"
            "        nearest_length_redistribution: true\n"
            "        replacement_if_needed: true\n"
            "      record_distribution_diagnostics: true\n",
        )
    )

    source = config.attack.fake_session_source
    assert source.type == FAKE_SESSION_SOURCE_TRAIN_TEMPLATE_CLEAN_EXACT_LENGTH_MATCHED
    assert source.train_template.reference_split == "train_sub"
    assert source.train_template.target_filtering == "none"
    assert source.train_template.fallback.nearest_length_redistribution is True


def test_unsupported_fake_session_source_type_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fake_session_source.type"):
        load_config(
            _write_config(
                tmp_path,
                "  fake_session_source:\n"
                "    type: uniform_baseline\n",
            )
        )


def test_unsupported_train_template_target_filtering_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_filtering"):
        load_config(
            _write_config(
                tmp_path,
                "  fake_session_source:\n"
                "    type: train_template_clean_exact_length_matched\n"
                "    train_template:\n"
                "      target_filtering: exclude_targets\n",
            )
        )


def test_train_template_replacement_true_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replacement=true is not supported"):
        load_config(
            _write_config(
                tmp_path,
                "  fake_session_source:\n"
                "    type: train_template_clean_exact_length_matched\n"
                "    train_template:\n"
                "      replacement: true\n",
            )
        )
