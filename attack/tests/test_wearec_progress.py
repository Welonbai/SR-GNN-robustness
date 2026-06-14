from __future__ import annotations

from attack.models.victim.subprocess_progress import _print_epoch_progress


def test_wearec_one_based_progress_is_not_shifted(capsys):
    seen: set[int] = set()
    _print_epoch_progress(
        "Epoch 1 train_loss=1.0",
        seen_epochs=seen,
        total_epochs=2,
        epoch_numbers_are_one_based=True,
    )
    assert capsys.readouterr().out.strip() == "1/2"


def test_existing_zero_based_progress_behavior_is_unchanged(capsys):
    seen: set[int] = set()
    _print_epoch_progress(
        "Epoch 1 train_loss=1.0",
        seen_epochs=seen,
        total_epochs=3,
    )
    assert capsys.readouterr().out.strip() == "2/3"
