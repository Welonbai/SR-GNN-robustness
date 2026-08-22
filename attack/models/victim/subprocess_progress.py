from __future__ import annotations

import codecs
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


_EPOCH_PATTERN = re.compile(r"\bepoch\s+(\d+)\b", re.IGNORECASE)


def resolve_subprocess_gpu_selector(
    configured_gpu_id: str | int,
    env: Mapping[str, str],
) -> str:
    """Resolve the selector passed to a child that resets CUDA visibility.

    Some third-party victim entry points assign ``CUDA_VISIBLE_DEVICES`` from
    their ``--gpu_id`` argument after the parent runner has already isolated a
    physical GPU.  When the parent exposes exactly one device, pass that same
    physical selector to the child so the child cannot undo the isolation.

    Multi-device visibility keeps the configured value for backward
    compatibility; those configurations may intentionally use a physical GPU
    selector rather than a logical index.
    """
    configured = str(configured_gpu_id).strip()
    inherited = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()
    visible = [value.strip() for value in inherited.split(",") if value.strip()]
    if len(visible) == 1:
        return visible[0]
    return configured


def run_subprocess_with_epoch_progress(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    model_name: str,
    target_item: int | None,
    total_epochs: int,
    epoch_numbers_are_one_based: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a victim subprocess while keeping console output compact.

    Full subprocess output is still captured in ``log_path``. The console only
    receives one training header plus one line per newly observed epoch.
    """
    target_label = "clean" if target_item is None else str(int(target_item))
    print(
        f"[victim] target={target_label} model={model_name} train epochs={int(total_epochs)}",
        flush=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    seen_epochs: set[int] = set()
    scan_tail = ""
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if process.stdout is None:
            raise RuntimeError("subprocess stdout pipe was not created.")
        while True:
            chunk = process.stdout.read1(4096)
            if not chunk:
                break
            log_handle.write(chunk)
            log_handle.flush()
            text = decoder.decode(chunk)
            scan_text = scan_tail + text
            _print_epoch_progress(
                scan_text,
                seen_epochs=seen_epochs,
                total_epochs=int(total_epochs),
                epoch_numbers_are_one_based=epoch_numbers_are_one_based,
            )
            scan_tail = scan_text[-128:]
        final_text = decoder.decode(b"", final=True)
        if final_text:
            scan_text = scan_tail + final_text
            _print_epoch_progress(
                scan_text,
                seen_epochs=seen_epochs,
                total_epochs=int(total_epochs),
                epoch_numbers_are_one_based=epoch_numbers_are_one_based,
            )
        returncode = process.wait()
    return subprocess.CompletedProcess(list(cmd), returncode)


def _print_epoch_progress(
    text: str,
    *,
    seen_epochs: set[int],
    total_epochs: int,
    epoch_numbers_are_one_based: bool = False,
) -> None:
    if total_epochs <= 0:
        return
    for match in _EPOCH_PATTERN.finditer(text):
        public_epoch = int(match.group(1))
        epoch_index = public_epoch - 1 if epoch_numbers_are_one_based else public_epoch
        if epoch_index < 0 or epoch_index >= total_epochs:
            continue
        if epoch_index in seen_epochs:
            continue
        seen_epochs.add(epoch_index)
        print(f"{epoch_index + 1}/{int(total_epochs)}", flush=True)


__all__ = ["resolve_subprocess_gpu_selector", "run_subprocess_with_epoch_progress"]
