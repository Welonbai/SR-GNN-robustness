from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys

from attack.common.config import Config
from attack.models.victim.base_runner import VictimRunnerBase
from attack.models.victim.registry import register_victim


class MiaSRecRunner(VictimRunnerBase):
    name = "miasrec"

    def __init__(self, config: Config, repo_root: str | Path | None = None) -> None:
        self.config = config
        self.repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    def build_model(self, opt=None):
        return None

    def load_dataset(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec runner does not load datasets directly.")

    def train(self, *args, **kwargs):
        export_root = kwargs.get("export_root")
        dataset_name = kwargs.get("dataset_name")
        run_dir = kwargs.get("run_dir")
        export_topk_path = kwargs.get("export_topk_path")
        topk = kwargs.get("topk")
        if (
            export_root is None
            or dataset_name is None
            or run_dir is None
            or export_topk_path is None
            or topk is None
        ):
            raise ValueError(
                "train() requires export_root, dataset_name, run_dir, export_topk_path, and topk."
            )
        return self.run(
            export_root=Path(export_root),
            dataset_name=str(dataset_name),
            run_dir=Path(run_dir),
            export_topk_path=Path(export_topk_path),
            topk=int(topk),
        )

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec evaluation is handled inside the subprocess.")

    def score_session(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec does not expose per-session scoring.")

    def predict_topk(self, *, predictions_path: Path, topk: int | None = None) -> list[list[int]]:
        predictions_path = Path(predictions_path)
        if not predictions_path.exists():
            raise FileNotFoundError(f"MiaSRec predictions not found: {predictions_path}")
        with predictions_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rankings = payload.get("rankings")
        if rankings is None:
            raise ValueError("MiaSRec predictions file missing rankings.")
        if topk is not None:
            rankings = [list(map(int, row[:topk])) for row in rankings]
        else:
            rankings = [list(map(int, row)) for row in rankings]
        return rankings

    def load_model(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec model loading is not supported.")

    def save_model(self, *args, **kwargs):
        raise NotImplementedError("MiaSRec model saving is not supported.")

    def run(
        self,
        *,
        export_root: Path,
        dataset_name: str,
        run_dir: Path,
        export_topk_path: Path,
        topk: int,
    ) -> dict[str, str | int]:
        miasrec_root = self.repo_root / "third_party" / "miasrec"
        if not miasrec_root.exists():
            raise FileNotFoundError(f"MiaSRec repository not found: {miasrec_root}")
        main_path = miasrec_root / "main.py"
        if not main_path.exists():
            raise FileNotFoundError(f"MiaSRec entrypoint missing: {main_path}")

        dataset_dir = export_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"MiaSRec dataset directory missing: {dataset_dir}")

        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "miasrec_stdout.log"
        checkpoint_dir = run_dir / "miasrec_checkpoints"
        log_root = miasrec_root / "log"
        tensorboard_root = miasrec_root / "log_tensorboard"
        saved_root = miasrec_root / "saved"
        log_snapshot = _snapshot_files(log_root)
        tb_snapshot = _snapshot_files(tensorboard_root)
        saved_snapshot = _snapshot_files(saved_root)

        override_path = run_dir / "miasrec_override.yaml"
        override_path.parent.mkdir(parents=True, exist_ok=True)
        with override_path.open("w", encoding="utf-8") as handle:
            handle.write(f"topk: [{int(topk)}]\n")
            handle.write(f"export_topk_path: {json.dumps(str(export_topk_path.resolve()))}\n")

        cmd = [
            sys.executable,
            "main.py",
            "--model",
            "miasrec",
            "--dataset",
            dataset_name,
            "--config2",
            str(override_path.resolve()),
            "--data_path",
            str(export_root.resolve()),
            "--checkpoint_dir",
            str(checkpoint_dir.resolve()),
        ]

        print(f"[miasrec] Starting subprocess. Log: {log_path}")
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=miasrec_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        _move_new_files(log_root, log_snapshot, run_dir / "miasrec_logs")
        _move_new_files(tensorboard_root, tb_snapshot, run_dir / "miasrec_tensorboard")
        _move_new_files(saved_root, saved_snapshot, run_dir / "miasrec_saved")

        if result.returncode != 0:
            raise RuntimeError(
                f"MiaSRec subprocess failed with code {result.returncode}. "
                f"See log: {log_path}"
            )
        if not export_topk_path.exists():
            raise RuntimeError(
                "MiaSRec did not export top-k predictions. "
                f"Missing: {export_topk_path}"
            )

        print(f"[miasrec] Completed. Predictions: {export_topk_path}")
        return {
            "returncode": int(result.returncode),
            "log_path": str(log_path),
            "export_topk_path": str(export_topk_path),
        }


def _snapshot_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


def _move_new_files(root: Path, before: set[Path], dest_root: Path) -> None:
    if not root.exists():
        return
    after = {path for path in root.rglob("*") if path.is_file()}
    new_files = [path for path in after if path not in before]
    if not new_files:
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    for path in new_files:
        relative = path.relative_to(root)
        target = dest_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    _prune_empty_dirs(root)


def _prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                continue


register_victim(MiaSRecRunner.name, MiaSRecRunner)


__all__ = ["MiaSRecRunner"]
