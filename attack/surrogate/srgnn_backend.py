from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from attack.common.config import Config
from attack.data.poisoned_dataset_builder import PoisonedDataset
from attack.models._srgnn_base import SRGNNBaseRunner
from attack.pipeline.core.pipeline_utils import build_srgnn_opt_from_train_config
from attack.position_opt.types import SurrogateScoreResult, TruncatedFineTuneConfig
from attack.surrogate.base import PoisonedTrainInput, SessionBatch
from pytorch_code.model import forward as srg_forward
from pytorch_code.model import trans_to_cpu, trans_to_cuda
from pytorch_code.utils import Data


@dataclass
class SRGNNModelHandle:
    # This handle represents the new MVP surrogate role. It deliberately wraps a
    # fresh SR-GNN instance cloned from a clean surrogate checkpoint, not the
    # poison-model instance used by fake-session generation.
    runner: SRGNNBaseRunner

    @property
    def model(self):
        if self.runner.model is None:
            raise RuntimeError("SR-GNN model is not initialized.")
        return self.runner.model


class SRGNNBackend:
    """Minimal SR-GNN surrogate backend for the position-opt MVP.

    This wrapper owns the new surrogate-model role used for retraining-aware
    scoring and truncated fine-tuning. It does not participate in fake-session
    generation and it is not the final universal backend/training design.
    """

    def __init__(
        self,
        config: Config,
        *,
        base_dir: str | Path | None = None,
        train_config: Mapping[str, Any] | None = None,
        n_node: int | None = None,
    ) -> None:
        self.config = config
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        # Phase 1 has no dedicated surrogate YAML section yet. Reusing the
        # existing SR-GNN train block here is only an architecture/optimizer
        # bootstrap so we can instantiate a load-compatible surrogate model.
        # It does not mean the poison model and clean surrogate are the same role.
        self.train_config = dict(train_config or config.attack.poison_model.params["train"])
        self.n_node = n_node
        self._clean_checkpoint_path: Path | None = None
        self._clean_state_dict: dict[str, Any] | None = None

    def load_clean_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = self._resolve_path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Clean surrogate checkpoint not found: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError("SR-GNN checkpoint must contain a state_dict-like mapping.")
        self._clean_checkpoint_path = checkpoint_path
        self._clean_state_dict = state_dict

    def clone_clean_model(self) -> SRGNNModelHandle:
        if self._clean_state_dict is None:
            if self._clean_checkpoint_path is None:
                raise RuntimeError("Call load_clean_checkpoint() before clone_clean_model().")
            self.load_clean_checkpoint(self._clean_checkpoint_path)

        runner = self._new_runner()
        runner.model.load_state_dict(self._clean_state_dict)
        runner.model.train()
        return SRGNNModelHandle(runner=runner)

    def fine_tune(
        self,
        model: object,
        poisoned_train_data: PoisonedTrainInput,
        *,
        fine_tune_config: TruncatedFineTuneConfig | None = None,
        eval_data: Any | None = None,
    ) -> dict[str, Any]:
        # This is the current MVP implementation of surrogate fine-tuning: clone
        # from a clean surrogate checkpoint, then take a small number of SR-GNN
        # optimizer steps on poisoned data. It is not poison-model generation
        # logic and it is not yet the final backend-agnostic training design.
        del eval_data
        handle = self._as_model_handle(model)
        config = fine_tune_config or TruncatedFineTuneConfig()
        sessions, labels = _coerce_poisoned_train_data(poisoned_train_data)
        if not sessions:
            return {
                "steps": 0,
                "epochs": 0,
                "step_loss": [],
                "train_loss": [],
                "avg_loss": None,
            }

        train_data = Data((sessions, labels), shuffle=True)
        torch_model = handle.model
        step_limit = int(config.steps)
        epoch_limit = int(config.epochs)
        if step_limit == 0 or epoch_limit == 0:
            return {
                "steps": 0,
                "epochs": 0,
                "step_loss": [],
                "train_loss": [],
                "avg_loss": None,
            }

        step_losses: list[float] = []
        epoch_losses: list[float] = []
        completed_steps = 0
        completed_epochs = 0

        for _ in range(epoch_limit):
            torch_model.train()
            epoch_loss_total = 0.0
            epoch_steps = 0
            for batch_indices in train_data.generate_batch(torch_model.batch_size):
                torch_model.optimizer.zero_grad()
                targets, scores = srg_forward(torch_model, batch_indices, train_data)
                targets_tensor = trans_to_cuda(torch.as_tensor(targets, dtype=torch.long))
                loss = torch_model.loss_function(scores, targets_tensor - 1)
                loss.backward()
                torch_model.optimizer.step()

                loss_value = float(loss.item())
                step_losses.append(loss_value)
                epoch_loss_total += loss_value
                epoch_steps += 1
                completed_steps += 1
                if completed_steps >= step_limit:
                    break

            if epoch_steps > 0:
                torch_model.scheduler.step()
                completed_epochs += 1
                epoch_losses.append(epoch_loss_total / epoch_steps)
            if completed_steps >= step_limit:
                break

        handle.runner.train_loss_history = [float(loss) for loss in epoch_losses]
        avg_loss = float(sum(step_losses) / len(step_losses)) if step_losses else None
        return {
            "steps": int(completed_steps),
            "epochs": int(completed_epochs),
            "step_loss": step_losses,
            "train_loss": epoch_losses,
            "avg_loss": avg_loss,
        }

    def score_target(
        self,
        model: object,
        eval_sessions: SessionBatch,
        target_item: int,
    ) -> SurrogateScoreResult:
        normalized_sessions = _normalize_sessions(eval_sessions)
        target_items = [int(target_item)] * len(normalized_sessions)
        return self._score_item_probabilities(model, normalized_sessions, target_items)

    def score_gt(
        self,
        model: object,
        eval_sessions: SessionBatch,
        ground_truth_items: Sequence[int],
    ) -> SurrogateScoreResult:
        normalized_sessions = _normalize_sessions(eval_sessions)
        if len(normalized_sessions) != len(ground_truth_items):
            raise ValueError("ground_truth_items must align 1:1 with eval_sessions.")
        return self._score_item_probabilities(
            model,
            normalized_sessions,
            [int(item) for item in ground_truth_items],
        )

    def _score_item_probabilities(
        self,
        model: object,
        sessions: list[list[int]],
        item_ids: Sequence[int],
    ) -> SurrogateScoreResult:
        handle = self._as_model_handle(model)
        data = Data((sessions, [1] * len(sessions)), shuffle=False)
        torch_model = handle.model
        values: list[float] = []
        cursor = 0

        torch_model.eval()
        with torch.no_grad():
            for batch_indices in data.generate_batch(torch_model.batch_size):
                _, scores = srg_forward(torch_model, batch_indices, data)
                probabilities = torch.softmax(scores, dim=1)
                batch_size = probabilities.shape[0]
                batch_items = item_ids[cursor : cursor + batch_size]
                item_tensor = torch.as_tensor(
                    [int(item) - 1 for item in batch_items],
                    dtype=torch.long,
                    device=probabilities.device,
                )
                if torch.any(item_tensor < 0) or torch.any(item_tensor >= probabilities.shape[1]):
                    raise ValueError("One or more item ids are outside the SR-GNN score range.")
                batch_scores = probabilities.gather(1, item_tensor.unsqueeze(1)).squeeze(1)
                values.extend(float(value) for value in trans_to_cpu(batch_scores).tolist())
                cursor += batch_size

        return SurrogateScoreResult.from_values(values)

    def _resolve_path(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else (self.base_dir / path)

    def _new_runner(self) -> SRGNNBaseRunner:
        runner = SRGNNBaseRunner(self.config, base_dir=self.base_dir, n_node=self.n_node)
        runner.build_model(build_srgnn_opt_from_train_config(self.train_config))
        return runner

    @staticmethod
    def _as_model_handle(model: object) -> SRGNNModelHandle:
        if not isinstance(model, SRGNNModelHandle):
            raise TypeError("SRGNNBackend expects an SRGNNModelHandle.")
        return model


def _coerce_poisoned_train_data(
    poisoned_train_data: PoisonedTrainInput,
) -> tuple[list[list[int]], list[int]]:
    if isinstance(poisoned_train_data, PoisonedDataset):
        sessions = poisoned_train_data.sessions
        labels = poisoned_train_data.labels
    else:
        sessions, labels = poisoned_train_data

    normalized_sessions = _normalize_sessions(sessions)
    normalized_labels = [int(label) for label in labels]
    if len(normalized_sessions) != len(normalized_labels):
        raise ValueError("poisoned_train_data sessions and labels must have the same length.")
    return normalized_sessions, normalized_labels


def _normalize_sessions(sessions: SessionBatch) -> list[list[int]]:
    normalized = [list(session) for session in sessions]
    if not normalized:
        raise ValueError("At least one session is required.")
    if any(len(session) == 0 for session in normalized):
        raise ValueError("Sessions must be non-empty.")
    return normalized


__all__ = ["SRGNNBackend", "SRGNNModelHandle"]
