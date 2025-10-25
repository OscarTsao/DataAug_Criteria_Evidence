from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

_ARTIFACT_ROOT = Path(__file__).resolve().parents[3] / "artifacts"
_DEFAULT_BEST_FILENAME = "best_model.pt"
_DEFAULT_LAST_FILENAME = "last_checkpoint.pt"


def ensure_artifact_dir(project: str) -> Path:
    """Ensure the artifact directory exists for a given project and return it."""
    path = _ARTIFACT_ROOT / project
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_artifact_path(project: str, filename: str) -> Path:
    """Construct an artifact path under the project's directory."""
    return ensure_artifact_dir(project) / filename


def save_best_model_state(
    project: str,
    model: nn.Module,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int | None = None,
    metric_name: str = "metric",
    metric_value: float | None = None,
    extra_state: dict[str, Any] | None = None,
    filename: str = _DEFAULT_BEST_FILENAME,
) -> Path:
    """Persist the best-performing model checkpoint and optional metadata."""
    checkpoint_path = get_artifact_path(project, filename)
    state: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        with contextlib.suppress(Exception):
            state["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state:
        state["extra_state"] = extra_state

    torch.save(state, checkpoint_path)

    if tokenizer is not None:
        tokenizer_dir = ensure_artifact_dir(project) / "tokenizer"
        tokenizer.save_pretrained(tokenizer_dir)

    return checkpoint_path


def save_training_state(
    project: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int,
    global_step: int,
    best_metric: float | None = None,
    best_metric_name: str | None = None,
    rng_state: dict[str, Any] | None = None,
    filename: str = _DEFAULT_LAST_FILENAME,
) -> Path:
    """Save the latest training state for resuming interrupted runs."""
    state: dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "best_metric": best_metric,
        "best_metric_name": best_metric_name,
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        with contextlib.suppress(Exception):
            state["scheduler_state_dict"] = scheduler.state_dict()
    if rng_state:
        state["rng_state"] = rng_state
    checkpoint_path = get_artifact_path(project, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def load_training_state(
    project: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    filename: str = _DEFAULT_LAST_FILENAME,
) -> dict[str, Any]:
    """Load a previously saved training state into the provided modules.

    Security Note:
        Uses weights_only=True for safe deserialization. Only load checkpoints
        from trusted sources to prevent arbitrary code execution (CVE-2022-45907).
    """
    checkpoint_path = get_artifact_path(project, filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        with contextlib.suppress(Exception):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def training_state_exists(project: str, filename: str = _DEFAULT_LAST_FILENAME) -> bool:
    """Return True if a saved training state exists for the project."""
    return get_artifact_path(project, filename).exists()


def load_best_model_state(
    project: str,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    filename: str = _DEFAULT_BEST_FILENAME,
) -> dict[str, Any]:
    """Load the best model checkpoint into the provided model/optimizers.

    Security Note:
        Uses weights_only=True for safe deserialization. Only load checkpoints
        from trusted sources to prevent arbitrary code execution (CVE-2022-45907).
    """
    checkpoint_path = get_artifact_path(project, filename)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        with contextlib.suppress(Exception):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


@dataclass
class BestModelSaver:
    """Track and persist the best model weights for a project."""

    project: str
    filename: str = _DEFAULT_BEST_FILENAME
    metric_name: str = "metric"
    mode: str = "max"
    best_metric: float | None = field(default=None, init=False)
    metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def _is_improvement(self, value: float) -> bool:
        if self.best_metric is None:
            return True
        if self.mode == "max":
            return value > self.best_metric
        if self.mode == "min":
            return value < self.best_metric
        raise ValueError("mode must be either 'max' or 'min'")

    def update(
        self,
        *,
        metric_value: float,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        epoch: int | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> bool:
        """Save the checkpoint if the provided metric improves the current best."""
        if not self._is_improvement(metric_value):
            return False

        save_best_model_state(
            self.project,
            model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric_name=self.metric_name,
            metric_value=metric_value,
            extra_state=extra_state,
            filename=self.filename,
        )
        self.best_metric = metric_value
        self.metadata = {
            "epoch": epoch,
            "metric_name": self.metric_name,
            "metric_value": metric_value,
        }
        if extra_state:
            self.metadata["extra_state"] = extra_state
        return True

    def load_existing(self) -> dict[str, Any] | None:
        """Load metadata from disk if a best model checkpoint already exists.

        Security Note:
            Uses weights_only=True for safe deserialization. Only load checkpoints
            from trusted sources to prevent arbitrary code execution (CVE-2022-45907).
        """
        path = get_artifact_path(self.project, self.filename)
        if not path.exists():
            return None
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metric_value = checkpoint.get("metric_value")
        if metric_value is not None:
            self.best_metric = metric_value
        self.metadata = {
            "epoch": checkpoint.get("epoch"),
            "metric_name": checkpoint.get("metric_name", self.metric_name),
            "metric_value": metric_value,
        }
        if "extra_state" in checkpoint:
            self.metadata["extra_state"] = checkpoint["extra_state"]
        return self.metadata


__all__ = [
    "BestModelSaver",
    "ensure_artifact_dir",
    "get_artifact_path",
    "save_best_model_state",
    "load_best_model_state",
    "save_training_state",
    "load_training_state",
    "training_state_exists",
]
