from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from Project.utils import (
    BestModelSaver,
    ensure_artifact_dir,
    get_artifact_path,
    load_best_model_state,
    load_training_state,
    save_best_model_state,
    save_training_state,
    training_state_exists,
)

PROJECT_NAME = "criteria"
DEFAULT_FILENAME = "best_model.pt"


def get_artifact_dir() -> str:
    """Ensure and return the project-specific artifact directory."""
    return str(ensure_artifact_dir(PROJECT_NAME))


def get_best_model_path(filename: str = DEFAULT_FILENAME) -> str:
    """Return the full path to the saved best model file."""
    return str(get_artifact_path(PROJECT_NAME, filename))


def save_best_model(
    model: nn.Module,
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metric_name: str = "validation_f1",
    metric_value: Optional[float] = None,
    extra_state: Optional[dict[str, Any]] = None,
    filename: str = DEFAULT_FILENAME,
) -> str:
    """Persist the model checkpoint under artifacts/criteria."""
    path = save_best_model_state(
        PROJECT_NAME,
        model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metric_name=metric_name,
        metric_value=metric_value,
        extra_state=extra_state,
        filename=filename,
    )
    return str(path)


def load_best_model(
    model: nn.Module,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    filename: str = DEFAULT_FILENAME,
) -> dict[str, Any]:
    """Load the best checkpoint weights into the provided modules."""
    return load_best_model_state(
        PROJECT_NAME,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        filename=filename,
    )


def best_model_saver(
    *,
    filename: str = DEFAULT_FILENAME,
    metric_name: str = "validation_f1",
    mode: str = "max",
) -> BestModelSaver:
    """Helper to construct a BestModelSaver pre-configured for this project."""
    return BestModelSaver(
        project=PROJECT_NAME,
        filename=filename,
        metric_name=metric_name,
        mode=mode,
    )


__all__ = [
    "PROJECT_NAME",
    "DEFAULT_FILENAME",
    "get_artifact_dir",
    "get_best_model_path",
    "save_best_model",
    "load_best_model",
    "best_model_saver",
    "save_training_state",
    "load_training_state",
    "training_state_exists",
]
