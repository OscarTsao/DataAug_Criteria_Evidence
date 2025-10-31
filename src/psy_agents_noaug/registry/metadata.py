#!/usr/bin/env python
"""Model metadata tracking (Phase 20).

This module provides:
- Comprehensive model metadata
- Training configuration tracking
- Performance metrics storage
- Hardware/environment information
- Dependency tracking
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata."""

    # Model identity
    model_name: str
    model_type: str
    version: str
    architecture: str

    # Training information
    trained_at: datetime
    training_duration_seconds: float
    total_parameters: int
    trainable_parameters: int

    # Performance metrics
    metrics: dict[str, float] = field(default_factory=dict)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Training configuration
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    optimizer: str = ""
    loss_function: str = ""
    batch_size: int = 0
    num_epochs: int = 0
    learning_rate: float = 0.0

    # Data information
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0
    data_version: str = ""
    preprocessing: dict[str, Any] = field(default_factory=dict)

    # Environment
    python_version: str = field(default_factory=lambda: platform.python_version())
    platform_info: str = field(default_factory=lambda: platform.platform())
    gpu_info: str = ""
    dependencies: dict[str, str] = field(default_factory=dict)

    # Lineage
    parent_model: str | None = None
    experiment_id: str = ""
    run_id: str = ""

    # Additional metadata
    tags: list[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            # Model identity
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "architecture": self.architecture,
            # Training information
            "trained_at": self.trained_at.isoformat(),
            "training_duration_seconds": self.training_duration_seconds,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            # Performance metrics
            "metrics": self.metrics,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            # Training configuration
            "hyperparameters": self.hyperparameters,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            # Data information
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "test_samples": self.test_samples,
            "data_version": self.data_version,
            "preprocessing": self.preprocessing,
            # Environment
            "python_version": self.python_version,
            "platform_info": self.platform_info,
            "gpu_info": self.gpu_info,
            "dependencies": self.dependencies,
            # Lineage
            "parent_model": self.parent_model,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            # Additional metadata
            "tags": self.tags,
            "description": self.description,
            "author": self.author,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            ModelMetadata instance
        """
        return cls(
            # Model identity
            model_name=data["model_name"],
            model_type=data["model_type"],
            version=data["version"],
            architecture=data["architecture"],
            # Training information
            trained_at=datetime.fromisoformat(data["trained_at"]),
            training_duration_seconds=data["training_duration_seconds"],
            total_parameters=data["total_parameters"],
            trainable_parameters=data["trainable_parameters"],
            # Performance metrics
            metrics=data.get("metrics", {}),
            validation_metrics=data.get("validation_metrics", {}),
            test_metrics=data.get("test_metrics", {}),
            # Training configuration
            hyperparameters=data.get("hyperparameters", {}),
            optimizer=data.get("optimizer", ""),
            loss_function=data.get("loss_function", ""),
            batch_size=data.get("batch_size", 0),
            num_epochs=data.get("num_epochs", 0),
            learning_rate=data.get("learning_rate", 0.0),
            # Data information
            train_samples=data.get("train_samples", 0),
            val_samples=data.get("val_samples", 0),
            test_samples=data.get("test_samples", 0),
            data_version=data.get("data_version", ""),
            preprocessing=data.get("preprocessing", {}),
            # Environment
            python_version=data.get("python_version", ""),
            platform_info=data.get("platform_info", ""),
            gpu_info=data.get("gpu_info", ""),
            dependencies=data.get("dependencies", {}),
            # Lineage
            parent_model=data.get("parent_model"),
            experiment_id=data.get("experiment_id", ""),
            run_id=data.get("run_id", ""),
            # Additional metadata
            tags=data.get("tags", []),
            description=data.get("description", ""),
            author=data.get("author", ""),
            notes=data.get("notes", ""),
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary of key metadata.

        Returns:
            Summary dictionary
        """
        return {
            "model": f"{self.model_name} v{self.version}",
            "architecture": self.architecture,
            "parameters": f"{self.total_parameters:,}",
            "trained": self.trained_at.strftime("%Y-%m-%d %H:%M"),
            "duration": f"{self.training_duration_seconds:.1f}s",
            "best_metric": (
                max(self.validation_metrics.items(), key=lambda x: x[1])
                if self.validation_metrics
                else None
            ),
            "environment": f"Python {self.python_version}",
        }


class MetadataManager:
    """Manager for model metadata."""

    def __init__(self):
        """Initialize metadata manager."""
        self.metadata_store: dict[str, ModelMetadata] = {}
        LOGGER.info("Initialized MetadataManager")

    def register_metadata(
        self,
        metadata: ModelMetadata,
    ) -> None:
        """Register model metadata.

        Args:
            metadata: Model metadata to register
        """
        key = f"{metadata.model_name}:{metadata.version}"
        self.metadata_store[key] = metadata
        LOGGER.info(f"Registered metadata for {key}")

    def get_metadata(
        self,
        model_name: str,
        version: str,
    ) -> ModelMetadata | None:
        """Get metadata for a model version.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            ModelMetadata if found
        """
        key = f"{model_name}:{version}"
        return self.metadata_store.get(key)

    def list_metadata(
        self,
        model_name: str | None = None,
    ) -> list[ModelMetadata]:
        """List all metadata.

        Args:
            model_name: Filter by model name (optional)

        Returns:
            List of ModelMetadata
        """
        metadata_list = list(self.metadata_store.values())

        if model_name:
            metadata_list = [m for m in metadata_list if m.model_name == model_name]

        return sorted(
            metadata_list,
            key=lambda m: m.trained_at,
            reverse=True,
        )

    def compare_metadata(
        self,
        model1_key: str,
        model2_key: str,
    ) -> dict[str, Any]:
        """Compare metadata between two models.

        Args:
            model1_key: First model key (name:version)
            model2_key: Second model key (name:version)

        Returns:
            Comparison dictionary
        """
        meta1 = self.metadata_store.get(model1_key)
        meta2 = self.metadata_store.get(model2_key)

        if not meta1 or not meta2:
            return {"error": "One or both models not found"}

        comparison = {
            "models": {
                "model1": model1_key,
                "model2": model2_key,
            },
            "parameters": {
                "model1": meta1.total_parameters,
                "model2": meta2.total_parameters,
                "difference": meta2.total_parameters - meta1.total_parameters,
            },
            "training_time": {
                "model1": meta1.training_duration_seconds,
                "model2": meta2.training_duration_seconds,
                "difference": meta2.training_duration_seconds
                - meta1.training_duration_seconds,
            },
        }

        # Compare metrics
        common_metrics = set(meta1.validation_metrics.keys()) & set(
            meta2.validation_metrics.keys()
        )
        metrics_comparison: dict[str, Any] = {}

        for metric in common_metrics:
            val1 = meta1.validation_metrics[metric]
            val2 = meta2.validation_metrics[metric]
            metrics_comparison[metric] = {
                "model1": val1,
                "model2": val2,
                "difference": val2 - val1,
                "improvement": ((val2 - val1) / val1 * 100) if val1 != 0 else 0,
            }

        comparison["metrics"] = metrics_comparison

        return comparison

    def get_best_model(
        self,
        model_name: str,
        metric: str,
        maximize: bool = True,
    ) -> ModelMetadata | None:
        """Get best model based on a metric.

        Args:
            model_name: Model name
            metric: Metric to optimize
            maximize: True to maximize, False to minimize

        Returns:
            Best ModelMetadata if found
        """
        metadata_list = self.list_metadata(model_name)

        if not metadata_list:
            return None

        # Filter models with the metric
        valid_models = [m for m in metadata_list if metric in m.validation_metrics]

        if not valid_models:
            return None

        return max(
            valid_models,
            key=lambda m: m.validation_metrics[metric] * (1 if maximize else -1),
        )


# Convenience function
def create_metadata(
    model_name: str,
    model_type: str,
    version: str,
    architecture: str,
    training_duration: float,
    total_params: int,
    trainable_params: int,
) -> ModelMetadata:
    """Create model metadata (convenience function).

    Args:
        model_name: Model name
        model_type: Model type
        version: Version string
        architecture: Architecture name
        training_duration: Training duration in seconds
        total_params: Total parameters
        trainable_params: Trainable parameters

    Returns:
        ModelMetadata instance
    """
    return ModelMetadata(
        model_name=model_name,
        model_type=model_type,
        version=version,
        architecture=architecture,
        trained_at=datetime.now(),
        training_duration_seconds=training_duration,
        total_parameters=total_params,
        trainable_parameters=trainable_params,
    )
