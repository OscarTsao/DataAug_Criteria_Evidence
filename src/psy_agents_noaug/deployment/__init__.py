"""Production deployment and model registry (Phase 14).

This module provides tools for deploying, versioning, and managing
optimized models in production environments.

Key Features:
- MLflow Model Registry integration
- Model versioning and tagging
- Model packaging and export
- Deployment configuration management
- Model serving utilities
- Production monitoring setup

Example Usage:

    # Register model to MLflow
    from psy_agents_noaug.deployment import ModelRegistry

    registry = ModelRegistry(tracking_uri="sqlite:///mlflow.db")

    # Register best model from HPO
    model_version = registry.register_model_from_checkpoint(
        model_name="criteria_classifier",
        checkpoint_path="outputs/checkpoints/best_model.pt",
        tags={"architecture": "roberta", "task": "criteria"},
    )

    # Transition to production
    registry.transition_model_stage(
        model_name="criteria_classifier",
        version=model_version,
        stage="Production",
    )

    # Load production model
    from psy_agents_noaug.deployment import ModelLoader

    loader = ModelLoader(registry=registry)
    model = loader.load_production_model("criteria_classifier")

    # Create deployment package
    from psy_agents_noaug.deployment import DeploymentPackager

    packager = DeploymentPackager()
    package_path = packager.create_deployment_package(
        model_name="criteria_classifier",
        version=model_version,
        output_dir="outputs/deployment/",
        include_dependencies=True,
    )

    # Deploy model
    from psy_agents_noaug.deployment import ModelDeployer

    deployer = ModelDeployer()
    deployment = deployer.deploy_model(
        package_path=package_path,
        deployment_config={
            "name": "criteria-classifier-v1",
            "replicas": 3,
            "resources": {"cpu": "1000m", "memory": "2Gi"},
        },
    )
"""

# Registry
# Deployment
from psy_agents_noaug.deployment.deployer import (
    DeploymentConfig,
    ModelDeployer,
    deploy_model,
)

# Loading
from psy_agents_noaug.deployment.loader import (
    ModelLoader,
    load_model_from_registry,
)

# Packaging
from psy_agents_noaug.deployment.packager import (
    DeploymentPackager,
    PackageConfig,
    create_deployment_package,
)
from psy_agents_noaug.deployment.registry import (
    ModelMetadata,
    ModelRegistry,
    get_production_models,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelMetadata",
    "get_production_models",
    # Loading
    "ModelLoader",
    "load_model_from_registry",
    # Packaging
    "DeploymentPackager",
    "PackageConfig",
    "create_deployment_package",
    # Deployment
    "ModelDeployer",
    "DeploymentConfig",
    "deploy_model",
]
