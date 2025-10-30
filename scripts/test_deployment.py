#!/usr/bin/env python
"""Test deployment functionality (Phase 14).

Quick test to validate model deployment and registry tools.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch  # noqa: E402
from torch import nn  # noqa: E402


# Simple test model
class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


def test_model_registry() -> None:
    """Test ModelRegistry."""
    print("\n" + "=" * 80)
    print("TEST 1: Model Registry")
    print("=" * 80)

    from psy_agents_noaug.deployment import ModelRegistry

    # Create temporary MLflow tracking
    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
        registry = ModelRegistry(tracking_uri=tracking_uri)

        print(f"\n✓ Created registry (tracking={tracking_uri})")

        # Create and save test model
        model = SimpleModel()
        model_path = Path(tmpdir) / "test_model.pt"
        torch.save(model, model_path)

        # Register model
        version = registry.register_model_from_checkpoint(
            model_name="test_model",
            checkpoint_path=model_path,
            tags={"test": "true"},
            description="Test model",
        )

        print(f"✓ Registered model version: {version}")

        # Get model version
        metadata = registry.get_model_version("test_model", version=version)
        print("✓ Retrieved metadata:")
        print(f"  Name: {metadata.name}")
        print(f"  Version: {metadata.version}")
        print(f"  Stage: {metadata.stage}")
        print(f"  Tags: {metadata.tags}")

        # Transition to staging
        registry.transition_model_stage("test_model", version, "Staging")
        print("✓ Transitioned to Staging")

        # List models
        models = registry.list_models()
        print(f"✓ Listed {len(models)} models: {models}")

    print("\n✅ TEST 1 PASSED: Model registry working correctly")


def test_deployment_packager() -> None:
    """Test DeploymentPackager."""
    print("\n" + "=" * 80)
    print("TEST 2: Deployment Packager")
    print("=" * 80)

    from psy_agents_noaug.deployment import DeploymentPackager

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model
        model = SimpleModel()
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model, model_path)

        # Create packager
        packager = DeploymentPackager()
        print("\n✓ Created packager")

        # Create package
        package_path = packager.create_deployment_package(
            model_path=model_path,
            output_dir=Path(tmpdir) / "packages",
            model_name="test_model",
            version="1.0.0",
            config={"test": True},
            include_dependencies=True,
        )

        print(f"✓ Created package: {package_path}")

        # Verify package contents
        required_files = [
            "model.pt",
            "config.json",
            "package.json",
            "requirements.txt",
            "inference.py",
            "README.md",
        ]

        for file_name in required_files:
            file_path = package_path / file_name
            assert file_path.exists(), f"Missing file: {file_name}"
            print(f"  ✓ {file_name}")

    print("\n✅ TEST 2 PASSED: Deployment packager working correctly")


def test_model_deployer() -> None:
    """Test ModelDeployer."""
    print("\n" + "=" * 80)
    print("TEST 3: Model Deployer")
    print("=" * 80)

    from psy_agents_noaug.deployment import DeploymentPackager, ModelDeployer

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create package first
        model = SimpleModel()
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model, model_path)

        packager = DeploymentPackager()
        package_path = packager.create_deployment_package(
            model_path=model_path,
            output_dir=Path(tmpdir) / "packages",
            model_name="test_model",
            version="1.0.0",
        )

        print("\n✓ Created package for deployment")

        # Create deployer
        deployer = ModelDeployer()
        print("✓ Created deployer")

        # Deploy model
        deployment_info = deployer.deploy_model(
            package_path=package_path,
            deployment_config={
                "name": "test-model-deployment",
                "replicas": 2,
                "resources": {"cpu": "500m", "memory": "1Gi"},
            },
        )

        print("✓ Deployed model:")
        print(f"  Status: {deployment_info['status']}")
        print(f"  Name: {deployment_info['name']}")
        print(f"  Replicas: {deployment_info['replicas']}")

        # Health check
        health = deployer.health_check("test-model-deployment")
        print("✓ Health check:")
        print(f"  Status: {health['status']}")
        print(
            f"  Replicas ready: {health['replicas_ready']}/{health['replicas_total']}"
        )

    print("\n✅ TEST 3 PASSED: Model deployer working correctly")


def test_production_models() -> None:
    """Test production model utilities."""
    print("\n" + "=" * 80)
    print("TEST 4: Production Models")
    print("=" * 80)

    from psy_agents_noaug.deployment import ModelRegistry, get_production_models

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
        registry = ModelRegistry(tracking_uri=tracking_uri)

        # Create and register model
        model = SimpleModel()
        model_path = Path(tmpdir) / "model.pt"
        torch.save(model, model_path)

        version = registry.register_model_from_checkpoint(
            model_name="prod_model",
            checkpoint_path=model_path,
        )

        print(f"\n✓ Registered model version {version}")

        # Transition to production
        registry.transition_model_stage("prod_model", version, "Production")
        print("✓ Transitioned to Production")

        # Get production models
        prod_models = get_production_models(tracking_uri=tracking_uri)
        print(f"✓ Found {len(prod_models)} production models")

        for name, metadata in prod_models.items():
            print(f"  {name}: version={metadata.version}, stage={metadata.stage}")

    print("\n✅ TEST 4 PASSED: Production model utilities working correctly")


def main() -> None:
    """Run all tests."""
    print("=" * 80)
    print("SUPERMAX Phase 14: Deployment Tests")
    print("=" * 80)

    try:
        test_model_registry()
        test_deployment_packager()
        test_model_deployer()
        test_production_models()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\nDeployment functionality is working correctly!")
        print("You can now use:")
        print("  - ModelRegistry for model versioning and lifecycle")
        print("  - DeploymentPackager for creating deployment packages")
        print("  - ModelDeployer for deploying models")
        print("  - Production model utilities")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
