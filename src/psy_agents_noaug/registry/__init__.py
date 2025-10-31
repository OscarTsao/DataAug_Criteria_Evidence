#!/usr/bin/env python
"""Model versioning and registry module (Phase 20).

This module provides production-ready model management including:
- Model versioning and tagging
- Model metadata tracking
- Promotion workflows (dev → staging → production)
- Model lineage and provenance
- MLflow registry integration

Key Features:
- Semantic versioning support
- Stage-based promotion (dev/staging/production)
- Comprehensive metadata tracking
- Lineage tracking for reproducibility
- Integration with MLflow Model Registry
"""

from __future__ import annotations

from psy_agents_noaug.registry.lineage import (
    LineageTracker,
    ModelLineage,
    track_lineage,
)
from psy_agents_noaug.registry.metadata import (
    MetadataManager,
    ModelMetadata,
    create_metadata,
)
from psy_agents_noaug.registry.promotion import (
    ModelPromoter,
    PromotionCriteria,
    PromotionWorkflow,
    Stage,
    promote_model,
)
from psy_agents_noaug.registry.versioning import (
    ModelRegistry,
    ModelVersion,
    SemanticVersion,
    create_version,
)

__all__ = [
    # Versioning
    "ModelRegistry",
    "ModelVersion",
    "SemanticVersion",
    "create_version",
    # Metadata
    "ModelMetadata",
    "MetadataManager",
    "create_metadata",
    # Promotion
    "ModelPromoter",
    "PromotionCriteria",
    "PromotionWorkflow",
    "Stage",
    "promote_model",
    # Lineage
    "LineageTracker",
    "ModelLineage",
    "track_lineage",
]
