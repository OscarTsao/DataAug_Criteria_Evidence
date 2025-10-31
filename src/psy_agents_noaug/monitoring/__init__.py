#!/usr/bin/env python
"""Model monitoring and observability (Phase 17).

This module provides comprehensive model monitoring including:
- Performance monitoring (latency, throughput, resource usage)
- Prediction drift detection
- Data drift detection
- Model health monitoring
- Alerting and notifications

Key Features:
- Real-time performance metrics
- Statistical drift detection
- Automated health checks
- Configurable alerting rules
"""

from __future__ import annotations

from psy_agents_noaug.monitoring.performance import (
    PerformanceMonitor,
    monitor_performance,
)
from psy_agents_noaug.monitoring.drift import (
    DriftDetector,
    detect_drift,
)
from psy_agents_noaug.monitoring.health import (
    HealthMonitor,
    HealthCheck,
    check_health,
)
from psy_agents_noaug.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    send_alert,
)

__all__ = [
    # Performance Monitoring
    "PerformanceMonitor",
    "monitor_performance",
    # Drift Detection
    "DriftDetector",
    "detect_drift",
    # Health Monitoring
    "HealthMonitor",
    "HealthCheck",
    "check_health",
    # Alerting
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "send_alert",
]
