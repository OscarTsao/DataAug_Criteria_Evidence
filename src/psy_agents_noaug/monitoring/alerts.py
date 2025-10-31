#!/usr/bin/env python
"""Alerting system for model monitoring (Phase 17).

This module provides alerting functionality including:
- Alert rules and conditions
- Multiple notification channels
- Alert aggregation and deduplication
- Severity levels
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""

    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    FILE = "file"


@dataclass
class AlertRule:
    """An alert rule definition."""

    name: str
    condition: Callable[[], bool]
    severity: AlertSeverity
    message_template: str
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    cooldown_minutes: int = 60  # Min time between alerts


@dataclass
class Alert:
    """A triggered alert."""

    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manage and dispatch alerts."""

    def __init__(
        self,
        alert_log_file: Path | str | None = None,
    ):
        """Initialize alert manager.

        Args:
            alert_log_file: File to log alerts to
        """
        self.rules: dict[str, AlertRule] = {}
        self.alert_history: list[Alert] = []
        self.last_alert_time: dict[str, datetime] = {}

        # Configure alert log file
        if alert_log_file:
            self.alert_log_file = Path(alert_log_file)
            self.alert_log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.alert_log_file = None

        LOGGER.info("Initialized AlertManager")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: Alert rule to add
        """
        self.rules[rule.name] = rule
        LOGGER.debug("Added alert rule: %s", rule.name)

    def _should_send_alert(self, rule_name: str, cooldown_minutes: int) -> bool:
        """Check if alert should be sent (respects cooldown).

        Args:
            rule_name: Name of the rule
            cooldown_minutes: Cooldown period in minutes

        Returns:
            True if alert should be sent
        """
        if rule_name not in self.last_alert_time:
            return True

        last_time = self.last_alert_time[rule_name]
        cooldown = timedelta(minutes=cooldown_minutes)

        return datetime.now() - last_time > cooldown

    def _send_to_log(self, alert: Alert) -> None:
        """Send alert to log.

        Args:
            alert: Alert to send
        """
        severity_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }

        LOGGER.log(
            severity_map[alert.severity],
            "ALERT [%s]: %s",
            alert.severity.value.upper(),
            alert.message,
        )

    def _send_to_file(self, alert: Alert) -> None:
        """Send alert to file.

        Args:
            alert: Alert to send
        """
        if self.alert_log_file is None:
            return

        alert_data = {
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata,
        }

        with self.alert_log_file.open("a") as f:
            f.write(json.dumps(alert_data) + "\n")

    def _send_to_email(self, alert: Alert) -> None:
        """Send alert via email.

        Args:
            alert: Alert to send
        """
        # Placeholder for email integration
        LOGGER.info("Would send email alert: %s", alert.message)

    def _send_to_slack(self, alert: Alert) -> None:
        """Send alert to Slack.

        Args:
            alert: Alert to send
        """
        # Placeholder for Slack integration
        LOGGER.info("Would send Slack alert: %s", alert.message)

    def _send_to_pagerduty(self, alert: Alert) -> None:
        """Send alert to PagerDuty.

        Args:
            alert: Alert to send
        """
        # Placeholder for PagerDuty integration
        LOGGER.info("Would send PagerDuty alert: %s", alert.message)

    def _dispatch_alert(self, alert: Alert, channels: list[AlertChannel]) -> None:
        """Dispatch alert to configured channels.

        Args:
            alert: Alert to dispatch
            channels: List of channels to send to
        """
        for channel in channels:
            if channel == AlertChannel.LOG:
                self._send_to_log(alert)
            elif channel == AlertChannel.FILE:
                self._send_to_file(alert)
            elif channel == AlertChannel.EMAIL:
                self._send_to_email(alert)
            elif channel == AlertChannel.SLACK:
                self._send_to_slack(alert)
            elif channel == AlertChannel.PAGERDUTY:
                self._send_to_pagerduty(alert)

    def check_rules(self, context: dict[str, Any] | None = None) -> list[Alert]:
        """Check all alert rules.

        Args:
            context: Optional context for message formatting

        Returns:
            List of triggered alerts
        """
        if context is None:
            context = {}

        triggered_alerts = []

        for rule in self.rules.values():
            try:
                # Check condition
                if not rule.condition():
                    continue

                # Check cooldown
                if not self._should_send_alert(rule.name, rule.cooldown_minutes):
                    LOGGER.debug(
                        "Alert %s in cooldown period",
                        rule.name,
                    )
                    continue

                # Create alert
                message = rule.message_template.format(**context)
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=message,
                    timestamp=datetime.now(),
                    metadata=context,
                )

                # Dispatch alert
                self._dispatch_alert(alert, rule.channels)

                # Record alert
                self.alert_history.append(alert)
                self.last_alert_time[rule.name] = alert.timestamp
                triggered_alerts.append(alert)

                LOGGER.info(
                    "Triggered alert: %s (%s)",
                    rule.name,
                    rule.severity.value,
                )

            except Exception as e:
                LOGGER.error("Error checking rule %s: %s", rule.name, e)

        return triggered_alerts

    def get_recent_alerts(
        self,
        hours: int = 24,
    ) -> list[Alert]:
        """Get recent alerts.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent alerts
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert
            for alert in self.alert_history
            if alert.timestamp > cutoff
        ]

    def get_summary(self) -> dict[str, Any]:
        """Get alerting summary.

        Returns:
            Summary dict
        """
        recent_alerts = self.get_recent_alerts(hours=24)

        # Count by severity
        severity_counts = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 0,
            AlertSeverity.CRITICAL: 0,
        }

        for alert in recent_alerts:
            severity_counts[alert.severity] += 1

        return {
            "timestamp": datetime.now().isoformat(),
            "total_rules": len(self.rules),
            "alerts_last_24h": len(recent_alerts),
            "severity_counts": {
                severity.value: count
                for severity, count in severity_counts.items()
            },
            "recent_alerts": [
                {
                    "rule_name": alert.rule_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in recent_alerts[-10:]  # Last 10 alerts
            ],
        }


def send_alert(
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    channels: list[AlertChannel] | None = None,
) -> None:
    """Send an alert (convenience function).

    Args:
        message: Alert message
        severity: Alert severity
        channels: Notification channels
    """
    if channels is None:
        channels = [AlertChannel.LOG]

    manager = AlertManager()

    alert = Alert(
        rule_name="ad_hoc",
        severity=severity,
        message=message,
        timestamp=datetime.now(),
    )

    manager._dispatch_alert(alert, channels)
