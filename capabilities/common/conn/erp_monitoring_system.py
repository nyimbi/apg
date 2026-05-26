"""
ERP Monitoring and Alerting System
Advanced monitoring for all ERP connections with intelligent alerting

This module provides comprehensive monitoring capabilities for ERP systems
including health checks, performance monitoring, and proactive alerting.
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import aiohttp
from collections import defaultdict, deque

from singer_taps.erp_registry import get_erp_registry, ERPSystemType
from .service import ConnectionManager
from .models import Connection, ConnectionStatus
from .notifications import NotificationManager

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics we monitor"""
    CONNECTION_STATUS = "connection_status"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    DATA_FRESHNESS = "data_freshness"
    AUTHENTICATION_STATUS = "authentication_status"
    SYNC_DURATION = "sync_duration"
    RECORD_COUNT = "record_count"


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    erp_system: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ERPSystemHealth:
    """Health status for an ERP system"""
    erp_system: str
    system_type: ERPSystemType
    overall_status: ConnectionStatus
    last_check: datetime
    metrics: Dict[MetricType, List[MetricPoint]] = field(default_factory=dict)
    active_alerts: List[Alert] = field(default_factory=list)
    uptime_percentage: float = 100.0
    avg_response_time: float = 0.0
    current_throughput: float = 0.0


class ERPMonitoringSystem:
    """Advanced ERP monitoring and alerting system"""

    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.registry = get_erp_registry()
        self.notification_manager = NotificationManager()

        # Monitoring state
        self.erp_health = {}  # erp_system -> ERPSystemHealth
        self.alerts = {}  # alert_id -> Alert
        self.monitoring_enabled = True
        self.check_interval = 60  # seconds
        self.metric_retention_hours = 24

        # Alerting thresholds
        self.thresholds = {
            MetricType.RESPONSE_TIME: {"warning": 5.0, "critical": 15.0},  # seconds
            MetricType.ERROR_RATE: {"warning": 5.0, "critical": 10.0},     # percentage
            MetricType.THROUGHPUT: {"warning": 10.0, "critical": 5.0},     # records/second (min)
            MetricType.DATA_FRESHNESS: {"warning": 3600, "critical": 7200} # seconds
        }

        # Callback functions for custom alerts
        self.alert_callbacks = []

        # Background monitoring task
        self.monitoring_task = None

    async def start_monitoring(self) -> None:
        """Start the monitoring system"""
        logger.info("Starting ERP monitoring system")

        self.monitoring_enabled = True

        # Initialize health status for all ERP connections
        await self._initialize_erp_health()

        # Start background monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("ERP monitoring system started")

    async def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        logger.info("Stopping ERP monitoring system")

        self.monitoring_enabled = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("ERP monitoring system stopped")

    async def _initialize_erp_health(self) -> None:
        """Initialize health status for all ERP connections"""
        connections = await self.connection_manager.list_connections()

        for connection in connections:
            if connection.connection_type.value == "erp":
                # Determine ERP system type from connection
                erp_system = self._get_erp_system_name(connection)
                system_type = self._get_system_type_from_connection(connection)

                if erp_system and system_type:
                    self.erp_health[erp_system] = ERPSystemHealth(
                        erp_system=erp_system,
                        system_type=system_type,
                        overall_status=connection.status,
                        last_check=datetime.now(timezone.utc),
                        metrics=defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute data
                    )

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_enabled:
            try:
                await self._perform_health_checks()
                await self._collect_metrics()
                await self._evaluate_alerts()
                await self._cleanup_old_data()

                # Wait for next check interval
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def _perform_health_checks(self) -> None:
        """Perform health checks on all ERP systems"""
        current_time = datetime.now(timezone.utc)

        for erp_system, health in self.erp_health.items():
            try:
                # Get connection for this ERP system
                connection = await self._get_connection_for_erp(erp_system)
                if not connection:
                    continue

                # Perform health check
                start_time = time.time()
                is_healthy = await connection.test_connection()
                response_time = time.time() - start_time

                # Update health status
                health.overall_status = ConnectionStatus.ACTIVE if is_healthy else ConnectionStatus.ERROR
                health.last_check = current_time
                health.avg_response_time = response_time

                # Record metrics
                self._record_metric(health, MetricType.CONNECTION_STATUS, 1.0 if is_healthy else 0.0)
                self._record_metric(health, MetricType.RESPONSE_TIME, response_time)

                # Calculate uptime
                await self._calculate_uptime(health)

            except Exception as e:
                logger.error(f"Health check failed for {erp_system}: {e}")
                health.overall_status = ConnectionStatus.ERROR
                self._record_metric(health, MetricType.CONNECTION_STATUS, 0.0)

    async def _collect_metrics(self) -> None:
        """Collect performance metrics from ERP systems"""
        for erp_system, health in self.erp_health.items():
            try:
                # Get connection metrics
                connection = await self._get_connection_for_erp(erp_system)
                if not connection:
                    continue

                # Collect various metrics
                await self._collect_throughput_metrics(health, connection)
                await self._collect_error_metrics(health, connection)
                await self._collect_data_freshness_metrics(health, connection)

            except Exception as e:
                logger.error(f"Metric collection failed for {erp_system}: {e}")

    async def _collect_throughput_metrics(self, health: ERPSystemHealth, connection: Connection) -> None:
        """Collect throughput metrics"""
        try:
            # Simulate throughput calculation - in real implementation,
            # this would track actual record processing rates
            current_throughput = await self._estimate_throughput(connection)
            health.current_throughput = current_throughput

            self._record_metric(health, MetricType.THROUGHPUT, current_throughput)

        except Exception as e:
            logger.warning(f"Failed to collect throughput metrics: {e}")

    async def _collect_error_metrics(self, health: ERPSystemHealth, connection: Connection) -> None:
        """Collect error rate metrics"""
        try:
            # Calculate error rate from recent sync attempts
            error_rate = await self._calculate_error_rate(connection)

            self._record_metric(health, MetricType.ERROR_RATE, error_rate)

        except Exception as e:
            logger.warning(f"Failed to collect error metrics: {e}")

    async def _collect_data_freshness_metrics(self, health: ERPSystemHealth, connection: Connection) -> None:
        """Collect data freshness metrics"""
        try:
            # Check how fresh the data is
            last_sync = await self._get_last_sync_time(connection)
            if last_sync:
                freshness_seconds = (datetime.now(timezone.utc) - last_sync).total_seconds()
                self._record_metric(health, MetricType.DATA_FRESHNESS, freshness_seconds)

        except Exception as e:
            logger.warning(f"Failed to collect freshness metrics: {e}")

    async def _evaluate_alerts(self) -> None:
        """Evaluate metrics against thresholds and generate alerts"""
        for erp_system, health in self.erp_health.items():
            try:
                # Check each metric type for threshold violations
                for metric_type, thresholds in self.thresholds.items():
                    if metric_type in health.metrics:
                        recent_metrics = list(health.metrics[metric_type])[-5:]  # Last 5 measurements
                        if recent_metrics:
                            avg_value = statistics.mean(point.value for point in recent_metrics)

                            # Evaluate thresholds
                            await self._check_metric_threshold(
                                health, metric_type, avg_value, thresholds
                            )

                # Check for connection status alerts
                if health.overall_status == ConnectionStatus.ERROR:
                    await self._create_alert(
                        health, MetricType.CONNECTION_STATUS, AlertSeverity.CRITICAL,
                        f"ERP system {erp_system} is not responding"
                    )

            except Exception as e:
                logger.error(f"Alert evaluation failed for {erp_system}: {e}")

    async def _check_metric_threshold(self, health: ERPSystemHealth, metric_type: MetricType,
                                    value: float, thresholds: Dict[str, float]) -> None:
        """Check if metric violates thresholds"""
        erp_system = health.erp_system

        # Determine severity based on thresholds
        severity = None
        if metric_type in [MetricType.RESPONSE_TIME, MetricType.ERROR_RATE, MetricType.DATA_FRESHNESS]:
            # Higher values are worse
            if value >= thresholds.get("critical", float('inf')):
                severity = AlertSeverity.CRITICAL
            elif value >= thresholds.get("warning", float('inf')):
                severity = AlertSeverity.WARNING
        elif metric_type == MetricType.THROUGHPUT:
            # Lower values are worse
            if value <= thresholds.get("critical", 0):
                severity = AlertSeverity.CRITICAL
            elif value <= thresholds.get("warning", 0):
                severity = AlertSeverity.WARNING

        if severity:
            message = self._format_threshold_alert_message(metric_type, value, thresholds)
            await self._create_alert(health, metric_type, severity, message)

    async def _create_alert(self, health: ERPSystemHealth, metric_type: MetricType,
                          severity: AlertSeverity, message: str) -> None:
        """Create and process a new alert"""
        alert_id = f"{health.erp_system}_{metric_type.value}_{int(time.time())}"

        # Check if similar alert already exists
        existing_alert = self._find_existing_alert(health.erp_system, metric_type, severity)
        if existing_alert:
            return  # Don't create duplicate alerts

        alert = Alert(
            alert_id=alert_id,
            erp_system=health.erp_system,
            severity=severity,
            metric_type=metric_type,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )

        # Store alert
        self.alerts[alert_id] = alert
        health.active_alerts.append(alert)

        # Send notifications
        await self._send_alert_notification(alert)

        # Execute custom callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Alert created: {alert.severity.value.upper()} - {alert.message}")

    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification"""
        try:
            notification_data = {
                "alert_id": alert.alert_id,
                "erp_system": alert.erp_system,
                "severity": alert.severity.value,
                "metric": alert.metric_type.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }

            # Send to notification manager
            await self.notification_manager.send_notification(
                channel="erp_alerts",
                message=f"ERP Alert: {alert.message}",
                data=notification_data,
                priority=alert.severity.value
            )

        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    async def resolve_alert(self, alert_id: str) -> bool:
        """Manually resolve an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)

            # Remove from active alerts
            erp_system = alert.erp_system
            if erp_system in self.erp_health:
                health = self.erp_health[erp_system]
                health.active_alerts = [a for a in health.active_alerts if a.alert_id != alert_id]

            logger.info(f"Alert {alert_id} resolved manually")
            return True

        return False

    async def _calculate_uptime(self, health: ERPSystemHealth) -> None:
        """Calculate uptime percentage for ERP system"""
        if MetricType.CONNECTION_STATUS in health.metrics:
            status_points = list(health.metrics[MetricType.CONNECTION_STATUS])
            if status_points:
                # Calculate uptime from recent status checks
                uptime_checks = [point.value for point in status_points[-100:]]  # Last 100 checks
                uptime = sum(uptime_checks) / len(uptime_checks) * 100
                health.uptime_percentage = uptime

    def _record_metric(self, health: ERPSystemHealth, metric_type: MetricType, value: float,
                      metadata: Optional[Dict] = None) -> None:
        """Record a metric point"""
        metric_point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            metadata=metadata or {}
        )

        health.metrics[metric_type].append(metric_point)

    async def _cleanup_old_data(self) -> None:
        """Clean up old metrics and resolved alerts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metric_retention_hours)

        # Clean up old metrics
        for health in self.erp_health.values():
            for metric_type, metrics in health.metrics.items():
                # Remove old metrics
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()

        # Clean up resolved alerts older than 7 days
        alert_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        resolved_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.resolved and alert.resolved_at and alert.resolved_at < alert_cutoff
        ]

        for alert_id in resolved_alerts:
            del self.alerts[alert_id]

    # Utility methods
    async def _get_connection_for_erp(self, erp_system: str) -> Optional[Connection]:
        """Get connection object for ERP system"""
        connections = await self.connection_manager.list_connections()
        for connection in connections:
            if self._get_erp_system_name(connection) == erp_system:
                return connection
        return None

    def _get_erp_system_name(self, connection: Connection) -> Optional[str]:
        """Extract ERP system name from connection"""
        # Implementation depends on how connections store ERP system info
        return connection.tap_config.get("erp_system_name") or connection.singer_tap

    def _get_system_type_from_connection(self, connection: Connection) -> Optional[ERPSystemType]:
        """Get ERP system type from connection"""
        # Map connection info to system type
        system_name = self._get_erp_system_name(connection)
        if system_name:
            for system_type in ERPSystemType:
                if system_name.lower() in system_type.value.lower():
                    return system_type
        return None

    async def _estimate_throughput(self, connection: Connection) -> float:
        """Estimate current throughput for connection"""
        # Simplified estimation - in real implementation would track actual sync rates
        return 50.0  # records per second

    async def _calculate_error_rate(self, connection: Connection) -> float:
        """Calculate error rate for connection"""
        # Simplified calculation - in real implementation would track sync failures
        return 2.0  # percentage

    async def _get_last_sync_time(self, connection: Connection) -> Optional[datetime]:
        """Get last successful sync time"""
        # Implementation would check actual sync history
        return datetime.now(timezone.utc) - timedelta(minutes=30)

    def _find_existing_alert(self, erp_system: str, metric_type: MetricType,
                           severity: AlertSeverity) -> Optional[Alert]:
        """Find existing unresolved alert for same condition"""
        for alert in self.alerts.values():
            if (alert.erp_system == erp_system and
                alert.metric_type == metric_type and
                alert.severity == severity and
                not alert.resolved):
                return alert
        return None

    def _format_threshold_alert_message(self, metric_type: MetricType, value: float,
                                      thresholds: Dict[str, float]) -> str:
        """Format alert message for threshold violation"""
        if metric_type == MetricType.RESPONSE_TIME:
            return f"Response time {value:.2f}s exceeds threshold (warning: {thresholds.get('warning', 0):.1f}s, critical: {thresholds.get('critical', 0):.1f}s)"
        elif metric_type == MetricType.ERROR_RATE:
            return f"Error rate {value:.1f}% exceeds threshold (warning: {thresholds.get('warning', 0):.1f}%, critical: {thresholds.get('critical', 0):.1f}%)"
        elif metric_type == MetricType.THROUGHPUT:
            return f"Throughput {value:.1f} records/sec below threshold (warning: {thresholds.get('warning', 0):.1f}, critical: {thresholds.get('critical', 0):.1f})"
        elif metric_type == MetricType.DATA_FRESHNESS:
            hours = value / 3600
            return f"Data freshness {hours:.1f} hours exceeds threshold"
        else:
            return f"Metric {metric_type.value} value {value} violates threshold"

    # Public API methods
    def get_erp_health_status(self, erp_system: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for ERP system(s)"""
        if erp_system:
            health = self.erp_health.get(erp_system)
            if health:
                return self._format_health_status(health)
            return {}
        else:
            return {
                erp_sys: self._format_health_status(health)
                for erp_sys, health in self.erp_health.items()
            }

    def _format_health_status(self, health: ERPSystemHealth) -> Dict[str, Any]:
        """Format health status for API response"""
        return {
            "erp_system": health.erp_system,
            "system_type": health.system_type.value,
            "overall_status": health.overall_status.value,
            "last_check": health.last_check.isoformat(),
            "uptime_percentage": health.uptime_percentage,
            "avg_response_time": health.avg_response_time,
            "current_throughput": health.current_throughput,
            "active_alerts": len(health.active_alerts),
            "alert_severities": [alert.severity.value for alert in health.active_alerts]
        }

    def get_active_alerts(self, erp_system: Optional[str] = None,
                         severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Get active alerts with optional filtering"""
        alerts = []
        for alert in self.alerts.values():
            if alert.resolved:
                continue

            if erp_system and alert.erp_system != erp_system:
                continue

            if severity and alert.severity != severity:
                continue

            alerts.append({
                "alert_id": alert.alert_id,
                "erp_system": alert.erp_system,
                "severity": alert.severity.value,
                "metric_type": alert.metric_type.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "age_minutes": (datetime.now(timezone.utc) - alert.timestamp).total_seconds() / 60
            })

        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add custom alert callback function"""
        self.alert_callbacks.append(callback)

    def update_thresholds(self, metric_type: MetricType, thresholds: Dict[str, float]) -> None:
        """Update alerting thresholds for a metric type"""
        self.thresholds[metric_type] = thresholds
        logger.info(f"Updated thresholds for {metric_type.value}: {thresholds}")


async def create_erp_monitoring_system(connection_manager: ConnectionManager) -> ERPMonitoringSystem:
    """Factory function to create and initialize ERP monitoring system"""
    monitoring_system = ERPMonitoringSystem(connection_manager)
    await monitoring_system.start_monitoring()
    return monitoring_system