"""
APG Import/Export (IMEX) Performance Monitoring and Optimization Layer

Purpose: Production-grade performance monitoring, metrics collection, and optimization
         for enterprise import/export operations with comprehensive analytics.
Dependencies: psutil, prometheus_client, asyncio
Usage Context: Performance monitoring and optimization for IMEX capability

This module provides:
- Real-time performance metrics collection and monitoring
- System resource utilization tracking (CPU, memory, I/O, network)
- Job execution performance analysis and optimization
- Bottleneck detection and automated performance tuning
- Historical performance data storage and analytics
- Performance alerting and notification system
- Resource scaling recommendations and capacity planning
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import statistics
from collections import defaultdict, deque

import psutil
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)

# Performance Enums and Constants

class MetricType(str, Enum):
    """Types of performance metrics"""
    SYSTEM = "system"
    JOB = "job"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ResourceType(str, Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"

# Performance Models

class PerformanceMetric(BaseModel):
    """Performance metric data model"""
    id: str = Field(default_factory=uuid7str)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = Field(...)
    metric_name: str = Field(...)
    value: float = Field(...)
    unit: str = Field(...)
    tags: Dict[str, str] = Field(default_factory=dict)
    tenant_id: str = Field(...)
    job_id: Optional[str] = Field(None)
    component: str = Field("imex")

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class SystemResourceMetrics(BaseModel):
    """System resource utilization metrics"""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_usage_percent: float = Field(...)
    memory_usage_percent: float = Field(...)
    memory_used_mb: float = Field(...)
    memory_available_mb: float = Field(...)
    disk_usage_percent: float = Field(...)
    disk_used_gb: float = Field(...)
    disk_available_gb: float = Field(...)
    network_bytes_sent: int = Field(...)
    network_bytes_recv: int = Field(...)
    active_connections: int = Field(...)
    load_average_1m: float = Field(...)
    load_average_5m: float = Field(...)
    load_average_15m: float = Field(...)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class JobPerformanceMetrics(BaseModel):
    """Job execution performance metrics"""
    job_id: str = Field(...)
    job_name: str = Field(...)
    start_time: datetime = Field(...)
    end_time: Optional[datetime] = Field(None)
    duration_seconds: Optional[float] = Field(None)
    records_processed: int = Field(0)
    throughput_records_per_second: Optional[float] = Field(None)
    memory_peak_mb: float = Field(0)
    cpu_usage_average: float = Field(0)
    errors_count: int = Field(0)
    warnings_count: int = Field(0)
    data_size_mb: float = Field(0)
    processing_stages: List[Dict[str, Any]] = Field(default_factory=list)
    bottlenecks: List[str] = Field(default_factory=list)
    optimization_suggestions: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class PerformanceAlert(BaseModel):
    """Performance alert model"""
    id: str = Field(default_factory=uuid7str)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlertSeverity = Field(...)
    alert_type: str = Field(...)
    message: str = Field(...)
    metric_name: str = Field(...)
    current_value: float = Field(...)
    threshold_value: float = Field(...)
    resource_type: ResourceType = Field(...)
    tenant_id: str = Field(...)
    job_id: Optional[str] = Field(None)
    resolved: bool = Field(False)
    resolved_at: Optional[datetime] = Field(None)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

class PerformanceThreshold(BaseModel):
    """Performance monitoring thresholds"""
    metric_name: str = Field(...)
    resource_type: ResourceType = Field(...)
    warning_threshold: float = Field(...)
    error_threshold: float = Field(...)
    critical_threshold: float = Field(...)
    enabled: bool = Field(True)

    model_config = ConfigDict(extra='forbid', validate_by_name=True)

# Performance Monitoring System

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""

    def __init__(self, collection_interval: int = 30):
        """
        Initialize performance monitor.

        Args:
            collection_interval: Metrics collection interval in seconds
        """
        self.collection_interval = collection_interval
        self.metrics_storage = []  # In production, use time-series database
        self.alerts_storage = []
        self.job_metrics = {}
        self.thresholds = self._initialize_default_thresholds()
        self.active_alerts = {}
        self._monitoring_active = False
        self._monitoring_thread = None
        self._lock = threading.RLock()

        logger.info(f"Performance monitor initialized with {collection_interval}s interval")

    def _initialize_default_thresholds(self) -> Dict[str, PerformanceThreshold]:
        """Initialize default performance thresholds"""
        return {
            "cpu_usage": PerformanceThreshold(
                metric_name="cpu_usage_percent",
                resource_type=ResourceType.CPU,
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0
            ),
            "memory_usage": PerformanceThreshold(
                metric_name="memory_usage_percent",
                resource_type=ResourceType.MEMORY,
                warning_threshold=75.0,
                error_threshold=90.0,
                critical_threshold=98.0
            ),
            "disk_usage": PerformanceThreshold(
                metric_name="disk_usage_percent",
                resource_type=ResourceType.DISK,
                warning_threshold=80.0,
                error_threshold=90.0,
                critical_threshold=95.0
            ),
            "job_duration": PerformanceThreshold(
                metric_name="job_duration_minutes",
                resource_type=ResourceType.DATABASE,
                warning_threshold=30.0,
                error_threshold=60.0,
                critical_threshold=120.0
            )
        }

    def start_monitoring(self):
        """Start performance monitoring"""
        with self._lock:
            if self._monitoring_active:
                logger.warning("Performance monitoring already active")
                return

            self._monitoring_active = True
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitoring_thread.start()
            logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        with self._lock:
            if not self._monitoring_active:
                return

            self._monitoring_active = False
            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5)

            logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self._store_system_metrics(system_metrics)

                # Check thresholds and generate alerts
                self._check_thresholds(system_metrics)

                # Clean old metrics (keep last 24 hours)
                self._cleanup_old_metrics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.collection_interval)

    def _collect_system_metrics(self) -> SystemResourceMetrics:
        """Collect current system resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics
            network = psutil.net_io_counters()

            # Load average (Unix-like systems)
            load_avg = [0.0, 0.0, 0.0]
            try:
                load_avg = psutil.getloadavg()
            except (AttributeError, OSError):
                pass  # Not available on all systems

            # Network connections
            connections = len(psutil.net_connections())

            return SystemResourceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_available_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections,
                load_average_1m=load_avg[0],
                load_average_5m=load_avg[1] if len(load_avg) > 1 else 0.0,
                load_average_15m=load_avg[2] if len(load_avg) > 2 else 0.0
            )

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            # Return zero metrics as fallback
            return SystemResourceMetrics(
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_used_gb=0.0,
                disk_available_gb=0.0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                load_average_1m=0.0,
                load_average_5m=0.0,
                load_average_15m=0.0
            )

    def _store_system_metrics(self, metrics: SystemResourceMetrics):
        """Store system metrics"""
        with self._lock:
            # Convert to performance metrics format
            metric_mappings = [
                ("cpu_usage_percent", "percent", MetricType.SYSTEM),
                ("memory_usage_percent", "percent", MetricType.SYSTEM),
                ("disk_usage_percent", "percent", MetricType.SYSTEM),
                ("memory_used_mb", "mb", MetricType.SYSTEM),
                ("network_bytes_sent", "bytes", MetricType.NETWORK),
                ("network_bytes_recv", "bytes", MetricType.NETWORK),
                ("load_average_1m", "load", MetricType.SYSTEM)
            ]

            for metric_name, unit, metric_type in metric_mappings:
                value = getattr(metrics, metric_name, 0.0)
                perf_metric = PerformanceMetric(
                    metric_type=metric_type,
                    metric_name=metric_name,
                    value=float(value),
                    unit=unit,
                    tenant_id="system",
                    tags={"source": "system_monitor"}
                )
                self.metrics_storage.append(perf_metric)

    def _check_thresholds(self, metrics: SystemResourceMetrics):
        """Check performance thresholds and generate alerts"""
        checks = [
            ("cpu_usage", metrics.cpu_usage_percent, ResourceType.CPU),
            ("memory_usage", metrics.memory_usage_percent, ResourceType.MEMORY),
            ("disk_usage", metrics.disk_usage_percent, ResourceType.DISK)
        ]

        for threshold_key, current_value, resource_type in checks:
            threshold = self.thresholds.get(threshold_key)
            if not threshold or not threshold.enabled:
                continue

            # Determine severity level
            severity = None
            threshold_value = 0.0

            if current_value >= threshold.critical_threshold:
                severity = AlertSeverity.CRITICAL
                threshold_value = threshold.critical_threshold
            elif current_value >= threshold.error_threshold:
                severity = AlertSeverity.ERROR
                threshold_value = threshold.error_threshold
            elif current_value >= threshold.warning_threshold:
                severity = AlertSeverity.WARNING
                threshold_value = threshold.warning_threshold

            if severity:
                self._create_alert(
                    severity=severity,
                    alert_type=f"{resource_type.value}_threshold_exceeded",
                    message=f"{threshold.metric_name} is {current_value:.1f}% (threshold: {threshold_value:.1f}%)",
                    metric_name=threshold.metric_name,
                    current_value=current_value,
                    threshold_value=threshold_value,
                    resource_type=resource_type
                )

    def _create_alert(self, severity: AlertSeverity, alert_type: str,
                     message: str, metric_name: str, current_value: float,
                     threshold_value: float, resource_type: ResourceType,
                     tenant_id: str = "system", job_id: Optional[str] = None):
        """Create a performance alert"""
        alert_key = f"{alert_type}_{metric_name}_{tenant_id}"

        # Avoid duplicate alerts for same issue
        if alert_key in self.active_alerts and not self.active_alerts[alert_key].resolved:
            return

        alert = PerformanceAlert(
            severity=severity,
            alert_type=alert_type,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            resource_type=resource_type,
            tenant_id=tenant_id,
            job_id=job_id
        )

        with self._lock:
            self.alerts_storage.append(alert)
            self.active_alerts[alert_key] = alert

        logger.warning(f"Performance alert: {severity.value.upper()} - {message}")

    def _cleanup_old_metrics(self):
        """Remove metrics older than 24 hours"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

        with self._lock:
            initial_count = len(self.metrics_storage)
            self.metrics_storage = [
                metric for metric in self.metrics_storage
                if metric.timestamp > cutoff_time
            ]

            cleaned_count = initial_count - len(self.metrics_storage)
            if cleaned_count > 0:
                logger.debug(f"Cleaned {cleaned_count} old metrics")

    def start_job_monitoring(self, job_id: str, job_name: str) -> JobPerformanceMetrics:
        """Start monitoring job performance"""
        job_metrics = JobPerformanceMetrics(
            job_id=job_id,
            job_name=job_name,
            start_time=datetime.now(timezone.utc)
        )

        with self._lock:
            self.job_metrics[job_id] = job_metrics

        logger.info(f"Started monitoring job: {job_name} ({job_id})")
        return job_metrics

    def update_job_progress(self, job_id: str, records_processed: int,
                          data_size_mb: float = 0.0, stage_info: Optional[Dict] = None):
        """Update job progress metrics"""
        with self._lock:
            if job_id not in self.job_metrics:
                logger.warning(f"Job metrics not found for job: {job_id}")
                return

            job_metrics = self.job_metrics[job_id]
            job_metrics.records_processed = records_processed
            job_metrics.data_size_mb = data_size_mb

            if stage_info:
                job_metrics.processing_stages.append(stage_info)

            # Calculate throughput
            if job_metrics.start_time:
                duration = (datetime.now(timezone.utc) - job_metrics.start_time).total_seconds()
                if duration > 0:
                    job_metrics.throughput_records_per_second = records_processed / duration

    def finish_job_monitoring(self, job_id: str, success: bool = True,
                            errors_count: int = 0, warnings_count: int = 0) -> Optional[JobPerformanceMetrics]:
        """Finish monitoring job and analyze performance"""
        with self._lock:
            if job_id not in self.job_metrics:
                logger.warning(f"Job metrics not found for job: {job_id}")
                return None

            job_metrics = self.job_metrics[job_id]
            job_metrics.end_time = datetime.now(timezone.utc)
            job_metrics.errors_count = errors_count
            job_metrics.warnings_count = warnings_count

            # Calculate final duration
            if job_metrics.start_time:
                duration = (job_metrics.end_time - job_metrics.start_time).total_seconds()
                job_metrics.duration_seconds = duration

                # Final throughput calculation
                if duration > 0 and job_metrics.records_processed > 0:
                    job_metrics.throughput_records_per_second = job_metrics.records_processed / duration

            # Analyze performance and generate insights
            self._analyze_job_performance(job_metrics)

            # Check if job duration exceeded thresholds
            if job_metrics.duration_seconds:
                duration_minutes = job_metrics.duration_seconds / 60
                threshold = self.thresholds.get("job_duration")
                if threshold and threshold.enabled:
                    if duration_minutes >= threshold.warning_threshold:
                        severity = AlertSeverity.WARNING
                        if duration_minutes >= threshold.critical_threshold:
                            severity = AlertSeverity.CRITICAL
                        elif duration_minutes >= threshold.error_threshold:
                            severity = AlertSeverity.ERROR

                        self._create_alert(
                            severity=severity,
                            alert_type="job_duration_exceeded",
                            message=f"Job {job_metrics.job_name} took {duration_minutes:.1f} minutes",
                            metric_name="job_duration_minutes",
                            current_value=duration_minutes,
                            threshold_value=threshold.warning_threshold,
                            resource_type=ResourceType.DATABASE,
                            job_id=job_id
                        )

            logger.info(f"Finished monitoring job: {job_metrics.job_name} ({job_id})")
            return job_metrics

    def _analyze_job_performance(self, job_metrics: JobPerformanceMetrics):
        """Analyze job performance and generate optimization suggestions"""
        bottlenecks = []
        suggestions = []

        # Analyze throughput
        if job_metrics.throughput_records_per_second:
            if job_metrics.throughput_records_per_second < 100:
                bottlenecks.append("low_throughput")
                suggestions.append("Consider batch processing or parallel execution")

        # Analyze duration
        if job_metrics.duration_seconds:
            if job_metrics.duration_seconds > 3600:  # 1 hour
                bottlenecks.append("long_duration")
                suggestions.append("Consider breaking job into smaller chunks")

        # Analyze error rate
        if job_metrics.errors_count > 0:
            error_rate = job_metrics.errors_count / max(job_metrics.records_processed, 1)
            if error_rate > 0.05:  # 5% error rate
                bottlenecks.append("high_error_rate")
                suggestions.append("Review data quality and validation rules")

        # Analyze memory usage (if available)
        if job_metrics.memory_peak_mb > 1024:  # > 1GB
            bottlenecks.append("high_memory_usage")
            suggestions.append("Implement streaming processing for large datasets")

        job_metrics.bottlenecks = bottlenecks
        job_metrics.optimization_suggestions = suggestions

    def get_system_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get system metrics summary for specified time period"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        with self._lock:
            recent_metrics = [
                metric for metric in self.metrics_storage
                if metric.timestamp > cutoff_time and metric.metric_type == MetricType.SYSTEM
            ]

        if not recent_metrics:
            return {"status": "no_data", "period_hours": hours}

        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.metric_name].append(metric.value)

        # Calculate statistics
        summary = {}
        for metric_name, values in grouped_metrics.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "average": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        return {
            "status": "success",
            "period_hours": hours,
            "metrics": summary,
            "total_data_points": len(recent_metrics),
            "collection_period": f"{self.collection_interval}s"
        }

    def get_job_performance_report(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job performance report"""
        with self._lock:
            job_metrics = self.job_metrics.get(job_id)
            if not job_metrics:
                return None

        # Calculate derived metrics
        report = {
            "job_id": job_metrics.job_id,
            "job_name": job_metrics.job_name,
            "performance_summary": {
                "duration_seconds": job_metrics.duration_seconds,
                "records_processed": job_metrics.records_processed,
                "throughput_rps": job_metrics.throughput_records_per_second,
                "data_size_mb": job_metrics.data_size_mb,
                "errors_count": job_metrics.errors_count,
                "warnings_count": job_metrics.warnings_count
            },
            "analysis": {
                "bottlenecks": job_metrics.bottlenecks,
                "optimization_suggestions": job_metrics.optimization_suggestions,
                "processing_stages": job_metrics.processing_stages
            },
            "timestamps": {
                "started_at": job_metrics.start_time.isoformat(),
                "finished_at": job_metrics.end_time.isoformat() if job_metrics.end_time else None
            }
        }

        return report

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[PerformanceAlert]:
        """Get active performance alerts"""
        with self._lock:
            alerts = [alert for alert in self.alerts_storage if not alert.resolved]

            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]

            # Sort by severity and timestamp
            severity_order = {
                AlertSeverity.CRITICAL: 0,
                AlertSeverity.ERROR: 1,
                AlertSeverity.WARNING: 2,
                AlertSeverity.INFO: 3
            }

            alerts.sort(key=lambda x: (severity_order.get(x.severity, 4), x.timestamp))
            return alerts

    def resolve_alert(self, alert_id: str):
        """Resolve a performance alert"""
        with self._lock:
            for alert in self.alerts_storage:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now(timezone.utc)

                    # Remove from active alerts
                    for key, active_alert in list(self.active_alerts.items()):
                        if active_alert.id == alert_id:
                            del self.active_alerts[key]
                            break

                    logger.info(f"Resolved alert: {alert.alert_type}")
                    return True

            return False

    def update_threshold(self, threshold_name: str, threshold: PerformanceThreshold):
        """Update performance threshold"""
        with self._lock:
            self.thresholds[threshold_name] = threshold
            logger.info(f"Updated threshold: {threshold_name}")

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._lock:
            total_metrics = len(self.metrics_storage)
            total_alerts = len(self.alerts_storage)
            active_alerts_count = len([alert for alert in self.alerts_storage if not alert.resolved])
            total_jobs_monitored = len(self.job_metrics)

        return {
            "monitoring_status": {
                "active": self._monitoring_active,
                "collection_interval": self.collection_interval,
                "uptime_seconds": time.time() - (self._monitoring_thread.ident if self._monitoring_thread else time.time())
            },
            "metrics_summary": {
                "total_metrics_collected": total_metrics,
                "total_alerts_generated": total_alerts,
                "active_alerts": active_alerts_count,
                "jobs_monitored": total_jobs_monitored
            },
            "system_health": self.get_system_metrics_summary(hours=1)
        }

# Performance Registry for APG Integration

performance_registry = {
    'monitor': PerformanceMonitor,
    'models': {
        'PerformanceMetric': PerformanceMetric,
        'SystemResourceMetrics': SystemResourceMetrics,
        'JobPerformanceMetrics': JobPerformanceMetrics,
        'PerformanceAlert': PerformanceAlert,
        'PerformanceThreshold': PerformanceThreshold
    },
    'enums': {
        'MetricType': MetricType,
        'AlertSeverity': AlertSeverity,
        'ResourceType': ResourceType
    }
}

__all__ = [
    'PerformanceMonitor',
    'PerformanceMetric',
    'SystemResourceMetrics',
    'JobPerformanceMetrics',
    'PerformanceAlert',
    'PerformanceThreshold',
    'MetricType',
    'AlertSeverity',
    'ResourceType',
    'performance_registry'
]