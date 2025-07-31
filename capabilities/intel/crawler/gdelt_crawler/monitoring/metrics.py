"""
GDELT Metrics Collection and Performance Monitoring
===================================================

Comprehensive metrics collection system for GDELT crawling operations
with performance monitoring, data quality tracking, and real-time analytics.

Key Features:
- **Performance Metrics**: Processing rates, response times, and throughput
- **Data Quality Metrics**: Validation rates, error counts, and data completeness
- **System Metrics**: Resource utilization and health indicators
- **Real-time Analytics**: Live dashboards and monitoring capabilities
- **Historical Tracking**: Time-series data for trend analysis
- **Custom Metrics**: Extensible framework for custom measurements

Metric Categories:
- **Processing Metrics**: ETL performance and throughput
- **Content Metrics**: Data quality and validation results
- **System Metrics**: Health and resource utilization
- **Business Metrics**: Conflict detection and event analysis

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import time
import psutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import threading
from statistics import mean, median

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    last_value: float
    last_updated: datetime


class MetricsCollector:
    """Base metrics collector with time-series storage."""
    
    def __init__(self, max_points: int = 10000):
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.RLock()
    
    def record(
        self,
        name: str,
        value: Union[int, float],
        tags: Dict[str, str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Record a metric data point."""
        with self.lock:
            point = MetricPoint(
                timestamp=datetime.now(timezone.utc),
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )
            self.metrics[name].append(point)
    
    def get_metric_summary(self, name: str, hours_back: int = 24) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self.lock:
            if name not in self.metrics:
                return None
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            points = [p for p in self.metrics[name] if p.timestamp >= cutoff_time]
            
            if not points:
                return None
            
            values = [p.value for p in points]
            
            return MetricSummary(
                name=name,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                mean_value=mean(values),
                median_value=median(values),
                std_dev=self._calculate_std_dev(values),
                last_value=values[-1],
                last_updated=points[-1].timestamp
            )
    
    def get_recent_points(self, name: str, limit: int = 100) -> List[MetricPoint]:
        """Get recent metric points."""
        with self.lock:
            if name not in self.metrics:
                return []
            return list(self.metrics[name])[-limit:]
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean_val = mean(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5


class GDELTMetrics:
    """
    Comprehensive metrics collection system for GDELT operations.
    
    Tracks performance, data quality, system health, and business metrics
    with real-time analytics and historical tracking capabilities.
    """
    
    def __init__(self):
        self.collector = MetricsCollector()
        self.start_time = datetime.now(timezone.utc)
        
        # Metric counters
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # System monitoring
        self._system_monitor_interval = 60  # seconds
        self._system_monitor_task: Optional[asyncio.Task] = None
        self._monitoring_active = False
        
        # Custom metric handlers
        self.custom_handlers: List[Callable[[str, Any], None]] = []
    
    async def start_monitoring(self):
        """Start background system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._system_monitor_task = asyncio.create_task(self._system_monitor_loop())
        logger.info("GDELT metrics monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._system_monitor_task:
            self._system_monitor_task.cancel()
            try:
                await self._system_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("GDELT metrics monitoring stopped")
    
    # ==================== ETL METRICS ====================
    
    def record_etl_completion(self, date: datetime, processed_counts: Dict[str, int]):
        """Record ETL completion metrics."""
        total_processed = sum(processed_counts.values())
        
        # Record overall metrics
        self.collector.record("etl.total_processed", total_processed)
        self.collector.record("etl.completion_rate", 1.0)  # Successful completion
        
        # Record per-dataset metrics
        for dataset, count in processed_counts.items():
            self.collector.record(f"etl.{dataset}_processed", count)
            self.counters[f"etl_{dataset}_total"] += count
        
        # Update counters
        self.counters["etl_runs_completed"] += 1
        self.counters["etl_total_processed"] += total_processed
        
        logger.info(f"ETL metrics recorded: {total_processed} records processed")
    
    def record_etl_failure(self, date: datetime, error: str):
        """Record ETL failure metrics."""
        self.collector.record("etl.failure_rate", 1.0)
        self.counters["etl_runs_failed"] += 1
        
        # Record error by type
        error_type = self._classify_error(error)
        self.collector.record(f"etl.errors.{error_type}", 1.0)
        
        logger.warning(f"ETL failure recorded for {date.date()}: {error}")
    
    def record_processing_time(self, operation: str, duration_seconds: float):
        """Record processing time metrics."""
        self.collector.record(f"processing.{operation}_duration", duration_seconds)
        self.timers[f"processing_{operation}"].append(duration_seconds)
        
        # Record performance categories
        if duration_seconds < 1.0:
            self.collector.record(f"processing.{operation}_fast", 1.0)
        elif duration_seconds < 10.0:
            self.collector.record(f"processing.{operation}_normal", 1.0)
        else:
            self.collector.record(f"processing.{operation}_slow", 1.0)
    
    # ==================== DATA QUALITY METRICS ====================
    
    def record_data_quality(self, dataset: str, stats: Dict[str, Any]):
        """Record data quality metrics."""
        # Validation rates
        total_records = stats.get('total_records', 0)
        valid_records = stats.get('valid_records', 0)
        
        if total_records > 0:
            validation_rate = valid_records / total_records
            self.collector.record(f"quality.{dataset}_validation_rate", validation_rate)
        
        # Error rates
        invalid_records = stats.get('invalid_records', 0)
        if total_records > 0:
            error_rate = invalid_records / total_records
            self.collector.record(f"quality.{dataset}_error_rate", error_rate)
        
        # Record counts
        self.collector.record(f"quality.{dataset}_total_records", total_records)
        self.collector.record(f"quality.{dataset}_valid_records", valid_records)
        self.collector.record(f"quality.{dataset}_invalid_records", invalid_records)
        
        # Update counters
        self.counters[f"quality_{dataset}_total"] += total_records
        self.counters[f"quality_{dataset}_valid"] += valid_records
        self.counters[f"quality_{dataset}_invalid"] += invalid_records
    
    def record_ml_processing(self, processed_count: int, high_confidence_count: int):
        """Record ML processing metrics."""
        self.collector.record("ml.records_processed", processed_count)
        self.collector.record("ml.high_confidence_records", high_confidence_count)
        
        if processed_count > 0:
            confidence_rate = high_confidence_count / processed_count
            self.collector.record("ml.high_confidence_rate", confidence_rate)
        
        # Update counters
        self.counters["ml_total_processed"] += processed_count
        self.counters["ml_high_confidence"] += high_confidence_count
    
    # ==================== CONFLICT METRICS ====================
    
    def record_conflict_detection(self, total_events: int, conflict_events: int, high_severity: int):
        """Record conflict detection metrics."""
        self.collector.record("conflicts.total_events", total_events)
        self.collector.record("conflicts.conflict_events", conflict_events)
        self.collector.record("conflicts.high_severity", high_severity)
        
        if total_events > 0:
            conflict_rate = conflict_events / total_events
            severity_rate = high_severity / total_events
            
            self.collector.record("conflicts.conflict_rate", conflict_rate)
            self.collector.record("conflicts.severity_rate", severity_rate)
        
        # Update counters
        self.counters["conflicts_total_events"] += total_events
        self.counters["conflicts_detected"] += conflict_events
        self.counters["conflicts_high_severity"] += high_severity
    
    def record_fatality_statistics(self, total_fatalities: int, events_with_fatalities: int):
        """Record fatality statistics."""
        self.collector.record("conflicts.total_fatalities", total_fatalities)
        self.collector.record("conflicts.events_with_fatalities", events_with_fatalities)
        
        if events_with_fatalities > 0:
            avg_fatalities = total_fatalities / events_with_fatalities
            self.collector.record("conflicts.avg_fatalities_per_event", avg_fatalities)
        
        # Update counters
        self.counters["fatalities_total"] += total_fatalities
        self.counters["fatal_events"] += events_with_fatalities
    
    # ==================== SYSTEM METRICS ====================
    
    async def _system_monitor_loop(self):
        """Background system monitoring loop."""
        while self._monitoring_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self._system_monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self._system_monitor_interval)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.collector.record("system.cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.collector.record("system.memory_percent", memory.percent)
            self.collector.record("system.memory_available_gb", memory.available / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.collector.record("system.disk_percent", disk.percent)
            self.collector.record("system.disk_free_gb", disk.free / (1024**3))
            
            # Network metrics (if available)
            net_io = psutil.net_io_counters()
            if net_io:
                self.collector.record("system.network_bytes_sent", net_io.bytes_sent)
                self.collector.record("system.network_bytes_recv", net_io.bytes_recv)
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            self.collector.record("process.memory_rss_mb", process_memory.rss / (1024**2))
            self.collector.record("process.memory_vms_mb", process_memory.vms / (1024**2))
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    # ==================== DOWNLOAD METRICS ====================
    
    def record_download_metrics(self, dataset: str, file_size_bytes: int, download_time: float):
        """Record file download metrics."""
        self.collector.record(f"downloads.{dataset}_size_mb", file_size_bytes / (1024**2))
        self.collector.record(f"downloads.{dataset}_time_seconds", download_time)
        
        if download_time > 0:
            speed_mbps = (file_size_bytes / (1024**2)) / download_time
            self.collector.record(f"downloads.{dataset}_speed_mbps", speed_mbps)
        
        # Update counters
        self.counters[f"downloads_{dataset}_count"] += 1
        self.counters[f"downloads_{dataset}_bytes"] += file_size_bytes
    
    def record_download_failure(self, dataset: str, error: str):
        """Record download failure metrics."""
        self.collector.record(f"downloads.{dataset}_failures", 1.0)
        self.counters[f"downloads_{dataset}_failures"] += 1
        
        # Classify error type
        error_type = self._classify_error(error)
        self.collector.record(f"downloads.errors.{error_type}", 1.0)
    
    # ==================== UTILITIES ====================
    
    def _classify_error(self, error: str) -> str:
        """Classify error into categories."""
        error_lower = error.lower()
        
        if any(term in error_lower for term in ['network', 'connection', 'timeout', 'dns']):
            return "network"
        elif any(term in error_lower for term in ['permission', 'access', 'auth']):
            return "permission"
        elif any(term in error_lower for term in ['disk', 'space', 'storage']):
            return "storage"
        elif any(term in error_lower for term in ['memory', 'out of memory']):
            return "memory"
        else:
            return "other"
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        self.counters[name] += value
        self.collector.record(f"counters.{name}", self.counters[name])
    
    def record_custom_metric(self, name: str, value: Union[int, float], **kwargs):
        """Record a custom metric."""
        self.collector.record(name, value, **kwargs)
        
        # Call custom handlers
        for handler in self.custom_handlers:
            try:
                handler(name, value)
            except Exception as e:
                logger.error(f"Custom metric handler error: {e}")
    
    def add_custom_handler(self, handler: Callable[[str, Any], None]):
        """Add a custom metric handler."""
        self.custom_handlers.append(handler)
    
    # ==================== REPORTING ====================
    
    async def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        
        summary = {
            "system": {
                "uptime_hours": uptime / 3600,
                "start_time": self.start_time.isoformat(),
                "monitoring_active": self._monitoring_active
            },
            "etl": {
                "runs_completed": self.counters.get("etl_runs_completed", 0),
                "runs_failed": self.counters.get("etl_runs_failed", 0),
                "total_processed": self.counters.get("etl_total_processed", 0),
                "success_rate": self._calculate_success_rate("etl_runs_completed", "etl_runs_failed")
            },
            "data_quality": {
                "total_records": sum(v for k, v in self.counters.items() if k.startswith("quality_") and k.endswith("_total")),
                "valid_records": sum(v for k, v in self.counters.items() if k.startswith("quality_") and k.endswith("_valid")),
                "invalid_records": sum(v for k, v in self.counters.items() if k.startswith("quality_") and k.endswith("_invalid"))
            },
            "ml_processing": {
                "total_processed": self.counters.get("ml_total_processed", 0),
                "high_confidence": self.counters.get("ml_high_confidence", 0)
            },
            "conflicts": {
                "total_events": self.counters.get("conflicts_total_events", 0),
                "conflicts_detected": self.counters.get("conflicts_detected", 0),
                "high_severity": self.counters.get("conflicts_high_severity", 0),
                "total_fatalities": self.counters.get("fatalities_total", 0),
                "fatal_events": self.counters.get("fatal_events", 0)
            },
            "downloads": self._get_download_summary(),
            "performance": self._get_performance_summary()
        }
        
        return summary
    
    def _calculate_success_rate(self, success_key: str, failure_key: str) -> float:
        """Calculate success rate percentage."""
        success = self.counters.get(success_key, 0)
        failure = self.counters.get(failure_key, 0)
        total = success + failure
        
        if total == 0:
            return 100.0
        
        return (success / total) * 100.0
    
    def _get_download_summary(self) -> Dict[str, Any]:
        """Get download metrics summary."""
        datasets = ['events', 'mentions', 'gkg']
        summary = {}
        
        for dataset in datasets:
            count_key = f"downloads_{dataset}_count"
            bytes_key = f"downloads_{dataset}_bytes"
            failures_key = f"downloads_{dataset}_failures"
            
            summary[dataset] = {
                "downloads": self.counters.get(count_key, 0),
                "total_bytes": self.counters.get(bytes_key, 0),
                "failures": self.counters.get(failures_key, 0)
            }
        
        return summary
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        performance = {}
        
        for operation, times in self.timers.items():
            if times:
                performance[operation] = {
                    "count": len(times),
                    "avg_seconds": mean(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times)
                }
        
        return performance
    
    def get_metric_history(self, metric_name: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get metric history for visualization."""
        points = self.collector.get_recent_points(metric_name, limit=1000)
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        filtered_points = [p for p in points if p.timestamp >= cutoff_time]
        
        return [
            {
                "timestamp": p.timestamp.isoformat(),
                "value": p.value,
                "tags": p.tags,
                "metadata": p.metadata
            }
            for p in filtered_points
        ]


# Factory function
def create_metrics_system() -> GDELTMetrics:
    """Create a GDELT metrics system."""
    return GDELTMetrics()


# Export components
__all__ = [
    'GDELTMetrics',
    'MetricsCollector',
    'MetricPoint',
    'MetricSummary',
    'create_metrics_system'
]