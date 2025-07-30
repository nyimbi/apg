"""
Performance Monitor Module
==========================

Comprehensive performance monitoring and health checking for YouTube crawler
with real-time metrics, alerts, and optimization recommendations.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # e.g., "value > 100", "avg_5m > 50"
    threshold: float
    severity: str = "warning"
    enabled: bool = True
    cooldown_minutes: int = 5


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores various performance metrics."""
    
    def __init__(self, max_points_per_metric: int = 1000):
        self.max_points = max_points_per_metric
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points_per_metric))
        self.metric_types: Dict[str, MetricType] = {}
        self.lock = threading.Lock()
        
        # Built-in system metrics
        self.system_metrics_enabled = True
        self.system_metrics_interval = 5.0
        self.system_task = None
    
    def start_system_monitoring(self):
        """Start collecting system metrics."""
        if self.system_task is None:
            self.system_task = asyncio.create_task(self._collect_system_metrics())
    
    def stop_system_monitoring(self):
        """Stop collecting system metrics."""
        if self.system_task:
            self.system_task.cancel()
            self.system_task = None
    
    async def _collect_system_metrics(self):
        """Collect system metrics continuously."""
        while True:
            try:
                current_time = time.time()
                
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                self.record_gauge("system.cpu.usage_percent", cpu_percent, {"host": "local"})
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.record_gauge("system.memory.usage_percent", memory.percent, {"host": "local"})
                self.record_gauge("system.memory.available_mb", memory.available / 1024 / 1024, {"host": "local"})
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.record_gauge("system.disk.usage_percent", disk_percent, {"host": "local"})
                
                # Network metrics (if available)
                try:
                    net_io = psutil.net_io_counters()
                    self.record_counter("system.network.bytes_sent", net_io.bytes_sent, {"host": "local"})
                    self.record_counter("system.network.bytes_recv", net_io.bytes_recv, {"host": "local"})
                except:
                    pass  # Network metrics might not be available
                
                await asyncio.sleep(self.system_metrics_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(self.system_metrics_interval)
    
    def record_counter(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonically increasing)."""
        self._record_metric(name, MetricType.COUNTER, value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric (point-in-time value)."""
        self._record_metric(name, MetricType.GAUGE, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric (for calculating percentiles)."""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration in seconds)."""
        self._record_metric(name, MetricType.TIMER, duration, labels)
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, 
                      labels: Optional[Dict[str, str]] = None):
        """Internal method to record a metric."""
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            )
            
            self.metrics[name].append(point)
            self.metric_types[name] = metric_type
    
    def get_metric_values(self, name: str, time_window: Optional[float] = None) -> List[float]:
        """Get values for a metric within a time window."""
        with self.lock:
            points = list(self.metrics.get(name, []))
        
        if time_window:
            cutoff_time = time.time() - time_window
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        return [p.value for p in points]
    
    def get_metric_summary(self, name: str, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get statistical summary of a metric."""
        values = self.get_metric_values(name, time_window)
        
        if not values:
            return {"count": 0}
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "percentile_95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            "percentile_99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            result = {}
            for name, points in self.metrics.items():
                if points:
                    latest_value = points[-1].value
                    summary = self.get_metric_summary(name, 300)  # Last 5 minutes
                    result[name] = {
                        "type": self.metric_types[name].value,
                        "latest_value": latest_value,
                        "points_count": len(points),
                        "summary_5m": summary
                    }
            return result
    
    def clear_metrics(self, name: Optional[str] = None):
        """Clear metrics (all or specific metric)."""
        with self.lock:
            if name:
                self.metrics[name].clear()
            else:
                self.metrics.clear()
                self.metric_types.clear()


class HealthChecker:
    """Performs health checks on various components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.check_interval = 30.0
        self.background_task = None
        self.running = False
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func
    
    def unregister_check(self, name: str):
        """Unregister a health check."""
        self.checks.pop(name, None)
        self.results.pop(name, None)
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="Check not found",
                timestamp=time.time(),
                duration=0.0
            )
        
        start_time = time.time()
        
        try:
            check_func = self.checks[name]
            
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            # Parse result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                message = "OK" if result else "Failed"
                metadata = {}
            elif isinstance(result, dict):
                status = HealthStatus[result.get('status', 'UNKNOWN').upper()]
                message = result.get('message', 'No message')
                metadata = result.get('metadata', {})
            else:
                status = HealthStatus.HEALTHY
                message = str(result)
                metadata = {}
            
            duration = time.time() - start_time
            
            health_result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                timestamp=time.time(),
                duration=duration,
                metadata=metadata
            )
            
            self.results[name] = health_result
            return health_result
            
        except Exception as e:
            duration = time.time() - start_time
            health_result = HealthCheckResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Check failed: {str(e)}",
                timestamp=time.time(),
                duration=duration
            )
            
            self.results[name] = health_result
            return health_result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        tasks = []
        for name in self.checks:
            task = asyncio.create_task(self.run_check(name))
            tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check execution failed: {str(e)}",
                    timestamp=time.time(),
                    duration=0.0
                )
        
        return results
    
    def start_background_checks(self):
        """Start running health checks in the background."""
        if not self.running:
            self.running = True
            self.background_task = asyncio.create_task(self._background_checker())
    
    def stop_background_checks(self):
        """Stop background health checks."""
        if self.running:
            self.running = False
            if self.background_task:
                self.background_task.cancel()
    
    async def _background_checker(self):
        """Background task to run health checks periodically."""
        while self.running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background health checker: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall health status based on all checks."""
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all health check results."""
        overall_status = self.get_overall_health()
        
        status_counts = defaultdict(int)
        for result in self.results.values():
            status_counts[result.status.value] += 1
        
        return {
            "overall_status": overall_status.value,
            "total_checks": len(self.results),
            "status_breakdown": dict(status_counts),
            "last_check_time": max((r.timestamp for r in self.results.values()), default=0),
            "checks": {name: {
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "timestamp": result.timestamp
            } for name, result in self.results.items()}
        }


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.operation_timers: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Background tasks
        self.alert_task = None
        self.running = False
    
    def start(self):
        """Start performance monitoring."""
        if not self.running:
            self.running = True
            self.metrics.start_system_monitoring()
            self.health_checker.start_background_checks()
            self.alert_task = asyncio.create_task(self._alert_processor())
            logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        if self.running:
            self.running = False
            self.metrics.stop_system_monitoring()
            self.health_checker.stop_background_checks()
            if self.alert_task:
                self.alert_task.cancel()
            logger.info("Performance monitoring stopped")
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        
        def check_cpu_usage():
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                return {"status": "critical", "message": f"CPU usage critical: {cpu_percent:.1f}%"}
            elif cpu_percent > 75:
                return {"status": "warning", "message": f"CPU usage high: {cpu_percent:.1f}%"}
            else:
                return {"status": "healthy", "message": f"CPU usage normal: {cpu_percent:.1f}%"}
        
        def check_memory_usage():
            memory = psutil.virtual_memory()
            if memory.percent > 95:
                return {"status": "critical", "message": f"Memory usage critical: {memory.percent:.1f}%"}
            elif memory.percent > 85:
                return {"status": "warning", "message": f"Memory usage high: {memory.percent:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Memory usage normal: {memory.percent:.1f}%"}
        
        def check_disk_space():
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            if percent_used > 95:
                return {"status": "critical", "message": f"Disk space critical: {percent_used:.1f}%"}
            elif percent_used > 85:
                return {"status": "warning", "message": f"Disk space high: {percent_used:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Disk space normal: {percent_used:.1f}%"}
        
        self.health_checker.register_check("cpu_usage", check_cpu_usage)
        self.health_checker.register_check("memory_usage", check_memory_usage)
        self.health_checker.register_check("disk_space", check_disk_space)
    
    # Timer context manager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, duration: float):
        """Record operation timing."""
        self.operation_timers[operation_name].append(duration)
        self.metrics.record_timer(f"operation.{operation_name}.duration", duration)
        
        # Keep only recent timings
        if len(self.operation_timers[operation_name]) > 1000:
            self.operation_timers[operation_name] = self.operation_timers[operation_name][-500:]
    
    def record_error(self, error_type: str, details: Optional[str] = None):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.metrics.record_counter(f"errors.{error_type}.count", 1)
        
        if details:
            logger.error(f"Error recorded - {error_type}: {details}")
    
    def record_api_request(self, endpoint: str, duration: float, status_code: int, success: bool):
        """Record API request metrics."""
        labels = {
            "endpoint": endpoint,
            "status_code": str(status_code),
            "success": str(success).lower()
        }
        
        self.metrics.record_timer("api.request.duration", duration, labels)
        self.metrics.record_counter("api.request.count", 1, labels)
        
        if not success:
            self.record_error("api_request_failed", f"{endpoint} returned {status_code}")
    
    def record_cache_operation(self, operation: str, hit: bool, duration: float):
        """Record cache operation metrics."""
        labels = {
            "operation": operation,
            "result": "hit" if hit else "miss"
        }
        
        self.metrics.record_timer("cache.operation.duration", duration, labels)
        self.metrics.record_counter("cache.operation.count", 1, labels)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
    
    async def _alert_processor(self):
        """Process alert rules continuously."""
        while self.running:
            try:
                await self._check_alert_rules()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert processor: {e}")
                await asyncio.sleep(10)
    
    async def _check_alert_rules(self):
        """Check all alert rules and trigger alerts."""
        current_time = time.time()
        
        for rule in self.alert_rules:
            if not rule.enabled:
                continue
            
            # Check if rule is in cooldown
            if rule.name in self.active_alerts:
                last_alert_time = self.active_alerts[rule.name].get('timestamp', 0)
                if current_time - last_alert_time < rule.cooldown_minutes * 60:
                    continue
            
            # Evaluate rule condition
            try:
                triggered = self._evaluate_alert_condition(rule)
                
                if triggered:
                    alert = {
                        "rule_name": rule.name,
                        "metric_name": rule.metric_name,
                        "condition": rule.condition,
                        "threshold": rule.threshold,
                        "severity": rule.severity,
                        "timestamp": current_time,
                        "message": f"Alert: {rule.name} - {rule.condition}"
                    }
                    
                    self.active_alerts[rule.name] = alert
                    self.alert_history.append(alert)
                    
                    logger.warning(f"ALERT TRIGGERED: {alert['message']}")
                
                elif rule.name in self.active_alerts:
                    # Alert resolved
                    resolved_alert = self.active_alerts.pop(rule.name)
                    resolved_alert['resolved_at'] = current_time
                    logger.info(f"ALERT RESOLVED: {resolved_alert['rule_name']}")
            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _evaluate_alert_condition(self, rule: AlertRule) -> bool:
        """Evaluate an alert rule condition."""
        # Get metric values
        if "avg_5m" in rule.condition:
            values = self.metrics.get_metric_values(rule.metric_name, 300)  # 5 minutes
            if not values:
                return False
            current_value = statistics.mean(values)
        elif "max_5m" in rule.condition:
            values = self.metrics.get_metric_values(rule.metric_name, 300)
            if not values:
                return False
            current_value = max(values)
        else:
            # Use latest value
            values = self.metrics.get_metric_values(rule.metric_name)
            if not values:
                return False
            current_value = values[-1]
        
        # Simple condition evaluation
        if ">" in rule.condition:
            return current_value > rule.threshold
        elif "<" in rule.condition:
            return current_value < rule.threshold
        elif "==" in rule.condition:
            return abs(current_value - rule.threshold) < 0.001
        
        return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # Operation performance
        operation_stats = {}
        for op_name, timings in self.operation_timers.items():
            if timings:
                operation_stats[op_name] = {
                    "count": len(timings),
                    "avg_duration": statistics.mean(timings),
                    "min_duration": min(timings),
                    "max_duration": max(timings),
                    "p95_duration": statistics.quantiles(timings, n=20)[18] if len(timings) >= 20 else max(timings)
                }
        
        return {
            "timestamp": time.time(),
            "system_metrics": self.metrics.get_all_metrics(),
            "health_status": self.health_checker.get_health_summary(),
            "operation_performance": operation_stats,
            "error_counts": dict(self.error_counts),
            "active_alerts": list(self.active_alerts.values()),
            "recent_alerts": list(self.alert_history)[-10:] if self.alert_history else []
        }
    
    def get_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Check system metrics
        cpu_values = self.metrics.get_metric_values("system.cpu.usage_percent", 300)
        if cpu_values and statistics.mean(cpu_values) > 80:
            recommendations.append("High CPU usage detected. Consider reducing concurrent operations.")
        
        memory_values = self.metrics.get_metric_values("system.memory.usage_percent", 300)
        if memory_values and statistics.mean(memory_values) > 85:
            recommendations.append("High memory usage detected. Consider implementing cache size limits.")
        
        # Check operation performance
        for op_name, timings in self.operation_timers.items():
            if len(timings) >= 10:
                avg_time = statistics.mean(timings[-10:])
                if avg_time > 5.0:  # More than 5 seconds
                    recommendations.append(f"Operation '{op_name}' is slow (avg: {avg_time:.2f}s). Consider optimization.")
        
        # Check error rates
        for error_type, count in self.error_counts.items():
            if count > 10:  # More than 10 errors
                recommendations.append(f"High error rate for '{error_type}': {count} errors. Investigation needed.")
        
        return recommendations
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        data = self.get_performance_summary()
        
        if format_type == "json":
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_operation_time(self.operation_name, duration)
            
            # Record error if exception occurred
            if exc_type:
                self.monitor.record_error(f"{self.operation_name}_exception", str(exc_val))


# Utility functions
def create_default_monitor() -> PerformanceMonitor:
    """Create a performance monitor with default configuration."""
    monitor = PerformanceMonitor()
    
    # Add some default alert rules
    monitor.add_alert_rule(AlertRule(
        name="high_cpu_usage",
        metric_name="system.cpu.usage_percent",
        condition="avg_5m > threshold",
        threshold=85.0,
        severity="warning"
    ))
    
    monitor.add_alert_rule(AlertRule(
        name="high_memory_usage",
        metric_name="system.memory.usage_percent",
        condition="avg_5m > threshold",
        threshold=90.0,
        severity="critical"
    ))
    
    return monitor


__all__ = [
    'PerformanceMonitor',
    'MetricsCollector',
    'HealthChecker',
    'OperationTimer',
    'MetricType',
    'HealthStatus',
    'AlertRule',
    'HealthCheckResult',
    'create_default_monitor'
]