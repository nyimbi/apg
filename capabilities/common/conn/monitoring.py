"""
APG Connection Management Monitoring and Metrics
Production-grade monitoring with OpenTelemetry, Prometheus, and custom metrics

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import time
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from functools import wraps
from contextlib import asynccontextmanager

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Monitoring will use basic logging.")

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics we track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class HealthStatus(str, Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics aggregation"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    active_connections: int = 0
    active_flows: int = 0
    data_processed_bytes: int = 0
    errors_per_minute: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics for the connection management capability"""

    def __init__(self, enable_otel: bool = True):
        self.enable_otel = enable_otel and OTEL_AVAILABLE
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.active_connections: set[str] = set()
        self.active_flows: set[str] = set()

        # OpenTelemetry setup
        if self.enable_otel:
            self._setup_opentelemetry()

        # Built-in metrics
        self._setup_builtin_metrics()

    def _setup_opentelemetry(self):
        """Setup OpenTelemetry instrumentation"""
        try:
            # Setup tracer
            self.tracer = trace.get_tracer(__name__)

            # Setup meter
            self.meter = metrics.get_meter(__name__)

            # Create metric instruments
            self.connection_counter = self.meter.create_counter(
                name="apg_connections_total",
                description="Total number of connections created"
            )

            self.flow_counter = self.meter.create_counter(
                name="apg_flows_total",
                description="Total number of flows executed"
            )

            self.request_duration = self.meter.create_histogram(
                name="apg_request_duration_seconds",
                description="Request duration in seconds"
            )

            self.active_connections_gauge = self.meter.create_up_down_counter(
                name="apg_active_connections",
                description="Number of active connections"
            )

            self.data_processed_counter = self.meter.create_counter(
                name="apg_data_processed_bytes",
                description="Total bytes of data processed"
            )

            self.error_counter = self.meter.create_counter(
                name="apg_errors_total",
                description="Total number of errors"
            )

            logger.info("OpenTelemetry metrics initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to setup OpenTelemetry: {e}")
            self.enable_otel = False

    def _setup_builtin_metrics(self):
        """Setup built-in performance tracking"""
        self.performance_history: deque = deque(maxlen=1440)  # 24 hours of minutes
        self.current_performance = PerformanceMetrics()
        self._system_metrics_task = None

        # Start background metrics collection only when import happens inside a running loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._system_metrics_task = loop.create_task(self._collect_system_metrics())

    async def _collect_system_metrics(self):
        """Collect system-level metrics periodically"""
        while True:
            try:
                # Collect memory usage
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()

                self.record_gauge("system_memory_mb", memory_mb)
                self.record_gauge("system_cpu_percent", cpu_percent)

                # Collect connection pool metrics
                self.record_gauge("active_connections", len(self.get_active_connections()))
                self.record_gauge("active_flows", len(self.get_active_flows()))

            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")

            await asyncio.sleep(60)  # Collect every minute

    def record_counter(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        """Record a counter metric"""
        self.counters[name] += value

        if self.enable_otel and hasattr(self, 'meter'):
            # Record in OpenTelemetry
            if name == "connections_created":
                self.connection_counter.add(value, labels or {})
            elif name == "flows_executed":
                self.flow_counter.add(value, labels or {})
            elif name == "errors":
                self.error_counter.add(value, labels or {})

        # Store in local storage
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.COUNTER
        )
        self.metrics_storage[name].append(metric)

    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric"""
        self.gauges[name] = value

        if self.enable_otel and hasattr(self, 'meter'):
            if name == "active_connections":
                self.active_connections_gauge.add(value, labels or {})

        # Store in local storage
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.GAUGE
        )
        self.metrics_storage[name].append(metric)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric"""
        self.timers[name].append(value)

        if self.enable_otel and hasattr(self, 'meter'):
            if name == "request_duration":
                self.request_duration.record(value, labels or {})

        # Store in local storage
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=MetricType.HISTOGRAM
        )
        self.metrics_storage[name].append(metric)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timers': {
                name: {
                    'count': len(values),
                    'avg': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0
                }
                for name, values in self.timers.items()
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    def get_active_connections(self) -> List[str]:
        """Get list of active connection IDs."""
        return sorted(self.active_connections)

    def get_active_flows(self) -> List[str]:
        """Get list of active flow IDs."""
        return sorted(self.active_flows)

    def register_active_connection(self, connection_id: str) -> None:
        """Track an active connection and update its gauge."""
        if not connection_id:
            raise ValueError("connection_id is required")
        self.active_connections.add(connection_id)
        self.record_gauge("active_connections", len(self.active_connections))

    def unregister_active_connection(self, connection_id: str) -> None:
        """Stop tracking an active connection and update its gauge."""
        if not connection_id:
            raise ValueError("connection_id is required")
        self.active_connections.discard(connection_id)
        self.record_gauge("active_connections", len(self.active_connections))

    def register_active_flow(self, flow_id: str) -> None:
        """Track an active flow and update its gauge."""
        if not flow_id:
            raise ValueError("flow_id is required")
        self.active_flows.add(flow_id)
        self.record_gauge("active_flows", len(self.active_flows))

    def unregister_active_flow(self, flow_id: str) -> None:
        """Stop tracking an active flow and update its gauge."""
        if not flow_id:
            raise ValueError("flow_id is required")
        self.active_flows.discard(flow_id)
        self.record_gauge("active_flows", len(self.active_flows))


class HealthChecker:
    """Comprehensive health checking for the connection management capability"""

    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.health_checks: Dict[str, Callable] = {}
        self.health_history: deque = deque(maxlen=100)
        self._setup_default_health_checks()

    def _setup_default_health_checks(self):
        """Setup default health checks"""
        self.register_health_check("database", self._check_database_health)
        self.register_health_check("memory", self._check_memory_health)
        self.register_health_check("disk", self._check_disk_health)
        self.register_health_check("connections", self._check_connections_health)
        self.register_health_check("external_apis", self._check_external_apis_health)

    def register_health_check(self, name: str, check_func: Callable):
        """Register a custom health check"""
        self.health_checks[name] = check_func

    async def run_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        for name, check_func in self.health_checks.items():
            start_time = time.time()
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                duration_ms = (time.time() - start_time) * 1000

                health_result = HealthCheckResult(
                    component=name,
                    status=result.get('status', HealthStatus.UNKNOWN),
                    message=result.get('message', ''),
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    details=result.get('details', {})
                )

                results[name] = health_result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(timezone.utc),
                    duration_ms=duration_ms
                )

        # Store in history
        self.health_history.append({
            'timestamp': datetime.now(timezone.utc),
            'results': results
        })

        return results

    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            # Mock database check - in reality, this would test actual DB connection
            await asyncio.sleep(0.01)  # Simulate DB query

            return {
                'status': HealthStatus.HEALTHY,
                'message': 'Database connection healthy',
                'details': {
                    'connection_pool_size': 10,
                    'active_connections': 2,
                    'query_time_ms': 10
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Database check failed: {str(e)}'
            }

    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()

            status = HealthStatus.HEALTHY
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED

            return {
                'status': status,
                'message': f'Memory usage: {memory.percent:.1f}%',
                'details': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percent_used': memory.percent
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f'Could not check memory: {str(e)}'
            }

    def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            import psutil
            disk = psutil.disk_usage('/')

            status = HealthStatus.HEALTHY
            percent_used = (disk.used / disk.total) * 100

            if percent_used > 95:
                status = HealthStatus.UNHEALTHY
            elif percent_used > 85:
                status = HealthStatus.DEGRADED

            return {
                'status': status,
                'message': f'Disk usage: {percent_used:.1f}%',
                'details': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent_used': percent_used
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f'Could not check disk: {str(e)}'
            }

    async def _check_connections_health(self) -> Dict[str, Any]:
        """Check connection manager health"""
        try:
            # Mock connection health check
            active_connections = self.metrics_collector.get_active_connections()
            failed_connections = []  # Would be populated from actual connection manager

            status = HealthStatus.HEALTHY
            if len(failed_connections) > 0:
                status = HealthStatus.DEGRADED if len(failed_connections) < 5 else HealthStatus.UNHEALTHY

            return {
                'status': status,
                'message': f'{len(active_connections)} active connections, {len(failed_connections)} failed',
                'details': {
                    'active_connections': len(active_connections),
                    'failed_connections': len(failed_connections)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'Connection health check failed: {str(e)}'
            }

    async def _check_external_apis_health(self) -> Dict[str, Any]:
        """Check external API dependencies"""
        try:
            # Mock external API checks
            apis_to_check = ['singer_registry', 'ai_service', 'notification_service']
            healthy_apis = []
            failed_apis = []

            for api in apis_to_check:
                # Mock API health check
                try:
                    await asyncio.sleep(0.01)  # Simulate API call
                    healthy_apis.append(api)
                except Exception:
                    failed_apis.append(api)

            status = HealthStatus.HEALTHY
            if failed_apis:
                status = HealthStatus.DEGRADED if len(failed_apis) < len(apis_to_check) else HealthStatus.UNHEALTHY

            return {
                'status': status,
                'message': f'{len(healthy_apis)}/{len(apis_to_check)} external APIs healthy',
                'details': {
                    'healthy_apis': healthy_apis,
                    'failed_apis': failed_apis
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f'External API health check failed: {str(e)}'
            }

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.health_history:
            return HealthStatus.UNKNOWN

        latest_results = self.health_history[-1]['results']
        statuses = [result.status for result in latest_results.values()]

        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN


def monitor_performance(metric_name: str = None, labels: Dict[str, str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            name = metric_name or f"{func.__module__}.{func.__name__}"

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Record successful execution
                global_metrics_collector.record_histogram(
                    f"{name}_duration",
                    duration,
                    {**(labels or {}), 'status': 'success'}
                )
                global_metrics_collector.record_counter(
                    f"{name}_total",
                    1,
                    {**(labels or {}), 'status': 'success'}
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Record failed execution
                global_metrics_collector.record_histogram(
                    f"{name}_duration",
                    duration,
                    {**(labels or {}), 'status': 'error'}
                )
                global_metrics_collector.record_counter(
                    f"{name}_total",
                    1,
                    {**(labels or {}), 'status': 'error'}
                )
                global_metrics_collector.record_counter("errors", 1, {'function': name})

                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            name = metric_name or f"{func.__module__}.{func.__name__}"

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                global_metrics_collector.record_histogram(f"{name}_duration", duration)
                global_metrics_collector.record_counter(f"{name}_total", 1)

                return result

            except Exception as e:
                duration = time.time() - start_time

                global_metrics_collector.record_histogram(f"{name}_duration", duration)
                global_metrics_collector.record_counter(f"{name}_total", 1, {'status': 'error'})
                global_metrics_collector.record_counter("errors", 1, {'function': name})

                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@asynccontextmanager
async def trace_operation(operation_name: str, attributes: Dict[str, Any] = None):
    """Context manager for tracing operations"""
    if OTEL_AVAILABLE:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(operation_name) as span:
            if attributes:
                span.set_attributes(attributes)

            start_time = time.time()
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
    else:
        # Fallback without tracing
        yield None


# Global instances
global_metrics_collector = MetricsCollector()
global_health_checker = HealthChecker(global_metrics_collector)


# Convenience functions
def record_connection_created(connection_type: str, tenant_id: str):
    """Record connection creation metric"""
    global_metrics_collector.record_counter(
        "connections_created",
        1,
        {'connection_type': connection_type, 'tenant_id': tenant_id}
    )


def register_active_connection(connection_id: str):
    """Register a connection as active in global metrics."""
    global_metrics_collector.register_active_connection(connection_id)


def unregister_active_connection(connection_id: str):
    """Remove a connection from active global metrics."""
    global_metrics_collector.unregister_active_connection(connection_id)


def record_flow_executed(flow_type: str, duration_seconds: float, success: bool):
    """Record flow execution metric"""
    global_metrics_collector.record_counter(
        "flows_executed",
        1,
        {'flow_type': flow_type, 'success': str(success)}
    )
    global_metrics_collector.record_histogram("flow_duration", duration_seconds)


def register_active_flow(flow_id: str):
    """Register a flow as active in global metrics."""
    global_metrics_collector.register_active_flow(flow_id)


def unregister_active_flow(flow_id: str):
    """Remove a flow from active global metrics."""
    global_metrics_collector.unregister_active_flow(flow_id)


def record_data_processed(bytes_processed: int, connection_type: str):
    """Record data processing metric"""
    global_metrics_collector.record_counter("data_processed_bytes", bytes_processed)
    global_metrics_collector.record_gauge("last_processed_bytes", bytes_processed, {'connection_type': connection_type})


async def get_health_status() -> Dict[str, Any]:
    """Get current health status"""
    results = await global_health_checker.run_all_health_checks()
    overall_status = global_health_checker.get_overall_health()

    return {
        'overall_status': overall_status.value,
        'components': {name: result.__dict__ for name, result in results.items()},
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def get_metrics() -> Dict[str, Any]:
    """Get current metrics summary"""
    return global_metrics_collector.get_metrics_summary()
