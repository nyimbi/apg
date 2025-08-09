"""
APG Capability Registry - Production Monitoring Setup

Comprehensive monitoring configuration for production deployments including
metrics collection, alerting, logging, and health checks.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

import psutil
import redis
import asyncpg
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, Info
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.exposition import MetricsHandler

# =============================================================================
# Monitoring Configuration
# =============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    enabled: bool = True
    port: int = 9090
    path: str = "/metrics"
    multiprocess_dir: str = "/tmp/prometheus_multiproc_dir"
    collect_interval: int = 15  # seconds
    
    # Database metrics
    db_enabled: bool = True
    db_slow_query_threshold: float = 1.0  # seconds
    
    # Redis metrics
    redis_enabled: bool = True
    
    # Application metrics
    app_metrics_enabled: bool = True
    
    # System metrics
    system_metrics_enabled: bool = True

@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str
    threshold: float
    duration: str
    severity: AlertSeverity
    description: str
    runbook_url: Optional[str] = None

# =============================================================================
# Custom Metrics Collector
# =============================================================================

class RegistryMetricsCollector:
    """Custom metrics collector for APG Capability Registry."""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        # Database connection info
        self.db_url = os.environ.get("DATABASE_URL")
        self.redis_url = os.environ.get("REDIS_URL")
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        # Application metrics
        self.capability_total = Counter(
            'registry_capabilities_total',
            'Total number of capabilities registered',
            ['category', 'status'],
            registry=self.registry
        )
        
        self.composition_total = Counter(
            'registry_compositions_total',
            'Total number of compositions created',
            ['type', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'registry_api_request_duration_seconds',
            'Time spent processing API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'registry_api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.active_users = Gauge(
            'registry_active_users',
            'Number of active users',
            registry=self.registry
        )
        
        self.websocket_connections = Gauge(
            'registry_websocket_connections',
            'Number of active WebSocket connections',
            registry=self.registry
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'registry_db_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'registry_db_query_duration_seconds',
            'Time spent executing database queries',
            ['operation'],
            registry=self.registry
        )
        
        self.db_slow_queries = Counter(
            'registry_db_slow_queries_total',
            'Number of slow database queries',
            ['operation'],
            registry=self.registry
        )
        
        # Redis metrics
        self.redis_operations = Counter(
            'registry_redis_operations_total',
            'Total Redis operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'registry_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'registry_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'registry_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        # Business metrics
        self.capability_search_requests = Counter(
            'registry_capability_searches_total',
            'Number of capability search requests',
            ['category'],
            registry=self.registry
        )
        
        self.composition_validations = Counter(
            'registry_composition_validations_total',
            'Number of composition validations',
            ['result'],
            registry=self.registry
        )
        
        # Info metrics
        self.app_info = Info(
            'registry_app_info',
            'Application information',
            registry=self.registry
        )
        
        self.app_info.info({
            'version': os.environ.get('APP_VERSION', '1.0.0'),
            'environment': os.environ.get('ENVIRONMENT', 'production'),
            'build_date': os.environ.get('BUILD_DATE', ''),
            'git_commit': os.environ.get('GIT_COMMIT', '')
        })
    
    async def collect_db_metrics(self):
        """Collect database-specific metrics."""
        if not self.config.db_enabled or not self.db_url:
            return
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Active connections
            result = await conn.fetchrow("SELECT count(*) as active FROM pg_stat_activity WHERE state = 'active'")
            if result:
                self.db_connections_active.set(result['active'])
            
            # Slow queries (from pg_stat_statements if available)
            try:
                slow_queries = await conn.fetch("""
                    SELECT query, calls, mean_time 
                    FROM pg_stat_statements 
                    WHERE mean_time > $1 
                    ORDER BY mean_time DESC 
                    LIMIT 10
                """, self.config.db_slow_query_threshold * 1000)  # Convert to ms
                
                for query in slow_queries:
                    self.db_slow_queries.labels(operation="unknown").inc()
            except:
                pass  # pg_stat_statements might not be available
            
            await conn.close()
            
        except Exception as e:
            print(f"Error collecting database metrics: {e}")
    
    async def collect_redis_metrics(self):
        """Collect Redis-specific metrics."""
        if not self.config.redis_enabled or not self.redis_url:
            return
        
        try:
            redis_client = redis.from_url(self.redis_url)
            info = redis_client.info()
            
            # Memory usage
            if 'used_memory' in info:
                # Redis memory is tracked separately from system memory
                pass
            
            # Hit rate
            if 'keyspace_hits' in info and 'keyspace_misses' in info:
                hits = info['keyspace_hits']
                misses = info['keyspace_misses']
                total = hits + misses
                if total > 0:
                    hit_rate = (hits / total) * 100
                    self.cache_hit_rate.set(hit_rate)
            
            redis_client.close()
            
        except Exception as e:
            print(f"Error collecting Redis metrics: {e}")
    
    def collect_system_metrics(self):
        """Collect system-level metrics."""
        if not self.config.system_metrics_enabled:
            return
        
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    async def collect_all_metrics(self):
        """Collect all metrics."""
        try:
            await self.collect_db_metrics()
            await self.collect_redis_metrics()
            self.collect_system_metrics()
        except Exception as e:
            print(f"Error in metrics collection: {e}")

# =============================================================================
# Health Check System
# =============================================================================

@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = None

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        self.redis_url = os.environ.get("REDIS_URL")
    
    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        start_time = time.time()
        
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Simple query to test connectivity
            await conn.fetchrow("SELECT 1")
            
            # Check for table existence
            tables = await conn.fetch("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('capabilities', 'compositions')
            """)
            
            response_time = (time.time() - start_time) * 1000
            
            await conn.close()
            
            if len(tables) >= 2:
                return HealthCheckResult(
                    name="database",
                    status="healthy",
                    message="Database connection successful, tables exist",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details={"tables_found": len(tables)}
                )
            else:
                return HealthCheckResult(
                    name="database",
                    status="degraded",
                    message="Database connected but missing tables",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details={"tables_found": len(tables)}
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="database",
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
    
    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance."""
        start_time = time.time()
        
        try:
            redis_client = redis.from_url(self.redis_url)
            
            # Test basic operations
            test_key = f"health_check_{int(time.time())}"
            redis_client.set(test_key, "test_value", ex=60)
            value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            info = redis_client.info()
            redis_client.close()
            
            if value == b"test_value":
                return HealthCheckResult(
                    name="redis",
                    status="healthy",
                    message="Redis connection and operations successful",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    details={
                        "redis_version": info.get("redis_version"),
                        "used_memory_human": info.get("used_memory_human")
                    }
                )
            else:
                return HealthCheckResult(
                    name="redis",
                    status="degraded",
                    message="Redis connected but operations failed",
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="redis",
                status="unhealthy",
                message=f"Redis connection failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        start_time = time.time()
        
        try:
            # Memory check
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # CPU check
            cpu_percent = psutil.cpu_percent(interval=1)
            
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on resource usage
            status = "healthy"
            issues = []
            
            if memory_usage_percent > 90:
                status = "degraded"
                issues.append(f"High memory usage: {memory_usage_percent:.1f}%")
            
            if disk_usage_percent > 85:
                status = "degraded" if status == "healthy" else "unhealthy"
                issues.append(f"High disk usage: {disk_usage_percent:.1f}%")
            
            if cpu_percent > 90:
                status = "degraded" if status == "healthy" else "unhealthy"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            message = "System resources within normal limits" if status == "healthy" else "; ".join(issues)
            
            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={
                    "memory_usage_percent": memory_usage_percent,
                    "disk_usage_percent": disk_usage_percent,
                    "cpu_usage_percent": cpu_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name="system_resources",
                status="unhealthy",
                message=f"System resource check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        # Run checks concurrently
        import asyncio
        
        db_check = self.check_database()
        redis_check = self.check_redis()
        
        # System check is synchronous, so run it separately
        system_result = self.check_system_resources()
        results["system_resources"] = system_result
        
        # Wait for async checks
        db_result, redis_result = await asyncio.gather(db_check, redis_check)
        
        results["database"] = db_result
        results["redis"] = redis_result
        
        return results
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> str:
        """Determine overall system status."""
        statuses = [result.status for result in results.values()]
        
        if all(status == "healthy" for status in statuses):
            return "healthy"
        elif any(status == "unhealthy" for status in statuses):
            return "unhealthy"
        else:
            return "degraded"

# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alert_rules = self._load_alert_rules()
        self.active_alerts = {}
    
    def _load_alert_rules(self) -> List[AlertRule]:
        """Load alert rules configuration."""
        return [
            AlertRule(
                name="HighErrorRate",
                condition="rate(registry_api_requests_total{status=~'5..'}[5m]) > 0.05",
                threshold=0.05,
                duration="2m",
                severity=AlertSeverity.CRITICAL,
                description="High error rate detected in API requests",
                runbook_url="https://docs.datacraft.co.ke/runbooks/high-error-rate"
            ),
            AlertRule(
                name="HighResponseTime",
                condition="histogram_quantile(0.95, rate(registry_api_request_duration_seconds_bucket[5m])) > 2",
                threshold=2.0,
                duration="5m",
                severity=AlertSeverity.WARNING,
                description="High API response time detected",
                runbook_url="https://docs.datacraft.co.ke/runbooks/high-response-time"
            ),
            AlertRule(
                name="DatabaseDown",
                condition="registry_db_connections_active == 0",
                threshold=0,
                duration="30s",
                severity=AlertSeverity.CRITICAL,
                description="Database connection lost",
                runbook_url="https://docs.datacraft.co.ke/runbooks/database-down"
            ),
            AlertRule(
                name="HighMemoryUsage",
                condition="registry_memory_usage_bytes / (1024^3) > 8",
                threshold=8.0,
                duration="5m",
                severity=AlertSeverity.WARNING,
                description="High memory usage detected",
                runbook_url="https://docs.datacraft.co.ke/runbooks/high-memory"
            ),
            AlertRule(
                name="LowCacheHitRate",
                condition="registry_cache_hit_rate < 70",
                threshold=70.0,
                duration="10m",
                severity=AlertSeverity.WARNING,
                description="Low cache hit rate detected",
                runbook_url="https://docs.datacraft.co.ke/runbooks/low-cache-hit"
            )
        ]
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            # This is a simplified alert checking logic
            # In production, you'd use Prometheus AlertManager
            if self._evaluate_condition(rule, metrics):
                alert = {
                    "rule": rule.name,
                    "severity": rule.severity.value,
                    "description": rule.description,
                    "timestamp": datetime.utcnow().isoformat(),
                    "runbook_url": rule.runbook_url
                }
                triggered_alerts.append(alert)
        
        return triggered_alerts
    
    def _evaluate_condition(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """Evaluate alert condition (simplified)."""
        # This is a very basic implementation
        # Real implementation would parse PromQL expressions
        
        if "HighErrorRate" in rule.name:
            error_rate = metrics.get("error_rate", 0)
            return error_rate > rule.threshold
        
        elif "HighResponseTime" in rule.name:
            response_time = metrics.get("p95_response_time", 0)
            return response_time > rule.threshold
        
        elif "DatabaseDown" in rule.name:
            db_connections = metrics.get("db_connections", 1)
            return db_connections <= rule.threshold
        
        elif "HighMemoryUsage" in rule.name:
            memory_gb = metrics.get("memory_usage_gb", 0)
            return memory_gb > rule.threshold
        
        elif "LowCacheHitRate" in rule.name:
            hit_rate = metrics.get("cache_hit_rate", 100)
            return hit_rate < rule.threshold
        
        return False

# =============================================================================
# Monitoring Dashboard Generator
# =============================================================================

def generate_grafana_dashboard() -> Dict[str, Any]:
    """Generate Grafana dashboard configuration."""
    return {
        "dashboard": {
            "id": None,
            "title": "APG Capability Registry",
            "tags": ["apg", "registry", "capabilities"],
            "timezone": "UTC",
            "refresh": "30s",
            "time": {
                "from": "now-1h",
                "to": "now"
            },
            "panels": [
                {
                    "id": 1,
                    "title": "API Request Rate",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "rate(registry_api_requests_total[5m])",
                            "legendFormat": "{{method}} {{endpoint}}"
                        }
                    ],
                    "yAxes": [
                        {
                            "label": "Requests/sec"
                        }
                    ]
                },
                {
                    "id": 2,
                    "title": "Response Time Percentiles",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.50, rate(registry_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.95, rate(registry_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile"
                        },
                        {
                            "expr": "histogram_quantile(0.99, rate(registry_api_request_duration_seconds_bucket[5m]))",
                            "legendFormat": "99th percentile"
                        }
                    ]
                },
                {
                    "id": 3,
                    "title": "Error Rate",
                    "type": "singlestat",
                    "targets": [
                        {
                            "expr": "rate(registry_api_requests_total{status=~'5..'}[5m]) / rate(registry_api_requests_total[5m])",
                            "legendFormat": "Error Rate"
                        }
                    ],
                    "thresholds": "0.01,0.05"
                },
                {
                    "id": 4,
                    "title": "Active Database Connections",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "registry_db_connections_active",
                            "legendFormat": "Active Connections"
                        }
                    ]
                },
                {
                    "id": 5,
                    "title": "Cache Hit Rate",
                    "type": "singlestat",
                    "targets": [
                        {
                            "expr": "registry_cache_hit_rate",
                            "legendFormat": "Hit Rate %"
                        }
                    ],
                    "thresholds": "70,90"
                },
                {
                    "id": 6,
                    "title": "System Resources",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "registry_cpu_usage_percent",
                            "legendFormat": "CPU %"
                        },
                        {
                            "expr": "registry_memory_usage_bytes / (1024^3)",
                            "legendFormat": "Memory GB"
                        }
                    ]
                },
                {
                    "id": 7,
                    "title": "Capabilities by Category",
                    "type": "piechart",
                    "targets": [
                        {
                            "expr": "registry_capabilities_total",
                            "legendFormat": "{{category}}"
                        }
                    ]
                },
                {
                    "id": 8,
                    "title": "Active WebSocket Connections",
                    "type": "graph",
                    "targets": [
                        {
                            "expr": "registry_websocket_connections",
                            "legendFormat": "WebSocket Connections"
                        }
                    ]
                }
            ]
        }
    }

def generate_prometheus_config() -> str:
    """Generate Prometheus configuration."""
    return """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'registry-app'
    static_configs:
      - targets: ['registry-app:9090']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
"""

# =============================================================================
# Main Monitoring Setup
# =============================================================================

async def setup_monitoring():
    """Setup comprehensive monitoring for the registry."""
    print("🔧 Setting up APG Capability Registry monitoring...")
    
    # Initialize metrics collector
    config = MetricsConfig()
    metrics_collector = RegistryMetricsCollector(config)
    
    # Initialize health checker
    health_checker = HealthChecker()
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    print("✅ Monitoring setup completed")
    print(f"   - Metrics endpoint: http://localhost:{config.port}{config.path}")
    print(f"   - Health checks: Available via /api/health")
    print(f"   - Alerts: {len(alert_manager.alert_rules)} rules configured")
    
    return {
        "metrics_collector": metrics_collector,
        "health_checker": health_checker,
        "alert_manager": alert_manager
    }

if __name__ == "__main__":
    import asyncio
    
    async def main():
        monitoring = await setup_monitoring()
        
        # Run a health check
        health_checker = monitoring["health_checker"]
        health_results = await health_checker.check_all()
        
        print("\n🏥 Health Check Results:")
        for name, result in health_results.items():
            status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(result.status, "❓")
            print(f"   {status_emoji} {name}: {result.status} ({result.response_time_ms:.0f}ms)")
            if result.message:
                print(f"      {result.message}")
        
        overall_status = health_checker.get_overall_status(health_results)
        print(f"\n🎯 Overall Status: {overall_status}")
    
    asyncio.run(main())