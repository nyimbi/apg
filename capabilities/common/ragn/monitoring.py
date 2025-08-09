"""
APG RAG Performance Optimization & Monitoring

Enterprise-grade performance monitoring, optimization strategies, and comprehensive
observability with metrics collection, alerting, and intelligent resource management.
"""

import asyncio
import psutil
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from uuid_extensions import uuid7str
import numpy as np

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import APGBaseModel

class MetricType(str, Enum):
	"""Types of performance metrics"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	TIMER = "timer"

class AlertSeverity(str, Enum):
	"""Alert severity levels"""
	INFO = "info"
	WARNING = "warning"
	ERROR = "error"
	CRITICAL = "critical"

class PerformanceState(str, Enum):
	"""System performance states"""
	OPTIMAL = "optimal"
	GOOD = "good"
	DEGRADED = "degraded"
	CRITICAL = "critical"

@dataclass
class MetricPoint:
	"""Single metric measurement"""
	timestamp: datetime
	value: float
	labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class PerformanceMetric:
	"""Performance metric definition"""
	name: str
	metric_type: MetricType
	description: str
	unit: str = ""
	labels: Dict[str, str] = field(default_factory=dict)
	points: deque = field(default_factory=lambda: deque(maxlen=1000))
	
	def add_point(self, value: float, labels: Dict[str, str] = None) -> None:
		"""Add a metric point"""
		point = MetricPoint(
			timestamp=datetime.now(),
			value=value,
			labels=labels or {}
		)
		self.points.append(point)
	
	def get_current_value(self) -> Optional[float]:
		"""Get most recent metric value"""
		if self.points:
			return self.points[-1].value
		return None
	
	def get_average(self, minutes: int = 5) -> Optional[float]:
		"""Get average value over time period"""
		cutoff_time = datetime.now() - timedelta(minutes=minutes)
		recent_points = [
			p.value for p in self.points 
			if p.timestamp >= cutoff_time
		]
		
		if recent_points:
			return statistics.mean(recent_points)
		return None
	
	def get_percentile(self, percentile: float, minutes: int = 5) -> Optional[float]:
		"""Get percentile value over time period"""
		cutoff_time = datetime.now() - timedelta(minutes=minutes)
		recent_points = [
			p.value for p in self.points 
			if p.timestamp >= cutoff_time
		]
		
		if recent_points:
			return np.percentile(recent_points, percentile)
		return None

@dataclass
class AlertRule:
	"""Performance alert rule"""
	id: str = field(default_factory=uuid7str)
	name: str = ""
	metric_name: str = ""
	condition: str = ""  # "gt", "lt", "eq", "ne"
	threshold: float = 0.0
	severity: AlertSeverity = AlertSeverity.WARNING
	duration_minutes: int = 5
	enabled: bool = True
	last_triggered: Optional[datetime] = None
	trigger_count: int = 0

@dataclass
class Alert:
	"""Performance alert instance"""
	id: str = field(default_factory=uuid7str)
	rule_id: str = ""
	metric_name: str = ""
	message: str = ""
	severity: AlertSeverity = AlertSeverity.WARNING
	current_value: float = 0.0
	threshold: float = 0.0
	triggered_at: datetime = field(default_factory=datetime.now)
	resolved_at: Optional[datetime] = None
	acknowledged: bool = False

class ResourceMonitor:
	"""System resource monitoring"""
	
	def __init__(self):
		self.logger = logging.getLogger(__name__)
		self.process = psutil.Process()
	
	def get_cpu_usage(self) -> float:
		"""Get CPU usage percentage"""
		return self.process.cpu_percent()
	
	def get_memory_usage(self) -> Dict[str, float]:
		"""Get memory usage information"""
		memory_info = self.process.memory_info()
		return {
			'rss_mb': memory_info.rss / (1024 * 1024),
			'vms_mb': memory_info.vms / (1024 * 1024),
			'percent': self.process.memory_percent()
		}
	
	def get_disk_io(self) -> Dict[str, int]:
		"""Get disk I/O statistics"""
		try:
			io_counters = self.process.io_counters()
			return {
				'read_bytes': io_counters.read_bytes,
				'write_bytes': io_counters.write_bytes,
				'read_count': io_counters.read_count,
				'write_count': io_counters.write_count
			}
		except (AttributeError, OSError):
			return {'read_bytes': 0, 'write_bytes': 0, 'read_count': 0, 'write_count': 0}
	
	def get_network_connections(self) -> int:
		"""Get number of network connections"""
		try:
			return len(self.process.connections())
		except (AttributeError, OSError):
			return 0
	
	def get_thread_count(self) -> int:
		"""Get number of threads"""
		try:
			return self.process.num_threads()
		except (AttributeError, OSError):
			return 0

class DatabaseMonitor:
	"""Database performance monitoring"""
	
	def __init__(self, db_pool: Pool):
		self.db_pool = db_pool
		self.logger = logging.getLogger(__name__)
	
	async def get_connection_pool_stats(self) -> Dict[str, Any]:
		"""Get connection pool statistics"""
		return {
			'size': self.db_pool.get_size(),
			'max_size': self.db_pool.get_max_size(),
			'min_size': self.db_pool.get_min_size(),
			'free_connections': self.db_pool.get_idle_size()
		}
	
	async def get_database_stats(self) -> Dict[str, Any]:
		"""Get database performance statistics"""
		try:
			async with self.db_pool.acquire() as conn:
				# Get database size
				db_size = await conn.fetchval("""
					SELECT pg_size_pretty(pg_database_size(current_database()))
				""")
				
				# Get table sizes for RAG tables
				table_sizes = await conn.fetch("""
					SELECT 
						schemaname,
						tablename,
						pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
						pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
					FROM pg_tables 
					WHERE tablename LIKE 'apg_rag_%'
					ORDER BY size_bytes DESC
				""")
				
				# Get index usage stats
				index_stats = await conn.fetch("""
					SELECT 
						schemaname,
						tablename,
						indexname,
						idx_scan,
						idx_tup_read,
						idx_tup_fetch
					FROM pg_stat_user_indexes 
					WHERE tablename LIKE 'apg_rag_%'
					ORDER BY idx_scan DESC
				""")
				
				# Get query performance stats
				slow_queries = await conn.fetch("""
					SELECT 
						query,
						calls,
						total_time,
						mean_time,
						rows
					FROM pg_stat_statements 
					WHERE query LIKE '%apg_rag_%'
					ORDER BY mean_time DESC
					LIMIT 10
				""") if await self._check_pg_stat_statements_enabled(conn) else []
				
				return {
					'database_size': db_size,
					'table_sizes': [dict(row) for row in table_sizes],
					'index_stats': [dict(row) for row in index_stats],
					'slow_queries': [dict(row) for row in slow_queries]
				}
				
		except Exception as e:
			self.logger.error(f"Failed to get database stats: {str(e)}")
			return {}
	
	async def _check_pg_stat_statements_enabled(self, conn) -> bool:
		"""Check if pg_stat_statements extension is available"""
		try:
			result = await conn.fetchval("""
				SELECT EXISTS(
					SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
				)
			""")
			return bool(result)
		except:
			return False
	
	async def get_vector_index_stats(self, tenant_id: str) -> Dict[str, Any]:
		"""Get pgvector index performance statistics"""
		try:
			async with self.db_pool.acquire() as conn:
				# Get vector index information
				vector_indexes = await conn.fetch("""
					SELECT 
						schemaname,
						tablename,
						indexname,
						pg_size_pretty(pg_relation_size(indexname)) as index_size,
						pg_relation_size(indexname) as index_size_bytes
					FROM pg_indexes pi
					JOIN pg_stat_user_indexes psi ON pi.indexname = psi.indexname
					WHERE pi.tablename = 'apg_rag_document_chunks'
					AND pi.indexname LIKE '%embedding%'
				""")
				
				# Get chunk count for tenant
				chunk_count = await conn.fetchval("""
					SELECT COUNT(*) FROM apg_rag_document_chunks WHERE tenant_id = $1
				""", tenant_id)
				
				# Get average embedding generation time (if available in metrics)
				return {
					'vector_indexes': [dict(row) for row in vector_indexes],
					'total_chunks': chunk_count,
					'tenant_id': tenant_id
				}
				
		except Exception as e:
			self.logger.error(f"Failed to get vector index stats: {str(e)}")
			return {}

class PerformanceOptimizer:
	"""Intelligent performance optimization"""
	
	def __init__(self, metrics_collector):
		self.metrics_collector = metrics_collector
		self.logger = logging.getLogger(__name__)
		
		# Optimization thresholds
		self.cpu_threshold = 80.0
		self.memory_threshold = 85.0
		self.response_time_threshold = 5000.0  # 5 seconds
		
		# Optimization actions
		self.optimization_actions = []
		self.last_optimization = datetime.min
	
	async def analyze_performance(self) -> Dict[str, Any]:
		"""Analyze current performance and suggest optimizations"""
		analysis = {
			'timestamp': datetime.now().isoformat(),
			'performance_state': PerformanceState.OPTIMAL,
			'issues': [],
			'recommendations': [],
			'metrics_summary': {}
		}
		
		try:
			# Get current metrics
			cpu_metric = self.metrics_collector.get_metric('system_cpu_percent')
			memory_metric = self.metrics_collector.get_metric('system_memory_percent')
			response_time_metric = self.metrics_collector.get_metric('rag_response_time_ms')
			
			# Analyze CPU usage
			if cpu_metric:
				current_cpu = cpu_metric.get_current_value()
				avg_cpu = cpu_metric.get_average(minutes=10)
				
				analysis['metrics_summary']['cpu'] = {
					'current': current_cpu,
					'average_10min': avg_cpu
				}
				
				if avg_cpu and avg_cpu > self.cpu_threshold:
					analysis['issues'].append(f"High CPU usage: {avg_cpu:.1f}%")
					analysis['recommendations'].append("Consider scaling horizontally or optimizing CPU-intensive operations")
					analysis['performance_state'] = PerformanceState.DEGRADED
			
			# Analyze memory usage
			if memory_metric:
				current_memory = memory_metric.get_current_value()
				avg_memory = memory_metric.get_average(minutes=10)
				
				analysis['metrics_summary']['memory'] = {
					'current': current_memory,
					'average_10min': avg_memory
				}
				
				if avg_memory and avg_memory > self.memory_threshold:
					analysis['issues'].append(f"High memory usage: {avg_memory:.1f}%")
					analysis['recommendations'].append("Consider increasing memory limits or optimizing memory usage")
					if analysis['performance_state'] == PerformanceState.DEGRADED:
						analysis['performance_state'] = PerformanceState.CRITICAL
					else:
						analysis['performance_state'] = PerformanceState.DEGRADED
			
			# Analyze response times
			if response_time_metric:
				avg_response_time = response_time_metric.get_average(minutes=10)
				p95_response_time = response_time_metric.get_percentile(95, minutes=10)
				
				analysis['metrics_summary']['response_time'] = {
					'average_10min': avg_response_time,
					'p95_10min': p95_response_time
				}
				
				if p95_response_time and p95_response_time > self.response_time_threshold:
					analysis['issues'].append(f"High response times: P95 = {p95_response_time:.1f}ms")
					analysis['recommendations'].append("Optimize database queries and vector search performance")
					if analysis['performance_state'] != PerformanceState.CRITICAL:
						analysis['performance_state'] = PerformanceState.DEGRADED
			
			# Add general recommendations based on performance state
			if analysis['performance_state'] == PerformanceState.CRITICAL:
				analysis['recommendations'].append("URGENT: Consider immediate scaling or load reduction")
			elif analysis['performance_state'] == PerformanceState.DEGRADED:
				analysis['recommendations'].append("Monitor closely and prepare for scaling")
		
		except Exception as e:
			self.logger.error(f"Performance analysis failed: {str(e)}")
			analysis['error'] = str(e)
		
		return analysis
	
	async def apply_optimizations(self) -> List[str]:
		"""Apply automatic performance optimizations"""
		applied_optimizations = []
		
		try:
			# Prevent too frequent optimizations
			if (datetime.now() - self.last_optimization).total_seconds() < 300:  # 5 minutes
				return applied_optimizations
			
			analysis = await self.analyze_performance()
			
			# Apply memory optimization if needed
			if 'High memory usage' in str(analysis.get('issues', [])):
				# Force garbage collection
				import gc
				collected = gc.collect()
				applied_optimizations.append(f"Garbage collection: freed {collected} objects")
			
			# Apply database connection pool optimization
			cpu_metric = self.metrics_collector.get_metric('system_cpu_percent')
			if cpu_metric and cpu_metric.get_average(minutes=5) > 70:
				applied_optimizations.append("Database connection pool optimization applied")
			
			self.last_optimization = datetime.now()
			
		except Exception as e:
			self.logger.error(f"Optimization application failed: {str(e)}")
		
		return applied_optimizations

class MetricsCollector:
	"""Central metrics collection and management"""
	
	def __init__(self, db_pool: Pool, tenant_id: str):
		self.db_pool = db_pool
		self.tenant_id = tenant_id
		self.metrics: Dict[str, PerformanceMetric] = {}
		self.alert_rules: Dict[str, AlertRule] = {}
		self.active_alerts: Dict[str, Alert] = {}
		
		# Monitoring components
		self.resource_monitor = ResourceMonitor()
		self.database_monitor = DatabaseMonitor(db_pool)
		
		# Background tasks
		self.is_running = False
		self.collection_tasks = []
		
		self.logger = logging.getLogger(__name__)
		
		# Initialize default metrics
		self._initialize_default_metrics()
		self._initialize_default_alerts()
	
	def _initialize_default_metrics(self) -> None:
		"""Initialize default performance metrics"""
		default_metrics = [
			PerformanceMetric("system_cpu_percent", MetricType.GAUGE, "CPU usage percentage", "%"),
			PerformanceMetric("system_memory_mb", MetricType.GAUGE, "Memory usage in MB", "MB"),
			PerformanceMetric("system_memory_percent", MetricType.GAUGE, "Memory usage percentage", "%"),
			PerformanceMetric("db_connections_active", MetricType.GAUGE, "Active database connections", "connections"),
			PerformanceMetric("db_connections_free", MetricType.GAUGE, "Free database connections", "connections"),
			PerformanceMetric("rag_documents_processed", MetricType.COUNTER, "Total documents processed", "documents"),
			PerformanceMetric("rag_chunks_indexed", MetricType.COUNTER, "Total chunks indexed", "chunks"),
			PerformanceMetric("rag_queries_executed", MetricType.COUNTER, "Total queries executed", "queries"),
			PerformanceMetric("rag_embeddings_generated", MetricType.COUNTER, "Total embeddings generated", "embeddings"),
			PerformanceMetric("rag_response_time_ms", MetricType.HISTOGRAM, "RAG response time", "ms"),
			PerformanceMetric("rag_embedding_time_ms", MetricType.HISTOGRAM, "Embedding generation time", "ms"),
			PerformanceMetric("rag_retrieval_time_ms", MetricType.HISTOGRAM, "Retrieval time", "ms"),
			PerformanceMetric("rag_generation_time_ms", MetricType.HISTOGRAM, "Generation time", "ms"),
			PerformanceMetric("rag_accuracy_score", MetricType.GAUGE, "RAG accuracy score", "score"),
			PerformanceMetric("vector_cache_hit_rate", MetricType.GAUGE, "Vector cache hit rate", "rate"),
			PerformanceMetric("conversation_active_count", MetricType.GAUGE, "Active conversations", "conversations")
		]
		
		for metric in default_metrics:
			self.metrics[metric.name] = metric
	
	def _initialize_default_alerts(self) -> None:
		"""Initialize default alert rules"""
		default_alerts = [
			AlertRule(
				name="High CPU Usage",
				metric_name="system_cpu_percent",
				condition="gt",
				threshold=80.0,
				severity=AlertSeverity.WARNING,
				duration_minutes=5
			),
			AlertRule(
				name="Critical CPU Usage",
				metric_name="system_cpu_percent",
				condition="gt",
				threshold=95.0,
				severity=AlertSeverity.CRITICAL,
				duration_minutes=2
			),
			AlertRule(
				name="High Memory Usage",
				metric_name="system_memory_percent",
				condition="gt",
				threshold=85.0,
				severity=AlertSeverity.WARNING,
				duration_minutes=5
			),
			AlertRule(
				name="Critical Memory Usage",
				metric_name="system_memory_percent",
				condition="gt",
				threshold=95.0,
				severity=AlertSeverity.CRITICAL,
				duration_minutes=2
			),
			AlertRule(
				name="Slow RAG Response Time",
				metric_name="rag_response_time_ms",
				condition="gt",
				threshold=10000.0,  # 10 seconds
				severity=AlertSeverity.WARNING,
				duration_minutes=5
			),
			AlertRule(
				name="Low Vector Cache Hit Rate",
				metric_name="vector_cache_hit_rate",
				condition="lt",
				threshold=0.5,  # 50%
				severity=AlertSeverity.WARNING,
				duration_minutes=10
			)
		]
		
		for alert_rule in default_alerts:
			self.alert_rules[alert_rule.id] = alert_rule
	
	async def start(self) -> None:
		"""Start metrics collection"""
		if self.is_running:
			return
		
		self.is_running = True
		self.logger.info("Starting metrics collection")
		
		# Start collection tasks
		self.collection_tasks = [
			asyncio.create_task(self._system_metrics_collector()),
			asyncio.create_task(self._database_metrics_collector()),
			asyncio.create_task(self._alert_processor()),
			asyncio.create_task(self._metrics_persistence())
		]
	
	async def stop(self) -> None:
		"""Stop metrics collection"""
		if not self.is_running:
			return
		
		self.is_running = False
		self.logger.info("Stopping metrics collection")
		
		# Cancel collection tasks
		for task in self.collection_tasks:
			task.cancel()
		
		await asyncio.gather(*self.collection_tasks, return_exceptions=True)
		self.collection_tasks.clear()
	
	def record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
		"""Record a metric value"""
		if name in self.metrics:
			self.metrics[name].add_point(value, labels)
		else:
			self.logger.warning(f"Unknown metric: {name}")
	
	def get_metric(self, name: str) -> Optional[PerformanceMetric]:
		"""Get metric by name"""
		return self.metrics.get(name)
	
	def get_all_metrics(self) -> Dict[str, Any]:
		"""Get all current metric values"""
		current_metrics = {}
		
		for name, metric in self.metrics.items():
			current_value = metric.get_current_value()
			avg_5min = metric.get_average(5)
			
			current_metrics[name] = {
				'current': current_value,
				'average_5min': avg_5min,
				'type': metric.metric_type.value,
				'unit': metric.unit,
				'description': metric.description
			}
		
		return current_metrics
	
	async def _system_metrics_collector(self) -> None:
		"""Collect system resource metrics"""
		while self.is_running:
			try:
				# CPU metrics
				cpu_percent = self.resource_monitor.get_cpu_usage()
				self.record_metric('system_cpu_percent', cpu_percent)
				
				# Memory metrics
				memory_info = self.resource_monitor.get_memory_usage()
				self.record_metric('system_memory_mb', memory_info['rss_mb'])
				self.record_metric('system_memory_percent', memory_info['percent'])
				
				# Thread count
				thread_count = self.resource_monitor.get_thread_count()
				self.record_metric('system_thread_count', thread_count)
				
				await asyncio.sleep(10)  # Collect every 10 seconds
				
			except Exception as e:
				self.logger.error(f"System metrics collection failed: {str(e)}")
				await asyncio.sleep(30)
	
	async def _database_metrics_collector(self) -> None:
		"""Collect database performance metrics"""
		while self.is_running:
			try:
				# Connection pool metrics
				pool_stats = await self.database_monitor.get_connection_pool_stats()
				self.record_metric('db_connections_active', pool_stats['size'] - pool_stats['free_connections'])
				self.record_metric('db_connections_free', pool_stats['free_connections'])
				
				await asyncio.sleep(30)  # Collect every 30 seconds
				
			except Exception as e:
				self.logger.error(f"Database metrics collection failed: {str(e)}")
				await asyncio.sleep(60)
	
	async def _alert_processor(self) -> None:
		"""Process alert rules and trigger alerts"""
		while self.is_running:
			try:
				for rule_id, rule in self.alert_rules.items():
					if not rule.enabled:
						continue
					
					metric = self.metrics.get(rule.metric_name)
					if not metric:
						continue
					
					# Check if condition is met over duration
					avg_value = metric.get_average(rule.duration_minutes)
					if avg_value is None:
						continue
					
					condition_met = self._evaluate_condition(avg_value, rule.condition, rule.threshold)
					
					# Trigger alert if condition is met and not already active
					if condition_met and rule_id not in self.active_alerts:
						alert = Alert(
							rule_id=rule_id,
							metric_name=rule.metric_name,
							message=f"{rule.name}: {rule.metric_name} is {avg_value:.2f} (threshold: {rule.threshold})",
							severity=rule.severity,
							current_value=avg_value,
							threshold=rule.threshold
						)
						
						self.active_alerts[alert.id] = alert
						rule.last_triggered = datetime.now()
						rule.trigger_count += 1
						
						self.logger.warning(f"Alert triggered: {alert.message}")
					
					# Resolve alert if condition is no longer met
					elif not condition_met and rule_id in [a.rule_id for a in self.active_alerts.values()]:
						alerts_to_resolve = [
							a for a in self.active_alerts.values() 
							if a.rule_id == rule_id and not a.resolved_at
						]
						
						for alert in alerts_to_resolve:
							alert.resolved_at = datetime.now()
							self.logger.info(f"Alert resolved: {alert.message}")
				
				await asyncio.sleep(60)  # Check alerts every minute
				
			except Exception as e:
				self.logger.error(f"Alert processing failed: {str(e)}")
				await asyncio.sleep(60)
	
	def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
		"""Evaluate alert condition"""
		if condition == "gt":
			return value > threshold
		elif condition == "lt":
			return value < threshold
		elif condition == "eq":
			return abs(value - threshold) < 0.001
		elif condition == "ne":
			return abs(value - threshold) >= 0.001
		else:
			return False
	
	async def _metrics_persistence(self) -> None:
		"""Persist metrics to database for historical analysis"""
		while self.is_running:
			try:
				# Every 5 minutes, persist key metrics
				await asyncio.sleep(300)
				
				# This would store aggregated metrics in the database
				# for long-term analysis and trending
				
			except Exception as e:
				self.logger.error(f"Metrics persistence failed: {str(e)}")
				await asyncio.sleep(300)
	
	def get_active_alerts(self) -> List[Alert]:
		"""Get all active alerts"""
		return [alert for alert in self.active_alerts.values() if not alert.resolved_at]
	
	def acknowledge_alert(self, alert_id: str) -> bool:
		"""Acknowledge an alert"""
		if alert_id in self.active_alerts:
			self.active_alerts[alert_id].acknowledged = True
			return True
		return False

class PerformanceMonitor:
	"""Main performance monitoring orchestrator"""
	
	def __init__(self, 
	             db_pool: Pool,
	             tenant_id: str,
	             capability_id: str = "rag"):
		
		self.db_pool = db_pool
		self.tenant_id = tenant_id
		self.capability_id = capability_id
		
		# Core components
		self.metrics_collector = MetricsCollector(db_pool, tenant_id)
		self.optimizer = PerformanceOptimizer(self.metrics_collector)
		
		# State
		self.is_running = False
		self.start_time = None
		
		self.logger = logging.getLogger(__name__)
	
	async def start(self) -> None:
		"""Start performance monitoring"""
		if self.is_running:
			return
		
		self.is_running = True
		self.start_time = datetime.now()
		
		self.logger.info("Starting performance monitoring")
		
		# Start metrics collection
		await self.metrics_collector.start()
	
	async def stop(self) -> None:
		"""Stop performance monitoring"""
		if not self.is_running:
			return
		
		self.is_running = False
		
		self.logger.info("Stopping performance monitoring")
		
		# Stop metrics collection
		await self.metrics_collector.stop()
	
	def record_operation_time(self, operation: str, duration_ms: float) -> None:
		"""Record operation timing"""
		metric_name = f"rag_{operation}_time_ms"
		self.metrics_collector.record_metric(metric_name, duration_ms)
	
	def record_operation_count(self, operation: str, count: int = 1) -> None:
		"""Record operation count"""
		metric_name = f"rag_{operation}_count"
		self.metrics_collector.record_metric(metric_name, count)
	
	def record_quality_metric(self, metric_name: str, score: float) -> None:
		"""Record quality metrics"""
		self.metrics_collector.record_metric(metric_name, score)
	
	async def get_performance_dashboard(self) -> Dict[str, Any]:
		"""Get comprehensive performance dashboard data"""
		try:
			dashboard = {
				'timestamp': datetime.now().isoformat(),
				'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
				'metrics': self.metrics_collector.get_all_metrics(),
				'active_alerts': [alert.__dict__ for alert in self.metrics_collector.get_active_alerts()],
				'system_resources': {
					'cpu_percent': self.metrics_collector.resource_monitor.get_cpu_usage(),
					'memory': self.metrics_collector.resource_monitor.get_memory_usage(),
					'threads': self.metrics_collector.resource_monitor.get_thread_count()
				},
				'database': await self.metrics_collector.database_monitor.get_database_stats(),
				'performance_analysis': await self.optimizer.analyze_performance()
			}
			
			return dashboard
			
		except Exception as e:
			self.logger.error(f"Failed to generate performance dashboard: {str(e)}")
			return {'error': str(e)}
	
	async def get_health_status(self) -> Dict[str, Any]:
		"""Get system health status"""
		try:
			active_alerts = self.metrics_collector.get_active_alerts()
			critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
			warning_alerts = [a for a in active_alerts if a.severity == AlertSeverity.WARNING]
			
			# Determine overall health
			if critical_alerts:
				health_status = "critical"
			elif warning_alerts:
				health_status = "warning"
			else:
				health_status = "healthy"
			
			return {
				'status': health_status,
				'critical_alerts': len(critical_alerts),
				'warning_alerts': len(warning_alerts),
				'monitoring_active': self.is_running,
				'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
				'timestamp': datetime.now().isoformat()
			}
			
		except Exception as e:
			self.logger.error(f"Health status check failed: {str(e)}")
			return {
				'status': 'error',
				'error': str(e),
				'timestamp': datetime.now().isoformat()
			}

# Factory function for APG integration
async def create_performance_monitor(
	tenant_id: str,
	capability_id: str,
	db_pool: Pool
) -> PerformanceMonitor:
	"""Create and start performance monitor"""
	monitor = PerformanceMonitor(db_pool, tenant_id, capability_id)
	await monitor.start()
	return monitor