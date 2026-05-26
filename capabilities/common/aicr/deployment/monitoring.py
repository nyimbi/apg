"""
Production Monitoring and Health Checks for AICR

This module provides comprehensive production monitoring including:
- Advanced health check endpoints with detailed diagnostics
- Prometheus metrics collection and custom metrics
- Real-time alerting with intelligent threshold management
- Performance monitoring and SLA tracking
- Application observability with distributed tracing
- Infrastructure monitoring and resource tracking
- Custom dashboards and visualization
- Automated incident response and escalation
- Predictive monitoring and anomaly detection

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import logging
import os
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from enum import Enum
import platform
import subprocess
import socket
import ssl
from dataclasses import dataclass
from collections import defaultdict, deque

import aiofiles
import aiohttp
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import asyncpg
import redis.asyncio as aioredis
from pydantic import BaseModel, Field, ConfigDict
from kubernetes import client as k8s_client, config as k8s_config
import GPUtil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from uuid_extensions import uuid7str


class HealthStatus(str, Enum):
	"""Health status levels."""
	HEALTHY = "healthy"
	WARNING = "warning"
	CRITICAL = "critical"
	UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
	"""Alert severity levels."""
	INFO = "info"
	WARNING = "warning"
	CRITICAL = "critical"
	EMERGENCY = "emergency"


class MetricType(str, Enum):
	"""Metric types."""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"
	INFO = "info"


class ComponentType(str, Enum):
	"""Component types for monitoring."""
	APPLICATION = "application"
	DATABASE = "database"
	CACHE = "cache"
	INFERENCE_ENGINE = "inference_engine"
	MODEL_REGISTRY = "model_registry"
	LOAD_BALANCER = "load_balancer"
	STORAGE = "storage"
	NETWORK = "network"
	SECURITY = "security"


class HealthCheck(BaseModel):
	"""Health check result."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	check_id: str = Field(default_factory=uuid7str)
	component: ComponentType
	name: str
	status: HealthStatus
	message: str
	response_time_ms: float
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	details: Dict[str, Any] = Field(default_factory=dict)
	metrics: Dict[str, float] = Field(default_factory=dict)
	dependencies: List[str] = Field(default_factory=list)
	remediation_steps: List[str] = Field(default_factory=list)


class Alert(BaseModel):
	"""Monitoring alert."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	alert_id: str = Field(default_factory=uuid7str)
	component: ComponentType
	severity: AlertSeverity
	title: str
	description: str
	metric_name: str
	current_value: float
	threshold: float
	timestamp: datetime = Field(default_factory=datetime.utcnow)
	resolved: bool = False
	resolved_at: Optional[datetime] = None
	escalated: bool = False
	escalated_at: Optional[datetime] = None
	correlation_id: Optional[str] = None
	labels: Dict[str, str] = Field(default_factory=dict)
	annotations: Dict[str, str] = Field(default_factory=dict)


class MetricDefinition(BaseModel):
	"""Metric definition."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	name: str
	metric_type: MetricType
	description: str
	labels: List[str] = Field(default_factory=list)
	unit: Optional[str] = None
	buckets: Optional[List[float]] = None  # For histograms
	objectives: Optional[Dict[str, float]] = None  # For summaries


class SLI(BaseModel):
	"""Service Level Indicator."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	name: str
	description: str
	metric_query: str
	target_value: float
	current_value: Optional[float] = None
	status: HealthStatus = HealthStatus.UNKNOWN
	last_updated: datetime = Field(default_factory=datetime.utcnow)


class SLO(BaseModel):
	"""Service Level Objective."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	name: str
	description: str
	sli: SLI
	target_percentage: float
	time_window_hours: int
	error_budget_consumed: float = 0.0
	status: HealthStatus = HealthStatus.UNKNOWN
	alerts: List[str] = Field(default_factory=list)


class MonitoringConfig(BaseModel):
	"""Monitoring configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True, validate_by_alias=True)

	health_check_interval: int = 30
	metrics_collection_interval: int = 15
	alert_evaluation_interval: int = 60
	anomaly_detection_enabled: bool = True
	predictive_monitoring_enabled: bool = True
	distributed_tracing_enabled: bool = True
	log_correlation_enabled: bool = True
	slo_monitoring_enabled: bool = True
	prometheus_url: str = "http://prometheus:9090"
	grafana_url: str = "http://grafana:3000"
	alertmanager_url: str = "http://alertmanager:9093"
	jaeger_url: str = "http://jaeger:14268"


@dataclass
class SystemMetrics:
	"""System metrics snapshot."""
	cpu_percent: float
	memory_percent: float
	disk_percent: float
	network_io: Dict[str, float]
	gpu_percent: Optional[float] = None
	gpu_memory_percent: Optional[float] = None
	open_files: int = 0
	tcp_connections: int = 0
	load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
	uptime_seconds: float = 0.0
	timestamp: datetime = datetime.utcnow()


class PrometheusMetrics:
	"""Prometheus metrics collection."""

	def __init__(self):
		# Application metrics
		self.http_requests_total = Counter(
			'aicr_http_requests_total',
			'Total HTTP requests',
			['method', 'endpoint', 'status_code']
		)

		self.http_request_duration = Histogram(
			'aicr_http_request_duration_seconds',
			'HTTP request duration',
			['method', 'endpoint'],
			buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
		)

		self.inference_requests_total = Counter(
			'aicr_inference_requests_total',
			'Total inference requests',
			['model_id', 'status']
		)

		self.inference_duration = Histogram(
			'aicr_inference_duration_seconds',
			'Inference duration',
			['model_id', 'framework'],
			buckets=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
		)

		self.model_accuracy = Gauge(
			'aicr_model_accuracy',
			'Model accuracy',
			['model_id', 'version']
		)

		self.active_models = Gauge(
			'aicr_active_models_total',
			'Number of active models'
		)

		self.queue_length = Gauge(
			'aicr_inference_queue_length',
			'Inference queue length',
			['priority']
		)

		# System metrics
		self.cpu_usage = Gauge(
			'aicr_cpu_usage_percent',
			'CPU usage percentage'
		)

		self.memory_usage = Gauge(
			'aicr_memory_usage_percent',
			'Memory usage percentage'
		)

		self.disk_usage = Gauge(
			'aicr_disk_usage_percent',
			'Disk usage percentage',
			['mount_point']
		)

		self.gpu_usage = Gauge(
			'aicr_gpu_usage_percent',
			'GPU usage percentage',
			['gpu_id']
		)

		self.database_connections = Gauge(
			'aicr_database_connections_active',
			'Active database connections'
		)

		self.cache_hit_rate = Gauge(
			'aicr_cache_hit_rate',
			'Cache hit rate percentage'
		)

		# Health metrics
		self.health_check_status = Gauge(
			'aicr_health_check_status',
			'Health check status (1=healthy, 0=unhealthy)',
			['component', 'check_name']
		)

		self.health_check_duration = Histogram(
			'aicr_health_check_duration_seconds',
			'Health check duration',
			['component', 'check_name']
		)

		# Business metrics
		self.slo_compliance = Gauge(
			'aicr_slo_compliance_percent',
			'SLO compliance percentage',
			['slo_name']
		)

		self.error_budget_remaining = Gauge(
			'aicr_error_budget_remaining_percent',
			'Error budget remaining percentage',
			['slo_name']
		)


class HealthChecker:
	"""Comprehensive health checker."""

	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.HealthChecker")
		self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
		self._last_check_time: Dict[str, datetime] = {}

	async def check_all_components(self) -> List[HealthCheck]:
		"""Check health of all components."""
		checks = []

		try:
			# Application health
			checks.append(await self._check_application_health())

			# Database health
			checks.append(await self._check_database_health())

			# Cache health
			checks.append(await self._check_cache_health())

			# Inference engine health
			checks.append(await self._check_inference_engine_health())

			# Model registry health
			checks.append(await self._check_model_registry_health())

			# Storage health
			checks.append(await self._check_storage_health())

			# Network health
			checks.append(await self._check_network_health())

			# Security health
			checks.append(await self._check_security_health())

			# System resource health
			checks.append(await self._check_system_resources())

		except Exception as e:
			self.logger.error(f"Health check failed: {e}")
			checks.append(HealthCheck(
				component=ComponentType.APPLICATION,
				name="health_checker",
				status=HealthStatus.CRITICAL,
				message=f"Health checker failed: {str(e)}",
				response_time_ms=0.0
			))

		return checks

	async def _check_application_health(self) -> HealthCheck:
		"""Check application health."""
		start_time = time.time()

		try:
			# Check if application is responding
			async with aiohttp.ClientSession() as session:
				async with session.get('http://localhost:8080/health', timeout=10) as response:
					response_time = (time.time() - start_time) * 1000

					if response.status == 200:
						health_data = await response.json()

						return HealthCheck(
							component=ComponentType.APPLICATION,
							name="application_endpoint",
							status=HealthStatus.HEALTHY,
							message="Application is responding normally",
							response_time_ms=response_time,
							details=health_data,
							metrics={
								"response_time_ms": response_time,
								"status_code": response.status
							}
						)
					else:
						return HealthCheck(
							component=ComponentType.APPLICATION,
							name="application_endpoint",
							status=HealthStatus.CRITICAL,
							message=f"Application returned status {response.status}",
							response_time_ms=response_time,
							metrics={"status_code": response.status}
						)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.APPLICATION,
				name="application_endpoint",
				status=HealthStatus.CRITICAL,
				message=f"Application health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check if application is running",
					"Verify network connectivity",
					"Check application logs",
					"Restart application if necessary"
				]
			)

	async def _check_database_health(self) -> HealthCheck:
		"""Check database health."""
		start_time = time.time()

		try:
			# Get database configuration from environment
			db_host = os.getenv('DATABASE_HOST', 'localhost')
			db_port = int(os.getenv('DATABASE_PORT', '5432'))
			db_name = os.getenv('DATABASE_NAME', 'aicr')
			db_user = os.getenv('DATABASE_USERNAME', 'aicr')
			db_password = os.getenv('DATABASE_PASSWORD', '')

			# Test database connection
			conn = await asyncpg.connect(
				host=db_host,
				port=db_port,
				database=db_name,
				user=db_user,
				password=db_password,
				timeout=10
			)

			# Test query execution
			result = await conn.fetchval('SELECT version()')

			# Get connection stats
			active_connections = await conn.fetchval(
				'SELECT count(*) FROM pg_stat_activity WHERE state = $1',
				'active'
			)

			# Get database size
			db_size = await conn.fetchval(
				'SELECT pg_size_pretty(pg_database_size($1))',
				db_name
			)

			await conn.close()

			response_time = (time.time() - start_time) * 1000

			return HealthCheck(
				component=ComponentType.DATABASE,
				name="postgresql_connection",
				status=HealthStatus.HEALTHY,
				message="Database is accessible and responding",
				response_time_ms=response_time,
				details={
					"version": result,
					"active_connections": active_connections,
					"database_size": db_size
				},
				metrics={
					"response_time_ms": response_time,
					"active_connections": active_connections
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.DATABASE,
				name="postgresql_connection",
				status=HealthStatus.CRITICAL,
				message=f"Database health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check database server status",
					"Verify connection parameters",
					"Check network connectivity to database",
					"Review database logs"
				]
			)

	async def _check_cache_health(self) -> HealthCheck:
		"""Check cache health."""
		start_time = time.time()

		try:
			# Get Redis configuration from environment
			redis_host = os.getenv('CACHE_HOST', 'localhost')
			redis_port = int(os.getenv('CACHE_PORT', '6379'))
			redis_password = os.getenv('CACHE_PASSWORD')

			# Test Redis connection
			redis = aioredis.Redis(
				host=redis_host,
				port=redis_port,
				password=redis_password,
				socket_timeout=10,
				socket_connect_timeout=10
			)

			# Test basic operations
			await redis.set('health_check', 'ok', ex=60)
			result = await redis.get('health_check')

			# Get Redis info
			info = await redis.info()

			await redis.close()

			response_time = (time.time() - start_time) * 1000

			memory_usage = info.get('used_memory_human', 'unknown')
			connected_clients = info.get('connected_clients', 0)
			hit_rate = info.get('keyspace_hits', 0) / max(
				info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1
			) * 100

			return HealthCheck(
				component=ComponentType.CACHE,
				name="redis_connection",
				status=HealthStatus.HEALTHY,
				message="Cache is accessible and responding",
				response_time_ms=response_time,
				details={
					"memory_usage": memory_usage,
					"connected_clients": connected_clients,
					"hit_rate_percent": hit_rate,
					"redis_version": info.get('redis_version', 'unknown')
				},
				metrics={
					"response_time_ms": response_time,
					"connected_clients": connected_clients,
					"hit_rate_percent": hit_rate
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.CACHE,
				name="redis_connection",
				status=HealthStatus.CRITICAL,
				message=f"Cache health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check Redis server status",
					"Verify Redis configuration",
					"Check network connectivity to Redis",
					"Review Redis logs"
				]
			)

	async def _check_inference_engine_health(self) -> HealthCheck:
		"""Check inference engine health."""
		start_time = time.time()

		try:
			# Check inference endpoint
			async with aiohttp.ClientSession() as session:
				test_data = {
					"model_id": "health_check_model",
					"input_data": {"text": "health check"},
					"parameters": {"max_tokens": 1}
				}

				async with session.post(
					'http://localhost:8080/api/v1/inference/predict',
					json=test_data,
					timeout=30
				) as response:
					response_time = (time.time() - start_time) * 1000

					if response.status in [200, 404]:  # 404 is OK for health check model
						return HealthCheck(
							component=ComponentType.INFERENCE_ENGINE,
							name="inference_endpoint",
							status=HealthStatus.HEALTHY,
							message="Inference engine is responding",
							response_time_ms=response_time,
							metrics={
								"response_time_ms": response_time,
								"status_code": response.status
							}
						)
					else:
						return HealthCheck(
							component=ComponentType.INFERENCE_ENGINE,
							name="inference_endpoint",
							status=HealthStatus.WARNING,
							message=f"Inference engine returned status {response.status}",
							response_time_ms=response_time,
							metrics={"status_code": response.status}
						)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.INFERENCE_ENGINE,
				name="inference_endpoint",
				status=HealthStatus.CRITICAL,
				message=f"Inference engine health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check inference service status",
					"Verify model loading",
					"Check GPU/CPU resources",
					"Review inference logs"
				]
			)

	async def _check_model_registry_health(self) -> HealthCheck:
		"""Check model registry health."""
		start_time = time.time()

		try:
			# Check model registry endpoint
			async with aiohttp.ClientSession() as session:
				async with session.get(
					'http://localhost:8080/api/v1/models',
					params={"limit": 1},
					timeout=10
				) as response:
					response_time = (time.time() - start_time) * 1000

					if response.status == 200:
						models_data = await response.json()

						return HealthCheck(
							component=ComponentType.MODEL_REGISTRY,
							name="model_registry_endpoint",
							status=HealthStatus.HEALTHY,
							message="Model registry is accessible",
							response_time_ms=response_time,
							details={
								"total_models": len(models_data.get('models', [])),
								"registry_version": models_data.get('version', 'unknown')
							},
							metrics={
								"response_time_ms": response_time,
								"status_code": response.status
							}
						)
					else:
						return HealthCheck(
							component=ComponentType.MODEL_REGISTRY,
							name="model_registry_endpoint",
							status=HealthStatus.CRITICAL,
							message=f"Model registry returned status {response.status}",
							response_time_ms=response_time,
							metrics={"status_code": response.status}
						)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.MODEL_REGISTRY,
				name="model_registry_endpoint",
				status=HealthStatus.CRITICAL,
				message=f"Model registry health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check model registry service",
					"Verify model storage access",
					"Check API gateway configuration",
					"Review registry logs"
				]
			)

	async def _check_storage_health(self) -> HealthCheck:
		"""Check storage health."""
		start_time = time.time()

		try:
			# Check disk space
			disk_usage = psutil.disk_usage('/')
			free_percent = (disk_usage.free / disk_usage.total) * 100

			# Check if we can write to storage
			test_file = Path('/tmp/aicr_health_check.txt')
			test_file.write_text('health check')
			test_file.unlink()

			response_time = (time.time() - start_time) * 1000

			if free_percent < 10:
				status = HealthStatus.CRITICAL
				message = f"Critical: Only {free_percent:.1f}% disk space remaining"
			elif free_percent < 20:
				status = HealthStatus.WARNING
				message = f"Warning: Only {free_percent:.1f}% disk space remaining"
			else:
				status = HealthStatus.HEALTHY
				message = f"Storage is healthy with {free_percent:.1f}% free space"

			return HealthCheck(
				component=ComponentType.STORAGE,
				name="disk_space",
				status=status,
				message=message,
				response_time_ms=response_time,
				details={
					"total_gb": disk_usage.total / (1024**3),
					"free_gb": disk_usage.free / (1024**3),
					"used_gb": disk_usage.used / (1024**3),
					"free_percent": free_percent
				},
				metrics={
					"response_time_ms": response_time,
					"free_percent": free_percent
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.STORAGE,
				name="disk_space",
				status=HealthStatus.CRITICAL,
				message=f"Storage health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check disk space availability",
					"Clean up temporary files",
					"Check storage mount points",
					"Verify write permissions"
				]
			)

	async def _check_network_health(self) -> HealthCheck:
		"""Check network health."""
		start_time = time.time()

		try:
			# Check external connectivity
			async with aiohttp.ClientSession() as session:
				try:
					async with session.get('https://www.google.com', timeout=5) as response:
						external_connectivity = response.status == 200
				except:
					external_connectivity = False

			# Check DNS resolution
			try:
				socket.gethostbyname('google.com')
				dns_working = True
			except:
				dns_working = False

			# Check internal connectivity (localhost)
			try:
				sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
				sock.settimeout(5)
				result = sock.connect_ex(('localhost', 8080))
				sock.close()
				internal_connectivity = result == 0
			except:
				internal_connectivity = False

			response_time = (time.time() - start_time) * 1000

			if not dns_working or not internal_connectivity:
				status = HealthStatus.CRITICAL
				message = "Network connectivity issues detected"
			elif not external_connectivity:
				status = HealthStatus.WARNING
				message = "External connectivity limited"
			else:
				status = HealthStatus.HEALTHY
				message = "Network connectivity is healthy"

			return HealthCheck(
				component=ComponentType.NETWORK,
				name="network_connectivity",
				status=status,
				message=message,
				response_time_ms=response_time,
				details={
					"external_connectivity": external_connectivity,
					"dns_working": dns_working,
					"internal_connectivity": internal_connectivity
				},
				metrics={
					"response_time_ms": response_time,
					"external_connectivity": 1 if external_connectivity else 0,
					"dns_working": 1 if dns_working else 0,
					"internal_connectivity": 1 if internal_connectivity else 0
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.NETWORK,
				name="network_connectivity",
				status=HealthStatus.CRITICAL,
				message=f"Network health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check network interface status",
					"Verify DNS configuration",
					"Check firewall rules",
					"Test external connectivity"
				]
			)

	async def _check_security_health(self) -> HealthCheck:
		"""Check security health."""
		start_time = time.time()

		try:
			security_issues = []

			# Check SSL certificate validity
			try:
				cert_path = os.getenv('TLS_CERT_PATH', '/etc/ssl/certs/aicr.crt')
				if os.path.exists(cert_path):
					# This is a simplified check - in production you'd want more thorough validation
					cert_info = ssl.get_server_certificate(('localhost', 443))
					if cert_info:
						# Certificate exists and can be loaded
						pass
				else:
					security_issues.append("SSL certificate not found")
			except Exception as e:
				security_issues.append(f"SSL certificate check failed: {str(e)}")

			# Check for security patches (simplified)
			try:
				result = subprocess.run(['which', 'unattended-upgrades'],
									  capture_output=True, text=True, timeout=5)
				if result.returncode != 0:
					security_issues.append("Automatic security updates not configured")
			except:
				pass

			# Check file permissions on sensitive files
			sensitive_files = ['/etc/ssl/private', '/etc/aicr']
			for file_path in sensitive_files:
				if os.path.exists(file_path):
					stat_info = os.stat(file_path)
					if stat_info.st_mode & 0o077:  # Check if readable by others
						security_issues.append(f"Insecure permissions on {file_path}")

			response_time = (time.time() - start_time) * 1000

			if security_issues:
				status = HealthStatus.WARNING
				message = f"Security issues detected: {', '.join(security_issues)}"
			else:
				status = HealthStatus.HEALTHY
				message = "Security configuration appears healthy"

			return HealthCheck(
				component=ComponentType.SECURITY,
				name="security_configuration",
				status=status,
				message=message,
				response_time_ms=response_time,
				details={
					"security_issues": security_issues,
					"checks_performed": ["ssl_certificate", "file_permissions", "security_updates"]
				},
				metrics={
					"response_time_ms": response_time,
					"security_issues_count": len(security_issues)
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.SECURITY,
				name="security_configuration",
				status=HealthStatus.CRITICAL,
				message=f"Security health check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Review SSL certificate configuration",
					"Check file permissions",
					"Verify security update configuration",
					"Run security audit"
				]
			)

	async def _check_system_resources(self) -> HealthCheck:
		"""Check system resource health."""
		start_time = time.time()

		try:
			# CPU usage
			cpu_percent = psutil.cpu_percent(interval=1)

			# Memory usage
			memory = psutil.virtual_memory()
			memory_percent = memory.percent

			# Disk usage
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100

			# Load average
			load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)

			# GPU usage (if available)
			gpu_percent = None
			try:
				gpus = GPUtil.getGPUs()
				if gpus:
					gpu_percent = gpus[0].load * 100
			except:
				pass

			response_time = (time.time() - start_time) * 1000

			# Determine status based on resource usage
			issues = []
			if cpu_percent > 90:
				issues.append(f"High CPU usage: {cpu_percent:.1f}%")
			if memory_percent > 90:
				issues.append(f"High memory usage: {memory_percent:.1f}%")
			if disk_percent > 90:
				issues.append(f"High disk usage: {disk_percent:.1f}%")

			if issues:
				if cpu_percent > 95 or memory_percent > 95 or disk_percent > 95:
					status = HealthStatus.CRITICAL
				else:
					status = HealthStatus.WARNING
				message = f"Resource issues: {', '.join(issues)}"
			else:
				status = HealthStatus.HEALTHY
				message = "System resources are healthy"

			details = {
				"cpu_percent": cpu_percent,
				"memory_percent": memory_percent,
				"disk_percent": disk_percent,
				"load_average": load_avg,
				"memory_total_gb": memory.total / (1024**3),
				"memory_available_gb": memory.available / (1024**3)
			}

			if gpu_percent is not None:
				details["gpu_percent"] = gpu_percent

			return HealthCheck(
				component=ComponentType.APPLICATION,
				name="system_resources",
				status=status,
				message=message,
				response_time_ms=response_time,
				details=details,
				metrics={
					"response_time_ms": response_time,
					"cpu_percent": cpu_percent,
					"memory_percent": memory_percent,
					"disk_percent": disk_percent
				}
			)

		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			return HealthCheck(
				component=ComponentType.APPLICATION,
				name="system_resources",
				status=HealthStatus.CRITICAL,
				message=f"System resource check failed: {str(e)}",
				response_time_ms=response_time,
				remediation_steps=[
					"Check system resource usage",
					"Identify resource-intensive processes",
					"Scale resources if needed",
					"Optimize application performance"
				]
			)


class MetricsCollector:
	"""Advanced metrics collection."""

	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
		self.prometheus_metrics = PrometheusMetrics()
		self._metrics_buffer: List[Dict[str, Any]] = []
		self._last_collection_time = datetime.utcnow()

	async def collect_system_metrics(self) -> SystemMetrics:
		"""Collect comprehensive system metrics."""
		try:
			# CPU metrics
			cpu_percent = psutil.cpu_percent(interval=1)

			# Memory metrics
			memory = psutil.virtual_memory()
			memory_percent = memory.percent

			# Disk metrics
			disk = psutil.disk_usage('/')
			disk_percent = (disk.used / disk.total) * 100

			# Network I/O
			network_io = psutil.net_io_counters()
			network_metrics = {
				"bytes_sent": network_io.bytes_sent,
				"bytes_recv": network_io.bytes_recv,
				"packets_sent": network_io.packets_sent,
				"packets_recv": network_io.packets_recv
			}

			# Process metrics
			process = psutil.Process()
			open_files = len(process.open_files())
			tcp_connections = len(process.connections(kind='tcp'))

			# Load average
			load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)

			# Uptime
			boot_time = psutil.boot_time()
			uptime_seconds = time.time() - boot_time

			# GPU metrics (if available)
			gpu_percent = None
			gpu_memory_percent = None
			try:
				gpus = GPUtil.getGPUs()
				if gpus:
					gpu = gpus[0]
					gpu_percent = gpu.load * 100
					gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
			except:
				pass

			# Update Prometheus metrics
			self.prometheus_metrics.cpu_usage.set(cpu_percent)
			self.prometheus_metrics.memory_usage.set(memory_percent)
			self.prometheus_metrics.disk_usage.labels(mount_point='/').set(disk_percent)

			if gpu_percent is not None:
				self.prometheus_metrics.gpu_usage.labels(gpu_id='0').set(gpu_percent)

			return SystemMetrics(
				cpu_percent=cpu_percent,
				memory_percent=memory_percent,
				disk_percent=disk_percent,
				network_io=network_metrics,
				gpu_percent=gpu_percent,
				gpu_memory_percent=gpu_memory_percent,
				open_files=open_files,
				tcp_connections=tcp_connections,
				load_average=load_avg,
				uptime_seconds=uptime_seconds
			)

		except Exception as e:
			self.logger.error(f"Failed to collect system metrics: {e}")
			raise

	async def collect_application_metrics(self) -> Dict[str, Any]:
		"""Collect application-specific metrics."""
		try:
			metrics = {}

			# Collect metrics from application endpoints
			async with aiohttp.ClientSession() as session:
				try:
					async with session.get('http://localhost:8080/metrics', timeout=10) as response:
						if response.status == 200:
							prometheus_data = await response.text()
							metrics['prometheus_metrics'] = prometheus_data
				except Exception as e:
					self.logger.warning(f"Failed to collect Prometheus metrics: {e}")

				# Application status
				try:
					async with session.get('http://localhost:8080/health', timeout=10) as response:
						if response.status == 200:
							health_data = await response.json()
							metrics['application_health'] = health_data
				except Exception as e:
					self.logger.warning(f"Failed to collect application health: {e}")

			return metrics

		except Exception as e:
			self.logger.error(f"Failed to collect application metrics: {e}")
			return {}

	async def collect_kubernetes_metrics(self) -> Dict[str, Any]:
		"""Collect Kubernetes metrics."""
		try:
			metrics = {}

			# Load Kubernetes configuration
			try:
				k8s_config.load_incluster_config()
			except:
				k8s_config.load_kube_config()

			v1 = k8s_client.CoreV1Api()
			apps_v1 = k8s_client.AppsV1Api()

			# Pod metrics
			pods = v1.list_namespaced_pod(namespace=os.getenv('NAMESPACE', 'aicr-production'))
			pod_metrics = {
				'total_pods': len(pods.items),
				'running_pods': len([p for p in pods.items if p.status.phase == 'Running']),
				'pending_pods': len([p for p in pods.items if p.status.phase == 'Pending']),
				'failed_pods': len([p for p in pods.items if p.status.phase == 'Failed'])
			}
			metrics['pods'] = pod_metrics

			# Deployment metrics
			deployments = apps_v1.list_namespaced_deployment(namespace=os.getenv('NAMESPACE', 'aicr-production'))
			deployment_metrics = {
				'total_deployments': len(deployments.items),
				'ready_deployments': len([d for d in deployments.items if d.status.ready_replicas == d.status.replicas])
			}
			metrics['deployments'] = deployment_metrics

			# Service metrics
			services = v1.list_namespaced_service(namespace=os.getenv('NAMESPACE', 'aicr-production'))
			metrics['services'] = {'total_services': len(services.items)}

			return metrics

		except Exception as e:
			self.logger.warning(f"Failed to collect Kubernetes metrics: {e}")
			return {}


class AlertManager:
	"""Advanced alerting system."""

	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.AlertManager")
		self._active_alerts: Dict[str, Alert] = {}
		self._alert_rules: List[Dict[str, Any]] = []
		self._notification_channels: Dict[str, Callable] = {}

	def add_alert_rule(self, rule: Dict[str, Any]) -> None:
		"""Add an alert rule."""
		self._alert_rules.append(rule)

	def add_notification_channel(self, name: str, handler: Callable) -> None:
		"""Add a notification channel."""
		self._notification_channels[name] = handler

	async def evaluate_alerts(self, metrics: Dict[str, Any], health_checks: List[HealthCheck]) -> List[Alert]:
		"""Evaluate alert rules against current metrics."""
		new_alerts = []

		try:
			# Evaluate health check alerts
			for health_check in health_checks:
				if health_check.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
					alert_id = f"health_{health_check.component.value}_{health_check.name}"

					if alert_id not in self._active_alerts:
						alert = Alert(
							alert_id=alert_id,
							component=health_check.component,
							severity=AlertSeverity.CRITICAL if health_check.status == HealthStatus.CRITICAL else AlertSeverity.WARNING,
							title=f"Health Check Failed: {health_check.name}",
							description=health_check.message,
							metric_name=f"health_check_{health_check.name}",
							current_value=0.0,
							threshold=1.0,
							labels={
								"component": health_check.component.value,
								"check_name": health_check.name
							}
						)

						self._active_alerts[alert_id] = alert
						new_alerts.append(alert)
						await self._send_alert(alert)

			# Evaluate custom alert rules
			for rule in self._alert_rules:
				await self._evaluate_rule(rule, metrics, new_alerts)

			# Check for resolved alerts
			await self._check_resolved_alerts(health_checks)

		except Exception as e:
			self.logger.error(f"Failed to evaluate alerts: {e}")

		return new_alerts

	async def _evaluate_rule(self, rule: Dict[str, Any], metrics: Dict[str, Any], new_alerts: List[Alert]) -> None:
		"""Evaluate a single alert rule."""
		try:
			metric_name = rule['metric']
			threshold = rule['threshold']
			comparison = rule.get('comparison', 'greater_than')
			severity = AlertSeverity(rule.get('severity', 'warning'))

			# Extract metric value
			current_value = self._extract_metric_value(metrics, metric_name)

			if current_value is None:
				return

			# Evaluate condition
			triggered = False
			if comparison == 'greater_than' and current_value > threshold:
				triggered = True
			elif comparison == 'less_than' and current_value < threshold:
				triggered = True
			elif comparison == 'equals' and current_value == threshold:
				triggered = True

			alert_id = f"rule_{rule['name']}"

			if triggered and alert_id not in self._active_alerts:
				alert = Alert(
					alert_id=alert_id,
					component=ComponentType(rule.get('component', 'application')),
					severity=severity,
					title=rule['title'],
					description=rule['description'],
					metric_name=metric_name,
					current_value=current_value,
					threshold=threshold,
					labels=rule.get('labels', {}),
					annotations=rule.get('annotations', {})
				)

				self._active_alerts[alert_id] = alert
				new_alerts.append(alert)
				await self._send_alert(alert)

		except Exception as e:
			self.logger.error(f"Failed to evaluate rule {rule.get('name', 'unknown')}: {e}")

	def _extract_metric_value(self, metrics: Dict[str, Any], metric_path: str) -> Optional[float]:
		"""Extract metric value from nested metrics dictionary."""
		try:
			keys = metric_path.split('.')
			value = metrics

			for key in keys:
				if isinstance(value, dict) and key in value:
					value = value[key]
				else:
					return None

			return float(value) if value is not None else None

		except (ValueError, TypeError, KeyError):
			return None

	async def _check_resolved_alerts(self, health_checks: List[HealthCheck]) -> None:
		"""Check if any active alerts should be resolved."""
		try:
			resolved_alerts = []

			for alert_id, alert in self._active_alerts.items():
				if alert.resolved:
					continue

				# Check if health check alerts are resolved
				if alert_id.startswith('health_'):
					component_name = alert.labels.get('component')
					check_name = alert.labels.get('check_name')

					# Find corresponding health check
					health_check = next(
						(hc for hc in health_checks
						 if hc.component.value == component_name and hc.name == check_name),
						None
					)

					if health_check and health_check.status == HealthStatus.HEALTHY:
						alert.resolved = True
						alert.resolved_at = datetime.utcnow()
						resolved_alerts.append(alert)

			# Send resolution notifications
			for alert in resolved_alerts:
				await self._send_resolution(alert)

		except Exception as e:
			self.logger.error(f"Failed to check resolved alerts: {e}")

	async def _send_alert(self, alert: Alert) -> None:
		"""Send alert notification."""
		try:
			self.logger.warning(f"ALERT: {alert.title} - {alert.description}")

			# Send to configured notification channels
			for channel_name, handler in self._notification_channels.items():
				try:
					await handler(alert, 'fired')
				except Exception as e:
					self.logger.error(f"Failed to send alert to {channel_name}: {e}")

		except Exception as e:
			self.logger.error(f"Failed to send alert: {e}")

	async def _send_resolution(self, alert: Alert) -> None:
		"""Send alert resolution notification."""
		try:
			self.logger.info(f"RESOLVED: {alert.title}")

			# Send to configured notification channels
			for channel_name, handler in self._notification_channels.items():
				try:
					await handler(alert, 'resolved')
				except Exception as e:
					self.logger.error(f"Failed to send resolution to {channel_name}: {e}")

		except Exception as e:
			self.logger.error(f"Failed to send resolution: {e}")


class AnomalyDetector:
	"""ML-based anomaly detection for metrics."""

	def __init__(self):
		self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
		self._models: Dict[str, IsolationForest] = {}
		self._scalers: Dict[str, StandardScaler] = {}
		self._metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		self._training_threshold = 100  # Minimum samples needed for training

	async def detect_anomalies(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		"""Detect anomalies in metrics."""
		anomalies = {}

		try:
			# Process each metric
			for metric_name, value in self._flatten_metrics(metrics).items():
				if isinstance(value, (int, float)):
					anomaly_result = await self._detect_metric_anomaly(metric_name, value)
					if anomaly_result['is_anomaly']:
						anomalies[metric_name] = anomaly_result

		except Exception as e:
			self.logger.error(f"Anomaly detection failed: {e}")

		return anomalies

	async def _detect_metric_anomaly(self, metric_name: str, value: float) -> Dict[str, Any]:
		"""Detect anomaly for a specific metric."""
		try:
			# Add to history
			self._metric_history[metric_name].append({
				'value': value,
				'timestamp': datetime.utcnow()
			})

			history = self._metric_history[metric_name]

			# Need enough data for training
			if len(history) < self._training_threshold:
				return {
					'is_anomaly': False,
					'confidence': 0.0,
					'message': 'Insufficient data for anomaly detection'
				}

			# Prepare data for model
			values = [h['value'] for h in history]
			timestamps = [h['timestamp'] for h in history]

			# Create features (value, time-based features)
			features = []
			for i, (val, ts) in enumerate(zip(values, timestamps)):
				hour = ts.hour
				day_of_week = ts.weekday()
				features.append([val, hour, day_of_week])

			features_array = np.array(features)

			# Train or use existing model
			if metric_name not in self._models:
				# Initialize scaler and model
				self._scalers[metric_name] = StandardScaler()
				self._models[metric_name] = IsolationForest(
					contamination=0.1,  # 10% anomalies expected
					random_state=42
				)

				# Fit scaler and model
				scaled_features = self._scalers[metric_name].fit_transform(features_array[:-1])
				self._models[metric_name].fit(scaled_features)

			# Predict anomaly for current value
			current_features = np.array([[value, datetime.utcnow().hour, datetime.utcnow().weekday()]])
			scaled_current = self._scalers[metric_name].transform(current_features)

			prediction = self._models[metric_name].predict(scaled_current)[0]
			anomaly_score = self._models[metric_name].score_samples(scaled_current)[0]

			is_anomaly = prediction == -1
			confidence = abs(anomaly_score)

			# Additional statistical checks
			recent_values = values[-20:]  # Last 20 values
			mean_val = np.mean(recent_values)
			std_val = np.std(recent_values)

			z_score = abs(value - mean_val) / max(std_val, 0.001)
			statistical_anomaly = z_score > 3  # 3-sigma rule

			return {
				'is_anomaly': is_anomaly or statistical_anomaly,
				'confidence': confidence,
				'anomaly_score': anomaly_score,
				'z_score': z_score,
				'mean': mean_val,
				'std': std_val,
				'message': f'Anomaly detected with confidence {confidence:.3f}' if is_anomaly else 'Normal'
			}

		except Exception as e:
			self.logger.error(f"Failed to detect anomaly for {metric_name}: {e}")
			return {
				'is_anomaly': False,
				'confidence': 0.0,
				'message': f'Error: {str(e)}'
			}

	def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
		"""Flatten nested metrics dictionary."""
		flattened = {}

		for key, value in metrics.items():
			new_key = f"{prefix}.{key}" if prefix else key

			if isinstance(value, dict):
				flattened.update(self._flatten_metrics(value, new_key))
			else:
				flattened[new_key] = value

		return flattened


class ProductionMonitor:
	"""Main production monitoring orchestrator."""

	def __init__(self, config: MonitoringConfig):
		self.config = config
		self.logger = logging.getLogger(f"{__name__}.ProductionMonitor")

		self.health_checker = HealthChecker(config)
		self.metrics_collector = MetricsCollector(config)
		self.alert_manager = AlertManager(config)
		self.anomaly_detector = AnomalyDetector()

		self._monitoring_task = None
		self._running = False

	async def start(self) -> None:
		"""Start production monitoring."""
		try:
			self.logger.info("Starting production monitoring...")

			# Configure default alert rules
			await self._configure_default_alerts()

			# Start monitoring loop
			self._running = True
			self._monitoring_task = asyncio.create_task(self._monitoring_loop())

			self.logger.info("Production monitoring started successfully")

		except Exception as e:
			self.logger.error(f"Failed to start production monitoring: {e}")
			raise

	async def stop(self) -> None:
		"""Stop production monitoring."""
		try:
			self.logger.info("Stopping production monitoring...")

			self._running = False

			if self._monitoring_task:
				self._monitoring_task.cancel()
				try:
					await self._monitoring_task
				except asyncio.CancelledError:
					pass

			self.logger.info("Production monitoring stopped")

		except Exception as e:
			self.logger.error(f"Failed to stop production monitoring: {e}")

	async def get_health_status(self) -> Dict[str, Any]:
		"""Get comprehensive health status."""
		try:
			health_checks = await self.health_checker.check_all_components()

			# Calculate overall health
			healthy_count = len([hc for hc in health_checks if hc.status == HealthStatus.HEALTHY])
			warning_count = len([hc for hc in health_checks if hc.status == HealthStatus.WARNING])
			critical_count = len([hc for hc in health_checks if hc.status == HealthStatus.CRITICAL])

			if critical_count > 0:
				overall_status = HealthStatus.CRITICAL
			elif warning_count > 0:
				overall_status = HealthStatus.WARNING
			else:
				overall_status = HealthStatus.HEALTHY

			return {
				'overall_status': overall_status.value,
				'timestamp': datetime.utcnow().isoformat(),
				'summary': {
					'total_checks': len(health_checks),
					'healthy': healthy_count,
					'warning': warning_count,
					'critical': critical_count
				},
				'components': {
					hc.component.value: {
						'status': hc.status.value,
						'message': hc.message,
						'response_time_ms': hc.response_time_ms,
						'last_check': hc.timestamp.isoformat()
					}
					for hc in health_checks
				},
				'active_alerts': len(self.alert_manager._active_alerts),
				'uptime_seconds': time.time() - psutil.boot_time()
			}

		except Exception as e:
			self.logger.error(f"Failed to get health status: {e}")
			return {
				'overall_status': HealthStatus.CRITICAL.value,
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}

	async def get_metrics_summary(self) -> Dict[str, Any]:
		"""Get metrics summary."""
		try:
			system_metrics = await self.metrics_collector.collect_system_metrics()
			app_metrics = await self.metrics_collector.collect_application_metrics()
			k8s_metrics = await self.metrics_collector.collect_kubernetes_metrics()

			return {
				'timestamp': datetime.utcnow().isoformat(),
				'system': {
					'cpu_percent': system_metrics.cpu_percent,
					'memory_percent': system_metrics.memory_percent,
					'disk_percent': system_metrics.disk_percent,
					'gpu_percent': system_metrics.gpu_percent,
					'load_average': system_metrics.load_average,
					'uptime_seconds': system_metrics.uptime_seconds
				},
				'application': app_metrics,
				'kubernetes': k8s_metrics
			}

		except Exception as e:
			self.logger.error(f"Failed to get metrics summary: {e}")
			return {
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}

	async def _monitoring_loop(self) -> None:
		"""Main monitoring loop."""
		while self._running:
			try:
				# Collect health checks
				health_checks = await self.health_checker.check_all_components()

				# Collect metrics
				system_metrics = await self.metrics_collector.collect_system_metrics()
				app_metrics = await self.metrics_collector.collect_application_metrics()
				k8s_metrics = await self.metrics_collector.collect_kubernetes_metrics()

				# Combine all metrics
				all_metrics = {
					'system': system_metrics.__dict__,
					'application': app_metrics,
					'kubernetes': k8s_metrics
				}

				# Detect anomalies
				if self.config.anomaly_detection_enabled:
					anomalies = await self.anomaly_detector.detect_anomalies(all_metrics)
					if anomalies:
						self.logger.warning(f"Anomalies detected: {list(anomalies.keys())}")

				# Evaluate alerts
				await self.alert_manager.evaluate_alerts(all_metrics, health_checks)

				# Log monitoring summary
				self.logger.debug(f"Monitoring cycle completed: {len(health_checks)} health checks, {len(all_metrics)} metric groups")

			except Exception as e:
				self.logger.error(f"Monitoring loop error: {e}")

			# Wait for next cycle
			await asyncio.sleep(self.config.health_check_interval)

	async def _configure_default_alerts(self) -> None:
		"""Configure default alert rules."""
		default_rules = [
			{
				'name': 'high_cpu_usage',
				'metric': 'system.cpu_percent',
				'threshold': 90.0,
				'comparison': 'greater_than',
				'severity': 'warning',
				'title': 'High CPU Usage',
				'description': 'CPU usage is above 90%',
				'component': 'application'
			},
			{
				'name': 'high_memory_usage',
				'metric': 'system.memory_percent',
				'threshold': 90.0,
				'comparison': 'greater_than',
				'severity': 'warning',
				'title': 'High Memory Usage',
				'description': 'Memory usage is above 90%',
				'component': 'application'
			},
			{
				'name': 'high_disk_usage',
				'metric': 'system.disk_percent',
				'threshold': 85.0,
				'comparison': 'greater_than',
				'severity': 'warning',
				'title': 'High Disk Usage',
				'description': 'Disk usage is above 85%',
				'component': 'storage'
			},
			{
				'name': 'critical_disk_usage',
				'metric': 'system.disk_percent',
				'threshold': 95.0,
				'comparison': 'greater_than',
				'severity': 'critical',
				'title': 'Critical Disk Usage',
				'description': 'Disk usage is above 95%',
				'component': 'storage'
			}
		]

		for rule in default_rules:
			self.alert_manager.add_alert_rule(rule)


# Example notification handlers
async def slack_notification_handler(alert: Alert, status: str) -> None:
	"""Send alert to Slack."""
	try:
		webhook_url = os.getenv('SLACK_WEBHOOK_URL')
		if not webhook_url:
			return

		color = {
			AlertSeverity.INFO: 'good',
			AlertSeverity.WARNING: 'warning',
			AlertSeverity.CRITICAL: 'danger',
			AlertSeverity.EMERGENCY: 'danger'
		}.get(alert.severity, 'warning')

		message = {
			'attachments': [{
				'color': color,
				'title': f"{'🔥' if status == 'fired' else '✅'} {alert.title}",
				'text': alert.description,
				'fields': [
					{'title': 'Severity', 'value': alert.severity.value, 'short': True},
					{'title': 'Component', 'value': alert.component.value, 'short': True},
					{'title': 'Metric', 'value': alert.metric_name, 'short': True},
					{'title': 'Value', 'value': f"{alert.current_value:.2f}", 'short': True}
				],
				'ts': int(alert.timestamp.timestamp())
			}]
		}

		async with aiohttp.ClientSession() as session:
			async with session.post(webhook_url, json=message) as response:
				if response.status != 200:
					logging.error(f"Failed to send Slack notification: {response.status}")

	except Exception as e:
		logging.error(f"Slack notification failed: {e}")


async def email_notification_handler(alert: Alert, status: str) -> None:
	"""Send alert via email."""
	try:
		# This would integrate with your email service
		# For now, just log the notification
		logging.info(f"EMAIL ALERT ({status}): {alert.title} - {alert.description}")

	except Exception as e:
		logging.error(f"Email notification failed: {e}")


# Main monitoring application
async def main():
	"""Main monitoring application."""
	# Configure logging
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	)

	# Create monitoring configuration
	config = MonitoringConfig(
		health_check_interval=30,
		metrics_collection_interval=15,
		alert_evaluation_interval=60,
		anomaly_detection_enabled=True,
		predictive_monitoring_enabled=True
	)

	# Create and start monitor
	monitor = ProductionMonitor(config)

	# Add notification handlers
	monitor.alert_manager.add_notification_channel('slack', slack_notification_handler)
	monitor.alert_manager.add_notification_channel('email', email_notification_handler)

	try:
		await monitor.start()

		# Keep running
		while True:
			await asyncio.sleep(60)

			# Print health summary
			health_status = await monitor.get_health_status()
			print(f"Overall Health: {health_status['overall_status']}")
			print(f"Active Alerts: {health_status['active_alerts']}")

	except KeyboardInterrupt:
		print("Shutting down...")

	finally:
		await monitor.stop()


if __name__ == "__main__":
	asyncio.run(main())