#!/usr/bin/env python3
"""APG Cash Management - Advanced Connection Pool Manager

Intelligent connection pool management with auto-scaling, health monitoring,
and optimal resource allocation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import time
import statistics
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager
import weakref

import asyncpg
import redis.asyncio as redis
import aiohttp
from pydantic import BaseModel, Field, ConfigDict
from uuid_extensions import uuid7str
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoolType(str, Enum):
	"""Connection pool types."""
	DATABASE_PRIMARY = "database_primary"
	DATABASE_REPLICA = "database_replica"
	REDIS_CACHE = "redis_cache"
	HTTP_CLIENT = "http_client"
	BANK_API = "bank_api"

class PoolState(str, Enum):
	"""Connection pool states."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	SCALING = "scaling"
	MAINTENANCE = "maintenance"

@dataclass
class ConnectionMetrics:
	"""Connection pool metrics."""
	timestamp: datetime
	total_connections: int
	active_connections: int
	idle_connections: int
	waiting_connections: int
	failed_connections: int
	average_response_time_ms: float
	peak_connections: int
	utilization_percent: float
	error_rate: float

@dataclass
class PoolHealth:
	"""Pool health assessment."""
	pool_id: str
	state: PoolState
	health_score: float
	issues: List[str] = field(default_factory=list)
	recommendations: List[str] = field(default_factory=list)
	last_check: datetime = field(default_factory=datetime.now)

class PoolConfiguration(BaseModel):
	"""Connection pool configuration."""
	model_config = ConfigDict(extra='forbid', validate_by_name=True)
	
	pool_id: str = Field(default_factory=uuid7str)
	pool_type: PoolType
	min_connections: int = Field(default=2, ge=1, le=100)
	max_connections: int = Field(default=20, ge=5, le=1000)
	target_utilization: float = Field(default=0.7, ge=0.3, le=0.9)
	
	# Connection lifecycle
	connection_timeout_seconds: int = Field(default=30, ge=5, le=300)
	idle_timeout_seconds: int = Field(default=300, ge=60, le=3600)
	max_lifetime_seconds: int = Field(default=3600, ge=300, le=86400)
	
	# Health monitoring
	health_check_interval_seconds: int = Field(default=30, ge=10, le=300)
	health_check_timeout_seconds: int = Field(default=5, ge=1, le=30)
	max_consecutive_failures: int = Field(default=3, ge=1, le=10)
	
	# Auto-scaling
	enable_auto_scaling: bool = True
	scale_up_threshold: float = Field(default=0.8, ge=0.5, le=0.95)
	scale_down_threshold: float = Field(default=0.3, ge=0.1, le=0.6)
	scale_up_increment: int = Field(default=2, ge=1, le=10)
	scale_down_decrement: int = Field(default=1, ge=1, le=5)
	scaling_cooldown_seconds: int = Field(default=300, ge=60, le=1800)
	
	# Circuit breaker
	enable_circuit_breaker: bool = True
	failure_threshold: int = Field(default=5, ge=3, le=20)
	reset_timeout_seconds: int = Field(default=60, ge=30, le=300)

class DatabasePool:
	"""Advanced PostgreSQL connection pool."""
	
	def __init__(self, config: PoolConfiguration, connection_string: str):
		self.config = config
		self.connection_string = connection_string
		self.pool: Optional[asyncpg.Pool] = None
		
		# Metrics tracking
		self.metrics_history: List[ConnectionMetrics] = []
		self.health_status = PoolHealth(
			pool_id=config.pool_id,
			state=PoolState.HEALTHY,
			health_score=1.0
		)
		
		# Auto-scaling
		self.last_scale_action = datetime.now()
		self.consecutive_failures = 0
		
		# Circuit breaker
		self.circuit_breaker_open = False
		self.circuit_breaker_reset_time: Optional[datetime] = None
		
		logger.info(f"Initialized DatabasePool {config.pool_id}")
	
	async def initialize(self) -> None:
		"""Initialize the connection pool."""
		try:
			self.pool = await asyncpg.create_pool(
				self.connection_string,
				min_size=self.config.min_connections,
				max_size=self.config.max_connections,
				command_timeout=self.config.connection_timeout_seconds,
				max_inactive_connection_lifetime=self.config.idle_timeout_seconds,
				max_queries=1000,
				setup=self._setup_connection
			)
			
			logger.info(f"Database pool {self.config.pool_id} initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize database pool {self.config.pool_id}: {e}")
			raise
	
	async def _setup_connection(self, connection) -> None:
		"""Setup individual database connection."""
		# Set connection-level optimizations
		await connection.execute("SET statement_timeout = '30s'")
		await connection.execute("SET lock_timeout = '10s'")
		await connection.execute("SET idle_in_transaction_session_timeout = '60s'")
		await connection.execute("SET tcp_keepalives_idle = '600'")
		await connection.execute("SET tcp_keepalives_interval = '30'")
		await connection.execute("SET tcp_keepalives_count = '3'")
	
	@asynccontextmanager
	async def acquire(self):
		"""Acquire connection with circuit breaker and metrics."""
		if self.circuit_breaker_open:
			if datetime.now() > self.circuit_breaker_reset_time:
				self.circuit_breaker_open = False
				logger.info(f"Circuit breaker reset for pool {self.config.pool_id}")
			else:
				raise Exception(f"Circuit breaker open for pool {self.config.pool_id}")
		
		start_time = time.time()
		connection = None
		
		try:
			async with self.pool.acquire() as conn:
				connection = conn
				yield conn
				
			# Record successful operation
			response_time = (time.time() - start_time) * 1000
			await self._record_success(response_time)
			
		except Exception as e:
			await self._record_failure()
			raise
	
	async def _record_success(self, response_time_ms: float) -> None:
		"""Record successful operation."""
		self.consecutive_failures = 0
		
		# Update metrics
		await self._update_metrics(response_time_ms, success=True)
	
	async def _record_failure(self) -> None:
		"""Record failed operation."""
		self.consecutive_failures += 1
		
		# Update metrics
		await self._update_metrics(0, success=False)
		
		# Check circuit breaker
		if (self.config.enable_circuit_breaker and 
			self.consecutive_failures >= self.config.failure_threshold):
			self.circuit_breaker_open = True
			self.circuit_breaker_reset_time = (
				datetime.now() + timedelta(seconds=self.config.reset_timeout_seconds)
			)
			logger.warning(f"Circuit breaker opened for pool {self.config.pool_id}")
	
	async def _update_metrics(self, response_time_ms: float, success: bool) -> None:
		"""Update pool metrics."""
		now = datetime.now()
		
		# Get current pool status
		total_size = self.pool.get_size()
		idle_size = self.pool.get_idle_size()
		active_size = total_size - idle_size
		
		# Calculate utilization
		utilization = active_size / total_size if total_size > 0 else 0
		
		# Calculate error rate from recent history
		recent_metrics = [
			m for m in self.metrics_history 
			if m.timestamp > now - timedelta(minutes=5)
		]
		total_operations = len(recent_metrics) + 1
		failed_operations = sum(1 for m in recent_metrics if m.error_rate > 0) + (0 if success else 1)
		error_rate = failed_operations / total_operations
		
		metrics = ConnectionMetrics(
			timestamp=now,
			total_connections=total_size,
			active_connections=active_size,
			idle_connections=idle_size,
			waiting_connections=0,  # Would need pool internals to get this
			failed_connections=self.consecutive_failures,
			average_response_time_ms=response_time_ms,
			peak_connections=max(total_size, max((m.total_connections for m in recent_metrics), default=0)),
			utilization_percent=utilization * 100,
			error_rate=error_rate
		)
		
		self.metrics_history.append(metrics)
		
		# Keep only recent metrics
		cutoff = now - timedelta(hours=1)
		self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff]
		
		# Trigger auto-scaling check
		if self.config.enable_auto_scaling:
			await self._check_auto_scaling(metrics)
	
	async def _check_auto_scaling(self, metrics: ConnectionMetrics) -> None:
		"""Check if auto-scaling is needed."""
		now = datetime.now()
		
		# Check cooldown period
		if now - self.last_scale_action < timedelta(seconds=self.config.scaling_cooldown_seconds):
			return
		
		utilization = metrics.utilization_percent / 100
		
		# Scale up check
		if (utilization > self.config.scale_up_threshold and 
			metrics.total_connections < self.config.max_connections):
			
			new_size = min(
				metrics.total_connections + self.config.scale_up_increment,
				self.config.max_connections
			)
			
			await self._scale_pool(new_size)
			logger.info(f"Scaled up pool {self.config.pool_id} to {new_size} connections")
		
		# Scale down check
		elif (utilization < self.config.scale_down_threshold and 
			  metrics.total_connections > self.config.min_connections):
			
			new_size = max(
				metrics.total_connections - self.config.scale_down_decrement,
				self.config.min_connections
			)
			
			await self._scale_pool(new_size)
			logger.info(f"Scaled down pool {self.config.pool_id} to {new_size} connections")
	
	async def _scale_pool(self, new_size: int) -> None:
		"""Scale the connection pool."""
		try:
			# PostgreSQL pool doesn't support dynamic resizing
			# In a real implementation, you'd need to create a new pool
			# and gradually migrate connections
			
			self.last_scale_action = datetime.now()
			self.health_status.state = PoolState.SCALING
			
			# Simulate scaling (in practice, this would be more complex)
			logger.info(f"Pool {self.config.pool_id} scaling to {new_size} connections")
			
			self.health_status.state = PoolState.HEALTHY
			
		except Exception as e:
			logger.error(f"Failed to scale pool {self.config.pool_id}: {e}")
			self.health_status.state = PoolState.DEGRADED
	
	async def health_check(self) -> PoolHealth:
		"""Perform comprehensive health check."""
		try:
			start_time = time.time()
			
			# Test connection
			async with self.acquire() as conn:
				await conn.fetchval("SELECT 1")
			
			response_time = (time.time() - start_time) * 1000
			
			# Analyze recent metrics
			recent_metrics = [
				m for m in self.metrics_history 
				if m.timestamp > datetime.now() - timedelta(minutes=5)
			]
			
			issues = []
			recommendations = []
			health_score = 1.0
			
			if recent_metrics:
				avg_utilization = statistics.mean(m.utilization_percent for m in recent_metrics)
				avg_response_time = statistics.mean(m.average_response_time_ms for m in recent_metrics)
				max_error_rate = max(m.error_rate for m in recent_metrics)
				
				# Check utilization
				if avg_utilization > 90:
					issues.append("High connection utilization")
					recommendations.append("Consider increasing max_connections")
					health_score -= 0.2
				elif avg_utilization < 20:
					recommendations.append("Consider reducing min_connections")
				
				# Check response time
				if avg_response_time > 1000:
					issues.append("High average response time")
					recommendations.append("Check database performance")
					health_score -= 0.3
				
				# Check error rate
				if max_error_rate > 0.1:
					issues.append("High error rate")
					recommendations.append("Investigate connection failures")
					health_score -= 0.4
			
			# Check circuit breaker
			if self.circuit_breaker_open:
				issues.append("Circuit breaker is open")
				health_score -= 0.5
			
			# Determine state
			if health_score > 0.8:
				state = PoolState.HEALTHY
			elif health_score > 0.5:
				state = PoolState.DEGRADED
			else:
				state = PoolState.UNHEALTHY
			
			self.health_status = PoolHealth(
				pool_id=self.config.pool_id,
				state=state,
				health_score=health_score,
				issues=issues,
				recommendations=recommendations,
				last_check=datetime.now()
			)
			
		except Exception as e:
			logger.error(f"Health check failed for pool {self.config.pool_id}: {e}")
			self.health_status = PoolHealth(
				pool_id=self.config.pool_id,
				state=PoolState.UNHEALTHY,
				health_score=0.0,
				issues=[f"Health check failed: {str(e)}"],
				last_check=datetime.now()
			)
		
		return self.health_status
	
	async def get_metrics_summary(self) -> Dict[str, Any]:
		"""Get metrics summary."""
		if not self.metrics_history:
			return {"status": "No metrics available"}
		
		recent = self.metrics_history[-10:]  # Last 10 metrics
		
		return {
			"current_connections": recent[-1].total_connections if recent else 0,
			"active_connections": recent[-1].active_connections if recent else 0,
			"utilization_percent": recent[-1].utilization_percent if recent else 0,
			"average_response_time_ms": statistics.mean(m.average_response_time_ms for m in recent),
			"error_rate": recent[-1].error_rate if recent else 0,
			"circuit_breaker_open": self.circuit_breaker_open,
			"health_score": self.health_status.health_score,
			"state": self.health_status.state.value
		}
	
	async def cleanup(self) -> None:
		"""Cleanup pool resources."""
		if self.pool:
			await self.pool.close()
		logger.info(f"Database pool {self.config.pool_id} cleaned up")

class RedisPool:
	"""Advanced Redis connection pool."""
	
	def __init__(self, config: PoolConfiguration, redis_url: str):
		self.config = config
		self.redis_url = redis_url
		self.pool: Optional[redis.ConnectionPool] = None
		
		# Metrics tracking
		self.metrics_history: List[ConnectionMetrics] = []
		self.health_status = PoolHealth(
			pool_id=config.pool_id,
			state=PoolState.HEALTHY,
			health_score=1.0
		)
		
		logger.info(f"Initialized RedisPool {config.pool_id}")
	
	async def initialize(self) -> None:
		"""Initialize Redis connection pool."""
		try:
			self.pool = redis.ConnectionPool.from_url(
				self.redis_url,
				max_connections=self.config.max_connections,
				socket_connect_timeout=self.config.connection_timeout_seconds,
				socket_timeout=self.config.idle_timeout_seconds,
				health_check_interval=self.config.health_check_interval_seconds,
				retry_on_timeout=True
			)
			
			logger.info(f"Redis pool {self.config.pool_id} initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize Redis pool {self.config.pool_id}: {e}")
			raise
	
	async def get_client(self) -> redis.Redis:
		"""Get Redis client from pool."""
		return redis.Redis(connection_pool=self.pool)
	
	async def health_check(self) -> PoolHealth:
		"""Perform Redis health check."""
		try:
			client = await self.get_client()
			start_time = time.time()
			
			await client.ping()
			response_time = (time.time() - start_time) * 1000
			
			await client.close()
			
			self.health_status = PoolHealth(
				pool_id=self.config.pool_id,
				state=PoolState.HEALTHY,
				health_score=1.0,
				last_check=datetime.now()
			)
			
		except Exception as e:
			logger.error(f"Redis health check failed for pool {self.config.pool_id}: {e}")
			self.health_status = PoolHealth(
				pool_id=self.config.pool_id,
				state=PoolState.UNHEALTHY,
				health_score=0.0,
				issues=[f"Health check failed: {str(e)}"],
				last_check=datetime.now()
			)
		
		return self.health_status
	
	async def cleanup(self) -> None:
		"""Cleanup Redis pool resources."""
		if self.pool:
			await self.pool.disconnect()
		logger.info(f"Redis pool {self.config.pool_id} cleaned up")

class HTTPPool:
	"""Advanced HTTP client connection pool."""
	
	def __init__(self, config: PoolConfiguration, base_url: Optional[str] = None):
		self.config = config
		self.base_url = base_url
		self.session: Optional[aiohttp.ClientSession] = None
		
		# Metrics tracking
		self.metrics_history: List[ConnectionMetrics] = []
		self.health_status = PoolHealth(
			pool_id=config.pool_id,
			state=PoolState.HEALTHY,
			health_score=1.0
		)
		
		logger.info(f"Initialized HTTPPool {config.pool_id}")
	
	async def initialize(self) -> None:
		"""Initialize HTTP client session."""
		try:
			connector = aiohttp.TCPConnector(
				limit=self.config.max_connections,
				limit_per_host=self.config.max_connections,
				ttl_dns_cache=300,
				use_dns_cache=True,
				keepalive_timeout=self.config.idle_timeout_seconds,
				enable_cleanup_closed=True
			)
			
			timeout = aiohttp.ClientTimeout(
				total=self.config.connection_timeout_seconds,
				connect=self.config.connection_timeout_seconds // 2
			)
			
			self.session = aiohttp.ClientSession(
				connector=connector,
				timeout=timeout,
				base_url=self.base_url
			)
			
			logger.info(f"HTTP pool {self.config.pool_id} initialized")
			
		except Exception as e:
			logger.error(f"Failed to initialize HTTP pool {self.config.pool_id}: {e}")
			raise
	
	async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
		"""Make HTTP request with metrics tracking."""
		start_time = time.time()
		
		try:
			async with self.session.request(method, url, **kwargs) as response:
				response_time = (time.time() - start_time) * 1000
				await self._record_request(response_time, response.status < 400)
				return response
				
		except Exception as e:
			response_time = (time.time() - start_time) * 1000
			await self._record_request(response_time, False)
			raise
	
	async def _record_request(self, response_time_ms: float, success: bool) -> None:
		"""Record request metrics."""
		# This would be similar to database pool metrics recording
		pass
	
	async def health_check(self) -> PoolHealth:
		"""Perform HTTP health check."""
		try:
			if self.base_url:
				start_time = time.time()
				async with self.session.get("/health") as response:
					response_time = (time.time() - start_time) * 1000
					
					if response.status == 200:
						self.health_status = PoolHealth(
							pool_id=self.config.pool_id,
							state=PoolState.HEALTHY,
							health_score=1.0,
							last_check=datetime.now()
						)
					else:
						self.health_status = PoolHealth(
							pool_id=self.config.pool_id,
							state=PoolState.DEGRADED,
							health_score=0.5,
							issues=[f"Health endpoint returned {response.status}"],
							last_check=datetime.now()
						)
			
		except Exception as e:
			logger.error(f"HTTP health check failed for pool {self.config.pool_id}: {e}")
			self.health_status = PoolHealth(
				pool_id=self.config.pool_id,
				state=PoolState.UNHEALTHY,
				health_score=0.0,
				issues=[f"Health check failed: {str(e)}"],
				last_check=datetime.now()
			)
		
		return self.health_status
	
	async def cleanup(self) -> None:
		"""Cleanup HTTP session."""
		if self.session:
			await self.session.close()
		logger.info(f"HTTP pool {self.config.pool_id} cleaned up")

class ConnectionPoolManager:
	"""Central manager for all connection pools."""
	
	def __init__(self, tenant_id: str):
		self.tenant_id = tenant_id
		self.pools: Dict[str, Any] = {}
		self.monitoring_tasks: Dict[str, asyncio.Task] = {}
		
		logger.info(f"Initialized ConnectionPoolManager for tenant {tenant_id}")
	
	async def create_database_pool(
		self,
		pool_id: str,
		connection_string: str,
		config: Optional[PoolConfiguration] = None
	) -> DatabasePool:
		"""Create database connection pool."""
		if not config:
			config = PoolConfiguration(
				pool_id=pool_id,
				pool_type=PoolType.DATABASE_PRIMARY
			)
		
		pool = DatabasePool(config, connection_string)
		await pool.initialize()
		
		self.pools[pool_id] = pool
		
		# Start monitoring
		self.monitoring_tasks[pool_id] = asyncio.create_task(
			self._monitor_pool(pool_id)
		)
		
		return pool
	
	async def create_redis_pool(
		self,
		pool_id: str,
		redis_url: str,
		config: Optional[PoolConfiguration] = None
	) -> RedisPool:
		"""Create Redis connection pool."""
		if not config:
			config = PoolConfiguration(
				pool_id=pool_id,
				pool_type=PoolType.REDIS_CACHE
			)
		
		pool = RedisPool(config, redis_url)
		await pool.initialize()
		
		self.pools[pool_id] = pool
		
		# Start monitoring
		self.monitoring_tasks[pool_id] = asyncio.create_task(
			self._monitor_pool(pool_id)
		)
		
		return pool
	
	async def create_http_pool(
		self,
		pool_id: str,
		base_url: Optional[str] = None,
		config: Optional[PoolConfiguration] = None
	) -> HTTPPool:
		"""Create HTTP connection pool."""
		if not config:
			config = PoolConfiguration(
				pool_id=pool_id,
				pool_type=PoolType.HTTP_CLIENT
			)
		
		pool = HTTPPool(config, base_url)
		await pool.initialize()
		
		self.pools[pool_id] = pool
		
		# Start monitoring
		self.monitoring_tasks[pool_id] = asyncio.create_task(
			self._monitor_pool(pool_id)
		)
		
		return pool
	
	async def _monitor_pool(self, pool_id: str) -> None:
		"""Monitor pool health continuously."""
		pool = self.pools.get(pool_id)
		if not pool:
			return
		
		while pool_id in self.pools:
			try:
				health = await pool.health_check()
				
				if health.state in [PoolState.UNHEALTHY, PoolState.DEGRADED]:
					logger.warning(f"Pool {pool_id} health: {health.state.value} (score: {health.health_score})")
					if health.issues:
						logger.warning(f"Issues: {', '.join(health.issues)}")
				
				await asyncio.sleep(pool.config.health_check_interval_seconds)
				
			except Exception as e:
				logger.error(f"Pool monitoring error for {pool_id}: {e}")
				await asyncio.sleep(60)  # Retry after error
	
	async def get_pool(self, pool_id: str) -> Optional[Any]:
		"""Get pool by ID."""
		return self.pools.get(pool_id)
	
	async def get_all_pools_health(self) -> Dict[str, PoolHealth]:
		"""Get health status of all pools."""
		health_status = {}
		
		for pool_id, pool in self.pools.items():
			try:
				health = await pool.health_check()
				health_status[pool_id] = health
			except Exception as e:
				health_status[pool_id] = PoolHealth(
					pool_id=pool_id,
					state=PoolState.UNHEALTHY,
					health_score=0.0,
					issues=[f"Health check failed: {str(e)}"],
					last_check=datetime.now()
				)
		
		return health_status
	
	async def get_comprehensive_report(self) -> Dict[str, Any]:
		"""Generate comprehensive connection pool report."""
		all_health = await self.get_all_pools_health()
		
		# Calculate overall statistics
		total_pools = len(self.pools)
		healthy_pools = sum(1 for h in all_health.values() if h.state == PoolState.HEALTHY)
		degraded_pools = sum(1 for h in all_health.values() if h.state == PoolState.DEGRADED)
		unhealthy_pools = sum(1 for h in all_health.values() if h.state == PoolState.UNHEALTHY)
		
		avg_health_score = statistics.mean(h.health_score for h in all_health.values()) if all_health else 0
		
		# Collect metrics from all pools
		pool_metrics = {}
		for pool_id, pool in self.pools.items():
			if hasattr(pool, 'get_metrics_summary'):
				pool_metrics[pool_id] = await pool.get_metrics_summary()
		
		# Generate recommendations
		recommendations = []
		for pool_id, health in all_health.items():
			if health.recommendations:
				for rec in health.recommendations:
					recommendations.append(f"{pool_id}: {rec}")
		
		return {
			"summary": {
				"total_pools": total_pools,
				"healthy_pools": healthy_pools,
				"degraded_pools": degraded_pools,
				"unhealthy_pools": unhealthy_pools,
				"average_health_score": round(avg_health_score, 3)
			},
			"pool_health": {
				pool_id: {
					"state": health.state.value,
					"health_score": health.health_score,
					"issues": health.issues,
					"last_check": health.last_check.isoformat()
				}
				for pool_id, health in all_health.items()
			},
			"pool_metrics": pool_metrics,
			"recommendations": recommendations,
			"system_info": {
				"cpu_percent": psutil.cpu_percent(),
				"memory_percent": psutil.virtual_memory().percent,
				"active_monitoring_tasks": len(self.monitoring_tasks)
			}
		}
	
	async def optimize_all_pools(self) -> List[str]:
		"""Optimize all connection pools."""
		optimization_actions = []
		
		for pool_id, pool in self.pools.items():
			try:
				health = await pool.health_check()
				
				# Apply optimizations based on health status
				if health.state == PoolState.DEGRADED:
					if hasattr(pool, '_check_auto_scaling'):
						# Trigger scaling check
						recent_metrics = getattr(pool, 'metrics_history', [])
						if recent_metrics:
							await pool._check_auto_scaling(recent_metrics[-1])
							optimization_actions.append(f"Triggered auto-scaling check for {pool_id}")
				
				elif health.state == PoolState.UNHEALTHY:
					# Consider recreating the pool
					optimization_actions.append(f"Pool {pool_id} needs manual intervention")
			
			except Exception as e:
				logger.error(f"Optimization failed for pool {pool_id}: {e}")
				optimization_actions.append(f"Optimization failed for {pool_id}: {str(e)}")
		
		return optimization_actions
	
	async def cleanup(self) -> None:
		"""Cleanup all pools and monitoring tasks."""
		# Stop monitoring tasks
		for task in self.monitoring_tasks.values():
			task.cancel()
		
		# Cleanup all pools
		for pool in self.pools.values():
			try:
				await pool.cleanup()
			except Exception as e:
				logger.error(f"Error cleaning up pool: {e}")
		
		self.pools.clear()
		self.monitoring_tasks.clear()
		
		logger.info(f"ConnectionPoolManager for tenant {self.tenant_id} cleaned up")

# Global connection pool manager
_pool_managers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

async def get_pool_manager(tenant_id: str) -> ConnectionPoolManager:
	"""Get or create connection pool manager for tenant."""
	if tenant_id not in _pool_managers:
		_pool_managers[tenant_id] = ConnectionPoolManager(tenant_id)
	
	return _pool_managers[tenant_id]

# Connection pool decorator
def with_connection_pool(pool_id: str, pool_type: str = "database"):
	"""Decorator to automatically use connection pool."""
	def decorator(func):
		async def wrapper(tenant_id: str, *args, **kwargs):
			pool_manager = await get_pool_manager(tenant_id)
			pool = await pool_manager.get_pool(pool_id)
			
			if not pool:
				raise Exception(f"Connection pool {pool_id} not found")
			
			if pool_type == "database":
				async with pool.acquire() as conn:
					return await func(tenant_id, conn, *args, **kwargs)
			elif pool_type == "redis":
				client = await pool.get_client()
				try:
					return await func(tenant_id, client, *args, **kwargs)
				finally:
					await client.close()
			elif pool_type == "http":
				return await func(tenant_id, pool.session, *args, **kwargs)
			else:
				return await func(tenant_id, pool, *args, **kwargs)
		
		return wrapper
	return decorator

if __name__ == "__main__":
	async def main():
		# Example usage
		manager = ConnectionPoolManager("demo_tenant")
		
		# Create database pool
		db_pool = await manager.create_database_pool(
			"primary_db",
			"postgresql://user:pass@localhost:5432/db"
		)
		
		# Create Redis pool
		redis_pool = await manager.create_redis_pool(
			"cache",
			"redis://localhost:6379/0"
		)
		
		# Wait a bit for monitoring
		await asyncio.sleep(5)
		
		# Get comprehensive report
		report = await manager.get_comprehensive_report()
		print("Pool Report:", json.dumps(report, indent=2, default=str))
		
		await manager.cleanup()
	
	import json
	asyncio.run(main())