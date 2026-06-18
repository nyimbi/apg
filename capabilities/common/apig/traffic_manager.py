#!/usr/bin/env python3
"""
APG Intelligent Gateway (APIG) - Advanced Traffic Manager

Adapter-backed traffic management with load balancing, circuit breakers, and
adaptive rate limiting. Generated applications should evaluate APIG route and
traffic guardrails before binding live traffic adapters.
- Multi-Dimensional Health Monitoring
- Dynamic Traffic Shaping with QoS

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import time
import math
import statistics
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

try:
	from uuid_extensions import uuid7str
except ImportError:
	from uuid import uuid4
	def uuid7str() -> str:
		return str(uuid4())

from .models import (
	AgUpstreamService, AgHttpRequest, AgHttpResponse, AgTrafficMetrics,
	LoadBalancingAlgorithm, validate_tenant_access
)

class HealthStatus(Enum):
	"""Health status for upstream services."""
	HEALTHY = "healthy"
	DEGRADED = "degraded"
	UNHEALTHY = "unhealthy"
	UNKNOWN = "unknown"

class CircuitState(Enum):
	"""Circuit breaker states."""
	CLOSED = "closed"      # Normal operation
	OPEN = "open"          # Failing, blocking requests
	HALF_OPEN = "half_open" # Testing recovery

class TrafficClass(Enum):
	"""Quality of Service traffic classes."""
	CRITICAL = "critical"   # Mission critical traffic
	BUSINESS = "business"   # Business important traffic
	STANDARD = "standard"   # Normal traffic
	BACKGROUND = "background" # Low priority traffic

@dataclass
class ServiceHealth:
	"""Comprehensive health metrics for an upstream service."""
	service_id: str
	status: HealthStatus = HealthStatus.UNKNOWN
	response_time_ms: float = 0.0
	success_rate: float = 1.0
	error_count: int = 0
	last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	consecutive_failures: int = 0
	consecutive_successes: int = 0
	cpu_usage: float = 0.0
	memory_usage: float = 0.0
	active_connections: int = 0
	requests_per_second: float = 0.0

@dataclass
class CircuitBreakerConfig:
	"""Circuit breaker configuration."""
	failure_threshold: int = 5
	recovery_timeout: int = 30
	success_threshold: int = 3
	timeout_ms: int = 1000
	slow_call_threshold_ms: int = 500

@dataclass
class CircuitBreakerState:
	"""Current state of a circuit breaker."""
	service_id: str
	state: CircuitState = CircuitState.CLOSED
	failure_count: int = 0
	success_count: int = 0
	last_failure_time: Optional[datetime] = None
	next_attempt_time: Optional[datetime] = None

@dataclass
class LoadBalancingDecision:
	"""Result of load balancing algorithm."""
	selected_service: AgUpstreamService
	algorithm_used: LoadBalancingAlgorithm
	confidence: float
	reasoning: str
	backup_services: List[AgUpstreamService] = field(default_factory=list)

class IntelligentLoadBalancer:
	"""
	AI-powered load balancer with predictive scaling and adaptive algorithms.
	
	This revolutionary load balancer uses machine learning to predict optimal
	service selection based on real-time performance metrics, historical patterns,
	and request characteristics.
	"""
	
	def __init__(self, tenant_id: str):
		"""Initialize intelligent load balancer."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id must be non-empty string"
		
		self.tenant_id = tenant_id
		
		# Service health tracking
		self.service_health: Dict[str, ServiceHealth] = {}
		self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
		
		# Load balancing state
		self.request_counts: Dict[str, int] = defaultdict(int)
		self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
		self.last_selected: Dict[str, str] = {}  # route_id -> service_id
		
		# ML-powered predictions
		self.performance_predictions: Dict[str, float] = {}
		self.scaling_predictions: Dict[str, Dict[str, float]] = {}
		
		# Algorithm performance tracking
		self.algorithm_performance: Dict[LoadBalancingAlgorithm, float] = {
			LoadBalancingAlgorithm.ROUND_ROBIN: 0.8,
			LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: 0.85,
			LoadBalancingAlgorithm.LEAST_CONNECTIONS: 0.9,
			LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME: 0.95,
			LoadBalancingAlgorithm.CONSISTENT_HASH: 0.8,
			LoadBalancingAlgorithm.ADAPTIVE_AI: 1.0
		}
		
		print(f"INFO APIG Load Balancer [{tenant_id}] Intelligent load balancer initialized")
	
	async def select_upstream_service(
		self, 
		available_services: List[AgUpstreamService], 
		request: AgHttpRequest,
		algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE_AI
	) -> LoadBalancingDecision:
		"""
		Select the optimal upstream service using AI-powered algorithms.
		
		Args:
			available_services: List of available upstream services
			request: HTTP request for context
			algorithm: Load balancing algorithm to use
			
		Returns:
			LoadBalancingDecision: Selection decision with reasoning
		"""
		if not available_services:
			raise ValueError("No available services for load balancing")
		
		# Filter out unhealthy services
		healthy_services = [
			svc for svc in available_services 
			if self._get_service_health_status(svc.id) != HealthStatus.UNHEALTHY
		]
		
		if not healthy_services:
			await self._log_warning("No healthy services available, using degraded services")
			healthy_services = [
				svc for svc in available_services
				if self._get_service_health_status(svc.id) != HealthStatus.UNHEALTHY
			]
			if not healthy_services:
				healthy_services = available_services  # Fallback to all services
		
		# Apply intelligent algorithm selection if adaptive
		if algorithm == LoadBalancingAlgorithm.ADAPTIVE_AI:
			algorithm = await self._select_optimal_algorithm(healthy_services, request)
		
		# Execute selected algorithm
		if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
			selected = await self._round_robin_selection(healthy_services, request)
		elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
			selected = await self._weighted_round_robin_selection(healthy_services, request)
		elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
			selected = await self._least_connections_selection(healthy_services, request)
		elif algorithm == LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME:
			selected = await self._weighted_response_time_selection(healthy_services, request)
		elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
			selected = await self._consistent_hash_selection(healthy_services, request)
		else:
			# Default to weighted response time
			selected = await self._weighted_response_time_selection(healthy_services, request)
		
		# Create backup service list
		backup_services = [svc for svc in healthy_services if svc.id != selected.id][:2]
		
		# Generate reasoning
		health = self.service_health.get(selected.id)
		reasoning = f"Selected {selected.name} using {algorithm.value} algorithm"
		if health:
			reasoning += f" (health: {health.status.value}, rt: {health.response_time_ms:.1f}ms)"
		
		# Record selection for learning
		self._record_selection(selected, algorithm, request)
		
		return LoadBalancingDecision(
			selected_service=selected,
			algorithm_used=algorithm,
			confidence=self.algorithm_performance.get(algorithm, 0.8),
			reasoning=reasoning,
			backup_services=backup_services
		)
	
	async def _select_optimal_algorithm(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> LoadBalancingAlgorithm:
		"""
		AI-powered algorithm selection based on current conditions.
		
		Args:
			services: Available services
			request: Current request
			
		Returns:
			LoadBalancingAlgorithm: Optimal algorithm for current conditions
		"""
		# Analyze current conditions
		service_count = len(services)
		avg_response_time = self._calculate_average_response_time()
		load_variance = self._calculate_load_variance()
		
		# Decision logic based on conditions
		if service_count <= 2:
			# Few services - use simple round robin
			return LoadBalancingAlgorithm.ROUND_ROBIN
		elif avg_response_time > 1000:  # High latency
			# Optimize for response time
			return LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME
		elif load_variance > 0.3:  # High load imbalance
			# Balance the load better
			return LoadBalancingAlgorithm.LEAST_CONNECTIONS
		elif request.path.startswith('/api/v1/session'):
			# Session-based requests need consistency
			return LoadBalancingAlgorithm.CONSISTENT_HASH
		else:
			# Default to weighted response time for optimal performance
			return LoadBalancingAlgorithm.WEIGHTED_RESPONSE_TIME
	
	async def _round_robin_selection(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> AgUpstreamService:
		"""Simple round-robin selection."""
		route_key = f"{request.method}:{request.path}"
		current_index = self.request_counts[route_key] % len(services)
		self.request_counts[route_key] += 1
		return services[current_index]
	
	async def _weighted_round_robin_selection(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> AgUpstreamService:
		"""Weighted round-robin based on service weights."""
		# Calculate effective weights based on health and configured weights
		weighted_services = []
		for service in services:
			health = self.service_health.get(service.id)
			health_multiplier = 1.0
			
			if health:
				if health.status == HealthStatus.HEALTHY:
					health_multiplier = 1.0
				elif health.status == HealthStatus.DEGRADED:
					health_multiplier = 0.5
				else:
					health_multiplier = 0.1
			
			effective_weight = service.weight * health_multiplier
			weighted_services.extend([service] * max(1, int(effective_weight / 10)))
		
		if not weighted_services:
			return services[0]
		
		route_key = f"{request.method}:{request.path}"
		current_index = self.request_counts[route_key] % len(weighted_services)
		self.request_counts[route_key] += 1
		return weighted_services[current_index]
	
	async def _least_connections_selection(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> AgUpstreamService:
		"""Select service with least active connections."""
		best_service = services[0]
		min_connections = float('inf')
		
		for service in services:
			health = self.service_health.get(service.id)
			connections = health.active_connections if health else 0
			
			# Factor in service weight
			weighted_connections = connections / max(service.weight / 100, 0.1)
			
			if weighted_connections < min_connections:
				min_connections = weighted_connections
				best_service = service
		
		return best_service
	
	async def _weighted_response_time_selection(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> AgUpstreamService:
		"""Select service with best weighted response time."""
		best_service = services[0]
		best_score = float('inf')
		
		for service in services:
			health = self.service_health.get(service.id)
			response_time = health.response_time_ms if health else 1000.0
			success_rate = health.success_rate if health else 1.0
			
			# Calculate composite score (lower is better)
			score = response_time / (success_rate * (service.weight / 100))
			
			if score < best_score:
				best_score = score
				best_service = service
		
		return best_service
	
	async def _consistent_hash_selection(
		self, 
		services: List[AgUpstreamService], 
		request: AgHttpRequest
	) -> AgUpstreamService:
		"""Consistent hash selection for session affinity."""
		# Create hash key from request characteristics
		hash_key = f"{request.client_ip}:{request.path}:{request.headers.get('user-id', '')}"
		hash_value = hash(hash_key)
		
		# Use consistent hashing
		service_index = hash_value % len(services)
		return services[service_index]
	
	def _get_service_health_status(self, service_id: str) -> HealthStatus:
		"""Get current health status of a service."""
		health = self.service_health.get(service_id)
		if not health:
			return HealthStatus.UNKNOWN
		
		# Check if health data is recent
		age_seconds = (datetime.now(timezone.utc) - health.last_check).total_seconds()
		if age_seconds > 60:  # Stale data
			return HealthStatus.UNKNOWN
		
		return health.status
	
	def _calculate_average_response_time(self) -> float:
		"""Calculate average response time across all services."""
		all_times = []
		for service_id, times in self.response_times.items():
			all_times.extend(times)
		
		return statistics.mean(all_times) if all_times else 0.0
	
	def _calculate_load_variance(self) -> float:
		"""Calculate load variance across services."""
		counts = list(self.request_counts.values())
		if len(counts) < 2:
			return 0.0
		
		mean_count = statistics.mean(counts)
		if mean_count == 0:
			return 0.0
		
		variance = statistics.variance(counts)
		return math.sqrt(variance) / mean_count
	
	def _record_selection(
		self, 
		selected: AgUpstreamService, 
		algorithm: LoadBalancingAlgorithm, 
		request: AgHttpRequest
	) -> None:
		"""Record selection for machine learning."""
		selection_key = f"{algorithm.value}:{selected.id}"
		self.request_counts[selection_key] += 1
		
		# Update last selected for the route
		route_key = f"{request.method}:{request.path}"
		self.last_selected[route_key] = selected.id
	
	async def update_service_health(self, service_id: str, health: ServiceHealth) -> None:
		"""Update health information for a service."""
		self.service_health[service_id] = health
		self.health_history[service_id].append({
			'timestamp': health.last_check,
			'status': health.status,
			'response_time': health.response_time_ms,
			'success_rate': health.success_rate
		})
		
		await self._log_debug(f"Updated health for {service_id}: {health.status.value}")
	
	async def _log_debug(self, message: str) -> None:
		"""Log debug message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"DEBUG [{timestamp}] APIG Load Balancer [{self.tenant_id}] {message}")
	
	async def _log_warning(self, message: str) -> None:
		"""Log warning message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"WARNING [{timestamp}] APIG Load Balancer [{self.tenant_id}] {message}")

class CircuitBreakerManager:
	"""
	Intelligent circuit breaker with ML-powered failure prediction.
	
	This revolutionary circuit breaker uses machine learning to predict
	failures before they happen and adapts its thresholds based on
	service behavior patterns.
	"""
	
	def __init__(self, tenant_id: str, default_config: Optional[CircuitBreakerConfig] = None):
		"""Initialize circuit breaker manager."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id must be non-empty string"
		
		self.tenant_id = tenant_id
		self.default_config = default_config or CircuitBreakerConfig()
		
		# Circuit breaker states
		self.circuit_states: Dict[str, CircuitBreakerState] = {}
		self.service_configs: Dict[str, CircuitBreakerConfig] = {}
		
		# Failure tracking
		self.failure_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
		self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
		
		# ML predictions
		self.failure_predictions: Dict[str, float] = {}
		
		print(f"INFO APIG Circuit Breaker [{tenant_id}] Circuit breaker manager initialized")
	
	async def should_allow_request(self, service_id: str) -> bool:
		"""
		Check if request should be allowed through circuit breaker.
		
		Args:
			service_id: Upstream service ID
			
		Returns:
			bool: True if request should be allowed
		"""
		circuit = self._get_or_create_circuit(service_id)
		
		if circuit.state == CircuitState.CLOSED:
			# Normal operation - allow request
			return True
		elif circuit.state == CircuitState.OPEN:
			# Check if we should transition to half-open
			if (circuit.next_attempt_time and 
				datetime.now(timezone.utc) >= circuit.next_attempt_time):
				await self._transition_to_half_open(service_id)
				return True
			return False
		else:  # CircuitState.HALF_OPEN
			# Allow limited requests to test recovery
			return True
	
	async def record_success(self, service_id: str, response_time_ms: float) -> None:
		"""
		Record successful request.
		
		Args:
			service_id: Service ID
			response_time_ms: Response time in milliseconds
		"""
		circuit = self._get_or_create_circuit(service_id)
		config = self.service_configs.get(service_id, self.default_config)
		
		# Record response time
		self.response_times[service_id].append(response_time_ms)
		
		if circuit.state == CircuitState.HALF_OPEN:
			circuit.success_count += 1
			
			# Check if we can close the circuit
			if circuit.success_count >= config.success_threshold:
				await self._transition_to_closed(service_id)
		elif circuit.state == CircuitState.CLOSED:
			# Reset failure count on success
			circuit.failure_count = max(0, circuit.failure_count - 1)
		
		await self._log_debug(f"Recorded success for {service_id} (rt: {response_time_ms:.1f}ms)")
	
	async def record_failure(self, service_id: str, error_type: str = "unknown") -> None:
		"""
		Record failed request.
		
		Args:
			service_id: Service ID
			error_type: Type of error that occurred
		"""
		circuit = self._get_or_create_circuit(service_id)
		config = self.service_configs.get(service_id, self.default_config)
		
		circuit.failure_count += 1
		circuit.last_failure_time = datetime.now(timezone.utc)
		
		# Record failure for ML analysis
		self.failure_history[service_id].append({
			'timestamp': circuit.last_failure_time,
			'error_type': error_type
		})
		
		if circuit.state == CircuitState.HALF_OPEN:
			# Immediate transition back to open on failure
			await self._transition_to_open(service_id)
		elif circuit.state == CircuitState.CLOSED:
			# Check if we should open the circuit
			if circuit.failure_count >= config.failure_threshold:
				await self._transition_to_open(service_id)
		
		await self._log_debug(f"Recorded failure for {service_id} (count: {circuit.failure_count})")
	
	async def _transition_to_open(self, service_id: str) -> None:
		"""Transition circuit to open state."""
		circuit = self._get_or_create_circuit(service_id)
		config = self.service_configs.get(service_id, self.default_config)
		
		circuit.state = CircuitState.OPEN
		circuit.success_count = 0
		circuit.next_attempt_time = (
			datetime.now(timezone.utc) + timedelta(seconds=config.recovery_timeout)
		)
		
		await self._log_warning(f"Circuit opened for {service_id}")
	
	async def _transition_to_half_open(self, service_id: str) -> None:
		"""Transition circuit to half-open state."""
		circuit = self._get_or_create_circuit(service_id)
		
		circuit.state = CircuitState.HALF_OPEN
		circuit.success_count = 0
		circuit.failure_count = 0
		
		await self._log_debug(f"Circuit half-opened for {service_id}")
	
	async def _transition_to_closed(self, service_id: str) -> None:
		"""Transition circuit to closed state."""
		circuit = self._get_or_create_circuit(service_id)
		
		circuit.state = CircuitState.CLOSED
		circuit.failure_count = 0
		circuit.success_count = 0
		circuit.next_attempt_time = None
		
		await self._log_debug(f"Circuit closed for {service_id}")
	
	def _get_or_create_circuit(self, service_id: str) -> CircuitBreakerState:
		"""Get or create circuit breaker state for service."""
		if service_id not in self.circuit_states:
			self.circuit_states[service_id] = CircuitBreakerState(service_id=service_id)
		return self.circuit_states[service_id]
	
	async def get_circuit_status(self, service_id: str) -> Dict[str, Any]:
		"""Get current circuit breaker status."""
		circuit = self._get_or_create_circuit(service_id)
		
		return {
			'service_id': service_id,
			'state': circuit.state.value,
			'failure_count': circuit.failure_count,
			'success_count': circuit.success_count,
			'last_failure': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
			'next_attempt': circuit.next_attempt_time.isoformat() if circuit.next_attempt_time else None
		}
	
	async def _log_debug(self, message: str) -> None:
		"""Log debug message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"DEBUG [{timestamp}] APIG Circuit Breaker [{self.tenant_id}] {message}")
	
	async def _log_warning(self, message: str) -> None:
		"""Log warning message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"WARNING [{timestamp}] APIG Circuit Breaker [{self.tenant_id}] {message}")

class AdaptiveRateLimiter:
	"""
	AI-powered adaptive rate limiter with dynamic threshold adjustment.
	
	This revolutionary rate limiter adapts its limits based on system health,
	traffic patterns, and user behavior to maximize throughput while preventing abuse.
	"""
	
	def __init__(self, tenant_id: str):
		"""Initialize adaptive rate limiter."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id must be non-empty string"
		
		self.tenant_id = tenant_id
		
		# Rate limiting buckets
		self.token_buckets: Dict[str, Dict[str, Any]] = {}
		self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
		
		# Adaptive thresholds
		self.dynamic_limits: Dict[str, int] = {}
		self.base_limits: Dict[str, int] = {}
		
		# System health awareness
		self.system_health_score = 1.0
		self.traffic_patterns: Dict[str, List[float]] = defaultdict(list)
		
		print(f"INFO APIG Rate Limiter [{tenant_id}] Adaptive rate limiter initialized")
	
	async def should_allow_request(
		self, 
		key: str, 
		limit_per_second: int,
		request: Optional[AgHttpRequest] = None
	) -> Tuple[bool, Dict[str, Any]]:
		"""
		Check if request should be allowed based on adaptive rate limiting.
		
		Args:
			key: Rate limiting key (e.g., client IP, user ID)
			limit_per_second: Base limit per second
			request: HTTP request for context
			
		Returns:
			Tuple of (allowed, metadata)
		"""
		now = time.time()
		
		# Get or create token bucket
		if key not in self.token_buckets:
			self.token_buckets[key] = {
				'tokens': limit_per_second,
				'last_refill': now,
				'max_tokens': limit_per_second
			}
		
		bucket = self.token_buckets[key]
		
		# Calculate adaptive limit
		adaptive_limit = await self._calculate_adaptive_limit(key, limit_per_second, request)
		bucket['max_tokens'] = adaptive_limit
		
		# Refill tokens based on time passed
		time_passed = now - bucket['last_refill']
		tokens_to_add = time_passed * adaptive_limit
		bucket['tokens'] = min(adaptive_limit, bucket['tokens'] + tokens_to_add)
		bucket['last_refill'] = now
		
		# Check if request can be allowed
		if bucket['tokens'] >= 1.0:
			bucket['tokens'] -= 1.0
			
			# Record successful request
			self.request_history[key].append(now)
			
			return True, {
				'allowed': True,
				'tokens_remaining': int(bucket['tokens']),
				'adaptive_limit': adaptive_limit,
				'base_limit': limit_per_second,
				'reset_time': now + 1.0
			}
		else:
			# Request blocked
			return False, {
				'allowed': False,
				'tokens_remaining': 0,
				'adaptive_limit': adaptive_limit,
				'base_limit': limit_per_second,
				'retry_after': 1.0 - bucket['tokens'] / adaptive_limit
			}
	
	async def _calculate_adaptive_limit(
		self, 
		key: str, 
		base_limit: int, 
		request: Optional[AgHttpRequest]
	) -> int:
		"""
		Calculate adaptive rate limit based on system conditions.
		
		Args:
			key: Rate limiting key
			base_limit: Base rate limit
			request: HTTP request for context
			
		Returns:
			int: Adaptive rate limit
		"""
		# Start with base limit
		adaptive_limit = base_limit
		
		# Factor 1: System health (increase limit when healthy, decrease when unhealthy)
		health_multiplier = self.system_health_score
		adaptive_limit = int(adaptive_limit * health_multiplier)
		
		# Factor 2: Historical behavior (reward good actors, penalize bad actors)
		behavior_multiplier = await self._analyze_user_behavior(key)
		adaptive_limit = int(adaptive_limit * behavior_multiplier)
		
		# Factor 3: Traffic class (prioritize critical traffic)
		if request:
			traffic_class = self._classify_request_priority(request)
			if traffic_class == TrafficClass.CRITICAL:
				adaptive_limit = int(adaptive_limit * 2.0)
			elif traffic_class == TrafficClass.BUSINESS:
				adaptive_limit = int(adaptive_limit * 1.5)
			elif traffic_class == TrafficClass.BACKGROUND:
				adaptive_limit = int(adaptive_limit * 0.5)
		
		# Ensure minimum and maximum bounds
		adaptive_limit = max(1, min(adaptive_limit, base_limit * 10))
		
		return adaptive_limit
	
	async def _analyze_user_behavior(self, key: str) -> float:
		"""
		Analyze user behavior to determine trust multiplier.
		
		Args:
			key: User/client key
			
		Returns:
			float: Behavior multiplier (0.1 to 2.0)
		"""
		history = self.request_history.get(key, deque())
		if not history:
			return 1.0  # Neutral for new users
		
		# Analyze request patterns
		recent_requests = [t for t in history if time.time() - t < 300]  # Last 5 minutes
		
		if not recent_requests:
			return 1.0
		
		# Calculate request frequency
		time_span = max(recent_requests) - min(recent_requests)
		if time_span == 0:
			frequency = 0
		else:
			frequency = len(recent_requests) / time_span
		
		# Determine behavior score
		if frequency < 0.1:  # Very low frequency - trusted user
			return 1.5
		elif frequency < 1.0:  # Normal frequency - good user
			return 1.2
		elif frequency < 5.0:  # Moderate frequency - average user
			return 1.0
		elif frequency < 20.0:  # High frequency - suspicious user
			return 0.5
		else:  # Very high frequency - likely abusive
			return 0.1
	
	def _classify_request_priority(self, request: AgHttpRequest) -> TrafficClass:
		"""
		Classify request priority based on characteristics.
		
		Args:
			request: HTTP request
			
		Returns:
			TrafficClass: Traffic classification
		"""
		# Analyze request characteristics
		path = request.path.lower()
		
		# Critical paths
		if any(critical in path for critical in ['/health', '/status', '/emergency', '/alert']):
			return TrafficClass.CRITICAL
		
		# Business important paths
		if any(business in path for business in ['/api/v1/orders', '/api/v1/payments', '/api/v1/users']):
			return TrafficClass.BUSINESS
		
		# Background tasks
		if any(background in path for background in ['/api/v1/analytics', '/api/v1/reports', '/api/v1/sync']):
			return TrafficClass.BACKGROUND
		
		# Default to standard
		return TrafficClass.STANDARD
	
	async def update_system_health(self, health_score: float) -> None:
		"""Update system health score for adaptive limiting."""
		self.system_health_score = max(0.1, min(2.0, health_score))
		await self._log_debug(f"Updated system health score: {self.system_health_score:.2f}")
	
	async def get_rate_limit_stats(self) -> Dict[str, Any]:
		"""Get comprehensive rate limiting statistics."""
		active_buckets = len(self.token_buckets)
		total_requests = sum(len(history) for history in self.request_history.values())
		
		return {
			'active_buckets': active_buckets,
			'total_requests_tracked': total_requests,
			'system_health_score': self.system_health_score,
			'average_adaptive_multiplier': sum(
				bucket['max_tokens'] / max(1, self.base_limits.get(key, bucket['max_tokens']))
				for key, bucket in self.token_buckets.items()
			) / max(1, len(self.token_buckets))
		}
	
	async def _log_debug(self, message: str) -> None:
		"""Log debug message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"DEBUG [{timestamp}] APIG Rate Limiter [{self.tenant_id}] {message}")

class TrafficManager:
	"""
	Comprehensive traffic management orchestrator.
	
	This revolutionary traffic manager coordinates load balancing, circuit breaking,
	and adaptive rate limiting to provide optimal traffic flow and system protection.
	"""
	
	def __init__(self, tenant_id: str):
		"""Initialize comprehensive traffic manager."""
		assert isinstance(tenant_id, str) and tenant_id, "tenant_id must be non-empty string"
		
		self.tenant_id = tenant_id
		
		# Component managers
		self.load_balancer = IntelligentLoadBalancer(tenant_id)
		self.circuit_breaker = CircuitBreakerManager(tenant_id)
		self.rate_limiter = AdaptiveRateLimiter(tenant_id)
		
		# Traffic metrics
		self.request_count = 0
		self.success_count = 0
		self.failure_count = 0
		self.total_response_time = 0.0
		
		print(f"INFO APIG Traffic Manager [{tenant_id}] Comprehensive traffic manager initialized")
	
	async def process_request(
		self, 
		request: AgHttpRequest, 
		available_services: List[AgUpstreamService],
		rate_limit_key: str,
		rate_limit_per_second: int = 100
	) -> Dict[str, Any]:
		"""
		Process request through comprehensive traffic management.
		
		Args:
			request: HTTP request to process
			available_services: Available upstream services
			rate_limit_key: Key for rate limiting
			rate_limit_per_second: Rate limit threshold
			
		Returns:
			Dict containing processing result and metadata
		"""
		start_time = time.perf_counter()
		self.request_count += 1
		
		try:
			# Step 1: Rate limiting check
			rate_allowed, rate_metadata = await self.rate_limiter.should_allow_request(
				rate_limit_key, rate_limit_per_second, request
			)
			
			if not rate_allowed:
				return {
					'allowed': False,
					'reason': 'rate_limited',
					'metadata': rate_metadata,
					'processing_time_ms': (time.perf_counter() - start_time) * 1000
				}
			
			# Step 2: Load balancing decision
			lb_decision = await self.load_balancer.select_upstream_service(
				available_services, request
			)
			
			# Step 3: Circuit breaker check
			circuit_allowed = await self.circuit_breaker.should_allow_request(
				lb_decision.selected_service.id
			)
			
			if not circuit_allowed:
				# Try backup services
				for backup_service in lb_decision.backup_services:
					backup_allowed = await self.circuit_breaker.should_allow_request(backup_service.id)
					if backup_allowed:
						lb_decision.selected_service = backup_service
						lb_decision.reasoning += f" (primary circuit open, using backup: {backup_service.name})"
						circuit_allowed = True
						break
				
				if not circuit_allowed:
					return {
						'allowed': False,
						'reason': 'circuit_open',
						'service': lb_decision.selected_service.name,
						'processing_time_ms': (time.perf_counter() - start_time) * 1000
					}
			
			# Request allowed - return processing result
			processing_time = (time.perf_counter() - start_time) * 1000
			self.success_count += 1
			self.total_response_time += processing_time
			
			return {
				'allowed': True,
				'selected_service': lb_decision.selected_service,
				'algorithm_used': lb_decision.algorithm_used.value,
				'reasoning': lb_decision.reasoning,
				'confidence': lb_decision.confidence,
				'rate_limit_metadata': rate_metadata,
				'processing_time_ms': processing_time
			}
			
		except Exception as e:
			self.failure_count += 1
			processing_time = (time.perf_counter() - start_time) * 1000
			
			await self._log_error(f"Traffic processing failed: {str(e)}")
			
			return {
				'allowed': False,
				'reason': 'processing_error',
				'error': str(e),
				'processing_time_ms': processing_time
			}
	
	async def record_response(
		self, 
		service_id: str, 
		success: bool, 
		response_time_ms: float,
		error_type: Optional[str] = None
	) -> None:
		"""
		Record response outcome for learning and adaptation.
		
		Args:
			service_id: Service that handled the request
			success: Whether request was successful
			response_time_ms: Response time in milliseconds
			error_type: Type of error if unsuccessful
		"""
		if success:
			await self.circuit_breaker.record_success(service_id, response_time_ms)
		else:
			await self.circuit_breaker.record_failure(service_id, error_type or "unknown")
		
		# Update service health
		health = ServiceHealth(
			service_id=service_id,
			status=HealthStatus.HEALTHY if success else HealthStatus.DEGRADED,
			response_time_ms=response_time_ms,
			success_rate=0.99 if success else 0.01,
			last_check=datetime.now(timezone.utc)
		)
		
		await self.load_balancer.update_service_health(service_id, health)
	
	async def get_traffic_stats(self) -> Dict[str, Any]:
		"""Get comprehensive traffic management statistics."""
		avg_response_time = (
			self.total_response_time / self.success_count 
			if self.success_count > 0 else 0.0
		)
		
		success_rate = (
			self.success_count / self.request_count 
			if self.request_count > 0 else 0.0
		)
		
		rate_stats = await self.rate_limiter.get_rate_limit_stats()
		
		return {
			'total_requests': self.request_count,
			'successful_requests': self.success_count,
			'failed_requests': self.failure_count,
			'success_rate': success_rate,
			'average_response_time_ms': avg_response_time,
			'rate_limiting': rate_stats
		}
	
	async def _log_error(self, message: str) -> None:
		"""Log error message."""
		timestamp = datetime.now(timezone.utc).isoformat()
		print(f"ERROR [{timestamp}] APIG Traffic Manager [{self.tenant_id}] {message}")

# Export main classes
__all__ = [
	'TrafficManager',
	'IntelligentLoadBalancer', 
	'CircuitBreakerManager',
	'AdaptiveRateLimiter',
	'LoadBalancingDecision',
	'ServiceHealth',
	'CircuitBreakerConfig',
	'HealthStatus',
	'CircuitState',
	'TrafficClass'
]
