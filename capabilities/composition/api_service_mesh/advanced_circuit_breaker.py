"""
APG API Service Mesh - Advanced Circuit Breaker Engine

Hystrix-style circuit breaker with advanced failure detection, 
adaptive thresholds, and intelligent recovery mechanisms.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import time
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics

import httpx
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from uuid_extensions import uuid7str

from .models import SMService, SMEndpoint, SMMetrics, SMAlert


class CircuitState(str, Enum):
	"""Circuit breaker states."""
	CLOSED = "closed"        # Normal operation
	OPEN = "open"           # Circuit is open, failing fast
	HALF_OPEN = "half_open" # Testing if service recovered


class FailureType(str, Enum):
	"""Types of failures detected."""
	TIMEOUT = "timeout"
	CONNECTION_ERROR = "connection_error"
	HTTP_ERROR = "http_error"
	LATENCY_SPIKE = "latency_spike"
	RATE_LIMIT = "rate_limit"
	SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass
class RequestResult:
	"""Request execution result."""
	success: bool
	duration_ms: float
	status_code: Optional[int] = None
	error_type: Optional[FailureType] = None
	timestamp: float = field(default_factory=time.time)


@dataclass
class CircuitMetrics:
	"""Circuit breaker metrics."""
	total_requests: int = 0
	failed_requests: int = 0
	success_requests: int = 0
	rejected_requests: int = 0
	timeout_requests: int = 0
	mean_response_time: float = 0.0
	p95_response_time: float = 0.0
	p99_response_time: float = 0.0
	error_percentage: float = 0.0
	throughput_per_second: float = 0.0


@dataclass
class CircuitConfig:
	"""Circuit breaker configuration."""
	failure_threshold: int = 5           # Number of failures to trip circuit
	success_threshold: int = 3           # Successful requests to close circuit
	timeout_ms: int = 5000              # Request timeout
	window_size_ms: int = 60000         # Sliding window size
	half_open_max_calls: int = 3        # Max calls in half-open state
	recovery_timeout_ms: int = 30000    # Time before trying half-open
	slow_call_threshold_ms: int = 1000  # Threshold for slow calls
	slow_call_rate_threshold: float = 0.5  # Percentage of slow calls to trip
	min_calls_threshold: int = 10       # Minimum calls before evaluation
	adaptive_threshold: bool = True     # Enable adaptive thresholds
	burst_detection: bool = True        # Enable burst failure detection
	exponential_backoff: bool = True    # Enable exponential backoff


class SlidingWindowMetrics:
	"""Sliding window for request metrics."""
	
	def __init__(self, window_size_ms: int = 60000):
		self.window_size_ms = window_size_ms
		self.requests: deque = deque()
		self.response_times: deque = deque()
		
	def add_request(self, result: RequestResult) -> None:
		"""Add request result to sliding window."""
		current_time = time.time() * 1000
		
		# Add new request
		self.requests.append({
			'timestamp': current_time,
			'success': result.success,
			'duration_ms': result.duration_ms,
			'error_type': result.error_type,
			'status_code': result.status_code
		})
		self.response_times.append(result.duration_ms)
		
		# Remove old requests outside window
		cutoff_time = current_time - self.window_size_ms
		while self.requests and self.requests[0]['timestamp'] < cutoff_time:
			self.requests.popleft()
		while self.response_times and len(self.response_times) > len(self.requests):
			self.response_times.popleft()
	
	def get_metrics(self) -> CircuitMetrics:
		"""Get current metrics from sliding window."""
		if not self.requests:
			return CircuitMetrics()
		
		total_requests = len(self.requests)
		failed_requests = sum(1 for req in self.requests if not req['success'])
		success_requests = total_requests - failed_requests
		
		error_percentage = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
		
		# Calculate response time percentiles
		if self.response_times:
			response_times = list(self.response_times)
			mean_response_time = statistics.mean(response_times)
			
			# Calculate percentiles
			sorted_times = sorted(response_times)
			p95_index = int(0.95 * len(sorted_times))
			p99_index = int(0.99 * len(sorted_times))
			
			p95_response_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
			p99_response_time = sorted_times[min(p99_index, len(sorted_times) - 1)]
		else:
			mean_response_time = p95_response_time = p99_response_time = 0.0
		
		# Calculate throughput
		window_duration_seconds = self.window_size_ms / 1000
		throughput_per_second = total_requests / window_duration_seconds if window_duration_seconds > 0 else 0
		
		return CircuitMetrics(
			total_requests=total_requests,
			failed_requests=failed_requests,
			success_requests=success_requests,
			error_percentage=error_percentage,
			mean_response_time=mean_response_time,
			p95_response_time=p95_response_time,
			p99_response_time=p99_response_time,
			throughput_per_second=throughput_per_second
		)


class BurstDetector:
	"""Detects burst failures and latency spikes."""
	
	def __init__(self, burst_window_ms: int = 5000):
		self.burst_window_ms = burst_window_ms
		self.recent_failures: deque = deque()
		self.recent_latencies: deque = deque()
		
	def add_failure(self, failure_type: FailureType, latency_ms: float) -> None:
		"""Add failure to burst detection."""
		current_time = time.time() * 1000
		
		self.recent_failures.append({
			'timestamp': current_time,
			'type': failure_type
		})
		self.recent_latencies.append({
			'timestamp': current_time,
			'latency': latency_ms
		})
		
		# Remove old entries
		cutoff_time = current_time - self.burst_window_ms
		while self.recent_failures and self.recent_failures[0]['timestamp'] < cutoff_time:
			self.recent_failures.popleft()
		while self.recent_latencies and self.recent_latencies[0]['timestamp'] < cutoff_time:
			self.recent_latencies.popleft()
	
	def detect_burst_failures(self, threshold: int = 3) -> bool:
		"""Detect if there's a burst of failures."""
		return len(self.recent_failures) >= threshold
	
	def detect_latency_spike(self, spike_threshold_ms: float = 2000) -> bool:
		"""Detect latency spikes."""
		if len(self.recent_latencies) < 3:
			return False
		
		recent_latencies = [entry['latency'] for entry in self.recent_latencies]
		avg_latency = statistics.mean(recent_latencies)
		
		return avg_latency > spike_threshold_ms


class AdaptiveThresholds:
	"""Adaptive threshold calculator based on historical performance."""
	
	def __init__(self):
		self.historical_error_rates: deque = deque(maxlen=100)
		self.historical_latencies: deque = deque(maxlen=1000)
		
	def update_history(self, error_rate: float, avg_latency: float) -> None:
		"""Update historical performance data."""
		self.historical_error_rates.append(error_rate)
		self.historical_latencies.append(avg_latency)
	
	def calculate_failure_threshold(self, base_threshold: int) -> int:
		"""Calculate adaptive failure threshold."""
		if len(self.historical_error_rates) < 10:
			return base_threshold
		
		avg_error_rate = statistics.mean(self.historical_error_rates)
		
		# If service is normally stable, be more sensitive to failures
		if avg_error_rate < 1.0:
			return max(3, base_threshold - 2)
		# If service is normally unstable, be less sensitive
		elif avg_error_rate > 10.0:
			return base_threshold + 3
		
		return base_threshold
	
	def calculate_latency_threshold(self, base_threshold_ms: float) -> float:
		"""Calculate adaptive latency threshold."""
		if len(self.historical_latencies) < 10:
			return base_threshold_ms
		
		# Use P95 of historical latencies as baseline
		sorted_latencies = sorted(self.historical_latencies)
		p95_index = int(0.95 * len(sorted_latencies))
		historical_p95 = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]
		
		# Adaptive threshold is 150% of historical P95
		adaptive_threshold = historical_p95 * 1.5
		
		return max(base_threshold_ms, adaptive_threshold)


class AdvancedCircuitBreaker:
	"""Advanced circuit breaker with failure detection and adaptive behavior."""
	
	def __init__(
		self,
		name: str,
		config: Optional[CircuitConfig] = None,
		redis_client: Optional[redis.Redis] = None,
		db_session: Optional[AsyncSession] = None
	):
		self.name = name
		self.config = config or CircuitConfig()
		self.redis_client = redis_client
		self.db_session = db_session
		
		# Circuit state
		self.state = CircuitState.CLOSED
		self.state_changed_at = time.time() * 1000
		self.failure_count = 0
		self.success_count = 0
		self.half_open_calls = 0
		
		# Metrics and detection
		self.metrics = SlidingWindowMetrics(self.config.window_size_ms)
		self.burst_detector = BurstDetector()
		self.adaptive_thresholds = AdaptiveThresholds()
		
		# Exponential backoff
		self.backoff_multiplier = 1.0
		self.max_backoff_ms = 300000  # 5 minutes
		
		# State change callbacks
		self.state_change_callbacks: List[Callable] = []
	
	async def call(self, func: Callable, *args, **kwargs) -> Any:
		"""Execute function with circuit breaker protection."""
		# Check if circuit should allow call
		if not await self._should_allow_call():
			await self._record_rejected_call()
			raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is open")
		
		start_time = time.time() * 1000
		success = False
		error_type = None
		status_code = None
		
		try:
			# Execute function with timeout
			result = await asyncio.wait_for(
				func(*args, **kwargs),
				timeout=self.config.timeout_ms / 1000
			)
			success = True
			
			# Check for HTTP status codes if result has them
			if hasattr(result, 'status_code'):
				status_code = result.status_code
				if status_code >= 400:
					success = False
					error_type = self._classify_http_error(status_code)
			
		except asyncio.TimeoutError:
			success = False
			error_type = FailureType.TIMEOUT
		except httpx.ConnectError:
			success = False
			error_type = FailureType.CONNECTION_ERROR
		except httpx.HTTPStatusError as e:
			success = False
			status_code = e.response.status_code
			error_type = self._classify_http_error(status_code)
		except Exception as e:
			success = False
			error_type = FailureType.SERVICE_UNAVAILABLE
		
		# Record result
		duration_ms = time.time() * 1000 - start_time
		request_result = RequestResult(
			success=success,
			duration_ms=duration_ms,
			status_code=status_code,
			error_type=error_type
		)
		
		await self._record_request_result(request_result)
		
		if not success:
			raise CircuitBreakerExecutionError(f"Request failed: {error_type}")
		
		return result
	
	async def _should_allow_call(self) -> bool:
		"""Check if circuit should allow the call."""
		current_time = time.time() * 1000
		
		if self.state == CircuitState.CLOSED:
			return True
		
		elif self.state == CircuitState.OPEN:
			# Check if enough time has passed to try half-open
			time_since_open = current_time - self.state_changed_at
			recovery_timeout = self._calculate_recovery_timeout()
			
			if time_since_open >= recovery_timeout:
				await self._transition_to_half_open()
				return True
			
			return False
		
		elif self.state == CircuitState.HALF_OPEN:
			# Allow limited calls in half-open state
			return self.half_open_calls < self.config.half_open_max_calls
		
		return False
	
	async def _record_request_result(self, result: RequestResult) -> None:
		"""Record request result and update circuit state."""
		# Add to metrics
		self.metrics.add_request(result)
		
		# Update burst detector
		if not result.success:
			self.burst_detector.add_failure(result.error_type, result.duration_ms)
		
		# Update state based on result
		if self.state == CircuitState.CLOSED:
			if result.success:
				self.failure_count = 0
			else:
				self.failure_count += 1
				
				# Check if should open circuit
				if await self._should_open_circuit():
					await self._transition_to_open()
		
		elif self.state == CircuitState.HALF_OPEN:
			self.half_open_calls += 1
			
			if result.success:
				self.success_count += 1
				
				# Check if should close circuit
				if self.success_count >= self.config.success_threshold:
					await self._transition_to_closed()
			else:
				# Any failure in half-open transitions back to open
				await self._transition_to_open()
		
		# Store metrics in database if available
		if self.db_session:
			await self._store_metrics()
	
	async def _should_open_circuit(self) -> bool:
		"""Determine if circuit should be opened."""
		metrics = self.metrics.get_metrics()
		
		# Not enough data yet
		if metrics.total_requests < self.config.min_calls_threshold:
			return False
		
		# Calculate adaptive thresholds
		failure_threshold = self.config.failure_threshold
		slow_call_threshold = self.config.slow_call_threshold_ms
		
		if self.config.adaptive_threshold:
			self.adaptive_thresholds.update_history(
				metrics.error_percentage,
				metrics.mean_response_time
			)
			failure_threshold = self.adaptive_thresholds.calculate_failure_threshold(failure_threshold)
			slow_call_threshold = self.adaptive_thresholds.calculate_latency_threshold(slow_call_threshold)
		
		# Check failure count threshold
		if self.failure_count >= failure_threshold:
			return True
		
		# Check error rate threshold
		if metrics.error_percentage > 50:  # 50% error rate
			return True
		
		# Check slow call rate
		slow_calls = sum(1 for req in self.metrics.requests 
						if req['duration_ms'] > slow_call_threshold)
		slow_call_rate = slow_calls / metrics.total_requests if metrics.total_requests > 0 else 0
		
		if slow_call_rate > self.config.slow_call_rate_threshold:
			return True
		
		# Check for burst failures
		if self.config.burst_detection and self.burst_detector.detect_burst_failures():
			return True
		
		# Check for latency spikes
		if self.burst_detector.detect_latency_spike(slow_call_threshold * 2):
			return True
		
		return False
	
	async def _transition_to_open(self) -> None:
		"""Transition circuit to open state."""
		self.state = CircuitState.OPEN
		self.state_changed_at = time.time() * 1000
		self.failure_count = 0
		self.success_count = 0
		self.half_open_calls = 0
		
		# Increase backoff for exponential recovery
		if self.config.exponential_backoff:
			self.backoff_multiplier = min(self.backoff_multiplier * 2, 32)
		
		await self._notify_state_change("open")
		await self._create_alert("Circuit breaker opened due to failures")
	
	async def _transition_to_half_open(self) -> None:
		"""Transition circuit to half-open state."""
		self.state = CircuitState.HALF_OPEN
		self.state_changed_at = time.time() * 1000
		self.success_count = 0
		self.half_open_calls = 0
		
		await self._notify_state_change("half_open")
	
	async def _transition_to_closed(self) -> None:
		"""Transition circuit to closed state."""
		self.state = CircuitState.CLOSED
		self.state_changed_at = time.time() * 1000
		self.failure_count = 0
		self.success_count = 0
		
		# Reset backoff multiplier on successful recovery
		self.backoff_multiplier = 1.0
		
		await self._notify_state_change("closed")
		await self._create_alert("Circuit breaker closed - service recovered")
	
	def _calculate_recovery_timeout(self) -> float:
		"""Calculate recovery timeout with exponential backoff."""
		base_timeout = self.config.recovery_timeout_ms
		
		if self.config.exponential_backoff:
			timeout = base_timeout * self.backoff_multiplier
			return min(timeout, self.max_backoff_ms)
		
		return base_timeout
	
	def _classify_http_error(self, status_code: int) -> FailureType:
		"""Classify HTTP error by status code."""
		if status_code == 429:
			return FailureType.RATE_LIMIT
		elif status_code >= 500:
			return FailureType.SERVICE_UNAVAILABLE
		else:
			return FailureType.HTTP_ERROR
	
	async def _record_rejected_call(self) -> None:
		"""Record that a call was rejected by circuit breaker."""
		if self.redis_client:
			await self.redis_client.incr(f"circuit_breaker:{self.name}:rejected")
	
	async def _notify_state_change(self, new_state: str) -> None:
		"""Notify listeners of state change."""
		for callback in self.state_change_callbacks:
			try:
				await callback(self.name, new_state, self.get_metrics())
			except Exception as e:
				print(f"Error in circuit breaker callback: {e}")
	
	async def _create_alert(self, message: str) -> None:
		"""Create alert for circuit breaker state change."""
		if self.db_session:
			try:
				alert = SMAlert(
					id=uuid7str(),
					service_id=self.name,
					alert_type="circuit_breaker",
					severity="warning" if self.state == CircuitState.HALF_OPEN else "error",
					message=message,
					metadata={
						"circuit_name": self.name,
						"state": self.state.value,
						"metrics": self.get_metrics()
					},
					created_at=datetime.now(timezone.utc)
				)
				self.db_session.add(alert)
				await self.db_session.commit()
			except Exception as e:
				print(f"Error creating circuit breaker alert: {e}")
	
	async def _store_metrics(self) -> None:
		"""Store circuit breaker metrics in database."""
		try:
			metrics = self.get_metrics()
			metric_record = SMMetrics(
				id=uuid7str(),
				service_id=self.name,
				metric_type="circuit_breaker",
				value=metrics['error_percentage'],
				metadata=metrics,
				timestamp=datetime.now(timezone.utc)
			)
			self.db_session.add(metric_record)
			await self.db_session.commit()
		except Exception as e:
			print(f"Error storing circuit breaker metrics: {e}")
	
	def get_metrics(self) -> Dict[str, Any]:
		"""Get current circuit breaker metrics."""
		circuit_metrics = self.metrics.get_metrics()
		
		return {
			"name": self.name,
			"state": self.state.value,
			"total_requests": circuit_metrics.total_requests,
			"failed_requests": circuit_metrics.failed_requests,
			"success_requests": circuit_metrics.success_requests,
			"rejected_requests": circuit_metrics.rejected_requests,
			"error_percentage": circuit_metrics.error_percentage,
			"mean_response_time": circuit_metrics.mean_response_time,
			"p95_response_time": circuit_metrics.p95_response_time,
			"p99_response_time": circuit_metrics.p99_response_time,
			"throughput_per_second": circuit_metrics.throughput_per_second,
			"failure_count": self.failure_count,
			"success_count": self.success_count,
			"state_changed_at": self.state_changed_at,
			"backoff_multiplier": self.backoff_multiplier
		}
	
	def add_state_change_callback(self, callback: Callable) -> None:
		"""Add callback for state changes."""
		self.state_change_callbacks.append(callback)
	
	async def force_open(self) -> None:
		"""Manually open the circuit breaker."""
		await self._transition_to_open()
	
	async def force_close(self) -> None:
		"""Manually close the circuit breaker."""
		await self._transition_to_closed()
	
	async def reset(self) -> None:
		"""Reset circuit breaker to initial state."""
		self.state = CircuitState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.half_open_calls = 0
		self.backoff_multiplier = 1.0
		self.metrics = SlidingWindowMetrics(self.config.window_size_ms)
		self.burst_detector = BurstDetector()


# =============================================================================
# Circuit Breaker Manager
# =============================================================================

class CircuitBreakerManager:
	"""Manages multiple circuit breakers."""
	
	def __init__(
		self,
		redis_client: Optional[redis.Redis] = None,
		db_session: Optional[AsyncSession] = None
	):
		self.redis_client = redis_client
		self.db_session = db_session
		self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
		self.default_config = CircuitConfig()
	
	def get_circuit_breaker(
		self,
		name: str,
		config: Optional[CircuitConfig] = None
	) -> AdvancedCircuitBreaker:
		"""Get or create circuit breaker."""
		if name not in self.circuit_breakers:
			self.circuit_breakers[name] = AdvancedCircuitBreaker(
				name=name,
				config=config or self.default_config,
				redis_client=self.redis_client,
				db_session=self.db_session
			)
		
		return self.circuit_breakers[name]
	
	def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
		"""Get metrics for all circuit breakers."""
		return {
			name: cb.get_metrics()
			for name, cb in self.circuit_breakers.items()
		}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Get health status of all circuit breakers."""
		health_data = {
			"total_circuits": len(self.circuit_breakers),
			"open_circuits": 0,
			"half_open_circuits": 0,
			"closed_circuits": 0,
			"circuits": {}
		}
		
		for name, cb in self.circuit_breakers.items():
			state = cb.state.value
			health_data["circuits"][name] = {
				"state": state,
				"metrics": cb.get_metrics()
			}
			
			if state == "open":
				health_data["open_circuits"] += 1
			elif state == "half_open":
				health_data["half_open_circuits"] += 1
			else:
				health_data["closed_circuits"] += 1
		
		return health_data


# =============================================================================
# Exceptions
# =============================================================================

class CircuitBreakerError(Exception):
	"""Base circuit breaker exception."""
	pass


class CircuitBreakerOpenError(CircuitBreakerError):
	"""Circuit breaker is open."""
	pass


class CircuitBreakerExecutionError(CircuitBreakerError):
	"""Error during circuit breaker execution."""
	pass