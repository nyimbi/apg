"""
Circuit Breaker Implementation
=============================

Production-grade circuit breaker for preventing cascade failures
and providing graceful degradation in the Google News crawler.

Circuit States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failure threshold exceeded, requests immediately fail
- HALF_OPEN: Testing recovery, limited requests allowed

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CircuitState(Enum):
	"""Circuit breaker states."""
	CLOSED = "closed"        # Normal operation
	OPEN = "open"           # Failing fast
	HALF_OPEN = "half_open" # Testing recovery

@dataclass
class CircuitBreakerConfig:
	"""Configuration for circuit breaker."""
	# Failure thresholds
	failure_threshold: int = 5          # Failures to open circuit
	success_threshold: int = 3          # Successes to close from half-open
	timeout_seconds: float = 60.0       # Time to wait before half-open
	
	# Request volume thresholds
	request_volume_threshold: int = 10   # Minimum requests before evaluation
	
	# Time window for failure rate calculation
	rolling_window_seconds: float = 60.0
	
	# Recovery testing
	half_open_max_calls: int = 3        # Max calls in half-open state
	
	# Monitoring
	enable_metrics: bool = True

class CircuitBreaker:
	"""
	Production-grade circuit breaker implementation with comprehensive
	failure detection and recovery mechanisms.
	"""
	
	def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
		"""Initialize circuit breaker."""
		self.config = config
		self.name = name
		
		# Circuit state
		self.state = CircuitState.CLOSED
		self.last_failure_time = None
		self.last_success_time = None
		
		# Counters
		self.failure_count = 0
		self.success_count = 0
		self.half_open_calls = 0
		
		# Request tracking for rolling window
		self.request_history = []  # List of (timestamp, success) tuples
		
		# Metrics
		self.metrics = {
			'total_requests': 0,
			'successful_requests': 0,
			'failed_requests': 0,
			'circuit_opens': 0,
			'circuit_half_opens': 0,
			'circuit_closes': 0,
			'fast_failures': 0,  # Requests failed due to open circuit
			'start_time': datetime.now()
		}
		
		# Thread safety
		self._lock = asyncio.Lock()
		
		logger.info(f"Circuit breaker '{name}' initialized: {config}")
	
	async def call(self, func: Callable, *args, **kwargs) -> Any:
		"""
		Execute function with circuit breaker protection.
		
		Args:
			func: Function to execute
			*args: Function arguments
			**kwargs: Function keyword arguments
			
		Returns:
			Function result if successful
			
		Raises:
			CircuitOpenError: If circuit is open
			Original exception: If function fails and circuit remains closed
		"""
		async with self._lock:
			self.metrics['total_requests'] += 1
			
			# Check if circuit is open
			if self.state == CircuitState.OPEN:
				if await self._should_attempt_reset():
					await self._transition_to_half_open()
				else:
					self.metrics['fast_failures'] += 1
					raise CircuitOpenError(f"Circuit breaker '{self.name}' is OPEN")
			
			# Check half-open call limit
			if self.state == CircuitState.HALF_OPEN:
				if self.half_open_calls >= self.config.half_open_max_calls:
					self.metrics['fast_failures'] += 1
					raise CircuitOpenError(f"Circuit breaker '{self.name}' half-open limit exceeded")
				self.half_open_calls += 1
		
		# Execute function
		start_time = time.time()
		try:
			result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
			
			# Record success
			execution_time = time.time() - start_time
			await self._record_success(execution_time)
			
			return result
			
		except Exception as e:
			# Record failure
			execution_time = time.time() - start_time
			await self._record_failure(e, execution_time)
			raise
	
	async def _record_success(self, execution_time: float) -> None:
		"""Record successful request and update circuit state."""
		async with self._lock:
			current_time = time.time()
			
			# Update counters
			self.success_count += 1
			self.last_success_time = current_time
			self.metrics['successful_requests'] += 1
			
			# Add to rolling window
			self.request_history.append((current_time, True))
			self._cleanup_request_history(current_time)
			
			# State transitions
			if self.state == CircuitState.HALF_OPEN:
				if self.success_count >= self.config.success_threshold:
					await self._transition_to_closed()
			
			logger.debug(f"Circuit breaker '{self.name}' recorded success (exec_time: {execution_time:.3f}s)")
	
	async def _record_failure(self, exception: Exception, execution_time: float) -> None:
		"""Record failed request and update circuit state."""
		async with self._lock:
			current_time = time.time()
			
			# Update counters
			self.failure_count += 1
			self.last_failure_time = current_time
			self.metrics['failed_requests'] += 1
			
			# Add to rolling window
			self.request_history.append((current_time, False))
			self._cleanup_request_history(current_time)
			
			# Check if we should open the circuit
			if self.state == CircuitState.CLOSED:
				if await self._should_open_circuit():
					await self._transition_to_open()
			elif self.state == CircuitState.HALF_OPEN:
				# Any failure in half-open immediately reopens circuit
				await self._transition_to_open()
			
			logger.warning(f"Circuit breaker '{self.name}' recorded failure: {type(exception).__name__} (exec_time: {execution_time:.3f}s)")
	
	async def _should_open_circuit(self) -> bool:
		"""Determine if circuit should be opened based on failure rate."""
		current_time = time.time()
		
		# Get recent requests within rolling window
		window_start = current_time - self.config.rolling_window_seconds
		recent_requests = [
			(timestamp, success) for timestamp, success in self.request_history
			if timestamp >= window_start
		]
		
		# Check minimum request volume
		if len(recent_requests) < self.config.request_volume_threshold:
			return False
		
		# Calculate failure rate
		failed_requests = sum(1 for _, success in recent_requests if not success)
		failure_rate = failed_requests / len(recent_requests)
		
		# Open if failure rate exceeds threshold
		threshold_rate = self.config.failure_threshold / len(recent_requests)
		should_open = failure_rate >= threshold_rate
		
		if should_open:
			logger.warning(f"Circuit breaker '{self.name}' should open: {failed_requests}/{len(recent_requests)} failures ({failure_rate:.2%})")
		
		return should_open
	
	async def _should_attempt_reset(self) -> bool:
		"""Check if enough time has passed to attempt reset."""
		if not self.last_failure_time:
			return True
		
		time_since_failure = time.time() - self.last_failure_time
		return time_since_failure >= self.config.timeout_seconds
	
	async def _transition_to_open(self) -> None:
		"""Transition circuit to OPEN state."""
		old_state = self.state
		self.state = CircuitState.OPEN
		self.failure_count = 0  # Reset for next evaluation
		self.success_count = 0
		self.metrics['circuit_opens'] += 1
		
		logger.warning(f"Circuit breaker '{self.name}' transitioned: {old_state.value} → OPEN")
	
	async def _transition_to_half_open(self) -> None:
		"""Transition circuit to HALF_OPEN state."""
		old_state = self.state
		self.state = CircuitState.HALF_OPEN
		self.half_open_calls = 0
		self.success_count = 0
		self.failure_count = 0
		self.metrics['circuit_half_opens'] += 1
		
		logger.info(f"Circuit breaker '{self.name}' transitioned: {old_state.value} → HALF_OPEN")
	
	async def _transition_to_closed(self) -> None:
		"""Transition circuit to CLOSED state."""
		old_state = self.state
		self.state = CircuitState.CLOSED
		self.failure_count = 0
		self.success_count = 0
		self.half_open_calls = 0
		self.metrics['circuit_closes'] += 1
		
		logger.info(f"Circuit breaker '{self.name}' transitioned: {old_state.value} → CLOSED")
	
	def _cleanup_request_history(self, current_time: float) -> None:
		"""Remove old requests from rolling window."""
		window_start = current_time - self.config.rolling_window_seconds
		self.request_history = [
			(timestamp, success) for timestamp, success in self.request_history
			if timestamp >= window_start
		]
	
	def get_state(self) -> CircuitState:
		"""Get current circuit state."""
		return self.state
	
	def is_closed(self) -> bool:
		"""Check if circuit is closed (normal operation)."""
		return self.state == CircuitState.CLOSED
	
	def is_open(self) -> bool:
		"""Check if circuit is open (failing fast)."""
		return self.state == CircuitState.OPEN
	
	def is_half_open(self) -> bool:
		"""Check if circuit is half-open (testing recovery)."""
		return self.state == CircuitState.HALF_OPEN
	
	def get_metrics(self) -> Dict[str, Any]:
		"""Get comprehensive circuit breaker metrics."""
		current_time = datetime.now()
		uptime = current_time - self.metrics['start_time']
		
		# Calculate failure rate from rolling window
		current_timestamp = time.time()
		window_start = current_timestamp - self.config.rolling_window_seconds
		recent_requests = [
			success for timestamp, success in self.request_history
			if timestamp >= window_start
		]
		
		current_failure_rate = 0.0
		if recent_requests:
			failures = sum(1 for success in recent_requests if not success)
			current_failure_rate = failures / len(recent_requests)
		
		return {
			'circuit_breaker_name': self.name,
			'current_state': self.state.value,
			'uptime_seconds': uptime.total_seconds(),
			'total_requests': self.metrics['total_requests'],
			'successful_requests': self.metrics['successful_requests'],
			'failed_requests': self.metrics['failed_requests'],
			'fast_failures': self.metrics['fast_failures'],
			'success_rate': (self.metrics['successful_requests'] / max(1, self.metrics['total_requests'])) * 100,
			'current_failure_rate': current_failure_rate * 100,
			'recent_requests_count': len(recent_requests),
			'state_transitions': {
				'opens': self.metrics['circuit_opens'],
				'half_opens': self.metrics['circuit_half_opens'],
				'closes': self.metrics['circuit_closes']
			},
			'current_counters': {
				'failure_count': self.failure_count,
				'success_count': self.success_count,
				'half_open_calls': self.half_open_calls
			},
			'timestamps': {
				'last_failure': self.last_failure_time,
				'last_success': self.last_success_time
			},
			'configuration': {
				'failure_threshold': self.config.failure_threshold,
				'success_threshold': self.config.success_threshold,
				'timeout_seconds': self.config.timeout_seconds,
				'request_volume_threshold': self.config.request_volume_threshold,
				'rolling_window_seconds': self.config.rolling_window_seconds
			}
		}
	
	async def reset(self) -> None:
		"""Manually reset circuit breaker to closed state."""
		async with self._lock:
			old_state = self.state
			await self._transition_to_closed()
			
			# Reset counters but preserve metrics
			self.last_failure_time = None
			self.last_success_time = None
			self.request_history.clear()
			
			logger.info(f"Circuit breaker '{self.name}' manually reset from {old_state.value}")

class CircuitOpenError(Exception):
	"""Exception raised when circuit breaker is open."""
	pass

def create_circuit_breaker(
	name: str = "default",
	failure_threshold: int = 5,
	timeout_seconds: float = 60.0,
	success_threshold: int = 3
) -> CircuitBreaker:
	"""Factory function to create circuit breaker with common settings."""
	config = CircuitBreakerConfig(
		failure_threshold=failure_threshold,
		success_threshold=success_threshold,
		timeout_seconds=timeout_seconds
	)
	return CircuitBreaker(config, name)

def create_google_news_circuit_breaker() -> CircuitBreaker:
	"""Create circuit breaker optimized for Google News operations."""
	config = CircuitBreakerConfig(
		failure_threshold=3,         # More sensitive for external API
		success_threshold=2,         # Faster recovery
		timeout_seconds=30.0,        # Shorter timeout for API
		request_volume_threshold=5,  # Lower volume threshold
		rolling_window_seconds=30.0, # Shorter window for responsiveness
		half_open_max_calls=2       # Conservative testing
	)
	return CircuitBreaker(config, "google_news_api")