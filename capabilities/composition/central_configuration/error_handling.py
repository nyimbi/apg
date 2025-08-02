"""
APG Central Configuration - Comprehensive Error Handling System

Production-ready error handling with structured logging, circuit breaker patterns,
retry logic, and comprehensive error recovery strategies.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import logging
import json
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import inspect
import time
import random
from contextlib import asynccontextmanager

# Error categorization
class ErrorCategory(Enum):
	"""Error categories for proper handling and metrics."""
	NETWORK_ERROR = "network_error"
	DATABASE_ERROR = "database_error"
	AUTHENTICATION_ERROR = "authentication_error"
	VALIDATION_ERROR = "validation_error"
	CONFIGURATION_ERROR = "configuration_error"
	EXTERNAL_SERVICE_ERROR = "external_service_error"
	SYSTEM_ERROR = "system_error"
	BUSINESS_LOGIC_ERROR = "business_logic_error"
	TEMPORARY_ERROR = "temporary_error"
	PERMANENT_ERROR = "permanent_error"

class ErrorSeverity(Enum):
	"""Error severity levels."""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"

@dataclass
class ErrorContext:
	"""Comprehensive error context information."""
	error_id: str
	timestamp: datetime
	category: ErrorCategory
	severity: ErrorSeverity
	service: str
	operation: str
	user_id: Optional[str] = None
	tenant_id: Optional[str] = None
	request_id: Optional[str] = None
	correlation_id: Optional[str] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	stack_trace: Optional[str] = None
	recovery_suggestions: List[str] = field(default_factory=list)

class ConfigurationError(Exception):
	"""Base exception for configuration-related errors."""
	def __init__(self, message: str, context: Optional[ErrorContext] = None):
		super().__init__(message)
		self.context = context

class NetworkError(ConfigurationError):
	"""Network-related errors with retry capabilities."""
	pass

class DatabaseError(ConfigurationError):
	"""Database operation errors."""
	pass

class ValidationError(ConfigurationError):
	"""Configuration validation errors."""
	pass

class AuthenticationError(ConfigurationError):
	"""Authentication and authorization errors."""
	pass

class ExternalServiceError(ConfigurationError):
	"""External service integration errors."""
	pass

@dataclass
class RetryConfig:
	"""Retry configuration for error handling."""
	max_attempts: int = 3
	base_delay: float = 1.0
	max_delay: float = 60.0
	exponential_base: float = 2.0
	jitter: bool = True
	retryable_exceptions: Tuple[Type[Exception], ...] = (NetworkError, ExternalServiceError)

@dataclass
class CircuitBreakerConfig:
	"""Circuit breaker configuration."""
	failure_threshold: int = 5
	reset_timeout: float = 60.0
	expected_exception: Type[Exception] = Exception

class CircuitBreakerState(Enum):
	"""Circuit breaker states."""
	CLOSED = "closed"
	OPEN = "open"
	HALF_OPEN = "half_open"

class CircuitBreaker:
	"""Circuit breaker implementation for fault tolerance."""
	
	def __init__(self, config: CircuitBreakerConfig):
		self.config = config
		self.state = CircuitBreakerState.CLOSED
		self.failure_count = 0
		self.last_failure_time = None
		self.success_count = 0
		
	async def call(self, func: Callable, *args, **kwargs):
		"""Execute function with circuit breaker protection."""
		if self.state == CircuitBreakerState.OPEN:
			if self._should_attempt_reset():
				self.state = CircuitBreakerState.HALF_OPEN
			else:
				raise ExternalServiceError("Circuit breaker is OPEN - service unavailable")
		
		try:
			result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
			self._on_success()
			return result
		except self.config.expected_exception as e:
			self._on_failure()
			raise
	
	def _should_attempt_reset(self) -> bool:
		"""Check if circuit breaker should attempt reset."""
		return (
			self.last_failure_time and
			time.time() - self.last_failure_time >= self.config.reset_timeout
		)
	
	def _on_success(self):
		"""Handle successful call."""
		self.failure_count = 0
		if self.state == CircuitBreakerState.HALF_OPEN:
			self.state = CircuitBreakerState.CLOSED
	
	def _on_failure(self):
		"""Handle failed call."""
		self.failure_count += 1
		self.last_failure_time = time.time()
		
		if self.failure_count >= self.config.failure_threshold:
			self.state = CircuitBreakerState.OPEN

class ErrorHandler:
	"""Comprehensive error handling system."""
	
	def __init__(self, service_name: str):
		self.service_name = service_name
		self.logger = logging.getLogger(f"apg.{service_name}.errors")
		self.error_counters: Dict[str, int] = {}
		self.circuit_breakers: Dict[str, CircuitBreaker] = {}
		self.error_history: List[ErrorContext] = []
		
		# Configure structured logging
		self._setup_logging()
	
	def _setup_logging(self):
		"""Setup structured logging for errors."""
		formatter = logging.Formatter(
			'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
		)
		
		# Console handler
		console_handler = logging.StreamHandler()
		console_handler.setFormatter(formatter)
		self.logger.addHandler(console_handler)
		self.logger.setLevel(logging.INFO)
	
	async def handle_error(
		self,
		exception: Exception,
		category: ErrorCategory,
		severity: ErrorSeverity,
		operation: str,
		context: Optional[Dict[str, Any]] = None,
		user_id: Optional[str] = None,
		tenant_id: Optional[str] = None
	) -> ErrorContext:
		"""Handle error with comprehensive logging and metrics."""
		error_context = ErrorContext(
			error_id=f"err_{int(time.time())}_{id(exception)}",
			timestamp=datetime.now(timezone.utc),
			category=category,
			severity=severity,
			service=self.service_name,
			operation=operation,
			user_id=user_id,
			tenant_id=tenant_id,
			metadata=context or {},
			stack_trace=traceback.format_exc(),
			recovery_suggestions=self._get_recovery_suggestions(category, exception)
		)
		
		# Log error with structured information
		log_level = self._get_log_level(severity)
		self.logger.log(
			log_level,
			f"Error in {operation}: {str(exception)}",
			extra={
				"error_id": error_context.error_id,
				"category": category.value,
				"severity": severity.value,
				"user_id": user_id,
				"tenant_id": tenant_id,
				"operation": operation,
				"metadata": context or {},
				"exception_type": type(exception).__name__
			}
		)
		
		# Update error metrics
		self._update_error_metrics(category, severity, operation)
		
		# Store error context for analysis
		self.error_history.append(error_context)
		
		# Trigger alerts for critical errors
		if severity == ErrorSeverity.CRITICAL:
			await self._trigger_alert(error_context)
		
		return error_context
	
	def _get_log_level(self, severity: ErrorSeverity) -> int:
		"""Get appropriate log level for error severity."""
		severity_map = {
			ErrorSeverity.LOW: logging.INFO,
			ErrorSeverity.MEDIUM: logging.WARNING,
			ErrorSeverity.HIGH: logging.ERROR,
			ErrorSeverity.CRITICAL: logging.CRITICAL
		}
		return severity_map.get(severity, logging.ERROR)
	
	def _get_recovery_suggestions(self, category: ErrorCategory, exception: Exception) -> List[str]:
		"""Generate recovery suggestions based on error category."""
		suggestions = []
		
		if category == ErrorCategory.NETWORK_ERROR:
			suggestions = [
				"Check network connectivity",
				"Verify service endpoints are accessible",
				"Consider retry with exponential backoff",
				"Check firewall and security group settings"
			]
		elif category == ErrorCategory.DATABASE_ERROR:
			suggestions = [
				"Verify database connection parameters",
				"Check database server status",
				"Ensure sufficient database permissions",
				"Consider connection pool settings"
			]
		elif category == ErrorCategory.AUTHENTICATION_ERROR:
			suggestions = [
				"Verify authentication credentials",
				"Check token expiration",
				"Ensure proper OAuth2 configuration",
				"Verify user permissions"
			]
		elif category == ErrorCategory.VALIDATION_ERROR:
			suggestions = [
				"Check input data format",
				"Verify required fields are present",
				"Ensure data meets validation rules",
				"Review configuration schema"
			]
		elif category == ErrorCategory.EXTERNAL_SERVICE_ERROR:
			suggestions = [
				"Check external service status",
				"Verify API endpoints and credentials",
				"Consider circuit breaker pattern",
				"Implement fallback mechanisms"
			]
		
		return suggestions
	
	def _update_error_metrics(self, category: ErrorCategory, severity: ErrorSeverity, operation: str):
		"""Update error metrics for monitoring."""
		metric_key = f"{category.value}_{severity.value}_{operation}"
		self.error_counters[metric_key] = self.error_counters.get(metric_key, 0) + 1
	
	async def _trigger_alert(self, error_context: ErrorContext):
		"""Trigger alerts for critical errors."""
		# In production, this would integrate with alerting systems
		# like PagerDuty, Slack, or custom notification systems
		self.logger.critical(
			f"CRITICAL ERROR ALERT: {error_context.error_id}",
			extra={
				"alert_type": "critical_error",
				"error_context": error_context.__dict__
			}
		)
	
	def get_circuit_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
		"""Get or create circuit breaker for service."""
		if service_name not in self.circuit_breakers:
			self.circuit_breakers[service_name] = CircuitBreaker(
				config or CircuitBreakerConfig()
			)
		return self.circuit_breakers[service_name]

# Decorators for error handling
def with_error_handling(
	category: ErrorCategory,
	severity: ErrorSeverity = ErrorSeverity.MEDIUM,
	retry_config: Optional[RetryConfig] = None
):
	"""Decorator for automatic error handling and retry logic."""
	def decorator(func):
		@wraps(func)
		async def async_wrapper(*args, **kwargs):
			error_handler = getattr(args[0], 'error_handler', None) if args else None
			if not error_handler:
				# Create basic error handler if none exists
				error_handler = ErrorHandler("unknown_service")
			
			retry_cfg = retry_config or RetryConfig()
			last_exception = None
			
			for attempt in range(retry_cfg.max_attempts):
				try:
					if asyncio.iscoroutinefunction(func):
						return await func(*args, **kwargs)
					else:
						return func(*args, **kwargs)
				except retry_cfg.retryable_exceptions as e:
					last_exception = e
					if attempt < retry_cfg.max_attempts - 1:
						delay = min(
							retry_cfg.base_delay * (retry_cfg.exponential_base ** attempt),
							retry_cfg.max_delay
						)
						if retry_cfg.jitter:
							delay += random.uniform(0, delay * 0.1)
						
						await asyncio.sleep(delay)
						continue
					break
				except Exception as e:
					# Non-retryable exception
					await error_handler.handle_error(
						e, category, severity, func.__name__
					)
					raise
			
			# All retry attempts failed
			if last_exception:
				await error_handler.handle_error(
					last_exception, category, severity, func.__name__
				)
				raise last_exception
		
		@wraps(func)
		def sync_wrapper(*args, **kwargs):
			if asyncio.iscoroutinefunction(func):
				return async_wrapper(*args, **kwargs)
			
			# Handle synchronous functions
			error_handler = getattr(args[0], 'error_handler', None) if args else None
			if not error_handler:
				error_handler = ErrorHandler("unknown_service")
			
			try:
				return func(*args, **kwargs)
			except Exception as e:
				# For sync functions, we can't use async error handling
				# So we'll do basic logging
				error_handler.logger.error(
					f"Error in {func.__name__}: {str(e)}",
					extra={
						"category": category.value,
						"severity": severity.value,
						"operation": func.__name__,
						"exception_type": type(e).__name__
					}
				)
				raise
		
		return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
	return decorator

@asynccontextmanager
async def error_context(
	operation: str,
	category: ErrorCategory,
	severity: ErrorSeverity = ErrorSeverity.MEDIUM,
	error_handler: Optional[ErrorHandler] = None
):
	"""Context manager for error handling."""
	handler = error_handler or ErrorHandler("context_manager")
	
	try:
		yield
	except Exception as e:
		await handler.handle_error(e, category, severity, operation)
		raise

# Utility functions
def is_retryable_error(exception: Exception) -> bool:
	"""Check if an error is retryable."""
	retryable_types = (
		NetworkError,
		ExternalServiceError,
		TimeoutError,
		ConnectionError
	)
	return isinstance(exception, retryable_types)

def get_error_category(exception: Exception) -> ErrorCategory:
	"""Automatically categorize errors based on exception type."""
	if isinstance(exception, (ConnectionError, TimeoutError)):
		return ErrorCategory.NETWORK_ERROR
	elif isinstance(exception, (ValueError, TypeError)):
		return ErrorCategory.VALIDATION_ERROR
	elif "authentication" in str(exception).lower() or "unauthorized" in str(exception).lower():
		return ErrorCategory.AUTHENTICATION_ERROR
	elif "database" in str(exception).lower() or "sql" in str(exception).lower():
		return ErrorCategory.DATABASE_ERROR
	else:
		return ErrorCategory.SYSTEM_ERROR

# Global error handler instance
_global_error_handler = ErrorHandler("apg_central_configuration")

def get_error_handler() -> ErrorHandler:
	"""Get the global error handler instance."""
	return _global_error_handler

# Performance monitoring for error handling
class ErrorMetrics:
	"""Error metrics collection and analysis."""
	
	def __init__(self):
		self.error_rates: Dict[str, List[float]] = {}
		self.response_times: Dict[str, List[float]] = {}
	
	def record_error_rate(self, operation: str, error_rate: float):
		"""Record error rate for operation."""
		if operation not in self.error_rates:
			self.error_rates[operation] = []
		self.error_rates[operation].append(error_rate)
	
	def get_average_error_rate(self, operation: str) -> float:
		"""Get average error rate for operation."""
		rates = self.error_rates.get(operation, [])
		return sum(rates) / len(rates) if rates else 0.0
	
	def get_error_trend(self, operation: str) -> str:
		"""Get error trend (improving, degrading, stable)."""
		rates = self.error_rates.get(operation, [])
		if len(rates) < 2:
			return "insufficient_data"
		
		recent_avg = sum(rates[-5:]) / min(5, len(rates))
		historical_avg = sum(rates[:-5]) / max(1, len(rates) - 5)
		
		if recent_avg < historical_avg * 0.8:
			return "improving"
		elif recent_avg > historical_avg * 1.2:
			return "degrading"
		else:
			return "stable"

# Global metrics instance
_global_metrics = ErrorMetrics()

def get_error_metrics() -> ErrorMetrics:
	"""Get the global error metrics instance."""
	return _global_metrics