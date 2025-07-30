"""
Retry Handler Implementation for Google News Crawler
===================================================

Configurable retry logic with exponential backoff, jitter,
and integration with circuit breaker patterns.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import random
import time
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class BackoffStrategy(Enum):
	"""Backoff strategies for retries."""
	FIXED = "fixed"
	LINEAR = "linear"
	EXPONENTIAL = "exponential"
	EXPONENTIAL_JITTER = "exponential_jitter"

@dataclass
class RetryConfig:
	"""Configuration for retry behavior."""
	# Basic retry settings
	max_attempts: int = 3
	initial_delay: float = 1.0
	max_delay: float = 60.0
	
	# Backoff strategy
	backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
	backoff_multiplier: float = 2.0
	
	# Jitter settings
	jitter_min: float = 0.5
	jitter_max: float = 1.5
	
	# Conditional retry
	retry_on_exceptions: tuple = (Exception,)
	dont_retry_on_exceptions: tuple = ()
	
	# Advanced settings
	enable_circuit_breaker_integration: bool = True
	respect_rate_limits: bool = True

class ExponentialBackoff:
	"""Exponential backoff calculator with jitter."""
	
	def __init__(self, 
				 initial_delay: float = 1.0,
				 max_delay: float = 60.0,
				 multiplier: float = 2.0,
				 jitter: bool = True):
		self.initial_delay = initial_delay
		self.max_delay = max_delay
		self.multiplier = multiplier
		self.jitter = jitter
	
	def calculate_delay(self, attempt: int) -> float:
		"""Calculate delay for given attempt number."""
		if attempt <= 0:
			return 0.0
		
		# Calculate base delay
		delay = self.initial_delay * (self.multiplier ** (attempt - 1))
		
		# Apply maximum delay cap
		delay = min(delay, self.max_delay)
		
		# Add jitter if enabled
		if self.jitter:
			jitter_factor = random.uniform(0.5, 1.5)
			delay *= jitter_factor
		
		return delay

class RetryHandler:
	"""
	Configurable retry handler with support for multiple backoff strategies
	and integration with other resilience patterns.
	"""
	
	def __init__(self, config: RetryConfig):
		"""Initialize retry handler with configuration."""
		self.config = config
		
		# Initialize backoff calculator
		if config.backoff_strategy in [BackoffStrategy.EXPONENTIAL, BackoffStrategy.EXPONENTIAL_JITTER]:
			self.backoff = ExponentialBackoff(
				initial_delay=config.initial_delay,
				max_delay=config.max_delay,
				multiplier=config.backoff_multiplier,
				jitter=(config.backoff_strategy == BackoffStrategy.EXPONENTIAL_JITTER)
			)
		
		# Retry statistics
		self.stats = {
			'total_attempts': 0,
			'successful_retries': 0,
			'failed_retries': 0,
			'retry_attempts_by_exception': {},
			'total_delay_time': 0.0
		}
		
		logger.info(f"Retry handler initialized: {config}")
	
	async def execute(self, 
					  func: Callable,
					  *args,
					  operation_name: str = "unknown",
					  **kwargs) -> Any:
		"""
		Execute function with retry logic.
		
		Args:
			func: Function to execute
			*args: Function arguments
			operation_name: Name for logging and stats
			**kwargs: Function keyword arguments
			
		Returns:
			Function result if successful
			
		Raises:
			Last exception if all retries fail
		"""
		last_exception = None
		
		for attempt in range(1, self.config.max_attempts + 1):
			self.stats['total_attempts'] += 1
			
			try:
				# Execute the function
				if asyncio.iscoroutinefunction(func):
					result = await func(*args, **kwargs)
				else:
					result = func(*args, **kwargs)
				
				# Success!
				if attempt > 1:
					self.stats['successful_retries'] += 1
					logger.info(f"Operation '{operation_name}' succeeded on attempt {attempt}")
				
				return result
				
			except Exception as e:
				last_exception = e
				
				# Check if we should retry this exception
				if not self._should_retry_exception(e):
					logger.warning(f"Operation '{operation_name}' failed with non-retryable exception: {type(e).__name__}")
					raise
				
				# Update stats
				exception_type = type(e).__name__
				if exception_type not in self.stats['retry_attempts_by_exception']:
					self.stats['retry_attempts_by_exception'][exception_type] = 0
				self.stats['retry_attempts_by_exception'][exception_type] += 1
				
				# Don't delay on the last attempt
				if attempt == self.config.max_attempts:
					self.stats['failed_retries'] += 1
					logger.error(f"Operation '{operation_name}' failed after {attempt} attempts: {e}")
					break
				
				# Calculate and apply delay
				delay = self._calculate_delay(attempt)
				if delay > 0:
					logger.warning(f"Operation '{operation_name}' failed on attempt {attempt}/{self.config.max_attempts}: {type(e).__name__}. Retrying in {delay:.2f}s")
					await asyncio.sleep(delay)
					self.stats['total_delay_time'] += delay
		
		# All attempts failed
		raise last_exception
	
	def _should_retry_exception(self, exception: Exception) -> bool:
		"""Determine if exception should trigger a retry."""
		# Check don't retry list first
		if self.config.dont_retry_on_exceptions:
			for exc_type in self.config.dont_retry_on_exceptions:
				if isinstance(exception, exc_type):
					return False
		
		# Check retry list
		for exc_type in self.config.retry_on_exceptions:
			if isinstance(exception, exc_type):
				return True
		
		return False
	
	def _calculate_delay(self, attempt: int) -> float:
		"""Calculate delay for given attempt."""
		if self.config.backoff_strategy == BackoffStrategy.FIXED:
			return self.config.initial_delay
		
		elif self.config.backoff_strategy == BackoffStrategy.LINEAR:
			delay = self.config.initial_delay * attempt
			return min(delay, self.config.max_delay)
		
		elif self.config.backoff_strategy in [BackoffStrategy.EXPONENTIAL, BackoffStrategy.EXPONENTIAL_JITTER]:
			return self.backoff.calculate_delay(attempt)
		
		else:
			return self.config.initial_delay
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive retry statistics."""
		total_attempts = self.stats['total_attempts']
		success_rate = 0.0
		if total_attempts > 0:
			successful_ops = total_attempts - self.stats['failed_retries']
			success_rate = (successful_ops / total_attempts) * 100
		
		return {
			'total_attempts': total_attempts,
			'successful_retries': self.stats['successful_retries'],
			'failed_retries': self.stats['failed_retries'],
			'success_rate_percent': success_rate,
			'total_delay_time_seconds': self.stats['total_delay_time'],
			'average_delay_per_retry': (
				self.stats['total_delay_time'] / max(1, self.stats['successful_retries'] + self.stats['failed_retries'])
			),
			'retry_attempts_by_exception': self.stats['retry_attempts_by_exception'].copy(),
			'configuration': {
				'max_attempts': self.config.max_attempts,
				'backoff_strategy': self.config.backoff_strategy.value,
				'initial_delay': self.config.initial_delay,
				'max_delay': self.config.max_delay,
				'backoff_multiplier': self.config.backoff_multiplier
			}
		}
	
	def reset_stats(self) -> None:
		"""Reset retry statistics."""
		self.stats = {
			'total_attempts': 0,
			'successful_retries': 0,
			'failed_retries': 0,
			'retry_attempts_by_exception': {},
			'total_delay_time': 0.0
		}
		logger.info("Retry handler statistics reset")

# Decorator for easy retry application
def retry(config: Optional[RetryConfig] = None,
		  max_attempts: int = 3,
		  initial_delay: float = 1.0,
		  backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER):
	"""
	Decorator to add retry logic to functions.
	
	Args:
		config: RetryConfig object (overrides individual parameters)
		max_attempts: Maximum retry attempts
		initial_delay: Initial delay between retries
		backoff_strategy: Backoff strategy to use
	"""
	if config is None:
		config = RetryConfig(
			max_attempts=max_attempts,
			initial_delay=initial_delay,
			backoff_strategy=backoff_strategy
		)
	
	def decorator(func):
		async def async_wrapper(*args, **kwargs):
			handler = RetryHandler(config)
			return await handler.execute(func, *args, operation_name=func.__name__, **kwargs)
		
		def sync_wrapper(*args, **kwargs):
			# For sync functions, create an async wrapper
			async def async_func():
				return func(*args, **kwargs)
			
			handler = RetryHandler(config)
			loop = asyncio.get_event_loop()
			return loop.run_until_complete(
				handler.execute(async_func, operation_name=func.__name__)
			)
		
		if asyncio.iscoroutinefunction(func):
			return async_wrapper
		else:
			return sync_wrapper
	
	return decorator

def create_retry_handler(
	max_attempts: int = 3,
	initial_delay: float = 1.0,
	max_delay: float = 60.0,
	backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
) -> RetryHandler:
	"""Factory function to create retry handler with common settings."""
	config = RetryConfig(
		max_attempts=max_attempts,
		initial_delay=initial_delay,
		max_delay=max_delay,
		backoff_strategy=backoff_strategy
	)
	return RetryHandler(config)

def create_google_news_retry_handler() -> RetryHandler:
	"""Create retry handler optimized for Google News operations."""
	config = RetryConfig(
		max_attempts=3,
		initial_delay=2.0,
		max_delay=30.0,
		backoff_strategy=BackoffStrategy.EXPONENTIAL_JITTER,
		backoff_multiplier=1.5,  # More conservative than default
		
		# Don't retry on authentication or client errors
		dont_retry_on_exceptions=(
			# Add specific exception types that shouldn't be retried
			ValueError,  # Usually indicates bad input
			TypeError,   # Programming errors
		),
		
		# Retry on network and server errors
		retry_on_exceptions=(
			Exception,  # Catch-all for now, can be refined
		)
	)
	return RetryHandler(config)