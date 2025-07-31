"""
Resilience and Error Handling for Google News Crawler
====================================================

Circuit breaker patterns, retry logic, and comprehensive error handling
for robust operation under various failure conditions.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

from .circuit_breaker import (
	CircuitBreaker,
	CircuitBreakerConfig,
	CircuitState,
	create_circuit_breaker,
	create_google_news_circuit_breaker
)

from .retry_handler import (
	RetryConfig,
	RetryHandler,
	ExponentialBackoff,
	create_retry_handler
)

from .error_handler import (
	ErrorHandler,
	ErrorCategory,
	ErrorContext,
	RecoveryStrategy,
	create_error_handler
)

__all__ = [
	'CircuitBreaker',
	'CircuitBreakerConfig', 
	'CircuitState',
	'create_circuit_breaker',
	'create_google_news_circuit_breaker',
	'RetryConfig',
	'RetryHandler',
	'ExponentialBackoff',
	'create_retry_handler',
	'ErrorHandler',
	'ErrorCategory',
	'ErrorContext',
	'RecoveryStrategy',
	'create_error_handler'
]