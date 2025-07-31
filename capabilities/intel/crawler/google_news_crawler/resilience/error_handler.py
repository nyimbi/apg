"""
Comprehensive Error Handling for Google News Crawler
===================================================

Centralized error handling with categorization, recovery strategies,
and integration with circuit breaker and retry mechanisms.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
	"""Categories of errors for different handling strategies."""
	# Network errors
	NETWORK_TIMEOUT = "network_timeout"
	NETWORK_CONNECTION = "network_connection" 
	NETWORK_DNS = "network_dns"
	
	# HTTP errors
	HTTP_CLIENT_ERROR = "http_client_error"  # 4xx
	HTTP_SERVER_ERROR = "http_server_error"  # 5xx
	HTTP_RATE_LIMITED = "http_rate_limited"   # 429
	HTTP_FORBIDDEN = "http_forbidden"         # 403
	HTTP_NOT_FOUND = "http_not_found"        # 404
	
	# Authentication errors
	AUTH_INVALID = "auth_invalid"
	AUTH_EXPIRED = "auth_expired"
	
	# Data parsing errors
	PARSE_JSON = "parse_json"
	PARSE_XML = "parse_xml"
	PARSE_HTML = "parse_html"
	PARSE_RSS = "parse_rss"
	
	# Database errors
	DB_CONNECTION = "db_connection"
	DB_QUERY = "db_query"
	DB_CONSTRAINT = "db_constraint"
	DB_TIMEOUT = "db_timeout"
	
	# External API errors
	API_QUOTA_EXCEEDED = "api_quota_exceeded"
	API_INVALID_REQUEST = "api_invalid_request"
	API_SERVICE_UNAVAILABLE = "api_service_unavailable"
	
	# Content quality errors
	CONTENT_EMPTY = "content_empty"
	CONTENT_INVALID = "content_invalid"
	CONTENT_TOO_SHORT = "content_too_short"
	
	# Configuration errors
	CONFIG_INVALID = "config_invalid"
	CONFIG_MISSING = "config_missing"
	
	# Unknown errors
	UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
	"""Recovery strategies for different error types."""
	RETRY_IMMEDIATELY = "retry_immediately"
	RETRY_WITH_BACKOFF = "retry_with_backoff"
	RETRY_AFTER_DELAY = "retry_after_delay"
	CIRCUIT_BREAKER = "circuit_breaker"
	FALLBACK_SOURCE = "fallback_source"
	SKIP_AND_CONTINUE = "skip_and_continue"
	FAIL_FAST = "fail_fast"
	DEGRADE_GRACEFULLY = "degrade_gracefully"
	ALERT_AND_CONTINUE = "alert_and_continue"

@dataclass
class ErrorContext:
	"""Context information for error handling."""
	error: Exception
	category: ErrorCategory
	source_url: Optional[str] = None
	operation: Optional[str] = None
	attempt_number: int = 1
	total_attempts: int = 1
	timestamp: datetime = field(default_factory=datetime.now)
	metadata: Dict[str, Any] = field(default_factory=dict)

class ErrorHandler:
	"""
	Comprehensive error handler with categorization and recovery strategies.
	"""
	
	def __init__(self):
		"""Initialize error handler."""
		self.error_stats = {
			'total_errors': 0,
			'errors_by_category': {},
			'errors_by_operation': {},
			'recovery_attempts': {},
			'start_time': datetime.now()
		}
		
		# Error categorization rules
		self.categorization_rules = self._setup_categorization_rules()
		
		# Recovery strategy mapping
		self.recovery_strategies = self._setup_recovery_strategies()
		
		# Custom error handlers
		self.custom_handlers: Dict[ErrorCategory, Callable] = {}
		
		logger.info("Error handler initialized")
	
	def _setup_categorization_rules(self) -> Dict[str, ErrorCategory]:
		"""Set up rules for categorizing exceptions."""
		return {
			# Network errors
			'TimeoutError': ErrorCategory.NETWORK_TIMEOUT,
			'asyncio.TimeoutError': ErrorCategory.NETWORK_TIMEOUT,
			'aiohttp.ServerTimeoutError': ErrorCategory.NETWORK_TIMEOUT,
			'ConnectionError': ErrorCategory.NETWORK_CONNECTION,
			'aiohttp.ClientConnectionError': ErrorCategory.NETWORK_CONNECTION,
			'aiohttp.ClientConnectorError': ErrorCategory.NETWORK_CONNECTION,
			'socket.gaierror': ErrorCategory.NETWORK_DNS,
			
			# HTTP errors (handled separately by status code)
			'aiohttp.ClientResponseError': ErrorCategory.HTTP_CLIENT_ERROR,
			
			# Authentication
			'aiohttp.ClientAuthenticationError': ErrorCategory.AUTH_INVALID,
			
			# Parsing errors
			'json.JSONDecodeError': ErrorCategory.PARSE_JSON,
			'xml.etree.ElementTree.ParseError': ErrorCategory.PARSE_XML,
			'bs4.FeatureNotFound': ErrorCategory.PARSE_HTML,
			'feedparser.exceptions.ParserError': ErrorCategory.PARSE_RSS,
			
			# Database errors
			'asyncpg.ConnectionDoesNotExistError': ErrorCategory.DB_CONNECTION,
			'asyncpg.InterfaceError': ErrorCategory.DB_CONNECTION,
			'asyncpg.PostgresError': ErrorCategory.DB_QUERY,
			'asyncpg.UniqueViolationError': ErrorCategory.DB_CONSTRAINT,
			'asyncpg.QueryCancelledError': ErrorCategory.DB_TIMEOUT,
			
			# Configuration errors
			'KeyError': ErrorCategory.CONFIG_MISSING,
			'ValueError': ErrorCategory.CONFIG_INVALID,
		}
	
	def _setup_recovery_strategies(self) -> Dict[ErrorCategory, RecoveryStrategy]:
		"""Set up recovery strategies for each error category."""
		return {
			# Network errors - retry with backoff
			ErrorCategory.NETWORK_TIMEOUT: RecoveryStrategy.RETRY_WITH_BACKOFF,
			ErrorCategory.NETWORK_CONNECTION: RecoveryStrategy.RETRY_WITH_BACKOFF,
			ErrorCategory.NETWORK_DNS: RecoveryStrategy.RETRY_AFTER_DELAY,
			
			# HTTP errors - different strategies by type
			ErrorCategory.HTTP_CLIENT_ERROR: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.HTTP_SERVER_ERROR: RecoveryStrategy.RETRY_WITH_BACKOFF,
			ErrorCategory.HTTP_RATE_LIMITED: RecoveryStrategy.RETRY_AFTER_DELAY,
			ErrorCategory.HTTP_FORBIDDEN: RecoveryStrategy.CIRCUIT_BREAKER,
			ErrorCategory.HTTP_NOT_FOUND: RecoveryStrategy.SKIP_AND_CONTINUE,
			
			# Authentication errors
			ErrorCategory.AUTH_INVALID: RecoveryStrategy.FAIL_FAST,
			ErrorCategory.AUTH_EXPIRED: RecoveryStrategy.ALERT_AND_CONTINUE,
			
			# Parsing errors - usually skip
			ErrorCategory.PARSE_JSON: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.PARSE_XML: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.PARSE_HTML: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.PARSE_RSS: RecoveryStrategy.FALLBACK_SOURCE,
			
			# Database errors
			ErrorCategory.DB_CONNECTION: RecoveryStrategy.RETRY_WITH_BACKOFF,
			ErrorCategory.DB_QUERY: RecoveryStrategy.RETRY_IMMEDIATELY,
			ErrorCategory.DB_CONSTRAINT: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.DB_TIMEOUT: RecoveryStrategy.RETRY_WITH_BACKOFF,
			
			# API errors
			ErrorCategory.API_QUOTA_EXCEEDED: RecoveryStrategy.CIRCUIT_BREAKER,
			ErrorCategory.API_INVALID_REQUEST: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.API_SERVICE_UNAVAILABLE: RecoveryStrategy.RETRY_WITH_BACKOFF,
			
			# Content quality errors
			ErrorCategory.CONTENT_EMPTY: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.CONTENT_INVALID: RecoveryStrategy.SKIP_AND_CONTINUE,
			ErrorCategory.CONTENT_TOO_SHORT: RecoveryStrategy.SKIP_AND_CONTINUE,
			
			# Configuration errors
			ErrorCategory.CONFIG_INVALID: RecoveryStrategy.FAIL_FAST,
			ErrorCategory.CONFIG_MISSING: RecoveryStrategy.DEGRADE_GRACEFULLY,
			
			# Unknown errors
			ErrorCategory.UNKNOWN: RecoveryStrategy.RETRY_WITH_BACKOFF,
		}
	
	def categorize_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
		"""
		Categorize an error based on its type and context.
		
		Args:
			error: The exception to categorize
			context: Additional context for categorization
			
		Returns:
			ErrorCategory: The categorized error type
		"""
		error_type = f"{error.__class__.__module__}.{error.__class__.__name__}"
		
		# Check for direct type matches
		if error_type in self.categorization_rules:
			return self.categorization_rules[error_type]
		
		# Check for class name matches
		class_name = error.__class__.__name__
		if class_name in self.categorization_rules:
			return self.categorization_rules[class_name]
		
		# Special handling for HTTP errors
		if hasattr(error, 'status'):
			status_code = error.status
			if 400 <= status_code < 500:
				if status_code == 429:
					return ErrorCategory.HTTP_RATE_LIMITED
				elif status_code == 403:
					return ErrorCategory.HTTP_FORBIDDEN
				elif status_code == 404:
					return ErrorCategory.HTTP_NOT_FOUND
				else:
					return ErrorCategory.HTTP_CLIENT_ERROR
			elif 500 <= status_code < 600:
				return ErrorCategory.HTTP_SERVER_ERROR
		
		# Content-specific errors
		error_message = str(error).lower()
		if 'empty' in error_message or 'no content' in error_message:
			return ErrorCategory.CONTENT_EMPTY
		elif 'too short' in error_message or 'insufficient content' in error_message:
			return ErrorCategory.CONTENT_TOO_SHORT
		elif 'quota' in error_message or 'rate limit' in error_message:
			return ErrorCategory.API_QUOTA_EXCEEDED
		
		return ErrorCategory.UNKNOWN
	
	def get_recovery_strategy(self, category: ErrorCategory) -> RecoveryStrategy:
		"""Get recovery strategy for error category."""
		return self.recovery_strategies.get(category, RecoveryStrategy.RETRY_WITH_BACKOFF)
	
	async def handle_error(self, 
						  error: Exception,
						  operation: str = "unknown",
						  source_url: Optional[str] = None,
						  attempt_number: int = 1,
						  **context_data) -> ErrorContext:
		"""
		Handle an error with appropriate categorization and logging.
		
		Args:
			error: The exception that occurred
			operation: Name of the operation that failed
			source_url: URL being processed when error occurred
			attempt_number: Current attempt number
			**context_data: Additional context data
			
		Returns:
			ErrorContext: Structured error context
		"""
		# Categorize the error
		category = self.categorize_error(error, context_data)
		
		# Create error context
		error_context = ErrorContext(
			error=error,
			category=category,
			source_url=source_url,
			operation=operation,
			attempt_number=attempt_number,
			metadata=context_data
		)
		
		# Update statistics
		self._update_error_stats(category, operation)
		
		# Log the error
		await self._log_error(error_context)
		
		# Execute custom handler if available
		if category in self.custom_handlers:
			try:
				await self.custom_handlers[category](error_context)
			except Exception as handler_error:
				logger.error(f"Custom error handler failed: {handler_error}")
		
		return error_context
	
	def _update_error_stats(self, category: ErrorCategory, operation: str) -> None:
		"""Update error statistics."""
		self.error_stats['total_errors'] += 1
		
		# Update category stats
		category_name = category.value
		if category_name not in self.error_stats['errors_by_category']:
			self.error_stats['errors_by_category'][category_name] = 0
		self.error_stats['errors_by_category'][category_name] += 1
		
		# Update operation stats
		if operation not in self.error_stats['errors_by_operation']:
			self.error_stats['errors_by_operation'][operation] = 0
		self.error_stats['errors_by_operation'][operation] += 1
	
	async def _log_error(self, context: ErrorContext) -> None:
		"""Log error with appropriate level and detail."""
		error = context.error
		category = context.category
		
		# Determine log level based on category
		if category in [ErrorCategory.CONFIG_INVALID, ErrorCategory.AUTH_INVALID]:
			log_level = logging.ERROR
		elif category in [ErrorCategory.NETWORK_TIMEOUT, ErrorCategory.HTTP_SERVER_ERROR]:
			log_level = logging.WARNING
		else:
			log_level = logging.INFO
		
		# Create log message
		message_parts = [
			f"Error in {context.operation}",
			f"Category: {category.value}",
			f"Type: {type(error).__name__}",
			f"Message: {str(error)}"
		]
		
		if context.source_url:
			# Extract domain for privacy
			try:
				domain = urlparse(context.source_url).netloc
				message_parts.append(f"Domain: {domain}")
			except:
				pass
		
		if context.attempt_number > 1:
			message_parts.append(f"Attempt: {context.attempt_number}")
		
		log_message = " | ".join(message_parts)
		
		# Log with stack trace for serious errors
		if log_level == logging.ERROR:
			logger.log(log_level, log_message, exc_info=True)
		else:
			logger.log(log_level, log_message)
	
	def register_custom_handler(self, category: ErrorCategory, handler: Callable) -> None:
		"""Register custom handler for specific error category."""
		self.custom_handlers[category] = handler
		logger.info(f"Registered custom handler for {category.value}")
	
	def should_retry(self, context: ErrorContext, max_attempts: int = 3) -> bool:
		"""
		Determine if operation should be retried based on error context.
		
		Args:
			context: Error context
			max_attempts: Maximum number of attempts allowed
			
		Returns:
			bool: True if should retry, False otherwise
		"""
		if context.attempt_number >= max_attempts:
			return False
		
		strategy = self.get_recovery_strategy(context.category)
		
		# Strategies that support retry
		retry_strategies = {
			RecoveryStrategy.RETRY_IMMEDIATELY,
			RecoveryStrategy.RETRY_WITH_BACKOFF,
			RecoveryStrategy.RETRY_AFTER_DELAY
		}
		
		return strategy in retry_strategies
	
	def get_retry_delay(self, context: ErrorContext, base_delay: float = 1.0) -> float:
		"""
		Calculate retry delay based on error context and attempt number.
		
		Args:
			context: Error context
			base_delay: Base delay in seconds
			
		Returns:
			float: Delay in seconds before retry
		"""
		strategy = self.get_recovery_strategy(context.category)
		
		if strategy == RecoveryStrategy.RETRY_IMMEDIATELY:
			return 0.0
		elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
			# Exponential backoff with jitter
			import random
			backoff_factor = 2 ** (context.attempt_number - 1)
			jitter = random.uniform(0.5, 1.5)
			return base_delay * backoff_factor * jitter
		elif strategy == RecoveryStrategy.RETRY_AFTER_DELAY:
			# Fixed delay based on error type
			delay_map = {
				ErrorCategory.NETWORK_DNS: 30.0,
				ErrorCategory.HTTP_RATE_LIMITED: 60.0,
				ErrorCategory.API_QUOTA_EXCEEDED: 300.0,
			}
			return delay_map.get(context.category, base_delay * 5)
		
		return base_delay
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get comprehensive error handling statistics."""
		current_time = datetime.now()
		uptime = current_time - self.error_stats['start_time']
		
		total_errors = self.error_stats['total_errors']
		
		return {
			'uptime_seconds': uptime.total_seconds(),
			'total_errors': total_errors,
			'errors_per_hour': (total_errors / max(1, uptime.total_seconds())) * 3600,
			'errors_by_category': self.error_stats['errors_by_category'].copy(),
			'errors_by_operation': self.error_stats['errors_by_operation'].copy(),
			'recovery_attempts': self.error_stats['recovery_attempts'].copy(),
			'most_common_errors': self._get_most_common_errors(),
			'error_trends': self._get_error_trends()
		}
	
	def _get_most_common_errors(self, limit: int = 5) -> List[Dict[str, Any]]:
		"""Get most common error categories."""
		sorted_errors = sorted(
			self.error_stats['errors_by_category'].items(),
			key=lambda x: x[1],
			reverse=True
		)
		
		return [
			{'category': category, 'count': count}
			for category, count in sorted_errors[:limit]
		]
	
	def _get_error_trends(self) -> Dict[str, Any]:
		"""Get error trend analysis."""
		# This would be enhanced with time-series data in production
		return {
			'trending_up': [],
			'trending_down': [],
			'stable': list(self.error_stats['errors_by_category'].keys())
		}

def create_error_handler() -> ErrorHandler:
	"""Factory function to create error handler."""
	return ErrorHandler()

# Common error handling decorators
def handle_network_errors(func):
	"""Decorator to handle common network errors."""
	async def wrapper(*args, **kwargs):
		try:
			return await func(*args, **kwargs)
		except (TimeoutError, ConnectionError) as e:
			logger.warning(f"Network error in {func.__name__}: {e}")
			raise
	return wrapper

def handle_parsing_errors(func):
	"""Decorator to handle parsing errors gracefully."""
	async def wrapper(*args, **kwargs):
		try:
			return await func(*args, **kwargs)
		except (ValueError, TypeError, AttributeError) as e:
			logger.warning(f"Parsing error in {func.__name__}: {e}")
			return None
	return wrapper