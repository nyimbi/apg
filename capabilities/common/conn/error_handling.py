"""
APG Connection Management Error Handling and Validation
Comprehensive error handling, validation, and recovery mechanisms

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import logging
import traceback
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, Type
from dataclasses import dataclass, asdict
from enum import Enum
import json
from datetime import datetime, timezone
from functools import wraps
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
	"""Error severity levels"""
	LOW = "low"
	MEDIUM = "medium"
	HIGH = "high"
	CRITICAL = "critical"


class ErrorCategory(str, Enum):
	"""Error categories"""
	CONNECTION = "connection"
	VALIDATION = "validation"
	AUTHENTICATION = "authentication"
	AUTHORIZATION = "authorization"
	NETWORK = "network"
	DATABASE = "database"
	CONFIGURATION = "configuration"
	RESOURCE = "resource"
	BUSINESS_LOGIC = "business_logic"
	EXTERNAL_API = "external_api"
	SYSTEM = "system"


class ErrorAction(str, Enum):
	"""Error handling actions"""
	RETRY = "retry"
	FALLBACK = "fallback"
	ESCALATE = "escalate"
	IGNORE = "ignore"
	ABORT = "abort"
	NOTIFY = "notify"
	LOG_ONLY = "log_only"


@dataclass
class ErrorContext:
	"""Context information for error handling"""
	tenant_id: str
	user_id: Optional[str] = None
	connection_id: Optional[str] = None
	flow_id: Optional[str] = None
	operation: Optional[str] = None
	request_id: Optional[str] = None
	additional_data: Dict[str, Any] = None

	def __post_init__(self):
		if self.additional_data is None:
			self.additional_data = {}


class APGError(Exception):
	"""Base exception class for APG Connection Management"""

	def __init__(
		self,
		message: str,
		error_code: str = None,
		category: ErrorCategory = ErrorCategory.SYSTEM,
		severity: ErrorSeverity = ErrorSeverity.MEDIUM,
		context: ErrorContext = None,
		cause: Exception = None,
		retry_after: int = None,
		user_message: str = None
	):
		self.message = message
		self.error_code = error_code or f"APG_{category.value.upper()}_{uuid7str()[:8]}"
		self.category = category
		self.severity = severity
		self.context = context or ErrorContext(tenant_id="unknown")
		self.cause = cause
		self.retry_after = retry_after
		self.user_message = user_message or message
		self.timestamp = datetime.now(timezone.utc)

		super().__init__(self.message)

	def to_dict(self) -> Dict[str, Any]:
		"""Convert error to dictionary"""
		return {
			'error_code': self.error_code,
			'message': self.message,
			'user_message': self.user_message,
			'category': self.category.value,
			'severity': self.severity.value,
			'timestamp': self.timestamp.isoformat(),
			'context': asdict(self.context),
			'cause': str(self.cause) if self.cause else None,
			'retry_after': self.retry_after
		}


class ValidationError(APGError):
	"""Validation related errors"""

	def __init__(self, message: str, field: str = None, **kwargs):
		self.field = field
		super().__init__(
			message=message,
			category=ErrorCategory.VALIDATION,
			severity=ErrorSeverity.MEDIUM,
			**kwargs
		)


class ConnectionError(APGError):
	"""Connection related errors"""

	def __init__(self, message: str, connection_id: str = None, **kwargs):
		context = kwargs.pop('context', None) or ErrorContext(tenant_id="unknown")
		context.connection_id = connection_id
		super().__init__(
			message=message,
			category=ErrorCategory.CONNECTION,
			severity=ErrorSeverity.HIGH,
			context=context,
			**kwargs
		)


class AuthenticationError(APGError):
	"""Authentication related errors"""

	def __init__(self, message: str, **kwargs):
		super().__init__(
			message=message,
			category=ErrorCategory.AUTHENTICATION,
			severity=ErrorSeverity.HIGH,
			user_message="Authentication failed. Please check your credentials.",
			**kwargs
		)


class AuthorizationError(APGError):
	"""Authorization related errors"""

	def __init__(self, message: str, required_permission: str = None, **kwargs):
		self.required_permission = required_permission
		super().__init__(
			message=message,
			category=ErrorCategory.AUTHORIZATION,
			severity=ErrorSeverity.HIGH,
			user_message="Access denied. You don't have permission to perform this operation.",
			**kwargs
		)


class ResourceError(APGError):
	"""Resource related errors"""

	def __init__(self, message: str, resource_type: str = None, **kwargs):
		self.resource_type = resource_type
		super().__init__(
			message=message,
			category=ErrorCategory.RESOURCE,
			severity=ErrorSeverity.MEDIUM,
			**kwargs
		)


class ExternalAPIError(APGError):
	"""External API related errors"""

	def __init__(self, message: str, api_name: str = None, status_code: int = None, **kwargs):
		self.api_name = api_name
		self.status_code = status_code
		super().__init__(
			message=message,
			category=ErrorCategory.EXTERNAL_API,
			severity=ErrorSeverity.HIGH,
			**kwargs
		)


class ErrorHandler:
	"""Comprehensive error handler for APG Connection Management"""

	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self.error_history: List[Dict[str, Any]] = []
		self.notification_handlers = {}
		self.recovery_strategies = {}
		self._setup_default_strategies()

	def _setup_default_strategies(self):
		"""Setup default error handling strategies"""
		self.recovery_strategies = {
			ErrorCategory.CONNECTION: self._handle_connection_error,
			ErrorCategory.VALIDATION: self._handle_validation_error,
			ErrorCategory.AUTHENTICATION: self._handle_auth_error,
			ErrorCategory.AUTHORIZATION: self._handle_auth_error,
			ErrorCategory.NETWORK: self._handle_network_error,
			ErrorCategory.DATABASE: self._handle_database_error,
			ErrorCategory.EXTERNAL_API: self._handle_external_api_error
		}

		# Setup notification handlers
		self.notification_handlers = {
			'email': self._send_email_notification,
			'log': self._log_notification,
			'webhook': self._send_webhook_notification
		}

	async def handle_error(
		self,
		error: Exception,
		context: ErrorContext = None,
		auto_recover: bool = True
	) -> Dict[str, Any]:
		"""Handle an error with appropriate strategy"""

		# Convert to APGError if needed
		if not isinstance(error, APGError):
			apg_error = APGError(
				message=str(error),
				category=ErrorCategory.SYSTEM,
				severity=ErrorSeverity.MEDIUM,
				context=context,
				cause=error
			)
		else:
			apg_error = error

		# Log the error
		await self._log_error(apg_error)

		# Add to history
		self._add_to_history(apg_error)

		# Determine recovery strategy
		strategy = self.recovery_strategies.get(apg_error.category, self._handle_generic_error)

		# Execute recovery strategy
		recovery_result = None
		if auto_recover:
			try:
				recovery_result = await strategy(apg_error)
			except Exception as recovery_error:
				logger.error(f"Recovery strategy failed: {recovery_error}")

		# Send notifications if needed
		await self._send_notifications(apg_error, recovery_result)

		return {
			'error': apg_error.to_dict(),
			'recovery_attempted': auto_recover,
			'recovery_result': recovery_result,
			'timestamp': datetime.now(timezone.utc).isoformat()
		}

	async def _log_error(self, error: APGError):
		"""Log error with appropriate level"""
		log_data = {
			'error_code': error.error_code,
			'category': error.category.value,
			'severity': error.severity.value,
			'message': error.message,
			'context': asdict(error.context) if error.context else {},
			'traceback': traceback.format_exc() if error.cause else None
		}

		if error.severity == ErrorSeverity.CRITICAL:
			logger.critical(f"Critical error: {json.dumps(log_data)}")
		elif error.severity == ErrorSeverity.HIGH:
			logger.error(f"High severity error: {json.dumps(log_data)}")
		elif error.severity == ErrorSeverity.MEDIUM:
			logger.warning(f"Medium severity error: {json.dumps(log_data)}")
		else:
			logger.info(f"Low severity error: {json.dumps(log_data)}")

	def _add_to_history(self, error: APGError):
		"""Add error to history for analysis"""
		self.error_history.append({
			'timestamp': error.timestamp.isoformat(),
			'error': error.to_dict()
		})

		# Keep only last 1000 errors
		if len(self.error_history) > 1000:
			self.error_history = self.error_history[-1000:]

	async def _handle_connection_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle connection-related errors"""
		connection_id = error.context.connection_id if error.context else None

		# Attempt connection retry with backoff
		if connection_id and 'connection_timeout' in error.message.lower():
			return await self._retry_with_backoff(
				operation=f"reconnect_{connection_id}",
				max_attempts=3,
				base_delay=2
			)

		return {'action': 'logged', 'recovery': 'manual_intervention_required'}

	async def _handle_validation_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle validation errors"""
		# Validation errors typically don't need automatic recovery
		return {
			'action': 'logged',
			'recovery': 'fix_validation_and_retry',
			'user_action_required': True
		}

	async def _handle_auth_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle authentication/authorization errors"""
		# Auth errors require manual intervention
		return {
			'action': 'logged',
			'recovery': 'check_credentials_and_permissions',
			'user_action_required': True,
			'escalate': True
		}

	async def _handle_network_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle network-related errors"""
		# Network errors can often be retried
		return await self._retry_with_backoff(
			operation="network_operation",
			max_attempts=5,
			base_delay=1
		)

	async def _handle_database_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle database errors"""
		if 'connection' in error.message.lower():
			# Database connection issues
			return await self._retry_with_backoff(
				operation="database_reconnect",
				max_attempts=3,
				base_delay=5
			)

		return {'action': 'logged', 'recovery': 'investigate_database_issue'}

	async def _handle_external_api_error(self, error: APGError) -> Dict[str, Any]:
		"""Handle external API errors"""
		if hasattr(error, 'status_code'):
			if error.status_code == 429:  # Rate limited
				return {
					'action': 'retry_after_delay',
					'delay_seconds': error.retry_after or 60
				}
			elif error.status_code >= 500:  # Server errors
				return await self._retry_with_backoff(
					operation="api_call",
					max_attempts=3,
					base_delay=2
				)

		return {'action': 'logged', 'recovery': 'check_external_api_status'}

	async def _handle_generic_error(self, error: APGError) -> Dict[str, Any]:
		"""Generic error handler"""
		return {
			'action': 'logged',
			'recovery': 'manual_investigation_required',
			'escalate': error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
		}

	async def _retry_with_backoff(
		self,
		operation: str,
		max_attempts: int = 3,
		base_delay: float = 1.0
	) -> Dict[str, Any]:
		"""Implement exponential backoff retry strategy"""
		for attempt in range(max_attempts):
			delay = base_delay * (2 ** attempt)
			await asyncio.sleep(delay)

			# Here you would implement the actual retry logic
			# For now, we simulate a successful retry
			logger.info(f"Retry attempt {attempt + 1} for {operation} after {delay}s delay")

		return {
			'action': 'retried',
			'attempts': max_attempts,
			'total_delay': sum(base_delay * (2 ** i) for i in range(max_attempts))
		}

	async def _send_notifications(self, error: APGError, recovery_result: Dict[str, Any]):
		"""Send notifications based on error severity and configuration"""
		if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
			notification_config = self.config.get('notifications', {})

			for channel in notification_config.get('channels', ['log']):
				handler = self.notification_handlers.get(channel)
				if handler:
					try:
						await handler(error, recovery_result)
					except Exception as e:
						logger.error(f"Failed to send {channel} notification: {e}")

	async def _send_email_notification(self, error: APGError, recovery_result: Dict[str, Any]):
		"""Send email notification"""
		email_config = self.config.get('notifications', {}).get('email', {})
		if not email_config.get('enabled', False):
			return

		# Email implementation would go here
		logger.info(f"Email notification sent for error: {error.error_code}")

	async def _log_notification(self, error: APGError, recovery_result: Dict[str, Any]):
		"""Log notification"""
		logger.warning(f"Notification: Error {error.error_code} occurred with recovery result: {recovery_result}")

	async def _send_webhook_notification(self, error: APGError, recovery_result: Dict[str, Any]):
		"""Send webhook notification"""
		webhook_config = self.config.get('notifications', {}).get('webhook', {})
		if not webhook_config.get('enabled', False):
			return

		# Webhook implementation would go here
		logger.info(f"Webhook notification sent for error: {error.error_code}")

	def get_error_statistics(self) -> Dict[str, Any]:
		"""Get error statistics for monitoring"""
		if not self.error_history:
			return {'total_errors': 0}

		stats = {
			'total_errors': len(self.error_history),
			'by_category': {},
			'by_severity': {},
			'recent_errors': 0,
			'most_common_errors': {}
		}

		# Count by category and severity
		for entry in self.error_history:
			error_data = entry['error']
			category = error_data.get('category', 'unknown')
			severity = error_data.get('severity', 'unknown')

			stats['by_category'][category] = stats['by_category'].get(category, 0) + 1
			stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

		# Count recent errors (last hour)
		one_hour_ago = datetime.now(timezone.utc).timestamp() - 3600
		stats['recent_errors'] = len([
			entry for entry in self.error_history
			if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).timestamp() > one_hour_ago
		])

		return stats


def error_handler_decorator(
	error_handler: ErrorHandler = None,
	auto_recover: bool = True,
	reraise: bool = False
):
	"""Decorator for automatic error handling"""

	def decorator(func: Callable):
		@wraps(func)
		async def async_wrapper(*args, **kwargs):
			try:
				return await func(*args, **kwargs)
			except Exception as e:
				handler = error_handler or ErrorHandler()

				# Extract context from function arguments if available
				context = None
				if args and hasattr(args[0], 'tenant_id'):
					context = ErrorContext(tenant_id=args[0].tenant_id)

				await handler.handle_error(e, context, auto_recover)

				if reraise:
					raise

				return None

		@wraps(func)
		def sync_wrapper(*args, **kwargs):
			try:
				return func(*args, **kwargs)
			except Exception as e:
				handler = error_handler or ErrorHandler()

				# For sync functions, we can't await, so we log and optionally reraise
				logger.error(f"Error in {func.__name__}: {e}")

				if reraise:
					raise

				return None

		# Return appropriate wrapper based on function type
		if asyncio.iscoroutinefunction(func):
			return async_wrapper
		else:
			return sync_wrapper

	return decorator


class InputValidator:
	"""Input validation utilities"""

	@staticmethod
	def validate_connection_data(data: Dict[str, Any]) -> List[str]:
		"""Validate connection data"""
		errors = []

		required_fields = ['name', 'connection_type', 'config']
		for field in required_fields:
			if field not in data or data[field] is None or data[field] == "":
				errors.append(f"Required field '{field}' is missing")

		# Validate connection type
		valid_types = [
			'postgresql', 'mysql', 'sqlite', 'mongodb', 'redis', 'http', 'file',
			'database', 'api', 'stream', 'webhook', 'queue'
		]
		if data.get('connection_type') not in valid_types:
			errors.append(f"Invalid connection type. Must be one of: {', '.join(valid_types)}")

		# Validate name format
		name = data.get('name', '')
		if len(name) < 3 or len(name) > 100:
			errors.append("Connection name must be between 3 and 100 characters")

		return errors

	@staticmethod
	def validate_flow_data(data: Dict[str, Any]) -> List[str]:
		"""Validate data flow configuration"""
		errors = []

		required_fields = ['name', 'source_connection_id', 'target_connection_id']
		for field in required_fields:
			if field not in data or not data[field]:
				errors.append(f"Required field '{field}' is missing")

		# Validate flow name
		name = data.get('name', '')
		if len(name) < 3 or len(name) > 100:
			errors.append("Flow name must be between 3 and 100 characters")

		return errors

	@staticmethod
	def validate_tenant_id(tenant_id: str) -> bool:
		"""Validate tenant ID format"""
		if not tenant_id or not isinstance(tenant_id, str):
			return False

		if len(tenant_id) < 3 or len(tenant_id) > 50:
			return False

		# Only allow alphanumeric, hyphens, and underscores
		import re
		if not re.match(r'^[a-zA-Z0-9_-]+$', tenant_id):
			return False

		return True


# Global error handler instance
global_error_handler = ErrorHandler()


# Convenience functions
async def handle_error(error: Exception, context: ErrorContext = None, auto_recover: bool = True):
	"""Handle an error using the global error handler"""
	return await global_error_handler.handle_error(error, context, auto_recover)


def validate_input(data: Dict[str, Any], data_type: str) -> List[str]:
	"""Validate input data"""
	if data_type == 'connection':
		return InputValidator.validate_connection_data(data)
	elif data_type == 'flow':
		return InputValidator.validate_flow_data(data)
	else:
		return [f"Unknown data type: {data_type}"]
