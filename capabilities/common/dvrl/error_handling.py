#!/usr/bin/env python3
"""
APG Data Virtualization (DVRL) Comprehensive Error Handling and Logging
Production-grade error handling with integrated monitoring and alerting

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import json
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union
from uuid_extensions import uuid7str

# Import real error handling implementations
try:
	from .real_implementations import RealErrorHandler, RealLoggingManager
	REAL_ERROR_HANDLING_AVAILABLE = True
except ImportError:
	REAL_ERROR_HANDLING_AVAILABLE = False

class DVRLErrorHandler:
	"""Enhanced DVRL error handler with production capabilities"""
	
	def __init__(self, tenant_id: str, user_id: str):
		self.tenant_id = tenant_id
		self.user_id = user_id
		
		# Use real error handler if available
		if REAL_ERROR_HANDLING_AVAILABLE:
			self.error_handler = RealErrorHandler(tenant_id, user_id, "DVRL")
			self.logging_manager = RealLoggingManager(tenant_id, "DVRL")
		else:
			self.error_handler = None
			self.logging_manager = None
		
		# Fallback error tracking
		self.error_history = []
		self.error_counts = {}
	
	async def handle_error(
		self, 
		error: Exception, 
		context: Dict[str, Any], 
		operation: str,
		severity: str = "ERROR"
	) -> Dict[str, Any]:
		"""Handle errors with comprehensive logging and recovery suggestions"""
		if self.error_handler:
			return await self.error_handler.handle_exception(error, context, operation, severity)
		else:
			# Fallback error handling
			return await self._fallback_error_handling(error, context, operation, severity)
	
	async def _fallback_error_handling(
		self, 
		error: Exception, 
		context: Dict[str, Any], 
		operation: str, 
		severity: str
	) -> Dict[str, Any]:
		"""Fallback error handling when real implementation is not available"""
		error_id = uuid7str()
		timestamp = datetime.now(timezone.utc).isoformat()
		
		error_info = {
			'error_id': error_id,
			'timestamp': timestamp,
			'tenant_id': self.tenant_id,
			'user_id': self.user_id,
			'operation': operation,
			'severity': severity,
			'error_type': type(error).__name__,
			'error_message': str(error),
			'context': context,
			'stack_trace': traceback.format_exc()
		}
		
		# Log error
		print(f"[{timestamp}] DVRL {severity} [{error_id}]: {operation} - {str(error)}")
		
		# Track error
		self.error_history.append(error_info)
		error_key = f"{operation}:{type(error).__name__}"
		self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
		
		return error_info
	
	async def log(
		self, 
		level: str, 
		message: str, 
		context: Optional[Dict[str, Any]] = None,
		operation: Optional[str] = None
	) -> None:
		"""Log message with appropriate level"""
		if self.logging_manager:
			await self.logging_manager.log(level, message, context, operation, self.user_id)
		else:
			# Fallback logging
			timestamp = datetime.now(timezone.utc).isoformat()
			operation_str = f" [{operation}]" if operation else ""
			context_str = f" | Context: {json.dumps(context, default=str)}" if context else ""
			print(f"[{timestamp}] DVRL {level.upper()}{operation_str}: {message}{context_str}")
	
	async def info(self, message: str, **kwargs) -> None:
		"""Log info message"""
		await self.log('INFO', message, **kwargs)
	
	async def warning(self, message: str, **kwargs) -> None:
		"""Log warning message"""
		await self.log('WARNING', message, **kwargs)
	
	async def error(self, message: str, **kwargs) -> None:
		"""Log error message"""
		await self.log('ERROR', message, **kwargs)
	
	async def critical(self, message: str, **kwargs) -> None:
		"""Log critical message"""
		await self.log('CRITICAL', message, **kwargs)
	
	async def get_error_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive error statistics"""
		if self.error_handler:
			return await self.error_handler.get_error_statistics()
		else:
			# Fallback statistics
			return {
				'total_errors': len(self.error_history),
				'error_counts': self.error_counts,
				'recent_errors': self.error_history[-10:] if self.error_history else []
			}

def error_handler_decorator(operation: str, severity: str = "ERROR"):
	"""Decorator to automatically handle errors in functions"""
	def decorator(func: Callable):
		@wraps(func)
		async def async_wrapper(*args, **kwargs):
			# Try to find error handler in instance
			error_handler = None
			if args and hasattr(args[0], 'error_handler'):
				error_handler = args[0].error_handler
			
			try:
				if asyncio.iscoroutinefunction(func):
					return await func(*args, **kwargs)
				else:
					return func(*args, **kwargs)
			except Exception as e:
				context = {
					'function': func.__name__,
					'args_count': len(args),
					'kwargs_keys': list(kwargs.keys())
				}
				
				if error_handler:
					await error_handler.handle_error(e, context, operation, severity)
				else:
					# Fallback error logging
					timestamp = datetime.now(timezone.utc).isoformat()
					print(f"[{timestamp}] DVRL {severity}: {operation} failed in {func.__name__}: {str(e)}")
				
				raise  # Re-raise the exception after logging
		
		@wraps(func)
		def sync_wrapper(*args, **kwargs):
			# Try to find error handler in instance
			error_handler = None
			if args and hasattr(args[0], 'error_handler'):
				error_handler = args[0].error_handler
			
			try:
				return func(*args, **kwargs)
			except Exception as e:
				context = {
					'function': func.__name__,
					'args_count': len(args),
					'kwargs_keys': list(kwargs.keys())
				}
				
				if error_handler:
					# Can't await in sync function, so create task
					asyncio.create_task(error_handler.handle_error(e, context, operation, severity))
				else:
					# Fallback error logging
					timestamp = datetime.now(timezone.utc).isoformat()
					print(f"[{timestamp}] DVRL {severity}: {operation} failed in {func.__name__}: {str(e)}")
				
				raise  # Re-raise the exception after logging
		
		if asyncio.iscoroutinefunction(func):
			return async_wrapper
		else:
			return sync_wrapper
	
	return decorator

class DVRLLoggingContext:
	"""Context manager for maintaining logging context"""
	
	def __init__(self, error_handler: DVRLErrorHandler, context: Dict[str, Any]):
		self.error_handler = error_handler
		self.context = context
		self.original_context = None
	
	async def __aenter__(self):
		if self.error_handler.logging_manager:
			await self.error_handler.logging_manager.set_context(self.context)
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		if self.error_handler.logging_manager:
			await self.error_handler.logging_manager.clear_context()
		
		# Handle any exceptions that occurred in the context
		if exc_type and exc_val:
			await self.error_handler.handle_error(
				exc_val, 
				self.context, 
				"logging_context",
				"ERROR"
			)
		
		return False  # Don't suppress exceptions

class DVRLPerformanceMonitor:
	"""Monitor performance and trigger alerts for slow operations"""
	
	def __init__(self, error_handler: DVRLErrorHandler):
		self.error_handler = error_handler
		self.performance_thresholds = {
			'query_execution': 5000,  # 5 seconds
			'schema_discovery': 10000,  # 10 seconds
			'connection_establishment': 3000,  # 3 seconds
			'data_federation': 15000  # 15 seconds
		}
	
	async def monitor_operation(
		self, 
		operation: str, 
		operation_type: str = "query_execution"
	):
		"""Context manager to monitor operation performance"""
		return DVRLOperationMonitor(self, operation, operation_type)

class DVRLOperationMonitor:
	"""Context manager for monitoring individual operations"""
	
	def __init__(self, perf_monitor: DVRLPerformanceMonitor, operation: str, operation_type: str):
		self.perf_monitor = perf_monitor
		self.operation = operation
		self.operation_type = operation_type
		self.start_time = None
	
	async def __aenter__(self):
		self.start_time = datetime.now(timezone.utc)
		await self.perf_monitor.error_handler.info(
			f"Starting operation: {self.operation}",
			operation=self.operation,
			context={'operation_type': self.operation_type}
		)
		return self
	
	async def __aexit__(self, exc_type, exc_val, exc_tb):
		end_time = datetime.now(timezone.utc)
		duration_ms = int((end_time - self.start_time).total_seconds() * 1000)
		
		# Check if operation exceeded threshold
		threshold = self.perf_monitor.performance_thresholds.get(self.operation_type, 30000)
		
		if duration_ms > threshold:
			await self.perf_monitor.error_handler.warning(
				f"Operation {self.operation} exceeded threshold",
				operation=self.operation,
				context={
					'duration_ms': duration_ms,
					'threshold_ms': threshold,
					'operation_type': self.operation_type
				}
			)
		else:
			await self.perf_monitor.error_handler.info(
				f"Completed operation: {self.operation}",
				operation=self.operation,
				context={
					'duration_ms': duration_ms,
					'operation_type': self.operation_type
				}
			)
		
		return False  # Don't suppress exceptions

class DVRLRetryHandler:
	"""Handle retries with exponential backoff and circuit breaker pattern"""
	
	def __init__(self, error_handler: DVRLErrorHandler):
		self.error_handler = error_handler
		self.retry_config = {
			'max_retries': 3,
			'base_delay': 1.0,
			'max_delay': 30.0,
			'exponential_base': 2.0
		}
		self.circuit_breaker_thresholds = {
			'failure_rate': 0.5,  # 50% failure rate
			'min_requests': 5,
			'time_window': 60  # 60 seconds
		}
	
	async def retry_operation(
		self, 
		operation: Callable,
		operation_name: str,
		*args, 
		**kwargs
	) -> Any:
		"""Retry operation with exponential backoff"""
		last_error = None
		
		for attempt in range(self.retry_config['max_retries'] + 1):
			try:
				if attempt > 0:
					delay = min(
						self.retry_config['base_delay'] * (self.retry_config['exponential_base'] ** (attempt - 1)),
						self.retry_config['max_delay']
					)
					await self.error_handler.info(
						f"Retrying {operation_name} (attempt {attempt + 1})",
						operation=operation_name,
						context={'delay_seconds': delay}
					)
					await asyncio.sleep(delay)
				
				if asyncio.iscoroutinefunction(operation):
					return await operation(*args, **kwargs)
				else:
					return operation(*args, **kwargs)
			
			except Exception as e:
				last_error = e
				
				await self.error_handler.handle_error(
					e,
					{
						'operation': operation_name,
						'attempt': attempt + 1,
						'max_attempts': self.retry_config['max_retries'] + 1
					},
					f"{operation_name}_retry",
					"WARNING" if attempt < self.retry_config['max_retries'] else "ERROR"
				)
				
				# Don't retry on certain types of errors
				if self._is_non_retryable_error(e):
					await self.error_handler.warning(
						f"Non-retryable error in {operation_name}",
						operation=operation_name,
						context={'error_type': type(e).__name__}
					)
					break
		
		# All retries failed
		if last_error:
			await self.error_handler.error(
				f"All retry attempts failed for {operation_name}",
				operation=operation_name,
				context={'total_attempts': self.retry_config['max_retries'] + 1}
			)
			raise last_error
	
	def _is_non_retryable_error(self, error: Exception) -> bool:
		"""Check if error should not be retried"""
		non_retryable_types = [
			'AuthenticationError',
			'PermissionError', 
			'ValidationError',
			'SyntaxError',
			'TypeError',
			'ValueError'
		]
		
		error_type = type(error).__name__
		error_message = str(error).lower()
		
		# Check error type
		if error_type in non_retryable_types:
			return True
		
		# Check error message for specific patterns
		non_retryable_patterns = [
			'permission denied',
			'access denied', 
			'authentication failed',
			'invalid syntax',
			'bad request'
		]
		
		return any(pattern in error_message for pattern in non_retryable_patterns)


# Global error handling utilities

async def safe_execute(
	operation: Callable,
	error_handler: DVRLErrorHandler,
	operation_name: str,
	*args,
	**kwargs
) -> Tuple[Any, Optional[Dict[str, Any]]]:
	"""Safely execute operation with comprehensive error handling"""
	try:
		if asyncio.iscoroutinefunction(operation):
			result = await operation(*args, **kwargs)
		else:
			result = operation(*args, **kwargs)
		
		return result, None
	
	except Exception as e:
		error_context = await error_handler.handle_error(
			e,
			{
				'operation': operation_name,
				'args_count': len(args),
				'kwargs_keys': list(kwargs.keys())
			},
			operation_name
		)
		return None, error_context

def create_error_handler(tenant_id: str, user_id: str) -> DVRLErrorHandler:
	"""Factory function to create error handler"""
	return DVRLErrorHandler(tenant_id, user_id)

# Export error handling components
__all__ = [
	"DVRLErrorHandler",
	"DVRLLoggingContext", 
	"DVRLPerformanceMonitor",
	"DVRLOperationMonitor",
	"DVRLRetryHandler",
	"error_handler_decorator",
	"safe_execute",
	"create_error_handler"
]