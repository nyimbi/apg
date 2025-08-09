"""
Logging configuration and utilities

© 2025 Datacraft. All rights reserved.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys
from datetime import datetime

from .constants import (
	LOG_LEVEL, LOG_FILE_PATH, LOG_MAX_SIZE, LOG_BACKUP_COUNT,
	DEBUG_MODE, VERBOSE_LOGGING
)


class ColoredFormatter(logging.Formatter):
	"""Colored console formatter for better readability"""
	
	# ANSI color codes
	COLORS = {
		'DEBUG': '\033[36m',      # Cyan
		'INFO': '\033[32m',       # Green
		'WARNING': '\033[33m',    # Yellow
		'ERROR': '\033[31m',      # Red
		'CRITICAL': '\033[35m',   # Magenta
		'RESET': '\033[0m'        # Reset
	}
	
	def format(self, record):
		"""Format log record with colors"""
		# Add color to levelname
		level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
		colored_levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
		
		# Create a copy of the record to avoid modifying the original
		record_copy = logging.makeLogRecord(record.__dict__)
		record_copy.levelname = colored_levelname
		
		return super().format(record_copy)


class ContextFilter(logging.Filter):
	"""Filter to add context information to log records"""
	
	def __init__(self, app_name: str = "APG-Mobile"):
		super().__init__()
		self.app_name = app_name
	
	def filter(self, record):
		"""Add context information to log record"""
		record.app_name = self.app_name
		record.timestamp = datetime.utcnow().isoformat()
		
		# Add thread information
		import threading
		record.thread_name = threading.current_thread().name
		
		# Add process information
		import os
		record.process_id = os.getpid()
		
		return True


def setup_logging(
	log_level: Optional[str] = None,
	log_file: Optional[Path] = None,
	enable_console: bool = True,
	enable_file: bool = True,
	app_name: str = "APG-Mobile"
) -> logging.Logger:
	"""Setup application logging configuration"""
	
	# Determine log level
	level = log_level or LOG_LEVEL
	if DEBUG_MODE:
		level = "DEBUG"
	elif VERBOSE_LOGGING:
		level = "INFO"
	
	# Convert string level to logging constant
	numeric_level = getattr(logging, level.upper(), logging.INFO)
	
	# Get root logger
	root_logger = logging.getLogger()
	root_logger.setLevel(numeric_level)
	
	# Clear existing handlers
	root_logger.handlers.clear()
	
	# Create context filter
	context_filter = ContextFilter(app_name)
	
	# Console handler
	if enable_console:
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(numeric_level)
		
		# Use colored formatter for console
		console_format = (
			"%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
		)
		if DEBUG_MODE or VERBOSE_LOGGING:
			console_format = (
				"%(asctime)s | %(levelname)-8s | %(name)-20s | "
				"%(funcName)s:%(lineno)d | %(message)s"
			)
		
		console_formatter = ColoredFormatter(
			console_format,
			datefmt='%H:%M:%S'
		)
		
		console_handler.setFormatter(console_formatter)
		console_handler.addFilter(context_filter)
		root_logger.addHandler(console_handler)
	
	# File handler
	if enable_file:
		file_path = log_file or LOG_FILE_PATH
		
		# Ensure log directory exists
		file_path.parent.mkdir(parents=True, exist_ok=True)
		
		# Use rotating file handler
		file_handler = logging.handlers.RotatingFileHandler(
			file_path,
			maxBytes=LOG_MAX_SIZE,
			backupCount=LOG_BACKUP_COUNT,
			encoding='utf-8'
		)
		file_handler.setLevel(numeric_level)
		
		# Detailed format for file logs
		file_format = (
			"%(timestamp)s | %(levelname)-8s | %(app_name)s | "
			"%(name)-25s | %(funcName)-15s:%(lineno)-4d | "
			"PID:%(process_id)d | %(thread_name)-10s | %(message)s"
		)
		
		file_formatter = logging.Formatter(file_format)
		file_handler.setFormatter(file_formatter)
		file_handler.addFilter(context_filter)
		root_logger.addHandler(file_handler)
	
	# Configure third-party loggers
	_configure_third_party_loggers(numeric_level)
	
	# Log startup message
	logger = logging.getLogger(__name__)
	logger.info(f"Logging configured - Level: {level}, Console: {enable_console}, File: {enable_file}")
	
	return root_logger


def _configure_third_party_loggers(level: int):
	"""Configure third-party library loggers"""
	
	# HTTP client libraries
	logging.getLogger("httpx").setLevel(max(level, logging.INFO))
	logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))
	logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
	
	# WebSocket libraries
	logging.getLogger("websockets").setLevel(max(level, logging.INFO))
	
	# Async libraries
	logging.getLogger("asyncio").setLevel(max(level, logging.WARNING))
	
	# BeeWare/Toga
	logging.getLogger("toga").setLevel(max(level, logging.INFO))
	
	# Database libraries
	logging.getLogger("sqlite3").setLevel(max(level, logging.WARNING))
	
	# Crypto libraries
	logging.getLogger("cryptography").setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
	"""Get logger with specified name"""
	return logging.getLogger(name)


def set_log_level(level: str):
	"""Change log level at runtime"""
	numeric_level = getattr(logging, level.upper(), logging.INFO)
	
	root_logger = logging.getLogger()
	root_logger.setLevel(numeric_level)
	
	# Update all handlers
	for handler in root_logger.handlers:
		handler.setLevel(numeric_level)
	
	logger = logging.getLogger(__name__)
	logger.info(f"Log level changed to: {level}")


def enable_debug_logging():
	"""Enable debug logging"""
	set_log_level("DEBUG")


def disable_debug_logging():
	"""Disable debug logging"""
	set_log_level("INFO")


class PerformanceLogger:
	"""Context manager for performance logging"""
	
	def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
		self.operation = operation
		self.logger = logger or logging.getLogger(__name__)
		self.start_time = None
	
	def __enter__(self):
		import time
		self.start_time = time.perf_counter()
		self.logger.debug(f"Starting operation: {self.operation}")
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		import time
		if self.start_time:
			duration = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
			if exc_type:
				self.logger.warning(f"Operation failed: {self.operation} ({duration:.2f}ms) - {exc_val}")
			else:
				self.logger.debug(f"Operation completed: {self.operation} ({duration:.2f}ms)")


def log_performance(operation: str, logger: Optional[logging.Logger] = None):
	"""Decorator for logging function performance"""
	def decorator(func):
		import functools
		import time
		
		@functools.wraps(func)
		def sync_wrapper(*args, **kwargs):
			perf_logger = logger or logging.getLogger(func.__module__)
			start_time = time.perf_counter()
			
			try:
				perf_logger.debug(f"Starting {operation}")
				result = func(*args, **kwargs)
				duration = (time.perf_counter() - start_time) * 1000
				perf_logger.debug(f"Completed {operation} ({duration:.2f}ms)")
				return result
			except Exception as e:
				duration = (time.perf_counter() - start_time) * 1000
				perf_logger.warning(f"Failed {operation} ({duration:.2f}ms) - {e}")
				raise
		
		@functools.wraps(func)
		async def async_wrapper(*args, **kwargs):
			perf_logger = logger or logging.getLogger(func.__module__)
			start_time = time.perf_counter()
			
			try:
				perf_logger.debug(f"Starting {operation}")
				result = await func(*args, **kwargs)
				duration = (time.perf_counter() - start_time) * 1000
				perf_logger.debug(f"Completed {operation} ({duration:.2f}ms)")
				return result
			except Exception as e:
				duration = (time.perf_counter() - start_time) * 1000
				perf_logger.warning(f"Failed {operation} ({duration:.2f}ms) - {e}")
				raise
		
		if asyncio.iscoroutinefunction(func):
			return async_wrapper
		else:
			return sync_wrapper
	
	return decorator


class LogCapture:
	"""Utility for capturing log messages in tests"""
	
	def __init__(self, logger_name: Optional[str] = None, level: int = logging.DEBUG):
		self.logger_name = logger_name
		self.level = level
		self.handler = None
		self.records = []
	
	def __enter__(self):
		import io
		
		# Create string buffer and handler
		self.buffer = io.StringIO()
		self.handler = logging.StreamHandler(self.buffer)
		self.handler.setLevel(self.level)
		
		# Add handler to logger
		logger = logging.getLogger(self.logger_name)
		logger.addHandler(self.handler)
		logger.setLevel(self.level)
		
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		if self.handler:
			logger = logging.getLogger(self.logger_name)
			logger.removeHandler(self.handler)
	
	def get_output(self) -> str:
		"""Get captured log output"""
		return self.buffer.getvalue()
	
	def get_records(self) -> list:
		"""Get captured log records"""
		return self.handler.buffer if hasattr(self.handler, 'buffer') else []


def setup_test_logging():
	"""Setup minimal logging for tests"""
	logging.basicConfig(
		level=logging.WARNING,
		format='%(levelname)s:%(name)s:%(message)s'
	)


# Import asyncio for performance logging
import asyncio