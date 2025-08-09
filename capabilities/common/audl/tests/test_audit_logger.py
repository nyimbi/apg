"""
Audit Logger Tests

Unit tests for the AuditLogger class and core functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock
from datetime import datetime

from .. import (
	AuditLogger, AuditEntry, AuditLevel, AuditEventType,
	get_audit_logger, init_audit_logging
)


class TestAuditEntry:
	"""Test AuditEntry model functionality"""
	
	def test_audit_entry_creation(self):
		"""Test creating AuditEntry with required fields"""
		entry = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login"
		)
		
		assert entry.level == AuditLevel.INFO
		assert entry.event_type == AuditEventType.USER_LOGIN
		assert entry.component == "auth"
		assert entry.action == "login"
		assert entry.success is True
		assert entry.id is not None
		assert entry.timestamp is not None

	def test_audit_entry_checksum_calculation(self):
		"""Test checksum calculation for integrity"""
		entry = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login"
		)
		
		# Checksum should be calculated
		assert entry.checksum is not None
		assert len(entry.checksum) == 64  # SHA-256 hex digest
		
		# Calculate expected checksum
		expected_checksum = entry._calculate_checksum()
		assert entry.checksum == expected_checksum

	def test_audit_entry_with_optional_fields(self):
		"""Test AuditEntry with all optional fields"""
		entry = AuditEntry(
			level=AuditLevel.WARNING,
			event_type=AuditEventType.SECURITY_EVENT,
			component="security",
			action="suspicious_activity",
			tenant_id="tenant123",
			user_id="user456",
			session_id="session789",
			resource="user_profile",
			resource_id="profile123",
			details={"threat_level": "medium"},
			ip_address="192.168.1.100",
			user_agent="Mozilla/5.0",
			success=False,
			error_message="Suspicious activity detected",
			duration_ms=150
		)
		
		assert entry.tenant_id == "tenant123"
		assert entry.user_id == "user456"
		assert entry.session_id == "session789"
		assert entry.resource == "user_profile"
		assert entry.resource_id == "profile123"
		assert entry.details == {"threat_level": "medium"}
		assert entry.ip_address == "192.168.1.100"
		assert entry.user_agent == "Mozilla/5.0"
		assert entry.success is False
		assert entry.error_message == "Suspicious activity detected"
		assert entry.duration_ms == 150


class TestAuditLogger:
	"""Test AuditLogger functionality"""
	
	@pytest.fixture
	def temp_config_dir(self):
		"""Create temporary config directory"""
		with tempfile.TemporaryDirectory() as temp_dir:
			yield Path(temp_dir)
	
	@pytest.fixture
	def audit_logger(self, temp_config_dir):
		"""Create AuditLogger instance"""
		return AuditLogger(temp_config_dir)
	
	def test_initialization(self, temp_config_dir):
		"""Test AuditLogger initialization"""
		logger = AuditLogger(temp_config_dir)
		
		assert logger._config_dir == temp_config_dir
		assert logger._enabled is True
		assert len(logger._log_handlers) == 0
		assert logger._context == {}

	def test_set_context(self, audit_logger):
		"""Test setting audit context"""
		audit_logger.set_context(
			tenant_id="tenant123",
			user_id="user456",
			session_id="session789"
		)
		
		assert audit_logger._context == {
			"tenant_id": "tenant123",
			"user_id": "user456",
			"session_id": "session789"
		}

	def test_add_remove_handler(self, audit_logger):
		"""Test adding and removing handlers"""
		def test_handler(entry):
			pass
		
		# Add handler
		audit_logger.add_handler(test_handler)
		assert test_handler in audit_logger._log_handlers
		
		# Remove handler
		audit_logger.remove_handler(test_handler)
		assert test_handler not in audit_logger._log_handlers

	async def test_basic_logging(self, audit_logger):
		"""Test basic audit logging"""
		handler_calls = []
		
		def test_handler(entry):
			handler_calls.append(entry)
		
		audit_logger.add_handler(test_handler)
		
		# Log an event
		entry = await audit_logger.log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login",
			user_id="user123"
		)
		
		assert entry is not None
		assert entry.level == AuditLevel.INFO
		assert entry.event_type == AuditEventType.USER_LOGIN
		assert entry.component == "auth"
		assert entry.action == "login"
		assert entry.user_id == "user123"
		
		# Handler should have been called
		assert len(handler_calls) == 1
		assert handler_calls[0] == entry

	async def test_logging_with_context(self, audit_logger):
		"""Test logging with preset context"""
		handler_calls = []
		
		def test_handler(entry):
			handler_calls.append(entry)
		
		audit_logger.add_handler(test_handler)
		audit_logger.set_context(tenant_id="tenant123", user_id="user456")
		
		# Log an event
		entry = await audit_logger.log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.DATA_ACCESS,
			component="api",
			action="read"
		)
		
		# Context should be included
		assert entry.tenant_id == "tenant123"
		assert entry.user_id == "user456"
		assert len(handler_calls) == 1

	async def test_async_handler(self, audit_logger):
		"""Test async handler support"""
		handler_calls = []
		
		async def async_handler(entry):
			handler_calls.append(entry)
		
		audit_logger.add_handler(async_handler)
		
		# Log an event
		entry = await audit_logger.log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.API_CALL,
			component="api",
			action="GET /users"
		)
		
		assert len(handler_calls) == 1
		assert handler_calls[0] == entry

	async def test_handler_error_handling(self, audit_logger):
		"""Test error handling in handlers"""
		def failing_handler(entry):
			raise Exception("Handler error")
		
		def working_handler(entry):
			working_handler.called = True
		working_handler.called = False
		
		audit_logger.add_handler(failing_handler)
		audit_logger.add_handler(working_handler)
		
		# Should not raise exception, but working handler should still be called
		entry = await audit_logger.log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login"
		)
		
		assert entry is not None
		assert working_handler.called is True

	async def test_disabled_logging(self, audit_logger):
		"""Test disabled logging"""
		handler_calls = []
		
		def test_handler(entry):
			handler_calls.append(entry)
		
		audit_logger.add_handler(test_handler)
		audit_logger.disable()
		
		# Log an event
		entry = await audit_logger.log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login"
		)
		
		# Should return None and not call handlers
		assert entry is None
		assert len(handler_calls) == 0
		
		# Re-enable
		audit_logger.enable()
		assert audit_logger.is_enabled() is True

	async def test_convenience_methods(self, audit_logger):
		"""Test convenience logging methods"""
		handler_calls = []
		
		def test_handler(entry):
			handler_calls.append(entry)
		
		audit_logger.add_handler(test_handler)
		
		# Test user login
		entry = await audit_logger.log_user_login(
			user_id="user123",
			success=True,
			ip_address="192.168.1.100"
		)
		
		assert entry.event_type == AuditEventType.USER_LOGIN
		assert entry.user_id == "user123"
		assert entry.success is True
		assert entry.ip_address == "192.168.1.100"
		
		# Test failed login
		entry = await audit_logger.log_user_login(
			user_id="user123",
			success=False,
			ip_address="192.168.1.100"
		)
		
		assert entry.event_type == AuditEventType.USER_FAILED_LOGIN
		assert entry.success is False
		
		# Test data access
		entry = await audit_logger.log_data_access(
			resource="users",
			resource_id="user123",
			action="read"
		)
		
		assert entry.event_type == AuditEventType.DATA_ACCESS
		assert entry.resource == "users"
		assert entry.resource_id == "user123"
		assert entry.action == "read"
		
		# Test security event
		entry = await audit_logger.log_security_event(
			action="brute_force_attempt",
			details={"attempts": 5}
		)
		
		assert entry.event_type == AuditEventType.SECURITY_EVENT
		assert entry.action == "brute_force_attempt"
		assert entry.details == {"attempts": 5}
		
		# Test API call
		entry = await audit_logger.log_api_call(
			endpoint="/api/users",
			method="GET",
			status_code=200,
			duration_ms=150
		)
		
		assert entry.event_type == AuditEventType.API_CALL
		assert entry.action == "GET /api/users"
		assert entry.success is True
		assert entry.duration_ms == 150
		assert entry.details["status_code"] == 200
		
		# All handlers should have been called
		assert len(handler_calls) == 5


class TestGlobalFunctions:
	"""Test global audit logging functions"""
	
	def test_get_audit_logger(self):
		"""Test getting global audit logger"""
		logger1 = get_audit_logger()
		logger2 = get_audit_logger()
		
		assert logger1 is logger2  # Should be singleton

	def test_init_audit_logging(self):
		"""Test initializing audit logging"""
		with tempfile.TemporaryDirectory() as temp_dir:
			logger = init_audit_logging(Path(temp_dir))
			assert isinstance(logger, AuditLogger)
			assert logger._config_dir == Path(temp_dir)

	async def test_convenience_functions(self):
		"""Test convenience audit logging functions"""
		from .. import (
			audit_log, audit_user_login, audit_data_access,
			audit_security_event, audit_api_call
		)
		
		# Test audit_log
		entry = await audit_log(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login"
		)
		assert entry.level == AuditLevel.INFO
		assert entry.event_type == AuditEventType.USER_LOGIN
		
		# Test other convenience functions
		entry = await audit_user_login("user123")
		assert entry.event_type == AuditEventType.USER_LOGIN
		
		entry = await audit_data_access("users", action="read")
		assert entry.event_type == AuditEventType.DATA_ACCESS
		
		entry = await audit_security_event("test_event")
		assert entry.event_type == AuditEventType.SECURITY_EVENT
		
		entry = await audit_api_call("/api/test", "GET", 200)
		assert entry.event_type == AuditEventType.API_CALL