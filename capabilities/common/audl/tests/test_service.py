"""
Audit Service Tests

Unit tests for the AuditService class and service layer functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock

from ..service import (
	AuditService, AuditQueryFilter, AuditQueryResult,
	DatabaseHandler, FileHandler, get_audit_service, init_audit_service
)
from .. import AuditEntry, AuditLevel, AuditEventType


class TestAuditQueryFilter:
	"""Test AuditQueryFilter dataclass"""
	
	def test_default_filter(self):
		"""Test default filter values"""
		filter_obj = AuditQueryFilter()
		
		assert filter_obj.tenant_id is None
		assert filter_obj.user_id is None
		assert filter_obj.component is None
		assert filter_obj.event_type is None
		assert filter_obj.level is None
		assert filter_obj.start_time is None
		assert filter_obj.end_time is None
		assert filter_obj.resource is None
		assert filter_obj.success is None
		assert filter_obj.limit == 1000
		assert filter_obj.offset == 0

	def test_custom_filter(self):
		"""Test custom filter values"""
		start_time = datetime.utcnow() - timedelta(days=7)
		end_time = datetime.utcnow()
		
		filter_obj = AuditQueryFilter(
			tenant_id="tenant123",
			user_id="user456",
			component="auth",
			event_type=AuditEventType.USER_LOGIN,
			level=AuditLevel.INFO,
			start_time=start_time,
			end_time=end_time,
			resource="user_profile",
			success=True,
			limit=500,
			offset=100
		)
		
		assert filter_obj.tenant_id == "tenant123"
		assert filter_obj.user_id == "user456"
		assert filter_obj.component == "auth"
		assert filter_obj.event_type == AuditEventType.USER_LOGIN
		assert filter_obj.level == AuditLevel.INFO
		assert filter_obj.start_time == start_time
		assert filter_obj.end_time == end_time
		assert filter_obj.resource == "user_profile"
		assert filter_obj.success is True
		assert filter_obj.limit == 500
		assert filter_obj.offset == 100


class TestDatabaseHandler:
	"""Test DatabaseHandler functionality"""
	
	@pytest.fixture
	def temp_db_path(self):
		"""Create temporary database path"""
		with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
			db_path = Path(tmp.name)
		yield db_path
		if db_path.exists():
			db_path.unlink()

	@pytest.fixture
	async def db_handler(self, temp_db_path):
		"""Create DatabaseHandler instance"""
		handler = DatabaseHandler(temp_db_path)
		await handler.initialize()
		return handler

	async def test_initialization(self, temp_db_path):
		"""Test database initialization"""
		handler = DatabaseHandler(temp_db_path)
		assert not handler._initialized
		
		await handler.initialize()
		assert handler._initialized
		assert temp_db_path.exists()

	async def test_store_entry(self, db_handler):
		"""Test storing audit entry"""
		entry = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login",
			user_id="user123",
			tenant_id="tenant456",
			ip_address="192.168.1.100"
		)
		
		await db_handler.store_entry(entry)
		
		# Verify entry was stored
		import aiosqlite
		async with aiosqlite.connect(db_handler.db_path) as db:
			async with db.execute("SELECT * FROM audit_logs WHERE id = ?", (entry.id,)) as cursor:
				row = await cursor.fetchone()
				assert row is not None
				assert row[0] == entry.id
				assert row[2] == entry.level.value
				assert row[3] == entry.event_type.value

	async def test_query_entries_basic(self, db_handler):
		"""Test basic query functionality"""
		# Store test entries
		entries = []
		for i in range(5):
			entry = AuditEntry(
				level=AuditLevel.INFO,
				event_type=AuditEventType.USER_LOGIN,
				component="auth",
				action=f"login_{i}",
				user_id=f"user{i}"
			)
			entries.append(entry)
			await db_handler.store_entry(entry)

		# Query all entries
		filter_criteria = AuditQueryFilter(limit=10)
		result = await db_handler.query_entries(filter_criteria)
		
		assert result.total_count == 5
		assert len(result.entries) == 5
		assert not result.has_more

	async def test_query_entries_with_filters(self, db_handler):
		"""Test query with various filters"""
		# Store test entries with different attributes
		entry1 = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login",
			user_id="user123",
			tenant_id="tenant1"
		)
		entry2 = AuditEntry(
			level=AuditLevel.WARNING,
			event_type=AuditEventType.SECURITY_EVENT,
			component="security",
			action="suspicious_activity",
			user_id="user456",
			tenant_id="tenant2"
		)
		
		await db_handler.store_entry(entry1)
		await db_handler.store_entry(entry2)

		# Filter by tenant
		filter_criteria = AuditQueryFilter(tenant_id="tenant1")
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 1
		assert result.entries[0].tenant_id == "tenant1"

		# Filter by event type
		filter_criteria = AuditQueryFilter(event_type=AuditEventType.SECURITY_EVENT)
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 1
		assert result.entries[0].event_type == AuditEventType.SECURITY_EVENT

		# Filter by level
		filter_criteria = AuditQueryFilter(level=AuditLevel.WARNING)
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 1
		assert result.entries[0].level == AuditLevel.WARNING

	async def test_query_entries_pagination(self, db_handler):
		"""Test query pagination"""
		# Store 10 test entries
		for i in range(10):
			entry = AuditEntry(
				level=AuditLevel.INFO,
				event_type=AuditEventType.DATA_ACCESS,
				component="api",
				action=f"read_{i}",
				resource=f"resource_{i}"
			)
			await db_handler.store_entry(entry)

		# Query first page
		filter_criteria = AuditQueryFilter(limit=3, offset=0)
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 10
		assert len(result.entries) == 3
		assert result.has_more is True

		# Query second page
		filter_criteria = AuditQueryFilter(limit=3, offset=3)
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 10
		assert len(result.entries) == 3
		assert result.has_more is True

		# Query last page
		filter_criteria = AuditQueryFilter(limit=3, offset=9)
		result = await db_handler.query_entries(filter_criteria)
		assert result.total_count == 10
		assert len(result.entries) == 1
		assert result.has_more is False


class TestFileHandler:
	"""Test FileHandler functionality"""
	
	@pytest.fixture
	def temp_log_dir(self):
		"""Create temporary log directory"""
		with tempfile.TemporaryDirectory() as temp_dir:
			yield Path(temp_dir)

	@pytest.fixture
	def file_handler(self, temp_log_dir):
		"""Create FileHandler instance"""
		return FileHandler(temp_log_dir)

	async def test_store_entry(self, file_handler, temp_log_dir):
		"""Test storing entry to file"""
		entry = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login",
			user_id="user123"
		)
		
		await file_handler.store_entry(entry)
		
		# Check that log file was created
		date_str = entry.timestamp.strftime("%Y-%m-%d")
		log_file = temp_log_dir / f"audit_{date_str}.jsonl"
		assert log_file.exists()
		
		# Check file contents
		with open(log_file, "r") as f:
			line = f.readline().strip()
			data = json.loads(line)
			assert data["id"] == entry.id
			assert data["level"] == entry.level.value
			assert data["action"] == entry.action

	async def test_multiple_entries_same_day(self, file_handler, temp_log_dir):
		"""Test storing multiple entries on the same day"""
		entries = []
		for i in range(3):
			entry = AuditEntry(
				level=AuditLevel.INFO,
				event_type=AuditEventType.DATA_ACCESS,
				component="api",
				action=f"read_{i}"
			)
			entries.append(entry)
			await file_handler.store_entry(entry)

		# Check that all entries are in the same file
		date_str = entries[0].timestamp.strftime("%Y-%m-%d")
		log_file = temp_log_dir / f"audit_{date_str}.jsonl"
		
		with open(log_file, "r") as f:
			lines = f.readlines()
			assert len(lines) == 3
			
		# Verify each entry
		for i, line in enumerate(lines):
			data = json.loads(line.strip())
			assert data["action"] == f"read_{i}"


class TestAuditService:
	"""Test AuditService functionality"""
	
	@pytest.fixture
	def temp_config_dir(self):
		"""Create temporary config directory"""
		with tempfile.TemporaryDirectory() as temp_dir:
			yield Path(temp_dir)

	@pytest.fixture
	async def audit_service(self, temp_config_dir):
		"""Create AuditService instance"""
		service = AuditService(temp_config_dir)
		await service.initialize()
		return service

	async def test_initialization(self, temp_config_dir):
		"""Test service initialization"""
		service = AuditService(temp_config_dir)
		assert service.config_dir == temp_config_dir
		assert isinstance(service.logger, object)
		assert isinstance(service.db_handler, DatabaseHandler)
		assert isinstance(service.file_handler, FileHandler)

	async def test_audit_logging_integration(self, audit_service):
		"""Test integration with audit logger"""
		# Log an event through the logger
		await audit_service.logger.log_user_login(
			user_id="user123",
			ip_address="192.168.1.100"
		)
		
		# Query the logged event
		filter_criteria = AuditQueryFilter(user_id="user123")
		result = await audit_service.query_audit_logs(filter_criteria)
		
		assert result.total_count == 1
		assert result.entries[0].user_id == "user123"
		assert result.entries[0].event_type == AuditEventType.USER_LOGIN

	async def test_get_audit_summary(self, audit_service):
		"""Test audit summary generation"""
		# Log some test events
		await audit_service.logger.log_user_login("user1", success=True)
		await audit_service.logger.log_user_login("user2", success=False)
		await audit_service.logger.log_data_access("users", action="read")
		await audit_service.logger.log_security_event("brute_force")

		# Get summary
		summary = await audit_service.get_audit_summary(days=1)
		
		assert summary["total_events"] == 4
		assert summary["success_count"] == 2
		assert summary["failure_count"] == 2
		assert summary["success_rate"] == 50.0
		assert "user_login" in summary["event_types"]
		assert "security_event" in summary["event_types"]
		assert "INFO" in summary["levels"]
		assert "WARNING" in summary["levels"]

	async def test_export_audit_logs_json(self, audit_service):
		"""Test JSON export functionality"""
		# Log test events
		await audit_service.logger.log_user_login("user1")
		await audit_service.logger.log_data_access("users")

		# Export as JSON
		filter_criteria = AuditQueryFilter(limit=10)
		export_data = await audit_service.export_audit_logs(filter_criteria, "json")
		
		data = json.loads(export_data)
		assert "total_count" in data
		assert "entries" in data
		assert data["total_count"] == 2
		assert len(data["entries"]) == 2

	async def test_export_audit_logs_csv(self, audit_service):
		"""Test CSV export functionality"""
		# Log test events
		await audit_service.logger.log_user_login("user1")
		
		# Export as CSV
		filter_criteria = AuditQueryFilter(limit=10)
		export_data = await audit_service.export_audit_logs(filter_criteria, "csv")
		
		lines = export_data.strip().split("\n")
		assert len(lines) >= 2  # Header + at least one data row
		assert "id,timestamp,level,event_type" in lines[0]

	async def test_cleanup_old_logs(self, audit_service):
		"""Test cleanup functionality"""
		# Create old entry by manually setting timestamp
		old_timestamp = datetime.utcnow() - timedelta(days=100)
		old_entry = AuditEntry(
			level=AuditLevel.INFO,
			event_type=AuditEventType.USER_LOGIN,
			component="auth",
			action="login",
			timestamp=old_timestamp
		)
		await audit_service.db_handler.store_entry(old_entry)

		# Create recent entry
		await audit_service.logger.log_user_login("user1")

		# Verify we have 2 entries
		filter_criteria = AuditQueryFilter(limit=10)
		result = await audit_service.query_audit_logs(filter_criteria)
		assert result.total_count == 2

		# Cleanup entries older than 30 days
		await audit_service.cleanup_old_logs(retention_days=30)

		# Verify only recent entry remains
		result = await audit_service.query_audit_logs(filter_criteria)
		assert result.total_count == 1

	async def test_verify_log_integrity(self, audit_service):
		"""Test log integrity verification"""
		# Log some entries
		await audit_service.logger.log_user_login("user1")
		await audit_service.logger.log_data_access("users")

		# Verify integrity
		integrity_report = await audit_service.verify_log_integrity()
		
		assert integrity_report["total_entries"] == 2
		assert integrity_report["verified_count"] == 2
		assert integrity_report["corrupted_count"] == 0
		assert integrity_report["integrity_percentage"] == 100.0


class TestGlobalServiceFunctions:
	"""Test global service functions"""
	
	def test_get_audit_service(self):
		"""Test getting global audit service"""
		service1 = get_audit_service()
		service2 = get_audit_service()
		
		assert service1 is service2  # Should be singleton

	async def test_init_audit_service(self):
		"""Test initializing audit service"""
		with tempfile.TemporaryDirectory() as temp_dir:
			service = await init_audit_service(Path(temp_dir))
			assert isinstance(service, AuditService)
			assert service.config_dir == Path(temp_dir)