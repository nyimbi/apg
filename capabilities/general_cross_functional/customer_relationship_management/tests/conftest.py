"""
APG Customer Relationship Management - Test Configuration

Pytest configuration and fixtures for comprehensive CRM testing
providing database setup, mock services, and test data factories.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import pytest
import asyncpg
import logging
from datetime import datetime, date
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock
from uuid_extensions import uuid7str

# Local imports
from ..models import (
	CRMContact, CRMAccount, CRMLead, CRMOpportunity, CRMActivity, CRMCampaign,
	ContactType, LeadSource, LeadStatus, OpportunityStage, ActivityType, ActivityStatus,
	CampaignType, CampaignStatus, Address, PhoneNumber
)
from ..database import DatabaseManager
from ..service import CRMService
from ..auth_integration import CRMAuthProvider, CRMUserContext, CRMRole, CRMPermission
from ..migrations.migration_manager import MigrationManager
from . import TEST_DATABASE_CONFIG, TEST_TENANT_ID, TEST_USER_ID


# Configure test logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture(scope="session")
async def test_database():
	"""Setup test database with schema"""
	# Create test database if it doesn't exist
	try:
		# Connect to default database to create test database
		default_config = TEST_DATABASE_CONFIG.copy()
		default_config["database"] = "postgres"
		
		conn = await asyncpg.connect(**default_config)
		
		# Check if test database exists
		db_exists = await conn.fetchval(
			"SELECT 1 FROM pg_database WHERE datname = $1",
			TEST_DATABASE_CONFIG["database"]
		)
		
		if not db_exists:
			await conn.execute(f'CREATE DATABASE "{TEST_DATABASE_CONFIG["database"]}"')
			logger.info(f"Created test database: {TEST_DATABASE_CONFIG['database']}")
		
		await conn.close()
		
		# Run migrations on test database
		migration_manager = MigrationManager(TEST_DATABASE_CONFIG)
		await migration_manager.initialize()
		await migration_manager.migrate_to_latest()
		await migration_manager.shutdown()
		
		yield TEST_DATABASE_CONFIG
		
		# Cleanup: Drop test database
		conn = await asyncpg.connect(**default_config)
		await conn.execute(f'DROP DATABASE IF EXISTS "{TEST_DATABASE_CONFIG["database"]}"')
		await conn.close()
		
	except Exception as e:
		logger.error(f"Failed to setup test database: {e}")
		pytest.skip(f"Cannot setup test database: {e}")


@pytest.fixture
async def db_connection(test_database):
	"""Create a database connection for testing"""
	conn = await asyncpg.connect(**test_database)
	
	# Start transaction for test isolation
	tx = conn.transaction()
	await tx.start()
	
	yield conn
	
	# Rollback transaction to clean up test data
	await tx.rollback()
	await conn.close()


@pytest.fixture
async def database_manager(test_database):
	"""Create database manager instance"""
	manager = DatabaseManager(test_database)
	await manager.initialize()
	yield manager
	await manager.shutdown()


@pytest.fixture
async def crm_service(database_manager):
	"""Create CRM service instance with mocked dependencies"""
	# Mock APG integrations
	mock_event_bus = AsyncMock()
	mock_ai_insights = AsyncMock()
	mock_analytics = AsyncMock()
	
	service = CRMService(
		database_manager=database_manager,
		event_bus=mock_event_bus,
		ai_insights=mock_ai_insights,
		analytics=mock_analytics
	)
	
	await service.initialize()
	yield service
	await service.shutdown()


@pytest.fixture
def mock_auth_provider():
	"""Create mock authentication provider"""
	provider = Mock(spec=CRMAuthProvider)
	provider.authenticate_user = AsyncMock()
	provider.validate_token = AsyncMock()
	provider.check_permission = AsyncMock(return_value=True)
	provider.create_token = AsyncMock(return_value="mock_jwt_token")
	return provider


@pytest.fixture
def test_user_context():
	"""Create test user context"""
	return CRMUserContext(
		user_id=TEST_USER_ID,
		username="testuser",
		email="test@example.com",
		tenant_id=TEST_TENANT_ID,
		roles=[CRMRole.SALES_REP],
		permissions={
			CRMPermission.CONTACT_CREATE,
			CRMPermission.CONTACT_READ,
			CRMPermission.CONTACT_UPDATE,
			CRMPermission.LEAD_CREATE,
			CRMPermission.LEAD_READ,
			CRMPermission.OPPORTUNITY_CREATE,
			CRMPermission.OPPORTUNITY_READ
		},
		territories=["territory_1"],
		department_id="sales",
		is_active=True,
		login_time=datetime.utcnow()
	)


# Test data factories

@pytest.fixture
def contact_factory():
	"""Factory for creating test contacts"""
	def _create_contact(**kwargs) -> CRMContact:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"first_name": "John",
			"last_name": "Doe",
			"email": "john.doe@example.com",
			"phone": "+1-555-0123",
			"company": "Test Company",
			"job_title": "Manager",
			"contact_type": ContactType.PROSPECT,
			"lead_source": LeadSource.WEBSITE,
			"lead_score": 75.5,
			"owner_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMContact(**defaults)
	
	return _create_contact


@pytest.fixture
def account_factory():
	"""Factory for creating test accounts"""
	def _create_account(**kwargs) -> CRMAccount:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"account_name": "Test Account Corp",
			"industry": "Technology",
			"annual_revenue": 1000000.00,
			"employee_count": 50,
			"website": "https://testaccount.com",
			"description": "A test account for testing purposes",
			"account_health_score": 80.0,
			"owner_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMAccount(**defaults)
	
	return _create_account


@pytest.fixture
def lead_factory():
	"""Factory for creating test leads"""
	def _create_lead(**kwargs) -> CRMLead:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"first_name": "Jane",
			"last_name": "Smith",
			"email": "jane.smith@example.com",
			"phone": "+1-555-0456",
			"company": "Lead Company",
			"job_title": "Director",
			"lead_source": LeadSource.REFERRAL,
			"lead_status": LeadStatus.NEW,
			"lead_score": 65.0,
			"budget": 50000.00,
			"timeline": "Q2 2025",
			"owner_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMLead(**defaults)
	
	return _create_lead


@pytest.fixture
def opportunity_factory():
	"""Factory for creating test opportunities"""
	def _create_opportunity(**kwargs) -> CRMOpportunity:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"opportunity_name": "Test Opportunity",
			"stage": OpportunityStage.QUALIFICATION,
			"amount": 75000.00,
			"probability": 40.0,
			"close_date": date(2025, 6, 30),
			"lead_source": LeadSource.WEBSITE,
			"description": "A test opportunity for testing",
			"owner_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMOpportunity(**defaults)
	
	return _create_opportunity


@pytest.fixture
def activity_factory():
	"""Factory for creating test activities"""
	def _create_activity(**kwargs) -> CRMActivity:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"activity_type": ActivityType.CALL,
			"subject": "Test Call",
			"description": "A test call activity",
			"status": ActivityStatus.PLANNED,
			"priority": "high",
			"due_date": datetime.utcnow(),
			"duration_minutes": 30,
			"owner_id": TEST_USER_ID,
			"assigned_to_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMActivity(**defaults)
	
	return _create_activity


@pytest.fixture
def campaign_factory():
	"""Factory for creating test campaigns"""
	def _create_campaign(**kwargs) -> CRMCampaign:
		defaults = {
			"id": uuid7str(),
			"tenant_id": TEST_TENANT_ID,
			"campaign_name": "Test Campaign",
			"campaign_type": CampaignType.EMAIL,
			"status": CampaignStatus.DRAFT,
			"description": "A test campaign for testing",
			"start_date": date.today(),
			"budget": 10000.00,
			"expected_response_rate": 5.0,
			"owner_id": TEST_USER_ID,
			"created_by": TEST_USER_ID,
			"updated_by": TEST_USER_ID,
			"created_at": datetime.utcnow(),
			"updated_at": datetime.utcnow()
		}
		defaults.update(kwargs)
		return CRMCampaign(**defaults)
	
	return _create_campaign


# Mock APG service fixtures

@pytest.fixture
def mock_event_bus():
	"""Mock APG event bus"""
	bus = AsyncMock()
	bus.publish = AsyncMock()
	bus.subscribe = AsyncMock()
	return bus


@pytest.fixture
def mock_ai_insights():
	"""Mock AI insights service"""
	insights = AsyncMock()
	insights.generate_contact_insights = AsyncMock()
	insights.calculate_lead_score = AsyncMock(return_value=75.0)
	insights.calculate_win_probability = AsyncMock(return_value=0.6)
	return insights


@pytest.fixture
def mock_analytics():
	"""Mock analytics service"""
	analytics = AsyncMock()
	analytics.track_event = AsyncMock()
	analytics.generate_report = AsyncMock()
	return analytics


@pytest.fixture
def mock_notification_service():
	"""Mock notification service"""
	service = AsyncMock()
	service.send_email = AsyncMock()
	service.send_sms = AsyncMock()
	return service


# Utility fixtures

@pytest.fixture
async def clean_database(db_connection):
	"""Clean all test data from database"""
	tables = [
		"crm_campaign_members",
		"crm_contact_relationships", 
		"crm_communications",
		"crm_ai_insights",
		"crm_activities",
		"crm_campaigns",
		"crm_opportunities",
		"crm_leads",
		"crm_accounts",
		"crm_contacts"
	]
	
	for table in tables:
		await db_connection.execute(f"DELETE FROM {table} WHERE tenant_id = $1", TEST_TENANT_ID)


@pytest.fixture
def sample_address():
	"""Sample address for testing"""
	return Address(
		street="123 Test Street",
		city="Test City",
		state="TS",
		postal_code="12345",
		country="Test Country"
	)


@pytest.fixture
def sample_phone():
	"""Sample phone number for testing"""
	return PhoneNumber(
		number="+1-555-0123",
		type="mobile",
		is_primary=True
	)


# Performance testing fixtures

@pytest.fixture
def performance_timer():
	"""Timer for performance testing"""
	import time
	
	class Timer:
		def __init__(self):
			self.start_time = None
			self.end_time = None
		
		def start(self):
			self.start_time = time.time()
		
		def stop(self):
			self.end_time = time.time()
		
		@property
		def elapsed(self):
			if self.start_time and self.end_time:
				return self.end_time - self.start_time
			return None
	
	return Timer()


# Async test helpers

def async_test(coro):
	"""Decorator for async test functions"""
	def wrapper(*args, **kwargs):
		loop = asyncio.get_event_loop()
		return loop.run_until_complete(coro(*args, **kwargs))
	return wrapper


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security