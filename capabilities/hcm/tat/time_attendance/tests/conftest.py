"""
Time & Attendance Capability Test Configuration

Pytest configuration and shared fixtures for comprehensive testing of the
revolutionary APG Time & Attendance capability with multi-tenant support.

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
"""

import asyncio
import pytest
import pytest_asyncio
from decimal import Decimal
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, AsyncGenerator
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Import application modules
from database import Base, DatabaseManager
from service import TimeAttendanceService
from config import TimeAttendanceConfig, Environment, TrackingMode
from models import (
	TAEmployee, TATimeEntry, TARemoteWorker, TAAIAgent, TAHybridCollaboration,
	TimeEntryStatus, TimeEntryType, WorkforceType, WorkMode, AIAgentType,
	ProductivityMetric, RemoteWorkStatus
)
from api import create_app


# Test Configuration
@pytest.fixture(scope="session")
def test_config():
	"""Test configuration with in-memory database"""
	config = TimeAttendanceConfig()
	config.environment = Environment.TESTING
	config.database.database_name = "test_time_attendance"
	config.database.echo_sql = False
	config.notifications.enabled = False
	
	# Enable all features for testing
	config.features.update({
		"ai_fraud_detection": True,
		"biometric_authentication": True,
		"remote_work_tracking": True,
		"ai_agent_management": True,
		"hybrid_collaboration": True,
		"iot_integration": True,
		"predictive_analytics": True
	})
	
	return config


# Database Fixtures
@pytest_asyncio.fixture(scope="session")
async def test_engine():
	"""Create test database engine with in-memory SQLite"""
	engine = create_async_engine(
		"sqlite+aiosqlite:///:memory:",
		poolclass=StaticPool,
		connect_args={"check_same_thread": False},
		echo=False
	)
	
	# Create all tables
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	yield engine
	
	await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
	"""Create database session for each test"""
	async_session = async_sessionmaker(
		bind=test_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	
	async with async_session() as session:
		yield session
		await session.rollback()


@pytest_asyncio.fixture
async def db_manager(test_config, test_engine):
	"""Database manager for testing"""
	manager = DatabaseManager(test_config)
	manager.engines['async'] = test_engine
	manager.sessions['async'] = async_sessionmaker(
		bind=test_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	
	yield manager


# Service Fixtures
@pytest_asyncio.fixture
async def service(test_config, db_manager):
	"""Time Attendance service for testing"""
	service = TimeAttendanceService(test_config)
	
	# Mock external integrations
	service._edm_client = AsyncMock()
	service._cv_client = AsyncMock()
	service._notification_client = AsyncMock()
	service._workflow_client = AsyncMock()
	service._fraud_detector = AsyncMock()
	service._predictor = AsyncMock()
	service._optimizer = AsyncMock()
	
	return service


# API Fixtures
@pytest.fixture
def test_client():
	"""FastAPI test client"""
	app = create_app()
	return TestClient(app)


@pytest_asyncio.fixture
async def authenticated_headers():
	"""Mock authentication headers"""
	return {
		"Authorization": "Bearer test_token_123",
		"Content-Type": "application/json"
	}


# Test Data Fixtures
@pytest.fixture
def tenant_data():
	"""Test tenant data"""
	return {
		"id": "test-tenant-001",
		"name": "Test Tenant Corporation",
		"schema_name": "ta_test_tenant_001"
	}


@pytest.fixture
def employee_data():
	"""Test employee data"""
	return {
		"id": str(uuid4()),
		"employee_number": "EMP001",
		"first_name": "John",
		"last_name": "Doe",
		"email": "john.doe@testcorp.com",
		"department_id": "dept-001",
		"hire_date": date.today() - timedelta(days=365),
		"employment_status": "active",
		"workforce_type": WorkforceType.HUMAN,
		"work_schedule": {
			"monday": {"start": "09:00", "end": "17:00"},
			"tuesday": {"start": "09:00", "end": "17:00"},
			"wednesday": {"start": "09:00", "end": "17:00"},
			"thursday": {"start": "09:00", "end": "17:00"},
			"friday": {"start": "09:00", "end": "17:00"}
		}
	}


@pytest.fixture
def time_entry_data(employee_data, tenant_data):
	"""Test time entry data"""
	return {
		"id": str(uuid4()),
		"tenant_id": tenant_data["id"],
		"employee_id": employee_data["id"],
		"entry_date": date.today(),
		"clock_in": datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
		"clock_out": datetime.now().replace(hour=17, minute=30, second=0, microsecond=0),
		"entry_type": TimeEntryType.REGULAR,
		"status": TimeEntryStatus.SUBMITTED,
		"device_info": {
			"device_id": "device-001",
			"device_type": "mobile",
			"os": "iOS 17.0",
			"app_version": "1.0.0"
		},
		"location": {
			"latitude": 40.7128,
			"longitude": -74.0060,
			"accuracy": 10.0
		}
	}


@pytest.fixture
def remote_worker_data(employee_data, tenant_data):
	"""Test remote worker data"""
	return {
		"id": str(uuid4()),
		"tenant_id": tenant_data["id"],
		"employee_id": employee_data["id"],
		"workspace_id": "ws-home-001",
		"work_mode": WorkMode.REMOTE_ONLY,
		"home_office_setup": {
			"location": "Home Office",
			"equipment": {
				"computer": "MacBook Pro M3",
				"monitor": "4K Display",
				"chair": "Ergonomic Chair",
				"desk": "Standing Desk"
			},
			"internet_speed": "100 Mbps",
			"lighting": "LED Desk Lamp"
		},
		"timezone": "America/New_York",
		"current_activity": RemoteWorkStatus.ACTIVE_WORKING,
		"productivity_metrics": []
	}


@pytest.fixture
def ai_agent_data(tenant_data):
	"""Test AI agent data"""
	return {
		"id": str(uuid4()),
		"tenant_id": tenant_data["id"],
		"agent_name": "Claude Assistant",
		"agent_type": AIAgentType.CONVERSATIONAL_AI,
		"agent_version": "3.5",
		"capabilities": [
			"natural_language_processing",
			"code_generation",
			"data_analysis",
			"customer_support"
		],
		"configuration": {
			"api_endpoints": {
				"chat": "/api/chat",
				"analyze": "/api/analyze"
			},
			"resource_limits": {
				"max_tokens": 100000,
				"rate_limit": "1000/hour"
			},
			"version": "3.5",
			"environment": "production",
			"cost_per_hour": 0.50
		},
		"deployment_environment": "production",
		"health_status": "healthy",
		"operational_cost_per_hour": Decimal("0.50")
	}


@pytest.fixture
def hybrid_collaboration_data(employee_data, ai_agent_data, tenant_data):
	"""Test hybrid collaboration data"""
	return {
		"id": str(uuid4()),
		"tenant_id": tenant_data["id"],
		"session_name": "Product Planning Session",
		"project_id": "proj-001",
		"session_type": "collaborative_work",
		"human_participants": [employee_data["id"]],
		"ai_participants": [ai_agent_data["id"]],
		"session_lead": employee_data["id"],
		"start_time": datetime.now(),
		"planned_duration_minutes": 90
	}


# Mock Data Generators
@pytest.fixture
def generate_employees():
	"""Generate multiple test employees"""
	def _generate(count: int = 5, tenant_id: str = "test-tenant-001") -> List[Dict[str, Any]]:
		employees = []
		for i in range(count):
			employees.append({
				"id": str(uuid4()),
				"tenant_id": tenant_id,
				"employee_number": f"EMP{i+1:03d}",
				"first_name": f"Employee{i+1}",
				"last_name": "Test",
				"email": f"employee{i+1}@testcorp.com",
				"department_id": f"dept-{(i % 3) + 1:03d}",
				"hire_date": date.today() - timedelta(days=365 - i*30),
				"employment_status": "active",
				"workforce_type": WorkforceType.HUMAN
			})
		return employees
	return _generate


@pytest.fixture
def generate_time_entries():
	"""Generate multiple test time entries"""
	def _generate(employee_id: str, days: int = 7, tenant_id: str = "test-tenant-001") -> List[Dict[str, Any]]:
		entries = []
		for i in range(days):
			entry_date = date.today() - timedelta(days=i)
			entries.append({
				"id": str(uuid4()),
				"tenant_id": tenant_id,
				"employee_id": employee_id,
				"entry_date": entry_date,
				"clock_in": datetime.combine(entry_date, datetime.min.time().replace(hour=9)),
				"clock_out": datetime.combine(entry_date, datetime.min.time().replace(hour=17, minute=30)),
				"entry_type": TimeEntryType.REGULAR,
				"status": TimeEntryStatus.APPROVED,
				"total_hours": Decimal("8.5"),
				"regular_hours": Decimal("8.0"),
				"overtime_hours": Decimal("0.5")
			})
		return entries
	return _generate


# Async Helper Fixtures
@pytest.fixture
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest_asyncio.fixture
async def async_mock():
	"""Create async mock helper"""
	return AsyncMock()


# Performance Testing Fixtures
@pytest.fixture
def performance_config():
	"""Configuration for performance testing"""
	return {
		"max_response_time": 500,  # milliseconds
		"concurrent_users": 10,
		"test_duration": 30,  # seconds
		"acceptable_error_rate": 0.01  # 1%
	}


# Integration Testing Fixtures
@pytest.fixture
def integration_endpoints():
	"""List of endpoints for integration testing"""
	return [
		"/api/human_capital_management/time_attendance/health",
		"/api/human_capital_management/time_attendance/config",
		"/api/human_capital_management/time_attendance/clock-in",
		"/api/human_capital_management/time_attendance/clock-out",
		"/api/human_capital_management/time_attendance/time-entries",
		"/api/human_capital_management/time_attendance/remote-work/start-session",
		"/api/human_capital_management/time_attendance/ai-agents/register",
		"/api/human_capital_management/time_attendance/collaboration/start-session"
	]


# Security Testing Fixtures
@pytest.fixture
def security_test_payloads():
	"""Test payloads for security testing"""
	return {
		"sql_injection": [
			"'; DROP TABLE ta_time_entries; --",
			"1' OR '1'='1",
			"admin'/*",
			"'; INSERT INTO ta_employees VALUES ('hacker'); --"
		],
		"xss": [
			"<script>alert('XSS')</script>",
			"javascript:alert('XSS')",
			"<img src=x onerror=alert('XSS')>"
		],
		"large_payload": "A" * 10000,
		"null_bytes": "\x00\x01\x02",
		"unicode_attacks": "￼￼￼￼￼"
	}


# Cleanup Fixtures
@pytest_asyncio.fixture(autouse=True)
async def cleanup_test_data(db_session):
	"""Cleanup test data after each test"""
	yield
	# Cleanup will happen automatically due to session rollback


# pytest configuration
def pytest_configure(config):
	"""Configure pytest settings"""
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)
	config.addinivalue_line(
		"markers", "security: mark test as security test"
	)
	config.addinivalue_line(
		"markers", "slow: mark test as slow running test"
	)


def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers automatically"""
	for item in items:
		# Add integration marker to integration tests
		if "integration" in str(item.fspath):
			item.add_marker(pytest.mark.integration)
		
		# Add performance marker to performance tests
		if "performance" in item.name.lower():
			item.add_marker(pytest.mark.performance)
		
		# Add security marker to security tests
		if "security" in item.name.lower():
			item.add_marker(pytest.mark.security)


# Export fixtures for use in tests
__all__ = [
	"test_config", "test_engine", "db_session", "db_manager", "service",
	"test_client", "authenticated_headers", "tenant_data", "employee_data",
	"time_entry_data", "remote_worker_data", "ai_agent_data",
	"hybrid_collaboration_data", "generate_employees", "generate_time_entries",
	"async_mock", "performance_config", "integration_endpoints",
	"security_test_payloads"
]