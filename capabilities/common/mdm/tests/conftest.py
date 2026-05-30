#!/usr/bin/env python3
"""
APG Master Data Management (MDM) - Test Configuration
pytest fixtures and test setup following APG standards

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import asyncio
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from uuid_extensions import uuid7str

# Test database setup
os.environ["TESTING"] = "1"
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/mdm_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"

try:
	from ..database import MDMDatabaseManager
	from ..service import MDMService
	from ..integrations import APGIntegrationManager
	from ..models import MdEntity, MdEntityVersion, MdGoldenRecord, EntityType, EntityStatus
	_RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
	_RUNTIME_IMPORT_ERROR = exc
	MDMDatabaseManager = MDMService = APGIntegrationManager = object
	MdEntity = MdEntityVersion = MdGoldenRecord = object
	EntityType = EntityStatus = object


@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
async def test_db_manager() -> AsyncGenerator[MDMDatabaseManager, None]:
	"""Create test database manager with clean database"""
	if _RUNTIME_IMPORT_ERROR is not None:
		pytest.skip(f"optional MDM runtime dependency unavailable: {_RUNTIME_IMPORT_ERROR.name}")
	db_manager = MDMDatabaseManager({
		"database_url": "postgresql://test:test@localhost:5432/mdm_test",
		"pool_size": 5,
		"max_overflow": 10
	})
	
	await db_manager.initialize()
	
	# Clean up any existing test data
	async with db_manager.get_session() as session:
		# Delete test data in reverse dependency order
		await session.execute("DELETE FROM md_cross_references WHERE tenant_id LIKE 'test-%'")
		await session.execute("DELETE FROM md_data_quality_assessments WHERE tenant_id LIKE 'test-%'")
		await session.execute("DELETE FROM md_golden_records WHERE tenant_id LIKE 'test-%'")
		await session.execute("DELETE FROM md_entity_versions WHERE tenant_id LIKE 'test-%'")
		await session.execute("DELETE FROM md_entities WHERE tenant_id LIKE 'test-%'")
		await session.commit()
	
	yield db_manager
	
	# Cleanup
	await db_manager.close()


@pytest.fixture
def test_tenant_id() -> str:
	"""Generate unique test tenant ID"""
	return f"test-{uuid7str()[:8]}"


@pytest.fixture
def test_user_id() -> str:
	"""Generate unique test user ID"""
	return f"test-user-{uuid7str()[:8]}"


@pytest.fixture
def mock_integration_manager() -> APGIntegrationManager:
	"""Create mock APG integration manager for testing"""
	if _RUNTIME_IMPORT_ERROR is not None:
		pytest.skip(f"optional MDM runtime dependency unavailable: {_RUNTIME_IMPORT_ERROR.name}")
	manager = MagicMock(spec=APGIntegrationManager)
	manager.is_initialized = True
	
	# Mock async methods
	manager.initialize = AsyncMock(return_value={
		'status': 'success',
		'components': {
			'event_publisher': True,
			'cache_manager': True,
			'audit_logger': True,
			'config_manager': True
		}
	})
	
	manager.shutdown = AsyncMock()
	manager.publish_entity_event = AsyncMock(return_value=True)
	manager.get_cached_entity = AsyncMock(return_value=None)
	manager.cache_entity = AsyncMock(return_value=True)
	manager.invalidate_entity_cache = AsyncMock(return_value=True)
	manager.get_config_value = AsyncMock(return_value=None)
	manager.update_config_value = AsyncMock(return_value=True)
	
	return manager


@pytest.fixture
async def test_mdm_service(test_db_manager: MDMDatabaseManager, 
                          mock_integration_manager: APGIntegrationManager) -> AsyncGenerator[MDMService, None]:
	"""Create test MDM service with mocked integrations"""
	if _RUNTIME_IMPORT_ERROR is not None:
		pytest.skip(f"optional MDM runtime dependency unavailable: {_RUNTIME_IMPORT_ERROR.name}")
	service = MDMService(
		db_manager=test_db_manager,
		integration_manager=mock_integration_manager,
		config={
			'enable_ai': False,  # Disable AI for unit tests
			'enable_caching': False,
			'enable_events': False
		}
	)
	
	await service.initialize()
	yield service
	await service.shutdown()


@pytest.fixture
def sample_entity_data(test_tenant_id: str) -> Dict[str, Any]:
	"""Sample entity data for testing"""
	return {
		"tenant_id": test_tenant_id,
		"entity_type": EntityType.PERSON,
		"entity_name": "John Doe",
		"entity_description": "Test person entity",
		"business_key": "PERSON-001",
		"source_system": "test_system",
		"status": EntityStatus.ACTIVE,
		"attributes": {
			"first_name": "John",
			"last_name": "Doe",
			"email": "john.doe@test.com",
			"phone": "+1-555-123-4567",
			"date_of_birth": "1985-06-15"
		},
		"tags": ["test", "person", "sample"],
		"data_classification": "internal"
	}


@pytest.fixture
def sample_customer_data(test_tenant_id: str) -> Dict[str, Any]:
	"""Sample customer entity data for testing"""
	return {
		"tenant_id": test_tenant_id,
		"entity_type": EntityType.CUSTOMER,
		"entity_name": "Acme Corporation",
		"entity_description": "Test customer entity",
		"business_key": "CUST-001",
		"source_system": "crm_system",
		"status": EntityStatus.ACTIVE,
		"attributes": {
			"company_name": "Acme Corporation",
			"industry": "Technology",
			"revenue": 5000000,
			"employees": 150,
			"website": "https://acme.com",
			"primary_contact": "jane.smith@acme.com"
		},
		"tags": ["customer", "enterprise", "technology"],
		"data_classification": "confidential"
	}


@pytest.fixture
def sample_product_data(test_tenant_id: str) -> Dict[str, Any]:
	"""Sample product entity data for testing"""
	return {
		"tenant_id": test_tenant_id,
		"entity_type": EntityType.PRODUCT,
		"entity_name": "Widget Pro 3000",
		"entity_description": "Professional widget for enterprise use",
		"business_key": "PROD-001",
		"source_system": "product_catalog",
		"status": EntityStatus.ACTIVE,
		"attributes": {
			"sku": "WP3000-PRO",
			"category": "Widgets",
			"price": 299.99,
			"weight": 2.5,
			"dimensions": "10x8x6 inches",
			"color": "Silver",
			"warranty_months": 24
		},
		"tags": ["product", "widget", "professional"],
		"data_classification": "public"
	}


@pytest.fixture
async def created_test_entity(test_mdm_service: MDMService, 
                             sample_entity_data: Dict[str, Any],
                             test_tenant_id: str,
                             test_user_id: str) -> Dict[str, Any]:
	"""Create a test entity in the database"""
	from ..models import MdEntityCreate
	
	entity_create = MdEntityCreate(**sample_entity_data)
	context = test_mdm_service.create_operation_context(
		tenant_id=test_tenant_id,
		user_id=test_user_id,
		operation_type="create_entity",
		source_system="pytest"
	)
	
	result = await test_mdm_service.entity_service.create_entity(entity_create, context)
	assert result["status"] == "success"
	
	# Get the full entity data
	entity_result = await test_mdm_service.entity_service.get_entity(
		result["entity_id"], test_tenant_id
	)
	assert entity_result["status"] == "success"
	
	return entity_result["entity"]


@pytest.fixture
async def multiple_test_entities(test_mdm_service: MDMService,
                               test_tenant_id: str,
                               test_user_id: str) -> list[Dict[str, Any]]:
	"""Create multiple test entities for batch operations"""
	from ..models import MdEntityCreate
	
	entities_data = [
		{
			"tenant_id": test_tenant_id,
			"entity_type": EntityType.PERSON,
			"entity_name": f"Person {i}",
			"business_key": f"PERSON-{i:03d}",
			"source_system": "test_system",
			"status": EntityStatus.ACTIVE,
			"attributes": {"index": i, "category": "person"},
			"tags": ["test", "batch"],
			"data_classification": "internal"
		}
		for i in range(1, 6)
	]
	
	created_entities = []
	context = test_mdm_service.create_operation_context(
		tenant_id=test_tenant_id,
		user_id=test_user_id,
		operation_type="batch_create",
		source_system="pytest"
	)
	
	for entity_data in entities_data:
		entity_create = MdEntityCreate(**entity_data)
		result = await test_mdm_service.entity_service.create_entity(entity_create, context)
		assert result["status"] == "success"
		
		entity_result = await test_mdm_service.entity_service.get_entity(
			result["entity_id"], test_tenant_id
		)
		assert entity_result["status"] == "success"
		created_entities.append(entity_result["entity"])
	
	return created_entities


@pytest.fixture
def quality_assessment_data() -> Dict[str, Any]:
	"""Sample quality assessment data"""
	return {
		"overall_score": 85.0,
		"quality_status": "good",
		"completeness_score": 90.0,
		"accuracy_score": 85.0,
		"consistency_score": 80.0,
		"validity_score": 88.0,
		"uniqueness_score": 95.0,
		"timeliness_score": 75.0,
		"assessment_duration_ms": 150.0,
		"quality_issues": [
			{
				"issue_type": "timeliness",
				"field": "last_updated",
				"severity": "medium",
				"message": "Data is 6 months old",
				"recommendation": "Update from source system"
			}
		],
		"recommendations": [
			"Update contact information",
			"Verify email address",
			"Standardize phone format"
		]
	}


@pytest.fixture
def duplicate_candidates_data() -> list[Dict[str, Any]]:
	"""Sample duplicate candidate data"""
	return [
		{
			"candidate_id": uuid7str(),
			"candidate_name": "John D. Doe",
			"candidate_business_key": "PERSON-002",
			"candidate_source_system": "hr_system",
			"match_score": 92.5,
			"confidence": "high",
			"matching_attributes": ["first_name", "last_name", "email"],
			"similarity_details": {
				"name_similarity": 95.0,
				"email_similarity": 100.0,
				"phone_similarity": 0.0
			},
			"recommended_action": "merge",
			"match_explanation": "High similarity in name and email"
		},
		{
			"candidate_id": uuid7str(),
			"candidate_name": "Johnny Doe",
			"candidate_business_key": "PERSON-003",
			"candidate_source_system": "customer_db",
			"match_score": 75.0,
			"confidence": "medium",
			"matching_attributes": ["last_name", "phone"],
			"similarity_details": {
				"name_similarity": 60.0,
				"email_similarity": 0.0,
				"phone_similarity": 100.0
			},
			"recommended_action": "review",
			"match_explanation": "Same phone number, similar name"
		}
	]


@pytest.fixture
def mock_ai_engines():
	"""Mock AI engines for testing without Ollama dependency"""
	from unittest.mock import AsyncMock, MagicMock
	
	# Mock EntityMatchingEngine
	matching_engine = MagicMock()
	matching_engine.find_duplicates = AsyncMock(return_value={
		"status": "success",
		"total_candidates": 2,
		"high_confidence_matches": 1,
		"medium_confidence_matches": 1,
		"low_confidence_matches": 0,
		"match_candidates": []
	})
	matching_engine.calculate_match_score = AsyncMock(return_value=85.0)
	
	# Mock QualityEngine
	quality_engine = MagicMock()
	quality_engine.assess_quality = AsyncMock(return_value={
		"status": "success",
		"overall_score": 85.0,
		"quality_status": "good",
		"completeness_score": 90.0,
		"accuracy_score": 85.0,
		"consistency_score": 80.0,
		"validity_score": 88.0,
		"uniqueness_score": 95.0,
		"timeliness_score": 75.0,
		"quality_issues": [],
		"recommendations": []
	})
	
	# Mock AnomalyEngine
	anomaly_engine = MagicMock()
	anomaly_engine.detect_anomalies = AsyncMock(return_value={
		"status": "success",
		"anomaly_score": 15.0,
		"is_anomalous": False,
		"anomaly_indicators": []
	})
	
	return {
		"matching_engine": matching_engine,
		"quality_engine": quality_engine,
		"anomaly_engine": anomaly_engine
	}


@pytest.fixture
def performance_benchmarks():
	"""Performance benchmarks for testing"""
	return {
		"entity_creation_max_ms": 100,
		"entity_retrieval_max_ms": 50,
		"quality_assessment_max_ms": 200,
		"duplicate_detection_max_ms": 500,
		"batch_operation_max_per_second": 100
	}


# Test utilities

def assert_entity_matches_data(entity: Dict[str, Any], expected_data: Dict[str, Any]):
	"""Assert that entity matches expected data"""
	assert entity["entity_type"] == expected_data["entity_type"]
	assert entity["entity_name"] == expected_data["entity_name"]
	assert entity["business_key"] == expected_data["business_key"]
	assert entity["source_system"] == expected_data["source_system"]
	assert entity["status"] == expected_data["status"]
	assert entity["data_classification"] == expected_data["data_classification"]
	
	# Check attributes subset match
	for key, value in expected_data["attributes"].items():
		assert entity["attributes"].get(key) == value
	
	# Check tags
	for tag in expected_data["tags"]:
		assert tag in entity["tags"]


def assert_valid_uuid7(uuid_str: str):
	"""Assert that string is valid UUID7"""
	import re
	pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
	assert re.match(pattern, uuid_str, re.IGNORECASE), f"Invalid UUID7: {uuid_str}"


def assert_recent_timestamp(timestamp: datetime, max_age_seconds: int = 5):
	"""Assert that timestamp is recent"""
	now = datetime.utcnow()
	age = (now - timestamp).total_seconds()
	assert age <= max_age_seconds, f"Timestamp too old: {age} seconds"


def assert_performance_within_limits(duration_ms: float, max_ms: float, operation: str):
	"""Assert that operation completed within performance limits"""
	assert duration_ms <= max_ms, f"{operation} took {duration_ms}ms, expected <= {max_ms}ms"
