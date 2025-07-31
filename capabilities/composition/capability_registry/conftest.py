"""
APG Capability Registry - Pytest Configuration

Shared fixtures and configuration for comprehensive testing of the capability registry.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Configure pytest-asyncio
pytest_asyncio.fixture(scope="session")

# =============================================================================
# Test Configuration
# =============================================================================

# Test database URL (in-memory SQLite for fast testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"

# APG test configuration
TEST_APG_CONFIG = {
	"platform_version": "2.0",
	"tenant_id": "test_tenant",
	"environment": "testing",
	"composition_engine": {
		"enabled": True,
		"auto_discovery": True,
		"intelligent_routing": True
	},
	"discovery_service": {
		"enabled": True,
		"auto_registration": True,
		"health_monitoring": False  # Disabled for testing
	},
	"registry_integration": {
		"sync_interval": 60,  # Shorter for testing
		"auto_sync": False,  # Manual control in tests
		"conflict_resolution": "test_mode"
	}
}

# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.new_event_loop()
	asyncio.set_event_loop(loop)
	yield loop
	loop.close()

@pytest.fixture(scope="session")
def test_data_dir():
	"""Provide path to test data directory."""
	TEST_DATA_DIR.mkdir(exist_ok=True)
	return TEST_DATA_DIR

@pytest.fixture(scope="session")
def temp_dir():
	"""Create temporary directory for test files."""
	with tempfile.TemporaryDirectory() as temp_dir:
		yield Path(temp_dir)

# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture(scope="session")
async def test_db_engine():
	"""Create test database engine."""
	engine = create_async_engine(
		TEST_DATABASE_URL,
		echo=False,  # Set to True for SQL debugging
		future=True
	)
	
	# Create all tables
	from .models import Base
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	
	yield engine
	
	# Cleanup
	await engine.dispose()

@pytest.fixture
async def db_session(test_db_engine):
	"""Create database session for each test."""
	async_session_maker = sessionmaker(
		bind=test_db_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	
	async with async_session_maker() as session:
		try:
			yield session
		finally:
			await session.rollback()

# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture
async def registry_service(db_session):
	"""Create registry service with test database session."""
	from .service import CRService
	
	service = CRService(tenant_id="test_tenant")
	service.db_session = db_session
	return service

@pytest.fixture
async def composition_engine(registry_service):
	"""Create composition engine with registry service."""
	from .composition_engine import IntelligentCompositionEngine
	
	engine = IntelligentCompositionEngine(tenant_id="test_tenant")
	await engine.set_registry_service(registry_service)
	return engine

@pytest.fixture
async def version_manager():
	"""Create version manager service."""
	from .version_manager import VersionManagerService
	
	return VersionManagerService(tenant_id="test_tenant")

@pytest.fixture
async def marketplace_service():
	"""Create marketplace service."""
	from .marketplace import MarketplaceService
	
	return MarketplaceService(tenant_id="test_tenant")

@pytest.fixture
async def apg_integration_service(registry_service):
	"""Create APG integration service."""
	from .apg_integration import APGIntegrationService
	
	service = APGIntegrationService(tenant_id="test_tenant")
	await service.set_registry_service(registry_service)
	return service

@pytest.fixture
async def mobile_service(registry_service, temp_dir):
	"""Create mobile offline service."""
	from .mobile_service import MobileOfflineService
	
	offline_db_path = str(temp_dir / "test_mobile.db")
	service = MobileOfflineService(
		tenant_id="test_tenant",
		offline_db_path=offline_db_path
	)
	await service.set_online_service(registry_service)
	return service

# =============================================================================
# API Client Fixtures
# =============================================================================

@pytest.fixture
def api_client():
	"""Create FastAPI test client."""
	from .api import api_app
	
	return TestClient(api_app)

@pytest.fixture
def api_headers():
	"""Provide API headers for testing."""
	return {
		"Authorization": "Bearer test_token",
		"Content-Type": "application/json",
		"Accept": "application/json"
	}

# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_capability_data():
	"""Sample capability data for testing."""
	return {
		"capability_code": "TEST_CAPABILITY",
		"capability_name": "Test Capability",
		"description": "A comprehensive test capability for automated testing and validation",
		"long_description": "This is a detailed test capability designed for comprehensive testing of the APG capability registry system.",
		"category": "foundation_infrastructure",
		"subcategory": "testing",
		"version": "1.0.0",
		"target_users": ["developers", "testers", "qa_engineers"],
		"business_value": "Enables automated testing and quality assurance processes",
		"use_cases": ["unit_testing", "integration_testing", "automated_validation"],
		"industry_focus": ["software_development", "technology"],
		"composition_keywords": ["testing", "validation", "automation", "quality"],
		"provides_services": ["test_execution", "result_validation", "coverage_analysis"],
		"data_models": ["TestResult", "TestCase", "TestSuite"],
		"api_endpoints": ["/api/test/execute", "/api/test/results", "/api/test/coverage"],
		"file_path": "/capabilities/testing/test_capability.py",
		"module_path": "capabilities.testing.test_capability",
		"documentation_path": "/docs/testing/test_capability.md",
		"repository_url": "https://github.com/datacraft/apg-testing",
		"multi_tenant": True,
		"audit_enabled": True,
		"security_integration": True,
		"performance_optimized": False,
		"ai_enhanced": False,
		"complexity_score": 2.5,
		"metadata": {
			"test_framework": "pytest",
			"coverage_threshold": 0.8,
			"protocols": ["apg.testing", "apg.validation"]
		}
	}

@pytest.fixture
def sample_composition_data():
	"""Sample composition data for testing."""
	return {
		"name": "Test Composition Suite",
		"description": "Comprehensive test composition for quality assurance workflows",
		"composition_type": "custom",
		"capability_ids": [],  # Will be populated in tests
		"business_requirements": [
			"automated_testing",
			"quality_assurance",
			"continuous_integration"
		],
		"compliance_requirements": ["iso_27001", "sox_compliance"],
		"target_users": ["developers", "qa_engineers", "devops_engineers"],
		"deployment_strategy": "standard",
		"is_template": False,
		"is_public": False,
		"shared_with_tenants": [],
		"configuration": {
			"test_timeout": 300,
			"parallel_execution": True,
			"failure_tolerance": 0.1
		},
		"environment_settings": {
			"test_environment": "staging",
			"data_isolation": True,
			"cleanup_after_test": True
		}
	}

@pytest.fixture
def sample_version_data():
	"""Sample version data for testing."""
	return {
		"version_number": "1.2.3",
		"major_version": 1,
		"minor_version": 2,
		"patch_version": 3,
		"prerelease": None,
		"build_metadata": None,
		"release_notes": "Bug fixes and performance improvements",
		"breaking_changes": [],
		"deprecations": [],
		"new_features": [
			"Enhanced validation engine",
			"Improved error reporting",
			"Performance optimizations"
		],
		"compatible_versions": ["1.0.0", "1.1.0", "1.2.0"],
		"incompatible_versions": ["0.9.0"],
		"backward_compatible": True,
		"forward_compatible": False,
		"quality_score": 0.92,
		"test_coverage": 0.88,
		"documentation_score": 0.85,
		"security_audit_passed": True,
		"status": "active",
		"support_level": "full",
		"api_changes": {
			"added_endpoints": ["/api/v2/enhanced"],
			"modified_endpoints": [],
			"removed_endpoints": []
		},
		"migration_path": {
			"from_previous": "automatic",
			"to_next": "manual_review_required"
		},
		"upgrade_instructions": "Run migration scripts and update configuration files"
	}

@pytest.fixture
def sample_dependency_data():
	"""Sample dependency data for testing."""
	return {
		"dependent_id": "cap_001",
		"dependency_id": "cap_002",
		"dependency_type": "required",
		"version_constraint": ">=1.0.0,<2.0.0",
		"is_circular": False,
		"resolution_order": 1,
		"conflict_resolution": "latest_wins",
		"metadata": {
			"dependency_reason": "Authentication services",
			"alternatives": ["oauth_provider", "ldap_integration"]
		}
	}

@pytest.fixture
def multiple_capabilities_data(sample_capability_data):
	"""Generate multiple capability data sets for testing."""
	capabilities = []
	
	# Foundation infrastructure capabilities
	for i in range(3):
		cap_data = sample_capability_data.copy()
		cap_data.update({
			"capability_code": f"FOUNDATION_CAP_{i:03d}",
			"capability_name": f"Foundation Capability {i}",
			"description": f"Foundation infrastructure capability number {i}",
			"category": "foundation_infrastructure",
			"subcategory": "core_services"
		})
		capabilities.append(cap_data)
	
	# Business operations capabilities
	for i in range(3):
		cap_data = sample_capability_data.copy()
		cap_data.update({
			"capability_code": f"BUSINESS_CAP_{i:03d}",
			"capability_name": f"Business Capability {i}",
			"description": f"Business operations capability number {i}",
			"category": "business_operations",
			"subcategory": "workflow_management"
		})
		capabilities.append(cap_data)
	
	# Analytics capabilities
	for i in range(2):
		cap_data = sample_capability_data.copy()
		cap_data.update({
			"capability_code": f"ANALYTICS_CAP_{i:03d}",
			"capability_name": f"Analytics Capability {i}",
			"description": f"Analytics and intelligence capability number {i}",
			"category": "analytics_intelligence",
			"subcategory": "data_analysis",
			"ai_enhanced": True
		})
		capabilities.append(cap_data)
	
	return capabilities

# =============================================================================
# Mock Configuration Fixtures
# =============================================================================

@pytest.fixture
def mock_apg_config():
	"""Mock APG configuration for testing."""
	return TEST_APG_CONFIG.copy()

@pytest.fixture
def mock_external_services():
	"""Mock external service responses."""
	return {
		"composition_engine": {
			"status": "healthy",
			"version": "2.0.0",
			"capabilities_registered": 50
		},
		"discovery_service": {
			"status": "healthy",
			"version": "2.0.0",
			"services_registered": 45
		},
		"marketplace": {
			"status": "healthy",
			"version": "1.5.0",
			"published_capabilities": 25
		}
	}

# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_test_data():
	"""Generate data for performance testing."""
	return {
		"bulk_capability_count": 100,
		"bulk_composition_count": 50,
		"search_test_queries": [
			"foundation",
			"business operations",
			"analytics",
			"test capability",
			"workflow management"
		],
		"performance_thresholds": {
			"capability_registration_ms": 1000,
			"composition_creation_ms": 2000,
			"search_response_ms": 500,
			"bulk_operation_s": 30
		}
	}

# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test(db_session):
	"""Cleanup database after each test."""
	yield
	
	# Rollback any changes
	await db_session.rollback()

@pytest.fixture
def cleanup_files():
	"""Track files created during tests for cleanup."""
	created_files = []
	
	def track_file(filepath):
		created_files.append(filepath)
		return filepath
	
	yield track_file
	
	# Cleanup created files
	for filepath in created_files:
		try:
			if os.path.exists(filepath):
				os.remove(filepath)
		except Exception:
			pass  # Ignore cleanup errors

# =============================================================================
# Utility Functions
# =============================================================================

def create_test_capability(override_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""Create test capability data with optional overrides."""
	base_data = {
		"capability_code": "UTIL_TEST_CAP",
		"capability_name": "Utility Test Capability",
		"description": "Utility-generated test capability",
		"category": "foundation_infrastructure",
		"version": "1.0.0",
		"multi_tenant": True,
		"audit_enabled": True,
		"security_integration": True
	}
	
	if override_data:
		base_data.update(override_data)
	
	return base_data

def create_test_composition(capability_ids: List[str], override_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
	"""Create test composition data with capability IDs."""
	base_data = {
		"name": "Utility Test Composition",
		"description": "Utility-generated test composition",
		"composition_type": "custom",
		"capability_ids": capability_ids,
		"is_template": False,
		"is_public": False
	}
	
	if override_data:
		base_data.update(override_data)
	
	return base_data

# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
	"""Configure pytest with custom markers and settings."""
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)
	config.addinivalue_line(
		"markers", "slow: mark test as slow running"
	)
	config.addinivalue_line(
		"markers", "apg_integration: mark test as APG integration test"
	)

def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers based on test names."""
	for item in items:
		# Mark integration tests
		if "integration" in item.name.lower():
			item.add_marker(pytest.mark.integration)
		
		# Mark performance tests
		if "performance" in item.name.lower() or "bulk" in item.name.lower():
			item.add_marker(pytest.mark.performance)
		
		# Mark APG integration tests
		if "apg" in item.name.lower():
			item.add_marker(pytest.mark.apg_integration)
		
		# Mark slow tests
		if "bulk" in item.name.lower() or "performance" in item.name.lower():
			item.add_marker(pytest.mark.slow)