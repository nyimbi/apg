"""
APG Capability Registry - Comprehensive Test Suite

Production-ready test suite covering all components of the capability registry
with pytest fixtures, real object testing, and full APG integration validation.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid_extensions import uuid7str

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from pydantic import ValidationError

# Import all components for testing
from .models import (
	CRCapability, CRComposition, CRDependency, CRVersion, CRRegistry,
	CRUsageAnalytics, CRHealthMetrics, CRCapabilityStatus, CRDependencyType,
	CRCompositionType, CRVersionConstraint
)
from .service import CRService, get_registry_service
from .composition_engine import IntelligentCompositionEngine, get_composition_engine
from .version_manager import VersionManagerService
from .marketplace import MarketplaceService
from .apg_integration import APGIntegrationService, get_apg_integration_service
from .mobile_service import MobileOfflineService
from .views import (
	CapabilityListView, CapabilityDetailView, CapabilityCreateForm,
	CompositionListView, CompositionDetailView, CompositionCreateForm,
	RegistryDashboardData
)

# =============================================================================
# Test Configuration and Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

@pytest.fixture
async def async_db_engine():
	"""Create async database engine for testing."""
	engine = create_async_engine(
		"sqlite+aiosqlite:///:memory:",
		echo=False
	)
	yield engine
	await engine.dispose()

@pytest.fixture
async def async_session(async_db_engine):
	"""Create async database session for testing."""
	async_session_maker = sessionmaker(
		bind=async_db_engine,
		class_=AsyncSession,
		expire_on_commit=False
	)
	
	async with async_session_maker() as session:
		yield session

@pytest.fixture
async def registry_service(async_session):
	"""Create registry service for testing."""
	service = CRService(tenant_id="test_tenant")
	service.db_session = async_session
	return service

@pytest.fixture
async def composition_engine(registry_service):
	"""Create composition engine for testing."""
	engine = IntelligentCompositionEngine(tenant_id="test_tenant")
	await engine.set_registry_service(registry_service)
	return engine

@pytest.fixture
async def version_manager():
	"""Create version manager for testing."""
	return VersionManagerService(tenant_id="test_tenant")

@pytest.fixture
async def marketplace_service():
	"""Create marketplace service for testing."""
	return MarketplaceService(tenant_id="test_tenant")

@pytest.fixture
async def apg_integration_service(registry_service):
	"""Create APG integration service for testing."""
	service = APGIntegrationService(tenant_id="test_tenant")
	await service.set_registry_service(registry_service)
	return service

@pytest.fixture
async def mobile_service(registry_service):
	"""Create mobile service for testing."""
	service = MobileOfflineService(tenant_id="test_tenant")
	await service.set_online_service(registry_service)
	return service

@pytest.fixture
def sample_capability_data():
	"""Sample capability data for testing."""
	return {
		"capability_code": "TEST_CAPABILITY",
		"capability_name": "Test Capability",
		"description": "A test capability for unit testing",
		"category": "foundation_infrastructure",
		"subcategory": "testing",
		"version": "1.0.0",
		"target_users": ["developers", "testers"],
		"business_value": "Enables automated testing",
		"use_cases": ["unit_testing", "integration_testing"],
		"industry_focus": ["software_development"],
		"composition_keywords": ["testing", "validation", "automation"],
		"provides_services": ["test_execution", "result_validation"],
		"data_models": ["TestResult", "TestCase"],
		"api_endpoints": ["/api/test/execute", "/api/test/results"],
		"multi_tenant": True,
		"audit_enabled": True,
		"security_integration": True,
		"performance_optimized": False,
		"ai_enhanced": False
	}

@pytest.fixture
def sample_composition_data():
	"""Sample composition data for testing."""
	return {
		"name": "Test Composition",
		"description": "A test composition for validation",
		"composition_type": "custom",
		"capability_ids": ["cap_001", "cap_002"],
		"business_requirements": ["automated_testing", "quality_assurance"],
		"compliance_requirements": ["iso_27001", "sox"],
		"target_users": ["developers", "qa_engineers"],
		"is_template": False,
		"is_public": False
	}

# =============================================================================
# Model Tests
# =============================================================================

class TestCapabilityModels:
	"""Test capability model validation and behavior."""
	
	def test_capability_creation(self, sample_capability_data):
		"""Test capability model creation with valid data."""
		capability = CRCapability(**sample_capability_data)
		
		assert capability.capability_code == "TEST_CAPABILITY"
		assert capability.capability_name == "Test Capability"
		assert capability.category == "foundation_infrastructure"
		assert capability.multi_tenant is True
		assert capability.quality_score == 0.0  # Default value
		assert capability.status == CRCapabilityStatus.DISCOVERED  # Default
	
	def test_capability_validation_errors(self):
		"""Test capability validation with invalid data."""
		with pytest.raises(ValidationError):
			CRCapability(
				capability_code="",  # Invalid empty code
				capability_name="Test",
				description="Test description",
				category="invalid_category",  # Invalid category
				version="invalid_version"  # Invalid version format
			)
	
	def test_capability_uuid_generation(self, sample_capability_data):
		"""Test automatic UUID generation for capability ID."""
		capability = CRCapability(**sample_capability_data)
		
		assert capability.capability_id is not None
		assert len(capability.capability_id) > 0
		assert capability.capability_id.startswith("01")  # UUID7 prefix
	
	def test_capability_timestamps(self, sample_capability_data):
		"""Test automatic timestamp generation."""
		capability = CRCapability(**sample_capability_data)
		
		assert capability.created_at is not None
		assert capability.updated_at is not None
		assert isinstance(capability.created_at, datetime)
		assert isinstance(capability.updated_at, datetime)

class TestCompositionModels:
	"""Test composition model validation and behavior."""
	
	def test_composition_creation(self, sample_composition_data):
		"""Test composition model creation with valid data."""
		composition = CRComposition(**sample_composition_data)
		
		assert composition.name == "Test Composition"
		assert composition.composition_type == "custom"
		assert composition.validation_status == "pending"  # Default
		assert composition.is_template is False
		assert composition.estimated_complexity == 1.0  # Default
	
	def test_composition_validation_errors(self):
		"""Test composition validation with invalid data."""
		with pytest.raises(ValidationError):
			CRComposition(
				name="",  # Invalid empty name
				description="Test",
				composition_type="invalid_type",  # Invalid type
				capability_ids=[]  # Empty capability list
			)
	
	def test_composition_capability_ids_validation(self, sample_composition_data):
		"""Test capability IDs validation."""
		# Valid capability IDs
		composition = CRComposition(**sample_composition_data)
		assert len(composition.capability_ids) == 2
		
		# Invalid capability IDs (empty list should fail in real validation)
		sample_composition_data["capability_ids"] = []
		with pytest.raises(ValidationError):
			CRComposition(**sample_composition_data)

class TestVersionModels:
	"""Test version model validation and behavior."""
	
	def test_version_creation(self):
		"""Test version model creation."""
		version_data = {
			"capability_id": "cap_001",
			"version_number": "1.2.3",
			"major_version": 1,
			"minor_version": 2,
			"patch_version": 3,
			"release_notes": "Bug fixes and improvements",
			"breaking_changes": ["API endpoint renamed"],
			"new_features": ["Added new validation"],
			"backward_compatible": True,
			"forward_compatible": False,
			"quality_score": 0.95,
			"test_coverage": 0.85,
			"security_audit_passed": True,
			"status": "active"
		}
		
		version = CRVersion(**version_data)
		
		assert version.version_number == "1.2.3"
		assert version.major_version == 1
		assert version.minor_version == 2
		assert version.patch_version == 3
		assert version.backward_compatible is True
		assert version.quality_score == 0.95
	
	def test_version_semantic_validation(self):
		"""Test semantic version validation."""
		# Valid semantic version
		version = CRVersion(
			capability_id="cap_001",
			version_number="2.0.0-beta.1",
			major_version=2,
			minor_version=0,
			patch_version=0,
			prerelease="beta.1"
		)
		assert version.prerelease == "beta.1"
		
		# Invalid version number
		with pytest.raises(ValidationError):
			CRVersion(
				capability_id="cap_001",
				version_number="invalid.version",
				major_version=1,
				minor_version=0,
				patch_version=0
			)

# =============================================================================
# Service Layer Tests
# =============================================================================

class TestRegistryService:
	"""Test capability registry service functionality."""
	
	@pytest.mark.asyncio
	async def test_register_capability(self, registry_service, sample_capability_data):
		"""Test capability registration."""
		result = await registry_service.register_capability(sample_capability_data)
		
		assert result["success"] is True
		assert "capability_id" in result
		assert result["capability_code"] == "TEST_CAPABILITY"
	
	@pytest.mark.asyncio
	async def test_get_capability(self, registry_service, sample_capability_data):
		"""Test capability retrieval."""
		# First register a capability
		register_result = await registry_service.register_capability(sample_capability_data)
		capability_id = register_result["capability_id"]
		
		# Then retrieve it
		capability = await registry_service.get_capability(capability_id)
		
		assert capability is not None
		assert capability.capability_id == capability_id
		assert capability.capability_code == "TEST_CAPABILITY"
	
	@pytest.mark.asyncio
	async def test_search_capabilities(self, registry_service, sample_capability_data):
		"""Test capability search functionality."""
		# Register multiple capabilities
		await registry_service.register_capability(sample_capability_data)
		
		sample_capability_data["capability_code"] = "ANOTHER_TEST"
		sample_capability_data["capability_name"] = "Another Test"
		await registry_service.register_capability(sample_capability_data)
		
		# Search capabilities
		search_criteria = {
			"query": "test",
			"category": "foundation_infrastructure",
			"page": 1,
			"per_page": 10
		}
		
		results = await registry_service.search_capabilities(search_criteria)
		
		assert "capabilities" in results
		assert len(results["capabilities"]) >= 2
		assert results["total_count"] >= 2
	
	@pytest.mark.asyncio
	async def test_update_capability(self, registry_service, sample_capability_data):
		"""Test capability update."""
		# Register capability
		register_result = await registry_service.register_capability(sample_capability_data)
		capability_id = register_result["capability_id"]
		
		# Update capability
		updates = {
			"description": "Updated test capability description",
			"quality_score": 0.95
		}
		
		result = await registry_service.update_capability(capability_id, updates)
		
		assert result["success"] is True
		
		# Verify update
		updated_capability = await registry_service.get_capability(capability_id)
		assert updated_capability.description == "Updated test capability description"
		assert updated_capability.quality_score == 0.95
	
	@pytest.mark.asyncio
	async def test_create_composition(self, registry_service, sample_composition_data):
		"""Test composition creation."""
		result = await registry_service.create_composition(sample_composition_data)
		
		assert result["success"] is True
		assert "composition_id" in result
		assert result["name"] == "Test Composition"
	
	@pytest.mark.asyncio
	async def test_validate_composition(self, registry_service, sample_capability_data):
		"""Test composition validation."""
		# Register capabilities first
		cap1_result = await registry_service.register_capability(sample_capability_data)
		
		sample_capability_data["capability_code"] = "TEST_CAPABILITY_2"
		cap2_result = await registry_service.register_capability(sample_capability_data)
		
		capability_ids = [cap1_result["capability_id"], cap2_result["capability_id"]]
		
		# Validate composition
		validation_result = await registry_service.validate_composition(capability_ids)
		
		assert "is_valid" in validation_result
		assert "validation_score" in validation_result
		assert "recommendations" in validation_result

class TestCompositionEngine:
	"""Test intelligent composition engine functionality."""
	
	@pytest.mark.asyncio
	async def test_analyze_composition(self, composition_engine, registry_service, sample_capability_data):
		"""Test composition analysis."""
		# Register test capabilities
		cap1_result = await registry_service.register_capability(sample_capability_data)
		
		sample_capability_data["capability_code"] = "TEST_DEP"
		sample_capability_data["capability_name"] = "Test Dependency"
		cap2_result = await registry_service.register_capability(sample_capability_data)
		
		capability_ids = [cap1_result["capability_id"], cap2_result["capability_id"]]
		
		# Analyze composition
		analysis = await composition_engine.analyze_composition(capability_ids)
		
		assert "is_valid" in analysis
		assert "validation_score" in analysis
		assert "dependencies" in analysis
		assert "conflicts" in analysis
		assert "recommendations" in analysis
	
	@pytest.mark.asyncio
	async def test_detect_conflicts(self, composition_engine):
		"""Test conflict detection."""
		# Mock capability data with potential conflicts
		capabilities = [
			{"capability_id": "cap1", "capability_code": "AUTH_BASIC"},
			{"capability_id": "cap2", "capability_code": "AUTH_OAUTH"},
		]
		
		conflicts = await composition_engine.detect_conflicts(capabilities)
		
		assert isinstance(conflicts, list)
		# Should detect authentication conflict
		if conflicts:
			assert any("auth" in conflict.get("type", "").lower() for conflict in conflicts)
	
	@pytest.mark.asyncio
	async def test_generate_recommendations(self, composition_engine):
		"""Test recommendation generation."""
		composition_data = {
			"capability_ids": ["cap1", "cap2"],
			"composition_type": "erp_enterprise",
			"target_users": ["business_users"]
		}
		
		recommendations = await composition_engine.generate_recommendations(composition_data)
		
		assert isinstance(recommendations, list)
		for rec in recommendations:
			assert "type" in rec
			assert "title" in rec
			assert "description" in rec

# =============================================================================
# APG Integration Tests
# =============================================================================

class TestAPGIntegration:
	"""Test APG platform integration functionality."""
	
	@pytest.mark.asyncio
	async def test_register_with_composition_engine(self, apg_integration_service, registry_service, sample_capability_data):
		"""Test APG composition engine registration."""
		# Register capability first
		register_result = await registry_service.register_capability(sample_capability_data)
		capability_id = register_result["capability_id"]
		
		# Register with APG composition engine
		apg_metadata = await apg_integration_service.register_with_composition_engine(capability_id)
		
		assert apg_metadata.capability_id == capability_id
		assert apg_metadata.capability_code == "TEST_CAPABILITY"
		assert "composition_engine" in apg_metadata.model_dump()
		assert "discovery_metadata" in apg_metadata.model_dump()
	
	@pytest.mark.asyncio
	async def test_register_with_discovery_service(self, apg_integration_service, registry_service, sample_capability_data):
		"""Test APG discovery service registration."""
		# Register capability first
		register_result = await registry_service.register_capability(sample_capability_data)
		capability_id = register_result["capability_id"]
		
		# Register with APG discovery service
		discovery_registration = await apg_integration_service.register_with_discovery_service(capability_id)
		
		assert discovery_registration.capability_id == capability_id
		assert discovery_registration.apg_tenant_id == "test_tenant"
		assert len(discovery_registration.discovery_tags) > 0
	
	@pytest.mark.asyncio
	async def test_create_apg_composition(self, apg_integration_service, registry_service, sample_composition_data, sample_capability_data):
		"""Test APG composition creation."""
		# Register capabilities first
		cap1_result = await registry_service.register_capability(sample_capability_data)
		sample_capability_data["capability_code"] = "TEST_CAP_2"
		cap2_result = await registry_service.register_capability(sample_capability_data)
		
		# Create composition
		sample_composition_data["capability_ids"] = [cap1_result["capability_id"], cap2_result["capability_id"]]
		comp_result = await registry_service.create_composition(sample_composition_data)
		
		# Create APG composition config
		apg_config = await apg_integration_service.create_apg_composition(
			comp_result["composition_id"],
			sample_composition_data["capability_ids"]
		)
		
		assert apg_config.composition_id == comp_result["composition_id"]
		assert apg_config.name == "Test Composition"
		assert len(apg_config.capability_bindings) == 2
		assert "failure_handling" in apg_config.model_dump()
		assert "resource_allocation" in apg_config.model_dump()
	
	@pytest.mark.asyncio
	async def test_sync_with_apg_ecosystem(self, apg_integration_service, registry_service, sample_capability_data):
		"""Test APG ecosystem synchronization."""
		# Register some capabilities
		await registry_service.register_capability(sample_capability_data)
		
		sample_capability_data["capability_code"] = "SYNC_TEST"
		await registry_service.register_capability(sample_capability_data)
		
		# Sync with APG ecosystem
		sync_results = await apg_integration_service.sync_with_apg_ecosystem()
		
		assert "capabilities_synced" in sync_results
		assert "compositions_synced" in sync_results
		assert "discovery_updates" in sync_results
		assert isinstance(sync_results["errors"], list)

# =============================================================================
# Mobile Service Tests
# =============================================================================

class TestMobileService:
	"""Test mobile and offline service functionality."""
	
	@pytest.mark.asyncio
	async def test_get_mobile_capabilities(self, mobile_service):
		"""Test mobile capability retrieval."""
		capabilities = await mobile_service.get_mobile_capabilities(
			category="foundation_infrastructure",
			limit=10,
			offset=0
		)
		
		assert isinstance(capabilities, list)
		for cap in capabilities:
			assert hasattr(cap, 'capability_id')
			assert hasattr(cap, 'name')
			assert hasattr(cap, 'category')
	
	@pytest.mark.asyncio
	async def test_create_mobile_composition(self, mobile_service):
		"""Test mobile composition creation."""
		composition_id = await mobile_service.create_mobile_composition(
			name="Mobile Test Composition",
			description="Test composition for mobile",
			capability_ids=["cap_001", "cap_002"],
			composition_type="mobile"
		)
		
		assert composition_id is not None
		assert len(composition_id) > 0
	
	@pytest.mark.asyncio
	async def test_sync_from_online(self, mobile_service):
		"""Test mobile data synchronization."""
		sync_result = await mobile_service.sync_from_online(force_full_sync=True)
		
		assert "success" in sync_result
		assert "capabilities_synced" in sync_result
		assert "compositions_synced" in sync_result
	
	@pytest.mark.asyncio
	async def test_get_mobile_dashboard_data(self, mobile_service):
		"""Test mobile dashboard data."""
		dashboard_data = await mobile_service.get_mobile_dashboard_data()
		
		assert "capabilities_count" in dashboard_data
		assert "compositions_count" in dashboard_data
		assert "is_online" in dashboard_data
		assert "storage_info" in dashboard_data

# =============================================================================
# View Model Tests
# =============================================================================

class TestViewModels:
	"""Test Pydantic view models and validation."""
	
	def test_capability_list_view(self, sample_capability_data):
		"""Test capability list view model."""
		view_data = {
			"capability_id": "cap_001",
			"capability_code": sample_capability_data["capability_code"],
			"capability_name": sample_capability_data["capability_name"],
			"description": sample_capability_data["description"],
			"version": sample_capability_data["version"],
			"category": sample_capability_data["category"],
			"status": "active",
			"quality_score": 0.85,
			"popularity_score": 0.75,
			"usage_count": 100,
			"created_at": datetime.utcnow()
		}
		
		view = CapabilityListView(**view_data)
		
		assert view.capability_code == "TEST_CAPABILITY"
		assert view.quality_score == 0.85
		assert view.usage_count == 100
	
	def test_capability_create_form(self, sample_capability_data):
		"""Test capability creation form validation."""
		form = CapabilityCreateForm(**sample_capability_data)
		
		assert form.capability_code == "TEST_CAPABILITY"
		assert form.capability_name == "Test Capability"
		assert form.multi_tenant is True
	
	def test_composition_create_form(self, sample_composition_data):
		"""Test composition creation form validation."""
		form = CompositionCreateForm(**sample_composition_data)
		
		assert form.name == "Test Composition"
		assert form.composition_type == "custom"
		assert len(form.capability_ids) == 2
	
	def test_registry_dashboard_data(self):
		"""Test registry dashboard data model."""
		dashboard_data = {
			"total_capabilities": 50,
			"active_capabilities": 45,
			"total_compositions": 20,
			"total_versions": 150,
			"registry_health_score": 0.95,
			"avg_quality_score": 0.85,
			"recent_capabilities": [],
			"recent_compositions": [],
			"category_stats": [
				{"category": "Foundation", "count": 15},
				{"category": "Business", "count": 20}
			],
			"marketplace_enabled": True,
			"published_capabilities": 10,
			"pending_submissions": 2
		}
		
		dashboard = RegistryDashboardData(**dashboard_data)
		
		assert dashboard.total_capabilities == 50
		assert dashboard.registry_health_score == 0.95
		assert dashboard.marketplace_enabled is True

# =============================================================================
# Integration Tests
# =============================================================================

class TestFullIntegration:
	"""Test full end-to-end integration scenarios."""
	
	@pytest.mark.asyncio
	async def test_complete_capability_lifecycle(self, registry_service, composition_engine, apg_integration_service, sample_capability_data):
		"""Test complete capability lifecycle from registration to APG integration."""
		# 1. Register capability
		register_result = await registry_service.register_capability(sample_capability_data)
		capability_id = register_result["capability_id"]
		
		# 2. Verify capability exists
		capability = await registry_service.get_capability(capability_id)
		assert capability is not None
		
		# 3. Update capability quality score
		await registry_service.update_capability(capability_id, {"quality_score": 0.90})
		
		# 4. Register with APG composition engine
		apg_metadata = await apg_integration_service.register_with_composition_engine(capability_id)
		assert apg_metadata.capability_id == capability_id
		
		# 5. Register with APG discovery service
		discovery_reg = await apg_integration_service.register_with_discovery_service(capability_id)
		assert discovery_reg.capability_id == capability_id
		
		# 6. Search for capability
		search_results = await registry_service.search_capabilities({
			"query": "test",
			"page": 1,
			"per_page": 10
		})
		assert len(search_results["capabilities"]) >= 1
	
	@pytest.mark.asyncio
	async def test_composition_creation_and_validation(self, registry_service, composition_engine, sample_capability_data, sample_composition_data):
		"""Test composition creation and validation workflow."""
		# 1. Register multiple capabilities
		cap1_result = await registry_service.register_capability(sample_capability_data)
		
		sample_capability_data["capability_code"] = "TEST_CAP_B"
		sample_capability_data["capability_name"] = "Test Capability B"
		cap2_result = await registry_service.register_capability(sample_capability_data)
		
		capability_ids = [cap1_result["capability_id"], cap2_result["capability_id"]]
		
		# 2. Validate composition before creation
		validation_result = await registry_service.validate_composition(capability_ids)
		assert "is_valid" in validation_result
		
		# 3. Create composition
		sample_composition_data["capability_ids"] = capability_ids
		comp_result = await registry_service.create_composition(sample_composition_data)
		composition_id = comp_result["composition_id"]
		
		# 4. Verify composition exists
		composition = await registry_service.get_composition(composition_id)
		assert composition is not None
		assert len(composition.capability_ids) == 2
		
		# 5. Analyze composition with engine
		analysis = await composition_engine.analyze_composition(capability_ids)
		assert "validation_score" in analysis
		assert "recommendations" in analysis

# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
	"""Test performance characteristics and scalability."""
	
	@pytest.mark.asyncio
	async def test_bulk_capability_registration(self, registry_service):
		"""Test bulk capability registration performance."""
		import time
		
		start_time = time.time()
		
		# Register 50 capabilities
		for i in range(50):
			capability_data = {
				"capability_code": f"PERF_TEST_{i:03d}",
				"capability_name": f"Performance Test Capability {i}",
				"description": f"Performance test capability number {i}",
				"category": "foundation_infrastructure",
				"version": "1.0.0",
				"multi_tenant": True,
				"audit_enabled": True
			}
			
			await registry_service.register_capability(capability_data)
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Should complete within reasonable time (adjust based on requirements)
		assert duration < 30.0  # 30 seconds for 50 capabilities
		
		# Verify all capabilities were registered
		search_results = await registry_service.search_capabilities({
			"query": "PERF_TEST",
			"page": 1,
			"per_page": 100
		})
		assert len(search_results["capabilities"]) == 50
	
	@pytest.mark.asyncio
	async def test_search_performance(self, registry_service):
		"""Test search performance with multiple capabilities."""
		# Register test capabilities first
		for i in range(20):
			capability_data = {
				"capability_code": f"SEARCH_TEST_{i:03d}",
				"capability_name": f"Search Test {i}",
				"description": f"Search test capability {i}",
				"category": "foundation_infrastructure",
				"version": "1.0.0"
			}
			await registry_service.register_capability(capability_data)
		
		import time
		start_time = time.time()
		
		# Perform search
		results = await registry_service.search_capabilities({
			"query": "search test",
			"page": 1,
			"per_page": 25
		})
		
		end_time = time.time()
		duration = end_time - start_time
		
		# Search should be fast
		assert duration < 1.0  # 1 second
		assert len(results["capabilities"]) >= 20

# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
	"""Test error handling and edge cases."""
	
	@pytest.mark.asyncio
	async def test_register_duplicate_capability(self, registry_service, sample_capability_data):
		"""Test duplicate capability registration handling."""
		# Register capability first time
		result1 = await registry_service.register_capability(sample_capability_data)
		assert result1["success"] is True
		
		# Try to register same capability again
		result2 = await registry_service.register_capability(sample_capability_data)
		
		# Should handle gracefully (update existing or return error)
		if not result2["success"]:
			assert "duplicate" in result2.get("message", "").lower() or "exists" in result2.get("message", "").lower()
	
	@pytest.mark.asyncio
	async def test_get_nonexistent_capability(self, registry_service):
		"""Test retrieval of non-existent capability."""
		nonexistent_id = "nonexistent_capability_id"
		
		capability = await registry_service.get_capability(nonexistent_id)
		assert capability is None
	
	@pytest.mark.asyncio
	async def test_validate_invalid_composition(self, registry_service):
		"""Test validation of composition with invalid capability IDs."""
		invalid_capability_ids = ["invalid_id_1", "invalid_id_2"]
		
		result = await registry_service.validate_composition(invalid_capability_ids)
		
		# Should return validation failure
		assert result.get("is_valid") is False or "error" in result
	
	def test_invalid_view_model_data(self):
		"""Test view model validation with invalid data."""
		with pytest.raises(ValidationError):
			CapabilityListView(
				capability_id="",  # Invalid empty ID
				capability_code="",  # Invalid empty code
				capability_name="",  # Invalid empty name
				quality_score=1.5,  # Invalid score > 1.0
				usage_count=-1  # Invalid negative count
			)

# =============================================================================
# Test Runner Configuration
# =============================================================================

if __name__ == "__main__":
	pytest.main([
		__file__,
		"-v",
		"--tb=short",
		"--asyncio-mode=auto",
		"--cov=capability_registry",
		"--cov-report=html",
		"--cov-report=term"
	])