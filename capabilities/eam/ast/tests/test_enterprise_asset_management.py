"""
Comprehensive tests for Enterprise Asset Management (EAM) capability

This test suite verifies the complete functionality of EAM capability including
data models, services, APIs, and APG platform integration following CLAUDE.md standards.
"""

import pytest
import asyncio
import json
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid_extensions import uuid7str
from unittest.mock import AsyncMock, MagicMock, patch

from capabilities.general_cross_functional.enterprise_asset_management import (
	get_capability_metadata, get_capability_id, get_required_dependencies,
	health_check, register_with_apg_composition_engine
)
from capabilities.general_cross_functional.enterprise_asset_management.models import (
	EAAsset, EALocation, EAWorkOrder, EAMaintenanceRecord, 
	EAInventory, EAContract, EAPerformanceRecord, EAAssetContract
)
from capabilities.general_cross_functional.enterprise_asset_management.service import (
	EAMAssetService, EAMWorkOrderService, EAMInventoryService, EAMAnalyticsService
)
from capabilities.general_cross_functional.enterprise_asset_management.views import (
	EAAssetCreateModel, EAAssetUpdateModel, EAWorkOrderCreateModel,
	EAInventoryCreateModel, EALocationCreateModel
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_tenant_id():
	"""Sample tenant ID for multi-tenant testing"""
	return uuid7str()

@pytest.fixture
def sample_location_data():
	"""Sample location data for testing"""
	return {
		"location_code": "SITE-001",
		"location_name": "Main Manufacturing Facility",
		"description": "Primary production facility with assembly lines",
		"location_type": "site",
		"address": "1234 Industrial Blvd",
		"city": "Manufacturing City",
		"state_province": "CA",
		"postal_code": "12345",
		"country_code": "US",
		"gps_latitude": Decimal("37.7749"),
		"gps_longitude": Decimal("-122.4194"),
		"floor_area_sqm": 5000.0,
		"max_capacity": 100,
		"cost_center": "CC-001"
	}

@pytest.fixture
def sample_asset_data(sample_tenant_id):
	"""Sample asset data for testing"""
	return {
		"tenant_id": sample_tenant_id,
		"asset_number": "ASSET-001",
		"asset_name": "CNC Machine #1",
		"description": "High-precision CNC machining center",
		"asset_type": "equipment",
		"asset_category": "production",
		"asset_class": "rotating",
		"criticality_level": "high",
		"manufacturer": "ACME Manufacturing",
		"model_number": "CNC-5000",
		"serial_number": "SN123456789",
		"year_manufactured": 2022,
		"purchase_cost": Decimal("250000.00"),
		"replacement_cost": Decimal("300000.00"),
		"installation_date": date(2023, 1, 15),
		"commissioning_date": date(2023, 2, 1),
		"maintenance_strategy": "predictive",
		"maintenance_frequency_days": 30,
		"has_digital_twin": True,
		"iot_enabled": True,
		"health_score": Decimal("95.5"),
		"condition_status": "excellent"
	}

@pytest.fixture
def sample_work_order_data(sample_tenant_id):
	"""Sample work order data for testing"""
	return {
		"tenant_id": sample_tenant_id,
		"title": "Preventive Maintenance - CNC Machine",
		"description": "Scheduled preventive maintenance including lubrication and calibration",
		"work_type": "maintenance",
		"priority": "medium",
		"work_category": "mechanical",
		"maintenance_type": "preventive",
		"safety_category": "low_risk",
		"estimated_hours": 4.0,
		"estimated_cost": Decimal("500.00"),
		"required_crew_size": 2,
		"scheduled_start": datetime.utcnow() + timedelta(days=1),
		"scheduled_end": datetime.utcnow() + timedelta(days=1, hours=4)
	}

@pytest.fixture
def sample_inventory_data(sample_tenant_id):
	"""Sample inventory data for testing"""
	return {
		"tenant_id": sample_tenant_id,
		"part_number": "PART-001",
		"description": "CNC Cutting Tool - Carbide Insert",
		"item_type": "spare_part",
		"category": "cutting_tools",
		"manufacturer": "Tool Corp",
		"manufacturer_part_number": "TC-123",
		"current_stock": 50,
		"minimum_stock": 10,
		"maximum_stock": 100,
		"reorder_point": 20,
		"unit_cost": Decimal("25.00"),
		"criticality": "high",
		"auto_reorder": True,
		"lead_time_days": 7
	}

@pytest.fixture
async def mock_auth_service():
	"""Mock auth service for testing"""
	mock = AsyncMock()
	mock.check_permission.return_value = True
	mock.get_current_user.return_value = {"user_id": "test_user", "tenant_id": uuid7str()}
	return mock

@pytest.fixture
async def mock_audit_service():
	"""Mock audit service for testing"""
	mock = AsyncMock()
	mock.log_action.return_value = True
	return mock

@pytest.fixture
async def eam_asset_service(mock_auth_service, mock_audit_service):
	"""EAM Asset Service for testing"""
	service = EAMAssetService()
	service.auth_service = mock_auth_service
	service.audit_service = mock_audit_service
	return service


# =============================================================================
# CAPABILITY METADATA TESTS
# =============================================================================

class TestEAMCapabilityMetadata:
	"""Test EAM capability metadata and APG integration"""
	
	def test_capability_metadata_structure(self):
		"""Test capability metadata has required structure"""
		metadata = get_capability_metadata()
		
		assert metadata["capability_id"] == "general_cross_functional.enterprise_asset_management"
		assert metadata["capability_name"] == "Enterprise Asset Management"
		assert metadata["version"] == "1.0.0"
		assert metadata["category"] == "general_cross_functional"
		
		# Test required fields
		required_fields = [
			"composition_type", "provides_services", "dependencies",
			"data_models", "api_endpoints", "ui_views", "permissions"
		]
		for field in required_fields:
			assert field in metadata
	
	def test_capability_dependencies(self):
		"""Test capability dependencies are correctly defined"""
		required_deps = get_required_dependencies()
		
		expected_required = [
			"auth_rbac", "audit_compliance", "fixed_asset_management",
			"predictive_maintenance", "digital_twin_marketplace",
			"document_management", "notification_engine"
		]
		
		for dep in expected_required:
			assert dep in required_deps
	
	def test_capability_id_consistency(self):
		"""Test capability ID consistency across functions"""
		capability_id = get_capability_id()
		metadata = get_capability_metadata()
		
		assert capability_id == metadata["capability_id"]
		assert capability_id == "general_cross_functional.enterprise_asset_management"
	
	@pytest.mark.asyncio
	async def test_health_check(self):
		"""Test capability health check functionality"""
		health_status = await health_check()
		
		assert health_status["capability_id"] == get_capability_id()
		assert health_status["version"] == "1.0.0"
		assert health_status["status"] == "healthy"
		assert "dependencies" in health_status
		assert "metrics" in health_status
		
		# Test dependency health status
		dependencies = health_status["dependencies"]
		for dep in get_required_dependencies():
			assert dep in dependencies
			assert dependencies[dep] == "healthy"


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestEAMDataModels:
	"""Test EAM data model functionality"""
	
	def test_asset_model_creation(self, sample_asset_data):
		"""Test asset model creation and validation"""
		asset = EAAsset(**sample_asset_data)
		
		assert asset.asset_number == "ASSET-001"
		assert asset.asset_name == "CNC Machine #1"
		assert asset.asset_type == "equipment"
		assert asset.criticality_level == "high"
		assert asset.purchase_cost == Decimal("250000.00")
		assert asset.has_digital_twin is True
		assert asset.health_score == Decimal("95.5")
	
	def test_location_model_creation(self, sample_location_data, sample_tenant_id):
		"""Test location model creation"""
		location_data = {**sample_location_data, "tenant_id": sample_tenant_id}
		location = EALocation(**location_data)
		
		assert location.location_code == "SITE-001"
		assert location.location_name == "Main Manufacturing Facility"
		assert location.location_type == "site"
		assert location.gps_latitude == Decimal("37.7749")
		assert location.floor_area_sqm == 5000.0
	
	def test_work_order_model_creation(self, sample_work_order_data):
		"""Test work order model creation"""
		work_order = EAWorkOrder(**sample_work_order_data)
		
		assert work_order.title == "Preventive Maintenance - CNC Machine"
		assert work_order.work_type == "maintenance"
		assert work_order.priority == "medium"
		assert work_order.estimated_hours == 4.0
		assert work_order.required_crew_size == 2
	
	def test_inventory_model_creation(self, sample_inventory_data):
		"""Test inventory model creation"""
		inventory = EAInventory(**sample_inventory_data)
		
		assert inventory.part_number == "PART-001"
		assert inventory.item_type == "spare_part"
		assert inventory.current_stock == 50
		assert inventory.reorder_point == 20
		assert inventory.auto_reorder is True
	
	def test_asset_hierarchy_relationships(self, sample_asset_data, sample_tenant_id):
		"""Test asset parent-child relationships"""
		# Create parent asset
		parent_data = {**sample_asset_data, "asset_number": "PARENT-001"}
		parent_asset = EAAsset(**parent_data)
		
		# Create child asset
		child_data = {
			**sample_asset_data,
			"asset_number": "CHILD-001",
			"parent_asset_id": parent_asset.asset_id
		}
		child_asset = EAAsset(**child_data)
		
		assert child_asset.parent_asset_id == parent_asset.asset_id
	
	def test_performance_record_model(self, sample_tenant_id):
		"""Test performance record model"""
		performance_data = {
			"tenant_id": sample_tenant_id,
			"asset_id": uuid7str(),
			"measurement_date": datetime.utcnow(),
			"measurement_period": "daily",
			"availability_percentage": Decimal("98.5"),
			"oee_overall": Decimal("85.0"),
			"health_score": Decimal("92.0"),
			"energy_consumption": Decimal("150.5"),
			"maintenance_cost": Decimal("2500.00"),
			"trend_direction": "stable"
		}
		
		performance = EAPerformanceRecord(**performance_data)
		
		assert performance.availability_percentage == Decimal("98.5")
		assert performance.oee_overall == Decimal("85.0")
		assert performance.trend_direction == "stable"


# =============================================================================
# PYDANTIC MODEL VALIDATION TESTS
# =============================================================================

class TestEAMPydanticModels:
	"""Test Pydantic model validation"""
	
	def test_asset_create_model_validation(self):
		"""Test asset creation model validation"""
		valid_data = {
			"asset_name": "Test Asset",
			"asset_type": "equipment",
			"asset_category": "production",
			"manufacturer": "Test Manufacturer",
			"purchase_cost": Decimal("10000.00"),
			"installation_date": date.today(),
			"commissioning_date": date.today() + timedelta(days=1)
		}
		
		model = EAAssetCreateModel(**valid_data)
		assert model.asset_name == "Test Asset"
		assert model.purchase_cost == Decimal("10000.00")
	
	def test_asset_create_model_date_validation(self):
		"""Test asset date validation logic"""
		invalid_data = {
			"asset_name": "Test Asset",
			"asset_type": "equipment",
			"asset_category": "production",
			"installation_date": date.today(),
			"commissioning_date": date.today() - timedelta(days=1)  # Before installation
		}
		
		with pytest.raises(ValueError, match="Commissioning date cannot be before installation date"):
			EAAssetCreateModel(**invalid_data)
	
	def test_work_order_create_model_validation(self):
		"""Test work order creation model validation"""
		valid_data = {
			"title": "Test Work Order",
			"description": "Test description",
			"work_type": "maintenance",
			"estimated_hours": 2.5,
			"estimated_cost": Decimal("500.00"),
			"scheduled_start": datetime.utcnow() + timedelta(hours=1),
			"scheduled_end": datetime.utcnow() + timedelta(hours=3)
		}
		
		model = EAWorkOrderCreateModel(**valid_data)
		assert model.title == "Test Work Order"
		assert model.estimated_hours == 2.5
	
	def test_inventory_create_model_stock_validation(self):
		"""Test inventory stock level validation"""
		invalid_data = {
			"part_number": "TEST-001",
			"description": "Test part",
			"item_type": "spare_part",
			"category": "test",
			"minimum_stock": 50,
			"maximum_stock": 25  # Invalid: minimum > maximum
		}
		
		with pytest.raises(ValueError, match="Minimum stock cannot exceed maximum stock"):
			EAInventoryCreateModel(**invalid_data)
	
	def test_location_create_model_coordinates(self):
		"""Test location GPS coordinate validation"""
		valid_data = {
			"location_code": "TEST-001",
			"location_name": "Test Location",
			"location_type": "site",
			"gps_latitude": Decimal("37.7749"),
			"gps_longitude": Decimal("-122.4194")
		}
		
		model = EALocationCreateModel(**valid_data)
		assert model.gps_latitude == Decimal("37.7749")
		assert model.gps_longitude == Decimal("-122.4194")


# =============================================================================
# SERVICE LAYER TESTS
# =============================================================================

class TestEAMAssetService:
	"""Test EAM Asset Service functionality"""
	
	@pytest.mark.asyncio
	async def test_create_asset(self, eam_asset_service, sample_asset_data):
		"""Test asset creation through service layer"""
		asset = await eam_asset_service.create_asset(sample_asset_data)
		
		assert asset.asset_name == sample_asset_data["asset_name"]
		assert asset.asset_type == sample_asset_data["asset_type"]
		assert asset.tenant_id == sample_asset_data["tenant_id"]
		
		# Verify audit logging was called
		eam_asset_service.audit_service.log_action.assert_called_once()
	
	@pytest.mark.asyncio
	async def test_asset_permission_checking(self, eam_asset_service, sample_asset_data):
		"""Test permission checking in asset operations"""
		# Mock permission denial
		eam_asset_service.auth_service.check_permission.return_value = False
		
		with pytest.raises(PermissionError):
			await eam_asset_service.create_asset(sample_asset_data)
	
	@pytest.mark.asyncio
	async def test_asset_search_with_filters(self, eam_asset_service, sample_tenant_id):
		"""Test asset search with various filters"""
		filters = {
			"asset_type": "equipment",
			"criticality_level": "high",
			"manufacturer": "ACME Manufacturing",
			"health_score_min": 90.0
		}
		
		# Mock database query result
		mock_results = [MagicMock(asset_name="Mock Asset")]
		with patch.object(eam_asset_service, '_execute_search_query', return_value=mock_results):
			results = await eam_asset_service.search_assets(sample_tenant_id, filters)
			
			assert len(results) == 1
			assert results[0].asset_name == "Mock Asset"
	
	@pytest.mark.asyncio
	async def test_asset_hierarchy_operations(self, eam_asset_service, sample_asset_data):
		"""Test asset hierarchy management"""
		# Create parent asset
		parent_asset = await eam_asset_service.create_asset(sample_asset_data)
		
		# Create child asset
		child_data = {
			**sample_asset_data,
			"asset_number": "CHILD-001",
			"parent_asset_id": parent_asset.asset_id
		}
		child_asset = await eam_asset_service.create_asset(child_data)
		
		# Test hierarchy retrieval
		hierarchy = await eam_asset_service.get_asset_hierarchy(parent_asset.asset_id)
		assert parent_asset.asset_id in hierarchy
	
	@pytest.mark.asyncio
	async def test_asset_health_score_update(self, eam_asset_service, sample_asset_data):
		"""Test asset health score updates"""
		asset = await eam_asset_service.create_asset(sample_asset_data)
		
		# Update health score
		new_health_data = {
			"health_score": Decimal("88.5"),
			"condition_status": "good",
			"change_reason": "Scheduled inspection results"
		}
		
		updated_asset = await eam_asset_service.update_asset_health(
			asset.asset_id, 
			new_health_data
		)
		
		assert updated_asset.health_score == Decimal("88.5")
		assert updated_asset.condition_status == "good"


class TestEAMWorkOrderService:
	"""Test EAM Work Order Service functionality"""
	
	@pytest.fixture
	async def work_order_service(self, mock_auth_service, mock_audit_service):
		"""Work Order Service for testing"""
		service = EAMWorkOrderService()
		service.auth_service = mock_auth_service
		service.audit_service = mock_audit_service
		return service
	
	@pytest.mark.asyncio
	async def test_create_work_order(self, work_order_service, sample_work_order_data):
		"""Test work order creation"""
		work_order = await work_order_service.create_work_order(sample_work_order_data)
		
		assert work_order.title == sample_work_order_data["title"]
		assert work_order.work_type == sample_work_order_data["work_type"]
		assert work_order.priority == sample_work_order_data["priority"]
		assert work_order.status == "draft"
	
	@pytest.mark.asyncio
	async def test_work_order_scheduling(self, work_order_service, sample_work_order_data):
		"""Test work order scheduling logic"""
		work_order = await work_order_service.create_work_order(sample_work_order_data)
		
		# Test scheduling
		schedule_data = {
			"assigned_to": "tech_001",
			"scheduled_start": datetime.utcnow() + timedelta(days=2),
			"scheduled_end": datetime.utcnow() + timedelta(days=2, hours=4)
		}
		
		scheduled_order = await work_order_service.schedule_work_order(
			work_order.work_order_id, 
			schedule_data
		)
		
		assert scheduled_order.assigned_to == "tech_001"
		assert scheduled_order.status == "scheduled"
	
	@pytest.mark.asyncio
	async def test_work_order_status_transitions(self, work_order_service, sample_work_order_data):
		"""Test work order status transitions"""
		work_order = await work_order_service.create_work_order(sample_work_order_data)
		
		# Test valid status transitions
		transitions = [
			("approved", "approved"),
			("assigned", "assigned"),
			("in_progress", "in_progress"),
			("completed", "completed")
		]
		
		for new_status, expected_status in transitions:
			updated_order = await work_order_service.update_work_order_status(
				work_order.work_order_id,
				new_status
			)
			assert updated_order.status == expected_status
	
	@pytest.mark.asyncio
	async def test_work_order_completion_with_results(self, work_order_service, sample_work_order_data):
		"""Test work order completion with results"""
		work_order = await work_order_service.create_work_order(sample_work_order_data)
		
		completion_data = {
			"work_performed": "Completed preventive maintenance as scheduled",
			"completion_notes": "All systems operating normally",
			"actual_hours": 3.5,
			"actual_cost": Decimal("450.00"),
			"quality_rating": 5,
			"parts_used": [{"part_id": "PART-001", "quantity": 2}]
		}
		
		completed_order = await work_order_service.complete_work_order(
			work_order.work_order_id,
			completion_data
		)
		
		assert completed_order.status == "completed"
		assert completed_order.actual_hours == 3.5
		assert completed_order.quality_rating == 5


class TestEAMInventoryService:
	"""Test EAM Inventory Service functionality"""
	
	@pytest.fixture
	async def inventory_service(self, mock_auth_service, mock_audit_service):
		"""Inventory Service for testing"""
		service = EAMInventoryService()
		service.auth_service = mock_auth_service
		service.audit_service = mock_audit_service
		return service
	
	@pytest.mark.asyncio
	async def test_create_inventory_item(self, inventory_service, sample_inventory_data):
		"""Test inventory item creation"""
		item = await inventory_service.create_inventory_item(sample_inventory_data)
		
		assert item.part_number == sample_inventory_data["part_number"]
		assert item.item_type == sample_inventory_data["item_type"]
		assert item.current_stock == sample_inventory_data["current_stock"]
	
	@pytest.mark.asyncio
	async def test_stock_movement_tracking(self, inventory_service, sample_inventory_data):
		"""Test stock movement tracking"""
		item = await inventory_service.create_inventory_item(sample_inventory_data)
		
		# Test stock adjustment
		movement_data = {
			"movement_type": "adjustment",
			"quantity": 10,
			"reason": "Physical count correction",
			"reference_id": "ADJ-001"
		}
		
		updated_item = await inventory_service.adjust_stock(
			item.inventory_id,
			movement_data
		)
		
		assert updated_item.current_stock == 60  # 50 + 10
	
	@pytest.mark.asyncio
	async def test_reorder_point_monitoring(self, inventory_service, sample_inventory_data):
		"""Test reorder point monitoring"""
		# Create item at reorder point
		reorder_data = {**sample_inventory_data, "current_stock": 20}
		item = await inventory_service.create_inventory_item(reorder_data)
		
		# Check reorder recommendations
		reorder_items = await inventory_service.get_reorder_recommendations(
			item.tenant_id
		)
		
		assert len(reorder_items) >= 1
		assert any(r.inventory_id == item.inventory_id for r in reorder_items)
	
	@pytest.mark.asyncio
	async def test_inventory_valuation(self, inventory_service, sample_inventory_data):
		"""Test inventory valuation calculations"""
		item = await inventory_service.create_inventory_item(sample_inventory_data)
		
		valuation = await inventory_service.calculate_inventory_valuation(
			item.tenant_id
		)
		
		assert "total_value" in valuation
		assert "item_count" in valuation
		assert valuation["total_value"] >= 0


class TestEAMAnalyticsService:
	"""Test EAM Analytics Service functionality"""
	
	@pytest.fixture
	async def analytics_service(self, mock_auth_service):
		"""Analytics Service for testing"""
		service = EAMAnalyticsService()
		service.auth_service = mock_auth_service
		return service
	
	@pytest.mark.asyncio
	async def test_asset_performance_analytics(self, analytics_service, sample_tenant_id):
		"""Test asset performance analytics"""
		analytics = await analytics_service.get_asset_performance_analytics(
			sample_tenant_id,
			time_period="last_30_days"
		)
		
		assert "average_health_score" in analytics
		assert "availability_percentage" in analytics
		assert "asset_count" in analytics
		assert "performance_trends" in analytics
	
	@pytest.mark.asyncio
	async def test_maintenance_effectiveness_analysis(self, analytics_service, sample_tenant_id):
		"""Test maintenance effectiveness analysis"""
		analysis = await analytics_service.analyze_maintenance_effectiveness(
			sample_tenant_id,
			time_period="last_quarter"
		)
		
		assert "preventive_ratio" in analysis
		assert "mean_time_between_failures" in analysis
		assert "maintenance_cost_per_asset" in analysis
		assert "effectiveness_score" in analysis
	
	@pytest.mark.asyncio
	async def test_cost_optimization_insights(self, analytics_service, sample_tenant_id):
		"""Test cost optimization insights"""
		insights = await analytics_service.get_cost_optimization_insights(
			sample_tenant_id
		)
		
		assert "high_cost_assets" in insights
		assert "maintenance_overspend" in insights
		assert "efficiency_opportunities" in insights
		assert "recommended_actions" in insights
	
	@pytest.mark.asyncio
	async def test_predictive_maintenance_insights(self, analytics_service, sample_tenant_id):
		"""Test predictive maintenance insights"""
		insights = await analytics_service.get_predictive_maintenance_insights(
			sample_tenant_id
		)
		
		assert "failure_predictions" in insights
		assert "maintenance_recommendations" in insights
		assert "risk_assessments" in insights
		assert "optimization_suggestions" in insights


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestEAMIntegration:
	"""Integration tests for complete EAM workflows"""
	
	@pytest.mark.asyncio
	async def test_complete_asset_lifecycle(self, eam_asset_service, sample_asset_data, sample_tenant_id):
		"""Test complete asset lifecycle from creation to retirement"""
		# 1. Create asset
		asset = await eam_asset_service.create_asset(sample_asset_data)
		assert asset.status == "active"
		
		# 2. Commission asset
		commissioned_asset = await eam_asset_service.commission_asset(
			asset.asset_id,
			{"commissioning_date": date.today()}
		)
		assert commissioned_asset.operational_status == "operational"
		
		# 3. Update asset health over time
		health_updates = [
			{"health_score": Decimal("90.0"), "condition_status": "good"},
			{"health_score": Decimal("85.0"), "condition_status": "fair"},
			{"health_score": Decimal("75.0"), "condition_status": "poor"}
		]
		
		for update in health_updates:
			await eam_asset_service.update_asset_health(asset.asset_id, update)
		
		# 4. Retire asset
		retired_asset = await eam_asset_service.retire_asset(
			asset.asset_id,
			{"retirement_date": date.today(), "retirement_reason": "End of useful life"}
		)
		assert retired_asset.status == "retired"
	
	@pytest.mark.asyncio
	async def test_maintenance_workflow_integration(self, sample_tenant_id):
		"""Test integrated maintenance workflow"""
		# This would test the complete flow from work order creation
		# through maintenance execution to performance impact
		
		work_order_service = EAMWorkOrderService()
		asset_service = EAMAssetService()
		
		# Mock services
		work_order_service.auth_service = AsyncMock()
		work_order_service.audit_service = AsyncMock()
		asset_service.auth_service = AsyncMock()
		asset_service.audit_service = AsyncMock()
		
		# Create work order for maintenance
		work_order_data = {
			"tenant_id": sample_tenant_id,
			"title": "Integration Test Maintenance",
			"description": "Test maintenance workflow",
			"work_type": "maintenance",
			"priority": "high"
		}
		
		work_order = await work_order_service.create_work_order(work_order_data)
		
		# Schedule and execute work order
		await work_order_service.schedule_work_order(
			work_order.work_order_id,
			{"assigned_to": "tech_001"}
		)
		
		await work_order_service.update_work_order_status(
			work_order.work_order_id,
			"completed"
		)
		
		# Verify completion
		completed_order = await work_order_service.get_work_order(
			work_order.work_order_id
		)
		assert completed_order.status == "completed"
	
	@pytest.mark.asyncio
	async def test_inventory_consumption_workflow(self, sample_tenant_id):
		"""Test inventory consumption through work orders"""
		inventory_service = EAMInventoryService()
		work_order_service = EAMWorkOrderService()
		
		# Mock services
		inventory_service.auth_service = AsyncMock()
		inventory_service.audit_service = AsyncMock()
		work_order_service.auth_service = AsyncMock()
		work_order_service.audit_service = AsyncMock()
		
		# Create inventory item
		inventory_data = {
			"tenant_id": sample_tenant_id,
			"part_number": "INT-001",
			"description": "Integration test part",
			"item_type": "spare_part",
			"category": "test",
			"current_stock": 100
		}
		
		item = await inventory_service.create_inventory_item(inventory_data)
		
		# Create work order that consumes inventory
		work_order_data = {
			"tenant_id": sample_tenant_id,
			"title": "Work Order with Parts",
			"description": "Test work order consuming parts",
			"work_type": "repair"
		}
		
		work_order = await work_order_service.create_work_order(work_order_data)
		
		# Complete work order with parts consumption
		completion_data = {
			"parts_used": [{"inventory_id": item.inventory_id, "quantity": 5}]
		}
		
		await work_order_service.complete_work_order(
			work_order.work_order_id,
			completion_data
		)
		
		# Verify inventory was reduced
		updated_item = await inventory_service.get_inventory_item(item.inventory_id)
		assert updated_item.current_stock == 95  # 100 - 5


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestEAMPerformance:
	"""Performance tests for EAM capability"""
	
	@pytest.mark.asyncio
	async def test_bulk_asset_operations(self, eam_asset_service, sample_tenant_id):
		"""Test bulk asset operations performance"""
		import time
		
		# Create multiple assets
		asset_count = 100
		assets_data = []
		
		for i in range(asset_count):
			asset_data = {
				"tenant_id": sample_tenant_id,
				"asset_number": f"BULK-{i:03d}",
				"asset_name": f"Bulk Test Asset {i}",
				"asset_type": "equipment",
				"asset_category": "test"
			}
			assets_data.append(asset_data)
		
		# Measure bulk creation time
		start_time = time.time()
		created_assets = await eam_asset_service.bulk_create_assets(assets_data)
		creation_time = time.time() - start_time
		
		assert len(created_assets) == asset_count
		assert creation_time < 10.0  # Should complete within 10 seconds
		
		# Test bulk search performance
		start_time = time.time()
		search_results = await eam_asset_service.search_assets(
			sample_tenant_id,
			{"asset_type": "equipment"}
		)
		search_time = time.time() - start_time
		
		assert len(search_results) >= asset_count
		assert search_time < 2.0  # Search should be fast
	
	@pytest.mark.asyncio
	async def test_concurrent_operations(self, sample_tenant_id):
		"""Test concurrent EAM operations"""
		import asyncio
		
		services = [
			EAMAssetService(),
			EAMWorkOrderService(),
			EAMInventoryService()
		]
		
		# Mock services
		for service in services:
			service.auth_service = AsyncMock()
			service.audit_service = AsyncMock()
		
		# Define concurrent operations
		async def create_assets():
			for i in range(10):
				asset_data = {
					"tenant_id": sample_tenant_id,
					"asset_number": f"CONC-A-{i}",
					"asset_name": f"Concurrent Asset {i}",
					"asset_type": "equipment",
					"asset_category": "test"
				}
				await services[0].create_asset(asset_data)
		
		async def create_work_orders():
			for i in range(10):
				wo_data = {
					"tenant_id": sample_tenant_id,
					"title": f"Concurrent WO {i}",
					"description": f"Test work order {i}",
					"work_type": "maintenance"
				}
				await services[1].create_work_order(wo_data)
		
		async def create_inventory():
			for i in range(10):
				inv_data = {
					"tenant_id": sample_tenant_id,
					"part_number": f"CONC-P-{i}",
					"description": f"Concurrent part {i}",
					"item_type": "spare_part",
					"category": "test"
				}
				await services[2].create_inventory_item(inv_data)
		
		# Run concurrent operations
		start_time = time.time()
		await asyncio.gather(
			create_assets(),
			create_work_orders(),
			create_inventory()
		)
		total_time = time.time() - start_time
		
		# Should handle concurrent operations efficiently
		assert total_time < 15.0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestEAMErrorHandling:
	"""Test error handling and resilience"""
	
	@pytest.mark.asyncio
	async def test_invalid_data_handling(self, eam_asset_service):
		"""Test handling of invalid data"""
		invalid_asset_data = {
			"asset_name": "",  # Empty name should fail
			"asset_type": "invalid_type",
			"purchase_cost": "not_a_number"
		}
		
		with pytest.raises(ValueError):
			await eam_asset_service.create_asset(invalid_asset_data)
	
	@pytest.mark.asyncio
	async def test_duplicate_asset_number_handling(self, eam_asset_service, sample_asset_data):
		"""Test handling of duplicate asset numbers"""
		# Create first asset
		await eam_asset_service.create_asset(sample_asset_data)
		
		# Try to create duplicate
		with pytest.raises(ValueError, match="Asset number already exists"):
			await eam_asset_service.create_asset(sample_asset_data)
	
	@pytest.mark.asyncio
	async def test_database_connection_failure(self, eam_asset_service, sample_asset_data):
		"""Test handling of database connection failures"""
		# Mock database failure
		with patch.object(eam_asset_service, '_execute_query', side_effect=ConnectionError("Database unavailable")):
			with pytest.raises(ConnectionError):
				await eam_asset_service.create_asset(sample_asset_data)
	
	@pytest.mark.asyncio
	async def test_permission_denied_handling(self, sample_asset_data):
		"""Test handling of permission denials"""
		service = EAMAssetService()
		
		# Mock permission denial
		mock_auth = AsyncMock()
		mock_auth.check_permission.side_effect = PermissionError("Access denied")
		service.auth_service = mock_auth
		
		with pytest.raises(PermissionError):
			await service.create_asset(sample_asset_data)


if __name__ == "__main__":
	pytest.main([__file__, "-v"])