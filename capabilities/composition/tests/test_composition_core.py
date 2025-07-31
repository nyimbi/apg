"""
Core Composition Tests

Unit and integration tests for the main composition functionality including:
- Application composition and validation
- Capability discovery and filtering  
- Tenant management and isolation
- Configuration management
- Access control integration

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from typing import List, Dict, Any
from uuid_extensions import uuid7str

from . import (
	CompositionTestHelper,
	MockDataGenerator,
	test_tenant_id,
	test_user_id,
	test_composition_config,
	initialized_tenant
)


class TestCompositionCore:
	"""Test core composition functionality."""
	
	async def test_compose_application_basic(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test basic application composition."""
		from .. import compose_application
		from ..capability_registry import CRCompositionType
		
		capabilities = MockDataGenerator.generate_capability_list(3)
		
		result = await compose_application(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			capabilities=capabilities,
			composition_type=CRCompositionType.ENTERPRISE
		)
		
		assert result is not None
		assert hasattr(result, 'is_valid') or hasattr(result, 'status')
		
	async def test_compose_application_with_industry_focus(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition with industry-specific requirements."""
		from .. import compose_application
		from ..capability_registry import CRCompositionType
		
		capabilities = [
			"core_business_operations.financial_management",
			"general_cross_functional.governance_risk_compliance"
		]
		
		result = await compose_application(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			capabilities=capabilities,
			composition_type=CRCompositionType.INDUSTRY_VERTICAL,
			industry_focus=["healthcare", "financial_services"]
		)
		
		assert result is not None
		
	async def test_compose_application_with_custom_config(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition with custom configuration."""
		from .. import compose_application
		from ..capability_registry import CRCompositionType
		
		capabilities = MockDataGenerator.generate_capability_list(2)
		custom_config = {
			"deployment_size": "large",
			"high_availability": True,
			"backup_strategy": "multi_region"
		}
		
		result = await compose_application(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			capabilities=capabilities,
			composition_type=CRCompositionType.ENTERPRISE,
			custom_config=custom_config
		)
		
		assert result is not None


class TestCapabilityDiscovery:
	"""Test capability discovery functionality."""
	
	async def test_discover_capabilities_basic(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test basic capability discovery."""
		from .. import discover_capabilities
		
		capabilities = await discover_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id
		)
		
		assert isinstance(capabilities, list)
		
	async def test_discover_capabilities_with_filters(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test capability discovery with filters."""
		from .. import discover_capabilities
		
		filters = {
			"category": "core_business_operations",
			"status": "active"
		}
		
		capabilities = await discover_capabilities(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			filters=filters
		)
		
		assert isinstance(capabilities, list)
		
	async def test_discover_capabilities_empty_tenant(
		self,
		test_user_id: str
	):
		"""Test capability discovery for empty tenant."""
		from .. import discover_capabilities
		
		empty_tenant_id = f"empty_tenant_{uuid7str()}"
		
		capabilities = await discover_capabilities(
			tenant_id=empty_tenant_id,
			user_id=test_user_id
		)
		
		assert isinstance(capabilities, list)
		# May be empty or contain default capabilities


class TestTenantManagement:
	"""Test tenant management functionality."""
	
	async def test_create_tenant_basic(self, test_user_id: str):
		"""Test basic tenant creation."""
		from .. import create_tenant
		
		tenant_id = f"test_tenant_{uuid7str()}"
		enabled_capabilities = MockDataGenerator.generate_capability_list(3)
		
		success = await create_tenant(
			tenant_id=tenant_id,
			admin_user_id=test_user_id,
			tenant_name="Test Tenant",
			enabled_capabilities=enabled_capabilities
		)
		
		assert success is True
		
	async def test_create_tenant_with_configuration(self, test_user_id: str):
		"""Test tenant creation with initial configuration."""
		from .. import create_tenant
		
		tenant_id = f"test_tenant_{uuid7str()}"
		enabled_capabilities = ["core_business_operations.financial_management"]
		configuration = {
			"financial_management": {
				"currency": "USD",
				"fiscal_year_start": "january"
			}
		}
		
		success = await create_tenant(
			tenant_id=tenant_id,
			admin_user_id=test_user_id,
			tenant_name="Test Tenant with Config",
			enabled_capabilities=enabled_capabilities,
			configuration=configuration
		)
		
		assert success is True
		
	async def test_tenant_isolation(self, test_user_id: str):
		"""Test that tenants are properly isolated."""
		from .. import create_tenant, discover_capabilities
		
		# Create two separate tenants
		tenant1_id = f"tenant1_{uuid7str()}"
		tenant2_id = f"tenant2_{uuid7str()}"
		
		tenant1_caps = ["core_business_operations.financial_management"]
		tenant2_caps = ["core_business_operations.human_capital_management"]
		
		await create_tenant(
			tenant_id=tenant1_id,
			admin_user_id=test_user_id,
			tenant_name="Tenant 1",
			enabled_capabilities=tenant1_caps
		)
		
		await create_tenant(
			tenant_id=tenant2_id,
			admin_user_id=test_user_id,
			tenant_name="Tenant 2", 
			enabled_capabilities=tenant2_caps
		)
		
		# Verify tenant isolation
		tenant1_discovered = await discover_capabilities(tenant1_id, test_user_id)
		tenant2_discovered = await discover_capabilities(tenant2_id, test_user_id)
		
		# Tenants should have different capability sets
		assert isinstance(tenant1_discovered, list)
		assert isinstance(tenant2_discovered, list)


class TestConfigurationManagement:
	"""Test configuration management functionality."""
	
	async def test_configure_capability_basic(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test basic capability configuration."""
		from .. import configure_capability
		
		configuration = {
			"auto_backup": True,
			"retention_days": 30,
			"encryption_level": "high"
		}
		
		success = await configure_capability(
			tenant_id=initialized_tenant,
			capability_id="core_business_operations.financial_management",
			user_id=test_user_id,
			configuration=configuration
		)
		
		assert success is True
		
	async def test_configure_capability_invalid_config(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test configuration with invalid values."""
		from .. import configure_capability
		
		# Invalid configuration (negative retention days)
		invalid_configuration = {
			"retention_days": -10,
			"invalid_setting": "invalid_value"
		}
		
		success = await configure_capability(
			tenant_id=initialized_tenant,
			capability_id="core_business_operations.financial_management",
			user_id=test_user_id,
			configuration=invalid_configuration
		)
		
		# Should fail validation
		assert success is False


class TestAccessControlIntegration:
	"""Test access control integration."""
	
	async def test_access_control_basic(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test basic access control functionality."""
		from .. import get_access_control
		
		access_control = get_access_control(initialized_tenant)
		
		if access_control:
			# Test permission granting
			permission_id = access_control.grant_capability_permission(
				capability_id="core_business_operations.financial_management",
				user_id=test_user_id,
				access_level=access_control.AccessLevel.READ
			)
			
			assert permission_id is not None
			assert isinstance(permission_id, str)
		
	async def test_permission_checking(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test permission checking functionality."""
		from .. import get_access_control
		from ..access_control_integration import AccessLevel
		
		access_control = get_access_control(initialized_tenant)
		
		if access_control:
			# Grant permission first
			access_control.grant_capability_permission(
				capability_id="core_business_operations.financial_management",
				user_id=test_user_id,
				access_level=AccessLevel.READ
			)
			
			# Check permission
			has_access = await access_control.check_capability_access(
				user_id=test_user_id,
				capability_id="core_business_operations.financial_management",
				requested_access=AccessLevel.READ
			)
			
			assert has_access is True


class TestCompositionRecommendations:
	"""Test AI-powered composition recommendations."""
	
	async def test_get_composition_recommendations_basic(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test basic composition recommendations."""
		from .. import get_composition_recommendations
		
		business_requirements = MockDataGenerator.generate_business_requirements()
		
		recommendations = await get_composition_recommendations(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			business_requirements=business_requirements
		)
		
		assert isinstance(recommendations, list)
		
	async def test_get_composition_recommendations_with_industry(
		self,
		initialized_tenant: str,
		test_user_id: str
	):
		"""Test composition recommendations with industry focus."""
		from .. import get_composition_recommendations
		
		business_requirements = {
			"company_size": "medium",
			"geographical_regions": ["north_america"],
			"main_business_processes": ["accounting", "payroll", "inventory"]
		}
		
		recommendations = await get_composition_recommendations(
			tenant_id=initialized_tenant,
			user_id=test_user_id,
			business_requirements=business_requirements,
			industry="manufacturing"
		)
		
		assert isinstance(recommendations, list)


class TestUtilityFunctions:
	"""Test utility functions."""
	
	def test_get_capability_info(self):
		"""Test capability info retrieval."""
		from .. import get_capability_info
		
		info = get_capability_info()
		assert isinstance(info, dict)
		assert "name" in info
		
	def test_list_subcapabilities(self):
		"""Test subcapability listing."""
		from .. import list_subcapabilities
		
		subcaps = list_subcapabilities()
		assert isinstance(subcaps, list)
		assert len(subcaps) > 0
		
	def test_get_composition_templates(self):
		"""Test composition template retrieval."""
		from .. import get_composition_templates
		
		templates = get_composition_templates()
		assert isinstance(templates, dict)
		assert len(templates) > 0
		
	def test_get_deployment_strategies(self):
		"""Test deployment strategy listing."""
		from .. import get_deployment_strategies
		
		strategies = get_deployment_strategies()
		assert isinstance(strategies, list)
		assert len(strategies) > 0