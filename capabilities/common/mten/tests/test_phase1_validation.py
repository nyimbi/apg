"""
Phase 1 Validation Tests for Multi-Tenant Management Capability

Tests core functionality of the MTen capability following CLAUDE.md standards.
No pytest.mark.asyncio decorators needed per CLAUDE.md guidelines.
"""

import asyncio
import pytest
from datetime import datetime, UTC
from typing import Dict, Any, Optional

from ..models import (
	Tenant, TenantStatus, TenantTier, 
	TenantTemplate, TenantMetrics,
	TenantConfiguration, ResourceAllocation
)
from ..service import MultiTenantManager
from ..views import (
	TenantCreateRequest, TenantResponse,
	TenantUpdateRequest, TenantListResponse
)


class TestTenantModels:
	"""Test core tenant data models and validation"""
	
	def test_tenant_creation_basic(self):
		"""Test basic tenant model creation with required fields"""
		tenant = Tenant(
			name="test-tenant",
			display_name="Test Tenant",
			organization_name="Test Org",
			contact_email="test@example.com",
			primary_domain="test.example.com",
			created_by="user123"
		)
		
		assert tenant.name == "test-tenant"
		assert tenant.status == TenantStatus.PROVISIONING
		assert tenant.tier == TenantTier.FREE
		assert tenant.id is not None
		assert len(tenant.id) > 20  # uuid7str should be long
	
	def test_tenant_name_validation(self):
		"""Test tenant name validation rules"""
		# Valid names
		valid_names = ["test-tenant", "tenant123", "my-org-2024"]
		for name in valid_names:
			tenant = Tenant(
				name=name,
				display_name="Test",
				organization_name="Test Org",
				contact_email="test@example.com", 
				primary_domain="test.example.com",
				created_by="user123"
			)
			assert tenant.name == name
	
	def test_resource_allocation_model(self):
		"""Test resource allocation configuration"""
		allocation = ResourceAllocation(
			cpu_cores=4,
			memory_gb=16,
			storage_gb=100,
			bandwidth_mbps=1000,
			database_connections=50
		)
		
		assert allocation.cpu_cores == 4
		assert allocation.memory_gb == 16
		assert allocation.total_compute_units() == 70  # Based on formula
		assert allocation.is_within_limits(TenantTier.PREMIUM) is True
	
	def test_tenant_metrics_model(self):
		"""Test tenant metrics tracking"""
		metrics = TenantMetrics(
			tenant_id="test-123",
			cpu_usage_percent=45.5,
			memory_usage_percent=67.2,
			storage_usage_gb=234.8,
			api_requests_per_minute=1250,
			active_users=89,
			data_transfer_gb=12.5
		)
		
		assert metrics.cpu_usage_percent == 45.5
		assert metrics.is_healthy() is True  # All metrics within normal ranges
		assert metrics.performance_score() > 70  # Should be good performance


class TestTenantService:
	"""Test core tenant management service functionality"""
	
	async def test_service_initialization(self):
		"""Test service initialization with APG integration"""
		service = MultiTenantManager(
			tenant_id="system",
			db_url="postgresql://test:test@localhost/test_mten",
			apg_auth_endpoint="http://localhost:8080/auth"
		)
		
		await service.initialize({
			'enable_ai_optimization': True,
			'provisioning_timeout_seconds': 60,
			'default_tier': 'free'
		})
		
		assert service.tenant_id == "system"
		assert service._ai_optimization_enabled is True
		assert service._provisioning_timeout == 60
	
	async def test_tenant_creation_flow(self):
		"""Test complete tenant creation workflow"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({})
		
		# Test tenant creation
		tenant = await service.create_tenant(
			name="test-org-2024",
			display_name="Test Organization 2024",
			organization_name="Test Corp",
			contact_email="admin@testcorp.com",
			primary_domain="testcorp.example.com",
			created_by="admin-user-123",
			tier=TenantTier.PREMIUM
		)
		
		assert tenant.name == "test-org-2024" 
		assert tenant.tier == TenantTier.PREMIUM
		assert tenant.status == TenantStatus.PROVISIONING
		assert tenant.created_by == "admin-user-123"
		assert tenant.created_at is not None
	
	async def test_tenant_provisioning_speed(self):
		"""Test <60 second provisioning requirement"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({'enable_fast_provisioning': True})
		
		start_time = datetime.now(UTC)
		
		tenant = await service.create_tenant(
			name="speed-test",
			display_name="Speed Test Tenant", 
			organization_name="Speed Test Org",
			contact_email="speed@test.com",
			primary_domain="speed.test.com",
			created_by="tester"
		)
		
		# Simulate provisioning completion
		provisioned_tenant = await service.complete_tenant_provisioning(tenant.id)
		
		end_time = datetime.now(UTC)
		provisioning_time = (end_time - start_time).total_seconds()
		
		assert provisioned_tenant.status == TenantStatus.ACTIVE
		assert provisioning_time < 60  # Must be under 60 seconds
	
	async def test_ai_powered_optimization(self):
		"""Test AI-powered tenant optimization"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({'enable_ai_optimization': True})
		
		# Create tenant with initial metrics
		tenant = await service.create_tenant(
			name="ai-test",
			display_name="AI Test Tenant",
			organization_name="AI Test Org", 
			contact_email="ai@test.com",
			primary_domain="ai.test.com",
			created_by="ai-tester"
		)
		
		# Generate optimization recommendations
		recommendations = await service.generate_optimization_recommendations(tenant.id)
		
		assert len(recommendations) > 0
		assert any(rec.category in ['resource_allocation', 'performance', 'cost'] 
				  for rec in recommendations)
		assert all(rec.confidence_score >= 0.7 for rec in recommendations)
	
	async def test_multi_cloud_deployment(self):
		"""Test multi-cloud deployment capabilities"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({
			'cloud_providers': ['aws', 'azure', 'gcp'],
			'enable_multi_cloud': True
		})
		
		tenant = await service.create_tenant(
			name="multi-cloud-test", 
			display_name="Multi-Cloud Test",
			organization_name="Multi-Cloud Org",
			contact_email="cloud@test.com",
			primary_domain="cloud.test.com",
			created_by="cloud-tester"
		)
		
		# Test cloud provider selection
		deployment_plan = await service.generate_deployment_plan(
			tenant.id, 
			preferred_regions=['us-east-1', 'europe-west1', 'eastus']
		)
		
		assert len(deployment_plan.cloud_deployments) >= 2
		assert any(dep.provider == 'aws' for dep in deployment_plan.cloud_deployments)
		assert deployment_plan.estimated_monthly_cost > 0


class TestTenantViews:
	"""Test Pydantic view models for API integration"""
	
	def test_tenant_create_request_validation(self):
		"""Test tenant creation request validation"""
		request = TenantCreateRequest(
			name="valid-tenant-name",
			display_name="Valid Tenant Display Name",
			organization_name="Valid Organization",
			contact_email="valid@example.com",
			primary_domain="valid.example.com"
		)
		
		assert request.name == "valid-tenant-name"
		assert request.tier == TenantTier.FREE  # Default value
		
		# Test serialization/deserialization
		request_dict = request.model_dump()
		restored_request = TenantCreateRequest.model_validate(request_dict)
		assert restored_request.name == request.name
	
	def test_tenant_response_model(self):
		"""Test tenant response model structure"""
		tenant = Tenant(
			name="response-test",
			display_name="Response Test",
			organization_name="Test Org", 
			contact_email="response@test.com",
			primary_domain="response.test.com",
			created_by="response-tester"
		)
		
		response = TenantResponse.from_tenant(tenant)
		
		assert response.id == tenant.id
		assert response.name == tenant.name
		assert response.status == tenant.status.value
		assert response.tier == tenant.tier.value
		assert response.created_at == tenant.created_at
	
	def test_tenant_list_response_pagination(self):
		"""Test tenant list response with pagination"""
		tenants = [
			Tenant(
				name=f"tenant-{i}",
				display_name=f"Tenant {i}",
				organization_name=f"Org {i}",
				contact_email=f"tenant{i}@test.com", 
				primary_domain=f"tenant{i}.test.com",
				created_by="list-tester"
			) for i in range(25)
		]
		
		response = TenantListResponse(
			tenants=[TenantResponse.from_tenant(t) for t in tenants[:10]],
			total_count=25,
			page=1,
			page_size=10,
			has_next=True
		)
		
		assert len(response.tenants) == 10
		assert response.total_count == 25
		assert response.has_next is True
		assert response.total_pages == 3


class TestAPGIntegration:
	"""Test APG ecosystem integration points"""
	
	async def test_auth_rbac_integration(self):
		"""Test integration with auth_rbac capability"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({
			'apg_auth_endpoint': 'http://localhost:8080/auth',
			'enable_rbac_integration': True
		})
		
		# Test tenant-scoped permissions
		permissions = await service.get_tenant_permissions(
			tenant_id="test-tenant-123",
			user_id="user-456"
		)
		
		assert 'tenant_admin' in permissions.roles
		assert 'tenant.read' in permissions.capabilities
		assert 'tenant.manage_users' in permissions.capabilities
	
	async def test_audit_compliance_integration(self):
		"""Test integration with audit_compliance capability"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({
			'enable_audit_logging': True,
			'compliance_framework': 'SOC2'
		})
		
		tenant = await service.create_tenant(
			name="audit-test",
			display_name="Audit Test",
			organization_name="Audit Org",
			contact_email="audit@test.com", 
			primary_domain="audit.test.com",
			created_by="audit-tester"
		)
		
		# Verify audit trail creation
		audit_entries = await service.get_tenant_audit_trail(tenant.id)
		
		assert len(audit_entries) > 0
		assert any(entry.action == 'tenant_created' for entry in audit_entries)
		assert all(entry.compliance_tags.get('SOC2') == 'tracked' for entry in audit_entries)


class TestPerformanceBenchmarks:
	"""Test performance benchmarks and SLA compliance"""
	
	async def test_concurrent_tenant_creation(self):
		"""Test concurrent tenant creation performance"""
		service = MultiTenantManager(tenant_id="system")
		await service.initialize({'enable_concurrent_provisioning': True})
		
		async def create_test_tenant(index: int) -> Tenant:
			return await service.create_tenant(
				name=f"concurrent-{index}",
				display_name=f"Concurrent Test {index}",
				organization_name=f"Concurrent Org {index}",
				contact_email=f"concurrent{index}@test.com",
				primary_domain=f"concurrent{index}.test.com", 
				created_by="concurrent-tester"
			)
		
		start_time = datetime.now(UTC)
		
		# Create 10 tenants concurrently
		tasks = [create_test_tenant(i) for i in range(10)]
		tenants = await asyncio.gather(*tasks)
		
		end_time = datetime.now(UTC)
		total_time = (end_time - start_time).total_seconds()
		
		assert len(tenants) == 10
		assert all(t.status == TenantStatus.PROVISIONING for t in tenants)
		assert total_time < 120  # Should handle 10 concurrent creations in <2 minutes
	
	async def test_resource_allocation_optimization(self):
		"""Test resource allocation optimization algorithms"""
		service = MultiTenantManager(tenant_id="system") 
		await service.initialize({'enable_ai_optimization': True})
		
		# Create tenants with varying resource needs
		tenants = []
		for i, tier in enumerate([TenantTier.FREE, TenantTier.PREMIUM, TenantTier.ENTERPRISE]):
			tenant = await service.create_tenant(
				name=f"resource-test-{i}",
				display_name=f"Resource Test {i}",
				organization_name=f"Resource Org {i}",
				contact_email=f"resource{i}@test.com",
				primary_domain=f"resource{i}.test.com",
				created_by="resource-tester",
				tier=tier
			)
			tenants.append(tenant)
		
		# Test resource optimization across all tenants
		optimization_plan = await service.optimize_global_resources()
		
		assert optimization_plan.total_cost_savings > 0
		assert optimization_plan.performance_improvement_percent > 5
		assert len(optimization_plan.tenant_adjustments) == len(tenants)


if __name__ == "__main__":
	# Run tests following CLAUDE.md guidelines
	# No pytest.mark.asyncio decorators needed
	loop = asyncio.get_event_loop()
	
	# Basic model tests (synchronous)
	model_tests = TestTenantModels()
	model_tests.test_tenant_creation_basic()
	model_tests.test_tenant_name_validation()
	model_tests.test_resource_allocation_model()
	model_tests.test_tenant_metrics_model()
	print("✓ Model tests passed")
	
	# View model tests (synchronous)
	view_tests = TestTenantViews()
	view_tests.test_tenant_create_request_validation()
	view_tests.test_tenant_response_model()
	view_tests.test_tenant_list_response_pagination()
	print("✓ View model tests passed")
	
	# Service tests (async)
	service_tests = TestTenantService()
	loop.run_until_complete(service_tests.test_service_initialization())
	loop.run_until_complete(service_tests.test_tenant_creation_flow())
	loop.run_until_complete(service_tests.test_ai_powered_optimization())
	loop.run_until_complete(service_tests.test_multi_cloud_deployment())
	print("✓ Service tests passed")
	
	# APG integration tests (async)
	apg_tests = TestAPGIntegration()
	loop.run_until_complete(apg_tests.test_auth_rbac_integration())
	loop.run_until_complete(apg_tests.test_audit_compliance_integration())
	print("✓ APG integration tests passed")
	
	# Performance benchmark tests (async)
	perf_tests = TestPerformanceBenchmarks()
	loop.run_until_complete(perf_tests.test_concurrent_tenant_creation())
	loop.run_until_complete(perf_tests.test_resource_allocation_optimization())
	print("✓ Performance benchmark tests passed")
	
	print("\n🎉 All Phase 1 validation tests completed successfully!")
	print("Multi-Tenant Management capability foundation is validated and ready for Phase 2.")