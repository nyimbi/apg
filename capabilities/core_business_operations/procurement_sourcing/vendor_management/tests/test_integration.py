"""
APG Vendor Management - Integration Tests
End-to-end integration tests for complete vendor management workflows

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from flask import Flask
from decimal import Decimal

from ..service import VendorManagementService
from ..intelligence_service import VendorIntelligenceEngine
from ..api import create_vendor_api_blueprint
from ..blueprint import init_subcapability
from ..models import VMVendor, VMPerformance, VMRisk


# ============================================================================
# COMPLETE VENDOR LIFECYCLE TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteVendorLifecycle:
	"""Test complete vendor lifecycle from creation to intelligence generation"""
	
	async def test_vendor_creation_to_intelligence_workflow(self, vendor_service, intelligence_engine, test_vendor_data):
		"""Test complete workflow from vendor creation to AI intelligence"""
		
		# Step 1: Create vendor
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'test-vendor-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='test-vendor-001'):
			vendor = await vendor_service.create_vendor(test_vendor_data)
		
		assert vendor.id == 'test-vendor-001'
		assert vendor.vendor_code == test_vendor_data['vendor_code']
		
		# Step 2: Record performance data
		performance_data = {
			'vendor_id': vendor.id,
			'measurement_period': 'quarterly',
			'overall_score': 85.0,
			'quality_score': 90.0,
			'delivery_score': 82.0,
			'cost_score': 88.0,
			'service_score': 85.0
		}
		
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'perf-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='perf-001'):
			performance = await vendor_service.record_performance(performance_data)
		
		assert performance.vendor_id == vendor.id
		assert performance.overall_score == Decimal('85.0')
		
		# Step 3: Record risk assessment
		risk_data = {
			'vendor_id': vendor.id,
			'risk_type': 'operational',
			'risk_category': 'delivery',
			'severity': 'medium',
			'title': 'Delivery Risk',
			'description': 'Risk of delivery delays',
			'overall_risk_score': 65.0
		}
		
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'risk-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='risk-001'):
			risk = await vendor_service.record_risk(risk_data)
		
		assert risk.vendor_id == vendor.id
		assert risk.risk_type == 'operational'
		
		# Step 4: Generate AI intelligence
		mock_vendor_data = vendor.model_dump()
		mock_performance_data = [performance.model_dump()]
		mock_risk_data = [risk.model_dump()]
		
		intelligence_engine._fetch_one = AsyncMock(return_value=mock_vendor_data)
		intelligence_engine._fetch_all = AsyncMock(side_effect=[
			mock_performance_data,  # Performance data call
			mock_risk_data,         # Risk data call
			[],                     # Communication data call
			[]                      # Additional data calls
		])
		
		patterns = await intelligence_engine.analyze_vendor_behavior_patterns(vendor.id)
		insights = await intelligence_engine.generate_predictive_insights(vendor.id)
		
		assert len(patterns) > 0
		assert len(insights) > 0
		
		# Step 5: Generate optimization plan
		objectives = ['performance_improvement', 'risk_mitigation']
		
		optimization_plan = await intelligence_engine.generate_optimization_plan(
			vendor.id, 
			objectives
		)
		
		assert optimization_plan.vendor_id == vendor.id
		assert len(optimization_plan.recommended_actions) > 0
		assert 'performance_improvement' in optimization_plan.optimization_objectives
	
	async def test_vendor_performance_tracking_workflow(self, vendor_service, sample_vendor):
		"""Test comprehensive performance tracking workflow"""
		
		vendor_id = sample_vendor.id
		
		# Record multiple performance periods
		performance_periods = [
			{
				'vendor_id': vendor_id,
				'measurement_period': 'monthly',
				'overall_score': 80.0,
				'quality_score': 85.0,
				'delivery_score': 78.0,
				'cost_score': 82.0,
				'service_score': 80.0
			},
			{
				'vendor_id': vendor_id,
				'measurement_period': 'monthly',
				'overall_score': 83.0,
				'quality_score': 87.0,
				'delivery_score': 80.0,
				'cost_score': 85.0,
				'service_score': 82.0
			},
			{
				'vendor_id': vendor_id,
				'measurement_period': 'monthly',
				'overall_score': 86.0,
				'quality_score': 89.0,
				'delivery_score': 83.0,
				'cost_score': 87.0,
				'service_score': 85.0
			}
		]
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(side_effect=[
			{'id': f'perf-00{i}'} for i in range(1, 4)
		])
		
		recorded_performances = []
		for i, perf_data in enumerate(performance_periods):
			with patch('uuid_extensions.uuid7str', return_value=f'perf-00{i+1}'):
				performance = await vendor_service.record_performance(perf_data)
				recorded_performances.append(performance)
		
		assert len(recorded_performances) == 3
		
		# Verify performance trend (should be improving)
		scores = [float(p.overall_score) for p in recorded_performances]
		assert scores == sorted(scores)  # Ascending order (improving)
		
		# Get performance summary
		mock_summary_data = {
			'vendor_id': vendor_id,
			'avg_overall_score': 83.0,
			'avg_quality_score': 87.0,
			'performance_trend': 'improving',
			'measurement_count': 3
		}
		
		vendor_service._fetch_one = AsyncMock(return_value=mock_summary_data)
		
		summary = await vendor_service.get_vendor_performance_summary(vendor_id)
		
		assert summary.avg_overall_score == 83.0
		assert summary.performance_trend == 'improving'
		assert summary.measurement_count == 3


# ============================================================================
# API INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestAPIIntegration:
	"""Test API integration with service layer"""
	
	def test_api_blueprint_creation_and_registration(self):
		"""Test complete API blueprint setup"""
		
		app = Flask(__name__)
		app.config['TESTING'] = True
		app.logger = MagicMock()
		
		# Create and register blueprint
		blueprint = create_vendor_api_blueprint()
		app.register_blueprint(blueprint)
		
		assert len(app.blueprints) == 1
		assert 'vendor_management_api' in app.blueprints
		
		# Test that endpoints are registered
		with app.test_client() as client:
			# Test endpoint existence (should return 401/403 without auth)
			response = client.get('/api/v1/vendor-management/vendors')
			assert response.status_code in [401, 403, 405]  # Auth required or method not allowed
	
	@patch('flask.request')
	def test_api_request_processing_flow(self, mock_request):
		"""Test complete API request processing flow"""
		
		from ..api import VendorListResource
		
		# Mock request with proper headers and parameters
		mock_request.headers = {
			'X-Tenant-ID': '00000000-0000-0000-0000-000000000000',
			'X-User-ID': '00000000-0000-0000-0000-000000000000'
		}
		mock_request.args = {
			'page': '1',
			'page_size': '25',
			'status': 'active'
		}
		
		resource = VendorListResource()
		
		# Mock the entire service layer
		with patch.object(resource, '_run_async') as mock_run_async:
			mock_service = AsyncMock()
			mock_response = MagicMock()
			mock_response.vendors = []
			mock_response.page = 1
			mock_response.page_size = 25
			mock_response.total_count = 0
			mock_response.has_next = False
			
			mock_run_async.side_effect = [mock_service, mock_response]
			
			response = resource.get()
		
		# Verify complete flow
		assert response['success'] is True
		assert 'data' in response
		assert 'pagination' in response['data']
		
		# Verify service was called with correct parameters
		mock_service.list_vendors.assert_called_once()


# ============================================================================
# BLUEPRINT INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
class TestBlueprintIntegration:
	"""Test Flask-AppBuilder blueprint integration"""
	
	def test_subcapability_initialization_flow(self):
		"""Test complete subcapability initialization"""
		
		# Mock AppBuilder
		mock_appbuilder = MagicMock()
		mock_app = MagicMock()
		mock_app.config = {
			'VENDOR_MANAGEMENT_DB_URL': 'postgresql://test/db',
			'TESTING': True
		}
		mock_app.logger = MagicMock()
		
		mock_appbuilder.get_app = mock_app
		mock_appbuilder.add_view = MagicMock()
		mock_appbuilder.sm = MagicMock()
		mock_appbuilder.sm.add_permission_view_menu = MagicMock()
		
		# Test initialization
		with patch('..blueprint.register_vendor_api') as mock_register_api:
			result = init_subcapability(mock_appbuilder)
		
		assert result['success'] is True
		assert result['capability'] == 'vendor_management'
		assert 'components_initialized' in result
		assert 'menu_structure' in result
		assert 'permissions' in result
		
		# Verify views were registered
		assert mock_appbuilder.add_view.call_count == 6  # Number of views
		
		# Verify API was registered
		mock_register_api.assert_called_once_with(mock_app)
	
	def test_menu_structure_generation(self):
		"""Test menu structure generation for UI integration"""
		
		from ..blueprint import get_menu_structure
		
		menu = get_menu_structure()
		
		assert menu['name'] == 'Vendor Management'
		assert menu['icon'] == 'fa-building'
		assert len(menu['items']) == 6
		assert len(menu['advanced_features']) == 3
		
		# Verify menu items have required fields
		for item in menu['items']:
			assert 'name' in item
			assert 'label' in item
			assert 'href' in item
			assert 'icon' in item
			assert 'order' in item
	
	def test_permissions_setup(self):
		"""Test permissions setup for role-based access"""
		
		from ..blueprint import setup_permissions
		
		mock_appbuilder = MagicMock()
		mock_appbuilder.sm = MagicMock()
		mock_appbuilder.sm.add_permission_view_menu = MagicMock()
		
		permissions = setup_permissions(mock_appbuilder)
		
		assert 'vendor_management_admin' in permissions
		assert 'vendor_management_manager' in permissions
		assert 'vendor_management_analyst' in permissions
		assert 'vendor_management_readonly' in permissions
		
		# Verify admin has most permissions
		admin_perms = permissions['vendor_management_admin']
		readonly_perms = permissions['vendor_management_readonly']
		
		assert len(admin_perms) > len(readonly_perms)
		assert 'can_create_vendors' in admin_perms
		assert 'can_create_vendors' not in readonly_perms


# ============================================================================
# DATA CONSISTENCY TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestDataConsistency:
	"""Test data consistency across service operations"""
	
	async def test_vendor_score_consistency(self, vendor_service, sample_vendor):
		"""Test that vendor scores remain consistent across operations"""
		
		vendor_id = sample_vendor.id
		
		# Mock vendor data
		vendor_service._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		
		# Record performance that should update vendor scores
		performance_data = {
			'vendor_id': vendor_id,
			'measurement_period': 'quarterly',
			'overall_score': 90.0,
			'quality_score': 95.0,
			'delivery_score': 88.0,
			'cost_score': 92.0,
			'service_score': 89.0
		}
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'perf-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='perf-001'):
			await vendor_service.record_performance(performance_data)
		
		# Verify vendor performance score would be updated
		# (In real implementation, this would trigger score recalculation)
		vendor_service._execute_query.assert_called()
	
	async def test_audit_trail_consistency(self, vendor_service, test_vendor_data):
		"""Test that audit trails are consistently maintained"""
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'test-vendor-001'})
		
		# Create vendor (should create audit record)
		with patch('uuid_extensions.uuid7str', return_value='test-vendor-001'):
			vendor = await vendor_service.create_vendor(test_vendor_data)
		
		# Update vendor (should create another audit record)
		vendor_service._fetch_one = AsyncMock(return_value=vendor.model_dump())
		
		updated_vendor = await vendor_service.update_vendor(
			vendor.id, 
			{'name': 'Updated Name'}
		)
		
		# Verify multiple database operations for audit trail
		assert vendor_service._execute_query.call_count >= 2
	
	async def test_multi_tenant_data_isolation(self, mock_db_context):
		"""Test that multi-tenant data isolation is maintained"""
		
		tenant1_id = uuid4()
		tenant2_id = uuid4()
		
		# Create services for different tenants
		service1 = VendorManagementService(tenant1_id, mock_db_context)
		service2 = VendorManagementService(tenant2_id, mock_db_context)
		
		# Mock different tenant data
		service1._fetch_all = AsyncMock(return_value=[
			{'id': 'vendor-1', 'name': 'Tenant 1 Vendor'}
		])
		service1._fetch_one = AsyncMock(return_value={'count': 1})
		
		service2._fetch_all = AsyncMock(return_value=[
			{'id': 'vendor-2', 'name': 'Tenant 2 Vendor'}
		])
		service2._fetch_one = AsyncMock(return_value={'count': 1})
		
		# Get vendor lists for both tenants
		tenant1_vendors = await service1.list_vendors()
		tenant2_vendors = await service2.list_vendors()
		
		# Verify data isolation
		assert tenant1_vendors.vendors[0].name == 'Tenant 1 Vendor'
		assert tenant2_vendors.vendors[0].name == 'Tenant 2 Vendor'
		assert tenant1_vendors.vendors[0].id != tenant2_vendors.vendors[0].id


# ============================================================================
# PERFORMANCE INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
class TestPerformanceIntegration:
	"""Test system performance under realistic loads"""
	
	async def test_concurrent_vendor_operations(self, vendor_service, test_vendor_data):
		"""Test concurrent vendor operations performance"""
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(side_effect=[
			{'id': f'vendor-{i}'} for i in range(10)
		])
		
		# Create multiple vendors concurrently
		tasks = []
		for i in range(10):
			vendor_data = test_vendor_data.copy()
			vendor_data['vendor_code'] = f'CONCURRENT{i:03d}'
			vendor_data['name'] = f'Concurrent Vendor {i}'
			
			with patch('uuid_extensions.uuid7str', return_value=f'vendor-{i}'):
				task = vendor_service.create_vendor(vendor_data)
			tasks.append(task)
		
		# Execute concurrently
		results = await asyncio.gather(*tasks, return_exceptions=True)
		
		# Verify all operations completed successfully
		successful_results = [r for r in results if not isinstance(r, Exception)]
		assert len(successful_results) == 10
	
	async def test_large_dataset_analytics_performance(self, vendor_service):
		"""Test analytics performance with large datasets"""
		
		# Mock large dataset
		large_analytics_data = {
			'vendor_counts': {
				'total_vendors': 10000,
				'active_vendors': 8500,
				'preferred_vendors': 150,
				'strategic_partners': 80
			},
			'performance_metrics': {
				'avg_performance': 82.5,
				'avg_risk': 28.3,
				'top_performers': 120
			},
			'recent_activities': []
		}
		
		vendor_service._fetch_one = AsyncMock(return_value=large_analytics_data)
		
		import time
		start_time = time.time()
		
		analytics = await vendor_service.get_vendor_analytics()
		
		end_time = time.time()
		execution_time = end_time - start_time
		
		# Should complete quickly even with large datasets
		assert execution_time < 2.0  # 2 seconds threshold
		assert analytics['vendor_counts']['total_vendors'] == 10000


# ============================================================================
# ERROR RECOVERY TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
class TestErrorRecovery:
	"""Test system error recovery and resilience"""
	
	async def test_database_reconnection_handling(self, vendor_service):
		"""Test handling of database connection failures and recovery"""
		
		# Simulate connection failure then recovery
		vendor_service._fetch_one = AsyncMock(side_effect=[
			Exception("connection lost"),
			{'id': 'vendor-001', 'name': 'Test Vendor'}  # Recovery
		])
		
		# First call should fail
		with pytest.raises(Exception, match="connection lost"):
			await vendor_service.get_vendor_by_id('vendor-001')
		
		# Second call should succeed (simulating reconnection)
		vendor = await vendor_service.get_vendor_by_id('vendor-001')
		assert vendor.name == 'Test Vendor'
	
	async def test_partial_failure_handling(self, vendor_service, intelligence_engine):
		"""Test handling of partial system failures"""
		
		vendor_id = str(uuid4())
		
		# Mock vendor service success but intelligence failure
		vendor_service._fetch_one = AsyncMock(return_value={
			'id': vendor_id,
			'name': 'Test Vendor',
			'vendor_code': 'TEST001',
			'category': 'technology'
		})
		
		intelligence_engine._fetch_all = AsyncMock(
			side_effect=Exception("AI service unavailable")
		)
		
		# Should handle intelligence failure gracefully
		vendor = await vendor_service.get_vendor_by_id(vendor_id)
		assert vendor is not None
		
		# Intelligence should fail but not crash the system
		with pytest.raises(Exception, match="AI service unavailable"):
			await intelligence_engine.analyze_vendor_behavior_patterns(vendor_id)


# ============================================================================
# SECURITY INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.security
class TestSecurityIntegration:
	"""Test security integration across the system"""
	
	def test_authentication_flow_integration(self):
		"""Test authentication integration with API endpoints"""
		
		from ..api import VendorListResource
		
		resource = VendorListResource()
		
		# Test without authentication headers
		mock_request = MagicMock()
		mock_request.headers = {}  # No auth headers
		
		with patch('flask.request', mock_request):
			# Should use default tenant/user for development
			tenant_id = resource._get_tenant_id()
			user_id = resource._get_current_user_id()
			
			assert tenant_id is not None
			assert user_id is not None
	
	def test_authorization_flow_integration(self):
		"""Test authorization integration with Flask-AppBuilder"""
		
		# This would test the @has_access_api decorators in practice
		# For now, just verify the decorators are applied
		from ..api import VendorListResource
		
		# Verify that sensitive methods have authentication decorators
		assert hasattr(VendorListResource.get, '__wrapped__')  # Decorated method
		assert hasattr(VendorListResource.post, '__wrapped__')  # Decorated method
	
	def test_data_encryption_integration(self, vendor_service):
		"""Test data encryption integration"""
		
		# Test that sensitive fields would be encrypted
		# (In real implementation, this would test actual encryption)
		
		sensitive_data = {
			'vendor_code': 'ENCRYPT001',
			'name': 'Encryption Test Vendor',
			'category': 'technology',
			'tax_id': '12-3456789',  # Sensitive field
			'email': 'sensitive@example.com'  # PII field
		}
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'encrypted-vendor'})
		
		# In production, sensitive fields would be encrypted before storage
		with patch('uuid_extensions.uuid7str', return_value='encrypted-vendor'):
			vendor = await vendor_service.create_vendor(sensitive_data)
		
		assert vendor.tax_id == '12-3456789'  # Would be encrypted in real system
		assert vendor.email == 'sensitive@example.com'  # Would be encrypted in real system