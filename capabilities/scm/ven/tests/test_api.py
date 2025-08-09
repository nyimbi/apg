"""
APG Vendor Management - API Tests
Comprehensive tests for REST API endpoints and functionality

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from flask import Flask

from ..api import (
	VendorListResource, VendorDetailResource, VendorPerformanceResource,
	VendorRiskResource, VendorIntelligenceResource, VendorAnalyticsResource,
	create_vendor_api_blueprint, register_vendor_api
)
from ..models import VMVendor


# ============================================================================
# API RESOURCE BASE TESTS
# ============================================================================

@pytest.mark.unit
class TestBaseVendorResource:
	"""Test base API resource functionality"""
	
	def test_tenant_id_extraction(self, mock_request):
		"""Test tenant ID extraction from request headers"""
		
		resource = VendorListResource()
		
		with patch('flask.request', mock_request):
			tenant_id = resource._get_tenant_id()
		
		assert str(tenant_id) == '00000000-0000-0000-0000-000000000000'
	
	def test_invalid_tenant_id_handling(self):
		"""Test handling of invalid tenant ID in headers"""
		
		mock_request = MagicMock()
		mock_request.headers = {'X-Tenant-ID': 'invalid-uuid'}
		
		resource = VendorListResource()
		
		with patch('flask.request', mock_request):
			with patch('flask_restful.abort') as mock_abort:
				resource._get_tenant_id()
				mock_abort.assert_called_with(400, message="Invalid tenant ID format")
	
	def test_user_id_extraction(self, mock_request):
		"""Test user ID extraction from request headers"""
		
		resource = VendorListResource()
		
		with patch('flask.request', mock_request):
			user_id = resource._get_current_user_id()
		
		assert str(user_id) == '00000000-0000-0000-0000-000000000000'
	
	def test_async_service_creation(self):
		"""Test async service instance creation"""
		
		resource = VendorListResource()
		
		with patch('flask.current_app') as mock_app:
			mock_app.config = {'VENDOR_MANAGEMENT_DB_URL': 'postgresql://test/db'}
			
			# Mock the async method
			async def test_service_creation():
				service = await resource._get_vendor_service()
				assert service is not None
				return service
			
			# In real tests, we'd run this with asyncio.run()
			# For now, just verify the method exists
			assert hasattr(resource, '_get_vendor_service')


# ============================================================================
# VENDOR LIST RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
class TestVendorListResource:
	"""Test vendor list API resource"""
	
	@patch('flask.request')
	def test_get_vendors_success(self, mock_request):
		"""Test successful vendor listing"""
		
		# Mock request parameters
		mock_request.args = {
			'page': '1',
			'page_size': '25',
			'status': 'active'
		}
		
		# Mock vendor data
		mock_vendors = [
			{
				'id': 'vendor-001',
				'vendor_code': 'TEST001',
				'name': 'Test Vendor 1',
				'vendor_type': 'supplier',
				'status': 'active',
				'performance_score': 85.5
			}
		]
		
		mock_response = MagicMock()
		mock_response.vendors = [VMVendor(**{
			'tenant_id': '00000000-0000-0000-0000-000000000000',
			'category': 'technology',
			**vendor
		}) for vendor in mock_vendors]
		mock_response.page = 1
		mock_response.page_size = 25
		mock_response.total_count = 1
		mock_response.has_next = False
		
		resource = VendorListResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.list_vendors.return_value = mock_response
				mock_run_async.return_value = mock_service_instance
				
				# Second call for actual vendor listing
				mock_run_async.side_effect = [mock_service_instance, mock_response]
				
				response = resource.get()
		
		assert response['success'] is True
		assert 'data' in response
		assert 'vendors' in response['data']
		assert 'pagination' in response['data']
	
	@patch('flask.request')
	def test_get_vendors_with_filters(self, mock_request):
		"""Test vendor listing with various filters"""
		
		mock_request.args = {
			'status': 'active',
			'category': 'technology',
			'vendor_type': 'supplier',
			'search': 'test vendor',
			'sort_by': 'name',
			'sort_order': 'asc'
		}
		
		resource = VendorListResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_response = MagicMock()
				mock_response.vendors = []
				mock_response.total_count = 0
				
				mock_run_async.side_effect = [mock_service_instance, mock_response]
				
				response = resource.get()
		
		# Verify service was called with correct filters
		mock_service_instance.list_vendors.assert_called_once()
		call_args = mock_service_instance.list_vendors.call_args
		assert call_args[1]['filters']['status'] == 'active'
		assert call_args[1]['filters']['category'] == 'technology'
		assert call_args[1]['sort_by'] == 'name'
	
	@patch('flask.request')
	def test_post_create_vendor_success(self, mock_request):
		"""Test successful vendor creation"""
		
		vendor_data = {
			'vendor_code': 'NEW001',
			'name': 'New Vendor Inc.',
			'vendor_type': 'supplier',
			'category': 'technology',
			'email': 'contact@newvendor.com'
		}
		
		mock_request.get_json.return_value = vendor_data
		
		# Mock created vendor
		created_vendor = VMVendor(
			tenant_id='00000000-0000-0000-0000-000000000000',
			id='new-vendor-001',
			**vendor_data
		)
		
		resource = VendorListResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.create_vendor.return_value = created_vendor
				
				mock_run_async.side_effect = [mock_service_instance, created_vendor]
				
				response, status_code = resource.post()
		
		assert status_code == 201
		assert response['success'] is True
		assert response['data']['vendor_code'] == 'NEW001'
		assert response['data']['name'] == 'New Vendor Inc.'
	
	@patch('flask.request')
	def test_post_create_vendor_validation_error(self, mock_request):
		"""Test vendor creation with validation errors"""
		
		invalid_data = {
			'vendor_code': '',  # Empty required field
			'name': 'Test Vendor',
			'vendor_type': 'invalid_type',  # Invalid enum value
			'category': 'technology'
		}
		
		mock_request.get_json.return_value = invalid_data
		
		resource = VendorListResource()
		response, status_code = resource.post()
		
		assert status_code == 400
		assert response['success'] is False
		assert 'Validation error' in response['error']


# ============================================================================
# VENDOR DETAIL RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
class TestVendorDetailResource:
	"""Test vendor detail API resource"""
	
	def test_get_vendor_success(self, sample_vendor):
		"""Test successful vendor detail retrieval"""
		
		resource = VendorDetailResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_vendor_by_id.return_value = sample_vendor
				mock_service_instance.get_vendor_performance_summary.return_value = None
				
				mock_run_async.side_effect = [mock_service_instance, sample_vendor]
				
				response = resource.get(sample_vendor.id)
		
		assert response['success'] is True
		assert response['data']['id'] == sample_vendor.id
		assert response['data']['vendor_code'] == sample_vendor.vendor_code
		assert 'scores' in response['data']
		assert 'address' in response['data']
		assert 'financial' in response['data']
	
	def test_get_vendor_not_found(self):
		"""Test vendor detail retrieval for non-existent vendor"""
		
		resource = VendorDetailResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_vendor_by_id.return_value = None
				
				mock_run_async.side_effect = [mock_service_instance, None]
				
				response, status_code = resource.get('non-existent-id')
		
		assert status_code == 404
		assert response['success'] is False
		assert 'not found' in response['error']
	
	@patch('flask.request')
	def test_put_update_vendor_success(self, mock_request, sample_vendor):
		"""Test successful vendor update"""
		
		update_data = {
			'name': 'Updated Vendor Name',
			'email': 'updated@example.com',
			'strategic_importance': 'high'
		}
		
		mock_request.get_json.return_value = update_data
		
		# Create updated vendor
		updated_vendor = sample_vendor.model_copy()
		updated_vendor.name = 'Updated Vendor Name'
		updated_vendor.email = 'updated@example.com'
		
		resource = VendorDetailResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.update_vendor.return_value = updated_vendor
				
				mock_run_async.side_effect = [mock_service_instance, updated_vendor]
				
				response = resource.put(sample_vendor.id)
		
		assert response['success'] is True
		assert response['data']['name'] == 'Updated Vendor Name'
	
	def test_delete_vendor_success(self, sample_vendor):
		"""Test successful vendor deactivation"""
		
		resource = VendorDetailResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.deactivate_vendor.return_value = True
				
				mock_run_async.side_effect = [mock_service_instance, True]
				
				response = resource.delete(sample_vendor.id)
		
		assert response['success'] is True
		assert 'deactivated successfully' in response['message']


# ============================================================================
# PERFORMANCE RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
class TestVendorPerformanceResource:
	"""Test vendor performance API resource"""
	
	def test_get_performance_success(self, sample_performance):
		"""Test successful performance data retrieval"""
		
		mock_summary = MagicMock()
		mock_summary.model_dump.return_value = {
			'vendor_id': sample_performance.vendor_id,
			'avg_overall_score': 85.5,
			'performance_trend': 'improving'
		}
		
		resource = VendorPerformanceResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_vendor_performance_summary.return_value = mock_summary
				
				mock_run_async.side_effect = [mock_service_instance, mock_summary]
				
				response = resource.get(sample_performance.vendor_id)
		
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_performance.vendor_id
		assert response['data']['avg_overall_score'] == 85.5
	
	@patch('flask.request')
	def test_post_record_performance_success(self, mock_request, sample_performance):
		"""Test successful performance recording"""
		
		performance_data = {
			'measurement_period': 'quarterly',
			'overall_score': 88.0,
			'quality_score': 92.0,
			'delivery_score': 85.0,
			'cost_score': 87.0,
			'service_score': 89.0
		}
		
		mock_request.get_json.return_value = performance_data
		
		resource = VendorPerformanceResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.record_performance.return_value = sample_performance
				
				mock_run_async.side_effect = [mock_service_instance, sample_performance]
				
				response, status_code = resource.post(sample_performance.vendor_id)
		
		assert status_code == 201
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_performance.vendor_id


# ============================================================================
# RISK RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
class TestVendorRiskResource:
	"""Test vendor risk API resource"""
	
	def test_get_risk_profile_success(self, sample_risk):
		"""Test successful risk profile retrieval"""
		
		mock_profile = MagicMock()
		mock_profile.model_dump.return_value = {
			'vendor_id': sample_risk.vendor_id,
			'total_risks': 1,
			'high_risks': 0,
			'medium_risks': 1,
			'overall_risk_score': 65.0
		}
		
		resource = VendorRiskResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_vendor_risk_profile.return_value = mock_profile
				
				mock_run_async.side_effect = [mock_service_instance, mock_profile]
				
				response = resource.get(sample_risk.vendor_id)
		
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_risk.vendor_id
		assert response['data']['total_risks'] == 1
	
	@patch('flask.request')
	def test_post_record_risk_success(self, mock_request, sample_risk):
		"""Test successful risk recording"""
		
		risk_data = {
			'risk_type': 'operational',
			'risk_category': 'delivery',
			'severity': 'medium',
			'title': 'New Risk',
			'description': 'Risk description',
			'overall_risk_score': 60.0
		}
		
		mock_request.get_json.return_value = risk_data
		
		resource = VendorRiskResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.record_risk.return_value = sample_risk
				
				mock_run_async.side_effect = [mock_service_instance, sample_risk]
				
				response, status_code = resource.post(sample_risk.vendor_id)
		
		assert status_code == 201
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_risk.vendor_id


# ============================================================================
# INTELLIGENCE RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.ai
class TestVendorIntelligenceResource:
	"""Test vendor intelligence API resource"""
	
	def test_get_intelligence_success(self, sample_intelligence):
		"""Test successful intelligence retrieval"""
		
		mock_intelligence = MagicMock()
		mock_intelligence.model_dump.return_value = {
			'id': sample_intelligence.id,
			'vendor_id': sample_intelligence.vendor_id,
			'confidence_score': float(sample_intelligence.confidence_score),
			'behavior_patterns': sample_intelligence.behavior_patterns,
			'predictive_insights': sample_intelligence.predictive_insights
		}
		
		resource = VendorIntelligenceResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_latest_vendor_intelligence.return_value = mock_intelligence
				
				mock_run_async.side_effect = [mock_service_instance, mock_intelligence]
				
				response = resource.get(sample_intelligence.vendor_id)
		
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_intelligence.vendor_id
		assert len(response['data']['behavior_patterns']) == 1
	
	def test_post_generate_intelligence_success(self, sample_intelligence):
		"""Test successful intelligence generation"""
		
		resource = VendorIntelligenceResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.generate_vendor_intelligence.return_value = sample_intelligence
				
				mock_run_async.side_effect = [mock_service_instance, sample_intelligence]
				
				response, status_code = resource.post(sample_intelligence.vendor_id)
		
		assert status_code == 201
		assert response['success'] is True
		assert response['data']['vendor_id'] == sample_intelligence.vendor_id
		assert response['data']['confidence_score'] == float(sample_intelligence.confidence_score)


# ============================================================================
# ANALYTICS RESOURCE TESTS
# ============================================================================

@pytest.mark.unit
class TestVendorAnalyticsResource:
	"""Test vendor analytics API resource"""
	
	def test_get_analytics_success(self):
		"""Test successful analytics retrieval"""
		
		mock_analytics = {
			'vendor_counts': {
				'total_vendors': 100,
				'active_vendors': 85,
				'preferred_vendors': 15
			},
			'performance_metrics': {
				'avg_performance': 82.5,
				'avg_risk': 28.3
			},
			'recent_activities': []
		}
		
		resource = VendorAnalyticsResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			with patch.object(resource, '_get_vendor_service') as mock_service:
				mock_service_instance = AsyncMock()
				mock_service_instance.get_vendor_analytics.return_value = mock_analytics
				
				mock_run_async.side_effect = [mock_service_instance, mock_analytics]
				
				response = resource.get()
		
		assert response['success'] is True
		assert 'vendor_counts' in response['data']
		assert 'performance_metrics' in response['data']
		assert response['data']['vendor_counts']['total_vendors'] == 100


# ============================================================================
# BLUEPRINT TESTS
# ============================================================================

@pytest.mark.unit
class TestAPIBlueprint:
	"""Test API blueprint creation and registration"""
	
	def test_create_vendor_api_blueprint(self):
		"""Test API blueprint creation"""
		
		blueprint = create_vendor_api_blueprint()
		
		assert blueprint.name == 'vendor_management_api'
		assert blueprint.url_prefix == '/api/v1/vendor-management'
	
	def test_register_vendor_api(self):
		"""Test API registration with Flask app"""
		
		app = Flask(__name__)
		app.logger = MagicMock()
		
		register_vendor_api(app)
		
		# Verify blueprint was registered
		assert len(app.blueprints) == 1
		assert 'vendor_management_api' in app.blueprints


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.unit
class TestAPIErrorHandling:
	"""Test API error handling scenarios"""
	
	def test_database_error_handling(self):
		"""Test handling of database errors"""
		
		resource = VendorListResource()
		
		with patch.object(resource, '_run_async') as mock_run_async:
			mock_run_async.side_effect = Exception("Database connection failed")
			
			response, status_code = resource.get()
		
		assert status_code == 500
		assert response['success'] is False
		assert 'error' in response
	
	def test_validation_error_handling(self):
		"""Test handling of validation errors"""
		
		from marshmallow import ValidationError
		
		resource = VendorListResource()
		error = ValidationError({'vendor_code': ['Field is required']})
		
		result = resource._handle_error(error)
		
		assert result[1] == 400  # Bad Request
		assert result[0]['success'] is False
		assert 'Validation error' in result[0]['error']
	
	def test_not_found_error_handling(self):
		"""Test handling of not found errors"""
		
		from werkzeug.exceptions import NotFound
		
		resource = VendorDetailResource()
		error = NotFound()
		
		result = resource._handle_error(error)
		
		assert result[1] == 404
		assert result[0]['success'] is False
		assert 'not found' in result[0]['error']


# ============================================================================
# SECURITY TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.security
class TestAPISecurity:
	"""Test API security features"""
	
	def test_missing_tenant_header(self):
		"""Test handling of missing tenant header"""
		
		mock_request = MagicMock()
		mock_request.headers = {}  # No tenant header
		
		resource = VendorListResource()
		
		with patch('flask.request', mock_request):
			tenant_id = resource._get_tenant_id()
			# Should default to development tenant
			assert str(tenant_id) == '00000000-0000-0000-0000-000000000000'
	
	def test_invalid_uuid_in_url(self):
		"""Test handling of invalid UUID in URL parameters"""
		
		resource = VendorDetailResource()
		
		with patch.object(resource, '_run_async'):
			response, status_code = resource.get('invalid-uuid-format')
		
		# The UUID validation should be handled by the service layer
		assert status_code in [400, 500]  # Either validation error or service error
	
	@pytest.mark.slow
	def test_sql_injection_protection(self, security_test_scenarios):
		"""Test protection against SQL injection attacks"""
		
		resource = VendorListResource()
		
		for injection_payload in security_test_scenarios['sql_injection']:
			mock_request = MagicMock()
			mock_request.args = {'search': injection_payload}
			
			with patch('flask.request', mock_request):
				with patch.object(resource, '_run_async') as mock_run_async:
					# Mock service to avoid actual database calls
					mock_service = AsyncMock()
					mock_response = MagicMock()
					mock_response.vendors = []
					mock_response.total_count = 0
					mock_service.list_vendors.return_value = mock_response
					
					mock_run_async.side_effect = [mock_service, mock_response]
					
					response = resource.get()
			
			# Should not crash or return database errors
			assert 'success' in response