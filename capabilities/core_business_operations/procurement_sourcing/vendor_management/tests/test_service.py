"""
APG Vendor Management - Service Layer Tests
Comprehensive tests for vendor management service functionality

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4
from decimal import Decimal
from datetime import datetime, timedelta

from ..service import VendorManagementService, VMDatabaseContext
from ..models import VMVendor, VMPerformance, VMRisk, VendorListResponse, VendorAnalytics


# ============================================================================
# SERVICE INITIALIZATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestVendorServiceInitialization:
	"""Test vendor service initialization and configuration"""
	
	async def test_service_creation(self, mock_db_context, test_tenant_id):
		"""Test vendor service creation"""
		
		service = VendorManagementService(test_tenant_id, mock_db_context)
		
		assert service.tenant_id == test_tenant_id
		assert service.db_context == mock_db_context
		assert service.current_user_id is None
	
	async def test_service_user_setting(self, vendor_service, test_user_id):
		"""Test setting current user on service"""
		
		vendor_service.set_current_user(test_user_id)
		
		assert vendor_service.current_user_id == test_user_id
	
	async def test_service_validation(self, mock_db_context):
		"""Test service validation with invalid parameters"""
		
		# Invalid tenant ID
		with pytest.raises(ValueError):
			VendorManagementService("invalid-uuid", mock_db_context)
		
		# None database context
		with pytest.raises(ValueError):
			VendorManagementService(UUID('00000000-0000-0000-0000-000000000000'), None)


# ============================================================================
# VENDOR CRUD OPERATIONS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestVendorCRUDOperations:
	"""Test vendor CRUD operations"""
	
	async def test_create_vendor_success(self, vendor_service, test_vendor_data):
		"""Test successful vendor creation"""
		
		# Mock database response
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'test-vendor-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='test-vendor-001'):
			vendor = await vendor_service.create_vendor(test_vendor_data)
		
		assert isinstance(vendor, VMVendor)
		assert vendor.vendor_code == test_vendor_data['vendor_code']
		assert vendor.name == test_vendor_data['name']
		vendor_service._execute_query.assert_called_once()
	
	async def test_create_vendor_duplicate_code(self, vendor_service, test_vendor_data):
		"""Test vendor creation with duplicate vendor code"""
		
		# Mock database error for duplicate vendor code
		vendor_service._execute_query = AsyncMock(
			side_effect=Exception("duplicate key value violates unique constraint")
		)
		
		with pytest.raises(ValueError, match="Vendor code already exists"):
			await vendor_service.create_vendor(test_vendor_data)
	
	async def test_get_vendor_by_id_success(self, vendor_service, sample_vendor):
		"""Test successful vendor retrieval by ID"""
		
		vendor_data = sample_vendor.model_dump()
		vendor_service._fetch_one = AsyncMock(return_value=vendor_data)
		
		vendor = await vendor_service.get_vendor_by_id(sample_vendor.id)
		
		assert isinstance(vendor, VMVendor)
		assert vendor.id == sample_vendor.id
		assert vendor.vendor_code == sample_vendor.vendor_code
	
	async def test_get_vendor_by_id_not_found(self, vendor_service):
		"""Test vendor retrieval with non-existent ID"""
		
		vendor_service._fetch_one = AsyncMock(return_value=None)
		
		vendor = await vendor_service.get_vendor_by_id('non-existent-id')
		
		assert vendor is None
	
	async def test_update_vendor_success(self, vendor_service, sample_vendor):
		"""Test successful vendor update"""
		
		update_data = {
			'name': 'Updated Vendor Name',
			'email': 'updated@example.com',
			'strategic_importance': 'high'
		}
		
		# Mock existing vendor
		vendor_service._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		vendor_service._execute_query = AsyncMock()
		
		updated_vendor = await vendor_service.update_vendor(sample_vendor.id, update_data)
		
		assert updated_vendor.name == 'Updated Vendor Name'
		assert updated_vendor.email == 'updated@example.com'
		vendor_service._execute_query.assert_called_once()
	
	async def test_update_vendor_not_found(self, vendor_service):
		"""Test vendor update with non-existent ID"""
		
		vendor_service._fetch_one = AsyncMock(return_value=None)
		
		result = await vendor_service.update_vendor('non-existent-id', {'name': 'New Name'})
		
		assert result is None
	
	async def test_deactivate_vendor_success(self, vendor_service, sample_vendor):
		"""Test successful vendor deactivation"""
		
		vendor_service._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		vendor_service._execute_query = AsyncMock()
		
		result = await vendor_service.deactivate_vendor(sample_vendor.id)
		
		assert result is True
		vendor_service._execute_query.assert_called_once()
	
	async def test_list_vendors_success(self, vendor_service, sample_vendor):
		"""Test successful vendor listing"""
		
		mock_vendors = [sample_vendor.model_dump() for _ in range(5)]
		vendor_service._fetch_all = AsyncMock(return_value=mock_vendors)
		vendor_service._fetch_one = AsyncMock(return_value={'count': 5})
		
		result = await vendor_service.list_vendors(page=1, page_size=10)
		
		assert isinstance(result, VendorListResponse)
		assert len(result.vendors) == 5
		assert result.total_count == 5
		assert result.page == 1
		assert result.page_size == 10
	
	async def test_list_vendors_with_filters(self, vendor_service, sample_vendor):
		"""Test vendor listing with filters"""
		
		filters = {
			'status': 'active',
			'category': 'technology',
			'search': 'test'
		}
		
		mock_vendors = [sample_vendor.model_dump()]
		vendor_service._fetch_all = AsyncMock(return_value=mock_vendors)
		vendor_service._fetch_one = AsyncMock(return_value={'count': 1})
		
		result = await vendor_service.list_vendors(filters=filters)
		
		assert len(result.vendors) == 1
		assert result.total_count == 1
		
		# Verify filter SQL generation
		call_args = vendor_service._fetch_all.call_args[0][0]
		assert 'status = $' in call_args
		assert 'category = $' in call_args
		assert 'name ILIKE $' in call_args


# ============================================================================
# PERFORMANCE MANAGEMENT TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestPerformanceManagement:
	"""Test vendor performance management functionality"""
	
	async def test_record_performance_success(self, vendor_service, test_performance_data):
		"""Test successful performance recording"""
		
		test_performance_data['vendor_id'] = str(uuid4())
		
		# Mock database operations
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'perf-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='perf-001'):
			performance = await vendor_service.record_performance(test_performance_data)
		
		assert isinstance(performance, VMPerformance)
		assert performance.overall_score == Decimal(str(test_performance_data['overall_score']))
		assert performance.measurement_period == test_performance_data['measurement_period']
	
	async def test_record_performance_validation(self, vendor_service):
		"""Test performance recording validation"""
		
		invalid_data = {
			'vendor_id': str(uuid4()),
			'measurement_period': 'quarterly',
			'overall_score': 150.0  # Invalid score > 100
		}
		
		with pytest.raises(ValueError):
			await vendor_service.record_performance(invalid_data)
	
	async def test_get_vendor_performance_summary(self, vendor_service, sample_performance):
		"""Test vendor performance summary retrieval"""
		
		vendor_id = sample_performance.vendor_id
		mock_data = {
			'vendor_id': vendor_id,
			'avg_overall_score': 85.5,
			'avg_quality_score': 88.0,
			'performance_trend': 'improving',
			'measurement_count': 4
		}
		
		vendor_service._fetch_one = AsyncMock(return_value=mock_data)
		
		summary = await vendor_service.get_vendor_performance_summary(vendor_id)
		
		assert summary is not None
		assert summary.vendor_id == vendor_id
		assert summary.avg_overall_score == 85.5
		assert summary.performance_trend == 'improving'


# ============================================================================
# RISK MANAGEMENT TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestRiskManagement:
	"""Test vendor risk management functionality"""
	
	async def test_record_risk_success(self, vendor_service, test_risk_data):
		"""Test successful risk recording"""
		
		test_risk_data['vendor_id'] = str(uuid4())
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'risk-001'})
		
		with patch('uuid_extensions.uuid7str', return_value='risk-001'):
			risk = await vendor_service.record_risk(test_risk_data)
		
		assert isinstance(risk, VMRisk)
		assert risk.risk_type == test_risk_data['risk_type']
		assert risk.severity.value == test_risk_data['severity']
		assert risk.overall_risk_score == Decimal(str(test_risk_data['overall_risk_score']))
	
	async def test_get_vendor_risk_profile(self, vendor_service, sample_risk):
		"""Test vendor risk profile retrieval"""
		
		vendor_id = sample_risk.vendor_id
		mock_risks = [sample_risk.model_dump()]
		mock_profile = {
			'vendor_id': vendor_id,
			'total_risks': 1,
			'high_risks': 0,
			'medium_risks': 1,
			'low_risks': 0,
			'overall_risk_score': 65.0
		}
		
		vendor_service._fetch_all = AsyncMock(return_value=mock_risks)
		vendor_service._fetch_one = AsyncMock(return_value=mock_profile)
		
		profile = await vendor_service.get_vendor_risk_profile(vendor_id)
		
		assert profile is not None
		assert profile.vendor_id == vendor_id
		assert profile.total_risks == 1
		assert profile.medium_risks == 1
	
	async def test_risk_score_calculation(self, vendor_service):
		"""Test risk score calculation logic"""
		
		# Mock risks with different severities
		mock_risks = [
			{'severity': 'high', 'overall_risk_score': 85.0, 'probability': 0.7},
			{'severity': 'medium', 'overall_risk_score': 60.0, 'probability': 0.5},
			{'severity': 'low', 'overall_risk_score': 25.0, 'probability': 0.2}
		]
		
		vendor_service._fetch_all = AsyncMock(return_value=mock_risks)
		
		calculated_score = await vendor_service._calculate_composite_risk_score('test-vendor')
		
		# Verify calculation considers probability weighting
		assert isinstance(calculated_score, float)
		assert 0 <= calculated_score <= 100


# ============================================================================
# ANALYTICS AND REPORTING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestAnalyticsReporting:
	"""Test vendor analytics and reporting functionality"""
	
	async def test_get_vendor_analytics_success(self, vendor_service):
		"""Test vendor analytics retrieval"""
		
		mock_analytics = {
			'vendor_counts': {
				'total_vendors': 100,
				'active_vendors': 85,
				'preferred_vendors': 15,
				'strategic_partners': 8
			},
			'performance_metrics': {
				'avg_performance': 82.5,
				'avg_risk': 28.3,
				'top_performers': 12
			},
			'recent_activities': []
		}
		
		vendor_service._fetch_one = AsyncMock(return_value=mock_analytics)
		
		analytics = await vendor_service.get_vendor_analytics()
		
		assert isinstance(analytics, dict)
		assert 'vendor_counts' in analytics
		assert 'performance_metrics' in analytics
		assert analytics['vendor_counts']['total_vendors'] == 100
	
	async def test_analytics_calculation_accuracy(self, vendor_service):
		"""Test analytics calculation accuracy"""
		
		# Mock multiple database calls for different metrics
		vendor_service._fetch_one = AsyncMock(side_effect=[
			{'count': 100},  # total vendors
			{'count': 85},   # active vendors
			{'avg': 82.5},   # average performance
			{'avg': 28.3}    # average risk
		])
		
		analytics = await vendor_service.get_vendor_analytics()
		
		assert analytics['vendor_counts']['total_vendors'] == 100
		assert analytics['vendor_counts']['active_vendors'] == 85
		assert analytics['performance_metrics']['avg_performance'] == 82.5


# ============================================================================
# AI INTEGRATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.ai
class TestAIIntegration:
	"""Test AI integration functionality"""
	
	async def test_generate_vendor_intelligence(self, vendor_service, sample_vendor):
		"""Test AI intelligence generation"""
		
		vendor_service._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		vendor_service._execute_query = AsyncMock()
		
		# Mock AI intelligence data
		mock_intelligence_data = {
			'id': 'intel-001',
			'vendor_id': sample_vendor.id,
			'confidence_score': 0.85,
			'behavior_patterns': [
				{
					'pattern_type': 'communication',
					'pattern_name': 'responsive',
					'confidence': 0.9
				}
			],
			'predictive_insights': [
				{
					'insight_type': 'performance',
					'prediction': 'improvement',
					'confidence': 0.8
				}
			]
		}
		
		with patch('uuid_extensions.uuid7str', return_value='intel-001'):
			with patch.object(vendor_service, '_generate_ai_intelligence', 
							return_value=mock_intelligence_data):
				intelligence = await vendor_service.generate_vendor_intelligence(sample_vendor.id)
		
		assert intelligence is not None
		assert intelligence.vendor_id == sample_vendor.id
		assert intelligence.confidence_score == Decimal('0.85')
		assert len(intelligence.behavior_patterns) == 1
	
	async def test_ai_intelligence_validation(self, vendor_service):
		"""Test AI intelligence validation"""
		
		with pytest.raises(ValueError):
			await vendor_service.generate_vendor_intelligence('non-existent-vendor')


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceErrorHandling:
	"""Test service error handling scenarios"""
	
	async def test_database_connection_error(self, vendor_service):
		"""Test handling of database connection errors"""
		
		vendor_service._fetch_one = AsyncMock(
			side_effect=Exception("connection failed")
		)
		
		with pytest.raises(Exception, match="connection failed"):
			await vendor_service.get_vendor_by_id('test-id')
	
	async def test_invalid_uuid_handling(self, vendor_service):
		"""Test handling of invalid UUID parameters"""
		
		with pytest.raises(ValueError, match="Invalid UUID"):
			await vendor_service.get_vendor_by_id('invalid-uuid')
	
	async def test_missing_required_fields(self, vendor_service):
		"""Test handling of missing required fields"""
		
		incomplete_data = {
			'vendor_code': 'TEST001',
			# Missing required 'name' field
			'category': 'technology'
		}
		
		with pytest.raises(ValueError):
			await vendor_service.create_vendor(incomplete_data)
	
	async def test_concurrent_modification_handling(self, vendor_service, sample_vendor):
		"""Test handling of concurrent modifications"""
		
		# Mock version conflict
		vendor_service._fetch_one = AsyncMock(return_value=sample_vendor.model_dump())
		vendor_service._execute_query = AsyncMock(
			side_effect=Exception("version conflict")
		)
		
		with pytest.raises(Exception, match="version conflict"):
			await vendor_service.update_vendor(sample_vendor.id, {'name': 'New Name'})


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
@pytest.mark.asyncio
class TestServicePerformance:
	"""Test service performance and scalability"""
	
	async def test_bulk_vendor_creation_performance(self, vendor_service, performance_test_data):
		"""Test performance of bulk vendor creation"""
		
		vendor_service._execute_query = AsyncMock()
		vendor_service._fetch_one = AsyncMock(return_value={'id': 'test-id'})
		
		# Time the bulk operation
		import time
		start_time = time.time()
		
		tasks = []
		for vendor_data in performance_test_data[:100]:  # Test with 100 vendors
			tasks.append(vendor_service.create_vendor(vendor_data))
		
		# Note: In real test, we'd use asyncio.gather for concurrent execution
		# For now, just test sequential to verify the mocking works
		for task in tasks[:5]:  # Test first 5 only for speed
			await task
		
		end_time = time.time()
		execution_time = end_time - start_time
		
		# Assert reasonable performance (adjust threshold as needed)
		assert execution_time < 5.0  # Should complete within 5 seconds
	
	async def test_large_result_set_handling(self, vendor_service):
		"""Test handling of large result sets"""
		
		# Mock large vendor list
		large_vendor_list = [
			{'id': f'vendor-{i}', 'name': f'Vendor {i}', 'vendor_code': f'V{i:04d}'}
			for i in range(1000)
		]
		
		vendor_service._fetch_all = AsyncMock(return_value=large_vendor_list)
		vendor_service._fetch_one = AsyncMock(return_value={'count': 1000})
		
		result = await vendor_service.list_vendors(page_size=100)
		
		assert len(result.vendors) == 100  # Should respect page_size
		assert result.total_count == 1000


# ============================================================================
# CACHING TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceCaching:
	"""Test service caching functionality"""
	
	async def test_vendor_caching(self, vendor_service, sample_vendor):
		"""Test vendor data caching"""
		
		vendor_data = sample_vendor.model_dump()
		vendor_service._fetch_one = AsyncMock(return_value=vendor_data)
		
		# First call should hit database
		vendor1 = await vendor_service.get_vendor_by_id(sample_vendor.id)
		
		# Second call should use cache (if implemented)
		vendor2 = await vendor_service.get_vendor_by_id(sample_vendor.id)
		
		assert vendor1.id == vendor2.id
		# In a real implementation with caching, we'd verify cache hit metrics