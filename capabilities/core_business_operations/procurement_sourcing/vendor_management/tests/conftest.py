"""
APG Vendor Management - Test Configuration
Pytest configuration and fixtures for comprehensive testing

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from ..models import VMVendor, VMPerformance, VMRisk, VMIntelligence
from ..service import VendorManagementService, VMDatabaseContext
from ..intelligence_service import VendorIntelligenceEngine


# ============================================================================
# DATABASE FIXTURES
# ============================================================================

@pytest_asyncio.fixture
async def mock_db_context() -> VMDatabaseContext:
	"""Mock database context for testing"""
	
	context = MagicMock(spec=VMDatabaseContext)
	context.connection_string = "postgresql://test:test@localhost/apg_test"
	context.pool = AsyncMock()
	context.get_connection = AsyncMock()
	
	return context


@pytest_asyncio.fixture
async def vendor_service(mock_db_context, test_tenant_id) -> VendorManagementService:
	"""Create vendor management service for testing"""
	
	service = VendorManagementService(test_tenant_id, mock_db_context)
	service.set_current_user(UUID('00000000-0000-0000-0000-000000000000'))
	
	# Mock database operations
	service._execute_query = AsyncMock()
	service._fetch_one = AsyncMock()
	service._fetch_all = AsyncMock()
	
	return service


@pytest_asyncio.fixture
async def intelligence_engine(mock_db_context, test_tenant_id) -> VendorIntelligenceEngine:
	"""Create intelligence engine for testing"""
	
	engine = VendorIntelligenceEngine(test_tenant_id, mock_db_context)
	engine.set_current_user(UUID('00000000-0000-0000-0000-000000000000'))
	
	# Mock AI operations
	engine._analyze_patterns = AsyncMock()
	engine._generate_predictions = AsyncMock()
	engine._optimize_relationships = AsyncMock()
	
	return engine


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def sample_vendor(test_tenant_id) -> VMVendor:
	"""Create sample vendor for testing"""
	
	return VMVendor(
		id='test-vendor-001',
		tenant_id=test_tenant_id,
		vendor_code='TEST001',
		name='Test Vendor Inc.',
		legal_name='Test Vendor Incorporated',
		display_name='Test Vendor',
		vendor_type='supplier',
		category='technology',
		subcategory='software',
		industry='information_technology',
		email='contact@testvendor.com',
		phone='+1-555-0123',
		website='https://testvendor.com',
		strategic_importance='standard',
		preferred_vendor=False,
		strategic_partner=False,
		performance_score=85.5,
		risk_score=25.0,
		intelligence_score=88.0,
		relationship_score=82.5
	)


@pytest.fixture
def sample_performance(test_tenant_id) -> VMPerformance:
	"""Create sample performance record for testing"""
	
	return VMPerformance(
		id='test-performance-001',
		tenant_id=test_tenant_id,
		vendor_id='test-vendor-001',
		measurement_period='quarterly',
		overall_score=85.5,
		quality_score=90.0,
		delivery_score=82.0,
		cost_score=88.0,
		service_score=85.0,
		on_time_delivery_rate=95.0,
		quality_rejection_rate=2.5,
		order_volume=1000000.0,
		order_count=50,
		total_spend=950000.0,
		average_order_value=19000.0
	)


@pytest.fixture
def sample_risk(test_tenant_id) -> VMRisk:
	"""Create sample risk record for testing"""
	
	return VMRisk(
		id='test-risk-001',
		tenant_id=test_tenant_id,
		vendor_id='test-vendor-001',
		risk_type='operational',
		risk_category='delivery',
		severity='medium',
		title='Potential delivery delays',
		description='Risk of delivery delays due to supply chain issues',
		root_cause='Dependency on single supplier',
		potential_impact='Project delays and cost overruns',
		overall_risk_score=65.0,
		financial_impact=50000.0,
		operational_impact=6,
		reputational_impact=4,
		mitigation_strategy='Diversify supplier base',
		mitigation_status='identified'
	)


@pytest.fixture
def sample_intelligence(test_tenant_id) -> VMIntelligence:
	"""Create sample intelligence record for testing"""
	
	return VMIntelligence(
		id='test-intelligence-001',
		tenant_id=test_tenant_id,
		vendor_id='test-vendor-001',
		model_version='v1.0',
		confidence_score=0.85,
		behavior_patterns=[
			{
				'pattern_type': 'communication',
				'pattern_name': 'responsive_communication',
				'confidence': 0.9,
				'description': 'Vendor responds quickly to communications'
			}
		],
		predictive_insights=[
			{
				'insight_type': 'performance_forecast',
				'prediction': 'performance_improvement',
				'confidence': 0.8,
				'time_horizon': 90,
				'description': 'Expected 5% improvement in overall performance'
			}
		],
		performance_forecasts={
			'next_quarter': {
				'overall_score': 88.0,
				'confidence': 0.75
			}
		},
		risk_assessments={
			'delivery_risk': {
				'probability': 0.15,
				'impact': 'medium',
				'confidence': 0.8
			}
		}
	)


# ============================================================================
# API FIXTURES
# ============================================================================

@pytest.fixture
def mock_flask_app():
	"""Mock Flask application for API testing"""
	
	app = MagicMock()
	app.config = {
		'VENDOR_MANAGEMENT_DB_URL': 'postgresql://test:test@localhost/apg_test',
		'VENDOR_AI_ENABLED': True,
		'VENDOR_CACHE_ENABLED': False,
		'TESTING': True
	}
	app.logger = MagicMock()
	
	return app


@pytest.fixture
def mock_request():
	"""Mock Flask request for API testing"""
	
	request = MagicMock()
	request.headers = {
		'X-Tenant-ID': '00000000-0000-0000-0000-000000000000',
		'X-User-ID': '00000000-0000-0000-0000-000000000000'
	}
	request.args = {}
	request.get_json = MagicMock(return_value={})
	
	return request


# ============================================================================
# PERFORMANCE TEST FIXTURES
# ============================================================================

@pytest.fixture
def performance_test_data():
	"""Generate large dataset for performance testing"""
	
	vendors = []
	for i in range(1000):
		vendors.append({
			'vendor_code': f'PERF{i:04d}',
			'name': f'Performance Test Vendor {i}',
			'vendor_type': 'supplier',
			'category': 'technology',
			'performance_score': 70.0 + (i % 30),
			'risk_score': 10.0 + (i % 40)
		})
	
	return vendors


# ============================================================================
# AI/ML TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_ai_models():
	"""Mock AI/ML models for testing"""
	
	models = {
		'performance_predictor': AsyncMock(),
		'risk_classifier': AsyncMock(),
		'behavior_analyzer': AsyncMock(),
		'optimization_engine': AsyncMock()
	}
	
	# Configure mock responses
	models['performance_predictor'].predict.return_value = {
		'prediction': 87.5,
		'confidence': 0.82
	}
	
	models['risk_classifier'].classify.return_value = {
		'risk_level': 'medium',
		'probability': 0.35,
		'confidence': 0.78
	}
	
	models['behavior_analyzer'].analyze.return_value = [
		{
			'pattern': 'consistent_performance',
			'strength': 0.85
		}
	]
	
	models['optimization_engine'].optimize.return_value = {
		'recommendations': [
			'Increase order frequency',
			'Negotiate better payment terms'
		],
		'expected_improvement': 0.15
	}
	
	return models


# ============================================================================
# SECURITY TEST FIXTURES
# ============================================================================

@pytest.fixture
def security_test_scenarios():
	"""Security test scenarios and payloads"""
	
	return {
		'sql_injection': [
			"'; DROP TABLE vm_vendor; --",
			"1' OR '1'='1",
			"admin'/*",
			" UNION SELECT * FROM vm_vendor--"
		],
		'xss_payloads': [
			"<script>alert('xss')</script>",
			"javascript:alert('xss')",
			"<img src=x onerror=alert('xss')>",
			"<svg onload=alert('xss')>"
		],
		'invalid_uuids': [
			"not-a-uuid",
			"",
			None,
			"12345",
			"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
		],
		'boundary_values': {
			'large_string': 'x' * 10000,
			'negative_numbers': [-1, -100, -999999],
			'extreme_scores': [-50, 150, 99999]
		}
	}


# ============================================================================
# CLEANUP FIXTURES
# ============================================================================

@pytest_asyncio.fixture
async def cleanup_test_data():
	"""Cleanup test data after tests"""
	
	# Setup phase
	yield
	
	# Cleanup phase
	# In a real test environment, this would clean up test database records
	pass