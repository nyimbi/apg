"""
APG Vendor Management - Test Suite
Comprehensive testing framework for AI-powered vendor lifecycle management

Author: Nyimbi Odero (nyimbi@gmail.com)
Copyright: © 2025 Datacraft (www.datacraft.co.ke)
"""

import pytest
import asyncio
import os
from typing import Dict, Any, AsyncGenerator
from uuid import UUID, uuid4

# Test configuration
TEST_CONFIG = {
	'DATABASE_URL': 'postgresql://test:test@localhost/apg_test',
	'REDIS_URL': 'redis://localhost:6379/1',
	'AI_MOCK_MODE': True,
	'TENANT_ID': '00000000-0000-0000-0000-000000000000',
	'USER_ID': '00000000-0000-0000-0000-000000000000'
}

# Test fixtures and utilities
class TestFixtures:
	"""Common test fixtures and utilities"""
	
	@staticmethod
	def get_test_tenant_id() -> UUID:
		return UUID(TEST_CONFIG['TENANT_ID'])
	
	@staticmethod
	def get_test_user_id() -> UUID:
		return UUID(TEST_CONFIG['USER_ID'])
	
	@staticmethod
	def get_test_vendor_data() -> Dict[str, Any]:
		return {
			'vendor_code': 'TEST001',
			'name': 'Test Vendor Inc.',
			'legal_name': 'Test Vendor Incorporated',
			'display_name': 'Test Vendor',
			'vendor_type': 'supplier',
			'category': 'technology',
			'subcategory': 'software',
			'industry': 'information_technology',
			'email': 'contact@testvendor.com',
			'phone': '+1-555-0123',
			'website': 'https://testvendor.com',
			'strategic_importance': 'standard',
			'preferred_vendor': False,
			'strategic_partner': False
		}
	
	@staticmethod
	def get_test_performance_data() -> Dict[str, Any]:
		return {
			'measurement_period': 'quarterly',
			'overall_score': 85.5,
			'quality_score': 90.0,
			'delivery_score': 82.0,
			'cost_score': 88.0,
			'service_score': 85.0,
			'on_time_delivery_rate': 95.0,
			'quality_rejection_rate': 2.5
		}
	
	@staticmethod
	def get_test_risk_data() -> Dict[str, Any]:
		return {
			'risk_type': 'operational',
			'risk_category': 'delivery',
			'severity': 'medium',
			'title': 'Potential delivery delays',
			'description': 'Risk of delivery delays due to supply chain issues',
			'root_cause': 'Dependency on single supplier',
			'potential_impact': 'Project delays and cost overruns',
			'overall_risk_score': 65.0,
			'financial_impact': 50000.0,
			'mitigation_strategy': 'Diversify supplier base'
		}


# Pytest configuration
def pytest_configure(config):
	"""Configure pytest with custom markers"""
	config.addinivalue_line("markers", "unit: Unit tests")
	config.addinivalue_line("markers", "integration: Integration tests")
	config.addinivalue_line("markers", "e2e: End-to-end tests")
	config.addinivalue_line("markers", "performance: Performance tests")
	config.addinivalue_line("markers", "ai: AI/ML specific tests")
	config.addinivalue_line("markers", "security: Security tests")
	config.addinivalue_line("markers", "slow: Slow running tests")


@pytest.fixture(scope="session")
def event_loop():
	"""Create event loop for async tests"""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def test_tenant_id() -> UUID:
	"""Get test tenant ID"""
	return TestFixtures.get_test_tenant_id()


@pytest.fixture
def test_user_id() -> UUID:
	"""Get test user ID"""
	return TestFixtures.get_test_user_id()


@pytest.fixture
def test_vendor_data() -> Dict[str, Any]:
	"""Get test vendor data"""
	return TestFixtures.get_test_vendor_data()


@pytest.fixture
def test_performance_data() -> Dict[str, Any]:
	"""Get test performance data"""
	return TestFixtures.get_test_performance_data()


@pytest.fixture
def test_risk_data() -> Dict[str, Any]:
	"""Get test risk data"""
	return TestFixtures.get_test_risk_data()