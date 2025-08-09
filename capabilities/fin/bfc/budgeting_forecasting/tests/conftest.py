"""
APG Budgeting & Forecasting - Test Configuration

Pytest configuration and shared fixtures for the APG Budgeting & Forecasting capability tests.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import AsyncMock, MagicMock
import logging

# Test framework imports
from pytest_httpserver import HTTPServer
import responses

# Core imports
from ..service import APGTenantContext, BFServiceConfig
from .. import create_budgeting_forecasting_capability
from ..models import BFBudgetType, BFBudgetStatus, BFLineType


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
	"""Configure pytest settings."""
	logging.disable(logging.CRITICAL)  # Disable logging during tests
	
	# Add custom markers
	config.addinivalue_line("markers", "integration: mark test as integration test")
	config.addinivalue_line("markers", "unit: mark test as unit test")
	config.addinivalue_line("markers", "slow: mark test as slow running")
	config.addinivalue_line("markers", "requires_db: mark test as requiring database")
	config.addinivalue_line("markers", "requires_ml: mark test as requiring ML components")
	config.addinivalue_line("markers", "requires_ai: mark test as requiring AI components")


def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add markers automatically."""
	for item in items:
		# Mark integration tests
		if "test_integration" in item.nodeid or "integration" in item.keywords:
			item.add_marker(pytest.mark.integration)
		
		# Mark unit tests
		if "test_unit" in item.nodeid or item.nodeid.startswith("test_"):
			item.add_marker(pytest.mark.unit)
		
		# Mark slow tests
		if "test_advanced_features" in item.nodeid or "performance" in item.keywords:
			item.add_marker(pytest.mark.slow)


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop_policy():
	"""Set event loop policy for the test session."""
	return asyncio.get_event_loop_policy()


@pytest.fixture(scope="function")
def event_loop():
	"""Create a new event loop for each test function."""
	loop = asyncio.new_event_loop()
	yield loop
	loop.close()


# =============================================================================
# Test Database Configuration
# =============================================================================

@pytest.fixture(scope="session")
def test_database_url():
	"""Get test database URL."""
	return os.getenv(
		"TEST_DATABASE_URL",
		"postgresql://test_user:test_pass@localhost:5432/test_apg_budgeting_forecasting"
	)


@pytest.fixture(scope="function")
def test_schema_name():
	"""Generate unique schema name for test isolation."""
	import uuid
	return f"test_schema_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Core Test Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def base_tenant_context():
	"""Create base tenant context for testing."""
	return APGTenantContext(
		tenant_id="test_tenant_base",
		user_id="test_user_base"
	)


@pytest.fixture(scope="function")
def multi_tenant_contexts():
	"""Create multiple tenant contexts for multi-tenant testing."""
	return [
		APGTenantContext(tenant_id="tenant_001", user_id="user_001"),
		APGTenantContext(tenant_id="tenant_002", user_id="user_002"),
		APGTenantContext(tenant_id="tenant_003", user_id="user_003")
	]


@pytest.fixture(scope="function")
def test_service_config(test_database_url, test_schema_name):
	"""Create test service configuration."""
	return BFServiceConfig(
		database_url=test_database_url,
		schema_name=test_schema_name,
		cache_enabled=False,  # Disable cache in tests
		audit_enabled=True,
		ml_enabled=True,
		ai_recommendations_enabled=True,
		real_time_collaboration_enabled=True,
		automated_monitoring_enabled=True,
		debug_mode=True
	)


@pytest.fixture(scope="function")
async def capability_instance(base_tenant_context, test_service_config):
	"""Create APG Budgeting & Forecasting capability instance."""
	capability = create_budgeting_forecasting_capability(base_tenant_context, test_service_config)
	yield capability
	# Cleanup would go here if needed


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def sample_budget_minimal():
	"""Minimal budget data for testing."""
	return {
		"budget_name": "Test Budget Minimal",
		"budget_type": BFBudgetType.ANNUAL.value,
		"fiscal_year": "2025",
		"total_amount": 100000.00,
		"base_currency": "USD"
	}


@pytest.fixture(scope="function")
def sample_budget_comprehensive():
	"""Comprehensive budget data for testing."""
	return {
		"budget_name": "Test Budget Comprehensive",
		"budget_type": BFBudgetType.ANNUAL.value,
		"fiscal_year": "2025",
		"total_amount": 1500000.00,
		"base_currency": "USD",
		"department_id": "dept_test_001",
		"description": "Comprehensive test budget with all features",
		"tags": ["test", "comprehensive", "annual"],
		"metadata": {
			"created_for": "testing",
			"test_scenario": "comprehensive"
		},
		"budget_lines": [
			{
				"line_name": "Personnel Costs",
				"category": "SALARIES",
				"amount": 800000.00,
				"line_type": BFLineType.EXPENSE.value,
				"description": "Employee salaries and wages",
				"allocation_method": "DIRECT"
			},
			{
				"line_name": "Marketing Budget",
				"category": "MARKETING",
				"amount": 300000.00,
				"line_type": BFLineType.EXPENSE.value,
				"description": "Marketing and advertising expenses",
				"allocation_method": "PERCENTAGE"
			},
			{
				"line_name": "Revenue Target",
				"category": "SALES",
				"amount": 2000000.00,
				"line_type": BFLineType.REVENUE.value,
				"description": "Annual revenue target",
				"allocation_method": "DIRECT"
			},
			{
				"line_name": "IT Infrastructure",
				"category": "TECHNOLOGY",
				"amount": 200000.00,
				"line_type": BFLineType.EXPENSE.value,
				"description": "IT infrastructure and software",
				"allocation_method": "HEADCOUNT"
			}
		]
	}


@pytest.fixture(scope="function")
def sample_budget_lines():
	"""Sample budget lines for testing."""
	return [
		{
			"line_name": "Sales Team Salaries",
			"category": "SALARIES",
			"amount": 300000.00,
			"line_type": BFLineType.EXPENSE.value
		},
		{
			"line_name": "Marketing Campaigns",
			"category": "MARKETING", 
			"amount": 150000.00,
			"line_type": BFLineType.EXPENSE.value
		},
		{
			"line_name": "Office Rent",
			"category": "FACILITIES",
			"amount": 120000.00,
			"line_type": BFLineType.EXPENSE.value
		},
		{
			"line_name": "Software Licenses",
			"category": "TECHNOLOGY",
			"amount": 80000.00,
			"line_type": BFLineType.EXPENSE.value
		}
	]


@pytest.fixture(scope="function")
def sample_template_data():
	"""Sample template data for testing."""
	return {
		"template_name": "Annual Department Budget Template",
		"template_category": "department",
		"template_scope": "department",
		"template_complexity": "standard",
		"description": "Standard annual budget template for departments",
		"template_lines": [
			{
				"line_name": "Personnel {{department_name}}",
				"category": "SALARIES",
				"amount_formula": "{{headcount}} * {{avg_salary}}",
				"line_type": BFLineType.EXPENSE.value
			},
			{
				"line_name": "Operating Expenses {{department_name}}",
				"category": "OPERATIONS",
				"amount_formula": "{{personnel_amount}} * 0.3",
				"line_type": BFLineType.EXPENSE.value
			}
		],
		"template_variables": [
			{"name": "department_name", "type": "string", "required": True},
			{"name": "headcount", "type": "integer", "required": True},
			{"name": "avg_salary", "type": "decimal", "required": True}
		]
	}


# =============================================================================
# Mock Service Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def mock_database():
	"""Mock database connection for unit tests."""
	mock_db = AsyncMock()
	mock_db.execute.return_value = AsyncMock()
	mock_db.fetch.return_value = []
	mock_db.fetchrow.return_value = None
	return mock_db


@pytest.fixture(scope="function")
def mock_cache():
	"""Mock cache service for testing."""
	mock_cache = MagicMock()
	mock_cache.get.return_value = None
	mock_cache.set.return_value = True
	mock_cache.delete.return_value = True
	return mock_cache


@pytest.fixture(scope="function")
def mock_ml_service():
	"""Mock ML service for testing."""
	mock_ml = AsyncMock()
	mock_ml.train_model.return_value = {
		"model_id": "mock_model_001",
		"accuracy": 0.85,
		"status": "trained"
	}
	mock_ml.predict.return_value = {
		"predictions": [100000, 105000, 110000],
		"confidence": 0.78
	}
	return mock_ml


@pytest.fixture(scope="function")
def mock_ai_service():
	"""Mock AI service for testing."""
	mock_ai = AsyncMock()
	mock_ai.generate_recommendations.return_value = {
		"recommendations": [
			{
				"recommendation_id": "rec_001",
				"title": "Mock Recommendation",
				"type": "cost_optimization",
				"estimated_impact": -25000
			}
		]
	}
	return mock_ai


# =============================================================================
# External Service Mocks
# =============================================================================

@pytest.fixture(scope="function")
def mock_http_server():
	"""Mock HTTP server for external API testing."""
	server = HTTPServer(host="127.0.0.1", port=0)
	server.start()
	yield server
	server.stop()


@pytest.fixture(scope="function")
def mock_industry_benchmark_api():
	"""Mock industry benchmark API responses."""
	with responses.RequestsMock() as rsps:
		# Mock benchmark API endpoints
		rsps.add(
			responses.GET,
			"https://api.industry-benchmarks.com/v1/metrics",
			json={
				"metrics": [
					{
						"metric_name": "Cost per Employee",
						"industry": "Technology",
						"median_value": 42000,
						"percentile_75": 48000,
						"sample_size": 1250
					}
				]
			},
			status=200
		)
		yield rsps


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def performance_timer():
	"""Timer fixture for performance testing."""
	import time
	
	class Timer:
		def __init__(self):
			self.start_time = None
			self.end_time = None
		
		def start(self):
			self.start_time = time.perf_counter()
		
		def stop(self):
			self.end_time = time.perf_counter()
		
		@property
		def elapsed(self):
			if self.start_time and self.end_time:
				return self.end_time - self.start_time
			return None
	
	return Timer()


@pytest.fixture(scope="function")
def large_dataset_generator():
	"""Generate large datasets for performance testing."""
	def generate_budget_lines(count: int = 1000):
		"""Generate large number of budget lines."""
		lines = []
		categories = ["SALARIES", "MARKETING", "OPERATIONS", "TECHNOLOGY", "FACILITIES"]
		
		for i in range(count):
			lines.append({
				"line_name": f"Budget Line {i:04d}",
				"category": categories[i % len(categories)],
				"amount": 1000.00 + (i * 10),
				"line_type": BFLineType.EXPENSE.value
			})
		
		return lines
	
	return generate_budget_lines


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(scope="function")
def temp_file_cleanup():
	"""Manage temporary file cleanup."""
	temp_files = []
	
	def create_temp_file(suffix=".tmp"):
		temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
		temp_files.append(temp_file.name)
		return temp_file.name
	
	yield create_temp_file
	
	# Cleanup
	for file_path in temp_files:
		try:
			os.unlink(file_path)
		except OSError:
			pass


@pytest.fixture(autouse=True)
def reset_logging():
	"""Reset logging configuration after each test."""
	yield
	# Reset logging to default state
	logging.disable(logging.NOTSET)
	
	# Clear any test-specific loggers
	for name in list(logging.Logger.manager.loggerDict.keys()):
		if name.startswith("test_") or "budgeting_forecasting" in name:
			logger = logging.getLogger(name)
			logger.handlers.clear()
			logger.setLevel(logging.NOTSET)


# =============================================================================
# Parametrized Fixtures
# =============================================================================

@pytest.fixture(params=[
	BFBudgetType.ANNUAL,
	BFBudgetType.QUARTERLY,
	BFBudgetType.MONTHLY
])
def budget_type_param(request):
	"""Parametrized budget type fixture."""
	return request.param


@pytest.fixture(params=["USD", "EUR", "GBP", "CAD"])
def currency_param(request):
	"""Parametrized currency fixture."""
	return request.param


@pytest.fixture(params=[
	{"tenant_count": 1, "user_count": 1},
	{"tenant_count": 3, "user_count": 5},
	{"tenant_count": 10, "user_count": 20}
])
def scale_param(request):
	"""Parametrized scale testing fixture."""
	return request.param


# =============================================================================
# Custom Pytest Helpers
# =============================================================================

def assert_service_response_success(response, expected_data_keys=None):
	"""Helper to assert service response success."""
	assert response is not None
	assert response.success == True
	assert response.message is not None
	assert response.data is not None
	
	if expected_data_keys:
		for key in expected_data_keys:
			assert key in response.data


def assert_service_response_failure(response, expected_error_count=None):
	"""Helper to assert service response failure."""
	assert response is not None
	assert response.success == False
	assert response.errors is not None
	assert len(response.errors) > 0
	
	if expected_error_count:
		assert len(response.errors) == expected_error_count


def create_test_budget_with_lines(capability, budget_data, line_count=3):
	"""Helper to create test budget with specified number of lines."""
	async def _create():
		# Add default budget lines if not provided
		if "budget_lines" not in budget_data and line_count > 0:
			budget_data["budget_lines"] = []
			for i in range(line_count):
				budget_data["budget_lines"].append({
					"line_name": f"Test Line {i+1}",
					"category": "TEST_CATEGORY",
					"amount": 10000.00 * (i + 1),
					"line_type": BFLineType.EXPENSE.value
				})
		
		response = await capability.create_budget(budget_data)
		assert_service_response_success(response, ["budget_id"])
		return response.data["budget_id"]
	
	return _create()


# Make helper functions available to tests
pytest.assert_service_response_success = assert_service_response_success
pytest.assert_service_response_failure = assert_service_response_failure
pytest.create_test_budget_with_lines = create_test_budget_with_lines