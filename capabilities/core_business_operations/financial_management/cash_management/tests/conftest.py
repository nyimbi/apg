"""APG Cash Management - Test Configuration

Comprehensive pytest configuration and fixtures for enterprise testing.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import asyncio
import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Test database setup
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# APG components
from ..cache import CashCacheManager
from ..events import CashEventManager
from ..service import CashManagementService
from ..models import CashAccount, CashFlow, CashPosition
from ..bank_integration import BankAPIConnection
from ..real_time_sync import RealTimeSyncEngine
from ..ai_forecasting import AIForecastingEngine
from ..analytics_dashboard import AnalyticsDashboard
from ..advanced_ml_models import AdvancedMLModelManager
from ..intelligent_optimization import IntelligentCashFlowOptimizer
from ..advanced_risk_analytics import AdvancedRiskAnalyticsEngine

# ============================================================================
# Test Configuration
# ============================================================================

# Test database URL for SQLite in-memory
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
TEST_TENANT_ID = "test_tenant_12345"
TEST_USER_ID = "test_user_67890"

@pytest.fixture(scope="session")
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()

# ============================================================================
# Database Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def async_engine():
	"""Create async database engine for testing."""
	engine = create_async_engine(
		TEST_DATABASE_URL,
		echo=False,  # Set to True for SQL debugging
		future=True
	)
	
	# Create tables (in real implementation, would use migrations)
	# For testing, we'll mock the database layer
	
	yield engine
	await engine.dispose()

@pytest_asyncio.fixture
async def async_session(async_engine):
	"""Create async database session for testing."""
	async_session_factory = sessionmaker(
		async_engine, class_=AsyncSession, expire_on_commit=False
	)
	
	async with async_session_factory() as session:
		yield session

# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_cash_accounts():
	"""Sample cash account data for testing."""
	return [
		{
			'id': 'ACC001',
			'tenant_id': TEST_TENANT_ID,
			'account_number': '123456789',
			'account_type': 'checking',
			'bank_id': 'BANK001',
			'current_balance': Decimal('100000.00'),
			'available_balance': Decimal('95000.00'),
			'currency': 'USD',
			'is_active': True,
			'created_at': datetime.now() - timedelta(days=30)
		},
		{
			'id': 'ACC002',
			'tenant_id': TEST_TENANT_ID,
			'account_number': '987654321',
			'account_type': 'savings',
			'bank_id': 'BANK001',
			'current_balance': Decimal('250000.00'),
			'available_balance': Decimal('250000.00'),
			'currency': 'USD',
			'is_active': True,
			'created_at': datetime.now() - timedelta(days=45)
		},
		{
			'id': 'ACC003',
			'tenant_id': TEST_TENANT_ID,
			'account_number': '555666777',
			'account_type': 'money_market',
			'bank_id': 'BANK002',
			'current_balance': Decimal('500000.00'),
			'available_balance': Decimal('480000.00'),
			'currency': 'USD',
			'is_active': True,
			'created_at': datetime.now() - timedelta(days=60)
		}
	]

@pytest.fixture
def sample_cash_flows():
	"""Sample cash flow data for testing."""
	base_date = datetime.now() - timedelta(days=30)
	flows = []
	
	for i in range(100):  # Generate 100 sample flows
		flow_date = base_date + timedelta(days=i % 30)
		amount = np.random.normal(5000, 2000)  # Random amounts
		
		flows.append({
			'id': f'FLOW{i:03d}',
			'tenant_id': TEST_TENANT_ID,
			'account_id': 'ACC001',
			'amount': Decimal(str(round(amount, 2))),
			'transaction_date': flow_date,
			'description': f'Test transaction {i}',
			'category': 'operating' if i % 2 == 0 else 'investment',
			'is_recurring': i % 10 == 0,
			'confidence_score': 0.85 + (i % 3) * 0.05
		})
	
	return flows

@pytest.fixture
def sample_returns_data():
	"""Sample returns data for risk analytics testing."""
	np.random.seed(42)  # For reproducible tests
	
	# Generate realistic daily returns (250 trading days)
	returns = np.random.normal(0.0008, 0.015, 250)  # ~20% annual vol, positive drift
	
	# Add some clustering (GARCH-like behavior)
	for i in range(1, len(returns)):
		if abs(returns[i-1]) > 0.02:  # If previous day was volatile
			returns[i] *= 1.5  # Increase current day's volatility
	
	return returns

@pytest.fixture
def sample_bank_data():
	"""Sample bank configuration data."""
	return [
		{
			'id': 'BANK001',
			'tenant_id': TEST_TENANT_ID,
			'bank_code': 'CHASE',
			'bank_name': 'JPMorgan Chase Bank',
			'swift_code': 'CHASUS33',
			'api_endpoint': 'https://api.chase.com/v1',
			'api_credentials': {
				'client_id': 'test_client_id',
				'client_secret': 'test_client_secret'
			},
			'is_active': True,
			'last_sync': datetime.now() - timedelta(hours=1)
		},
		{
			'id': 'BANK002',
			'tenant_id': TEST_TENANT_ID,
			'bank_code': 'WELLS_FARGO',
			'bank_name': 'Wells Fargo Bank',
			'swift_code': 'WFBIUS6S',
			'api_endpoint': 'https://api.wellsfargo.com/v1',
			'api_credentials': {
				'client_id': 'test_wf_client_id',
				'client_secret': 'test_wf_client_secret'
			},
			'is_active': True,
			'last_sync': datetime.now() - timedelta(hours=2)
		}
	]

# ============================================================================
# Service Component Fixtures
# ============================================================================

@pytest_asyncio.fixture
async def mock_cache_manager():
	"""Mock cache manager for testing."""
	cache = AsyncMock(spec=CashCacheManager)
	
	# Mock cache operations
	cache.get = AsyncMock(return_value=None)
	cache.set = AsyncMock(return_value=True)
	cache.delete = AsyncMock(return_value=True)
	cache.exists = AsyncMock(return_value=False)
	cache.invalidate_pattern = AsyncMock(return_value=True)
	
	return cache

@pytest_asyncio.fixture
async def mock_event_manager():
	"""Mock event manager for testing."""
	events = AsyncMock(spec=CashEventManager)
	
	# Mock event operations
	events.emit_cash_flow_created = AsyncMock()
	events.emit_balance_updated = AsyncMock()
	events.emit_forecast_generated = AsyncMock()
	events.emit_optimization_completed = AsyncMock()
	events.emit_risk_alert = AsyncMock()
	events.emit_sync_completed = AsyncMock()
	
	return events

@pytest_asyncio.fixture
async def cash_service(mock_cache_manager, mock_event_manager):
	"""Cash management service instance for testing."""
	service = CashManagementService(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	# Mock database operations
	service.db = AsyncMock()
	
	return service

@pytest_asyncio.fixture
async def bank_integration(mock_cache_manager, mock_event_manager):
	"""Bank integration service for testing."""
	integration = BankAPIConnection(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	# Mock HTTP client
	integration.session = AsyncMock()
	
	return integration

@pytest_asyncio.fixture
async def sync_engine(bank_integration, mock_cache_manager, mock_event_manager):
	"""Real-time sync engine for testing."""
	sync = RealTimeSyncEngine(
		tenant_id=TEST_TENANT_ID,
		bank_integration=bank_integration,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	return sync

@pytest_asyncio.fixture
async def ai_forecasting(mock_cache_manager, mock_event_manager):
	"""AI forecasting engine for testing."""
	forecasting = AIForecastingEngine(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	return forecasting

@pytest_asyncio.fixture
async def ml_manager(mock_cache_manager, mock_event_manager):
	"""ML model manager for testing."""
	ml_manager = AdvancedMLModelManager(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	return ml_manager

@pytest_asyncio.fixture
async def optimization_engine(mock_cache_manager, mock_event_manager, ml_manager):
	"""Optimization engine for testing."""
	optimizer = IntelligentCashFlowOptimizer(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager,
		ml_manager=ml_manager
	)
	
	return optimizer

@pytest_asyncio.fixture
async def risk_analytics(mock_cache_manager, mock_event_manager):
	"""Risk analytics engine for testing."""
	risk = AdvancedRiskAnalyticsEngine(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager
	)
	
	return risk

@pytest_asyncio.fixture
async def analytics_dashboard(mock_cache_manager, mock_event_manager, ai_forecasting):
	"""Analytics dashboard for testing."""
	dashboard = AnalyticsDashboard(
		tenant_id=TEST_TENANT_ID,
		cache_manager=mock_cache_manager,
		event_manager=mock_event_manager,
		ai_forecasting=ai_forecasting
	)
	
	return dashboard

# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def portfolio_data_generator():
	"""Generator for portfolio test data."""
	def generate_portfolio(num_accounts: int = 5, total_value: float = 1000000.0):
		portfolio = {}
		
		for i in range(num_accounts):
			account_id = f"ACC{i:03d}"
			account_type = ['checking', 'savings', 'money_market', 'investment'][i % 4]
			balance = total_value / num_accounts * (0.8 + 0.4 * np.random.random())
			
			portfolio[account_id] = {
				'balance': balance,
				'type': account_type,
				'liquidity_score': 1.0 if account_type == 'checking' else 0.8,
				'risk_score': 0.05 if account_type in ['checking', 'savings'] else 0.15
			}
		
		return portfolio
	
	return generate_portfolio

@pytest.fixture
def time_series_generator():
	"""Generator for time series test data."""
	def generate_time_series(
		length: int = 100,
		start_date: datetime = None,
		frequency: str = 'D',
		trend: float = 0.0,
		seasonality: bool = True,
		noise_level: float = 0.1
	):
		if start_date is None:
			start_date = datetime.now() - timedelta(days=length)
		
		dates = pd.date_range(start=start_date, periods=length, freq=frequency)
		
		# Generate base time series
		t = np.arange(length)
		values = trend * t
		
		# Add seasonality
		if seasonality:
			values += 0.1 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
			values += 0.05 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
		
		# Add noise
		values += np.random.normal(0, noise_level, length)
		
		return pd.DataFrame({
			'date': dates,
			'value': values
		})
	
	return generate_time_series

# ============================================================================
# Mock External Services
# ============================================================================

@pytest.fixture
def mock_bank_api_responses():
	"""Mock responses from bank APIs."""
	return {
		'account_balance': {
			'account_id': 'ACC001',
			'balance': 100000.00,
			'available_balance': 95000.00,
			'currency': 'USD',
			'last_updated': datetime.now().isoformat()
		},
		'transactions': [
			{
				'transaction_id': 'TXN001',
				'amount': 5000.00,
				'description': 'Wire transfer',
				'transaction_date': datetime.now().isoformat(),
				'status': 'completed'
			},
			{
				'transaction_id': 'TXN002',
				'amount': -2500.00,
				'description': 'ACH payment',
				'transaction_date': (datetime.now() - timedelta(days=1)).isoformat(),
				'status': 'completed'
			}
		],
		'account_details': {
			'account_number': '123456789',
			'account_type': 'checking',
			'bank_name': 'Test Bank',
			'routing_number': '123456789'
		}
	}

@pytest.fixture
def mock_market_data():
	"""Mock market data for testing."""
	return {
		'interest_rates': {
			'fed_funds_rate': 0.025,
			'treasury_1m': 0.02,
			'treasury_3m': 0.021,
			'treasury_6m': 0.022,
			'treasury_1y': 0.024
		},
		'fx_rates': {
			'USD/EUR': 0.85,
			'USD/GBP': 0.75,
			'USD/JPY': 110.0
		},
		'volatility_indices': {
			'vix': 18.5,
			'move': 85.2
		}
	}

# ============================================================================
# Performance Test Fixtures
# ============================================================================

@pytest.fixture
def performance_test_data():
	"""Large dataset for performance testing."""
	return {
		'large_cash_flows': [
			{
				'id': f'FLOW{i:06d}',
				'amount': Decimal(str(round(np.random.normal(1000, 500), 2))),
				'date': datetime.now() - timedelta(days=np.random.randint(0, 365))
			}
			for i in range(10000)  # 10K records
		],
		'large_portfolio': {
			f'ACC{i:04d}': {
				'balance': np.random.exponential(50000),
				'type': ['checking', 'savings', 'investment'][i % 3]
			}
			for i in range(1000)  # 1K accounts
		}
	}

# ============================================================================
# Test Utilities
# ============================================================================

@pytest.fixture
def assert_helpers():
	"""Helper functions for test assertions."""
	class AssertHelpers:
		@staticmethod
		def assert_decimal_equal(actual: Decimal, expected: Decimal, places: int = 2):
			"""Assert decimal equality within specified decimal places."""
			assert abs(actual - expected) < Decimal('0.1') ** places
		
		@staticmethod
		def assert_datetime_close(actual: datetime, expected: datetime, delta_seconds: int = 60):
			"""Assert datetime values are close within delta."""
			assert abs((actual - expected).total_seconds()) <= delta_seconds
		
		@staticmethod
		def assert_percentage_close(actual: float, expected: float, tolerance_pct: float = 1.0):
			"""Assert percentage values are close within tolerance."""
			if expected == 0:
				assert abs(actual) <= tolerance_pct / 100
			else:
				assert abs((actual - expected) / expected) <= tolerance_pct / 100
		
		@staticmethod
		def assert_risk_metrics_valid(metrics: Dict[str, Any]):
			"""Assert risk metrics are valid."""
			# VaR should be positive
			var_data = metrics.get('value_at_risk', {})
			for var_key, var_results in var_data.items():
				for method, result in var_results.items():
					if isinstance(result, dict) and 'value' in result:
						assert result['value'] >= 0, f"VaR should be positive: {var_key}.{method}"
			
			# Volatility should be positive
			perf_ratios = metrics.get('performance_ratios', {})
			if 'annual_volatility' in perf_ratios:
				assert perf_ratios['annual_volatility'] >= 0, "Volatility should be positive"
		
		@staticmethod
		def assert_optimization_result_valid(result):
			"""Assert optimization result is valid."""
			assert result.success is not None
			assert isinstance(result.objective_value, (int, float))
			assert isinstance(result.optimal_solution, dict)
			assert result.confidence_score >= 0.0
			assert result.confidence_score <= 1.0
	
	return AssertHelpers()

# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
async def cleanup_after_test():
	"""Cleanup after each test."""
	yield
	
	# Clear any global state
	# Reset random seeds for deterministic tests
	np.random.seed(42)
	
	# Clear any cached data (if using real cache)
	# await cache.clear_all()

# ============================================================================
# Markers and Test Configuration
# ============================================================================

# Custom pytest markers
pytest_plugins = []

def pytest_configure(config):
	"""Configure pytest with custom markers."""
	config.addinivalue_line("markers", "unit: mark test as unit test")
	config.addinivalue_line("markers", "integration: mark test as integration test")
	config.addinivalue_line("markers", "performance: mark test as performance test")
	config.addinivalue_line("markers", "slow: mark test as slow running")
	config.addinivalue_line("markers", "ml: mark test as machine learning test")
	config.addinivalue_line("markers", "risk: mark test as risk analytics test")
	config.addinivalue_line("markers", "optimization: mark test as optimization test")

# Test timeouts
def pytest_collection_modifyitems(config, items):
	"""Modify test collection to add timeouts."""
	for item in items:
		# Add timeout for slow tests
		if item.get_closest_marker("slow"):
			item.add_marker(pytest.mark.timeout(300))  # 5 minutes
		elif item.get_closest_marker("performance"):
			item.add_marker(pytest.mark.timeout(600))  # 10 minutes
		else:
			item.add_marker(pytest.mark.timeout(60))   # 1 minute default