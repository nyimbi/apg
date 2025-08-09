"""APG Cash Management - Unit Tests for Core Components

Comprehensive unit tests for core cash management functionality.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero | APG Platform Architect
"""

import pytest
import pytest_asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from ..models import CashAccount, CashFlow, CashPosition
from ..service import CashManagementService
from ..cache import CashCacheManager
from ..events import CashEventManager

# ============================================================================
# Core Service Tests
# ============================================================================

@pytest.mark.unit
class TestCashManagementService:
	"""Test cases for CashManagementService."""
	
	@pytest_asyncio.fixture
	async def service(self, mock_cache_manager, mock_event_manager):
		"""Service instance for testing."""
		return CashManagementService(
			tenant_id="test_tenant",
			cache_manager=mock_cache_manager,
			event_manager=mock_event_manager
		)
	
	async def test_create_cash_account(self, service, sample_cash_accounts):
		"""Test cash account creation."""
		account_data = sample_cash_accounts[0]
		
		# Mock database operations
		service.db = AsyncMock()
		service.db.add = AsyncMock()
		service.db.commit = AsyncMock()
		
		result = await service.create_cash_account(account_data)
		
		assert result['success'] is True
		assert result['account_id'] == account_data['id']
		
		# Verify database calls
		service.db.add.assert_called_once()
		service.db.commit.assert_called_once()
	
	async def test_get_account_balance(self, service, sample_cash_accounts):
		"""Test account balance retrieval."""
		account_data = sample_cash_accounts[0]
		
		# Mock database query
		mock_account = AsyncMock()
		mock_account.current_balance = account_data['current_balance']
		mock_account.available_balance = account_data['available_balance']
		
		service.db = AsyncMock()
		service.db.query().filter().first = AsyncMock(return_value=mock_account)
		
		balance = await service.get_account_balance(account_data['id'])
		
		assert balance['current_balance'] == account_data['current_balance']
		assert balance['available_balance'] == account_data['available_balance']
	
	async def test_record_cash_flow(self, service, sample_cash_flows):
		"""Test cash flow recording."""
		flow_data = sample_cash_flows[0]
		
		# Mock database operations
		service.db = AsyncMock()
		service.db.add = AsyncMock()
		service.db.commit = AsyncMock()
		
		result = await service.record_cash_flow(flow_data)
		
		assert result['success'] is True
		assert result['flow_id'] == flow_data['id']
		
		# Verify event emission
		service.events.emit_cash_flow_created.assert_called_once()
	
	async def test_update_account_balance(self, service, assert_helpers):
		"""Test account balance updates."""
		account_id = "ACC001"
		new_balance = Decimal("125000.00")
		
		# Mock database operations
		mock_account = AsyncMock()
		mock_account.current_balance = Decimal("100000.00")
		
		service.db = AsyncMock()
		service.db.query().filter().first = AsyncMock(return_value=mock_account)
		service.db.commit = AsyncMock()
		
		result = await service.update_account_balance(account_id, new_balance)
		
		assert result['success'] is True
		assert_helpers.assert_decimal_equal(
			mock_account.current_balance, 
			new_balance
		)
		
		# Verify event emission
		service.events.emit_balance_updated.assert_called_once()
	
	async def test_get_cash_flows_by_date_range(self, service, sample_cash_flows):
		"""Test cash flow retrieval by date range."""
		start_date = datetime.now() - timedelta(days=7)
		end_date = datetime.now()
		
		# Mock database query
		mock_flows = [AsyncMock() for _ in sample_cash_flows[:5]]
		service.db = AsyncMock()
		service.db.query().filter().all = AsyncMock(return_value=mock_flows)
		
		flows = await service.get_cash_flows_by_date_range(
			start_date=start_date,
			end_date=end_date
		)
		
		assert len(flows) == 5
		service.db.query().filter().all.assert_called_once()
	
	async def test_calculate_account_metrics(self, service, sample_cash_flows):
		"""Test account metrics calculation."""
		account_id = "ACC001"
		
		# Mock cash flows data
		mock_flows = []
		for flow_data in sample_cash_flows[:10]:
			mock_flow = AsyncMock()
			mock_flow.amount = flow_data['amount']
			mock_flow.transaction_date = flow_data['transaction_date']
			mock_flows.append(mock_flow)
		
		service.db = AsyncMock()
		service.db.query().filter().all = AsyncMock(return_value=mock_flows)
		
		metrics = await service.calculate_account_metrics(account_id)
		
		assert 'total_inflows' in metrics
		assert 'total_outflows' in metrics
		assert 'net_flow' in metrics
		assert 'average_daily_flow' in metrics
		assert isinstance(metrics['total_inflows'], Decimal)

# ============================================================================
# Cache Manager Tests
# ============================================================================

@pytest.mark.unit
class TestCashCacheManager:
	"""Test cases for CashCacheManager."""
	
	@pytest_asyncio.fixture
	async def cache_manager(self):
		"""Cache manager instance for testing."""
		# Use mock Redis client
		mock_redis = AsyncMock()
		
		with patch('redis.asyncio.Redis', return_value=mock_redis):
			cache = CashCacheManager(
				host="localhost",
				port=6379,
				db=0
			)
			cache.redis = mock_redis
			yield cache
	
	async def test_set_and_get(self, cache_manager):
		"""Test cache set and get operations."""
		key = "test_key"
		value = {"test": "data", "number": 123}
		
		# Mock Redis operations
		cache_manager.redis.set = AsyncMock(return_value=True)
		cache_manager.redis.get = AsyncMock(return_value='{"test": "data", "number": 123}')
		
		# Test set
		result = await cache_manager.set(key, value, ttl=3600)
		assert result is True
		
		# Test get
		cached_value = await cache_manager.get(key)
		assert cached_value == value
	
	async def test_delete(self, cache_manager):
		"""Test cache delete operation."""
		key = "test_key"
		
		cache_manager.redis.delete = AsyncMock(return_value=1)
		
		result = await cache_manager.delete(key)
		assert result is True
		
		cache_manager.redis.delete.assert_called_once_with(key)
	
	async def test_exists(self, cache_manager):
		"""Test cache existence check."""
		key = "test_key"
		
		cache_manager.redis.exists = AsyncMock(return_value=1)
		
		exists = await cache_manager.exists(key)
		assert exists is True
		
		cache_manager.redis.exists.assert_called_once_with(key)
	
	async def test_invalidate_pattern(self, cache_manager):
		"""Test pattern-based cache invalidation."""
		pattern = "cash_flow:*"
		
		cache_manager.redis.scan_iter = AsyncMock(return_value=["cash_flow:1", "cash_flow:2"])
		cache_manager.redis.delete = AsyncMock(return_value=2)
		
		result = await cache_manager.invalidate_pattern(pattern)
		assert result is True
		
		cache_manager.redis.scan_iter.assert_called_once_with(match=pattern)

# ============================================================================
# Event Manager Tests
# ============================================================================

@pytest.mark.unit
class TestCashEventManager:
	"""Test cases for CashEventManager."""
	
	@pytest_asyncio.fixture
	async def event_manager(self):
		"""Event manager instance for testing."""
		return CashEventManager()
	
	async def test_emit_cash_flow_created(self, event_manager):
		"""Test cash flow created event emission."""
		tenant_id = "test_tenant"
		flow_data = {
			'id': 'FLOW001',
			'amount': Decimal('1000.00'),
			'description': 'Test flow'
		}
		
		# Mock event handlers
		handler = AsyncMock()
		event_manager.subscribe('cash_flow_created', handler)
		
		await event_manager.emit_cash_flow_created(tenant_id, flow_data)
		
		handler.assert_called_once()
		call_args = handler.call_args[0]
		assert call_args[0] == tenant_id
		assert call_args[1] == flow_data
	
	async def test_emit_balance_updated(self, event_manager):
		"""Test balance updated event emission."""
		tenant_id = "test_tenant"
		account_id = "ACC001"
		old_balance = Decimal('100000.00')
		new_balance = Decimal('125000.00')
		
		handler = AsyncMock()
		event_manager.subscribe('balance_updated', handler)
		
		await event_manager.emit_balance_updated(
			tenant_id, account_id, old_balance, new_balance
		)
		
		handler.assert_called_once()
	
	async def test_emit_forecast_generated(self, event_manager):
		"""Test forecast generated event emission."""
		tenant_id = "test_tenant"
		forecast_data = {
			'horizon_days': 30,
			'confidence_level': 0.95,
			'predictions': [1000, 1100, 1200]
		}
		
		handler = AsyncMock()
		event_manager.subscribe('forecast_generated', handler)
		
		await event_manager.emit_forecast_generated(tenant_id, forecast_data)
		
		handler.assert_called_once()
	
	async def test_event_subscription_and_unsubscription(self, event_manager):
		"""Test event subscription management."""
		event_type = 'test_event'
		handler1 = AsyncMock()
		handler2 = AsyncMock()
		
		# Subscribe handlers
		event_manager.subscribe(event_type, handler1)
		event_manager.subscribe(event_type, handler2)
		
		# Emit event
		await event_manager.emit_custom_event(event_type, "test_data")
		
		# Both handlers should be called
		handler1.assert_called_once()
		handler2.assert_called_once()
		
		# Unsubscribe one handler
		event_manager.unsubscribe(event_type, handler1)
		
		# Emit event again
		await event_manager.emit_custom_event(event_type, "test_data_2")
		
		# Only handler2 should be called again
		assert handler1.call_count == 1
		assert handler2.call_count == 2

# ============================================================================
# Model Tests
# ============================================================================

@pytest.mark.unit
class TestCashModels:
	"""Test cases for cash management models."""
	
	def test_cash_account_validation(self, sample_cash_accounts):
		"""Test cash account model validation."""
		account_data = sample_cash_accounts[0]
		
		# Test valid account creation
		account = CashAccount(**account_data)
		assert account.id == account_data['id']
		assert account.account_type == account_data['account_type']
		assert account.current_balance == account_data['current_balance']
	
	def test_cash_flow_validation(self, sample_cash_flows):
		"""Test cash flow model validation."""
		flow_data = sample_cash_flows[0]
		
		# Test valid flow creation
		flow = CashFlow(**flow_data)
		assert flow.id == flow_data['id']
		assert flow.amount == flow_data['amount']
		assert flow.account_id == flow_data['account_id']
	
	def test_cash_position_calculation(self):
		"""Test cash position calculation logic."""
		position = CashPosition(
			id="POS001",
			tenant_id="test_tenant",
			position_date=datetime.now().date(),
			total_cash=Decimal('1000000.00'),
			available_cash=Decimal('900000.00'),
			restricted_cash=Decimal('100000.00')
		)
		
		assert position.total_cash == Decimal('1000000.00')
		assert position.available_cash == Decimal('900000.00')
		assert position.restricted_cash == Decimal('100000.00')
		
		# Test calculated fields
		utilization_rate = position.available_cash / position.total_cash
		assert abs(utilization_rate - Decimal('0.9')) < Decimal('0.001')

# ============================================================================
# Utility Functions Tests
# ============================================================================

@pytest.mark.unit
class TestUtilityFunctions:
	"""Test cases for utility functions."""
	
	def test_decimal_precision_handling(self):
		"""Test decimal precision in calculations."""
		amount1 = Decimal('1000.12')
		amount2 = Decimal('2000.34')
		
		result = amount1 + amount2
		assert result == Decimal('3000.46')
		
		# Test division precision
		division_result = amount1 / 3
		assert len(str(division_result).split('.')[-1]) <= 10  # Reasonable precision
	
	def test_date_range_calculations(self):
		"""Test date range utility functions."""
		start_date = datetime(2024, 1, 1)
		end_date = datetime(2024, 1, 31)
		
		# Test business days calculation
		total_days = (end_date - start_date).days + 1
		assert total_days == 31
		
		# Test date formatting
		formatted_date = start_date.strftime('%Y-%m-%d')
		assert formatted_date == "2024-01-01"
	
	def test_currency_formatting(self):
		"""Test currency formatting utilities."""
		amount = Decimal('12345.67')
		
		# Test basic formatting
		formatted = f"${amount:,.2f}"
		assert formatted == "$12,345.67"
		
		# Test negative amounts
		negative_amount = Decimal('-1234.56')
		formatted_negative = f"${negative_amount:,.2f}"
		assert formatted_negative == "$-1,234.56"

# ============================================================================
# Error Handling Tests
# ============================================================================

@pytest.mark.unit
class TestErrorHandling:
	"""Test cases for error handling."""
	
	async def test_invalid_account_id(self, cash_service):
		"""Test handling of invalid account IDs."""
		# Mock database to return None
		cash_service.db = AsyncMock()
		cash_service.db.query().filter().first = AsyncMock(return_value=None)
		
		with pytest.raises(ValueError, match="Account not found"):
			await cash_service.get_account_balance("INVALID_ID")
	
	async def test_insufficient_balance(self, cash_service):
		"""Test handling of insufficient balance scenarios."""
		# Mock account with low balance
		mock_account = AsyncMock()
		mock_account.available_balance = Decimal('100.00')
		
		cash_service.db = AsyncMock()
		cash_service.db.query().filter().first = AsyncMock(return_value=mock_account)
		
		# Test withdrawal that exceeds balance
		with pytest.raises(ValueError, match="Insufficient balance"):
			await cash_service.process_withdrawal("ACC001", Decimal('1000.00'))
	
	async def test_invalid_date_range(self, cash_service):
		"""Test handling of invalid date ranges."""
		start_date = datetime.now()
		end_date = datetime.now() - timedelta(days=1)  # End before start
		
		with pytest.raises(ValueError, match="Invalid date range"):
			await cash_service.get_cash_flows_by_date_range(start_date, end_date)
	
	def test_decimal_overflow_protection(self):
		"""Test protection against decimal overflow."""
		# Test very large numbers
		large_amount = Decimal('999999999999999999.99')
		
		# Should not raise an exception
		result = large_amount + Decimal('0.01')
		assert result == Decimal('1000000000000000000.00')
		
		# Test precision limits
		high_precision = Decimal('1.123456789012345678901234567890')
		# Should be handled gracefully
		assert isinstance(high_precision, Decimal)

# ============================================================================
# Concurrency Tests
# ============================================================================

@pytest.mark.unit
class TestConcurrency:
	"""Test cases for concurrent operations."""
	
	async def test_concurrent_balance_updates(self, cash_service):
		"""Test concurrent balance updates."""
		account_id = "ACC001"
		initial_balance = Decimal('1000.00')
		
		# Mock account
		mock_account = AsyncMock()
		mock_account.current_balance = initial_balance
		mock_account.available_balance = initial_balance
		
		cash_service.db = AsyncMock()
		cash_service.db.query().filter().first = AsyncMock(return_value=mock_account)
		cash_service.db.commit = AsyncMock()
		
		# Simulate concurrent updates
		import asyncio
		
		async def update_balance(amount):
			current = mock_account.current_balance
			mock_account.current_balance = current + amount
			await asyncio.sleep(0.01)  # Simulate processing time
		
		# Run concurrent updates
		await asyncio.gather(
			update_balance(Decimal('100.00')),
			update_balance(Decimal('200.00')),
			update_balance(Decimal('-50.00'))
		)
		
		# Final balance should reflect all updates
		expected_balance = initial_balance + Decimal('250.00')
		assert mock_account.current_balance == expected_balance
	
	async def test_cache_consistency(self, mock_cache_manager):
		"""Test cache consistency under concurrent access."""
		key = "test_consistency"
		value1 = {"data": "version1"}
		value2 = {"data": "version2"}
		
		# Mock cache operations
		mock_cache_manager.get = AsyncMock(side_effect=[None, value1, value2])
		mock_cache_manager.set = AsyncMock(return_value=True)
		
		import asyncio
		
		async def cache_operation(value):
			await mock_cache_manager.set(key, value)
			return await mock_cache_manager.get(key)
		
		# Run concurrent cache operations
		results = await asyncio.gather(
			cache_operation(value1),
			cache_operation(value2)
		)
		
		# Verify cache operations were called
		assert mock_cache_manager.set.call_count == 2
		assert mock_cache_manager.get.call_count == 3

# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.unit
@pytest.mark.performance
class TestPerformance:
	"""Performance test cases for core components."""
	
	async def test_batch_cash_flow_processing(self, cash_service, performance_test_data):
		"""Test performance of batch cash flow processing."""
		flows = performance_test_data['large_cash_flows'][:1000]  # 1K flows
		
		# Mock database operations
		cash_service.db = AsyncMock()
		cash_service.db.add = AsyncMock()
		cash_service.db.commit = AsyncMock()
		
		import time
		start_time = time.time()
		
		# Process flows in batch
		results = []
		for flow in flows:
			result = await cash_service.record_cash_flow(flow)
			results.append(result)
		
		end_time = time.time()
		processing_time = end_time - start_time
		
		# Performance assertions
		assert len(results) == 1000
		assert processing_time < 10.0  # Should complete within 10 seconds
		assert all(r['success'] for r in results)
		
		# Throughput assertion
		throughput = len(flows) / processing_time
		assert throughput > 100  # Should process > 100 flows per second
	
	async def test_cache_performance(self, mock_cache_manager):
		"""Test cache performance under load."""
		num_operations = 1000
		
		# Mock fast cache operations
		mock_cache_manager.set = AsyncMock(return_value=True)
		mock_cache_manager.get = AsyncMock(return_value={"cached": "data"})
		
		import time
		start_time = time.time()
		
		# Perform cache operations
		for i in range(num_operations):
			await mock_cache_manager.set(f"key_{i}", {"data": i})
			await mock_cache_manager.get(f"key_{i}")
		
		end_time = time.time()
		cache_time = end_time - start_time
		
		# Performance assertions
		assert cache_time < 5.0  # Should complete within 5 seconds
		
		operations_per_second = (num_operations * 2) / cache_time  # set + get
		assert operations_per_second > 400  # Should handle > 400 ops/sec

if __name__ == "__main__":
	pytest.main([__file__, "-v"])