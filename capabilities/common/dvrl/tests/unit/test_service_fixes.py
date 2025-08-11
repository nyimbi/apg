#!/usr/bin/env python3
"""
Unit Tests for DVRL Service Fixes and Improvements
Tests for fixed null returns, improved error handling, and timing implementations

Author: APG Platform Team  
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from capabilities.common.dvrl.service import DVRLService
from capabilities.common.dvrl.error_handling import (
    ServiceUnavailableError, OperationError, RegistrationError
)


class TestServiceErrorHandlingFixes:
    """Test suite for fixed service error handling"""
    
    @pytest.fixture
    async def dvrl_service(self):
        """Create DVRL service instance for testing"""
        config = {
            'tenant_id': 'test_tenant',
            'user_id': 'test_user',
            'enable_cache': True
        }
        service = DVRLService(config)
        await service.initialize()
        return service
    
    async def test_get_available_singer_taps_without_manager(self, dvrl_service):
        """Test get_available_singer_taps raises error when manager unavailable"""
        # Ensure singer manager is not available
        dvrl_service.singer_tap_manager = None
        
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await dvrl_service.get_available_singer_taps()
        
        assert "Singer.io integration is not configured or available" in str(exc_info.value)
        assert exc_info.value.error_code == "ServiceUnavailableError"
    
    async def test_get_available_singer_taps_with_manager_success(self, dvrl_service):
        """Test get_available_singer_taps with working manager"""
        # Mock singer manager
        mock_manager = AsyncMock()
        mock_manager.get_available_taps.return_value = {
            'tap-postgres': {'description': 'PostgreSQL tap'},
            'tap-mysql': {'description': 'MySQL tap'}
        }
        dvrl_service.singer_tap_manager = mock_manager
        
        result = await dvrl_service.get_available_singer_taps()
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'tap-postgres' in result
        assert 'tap-mysql' in result
        mock_manager.get_available_taps.assert_called_once()
    
    async def test_get_available_singer_taps_manager_returns_none(self, dvrl_service):
        """Test get_available_singer_taps when manager returns None"""
        # Mock singer manager that returns None
        mock_manager = AsyncMock()
        mock_manager.get_available_taps.return_value = None
        dvrl_service.singer_tap_manager = mock_manager
        
        result = await dvrl_service.get_available_singer_taps()
        
        # Should return empty dict instead of None
        assert result == {}
        assert isinstance(result, dict)
    
    async def test_get_available_singer_taps_manager_exception(self, dvrl_service):
        """Test get_available_singer_taps when manager raises exception"""
        # Mock singer manager that raises exception
        mock_manager = AsyncMock()
        mock_manager.get_available_taps.side_effect = Exception("Connection timeout")
        dvrl_service.singer_tap_manager = mock_manager
        
        with pytest.raises(OperationError) as exc_info:
            await dvrl_service.get_available_singer_taps()
        
        assert "Failed to retrieve available Singer taps" in str(exc_info.value)
        assert "Connection timeout" in str(exc_info.value)
    
    async def test_register_singer_tap_data_source_without_manager(self, dvrl_service):
        """Test register_singer_tap_data_source raises error when manager unavailable"""
        dvrl_service.singer_tap_manager = None
        
        with pytest.raises(ServiceUnavailableError) as exc_info:
            await dvrl_service.register_singer_tap_data_source(
                'tap-postgres', 
                {'host': 'localhost'}
            )
        
        assert "Singer.io integration is not configured or available" in str(exc_info.value)
    
    async def test_register_singer_tap_data_source_success(self, dvrl_service):
        """Test successful singer tap data source registration"""
        # Mock singer manager and dependencies
        mock_manager = AsyncMock()
        mock_connector = AsyncMock()
        mock_manager.create_tap_connector.return_value = mock_connector
        dvrl_service.singer_tap_manager = mock_manager
        
        # Mock register_data_source method
        mock_data_source = Mock()
        mock_data_source.name = 'singer_tap-postgres'
        mock_data_source.id = 'test_id'
        
        with patch.object(dvrl_service, 'register_data_source', return_value=mock_data_source):
            result = await dvrl_service.register_singer_tap_data_source(
                'tap-postgres',
                {'host': 'localhost', 'port': 5432}
            )
        
        assert result is not None
        assert result.name == 'singer_tap-postgres'
        mock_manager.create_tap_connector.assert_called_once_with(
            'tap-postgres', 
            {'host': 'localhost', 'port': 5432}
        )
    
    async def test_register_singer_tap_data_source_connector_creation_fails(self, dvrl_service):
        """Test register_singer_tap_data_source when connector creation fails"""
        mock_manager = AsyncMock()
        mock_manager.create_tap_connector.return_value = None  # Creation failed
        dvrl_service.singer_tap_manager = mock_manager
        
        with pytest.raises(RegistrationError) as exc_info:
            await dvrl_service.register_singer_tap_data_source(
                'tap-postgres',
                {'host': 'localhost'}
            )
        
        assert "Failed to register Singer tap 'tap-postgres'" in str(exc_info.value)
    
    async def test_register_singer_tap_data_source_exception(self, dvrl_service):
        """Test register_singer_tap_data_source with exception during registration"""
        mock_manager = AsyncMock()
        mock_connector = AsyncMock()
        mock_manager.create_tap_connector.return_value = mock_connector
        dvrl_service.singer_tap_manager = mock_manager
        
        # Mock register_data_source to raise exception
        with patch.object(dvrl_service, 'register_data_source', side_effect=Exception("Database error")):
            with pytest.raises(RegistrationError) as exc_info:
                await dvrl_service.register_singer_tap_data_source(
                    'tap-postgres',
                    {'host': 'localhost'}
                )
        
        assert "Failed to register Singer tap 'tap-postgres'" in str(exc_info.value)
        assert "Database error" in str(exc_info.value)


class TestCacheCheckImprovements:
    """Test suite for improved cache check functionality"""
    
    @pytest.fixture
    async def dvrl_service(self):
        """Create DVRL service with cache service for testing"""
        config = {
            'tenant_id': 'test_tenant', 
            'user_id': 'test_user',
            'enable_cache': True
        }
        service = DVRLService(config)
        await service.initialize()
        return service
    
    async def test_check_query_cache_apg_cache_hit(self, dvrl_service):
        """Test cache check with APG cache service hit"""
        query_hash = "test_query_hash"
        cached_data = {
            'result': {'data': [{'id': 1, 'name': 'test'}]},
            'metadata': {'rows': 1},
            'query_id': 'cached_query_123'
        }
        
        # Mock APG cache service
        dvrl_service.cache_service = AsyncMock()
        dvrl_service.cache_service.get.return_value = cached_data
        
        result = await dvrl_service._check_query_cache(query_hash)
        
        assert result is not None
        assert result['cached'] is True
        assert result['result'] == cached_data['result'] 
        assert result['metadata'] == cached_data['metadata']
        assert result['query_id'] == cached_data['query_id']
        assert result['cache_source'] == 'apg_cache_service'
        
        dvrl_service.cache_service.get.assert_called_once_with(f"dvrl_cache_{query_hash}")
    
    async def test_check_query_cache_local_cache_hit(self, dvrl_service):
        """Test cache check with local cache hit"""
        query_hash = "test_query_hash"
        
        # Setup local cache entry
        from datetime import timedelta
        cache_entry = Mock()
        cache_entry.result = {'data': [{'id': 2, 'name': 'local'}]}
        cache_entry.metadata = {'rows': 1}
        cache_entry.query_id = 'local_query_456'
        cache_entry.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)  # Not expired
        cache_entry.hit_count = 0
        cache_entry.last_accessed = datetime.now(timezone.utc)
        
        dvrl_service.query_cache[query_hash] = cache_entry
        
        # Mock APG cache service to return None (cache miss)
        dvrl_service.cache_service = AsyncMock()
        dvrl_service.cache_service.get.return_value = None
        
        result = await dvrl_service._check_query_cache(query_hash)
        
        assert result is not None
        assert result['cached'] is True
        assert result['result'] == cache_entry.result
        assert result['metadata'] == cache_entry.metadata
        assert result['query_id'] == cache_entry.query_id
        assert result['cache_source'] == 'local_cache'
        
        # Check that hit count was incremented
        assert cache_entry.hit_count == 1
    
    async def test_check_query_cache_local_cache_expired(self, dvrl_service):
        """Test cache check with expired local cache entry"""
        query_hash = "expired_query_hash"
        
        # Setup expired local cache entry
        from datetime import timedelta
        cache_entry = Mock()
        cache_entry.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)  # Expired
        
        dvrl_service.query_cache[query_hash] = cache_entry
        
        # Mock APG cache service to return None
        dvrl_service.cache_service = AsyncMock()
        dvrl_service.cache_service.get.return_value = None
        
        result = await dvrl_service._check_query_cache(query_hash)
        
        # Should return None for cache miss
        assert result is None
        
        # Expired entry should be removed
        assert query_hash not in dvrl_service.query_cache
    
    async def test_check_query_cache_complete_miss(self, dvrl_service):
        """Test cache check with complete cache miss"""
        query_hash = "missing_query_hash"
        
        # Mock APG cache service to return None
        dvrl_service.cache_service = AsyncMock()
        dvrl_service.cache_service.get.return_value = None
        
        # Ensure no local cache entry
        dvrl_service.query_cache = {}
        
        result = await dvrl_service._check_query_cache(query_hash)
        
        assert result is None
    
    async def test_check_query_cache_exception_handling(self, dvrl_service):
        """Test cache check with exception from cache service"""
        query_hash = "error_query_hash"
        
        # Mock APG cache service to raise exception
        dvrl_service.cache_service = AsyncMock()
        dvrl_service.cache_service.get.side_effect = Exception("Cache service unavailable")
        
        result = await dvrl_service._check_query_cache(query_hash)
        
        # Should return None on exception (cache miss)
        assert result is None


class TestTransactionCleanupImprovements:
    """Test suite for improved transaction cleanup error handling"""
    
    @pytest.fixture
    async def dvrl_service(self):
        """Create DVRL service for testing"""
        config = {'tenant_id': 'test_tenant', 'user_id': 'test_user'}
        service = DVRLService(config)
        await service.initialize()
        return service
    
    async def test_rollback_transaction_success_logging(self, dvrl_service):
        """Test that successful rollbacks are logged"""
        # Setup transaction
        tx_id = "test_transaction_123"
        mock_connector = AsyncMock()
        mock_connector.rollback_transaction.return_value = None  # Successful rollback
        
        dvrl_service.active_transactions[tx_id] = {
            'rollback_points': {
                'ds1': {'transaction_handle': 'tx_handle_1'}
            },
            'connectors': {
                'ds1': mock_connector
            }
        }
        
        with patch.object(dvrl_service, '_log_info', new_callable=AsyncMock) as mock_log:
            await dvrl_service.rollback_federated_transaction(tx_id)
        
        # Check that success was logged
        mock_log.assert_called_with(
            "Successfully rolled back transaction for data source: ds1"
        )
        mock_connector.rollback_transaction.assert_called_once_with('tx_handle_1')
    
    async def test_rollback_transaction_failure_logging(self, dvrl_service):
        """Test that rollback failures are logged as warnings"""
        # Setup transaction
        tx_id = "test_transaction_456"
        mock_connector = AsyncMock()
        mock_connector.rollback_transaction.side_effect = Exception("Rollback failed")
        
        dvrl_service.active_transactions[tx_id] = {
            'rollback_points': {
                'ds2': {'transaction_handle': 'tx_handle_2'}
            },
            'connectors': {
                'ds2': mock_connector
            }
        }
        
        with patch.object(dvrl_service, '_log_warning', new_callable=AsyncMock) as mock_log:
            await dvrl_service.rollback_federated_transaction(tx_id)
        
        # Check that failure was logged as warning
        mock_log.assert_called_with(
            "Failed to rollback transaction for data source ds2: Rollback failed"
        )
        mock_connector.rollback_transaction.assert_called_once_with('tx_handle_2')
    
    async def test_cleanup_expired_transactions_success_logging(self, dvrl_service):
        """Test that successful transaction cleanup is logged"""
        # Setup expired transaction
        from datetime import timedelta
        tx_id = "expired_transaction_789"
        expired_time = datetime.now(timezone.utc) - timedelta(hours=2)
        
        dvrl_service.active_transactions[tx_id] = {
            'created_at': expired_time,
            'rollback_points': {},
            'connectors': {}
        }
        
        with patch.object(dvrl_service, 'rollback_federated_transaction', new_callable=AsyncMock) as mock_rollback:
            with patch.object(dvrl_service, '_log_info', new_callable=AsyncMock) as mock_log:
                expired_count = await dvrl_service.cleanup_expired_transactions()
        
        assert expired_count == 1
        mock_rollback.assert_called_once_with(tx_id)
        mock_log.assert_called_with(
            f"Successfully cleaned up expired transaction: {tx_id}"
        )
    
    async def test_cleanup_expired_transactions_failure_logging(self, dvrl_service):
        """Test that failed transaction cleanup is logged as warning"""
        # Setup expired transaction 
        from datetime import timedelta
        tx_id = "failed_cleanup_abc"
        expired_time = datetime.now(timezone.utc) - timedelta(hours=3)
        
        dvrl_service.active_transactions[tx_id] = {
            'created_at': expired_time,
            'rollback_points': {},
            'connectors': {}
        }
        
        with patch.object(dvrl_service, 'rollback_federated_transaction', 
                         new_callable=AsyncMock, 
                         side_effect=Exception("Cleanup failed")) as mock_rollback:
            with patch.object(dvrl_service, '_log_warning', new_callable=AsyncMock) as mock_log:
                expired_count = await dvrl_service.cleanup_expired_transactions()
        
        # Expired count should be 0 since cleanup failed
        assert expired_count == 0
        mock_rollback.assert_called_once_with(tx_id)
        mock_log.assert_called_with(
            f"Failed to cleanup expired transaction {tx_id}: Cleanup failed"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])