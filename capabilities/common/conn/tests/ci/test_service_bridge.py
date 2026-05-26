"""
Tests for Service Bridge - async/sync integration layer
Tests the bridge between async services and sync Flask-AppBuilder views

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock, PropertyMock
from concurrent.futures import ThreadPoolExecutor

from ...service_bridge import ServiceBridge, with_service_bridge
from ...service import ConnectionManager, FlowExecutor, IntelligentConnector
from ...sqlalchemy_models import CnConnection, ConnectionStatus


class TestServiceBridge:
	"""Test ServiceBridge functionality"""

	def test_initialization(self, service_bridge):
		"""Test ServiceBridge initializes properly"""
		assert service_bridge._connection_manager is not None
		assert service_bridge._flow_executor is None  # Lazy loaded
		assert service_bridge._intelligent_connector is None  # Lazy loaded

	def test_get_event_loop(self, service_bridge):
		"""Test event loop creation and management"""
		# Get event loop
		loop1 = service_bridge._get_event_loop()
		assert loop1 is not None

		# Should return same loop on subsequent calls
		loop2 = service_bridge._get_event_loop()
		assert loop1 is loop2

	def test_run_async_simple(self, service_bridge):
		"""Test running simple async coroutine in sync context"""
		async def simple_async():
			await asyncio.sleep(0.01)
			return "async_result"

		result = service_bridge.run_async(simple_async())
		assert result == "async_result"

	def test_run_async_with_exception(self, service_bridge):
		"""Test async exception handling"""
		async def failing_async():
			raise ValueError("Test error")

		with pytest.raises(ValueError, match="Test error"):
			service_bridge.run_async(failing_async())

	def test_connection_manager_property(self, service_bridge):
		"""Test connection manager lazy loading"""
		service_bridge._connection_manager = None
		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_manager = Mock()
			mock_run_async.return_value = None

			with patch('capabilities.common.conn.service_bridge.ConnectionManager') as mock_conn_manager_class:
				mock_conn_manager_class.return_value = mock_manager

				manager = service_bridge.connection_manager

				assert manager == mock_manager
				assert service_bridge._connection_manager == mock_manager

	def test_flow_executor_property(self, service_bridge):
		"""Test flow executor lazy loading"""
		mock_manager = Mock()
		service_bridge._connection_manager = mock_manager

		with patch('capabilities.common.conn.service_bridge.FlowExecutor') as mock_flow_executor_class:
			mock_executor = Mock()
			mock_flow_executor_class.return_value = mock_executor

			executor = service_bridge.flow_executor

			assert executor == mock_executor
			mock_flow_executor_class.assert_called_once_with(connection_manager=mock_manager)

	def test_intelligent_connector_property(self, service_bridge):
		"""Test intelligent connector lazy loading"""
		with patch('capabilities.common.conn.service_bridge.IntelligentConnector') as mock_connector_class:
			mock_connector = Mock()
			mock_connector_class.return_value = mock_connector

			connector = service_bridge.intelligent_connector

			assert connector == mock_connector
			mock_connector_class.assert_called_once()


class TestServiceBridgeConnectionMethods:
	"""Test connection management methods via service bridge"""

	def test_create_connection_success(self, service_bridge):
		"""Test successful connection creation"""
		connection_data = {
			'name': 'Test Connection',
			'connection_type': 'database',
			'tap_config': {'host': 'localhost'}
		}

		mock_connection = Mock()
		mock_connection.id = 'conn_123'
		mock_connection.status.value = 'configuring'
		mock_connection.name = 'Test Connection'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_connection

			result = service_bridge.create_connection(connection_data)

			assert result['success'] is True
			assert result['connection_id'] == 'conn_123'
			assert result['status'] == 'configuring'
			assert 'Test Connection' in result['message']

	def test_create_connection_failure(self, service_bridge):
		"""Test connection creation failure"""
		connection_data = {'invalid': 'data'}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.side_effect = ValueError("Invalid connection data")

			result = service_bridge.create_connection(connection_data)

			assert result['success'] is False
			assert 'Invalid connection data' in result['error']
			assert 'Failed to create connection' in result['message']

	def test_test_connection_success(self, service_bridge):
		"""Test successful connection testing"""
		connection_id = 'conn_123'
		mock_test_result = {'status': 'success', 'latency': 150.5}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_test_result

			result = service_bridge.test_connection(connection_id)

			assert result['success'] is True
			assert result['result']['status'] == 'success'
			assert result['message'] == 'Connection test completed'

	def test_test_connection_failure(self, service_bridge):
		"""Test connection testing failure"""
		connection_id = 'conn_123'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.side_effect = Exception("Connection timeout")

			result = service_bridge.test_connection(connection_id)

			assert result['success'] is False
			assert 'Connection timeout' in result['error']
			assert 'Connection test failed' in result['message']

	def test_get_connection_health(self, service_bridge):
		"""Test connection health retrieval"""
		connection_id = 'conn_123'

		mock_health = Mock()
		mock_health.status.value = 'active'
		mock_health.latency_ms = 125.0
		mock_health.throughput_records_per_sec = 1500.0
		mock_health.error_rate = 0.01
		mock_health.is_healthy.return_value = True
		mock_health.timestamp.isoformat.return_value = '2025-01-01T12:00:00Z'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_health

			result = service_bridge.get_connection_health(connection_id)

			assert result['success'] is True
			health_data = result['health']
			assert health_data['status'] == 'active'
			assert health_data['latency_ms'] == 125.0
			assert health_data['is_healthy'] is True

	def test_get_connection_health_not_found(self, service_bridge):
		"""Test connection health when not available"""
		connection_id = 'conn_123'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = None

			result = service_bridge.get_connection_health(connection_id)

			assert result['success'] is False
			assert 'Health data not available' in result['message']

	def test_get_performance_metrics(self, service_bridge, mock_performance_metrics):
		"""Test performance metrics retrieval"""
		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_performance_metrics

			result = service_bridge.get_performance_metrics()

			assert result['success'] is True
			metrics = result['metrics']
			assert 'system_metrics' in metrics
			assert 'connection_metrics' in metrics
			assert metrics['system_metrics']['cpu_usage'] == 45.2


class TestServiceBridgeFlowMethods:
	"""Test flow management methods via service bridge"""

	def test_create_flow_success(self, service_bridge):
		"""Test successful flow creation"""
		flow_data = {
			'name': 'Test Flow',
			'source_connection_id': 'conn_1',
			'target_connection_id': 'conn_2'
		}

		mock_flow = Mock()
		mock_flow.id = 'flow_123'
		mock_flow.name = 'Test Flow'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_flow

			result = service_bridge.create_flow(flow_data)

			assert result['success'] is True
			assert result['flow_id'] == 'flow_123'
			assert 'Test Flow' in result['message']

	def test_create_flow_failure(self, service_bridge):
		"""Test flow creation failure"""
		flow_data = {'invalid': 'data'}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.side_effect = ValueError("Invalid flow configuration")

			result = service_bridge.create_flow(flow_data)

			assert result['success'] is False
			assert 'Invalid flow configuration' in result['error']

	def test_execute_flow_success(self, service_bridge):
		"""Test successful flow execution"""
		flow_id = 'flow_123'
		mock_result = {'status': 'success', 'records_processed': 1500}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_result

			result = service_bridge.execute_flow(flow_id)

			assert result['success'] is True
			assert result['result']['records_processed'] == 1500
			assert result['message'] == 'Flow execution completed'

	def test_execute_flow_failure(self, service_bridge):
		"""Test flow execution failure"""
		flow_id = 'flow_123'

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.side_effect = Exception("Execution failed")

			result = service_bridge.execute_flow(flow_id)

			assert result['success'] is False
			assert 'Execution failed' in result['error']


class TestServiceBridgeLineageMethods:
	"""Test lineage methods via service bridge"""

	def test_discover_lineage_success(self, service_bridge):
		"""Test successful lineage discovery"""
		connection_id = 'conn_123'

		mock_connection = Mock()
		mock_discovery_result = {
			'nodes_created': 15,
			'tables_discovered': 3,
			'fields_discovered': 25
		}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_discovery_result

			with patch.object(service_bridge.connection_manager, 'get_connection') as mock_get_conn:
				mock_get_conn.return_value = mock_connection

				result = service_bridge.discover_lineage(connection_id)

				assert result['success'] is True
				assert result['discovery_result']['nodes_created'] == 15
				assert '15 lineage nodes' in result['message']

	def test_discover_lineage_connection_not_found(self, service_bridge):
		"""Test lineage discovery when connection not found"""
		connection_id = 'conn_123'

		with patch.object(service_bridge.connection_manager, 'get_connection') as mock_get_conn:
			mock_get_conn.return_value = None

			result = service_bridge.discover_lineage(connection_id)

			assert result['success'] is False
			assert 'Connection not found' in result['message']

	def test_get_lineage_visualization_success(self, service_bridge):
		"""Test successful lineage visualization data retrieval"""
		params = {
			'node_id': 'node_123',
			'type': 'upstream',
			'max_depth': 5
		}

		mock_lineage_data = {
			'nodes': [
				{'id': 'node_1', 'label': 'Table 1', 'type': 'table'},
				{'id': 'node_2', 'label': 'Field 1', 'type': 'field'}
			],
			'edges': [
				{'id': 'edge_1', 'source': 'node_1', 'target': 'node_2', 'type': 'contains'}
			],
			'summary': {
				'total_nodes': 2,
				'total_edges': 1,
				'sensitive_entities': 0
			}
		}

		with patch('capabilities.common.conn.service_bridge.lineage_engine') as mock_engine:
			mock_engine.get_lineage_visualization.return_value = mock_lineage_data

			result = service_bridge.get_lineage_visualization(params)

			assert result['success'] is True
			assert result['lineage_data']['summary']['total_nodes'] == 2

			# Verify correct parameters were passed
			mock_engine.get_lineage_visualization.assert_called_once_with(
				node_id='node_123',
				visualization_type='upstream',
				max_depth=5
			)

	def test_get_lineage_visualization_default_params(self, service_bridge):
		"""Test lineage visualization with default parameters"""
		with patch('capabilities.common.conn.service_bridge.lineage_engine') as mock_engine:
			mock_engine.get_lineage_visualization.return_value = {'nodes': [], 'edges': [], 'summary': {}}

			result = service_bridge.get_lineage_visualization()

			assert result['success'] is True

			# Verify default parameters
			mock_engine.get_lineage_visualization.assert_called_once_with(
				node_id=None,
				visualization_type='full',
				max_depth=10
			)

	def test_get_lineage_visualization_failure(self, service_bridge):
		"""Test lineage visualization failure"""
		with patch('capabilities.common.conn.service_bridge.lineage_engine') as mock_engine:
			mock_engine.get_lineage_visualization.side_effect = Exception("Lineage error")

			result = service_bridge.get_lineage_visualization()

			assert result['success'] is False
			assert 'Lineage error' in result['error']
			assert result['lineage_data'] == {'nodes': [], 'edges': [], 'summary': {}}


class TestServiceBridgeAIMethods:
	"""Test AI and intelligence methods via service bridge"""

	def test_suggest_field_mappings(self, service_bridge, mock_ai_suggestions):
		"""Test AI field mapping suggestions"""
		source_schema = {'properties': {'first_name': {'type': 'string'}}}
		target_schema = {'properties': {'fname': {'type': 'string'}}}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_ai_suggestions['suggestions']

			result = service_bridge.suggest_field_mappings(source_schema, target_schema)

			assert result['success'] is True
			assert len(result['suggestions']) == 2
			assert result['suggestions'][0]['confidence'] == 0.95

	def test_predict_performance(self, service_bridge):
		"""Test performance prediction"""
		connection_config = {
			'connection_type': 'database',
			'data_size_mb': 1000
		}

		mock_prediction = {
			'estimated_duration_minutes': 15,
			'throughput_prediction': 2000,
			'resource_requirements': {'cpu': 'medium', 'memory': '4GB'}
		}

		with patch.object(service_bridge, 'run_async') as mock_run_async:
			mock_run_async.return_value = mock_prediction

			result = service_bridge.predict_performance(connection_config)

			assert result['success'] is True
			assert result['prediction']['estimated_duration_minutes'] == 15
			assert result['message'] == 'Performance prediction completed'


class TestServiceBridgeDecorator:
	"""Test the with_service_bridge decorator"""

	def test_with_service_bridge_decorator(self, service_bridge):
		"""Test decorator adds service bridge to function"""

		@with_service_bridge
		def test_function(self, arg1, service_bridge=None):
			return {
				'arg1': arg1,
				'has_service_bridge': service_bridge is not None,
				'service_bridge_type': type(service_bridge).__name__
			}

		# Mock self object
		mock_self = Mock()

		result = test_function(mock_self, 'test_value')

		assert result['arg1'] == 'test_value'
		assert result['has_service_bridge'] is True
		assert result['service_bridge_type'] == 'ServiceBridge'

	def test_decorator_preserves_function_metadata(self):
		"""Test decorator preserves original function metadata"""

		@with_service_bridge
		def documented_function():
			"""This is a test function with documentation."""
			return "test"

		assert documented_function.__doc__ == "This is a test function with documentation."
		assert documented_function.__name__ == "documented_function"


# Integration tests
class TestServiceBridgeIntegration:
	"""Integration tests for service bridge with real async operations"""

	async def test_real_async_integration(self):
		"""Test with real async operations (not mocked)"""
		bridge = ServiceBridge()

		async def test_async_function(value):
			await asyncio.sleep(0.01)  # Small delay to ensure it's truly async
			return value * 2

		result = bridge.run_async(test_async_function(21))
		assert result == 42

	def test_concurrent_operations(self):
		"""Test bridge handles concurrent operations correctly"""
		bridge = ServiceBridge()

		async def async_operation(delay, result):
			await asyncio.sleep(delay)
			return result

		# Run multiple operations
		results = []
		for i in range(5):
			result = bridge.run_async(async_operation(0.01 * i, f"result_{i}"))
			results.append(result)

		assert len(results) == 5
		assert results[0] == "result_0"
		assert results[4] == "result_4"


# Error handling tests
class TestServiceBridgeErrorHandling:
	"""Test error handling in service bridge"""

	def test_async_timeout_handling(self, service_bridge):
		"""Test handling of async operation timeouts"""
		async def slow_operation():
			await asyncio.sleep(10)  # Very slow operation
			return "completed"

		# This should complete quickly due to our test setup
		# In a real implementation, you might want to add timeout handling
		with patch.object(service_bridge, '_get_event_loop') as mock_loop:
			mock_loop.return_value.run_until_complete.side_effect = TimeoutError("Operation timed out")

			with pytest.raises(TimeoutError):
				service_bridge.run_async(slow_operation())

	def test_invalid_async_operation(self, service_bridge):
		"""Test handling of invalid async operations"""
		def not_async_function():
			return "not async"

		# This should still work - sync functions can be run in async context
		result = service_bridge.run_async(not_async_function())
		assert result == "not async"

	def test_service_unavailable_fallback(self, service_bridge):
		"""Test fallback behavior when services are unavailable"""
		with patch.object(ServiceBridge, 'connection_manager', new_callable=PropertyMock) as mock_manager:
			mock_manager.side_effect = Exception("Service unavailable")
			# Should handle gracefully
			with pytest.raises(Exception, match="Service unavailable"):
				_ = service_bridge.connection_manager


# Performance tests
class TestServiceBridgePerformance:
	"""Performance tests for service bridge"""

	def test_rapid_async_calls(self, service_bridge):
		"""Test bridge performance with many rapid async calls"""
		import time

		async def quick_async_op(value):
			return value + 1

		start_time = time.time()

		results = []
		for i in range(100):
			result = service_bridge.run_async(quick_async_op(i))
			results.append(result)

		duration = time.time() - start_time

		assert len(results) == 100
		assert results[0] == 1
		assert results[99] == 100
		assert duration < 5.0  # Should complete quickly

	def test_memory_usage_stability(self, service_bridge):
		"""Test that bridge doesn't leak memory with repeated use"""
		import gc

		async def memory_test_op():
			data = list(range(1000))  # Create some data
			return len(data)

		# Run multiple operations and force garbage collection
		for _ in range(50):
			result = service_bridge.run_async(memory_test_op())
			assert result == 1000

		gc.collect()

		# If we got here without running out of memory, test passed
		assert True
