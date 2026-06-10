"""
Comprehensive tests for Connection Management service layer
Tests the core business logic, async operations, and integration points

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import UUID

from ...service import ConnectionManager, FlowExecutor, IntelligentConnector
from ...sqlalchemy_models import (
    CnConnection, CnDataFlow, CnHealthMetric, CnExecutionLog,
    ConnectionStatus, ConnectionType, SyncMode, ExecutionStatus
)


class TestConnectionManager:
	"""Test ConnectionManager functionality"""

	async def test_initialization(self, connection_manager):
		"""Test ConnectionManager initializes properly"""
		assert connection_manager.initialized
		assert connection_manager.singer_registry is not None
		assert connection_manager.health_monitor is not None
		assert connection_manager.performance_tracker is not None

	async def test_create_connection_success(self, connection_manager, sample_connection_data):
		"""Test successful connection creation"""
		connection = await connection_manager.create_connection(sample_connection_data)

		assert connection is not None
		assert isinstance(connection.id, UUID)
		assert connection.name == sample_connection_data['name']
		assert connection.connection_type == sample_connection_data['connection_type']
		assert connection.status == ConnectionStatus.CONFIGURING
		assert connection.tenant_id == 'default'

	async def test_create_connection_validation_error(self, connection_manager):
		"""Test connection creation with invalid data"""
		invalid_data = {
			'name': '',  # Invalid empty name
			'connection_type': 'invalid_type'  # Invalid type
		}

		with pytest.raises(ValueError) as exc_info:
			await connection_manager.create_connection(invalid_data)

		assert 'name' in str(exc_info.value) or 'connection_type' in str(exc_info.value)

	async def test_test_connection_success(self, connection_manager, sample_connection):
		"""Test successful connection testing"""
		# Mock successful connection test
		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 0
			mock_run.return_value.stdout = '{"status": "success", "message": "Connection successful"}'

			result = await connection_manager.test_connection_sync(str(sample_connection.id))

			assert result['status'] == 'success'
			assert 'message' in result

	async def test_test_connection_failure(self, connection_manager, sample_connection):
		"""Test connection test failure"""
		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 1
			mock_run.return_value.stderr = 'Connection failed: Invalid credentials'

			result = await connection_manager.test_connection_sync(str(sample_connection.id))

			assert result['status'] == 'error'
			assert 'Connection failed' in result['error']

	async def test_get_connection_health(self, connection_manager, sample_connection, db_session):
		"""Test connection health retrieval"""
		# Create sample health metric
		health_metric = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ACTIVE,
			latency_ms=150.5,
			throughput_records_per_sec=1200.0,
			error_rate=0.02,
			timestamp=datetime.now(timezone.utc)
		)
		db_session.add(health_metric)
		db_session.commit()

		# Mock database session in manager
		connection_manager.db_session = db_session

		health = await connection_manager.get_connection_health(str(sample_connection.id))

		assert health is not None
		assert health.connection_id == str(sample_connection.id)
		assert health.status == ConnectionStatus.ACTIVE
		assert health.latency_ms == 150.5
		assert health.is_healthy()

	async def test_get_performance_metrics(self, connection_manager, mock_performance_metrics):
		"""Test performance metrics collection"""
		with patch.object(connection_manager.performance_tracker, 'get_system_metrics') as mock_metrics:
			mock_metrics.return_value = mock_performance_metrics

			metrics = await connection_manager.get_performance_metrics()

			assert 'system_metrics' in metrics
			assert 'connection_metrics' in metrics
			assert metrics['system_metrics']['cpu_usage'] == 45.2
			assert metrics['connection_metrics']['total_connections'] == 15

	async def test_discover_schema_singer(self, connection_manager, sample_connection, mock_singer_discovery):
		"""Test schema discovery using Singer.io"""
		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 0
			mock_run.return_value.stdout = f'{mock_singer_discovery}'

			schema = await connection_manager.discover_schema(str(sample_connection.id))

			assert 'streams' in schema
			assert len(schema['streams']) == 2
			assert schema['streams'][0]['tap_stream_id'] == 'users'

	async def test_start_health_monitoring(self, connection_manager, sample_connection):
		"""Test health monitoring startup"""
		with patch.object(connection_manager.health_monitor, 'start_monitoring') as mock_start:
			mock_start.return_value = True

			result = await connection_manager.start_health_monitoring(str(sample_connection.id))

			assert result is True
			mock_start.assert_called_once_with(str(sample_connection.id))

	async def test_update_connection(self, connection_manager, sample_connection):
		"""Test connection update"""
		update_data = {
			'name': 'Updated Connection Name',
			'description': 'Updated description',
			'batch_size': 2000
		}

		updated_connection = await connection_manager.update_connection(
			str(sample_connection.id), update_data
		)

		assert updated_connection.name == update_data['name']
		assert updated_connection.description == update_data['description']
		assert updated_connection.batch_size == update_data['batch_size']

	async def test_delete_connection(self, connection_manager, sample_connection, db_session):
		"""Test connection deletion"""
		connection_id = str(sample_connection.id)
		connection_manager.db_session = db_session

		result = await connection_manager.delete_connection(connection_id)

		assert result is True

		# Verify connection is marked as deleted or removed
		deleted_conn = db_session.query(CnConnection).filter(
			CnConnection.id == sample_connection.id
		).first()

		# Depending on implementation, connection might be soft-deleted or removed
		assert deleted_conn is None or deleted_conn.status == ConnectionStatus.INACTIVE


class TestFlowExecutor:
	"""Test FlowExecutor functionality"""

	@pytest.fixture
	async def flow_executor(self, connection_manager):
		"""Create FlowExecutor instance"""
		return FlowExecutor(connection_manager=connection_manager)

	async def test_create_flow(self, flow_executor, sample_flow_data, db_session):
		"""Test flow creation"""
		flow_executor.db_session = db_session

		flow = await flow_executor.create_flow(sample_flow_data)

		assert flow is not None
		assert flow.name == sample_flow_data['name']
		assert flow.source_connection_id == sample_flow_data['source_connection_id']
		assert flow.target_connection_id == sample_flow_data['target_connection_id']
		assert flow.field_mappings == sample_flow_data['field_mappings']

	async def test_execute_flow_once_success(self, flow_executor, sample_flow, mock_singer_tap_response):
		"""Test successful flow execution"""
		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 0
			mock_run.return_value.stdout = str(mock_singer_tap_response)

			result = await flow_executor.execute_flow_once(str(sample_flow.id))

			assert result['status'] == 'success'
			assert 'execution_id' in result
			assert result['records_processed'] > 0

	async def test_execute_flow_once_failure(self, flow_executor, sample_flow):
		"""Test flow execution failure"""
		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 1
			mock_run.return_value.stderr = 'Singer tap failed: Connection timeout'

			result = await flow_executor.execute_flow_once(str(sample_flow.id))

			assert result['status'] == 'error'
			assert 'Connection timeout' in result['error']

	async def test_validate_flow(self, flow_executor, sample_flow):
		"""Test flow validation"""
		validation_result = await flow_executor.validate_flow(str(sample_flow.id))

		assert 'valid' in validation_result
		assert 'errors' in validation_result
		assert 'warnings' in validation_result

		# Should be valid for our sample flow
		assert validation_result['valid'] is True
		assert len(validation_result['errors']) == 0

	async def test_get_flow_execution_history(self, flow_executor, sample_flow, db_session):
		"""Test flow execution history retrieval"""
		# Create sample execution logs
		log1 = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.SUCCESS,
			started_at=datetime.now(timezone.utc) - timedelta(hours=2),
			completed_at=datetime.now(timezone.utc) - timedelta(hours=1),
			records_processed=1500,
			execution_details={'tap': 'tap-postgres', 'target': 'target-warehouse'}
		)

		log2 = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.FAILED,
			started_at=datetime.now(timezone.utc) - timedelta(hours=4),
			completed_at=datetime.now(timezone.utc) - timedelta(hours=3),
			error_message='Connection timeout',
			execution_details={'tap': 'tap-postgres'}
		)

		db_session.add(log1)
		db_session.add(log2)
		db_session.commit()

		flow_executor.db_session = db_session

		history = await flow_executor.get_flow_execution_history(str(sample_flow.id))

		assert len(history) == 2
		assert history[0]['status'] == 'SUCCESS'  # Should be sorted by date desc
		assert history[0]['records_processed'] == 1500
		assert history[1]['status'] == 'FAILED'
		assert 'Connection timeout' in history[1]['error_message']

	async def test_schedule_flow(self, flow_executor, sample_flow):
		"""Test flow scheduling"""
		with patch.object(flow_executor, 'scheduler') as mock_scheduler:
			mock_scheduler.add_job.return_value = Mock(id='job_123')

			job_id = await flow_executor.schedule_flow(str(sample_flow.id))

			assert job_id is not None
			mock_scheduler.add_job.assert_called_once()

	async def test_stop_flow_execution(self, flow_executor, sample_flow):
		"""Test stopping flow execution"""
		# Mock running execution
		execution_id = 'exec_123'

		with patch.object(flow_executor, 'running_executions') as mock_executions:
			mock_process = Mock()
			mock_executions.__contains__.return_value = True
			mock_executions.__getitem__.return_value = mock_process

			result = await flow_executor.stop_flow_execution(execution_id)

			assert result is True
			mock_process.terminate.assert_called_once()


class TestIntelligentConnector:
	"""Test IntelligentConnector AI-powered features"""

	@pytest.fixture
	def intelligent_connector(self):
		"""Create IntelligentConnector instance"""
		return IntelligentConnector()

	async def test_suggest_field_mappings(self, intelligent_connector, mock_ai_suggestions):
		"""Test AI-powered field mapping suggestions"""
		source_schema = {
			'properties': {
				'first_name': {'type': 'string'},
				'email_address': {'type': 'string'},
				'created_date': {'type': 'string', 'format': 'date'}
			}
		}

		target_schema = {
			'properties': {
				'fname': {'type': 'string'},
				'email': {'type': 'string'},
				'created_at': {'type': 'string', 'format': 'date-time'}
			}
		}

		with patch.object(intelligent_connector, 'ai_service') as mock_ai:
			mock_ai.suggest_mappings.return_value = mock_ai_suggestions

			suggestions = await intelligent_connector.suggest_field_mappings(
				source_schema, target_schema
			)

			assert len(suggestions['suggestions']) == 2
			assert suggestions['suggestions'][0]['confidence'] == 0.95
			assert suggestions['suggestions'][1]['source_field'] == 'email_address'

	async def test_predict_performance(self, intelligent_connector):
		"""Test performance prediction"""
		connection_config = {
			'connection_type': 'database',
			'data_size_mb': 500,
			'batch_size': 1000,
			'network_latency_ms': 50
		}

		prediction = await intelligent_connector.predict_performance(connection_config)

		assert 'estimated_duration_minutes' in prediction
		assert 'throughput_prediction' in prediction
		assert 'resource_requirements' in prediction
		assert prediction['estimated_duration_minutes'] > 0

	async def test_optimize_batch_size(self, intelligent_connector):
		"""Test batch size optimization"""
		performance_history = [
			{'batch_size': 500, 'throughput': 800, 'latency': 120},
			{'batch_size': 1000, 'throughput': 1200, 'latency': 150},
			{'batch_size': 1500, 'throughput': 1100, 'latency': 200},
		]

		optimal_batch_size = await intelligent_connector.optimize_batch_size(
			performance_history
		)

		assert isinstance(optimal_batch_size, int)
		assert 500 <= optimal_batch_size <= 2000  # Reasonable range

	async def test_detect_schema_drift(self, intelligent_connector):
		"""Test schema drift detection"""
		old_schema = {
			'properties': {
				'id': {'type': 'integer'},
				'name': {'type': 'string'},
				'email': {'type': 'string'}
			}
		}

		new_schema = {
			'properties': {
				'id': {'type': 'integer'},
				'name': {'type': 'string'},
				'email': {'type': 'string'},
				'phone': {'type': 'string'},  # New field
				'address': {'type': 'string'}  # New field
			}
		}

		drift_result = await intelligent_connector.detect_schema_drift(old_schema, new_schema)

		assert 'drift_detected' in drift_result
		assert drift_result['drift_detected'] is True
		assert 'added_fields' in drift_result
		assert len(drift_result['added_fields']) == 2
		assert 'phone' in drift_result['added_fields']

	async def test_generate_data_quality_rules(self, intelligent_connector):
		"""Test data quality rule generation"""
		sample_data = [
			{'id': 1, 'email': 'user1@example.com', 'age': 25},
			{'id': 2, 'email': 'user2@example.com', 'age': 30},
			{'id': 3, 'email': 'invalid-email', 'age': -5},  # Invalid data
		]

		quality_rules = await intelligent_connector.generate_data_quality_rules(sample_data)

		assert 'rules' in quality_rules
		assert len(quality_rules['rules']) > 0

		# Should detect email format rule
		email_rules = [rule for rule in quality_rules['rules'] if 'email' in rule['field']]
		assert len(email_rules) > 0

		# Should detect age range rule
		age_rules = [rule for rule in quality_rules['rules'] if 'age' in rule['field']]
		assert len(age_rules) > 0


# Integration tests
class TestServiceIntegration:
	"""Integration tests between service components"""

	async def test_connection_to_flow_integration(self, connection_manager, sample_connection_data):
		"""Test creating connection and using it in flow"""
		# Create connection
		connection = await connection_manager.create_connection(sample_connection_data)

		# Create flow using the connection
		flow_executor = FlowExecutor(connection_manager=connection_manager)

		flow_data = {
			'name': 'Integration Test Flow',
			'source_connection_id': str(connection.id),
			'target_connection_id': str(connection.id),
			'field_mappings': {'id': 'user_id'},
			'enabled': True
		}

		flow = await flow_executor.create_flow(flow_data)

		assert flow.source_connection_id == str(connection.id)
		assert str(connection.id) in [flow.source_connection_id, flow.target_connection_id]

	async def test_health_monitoring_flow(self, connection_manager, sample_connection, db_session):
		"""Test complete health monitoring workflow"""
		connection_manager.db_session = db_session

		# Start monitoring
		monitoring_started = await connection_manager.start_health_monitoring(
			str(sample_connection.id)
		)
		assert monitoring_started

		# Simulate health check
		with patch.object(connection_manager, 'test_connection_sync') as mock_test:
			mock_test.return_value = {'status': 'success', 'latency': 125.5}

			await connection_manager._perform_health_check(str(sample_connection.id))

		# Verify health metric was recorded
		health = await connection_manager.get_connection_health(str(sample_connection.id))
		assert health is not None

	async def test_performance_tracking_integration(self, connection_manager, sample_flow):
		"""Test performance tracking during flow execution"""
		flow_executor = FlowExecutor(connection_manager=connection_manager)

		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 0
			mock_run.return_value.stdout = '{"records": 1000, "duration": 45.2}'

			# Execute flow
			result = await flow_executor.execute_flow_once(str(sample_flow.id))

			assert result['status'] == 'success'

			# Verify performance metrics were updated
			metrics = await connection_manager.get_performance_metrics()
			assert 'flow_metrics' in metrics


# Performance tests
class TestServicePerformance:
	"""Performance and stress tests"""

	async def test_concurrent_connections(self, connection_manager):
		"""Test handling multiple concurrent connection operations"""
		connection_data_templates = [
			{'name': f'Connection {i}', 'connection_type': ConnectionType.DATABASE}
			for i in range(10)
		]

		# Create connections concurrently
		tasks = [
			connection_manager.create_connection(template)
			for template in connection_data_templates
		]

		connections = await asyncio.gather(*tasks, return_exceptions=True)


		assert len(connections) == 10
		assert all(conn.name.startswith('Connection') for conn in connections)

	async def test_large_schema_discovery(self, connection_manager, sample_connection):
		"""Test schema discovery with large number of tables/fields"""
		# Mock large schema response
		large_schema = {
			'streams': [
				{
					'tap_stream_id': f'table_{i}',
					'schema': {
						'properties': {
							f'field_{j}': {'type': 'string'}
							for j in range(50)  # 50 fields per table
						}
					}
				}
				for i in range(100)  # 100 tables
			]
		}

		with patch('subprocess.run') as mock_run:
			mock_run.return_value.returncode = 0
			mock_run.return_value.stdout = str(large_schema)

			schema = await connection_manager.discover_schema(str(sample_connection.id))

			assert len(schema['streams']) == 100
			assert len(schema['streams'][0]['schema']['properties']) == 50

	async def test_health_monitoring_load(self, connection_manager, db_session):
		"""Test health monitoring under load"""
		connection_manager.db_session = db_session

		# Create multiple connections
		connections = []
		for i in range(5):
			conn_data = {
				'name': f'Load Test Connection {i}',
				'connection_type': ConnectionType.DATABASE
			}
			conn = await connection_manager.create_connection(conn_data)
			connections.append(conn)

		# Start monitoring for all connections
		monitoring_tasks = [
			connection_manager.start_health_monitoring(str(conn.id))
			for conn in connections
		]

		results = await asyncio.gather(*monitoring_tasks, return_exceptions=True)

		assert all(results)  # All should start successfully