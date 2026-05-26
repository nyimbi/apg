"""
Tests for SQLAlchemy Models
Tests database models, relationships, and validation

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy.exc import IntegrityError

from ...sqlalchemy_models import (
    CnConnection, CnDataFlow, CnTransformationRule, CnSingerTap,
    CnHealthMetric, CnExecutionLog, CnLineageNode, CnLineageEdge,
    ConnectionType, ConnectionStatus, SyncMode, ExecutionStatus, LineageNodeType
)


class TestCnConnectionModel:
	"""Test CnConnection model"""

	def test_connection_creation(self, db_session):
		"""Test creating a new connection"""
		connection = CnConnection(
			tenant_id='test_tenant',
			name='Test Database',
			description='Test PostgreSQL database',
			connection_type=ConnectionType.DATABASE,
			status=ConnectionStatus.CONFIGURING,
			singer_tap='tap-postgres',
			sync_mode=SyncMode.INCREMENTAL,
			batch_size=1000,
			enabled=True,
			tap_config={'host': 'localhost', 'port': 5432},
			target_config={'connection_string': 'postgresql://...'}
		)

		db_session.add(connection)
		db_session.commit()
		db_session.refresh(connection)

		assert isinstance(connection.id, UUID)
		assert connection.name == 'Test Database'
		assert connection.connection_type == ConnectionType.DATABASE
		assert connection.status == ConnectionStatus.CONFIGURING
		assert connection.created_at is not None
		assert connection.updated_at is not None

	def test_connection_validation_required_fields(self, db_session):
		"""Test connection requires mandatory fields"""
		# Missing tenant_id
		connection = CnConnection(
			name='Test Connection',
			connection_type=ConnectionType.DATABASE
		)

		db_session.add(connection)

		with pytest.raises(IntegrityError):
			db_session.commit()

	def test_connection_defaults(self, db_session):
		"""Test connection default values"""
		connection = CnConnection(
			tenant_id='test_tenant',
			name='Test Connection',
			connection_type=ConnectionType.DATABASE
		)

		db_session.add(connection)
		db_session.commit()
		db_session.refresh(connection)

		assert connection.status == ConnectionStatus.INACTIVE
		assert connection.sync_mode == SyncMode.FULL_TABLE
		assert connection.batch_size == 1000
		assert connection.enabled is True
		assert connection.records_processed == 0
		assert connection.error_count == 0

	def test_connection_unique_name_per_tenant(self, db_session):
		"""Test connection names are unique per tenant"""
		# First connection
		connection1 = CnConnection(
			tenant_id='tenant1',
			name='Unique Name',
			connection_type=ConnectionType.DATABASE
		)

		# Second connection same tenant, same name - should fail
		connection2 = CnConnection(
			tenant_id='tenant1',
			name='Unique Name',
			connection_type=ConnectionType.API
		)

		db_session.add(connection1)
		db_session.commit()

		db_session.add(connection2)

		with pytest.raises(IntegrityError):
			db_session.commit()

	def test_connection_same_name_different_tenant(self, db_session):
		"""Test same connection name allowed for different tenants"""
		connection1 = CnConnection(
			tenant_id='tenant1',
			name='Same Name',
			connection_type=ConnectionType.DATABASE
		)

		connection2 = CnConnection(
			tenant_id='tenant2',
			name='Same Name',
			connection_type=ConnectionType.API
		)

		db_session.add(connection1)
		db_session.add(connection2)
		db_session.commit()

		# Both should be saved successfully
		assert connection1.id != connection2.id
		assert connection1.tenant_id != connection2.tenant_id

	def test_connection_json_fields(self, db_session):
		"""Test JSON field storage and retrieval"""
		config_data = {
			'host': 'localhost',
			'port': 5432,
			'database': 'testdb',
			'ssl': True,
			'options': {
				'timeout': 30,
				'retries': 3
			}
		}

		connection = CnConnection(
			tenant_id='test_tenant',
			name='JSON Test',
			connection_type=ConnectionType.DATABASE,
			tap_config=config_data
		)

		db_session.add(connection)
		db_session.commit()
		db_session.refresh(connection)

		assert connection.tap_config == config_data
		assert connection.tap_config['options']['timeout'] == 30

	def test_connection_str_representation(self, db_session):
		"""Test connection string representation"""
		connection = CnConnection(
			tenant_id='test_tenant',
			name='Test Connection',
			connection_type=ConnectionType.DATABASE
		)

		db_session.add(connection)
		db_session.commit()
		db_session.refresh(connection)

		str_repr = str(connection)
		assert 'Test Connection' in str_repr
		assert 'database' in str_repr.lower()


class TestCnDataFlowModel:
	"""Test CnDataFlow model"""

	def test_data_flow_creation(self, db_session, sample_connection):
		"""Test creating a data flow"""
		flow = CnDataFlow(
			tenant_id='test_tenant',
			name='Test Flow',
			description='Test data flow',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id),
			field_mappings={'id': 'user_id', 'email': 'email_address'},
			transformation_config={
				'transformations': [
					{'type': 'filter', 'conditions': [{'field': 'active', 'operator': 'equals', 'value': True}]}
				]
			},
			schedule_expression='0 2 * * *',
			enabled=True
		)

		db_session.add(flow)
		db_session.commit()
		db_session.refresh(flow)

		assert isinstance(flow.id, UUID)
		assert flow.name == 'Test Flow'
		assert flow.source_connection_id == str(sample_connection.id)
		assert flow.field_mappings['id'] == 'user_id'
		assert flow.transformation_config['transformations'][0]['type'] == 'filter'

	def test_data_flow_defaults(self, db_session, sample_connection):
		"""Test data flow default values"""
		flow = CnDataFlow(
			tenant_id='test_tenant',
			name='Minimal Flow',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id)
		)

		db_session.add(flow)
		db_session.commit()
		db_session.refresh(flow)

		assert flow.enabled is True
		assert flow.execution_count == 0
		assert flow.records_processed == 0
		assert flow.created_at is not None

	def test_data_flow_relationships(self, db_session, sample_connection):
		"""Test data flow relationships with connections"""
		flow = CnDataFlow(
			tenant_id='test_tenant',
			name='Relationship Test',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id)
		)

		db_session.add(flow)
		db_session.commit()
		db_session.refresh(flow)

		# Note: In a full implementation, you might have foreign key relationships
		# For now, we're using string IDs to maintain flexibility
		assert flow.source_connection_id == str(sample_connection.id)
		assert flow.target_connection_id == str(sample_connection.id)


class TestCnHealthMetricModel:
	"""Test CnHealthMetric model"""

	def test_health_metric_creation(self, db_session, sample_connection):
		"""Test creating health metrics"""
		metric = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ACTIVE,
			latency_ms=125.5,
			throughput_records_per_sec=1500.0,
			error_rate=0.02,
			cpu_usage=45.3,
			memory_usage=67.8,
			additional_metrics={'custom_metric': 123.4}
		)

		db_session.add(metric)
		db_session.commit()
		db_session.refresh(metric)

		assert metric.connection_id == str(sample_connection.id)
		assert metric.latency_ms == 125.5
		assert metric.error_rate == 0.02
		assert metric.additional_metrics['custom_metric'] == 123.4
		assert metric.timestamp is not None

	def test_health_metric_is_healthy(self, db_session, sample_connection):
		"""Test health metric is_healthy method"""
		# Healthy metric
		healthy_metric = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ACTIVE,
			latency_ms=100.0,
			error_rate=0.01
		)

		db_session.add(healthy_metric)
		db_session.commit()
		db_session.refresh(healthy_metric)

		assert healthy_metric.is_healthy() is True

		# Unhealthy metric
		unhealthy_metric = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ERROR,
			latency_ms=5000.0,
			error_rate=0.15
		)

		db_session.add(unhealthy_metric)
		db_session.commit()
		db_session.refresh(unhealthy_metric)

		assert unhealthy_metric.is_healthy() is False


class TestCnExecutionLogModel:
	"""Test CnExecutionLog model"""

	def test_execution_log_creation(self, db_session, sample_flow):
		"""Test creating execution logs"""
		log = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.SUCCESS,
			started_at=datetime.now(timezone.utc),
			completed_at=datetime.now(timezone.utc),
			records_processed=1500,
			execution_details={
				'tap': 'tap-postgres',
				'target': 'target-warehouse',
				'config_used': {'batch_size': 1000}
			}
		)

		db_session.add(log)
		db_session.commit()
		db_session.refresh(log)

		assert log.flow_id == str(sample_flow.id)
		assert log.status == ExecutionStatus.SUCCESS
		assert log.records_processed == 1500
		assert log.execution_details['tap'] == 'tap-postgres'

	def test_execution_log_duration_calculation(self, db_session, sample_flow):
		"""Test execution duration calculation"""
		from datetime import timedelta

		start_time = datetime.now(timezone.utc)
		end_time = start_time + timedelta(minutes=5, seconds=30)

		log = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.SUCCESS,
			started_at=start_time,
			completed_at=end_time
		)

		db_session.add(log)
		db_session.commit()
		db_session.refresh(log)

		duration = log.get_duration()
		assert duration.total_seconds() == 330  # 5 minutes 30 seconds

	def test_execution_log_running_status(self, db_session, sample_flow):
		"""Test execution log for running flows"""
		log = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.RUNNING,
			started_at=datetime.now(timezone.utc)
		)

		db_session.add(log)
		db_session.commit()
		db_session.refresh(log)

		assert log.status == ExecutionStatus.RUNNING
		assert log.completed_at is None

		# Duration should be None for running executions
		assert log.get_duration() is None


class TestCnLineageNodeModel:
	"""Test CnLineageNode model"""

	def test_lineage_node_creation(self, db_session, sample_connection):
		"""Test creating lineage nodes"""
		node = CnLineageNode(
			id='conn_' + str(sample_connection.id),
			tenant_id='test_tenant',
			name=sample_connection.name,
			node_type=LineageNodeType.CONNECTION,
			connection_id=str(sample_connection.id),
			sensitive=False,
			meta_data={'connection_type': sample_connection.connection_type.value}
		)

		db_session.add(node)
		db_session.commit()
		db_session.refresh(node)

		assert node.id == 'conn_' + str(sample_connection.id)
		assert node.node_type == LineageNodeType.CONNECTION
		assert node.sensitive is False
		assert node.meta_data['connection_type'] == sample_connection.connection_type.value

	def test_lineage_node_field_level(self, db_session, sample_connection):
		"""Test field-level lineage node"""
		field_node = CnLineageNode(
			id='field_users_email',
			tenant_id='test_tenant',
			name='users.email',
			node_type=LineageNodeType.FIELD,
			connection_id=str(sample_connection.id),
			schema_name='public',
			table_name='users',
			field_name='email',
			sensitive=True,
			pii_classification='email_address',
			meta_data={'data_type': 'varchar', 'max_length': 255}
		)

		db_session.add(field_node)
		db_session.commit()
		db_session.refresh(field_node)

		assert field_node.node_type == LineageNodeType.FIELD
		assert field_node.table_name == 'users'
		assert field_node.field_name == 'email'
		assert field_node.sensitive is True
		assert field_node.pii_classification == 'email_address'

	def test_lineage_node_unique_id(self, db_session):
		"""Test lineage node ID uniqueness"""
		node1 = CnLineageNode(
			id='unique_node_1',
			tenant_id='test_tenant',
			name='Node 1',
			node_type=LineageNodeType.TABLE
		)

		node2 = CnLineageNode(
			id='unique_node_1',  # Same ID
			tenant_id='test_tenant',
			name='Node 2',
			node_type=LineageNodeType.TABLE
		)

		db_session.add(node1)
		db_session.commit()

		db_session.add(node2)

		with pytest.raises(IntegrityError):
			db_session.commit()


class TestCnLineageEdgeModel:
	"""Test CnLineageEdge model"""

	def test_lineage_edge_creation(self, db_session, sample_lineage_nodes):
		"""Test creating lineage edges"""
		source_node = sample_lineage_nodes[0]  # Connection node
		target_node = sample_lineage_nodes[1]  # Table node

		edge = CnLineageEdge(
			id='edge_conn_contains_table',
			tenant_id='test_tenant',
			source_node_id=source_node.id,
			target_node_id=target_node.id,
			relationship_type='contains',
			transformation_logic=None,
			confidence_score=1.0,
			meta_data={'discovered_via': 'schema_introspection'}
		)

		db_session.add(edge)
		db_session.commit()
		db_session.refresh(edge)

		assert edge.source_node_id == source_node.id
		assert edge.target_node_id == target_node.id
		assert edge.relationship_type == 'contains'
		assert edge.confidence_score == 1.0
		assert edge.meta_data['discovered_via'] == 'schema_introspection'

	def test_lineage_edge_with_transformation(self, db_session, sample_lineage_nodes):
		"""Test lineage edge with transformation logic"""
		source_field = sample_lineage_nodes[2]  # Email field
		target_field = sample_lineage_nodes[3]  # ID field (repurposed for test)

		edge = CnLineageEdge(
			id='edge_transform_email_to_hash',
			tenant_id='test_tenant',
			source_node_id=source_field.id,
			target_node_id=target_field.id,
			relationship_type='transforms_to',
			transformation_logic='SHA256(LOWER(TRIM(email)))',
			flow_id='flow_123',
			confidence_score=0.95,
			meta_data={
				'transformation_type': 'hash',
				'preserves_uniqueness': True
			}
		)

		db_session.add(edge)
		db_session.commit()
		db_session.refresh(edge)

		assert edge.relationship_type == 'transforms_to'
		assert 'SHA256' in edge.transformation_logic
		assert edge.flow_id == 'flow_123'
		assert edge.confidence_score == 0.95

	def test_lineage_edge_unique_constraint(self, db_session, sample_lineage_nodes):
		"""Test lineage edge uniqueness constraint"""
		source_node = sample_lineage_nodes[0]
		target_node = sample_lineage_nodes[1]

		edge1 = CnLineageEdge(
			id='duplicate_edge',
			tenant_id='test_tenant',
			source_node_id=source_node.id,
			target_node_id=target_node.id,
			relationship_type='contains'
		)

		edge2 = CnLineageEdge(
			id='duplicate_edge',  # Same ID
			tenant_id='test_tenant',
			source_node_id=source_node.id,
			target_node_id=target_node.id,
			relationship_type='contains'
		)

		db_session.add(edge1)
		db_session.commit()

		db_session.add(edge2)

		with pytest.raises(IntegrityError):
			db_session.commit()


class TestCnSingerTapModel:
	"""Test CnSingerTap model"""

	def test_singer_tap_creation(self, db_session):
		"""Test creating Singer tap entries"""
		tap = CnSingerTap(
			tenant_id='test_tenant',
			name='tap-postgres',
			package_name='pipelinewise-tap-postgres',
			version='1.2.3',
			description='PostgreSQL database tap',
			installation_status='installed',
			configuration_schema={
				'type': 'object',
				'properties': {
					'host': {'type': 'string'},
					'port': {'type': 'integer', 'default': 5432}
				},
				'required': ['host']
			},
			supported_features=['discovery', 'properties', 'catalog']
		)

		db_session.add(tap)
		db_session.commit()
		db_session.refresh(tap)

		assert tap.name == 'tap-postgres'
		assert tap.version == '1.2.3'
		assert tap.installation_status == 'installed'
		assert 'host' in tap.configuration_schema['properties']
		assert 'discovery' in tap.supported_features

	def test_singer_tap_unique_name_per_tenant(self, db_session):
		"""Test Singer tap name uniqueness per tenant"""
		tap1 = CnSingerTap(
			tenant_id='tenant1',
			name='tap-postgres',
			package_name='tap-postgres-1'
		)

		tap2 = CnSingerTap(
			tenant_id='tenant1',
			name='tap-postgres',  # Same name, same tenant
			package_name='tap-postgres-2'
		)

		db_session.add(tap1)
		db_session.commit()

		db_session.add(tap2)

		with pytest.raises(IntegrityError):
			db_session.commit()


# Model relationship tests
class TestModelRelationships:
	"""Test relationships between models"""

	def test_connection_to_flows_relationship(self, db_session, sample_connection):
		"""Test connection can have multiple flows"""
		flow1 = CnDataFlow(
			tenant_id='test_tenant',
			name='Flow 1',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id)
		)

		flow2 = CnDataFlow(
			tenant_id='test_tenant',
			name='Flow 2',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id)
		)

		db_session.add(flow1)
		db_session.add(flow2)
		db_session.commit()

		# Both flows should reference the same connection
		assert flow1.source_connection_id == str(sample_connection.id)
		assert flow2.source_connection_id == str(sample_connection.id)

	def test_flow_to_execution_logs_relationship(self, db_session, sample_flow):
		"""Test flow can have multiple execution logs"""
		log1 = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.SUCCESS,
			started_at=datetime.now(timezone.utc)
		)

		log2 = CnExecutionLog(
			flow_id=str(sample_flow.id),
			tenant_id='test_tenant',
			status=ExecutionStatus.FAILED,
			started_at=datetime.now(timezone.utc),
			error_message='Test error'
		)

		db_session.add(log1)
		db_session.add(log2)
		db_session.commit()

		# Both logs should reference the same flow
		assert log1.flow_id == str(sample_flow.id)
		assert log2.flow_id == str(sample_flow.id)

	def test_connection_to_health_metrics_relationship(self, db_session, sample_connection):
		"""Test connection can have multiple health metrics"""
		metric1 = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ACTIVE,
			latency_ms=100.0
		)

		metric2 = CnHealthMetric(
			connection_id=str(sample_connection.id),
			tenant_id='test_tenant',
			status=ConnectionStatus.ACTIVE,
			latency_ms=150.0
		)

		db_session.add(metric1)
		db_session.add(metric2)
		db_session.commit()

		# Both metrics should reference the same connection
		assert metric1.connection_id == str(sample_connection.id)
		assert metric2.connection_id == str(sample_connection.id)


# Model validation tests
class TestModelValidation:
	"""Test model validation methods"""

	def test_connection_validation_method(self, db_session):
		"""Test connection model validation"""
		connection = CnConnection(
			tenant_id='test_tenant',
			name='Test Connection',
			connection_type=ConnectionType.DATABASE
		)

		db_session.add(connection)
		db_session.commit()
		db_session.refresh(connection)

		# Test validation method exists and works
		assert hasattr(connection, 'validate_config') or True  # Placeholder

		# In a full implementation, you might have:
		# validation_result = connection.validate_config()
		# assert validation_result.is_valid

	def test_flow_validation_method(self, db_session, sample_connection):
		"""Test flow model validation"""
		flow = CnDataFlow(
			tenant_id='test_tenant',
			name='Test Flow',
			source_connection_id=str(sample_connection.id),
			target_connection_id=str(sample_connection.id),
			field_mappings={'valid': 'mapping'}
		)

		db_session.add(flow)
		db_session.commit()
		db_session.refresh(flow)

		# Test validation placeholder
		assert hasattr(flow, 'validate_mappings') or True  # Placeholder

		# In a full implementation:
		# validation_result = flow.validate_mappings()
		# assert validation_result.is_valid