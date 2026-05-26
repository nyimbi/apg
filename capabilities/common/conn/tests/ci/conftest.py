"""
Test configuration and fixtures for Connection Management capability
Provides shared fixtures for database sessions, mock data, and test utilities

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import pytest
import tempfile
import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ...sqlalchemy_models import Base, CnConnection, CnDataFlow, CnLineageNode, CnLineageEdge
from ...sqlalchemy_models import ConnectionType, ConnectionStatus, SyncMode, LineageNodeType
from ...service import ConnectionManager, FlowExecutor, IntelligentConnector
from ...service_bridge import ServiceBridge
from ...lineage_engine import DataLineageEngine


@pytest.fixture
def event_loop():
	"""Create an instance of the default event loop for the test session."""
	loop = asyncio.get_event_loop_policy().new_event_loop()
	yield loop
	loop.close()


@pytest.fixture
def db_engine():
	"""Create in-memory SQLite database engine for testing"""
	engine = create_engine(
		"sqlite:///:memory:",
		connect_args={
			"check_same_thread": False,
		},
		poolclass=StaticPool,
		echo=False  # Set to True for SQL debugging
	)

	# Create all tables
	Base.metadata.create_all(engine)

	yield engine

	# Cleanup
	Base.metadata.drop_all(engine)
	engine.dispose()


@pytest.fixture
def db_session(db_engine):
	"""Create database session for testing"""
	SessionLocal = sessionmaker(bind=db_engine)
	session = SessionLocal()

	yield session

	session.rollback()
	session.close()


@pytest.fixture
def sample_connection_data():
	"""Sample connection configuration data"""
	return {
		'name': 'Test PostgreSQL Connection',
		'description': 'Test database connection',
		'connection_type': ConnectionType.DATABASE,
		'singer_tap': 'tap-postgres',
		'sync_mode': SyncMode.INCREMENTAL,
		'batch_size': 1000,
		'enabled': True,
		'tap_config': {
			'host': 'localhost',
			'port': 5432,
			'database': 'testdb',
			'user': 'testuser',
			'password': 'testpass'
		},
		'target_config': {
			'connection_string': 'postgresql://user:pass@localhost/target'
		}
	}


@pytest.fixture
def sample_connection(db_session, sample_connection_data):
	"""Create a sample connection in database"""
	connection = CnConnection(
		tenant_id='test_tenant',
		name=sample_connection_data['name'],
		description=sample_connection_data['description'],
		connection_type=sample_connection_data['connection_type'],
		status=ConnectionStatus.ACTIVE,
		singer_tap=sample_connection_data['singer_tap'],
		sync_mode=sample_connection_data['sync_mode'],
		batch_size=sample_connection_data['batch_size'],
		enabled=sample_connection_data['enabled'],
		tap_config=sample_connection_data['tap_config'],
		target_config=sample_connection_data['target_config']
	)

	db_session.add(connection)
	db_session.commit()
	db_session.refresh(connection)

	return connection


@pytest.fixture
def sample_flow_data(sample_connection):
	"""Sample data flow configuration"""
	return {
		'name': 'Test Data Flow',
		'description': 'Test flow from postgres to warehouse',
		'source_connection_id': str(sample_connection.id),
		'target_connection_id': str(sample_connection.id),  # Using same for simplicity
		'field_mappings': {
			'users.id': 'customers.customer_id',
			'users.email': 'customers.email_address',
			'users.name': 'customers.full_name'
		},
		'transformation_config': {
			'transformations': [
				{
					'type': 'filter',
					'conditions': [
						{'field': 'active', 'operator': 'equals', 'value': True}
					]
				},
				{
					'type': 'map',
					'mappings': {
						'first_name,last_name': 'full_name'
					}
				}
			]
		},
		'schedule_expression': '0 2 * * *',  # Daily at 2 AM
		'enabled': True
	}


@pytest.fixture
def sample_flow(db_session, sample_flow_data):
	"""Create a sample data flow in database"""
	flow = CnDataFlow(
		tenant_id='test_tenant',
		name=sample_flow_data['name'],
		description=sample_flow_data['description'],
		source_connection_id=sample_flow_data['source_connection_id'],
		target_connection_id=sample_flow_data['target_connection_id'],
		field_mappings=sample_flow_data['field_mappings'],
		transformation_config=sample_flow_data['transformation_config'],
		schedule_expression=sample_flow_data['schedule_expression'],
		enabled=sample_flow_data['enabled']
	)

	db_session.add(flow)
	db_session.commit()
	db_session.refresh(flow)

	return flow


@pytest.fixture
def sample_lineage_nodes(db_session, sample_connection):
	"""Create sample lineage nodes for testing"""
	nodes = []

	# Connection node
	conn_node = CnLineageNode(
		id=f"conn_{sample_connection.id}",
		tenant_id='test_tenant',
		name=sample_connection.name,
		node_type=LineageNodeType.CONNECTION,
		connection_id=str(sample_connection.id),
		sensitive=False,
		meta_data={'connection_type': sample_connection.connection_type.value}
	)
	nodes.append(conn_node)

	# Table node
	table_node = CnLineageNode(
		id="test_table_users",
		tenant_id='test_tenant',
		name="users",
		node_type=LineageNodeType.TABLE,
		connection_id=str(sample_connection.id),
		schema_name="public",
		table_name="users",
		sensitive=False,
		meta_data={'record_count': 1000}
	)
	nodes.append(table_node)

	# Field nodes with sensitive data
	email_field = CnLineageNode(
		id="test_field_users_email",
		tenant_id='test_tenant',
		name="users.email",
		node_type=LineageNodeType.FIELD,
		connection_id=str(sample_connection.id),
		schema_name="public",
		table_name="users",
		field_name="email",
		sensitive=True,
		pii_classification="email_address",
		meta_data={'data_type': 'varchar', 'nullable': False}
	)
	nodes.append(email_field)

	id_field = CnLineageNode(
		id="test_field_users_id",
		tenant_id='test_tenant',
		name="users.id",
		node_type=LineageNodeType.FIELD,
		connection_id=str(sample_connection.id),
		schema_name="public",
		table_name="users",
		field_name="id",
		sensitive=False,
		meta_data={'data_type': 'integer', 'primary_key': True}
	)
	nodes.append(id_field)

	# Add all nodes to session
	for node in nodes:
		db_session.add(node)

	db_session.commit()

	return nodes


@pytest.fixture
def sample_lineage_edges(db_session, sample_lineage_nodes):
	"""Create sample lineage edges for testing"""
	edges = []

	# Connection contains table
	conn_table_edge = CnLineageEdge(
		id="edge_conn_contains_table",
		tenant_id='test_tenant',
		source_node_id=sample_lineage_nodes[0].id,  # Connection
		target_node_id=sample_lineage_nodes[1].id,  # Table
		relationship_type="contains",
		confidence_score=1.0,
		meta_data={'discovered_via': 'schema_introspection'}
	)
	edges.append(conn_table_edge)

	# Table contains fields
	table_email_edge = CnLineageEdge(
		id="edge_table_contains_email",
		tenant_id='test_tenant',
		source_node_id=sample_lineage_nodes[1].id,  # Table
		target_node_id=sample_lineage_nodes[2].id,  # Email field
		relationship_type="contains",
		confidence_score=1.0,
		meta_data={'field_type': 'varchar'}
	)
	edges.append(table_email_edge)

	table_id_edge = CnLineageEdge(
		id="edge_table_contains_id",
		tenant_id='test_tenant',
		source_node_id=sample_lineage_nodes[1].id,  # Table
		target_node_id=sample_lineage_nodes[3].id,  # ID field
		relationship_type="contains",
		confidence_score=1.0,
		meta_data={'field_type': 'integer', 'primary_key': True}
	)
	edges.append(table_id_edge)

	# Add all edges to session
	for edge in edges:
		db_session.add(edge)

	db_session.commit()

	return edges


@pytest.fixture
async def connection_manager(db_session):
	"""Create ConnectionManager instance for testing"""
	manager = ConnectionManager()
	await manager.initialize()

	# Mock database session
	manager.db_session = db_session

	return manager


@pytest.fixture
def service_bridge(connection_manager):
	"""Create ServiceBridge instance for testing"""
	bridge = ServiceBridge()
	bridge._connection_manager = connection_manager
	bridge._initialized = True

	return bridge


@pytest.fixture
def lineage_engine(db_session):
	"""Create DataLineageEngine instance for testing"""
	engine = DataLineageEngine(db_session=db_session)
	return engine


@pytest.fixture
def mock_singer_discovery():
	"""Mock Singer.io catalog discovery"""
	return {
		'streams': [
			{
				'tap_stream_id': 'users',
				'schema': {
					'properties': {
						'id': {'type': 'integer'},
						'email': {'type': 'string'},
						'first_name': {'type': 'string'},
						'last_name': {'type': 'string'},
						'created_at': {'type': 'string', 'format': 'date-time'}
					}
				},
				'metadata': [
					{'breadcrumb': [], 'metadata': {'table-key-properties': ['id']}}
				]
			},
			{
				'tap_stream_id': 'orders',
				'schema': {
					'properties': {
						'id': {'type': 'integer'},
						'user_id': {'type': 'integer'},
						'total': {'type': 'number'},
						'status': {'type': 'string'},
						'created_at': {'type': 'string', 'format': 'date-time'}
					}
				}
			}
		]
	}


@pytest.fixture
def mock_flask_app():
	"""Mock Flask application for view testing"""
	from flask import Flask
	from flask_appbuilder import AppBuilder
	from flask_sqlalchemy import SQLAlchemy

	app = Flask(__name__)
	app.config['SECRET_KEY'] = 'test-secret-key'
	app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
	app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
	db = SQLAlchemy()
	db.init_app(app)

	with app.app_context():
		db.create_all()
		appbuilder = AppBuilder(app, db.session)
		appbuilder.sm.has_access = lambda permission_name, view_name: True
		yield app, appbuilder


@pytest.fixture
def mock_performance_metrics():
	"""Mock performance metrics data"""
	return {
		'system_metrics': {
			'cpu_usage': 45.2,
			'memory_usage': 67.8,
			'disk_usage': 23.1
		},
		'connection_metrics': {
			'total_connections': 15,
			'active_connections': 12,
			'error_connections': 1,
			'avg_latency_ms': 234.5,
			'throughput_rps': 1247.3
		},
		'flow_metrics': {
			'total_flows': 8,
			'running_flows': 3,
			'successful_executions_24h': 156,
			'failed_executions_24h': 2
		}
	}


# Test utilities
class AsyncContextManager:
	"""Helper for async context manager testing"""

	def __init__(self, return_value=None):
		self.return_value = return_value

	async def __aenter__(self):
		return self.return_value

	async def __aexit__(self, *args):
		pass


def create_mock_async_function(return_value=None, side_effect=None):
	"""Create a mock async function with optional return value or side effect"""
	mock = AsyncMock()

	if side_effect:
		mock.side_effect = side_effect
	else:
		mock.return_value = return_value

	return mock


def assert_connection_data(connection: CnConnection, expected_data: Dict[str, Any]):
	"""Assert connection object matches expected data"""
	assert connection.name == expected_data['name']
	assert connection.description == expected_data['description']
	assert connection.connection_type == expected_data['connection_type']
	assert connection.singer_tap == expected_data['singer_tap']
	assert connection.sync_mode == expected_data['sync_mode']
	assert connection.batch_size == expected_data['batch_size']
	assert connection.enabled == expected_data['enabled']


def assert_lineage_structure(nodes: List, edges: List, expected_structure: Dict[str, Any]):
	"""Assert lineage structure matches expected format"""
	assert len(nodes) == expected_structure.get('node_count', 0)
	assert len(edges) == expected_structure.get('edge_count', 0)

	# Check node types
	node_types = {node['type'] for node in nodes}
	expected_types = set(expected_structure.get('node_types', []))
	assert node_types == expected_types

	# Check sensitive data presence
	sensitive_nodes = [node for node in nodes if node.get('metadata', {}).get('sensitive')]
	assert len(sensitive_nodes) == expected_structure.get('sensitive_count', 0)


# Mock external service responses
@pytest.fixture
def mock_singer_tap_response():
	"""Mock response from Singer tap execution"""
	return {
		'status': 'success',
		'records_processed': 1500,
		'execution_time': 45.2,
		'state': {'bookmarks': {'users': {'created_at': '2025-01-01T00:00:00Z'}}},
		'schema': {
			'users': {
				'type': 'SCHEMA',
				'stream': 'users',
				'schema': {
					'properties': {
						'id': {'type': 'integer'},
						'email': {'type': 'string'}
					}
				}
			}
		}
	}


@pytest.fixture
def mock_ai_suggestions():
	"""Mock AI-powered field mapping suggestions"""
	return {
		'suggestions': [
			{
				'source_field': 'first_name',
				'target_field': 'fname',
				'confidence': 0.95,
				'reasoning': 'Semantic similarity and naming pattern match'
			},
			{
				'source_field': 'email_address',
				'target_field': 'email',
				'confidence': 0.99,
				'reasoning': 'Exact semantic match for email field'
			}
		],
		'transformation_suggestions': [
			{
				'type': 'data_type_conversion',
				'field': 'created_date',
				'suggestion': 'Convert to ISO datetime format',
				'confidence': 0.87
			}
		]
	}
