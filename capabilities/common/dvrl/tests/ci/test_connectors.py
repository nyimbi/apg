#!/usr/bin/env python3
"""
Tests for Universal Connector Framework
Basic functionality tests for the connector system

Author: APG Platform Team
Copyright: © 2025 Datacraft
"""

import asyncio
import pytest
from typing import Dict, Any

from ..models import DataSource, DataSourceType, DataSourceStatus
from ..connectors import (
	ConnectorFactory, UniversalConnectorManager, SQLDatabaseConnector,
	NoSQLConnector, APIConnector, StreamingConnector, ConnectionCapability
)


@pytest.fixture
def sample_postgresql_config() -> Dict[str, Any]:
	"""Sample PostgreSQL data source configuration"""
	return {
		'name': 'test_postgresql',
		'description': 'Test PostgreSQL database',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {
			'host': 'localhost',
			'port': 5432,
			'database': 'testdb',
			'username': 'testuser',
			'password': 'testpass'
		},
		'host': 'localhost',
		'port': 5432,
		'database': 'testdb'
	}


@pytest.fixture  
def sample_mongodb_config() -> Dict[str, Any]:
	"""Sample MongoDB data source configuration"""
	return {
		'name': 'test_mongodb',
		'description': 'Test MongoDB database',
		'type': DataSourceType.MONGODB,
		'connection_config': {
			'host': 'localhost',
			'port': 27017,
			'database': 'testdb',
			'username': 'testuser',
			'password': 'testpass'
		},
		'host': 'localhost',
		'port': 27017,
		'database': 'testdb'
	}


@pytest.fixture
def sample_api_config() -> Dict[str, Any]:
	"""Sample REST API data source configuration"""
	return {
		'name': 'test_rest_api',
		'description': 'Test REST API',
		'type': DataSourceType.REST_API,
		'connection_config': {
			'base_url': 'https://api.example.com',
			'auth_type': 'bearer',
			'api_key': 'test-api-key'
		}
	}


async def test_sql_connector_creation():
	"""Test SQL database connector creation and basic functionality"""
	config = {
		'name': 'test_postgresql',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost', 'port': 5432}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = ConnectorFactory.create_connector(data_source, 'test-tenant', 'test-user')
	
	assert isinstance(connector, SQLDatabaseConnector)
	assert connector.data_source.name == 'test_postgresql'
	assert connector.tenant_id == 'test-tenant'
	assert connector.user_id == 'test-user'


async def test_nosql_connector_creation():
	"""Test NoSQL database connector creation"""
	config = {
		'name': 'test_mongodb',
		'type': DataSourceType.MONGODB,
		'connection_config': {'host': 'localhost', 'port': 27017}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = ConnectorFactory.create_connector(data_source, 'test-tenant', 'test-user')
	
	assert isinstance(connector, NoSQLConnector)
	assert connector.data_source.type == DataSourceType.MONGODB


async def test_api_connector_creation():
	"""Test API connector creation"""
	config = {
		'name': 'test_api',
		'type': DataSourceType.REST_API,
		'connection_config': {'base_url': 'https://api.example.com'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = ConnectorFactory.create_connector(data_source, 'test-tenant', 'test-user')
	
	assert isinstance(connector, APIConnector)
	assert connector.data_source.type == DataSourceType.REST_API


async def test_streaming_connector_creation():
	"""Test streaming connector creation"""
	config = {
		'name': 'test_bytewax',
		'type': DataSourceType.BYTEWAX,
		'connection_config': {'streams': ['orders']}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = ConnectorFactory.create_connector(data_source, 'test-tenant', 'test-user')
	
	assert isinstance(connector, StreamingConnector)
	assert connector.data_source.type == DataSourceType.BYTEWAX


async def test_sql_connector_capabilities():
	"""Test SQL connector capabilities discovery"""
	config = {
		'name': 'test_postgresql',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = SQLDatabaseConnector(data_source, 'test-tenant', 'test-user')
	capabilities = await connector.get_capabilities()
	
	assert ConnectionCapability.BATCH_READ in capabilities
	assert ConnectionCapability.BATCH_WRITE in capabilities
	assert ConnectionCapability.TRANSACTION_SUPPORT in capabilities
	assert ConnectionCapability.SCHEMA_INTROSPECTION in capabilities
	assert ConnectionCapability.QUERY_PUSHDOWN in capabilities


async def test_connector_connection_lifecycle():
	"""Test connector connection lifecycle"""
	config = {
		'name': 'test_db',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = SQLDatabaseConnector(data_source, 'test-tenant', 'test-user')
	
	# Test connection
	connected = await connector.connect()
	assert connected == True
	
	# Test health check
	health_result = await connector.test_connection()
	assert health_result == True
	
	# Test disconnection
	disconnected = await connector.disconnect()
	assert disconnected == True


async def test_schema_discovery():
	"""Test automatic schema discovery"""
	config = {
		'name': 'test_db',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = SQLDatabaseConnector(data_source, 'test-tenant', 'test-user')
	await connector.connect()
	
	schema = await connector.discover_schema()
	
	assert schema is not None
	assert schema.data_source_id == data_source.id
	assert len(schema.tables) > 0
	assert schema.discovery_method == "sql_introspection"
	assert schema.confidence_score > 0.8


async def test_query_execution():
	"""Test query execution through connector"""
	config = {
		'name': 'test_db',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	connector = SQLDatabaseConnector(data_source, 'test-tenant', 'test-user')
	await connector.connect()
	
	result = await connector.execute_query("SELECT * FROM users LIMIT 10")
	
	assert 'query' in result
	assert 'results' in result
	assert 'row_count' in result
	assert 'execution_time_ms' in result
	assert result['row_count'] > 0


async def test_universal_connector_manager():
	"""Test universal connector manager functionality"""
	manager = UniversalConnectorManager('test-tenant', 'test-user')
	
	# Create test data source
	config = {
		'name': 'test_manager_db',
		'type': DataSourceType.POSTGRESQL,
		'connection_config': {'host': 'localhost'}
	}
	
	data_source = DataSource(
		tenant_id='test-tenant',
		created_by='test-user',
		**config
	)
	
	# Create connector through manager
	connector = await manager.create_connector(data_source)
	
	assert connector is not None
	assert data_source.id in manager.active_connectors
	
	# Test retrieval
	retrieved_connector = await manager.get_connector(data_source.id)
	assert retrieved_connector == connector
	
	# Test health check
	health_results = await manager.health_check_all()
	assert data_source.id in health_results
	
	# Test schema discovery
	schemas = await manager.discover_all_schemas()
	assert data_source.id in schemas
	
	# Test connector statistics
	stats = await manager.get_connector_stats()
	assert stats['total_connectors'] == 1
	assert 'connector_types' in stats
	assert 'health_summary' in stats
	
	# Test removal
	removed = await manager.remove_connector(data_source.id)
	assert removed == True
	assert data_source.id not in manager.active_connectors


async def test_connector_factory_supported_types():
	"""Test connector factory supported types"""
	supported_types = ConnectorFactory.get_supported_types()
	
	assert DataSourceType.POSTGRESQL in supported_types
	assert DataSourceType.MONGODB in supported_types
	assert DataSourceType.REST_API in supported_types
	assert DataSourceType.BYTEWAX in supported_types
	
	assert len(supported_types) > 10  # Should support many types


async def test_connector_error_handling():
	"""Test connector error handling for unsupported types"""
	# Create data source with unsupported type
	config = {
		'name': 'test_unsupported',
		'type': 'UNSUPPORTED_TYPE',  # This should cause an error
		'connection_config': {}
	}
	
	# This will create a DataSource but the factory should reject it
	with pytest.raises((ValueError, KeyError)):
		# Create a mock data source with invalid type
		data_source = DataSource(
			tenant_id='test-tenant',
			created_by='test-user',
			name='test_unsupported',
			type='INVALID_TYPE',
			connection_config={}
		)
		ConnectorFactory.create_connector(data_source, 'test-tenant', 'test-user')


if __name__ == "__main__":
	# Run basic tests
	loop = asyncio.get_event_loop()
	
	print("Testing SQL Connector Creation...")
	loop.run_until_complete(test_sql_connector_creation())
	print("✓ SQL Connector Creation test passed")
	
	print("Testing NoSQL Connector Creation...")
	loop.run_until_complete(test_nosql_connector_creation())
	print("✓ NoSQL Connector Creation test passed")
	
	print("Testing API Connector Creation...")
	loop.run_until_complete(test_api_connector_creation())
	print("✓ API Connector Creation test passed")
	
	print("Testing Connector Capabilities...")
	loop.run_until_complete(test_sql_connector_capabilities())
	print("✓ Connector Capabilities test passed")
	
	print("Testing Connection Lifecycle...")
	loop.run_until_complete(test_connector_connection_lifecycle())
	print("✓ Connection Lifecycle test passed")
	
	print("Testing Schema Discovery...")
	loop.run_until_complete(test_schema_discovery())
	print("✓ Schema Discovery test passed")
	
	print("Testing Query Execution...")
	loop.run_until_complete(test_query_execution())
	print("✓ Query Execution test passed")
	
	print("Testing Universal Connector Manager...")
	loop.run_until_complete(test_universal_connector_manager())
	print("✓ Universal Connector Manager test passed")
	
	print("Testing Factory Supported Types...")
	loop.run_until_complete(test_connector_factory_supported_types())
	print("✓ Factory Supported Types test passed")
	
	print("\n🎉 All connector framework tests passed!")
