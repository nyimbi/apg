#!/usr/bin/env python3
"""
Unit tests for DVRL database connectors
Tests the production client implementations for all database types
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

# Import test dependencies
import aiohttp
from aioresponses import aioresponses

# Import the modules under test
from dvrl.connectors import (
	SQLDatabaseConnector,
	NoSQLConnector,
	APIConnector,
	StreamingConnector,
	ConnectorFactory,
	UniversalConnectorManager,
	ConnectionCapability,
	ConnectionHealth
)
from dvrl.models import DataSource, DataSourceType, DataSourceStatus


class TestSQLDatabaseConnector:
	"""Test the SQL database connector implementations"""
	
	@pytest.fixture
	def postgresql_datasource(self):
		"""Create a PostgreSQL data source for testing"""
		return DataSource(
			id="pg_test_1",
			name="Test PostgreSQL",
			type=DataSourceType.POSTGRESQL,
			connection_config={
				'host': 'localhost',
				'port': 5432,
				'database': 'testdb',
				'username': 'testuser',
				'password': 'testpass'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	def mysql_datasource(self):
		"""Create a MySQL data source for testing"""
		return DataSource(
			id="mysql_test_1",
			name="Test MySQL",
			type=DataSourceType.MYSQL,
			connection_config={
				'host': 'localhost',
				'port': 3306,
				'database': 'testdb',
				'username': 'testuser',
				'password': 'testpass'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	async def postgres_connector(self, postgresql_datasource):
		"""Create PostgreSQL connector for testing"""
		connector = SQLDatabaseConnector(postgresql_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	@pytest.fixture
	async def mysql_connector(self, mysql_datasource):
		"""Create MySQL connector for testing"""
		connector = SQLDatabaseConnector(mysql_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	async def test_postgresql_connection_string_building(self, postgres_connector):
		"""Test PostgreSQL connection string construction"""
		conn_string = postgres_connector._build_connection_string()
		
		assert 'postgresql+asyncpg' in conn_string
		assert 'testuser:testpass' in conn_string
		assert 'localhost:5432' in conn_string
		assert 'testdb' in conn_string
	
	async def test_mysql_connection_string_building(self, mysql_connector):
		"""Test MySQL connection string construction"""
		conn_string = mysql_connector._build_connection_string()
		
		assert 'mysql+aiomysql' in conn_string
		assert 'testuser:testpass' in conn_string
		assert 'localhost:3306' in conn_string
		assert 'testdb' in conn_string
	
	@patch('dvrl.connectors.asyncpg.create_pool')
	async def test_postgresql_connect_success(self, mock_create_pool, postgres_connector):
		"""Test successful PostgreSQL connection"""
		mock_pool = AsyncMock()
		mock_create_pool.return_value = mock_pool
		
		# Mock successful connection test
		mock_conn = AsyncMock()
		mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
		mock_conn.execute = AsyncMock()
		
		success = await postgres_connector.connect()
		
		assert success is True
		assert postgres_connector.connection_pool is not None
		assert postgres_connector.connection_metadata['database_type'] == 'postgresql'
		mock_create_pool.assert_called_once()
	
	@patch('dvrl.connectors.aiomysql.create_pool')
	async def test_mysql_connect_success(self, mock_create_pool, mysql_connector):
		"""Test successful MySQL connection"""
		mock_pool = AsyncMock()
		mock_create_pool.return_value = mock_pool
		
		# Mock successful connection test
		mock_conn = AsyncMock()
		mock_cursor = AsyncMock()
		mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
		mock_conn.cursor.return_value.__aenter__.return_value = mock_cursor
		mock_cursor.execute = AsyncMock()
		
		success = await mysql_connector.connect()
		
		assert success is True
		assert postgres_connector.connection_pool is not None
		assert mysql_connector.connection_metadata['database_type'] == 'mysql'
		mock_create_pool.assert_called_once()
	
	@patch('dvrl.connectors.asyncpg.create_pool')
	async def test_postgresql_schema_discovery(self, mock_create_pool, postgres_connector):
		"""Test PostgreSQL schema discovery"""
		# Mock connection pool and query results
		mock_pool = AsyncMock()
		mock_create_pool.return_value = mock_pool
		postgres_connector.connection_pool = mock_pool
		
		mock_conn = AsyncMock()
		mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
		
		# Mock schema discovery results
		mock_rows = [
			{
				'table_name': 'users',
				'table_type': 'BASE TABLE',
				'column_name': 'id',
				'data_type': 'integer',
				'is_nullable': 'NO',
				'column_default': None,
				'constraint_type': 'PRIMARY KEY'
			},
			{
				'table_name': 'users',
				'table_type': 'BASE TABLE',
				'column_name': 'username',
				'data_type': 'character varying',
				'is_nullable': 'NO',
				'column_default': None,
				'constraint_type': None
			}
		]
		mock_conn.fetch.return_value = mock_rows
		
		schema = await postgres_connector.discover_schema()
		
		assert schema.data_source_id == "pg_test_1"
		assert len(schema.tables) == 1
		assert schema.tables[0]['name'] == 'users'
		assert len(schema.tables[0]['columns']) == 2
		assert schema.discovery_method == 'postgresql_introspection'
		assert schema.confidence_score == 0.98
	
	@patch('dvrl.connectors.asyncpg.create_pool')
	async def test_postgresql_query_execution(self, mock_create_pool, postgres_connector):
		"""Test PostgreSQL query execution"""
		# Mock connection pool
		mock_pool = AsyncMock()
		mock_create_pool.return_value = mock_pool
		postgres_connector.connection_pool = mock_pool
		
		mock_conn = AsyncMock()
		mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
		
		# Mock query results
		mock_result = [
			{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
			{'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
		]
		mock_conn.fetch.return_value = mock_result
		
		result = await postgres_connector.execute_query("SELECT * FROM users", {'limit': 10})
		
		assert result['database_type'] == 'postgresql'
		assert result['row_count'] == 2
		assert len(result['results']) == 2
		assert result['results'][0]['name'] == 'Alice'
		assert 'execution_time_ms' in result
		assert result['columns'] == ['id', 'name', 'email']
	
	async def test_sql_capabilities(self, postgres_connector):
		"""Test SQL database capabilities"""
		capabilities = await postgres_connector.get_capabilities()
		
		assert ConnectionCapability.BATCH_READ in capabilities
		assert ConnectionCapability.BATCH_WRITE in capabilities
		assert ConnectionCapability.TRANSACTION_SUPPORT in capabilities
		assert ConnectionCapability.SCHEMA_INTROSPECTION in capabilities
		assert ConnectionCapability.QUERY_PUSHDOWN in capabilities
		assert ConnectionCapability.AGGREGATION_PUSHDOWN in capabilities
		assert ConnectionCapability.JOIN_PUSHDOWN in capabilities
		assert ConnectionCapability.LIMIT_PUSHDOWN in capabilities
		assert ConnectionCapability.FULL_TEXT_SEARCH in capabilities  # PostgreSQL specific


class TestNoSQLConnector:
	"""Test the NoSQL database connector implementations"""
	
	@pytest.fixture
	def mongodb_datasource(self):
		"""Create a MongoDB data source for testing"""
		return DataSource(
			id="mongo_test_1",
			name="Test MongoDB",
			type=DataSourceType.MONGODB,
			connection_config={
				'host': 'localhost',
				'port': 27017,
				'database': 'testdb',
				'username': 'testuser',
				'password': 'testpass'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	def elasticsearch_datasource(self):
		"""Create an Elasticsearch data source for testing"""
		return DataSource(
			id="es_test_1",
			name="Test Elasticsearch",
			type=DataSourceType.ELASTICSEARCH,
			connection_config={
				'host': 'localhost',
				'port': 9200,
				'scheme': 'http',
				'username': 'elastic',
				'password': 'testpass'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	async def mongodb_connector(self, mongodb_datasource):
		"""Create MongoDB connector for testing"""
		connector = NoSQLConnector(mongodb_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	@pytest.fixture
	async def elasticsearch_connector(self, elasticsearch_datasource):
		"""Create Elasticsearch connector for testing"""
		connector = NoSQLConnector(elasticsearch_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	@patch('dvrl.connectors.motor.motor_asyncio.AsyncIOMotorClient')
	async def test_mongodb_connect_success(self, mock_motor_client, mongodb_connector):
		"""Test successful MongoDB connection"""
		mock_client = AsyncMock()
		mock_database = AsyncMock()
		mock_client.__getitem__.return_value = mock_database
		mock_database.command = AsyncMock(return_value={'ok': 1})
		mock_motor_client.return_value = mock_client
		
		success = await mongodb_connector.connect()
		
		assert success is True
		assert mongodb_connector.client is not None
		assert mongodb_connector.database is not None
		mock_motor_client.assert_called_once()
		mock_database.command.assert_called_once_with('ping')
	
	async def test_elasticsearch_connect_success(self, elasticsearch_connector):
		"""Test successful Elasticsearch connection"""
		with aioresponses() as mock_resp:
			mock_resp.get(
				'http://localhost:9200/_cluster/health',
				payload={'status': 'green'}
			)
			
			success = await elasticsearch_connector.connect()
			
			assert success is True
			assert elasticsearch_connector.client is not None
			assert elasticsearch_connector.base_url == 'http://localhost:9200'
	
	@patch('dvrl.connectors.motor.motor_asyncio.AsyncIOMotorClient')
	async def test_mongodb_schema_discovery(self, mock_motor_client, mongodb_connector):
		"""Test MongoDB schema discovery"""
		# Mock client setup
		mock_client = AsyncMock()
		mock_database = AsyncMock()
		mock_client.__getitem__.return_value = mock_database
		mock_motor_client.return_value = mock_client
		mongodb_connector.client = mock_client
		mongodb_connector.database = mock_database
		
		# Mock collection discovery
		mock_database.list_collection_names.return_value = ['users', 'orders']
		
		# Mock collection stats
		mock_database.command.side_effect = [
			{'count': 1000, 'avgObjSize': 256},  # users stats
			{'count': 500, 'avgObjSize': 512}    # orders stats
		]
		
		# Mock sample documents
		mock_users_collection = AsyncMock()
		mock_orders_collection = AsyncMock()
		mock_database.__getitem__.side_effect = lambda name: {
			'users': mock_users_collection,
			'orders': mock_orders_collection
		}[name]
		
		# Mock find cursors
		async def mock_users_cursor():
			yield {'_id': 'user1', 'username': 'alice', 'email': 'alice@example.com', 'age': 25}
		
		async def mock_orders_cursor():
			yield {'_id': 'order1', 'user_id': 'user1', 'amount': 99.99, 'items': ['item1', 'item2']}
		
		mock_users_collection.find.return_value.limit.return_value = mock_users_cursor()
		mock_orders_collection.find.return_value.limit.return_value = mock_orders_cursor()
		
		schema = await mongodb_connector.discover_schema()
		
		assert schema.data_source_id == "mongo_test_1"
		assert len(schema.tables) == 2
		assert schema.discovery_method == 'mongodb_introspection'
		
		# Check users collection
		users_collection = next(c for c in schema.tables if c['name'] == 'users')
		assert users_collection['type'] == 'collection'
		assert users_collection['document_count'] == 1000
		assert 'sample_document' in users_collection
	
	@patch('dvrl.connectors.motor.motor_asyncio.AsyncIOMotorClient')
	async def test_mongodb_query_execution(self, mock_motor_client, mongodb_connector):
		"""Test MongoDB query execution"""
		# Setup mock client
		mock_client = AsyncMock()
		mock_database = AsyncMock()
		mock_collection = AsyncMock()
		mock_client.__getitem__.return_value = mock_database
		mock_database.__getitem__.return_value = mock_collection
		mock_motor_client.return_value = mock_client
		
		mongodb_connector.client = mock_client
		mongodb_connector.database = mock_database
		
		# Mock find results
		async def mock_cursor():
			yield {'_id': 'user1', 'name': 'Alice', 'age': 25}
			yield {'_id': 'user2', 'name': 'Bob', 'age': 30}
		
		mock_collection.find.return_value.limit.return_value = mock_cursor()
		
		# Test find query
		result = await mongodb_connector.execute_query('users.find({"age": {"$gte": 18}})', {'collection': 'users'})
		
		assert result['database_type'] == 'mongodb'
		assert result['document_count'] == 2
		assert len(result['results']) == 2
		assert result['results'][0]['name'] == 'Alice'
		assert result['query_type'] == 'mongodb_find'
	
	async def test_elasticsearch_query_execution(self, elasticsearch_connector):
		"""Test Elasticsearch query execution"""
		with aioresponses() as mock_resp:
			# Mock health check for connection
			mock_resp.get(
				'http://localhost:9200/_cluster/health',
				payload={'status': 'green'}
			)
			
			# Connect first
			await elasticsearch_connector.connect()
			
			# Mock search response
			search_response = {
				'hits': {
					'total': {'value': 2},
					'hits': [
						{
							'_id': 'doc1',
							'_score': 1.0,
							'_source': {'name': 'Alice', 'age': 25}
						},
						{
							'_id': 'doc2',
							'_score': 0.8,
							'_source': {'name': 'Bob', 'age': 30}
						}
					]
				}
			}
			
			mock_resp.post(
				'http://localhost:9200/users/_search',
				payload=search_response
			)
			
			# Test search query
			query = '{"query": {"match": {"name": "Alice"}}}'
			result = await elasticsearch_connector.execute_query(query, {'index': 'users'})
			
			assert result['database_type'] == 'elasticsearch'
			assert result['document_count'] == 2
			assert result['results'][0]['name'] == 'Alice'
			assert result['results'][0]['_score'] == 1.0
			assert result['query_type'] == 'elasticsearch_search'
	
	async def test_nosql_capabilities(self, mongodb_connector):
		"""Test NoSQL database capabilities"""
		capabilities = await mongodb_connector.get_capabilities()
		
		assert ConnectionCapability.BATCH_READ in capabilities
		assert ConnectionCapability.BATCH_WRITE in capabilities
		assert ConnectionCapability.SCHEMA_INTROSPECTION in capabilities
		assert ConnectionCapability.FULL_TEXT_SEARCH in capabilities  # MongoDB specific
		assert ConnectionCapability.AGGREGATION_PUSHDOWN in capabilities


class TestAPIConnector:
	"""Test the API connector implementations"""
	
	@pytest.fixture
	def rest_api_datasource(self):
		"""Create a REST API data source for testing"""
		return DataSource(
			id="api_test_1",
			name="Test REST API",
			type=DataSourceType.REST_API,
			connection_config={
				'base_url': 'https://api.example.com',
				'auth_type': 'bearer',
				'token': 'test-token-123'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	def graphql_api_datasource(self):
		"""Create a GraphQL API data source for testing"""
		return DataSource(
			id="graphql_test_1",
			name="Test GraphQL API",
			type=DataSourceType.GRAPHQL,
			connection_config={
				'base_url': 'https://api.example.com',
				'auth_type': 'api_key',
				'api_key': 'test-api-key-456',
				'api_key_header': 'X-API-Key'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	@pytest.fixture
	async def rest_connector(self, rest_api_datasource):
		"""Create REST API connector for testing"""
		connector = APIConnector(rest_api_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	@pytest.fixture
	async def graphql_connector(self, graphql_api_datasource):
		"""Create GraphQL API connector for testing"""
		connector = APIConnector(graphql_api_datasource, "test-tenant", "test-user")
		yield connector
		await connector.disconnect()
	
	async def test_rest_api_connect_success(self, rest_connector):
		"""Test successful REST API connection"""
		with aioresponses() as mock_resp:
			# Mock health check
			mock_resp.get(
				'https://api.example.com/',
				payload={'status': 'ok'}
			)
			
			# Mock OpenAPI spec discovery attempts (all fail)
			for path in ['/swagger.json', '/openapi.json', '/api-docs', '/docs/swagger.json']:
				mock_resp.get(f'https://api.example.com{path}', status=404)
			
			success = await rest_connector.connect()
			
			assert success is True
			assert rest_connector.session is not None
			assert rest_connector.base_url == 'https://api.example.com'
			assert 'Bearer test-token-123' in rest_connector.auth_headers['Authorization']
	
	async def test_graphql_connect_with_schema_discovery(self, graphql_connector):
		"""Test GraphQL connection with schema discovery"""
		with aioresponses() as mock_resp:
			# Mock health check
			mock_resp.get(
				'https://api.example.com/',
				payload={'status': 'ok'}
			)
			
			# Mock GraphQL introspection
			introspection_response = {
				'data': {
					'__schema': {
						'queryType': {'name': 'Query'},
						'mutationType': {'name': 'Mutation'},
						'types': [
							{'name': 'User', 'kind': 'OBJECT'},
							{'name': 'Order', 'kind': 'OBJECT'},
							{'name': 'Product', 'kind': 'OBJECT'}
						]
					}
				}
			}
			
			mock_resp.post(
				'https://api.example.com/graphql',
				payload=introspection_response
			)
			
			success = await graphql_connector.connect()
			
			assert success is True
			assert 'schema_types' in graphql_connector.connection_metadata
			assert 'User' in graphql_connector.connection_metadata['schema_types']
			assert 'Order' in graphql_connector.connection_metadata['schema_types']
	
	async def test_rest_api_query_execution(self, rest_connector):
		"""Test REST API query execution"""
		with aioresponses() as mock_resp:
			# Mock connection setup
			mock_resp.get('https://api.example.com/', payload={'status': 'ok'})
			for path in ['/swagger.json', '/openapi.json', '/api-docs', '/docs/swagger.json']:
				mock_resp.get(f'https://api.example.com{path}', status=404)
			
			await rest_connector.connect()
			
			# Mock API response
			api_response = {
				'data': [
					{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
					{'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}
				],
				'total': 2,
				'page': 1
			}
			
			mock_resp.get(
				'https://api.example.com/users',
				payload=api_response
			)
			
			result = await rest_connector.execute_query('GET /users', {'query_params': {'limit': 10}})
			
			assert result['api_call_type'] == 'GET'
			assert result['status_code'] == 200
			assert result['record_count'] == 2
			assert result['response'][0]['name'] == 'Alice'
	
	async def test_graphql_query_execution(self, graphql_connector):
		"""Test GraphQL query execution"""
		with aioresponses() as mock_resp:
			# Mock connection setup
			mock_resp.get('https://api.example.com/', payload={'status': 'ok'})
			mock_resp.post('https://api.example.com/graphql', payload={'data': {'__schema': {'types': []}}})
			
			await graphql_connector.connect()
			
			# Mock GraphQL response
			graphql_response = {
				'data': {
					'users': [
						{'id': '1', 'name': 'Alice', 'email': 'alice@example.com'},
						{'id': '2', 'name': 'Bob', 'email': 'bob@example.com'}
					]
				}
			}
			
			mock_resp.post(
				'https://api.example.com/graphql',
				payload=graphql_response
			)
			
			query = '{ users { id name email } }'
			result = await graphql_connector.execute_query(query, {'variables': {}})
			
			assert result['api_call_type'] == 'GRAPHQL'
			assert result['status_code'] == 200
			assert result['record_count'] == 1  # GraphQL returns nested data as single result
			assert 'users' in result['response'][0]
	
	async def test_api_capabilities(self, rest_connector):
		"""Test API connector capabilities"""
		capabilities = await rest_connector.get_capabilities()
		
		assert ConnectionCapability.BATCH_READ in capabilities
		assert ConnectionCapability.SCHEMA_INTROSPECTION in capabilities


class TestConnectorFactory:
	"""Test the connector factory"""
	
	def test_create_sql_connector(self):
		"""Test creating SQL connectors"""
		pg_datasource = DataSource(
			id="test_pg",
			name="Test PG",
			type=DataSourceType.POSTGRESQL,
			connection_config={},
			tenant_id="test",
			created_by="test"
		)
		
		connector = ConnectorFactory.create_connector(pg_datasource, "test", "test")
		
		assert isinstance(connector, SQLDatabaseConnector)
		assert connector.data_source.type == DataSourceType.POSTGRESQL
	
	def test_create_nosql_connector(self):
		"""Test creating NoSQL connectors"""
		mongo_datasource = DataSource(
			id="test_mongo",
			name="Test Mongo",
			type=DataSourceType.MONGODB,
			connection_config={},
			tenant_id="test",
			created_by="test"
		)
		
		connector = ConnectorFactory.create_connector(mongo_datasource, "test", "test")
		
		assert isinstance(connector, NoSQLConnector)
		assert connector.data_source.type == DataSourceType.MONGODB
	
	def test_create_api_connector(self):
		"""Test creating API connectors"""
		api_datasource = DataSource(
			id="test_api",
			name="Test API",
			type=DataSourceType.REST_API,
			connection_config={},
			tenant_id="test",
			created_by="test"
		)
		
		connector = ConnectorFactory.create_connector(api_datasource, "test", "test")
		
		assert isinstance(connector, APIConnector)
		assert connector.data_source.type == DataSourceType.REST_API
	
	def test_unsupported_connector_type(self):
		"""Test creating connector for unsupported type"""
		# Create a mock data source with unsupported type
		mock_datasource = Mock()
		mock_datasource.type = "UNSUPPORTED_TYPE"
		
		with pytest.raises(ValueError, match="No connector available"):
			ConnectorFactory.create_connector(mock_datasource, "test", "test")


class TestUniversalConnectorManager:
	"""Test the universal connector manager"""
	
	@pytest.fixture
	def connector_manager(self):
		"""Create connector manager for testing"""
		return UniversalConnectorManager("test-tenant", "test-user")
	
	@pytest.fixture
	def test_datasource(self):
		"""Create test data source"""
		return DataSource(
			id="test_ds_1",
			name="Test Data Source",
			type=DataSourceType.POSTGRESQL,
			connection_config={
				'host': 'localhost',
				'port': 5432,
				'database': 'testdb',
				'username': 'testuser',
				'password': 'testpass'
			},
			tenant_id="test-tenant",
			created_by="test-user"
		)
	
	async def test_create_connector_success(self, connector_manager, test_datasource):
		"""Test successful connector creation"""
		with patch.object(SQLDatabaseConnector, 'connect', return_value=True):
			with patch.object(SQLDatabaseConnector, 'get_capabilities', return_value=[ConnectionCapability.BATCH_READ]):
				connector = await connector_manager.create_connector(test_datasource)
				
				assert isinstance(connector, SQLDatabaseConnector)
				assert test_datasource.id in connector_manager.active_connectors
				assert connector_manager.active_connectors[test_datasource.id] == connector
	
	async def test_create_connector_connection_failure(self, connector_manager, test_datasource):
		"""Test connector creation with connection failure"""
		with patch.object(SQLDatabaseConnector, 'connect', return_value=False):
			with pytest.raises(ConnectionError, match="Failed to connect to data source"):
				await connector_manager.create_connector(test_datasource)
	
	async def test_get_connector(self, connector_manager, test_datasource):
		"""Test retrieving existing connector"""
		with patch.object(SQLDatabaseConnector, 'connect', return_value=True):
			with patch.object(SQLDatabaseConnector, 'get_capabilities', return_value=[]):
				created_connector = await connector_manager.create_connector(test_datasource)
				retrieved_connector = await connector_manager.get_connector(test_datasource.id)
				
				assert created_connector == retrieved_connector
	
	async def test_remove_connector(self, connector_manager, test_datasource):
		"""Test connector removal"""
		with patch.object(SQLDatabaseConnector, 'connect', return_value=True):
			with patch.object(SQLDatabaseConnector, 'get_capabilities', return_value=[]):
				with patch.object(SQLDatabaseConnector, 'disconnect', return_value=True):
					await connector_manager.create_connector(test_datasource)
					
					success = await connector_manager.remove_connector(test_datasource.id)
					
					assert success is True
					assert test_datasource.id not in connector_manager.active_connectors
	
	async def test_health_check_all(self, connector_manager, test_datasource):
		"""Test health check on all connectors"""
		with patch.object(SQLDatabaseConnector, 'connect', return_value=True):
			with patch.object(SQLDatabaseConnector, 'get_capabilities', return_value=[]):
				with patch.object(SQLDatabaseConnector, 'health_check', return_value=ConnectionHealth.HEALTHY):
					await connector_manager.create_connector(test_datasource)
					
					health_results = await connector_manager.health_check_all()
					
					assert test_datasource.id in health_results
					assert health_results[test_datasource.id] == ConnectionHealth.HEALTHY
	
	async def test_get_connector_stats(self, connector_manager, test_datasource):
		"""Test getting connector statistics"""
		mock_stats = {
			'connector_type': 'SQLDatabaseConnector',
			'data_source_id': test_datasource.id,
			'health_status': 'healthy',
			'capabilities': ['batch_read']
		}
		
		with patch.object(SQLDatabaseConnector, 'connect', return_value=True):
			with patch.object(SQLDatabaseConnector, 'get_capabilities', return_value=[ConnectionCapability.BATCH_READ]):
				with patch.object(SQLDatabaseConnector, 'get_connection_stats', return_value=mock_stats):
					await connector_manager.create_connector(test_datasource)
					
					stats = await connector_manager.get_connector_stats()
					
					assert stats['total_connectors'] == 1
					assert 'SQLDatabaseConnector' in stats['connector_types']
					assert stats['connector_types']['SQLDatabaseConnector'] == 1
					assert test_datasource.id in stats['connectors']


if __name__ == '__main__':
	# Run tests with: python -m pytest tests/unit/test_connectors.py -v
	pytest.main([__file__, '-v'])